// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use reqwest::{Client, RequestBuilder, Url};
use std::time::Duration;

/// Cancels unfinished engine work without delaying request cleanup.
#[derive(Default)]
pub struct AbortOnDrop(Option<RequestBuilder>);

impl AbortOnDrop {
    // Only router-minted IDs are safe: the engine aborts by prefix.
    pub(super) fn new(
        client: &Client,
        worker: &Url,
        headers: &HeaderMap,
        rid: Option<&str>,
    ) -> Self {
        let admin_key = std::env::var("SGL_ROUTER_WORKER_ABORT_API_KEY").ok();
        Self::with_admin_key(client, worker, headers, rid, admin_key.as_deref())
    }

    fn with_admin_key(
        client: &Client,
        worker: &Url,
        headers: &HeaderMap,
        rid: Option<&str>,
        admin_key: Option<&str>,
    ) -> Self {
        Self(rid.filter(|rid| !rid.is_empty()).and_then(|rid| {
            let mut request = client
                .post(worker.join("/abort_request").expect("validated worker URL"))
                .json(&serde_json::json!({"rid": rid, "abort_all": false}))
                .timeout(Duration::from_secs(5));
            let auth = match abort_auth(admin_key, headers) {
                Ok(auth) => auth,
                Err(_) => {
                    // Never print a management credential or fall back to an unrelated client key.
                    tracing::error!("invalid configured worker abort credential; abort suppressed");
                    return None;
                }
            };
            if let Some(auth) = auth {
                request = request.header(AUTHORIZATION, auth);
            }
            Some(request)
        }))
    }

    pub fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        let Some(request) = self.0.take() else { return };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Err(error) = request.send().await.and_then(|r| r.error_for_status()) {
                    tracing::warn!(%error, "engine abort failed");
                }
            });
        }
    }
}

/// Backend abort authorization is independent of inference-client credentials.
/// Supplying this credential changes only /abort_request, never inference headers.
fn abort_auth(
    admin_key: Option<&str>,
    headers: &HeaderMap,
) -> Result<Option<HeaderValue>, axum::http::header::InvalidHeaderValue> {
    match admin_key {
        Some(key) => {
            let mut value: HeaderValue = format!("Bearer {key}").parse()?;
            value.set_sensitive(true);
            Ok(Some(value))
        }
        None => Ok(headers.get(AUTHORIZATION).cloned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_admin_key_overrides_client_key_only_for_abort() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer client-key"));
        let auth = abort_auth(Some("backend-admin"), &headers)
            .unwrap()
            .unwrap();
        assert_eq!(auth, "Bearer backend-admin");
        assert!(auth.is_sensitive());
        assert!(!format!("{auth:?}").contains("backend-admin"));
        assert_eq!(headers[AUTHORIZATION], "Bearer client-key");
    }

    #[test]
    fn unconfigured_backend_retains_legacy_client_auth() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer client-key"));
        assert_eq!(
            abort_auth(None, &headers).unwrap(),
            Some(headers[AUTHORIZATION].clone())
        );
        assert!(abort_auth(None, &HeaderMap::new()).unwrap().is_none());
    }

    #[test]
    fn malformed_management_key_does_not_fall_back_to_client() {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer client-key"));
        assert!(abort_auth(Some("bad\nkey"), &headers).is_err());
    }

    #[test]
    fn abort_request_uses_management_key_and_keeps_the_router_minted_rid() {
        let mut guard = AbortOnDrop::with_admin_key(
            &Client::new(),
            &Url::parse("http://worker:8000").unwrap(),
            &HeaderMap::new(),
            Some("router-minted-id"),
            Some("backend-admin"),
        );
        let request = guard.0.take().unwrap().build().unwrap();
        assert_eq!(request.url().path(), "/abort_request");
        assert_eq!(request.headers()[AUTHORIZATION], "Bearer backend-admin");
        let body: serde_json::Value =
            serde_json::from_slice(request.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(body["rid"], "router-minted-id");
        assert_eq!(body["abort_all"], false);
    }

    #[test]
    fn a_missing_rid_never_sends_a_broad_abort() {
        let guard = AbortOnDrop::with_admin_key(
            &Client::new(),
            &Url::parse("http://worker:8000").unwrap(),
            &HeaderMap::new(),
            None,
            Some("backend-admin"),
        );
        assert!(guard.0.is_none());
    }
}
