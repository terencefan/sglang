//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;
const MAX_LOAD_SNAPSHOT_AGE: Duration = Duration::from_secs(30);

fn parse_total_token_load(json: &Value) -> Option<isize> {
    if let Some(loads) = json.get("loads") {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        let max_age = MAX_LOAD_SNAPSHOT_AGE.as_secs_f64();
        let total = loads.as_array()?.iter().try_fold(0_u64, |total, load| {
            // Newer SGLang servers return a scheduler snapshot timestamp for every
            // DP rank.  The HTTP endpoint can remain responsive while that snapshot
            // is frozen, so a successful response alone is not proof of fresh load.
            // Keep accepting timestamp-less payloads for compatibility with the
            // first version of the `loads` schema.
            if let Some(timestamp) = load.get("timestamp") {
                let timestamp = timestamp.as_f64()?;
                if !timestamp.is_finite() || (now - timestamp).abs() > max_age {
                    return None;
                }
            }

            total.checked_add(load.get("num_total_tokens")?.as_u64()?)
        })?;

        return isize::try_from(total).ok();
    }

    let total = json.get("aggregate")?.get("total_tokens")?.as_u64()?;
    isize::try_from(total).ok()
}

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => parse_total_token_load(&json).unwrap_or(-1),
                _ => -1,
            },
            _ => -1,
        }
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads.into_iter().filter(|load| load.load >= 0) {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    use axum::{extract::Query, http::StatusCode, routing::get, Json, Router};
    use serde_json::{json, Value};
    use tokio::net::TcpListener;

    use super::{parse_total_token_load, WorkerManager};

    #[test]
    fn parses_current_loads_schema() {
        let response = json!({
            "timestamp": "2026-08-13T00:00:00Z",
            "version": "0.5.16",
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 120},
                {"dp_rank": 1, "num_total_tokens": 80}
            ]
        });

        assert_eq!(parse_total_token_load(&response), Some(200));
    }

    #[test]
    fn accepts_empty_loads_as_zero_load() {
        assert_eq!(parse_total_token_load(&json!({"loads": []})), Some(0));
    }

    #[test]
    fn falls_back_to_legacy_aggregate_schema() {
        let response = json!({"aggregate": {"total_tokens": 200}});

        assert_eq!(parse_total_token_load(&response), Some(200));
    }

    #[test]
    fn current_schema_takes_precedence_over_legacy_aggregate() {
        let response = json!({
            "loads": [{"num_total_tokens": 200}],
            "aggregate": {"total_tokens": 999}
        });

        assert_eq!(parse_total_token_load(&response), Some(200));
    }

    #[test]
    fn rejects_malformed_current_schema_without_legacy_fallback() {
        let response = json!({
            "loads": [{"num_total_tokens": 120}, {"num_total_tokens": "80"}],
            "aggregate": {"total_tokens": 200}
        });

        assert_eq!(parse_total_token_load(&response), None);
    }

    #[test]
    fn rejects_negative_or_overflowing_loads() {
        assert_eq!(
            parse_total_token_load(&json!({"loads": [{"num_total_tokens": -1}]})),
            None
        );
        assert_eq!(
            parse_total_token_load(&json!({
                "loads": [
                    {"num_total_tokens": i64::MAX},
                    {"num_total_tokens": 1}
                ]
            })),
            None
        );
    }

    #[test]
    fn rejects_missing_load_fields() {
        assert_eq!(parse_total_token_load(&json!({})), None);
        assert_eq!(
            parse_total_token_load(&json!({"loads": [{"dp_rank": 0}]})),
            None
        );
        assert_eq!(
            parse_total_token_load(&json!({"aggregate": {"requests": 1}})),
            None
        );
    }

    #[test]
    fn rejects_stale_or_invalid_scheduler_snapshots() {
        assert_eq!(
            parse_total_token_load(&json!({
                "loads": [{"timestamp": 0.0, "num_total_tokens": 0}]
            })),
            None
        );
        assert_eq!(
            parse_total_token_load(&json!({
                "loads": [{"timestamp": "not-a-number", "num_total_tokens": 0}]
            })),
            None
        );
    }

    #[test]
    fn accepts_fresh_scheduler_snapshot() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        assert_eq!(
            parse_total_token_load(&json!({
                "loads": [{"timestamp": now, "num_total_tokens": 123}]
            })),
            Some(123)
        );
    }

    async fn current_loads_handler(
        Query(query): Query<HashMap<String, String>>,
    ) -> Result<Json<Value>, StatusCode> {
        if query.get("include").map(String::as_str) != Some("core") {
            return Err(StatusCode::BAD_REQUEST);
        }

        Ok(Json(json!({
            "timestamp": "2026-08-13T00:00:00Z",
            "version": "0.5.16",
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 120},
                {"dp_rank": 1, "num_total_tokens": 80}
            ]
        })))
    }

    #[tokio::test]
    async fn requests_v1_loads_core_and_parses_current_schema() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/loads", get(current_loads_handler));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let client = reqwest::Client::new();
        let load =
            WorkerManager::parse_load_response(&client, &format!("http://{address}"), None).await;

        server.abort();
        assert_eq!(load, 200);
    }

    #[tokio::test]
    async fn treats_unsuccessful_or_invalid_responses_as_failed_loads() {
        for app in [
            Router::new().route(
                "/v1/loads",
                get(|| async { StatusCode::SERVICE_UNAVAILABLE }),
            ),
            Router::new().route("/v1/loads", get(|| async { "not json" })),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });

            let client = reqwest::Client::new();
            let load =
                WorkerManager::parse_load_response(&client, &format!("http://{address}"), None)
                    .await;

            server.abort();
            assert_eq!(load, -1);
        }
    }
}
