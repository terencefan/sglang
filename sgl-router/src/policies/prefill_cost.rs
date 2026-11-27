//! Optional cold-input quotes. Unknown cache reloads retain the existing route.
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum CostCoefficients {
    Local([u64; 5]),
    Shared([u64; 7]),
}
impl From<[u64; 5]> for CostCoefficients {
    fn from(value: [u64; 5]) -> Self {
        Self::Local(value)
    }
}
impl From<[u64; 7]> for CostCoefficients {
    fn from(value: [u64; 7]) -> Self {
        Self::Shared(value)
    }
}
impl CostCoefficients {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Local(v) => v,
            Self::Shared(v) => v,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub enum FeatureContract {
    #[default]
    #[serde(rename = "local-v1")]
    Local,
    #[serde(rename = "shared-dp-v2")]
    Shared,
    #[serde(other)]
    Unsupported,
}
fn one() -> u64 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct PrefillCostSummary {
    pub schema: u64,
    pub model: u64,
    pub boot: u64,
    pub sequence: u64,
    pub coefficients_ps: CostCoefficients,
    #[serde(default = "one")]
    pub dp_degree: u64,
    #[serde(default)]
    pub feature_contract: FeatureContract,
    pub queue_us: u64,
    pub remaining_us: u64,
}

impl PrefillCostSummary {
    pub fn valid(&self) -> bool {
        let contract = match (self.schema, self.feature_contract, self.coefficients_ps) {
            (1, FeatureContract::Local, CostCoefficients::Local(_)) => self.dp_degree == 1,
            (2, FeatureContract::Shared, CostCoefficients::Shared(_)) => {
                (1..=64).contains(&self.dp_degree)
            }
            _ => false,
        };
        contract
            && self.model > 0
            && self.sequence > 0
            && self.coefficients_ps.as_slice().iter().any(|&c| c > 0)
    }

    /// Analytical sum of per-request 8192-row chunks, using saturating u128
    /// arithmetic. Quotes cannot wrap into an artificially cheap destination.
    pub fn compute_us(&self, rows: u64, prefix: u64) -> u64 {
        if rows == 0 {
            return 0;
        }
        let values = self.coefficients_ps.as_slice();
        let [a, mut b, c, d, e] =
            [values[0], values[1], values[2], values[3], values[4]].map(u128::from);
        if let CostCoefficients::Shared(v) = self.coefficients_ps {
            // Same saturated-peer estimate used by the Python queue/admission
            // path. Actual current-forward prediction uses the measured DP rows.
            b = b
                .saturating_add(u128::from(v[5]).saturating_mul(u128::from(self.dp_degree)))
                .saturating_add(u128::from(v[6]));
        }
        let n = u128::from(rows / 8192);
        let tail = u128::from(rows % 8192);
        let prefix = u128::from(prefix);
        let sum_prefix = n.saturating_mul(prefix).saturating_add(
            8192u128
                .saturating_mul(n)
                .saturating_mul(n.saturating_sub(1))
                / 2,
        );
        let full = n.saturating_mul(
            a.saturating_add(b.saturating_mul(8192))
                .saturating_add(c.saturating_mul(8192 * 8192))
                .saturating_add(e),
        );
        let mut ps = full.saturating_add(d.saturating_mul(8192).saturating_mul(sum_prefix));
        if tail > 0 {
            ps = ps
                .saturating_add(a)
                .saturating_add(b.saturating_mul(tail))
                .saturating_add(c.saturating_mul(tail).saturating_mul(tail))
                .saturating_add(
                    d.saturating_mul(tail)
                        .saturating_mul(prefix.saturating_add(n.saturating_mul(8192))),
                )
                .saturating_add(e);
        }
        (ps.saturating_add(999999) / 1000000).min(u128::from(u64::MAX)) as u64
    }

    pub fn finish_us(&self, prompt: u64, unseen_dispatches: u64) -> Option<u64> {
        self.valid().then(|| {
            self.remaining_us
                .saturating_add(self.queue_us)
                .saturating_add(
                    self.compute_us(prompt, 0)
                        .saturating_mul(unseen_dispatches.saturating_add(1)),
                )
        })
    }
}

/// Require a >10% improvement; ties retain the existing decision.
pub fn materially_better(candidate: u64, baseline: u64) -> bool {
    u128::from(candidate) * 10 < u128::from(baseline) * 9
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> PrefillCostSummary {
        PrefillCostSummary {
            schema: 1,
            model: 1,
            boot: 1,
            sequence: 1,
            coefficients_ps: [0, 1000000, 0, 0, 0].into(),
            dp_degree: 1,
            feature_contract: FeatureContract::Local,
            queue_us: 10,
            remaining_us: 20,
        }
    }
    #[test]
    fn chunks_and_dispatch_reservations() {
        let s = sample();
        assert_eq!(s.compute_us(20000, 0), 20000);
        assert_eq!(s.finish_us(100, 2), Some(330));
    }
    #[test]
    fn reject_invalid_and_keep_near_ties() {
        let mut s = sample();
        s.schema = 2;
        assert_eq!(s.finish_us(100, 0), None);
        assert!(!materially_better(90, 100));
        assert!(materially_better(89, 100));
    }
    #[test]
    fn shared_contract_roundtrips_and_matches_python_chunk_quote() {
        let value = serde_json::json!({"schema":2,"feature_contract":"shared-dp-v2",
            "dp_degree":8,"model":1,"boot":1,"sequence":1,
            "coefficients_ps":[1000000000u64,2000000,100,1,100000000,3000000,4000000],
            "queue_us":0,"remaining_us":0});
        let quote: PrefillCostSummary = serde_json::from_value(value.clone()).unwrap();
        assert!(quote.valid());
        assert_eq!(quote.compute_us(20000,4096),618238);
        for (key,bad) in [("dp_degree",serde_json::json!(0)),
            ("dp_degree",serde_json::json!(65)),
            ("schema",serde_json::json!(1)),
            ("feature_contract",serde_json::json!("future-unknown")),
            ("coefficients_ps",serde_json::json!([0,1000000,0,0,0]))] {
            let mut changed=value.clone();changed[key]=bad;
            let parsed: PrefillCostSummary=serde_json::from_value(changed).unwrap();
            assert!(!parsed.valid());
        }
    }

    #[test]
    fn shared_pathological_work_saturates() {
        let mut s=sample();s.schema=2;s.feature_contract=FeatureContract::Shared;
        s.dp_degree=8;s.coefficients_ps=[u64::MAX;7].into();
        assert!(s.valid());assert_eq!(s.compute_us(u64::MAX,u64::MAX),u64::MAX);
    }

    #[test]
    fn pathological_work_does_not_wrap() {
        let mut s = sample();
        s.coefficients_ps = [u64::MAX; 5].into();
        assert_eq!(s.compute_us(u64::MAX, u64::MAX), u64::MAX);
    }
}
