//! Optional cold-input quotes. Unknown cache reloads retain the existing route.
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct PrefillCostSummary {
    pub schema: u64,
    pub model: u64,
    pub boot: u64,
    pub sequence: u64,
    pub coefficients_ps: [u64; 5],
    pub queue_us: u64,
    pub remaining_us: u64,
}

impl PrefillCostSummary {
    pub fn valid(&self) -> bool {
        self.schema == 1
            && self.model > 0
            && self.sequence > 0
            && self.coefficients_ps.iter().any(|&c| c > 0)
    }

    /// Analytical sum of per-request 8192-row chunks, using saturating u128
    /// arithmetic. Quotes cannot wrap into an artificially cheap destination.
    pub fn compute_us(&self, rows: u64, prefix: u64) -> u64 {
        if rows == 0 {
            return 0;
        }
        let [a, b, c, d, e] = self.coefficients_ps.map(u128::from);
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
            coefficients_ps: [0, 1000000, 0, 0, 0],
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
    fn pathological_work_does_not_wrap() {
        let mut s = sample();
        s.coefficients_ps = [u64::MAX; 5];
        assert_eq!(s.compute_us(u64::MAX, u64::MAX), u64::MAX);
    }
}
