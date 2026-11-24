// SPDX-License-Identifier: Apache-2.0
//! Router-local, per-prefill-worker admission accounting. The single mutex in
//! MetricsRegistry covers selection and booking together, across all DP ranks.
//! Baseline/extra are estimates at routing time, not measured GPU execution.
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

pub(crate) const WINDOW: Duration = Duration::from_secs(60);
pub(crate) const RESIDENCE: Duration = Duration::from_secs(60);
const MAX_RESIDENTS: usize = 4096;

#[derive(Debug)]
struct Bucket {
    millisecond: u64,
    baseline: u128,
    extra: u128,
}
#[derive(Debug)]
struct Resident {
    rank: u32,
    expires: Instant,
}

#[derive(Debug)]
pub(crate) struct PrefillGuard {
    epoch: Instant,
    buckets: VecDeque<Bucket>,
    baseline: u128,
    extra: u128,
    residents: HashMap<[u8; 32], Resident>,
}
impl PrefillGuard {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            epoch: now,
            buckets: VecDeque::new(),
            baseline: 0,
            extra: 0,
            residents: HashMap::new(),
        }
    }
    fn tick(&self, now: Instant) -> u64 {
        u64::try_from(now.saturating_duration_since(self.epoch).as_millis()).unwrap_or(u64::MAX)
    }
    fn prune(&mut self, now: Instant) {
        let tick = self.tick(now);
        while self
            .buckets
            .front()
            .is_some_and(|b| tick.saturating_sub(b.millisecond) >= WINDOW.as_millis() as u64)
        {
            let b = self.buckets.pop_front().unwrap();
            self.baseline -= b.baseline;
            self.extra -= b.extra;
        }
        self.residents.retain(|_, r| now < r.expires);
    }
    /// Windowed admission: old credits age out. If that makes the historical
    /// ratio exceed 10%, no additional extra work is admitted until it recovers.
    /// No retroactive claim is made that every future sliding window stays <=10%.
    pub(crate) fn allowance(&mut self, now: Instant, baseline: u64) -> u64 {
        self.prune(now);
        u64::try_from(((self.baseline + u128::from(baseline)) / 10).saturating_sub(self.extra))
            .unwrap_or(u64::MAX)
    }
    pub(crate) fn resident(&mut self, now: Instant, key: Option<[u8; 32]>) -> Option<u32> {
        self.prune(now);
        key.and_then(|k| self.residents.get(&k).map(|r| r.rank))
    }
    pub(crate) fn book(
        &mut self,
        now: Instant,
        baseline: u64,
        extra: u64,
        key: Option<[u8; 32]>,
        cached: u32,
        selected: u32,
    ) {
        self.prune(now);
        let tick = self.tick(now);
        if self.buckets.back().is_none_or(|b| b.millisecond != tick) {
            self.buckets.push_back(Bucket {
                millisecond: tick,
                baseline: 0,
                extra: 0,
            });
        }
        let b = self.buckets.back_mut().unwrap();
        b.baseline += u128::from(baseline);
        b.extra += u128::from(extra);
        self.baseline += u128::from(baseline);
        self.extra += u128::from(extra);
        if let Some(k) = key {
            // Keeping the same target must not refresh the lease forever.
            if self.residents.get(&k).is_some_and(|r| r.rank == selected) {
                return;
            }
            if selected != cached || self.residents.contains_key(&k) {
                if self.residents.len() >= MAX_RESIDENTS && !self.residents.contains_key(&k) {
                    if let Some(old) = self
                        .residents
                        .iter()
                        .min_by_key(|(_, r)| r.expires)
                        .map(|(k, _)| *k)
                    {
                        self.residents.remove(&old);
                    }
                }
                self.residents.insert(
                    k,
                    Resident {
                        rank: selected,
                        expires: now + RESIDENCE,
                    },
                );
            }
        }
    }
    pub(crate) fn snapshot(&mut self, now: Instant) -> (u128, u128, usize) {
        self.prune(now);
        (self.baseline, self.extra, self.residents.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_credit_and_expiry_are_not_an_unbounded_token_bank() {
        let t = Instant::now();
        let mut g = PrefillGuard::new(t);
        assert_eq!(g.allowance(t, 100), 10);
        g.book(t, 100_000, 0, None, 0, 0);
        assert_eq!(g.allowance(t, 0), 10_000);
        g.book(t + Duration::from_secs(50), 0, 9000, None, 0, 1);
        assert_eq!(g.allowance(t + Duration::from_secs(51), 0), 1000);
        assert_eq!(g.allowance(t + Duration::from_secs(61), 100), 0);
        assert_eq!(g.allowance(t + Duration::from_secs(111), 100), 10);
    }
    #[test]
    fn residence_is_scoped_and_expires_without_refresh_on_reuse() {
        let t = Instant::now();
        let mut g = PrefillGuard::new(t);
        let a = Some([1; 32]);
        let b = Some([2; 32]);
        g.book(t, 10000, 100, a, 0, 1);
        assert_eq!(g.resident(t, a), Some(1));
        assert_eq!(g.resident(t, b), None);
        g.book(t + Duration::from_secs(40), 10000, 100, a, 0, 1);
        assert_eq!(g.resident(t + Duration::from_secs(60), a), None);
    }
    #[test]
    fn concurrent_admissions_cannot_spend_the_same_credit() {
        use std::sync::{Arc, Mutex};
        let t = Instant::now();
        let g = Arc::new(Mutex::new(PrefillGuard::new(t)));
        g.lock().unwrap().book(t, 100000, 0, None, 0, 0);
        let threads: Vec<_> = (0..32)
            .map(|_| {
                let g = g.clone();
                std::thread::spawn(move || {
                    let mut g = g.lock().unwrap();
                    if g.allowance(t, 0) >= 1000 {
                        g.book(t, 0, 1000, None, 0, 1);
                        1
                    } else {
                        0
                    }
                })
            })
            .collect();
        assert_eq!(
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .sum::<usize>(),
            10
        );
    }
    #[test]
    fn baseline_math_does_not_overflow_at_u64_limits() {
        let t = Instant::now();
        let mut g = PrefillGuard::new(t);
        g.book(t, u64::MAX, 0, None, 0, 0);
        g.book(t, u64::MAX, 0, None, 0, 0);
        assert_eq!(g.allowance(t, 0), u64::MAX / 5);
    }
}
