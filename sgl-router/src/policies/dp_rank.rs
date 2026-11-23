// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-rank selection inside an already selected worker (`--dp-aware`).
//!
//! First match wins: a sticky/session key hashes to a fixed rank, so router
//! replicas agree without shared state; else the rank with the deepest cached
//! prefix; else [`DpRankPolicy`] picks: the fewest router in-flight requests,
//! the fewest KV tokens the rank is expected to hold by the engine's load
//! feed, or the worker's ranks in turn.

use crate::config::DpRankPolicy;
use crate::state::load_monitor::engine_reported_load::RankLoad;
use crate::workers::Worker;
use sha2::{Digest, Sha256};

/// `rank_tokens` for [`select_dp_rank`]: the KV each rank is expected to
/// hold, its committed tokens plus every request it holds, and every stamped
/// request this router sent it since the reading, at a typical request's KV.
/// KV in use lags: a rank whose requests are all young holds
/// little now and the most later, so ranking by it herds. Charging each
/// request its expected size ranks by where the rank is heading instead, and
/// a request dispatched between two reports counts before the engine has seen
/// it.
///
/// A typical request is the median request's KV: the ranks' mean KV per
/// request, each counted once per request it holds, at the lower median. Not
/// the worker-wide mean, and not the median rank: a few long-context requests,
/// even spread over most ranks, would otherwise inflate what every short
/// request is charged, and a rank near its KV cap would outrank one that is
/// nearly empty. Ranks whose requests hold no KV yet (queued on an older
/// publisher, which reports KV in use only) do not vote, or a two-rank worker
/// would charge nothing and send every request to that rank until its KV
/// shows up.
pub fn expected_rank_tokens(worker: &Worker, loads: &[RankLoad]) -> Vec<(u32, u64)> {
    let mut means: Vec<(u64, u64)> = loads
        .iter()
        .filter(|load| load.num_reqs > 0)
        .map(|load| (load.num_tokens / load.num_reqs, load.num_reqs))
        .filter(|&(mean, _)| mean > 0)
        .collect();
    means.sort_unstable();
    let requests: u64 = means.iter().map(|&(_, reqs)| reqs).sum();
    let median = requests.saturating_sub(1) / 2;
    let mut counted = 0;
    let per_req = means
        .iter()
        .find(|&&(_, reqs)| {
            counted += reqs;
            counted > median
        })
        .map_or(0, |&(mean, _)| mean);
    loads
        .iter()
        .map(|load| {
            let sent = worker.dp_rank_dispatched_since(load.rank, load.at) as u64;
            let held = load.num_reqs.saturating_add(sent);
            (
                load.rank,
                load.num_tokens.saturating_add(held.saturating_mul(per_req)),
            )
        })
        .collect()
}

/// Allow a cached request to spill only when saved queued work exceeds its
/// recomputation cost plus four cache pages of hysteresis. Cache depths include
/// host-backed entries: a GPU miss alone must not be charged as full prefill.
/// Call with fresh reports; incomplete/native-field-missing reports fail closed.
/// Dispatch stamps bridge load-publisher updates conservatively at this prompt's
/// full size, so a burst cannot repeatedly choose the same apparently idle rank.
pub fn prefill_work_rank(
    worker: &Worker,
    cached_rank: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    block_size: usize,
    loads: &[RankLoad],
) -> u32 {
    let ranks = worker.dp_ranks();
    if cached_rank >= ranks || block_size == 0 || prompt_tokens == 0 {
        return cached_rank;
    }
    let cached_tokens = |rank: u32| {
        prefix_depths
            .iter()
            .filter(|&&(r, _)| r == rank)
            .map(|&(_, depth)| depth)
            .max()
            .unwrap_or(0)
            .saturating_mul(block_size)
            .min(prompt_tokens)
    };
    let baseline_uncached = prompt_tokens - cached_tokens(cached_rank);
    // Busy-to-busy moves may at most double the request's necessary uncached
    // work (plus four pages). A full cold copy needs genuinely spare capacity,
    // not merely a shorter queue. Otherwise a saturated fleet keeps moving
    // long hot prefixes and amplifies its own prefill backlog.
    let extra_budget = baseline_uncached.saturating_add(block_size.saturating_mul(4));
    let eligible = |rank: u32| {
        if rank == cached_rank {
            return true;
        }
        let extra = cached_tokens(cached_rank).saturating_sub(cached_tokens(rank));
        if extra <= extra_budget {
            return true;
        }
        loads.iter().find(|l| l.rank == rank).is_some_and(|l| {
            l.num_reqs == 0
                && l.pending_prefill_tokens == Some(0)
                && worker.dp_rank_inflight(rank) == 0
                && worker.dp_rank_dispatched_since(rank, l.at) == 0
        })
    };
    let cost = |rank: u32| -> Option<u64> {
        let load = loads.iter().find(|load| load.rank == rank)?;
        let pending = load.pending_prefill_tokens?;
        let depth = prefix_depths
            .iter()
            .filter(|&&(r, _)| r == rank)
            .map(|&(_, depth)| depth)
            .max()
            .unwrap_or(0);
        let cached = depth.saturating_mul(block_size).min(prompt_tokens);
        let dispatched = worker.dp_rank_dispatched_since(rank, load.at) as u64;
        Some(
            pending
                .saturating_add(dispatched.saturating_mul(prompt_tokens as u64))
                .saturating_add((prompt_tokens - cached) as u64),
        )
    };
    let Some(costs) = (0..ranks)
        .map(|rank| cost(rank).map(|c| (rank, c)))
        .collect::<Option<Vec<_>>>()
    else {
        return cached_rank;
    };
    let current = costs[cached_rank as usize].1;
    let (best, best_cost) = costs
        .into_iter()
        .filter(|&(rank, _)| eligible(rank))
        .min_by_key(|&(rank, c)| (c, worker.dp_rank_inflight(rank), rank != cached_rank))
        .unwrap();
    if best_cost.saturating_add((block_size as u64).saturating_mul(4)) < current {
        best
    } else {
        cached_rank
    }
}

/// `None` for a single-rank worker. `prefix_depths` is `(rank, cached blocks)`;
/// `preferred` breaks a load tie among the deepest ranks (media affinity's
/// rendezvous rank, so replicas agree) and never outranks load or depth;
/// `rank_tokens` is `(rank, expected KV tokens)` from the engine's load feed, read
/// only under `least_tokens`, and ignored unless it covers every candidate so a
/// rank with no reading never looks idle.
pub fn select_dp_rank(
    worker: &Worker,
    affinity_key: Option<&str>,
    prefix_depths: &[(u32, usize)],
    preferred: Option<u32>,
    rank_tokens: &[(u32, u64)],
    policy: DpRankPolicy,
) -> Option<u32> {
    let ranks = worker.dp_ranks();
    if ranks <= 1 {
        return None;
    }
    if let Some(key) = affinity_key {
        let digest = Sha256::digest(key.as_bytes());
        let hash = u64::from_be_bytes(digest[..8].try_into().unwrap());
        return Some((hash % u64::from(ranks)) as u32);
    }
    let cached: Vec<_> = prefix_depths
        .iter()
        .filter(|&&(rank, depth)| rank < ranks && depth > 0)
        .collect();
    let candidates: Vec<u32> = match cached.iter().map(|&&(_, depth)| depth).max() {
        Some(deepest) => cached
            .iter()
            .filter(|&&&(_, depth)| depth == deepest)
            .map(|&&(rank, _)| rank)
            .collect(),
        None => (0..ranks).collect(),
    };
    if policy == DpRankPolicy::RoundRobin {
        return Some(candidates[worker.next_dp_rank_turn() % candidates.len()]);
    }
    let tokens_of = |rank: u32| {
        rank_tokens
            .iter()
            .find(|&&(reported, _)| reported == rank)
            .map(|&(_, tokens)| tokens)
    };
    let by_tokens = policy == DpRankPolicy::LeastTokens
        && candidates.iter().all(|&rank| tokens_of(rank).is_some());
    let key = |rank: u32| {
        let tokens = if by_tokens {
            tokens_of(rank).unwrap_or(0)
        } else {
            0
        };
        (
            tokens,
            worker.dp_rank_inflight(rank),
            preferred != Some(rank),
        )
    };
    // Random start so equally loaded ranks share traffic.
    let start = rand::random::<usize>() % candidates.len();
    candidates[start..]
        .iter()
        .chain(&candidates[..start])
        .copied()
        .min_by_key(|&rank| key(rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::{EngineProfile, WireProtocol};
    use std::time::{Duration, Instant};

    fn worker(dp_ranks: u32) -> Worker {
        let spec = WorkerSpec {
            id: WorkerId("w".into()),
            url: "http://w".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
            version_group: None,
            services: Default::default(),
        };
        let profile = EngineProfile {
            protocol: WireProtocol::default(),
            dp_ranks,
            openai: None,
            grpc_port: None,
            supports_inline_system: false,
        };
        Worker::with_cb_config(spec, None, profile)
    }

    const LEAST: DpRankPolicy = DpRankPolicy::LeastInFlight;
    const TOKENS: DpRankPolicy = DpRankPolicy::LeastTokens;

    #[test]
    fn prefill_work_balances_queue_against_cache_and_unreported_dispatches() {
        let w = worker(2);
        let at = Instant::now() - Duration::from_millis(1);
        let reading = |rank, pending| RankLoad {
            rank,
            num_tokens: 0,
            num_reqs: 0,
            pending_prefill_tokens: pending,
            at,
        };
        let cache = [(0, 39)]; // 9984 of 10000 tokens reusable on rank 0.
        let mut loads = [reading(0, Some(8000)), reading(1, Some(0))];
        // A shorter queue is not worth recomputing the prompt.
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 0);
        loads[0].pending_prefill_tokens = Some(30000);
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 1);
        // A burst must be counted before its next load-publisher update.
        let _sent: Vec<_> = (0..3).map(|_| w.dp_rank_timestamped_guard(1)).collect();
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 0);
        // Complete the prefill leg before considering a cold target idle again.
        drop(_sent);
        for load in &mut loads {
            load.at = Instant::now() + Duration::from_secs(1);
        }
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 1);
    }

    #[test]
    fn prefill_work_fails_closed_and_preserves_cache_on_small_gains() {
        let w = worker(2);
        let at = Instant::now();
        let mut loads = [
            RankLoad {
                rank: 0,
                num_tokens: 0,
                num_reqs: 0,
                pending_prefill_tokens: Some(10500),
                at,
            },
            RankLoad {
                rank: 1,
                num_tokens: 0,
                num_reqs: 0,
                pending_prefill_tokens: Some(0),
                at,
            },
        ];
        let cache = [(0, usize::MAX), (99, usize::MAX)];
        // Overflowing depths clamp to the prompt; 500 tokens is below hysteresis.
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 0);
        loads[0].pending_prefill_tokens = Some(u64::MAX);
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 1);
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads[..1]), 0);
        loads[1].pending_prefill_tokens = None;
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 0);
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 0, &loads), 0);
    }

    #[test]
    fn saturated_ranks_preserve_hot_prefix_but_balance_warm_replicas() {
        let w = worker(2);
        let at = Instant::now();
        let mut loads = [
            RankLoad {
                rank: 0,
                num_tokens: 0,
                num_reqs: 10,
                pending_prefill_tokens: Some(1_000_000),
                at,
            },
            RankLoad {
                rank: 1,
                num_tokens: 0,
                num_reqs: 1,
                pending_prefill_tokens: Some(1),
                at,
            },
        ];
        // An attractive short queue must not repeatedly recompute a 100k prefix.
        assert_eq!(
            prefill_work_rank(&w, 0, &[(0, 390)], 100_000, 256, &loads),
            0
        );
        // A similarly warm replica can still take load while busy.
        assert_eq!(
            prefill_work_rank(&w, 0, &[(0, 390), (1, 388)], 100_000, 256, &loads),
            1
        );
        // A truly idle rank can expand capacity: balancing is not disabled.
        loads[1].num_reqs = 0;
        loads[1].pending_prefill_tokens = Some(0);
        assert_eq!(
            prefill_work_rank(&w, 0, &[(0, 390)], 100_000, 256, &loads),
            1
        );
        // A dispatch not yet published consumes that idle expansion slot.
        let _sent = w.dp_rank_timestamped_guard(1);
        assert_eq!(
            prefill_work_rank(&w, 0, &[(0, 390)], 100_000, 256, &loads),
            0
        );
    }

    #[test]
    fn cold_requests_balance_without_extra_recomputation() {
        let w = worker(2);
        let at = Instant::now();
        let loads = [
            RankLoad {
                rank: 0,
                num_tokens: 0,
                num_reqs: 20,
                pending_prefill_tokens: Some(1_000_000),
                at,
            },
            RankLoad {
                rank: 1,
                num_tokens: 0,
                num_reqs: 2,
                pending_prefill_tokens: Some(10_000),
                at,
            },
        ];
        assert_eq!(prefill_work_rank(&w, 0, &[], 100_000, 256, &loads), 1);
    }

    #[test]
    fn affinity_then_prefix_then_load() {
        let w = worker(4);
        assert_eq!(
            select_dp_rank(&worker(1), Some("k"), &[], None, &[], LEAST),
            None
        );
        // Pinned: replicas and releases must agree on the mapping.
        assert_eq!(
            select_dp_rank(&w, Some("session-42"), &[(0, 9)], None, &[], LEAST),
            Some(1)
        );
        assert_eq!(
            select_dp_rank(&w, None, &[(1, 2), (3, 5), (9, 9)], None, &[], LEAST),
            Some(3)
        );
        let _busy = [0, 1, 3].map(|rank| w.dp_rank_guard(rank));
        assert_eq!(select_dp_rank(&w, None, &[], None, &[], LEAST), Some(2));
    }

    #[test]
    fn least_tokens_reads_engine_load_and_falls_back_to_in_flight() {
        let w = worker(4);
        let tokens = [(0, 50), (1, 10), (2, 10), (3, 90)];
        let _busy_1 = w.dp_rank_guard(1);
        // Fewest tokens; the router's in-flight count breaks the tie.
        assert_eq!(
            select_dp_rank(&w, None, &[], None, &tokens, TOKENS),
            Some(2)
        );
        // A cached prefix still outranks load.
        assert_eq!(
            select_dp_rank(&w, None, &[(3, 4)], None, &tokens, TOKENS),
            Some(3)
        );
        // Tokens outrank in-flight: rank 0 is idle but holds more KV.
        let _busy_2 = [2, 2].map(|rank| w.dp_rank_guard(rank));
        assert_eq!(
            select_dp_rank(&w, None, &[], None, &tokens, TOKENS),
            Some(1)
        );
        // No reading, or one that misses a candidate: least in flight (0 or 3).
        for partial in [&[][..], &[(1, 0)][..]] {
            let picked = select_dp_rank(&w, None, &[], None, partial, TOKENS).unwrap();
            assert!(picked == 0 || picked == 3, "{picked}");
        }
        // least_in_flight never reads the tokens.
        let picked = select_dp_rank(&w, None, &[], None, &tokens, LEAST).unwrap();
        assert!(picked == 0 || picked == 3, "{picked}");
    }

    #[test]
    fn preferred_rank_breaks_only_a_load_tie() {
        let w = worker(4);
        let tied = [(1, 5), (2, 5)];
        assert_eq!(
            select_dp_rank(&w, None, &tied, Some(2), &[], LEAST),
            Some(2)
        );
        assert_eq!(
            select_dp_rank(&w, None, &tied, Some(1), &[], LEAST),
            Some(1)
        );
        // Never over a deeper rank.
        assert_eq!(
            select_dp_rank(&w, None, &[(1, 5), (2, 4)], Some(2), &[], LEAST),
            Some(1)
        );
        // Load decides first: the preferred rank is busier.
        let _busy = w.dp_rank_guard(2);
        assert_eq!(
            select_dp_rank(&w, None, &tied, Some(2), &[], LEAST),
            Some(1)
        );
        let tokens = [(1, 50), (2, 10)];
        assert_eq!(
            select_dp_rank(&w, None, &tied, Some(1), &tokens, TOKENS),
            Some(2)
        );
    }

    #[test]
    fn expected_tokens_charge_every_request_at_the_median_request() {
        let w = worker(2);
        let reading = |rank, num_tokens, num_reqs, at| RankLoad {
            rank,
            num_tokens,
            num_reqs,
            pending_prefill_tokens: None,
            at,
        };
        let at = Instant::now();
        // Both ranks average 100 KV per request. Rank 1 holds more now; rank
        // 0's one request is charged the same.
        let loads = [reading(0, 100, 1, at), reading(1, 400, 4, at)];
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 200), (1, 800)]);
        // Stamped dispatches since the reading count as requests it will hold.
        let _sent = [0, 0, 0, 0].map(|rank| w.dp_rank_timestamped_guard(rank));
        let _plain = w.dp_rank_guard(0);
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 600), (1, 800)]);
        // A newer reading already covers them.
        let newer = Instant::now() + Duration::from_secs(1);
        let loads = [reading(0, 100, 1, newer), reading(1, 400, 4, newer)];
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 200), (1, 800)]);
        // No request on the worker: nothing to scale by.
        let loads = [reading(0, 0, 0, at), reading(1, 0, 0, at)];
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 0), (1, 0)]);
        // Mixed sizes: one long-context request on rank 0 must not inflate
        // what rank 1's eight short ones are charged, or the rank near its
        // cap would look lighter than the nearly empty one.
        let loads = [reading(0, 30_000, 1, newer), reading(1, 4_000, 8, newer)];
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 30_500), (1, 8_000)]);
        // Nor when long-context ranks are the majority: the median request is
        // still a short one, since requests vote, not ranks.
        let w3 = worker(3);
        let loads = [
            reading(0, 30_000, 1, newer),
            reading(1, 30_000, 1, newer),
            reading(2, 4_000, 8, newer),
        ];
        assert_eq!(
            expected_rank_tokens(&w3, &loads),
            [(0, 30_500), (1, 30_500), (2, 8_000)]
        );
        // A rank whose requests hold no KV yet does not set the charge to 0:
        // its three queued requests count at the other rank's 1000 each.
        let loads = [reading(0, 0, 3, newer), reading(1, 5_000, 5, newer)];
        assert_eq!(expected_rank_tokens(&w, &loads), [(0, 3_000), (1, 10_000)]);
    }
}
