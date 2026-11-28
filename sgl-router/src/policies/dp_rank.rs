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

/// Experimental P-only queue-first selection. The caller supplies fresh reports.
/// A complete native snapshot is mandatory; otherwise retain the old choice.
/// Dispatch stamps reserve full prompt work until the next report, preventing
/// a burst from herding onto a rank whose idle reading has not caught up.
/// L3 reuse is not assumed to be free: acceptance measures actual cache hits,
/// cold throughput and TTFT. Cache depth only breaks equal-work ties.
pub fn prefill_queue_first_rank(
    worker: &Worker,
    fallback: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    loads: &[RankLoad],
) -> u32 {
    let ranks = worker.dp_ranks();
    if ranks <= 1
        || fallback >= ranks
        || prompt_tokens == 0
        || loads.len() != ranks as usize
        || (0..ranks).any(|r| loads.iter().filter(|l| l.rank == r).count() != 1)
        || loads.iter().any(|l| l.pending_prefill_tokens.is_none())
    {
        return fallback;
    }
    let start = worker.next_dp_rank_turn() % ranks as usize;
    (0..ranks)
        .map(|i| ((i as usize + start) % ranks as usize) as u32)
        .min_by_key(|&rank| {
            let load = loads.iter().find(|l| l.rank == rank).unwrap();
            let sent = worker.dp_rank_dispatched_since(rank, load.at) as u64;
            let queued = load
                .pending_prefill_tokens
                .unwrap()
                .saturating_add(sent.saturating_mul(prompt_tokens as u64));
            let depth = prefix_depths
                .iter()
                .filter(|&&(r, _)| r == rank)
                .map(|&(_, d)| d)
                .max()
                .unwrap_or(0);
            (
                queued,
                std::cmp::Reverse(depth),
                load.num_reqs.saturating_add(sent),
                worker.dp_rank_inflight(rank),
            )
        })
        .unwrap_or(fallback)
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
    prefill_work_rank_impl(
        worker,
        cached_rank,
        prefix_depths,
        prompt_tokens,
        block_size,
        loads,
        u64::MAX,
        None,
        false,
    )
}

/// Old rule is evaluated only as a counterfactual; never used to serve traffic.
pub(crate) fn legacy_prefill_work_rank(
    worker: &Worker,
    cached_rank: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    block_size: usize,
    loads: &[RankLoad],
) -> u32 {
    prefill_work_rank_impl(
        worker,
        cached_rank,
        prefix_depths,
        prompt_tokens,
        block_size,
        loads,
        u64::MAX,
        None,
        true,
    )
}

pub(crate) fn budgeted_prefill_work_rank(
    worker: &Worker,
    cached_rank: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    block_size: usize,
    loads: &[RankLoad],
    allowance: u64,
    resident: Option<u32>,
) -> u32 {
    prefill_work_rank_impl(
        worker,
        cached_rank,
        prefix_depths,
        prompt_tokens,
        block_size,
        loads,
        allowance,
        resident,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn prefill_work_rank_impl(
    worker: &Worker,
    cached_rank: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    block_size: usize,
    loads: &[RankLoad],
    allowance: u64,
    resident: Option<u32>,
    legacy: bool,
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
    // Every destination, including an idle one, pays the same 25% + four
    // pages cap. Attention-DP idle does not mean the EP group has spare compute.
    let extra_budget = (if legacy {
        baseline_uncached
    } else {
        baseline_uncached / 4
    })
    .saturating_add(block_size.saturating_mul(4));
    let eligible = |rank: u32| {
        if rank == cached_rank {
            return true;
        }
        let extra = cached_tokens(cached_rank).saturating_sub(cached_tokens(rank));
        if extra <= extra_budget && (extra as u64) <= allowance {
            return true;
        }
        legacy
            && loads.iter().find(|l| l.rank == rank).is_some_and(|l| {
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
    let selected = if best_cost.saturating_add((block_size as u64).saturating_mul(4)) < current {
        best
    } else {
        cached_rank
    };
    // Residence is a preference within the existing four-page hysteresis, not
    // a hard session route: a clearly better queue/cache choice still wins.
    if !legacy {
        if let Some(r) = resident.filter(|&r| r < ranks && eligible(r)) {
            if let (Some(rc), Some(sc)) = (cost(r), cost(selected)) {
                if rc <= sc.saturating_add((block_size as u64).saturating_mul(4)) {
                    return r;
                }
            }
        }
    }
    selected
}

/// Escape the per-request cache cap only for sustained queue pressure. The
/// caller still serializes and charges the same worker-wide 10% rolling budget.
/// Times are scheduling estimates, not GPU timings: extra/min(rate) also prices
/// the additional work shared by the EP group rather than treating idle attention
/// ranks as free GPUs. Explicit sticky routes never enter this path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefillOverflow {
    pub rank: u32,
    pub source_queue_ms: u64,
    pub predicted_finish_ms: u64,
    pub extra_tokens: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_overflow_rank(
    worker: &Worker,
    cached_rank: u32,
    normal_rank: u32,
    prefix_depths: &[(u32, usize)],
    prompt_tokens: usize,
    block_size: usize,
    loads: &[RankLoad],
    allowance: u64,
    congested: &[u32],
    target_ready: impl Fn(u32) -> bool,
) -> Option<PrefillOverflow> {
    if !congested.contains(&cached_rank)
        || block_size == 0
        || loads.len() != worker.dp_ranks() as usize
    {
        return None;
    }
    let cached = |rank| {
        prefix_depths
            .iter()
            .filter(|&&(r, _)| r == rank)
            .map(|&(_, d)| d.saturating_mul(block_size).min(prompt_tokens))
            .max()
            .unwrap_or(0) as u64
    };
    let reading = |rank| loads.iter().find(|l| l.rank == rank);
    // Incomplete rates fail closed; idle ranks retain their most recent measured rate.
    let slowest = loads
        .iter()
        .map(|l| l.prefill_tokens_per_second.filter(|&r| r > 0))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min()?;
    let millis = |tokens: u64, rate: u64| -> u64 {
        (u128::from(tokens) * 1000 / u128::from(rate)).min(u128::from(u64::MAX)) as u64
    };
    let queue = |l: &RankLoad| -> Option<u64> {
        let dispatched = worker.dp_rank_dispatched_since(l.rank, l.at) as u64;
        Some(
            l.pending_prefill_tokens?
                .saturating_add(dispatched.saturating_mul(prompt_tokens as u64)),
        )
    };
    let finish = |l: &RankLoad| -> Option<u64> {
        Some(millis(
            queue(l)?.saturating_add((prompt_tokens as u64).saturating_sub(cached(l.rank))),
            l.prefill_tokens_per_second?,
        ))
    };
    let source = reading(cached_rank)?;
    let source_queue_ms = millis(queue(source)?, source.prefill_tokens_per_second?);
    if source_queue_ms < 10_000 {
        return None;
    }
    let source_finish = finish(source)?;
    let normal_finish = finish(reading(normal_rank)?)?;
    let mut best = None;
    for l in loads {
        if l.rank == cached_rank
            || l.rank >= worker.dp_ranks()
            || l.num_reqs > 4
            || worker.dp_rank_dispatched_since(l.rank, l.at) > 0
            || !target_ready(l.rank)
        {
            continue;
        }
        let extra = cached(cached_rank).saturating_sub(cached(l.rank));
        if extra == 0 || extra > allowance || extra > 131_072 {
            continue;
        }
        let predicted_finish_ms = finish(l)?.saturating_add(millis(extra, slowest));
        // Require both 2x improvement over staying, and >=5s over the normal rule.
        if predicted_finish_ms.saturating_mul(2) >= source_finish
            || predicted_finish_ms.saturating_add(5000) >= normal_finish
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|b: &PrefillOverflow| predicted_finish_ms < b.predicted_finish_ms)
        {
            best = Some(PrefillOverflow {
                rank: l.rank,
                source_queue_ms,
                predicted_finish_ms,
                extra_tokens: extra,
            });
        }
    }
    best
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

    #[test]
    fn queue_first_spills_despite_deep_local_cache() {
        let w = worker(2);
        let at = Instant::now();
        let mut loads = vec![
            RankLoad {
                rank: 0,
                num_tokens: 0,
                num_reqs: 8,
                pending_prefill_tokens: Some(100_000),
                prefill_tokens_per_second: None,
                at,
            },
            RankLoad {
                rank: 1,
                num_tokens: 0,
                num_reqs: 0,
                pending_prefill_tokens: Some(0),
                prefill_tokens_per_second: None,
                at,
            },
        ];
        assert_eq!(
            prefill_queue_first_rank(&w, 0, &[(0, 390)], 100_000, &loads),
            1
        );
        loads[0].pending_prefill_tokens = Some(0);
        assert_eq!(
            prefill_queue_first_rank(&w, 0, &[(0, 390)], 100_000, &loads),
            0
        );
        let _sent = w.dp_rank_timestamped_guard(0);
        assert_eq!(
            prefill_queue_first_rank(&w, 0, &[(0, 390)], 100_000, &loads),
            1
        );
    }

    #[test]
    fn queue_first_incomplete_or_legacy_loads_fail_closed() {
        let w = worker(2);
        let load = RankLoad {
            rank: 0,
            num_tokens: 0,
            num_reqs: 8,
            pending_prefill_tokens: Some(100_000),
            prefill_tokens_per_second: None,
            at: Instant::now(),
        };
        assert_eq!(prefill_queue_first_rank(&w, 0, &[], 8192, &[load]), 0);
        let mut other = load;
        other.rank = 1;
        other.pending_prefill_tokens = None;
        assert_eq!(
            prefill_queue_first_rank(&w, 0, &[], 8192, &[load, other]),
            0
        );
        assert_eq!(prefill_queue_first_rank(&w, 0, &[], 8192, &[load, load]), 0);
    }

    #[test]
    fn prefill_budget_caps_extra_work_even_for_idle_targets() {
        let w = worker(2);
        let at = Instant::now();
        let loads = [
            RankLoad {
                prefill_tokens_per_second: None,
                rank: 0,
                num_tokens: 0,
                num_reqs: 1,
                pending_prefill_tokens: Some(100_000),
                at,
            },
            RankLoad {
                prefill_tokens_per_second: None,
                rank: 1,
                num_tokens: 0,
                num_reqs: 0,
                pending_prefill_tokens: Some(0),
                at,
            },
        ];
        let expensive = [(0, 351), (1, 320)]; // baseline 10144, extra 7936
        assert_eq!(
            legacy_prefill_work_rank(&w, 0, &expensive, 100_000, 256, &loads),
            1
        );
        assert_eq!(
            prefill_work_rank(&w, 0, &expensive, 100_000, 256, &loads),
            0
        );
        let cheap = [(0, 351), (1, 338)]; // extra 3328 <= 25% * 10144 + 1024
        assert_eq!(prefill_work_rank(&w, 0, &cheap, 100_000, 256, &loads), 1);
        assert_eq!(
            budgeted_prefill_work_rank(&w, 0, &cheap, 100_000, 256, &loads, 1000, None),
            0
        );
        assert_eq!(
            budgeted_prefill_work_rank(&w, 0, &cheap, 100_000, 256, &loads, 1000, Some(1)),
            0
        );
        let warm = [(0, 351), (1, 351)];
        assert_eq!(
            budgeted_prefill_work_rank(&w, 0, &warm, 100_000, 256, &loads, 0, None),
            1
        );
    }

    #[test]
    fn prefill_residence_yields_to_material_queue_improvement() {
        let w = worker(2);
        let at = Instant::now();
        let mut loads = [
            RankLoad {
                prefill_tokens_per_second: None,
                rank: 0,
                num_tokens: 0,
                num_reqs: 1,
                pending_prefill_tokens: Some(0),
                at,
            },
            RankLoad {
                prefill_tokens_per_second: None,
                rank: 1,
                num_tokens: 0,
                num_reqs: 1,
                pending_prefill_tokens: Some(500),
                at,
            },
        ];
        let warm = [(0, 351), (1, 351)];
        assert_eq!(
            budgeted_prefill_work_rank(&w, 0, &warm, 100_000, 256, &loads, 0, Some(1)),
            1
        );
        loads[1].pending_prefill_tokens = Some(2000);
        assert_eq!(
            budgeted_prefill_work_rank(&w, 0, &warm, 100_000, 256, &loads, 0, Some(1)),
            0
        );
    }

    #[test]
    fn sustained_overflow_prices_time_shared_work_and_preserves_total_budget() {
        let w = worker(3);
        let at = Instant::now();
        let cache = [(0, 390)]; // 99,840 cached; cold copy exceeds old per-request cap.
        let mut loads: Vec<_> = (0..3)
            .map(|rank| RankLoad {
                rank,
                num_tokens: 0,
                num_reqs: if rank == 0 { 64 } else { 1 },
                pending_prefill_tokens: Some(if rank == 0 { 1_300_000 } else { 0 }),
                prefill_tokens_per_second: Some(20_000),
                at,
            })
            .collect();
        let choose = |loads: &[RankLoad], budget, congested: &[u32]| {
            prefill_overflow_rank(
                &w,
                0,
                0,
                &cache,
                100_000,
                256,
                loads,
                budget,
                congested,
                |_| true,
            )
        };
        assert!(choose(&loads, 200_000, &[]).is_none());
        assert!(choose(&loads, 99_000, &[0]).is_none());
        let decision = choose(&loads, 100_000, &[0]).unwrap();
        assert_eq!(decision.rank, 1);
        assert_eq!(decision.extra_tokens, 99_840);
        assert_eq!(decision.source_queue_ms, 65_000);
        assert!(decision.predicted_finish_ms < 10_000);
        // Missing/reset throughput cannot make a destination appear infinitely fast.
        loads[2].prefill_tokens_per_second = None;
        assert!(choose(&loads, 200_000, &[0]).is_none());
        loads[2].prefill_tokens_per_second = Some(20_000);
        loads[1].prefill_tokens_per_second = Some(1000);
        loads[2].prefill_tokens_per_second = Some(1000);
        assert!(choose(&loads, 200_000, &[0]).is_none());
        for l in &mut loads {
            l.prefill_tokens_per_second = Some(20_000);
        }
        // A stamped dispatch closes the destination until the next engine reading.
        let a = w.dp_rank_timestamped_guard(1);
        let b = w.dp_rank_timestamped_guard(2);
        assert!(choose(&loads, 200_000, &[0]).is_none());
        drop((a, b));
        // Cooldown and the absolute copy bound remain admission constraints.
        assert!(prefill_overflow_rank(
            &w,
            0,
            0,
            &cache,
            100_000,
            256,
            &loads,
            200_000,
            &[0],
            |_| false
        )
        .is_none());
        assert!(prefill_overflow_rank(
            &w,
            0,
            0,
            &[(0, 600)],
            160_000,
            256,
            &loads,
            200_000,
            &[0],
            |_| true
        )
        .is_none());
    }

    const LEAST: DpRankPolicy = DpRankPolicy::LeastInFlight;
    const TOKENS: DpRankPolicy = DpRankPolicy::LeastTokens;

    #[test]
    fn prefill_work_balances_queue_against_cache_and_unreported_dispatches() {
        let w = worker(2);
        let at = Instant::now() - Duration::from_millis(1);
        let reading = |rank, pending| RankLoad {
            prefill_tokens_per_second: None,
            rank,
            num_tokens: 0,
            num_reqs: 0,
            pending_prefill_tokens: pending,
            at,
        };
        let cache = [(0, 39), (1, 36)]; // A slightly less warm replica can take work.
        let mut loads = [reading(0, Some(500)), reading(1, Some(0))];
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
                prefill_tokens_per_second: None,
                rank: 0,
                num_tokens: 0,
                num_reqs: 0,
                pending_prefill_tokens: Some(10500),
                at,
            },
            RankLoad {
                prefill_tokens_per_second: None,
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
        assert_eq!(prefill_work_rank(&w, 0, &cache, 10000, 256, &loads), 0);
        let cache = [(0, usize::MAX), (1, usize::MAX)];
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
                prefill_tokens_per_second: None,
                rank: 0,
                num_tokens: 0,
                num_reqs: 10,
                pending_prefill_tokens: Some(1_000_000),
                at,
            },
            RankLoad {
                prefill_tokens_per_second: None,
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
        // Idle cannot bypass the extra-compute budget in an EP group.
        loads[1].num_reqs = 0;
        loads[1].pending_prefill_tokens = Some(0);
        assert_eq!(
            prefill_work_rank(&w, 0, &[(0, 390)], 100_000, 256, &loads),
            0
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
                prefill_tokens_per_second: None,
                rank: 0,
                num_tokens: 0,
                num_reqs: 20,
                pending_prefill_tokens: Some(1_000_000),
                at,
            },
            RankLoad {
                prefill_tokens_per_second: None,
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
            prefill_tokens_per_second: None,
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
