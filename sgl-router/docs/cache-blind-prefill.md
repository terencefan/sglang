# Cache-blind Prefill routing

The selected DeepSeek profile uses `--policy power_of_two --dp-aware --dp-rank-policy prefill_queue_only --decode-dp-rank-policy least_in_flight`. Keep the existing model, tokenizer, authentication, timeouts, worker discovery and health configuration.

In this mode the Router does not subscribe to KV events, build a prefix tree, bootstrap peer cache snapshots, look up cache depth, or record per-rank prefix placements. Native engine load subscriptions remain active: queue-only decisions still need fresh eight-rank load snapshots. P/D bootstrap pairing and request cancellation are unchanged. Engine L1 radix cache and shared Mooncake L3 remain enabled.

Cache-aware source modules remain available for other profiles and rollback; they are not instantiated in this profile. Remove cache-specific bootstrap peers and indexer options from the deployment profile. Do not remove the tokenizer until the message-adapter and token-budget requirements are separately verified.

The isolated three-Router C64 comparison measured 18.83 requests/s for cache-blind vs 14.62 baseline and 13.80 rollback, with equal aggregate cached-token volume. The user accepts moderate tail-latency regression as a known risk. One separate cross-rank comparison lost 256 cached tokens; cross-rank losslessness is not guaranteed. No production restart has been performed by this change.
