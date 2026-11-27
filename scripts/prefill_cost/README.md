# H200 Prefill cost calibration

Status: **shadow collection only; not a completed adaptive scheduler**.
The base is the deployed Mooncake cleanup-race fix (29b12b1a92), rather than an
older optimization worktree. New behavior is off by default. No deployment,
GPU benchmark, throughput gain, DP agreement, or Router promotion is claimed.

## Implemented

- Nonnegative CPU batch-cost contract, including prefix, replay and batch size.
- Aligned time-budget search capped at 8192, without modifying KV admission.
- Scheduler forward wrapper with opt-in CUDA events; polls `query()` and never
  synchronizes the device. At most 64 outstanding samples, default every 32
  Prefill rounds. Request identities and prompt contents are not logged.
- Execution-signed model loading; incompatible artifacts leave shadow prediction
  disabled while preserving timing collection.
- CPU-only nonnegative fitting and chronological 70/30 holdout; compares the
  same holdout against a token/rate baseline. Cache-load-pending samples are
  excluded. Calibration never automatically authorizes serving control.

## Run after arranging a GPU validation window

Set `SGLANG_PREFILL_TIME_BUDGET_MODE=shadow`; optionally set
`SGLANG_PREFILL_COST_SAMPLE_INTERVAL=32` and
`SGLANG_PREFILL_COST_MODEL=/path/to/model.json`.
The engine prints `PREFILL_COST_SIGNATURE` and bounded pending events produce
`PREFILL_COST_SAMPLE` JSON log records. Off mode constructs no events and never
walks the requests. Logs must be retained/exported with bounded rotation by the
existing serving log collector; the sampler does not write unbounded files.

Fit a captured chronological log on CPU (Python + NumPy):

```
python scripts/prefill_cost/calibrate.py samples.log \
  --signature EXECUTION_HASH --output model.json
```

`prediction_gate` tests P95 relative error <=20% and >=20% improvement versus
baseline. `serving_gate` remains false: coverage, GPU overhead and end-to-end
quality/performance acceptance are separate evidence.

### Measurement limitations

`gpu_stream_ms` measures the scheduler's forward-stream span, including stream
waits and host submission gaps. It is not isolated attention/MoE kernel time,
nor complete request TTFT. Do not train an isolated compute or reload model
from this span. Encoder SWA replay uses the existing 128-token replay rule.
DP sync wait is separate CPU metadata where available (zero is not evidence of
no wait). Pending final samples may remain uncollected until another scheduler
step; there is deliberately no shutdown CUDA synchronization.

## Remaining implementation before enforce

1. Isolate attention, MoE, H2D, scheduling, sync and PD handoff costs on H200;
   add aggregate DP rows/padding, latency histograms and bounded diagnostic
   sampling; measure shadow overhead <=1%.
2. Introduce a whole-batch time-budget gate in PrefillAdder after cache match,
   including continuation/ignore-EOS/replay paths, page and workspace bounds,
   minimum progress and age protection.
3. Carry readiness, signature and budget epoch through the existing MLP metadata
   exchange. All ranks, including idle ranks, must agree before enforce; no
   per-candidate collective. Preserve the old tensor protocol until its tests
   and mixed-version deployment procedure are ready.
4. Extend the existing load feed and Rust Router with optional signed cost
   summaries, dispatch reconciliation, freshness fallback and 10% hysteresis;
   retain the current cache-affinity/overflow candidate constraints.
5. Paired fixed4096/fixed8192, engine-only, Router-only and combined replays;
   cold/warm/mixed inputs, concurrency64/200, one-hour final run and drain.
   Require >=5% cold throughput improvement, <=5% tail regressions, no new
   errors or correctness failures. Keep memory, chunk cap and Decode fixed.

An `enforce` environment value is rejected explicitly, rather than silently
pretending to execute an unimplemented or unvalidated control policy.

## Local verification

```
python3 -m pytest -q \
  test/registered/unit/managers/scheduler_components/test_prefill_cost_model.py \
  test/registered/unit/managers/scheduler_components/test_prefill_cost_runtime.py
```

CPU event doubles verify nonblocking behavior only; they are not GPU acceptance.
