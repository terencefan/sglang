# H200 Prefill cost calibration

Status: **candidate implementation; GPU acceptance pending**.
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

## Acceptance gates still open

- Actual H200 shadow overhead and forward-stream prediction accuracy, including
  coverage of four context cohorts and DP aggregate shapes. Stream spans are
  never presented as isolated kernel timing.
- Model calibration and isolated A/B/A replay for fixed4096, fixed8192, engine
  only, Router only and combined control. Keep KV/workspace capacity fixed.
- Cold, warm and mixed replay; the first Router candidate deliberately falls
  back for unknown warm-prefix reload costs. No L2 bandwidth is invented.
- Quality and handoff checks, failure-free one-hour concurrency200 run plus
  drain, GPU-memory/power/SM telemetry and a documented promote/reject decision.
  Cold throughput gain must reach 5%; short/long P95 regressions must stay <=5%.

The candidate implements PrefillAdder gates for new requests, continuations
and ignore-EOS, progress exceptions, DP model/time votes in the existing MLP
exchange, and optional backward-compatible cold-cost load frames. The CPU
probe asserts candidate import origins and runs with GPU visibility disabled;
its Gloo exchange test checks the added metadata columns over 2–8 ranks.

`enforce` is an experimental candidate mode requiring a same-signature model
with a passed prediction gate, a positive `SGLANG_PREFILL_TIME_BUDGET_MS`, and
DP-wide agreement through the existing MLP metadata exchange. Unready or
incompatible ranks disable control. It has not been promoted to production.
The new cold-only Router quote mode is independently controlled with
`SGLANG_ROUTER_PREFILL_COST_MODE=off|shadow|enforce`; unknown cache reloads,
missing quotes, mixed models and stale load retain the original route.

For fixed-cap calibration, `SGLANG_PREFILL_FIXED_CHUNK_ROWS=4096|8192` limits
scheduler admission after workspace and KV sizing. Keep the launch-time global
chunk at 65536, so memory capacity remains identical between comparisons.
No value (or zero) preserves the original scheduler cap.

## Local verification

```
python3 -m pytest -q \
  test/registered/unit/managers/scheduler_components/test_prefill_cost_model.py \
  test/registered/unit/managers/scheduler_components/test_prefill_cost_runtime.py
```

CPU event doubles verify nonblocking behavior only; they are not GPU acceptance.

## In-process acceptance profiles

`SGLANG_PREFILL_COST_CONTROL` may point to a task-owned local JSON file. It is
polled at most four times per second; replacing it atomically with an increasing
`generation` changes only mode, fixed_rows, target_ms and model_path. Models
must match the execution signature and pass the holdout gate for enforce. DP
control remains disabled until every rank votes for the same model. Invalid
profiles disable control; they do not terminate serving. Record all eight
`PREFILL_COST_CONTROL` receipts and finish warm-up before measuring a profile.

Example shadow profile: `{"generation":1,"mode":"shadow","fixed_rows":8192}`.
Example candidate: `{"generation":2,"mode":"enforce","target_ms":100,
"model_path":"/task/model.json"}`. The target is illustrative: use measured
fixed4096 P75/P90, never this example as an H200 calibration constant.
This allows A/B/A acceptance without repeatedly loading the model or changing
KV/workspace capacity. Initial installation still requires a controlled P
restart; production ingress must be drained before that operation.

## Optional bounded Router CPU work

`SGLANG_ROUTER_CPU_WORK_THREADS=4` opts into a dedicated pool for expensive
request parsing, template render, tokenization and body rebuild. The default
remains the existing inline blocking offload. This bounds active CPU scratch
allocations without capping HTTP or engine concurrency. Tokio retains the I/O
core handoff while callers wait for a pool slot. Nested CPU helpers execute on
the same pool thread. This is a separate test-ingress safeguard: compare both
Router arms with the same setting, verify real tokenizer parity and memory
under concurrency200, and do not attribute its effect to Prefill kernel speed.
No production memory or performance acceptance is implied by CPU unit tests.
