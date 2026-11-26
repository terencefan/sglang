# Bounded MXFP4 Prefill workspace

The 2026-11-26 cold-prefix200-client run failed in SM90 FlashInfer/CUTLASS MoE: a15.03GiB scratch allocation encountered2.84GiB free; PyTorch also reported16.14GiB reserved but unallocated. The completed8-client smoke had not established sustained-load safety. Allocation failure is established; fragmentation is not independently proven.

Enable `SGLANG_FLASHINFER_MXFP4_PREFILL_PREALLOC_WORKSPACE=1` to reserve one target-runner-owned workspace after weights/transforms and before KV sizing. All compatible MXFP4 layers query the deployed FlashInfer size API with their exact geometry/placement/finalize flags; allocate the maximum once. The row bound is the registered per-rank prefill capacity multiplied by Attention DP size, not the8192-token autotuning cap. For the current8192×DP8 configuration this covers65536 rows.

The quantization adapter passes that buffer to FlashInfer's existing `workspace_buffer` interface. The default remains off; Decode and draft runners retain their existing calls. Same-stream layers reuse the buffer in order; cross-stream access records completion on the previous stream and waits on the next under a host enqueue lock. Capture is explicitly rejected. Oversized rows or changed dtype/device/hidden dimension fail instead of growing scratch or silently returning to dynamic allocation.

Scope: eager Prefill, PP1/CP1, current SM90 BF16-activation/W4A16 MXFP4 backend. Humming W4A8, SM120, MegaMoE and prefill CUDA Graph require separate integration/validation. No global tensor cache, no per-layer15GiB allocation, no model mathematics or sampling changes.

Preallocation makes the cost visible to KV sizing and avoids repeated large allocations; it does **not** remove scratch capacity or prove the200-client workload fits. KV and host HiCache capacity may decrease at unchanged static fraction. Startup may fail early if weights+scratch leave insufficient KV capacity. Do not infer SM Active, throughput or one-hour reliability improvements from CPU tests.

## Validation

CPU contracts:

```bash
python3 -m unittest discover -s test/registered/unit/layers/moe -p test_mxfp4_prefill_workspace.py -v
```

Tests cover bounded allocation, multi-layer sharing, geometry-specific size planning, actual adapter buffer forwarding, unchanged opt-out calls, dtype/device guards, cross-stream event ordering, exception safety and explicit graph/unsupported-path rejection.

`validate_gpu.py` requires one isolated, otherwise unused H200. It uses the deployed FlashInfer0.7.4 and synthetic384-expert/EP8 geometry on a single expert shard, compares exact outputs against the dynamic-workspace call for128/8192/32768/65536 rows, exercises two streams, records peak additional allocated bytes and requires a stable scratch pointer. `run_rjob.py` adds power/utilization/memory samples and1200-second execution timeout. `_SUCCESS` is the application acceptance marker; scheduling alone is not a pass.

The owning `submit_rjob.py` defaults to dry run, explicitly selects `ailab-datatools/data_tools_gpu_pool`,1GPU/8CPU/32GiB, and uses the exact production cu130 image to reproduce this failure (not a runtime upgrade). The initial8CPU submission `ds-moe-ws-1126-212546-50702109` was stopped after a replacement was accepted. Active validation: `ds-moe-ws2-1126-213232-56384722`,1GPU/2CPU/32GiB, in the same public pool; artifacts under `benchmark/experiments/mxfp4-workspace-cpu2-20261126T213232`. The CPU override is for this synthetic kernel driver after CPU-capacity scheduling failures; host JIT/OMP concurrency is limited to2. No private-pool online allocation is used.

After single-GPU validation, acceptance still requires an isolated8-rank startup to verify KV sizing, then staged long-input PD validation and a fresh uninterrupted200-client one-hour window. Do not apply the opt-in flag to the current image: it does not contain this patch. Rollback is the previous image/profile or disabling the flag in a verified patched image.
