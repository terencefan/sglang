"""Maintenance-only synthetic EP-shard tuning; not a serving throughput result."""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--cap", type=int, choices=[8192, 32768, 65536], required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--no-fused-finalize", action="store_true")
    a = p.parse_args()
    manifest = json.loads((a.root / "devices.json").read_text())
    assert a.gpu in manifest["gpu_uuids"] and len(manifest["gpu_uuids"]) == 8
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + a.gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    assert not apps, "GPU still has compute processes"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
    from flashinfer.fused_moe import interleave_moe_scales_for_sm90_mixed_gemm as isc
    from flashinfer.fused_moe import interleave_moe_weights_for_sm90_mixed_gemm as iw
    from flashinfer.fused_moe.core import ActivationType

    assert importlib.metadata.version("flashinfer-python") == "0.7.4"
    out = a.root / "results" / a.gpu / str(a.cap)
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.mem_get_info()[0] > 80 * 1024**3, (
        "requires80GiBfree before synthetic weights/tuning"
    )
    torch.manual_seed(20261126)
    e, h, intermediate, topk, ep = 48, 5120, 2304, 6, 8
    device = torch.device("cuda:0")
    w13 = iw(
        torch.randint(
            0, 256, (e, 2 * intermediate, h // 2), dtype=torch.uint8, device=device
        ),
        "fp4",
    )
    w2 = iw(
        torch.randint(
            0, 256, (e, h, intermediate // 2), dtype=torch.uint8, device=device
        ),
        "fp4",
    )
    scales = [
        isc(
            torch.full(
                (e, 2 * intermediate, h // 32), 120, dtype=torch.uint8, device=device
            ),
            group_size=32,
        ).view(torch.int32),
        isc(
            torch.full(
                (e, h, intermediate // 32), 120, dtype=torch.uint8, device=device
            ),
            group_size=32,
        ).view(torch.int32),
    ]
    kwargs = dict(
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=scales,
        tp_size=1,
        tp_rank=0,
        ep_size=ep,
        ep_rank=0,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        use_fused_finalize=not a.no_fused_finalize,
        tune_max_num_tokens=a.cap,
        swiglu_limit=torch.full((e,), 10.0, dtype=torch.float32, device=device),
    )
    size = cutlass_fused_moe_workspace_size(
        65536,
        h,
        intermediate,
        e * ep,
        topk,
        x_dtype=torch.bfloat16,
        weight_dtype=w13.dtype,
        output_dtype=torch.bfloat16,
        tp_size=1,
        ep_size=ep,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        use_fused_finalize=not a.no_fused_finalize,
        device=device,
    )
    scratch = torch.empty(size, dtype=torch.uint8, device=device)
    records = []
    for distribution in ["uniform", "skew"]:
        for n in [128, 8192, 32768, 65536]:
            torch.manual_seed(20261126 + n)
            x = torch.randn(n, h, dtype=torch.bfloat16, device=device) * 0.01
            ids = (
                (
                    torch.arange(n, device=device)[:, None] * 7
                    + torch.arange(topk, device=device)[None, :]
                )
                % (e * ep)
            ).to(torch.int32)
            if distribution == "skew":
                ids[: int(n * 0.75)] = torch.arange(
                    topk, device=device, dtype=torch.int32
                )
            probabilities = torch.full(
                (n, topk), 1 / topk, dtype=torch.float32, device=device
            )
            reference = torch.empty_like(x)
            actual = torch.empty_like(x)
            call = dict(
                input=x,
                token_selected_experts=ids,
                token_final_scales=probabilities,
                **kwargs,
            )
            baseline = dict(call, tune_max_num_tokens=8192)
            AutoTuner.get().clear_cache()
            with autotune(tune_mode=False):
                cutlass_fused_moe(
                    **baseline, output=reference, workspace_buffer=scratch
                )
            cache = str(
                (a.cache_root or a.root)
                / "results"
                / a.gpu
                / str(a.cap)
                / "tactics.json"
            )
            AutoTuner.get().clear_cache()
            start = time.time()
            with autotune(tune_mode=a.cache_root is None, cache=cache):
                cutlass_fused_moe(**call, output=actual, workspace_buffer=scratch)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(actual).all().item())
            exact = finite and torch.equal(actual, reference)
            with autotune(tune_mode=False, cache=cache):
                for _ in range(5):
                    cutlass_fused_moe(**call, output=actual, workspace_buffer=scratch)
                torch.cuda.synchronize()
                measure = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    first, last = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    first.record()
                    for _ in range(100):
                        cutlass_fused_moe(
                            **call, output=actual, workspace_buffer=scratch
                        )
                    last.record()
                    last.synchronize()
                    measure.append(first.elapsed_time(last) / 100)
            baseline_times = []
            tuned_times = []
            if a.cache_root is not None:
                for repeat in range(15):
                    for candidate in (
                        [False, True] if repeat % 2 == 0 else [True, False]
                    ):
                        AutoTuner.get().clear_cache()
                        context = autotune(
                            tune_mode=False, cache=cache if candidate else None
                        )
                        with context:
                            args = call if candidate else baseline
                            for _ in range(3):
                                cutlass_fused_moe(
                                    **args,
                                    output=actual if candidate else reference,
                                    workspace_buffer=scratch,
                                )
                            first = torch.cuda.Event(enable_timing=True)
                            last = torch.cuda.Event(enable_timing=True)
                            first.record()
                            for _ in range(10):
                                cutlass_fused_moe(
                                    **args,
                                    output=actual if candidate else reference,
                                    workspace_buffer=scratch,
                                )
                            last.record()
                            last.synchronize()
                            (tuned_times if candidate else baseline_times).append(
                                first.elapsed_time(last) / 10
                            )
                exact = torch.equal(actual, reference)
            rec = dict(
                default_median_ms=statistics.median(baseline_times)
                if baseline_times
                else None,
                paired_tuned_median_ms=statistics.median(tuned_times)
                if tuned_times
                else None,
                speedup=statistics.median(
                    [b / t for b, t in zip(baseline_times, tuned_times)]
                )
                if baseline_times
                else None,
                max_abs_error=(actual.float() - reference.float()).abs().max().item(),
                tolerance_pass=bool(
                    torch.allclose(actual, reference, rtol=0.01, atol=0.01)
                ),
                distribution=distribution,
                rows=n,
                exact_equal=exact,
                finite=finite,
                tuning_started_at=start,
                mean_call_ms_per_round=measure,
                median_ms=statistics.median(measure),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                free_bytes=torch.cuda.mem_get_info()[0],
            )
            records.append(rec)
            (out / "results.json").write_text(
                json.dumps(
                    dict(
                        gpu=a.gpu,
                        cap=a.cap,
                        fused_finalize=not a.no_fused_finalize,
                        workspace_bytes=size,
                        scope="synthetic uniform/skew EP shard; NOT real-request or distributed acceptance",
                        records=records,
                    ),
                    indent=2,
                )
            )
            print(json.dumps(rec), flush=True)
            del x, ids, probabilities, reference, actual, call, baseline
    (out / "_COMPLETE").write_text(str(time.time()))


if __name__ == "__main__":
    main()
