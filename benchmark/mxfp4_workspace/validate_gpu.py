"""Isolated one-H200 numerical/stream/allocation check; never use an online GPU."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workspace-module", type=Path, required=True)
    p.add_argument("--run-on-idle-gpu", action="store_true")
    a = p.parse_args()
    if not a.run_on_idle_gpu:
        p.error("explicit --run-on-idle-gpu required")
    devices = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(devices) != 1 or "H200" not in devices[0]:
        raise RuntimeError("requires exactly one allocated H200")
    gpu = devices[0].split(",")[0].strip()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if apps:
        raise RuntimeError("refusing a GPU with existing compute processes")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
    from flashinfer.fused_moe import interleave_moe_scales_for_sm90_mixed_gemm as isc
    from flashinfer.fused_moe import interleave_moe_weights_for_sm90_mixed_gemm as iw
    from flashinfer.fused_moe.core import ActivationType

    if importlib.metadata.version("flashinfer-python") != "0.7.4":
        raise RuntimeError(
            "revalidate API for FlashInfer version other than deployed0.7.4"
        )
    a.out.mkdir(exist_ok=False, parents=True)
    spec = importlib.util.spec_from_file_location(
        "workspace_candidate", a.workspace_module
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
        use_fused_finalize=True,
        tune_max_num_tokens=8192,
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
        use_fused_finalize=True,
        device=device,
    )
    arena = module.PrefillWorkspace(size, 65536, device)
    before_ptr = arena.buffer.data_ptr()
    results = []
    for n in [128, 8192, 32768, 65536]:
        x = torch.randn(n, h, dtype=torch.bfloat16, device=device) * 0.01
        ids = (
            (
                torch.arange(n, device=device)[:, None] * 7
                + torch.arange(topk, device=device)[None, :]
            )
            % (e * ep)
        ).to(torch.int32)
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
        cutlass_fused_moe(**call, output=reference)
        with arena.use(n) as scratch:
            cutlass_fused_moe(**call, output=actual, workspace_buffer=scratch)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # Warmup precedes measurement. Output tensors and scratch are already allocated.
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        tick = time.perf_counter()
        for _ in range(5):
            with arena.use(n) as scratch:
                cutlass_fused_moe(**call, output=actual, workspace_buffer=scratch)
        torch.cuda.synchronize()
        results.append(
            {
                "rows": n,
                "exact_equal": True,
                "seconds_5_calls": time.perf_counter() - tick,
                "peak_extra_bytes": torch.cuda.max_memory_allocated() - allocated,
            }
        )
        if arena.buffer.data_ptr() != before_ptr:
            raise RuntimeError("scratch pointer changed")
        if n == 128:
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            outputs = [torch.empty_like(x), torch.empty_like(x)]
            # Ensure inputs/reference are ready before submitting on side streams.
            torch.cuda.synchronize()
            for _ in range(4):
                for stream, out in zip(streams, outputs):
                    with torch.cuda.stream(stream), arena.use(n) as scratch:
                        cutlass_fused_moe(**call, output=out, workspace_buffer=scratch)
            torch.cuda.synchronize()
            for out in outputs:
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
        if n == 128:
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            graph_out = torch.empty_like(x)
            with arena.execution_scope():
                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        with arena.use(n) as scratch:
                            cutlass_fused_moe(
                                **call, output=graph_out, workspace_buffer=scratch
                            )
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=capture_stream):
                    with arena.use(n) as scratch:
                        cutlass_fused_moe(
                            **call, output=graph_out, workspace_buffer=scratch
                        )
                # Replay does not call arena.use: every caller must guard it.
                for stream in streams * 3:
                    with torch.cuda.stream(stream), arena.execution_scope():
                        graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(graph_out, reference, rtol=0, atol=0)
            assert arena.buffer.data_ptr() == before_ptr
        del x, ids, probabilities, reference, actual, call
    if max(x["peak_extra_bytes"] for x in results) > 128 * 1024**2:
        raise RuntimeError("unexpected runtime scratch growth; inspect allocator trace")
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "gpu": gpu,
                "workspace_bytes": size,
                "results": results,
                "two_stream_exact_equal": True,
                "graph_capture_and_cross_stream_replay_exact_equal": True,
                "limits": "synthetic EP shard on one GPU; not distributed serving or sustained-load acceptance",
            },
            indent=2,
        )
    )
    (a.out / "_SUCCESS").write_text(str(time.time()))
    print("WORKSPACE_GPU_VALIDATION_SUCCESS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
