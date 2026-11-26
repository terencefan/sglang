"""Isolated H200 shortlist benchmark. Defaults to CPU-only description.
Never loads a serving model, changes a cache, or runs on a GPU with compute PIDs.
Synthetic per-EP-shard fixtures qualify kernel candidates, not model quality or E2E gain.
"""

import argparse
import itertools
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-on-idle-gpu", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--rows", type=int, nargs="+", default=[1536, 1792, 1920, 2048])
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--fused-finalize", action="store_true")
    a = p.parse_args()
    plan = {
        "H": 5120,
        "I": 2304,
        "local_experts": 48,
        "global_experts": 384,
        "topk": 6,
        "ep_size": 8,
        "ep_rank": 0,
        "rows": a.rows,
        "gemm1_tactics": [64, 73],
        "gemm2_tactics": [180, 189],
        "reference": "snapshot cache: 1536/1792 -> [64,180]; 2048 -> [73,189]",
        "distributions": ["uniform", "local_hot"],
        "l2_policy": ["hot", "flush_128MiB"],
        "gamma": 5,
        "status": "Prepared, not GPU validated",
        "limits": "Synthetic routing; baseline cache does not prove resident graph tactic IDs",
    }
    if not a.run_on_idle_gpu:
        print(json.dumps(plan, indent=2))
        return
    if not a.out:
        p.error("--out is required when running")
    if a.repeats < 3 or any(n < 1 or n > 4096 for n in a.rows):
        p.error("Invalid rows/repeats")
    # Fail before torch import or CUDA allocation when there is existing GPU work.
    inventory = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if a.device is None:
        if len(inventory) != 1:
            raise SystemExit("Require one allocated GPU or an explicit GPU UUID")
        a.device = inventory[0].split(",", 1)[0].strip()
    if not a.device.startswith("GPU-"):
        raise SystemExit(
            "Use a GPU UUID; physical index could select an unallocated device"
        )
    proc = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + a.device,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if proc:
        raise SystemExit(
            "GPU has compute PIDs; refusing to share GPU with inference: " + proc
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = a.device
    import importlib.metadata

    import torch
    from flashinfer.fused_moe import (
        cutlass_fused_moe,
    )
    from flashinfer.fused_moe import interleave_moe_scales_for_sm90_mixed_gemm as isc
    from flashinfer.fused_moe import interleave_moe_weights_for_sm90_mixed_gemm as iw
    from flashinfer.fused_moe.core import ActivationType

    if importlib.metadata.version("flashinfer-python") != "0.7.4":
        raise RuntimeError(
            "Reinspect installed API/tactic IDs for this FlashInfer version"
        )
    assert "H200" in torch.cuda.get_device_name(0)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "plan.json").write_text(json.dumps(plan, indent=2))
    torch.manual_seed(20261126)
    H, I, E, K = 5120, 2304, 48, 6
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.full((E, 2 * I, H // 32), 122, device="cuda", dtype=torch.uint8)
    s2 = torch.full((E, H, I // 32), 122, device="cuda", dtype=torch.uint8)
    w1, w2 = iw(w1, "fp4"), iw(w2, "fp4")
    s1, s2 = isc(s1, group_size=32), isc(s2, group_size=32)
    scales = [s1.view(torch.int32), s2.view(torch.int32)]
    clamp = torch.full((E,), 10.0, device="cuda", dtype=torch.float32)
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for n, distribution in itertools.product(a.rows, plan["distributions"]):
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = torch.rand(n, 384, device="cuda")
        if distribution == "local_hot":
            ids[:, :12] *= 3
        ids = ids.topk(K, dim=-1).indices.to(torch.int32)
        weights = torch.softmax(torch.randn(n, K, device="cuda"), dim=-1)
        out = torch.empty_like(x)

        def run(pair, output=None, *, x=x, ids=ids, weights=weights, out=out, n=n):
            return cutlass_fused_moe(
                input=x,
                token_selected_experts=ids,
                token_final_scales=weights,
                fc1_expert_weights=w1,
                fc2_expert_weights=w2,
                output_dtype=torch.bfloat16,
                quant_scales=scales,
                swiglu_limit=clamp,
                tp_size=1,
                tp_rank=0,
                ep_size=8,
                ep_rank=0,
                use_w4_group_scaling=True,
                use_mxfp8_act_scaling=False,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1 << (n - 1).bit_length(),
                use_fused_finalize=a.fused_finalize,
                profile_ids=list(pair),
                output=out if output is None else output,
            )

        reference_pair = (64, 180) if n <= 1792 else (73, 189)
        run(reference_pair)
        torch.cuda.synchronize()
        reference = out.clone()
        reference_output = torch.empty_like(out)
        with torch.cuda.graph(reference_graph := torch.cuda.CUDAGraph()):
            run(reference_pair, reference_output)
        for pair in itertools.product(plan["gemm1_tactics"], plan["gemm2_tactics"]):
            row = {
                "rows": n,
                "distribution": distribution,
                "tactics": pair,
                "reference_tactics": reference_pair,
            }
            try:
                for _ in range(3):
                    run(pair)
                torch.cuda.synchronize()
                assert torch.isfinite(out).all()
                delta = out.float() - reference.float()
                denom = reference.float().square().mean().sqrt().clamp_min(1e-12)
                row.update(
                    relative_rmse=float(delta.square().mean().sqrt() / denom),
                    max_abs=float(delta.abs().max()),
                    bitwise_equal=bool(torch.equal(out, reference)),
                )
                # Compatibility screen only; model-level fixed-token and end-to-end gates still required.
                torch.testing.assert_close(out, reference, rtol=0.01, atol=0.01)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(pair)
                times = {}
                for cold in [False, True]:
                    samples = []
                    baseline_samples = []
                    ratios = []
                    for repeat in range(a.repeats):
                        pair_times = {}
                        for name, g in (
                            [("baseline", reference_graph), ("candidate", graph)]
                            if repeat % 2 == 0
                            else [("candidate", graph), ("baseline", reference_graph)]
                        ):
                            if cold:
                                flush.fill_(1)
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            g.replay()
                            end.record()
                            end.synchronize()
                            pair_times[name] = start.elapsed_time(end)
                        samples.append(pair_times["candidate"])
                        baseline_samples.append(pair_times["baseline"])
                        ratios.append(pair_times["baseline"] / pair_times["candidate"])
                    times["flush_128MiB" if cold else "hot"] = {
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                        "baseline_samples_ms": baseline_samples,
                        "paired_speedup_median": statistics.median(ratios),
                    }
                torch.testing.assert_close(out, reference, rtol=0.01, atol=0.01)
                row.update(passed=True, timing=times)
            except Exception as e:  # noqa: BLE001 - record rejected GPU candidates
                row.update(passed=False, error=repr(e))
            results.append(row)
            (a.out / "results.json").write_text(json.dumps(results, indent=2))
        del x, ids, weights, out, reference, reference_output, reference_graph
    (a.out / "finished.json").write_text(
        json.dumps(
            {
                "scope": "isolated synthetic shard; no serving change",
                "cases": len(results),
                "passed": all(r["passed"] for r in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
