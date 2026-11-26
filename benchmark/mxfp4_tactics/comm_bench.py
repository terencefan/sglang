"""Eight-GPU PyNCCL ReduceScatter equivalence/timing and custom-AR eligibility."""

import argparse
import datetime
import json
import os
import statistics
from pathlib import Path


def main():
    import torch
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--with-custom", action="store_true")
    p.add_argument("--custom-runtime", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    assert dist.get_world_size() == 8
    comm = PyNcclCommunicator(dist.group.WORLD, device=device)
    assert comm.available
    car = None
    if a.with_custom:
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(
            world_group=SimpleNamespace(local_rank=rank, barrier=lambda: dist.barrier())
        ):
            cls = CustomAllreduce
            if a.custom_runtime:
                from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                    dispatch_custom_allreduce,
                )

                cls = dispatch_custom_allreduce(dist.group.WORLD, device=device)
            car = cls(dist.group.WORLD, device=device)
    records = []

    def measure(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        for _ in range(31):
            first, last = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            first.record()
            fn()
            last.record()
            last.synchronize()
            times.append(first.elapsed_time(last))
        return statistics.median(times)

    for n in [8, 16, 32, 64, 128, 192, 224, 240, 256, 512, 1024, 4096, 8192]:
        x = torch.full(
            (n * 8, 5120), float(rank + 1), dtype=torch.bfloat16, device=device
        )
        baseline = torch.empty((n, 5120), dtype=x.dtype, device=device)
        native = torch.empty_like(baseline)
        old = lambda baseline=baseline, x=x, n=n: comm.reduce_scatter(
            baseline, x, sizes=[n] * 8
        )
        new = lambda native=native, x=x: comm.reduce_scatter(native, x, sizes=None)
        with comm.change_state(enable=True):
            old()
            new()
            torch.cuda.synchronize()
            assert torch.equal(baseline, native) and bool((native == 36).all())
            ms_old = measure(old)
            ms_new = measure(new)
            dist.barrier()
            torch.cuda.synchronize()
            with torch.cuda.graph(g1 := torch.cuda.CUDAGraph()):
                old()
            with torch.cuda.graph(g2 := torch.cuda.CUDAGraph()):
                new()
            g1.replay()
            g2.replay()
            torch.cuda.synchronize()
            assert torch.equal(baseline, native)
            graph_old = measure(g1.replay)
            graph_new = measure(g2.replay)
        random_checks = []
        for seed in [20261127, 20261128, 20261129]:
            torch.manual_seed(seed + rank)
            x.normal_()
            with comm.change_state(enable=True):
                old()
                new()
                torch.cuda.synchronize()
                eager_equal = torch.equal(baseline, native)
                max_abs = (baseline.float() - native.float()).abs().max().item()
                g1.replay()
                g2.replay()
                torch.cuda.synchronize()
                graph_equal = torch.equal(baseline, native)
            random_checks.append(
                dict(
                    seed=seed,
                    eager_equal=eager_equal,
                    graph_equal=graph_equal,
                    max_abs=max_abs,
                )
            )
        x.fill_(float(rank + 1))
        # Existing custom AR's own size/topology gate decides eligibility.
        ar_input = x[:n]
        custom_supported = (
            car is not None and not car.disabled and car.should_custom_ar(ar_input)
        )
        row = {
            "random_checks": random_checks,
            "local_rows": n,
            "dtype": "bf16",
            "equal_exact": True,
            "baseline_ms": ms_old,
            "native_ms": ms_new,
            "graph_baseline_ms": graph_old,
            "graph_native_ms": graph_new,
            "custom_class": type(car).__name__ if car is not None else None,
            "custom_ar_eligible": bool(custom_supported) if car is not None else None,
            "allreduce_bytes": ar_input.numel() * ar_input.element_size(),
        }
        if custom_supported:
            out = car.custom_all_reduce(ar_input)
            torch.cuda.synchronize()
            assert out is not None and bool((out == 36).all())
            with comm.change_state(enable=True):
                other = torch.empty_like(ar_input)
                row["nccl_allreduce_ms"] = measure(
                    lambda: comm.outplace_all_reduce(ar_input, other)
                )
            row["custom_allreduce_ms"] = measure(
                lambda: car.custom_all_reduce(ar_input)
            )
            with car.capture():
                with torch.cuda.graph(ga := torch.cuda.CUDAGraph()):
                    custom_out = car.custom_all_reduce(ar_input)
            with comm.change_state(enable=True):
                with torch.cuda.graph(gn := torch.cuda.CUDAGraph()):
                    comm.outplace_all_reduce(ar_input, other)
            ga.replay()
            gn.replay()
            torch.cuda.synchronize()
            assert custom_out is not None and torch.equal(custom_out, other)
            row["custom_graph_ms"] = measure(ga.replay)
            row["nccl_ar_graph_ms"] = measure(gn.replay)
            torch.manual_seed(20261127 + rank)
            ar_input.normal_()
            ga.replay()
            gn.replay()
            torch.cuda.synchronize()
            row["custom_random_exact"] = torch.equal(custom_out, other)
            row["custom_random_max_abs"] = (
                (custom_out.float() - other.float()).abs().max().item()
            )
            row["custom_random_tolerance"] = bool(
                torch.allclose(custom_out, other, rtol=0.01, atol=0.01)
            )
            del ga, gn, custom_out

        records.append(row)
        (a.out / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
        print(json.dumps({"rank": rank, **row}), flush=True)
        del g1, g2, x, baseline, native
    # Unequal and empty shards retain the variable-size implementation.
    for sizes in [
        [0] * 8,
        [0, 6, 12, 0, 18, 24, 30, 36],
        [180, 192, 198, 210, 216, 222, 228, 234],
    ]:
        x = torch.full(
            (sum(sizes), 5120), float(rank + 1), dtype=torch.bfloat16, device=device
        )
        out = torch.empty((sizes[rank], 5120), dtype=x.dtype, device=device)
        with comm.change_state(enable=True):
            comm.reduce_scatter(out, x, sizes=sizes)
        torch.cuda.synchronize()
        assert bool((out == 36).all())
    if car is not None and not car.disabled:
        inputs = [
            torch.full(
                (192, 5120),
                float(rank + 1) * scale,
                dtype=torch.bfloat16,
                device=device,
            )
            for scale in [1, 10]
        ]
        graphs = []
        outputs = []
        for data in inputs:
            with car.capture():
                with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
                    result = car.custom_all_reduce(data)
            graphs.append(graph)
            outputs.append(result)
        dist.barrier()
        graphs[rank % 2].replay()
        torch.cuda.synchronize()
        selected = outputs[rank % 2]
        mixed = dict(
            rank=rank,
            custom_class=type(car).__name__,
            graph_choice=rank % 2,
            expected=216.0,
            exact=bool((selected == 216).all()),
            min=float(selected.min().item()),
            max=float(selected.max().item()),
        )
        (a.out / f"mixed-graph-rank{rank}.json").write_text(json.dumps(mixed, indent=2))
        print(json.dumps({"mixed_graph_contract": mixed}), flush=True)
    dist.barrier()
    if car is not None:
        car.close()
    dist.barrier()
    dist.destroy_process_group()
    (a.out / f"_COMPLETE_rank{rank}").write_text("ok")


if __name__ == "__main__":
    main()
