"""Opt-in, bounded asynchronous Prefill timing collection.

Shadow estimates deliberately cannot change admission until H200 calibration
and DP-wide budget coordination have been validated.
"""

import hashlib
import json
import logging
import os
import time
from collections import deque
from functools import wraps
from pathlib import Path

from sglang.srt.managers.scheduler_components.prefill_cost_model import (
    PrefillCostModel,
    PrefillWork,
)

logger = logging.getLogger(__name__)


def execution_signature(scheduler):
    runner = scheduler.tp_worker.model_runner
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    source = Path(__file__).parents[1]
    hashes = {}
    for name in (
        "scheduler.py",
        "schedule_policy.py",
        "scheduler_components/prefill_cost_model.py",
    ):
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    import torch

    identity = {
        "model": str(runner.model_config.model_path),
        "config": runner.model_config.hf_config.to_dict(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tp": p.tp_size,
        "pp": p.pp_size,
        "cp": p.attn_cp_size,
        "dp": p.num_dp_ranks,
        "args": vars(scheduler.server_args),
        "source": hashes,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


class PrefillCostRuntime:
    def __init__(self, scheduler):
        self.mode = os.environ.get("SGLANG_PREFILL_TIME_BUDGET_MODE", "off")
        if self.mode not in ("off", "shadow"):
            raise ValueError(
                "Prefill time-budget enforce is not GPU-validated; use off or shadow"
            )
        self.pending = deque(maxlen=64)
        self.step = 0
        self.dropped = 0
        self.model = None
        self.signature = None
        self.rank = 0
        if self.mode == "off":
            return
        self.signature = execution_signature(scheduler)
        from sglang.srt.runtime_context import get_parallel

        self.rank = get_parallel().attn_dp_rank
        self.interval = int(os.environ.get("SGLANG_PREFILL_COST_SAMPLE_INTERVAL", "32"))
        if self.interval < 1:
            raise ValueError("sample interval must be positive")
        path = os.environ.get("SGLANG_PREFILL_COST_MODEL")
        if path:
            try:
                artifact = json.loads(Path(path).read_text())
                if artifact["signature"] != self.signature:
                    raise ValueError("execution signature mismatch")
                self.model = PrefillCostModel(
                    self.signature, tuple(artifact["coefficients"])
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Prefill cost model unavailable: %s", exc)
        logger.info("PREFILL_COST_SIGNATURE %s", self.signature)

    def poll(self):
        while self.pending and self.pending[0][1].query():
            start, end, sample = self.pending.popleft()
            sample["gpu_stream_ms"] = start.elapsed_time(end)
            sample["dropped_samples"] = self.dropped
            logger.info("PREFILL_COST_SAMPLE %s", json.dumps(sample, sort_keys=True))

    def begin(self, scheduler, batch):
        self.poll()
        self.step += 1
        if self.step % self.interval or len(self.pending) == self.pending.maxlen:
            if len(self.pending) == self.pending.maxlen:
                self.dropped += 1
            return None
        resets = getattr(batch, "encoder_swa_reset", None)
        work = [
            PrefillWork(
                int(r.extend_len),
                int(r.prefix_len),
                min(int(r.prefix_len), 128) if resets is not None and resets[i] else 0,
            )
            for i, r in enumerate(batch.reqs)
        ]
        import torch

        stream = (
            scheduler.forward_stream
            if scheduler.enable_overlap
            else torch.cuda.current_stream()
        )
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record(stream)
        sample = {
            "schema": 1,
            "signature": self.signature,
            "rank": self.rank,
            "boot": f"{os.getpid()}-{self.boot_ns}",
            "step": self.step,
            "work": [
                {"rows": w.rows, "prefix": w.prefix, "replay": w.replay} for w in work
            ],
            "predicted_ms": self.model.predict_ms(work) if self.model else None,
            "measurement": "scheduler_forward_stream_span",
            "hicache_pending": batch.hicache_consumer_index >= 0,
            "sync_wait_ms": 1000 * float(getattr(batch, "dp_sync_wait_seconds", 0)),
            "cpu_launch_ns": time.monotonic_ns(),
        }
        return start, end, sample, stream

    boot_ns = time.time_ns()

    def finish(self, token):
        if token is not None:
            start, end, sample, stream = token
            end.record(stream)
            self.pending.append((start, end, sample))


def observe_prefill_forward(fn):
    @wraps(fn)
    def wrapped(scheduler, batch, *args, **kwargs):
        # No event construction, signature hashing, or request walk when off.
        if os.environ.get("SGLANG_PREFILL_TIME_BUDGET_MODE", "off") == "off":
            return fn(scheduler, batch, *args, **kwargs)
        runtime = getattr(scheduler, "_prefill_cost_runtime", None)
        if runtime is None:
            runtime = scheduler._prefill_cost_runtime = PrefillCostRuntime(scheduler)
        token = None
        if batch.forward_mode.is_extend_without_speculative() and batch.reqs:
            token = runtime.begin(scheduler, batch)
        else:
            runtime.poll()
        result = fn(scheduler, batch, *args, **kwargs)
        runtime.finish(token)
        return result

    return wrapped
