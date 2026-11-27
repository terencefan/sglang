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

from sglang.srt.managers.scheduler_components.prefill_budget import PrefillBudget
from sglang.srt.managers.scheduler_components.prefill_cost_model import (
    PrefillCostModel,
    PrefillWork,
    remaining_cost_ms,
)

logger = logging.getLogger(__name__)


def execution_signature(scheduler):
    runner = scheduler.tp_worker.model_runner
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    source = Path(__file__).parents[2]
    hashes = {}
    for name in (
        "managers/scheduler.py",
        "managers/schedule_policy.py",
        "managers/scheduler_components/dp_attn.py",
        "managers/scheduler_components/prefill_cost_model.py",
        "managers/scheduler_components/prefill_cost_runtime.py",
        "managers/scheduler_components/prefill_budget.py",
        "model_executor/model_runner.py",
        "layers/moe/moe_runner/flashinfer_cutlass.py",
        "layers/quantization/mxfp4_flashinfer_cutlass_moe.py",
        "layers/moe/moe_runner/mxfp4_workspace.py",
        "layers/moe/moe_runner/mxfp4_tactics.py",
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
        "args": {
            k: v
            for k, v in vars(scheduler.server_args).items()
            if k
            in (
                "attention_backend",
                "moe_runner_backend",
                "moe_a2a_backend",
                "mem_fraction_static",
                "chunked_prefill_size",
                "page_size",
                "dtype",
                "quantization",
                "disable_cuda_graph",
                "enable_dp_attention",
            )
        },
        "source": hashes,
        "kernel_environment": {
            k: os.environ.get(k)
            for k in (
                "SGLANG_FLASHINFER_MXFP4_PREFILL_TACTICS_SHA256",
                "SGLANG_FLASHINFER_MXFP4_PREFILL_PREALLOC_WORKSPACE",
                "SGLANG_RAGGED_VERIFY_MODE",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


class PrefillCostRuntime:
    def __init__(self, scheduler):
        self.mode = os.environ.get("SGLANG_PREFILL_TIME_BUDGET_MODE", "off")
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError("invalid Prefill time-budget mode")
        self.budget = PrefillBudget(self.mode)
        self.pending = deque(maxlen=64)
        self.step = 0
        self.dropped = 0
        self.model = None
        self.signature = None
        self.rank = 0
        self.queue_us = 0
        self.queue_complete = False
        self.quote_sequence = 0
        self.active_until_ns = 0
        if self.mode == "off":
            return
        self.signature = execution_signature(scheduler)
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if parallel.pp_size != 1 or parallel.attn_cp_size != 1:
            raise ValueError("Prefill time budgets currently require PP1 and CP1")
        self.rank = parallel.attn_dp_rank
        self.interval = int(os.environ.get("SGLANG_PREFILL_COST_SAMPLE_INTERVAL", "32"))
        if self.interval < 1:
            raise ValueError("sample interval must be positive")
        path = os.environ.get("SGLANG_PREFILL_COST_MODEL")
        if path:
            try:
                artifact = json.loads(Path(path).read_text())
                if artifact["signature"] != self.signature:
                    raise ValueError("execution signature mismatch")
                if self.mode == "enforce" and not artifact.get(
                    "prediction_gate", False
                ):
                    raise ValueError("model has not passed prediction holdout gate")
                self.model = PrefillCostModel(
                    self.signature, tuple(artifact["coefficients"])
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Prefill cost model unavailable: %s", exc)
        self.budget.model = self.model
        self.budget.target_ms = float(
            os.environ.get("SGLANG_PREFILL_TIME_BUDGET_MS", "0")
        )
        self.budget.__post_init__()
        if self.mode == "enforce" and self.budget.target_ms <= 0:
            raise ValueError("enforce requires positive SGLANG_PREFILL_TIME_BUDGET_MS")
        self.fixed_rows = int(os.environ.get("SGLANG_PREFILL_FIXED_CHUNK_ROWS", "0"))
        self.control_path = os.environ.get("SGLANG_PREFILL_COST_CONTROL")
        self.control_next_poll = 0.0
        self.control_mtime = None
        self.control_generation = 0
        self.refresh_control()
        logger.info("PREFILL_COST_SIGNATURE %s", self.signature)

    def refresh_control(self):
        # Optional local-file control is for isolated acceptance runs. It does
        # not modify source, memory sizing or the public inference API.
        if not self.control_path or time.monotonic() < self.control_next_poll:
            return
        self.control_next_poll = time.monotonic() + 0.25
        try:
            path = Path(self.control_path)
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp == self.control_mtime:
                return
            if stat.st_size > 65536:
                raise ValueError("control document too large")
            cfg = json.loads(path.read_text())
            generation = int(cfg["generation"])
            if generation <= self.control_generation:
                return
            mode = cfg["mode"]
            fixed = int(cfg.get("fixed_rows", 0))
            if mode not in ("off", "shadow", "enforce") or fixed not in (0, 4096, 8192):
                raise ValueError("invalid control profile")
            target = float(cfg.get("target_ms", 0))
            model = None
            if cfg.get("model_path"):
                model_file = Path(cfg["model_path"])
                if model_file.stat().st_size > 1048576:
                    raise ValueError("cost artifact too large")
                artifact = json.loads(model_file.read_text())
                if artifact["signature"] != self.signature:
                    raise ValueError("execution signature mismatch")
                if mode == "enforce" and not artifact.get("prediction_gate", False):
                    raise ValueError("prediction holdout gate not passed")
                model = PrefillCostModel(
                    self.signature, tuple(artifact["coefficients"])
                )
            budget = PrefillBudget(mode, model, target)
            if mode == "enforce" and (model is None or target <= 0):
                raise ValueError("enforce requires a calibrated model and time target")
            self.mode, self.model, self.budget = mode, model, budget
            self.fixed_rows = fixed
            self.control_mtime = stamp
            self.control_generation = generation
            self.queue_complete = False
            logger.info(
                "PREFILL_COST_CONTROL %s",
                json.dumps(
                    {
                        "generation": generation,
                        "rank": self.rank,
                        "mode": mode,
                        "fixed_rows": fixed,
                        "target_ms": target,
                        "model_ready": model is not None,
                    },
                    sort_keys=True,
                ),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            # A broken experiment profile disables control, never serving.
            self.budget = PrefillBudget()
            self.mode, self.model, self.fixed_rows = "off", None, 0
            self.queue_complete = False
            logger.warning("Prefill cost control disabled: %s", exc)

    def queue_quote(self, event, req=None, uncached=0):
        if event == "begin":
            self.queue_us = 0
            self.queue_complete = self.model is not None
        elif event in ("request", "continuation") and self.model is not None:
            # Cache residency/loading is not known precisely here. Warm queues
            # remain on the old route until a reload model has been calibrated.
            prefix = max(0, int(req.seqlen) - int(uncached))
            if prefix and event != "continuation":
                self.queue_complete = False
            self.queue_us += int(
                1000 * remaining_cost_ms(self.model, PrefillWork(int(uncached), prefix))
            )
        elif event == "end":
            self.quote_sequence += 1

    def quote(self):
        if self.model is None or not self.queue_complete:
            return None
        key = json.dumps(
            [self.model.signature, self.model.coefficients], sort_keys=True
        )
        return {
            "schema": 1,
            "model": int.from_bytes(hashlib.sha256(key.encode()).digest()[:7], "big")
            or 1,
            "sequence": self.quote_sequence,
            "boot": self.boot_ns,
            "coefficients_ps": [round(c * 1000000000) for c in self.model.coefficients],
            "queue_us": self.queue_us,
            "remaining_us": max(
                0, (self.active_until_ns - time.monotonic_ns()) // 1000
            ),
        }

    def poll(self):
        while self.pending and self.pending[0][1].query():
            start, end, sample = self.pending.popleft()
            sample["gpu_stream_ms"] = start.elapsed_time(end)
            sample["dropped_samples"] = self.dropped
            logger.info("PREFILL_COST_SAMPLE %s", json.dumps(sample, sort_keys=True))

    def begin(self, scheduler, batch):
        self.poll()
        self.step += 1
        if self.model is not None:
            work = [
                PrefillWork(int(r.extend_len), int(r.prefix_len)) for r in batch.reqs
            ]
            self.active_until_ns = time.monotonic_ns() + int(
                self.model.predict_ms(work) * 1000000
            )
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
            "sync_wait_ms": (
                1000 * batch.dp_balance_stats.sync_wait_seconds
                if getattr(batch, "dp_balance_stats", None) is not None
                else None
            ),
            "budget_mode": self.budget.mode,
            "fixed_rows": self.fixed_rows,
            "control_generation": self.control_generation,
            "group_budget_us": self.budget.group_budget_us,
            "cpu_launch_ns": time.monotonic_ns(),
            "global_num_tokens": list(batch.global_num_tokens or []),
            "local_num_tokens": int(batch.extend_num_tokens),
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
        runtime.refresh_control()
        if runtime.mode == "off":
            return fn(scheduler, batch, *args, **kwargs)
        token = None
        if batch.forward_mode.is_extend_without_speculative() and batch.reqs:
            token = runtime.begin(scheduler, batch)
        else:
            runtime.poll()
        result = fn(scheduler, batch, *args, **kwargs)
        if token is not None:
            token[2]["cuda_graph_replay"] = bool(getattr(result, "can_run_cuda_graph", False))
        runtime.finish(token)
        return result

    return wrapped
