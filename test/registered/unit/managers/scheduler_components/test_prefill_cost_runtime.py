"""CPU contract checks with asynchronous event doubles; no fake GPU timings."""

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
MODEL_NAME = "sglang.srt.managers.scheduler_components.prefill_cost_model"
spec = importlib.util.spec_from_file_location(
    MODEL_NAME,
    ROOT / "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py",
)
model = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {MODEL_NAME: model}):
    spec.loader.exec_module(model)
spec = importlib.util.spec_from_file_location(
    "cost_runtime_test",
    ROOT / "python/sglang/srt/managers/scheduler_components/prefill_cost_runtime.py",
)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
with patch.dict(sys.modules, {MODEL_NAME: model}):
    spec.loader.exec_module(runtime)
spec = importlib.util.spec_from_file_location(
    "cost_calibration_test", ROOT / "scripts/prefill_cost/calibrate.py"
)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class RuntimeTests(TestCase):
    def test_off_never_inspects_batch_or_constructs_events(self):
        calls = []

        @runtime.observe_prefill_forward
        def run(owner, batch):
            calls.append(batch)
            return 42

        with patch.dict("os.environ", {"SGLANG_PREFILL_TIME_BUDGET_MODE": "off"}):
            self.assertEqual(run(object(), None), 42)
        self.assertEqual(calls, [None])

    def test_unsupported_mode_fails_explicitly(self):
        with (
            patch.dict("os.environ", {"SGLANG_PREFILL_TIME_BUDGET_MODE": "enforce"}),
            self.assertRaises(ValueError),
        ):
            runtime.PrefillCostRuntime(object())

    def test_poll_queries_without_synchronize(self):
        rt = runtime.PrefillCostRuntime.__new__(runtime.PrefillCostRuntime)
        rt.pending = deque()
        rt.dropped = 0
        done = [False]
        end = SimpleNamespace(query=lambda: done[0])
        start = SimpleNamespace(elapsed_time=lambda e: 10.0)
        sample = {}
        rt.pending.append((start, end, sample))
        rt.poll()
        self.assertEqual(len(rt.pending), 1)
        done[0] = True
        rt.poll()
        self.assertEqual(len(rt.pending), 0)
        self.assertEqual(sample["gpu_stream_ms"], 10.0)

    def test_chronological_fit_and_serving_gate(self):
        samples = []
        for i in range(200):
            rows = 128 * (1 + i % 32)
            prefix = 1024 * (1 + i % 17)
            samples.append(
                {
                    "signature": "h200",
                    "hicache_pending": False,
                    "work": [{"rows": rows, "prefix": prefix}],
                    "gpu_stream_ms": 1 + 0.001 * rows + 0.000001 * rows * prefix,
                }
            )
        result = calibration.fit(samples, "h200")
        self.assertEqual(result["train_samples"], 140)
        self.assertLess(result["p95_error"], 0.05)
        self.assertFalse(result["serving_gate"])
        with self.assertRaises(ValueError):
            calibration.fit(samples, "other-hardware")


if __name__ == "__main__":
    main()
