"""CPU contract checks with asynchronous event doubles; no fake GPU timings."""

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
try:
    from sglang.srt.managers.scheduler_components import prefill_budget as budget
    from sglang.srt.managers.scheduler_components import prefill_cost_model as model
    from sglang.srt.managers.scheduler_components import prefill_cost_runtime as runtime
except ModuleNotFoundError:
    MODEL_NAME = "sglang.srt.managers.scheduler_components.prefill_cost_model"
    spec = importlib.util.spec_from_file_location(
        MODEL_NAME,
        ROOT / "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py",
    )
    model = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {MODEL_NAME: model}):
        spec.loader.exec_module(model)
    BUDGET_NAME = "sglang.srt.managers.scheduler_components.prefill_budget"
    spec = importlib.util.spec_from_file_location(
        BUDGET_NAME,
        ROOT / "python/sglang/srt/managers/scheduler_components/prefill_budget.py",
    )
    budget = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {MODEL_NAME: model, BUDGET_NAME: budget}):
        spec.loader.exec_module(budget)

    spec = importlib.util.spec_from_file_location(
        "cost_runtime_test",
        ROOT
        / "python/sglang/srt/managers/scheduler_components/prefill_cost_runtime.py",
    )
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    with patch.dict(sys.modules, {MODEL_NAME: model, BUDGET_NAME: budget}):
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
            patch.dict("os.environ", {"SGLANG_PREFILL_TIME_BUDGET_MODE": "invalid"}),
            self.assertRaises(ValueError),
        ):
            runtime.PrefillCostRuntime(object())

    def test_consensus_fails_closed(self):
        self.assertEqual(budget.consensus_budget([(1, 100), (1, 80)]), 80)
        for votes in ([], [(1, 100), (0, 0)], [(1, 100), (2, 100)], [(1, 0)]):
            self.assertEqual(budget.consensus_budget(votes), 0)

    def test_budget_requires_group_agreement_and_preserves_progress(self):
        b = budget.PrefillBudget(
            "enforce", model.PrefillCostModel("h200", (0, 1, 0, 0, 0)), 100
        )
        args = {"prefix": 0, "replay": 0, "remaining": 1000, "page_size": 128}
        self.assertIsNone(b.limit([], **args))
        b.receive([b.vote()] * 8)
        self.assertEqual(b.limit([], **args), 128)
        self.assertEqual(b.limit([model.PrefillWork(128, 0)], **args), 0)
        b.receive([])
        self.assertIsNone(b.limit([], **args))

    def test_control_file_transition_and_invalid_profile_fallback(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            rt = runtime.PrefillCostRuntime.__new__(runtime.PrefillCostRuntime)
            rt.signature = "h200"
            rt.rank = 0
            rt.mode = "shadow"
            rt.model = None
            rt.budget = budget.PrefillBudget()
            rt.control_path = str(path)
            rt.control_next_poll = 0
            rt.control_mtime = None
            rt.control_generation = 0
            path.write_text(
                json.dumps({"generation": 1, "mode": "shadow", "fixed_rows": 4096})
            )
            rt.refresh_control()
            self.assertEqual((rt.mode, rt.fixed_rows), ("shadow", 4096))
            model_file = Path(directory) / "model.json"
            model_file.write_text(
                json.dumps(
                    {
                        "signature": "h200",
                        "coefficients": [0, 1, 0, 0, 0],
                        "prediction_gate": True,
                    }
                )
            )
            path.write_text(
                json.dumps(
                    {
                        "generation": 2,
                        "mode": "enforce",
                        "target_ms": 100,
                        "model_path": str(model_file),
                    }
                )
            )
            rt.control_next_poll = 0
            rt.refresh_control()
            self.assertEqual(rt.mode, "enforce")
            self.assertEqual(rt.budget.group_budget_us, 0)
            rt.budget.receive([rt.budget.vote()] * 8)
            self.assertEqual(rt.budget.group_budget_us, 100000)
            path.write_text(json.dumps({"generation": 3, "mode": "invalid"}))
            rt.control_next_poll = 0
            rt.refresh_control()
            self.assertEqual(
                (rt.mode, rt.fixed_rows, rt.budget.group_budget_us), ("off", 0, 0)
            )

    def test_runtime_initialization_preserves_unready_fallback(self):
        parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dp_rank=0)
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_parallel=lambda: parallel
                    )
                },
            ),
            patch.object(runtime, "execution_signature", return_value="h200"),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_PREFILL_TIME_BUDGET_MODE": "shadow",
                    "SGLANG_PREFILL_COST_CONTROL": "",
                    "SGLANG_PREFILL_COST_MODEL": "",
                    "SGLANG_PREFILL_FIXED_CHUNK_ROWS": "0",
                },
            ),
        ):
            rt = runtime.PrefillCostRuntime(object())
            self.assertEqual(rt.pending.maxlen, 64)
            self.assertEqual(rt.budget.vote(), (0, 0))
            self.assertEqual(rt.fixed_rows, 0)

    def test_event_collection_uses_actual_request_shape_and_replay(self):
        recorded = []

        class Event:
            def record(self, stream):
                recorded.append(stream)

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda: "test-stream", Event=lambda **kwargs: Event()
            )
        )
        rt = runtime.PrefillCostRuntime.__new__(runtime.PrefillCostRuntime)
        rt.pending = deque(maxlen=64)
        rt.step, rt.interval, rt.dropped = 0, 1, 0
        rt.signature, rt.rank = "h200", 0
        rt.model = None
        rt.budget = budget.PrefillBudget()
        rt.fixed_rows, rt.control_generation = 8192, 1
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(extend_len=256, prefix_len=512)],
            encoder_swa_reset=[True],
            hicache_consumer_index=-1,
            global_num_tokens=[384] * 8,
            extend_num_tokens=256,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            token = rt.begin(SimpleNamespace(enable_overlap=False), batch)
            rt.finish(token)
        self.assertEqual(recorded, ["test-stream", "test-stream"])
        self.assertEqual(
            token[2]["work"], [{"rows": 256, "prefix": 512, "replay": 128}]
        )
        self.assertEqual(len(rt.pending), 1)

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

    def test_shared_runtime_uses_actual_peers_and_suppresses_unknown_idle_quote(self):
        rt = runtime.PrefillCostRuntime.__new__(runtime.PrefillCostRuntime)
        rt.signature = "h200"
        rt.dp_degree = 8
        rt.model = model.PrefillCostModel("h200", (1, 0.01, 0, 0, 0, 0.02, 0.03), 8)
        rt.budget = budget.PrefillBudget("enforce", rt.model, 1000)
        rt.queue_complete = True
        rt.queue_us = 0
        rt.quote_sequence = 1
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(extend_len=317, prefix_len=98048)],
            global_num_tokens=[317, 8192, 0, 0, 0, 0, 0, 0],
        )
        local = rt.model.predict_ms([model.PrefillWork(317, 98048)])
        rt.update_activity(batch)
        self.assertGreater(rt.predicted_ms, local)
        quote = rt.quote()
        self.assertEqual(quote["schema"], 2)
        self.assertEqual(quote["dp_degree"], 8)
        self.assertEqual(len(quote["coefficients_ps"]), 7)
        rt.update_activity(
            SimpleNamespace(reqs=[], global_num_tokens=batch.global_num_tokens)
        )
        self.assertIsNone(rt.quote())
        rt.update_activity(SimpleNamespace(reqs=[], global_num_tokens=[0] * 8))
        self.assertIsNotNone(rt.quote())
        rt.update_activity(SimpleNamespace(reqs=batch.reqs, global_num_tokens=[]))
        self.assertIsNone(rt.model)
        self.assertEqual(rt.budget.vote(), (0, 0))

    def test_shared_model_contract_is_explicit_and_degree_bound(self):
        rt = runtime.PrefillCostRuntime.__new__(runtime.PrefillCostRuntime)
        rt.signature = "h200"
        rt.dp_degree = 8
        artifact = {
            "schema": 2,
            "feature_contract": "shared-dp-v2",
            "dp_degree": 8,
            "signature": "h200",
            "coefficients": [1, 0.01, 0, 0, 0, 0.02, 0.03],
        }
        self.assertTrue(rt.load_model(artifact).shared)
        for changes in (
            {"schema": 1},
            {"feature_contract": "unknown"},
            {"dp_degree": 4},
            {"coefficients": [1, 0.01, 0, 0, 0]},
        ):
            with self.assertRaises(ValueError):
                rt.load_model({**artifact, **changes})

    def test_chronological_fit_and_serving_gate(self):
        samples = []
        for i in range(200):
            rows = 128 * (1 + i % 32)
            prefix = 1024 * (1 + i % 17)
            samples.append(
                {
                    "signature": "h200",
                    "rank": i % 8,
                    "boot": f"rank-{i % 8}-boot-a",
                    "hicache_pending": False,
                    "work": [{"rows": rows, "prefix": prefix}],
                    "gpu_stream_ms": 1 + 0.001 * rows + 0.000001 * rows * prefix,
                    "global_num_tokens": [rows] * 8,
                    "cpu_launch_ns": (i + 1) * 10_000_000_000,
                }
            )
        result = calibration.fit(samples, "h200")
        self.assertEqual(result["train_samples"], 140)
        self.assertEqual(result["schema"], 2)
        self.assertEqual(result["dp_degree"], 8)
        self.assertEqual(len(result["coefficients"]), 7)
        self.assertLess(result["p95_error"], 0.05)
        self.assertTrue(result["temporal_holdout_verified"])
        self.assertFalse(result["serving_gate"])
        with self.assertRaises(ValueError):
            calibration.fit(samples, "other-hardware")


if __name__ == "__main__":
    main()
