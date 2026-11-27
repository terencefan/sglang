import importlib.util
import sys
import unittest
from pathlib import Path

path = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py"
)
try:
    from sglang.srt.managers.scheduler_components import prefill_cost_model as m
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("prefill_cost_model_under_test", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)


class CostModelTests(unittest.TestCase):
    def test_whole_batch_budget(self):
        model = m.PrefillCostModel("h200", (0, 1, 0, 0, 0))
        self.assertEqual(
            model.chunk_rows(
                [m.PrefillWork(128, 0)],
                prefix=0,
                replay=0,
                remaining=1000,
                budget_ms=400,
                page_size=128,
            ),
            256,
        )

    def test_prefix_and_replay_cost(self):
        model = m.PrefillCostModel("h200", (0, 1, 0, 0.01, 0))
        self.assertGreater(
            model.predict_ms([m.PrefillWork(128, 100)]),
            model.predict_ms([m.PrefillWork(128, 0)]),
        )
        self.assertGreater(
            model.predict_ms([m.PrefillWork(128, 0, 10)]),
            model.predict_ms([m.PrefillWork(128, 0)]),
        )

    def test_caps_tail_and_no_fit(self):
        model = m.PrefillCostModel("h200", (0, 1, 0, 0, 0))

        def choose(n, budget):
            return model.chunk_rows(
                [],
                prefix=0,
                replay=0,
                remaining=n,
                budget_ms=budget,
                page_size=128,
                hard_cap=32768,
            )

        self.assertEqual(choose(32768, 100000), 8192)
        self.assertEqual(choose(17, 100), 17)
        self.assertEqual(choose(1024, 1), 0)

    def test_nonfinal_cap_is_aligned(self):
        model = m.PrefillCostModel("h200", (0, 1, 0, 0, 0))
        self.assertEqual(
            model.chunk_rows(
                [],
                prefix=0,
                replay=0,
                remaining=1000,
                budget_ms=10000,
                page_size=128,
                hard_cap=300,
            ),
            256,
        )

    def test_invalid_budget_and_negative_work(self):
        model = m.PrefillCostModel("h200", (0, 1, 0, 0, 0))
        for budget in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.chunk_rows(
                    [],
                    prefix=0,
                    replay=0,
                    remaining=1000,
                    budget_ms=budget,
                    page_size=128,
                )
        with self.assertRaises(ValueError):
            m.PrefillWork(-1, 10)

    def test_invalid_coefficients(self):
        for x in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m.PrefillCostModel("h200", (x, 0, 0, 0, 0))

    def test_shared_rows_change_prediction_and_identity(self):
        model = m.PrefillCostModel("h200", (1, 2, 3, 4, 5, 6, 7), 8)
        work = [m.PrefillWork(10, 20, 2)]
        small = model.predict_ms(work, global_num_tokens=[12] * 8)
        large = model.predict_ms(work, global_num_tokens=[12, 8192, 0, 0, 0, 0, 0, 0])
        self.assertGreater(large, small)
        self.assertEqual(model.predict_ms(work), small)
        self.assertNotEqual(
            model.fingerprint,
            m.PrefillCostModel("h200", model.coefficients, 4).fingerprint,
        )
        for shape in ([12], [-1] * 8, [1.5] * 8):
            with self.assertRaises(ValueError):
                model.predict_ms(work, global_num_tokens=shape)
        self.assertGreater(
            model.predict_ms([], global_num_tokens=[0, 8192, 0, 0, 0, 0, 0, 0]), 0
        )

    def test_shared_quote_matches_chunk_sum(self):
        model = m.PrefillCostModel("h200", (1, 0.002, 1e-7, 1e-9, 0.1, 0.003, 0.004), 8)
        work = m.PrefillWork(20000, 4096)
        expected = sum(
            model.predict_ms([m.PrefillWork(min(8192, 20000 - i), 4096 + i)])
            for i in range(0, 20000, 8192)
        )
        self.assertAlmostEqual(m.remaining_cost_ms(model, work), expected)

    def test_quote(self):
        self.assertEqual(
            m.finish_quote_ms(
                remaining_ms=1, queued_ms=2, dispatched_ms=3, compute_ms=4, reload_ms=5
            ),
            15,
        )


if __name__ == "__main__":
    unittest.main()
