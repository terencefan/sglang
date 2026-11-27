import importlib.util
import sys
import unittest
from pathlib import Path

path = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py"
)
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

    def test_quote(self):
        self.assertEqual(
            m.finish_quote_ms(
                remaining_ms=1, queued_ms=2, dispatched_ms=3, compute_ms=4, reload_ms=5
            ),
            15,
        )


if __name__ == "__main__":
    unittest.main()
