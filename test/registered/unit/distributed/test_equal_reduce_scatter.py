"""CPU contract checks of extracted candidate; does not import SGLang/CUDA.
Backend dispatch/output-buffer tests only. NCCL reduction numerics require GPU validation.
"""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


class Tensor:
    def __init__(self, n, d=4):
        self.shape = (n, d)
        self.dtype = "bf16"
        self.device = "fake"


class Backend:
    def __init__(self):
        self.calls = []

    def change_state(self, **kwargs):
        return contextlib.nullcontext()

    def reduce_scatter(self, output, input_, sizes):
        self.calls.append((output, input_, sizes))


def load(path):
    tree = ast.parse(path.read_text())
    cls = next(
        c
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "GroupCoordinator"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "reduce_scatterv"
    )
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    envs = SimpleNamespace(
        SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(get=lambda: False),
        SGLANG_REDUCE_SCATTERV_EQUAL_SIZE_FASTPATH=SimpleNamespace(get=lambda: True),
    )
    ns = {"envs": envs}
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
    return ns["reduce_scatterv"], envs


class Contract(unittest.TestCase):
    def setUp(self):
        self.fn, self.envs = load(
            ROOT / "python/sglang/srt/distributed/parallel_state.py"
        )
        self.backend = Backend()
        self.coord = SimpleNamespace(
            world_size=8, rank_in_group=3, require_pynccl_comm=lambda _: self.backend
        )

    def call(self, sizes, input_rows, output_rows):
        inp, out = Tensor(input_rows), Tensor(output_rows)
        result = self.fn(self.coord, inp, output=out, sizes=sizes)
        self.assertIs(result, out)
        self.assertIs(self.backend.calls[-1][0], out)
        return self.backend.calls[-1][2]

    def test_disabled_keeps_existing_dispatch(self):
        self.envs.SGLANG_REDUCE_SCATTERV_EQUAL_SIZE_FASTPATH.get = lambda: False
        self.assertEqual(self.call([240] * 8, 1920, 240), [240] * 8)

    def test_equal_graph_shards_use_native(self):
        self.assertIsNone(self.call([240] * 8, 1920, 240))

    def test_unequal_shards_keep_root_lengths(self):
        s = [180, 192, 198, 210, 216, 222, 228, 234]
        self.assertEqual(self.call(s, sum(s), 210), s)

    def test_empty_rank_does_not_turn_variable_into_native(self):
        s = [0, 6, 12, 0, 18, 24, 30, 36]
        self.assertEqual(self.call(s, sum(s), 0), s)

    def test_all_empty_retains_zero_sized_output(self):
        self.assertIsNone(self.call([0] * 8, 0, 0))

    def test_implicit_equal_sizes_unchanged(self):
        self.assertIsNone(self.call(None, 1920, 240))

    def test_invalid_sizes_rejected_before_backend(self):
        for sizes, rows, outrows in [
            ([240] * 7, 1680, 240),
            ([240] * 8, 1900, 240),
            ([240] * 8, 1920, 239),
        ]:
            with self.subTest(sizes=sizes, rows=rows, outrows=outrows):
                with self.assertRaises(AssertionError):
                    self.call(sizes, rows, outrows)
        self.assertEqual(self.backend.calls, [])

    def test_deterministic_path_kept_before_native(self):
        marker = object()
        self.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get = lambda: True
        self.coord._deterministic_reduce_scatterv = lambda *a: marker
        self.assertIs(self.fn(self.coord, Tensor(1920), Tensor(240), [240] * 8), marker)
        self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
