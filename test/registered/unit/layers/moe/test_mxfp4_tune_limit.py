"""Check actual adapter tune selection without importing CUDA dependencies."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py"
)


class TuneLimitTests(unittest.TestCase):
    def select(self, role, limit, rows=65536, capacity=8192, mxfp4=True):
        node = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_cutlass_moe_tune_max_num_tokens"
        )
        scope = dict(
            get_disagg=lambda: SimpleNamespace(disaggregation_mode=role),
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            max_prefill_buffer_tokens=lambda: capacity,
            deployment_attn_dp_size=lambda: 8,
            _FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS=8192,
            envs=SimpleNamespace(
                SGLANG_FLASHINFER_MXFP4_PREFILL_TUNE_MAX_TOKENS=SimpleNamespace(
                    get=lambda: limit
                )
            ),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
            scope,
        )
        return scope[node.name](rows, mxfp4=mxfp4)

    def test_prefill_candidates_and_capacity(self):
        for n in (8192, 32768, 65536):
            self.assertEqual(self.select("prefill", n), n)
        self.assertEqual(self.select("prefill", 65536, capacity=4096), 32768)

    def test_decode_and_other_backends_unchanged(self):
        self.assertEqual(self.select("decode", 65536, rows=129), 256)
        self.assertEqual(self.select("prefill", 65536, mxfp4=False), 8192)
        self.assertEqual(self.select("null", 65536), 8192)

    def test_invalid_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.select("prefill", 16384)
