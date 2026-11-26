"""CPU contract checks for bounded scratch lifetime and stream hand-off."""

import ast
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/moe_runner/mxfp4_workspace.py"
)
spec = importlib.util.spec_from_file_location("mxfp4_workspace_under_test", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.stream_id = 1
        self.capture = False
        self.tensor = SimpleNamespace(
            record_stream=lambda s: self.calls.append(("record_stream", s.cuda_stream))
        )

    def allocate(self, size, device):
        self.calls.append(("allocate", size, device))
        return self.tensor

    def event(self):
        return SimpleNamespace(
            record=lambda s: self.calls.append(("event_record", s.cuda_stream))
        )

    def stream(self, device):
        return SimpleNamespace(
            cuda_stream=self.stream_id,
            wait_event=lambda e: self.calls.append(("wait", self.stream_id)),
        )

    def capturing(self):
        return self.capture


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.arena = m.PrefillWorkspace(100, 64, "cuda:0", backend=self.backend)

    def test_layers_reuse_one_allocation(self):
        for rows in [64, 17, 1, 64]:
            with self.arena.use(rows) as buffer:
                self.assertIs(buffer, self.backend.tensor)
        self.assertEqual(sum(c[0] == "allocate" for c in self.backend.calls), 1)
        self.assertFalse(any(c[0] == "event_record" for c in self.backend.calls))

    def test_cross_stream_wait_precedes_next_use(self):
        with self.arena.use(8):
            pass
        self.backend.stream_id = 2
        with self.arena.use(8):
            self.assertEqual(
                self.backend.calls[-3:],
                [("event_record", 1), ("wait", 2), ("record_stream", 2)],
            )
        self.backend.stream_id = 1
        with self.arena.use(8):
            self.assertEqual(
                self.backend.calls[-3:],
                [("event_record", 2), ("wait", 1), ("record_stream", 1)],
            )

    def test_exception_keeps_stream_dependency(self):
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            with self.arena.use(8):
                raise RuntimeError("kernel")
        self.backend.stream_id = 2
        with self.arena.use(8):
            self.assertIn(("event_record", 1), self.backend.calls)

    def test_capacity_never_grows_or_falls_back(self):
        for rows in [65, 0, -1]:
            with self.assertRaises(ValueError):
                with self.arena.use(rows):
                    self.fail("invalid rows reached kernel")
        self.assertEqual(len(self.backend.calls), 1)

    def test_capture_rejected_without_growth(self):
        self.backend.capture = True
        with self.assertRaisesRegex(RuntimeError, "guarded runner"):
            with self.arena.use(8):
                self.fail("capture reached kernel")
        self.assertEqual(len(self.backend.calls), 1)

    def test_guarded_capture_and_replay_stream_handoff(self):
        with self.arena.execution_scope():
            self.backend.stream_id = 2
            with self.arena.use(8):  # capture warmup
                pass
            before = list(self.backend.calls)
            self.backend.capture = True
            with self.arena.use(8):
                pass
            self.assertEqual(before, self.backend.calls)
            self.backend.capture = False
            self.backend.stream_id = 1
            with self.arena.execution_scope():  # replay skips use()
                self.assertEqual(
                    self.backend.calls[-3:],
                    [("event_record", 2), ("wait", 1), ("record_stream", 1)],
                )
        self.assertIsNone(m._active_graph_workspace.get())

    def test_capture_on_unprepared_stream_rejected(self):
        with self.arena.execution_scope():
            self.backend.stream_id = 2
            self.backend.capture = True
            with self.assertRaisesRegex(RuntimeError, "guarded runner"):
                with self.arena.use(8):
                    pass
        self.assertIsNone(m._active_graph_workspace.get())

    def test_execution_scope_exception_resets_owner(self):
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            with self.arena.execution_scope():
                raise RuntimeError("kernel")
        self.assertIsNone(m._active_graph_workspace.get())

    def test_dtype_device_guard(self):
        binding = m.WorkspaceBinding(self.arena, "bf16", 5120)
        for dtype, device in [("fp16", "cuda:0"), ("bf16", "cuda:1")]:
            with self.assertRaises(ValueError):
                binding.use(
                    SimpleNamespace(dtype=dtype, device=device, shape=(8, 5120))
                )

    def test_invalid_startup_capacity(self):
        with self.assertRaises(ValueError):
            m.PrefillWorkspace(0, 8, "cuda:0", backend=self.backend)


class StartupTests(unittest.TestCase):
    def setup_modules(self, *, role="prefill", draft=False, graph="disabled"):
        self.backend = FakeBackend()
        self.size_calls = []
        self.method_type = type("Method", (), {})
        device = "cuda:0"
        self.methods = []
        layers = []
        for top_k in [6, 6, 3]:
            method = self.method_type()
            method._use_mxfp8_act_scaling = method._use_sm90_humming = (
                method._use_swiglu_step
            ) = False
            method.moe_runner_config = SimpleNamespace(
                hidden_size=5120,
                intermediate_size_per_partition=2304,
                num_experts=384,
                top_k=top_k,
                params_dtype="bf16",
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=8,
                moe_ep_rank=0,
            )
            layers.append(
                SimpleNamespace(
                    quant_method=method,
                    w13_weight=SimpleNamespace(device=device, dtype="uint8"),
                )
            )
            self.methods.append(method)
        self.runner = SimpleNamespace(
            is_draft_worker=draft, model=SimpleNamespace(modules=lambda: iter(layers))
        )
        flag = SimpleNamespace(get=lambda: True)
        env = SimpleNamespace(
            SGLANG_FLASHINFER_MXFP4_PREFILL_PREALLOC_WORKSPACE=flag,
            SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=flag,
        )

        def size(*args, **kwargs):
            self.size_calls.append((args, kwargs))
            self.backend.calls.append(("size", args[0], args[4]))
            return args[4] * 100

        def fused(*, workspace_buffer=None):
            pass

        return {
            "sglang.srt.environ": SimpleNamespace(envs=env),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_disagg=lambda: SimpleNamespace(disaggregation_mode=role),
                get_exec=lambda: SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            prefill=SimpleNamespace(backend=graph)
                        )
                    )
                ),
                get_parallel=lambda: SimpleNamespace(pp_size=1, attn_cp_size=1),
                max_prefill_buffer_tokens=lambda: 8192,
            ),
            "sglang.srt.layers.dp_attention": SimpleNamespace(
                deployment_attn_dp_size=lambda: 8
            ),
            "sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe": SimpleNamespace(
                Mxfp4FlashinferCutlassMoEMethod=self.method_type
            ),
            "torch": SimpleNamespace(bfloat16="bf16"),
            "flashinfer.fused_moe": SimpleNamespace(
                cutlass_fused_moe=fused, cutlass_fused_moe_workspace_size=size
            ),
            "flashinfer.fused_moe.core": SimpleNamespace(
                ActivationType=SimpleNamespace(Swiglu=0, SwigluStep=1)
            ),
        }

    def test_plan_all_geometries_before_single_allocation(self):
        modules = self.setup_modules()
        with (
            patch.dict(sys.modules, modules),
            patch.object(m, "_TorchBackend", return_value=self.backend),
        ):
            m.initialize_prefill_workspace(self.runner)
        self.assertEqual(len(self.size_calls), 2)
        self.assertTrue(all(args[0] == 65536 for args, _ in self.size_calls))
        self.assertTrue(
            all(
                kw["use_w4_group_scaling"] and kw["ep_size"] == 8
                for _, kw in self.size_calls
            )
        )
        self.assertEqual(
            [c for c in self.backend.calls if c[0] == "allocate"],
            [("allocate", 600, "cuda:0")],
        )
        self.assertTrue(
            all(
                method._prefill_workspace.arena is self.runner.mxfp4_prefill_workspace
                for method in self.methods
            )
        )
        self.assertEqual(
            [c[0] for c in self.backend.calls], ["size", "size", "allocate"]
        )

    def test_decode_and_draft_are_not_changed(self):
        for role, draft in [("decode", False), ("prefill", True)]:
            modules = self.setup_modules(role=role, draft=draft)
            with patch.dict(sys.modules, modules):
                m.initialize_prefill_workspace(self.runner)
            self.assertFalse(self.size_calls)
            self.assertFalse(
                any(hasattr(method, "_prefill_workspace") for method in self.methods)
            )

    def test_graph_requires_explicit_rejection(self):
        modules = self.setup_modules(graph="full")
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(ValueError, "eager Prefill"),
        ):
            m.initialize_prefill_workspace(self.runner)
        self.assertFalse(self.backend.calls)

    def test_humming_is_not_silently_reused(self):
        modules = self.setup_modules()
        self.methods[0]._use_sm90_humming = True
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(ValueError, "SM90 W4A16"),
        ):
            m.initialize_prefill_workspace(self.runner)
        self.assertFalse(self.backend.calls)


class ForwardAdapterTests(unittest.TestCase):
    def test_existing_adapter_forwards_only_the_opted_in_buffer(self):
        source = PATH.with_name("flashinfer_cutlass.py")
        tree = ast.parse(source.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_fused_experts_flashinfer_mxfp4_cutlass"
        )
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        calls = []
        backend = FakeBackend()
        arena = m.PrefillWorkspace(100, 64, "cuda:0", backend=backend)
        x = SimpleNamespace(shape=(8, 5120), dtype="bf16", device="cuda:0")
        ids = SimpleNamespace(to=lambda dtype: "ids", view=lambda dtype: "scales")
        activation = SimpleNamespace(Swiglu=0, SwigluStep=1)
        namespace = {
            "torch": SimpleNamespace(
                bfloat16="bf16", int32="int32", empty=lambda *a, **kw: "output"
            ),
            "_flashinfer_cutlass_fused_moe": lambda: (
                lambda **kw: calls.append(kw),
                activation,
            ),
            "use_symmetric_memory": lambda *a, **kw: nullcontext(),
            "is_allocation_symmetric": lambda: False,
            "get_parallel": lambda: SimpleNamespace(tp_group=None),
            "envs": SimpleNamespace(
                SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=SimpleNamespace(get=lambda: True)
            ),
            "_cutlass_moe_tune_max_num_tokens": lambda n: 8192,
            "nullcontext": nullcontext,
        }
        exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
        quant = SimpleNamespace(
            padded_hidden=None,
            mxfp4_weight_global_scale=None,
            w13_humming_residual_scale=None,
            w2_humming_residual_scale=None,
            humming_fc2_act_scale=None,
            w13_weight=None,
            w2_weight=None,
            w13_weight_scale=ids,
            w2_weight_scale=ids,
            w13_bias=None,
            w2_bias=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            use_swiglu_step=False,
            prefill_workspace=None,
        )
        runner = SimpleNamespace(
            moe_tp_size=1, moe_tp_rank=0, moe_ep_size=8, moe_ep_rank=0
        )
        dispatch = SimpleNamespace(
            hidden_states=x,
            topk_output=SimpleNamespace(topk_ids=ids, topk_weights=None),
        )
        modules = {
            "sglang.srt.layers.moe.token_dispatcher.standard": SimpleNamespace(
                StandardCombineInput=lambda **kw: kw
            ),
            "sglang.srt.layers.moe.topk": SimpleNamespace(
                TopKOutputChecker=SimpleNamespace(format_is_bypassed=lambda x: False)
            ),
        }
        with patch.dict(sys.modules, modules):
            namespace[function.name](dispatch, quant, runner)
            self.assertNotIn("workspace_buffer", calls[-1])
            quant.prefill_workspace = m.WorkspaceBinding(arena, "bf16", 5120)
            namespace[function.name](dispatch, quant, runner)
            self.assertIs(calls[-1]["workspace_buffer"], arena.buffer)
        self.assertEqual(sum(c[0] == "allocate" for c in backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
