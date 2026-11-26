"""Model-owned, bounded scratch for SM90 MXFP4 Prefill.

No process-global tensor cache: all layers of one target runner share one arena.
Allocate before KV sizing. Never grow or silently fall back on the hot path.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock

_active_graph_workspace = ContextVar("active_mxfp4_graph_workspace", default=None)


class _TorchBackend:
    def __init__(self):
        import torch

        self.torch = torch

    def allocate(self, size, device):
        return self.torch.empty(size, dtype=self.torch.uint8, device=device)

    def event(self):
        return self.torch.cuda.Event()

    def stream(self, device):
        return self.torch.cuda.current_stream(device)

    def capturing(self):
        return self.torch.cuda.is_current_stream_capturing()


class PrefillWorkspace:
    """One scratch buffer; serialize enqueueing and cross-stream GPU accesses.

    Same-stream layers need no event operations. On a stream change, record
    completion on the previous stream, then enqueue a wait on the next one.
    Graph capture and replay must both hold execution_scope. Captured kernels
    use the same stable pointer; replay must establish stream ownership again.
    """

    def __init__(self, size_bytes, max_rows, device, *, backend=None):
        if size_bytes <= 0 or max_rows <= 0:
            raise ValueError("workspace bytes and row bound must be positive")
        self.backend = backend or _TorchBackend()
        self.device = device
        self.size_bytes = size_bytes
        self.max_rows = max_rows
        self.buffer = self.backend.allocate(size_bytes, device)
        self._completion = self.backend.event()
        self._previous_stream = None
        self._lock = RLock()

    def _handoff(self, stream):
        previous = self._previous_stream
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            self._completion.record(previous)
            stream.wait_event(self._completion)
        self.buffer.record_stream(stream)
        self._previous_stream = stream

    @contextmanager
    def execution_scope(self):
        """Serialize a graph invocation, including replays that skip Python use()."""
        with self._lock:
            if self.backend.capturing():
                raise RuntimeError(
                    "workspace execution scope must start outside capture"
                )
            self._handoff(self.backend.stream(self.device))
            token = _active_graph_workspace.set(self)
            try:
                yield
            finally:
                _active_graph_workspace.reset(token)

    @contextmanager
    def use(self, rows):
        if not 0 < rows <= self.max_rows:
            raise ValueError(
                f"MXFP4 Prefill workspace row bound exceeded: {rows} > {self.max_rows}; "
                "restart with a sufficient prefill capacity"
            )
        with self._lock:
            stream = self.backend.stream(self.device)
            if self.backend.capturing():
                if (
                    _active_graph_workspace.get() is not self
                    or self._previous_stream is None
                    or self._previous_stream.cuda_stream != stream.cuda_stream
                ):
                    raise RuntimeError(
                        "MXFP4 workspace capture requires a guarded runner"
                    )
                # Warmup established ownership on this capture stream. Do not
                # record events or allocator lifetime operations inside capture.
            else:
                self._handoff(stream)
            yield self.buffer


@dataclass(frozen=True)
class WorkspaceBinding:
    arena: PrefillWorkspace
    input_dtype: object
    hidden_size: int

    def use(self, x):
        if (
            x.dtype != self.input_dtype
            or x.device != self.arena.device
            or len(x.shape) != 2
            or x.shape[1] != self.hidden_size
        ):
            raise ValueError("MXFP4 Prefill workspace input dtype/device changed")
        return self.arena.use(x.shape[0])


def initialize_prefill_workspace(model_runner):
    """Opt in only the target-only SM90 W4A16 Prefill path."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    if not envs.SGLANG_FLASHINFER_MXFP4_PREFILL_PREALLOC_WORKSPACE.get():
        return
    if get_disagg().disaggregation_mode != "prefill" or model_runner.is_draft_worker:
        return

    import inspect
    import logging

    import torch

    from sglang.srt.layers.dp_attention import deployment_attn_dp_size
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.runtime_context import (
        get_exec,
        get_parallel,
        max_prefill_buffer_tokens,
    )

    parallel = get_parallel()
    if (
        get_exec().graph.cuda_graph_config.prefill.backend != "disabled"
        or parallel.pp_size != 1
        or parallel.attn_cp_size != 1
    ):
        raise ValueError(
            "MXFP4 workspace preallocation requires eager Prefill, PP1, CP1"
        )
    layers = []
    for layer in model_runner.model.modules():
        method = getattr(layer, "quant_method", None)
        if isinstance(method, Mxfp4FlashinferCutlassMoEMethod):
            if method._use_mxfp8_act_scaling or method._use_sm90_humming:
                raise ValueError(
                    "workspace preallocation currently requires SM90 W4A16"
                )
            if getattr(layer, "_mega_moe_weights_built", False):
                raise ValueError("workspace preallocation does not support MegaMoE")
            layers.append((layer, method))
    if not layers:
        raise ValueError(
            "no SM90 CUTLASS MXFP4 layers found for workspace preallocation"
        )

    from flashinfer.fused_moe import (
        cutlass_fused_moe,
        cutlass_fused_moe_workspace_size,
    )
    from flashinfer.fused_moe.core import ActivationType

    if "workspace_buffer" not in inspect.signature(cutlass_fused_moe).parameters:
        raise RuntimeError("FlashInfer must support cutlass_fused_moe workspace_buffer")
    tails = [
        layer.decoder_replay_tail
        for layer in model_runner.model.modules()
        if getattr(layer, "decoder_replay_tail", None) is not None
    ]
    max_rows = (
        max([max_prefill_buffer_tokens()] + [tail.runner.max_rows for tail in tails])
        * deployment_attn_dp_size()
    )
    if max_rows <= 0:
        raise ValueError("a bounded prefill buffer capacity is required")
    sizes = {}
    device = layers[0][0].w13_weight.device
    for layer, method in layers:
        config = method.moe_runner_config
        if layer.w13_weight.device != device or config.params_dtype != torch.bfloat16:
            raise ValueError("workspace requires one CUDA device and BF16 activations")
        activation = (
            ActivationType.SwigluStep
            if method._use_swiglu_step
            else ActivationType.Swiglu
        )
        # Match every shape/runner flag of _fused_experts_flashinfer_mxfp4_cutlass.
        key = (
            config.hidden_size,
            config.intermediate_size_per_partition,
            config.num_experts,
            config.top_k,
            layer.w13_weight.dtype,
            activation,
            config.moe_tp_size,
            config.moe_tp_rank,
            config.moe_ep_size,
            config.moe_ep_rank,
        )
        if key not in sizes:
            sizes[key] = cutlass_fused_moe_workspace_size(
                max_rows,
                config.hidden_size,
                config.intermediate_size_per_partition,
                config.num_experts,
                config.top_k,
                x_dtype=torch.bfloat16,
                weight_dtype=layer.w13_weight.dtype,
                output_dtype=torch.bfloat16,
                activation_type=activation,
                tp_size=config.moe_tp_size,
                tp_rank=config.moe_tp_rank,
                ep_size=config.moe_ep_size,
                ep_rank=config.moe_ep_rank,
                use_w4_group_scaling=True,
                use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
                device=device,
            )
    arena = PrefillWorkspace(max(sizes.values()), max_rows, device)
    for _, method in layers:
        method._prefill_workspace = WorkspaceBinding(
            arena, torch.bfloat16, method.moe_runner_config.hidden_size
        )
    for tail in tails:
        tail.runner.execution_scope = arena.execution_scope
    # Explicit ownership, introspection, and lifetime of the target runner.
    model_runner.mxfp4_prefill_workspace = arena
    logging.getLogger(__name__).info(
        "Preallocated shared MXFP4 Prefill workspace before KV sizing: "
        "%.3f GiB, max_rows=%d, layers=%d, device=%s",
        arena.size_bytes / (1 << 30),
        max_rows,
        len(layers),
        device,
    )
