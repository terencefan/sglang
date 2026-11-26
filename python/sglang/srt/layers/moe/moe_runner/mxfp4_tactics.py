"""Opt-in replay of a validated target-Prefill MXFP4 tactic cache."""

import ast
import hashlib
import json
import logging
from pathlib import Path


def validate_cache(path, expected_sha256):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("MXFP4 tactic cache must be an absolute regular file")
    raw = path.read_bytes()
    if len(expected_sha256) != 64 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("MXFP4 tactic cache checksum mismatch")
    cache = json.loads(raw)
    metadata = cache.get("_metadata", {})
    if (
        metadata.get("flashinfer_version") != "0.7.4"
        or metadata.get("gpu") != "NVIDIA H200"
    ):
        raise ValueError("MXFP4 tactic cache requires FlashInfer0.7.4 on H200")
    stages = set()
    count = 0
    for key, value in cache.items():
        if key == "_metadata":
            continue
        parsed = ast.literal_eval(
            key.replace("torch.bfloat16", "'bf16'").replace("torch.uint8", "'u8'")
        )
        op, runner, shapes, extras = parsed
        if (
            op not in ("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2")
            or runner != "MoERunner"
            or shapes[0][0] not in (8192, 32768, 65536)
            or shapes
            != ((shapes[0][0], 5120), (48, 4608, 2560), (0,), (48, 5120, 1152), (0,))
            or extras
            != (
                "bf16",
                "u8",
                "bf16",
                6,
                1,
                8,
                1,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                3,
                False,
                True,
            )
            or value[0] != "MoERunner"
            or not isinstance(value[1], int)
        ):
            raise ValueError("cache contains an unvalidated MXFP4 shape or mode")
        stages.add((shapes[0][0], op))
        count += 1
    if count < 2 or any(
        (rows, f"trtllm::fused_moe::gemm{stage}") not in stages
        for rows, _ in stages
        for stage in (1, 2)
    ):
        raise ValueError("cache must contain both GEMM stages")
    return path, count


def load_prefill_tactics(model_runner):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg

    path = envs.SGLANG_FLASHINFER_MXFP4_PREFILL_TACTICS_PATH.get()
    if not path or model_runner.is_draft_worker:
        return
    if get_disagg().disaggregation_mode != "prefill":
        raise ValueError("MXFP4 Prefill tactic replay requires Prefill role")
    path, count = validate_cache(
        path, envs.SGLANG_FLASHINFER_MXFP4_PREFILL_TACTICS_SHA256.get()
    )
    from flashinfer.autotuner import AutoTuner

    if not AutoTuner.get().load_configs(str(path)):
        raise ValueError("FlashInfer rejected MXFP4 tactic cache metadata")
    logging.getLogger(__name__).info(
        "Loaded %s validated Prefill MXFP4 tactic records from %s", count, path
    )
