import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/moe/moe_runner/mxfp4_tactics.py"
)
spec = importlib.util.spec_from_file_location("tactics", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cache_file(tmp_path):
    extras = "(torch.bfloat16, torch.uint8, torch.bfloat16, 6, 1, 8, 1, False, False, True, False, False, False, True, 3, False, False)"
    d = {"_metadata": {"flashinfer_version": "0.7.4", "gpu": "NVIDIA H200"}}
    for stage in [1, 2]:
        key = f"('trtllm::fused_moe::gemm{stage}', 'MoERunner', ((8192, 5120), (48, 4608, 2560), (0,), (48, 5120, 1152), (0,)), {extras})"
        d[key] = ["MoERunner", 73 if stage == 1 else 189]
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(d))
    return p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_validated_chunk_cache(tmp_path):
    p = cache_file(tmp_path)
    assert module.validate_cache(p, sha(p)) == (p, 2)


def test_checksum_guard(tmp_path):
    p = cache_file(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        module.validate_cache(p, "0" * 64)


def test_unvalidated_tail_shape(tmp_path):
    p = cache_file(tmp_path)
    p.write_text(p.read_text().replace("8192", "128"))
    with pytest.raises(ValueError, match="unvalidated"):
        module.validate_cache(p, sha(p))


def test_draft_topk_rejected(tmp_path):
    p = cache_file(tmp_path)
    p.write_text(p.read_text().replace("6, 1, 8", "3, 1, 8"))
    with pytest.raises(ValueError, match="unvalidated"):
        module.validate_cache(p, sha(p))


def test_metadata_rejected(tmp_path):
    p = cache_file(tmp_path)
    p.write_text(p.read_text().replace("0.7.4", "0.7.3"))
    with pytest.raises(ValueError, match="requires"):
        module.validate_cache(p, sha(p))


def test_symlink_rejected(tmp_path):
    p = cache_file(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(p)
    with pytest.raises(ValueError, match="regular"):
        module.validate_cache(link, sha(p))


def test_finalize_mode_must_match_runtime(tmp_path):
    p = cache_file(tmp_path)
    with pytest.raises(ValueError, match="unvalidated"):
        module.validate_cache(p, sha(p), fused_finalize=True)
