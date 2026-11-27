"""CPU-only calibration acceptance guards; no Torch or serving imports."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "calibration_under_test", Path(__file__).with_name("calibrate.py")
)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def samples():
    rows = []
    for i in range(200):
        q, prefix = 128 * (1 + i % 32), 1024 * (1 + i % 17)
        rows.append(
            {
                "signature": "h200",
                "rank": i % 8,
                "boot": f"rank-{i % 8}-boot-a",
                "hicache_pending": False,
                "cpu_launch_ns": (i + 1) * 10_000_000_000,
                "global_num_tokens": [q] * 8,
                "work": [{"rows": q, "prefix": prefix}],
                "gpu_stream_ms": 1 + 0.001 * q + 0.000001 * q * prefix,
            }
        )
    return rows


def test_launch_clock_orders_interleaved_rank_logs_without_mutating_input():
    ordered = samples()
    interleaved = ordered[::2] + ordered[1::2]
    before = [s["cpu_launch_ns"] for s in interleaved]
    expected = calibration.fit(ordered, "h200")
    actual = calibration.fit(interleaved, "h200")
    assert actual == expected
    assert actual["temporal_holdout_verified"]
    assert actual["holdout_gap_ms_each_side"] >= 1000
    assert actual["p95_error"] < 0.05
    assert actual["train_samples"] == 140
    assert actual["test_samples"] == 59
    assert not actual["serving_gate"]
    assert [s["cpu_launch_ns"] for s in interleaved] == before


@pytest.mark.parametrize("missing_count", [1, 200])
def test_missing_clock_cannot_silently_use_log_order(missing_count):
    rows = samples()
    for row in rows[:missing_count]:
        row.pop("cpu_launch_ns")
    with pytest.raises(ValueError, match="requires launch clocks"):
        calibration.fit(rows, "h200")


@pytest.mark.parametrize("invalid", [True, -1, 0, 1.5, 2**63])
def test_invalid_clock_rejected(invalid):
    rows = samples()
    rows[0]["cpu_launch_ns"] = invalid
    with pytest.raises(ValueError, match="invalid launch clock"):
        calibration.fit(rows, "h200")


def test_simultaneous_rank_records_do_not_form_a_temporal_holdout():
    rows = samples()
    for row in rows:
        row["cpu_launch_ns"] = 10_000_000_000
    with pytest.raises(ValueError, match="timing-gap-separated"):
        calibration.fit(rows, "h200")


def test_legacy_without_clocks_is_diagnostic_only():
    rows = samples()
    for row in rows:
        row.pop("cpu_launch_ns")
    artifact = calibration.fit(rows, "h200", shared=False)
    assert artifact["p95_error"] < 0.05
    assert not artifact["temporal_holdout_verified"]
    assert not artifact["prediction_gate"]
    assert not artifact["serving_gate"]


@pytest.mark.parametrize("key", ["rows", "prefix", "replay"])
def test_negative_work_rejected(key):
    rows = samples()
    rows[0]["work"][0][key] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        calibration.fit(rows, "h200")


def test_empty_training_work_cannot_produce_nan_baseline():
    rows = samples()
    for row in rows:
        row["work"] = []
    with pytest.raises(ValueError, match="positive training work"):
        calibration.fit(rows, "h200")


def test_same_rank_restart_cannot_be_merged_into_calibration():
    rows = samples()
    rows[-1]["boot"] = "rank-7-boot-b"
    with pytest.raises(ValueError, match="mixed rank boots"):
        calibration.fit(rows, "h200")


def test_missing_rank_cannot_pass_on_only_the_fastest_ranks():
    rows = [s for s in samples() if s["rank"] != 7]
    with pytest.raises(ValueError, match="requires all DP ranks"):
        calibration.fit(rows, "h200")


def test_overall_rank_coverage_does_not_replace_holdout_rank_coverage():
    rows = samples()
    for row in rows[140:]:
        if row["rank"] == 7:
            row.update(rank=0, boot="rank-0-boot-a")
    with pytest.raises(ValueError, match="both training and holdout"):
        calibration.fit(rows, "h200")
