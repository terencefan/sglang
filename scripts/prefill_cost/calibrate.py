"""Fit nonnegative Prefill timing coefficients from chronological shadow logs.

Usage: python calibrate.py samples.log --signature HASH --output model.json
GPU stream span is not isolated kernel time; do not mix cache-load samples.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def nnls(x, y):
    """Nonnegative coordinate descent on column-scaled CPU features."""
    coefficients = np.zeros(x.shape[1])
    residual = y.copy()
    for _ in range(10000):
        previous = coefficients.copy()
        for j in range(x.shape[1]):
            col = x[:, j]
            norm = col @ col
            if norm == 0:
                continue
            value = max(0.0, coefficients[j] + (col @ residual) / norm)
            residual += col * (coefficients[j] - value)
            coefficients[j] = value
        if np.max(np.abs(previous - coefficients)) < 1e-9:
            break
    return coefficients, float(np.linalg.norm(residual))


def features(work, global_num_tokens=None):
    q = [w["rows"] + w.get("replay", 0) for w in work]
    result = [
        1,
        sum(q),
        sum(x * x for x in q),
        sum(x * w["prefix"] for x, w in zip(q, work)),
        len(work),
    ]
    if global_num_tokens is not None:
        if not global_num_tokens or any(
            not isinstance(x, int) or x < 0 for x in global_num_tokens
        ):
            raise ValueError("invalid shared DP rows")
        result += [sum(global_num_tokens), max(global_num_tokens)]
    return result


def fit(samples, signature, *, shared=True):
    samples = [
        s for s in samples if s["signature"] == signature and not s["hicache_pending"]
    ]
    if len(samples) < 100:
        raise ValueError("at least 100 same-signature non-reload samples required")
    if all("cpu_launch_ns" in s for s in samples):
        samples.sort(key=lambda s: s["cpu_launch_ns"])
    degree = 1
    if shared:
        shapes = [s.get("global_num_tokens") for s in samples]
        if (
            any(not shape for shape in shapes)
            or len({len(shape) for shape in shapes}) != 1
        ):
            raise ValueError("shared model requires consistent DP rows on every sample")
        degree = len(shapes[0])
        if not 1 <= degree <= 64:
            raise ValueError("invalid DP degree")
    split = int(len(samples) * 0.7)
    x = np.asarray(
        [
            features(s["work"], s["global_num_tokens"] if shared else None)
            for s in samples
        ],
        dtype=float,
    )
    train = np.arange(len(samples)) < split
    test = ~train
    gap_ms = 0
    if all("cpu_launch_ns" in s for s in samples):
        clock = np.asarray([s["cpu_launch_ns"] for s in samples], dtype=np.int64)
        gap_ms = max(1000, 2 * max(s["gpu_stream_ms"] for s in samples))
        boundary = clock[split]
        train = clock < boundary - gap_ms * 1000000
        test = clock > boundary + gap_ms * 1000000
        if train.sum() < 50 or test.sum() < 20:
            raise ValueError("not enough timing-gap-separated holdout samples")
    y = np.asarray([s["gpu_stream_ms"] for s in samples], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("invalid timing samples")
    scale = np.maximum(x[train].max(axis=0), 1)
    coefficients, _ = nnls(x[train] / scale, y[train])
    coefficients /= scale
    err = np.abs(x[test] @ coefficients - y[test]) / y[test]
    # Fair baseline: least-squares ms/row on the identical training split.
    rate = (x[train, 1] @ y[train]) / (x[train, 1] @ x[train, 1])
    old = np.abs(x[test, 1] * rate - y[test]) / y[test]
    p95, old95 = float(np.quantile(err, 0.95)), float(np.quantile(old, 0.95))
    return {
        "schema": 2 if shared else 1,
        "feature_contract": "shared-dp-v2" if shared else "local-v1",
        "dp_degree": degree,
        "holdout_gap_ms_each_side": gap_ms,
        "signature": signature,
        "coefficients": coefficients.tolist(),
        "samples": len(samples),
        "train_samples": int(train.sum()),
        "test_samples": int(test.sum()),
        "p50_error": float(np.quantile(err, 0.5)),
        "p95_error": p95,
        "baseline_p95_error": old95,
        "prediction_gate": p95 <= 0.2 and p95 <= 0.8 * old95,
        "serving_gate": False,
        "measurement": "scheduler_forward_stream_span",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--signature", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--legacy-local",
        action="store_true",
        help="DP1 diagnostic only; DP serving rejects local-only artifacts",
    )
    args = p.parse_args()
    samples = []
    for line in Path(args.input).read_text().splitlines():
        if "PREFILL_COST_SAMPLE " in line:
            samples.append(json.loads(line.split("PREFILL_COST_SAMPLE ", 1)[1]))
    artifact = fit(samples, args.signature, shared=not args.legacy_local)
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact))


if __name__ == "__main__":
    main()
