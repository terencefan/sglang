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


def features(work):
    q = [w["rows"] + w.get("replay", 0) for w in work]
    return [
        1,
        sum(q),
        sum(x * x for x in q),
        sum(x * w["prefix"] for x, w in zip(q, work)),
        len(work),
    ]


def fit(samples, signature):
    samples = [
        s for s in samples if s["signature"] == signature and not s["hicache_pending"]
    ]
    if len(samples) < 100:
        raise ValueError("at least 100 same-signature non-reload samples required")
    split = int(len(samples) * 0.7)
    x = np.asarray([features(s["work"]) for s in samples], dtype=float)
    y = np.asarray([s["gpu_stream_ms"] for s in samples], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("invalid timing samples")
    scale = np.maximum(x[:split].max(axis=0), 1)
    coefficients, _ = nnls(x[:split] / scale, y[:split])
    coefficients /= scale
    err = np.abs(x[split:] @ coefficients - y[split:]) / y[split:]
    # Fair baseline: least-squares ms/row on the identical training split.
    rate = (x[:split, 1] @ y[:split]) / (x[:split, 1] @ x[:split, 1])
    old = np.abs(x[split:, 1] * rate - y[split:]) / y[split:]
    p95, old95 = float(np.quantile(err, 0.95)), float(np.quantile(old, 0.95))
    return {
        "schema": 1,
        "signature": signature,
        "coefficients": coefficients.tolist(),
        "samples": len(samples),
        "train_samples": split,
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
    args = p.parse_args()
    samples = []
    for line in Path(args.input).read_text().splitlines():
        if "PREFILL_COST_SAMPLE " in line:
            samples.append(json.loads(line.split("PREFILL_COST_SAMPLE ", 1)[1]))
    artifact = fit(samples, args.signature)
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact))


if __name__ == "__main__":
    main()
