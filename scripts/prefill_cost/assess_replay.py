"""Assess complete paired windows; faster throughput cannot hide failures/tails."""

import argparse
import hashlib
import json
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


def bucket(tokens):
    if tokens < 16384:
        return "lt16k"
    if tokens < 65536:
        return "16k-64k"
    if tokens < 262144:
        return "64k-256k"
    return "ge256k"


def summarize(root):
    summary = json.loads((root / "summary.json").read_text())
    rows = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    good = [r for r in rows if r.get("success")]
    elapsed = (
        summary["finished_at"]
        - json.loads((root / "config.json").read_text())["started_at"]
    )
    bins = {}
    for name in ("lt16k", "16k-64k", "64k-256k", "ge256k"):
        selected = [r for r in good if bucket(r["usage"]["prompt_tokens"]) == name]
        bins[name] = {
            "count": len(selected),
            "ttft_p95": percentile([r["ttft_seconds"] for r in selected], 0.95),
            "completion_p95": percentile(
                [r["elapsed_seconds"] for r in selected], 0.95
            ),
        }
    complete = (
        summary.get("duration_requirement_met", False)
        and summary.get("active") == 0
        and summary.get("failed") == 0
        and len(good) == len(rows)
        and len(rows) == summary["started_requests"]
        and bool(rows)
    )
    config = json.loads((root / "config.json").read_text())
    return {
        "workload": config.get("workload"),
        "concurrency": config["concurrency"],
        "max_tokens": config["max_tokens"],
        "cache_mode": config.get("cache_mode", "cold"),
        "source_manifest_sha256": hashlib.sha256(
            (root / "sources.json").read_bytes()
        ).hexdigest(),
        "complete": complete,
        "seconds_including_drain": elapsed,
        "prompt_tokens_per_s": sum(r["usage"]["prompt_tokens"] for r in good) / elapsed,
        "requests": len(rows),
        "failed": len(rows) - len(good),
        "bins": bins,
    }


def assess(base, candidate, *, quality_passed, gpu_health_passed):
    reasons = []
    for key in (
        "workload",
        "concurrency",
        "max_tokens",
        "cache_mode",
        "source_manifest_sha256",
    ):
        if base.get(key) != candidate.get(key):
            reasons.append("comparison_mismatch_" + key)
    if not base["complete"] or not candidate["complete"]:
        reasons.append("incomplete_or_failed_window")
    gain = candidate["prompt_tokens_per_s"] / base["prompt_tokens_per_s"] - 1
    if gain < 0.05:
        reasons.append("cold_throughput_gain_below_5pct")
    for name in ("lt16k", "ge256k"):
        b, c = base["bins"][name], candidate["bins"][name]
        if min(b["count"], c["count"]) < 20:
            reasons.append(name + "_insufficient_tail_samples")
        for metric in ("ttft_p95", "completion_p95"):
            if b[metric] is None or c[metric] is None or c[metric] > 1.05 * b[metric]:
                reasons.append(name + "_" + metric + "_regression_or_missing")
    if not quality_passed:
        reasons.append("quality_not_accepted")
    if not gpu_health_passed:
        reasons.append("gpu_health_not_accepted")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "throughput_gain": gain,
        "baseline": base,
        "candidate": candidate,
        "note": "Client TTFT includes PD handoff and first Decode; completion includes bounded output.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--quality-evidence", type=Path, required=True)
    p.add_argument("--gpu-evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = assess(
        summarize(args.baseline),
        summarize(args.candidate),
        quality_passed=json.loads(args.quality_evidence.read_text()).get("accepted")
        is True,
        gpu_health_passed=json.loads(args.gpu_evidence.read_text()).get("accepted")
        is True,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": result["accepted"], "reasons": result["reasons"]}))


if __name__ == "__main__":
    main()
