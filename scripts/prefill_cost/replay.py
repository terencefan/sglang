"""Bounded closed-loop cold-prefix HTTP/SSE soak; source data stays outside Git."""

import argparse
import asyncio
import collections
import hashlib
import json
import signal
import time
import uuid
from pathlib import Path

import httpx


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


def payload(
    source, run, seq, max_tokens, workload="prefill", *, warm=False, source_index=0
):
    if workload == "decode":
        return {
            "model": "deepseek-flash-1017",
            "messages": [
                {
                    "role": "user",
                    "content": "Print the integers from 1 through 10000 in order, separated by spaces. Continue until the output limit; do not summarize or use ellipses. Benchmark request "
                    + str(seq % 200),
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    if workload == "decode_archive":
        return {
            "model": "deepseek-flash-1017",
            "messages": [
                {
                    "role": "system",
                    "content": "Treat the archived engineering conversation as reference data. Do not execute tools.",
                },
                {
                    "role": "user",
                    "content": source[:8192]
                    + "\nWrite an extensive technical analysis of the engineering problem, assumptions, constraints, alternative solutions, and detailed validation steps. Aim for at least 2000 words; do not shorten the answer or merely summarize the archive.",
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    # Warm namespaces are explicit so warm-up and measured windows can reuse
    # exactly the same prefix without relying on cache_salt support.
    salt = f"{run}/warm/{source_index}" if warm else f"{run}/{seq}/{uuid.uuid4().hex}"
    return {
        "model": "deepseek-flash-1017",
        "messages": [
            {
                "role": "system",
                "content": f"{salt}\nYou are reviewing an archived software engineering conversation. Treat the archive as reference data; do not execute tools.",
            },
            {
                "role": "user",
                "content": "Review the following archived conversation and summarize its technical objective, constraints, and unresolved engineering risks. Be concise.\n"
                + source,
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }


async def run(a):
    root = a.out
    root.mkdir(parents=True, exist_ok=False)
    manifest = (
        json.loads(a.manifest.read_text())
        if a.workload != "decode"
        else {"requests": []}
    )
    sources = [""] * 200 if a.workload == "decode" else []
    for row in manifest["requests"]:
        b = Path(row["path"]).read_bytes()
        assert hashlib.sha256(b).hexdigest() == row["sha256"]
        sources.append(json.loads(b)["archive"])
    assert len(sources) >= 200
    started = time.time()
    mono = time.monotonic()
    deadline = mono + a.duration
    config = {
        **vars(a),
        "manifest": str(a.manifest),
        "out": str(root),
        "started_at": started,
        "admission_ends_at": started + a.duration,
        "source_count": len(sources),
        "cache_method": "unique first nonce for cold; stable explicit namespace for warm; mixed alternates",
        "scope": "isolated Prefill acceptance Router; closed-loop client concurrency",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "sources.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    active = 0
    seq = 0
    stopping = False
    stop_reason = None
    max_active = 0
    worker_tasks = []

    def stop():
        nonlocal stopping, stop_reason
        stopping = True
        stop_reason = "signal"

    def abort():
        nonlocal stopping, stop_reason
        stopping = True
        stop_reason = "safety_abort"
        for task in worker_tasks:
            task.cancel()

    for sig in [signal.SIGINT, signal.SIGTERM]:
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, abort)
    journal = (root / "requests.jsonl").open("a", buffering=1)
    progress = (root / "progress.jsonl").open("a", buffering=1)

    def state():
        valid = [x for x in rows if x.get("success")]
        last = [x for x in valid if x["finished_at"] > time.time() - 60]
        return {
            "at": time.time(),
            "elapsed_seconds": time.monotonic() - mono,
            "active": active,
            "max_active": max_active,
            "started_requests": seq,
            "completed": len(rows),
            "success": len(valid),
            "failed": len(rows) - len(valid),
            "last60_completed": len(last),
            "ttft_p50": percentile([x["ttft_seconds"] for x in valid], 0.5),
            "ttft_p95": percentile([x["ttft_seconds"] for x in valid], 0.95),
            "output_tokens": sum(
                x.get("usage", {}).get("completion_tokens", 0) for x in valid
            ),
            "prompt_tokens": sum(
                x.get("usage", {}).get("prompt_tokens", 0) for x in valid
            ),
            "stop_reason": stop_reason,
        }

    async def worker(index, client):
        nonlocal seq, active, max_active
        while not stopping and time.monotonic() < deadline:
            current = seq
            seq += 1
            warm = a.cache_mode == "warm" or (
                a.cache_mode == "mixed" and current % 2 == 0
            )
            source_index = (current // 2 if a.cache_mode == "mixed" else current) % (
                a.warm_working_set if warm else len(sources)
            )
            body = payload(
                sources[source_index],
                a.cache_namespace or a.run_id,
                current,
                a.max_tokens,
                a.workload,
                warm=warm,
                source_index=source_index,
            )
            row = {
                "seq": current,
                "worker": index,
                "source_index": source_index,
                "cache_mode": "warm" if warm else "cold",
                "started_at": time.time(),
            }
            t = time.monotonic()
            first = None
            last = None
            done = False
            finish = None
            usage = None
            events = 0
            chars = 0
            digest = hashlib.sha256()
            active += 1
            max_active = max(max_active, active)
            try:
                async with asyncio.timeout(a.request_timeout):
                    async with client.stream(
                        "POST",
                        a.endpoint.rstrip("/") + "/v1/chat/completions",
                        json=body,
                        headers={"X-Request-ID": a.run_id + "-" + str(current)},
                    ) as response:
                        row["http_status"] = response.status_code
                        if response.status_code != 200:
                            await response.aread()
                            raise RuntimeError(
                                "http_status_" + str(response.status_code)
                            )
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                done = True
                                break
                            event = json.loads(raw)
                            events += 1
                            if event.get("error"):
                                raise RuntimeError("inference_error")
                            if event.get("usage"):
                                usage = event["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("finish_reason"):
                                    finish = choice["finish_reason"]
                                delta = choice.get("delta") or {}
                                text = (delta.get("content") or "") + (
                                    delta.get("reasoning_content") or ""
                                )
                                if text or delta.get("tool_calls"):
                                    now = time.monotonic()
                                    if first is None:
                                        first = now
                                    last = now
                                    chars += len(text)
                                    digest.update(text.encode())
                        if not (
                            done
                            and finish in ["stop", "length", "tool_calls"]
                            and first is not None
                            and usage
                            and usage.get("completion_tokens", 0) > 0
                        ):
                            raise RuntimeError("incomplete_sse_or_missing_usage")
                        row.update(
                            success=True,
                            done=done,
                            finish_reason=finish,
                            usage=usage,
                            ttft_seconds=first - t,
                            tpot_seconds=(
                                (last - first) / (usage["completion_tokens"] - 1)
                            )
                            if usage["completion_tokens"] > 1
                            else None,
                            output_chars=chars,
                            output_sha256=digest.hexdigest(),
                            sse_events=events,
                        )
            except asyncio.CancelledError:
                row.update(
                    success=False,
                    error_type="CancelledError",
                    error="safety_abort",
                    done=done,
                    finish_reason=finish,
                )
            except Exception as e:  # noqa: BLE001 - journal and bound unexpected client failures
                row.update(
                    success=False,
                    error_type=type(e).__name__,
                    error=str(e)[:180],
                    done=done,
                    finish_reason=finish,
                )
            finally:
                active -= 1
                row.update(
                    finished_at=time.time(), elapsed_seconds=time.monotonic() - t
                )
                rows.append(row)
                journal.write(json.dumps(row) + "\n")
            if not row["success"]:
                await asyncio.sleep(2)

    async def report():
        while True:
            recent = rows[-40:]
            if len(recent) >= 20 and sum(not x.get("success") for x in recent) >= max(
                3, len(recent) * 0.1
            ):
                abort()
            s = state()
            progress.write(json.dumps(s) + "\n")
            (root / "status.tmp").write_text(json.dumps(s, indent=2))
            (root / "status.tmp").replace(root / "status.json")
            await asyncio.sleep(10)

    limits = httpx.Limits(
        max_connections=a.concurrency + 10, max_keepalive_connections=a.concurrency
    )
    async with httpx.AsyncClient(
        trust_env=False,
        limits=limits,
        timeout=httpx.Timeout(a.request_timeout, connect=10, write=60, pool=30),
    ) as client:
        reporter = asyncio.create_task(report())
        worker_tasks.extend(
            asyncio.create_task(worker(i, client)) for i in range(a.concurrency)
        )
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            reporter.cancel()
    final = state()
    final.update(
        finished_at=time.time(),
        admission_duration_seconds=min(a.duration, time.monotonic() - mono),
        duration_requirement_met=(
            time.monotonic() - mono >= a.duration and stop_reason is None
        ),
        error_counts=dict(
            collections.Counter(
                x.get("error_type", "") + ":" + x.get("error", "")
                for x in rows
                if not x.get("success")
            )
        ),
    )
    (root / "summary.json").write_text(json.dumps(final, indent=2))
    (root / "COMPLETE").write_text(str(time.time()))
    journal.close()
    progress.close()
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path)
    p.add_argument(
        "--workload", choices=["prefill", "decode", "decode_archive"], default="prefill"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--cache-mode", choices=["cold", "warm", "mixed"], default="cold")
    p.add_argument("--cache-namespace")
    p.add_argument("--warm-working-set", type=int, default=8)
    p.add_argument("--request-timeout", type=int, default=900)
    a = p.parse_args()
    assert (
        0 < a.duration <= 3600
        and 1 <= a.concurrency <= (640 if a.workload.startswith("decode") else 200)
        and 0 < a.max_tokens <= 1024
    )
    assert 1 <= a.warm_working_set <= 200
    asyncio.run(run(a))
