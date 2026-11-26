"""Persistent maintenance entry: resumable experiments, explicit serving transition."""

import argparse
import fcntl
import hashlib
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    plan = json.loads((root / "suite-plan.json").read_text())
    stop = threading.Event()
    state = {"phase": "starting", "at": time.time()}

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {
                    "role": "maintenance-experiment",
                    "inference_available": False,
                    **state,
                }
            ).encode()
            self.send_response(
                200 if self.path in ["/health", "/health_generate", "/"] else 503
            )
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def publish(phase, **kw):
        state.clear()
        state.update(phase=phase, at=time.time(), **kw)
        write(root / "suite-state.json", state)

    def mode_boot():
        mode = (
            json.loads((root / "mode.json").read_text())
            if (root / "mode.json").exists()
            else {}
        )
        if mode.get("mode") == "serve":
            boot = mode["boot"]
            assert boot in plan["allowed_serving_boots"]
            server.shutdown()
            os.execv(sys.executable, [sys.executable, "-B", "-u", boot])

    mode_boot()
    assert socket.gethostname() == plan["expected_hostname"]
    for name, sha in plan["sha256"].items():
        assert hashlib.sha256((root / "code" / name).read_bytes()).hexdigest() == sha, (
            name
        )
    ledger = (
        json.loads((root / "ledger.json").read_text())
        if (root / "ledger.json").exists()
        else {}
    )
    # A single ModelSet process owns this experiment allocation.
    lease = (root / "suite.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)
    inventory = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines()
    )
    assert len(inventory) == 8 and all("H200" in x for x in inventory)
    devices = [x.split(",")[0].strip() for x in inventory]

    publish("waiting-for-idle-devices")
    for attempt in range(90):
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        if not active:
            break
        time.sleep(1)
    else:
        publish("blocked", reason="assigned GPUs still have compute processes")
        while not stop.wait(5):
            mode_boot()
        return

    def sample():
        while not stop.wait(5):
            try:
                csv = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,utilization.gpu,memory.used,memory.free,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                with (root / "gpu-samples.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "at": time.time(),
                                "phase": state.copy(),
                                "csv": csv.strip(),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

    threading.Thread(target=sample, daemon=True).start()
    for job in plan["jobs"]:
        name = job["name"]
        old = ledger.get(name, {})
        if old.get("status") in ["complete", "failed"]:
            continue
        attempt = old.get("attempt", 0) + 1
        if attempt > 2:
            ledger[name] = {"status": "failed", "error": "two attempts interrupted"}
            write(root / "ledger.json", ledger)
            continue
        trial = root / "runs" / name / str(attempt)
        trial.mkdir(parents=True, exist_ok=True)
        write(trial / "devices.json", {"gpu_uuids": devices})
        children = []
        ledger[name] = {
            "status": "running",
            "attempt": attempt,
            "started_at": time.time(),
            "trial": str(trial),
        }
        write(root / "ledger.json", ledger)
        publish("testing", job=name, attempt=attempt)
        try:
            for index, command in enumerate(job["commands"]):
                args = [
                    s.replace("{trial}", str(trial)).replace("{root}", str(root))
                    for s in command
                ]
                log = (trial / f"process{index}.log").open("ab")
                env = {
                    **os.environ,
                    "MAX_JOBS": "1",
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                }
                child = subprocess.Popen(
                    [sys.executable, "-B", "-u", *args],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append((child, log))
            deadline = min(
                time.time() + job["timeout"], plan["experiment_deadline_unix"]
            )
            while any(c.poll() is None for c, _ in children):
                if stop.is_set() or (root / "ABORT").exists() or time.time() > deadline:
                    raise RuntimeError("job cancelled or bounded deadline")
                if any(c.poll() not in (None, 0) for c, _ in children):
                    raise RuntimeError("one worker failed")
                time.sleep(1)
            if any(c.returncode != 0 for c, _ in children):
                raise RuntimeError("worker failed")
            ledger[name].update(status="complete", finished_at=time.time())
        except Exception as exc:
            ledger[name].update(
                status="failed", error=str(exc), finished_at=time.time()
            )
        finally:
            for c, _ in children:
                if c.poll() is None:
                    try:
                        os.killpg(c.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for c, log in children:
                try:
                    c.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(c.pid, signal.SIGKILL)
                    c.wait(timeout=15)
                log.close()
            write(root / "ledger.json", ledger)
        if (
            stop.is_set()
            or (root / "ABORT").exists()
            or time.time() > plan["experiment_deadline_unix"]
        ):
            break
    publish("experiments-finished", jobs=ledger)
    (root / "EXPERIMENTS_FINISHED").write_text(str(time.time()))
    # Do not resume serving during the user-specified maintenance window.
    while not stop.wait(5):
        mode_boot()
    server.shutdown()


if __name__ == "__main__":
    main()
