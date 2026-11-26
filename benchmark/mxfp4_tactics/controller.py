"""Bounded, one-shot maintenance job; restore verified boot before GPU work."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic_json(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def restore_boot(root, plan):
    target, saved = Path(plan["boot_path"]), root / "original-boot.py"
    if digest(saved) != plan["original_boot_sha256"]:
        raise RuntimeError("original boot checksum mismatch")
    if digest(target) not in (plan["original_boot_sha256"], plan["wrapper_sha256"]):
        raise RuntimeError("refusing unknown boot change")
    tmp = target.with_suffix(".restore-tmp")
    tmp.write_bytes(saved.read_bytes())
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)


def stop_children(children):
    for p, log in children:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for p, log in children:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
        log.close()


def run_matrix(root, plan):
    if socket.gethostname() != plan["expected_hostname"]:
        raise RuntimeError("unexpected GPU host")
    lines = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    devices = []
    for line in lines:
        uuid, name, used = [x.strip() for x in line.split(",")]
        if "H200" not in name or float(used) >= 1024:
            raise RuntimeError("GPU not idle after stopping Prefill")
        devices.append(uuid)
    if len(devices) != 8 or len(set(devices)) != 8:
        raise RuntimeError("requires eight assigned H200s")
    atomic_json(root / "devices.json", {"gpu_uuids": devices})
    for name, sha in plan["code_sha256"].items():
        if digest(root / name) != sha:
            raise RuntimeError("test source checksum mismatch")
    deadline = time.monotonic() + 1800
    next_sample = 0.0
    for cap in [8192, 32768, 65536]:
        children = []
        atomic_json(
            root / "state.json", {"phase": "testing", "cap": cap, "at": time.time()}
        )
        try:
            for gpu in devices:
                log = (root / (gpu + "-cap" + str(cap) + ".log")).open("ab")
                env = {
                    **os.environ,
                    "MAX_JOBS": "1",
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                }
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(root / "worker.py"),
                        "--root",
                        str(root),
                        "--gpu",
                        gpu,
                        "--cap",
                        str(cap),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append((p, log))
            while any(p.poll() is None for p, _ in children):
                if any(p.poll() not in (None, 0) for p, _ in children):
                    raise RuntimeError(
                        "worker failed; stop remaining workers and restore serving"
                    )
                if time.monotonic() > deadline or (root / "ABORT").exists():
                    raise RuntimeError("bounded test deadline or explicit abort")
                if time.monotonic() >= next_sample:
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
                                    {"at": time.time(), "cap": cap, "csv": csv.strip()}
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    next_sample = time.monotonic() + 5
                time.sleep(1)
            if any(p.returncode != 0 for p, _ in children):
                raise RuntimeError("worker failed")
        finally:
            stop_children(children)
    (root / "_COMPLETE").write_text(str(time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    # Restore before testing, so any later container restart serves baseline.
    restore_boot(root, plan)
    if not (root / "CLAIMED").exists():
        (root / "CLAIMED").open("x").close()
        try:
            run_matrix(root, plan)
        except Exception as exc:
            atomic_json(root / "failure.json", {"error": str(exc), "at": time.time()})
            (root / "_FAILED").write_text(str(time.time()))
    atomic_json(root / "state.json", {"phase": "restoring-serving", "at": time.time()})
    os.execv(sys.executable, [sys.executable, "-B", "-u", plan["boot_path"]])


if __name__ == "__main__":
    main()
