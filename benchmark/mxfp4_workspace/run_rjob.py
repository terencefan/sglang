"""Bounded isolated GPU validation, with terminal markers and power samples."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    # Synthetic kernel validation needs only a small host submission/JIT pool.
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    out = a.root / "output"
    out.mkdir(exist_ok=False)
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            try:
                data = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.free,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                samples.append({"at": time.time(), "csv": data.strip()})
            except Exception as e:
                samples.append({"at": time.time(), "error": type(e).__name__})
            stop.wait(1)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": time.time(),
                "host": os.uname().nodename,
                "scope": "one isolated H200; synthetic per-EP shard, no serving changes",
                "sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in a.root.glob("*.py")
                },
            },
            indent=2,
        )
    )
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with (out / "validation.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(a.root / "validate_gpu.py"),
                    "--out",
                    str(out / "validation"),
                    "--workspace-module",
                    str(a.root / "mxfp4_workspace.py"),
                    "--run-on-idle-gpu",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=True,
            )
        assert (out / "validation/_SUCCESS").exists()
        (out / "_SUCCESS").write_text(str(time.time()))
    except BaseException:
        (out / "error.txt").write_text(traceback.format_exc())
        (out / "_FAILED").write_text(str(time.time()))
        raise
    finally:
        stop.set()
        thread.join(timeout=8)
        (out / "gpu-samples.json").write_text(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
