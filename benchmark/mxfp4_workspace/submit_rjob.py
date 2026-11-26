"""Submit only to the explicit public task pool; dry run by default."""

import argparse
import json
import subprocess
from pathlib import Path

IMAGE = "registry.h.pjlab.org.cn/ailab-dataprojects/deepseek-sglang@sha256:9f1d906c44bc672a27187f5713027e61da44cd9cefd24646bbd025fa0d9d3b7d"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--cpu", type=int, choices=[2, 4, 8], default=8)
    a = p.parse_args()
    if not a.root.is_absolute() or not all(
        (a.root / name).is_file()
        for name in ["run_rjob.py", "validate_gpu.py", "mxfp4_workspace.py"]
    ):
        p.error("scripts must exist on mounted GPFS")
    cmd = [
        "rjob",
        "submit",
        "--name",
        a.name,
        "--namespace",
        "ailab-datatools",
        "--charged-group",
        "data_tools_gpu_pool",
        "--image",
        IMAGE,
        "--replicas",
        "1",
        "--cpu",
        str(a.cpu),
        "--memory",
        "32768",
        "--gpu",
        "1",
        "--preemptible",
        "best-effort",
        "--restart-policy",
        "never",
        "--auto-delete-duration",
        "24h",
        "--mount",
        "gpfs://gpfs2/xyz-data-share:/mnt/shared-storage-gpfs2/xyz-data-share",
        "--",
        "/opt/sglang/bin/python3",
        "-u",
        str(a.root / "run_rjob.py"),
        "--root",
        str(a.root),
    ]
    print(json.dumps({"command": cmd, "apply": a.apply}), flush=True)
    if a.apply:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
