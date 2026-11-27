"""Stage a hash-verified, immutable source bundle without editing live sources."""

import argparse
import base64
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

FILES = [
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/schedule_policy.py",
    "python/sglang/srt/managers/scheduler_components/dp_attn.py",
    "python/sglang/srt/managers/scheduler_components/load_inquirer.py",
    "python/sglang/srt/managers/scheduler_components/load_publisher.py",
    "test/registered/unit/managers/test_prefill_adder.py",
    "test/registered/unit/managers/test_schedule_policy.py",
    "test/registered/unit/managers/test_load_inquirer.py",
    "test/registered/unit/managers/test_loadstat_wire.py",
    "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py",
    "python/sglang/srt/managers/scheduler_components/prefill_cost_runtime.py",
    "python/sglang/srt/managers/scheduler_components/prefill_budget.py",
    "scripts/prefill_cost/calibrate.py",
    "test/registered/unit/managers/scheduler_components/test_prefill_cost_model.py",
    "test/registered/unit/managers/scheduler_components/test_prefill_cost_runtime.py",
    "test/registered/unit/managers/scheduler_components/test_dp_attn.py",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--destination", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in FILES
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for rel in FILES:
            t.add(root / rel, arcname=rel)
    payload = base64.b64encode(buf.getvalue()).decode()
    code = """import base64,hashlib,io,json,tarfile
from pathlib import Path
root=Path(DESTINATION)
assert root.is_absolute() and not root.exists()
root.mkdir(parents=True,mode=0o700)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))) as t:
 for item in t.getmembers():
  assert item.isfile() and item.name in MANIFEST
 t.extractall(root,filter='data')
for rel,digest in MANIFEST.items():
 assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
(root/'source-manifest.json').write_text(json.dumps(MANIFEST,indent=2))
print(json.dumps({'staged':str(root),'files':len(MANIFEST)}))
"""
    code = (
        "DESTINATION="
        + repr(args.destination)
        + "\nPAYLOAD="
        + repr(payload)
        + "\nMANIFEST="
        + repr(manifest)
        + "\n"
        + code
    )
    subprocess.run(
        ["ssh", args.host, "python3", "-"], input=code, text=True, check=True
    )


if __name__ == "__main__":
    main()
