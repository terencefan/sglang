"""CPU compatibility probe inside the current image; never patches live sources."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    root = args.root
    hooks = root / "cpu-hooks"
    hooks.mkdir(exist_ok=True)
    mapping = {
        "sglang.srt.managers.scheduler_components." + name: str(
            root / "python/sglang/srt/managers/scheduler_components" / f"{name}.py"
        )
        for name in (
            "dp_attn",
            "prefill_cost_model",
            "prefill_cost_runtime",
            "prefill_budget",
            "load_inquirer",
            "load_publisher",
        )
    }
    mapping.update(
        {
            "sglang.srt.managers." + name: str(
                root / "python/sglang/srt/managers" / f"{name}.py"
            )
            for name in ("schedule_policy", "scheduler")
        }
    )
    (hooks / "sitecustomize.py").write_text(
        """import importlib.abc,importlib.util,sys
MAPPING = """
        + repr(mapping)
        + """
class CandidateFinder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in MAPPING:return importlib.util.spec_from_file_location(fullname,MAPPING[fullname])
sys.meta_path.insert(0,CandidateFinder())
"""
    )
    import os

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "999",
        "PYTHONPATH": str(hooks),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SGLANG_PREFILL_TIME_BUDGET_MODE": "off",
    }
    tests = [
        str(
            root
            / "test/registered/unit/managers/scheduler_components"
            / f"test_{name}.py"
        )
        for name in ("prefill_cost_model", "prefill_cost_runtime", "dp_attn")
    ]
    tests += [
        str(root / "test/registered/unit/managers" / f"test_{name}.py")
        for name in (
            "prefill_adder",
            "schedule_policy",
            "load_inquirer",
            "loadstat_wire",
        )
    ]
    pytest_args = ["-q", *tests, "--disable-warnings", "--tb=short", "--assert=plain"]
    # pytest's rewriting finder can resolve installed modules before our source
    # finder. Preload and assert origins before importing pytest; no passing
    # baseline tests may be reported as candidate compatibility evidence.
    child = "import importlib,json,torch\nassert torch.cuda.device_count()==0\n"
    child += "mapping=" + repr(mapping) + "\n"
    child += "for name,path in mapping.items():\n m=importlib.import_module(name)\n assert m.__file__==path,(name,m.__file__,path)\n"
    child += "print('CANDIDATE_IMPORT_ORIGINS_VERIFIED',len(mapping),flush=True)\n"
    child += "import pytest\nraise SystemExit(pytest.main(" + repr(pytest_args) + "))\n"
    command = ["/opt/sglang/bin/python3", "-c", child]
    result = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    (root / "cpu-probe.log").write_text(result.stdout)
    print(result.stdout[-12000:])
    print(
        json.dumps(
            {
                "exit_code": result.returncode,
                "gpu_hidden": True,
                "source_root": str(root),
            }
        )
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
