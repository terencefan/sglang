"""Compose a reversible candidate with the existing serving overlay; no restart."""

import argparse
import base64
import hashlib
import subprocess
from pathlib import Path

SOURCES = [
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/schedule_policy.py",
    "python/sglang/srt/managers/scheduler_components/dp_attn.py",
    "python/sglang/srt/managers/scheduler_components/load_inquirer.py",
    "python/sglang/srt/managers/scheduler_components/load_publisher.py",
    "python/sglang/srt/managers/scheduler_components/prefill_cost_model.py",
    "python/sglang/srt/managers/scheduler_components/prefill_cost_runtime.py",
    "python/sglang/srt/managers/scheduler_components/prefill_budget.py",
]
BASE = "e96e96757018ce842d570781ec8312f8c470493b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--previous-root", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--generation", required=True, type=int)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCES], cwd=repo, text=True
    )
    if dirty:
        raise SystemExit("commit candidate sources before staging an online bundle")
    files = []
    for rel in SOURCES:
        candidate = (repo / rel).read_bytes()
        original = subprocess.run(
            ["git", "show", BASE + ":" + rel],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        if original.returncode and rel not in [
            p
            for p in SOURCES
            if "prefill_cost_" in p or p.endswith("/prefill_budget.py")
        ]:
            raise SystemExit("missing baseline source: " + rel)
        baseline = original.stdout if original.returncode == 0 else None
        files.append(
            {
                "path": rel,
                "base_sha256": hashlib.sha256(baseline).hexdigest()
                if baseline is not None
                else None,
                "sha256": hashlib.sha256(candidate).hexdigest(),
                "baseline": base64.b64encode(baseline).decode()
                if baseline is not None
                else None,
                "candidate": base64.b64encode(candidate).decode(),
            }
        )
    settings = {
        "previous": args.previous_root,
        "destination": args.destination,
        "generation": args.generation,
        "commit": commit,
        "base": BASE,
        "files": files,
    }
    remote = """import base64,hashlib,json,os,shutil
from pathlib import Path
cfg=SETTINGS
old,new=Path(cfg['previous']),Path(cfg['destination'])
assert old.is_dir() and old.parent==new.parent and not new.exists()
previous=json.loads((old/'desired.json').read_text())
assert previous['role']=='prefill' and previous['source_commit']==cfg['base']
assert cfg['generation']>previous['generation']
manifest=json.loads((old/'overlay-manifest.json').read_text())
assert manifest['base_commit']==cfg['base']
old_paths={row['path'] for row in manifest['files']}
assert not old_paths.intersection(row['path'] for row in cfg['files'])
new.mkdir(mode=0o700)
for name in ['launch.py','online_overlay.py','restore_jit_cache.py','boot.py','rdma_metrics.py','forward_batch_metrics.py','.profile-admin.key','prefill-tactics-8192.json']:
 shutil.copy2(old/name,new/name)
assert (new/'.profile-admin.key').stat().st_mode & 0o077 == 0
for kind in ['baseline','candidate']:shutil.copytree(old/kind,new/kind)
for row in cfg['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 for kind in ['baseline','candidate']:
  if row[kind] is None:continue
  data=base64.b64decode(row[kind]);expected=row['base_sha256' if kind=='baseline' else 'sha256']
  assert hashlib.sha256(data).hexdigest()==expected
  target=new/kind/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  compile(data,str(target),'exec')
 manifest['files'].append({k:row[k] for k in ['path','base_sha256','sha256']})
manifest['previous_root']=str(old);manifest['patch_commit']=cfg['commit']
(new/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2))
previous['generation']=cfg['generation'];previous['patch_commit']=cfg['commit']
previous['profile']='prefill-time-budget-acceptance'
control='/tmp/prefill-cost-control-20261127.json'
previous['environment'].update({'SGLANG_PREFILL_TIME_BUDGET_MODE':'shadow',
 'SGLANG_PREFILL_COST_SAMPLE_INTERVAL':'32','SGLANG_PREFILL_COST_CONTROL':control})
(new/'desired.json').write_text(json.dumps(previous,indent=2))
(new/'control-initial.json').write_text(json.dumps({'generation':1,'mode':'off','fixed_rows':8192}))
boot=(new/'boot.py').read_text()
anchor="r=Path(__file__).resolve().parent"
assert boot.count(anchor)==1
init=chr(10).join([
 "", "control=Path("+repr(control)+")",
 "tmp=control.with_name(control.name+'.tmp-'+str(os.getpid()))",
 "fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)",
 "with os.fdopen(fd,'wb') as f:f.write((r/'control-initial.json').read_bytes())",
 "os.replace(tmp,control)", ""])
boot=boot.replace(anchor,anchor+init)
compile(boot,str(new/'boot.py'),'exec');(new/'boot.py').write_text(boot)
receipt={'root':str(new),'previous_root':str(old),'generation':cfg['generation'],
 'source_commit':cfg['commit'],'files':len(manifest['files']),
 'command':['python3 -B -u '+str(new/'boot.py')+' --root '+str(new)],
 'public_ingress_changed':False,'live_sources_changed':False}
(new/'stage-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
""".replace("SETTINGS", repr(settings), 1)
    subprocess.run(
        ["ssh", args.host, "python3", "-"], input=remote, text=True, check=True
    )


if __name__ == "__main__":
    main()
