"""Hash-guarded reversible source overlay for an explicitly authorized online test."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(artifact, source):
    artifact, source = Path(artifact), Path(source)
    manifest = json.loads((artifact / "overlay-manifest.json").read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != manifest["base_commit"]:
        raise ValueError("unexpected image source commit")
    dirty = set(
        subprocess.check_output(
            ["git", "diff", "--name-only"], cwd=source, text=True
        ).splitlines()
    )
    expected = {
        r["path"]
        for r in manifest["files"]
        if r["base_sha256"] is not None and r["base_sha256"] != r["sha256"]
    }
    if dirty != expected:
        raise ValueError(
            "unexpected modified source paths: " + repr(sorted(dirty ^ expected))
        )
    for row in manifest["files"]:
        rel = Path(row["path"])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or digest(source / rel) != row["sha256"]
        ):
            raise ValueError("candidate source hash mismatch")
    return manifest


def apply(artifact, source):
    artifact, source = Path(artifact), Path(source)
    manifest = json.loads((artifact / "overlay-manifest.json").read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != manifest["base_commit"]:
        raise ValueError("unexpected image source commit")
    allowed = {r["path"] for r in manifest["files"] if r["base_sha256"] is not None}
    dirty = set(
        subprocess.check_output(
            ["git", "diff", "--name-only"], cwd=source, text=True
        ).splitlines()
    )
    if not dirty <= allowed:
        raise ValueError("refusing unknown source modifications")
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("invalid overlay path")
        target = source / rel
        candidate = artifact / "candidate" / rel
        if target.is_symlink() or not target.resolve().is_relative_to(source.resolve()):
            raise ValueError("overlay target escapes source")
        current = digest(target) if target.exists() else None
        if (
            current not in (row["base_sha256"], row["sha256"])
            or digest(candidate) != row["sha256"]
        ):
            raise ValueError("baseline or candidate hash mismatch: " + row["path"])
    # Check every input before mutating any source file. Re-entry is idempotent.
    for row in manifest["files"]:
        target = source / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) == row["sha256"]:
            continue
        tmp = target.with_name(target.name + ".overlay-" + str(os.getpid()))
        tmp.write_bytes((artifact / "candidate" / row["path"]).read_bytes())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    verify(artifact, source)
    (artifact / "overlay-applied.json").write_text(
        json.dumps(
            {
                "base_commit": manifest["base_commit"],
                "patch_commit": manifest["patch_commit"],
                "files": manifest["files"],
            },
            indent=2,
        )
    )


def restore(artifact, source):
    """Restore only the exact original bytes recorded for this overlay."""
    artifact, source = Path(artifact), Path(source)
    manifest = json.loads((artifact / "overlay-manifest.json").read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != manifest["base_commit"]:
        raise ValueError("unexpected image source commit")
    for row in manifest["files"]:
        rel = Path(row["path"])
        target = source / rel
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or target.is_symlink()
            or not target.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError("invalid rollback path")
        current = digest(target) if target.exists() else None
        if current not in (row["sha256"], row["base_sha256"]):
            raise ValueError("refusing rollback of unknown source")
        if row["base_sha256"] is not None:
            if digest(artifact / "baseline" / rel) != row["base_sha256"]:
                raise ValueError("rollback backup hash mismatch")
    for row in manifest["files"]:
        target = source / row["path"]
        if row["base_sha256"] is None:
            if target.exists():
                target.unlink()
        else:
            tmp = target.with_name(target.name + ".rollback-" + str(os.getpid()))
            tmp.write_bytes((artifact / "baseline" / row["path"]).read_bytes())
            os.chmod(tmp, 0o644)
            os.replace(tmp, target)
    if subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=source, text=True
    ).strip():
        raise ValueError("source is not clean after rollback")
