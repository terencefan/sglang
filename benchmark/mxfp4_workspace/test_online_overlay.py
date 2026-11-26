"""The authorized online overlay must fail closed and restore original bytes."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "online_overlay_under_test", Path(__file__).with_name("online_overlay.py")
)
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.git("init", "-q")
        (self.source / "old.py").write_text("original\n")
        self.git("add", "old.py")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        for directory in ["candidate", "baseline"]:
            (self.artifact / directory).mkdir()
        (self.artifact / "baseline/old.py").write_text("original\n")
        (self.artifact / "candidate/old.py").write_text("candidate\n")
        (self.artifact / "candidate/new.py").write_text("new\n")
        self.manifest = {
            "base_commit": self.git("rev-parse", "HEAD").strip(),
            "patch_commit": "fixture",
            "files": [
                {
                    "path": "old.py",
                    "base_sha256": overlay.digest(self.source / "old.py"),
                    "sha256": overlay.digest(self.artifact / "candidate/old.py"),
                },
                {
                    "path": "new.py",
                    "base_sha256": None,
                    "sha256": overlay.digest(self.artifact / "candidate/new.py"),
                },
            ],
        }
        self.save_manifest()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.source, text=True, stderr=subprocess.DEVNULL
        )

    def save_manifest(self):
        (self.artifact / "overlay-manifest.json").write_text(json.dumps(self.manifest))

    def test_apply_reentry_and_rollback(self):
        overlay.apply(self.artifact, self.source)
        overlay.verify(self.artifact, self.source)
        overlay.apply(self.artifact, self.source)
        overlay.restore(self.artifact, self.source)
        self.assertEqual((self.source / "old.py").read_text(), "original\n")
        self.assertFalse((self.source / "new.py").exists())
        self.assertEqual(self.git("diff", "--name-only"), "")

    def test_tamper_rejected_before_any_write(self):
        (self.artifact / "candidate/new.py").write_text("tampered")
        with self.assertRaises(ValueError):
            overlay.apply(self.artifact, self.source)
        self.assertEqual((self.source / "old.py").read_text(), "original\n")

    def test_unknown_source_not_overwritten(self):
        (self.source / "old.py").write_text("unrelated change")
        with self.assertRaises(ValueError):
            overlay.apply(self.artifact, self.source)
        self.assertEqual((self.source / "old.py").read_text(), "unrelated change")

    def test_rollback_does_not_overwrite_unknown_edit(self):
        overlay.apply(self.artifact, self.source)
        (self.source / "old.py").write_text("unrelated change")
        with self.assertRaises(ValueError):
            overlay.restore(self.artifact, self.source)
        self.assertEqual((self.source / "old.py").read_text(), "unrelated change")
        self.assertTrue((self.source / "new.py").exists())


if __name__ == "__main__":
    unittest.main()
