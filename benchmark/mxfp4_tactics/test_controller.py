import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller import main, restore_boot


class RestoreTests(unittest.TestCase):
    def test_restore_only_known_bytes_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = root / "original-boot.py"
            target = root / "boot.py"
            saved.write_bytes(b"original")
            target.write_bytes(b"wrapper")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            plan = {
                "boot_path": str(target),
                "original_boot_sha256": sha(saved),
                "wrapper_sha256": sha(target),
            }
            restore_boot(root, plan)
            self.assertEqual(target.read_bytes(), b"original")
            restore_boot(root, plan)
            target.write_bytes(b"unknown")
            with self.assertRaises(RuntimeError):
                restore_boot(root, plan)
            self.assertEqual(target.read_bytes(), b"unknown")
            saved.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                restore_boot(root, plan)

    def test_failure_still_executes_original_serving_boot(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = root / "original-boot.py"
            target = root / "boot.py"
            saved.write_bytes(b"original")
            target.write_bytes(b"wrapper")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            plan = {
                "boot_path": str(target),
                "original_boot_sha256": sha(saved),
                "wrapper_sha256": sha(target),
            }
            (root / "plan.json").write_text(json.dumps(plan))
            with (
                patch("sys.argv", ["controller", "--root", str(root)]),
                patch(
                    "controller.run_matrix", side_effect=RuntimeError("test failed")
                ) as run,
                patch("controller.os.execv") as execute,
            ):
                main()
                self.assertEqual(target.read_bytes(), b"original")
                self.assertTrue((root / "_FAILED").exists())
                self.assertEqual(execute.call_args.args[1][-1], str(target))
                main()
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
