"""Offline checks for the Linux mini-corpus bash sandbox."""
from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The bash sandbox does not use BM25.  Stub retrieval so these tests also run
# on a machine that has not installed the corpus/indexing dependencies.
retrieval_stub = types.ModuleType("rise.retrieval")
retrieval_stub.retrieve = lambda *args, **kwargs: None
with patch.dict(sys.modules, {
    "bm25s": types.ModuleType("bm25s"),
    "rise.retrieval": retrieval_stub,
}):
    tools = importlib.import_module("rise.tools")


class LinuxBashSandboxTests(unittest.TestCase):
    def test_linux_command_mounts_only_system_and_current_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            with patch.object(tools.sys, "platform", "linux"):
                cmd = tools._sandboxed_bash_command(work, "cat visible.txt")
            resolved_work = str(work.resolve())
            self.assertEqual(cmd[0], "/usr/bin/bwrap")
            self.assertIn("--unshare-net", cmd)
            self.assertIn("--unshare-pid", cmd)
            self.assertNotIn("--clearenv", cmd)  # unsupported by bwrap 0.2.1
            self.assertIn(["--tmpfs", "/tmp"], [cmd[i:i + 2] for i in range(len(cmd) - 1)])
            mounts = [cmd[i + 1:i + 3] for i, arg in enumerate(cmd) if arg == "--ro-bind"]
            self.assertIn([resolved_work, resolved_work], mounts)
            self.assertNotIn(["/", "/"], mounts)
            self.assertFalse(any(mount[0] == str(Path(tmp)) for mount in mounts))
            self.assertEqual(cmd[-3:], ["/bin/bash", "-c", "cat visible.txt"])

    def test_macos_command_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with patch.object(tools.sys, "platform", "darwin"):
                cmd = tools._sandboxed_bash_command(work, "pwd")
            self.assertEqual(cmd[:2], ["/usr/bin/sandbox-exec", "-p"])
            self.assertEqual(cmd[-3:], ["/bin/bash", "-c", "pwd"])

    def test_linux_can_mount_only_bundled_ripgrep_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "src" / "rise" / "tools.py"
            bundled_rg = root / "bin" / "rg"
            bundled_rg.parent.mkdir()
            bundled_rg.write_text("#!/bin/sh\n", encoding="utf-8")
            bundled_rg.chmod(0o755)
            work = root / "work"
            work.mkdir()
            with patch.object(tools.sys, "platform", "linux"), patch.object(
                tools, "__file__", str(module_path)
            ):
                cmd = tools._sandboxed_bash_command(work, "rg --version")
            mounts = [cmd[i + 1:i + 3] for i, arg in enumerate(cmd) if arg == "--ro-bind"]
            self.assertIn([str(bundled_rg.resolve()), "/opt/rise-tools/rg"], mounts)
            self.assertIn("/opt/rise-tools:/usr/local/bin:/usr/bin:/bin", cmd)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and Path("/usr/bin/bwrap").is_file(),
        "requires Linux bubblewrap",
    )
    def test_real_linux_sandbox_read_visibility_and_write_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            (work / "visible.txt").write_text("WORKSPACE_VISIBLE\n", encoding="utf-8")
            hidden = root / "outside.txt"
            hidden.write_text("OUTSIDE_SECRET\n", encoding="utf-8")
            bash = tools.make_bash_tool(work, enable_sandbox=True)

            self.assertIn("WORKSPACE_VISIBLE", bash({"command": "cat visible.txt"}))
            outside_result = bash({"command": f"cat {hidden}"})
            self.assertNotIn("OUTSIDE_SECRET", outside_result)
            self.assertIn("[exit=", outside_result)

            write_result = bash({"command": "printf changed > visible.txt"})
            self.assertIn("[exit=", write_result)
            self.assertEqual((work / "visible.txt").read_text(encoding="utf-8"), "WORKSPACE_VISIBLE\n")

            self.assertIn("SCRATCH_OK", bash({
                "command": "printf SCRATCH_OK > /tmp/scratch && cat /tmp/scratch",
            }))
            with patch.dict("os.environ", {"RISE_SANDBOX_TEST_SECRET": "HIDDEN_FROM_BASH"}):
                self.assertEqual(
                    bash({"command": "printf '%s' \"${RISE_SANDBOX_TEST_SECRET-unset}\""}),
                    "unset",
                )


if __name__ == "__main__":
    unittest.main()
