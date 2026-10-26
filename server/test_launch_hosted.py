"""Detached launch must preserve credentials only in the worker environment."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch_hosted as launch


class HostedLaunchTests(unittest.TestCase):
    def test_detached_worker_receipt_and_command_contain_no_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = Path(folder)
            manifest = {"manifest_sha256": "fixture-manifest"}
            config = {"python": "/runtime/python"}
            with patch.object(launch.pilot, "REPORTS", reports), \
                 patch.object(launch.pilot, "verify_manifest", return_value=(manifest, config)), \
                 patch.object(launch.pilot, "require_preflight"), \
                 patch.object(launch.pilot, "runtime_environment", return_value={"DASHSCOPE_API_KEY": "test-secret"}), \
                 patch.object(launch.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as popen:
                result = launch.launch()
            self.assertNotIn("test-secret", json.dumps(result))
            self.assertNotIn("test-secret", (reports / "launch_receipt.json").read_text())
            self.assertNotIn("test-secret", json.dumps(popen.call_args.args[0]))
            self.assertEqual(popen.call_args.kwargs["env"]["DASHSCOPE_API_KEY"], "test-secret")
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["stdin"], launch.subprocess.DEVNULL)

    def test_existing_live_launch_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = Path(folder)
            (reports / "launch_receipt.json").write_text(json.dumps({"pid": 123}))
            with patch.object(launch.pilot, "REPORTS", reports), \
                 patch.object(launch.pilot, "verify_manifest", return_value=({}, {})), \
                 patch.object(launch.pilot, "require_preflight"), \
                 patch.object(launch.pilot, "runtime_environment", return_value={}), \
                 patch.object(launch.os, "kill"), \
                 patch.object(launch.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    launch.launch()
                popen.assert_not_called()
