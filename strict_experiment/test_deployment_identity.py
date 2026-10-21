import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from strict_experiment import deployment_identity as identity


class DeploymentIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"):
            (self.model / name).write_text("fixture " + name)
        self.python = self.root / "python3.12"
        self.python.write_text("fake Python binary fixture")
        self.launcher = self.root / "vllm"
        self.launcher.write_text("#!" + str(self.python) + "\n# launcher fixture\n")
        self.plugin = self.root / "qwen35_official.py"
        self.plugin.write_text("# fixture plugin")
        (self.root / "official_qwen3_reasoning_parser.py").write_text("# fixture official parser")
        self.command = [str(self.launcher), "serve", str(self.model), "--served-model-name", identity.MODEL,
            "--host", "127.0.0.1", "--port", "8000", "--reasoning-parser-plugin", str(self.plugin),
            "--reasoning-parser", "qwen35_official", "--tool-call-parser", "qwen3_coder",
            "--max-num-batched-tokens", "8192"]
        self.service_path = self.root / "service_config.json"
        self.service = {"command": self.command, "model": identity.MODEL}
        self.service_path.write_text(json.dumps(self.service))
        self.config = {"base_url": "http://127.0.0.1:8000/v1", "model_identity": {"service_config": str(self.service_path)}}
        self.process = {"pid": 132, "process_start_ticks": 400,
            "actual_command": [str(self.python), *self.command], "executable": str(self.python)}
        self.packages = {"packages": {"vllm": "0.17.0", "torch": "2.10.0", "transformers": "4.57.6"},
            "vllm_package_root": "/fixture/vllm", "source_files": [{"path": "/fixture/parser.py", "sha256": "source-sha", "bytes": 5}]}
        self.models = {"id": identity.MODEL, "root": str(self.model), "parent": None, "max_model_len": 262144}
        for name, value in (("_process", self.process), ("_python_identity", self.packages), ("_get_models", self.models)):
            patch = mock.patch.object(identity, name, return_value=value)
            patch.start(); self.addCleanup(patch.stop)

    def freeze(self):
        receipt = identity.collect_service_identity(self.service_path, pid=132)
        self.service["actual_identity"] = receipt
        self.service_path.write_text(json.dumps(self.service))
        return receipt

    def test_live_receipt_and_recheck_accept_same_process(self):
        receipt = self.freeze()
        self.assertFalse(receipt["weight_shards_rehashed"])
        self.assertEqual(len(receipt["model_metadata_files"]), 5)
        self.assertEqual(len(receipt["parser_files"]), 2)
        result = identity.verify_state(self.config)
        self.assertTrue(result["verified"])
        self.assertEqual(result["identity_sha256"], receipt["identity_sha256"])

    def test_pid_reuse_and_command_drift_rejected(self):
        self.freeze()
        for field, changed in (("process_start_ticks", 401), ("actual_command", [str(self.python), *self.command, "--changed"])):
            value = copy.deepcopy(self.process); value[field] = changed
            with mock.patch.object(identity, "_process", return_value=value), self.assertRaises(RuntimeError):
                identity.verify_state(self.config)

    def test_alias_cannot_hide_wrong_model_root(self):
        other = self.root / "other_model"
        other.mkdir()
        with mock.patch.object(identity, "_get_models", return_value={**self.models, "root": str(other)}), self.assertRaisesRegex(RuntimeError, "model root"):
            identity.collect_service_identity(self.service_path, 132)

    def test_parser_template_package_and_source_changes_rejected(self):
        self.freeze()
        for path in (self.plugin, self.model / "chat_template.jinja"):
            original = path.read_text(); path.write_text("changed bytes")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                identity.verify_state(self.config)
            path.write_text(original)
        package = copy.deepcopy(self.packages); package["packages"]["vllm"] = "changed"
        with mock.patch.object(identity, "_python_identity", return_value=package), self.assertRaises(RuntimeError):
            identity.verify_state(self.config)
        package = copy.deepcopy(self.packages); package["source_files"][0]["sha256"] = "changed"
        with mock.patch.object(identity, "_python_identity", return_value=package), self.assertRaises(RuntimeError):
            identity.verify_state(self.config)

    def test_missing_receipt_and_wrong_endpoint_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "actual_identity"):
            identity.verify_state(self.config)
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, "endpoint"):
            identity.verify_state({**self.config, "base_url": "http://127.0.0.1:8001/v1"})

    def test_proc_start_ticks_and_live_executable_are_measured(self):
        proc = self.root / "proc" / "132"
        proc.mkdir(parents=True)
        fields = ["S"] + ["0"] * 18 + ["7654"]
        (proc / "stat").write_text("132 (complex process name) " + " ".join(fields))
        (proc / "cmdline").write_bytes(b"/example/python\0/example/vllm\0serve\0")
        (proc / "exe").symlink_to(self.python)
        # Use the real function, which setUp replaced on the module.
        real_process = DeploymentIdentityTest.real_process
        with mock.patch.object(identity, "PROC", self.root / "proc"):
            result = real_process(132)
        self.assertEqual(result["process_start_ticks"], 7654)
        self.assertEqual(result["actual_command"], ["/example/python", "/example/vllm", "serve"])
        self.assertEqual(result["executable"], str(self.python))


DeploymentIdentityTest.real_process = staticmethod(identity._process)


if __name__ == "__main__":
    unittest.main()
