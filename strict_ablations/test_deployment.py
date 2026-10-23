"""Pure offline deployment-epoch tests; no model, GPU, Docker or HTTP calls."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from strict_ablations import deployment


class DeploymentEpochTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous = self.receipt()
        self.actual = self.receipt(pid=55, process_start_ticks=999)
        self.service = self.root / "service_config.json"
        self.service.write_text(json.dumps({"command": self.previous["configured_command"],
            "actual_identity": self.previous}))
        self.telemetry = self.root / "capacity.telemetry.jsonl"
        self.telemetry.write_text(json.dumps({"gpu": {"rows": [["0", "GPU SKU", "80000", "123", "79877", "0"]]}}) + "\n")
        self.capacity = self.root / "capacity.json"
        self.capacity.write_text(json.dumps({"telemetry": {"path": str(self.telemetry)},
            "certification_scope": "Only the two measured C4 admission/coexistence workloads"}))
        self.config = {"model_identity": {"service_config": str(self.service), "revision": "d" * 40},
            "capacity_report": str(self.capacity), "sandbox_backend": "docker",
            "runtime_environment": {"RISE_DOCKER_IMAGE": "sha256:" + "e" * 64,
                "DOCKER_HOST": "unix:///explicit-frozen-socket"}}
        self.baseline = {"model_revision": "d" * 40,
            "model_files": {"service_config": deployment.shared.file_sha(self.service)},
            "capacity_report": {"path": str(self.capacity.resolve()), "sha256": deployment.shared.file_sha(self.capacity),
                "passed": True, "report_sha256": "f" * 64, "certification_scope": "Only measured workloads"}}
        self.host = {"passed": True, "original_capacity_sha256": self.baseline["capacity_report"]["sha256"],
            "original_telemetry_sha256": deployment.shared.file_sha(self.telemetry),
            "gpu": [{"index": 0, "name": "GPU SKU", "memory_total_mib": 80000, "uuid": "GPU-uuid", "driver_version": "observed"}],
            "docker_image_id": self.config["runtime_environment"]["RISE_DOCKER_IMAGE"],
            "host": {"physical_memory_bytes": 1000000}, "long_capacity_probe_repeated": False}
        for patcher in (
            mock.patch.object(deployment.shared, "validate_config"),
            mock.patch.object(deployment.shared, "verify_capacity_report", return_value=copy.deepcopy(self.baseline["capacity_report"])),
            mock.patch.object(deployment.identity, "collect_service_identity", return_value=self.actual),
            mock.patch.object(deployment, "host_capacity_evidence", return_value=self.host),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def receipt(**changes):
        value = {"version": deployment.identity.VERSION, "pid": 132, "process_start_ticks": 100,
            "configured_command": ["/same/python", "-m", "vllm", "serve", "/same/model", "--max-num-seqs", "4"],
            "actual_command": ["/same/python", "-m", "vllm", "serve", "/same/model", "--max-num-seqs", "4"],
            "executable": "/same/python", "parser_files": [{"sha256": "unchanged"}],
            "model_metadata_files": [{"sha256": "unchanged-template"}], "model_root": "/same/model",
            "packages": {"vllm": "same", "torch": "same", "transformers": "same"}, **changes}
        value["identity_sha256"] = deployment.identity.digest(value)
        return value

    def prepare(self):
        return deployment.prepare_epoch(self.config, self.baseline, pid=55)

    @staticmethod
    def reseal(value):
        value.pop("epoch_sha256", None)
        value["epoch_sha256"] = deployment.identity.digest(value)
        return value

    def test_explicit_current_pid_required_no_old_pid_guess(self):
        for pid in (None, True, 0, -1, "55"):
            with self.subTest(pid=pid), self.assertRaisesRegex(RuntimeError, "service-pid"):
                deployment.prepare_epoch(self.config, self.baseline, pid=pid)

    def test_new_pid_accepted_and_old_service_file_remains_unchanged(self):
        original = self.service.read_bytes()
        proof = self.prepare()
        verified = deployment.verify_epoch(self.config, self.baseline, proof)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["pid"], 55)
        self.assertEqual(verified["process_start_ticks"], 999)
        self.assertEqual(self.service.read_bytes(), original)
        self.assertEqual(json.loads(original)["actual_identity"]["pid"], 132)
        self.assertFalse(verified["long_capacity_probe_repeated"])
        deployment.identity.collect_service_identity.assert_called_with(str(self.service), 55)

    def test_old_service_config_cannot_be_rewritten_for_new_pid(self):
        self.service.write_text(json.dumps({"command": self.actual["configured_command"], "actual_identity": self.actual}))
        with self.assertRaisesRegex(RuntimeError, "service_config bytes"):
            self.prepare()

    def test_static_changes_including_added_fields_are_rejected(self):
        for change in ({"configured_command": ["different-argv"]}, {"actual_command": ["different-argv"]},
                       {"packages": {"vllm": "different"}}, {"parser_files": [{"sha256": "changed"}]},
                       {"model_metadata_files": [{"sha256": "changed-template"}]},
                       {"model_root": "/other/model"}, {"new_static_field": True}):
            with self.subTest(change=change), mock.patch.object(deployment.identity,
                    "collect_service_identity", return_value=self.receipt(pid=55, process_start_ticks=999, **change)):
                with self.assertRaisesRegex(RuntimeError, "static"):
                    self.prepare()

    def test_epoch_detects_pid_reuse_and_post_verification_software_drift(self):
        proof = self.prepare()
        for change in ({"process_start_ticks": 1000}, {"packages": {"vllm": "changed"}}):
            actual = self.receipt(pid=55, **change)
            with self.subTest(change=change), mock.patch.object(deployment.identity, "collect_service_identity", return_value=actual):
                with self.assertRaisesRegex(RuntimeError, "drifted|reused"):
                    deployment.verify_epoch(self.config, self.baseline, proof)

    def test_capacity_bytes_and_fresh_validation_must_match_baseline(self):
        original = self.capacity.read_bytes()
        self.capacity.write_bytes(original + b" ")
        with self.assertRaisesRegex(RuntimeError, "capacity report bytes"):
            self.prepare()
        self.capacity.write_bytes(original)
        with mock.patch.object(deployment.shared, "verify_capacity_report", return_value={"passed": True}):
            with self.assertRaisesRegex(RuntimeError, "capacity identity"):
                self.prepare()
        with mock.patch.object(deployment.shared, "verify_capacity_report", side_effect=RuntimeError("invalid old workload")):
            with self.assertRaisesRegex(RuntimeError, "workload"):
                self.prepare()

    def test_config_baseline_and_proof_tamper_rejected(self):
        proof = self.prepare()
        changed = copy.deepcopy(proof)
        changed["actual_identity"]["pid"] = 77
        with self.assertRaisesRegex(RuntimeError, "digest"):
            deployment.verify_epoch(self.config, self.baseline, changed)
        for field, value in (("allowed_lifecycle_fields", ["pid", "packages"]),
                             ("service_config_sha256", "a" * 64), ("operator_source_sha256", "b" * 64)):
            changed = self.reseal({**copy.deepcopy(proof), field: value})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                deployment.verify_epoch(self.config, self.baseline, changed)
        with self.assertRaisesRegex(RuntimeError, "another config"):
            deployment.verify_epoch({**self.config, "new": "budget"}, self.baseline, proof)
        with self.assertRaisesRegex(RuntimeError, "another config"):
            deployment.verify_epoch(self.config, {**self.baseline, "other": "baseline"}, proof)

    def test_host_drift_invalidates_existing_epoch(self):
        proof = self.prepare()
        for field, value in (("gpu", []), ("docker_image_id", "different"),
                             ("original_telemetry_sha256", "c" * 64), ("host", {"physical_memory_bytes": 1})):
            changed = {**copy.deepcopy(self.host), field: value}
            with self.subTest(field=field), mock.patch.object(deployment, "host_capacity_evidence", return_value=changed):
                with self.assertRaisesRegex(RuntimeError, "capacity evidence"):
                    deployment.verify_epoch(self.config, self.baseline, proof)


class HostCapacityEvidenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.telemetry = self.root / "old.jsonl"
        self.telemetry.write_text(json.dumps({"gpu": {"rows": [["0", "GPU SKU", "80000", "5", "79995", "0"]]}}) + "\n")
        self.capacity = self.root / "capacity.json"
        self.capacity.write_text(json.dumps({"telemetry": {"path": str(self.telemetry)}, "certification_scope": "Original limited scope"}))
        self.image = "sha256:" + "a" * 64
        self.config = {"capacity_report": str(self.capacity), "sandbox_backend": "docker",
            "runtime_environment": {"RISE_DOCKER_IMAGE": self.image, "DOCKER_HOST": "unix:///frozen"}}

    def probe(self, gpu="0, GPU-uuid, GPU SKU, 80000, 555.1\n", image=None):
        image = self.image if image is None else image
        def run(command, **kwargs):
            self.assertTrue(kwargs["check"])
            if command[0] == "nvidia-smi":
                self.assertIn("--query-gpu=index,uuid,name,memory.total,driver_version", command)
                return SimpleNamespace(stdout=gpu)
            self.assertEqual(command, ["/usr/bin/docker", "image", "inspect", self.image, "--format", "{{.Id}}"])
            self.assertEqual(kwargs["env"]["DOCKER_HOST"], "unix:///frozen")
            return SimpleNamespace(stdout=image + "\n")
        with mock.patch.object(deployment.subprocess, "run", side_effect=run), \
                mock.patch.object(deployment, "_host_snapshot", return_value={"cpu_count": 4, "physical_memory_bytes": 1000000}):
            return deployment.host_capacity_evidence(self.config)

    def test_read_only_host_probe_records_historical_limits(self):
        result = self.probe()
        self.assertTrue(result["passed"])
        self.assertEqual(result["gpu"][0]["driver_version"], "555.1")
        self.assertEqual(result["original_telemetry_sha256"], deployment.shared.file_sha(self.telemetry))
        self.assertIn("did not record", result["historical_telemetry_binding"])
        self.assertEqual(result["capacity_scope"], "Original limited scope")
        self.assertFalse(result["long_capacity_probe_repeated"])

    def test_gpu_index_sku_memory_and_image_must_match(self):
        for gpu in ("1, UUID, GPU SKU, 80000, 555\n", "0, UUID, different GPU, 80000, 555\n", "0, UUID, GPU SKU, 40000, 555\n", ""):
            with self.subTest(gpu=gpu), self.assertRaisesRegex(RuntimeError, "GPU"):
                self.probe(gpu=gpu)
        with self.assertRaisesRegex(RuntimeError, "Docker image"):
            self.probe(image="sha256:" + "b" * 64)


class SuiteDeploymentIntegrationTest(unittest.TestCase):
    """Real new Suite glue with all live identity/HTTP/resource reads mocked."""
    def setUp(self):
        from strict_ablations import runner
        self.runner = runner
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = runner.Suite(self.root / "suite")
        self.config = {"model": runner.MODEL, "max_turns": 100, "context_length": 262144,
            "base_url": "http://offline.invalid/v1", "runtime_environment": {}}
        self.baseline = {"model_revision": "a" * 40,
            "model_files": {"weights_manifest": "b" * 64, "tokenizer_config": "c" * 64,
                "service_config": "d" * 64},
            "capacity_report": {"path": "/unchanged/capacity.json", "sha256": "e" * 64}}
        self.epoch = {"version": deployment.VERSION, "config_sha256": runner.digest(self.config),
            "baseline_resource_identity_sha256": runner.digest(self.baseline),
            "actual_identity": {"pid": 55, "process_start_ticks": 999}}
        self.epoch["epoch_sha256"] = runner.digest(self.epoch)
        self.control_root = self.root / "control"
        self.legacy_sources = {"legacy.py": "f" * 64}
        self.sources = {**self.legacy_sources, "strict_ablations/new.py": "0" * 64}
        runner.atomic_json(self.control_root / "reports/strict_qwen35/run_manifest.json",
            {"source_sha256": self.legacy_sources, "resource_identity": self.baseline})
        self.bundle = {"version": "frozen-control", "manifest_sha256": "1" * 64,
            "public_export_sha256": "2" * 64, "integrity_receipt_sha256": "3" * 64,
            "model": runner.MODEL, "cohort_sha256": runner.shared.DATASET_SHA,
            "contract_sha256": "4" * 64, "config_sha256": runner.digest(self.config),
            "common_config": self.config, "records": []}
        self.rows = [{"query_id": qid, "query": "synthetic question"} for qid in range(100)]

    def prepare_context(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(self.suite, "offline_preflight", return_value={"passed": True}))
        stack.enter_context(mock.patch.object(self.runner, "verify_control_bundle", return_value=self.bundle))
        stack.enter_context(mock.patch.object(self.runner.shared, "source_hashes", return_value=self.legacy_sources))
        stack.enter_context(mock.patch.object(self.runner, "source_hashes", return_value=self.sources))
        stack.enter_context(mock.patch.object(self.runner.shared, "resource_identity", return_value=self.baseline))
        stack.enter_context(mock.patch.object(self.runner.shared, "validate_config"))
        stack.enter_context(mock.patch.object(self.runner.shared, "validate_queries", return_value=self.rows))
        stack.enter_context(mock.patch.object(deployment, "prepare_epoch", return_value=self.epoch))
        return stack

    def service_manifest(self):
        return {"config": self.config, "resource_identity": self.baseline,
            "control_model_inputs": self.runner.build_contract(self.runner.CONTROL)}

    def test_static_resource_check_rejects_service_and_capacity_replacement(self):
        self.runner.same_static_resources(copy.deepcopy(self.baseline), self.baseline)
        for field in ("service_config", "weights_manifest", "tokenizer_config"):
            changed = copy.deepcopy(self.baseline)
            changed["model_files"][field] = "9" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.runner.same_static_resources(changed, self.baseline)
        changed = copy.deepcopy(self.baseline)
        changed["capacity_report"]["sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            self.runner.same_static_resources(changed, self.baseline)
        with self.assertRaises(ValueError):
            self.runner.same_static_resources({**self.baseline, "new_resource": True}, self.baseline)

    def test_prepare_binds_initial_epoch_and_offline_manifest_detects_tamper(self):
        config = {"version": self.runner.SUITE_VERSION, "control_root": str(self.control_root),
                  "common_config": self.config}
        original_control = (self.control_root / "reports/strict_qwen35/run_manifest.json").read_bytes()
        with self.prepare_context(), mock.patch.object(deployment, "verify_epoch",
                side_effect=AssertionError("Offline manifest verification must not inspect a live process")):
            manifest = self.suite.prepare(config, service_pid=55)
            deployment.prepare_epoch.assert_called_once_with(self.config, self.baseline, pid=55)
            self.assertEqual(manifest["initial_service_epoch_sha256"], self.epoch["epoch_sha256"])
            self.assertEqual(manifest["manifest_sha256"], self.runner.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}))
            archive = self.suite.reports / "service_epochs" / (self.epoch["epoch_sha256"] + ".json")
            self.assertEqual(self.runner.load(archive), self.epoch)
            self.assertEqual(self.runner.load(self.suite.reports / "service_epoch.json"), self.epoch)
            self.assertEqual(self.suite.verify_manifest(resources=False), manifest)
            changed = copy.deepcopy(self.epoch)
            changed["actual_identity"]["pid"] = 77
            self.runner.atomic_json(archive, changed)
            with self.assertRaisesRegex(RuntimeError, "initial process epoch"):
                self.suite.verify_manifest(resources=False)
        self.assertEqual((self.control_root / "reports/strict_qwen35/run_manifest.json").read_bytes(), original_control)

    def test_epoch_archive_is_immutable_and_rollover_preserves_initial_proof(self):
        self.suite.save_epoch(self.epoch)
        self.suite.save_epoch(copy.deepcopy(self.epoch))
        with self.assertRaisesRegex(RuntimeError, "immutable service epoch"):
            self.suite.save_epoch({**copy.deepcopy(self.epoch), "different": True})
        second = copy.deepcopy(self.epoch)
        second.pop("epoch_sha256")
        second["actual_identity"]["pid"] = 66
        second["epoch_sha256"] = self.runner.digest(second)
        self.suite.save_epoch(second)
        self.assertEqual(self.runner.load(self.suite.reports / "service_epoch.json"), second)
        self.assertEqual(self.runner.load(self.suite.reports / "service_epochs" /
            (self.epoch["epoch_sha256"] + ".json")), self.epoch)

    def test_verify_service_checks_epoch_before_control_tokenization(self):
        self.suite.save_epoch(self.epoch)
        manifest = self.service_manifest()
        events = []
        def verify(config, baseline, proof):
            self.assertEqual((config, baseline, proof), (self.config, self.baseline, self.epoch))
            events.append("epoch")
            return {"verified": True, "pid": 55}
        def request(value, *, timeout):
            events.append("tokenize")
            self.assertEqual(timeout, 30)
            self.assertEqual(value.full_url, "http://offline.invalid/tokenize")
            body = json.loads(value.data)
            self.assertEqual(body["model"], self.runner.MODEL)
            self.assertEqual(body["messages"][0], {"role": "system",
                "content": manifest["control_model_inputs"]["system_prompt"]})
            self.assertEqual(body["tools"], manifest["control_model_inputs"]["tool_specs"])
            self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
            return io.BytesIO(json.dumps({"count": 100, "max_model_len": 262144}).encode())
        with mock.patch.object(deployment, "verify_epoch", side_effect=verify), \
                mock.patch.object(self.runner, "urlopen", side_effect=request):
            result = self.suite.verify_service(manifest)
        self.assertEqual(events, ["epoch", "tokenize"])
        self.assertEqual(result["epoch_sha256"], self.epoch["epoch_sha256"])
        self.assertEqual(result["deployment_identity"]["pid"], 55)
        with mock.patch.object(deployment, "verify_epoch", side_effect=RuntimeError("stale PID")), \
                mock.patch.object(self.runner, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "stale PID"):
                self.suite.verify_service(manifest)
            network.assert_not_called()

    def test_verify_service_rejects_unarchived_epoch_and_wrong_token_budget(self):
        self.suite.save_epoch(self.epoch)
        manifest = self.service_manifest()
        archive = self.suite.reports / "service_epochs" / (self.epoch["epoch_sha256"] + ".json")
        self.runner.atomic_json(archive, {**self.epoch, "changed": True})
        with mock.patch.object(deployment, "verify_epoch") as verify, mock.patch.object(self.runner, "urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "not archived"):
                self.suite.verify_service(manifest)
            verify.assert_not_called()
            network.assert_not_called()
        self.runner.atomic_json(archive, self.epoch)
        for body in ({"count": True, "max_model_len": 262144}, {"count": 100, "max_model_len": 1000}):
            with self.subTest(body=body), mock.patch.object(deployment, "verify_epoch", return_value={"verified": True}), \
                    mock.patch.object(self.runner, "urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
                with self.assertRaisesRegex(RuntimeError, "tokenizer/context"):
                    self.suite.verify_service(manifest)

    def test_run_cannot_reuse_preflight_from_another_process_epoch(self):
        manifest = {"manifest_sha256": "a" * 64, "config": self.config}
        self.runner.atomic_json(self.suite.reports / "preflight_live.json", {
            "passed": True, "manifest_sha256": manifest["manifest_sha256"],
            "service": {"epoch_sha256": "old"}})
        with mock.patch.object(self.suite, "verify_manifest", return_value=manifest), \
                mock.patch.object(self.suite, "verify_service", return_value={"epoch_sha256": "new"}), \
                mock.patch.object(self.runner, "ThreadPoolExecutor") as workers:
            with self.assertRaisesRegex(RuntimeError, "own live preflight"):
                self.suite.run("pilot")
            workers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
