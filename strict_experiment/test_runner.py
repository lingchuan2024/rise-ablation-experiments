"""Controls that prevent outcome-selected retries and silent condition drift."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from strict_experiment import runner


class RunnerContractsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        for name, value in (("RUNS", folder / "runs"), ("REPORTS", folder / "reports")):
            patch = mock.patch.object(runner, name, value)
            patch.start(); self.addCleanup(patch.stop)
        self.rows = runner.validate_queries()
        self.contracts = runner.contracts()
        config = runner.config_template()
        self.manifest = {"version": runner.VERSION, "manifest_sha256": "frozen-manifest",
            "config": config, "config_sha256": runner.digest(config),
            "contracts": self.contracts, "query_ids": [row["query_id"] for row in self.rows]}
        self.calls = []
        self.runtime = SimpleNamespace(run_strict_query=self.fake_runtime)
        module_patch = mock.patch.dict(sys.modules, {"strict_experiment.runtime": self.runtime})
        module_patch.start(); self.addCleanup(module_patch.stop)

    def fake_runtime(self, *, arm, query, config, output_dir, expected_contract):
        self.calls.append((arm, query["query_id"]))
        self.assertEqual(set(query), {"query_id", "query"})  # No gold reaches the model.
        request = {"arm": arm, "query_id": query["query_id"], "model": runner.MODEL,
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "system_prompt": expected_contract["system_prompt"],
            "tool_specs": expected_contract["tool_specs"],
            "user_prompt": expected_contract["user_prompt_template"].format(question=query["query"]),
            "common_config_sha256": runner.digest(config),
            **{key: expected_contract[key] for key in ("system_prompt_sha256", "tool_specs_sha256", "contract_sha256")}}
        runner.atomic_json(output_dir / "request_contracts" / f"qid_{query['query_id']}.json", request)
        ledger = output_dir / "api_ledgers" / f"qid_{query['query_id']}.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"status": "ok", "requested_model": runner.MODEL,
            "reported_model": runner.MODEL, "contract_sha256": expected_contract["contract_sha256"],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15}}) + "\n")
        return {"model": runner.MODEL, "final_text": "", "terminated_by": "max_turns",
            "n_turns": 100, "elapsed_seconds": 12, "input_tokens": 30,
            "cached_input_tokens": 10, "output_tokens": 15, "reasoning_tokens": 11,
            "reranker_summary": {}, "execution_checks": {"contract_verified": True, "model_verified": True, "token_ledger_verified": True}}

    def make_pilot(self):
        for query in self.rows[:10]:
            for arm in runner.ARMS:
                runner.run_one(self.manifest, arm, query)

    def test_common_config_rejects_extra_arm_overrides_and_wrong_model(self):
        config = runner.config_template()
        self.assertIs(runner.validate_config(config), config)
        for key, value in (("model", "qwen3.8-flash"), ("per_call_max_tokens", 32000),
                ("run_max_attempts", 2), ("A", {"max_turns": 101})):
            changed = copy.deepcopy(config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_config(changed)

    def test_no_secret_or_wrong_profile_can_enter_frozen_environment(self):
        config = runner.config_template()
        config["runtime_environment"]["DASHSCOPE_API_KEY"] = "not-a-real-key"
        with self.assertRaises(ValueError): runner.validate_config(config)
        del config["runtime_environment"]["DASHSCOPE_API_KEY"]
        config["runtime_environment"]["RISE_QWEN35_PROFILE"] = "legacy"
        with self.assertRaises(ValueError): runner.validate_config(config)

    def test_prepare_cannot_claim_missing_model_is_ready(self):
        with self.assertRaises((ValueError, RuntimeError)):
            runner.prepare(runner.config_template())
        self.assertFalse((runner.REPORTS / "run_manifest.json").exists())

    def test_empty_natural_terminal_is_immutable_and_never_rerun(self):
        query = self.rows[0]
        first = runner.run_one(self.manifest, "A", query)
        path = runner.terminal_path("A", query["query_id"])
        original = path.read_bytes()
        self.assertEqual(first["terminated_by"], "max_turns")
        second = runner.run_one(self.manifest, "A", query, resume_infrastructure=True)
        self.assertEqual(second["status"], "reused_immutable_terminal")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            runner.atomic_json(path, {"final_text": "a better answer"}, immutable=True)

    def test_mutated_final_is_rejected_instead_of_refreshed(self):
        query = self.rows[0]
        runner.run_one(self.manifest, "A", query)
        path = runner.terminal_path("A", query["query_id"])
        row = runner.load(path); row["final_text"] = "changed"
        runner.atomic_json(path, row)
        with self.assertRaises(RuntimeError): runner.run_one(self.manifest, "A", query)
        self.assertEqual(len(self.calls), 1)

    def test_all_twenty_natural_empty_failures_still_pass_execution_gate(self):
        self.make_pilot()
        gate = runner.pilot_gate(self.manifest)
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["correctness_used"])
        self.assertEqual(len(gate["checked_rows"]), 20)

    def test_actual_sent_prompt_drift_blocks_gate_even_with_success_flag(self):
        self.make_pilot()
        path = runner.RUNS / "B/request_contracts/qid_70.json"
        record = runner.load(path)
        record["system_prompt"] += "\nTry harder."
        runner.atomic_json(path, record)
        with self.assertRaisesRegex(RuntimeError, "actual sent prompt"):
            runner.pilot_gate(self.manifest)

    def test_reported_model_drift_in_actual_ledger_blocks_gate(self):
        self.make_pilot()
        path = runner.RUNS / "B/api_ledgers/qid_70.jsonl"
        row = json.loads(path.read_text()); row["reported_model"] = "another-model"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "response reports another model"):
            runner.pilot_gate(self.manifest)

    def test_partial_api_execution_without_checkpoint_cannot_start_over(self):
        query = self.rows[0]
        output = runner.RUNS / "A"
        ledger = output / "api_ledgers" / f"qid_{query['query_id']}.jsonl"
        ledger.parent.mkdir(parents=True); ledger.write_text("{}\n")
        with self.assertRaises(RuntimeError): runner.run_one(self.manifest, "A", query)
        with self.assertRaisesRegex(RuntimeError, "fresh retry is forbidden"):
            runner.run_one(self.manifest, "A", query, resume_infrastructure=True)
        self.assertEqual(self.calls, [])

    def test_checkpoint_identity_and_consumed_budget_are_required_to_resume(self):
        query = self.rows[0]
        contract = self.contracts["B"]
        checkpoint = runner.RUNS / "B/_checkpoints/qid_70.json"
        identity = {"model": runner.MODEL, "query_id": str(query["query_id"]),
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "system_prompt_sha256": contract["system_prompt_sha256"], "tool_specs_sha256": contract["tool_specs_sha256"]}
        runner.atomic_json(checkpoint, {"identity": identity, "elapsed_seconds": 100})
        with self.assertRaises(RuntimeError): runner.run_one(self.manifest, "B", query)
        invalid = {"identity": {**identity, "query_id": "81"}, "elapsed_seconds": 100}
        runner.atomic_json(checkpoint, invalid)
        with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
            runner.run_one(self.manifest, "B", query, resume_infrastructure=True)
        runner.atomic_json(checkpoint, {"identity": identity, "elapsed_seconds": -1})
        with self.assertRaisesRegex(RuntimeError, "consumed wall-clock"):
            runner.run_one(self.manifest, "B", query, resume_infrastructure=True)
        self.assertEqual(self.calls, [])

    def test_full_stage_executes_only_remaining180_and_keeps_pilot_failures(self):
        self.make_pilot()
        original = {str(path): path.read_bytes() for path in runner.RUNS.glob("*/_per_query/*.json")}
        self.calls.clear()
        runner.atomic_json(runner.REPORTS / "preflight_live.json", {"passed": True,
            "manifest_sha256": self.manifest["manifest_sha256"]})
        with mock.patch.object(runner, "verify_manifest", return_value=self.manifest), \
                mock.patch.object(runner, "verify_service", return_value={}), mock.patch.dict(os.environ):
            receipt = runner.run("full")
        self.assertEqual(receipt["new_terminal_rows"], 180)
        self.assertEqual(len(self.calls), 180)
        self.assertEqual(len(list(runner.RUNS.glob("*/_per_query/*.json"))), 200)
        self.assertTrue(all(qid not in {row["query_id"] for row in self.rows[:10]} for _, qid in self.calls))
        for path, content in original.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_real_source_freeze_includes_shared_backend_and_runner(self):
        hashes = runner.source_hashes()
        for path in ("server/shared_reranker.py", "strict_experiment/runner.py", "strict_experiment/test_runner.py"):
            self.assertIn(path, hashes)

    def test_usage_drift_blocks_promotion_even_when_flags_claim_success(self):
        self.make_pilot()
        path = runner.RUNS / "B/api_ledgers/qid_70.jsonl"
        row = json.loads(path.read_text()); row["usage"]["completion_tokens"] = 999
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "token ledger differs"):
            runner.pilot_gate(self.manifest)

    def test_run_all_promotes_without_a_second_permission_or_pilot_replay(self):
        with mock.patch.object(runner, "run", side_effect=[{"stage": "pilot"}, {"stage": "full"}]) as run:
            receipt = runner.run_all()
        self.assertEqual(run.call_args_list, [mock.call("pilot", resume_infrastructure=False), mock.call("full", resume_infrastructure=False)])
        self.assertTrue(receipt["pilot_results_included_in_formal_total"])
        self.assertEqual(receipt["question_runs"], 200)

    def test_resource_receipt_checks_real_bytes_and_exact_directory_scope(self):
        folder = Path(self.temporary.name) / "resource"
        folder.mkdir(); (folder / "one.txt").write_text("original")
        receipt = Path(self.temporary.name) / "receipt.json"
        runner.write_resource_receipt("corpus", folder, receipt)
        verified = runner.verify_resource_receipt(receipt, "corpus", expected_root=folder, expected_paths={"one.txt"})
        self.assertTrue(verified["actual_bytes_verified"])
        (folder / "one.txt").write_text("modified")  # Same size, different bytes.
        with self.assertRaisesRegex(RuntimeError, "Actual resource bytes differ"):
            runner.verify_resource_receipt(receipt, "corpus", expected_root=folder)
        (folder / "one.txt").write_text("original")
        (folder / "extra.txt").write_text("extra")
        with self.assertRaisesRegex(RuntimeError, "exact resource file set"):
            runner.verify_resource_receipt(receipt, "corpus", expected_root=folder)

    def test_fake_resource_receipt_and_wrong_model_provenance_are_rejected(self):
        folder = Path(self.temporary.name) / "resource"
        folder.mkdir(); (folder / "weights.bin").write_bytes(b"actual-model")
        receipt = Path(self.temporary.name) / "receipt.json"
        revision = "a" * 40
        runner.write_resource_receipt("model_weights", folder, receipt, model_revision=revision)
        with self.assertRaisesRegex(RuntimeError, "another model or revision"):
            runner.verify_resource_receipt(receipt, "model_weights", model_revision="b" * 40)
        fabricated = runner.load(receipt)
        fabricated["files"][0]["sha256"] = "0" * 64
        fabricated["content_sha256"] = runner.digest(fabricated["files"])
        runner.atomic_json(receipt, fabricated)
        with self.assertRaisesRegex(RuntimeError, "Actual resource bytes differ"):
            runner.verify_resource_receipt(receipt, "model_weights", model_revision=revision)

    def test_corpus_metadata_is_hashed_but_not_confused_with_retrieved_text_scope(self):
        folder = Path(self.temporary.name) / "resource"
        folder.mkdir(); (folder / "one.txt").write_text("doc")
        (folder / "complete.json").write_text('{"ready":true}')
        receipt = Path(self.temporary.name) / "receipt.json"
        runner.write_resource_receipt("corpus", folder, receipt)
        verified = runner.verify_resource_receipt(receipt, "corpus", expected_root=folder, expected_paths={"one.txt"})
        self.assertEqual(verified["files"], 2)
        self.assertEqual(verified["metadata_files"], 1)
        (folder / "complete.json").write_text('{"ready":false}')
        with self.assertRaises(RuntimeError):
            runner.verify_resource_receipt(receipt, "corpus", expected_root=folder, expected_paths={"one.txt"})

    def make_cleanup_workspace(self, arm="B"):
        folder = Path(self.temporary.name)
        corpus = folder / "corpus"; corpus.mkdir()
        (corpus / "example.org").mkdir()
        source = corpus / "example.org/document.txt"
        source.write_text("FROZEN_SOURCE_BODY")
        receipt = folder / "corpus_receipt.json"
        runner.write_resource_receipt("corpus", corpus, receipt)
        self.manifest["config"].update(corpus=str(corpus), corpus_manifest=str(receipt))
        self.manifest["config_sha256"] = runner.digest(self.manifest["config"])
        query = self.rows[0]
        runner.run_one(self.manifest, arm, query)
        output = runner.RUNS / arm
        working = output / "_working/qid_70"
        (working / "example.org").mkdir(parents=True)
        os.link(source, working / "example.org/document.txt")
        derived = output / "_rerank_runtime/qid_70"
        derived.mkdir(parents=True); (derived / "derived_cache.json").write_text('{"cache":1}')
        protected = []
        for relative in ("_checkpoints/qid_70.json", "_traces/qid_70/single.json"):
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PRIVATE_TRACE_STAYS")
            protected.append(path)
        return query, source, working, derived, protected

    def test_only_terminal_workspaces_are_released_after_recovery_map_is_saved(self):
        query, source, working, derived, protected = self.make_cleanup_workspace()
        terminal_bytes = runner.terminal_path("B", 70).read_bytes()
        receipt = runner.post_terminal_maintenance(self.manifest, "B", query)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["source_file_count"], 1)
        self.assertFalse(working.exists()); self.assertFalse(derived.exists())
        self.assertEqual(source.read_text(), "FROZEN_SOURCE_BODY")
        self.assertTrue(all(path.read_text() == "PRIVATE_TRACE_STAYS" for path in protected))
        self.assertEqual(runner.terminal_path("B", 70).read_bytes(), terminal_bytes)
        archive = runner.load(runner.RUNS / "B/_maintenance/source_manifests/qid_70.json")
        self.assertEqual(archive["source_files"][0]["sha256"], runner.file_sha(source))
        self.assertEqual(archive["source_files"][0]["source_path"], str(source.resolve()))
        self.assertEqual(runner.post_terminal_maintenance(self.manifest, "B", query), receipt)

    def test_cleanup_refuses_unfinished_or_untracked_workspaces(self):
        query, source, working, derived, _ = self.make_cleanup_workspace()
        (working / "unexpected.txt").write_text("not-a-corpus-hardlink")
        with self.assertRaisesRegex(RuntimeError, "untracked or non-source"):
            runner.post_terminal_maintenance(self.manifest, "B", query)
        self.assertTrue(working.exists()); self.assertTrue(derived.exists()); self.assertTrue(source.exists())
        runner.terminal_path("B", 70).unlink()
        with self.assertRaises(FileNotFoundError):
            runner.post_terminal_maintenance(self.manifest, "B", query)
        self.assertTrue(working.exists())

    def test_cleanup_interruption_resumes_without_rerunning_natural_terminal(self):
        query, _, working, derived, _ = self.make_cleanup_workspace()
        original_remove = runner.shutil.rmtree
        def fail_on_runtime(path):
            if path == derived.resolve():
                raise OSError("simulated filesystem interruption")
            return original_remove(path)
        with mock.patch.object(runner.shutil, "rmtree", side_effect=fail_on_runtime):
            with self.assertRaises(OSError): runner.post_terminal_maintenance(self.manifest, "B", query)
        self.assertFalse(working.exists()); self.assertTrue(derived.exists())
        calls_before = len(self.calls)
        result = runner.run_one(self.manifest, "B", query, resume_infrastructure=True)
        self.assertEqual(result["status"], "reused_immutable_terminal")
        self.assertEqual(len(self.calls), calls_before)
        self.assertFalse(derived.exists())


if __name__ == "__main__":
    unittest.main()
