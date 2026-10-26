"""Offline runner admission, immutable-terminal and artifact-verification tests.

Fixtures use the actual RequestRecorder to produce per-question contracts and
numeric ledgers. Model transport, resource attestation and scheduling probes
are replaced locally; no model is started or contacted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from strict_ablations import control, runner
from strict_ablations.prompt import ARMS, CONTROL, build_contract
from strict_ablations.runtime import RequestRecorder
from strict_ablations.test_artifacts import fixture_config, write_fixture_artifacts

ROOT = Path(__file__).resolve().parents[1]


class FixtureSuite(runner.Suite):
    """Replace only external manifest/resource admission for scheduler tests."""
    def verify_manifest(self, *, resources=True):
        return copy.deepcopy(self.fixture_manifest)


class StrictAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.suite = FixtureSuite(self.root)
        self.rows = [{"query_id": qid, "query": f"Question {qid}", "answer": f"Gold {qid}"}
                     for qid in range(100)]
        self.config = fixture_config()
        payload = {"version": runner.SUITE_VERSION, "config": self.config,
            "config_sha256": runner.digest(self.config),
            "contracts": {arm: build_contract(arm) for arm in ARMS},
            "cohort_sha256": runner.shared.DATASET_SHA,
            "query_ids": list(range(100)), "pilot_query_ids": list(range(10))}
        self.manifest = {**payload, "manifest_sha256": runner.digest(payload)}
        self.suite.fixture_manifest = self.manifest
        self.write(self.suite.reports / "run_manifest.json", self.manifest)

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def runtime_row(self, arm, query, *, reason="done", final="A deliberately incorrect answer"):
        return {"arm": arm, "query_id": query["query_id"], "model": runner.MODEL,
            "final_text": final, "terminated_by": reason, "error": "", "n_turns": 1,
            "n_tool_calls": 0, "tool_call_breakdown": {}, "elapsed_seconds": 3.0,
            "input_tokens": 50, "output_tokens": 10, "cached_input_tokens": 20,
            "raw_non_cached_input_tokens": 30, "reasoning_tokens": 4,
            "api_ledger": {"attempts": 1, "successful_calls": 1,
                "attempts_without_usage": 0, "input_tokens": 50, "output_tokens": 10,
                "agent_totals_match": True},
            "execution_checks": {"contract_verified": True, "model_verified": True,
                                  "token_ledger_verified": True},
            "reranker_summary": {"rerank_calls": 0, "scoring_calls": 0,
                "scored_passages": 0, "input_tokens": 0,
                "scoring_seconds": 0.0, "preparation_seconds": 0.0},
            "ablation_summary": {"arm": arm, "search_invocations": 0,
                "retrieval_calls": 0, "retrieval_queries": 0,
                "independent_rerank_calls": 0, "independent_rerank_page_calls": 0,
                "search_page_calls": 0, "automatic_rerank_calls": 0,
                "automatic_rerank_errors": 0, "automatic_empty_scopes": 0}}

    def seal_terminal(self, arm, query, **kwargs):
        row = {**self.runtime_row(arm, query, **kwargs), "version": runner.SUITE_VERSION,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "config_sha256": self.manifest["config_sha256"],
            "contract_sha256": self.manifest["contracts"][arm]["contract_sha256"],
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest()}
        row["final_sha256"] = hashlib.sha256(row["final_text"].encode()).hexdigest()
        row["row_sha256"] = runner.digest(row)
        return row

    def write_artifacts(self, arm, query, *, terminal=None):
        """Capture a real checked request against a zero-network local client."""
        contract = self.manifest["contracts"][arm]
        output = self.suite.runs / arm
        usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        response = SimpleNamespace(model=runner.MODEL, id="offline-recorder",
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: copy.deepcopy(usage)))
        send = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
        recorder = RequestRecorder(client, arm=arm, query=query, config=self.config,
            contract=contract, output_dir=output)
        recorder.chat.completions.create(model=runner.MODEL,
            messages=[{"role": "system", "content": contract["system_prompt"]},
                      {"role": "user", "content": contract["user_prompt_template"].format(question=query["query"])}],
            tools=contract["tool_specs"], max_tokens=32768)
        send.assert_called_once()
        qid = query["query_id"]
        terminal = terminal if terminal is not None else self.runtime_row(arm, query)
        write_fixture_artifacts(output, arm=arm, query=query, config=self.config, terminal=terminal)
        return terminal

    def write_terminal(self, arm, query, *, artifacts=True, **kwargs):
        row = self.seal_terminal(arm, query, **kwargs)
        if artifacts:
            self.write_artifacts(arm, query, terminal=row)
        row.pop("row_sha256", None)
        row["row_sha256"] = runner.digest(row)
        self.write(self.suite.terminal_path(arm, query["query_id"]), row)
        return row

    def ready_file(self):
        self.write(self.suite.reports / "preflight_live.json", {
            "passed": True, "manifest_sha256": self.manifest["manifest_sha256"],
            "service": {"epoch_sha256": "fixture-epoch"}})

    def test_new_paths_ignore_all_legacy_thirty_and_b100_terminals(self):
        decoys = []
        for arm in ARMS:
            for query in self.rows[:10]:
                path = self.root / "runs" / arm / "_per_query" / f"qid_{query['query_id']}.json"
                self.write(path, {"old_v14": True})
                decoys.append(path)
        old_b = self.root / "runs/strict_qwen35/B/_per_query/qid_0.json"
        self.write(old_b, {"old_B100": True}); decoys.append(old_b)
        before = {path: path.read_bytes() for path in decoys}
        with patch.object(runner.shared, "validate_queries", return_value=self.rows):
            report = self.suite.status()
        self.assertEqual(report["completed_new_runs"], 0)
        self.assertTrue(all(len(group["pending"]) == 100 for group in report["arms"].values()))
        query, arm = self.rows[0], ARMS[0]
        def execute(**kwargs):
            self.assertEqual(kwargs["output_dir"], self.root / "runs" / runner.DIRECTORY / arm)
            self.assertEqual(kwargs["report_dir"], self.root / "reports" / runner.DIRECTORY)
            self.assertEqual(kwargs["query"], {"query_id": 0, "query": query["query"]})
            return self.write_artifacts(arm, query, terminal=self.runtime_row(arm, query))
        with patch("strict_ablations.runtime.run_query", side_effect=execute) as run_query:
            result = self.suite.run_one(self.manifest, arm, query)
        self.assertEqual(result["status"], "terminal_saved")
        run_query.assert_called_once()
        self.assertTrue(self.suite.terminal_path(arm, 0).is_file())
        self.assertEqual(before, {path: path.read_bytes() for path in decoys})

    def test_control_cannot_have_a_new_terminal_or_be_scheduled(self):
        with patch("strict_ablations.runtime.run_query") as execute:
            with self.assertRaises(ValueError):
                self.suite.terminal_path(CONTROL, 0)
            with self.assertRaises(ValueError):
                self.suite.run_one(self.manifest, CONTROL, self.rows[0])
        execute.assert_not_called()

    def test_prepare_refuses_existing_manifest_or_runs_without_overwrite(self):
        config = {"version": runner.SUITE_VERSION, "control_root": str(self.root / "control"),
                  "common_config": self.config}
        for case in ("manifest", "runs"):
            suite = runner.Suite(self.root / case)
            sentinel = (suite.reports / "run_manifest.json" if case == "manifest"
                        else suite.runs / "preserve.txt")
            sentinel.parent.mkdir(parents=True); sentinel.write_text("original bytes")
            with patch.object(suite, "offline_preflight", return_value={}), \
                 patch.object(runner, "verify_control_bundle") as verify, \
                 patch.object(runner.shared, "resource_identity") as resources:
                with self.subTest(case=case), self.assertRaisesRegex(RuntimeError, "cannot be rebound"):
                    suite.prepare(config)
            self.assertEqual(sentinel.read_text(), "original bytes")
            verify.assert_not_called(); resources.assert_not_called()

    def test_prepare_freezes_three_new_arms_without_rewriting_control(self):
        suite = runner.Suite(self.root / "fresh")
        control_root = self.root / "control"
        control_manifest = {"source_sha256": {"frozen.py": "a" * 64}, "resource_identity": {}}
        control_path = control_root / control.MANIFEST_PATH
        self.write(control_path, control_manifest)
        before = control_path.read_bytes()
        bundle = {"version": "fixture", "manifest_sha256": "b" * 64,
            "public_export_sha256": "c" * 64, "integrity_receipt_sha256": "d" * 64,
            "model": runner.MODEL, "cohort_sha256": runner.shared.DATASET_SHA,
            "contract_sha256": build_contract(CONTROL)["contract_sha256"],
            "config_sha256": runner.digest(self.config)}
        config = {"version": runner.SUITE_VERSION, "control_root": str(control_root),
                  "common_config": self.config}
        with patch.object(suite, "offline_preflight", return_value={}), \
             patch.object(runner, "verify_control_bundle", return_value=bundle), \
             patch.object(runner.shared, "source_hashes", return_value=control_manifest["source_sha256"]), \
             patch.object(runner, "source_hashes", return_value={"strict_ablations/runtime.py": "e" * 64}), \
             patch("strict_ablations.deployment.prepare_epoch", return_value={"epoch_sha256": "f" * 64}), \
             patch.object(runner.shared, "resource_identity", return_value={}), \
             patch.object(runner.shared, "validate_queries", return_value=self.rows):
            manifest = suite.prepare(config)
            with self.assertRaises(RuntimeError):
                suite.prepare(config)
        self.assertEqual(manifest["config"], self.config)
        self.assertEqual(set(manifest["contracts"]), set(ARMS))
        self.assertEqual(manifest["question_runs"], 300)
        self.assertEqual(manifest["reused_control_questions"], 100)
        self.assertEqual(control_path.read_bytes(), before)
        self.assertEqual(runner.load(suite.reports / "run_manifest.json"), manifest)
        self.assertFalse(suite.runs.exists())

    def test_natural_failure_terminal_is_reused_without_model_or_file_changes(self):
        arm, query = ARMS[0], self.rows[0]
        self.write_terminal(arm, query, reason="max_turns", final="")
        before = {path: path.read_bytes() for path in self.suite.runs.rglob("*") if path.is_file()}
        with patch("strict_ablations.runtime.run_query") as execute:
            result = self.suite.run_one(self.manifest, arm, query)
        self.assertEqual(result["status"], "reused_immutable_terminal")
        execute.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in self.suite.runs.rglob("*") if path.is_file()})

    def test_existing_terminal_requires_all_artifacts_and_actual_contract(self):
        arm, query = ARMS[0], self.rows[0]
        self.write_terminal(arm, query)
        output = self.suite.runs / arm
        paths = (output / "request_contracts/qid_0.json", output / "api_ledgers/qid_0.jsonl",
                 output / "_traces/qid_0/single.json", output / "_checkpoints/qid_0.json")
        with patch("strict_ablations.runtime.run_query") as execute:
            for path in paths:
                before = path.read_bytes(); path.unlink()
                with self.subTest(missing=path), self.assertRaises((RuntimeError, ValueError, OSError)):
                    self.suite.run_one(self.manifest, arm, query)
                path.write_bytes(before)
            contract_path = paths[0]
            record = runner.load(contract_path); record["system_prompt"] += "extra instruction"
            self.write(contract_path, record)
            with self.assertRaisesRegex(RuntimeError, "Actual sent prompt/tools"):
                self.suite.run_one(self.manifest, arm, query)
        execute.assert_not_called()

    def test_resigned_final_tamper_still_fails_final_bytes_binding(self):
        arm, query = ARMS[0], self.rows[0]
        row = self.write_terminal(arm, query)
        row["final_text"] = "tampered public answer"
        row.pop("row_sha256"); row["row_sha256"] = runner.digest(row)
        self.write(self.suite.terminal_path(arm, 0), row)
        with patch("strict_ablations.runtime.run_query") as execute:
            with self.assertRaisesRegex(RuntimeError, "Public final bytes"):
                self.suite.run_one(self.manifest, arm, query)
        execute.assert_not_called()

    def test_actual_ledger_numeric_usage_and_model_tamper_are_rejected(self):
        arm, query = ARMS[0], self.rows[0]
        row = self.write_terminal(arm, query)
        path = self.suite.runs / arm / "api_ledgers/qid_0.jsonl"
        original = path.read_bytes()
        changes = (lambda r: r["usage"].update(prompt_tokens=51),
                   lambda r: r["usage"].update(completion_tokens=True),
                   lambda r: r.update(requested_model="wrong-model"),
                   lambda r: r.update(reported_model="wrong-model"),
                   lambda r: r.update(contract_sha256="0" * 64))
        for mutate in changes:
            entry = json.loads(original); mutate(entry); self.write(path, entry)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                self.suite.verify_execution_artifacts(self.manifest, arm, query, row)
            path.write_bytes(original)

    def test_terminal_attempt_counters_must_match_the_actual_jsonl(self):
        arm, query = ARMS[0], self.rows[0]
        original = self.write_terminal(arm, query)
        for field, value in (("attempts", 2), ("successful_calls", 0), ("attempts_without_usage", 1)):
            row = copy.deepcopy(original); row["api_ledger"][field] = value
            row.pop("row_sha256"); row["row_sha256"] = runner.digest(row)
            # Basic metric validation alone cannot prove these counters; the
            # request-artifact verifier must bind them to the actual ledger.
            self.suite.validate_terminal(row, self.manifest, arm, query)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.suite.verify_execution_artifacts(self.manifest, arm, query, row)

    def test_pilot_requires_all_thirty_actual_identity_and_usage_artifacts(self):
        with patch.object(runner.shared, "validate_queries", return_value=self.rows):
            with self.assertRaises((RuntimeError, OSError)):
                self.suite.pilot_gate(self.manifest)
            for query in self.rows[:10]:
                for arm in ARMS:
                    self.write_terminal(arm, query, reason="max_turns", final="")
            result = self.suite.pilot_gate(self.manifest)
            self.assertTrue(result["passed"])
            self.assertFalse(result["correctness_used"])
            self.assertEqual(len(result["checked_rows"]), 30)
            self.assertEqual(result["new_pilot_runs"], 30)
            self.assertTrue(all(record["recorded_requests"] == 1 for record in result["checked_rows"]))
            missing = self.suite.runs / ARMS[-1] / "api_ledgers/qid_9.jsonl"
            missing.unlink()
            with self.assertRaises((RuntimeError, OSError)):
                self.suite.pilot_gate(self.manifest)

    def test_false_execution_flags_cannot_promote_even_with_artifacts(self):
        arm, query = ARMS[0], self.rows[0]
        row = self.write_terminal(arm, query)
        row["execution_checks"]["token_ledger_verified"] = False
        row.pop("row_sha256"); row["row_sha256"] = runner.digest(row)
        with self.assertRaisesRegex(RuntimeError, "verification is missing"):
            self.suite.validate_terminal(row, self.manifest, arm, query)

    def test_partial_request_without_checkpoint_cannot_get_a_fresh_budget(self):
        arm, query = ARMS[0], self.rows[0]
        for directory, extension in (("api_ledgers", ".jsonl"), ("request_contracts", ".json")):
            path = self.suite.runs / arm / directory / ("qid_0" + extension)
            self.write(path, {"admitted_request": True})
            with patch("strict_ablations.runtime.run_query") as execute:
                for resume in (False, True):
                    with self.subTest(directory=directory, resume=resume), self.assertRaises(RuntimeError):
                        self.suite.run_one(self.manifest, arm, query, resume_infrastructure=resume)
                execute.assert_not_called()
            path.unlink()

    def test_infrastructure_marker_requires_explicit_resume_and_same_manifest(self):
        arm, query = ARMS[0], self.rows[0]
        path = self.suite.runs / arm / "_infrastructure/qid_0.json"
        for marker, resume in ((self.manifest["manifest_sha256"], False), ("f" * 64, True)):
            self.write(path, {"manifest_sha256": marker})
            with self.subTest(marker=marker, resume=resume), \
                 patch("strict_ablations.runtime.run_query", side_effect=RuntimeError("unexpected runtime admission")) as execute:
                with self.assertRaises(RuntimeError):
                    self.suite.run_one(self.manifest, arm, query, resume_infrastructure=resume)
                self.assertEqual(execute.call_count, 0, "Infrastructure marker admitted a model execution")

    def test_status_marks_ledger_only_partial_as_interrupted(self):
        arm = ARMS[0]
        self.write(self.suite.runs / arm / "api_ledgers/qid_0.jsonl", {"admitted": True})
        with patch.object(runner.shared, "validate_queries", return_value=self.rows):
            status = self.suite.status()
        self.assertIn(0, status["arms"][arm]["interrupted"])
        self.assertNotIn(0, status["arms"][arm]["pending"])

    def test_full_stage_rejects_missing_pilot_before_admitting_any_question(self):
        self.ready_file()
        with patch.object(self.suite, "pilot_gate", side_effect=RuntimeError("pilot not complete")) as gate, \
             patch.object(self.suite, "run_one") as execute, \
             patch.object(self.suite, "verify_service", return_value={"epoch_sha256": "fixture-epoch"}), \
             patch.object(runner.shared, "validate_queries", return_value=self.rows):
            with self.assertRaisesRegex(RuntimeError, "pilot not complete"):
                self.suite.run("full")
        gate.assert_called_once(); execute.assert_not_called()
        self.assertFalse((self.suite.reports / "full_completion.json").exists())

    def test_stage_requires_its_own_manifest_bound_live_preflight(self):
        for ready in ({"passed": False, "manifest_sha256": self.manifest["manifest_sha256"]},
                      {"passed": True, "manifest_sha256": "f" * 64}):
            self.write(self.suite.reports / "preflight_live.json", ready)
            with patch.object(self.suite, "run_one") as execute, \
                 patch.object(self.suite, "verify_service") as service:
                with self.subTest(ready=ready), self.assertRaisesRegex(RuntimeError, "own live preflight"):
                    self.suite.run("pilot")
                execute.assert_not_called(); service.assert_not_called()


class RealOfflineAdmissionTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / control.PUBLIC_PATH).exists(), "Local pinned B100 bundle unavailable")
    def test_export_copies_exactly_six_pinned_public_files_and_refuses_overwrite(self):
        paths = (control.MANIFEST_PATH, control.PUBLIC_PATH, control.METADATA_PATH,
                 control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH)
        before = {path: (ROOT / path).read_bytes() for path in paths}
        with tempfile.TemporaryDirectory() as directory, \
             patch("socket.create_connection", side_effect=AssertionError("network forbidden")) as network, \
             patch("rise.decompose.make_client", side_effect=AssertionError("model forbidden")) as model:
            output = Path(directory) / "public-control"
            result = runner.export_control_bundle(ROOT, output)
            self.assertEqual(result["files"], 6)
            self.assertEqual(result["model_requests"], 0)
            self.assertFalse(result["private_trajectories_included"])
            self.assertEqual({p.relative_to(output) for p in output.rglob("*") if p.is_file()}, set(paths))
            self.assertEqual(before, {path: (output / path).read_bytes() for path in paths})
            self.assertEqual(len(control.verify_control_bundle(output)["records"]), 100)
            with self.assertRaises(FileExistsError):
                runner.export_control_bundle(ROOT, output)
            self.assertEqual(before, {path: (output / path).read_bytes() for path in paths})
            public = output / control.PUBLIC_PATH
            public.write_bytes(public.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                control.verify_control_bundle(output)
            network.assert_not_called(); model.assert_not_called()
        self.assertEqual(before, {path: (ROOT / path).read_bytes() for path in paths})

    @unittest.skipUnless((ROOT / control.PUBLIC_PATH).exists(), "Local pinned B100 bundle unavailable")
    def test_real_control_and_offline_preflight_make_zero_network_or_model_calls(self):
        paths = (control.MANIFEST_PATH, control.PUBLIC_PATH, control.METADATA_PATH,
                 control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH)
        before = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
        with tempfile.TemporaryDirectory() as directory, \
             patch("socket.create_connection", side_effect=AssertionError("offline network forbidden")) as network, \
             patch.object(runner.shared, "urlopen", side_effect=AssertionError("offline HTTP forbidden")) as http, \
             patch.object(runner.shared, "verify_service", side_effect=AssertionError("live verification forbidden")) as service, \
             patch("strict_ablations.runtime.make_service", side_effect=AssertionError("neural backend forbidden")) as neural, \
             patch("rise.decompose.make_client", side_effect=AssertionError("model client forbidden")) as model:
            bundle = control.verify_control_bundle(ROOT)
            suite = runner.Suite(directory)
            config = suite.template(ROOT)
            result = suite.offline_preflight(config)
            self.assertEqual(len(bundle["records"]), 100)
            self.assertTrue(result["passed"])
            self.assertFalse(result["live_ready"])
            self.assertEqual(result["model_requests"], 0)
            self.assertEqual(result["new_question_runs"], 300)
            self.assertEqual(result["reused_control_questions"], 100)
            self.assertEqual(result["control_reference"]["manifest_sha256"], control.MANIFEST_SHA256)
            self.assertFalse(suite.runs.exists())
            self.assertFalse(suite.reports.exists())
            for call in (network, http, service, neural, model):
                call.assert_not_called()
        self.assertEqual(before, {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths})


if __name__ == "__main__":
    unittest.main()
