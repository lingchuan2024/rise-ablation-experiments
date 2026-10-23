"""Offline scope, admission and scheduling checks for the automatic20 pilot."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from strict_ablations import auto20, control, runner
from strict_ablations import test_runner as fixtures


class FixtureAuto20(auto20.Auto20Suite):
    def verify_manifest(self, *, resources=True):
        return copy.deepcopy(self.fixture_manifest)


class Automatic20Tests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.StrictAblationRunnerTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.rows = self.helper.rows
        self.suite = FixtureAuto20(self.helper.root)
        self.helper.suite = self.suite
        payload = {k: v for k, v in self.helper.manifest.items() if k != "manifest_sha256"}
        payload.update(self.suite.execution_scope(self.rows))
        payload["analysis_contract"] = {"version": "strict-auto20-analysis-v1", "rubric_sha256": "a" * 64}
        self.manifest = {**payload, "manifest_sha256": runner.digest(payload)}
        self.helper.manifest = self.manifest
        self.suite.fixture_manifest = self.manifest
        self.helper.write(self.suite.reports / "run_manifest.json", self.manifest)
        question_patch = patch.object(runner.shared, "validate_queries", return_value=self.rows)
        question_patch.start(); self.addCleanup(question_patch.stop)

    def test_scope_is_first20_only_and_main_suite_keeps300(self):
        expected = self.suite.execution_scope(self.rows)
        self.assertEqual(expected["selected_query_ids"], list(range(20)))
        self.assertEqual(expected["question_runs"], 20)
        self.assertEqual(expected["reused_control_questions"], 20)
        self.assertEqual(expected["execution_plan"]["arms"], [auto20.ARM])
        self.assertEqual(self.suite.probe_arms, (auto20.ARM,))
        self.assertEqual(runner.Suite().execution_scope(self.rows)["question_runs"], 300)
        self.assertEqual(runner.Suite().probe_arms, runner.ARMS)
        self.assertIn("strict_auto20_v1", str(self.suite.runs))
        self.assertNotEqual(self.suite.runs, runner.Suite(self.helper.root).runs)

    def test_forbidden_arm_qid_and_full_stage_cannot_reach_solver(self):
        with patch("strict_ablations.runtime.run_query") as execute:
            for arm, query in (("control", self.rows[0]), ("no_bash", self.rows[0]),
                               ("search_paging", self.rows[0]), (auto20.ARM, self.rows[20])):
                with self.subTest(arm=arm, qid=query["query_id"]), self.assertRaises(ValueError):
                    self.suite.run_one(self.manifest, arm, query)
            for stage in ("full", "pilot"):
                with self.assertRaises(ValueError):
                    self.suite.run(stage)
        execute.assert_not_called()

    def test_prepare_freezes20_scope_and_pilot_rubric(self):
        suite = auto20.Auto20Suite(self.helper.root / "fresh")
        old_root = self.helper.root / "old-control"
        old_manifest = {"source_sha256": {"legacy.py": "b" * 64}, "resource_identity": {}}
        old_path = old_root / control.MANIFEST_PATH
        self.helper.write(old_path, old_manifest)
        original = old_path.read_bytes()
        bundle = {key: "c" * 64 for key in ("manifest_sha256", "public_export_sha256",
            "integrity_receipt_sha256", "cohort_sha256", "contract_sha256", "config_sha256")}
        bundle.update(version="fixture", model=runner.MODEL)
        rubric = self.manifest["analysis_contract"]
        config = {"version": runner.SUITE_VERSION, "control_root": str(old_root),
                  "common_config": self.helper.config}
        with patch.object(suite, "offline_preflight", return_value={}), \
             patch.object(suite, "analysis_contract", return_value=rubric), \
             patch.object(runner, "verify_control_bundle", return_value=bundle), \
             patch.object(runner.shared, "source_hashes", return_value=old_manifest["source_sha256"]), \
             patch.object(runner, "source_hashes", return_value={"pilot.py": "d" * 64}), \
             patch.object(runner.shared, "resource_identity", return_value={}), \
             patch.object(runner.deployment, "prepare_epoch", return_value={"epoch_sha256": "e" * 64}):
            manifest = suite.prepare(config, service_pid=123)
        self.assertEqual(manifest["question_runs"], 20)
        self.assertEqual(manifest["selected_query_ids"], list(range(20)))
        self.assertEqual(len(manifest["query_ids"]), 100)
        self.assertEqual(manifest["analysis_contract"], rubric)
        self.assertEqual(original, old_path.read_bytes())
        self.assertFalse(suite.runs.exists())

    def test_scope_tampering_is_rejected_after_common_manifest_checks(self):
        suite = auto20.Auto20Suite(self.helper.root)
        changes = ({"question_runs": 300}, {"reused_control_questions": 100},
                   {"selected_query_ids": list(range(1, 21))},
                   {"execution_plan": {**self.manifest["execution_plan"], "arms": list(runner.ARMS)}})
        for change in changes:
            modified = {**self.manifest, **change}
            with patch.object(runner.Suite, "verify_manifest", return_value=modified):
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "scope changed"):
                    suite.verify_manifest(resources=False)

    def test_run20_schedules_exact20_at_most_c4_and_reuses_natural_failures(self):
        self.helper.ready_file()
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        def execute(manifest, arm, query, **kwargs):
            self.assertEqual(arm, auto20.ARM)
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.005)
                self.helper.write_terminal(arm, query, reason="max_turns", final="")
                return {"arm": arm, "query_id": query["query_id"], "status": "terminal_saved"}
            finally:
                with lock:
                    state["active"] -= 1

        with patch.object(self.suite, "verify_service", return_value={"epoch_sha256": "fixture-epoch"}), \
             patch.object(self.suite, "run_one", side_effect=execute) as run_one, \
             patch.object(self.suite, "pilot_gate", side_effect=AssertionError("all-arm gate forbidden")):
            receipt = self.suite.run20()
        self.assertEqual(run_one.call_count, 20)
        self.assertEqual({call.args[2]["query_id"] for call in run_one.call_args_list}, set(range(20)))
        self.assertLessEqual(state["peak"], 4)
        self.assertGreater(state["peak"], 1)
        self.assertEqual(receipt["question_runs"], 20)
        self.assertFalse(receipt["full_three_arm_suite_complete"])
        self.assertEqual(receipt["control_model_requests"], receipt["other_arm_model_requests"])
        before = {p: p.read_bytes() for p in self.suite.runs.rglob("*") if p.is_file()}
        with patch.object(self.suite, "verify_service", return_value={"epoch_sha256": "fixture-epoch"}), \
             patch("strict_ablations.runtime.run_query") as generate:
            self.suite.run20()
        generate.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.suite.runs.rglob("*") if p.is_file()})

    def test_preflight_manifest_and_process_epoch_bind_before_workers(self):
        for ready in ({"passed": False}, {"passed": True, "manifest_sha256": "wrong"},
                      {"passed": True, "manifest_sha256": self.manifest["manifest_sha256"],
                       "service": {"epoch_sha256": "old-epoch"}}):
            self.helper.write(self.suite.reports / "preflight_live.json", ready)
            with patch.object(self.suite, "verify_service", return_value={"epoch_sha256": "fixture-epoch"}), \
                 patch.object(self.suite, "run_one") as execute:
                with self.subTest(ready=ready), self.assertRaisesRegex(RuntimeError, "live preflight"):
                    self.suite.run20()
                execute.assert_not_called()

    def test_status_ignores_other_suites_and_marks_ledger_only_interruption(self):
        old = runner.Suite(self.helper.root).runs / auto20.ARM / "_per_query/qid_0.json"
        self.helper.write(old, {"decoy": True})
        ledger = self.suite.runs / auto20.ARM / "api_ledgers/qid_1.jsonl"
        ledger.parent.mkdir(parents=True); ledger.write_text("{}\n")
        status = self.suite.status()
        self.assertEqual(status["completed_new_runs"], 0)
        self.assertEqual(status["arms"][auto20.ARM]["interrupted"], [1])
        self.assertEqual(len(status["arms"][auto20.ARM]["pending"]), 19)
        self.assertEqual(set(status["arms"]), {auto20.ARM})

    def test_live_probe_only_admits_auto_and_requires_actual_neural_scores(self):
        from strict_ablations import runtime
        seen = []
        commands = {"printf RISE_STRICT_SANDBOX_READY": "RISE_STRICT_SANDBOX_READY",
            "printf RISE_TMP_PROBE >/tmp/rise-strict-probe": "",
            "test ! -e /tmp/rise-strict-probe && printf FRESH_TMP": "FRESH_TMP",
            "rg --version | head -n 1": "ripgrep fixture"}

        def build(arm, **kwargs):
            seen.append(arm)
            work = kwargs["working_dir"]
            def search(args):
                (work / "fixture.txt").write_text("fixture evidence")
                return json.dumps({"status": "ok"})
            class Surface(dict):
                summary = {"automatic_rerank_calls": 1}
            return Surface(search=search, bash=lambda args: commands[args["command"]], read=Mock())

        manifest = copy.deepcopy(self.manifest)
        manifest["config"]["corpus"] = str(self.helper.root / "unused-corpus")
        with patch.object(runtime, "resources", return_value=(None, [], {})), \
             patch.object(runtime, "health_state", return_value=None), \
             patch.object(runtime, "make_service", return_value=SimpleNamespace(
                 summary={"scoring_calls": 0, "scored_passages": 0})), \
             patch.object(runtime, "build_tools", side_effect=build), \
             patch("rise.decompose.make_client", side_effect=AssertionError("model forbidden")) as model:
            with self.assertRaisesRegex(RuntimeError, "Automatic search surface failed"):
                self.suite._probe_surfaces(manifest)
        self.assertEqual(seen, [auto20.ARM])
        model.assert_not_called()


class RealPilotOfflineTests(unittest.TestCase):
    @unittest.skipUnless((runner.ROOT / control.PUBLIC_PATH).exists(), "Pinned local B100 public bundle missing")
    def test_offline_admission_reports20_with_zero_network(self):
        suite = auto20.Auto20Suite()
        with patch("socket.create_connection", side_effect=AssertionError("network forbidden")) as network, \
             patch("rise.decompose.make_client", side_effect=AssertionError("model forbidden")) as model:
            report = suite.offline_preflight(control_root=runner.ROOT)
        self.assertEqual(report["arms"], [auto20.ARM])
        self.assertEqual(report["new_question_runs"], 20)
        self.assertEqual(report["reused_control_questions"], 20)
        self.assertEqual(report["selected_query_ids"], [q["query_id"] for q in auto20.selected_queries()])
        self.assertFalse(report["live_ready"])
        network.assert_not_called(); model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
