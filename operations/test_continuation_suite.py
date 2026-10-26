"""Scheduling, immutability and scope guards for the administrative continuation."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from operations import continuation_suite as mod


class FakeSuite(mod.ContinuationSuite):
    def __init__(self, root, *, fail_qid=None):
        super().__init__(root)
        self.fail_qid = fail_qid
        self.started = []
        self.finished = []
        self.active_count = self.peak = 0
        self.mutex = threading.Lock()
        self.stage_checks = []
        self.reports.mkdir(parents=True)
        reuse = Path(root) / "old"
        (reuse / "reports/strict_auto20_v1").mkdir(parents=True)
        (reuse / "reports/strict_auto20_v1/worker.lock").write_bytes(b"immutable old lock")
        self.manifest = {"manifest_sha256": "test", "config": {"concurrency": 4},
            "reused_auto20_reference": {"root": str(reuse)}}
        mod.runner.atomic_json(self.reports / "run_manifest.json", self.manifest)

    def verify_manifest(self, *, resources=True):
        return self.manifest

    def _stage_ready(self, manifest):
        self.stage_checks.append(len(self.finished))

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        self.terminal_path(arm, query["query_id"])
        with self.mutex:
            if arm == "search_paging":
                self.assert_auto_complete()
            self.started.append((arm, query["query_id"]))
            self.active_count += 1
            self.peak = max(self.peak, self.active_count)
        try:
            if query["query_id"] == self.fail_qid:
                raise RuntimeError("simulated infrastructure interruption")
            time.sleep(0.05 if self.fail_qid is not None else 0.002)
            with self.mutex:
                self.finished.append((arm, query["query_id"]))
            return {"status": "terminal_saved"}
        finally:
            with self.mutex:
                self.active_count -= 1

    def assert_auto_complete(self):
        expected = {("auto_rerank", q["query_id"]) for q in mod.selected_queries("auto_rerank")}
        if not expected.issubset(set(self.finished)):
            raise AssertionError("Paging started before all automatic80 terminals")

    def _terminal_bindings(self, manifest, stage):
        expected = {(stage["arm"], q["query_id"]) for q in stage["queries"]}
        if not expected.issubset(set(self.finished)):
            raise AssertionError("Stage receipt before every terminal")
        return [{"arm": arm, "query_id": qid, "row_sha256": str(qid),
            "file_sha256": str(qid), "execution_artifacts": {}} for arm, qid in sorted(expected)]

    def status(self):
        return {"completed_new_runs": len(self.finished)}


class ContinuationTests(unittest.TestCase):
    def test_scope_is_80_then20_not_old20(self):
        rows = mod.runner.shared.validate_queries()
        self.assertEqual([s["name"] for s in mod.stages()], ["auto80", "paging20"])
        self.assertEqual(mod.selected_queries("auto_rerank"), rows[20:])
        self.assertEqual(mod.selected_queries("search_paging"), rows[:20])
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            for arm, qid in (("auto_rerank", rows[0]["query_id"]),
                             ("search_paging", rows[20]["query_id"]),
                             ("no_bash", rows[0]["query_id"]),
                             ("control", rows[0]["query_id"])):
                with self.assertRaises(ValueError): suite.terminal_path(arm, qid)

    def test_dispatch_c4_strict_stage_order_and_terminal_bindings(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d)
            result = suite.run()
            self.assertEqual(len(suite.started), 100)
            self.assertEqual(len(set(suite.started)), 100)
            self.assertEqual(suite.peak, 4)
            self.assertEqual(suite.stage_checks, [0, 80])
            self.assertEqual(result["terminal_rows_checked"], 100)
            self.assertEqual([len(s["terminal_rows"]) for s in result["stages"]], [80, 20])
            self.assertEqual(json.loads((suite.reports / "phase_state.json").read_text())["phase"], "complete")
            old = Path(suite.manifest["reused_auto20_reference"]["root"])
            self.assertEqual((old / "reports/strict_auto20_v1/worker.lock").read_bytes(), b"immutable old lock")

    def test_failure_stops_new_dispatch_and_forbids_paging(self):
        first = mod.selected_queries("auto_rerank")[0]["query_id"]
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d, fail_qid=first)
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                suite.run()
            self.assertLessEqual(len(suite.started), 4)
            self.assertTrue(all(arm == "auto_rerank" for arm, _ in suite.started))
            self.assertFalse((suite.reports / "auto80_completion.json").exists())
            self.assertFalse((suite.reports / "continuation_completion.json").exists())
            self.assertEqual(json.loads((suite.reports / "phase_state.json").read_text())["phase"], "interrupted")

    def test_held_old_lock_blocks_execution_without_modification(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d)
            old = Path(suite.manifest["reused_auto20_reference"]["root"]) / "reports/strict_auto20_v1/worker.lock"
            with old.open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError): suite.run()
            self.assertEqual(suite.started, [])
            self.assertEqual(old.read_bytes(), b"immutable old lock")

    def test_unexpected_arm_and_reused_query_artifacts_are_rejected(self):
        old_qid = mod.selected_queries("search_paging")[0]["query_id"]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            bad = suite.runs / "auto_rerank" / "request_contracts" / f"qid_{old_qid}.json"
            bad.parent.mkdir(parents=True); bad.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "outside"): suite.assert_run_scope()
            bad.unlink()
            (suite.runs / "no_bash").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Forbidden arm"): suite.assert_run_scope()

    def test_administrative_code_drift_blocks_before_solver(self):
        query = mod.selected_queries("auto_rerank")[0]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            with patch.object(mod.runner.Suite, "run_one") as base:
                with self.assertRaisesRegex(RuntimeError, "drifted"):
                    suite.run_one({"administrative_source_sha256": {}}, "auto_rerank", query)
                base.assert_not_called()

    def test_stage_receipt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d)
            suite.run()
            path = suite.reports / "auto80_completion.json"
            receipt = json.loads(path.read_text())
            receipt["terminal_rows"][0]["row_sha256"] = "changed"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, "immutable stage"):
                suite._stage_receipt(suite.manifest, mod.stages()[0])

    def test_completed_row_is_reused_without_solver_even_on_resume(self):
        query = mod.selected_queries("auto_rerank")[0]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            path = suite.terminal_path("auto_rerank", query["query_id"])
            path.parent.mkdir(parents=True); path.write_text('{}')
            with patch.object(suite, "validate_terminal", return_value={}) as validate, \
                 patch.object(suite, "verify_execution_artifacts", return_value={}), \
                 patch("strict_ablations.runtime.run_query") as solver:
                result = suite.run_one({"administrative_source_sha256": mod.admin_hashes()},
                    "auto_rerank", query, resume_infrastructure=True)
                self.assertEqual(result["status"], "reused_immutable_terminal")
                solver.assert_not_called(); validate.assert_called_once()

    def test_partial_without_checkpoint_cannot_receive_fresh_budget(self):
        query = mod.selected_queries("auto_rerank")[0]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            partial = suite.runs / "auto_rerank" / "api_ledgers" / f"qid_{query['query_id']}.jsonl"
            partial.parent.mkdir(parents=True); partial.write_text('{}\n')
            with patch("strict_ablations.runtime.run_query") as solver:
                with self.assertRaisesRegex(RuntimeError, "fresh-budget retry is forbidden"):
                    suite.run_one({"administrative_source_sha256": mod.admin_hashes()},
                        "auto_rerank", query, resume_infrastructure=True)
                solver.assert_not_called()


    def test_status_ignores_active_atomic_write_beside_committed_checkpoint(self):
        qid = mod.selected_queries("auto_rerank")[0]["query_id"]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            suite.reports.mkdir(parents=True)
            (suite.reports / "run_manifest.json").write_text('{}')
            cp = suite.runs / "auto_rerank" / "_checkpoints" / f"qid_{qid}.json"
            cp.parent.mkdir(parents=True); cp.write_text('{"committed":true}')
            temporary = cp.with_name(cp.name + ".abcd123_.tmp")
            temporary.write_text('{"incomplete":')
            def verified(**kwargs):
                suite.assert_run_scope()
                return {}
            with patch.object(suite, "verify_manifest", side_effect=verified):
                status = suite.status()
            self.assertEqual(status["completed_new_runs"], 0)
            self.assertIn(qid, status["arms"]["auto_rerank"]["interrupted"])
            self.assertEqual(temporary.read_text(), '{"incomplete":')
            self.assertEqual(cp.read_text(), '{"committed":true}')

    def test_orphan_temporary_is_neither_terminal_nor_committed_partial(self):
        qid = mod.selected_queries("auto_rerank")[0]["query_id"]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            suite.reports.mkdir(parents=True)
            (suite.reports / "run_manifest.json").write_text('{}')
            for directory, suffix in (("_per_query", ".json.abcd123_"),
                    ("_infrastructure", ".json.abcd123_"),
                    ("_checkpoints", ".json.abcd123_.tmp"),
                    ("request_contracts", ".json.tmp")):
                path = suite.runs / "auto_rerank" / directory / (f"qid_{qid}" + suffix)
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{')
            def verified(**kwargs):
                suite.assert_run_scope()
                return {}
            with patch.object(suite, "verify_manifest", side_effect=verified):
                status = suite.status()
            self.assertEqual(status["completed_new_runs"], 0)
            self.assertEqual(status["arms"]["auto_rerank"]["interrupted"], [])
            self.assertIn(qid, status["arms"]["auto_rerank"]["pending"])

    def test_committed_checkpoint_resumes_with_crash_leftover_untouched(self):
        query = mod.selected_queries("auto_rerank")[0]
        with tempfile.TemporaryDirectory() as d:
            suite = mod.ContinuationSuite(d)
            cp = suite.runs / "auto_rerank" / "_checkpoints" / f"qid_{query['query_id']}.json"
            cp.parent.mkdir(parents=True); cp.write_text('{"elapsed_seconds":23}')
            orphan = cp.with_name(cp.name + ".abcd123_.tmp"); orphan.write_text('{')
            manifest = {"administrative_source_sha256": mod.admin_hashes(),
                "manifest_sha256": "m", "config_sha256": "c", "config": {},
                "contracts": {"auto_rerank": {"contract_sha256": "t"}}}
            suite.assert_run_scope()
            with patch("strict_ablations.runtime.run_query", return_value={"final_text": "answer", "terminated_by": "done"}) as solver, \
                 patch.object(suite, "artifact_hashes", return_value={}), \
                 patch.object(suite, "validate_terminal", return_value={}), \
                 patch.object(suite, "verify_execution_artifacts", return_value={}):
                result = suite.run_one(manifest, "auto_rerank", query, resume_infrastructure=True)
                self.assertEqual(result["status"], "terminal_saved")
                solver.assert_called_once()
                self.assertEqual(solver.call_args.kwargs["output_dir"], suite.runs / "auto_rerank")
            self.assertEqual(cp.read_text(), '{"elapsed_seconds":23}')
            self.assertEqual(orphan.read_text(), '{')

    def test_temporary_names_are_directory_specific_and_require_authorized_qid(self):
        qid = mod.selected_queries("auto_rerank")[0]["query_id"]
        reused_qid = mod.selected_queries("search_paging")[0]["query_id"]
        cases = [("_checkpoints", f"qid_{reused_qid}.json.abcd123_.tmp"),
                 ("_checkpoints", f"qid_{qid}.json.tmp"),
                 ("request_contracts", f"qid_{qid}.json.abcd123_.tmp"),
                 ("api_ledgers", f"qid_{qid}.jsonl.tmp"),
                 ("_per_query", f"qid_{qid}.json.anythingelse"),
                 ("_per_query", f"qid_0{qid}.json.abcd123_")]
        for directory, name in cases:
            with self.subTest(directory=directory, name=name), tempfile.TemporaryDirectory() as d:
                suite = mod.ContinuationSuite(d)
                p = suite.runs / "auto_rerank" / directory / name
                p.parent.mkdir(parents=True); p.write_text('{}')
                with self.assertRaises(RuntimeError): suite.assert_run_scope()

    def test_temporary_symlink_and_directory_are_rejected(self):
        qid = mod.selected_queries("auto_rerank")[0]["query_id"]
        for kind in ("symlink", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                suite = mod.ContinuationSuite(d)
                p = suite.runs / "auto_rerank" / "_checkpoints" / f"qid_{qid}.json.abcd123_.tmp"
                p.parent.mkdir(parents=True)
                if kind == "symlink": p.symlink_to(Path(d) / "nonexistent")
                else: p.mkdir()
                with self.assertRaisesRegex(RuntimeError, "Unsafe"): suite.assert_run_scope()



if __name__ == "__main__":
    unittest.main()
