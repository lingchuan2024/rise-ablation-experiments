"""No network, GPU, A solver call or benchmark model generation in these tests."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from operations import resume_b_only as resume
from strict_experiment import runner


class ResumeBOnlyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name, path in (("RUNS", self.folder / "runs"), ("REPORTS", self.folder / "reports")):
            patch = mock.patch.object(runner, name, path); patch.start(); self.addCleanup(patch.stop)
        self.epoch_dir = runner.REPORTS / "operations/epoch_test"
        self.epoch_dir.mkdir(parents=True)
        self.manifest = {"manifest_sha256": "original-manifest", "config_sha256": "original-config",
            "config": {"concurrency": 4}}

    def receipt(self, **changes):
        receipt = {"version": "static-model-identity", "pid": 132, "process_start_ticks": 100,
            "configured_command": ["same-vllm", "serve", "same-model", "--max-num-seqs", "4"],
            "parser_files": [{"sha256": "unchanged-parser"}], "model_root": "same-model",
            "packages": {"vllm": "0.17", "torch": "same", "transformers": "same"}, **changes}
        receipt["identity_sha256"] = resume.identity.digest(receipt)
        return receipt

    def test_restart_accepts_only_real_lifecycle_changes(self):
        previous = self.receipt()
        actual = self.receipt(pid=55, process_start_ticks=999)
        self.assertEqual(resume.assert_lifecycle_only(previous, actual), resume.identity.digest(resume.static_identity(previous)))
        for changes in ({"configured_command": ["changed-argv"]}, {"parser_files": [{"sha256": "changed"}]},
                {"packages": {"vllm": "different"}}, {"model_root": "different-model"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "static"):
                resume.assert_lifecycle_only(previous, self.receipt(pid=55, process_start_ticks=999, **changes))

    def test_epoch_returns_new_pid_and_never_forges_the_old_process(self):
        previous, actual = self.receipt(), self.receipt(pid=55, process_start_ticks=999)
        service = self.folder / "service.json"
        service.write_text(json.dumps({"actual_identity": previous}))
        self.manifest["config"]["model_identity"] = {"service_config": str(service)}
        with mock.patch.object(resume.identity, "collect_service_identity", return_value=actual) as collect, \
                mock.patch.object(resume, "host_capacity_evidence", return_value={"passed": True}):
            epoch = resume.create_epoch(self.manifest, self.epoch_dir, 55)
            measured = resume.verify_epoch(self.manifest, self.epoch_dir)
        self.assertEqual(epoch["actual_identity"]["pid"], 55)
        self.assertEqual(measured["pid"], 55)
        self.assertEqual(measured["process_start_ticks"], 999)
        self.assertEqual(json.loads(service.read_text())["actual_identity"]["pid"], 132)
        self.assertEqual(collect.call_args.args, (str(service), 55))

    def test_verify_state_binding_is_temporary_and_returns_real_epoch(self):
        original = resume.identity.verify_state
        with mock.patch.object(resume, "verify_epoch", return_value={"pid": 55, "verified": True}):
            with resume.bind_epoch(self.manifest, self.epoch_dir):
                self.assertEqual(resume.identity.verify_state(self.manifest["config"])["pid"], 55)
        self.assertIs(resume.identity.verify_state, original)

    def test_blocked_v1_audit_is_rejected_even_if_items_look_clean(self):
        path = self.epoch_dir / "audit.json"
        path.write_text(json.dumps({"manifest_sha256": "original-manifest", "config_sha256": "original-config",
            "passed": False, "unfinished_admitted": []}))
        with self.assertRaisesRegex(RuntimeError, "passed audit"):
            resume.audit_entries(self.manifest, path)

    def partial_fixture(self):
        paths = resume.partial_paths(376)
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fixture": name, "elapsed_seconds": 2818.406759}))
        return {"query_id": 376, "decision": "resume_same_checkpoint", "problems": [],
            "file_sha256": {name: runner.file_sha(path) for name, path in paths.items()},
            "successful_ledger_tail_not_in_checkpoint": {"rows": 0}, "budget_charge_seconds": 0,
            "unknown_inflight_tokens": None}

    def test_partial_resume_requires_exact_audit_bytes_and_no_unknown_success_tail(self):
        item = self.partial_fixture()
        self.assertIs(resume.verify_partial_audit(self.manifest, {"query_id": 376}, {376: item}), item)
        unknown = copy.deepcopy(item); unknown["successful_ledger_tail_not_in_checkpoint"]["rows"] = None
        with self.assertRaisesRegex(RuntimeError, "unknown successful"):
            resume.verify_partial_audit(self.manifest, {"query_id": 376}, {376: unknown})
        resume.partial_paths(376)["ledger"].write_text("changed after audit")
        with self.assertRaisesRegex(RuntimeError, "changed after audit"):
            resume.verify_partial_audit(self.manifest, {"query_id": 376}, {376: item})

    def test_zero_charge_uses_original_checkpoint_loader_and_preserves_snapshot(self):
        from rise.turn_checkpoint import TurnCheckpoint
        item = self.partial_fixture(); path = resume.partial_paths(376)["checkpoint"]
        original, data = TurnCheckpoint.load, path.read_bytes()
        with resume.budget_overlays(self.epoch_dir, {376: item}):
            self.assertIs(TurnCheckpoint.load, original)
            self.assertEqual(path.read_bytes(), data)
        backup = self.epoch_dir / "private_checkpoint_backups/qid_376.json"
        self.assertEqual(backup.read_bytes(), data)
        self.assertEqual(path.read_bytes(), data)

    def test_b_only_schedules_72_remaining_and_leaves_all_a_artifacts_unchanged(self):
        rows = [{"query_id": qid, "query": "synthetic fixture"} for qid in range(1, 101)]
        for qid in range(1, 30):
            path = runner.RUNS / "A/_per_query" / f"qid_{qid}.json"
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("FROZEN_A")
        for qid in range(1, 29):
            runner.atomic_json(runner.terminal_path("B", qid), {"query_id": qid})
        before = resume.a_snapshot()
        epoch = {"epoch_sha256": "epoch-hash", "a_before": before,
            "actual_identity": {"pid": 55, "process_start_ticks": 999}}
        runner.atomic_json(self.epoch_dir / "b_epoch.json", epoch)
        runner.atomic_json(self.epoch_dir / "b_short_preflight.json", {"passed": True, "epoch_sha256": "epoch-hash"})
        audit = self.epoch_dir / "audit.json"
        runner.atomic_json(audit, {"passed": True, "manifest_sha256": "original-manifest", "config_sha256": "original-config", "unfinished_admitted": []})
        calls = []
        def one(manifest, arm, query, **kwargs):
            self.assertEqual(arm, "B"); calls.append(query["query_id"])
            runner.atomic_json(runner.terminal_path("B", query["query_id"]), {"query_id": query["query_id"]}, immutable=True)
            return {"status": "terminal_saved"}
        with mock.patch.object(runner, "validate_queries", return_value=rows), \
                mock.patch.object(runner, "validate_terminal", side_effect=lambda row, *args: row), \
                mock.patch.object(runner, "verify_execution_artifacts"), \
                mock.patch.object(runner, "verify_manifest", return_value=self.manifest), \
                mock.patch.object(runner, "run_one", side_effect=one), \
                mock.patch.object(resume, "verify_service_b"), mock.patch.object(resume, "verify_epoch"), \
                mock.patch.object(resume.identity, "_process", return_value={"pid": 55, "process_start_ticks": 999}), \
                redirect_stdout(io.StringIO()):
            receipt = resume.run_b(self.manifest, self.epoch_dir, audit)
        self.assertEqual(receipt["b_terminal_count"], 100)
        self.assertEqual(receipt["a_started"], 0)
        self.assertEqual(len(calls), 72)
        self.assertEqual(set(calls), set(range(29, 101)))
        self.assertEqual(resume.a_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
