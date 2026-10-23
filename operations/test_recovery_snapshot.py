"""Recovery backup tests: synthetic new answers, real immutable old evidence.

No production identity/hash/artifact validator is mocked. Only Linux /proc is
mapped to a fixture. Private old CP/API bytes are copied without decoding them.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from operations import recovery_snapshot as snap


class RecoverySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def archive(self, *, name="reports/public.json", member_type=tarfile.REGTYPE, duplicate=False):
        archive, receipt = self.root / "snapshot.tar.gz", self.root / "receipt.json"
        data = b'{"public_final":"synthetic"}\n'
        with tarfile.open(archive, "w:gz") as tar:
            item = tarfile.TarInfo(name)
            item.type, item.linkname = member_type, "outside"
            item.size = len(data) if member_type == tarfile.REGTYPE else 0
            tar.addfile(item, io.BytesIO(data) if item.size else None)
            if duplicate: tar.addfile(item, io.BytesIO(data) if item.size else None)
        value = {"version": snap.VERSION, "ready_for_shutdown": True, "counts": snap.COUNTS,
                 "pipeline_phase": "complete", "worker_locks_acquired": 3, "active_questions": 0,
                 "worker_exit_verification": {"worker_exited": True}, "original_full_three_arm_suite_complete": False,
                 "control_model_requests": 0, "reused_auto20_model_requests": 0, "observed_final310_new_model_requests": 0,
                 "files": {name: {"bytes": item.size, "sha256": snap._sha(data)}},
                 "archive": {"sha256": snap.runner.file_sha(archive), "bytes": archive.stat().st_size}}
        value["receipt_sha256"] = snap.runner.digest(value)
        snap._write_new(receipt, value)
        return archive, receipt

    def test_safe_raw_byte_extraction_with_independent_receipt(self):
        archive, receipt = self.archive()
        result = snap.safe_extract(archive, receipt, self.root / "out",
            expected_receipt_sha256=snap.runner.file_sha(receipt), expected_receipt_bytes=receipt.stat().st_size)
        self.assertTrue(result["archive_bytes_verified"])
        self.assertEqual((self.root / "out/reports/public.json").read_bytes(), b'{"public_final":"synthetic"}\n')
        with self.assertRaisesRegex(RuntimeError, "previously nonexistent"):
            snap.safe_extract(archive, receipt, self.root / "out")

    def test_unsafe_tar_paths_and_links_are_rejected_before_writing(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("a//b", tarfile.REGTYPE), ("a/../b", tarfile.REGTYPE),
                           ("link", tarfile.LNKTYPE), ("link", tarfile.SYMTYPE), ("dir", tarfile.DIRTYPE)]:
            with self.subTest(name=name, kind=kind):
                for filename in ("snapshot.tar.gz", "receipt.json"): (self.root / filename).unlink(missing_ok=True)
                archive, receipt = self.archive(name=name, member_type=kind)
                with self.assertRaises(RuntimeError): snap.safe_extract(archive, receipt, self.root / "out")
                self.assertFalse((self.root / "out").exists())

    def test_duplicate_members_rejected(self):
        archive, receipt = self.archive(duplicate=True)
        with self.assertRaisesRegex(RuntimeError, "membership"):
            snap.safe_extract(archive, receipt, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_independent_archive_and_receipt_corruption_rejected(self):
        archive, receipt = self.archive()
        for kwargs in ({"expected_receipt_sha256": "0" * 64}, {"expected_receipt_bytes": 1}):
            with self.assertRaisesRegex(RuntimeError, "independent receipt"):
                snap.safe_extract(archive, receipt, self.root / "out", **kwargs)
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "archive SHA/byte"):
            snap.safe_extract(archive, receipt, self.root / "out")

    def test_count_conflation_or_unfinished_safety_state_rejected(self):
        _, receipt = self.archive()
        original = snap.runner.load(receipt)
        for change in ({"counts": {**snap.COUNTS, "validated_natural_terminals": 100}},
                       {"active_questions": 1}, {"worker_locks_acquired": 1},
                       {"pipeline_phase": "running_paging20"}, {"observed_final310_new_model_requests": 1}):
            value = {**original, **change}
            value["receipt_sha256"] = snap.runner.digest({k: v for k, v in value.items() if k != "receipt_sha256"})
            receipt.unlink(); snap._write_new(receipt, value)
            with self.assertRaises(RuntimeError): snap._receipt(receipt)

    def test_worker_exit_uses_start_ticks_and_boot(self):
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True)
        (proc / "sys/kernel/random/boot_id").write_text("test-boot")
        (proc / "123").mkdir()
        stat = proc / "123/stat"
        stat.write_text("123 (worker (name)) " + " ".join(["R"] + ["0"] * 18 + ["456"]))
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "test-boot"}
        with self.assertRaisesRegex(RuntimeError, "still running"):
            snap.verify_worker_exit(identity, proc_root=proc)
        stat.write_text("123 (worker) " + " ".join(["R"] + ["0"] * 18 + ["999"]))
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=proc)["reason"], "pid_reused")
        stat.unlink()
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=proc)["reason"], "pid_absent")

    def test_workspace_pool_manifests_are_excluded_but_small_receipts_are_retained(self):
        reports = self.root / "reports" / snap.DIRECTORY
        files = ["workspace_backups/293/workspace_manifest.json", "workspace_backups/293/workspaces.tar.gz",
                 "workspace_backups/293/receipt.json", "workspace_backups/293/deletion_receipt.json",
                 "workspace_backups/293/local_acknowledgement.json", "snapshots/old/public_finals.json",
                 "snapshots/old/workspace_backups/293/receipt.json", "downloaded/old/workspace_backups/293/receipt.json",
                 "downloaded/old/receipt.json", "phase_state.json"]
        for name in files:
            path = reports / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"{}")
        huge = reports / "workspace_backups/294/receipt.json"
        huge.parent.mkdir(parents=True); huge.write_bytes(b"x" * (1024 * 1024 + 1))
        selected = {p.relative_to(reports).as_posix() for p in snap._report_files(self.root, reports / "snapshots/new")}
        self.assertEqual(selected, {"workspace_backups/293/receipt.json", "workspace_backups/293/deletion_receipt.json",
                                    "workspace_backups/293/local_acknowledgement.json", "phase_state.json"})

    def test_no_ready_receipt_is_published_when_final_verification_fails(self):
        output = self.root / "snapshot_receipt.json"
        with patch.object(snap, "_verify_snapshot", side_effect=RuntimeError("Synthetic failed verification")):
            with self.assertRaisesRegex(RuntimeError, "Synthetic failed"):
                snap._publish_verified_receipt(self.root, output, {}, self.root / "archive.tar.gz")
        self.assertFalse(output.exists())
        with patch.object(snap, "_verify_snapshot", return_value={"verified": False}):
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                snap._publish_verified_receipt(self.root, output, {}, self.root / "archive.tar.gz")
        self.assertFalse(output.exists())

    def test_control_and_observed_final_evidence_labels_are_distinct(self):
        queries = snap.runner.shared.validate_queries()
        def row(arm, query):
            return {"arm": arm, "query_id": query["query_id"], "question": query["query"], "gold_answer": query["answer"],
                    "final_text": "synthetic", "final_sha256": "a" * 64, "row_sha256": "b" * 64,
                    "terminal_row_sha256": "b" * 64, "manifest_sha256": "c" * 64, "terminated_by": "done"}
        new = [row(arm, query) for arm, query in snap.prior.expected_jobs() if (arm, query["query_id"]) != ("auto_rerank", 310)]
        old = [row("auto_rerank", query) for query in queries[:20]]
        outcomes = [{"kind": "materialized_observed_final", "observation": {"final_text": "synthetic", "receipt_sha256": "e" * 64},
                     "final_sha256": "a" * 64, "outcome_sha256": "d" * 64, "manifest_sha256": "c" * 64}]
        control = {"records": [row("control", query) for query in queries], "manifest_sha256": "c" * 64}
        finals = snap._public_finals(new, [], old, outcomes, control)
        self.assertEqual(len(finals), 220)
        for record in finals:
            if record["arm"] == "control":
                self.assertEqual(record["archived_evidence"], "only_public_control_bundle")
                self.assertNotIn("complete_trace_checkpoint", record)
            elif record["query_id"] == 310:
                self.assertFalse(record["committed_source_terminal"])
                self.assertFalse(record["complete_trace_checkpoint"])
                self.assertEqual(record["outcome_kind"], "materialized_observed_final")
            else:
                self.assertEqual(record["archived_evidence"], "complete_trace_checkpoint")

    def build_fixture(self):
        repo = snap.SOURCE_ROOT
        incident = repo / "reports/strict_continue_v1/incidents/inode_quota_20261023_2311/extracted"
        recovery_data = repo / "reports/strict_continue_v1/recovery/resume_20261024"
        old_auto = repo / "reports/strict_auto20_v1/downloaded/completed_20261023_1600/artifacts"
        if not all(p.exists() for p in (incident, recovery_data, old_auto)):
            self.skipTest("Local immutable incident/auto20 backup fixtures are unavailable")
        server = self.root / "server"
        server.mkdir()
        source_root, auto_root, control_root = server / "source", server / "old20", server / "control"
        plan = snap.runner.load(recovery_data / "recovery_plan.json")
        for name in plan["source_pins"]:
            if name == "operations/recovery_evidence/incident_manifest.json":
                original = incident / "incident_manifest.json"
            elif name.startswith("operations/recovery_evidence/"):
                original = recovery_data / Path(name).name
            else:
                original = incident / "artifacts" / name
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)  # Opaque CP/API data; never decode it.
        source = snap.runner.load(source_root / "reports/strict_continue_v1/run_manifest.json")
        snap._pins(source_root, plan["source_pins"])
        snap._copy_pinned(old_auto, auto_root, source["reused_auto20_reference"]["files"])
        (auto_root / "reports/strict_auto20_v1/worker.lock").touch()
        snap._copy_pinned(repo, control_root, snap.prior.CONTROL_FILES)
        manifest = copy.deepcopy(source)
        manifest.pop("manifest_sha256")
        manifest.update(recovery_version=snap.recovery.VERSION, directory=snap.DIRECTORY,
            recovery_plan=plan, recovery_plan_sha256=snap.runner.digest(plan),
            source_continuation_manifest_sha256=snap.recovery.SOURCE_MANIFEST_SHA,
            administrative_source_sha256=snap.recovery.admin_hashes(), question_runs=100,
            new_scheduled_outcomes=97, reused_continuation_terminals=snap.recovery.REUSED_QIDS,
            execution_plan={"version": snap.recovery.VERSION,
                "stages": [{"name": s["name"], "arm": s["arm"], "query_ids": [q["query_id"] for q in s["queries"]]}
                           for s in snap.recovery.plan_stages(plan)],
                "selection_rule": snap.recovery.SELECTION, "original_full_three_arm_suite_complete": False},
            protocol_amendment=True, original_single_attempt_policy_preserved=False,
            storage_admission=snap.recovery.STORAGE_ADMISSION, scheduling="Synthetic offline fixture", terminal_policy="Synthetic offline fixture")
        manifest["manifest_sha256"] = snap.runner.digest(manifest)
        reports = server / "reports" / snap.DIRECTORY
        snap._write_new(reports / "run_manifest.json", manifest)
        (reports / "worker.lock").touch()
        epoch_path = "service_epochs/" + manifest["initial_service_epoch_sha256"] + ".json"
        snap._copy_pinned(source_root / "reports/strict_continue_v1", reports, [epoch_path])
        snap._copy_pinned(repo, server, [*manifest["source_sha256"], *manifest["administrative_source_sha256"],
                                       "operations/recovery_snapshot.py", "operations/continuation_snapshot.py"])
        from strict_ablations.test_runner import StrictAblationRunnerTests
        fixture = StrictAblationRunnerTests()
        suite = snap.recovery.RecoverySuite(server)
        fixture.root, fixture.suite = server, suite
        outcomes = []
        for arm, query in snap.jobs(plan):
            qid = query["query_id"]
            item = snap.recovery.question_plan(plan, qid) if arm == "auto_rerank" else None
            common = {"version": snap.recovery.VERSION, "arm": arm, "query_id": qid,
                      "manifest_sha256": manifest["manifest_sha256"], "selection_rule": snap.recovery.SELECTION,
                      "recovery_strategy": item["strategy"] if item else "original_first_attempt"}
            if (arm, qid) == ("auto_rerank", 310):
                observed = snap.recovery.validate_observed_final(source_root, item, plan)
                payload = {**common, "kind": "materialized_observed_final", "observation": observed,
                    "final_sha256": observed["final_sha256"], "natural_terminal": False,
                    "observed_natural_response": True, "new_model_requests": 0, "complete_trace_checkpoint": False}
            else:
                view = suite.effective_manifest(manifest, arm, qid)
                fixture.manifest, fixture.config = view, view["config"]
                row = fixture.write_terminal(arm, query, final="Synthetic offline backup test answer")
                # API diagnostics are deliberately opaque; backup must not parse them.
                (suite.runs / arm / "_traces" / f"qid_{qid}" / "single.api.jsonl").write_bytes(b"opaque test diagnostic bytes\n")
                payload = {**common, "kind": "natural_terminal", "natural_terminal": True,
                    "terminal_sha256": snap.runner.file_sha(suite.terminal_path(arm, qid)), "final_sha256": row["final_sha256"],
                    "effective_config_sha256": view["config_sha256"], "same_common_configuration": view["config"] == manifest["config"],
                    "original_single_attempt_policy_preserved": item is None, "accounting": suite.accounting(row, item)}
            outcomes.append(suite._save_outcome(payload))
        stages = []
        for stage in snap.recovery.plan_stages(plan):
            value = {"version": snap.recovery.VERSION, "passed": True, "stage": stage["name"],
                "manifest_sha256": manifest["manifest_sha256"], "selected_outcomes": [
                    {k: o[k] for k in ("query_id", "kind", "outcome_sha256")} for o in outcomes if o["arm"] == stage["arm"]]}
            stages.append(value)
            snap._write_new(reports / (stage["name"] + "_completion.json"), value)
        complete = {"version": snap.recovery.VERSION, "passed": True, "manifest_sha256": manifest["manifest_sha256"],
            "stages": stages, "new_outcome_counts": snap.KINDS, "validated_natural_terminals": 99,
            "target_coverage": 100, "target_count": 100, "control_model_requests": 0,
            "old_auto20_model_requests": 0, "observed_final310_new_model_requests": 0,
            "original_full_three_arm_suite_complete": False}
        snap._write_new(reports / "recovery_completion.json", complete)
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "fixture-boot"}
        snap._write_new(reports / "phase_state.json", {"phase": "complete", "worker_identity": identity,
            "manifest_sha256": manifest["manifest_sha256"], "receipt_sha256": snap.runner.file_sha(reports / "recovery_completion.json")})
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True)
        (proc / "sys/kernel/random/boot_id").write_text("fixture-boot")
        return server, source_root, auto_root, control_root, proc, manifest

    def test_full_export_delete_original_server_extract_verify(self):
        server, source, auto20, control, proc, manifest = self.build_fixture()
        reports = server / "reports" / snap.DIRECTORY
        original_manifest_bytes = (reports / "run_manifest.json").read_bytes()
        old_manifest_bytes = (source / "reports/strict_continue_v1/run_manifest.json").read_bytes()
        # Real admission catches changed contracts before backup, without patching validators.
        contract = server / f"runs/{snap.DIRECTORY}/auto_rerank/request_contracts/qid_293.json"
        previous = contract.read_bytes()
        value = snap.runner.load(contract); value["system_prompt"] += " altered fixture"
        contract.unlink(); snap._write_new(contract, value)
        with self.assertRaisesRegex(RuntimeError, "prompt/tools differ"):
            snap._outcomes(server, manifest, source)
        contract.write_bytes(previous)
        exit_check = snap.verify_worker_exit
        with patch.object(snap, "verify_worker_exit", side_effect=lambda identity: exit_check(identity, proc_root=proc)):
            result = snap.export_snapshot(server, reports / "snapshots/complete", source_root=source,
                reused_auto20_root=auto20, control_root=control)
        download = self.root / "download"
        download.mkdir()
        archive, receipt = download / "archive.tar.gz", download / "snapshot_receipt.json"
        shutil.copyfile(result["archive"], archive); shutil.copyfile(result["receipt"], receipt)
        shutil.rmtree(server)
        self.assertFalse(server.exists())
        extracted = download / "artifacts"
        snap.safe_extract(archive, receipt, extracted, expected_receipt_sha256=result["receipt_file_sha256"],
                          expected_receipt_bytes=result["receipt_bytes"])
        # Any accidental server/resource path access is a test failure after deletion.
        ordinary_open = Path.open
        def no_server_open(path, *args, **kwargs):
            self.assertFalse(str(path).startswith("/root/data/"), "Offline verification accessed a server path")
            self.assertFalse(path.is_relative_to(server), "Offline verification accessed the deleted fixture server")
            return ordinary_open(path, *args, **kwargs)
        with patch.object(Path, "open", new=no_server_open):
            verified = snap.load_verified_snapshot(extracted, receipt=receipt, archive=archive,
                expected_receipt_sha256=result["receipt_file_sha256"], expected_receipt_bytes=result["receipt_bytes"])
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["validated_natural_terminals"], 99)
        self.assertEqual(verified["materialized_observed_finals"], 1)
        self.assertEqual(verified["public_finals"], 220)
        self.assertTrue(verified["archive_bytes_verified"])
        proof = snap.runner.load(receipt)
        self.assertEqual((extracted / f"reports/{snap.DIRECTORY}/run_manifest.json").read_bytes(), original_manifest_bytes)
        self.assertEqual((extracted / proof["bundles"]["source_continuation"] / "reports/strict_continue_v1/run_manifest.json").read_bytes(), old_manifest_bytes)
        self.assertFalse((extracted / f"runs/{snap.DIRECTORY}/auto_rerank/_per_query/qid_310.json").exists())
        for relative in manifest["recovery_plan"]["source_pins"]:
            self.assertIn(proof["bundles"]["source_continuation"] + "/" + relative, proof["files"])
        # Merely having a required artifact on disk is insufficient: receipt membership is mandatory.
        altered = copy.deepcopy(proof)
        del altered["files"][f"runs/{snap.DIRECTORY}/auto_rerank/_checkpoints/qid_293.json"]
        altered["receipt_sha256"] = snap.runner.digest({k: v for k, v in altered.items() if k != "receipt_sha256"})
        altered_path = download / "altered_receipt.json"
        snap._write_new(altered_path, altered)
        with self.assertRaisesRegex(RuntimeError, "outside the independent receipt"):
            snap.load_verified_snapshot(extracted, receipt=altered_path, archive=archive)


if __name__ == "__main__": unittest.main()
