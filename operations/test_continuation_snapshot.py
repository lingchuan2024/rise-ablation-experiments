"""Backup boundary tests; no solver, GPU, network, or private reasoning output."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from operations import continuation_snapshot as snap


class SnapshotBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, *, name="reports/public.json", member_type=tarfile.REGTYPE):
        archive = self.root / "snapshot.tar.gz"
        data = b'{"public_final":"answer"}\n'
        with tarfile.open(archive, "w:gz") as tar:
            item = tarfile.TarInfo(name)
            item.type = member_type
            item.linkname = "outside"
            item.size = len(data) if member_type == tarfile.REGTYPE else 0
            tar.addfile(item, io.BytesIO(data) if member_type == tarfile.REGTYPE else None)
        proof = {"version": snap.VERSION, "ready_for_shutdown": True, "terminal_rows": 100,
            "reused_auto20_rows": 20, "pipeline_phase": "complete", "worker_lock_acquired": True,
            "worker_exit_verification": {"worker_exited": True}, "control_model_requests": 0,
            "other_arm_model_requests": 0, "reused_auto20_model_requests": 0,
            "files": {name: {"bytes": item.size, "sha256": snap._sha(data)}},
            "archive": {"sha256": snap.runner.file_sha(archive), "bytes": archive.stat().st_size}}
        proof["receipt_sha256"] = snap.runner.digest(proof)
        receipt = self.root / "receipt.json"
        snap._write_new(receipt, proof)
        return archive, receipt

    def test_valid_archive_independent_receipt_and_raw_bytes(self):
        archive, receipt = self.archive()
        output = self.root / "extracted"
        result = snap.safe_extract(archive, receipt, output,
            expected_receipt_sha256=snap.runner.file_sha(receipt), expected_receipt_bytes=receipt.stat().st_size)
        self.assertTrue(result["archive_bytes_verified"])
        self.assertEqual((output / "reports/public.json").read_bytes(), b'{"public_final":"answer"}\n')
        with self.assertRaisesRegex(RuntimeError, "previously nonexistent"):
            snap.safe_extract(archive, receipt, output)

    def test_download_corruption_is_rejected_before_extraction(self):
        archive, receipt = self.archive()
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "SHA/byte size"):
            snap.safe_extract(archive, receipt, self.root / "extracted")
        self.assertFalse((self.root / "extracted").exists())

    def test_independent_receipt_mismatch(self):
        archive, receipt = self.archive()
        with self.assertRaisesRegex(RuntimeError, "receipt SHA differs"):
            snap.safe_extract(archive, receipt, self.root / "extracted", expected_receipt_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "receipt byte size"):
            snap.safe_extract(archive, receipt, self.root / "extracted", expected_receipt_bytes=1)

    def test_reject_tar_links(self):
        for member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(member_type=member_type):
                for filename in ("snapshot.tar.gz", "receipt.json"):
                    (self.root / filename).unlink(missing_ok=True)
                archive, receipt = self.archive(member_type=member_type)
                with self.assertRaisesRegex(RuntimeError, "unsafe member"):
                    snap.safe_extract(archive, receipt, self.root / "extracted")
                self.assertFalse((self.root / "extracted").exists())

    def test_reject_traversal_before_creating_output(self):
        for name in ("../escape", "/absolute", "double//slash", "x/../escape"):
            with self.subTest(name=name):
                for filename in ("snapshot.tar.gz", "receipt.json"):
                    (self.root / filename).unlink(missing_ok=True)
                archive, receipt = self.archive(name=name)
                with self.assertRaises(RuntimeError):
                    snap.safe_extract(archive, receipt, self.root / "extracted")
                self.assertFalse((self.root / "extracted").exists())

    def test_symlink_parent_rejected(self):
        (self.root / "alias").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            snap._safe(self.root, "alias/file")

    def test_existing_twenty_is_not_continuation_completion(self):
        for arm in ("auto_rerank", "search_paging"):
            (self.root / "runs" / snap.DIRECTORY / arm / "_per_query").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Exactly auto80 and paging20"):
            snap._verify_rows(self.root, {})

    def test_running_phase_rejected(self):
        reports = self.root / "reports" / snap.DIRECTORY
        reports.mkdir(parents=True)
        snap._write_new(reports / "phase_state.json", {"phase": "running_paging20", "manifest_sha256": "example"})
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            snap._verify_completion(self.root, {"manifest_sha256": "example"}, [])

    def process(self, *, state="R", ticks="456", boot="original"):
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
        (proc / "sys/kernel/random/boot_id").write_text(boot)
        (proc / "123").mkdir(exist_ok=True)
        (proc / "123/stat").write_text("123 (worker (with spaces)) " + " ".join([state] + ["0"] * 18 + [ticks]))
        return proc

    def test_live_worker_blocks_even_if_phase_complete(self):
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "original"}
        with self.assertRaisesRegex(RuntimeError, "still running"):
            snap.verify_worker_exit(identity, proc_root=self.process())

    def test_pid_reuse_and_missing_worker_are_distinguished(self):
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "original"}
        proc = self.process(ticks="789")
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=proc)["reason"], "pid_reused")
        (proc / "123/stat").unlink()
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=proc)["reason"], "pid_absent")

    def test_real_completed_auto20_reusable_without_server_paths(self):
        old_root = snap.SOURCE_ROOT / "reports/strict_auto20_v1/downloaded/completed_20261023_1600/artifacts"
        if not old_root.exists():
            self.skipTest("Existing local verified auto20 backup is unavailable")
        old = snap.runner.load(old_root / "reports/strict_auto20_v1/run_manifest.json")
        reference = snap.verify_reused_auto20(old_root, common_config=old["config"],
                                            control_reference=old["control_reference"])
        manifest = {"config": old["config"], "source_sha256": old["source_sha256"],
                    "control_reference": old["control_reference"], "reused_auto20_reference": reference}
        self.assertEqual(len(snap._verify_reused(old_root, manifest)), 20)
        self.assertEqual(len(snap.runner.source_hashes()), 135)
        changed = copy.deepcopy(manifest)
        changed["reused_auto20_reference"]["files"]["reports/strict_auto20_v1/run_manifest.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "reference differs"):
            snap._verify_reused(old_root, changed)

    def test_complete_offline_restore_after_original_server_tree_deleted(self):
        """Real old20/control bytes plus synthetic new metadata test backup plumbing.

        The frozen fixture writes actual RequestRecorder/AgentRun/ToolSurface
        schemas using a zero-network mock transport. No hash, row, contract,
        checkpoint, or artifact validator is patched. These are synthetic test
        runs, never experimental answers; only the Linux /proc location is mocked.
        """
        original_old = snap.SOURCE_ROOT / "reports/strict_auto20_v1/downloaded/completed_20261023_1600/artifacts"
        if not original_old.exists():
            self.skipTest("Existing local verified auto20 backup is unavailable")
        server = self.root / "server"
        server.mkdir()
        old = snap.runner.load(original_old / "reports/strict_auto20_v1/run_manifest.json")
        reference = snap.verify_reused_auto20(original_old, common_config=old["config"],
                                            control_reference=old["control_reference"])
        reuse_root, control_root = server / "old_auto20", server / "old_control"
        snap._copy_pinned(original_old, reuse_root, reference["files"])
        snap._copy_pinned(snap.SOURCE_ROOT, control_root, snap.CONTROL_FILES)
        reference["root"] = str(reuse_root)
        suite = snap.ContinuationSuite(server)
        manifest = {**copy.deepcopy(old), **suite.execution_scope(snap.runner.shared.validate_queries()),
                    "control_root": str(control_root), "reused_auto20_reference": reference,
                    "analysis_contract": suite.analysis_contract()}
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = snap.runner.digest(manifest)
        reports = server / "reports" / snap.DIRECTORY
        snap._write_new(reports / "run_manifest.json", manifest)
        (reports / "worker.lock").touch()
        epoch_relative = "service_epochs/" + manifest["initial_service_epoch_sha256"] + ".json"
        snap._copy_pinned(original_old / "reports/strict_auto20_v1", reports, [epoch_relative])
        snap._copy_pinned(snap.SOURCE_ROOT, server,
                         [*manifest["source_sha256"], *manifest["administrative_source_sha256"],
                          "operations/continuation_snapshot.py"])
        from strict_ablations.test_runner import StrictAblationRunnerTests
        fixture = StrictAblationRunnerTests()
        fixture.root, fixture.suite = server, snap.ArtifactSuite(server)
        fixture.manifest, fixture.config = manifest, manifest["config"]
        rows = [fixture.write_terminal(arm, query, final="Synthetic backup test answer")
                for arm, query in snap.expected_jobs()]
        # Confirm the fixture really traverses the production request-contract gate.
        arm, query = snap.expected_jobs()[0]
        contract_path = server / "runs" / snap.DIRECTORY / arm / "request_contracts" / f"qid_{query['query_id']}.json"
        original_contract = contract_path.read_bytes()
        changed_contract = json.loads(original_contract)
        changed_contract["system_prompt"] += " synthetic contract drift"
        contract_path.write_text(json.dumps(changed_contract))
        with self.assertRaisesRegex(RuntimeError, "prompt/tools differ"):
            fixture.suite.verify_execution_artifacts(manifest, arm, query, rows[0])
        contract_path.write_bytes(original_contract)
        stage_receipts = []
        for name, arm, count in (("auto80", "auto_rerank", 80), ("paging20", "search_paging", 20)):
            stage = {"passed": True, "manifest_sha256": manifest["manifest_sha256"], "question_runs": count,
                "terminal_rows_checked": count, "natural_error_reruns": 0, "control_model_requests": 0,
                "other_arm_model_requests": 0, "reused_auto20_model_requests": 0,
                "terminal_rows": [{"arm": arm, "query_id": row["query_id"], "row_sha256": row["row_sha256"],
                    "file_sha256": snap.runner.file_sha(server / "runs" / snap.DIRECTORY / arm /
                                                        "_per_query" / f"qid_{row['query_id']}.json"),
                    "execution_artifacts": row["execution_artifacts"]} for row in rows if row["arm"] == arm]}
            snap._write_new(reports / (name + "_completion.json"), stage)
            stage_receipts.append(stage)
        completion = {"passed": True, "manifest_sha256": manifest["manifest_sha256"], "question_runs": 100,
            "terminal_rows_checked": 100, "natural_error_reruns": 0, "control_model_requests": 0,
            "other_arm_model_requests": 0, "reused_auto20_model_requests": 0, "stages": stage_receipts}
        snap._write_new(reports / "continuation_completion.json", completion)
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "original"}
        snap._write_new(reports / "phase_state.json", {"phase": "complete", "worker_identity": identity,
            "manifest_sha256": manifest["manifest_sha256"], "terminal_rows_checked": 100,
            "receipt_sha256": snap.runner.file_sha(reports / "continuation_completion.json")})
        proc = self.process()
        (proc / "123/stat").unlink()
        original_exit_check = snap.verify_worker_exit
        with patch.object(snap, "verify_worker_exit", side_effect=lambda value: original_exit_check(value, proc_root=proc)):
            result = snap.export_snapshot(server, reports / "snapshots/final")
            download = self.root / "download"
            download.mkdir()
            archive, receipt = download / "snapshot.tar.gz", download / "receipt.json"
            shutil.copyfile(result["archive"], archive)
            shutil.copyfile(result["receipt"], receipt)
            shutil.rmtree(server)
            extracted = download / "artifacts"
            snap.safe_extract(archive, receipt, extracted,
                expected_receipt_sha256=result["receipt_file_sha256"], expected_receipt_bytes=result["receipt_bytes"])
            verified = snap.load_verified_snapshot(extracted, receipt=receipt, archive=archive,
                expected_receipt_sha256=result["receipt_file_sha256"], expected_receipt_bytes=result["receipt_bytes"])
        self.assertFalse(server.exists())
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["terminal_rows"], 100)
        self.assertEqual(verified["merged_auto_rerank_rows"], 100)
        self.assertEqual(verified["public_finals"], 220)
        self.assertTrue(verified["archive_bytes_verified"])


if __name__ == "__main__":
    unittest.main()
