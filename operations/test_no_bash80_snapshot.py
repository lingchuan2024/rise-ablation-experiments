"""Offline backup boundaries and an actual server-deletion restore exercise.

Only synthetic new answers use real frozen artifact schemas and mock transport.
No model, serving process, corpus, index, or remote filesystem is contacted.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from operations import no_bash80_snapshot as snap


class NoBash80SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def archive(self, *, name="reports/public.json", member_type=tarfile.REGTYPE, duplicate=False):
        archive, receipt = self.root / "snapshot.tar.gz", self.root / "receipt.json"
        data = b'{"public_final":"synthetic"}\n'
        with tarfile.open(archive, "w:gz") as tar:
            item = tarfile.TarInfo(name)
            item.type, item.linkname = member_type, "outside"
            item.size = len(data) if member_type == tarfile.REGTYPE else 0
            tar.addfile(item, io.BytesIO(data) if item.size else None)
            if duplicate:
                tar.addfile(item, io.BytesIO(data) if item.size else None)
        value = {"version": snap.VERSION, "ready_for_shutdown": True, "counts": snap.COUNTS,
            "pipeline_phase": "complete", "worker_locks_acquired": 3, "active_questions": 0,
            "worker_lock_paths": list(snap.LOCK_FILES),
            "worker_exit_verification": {"worker_exited": True}, "control_model_requests": 0,
            "other_arm_model_requests": 0, "natural_error_reruns": 0,
            "files": {name: {"bytes": item.size, "sha256": snap._sha(data)}},
            "archive": {"sha256": snap.runner.file_sha(archive), "bytes": archive.stat().st_size}}
        value["receipt_sha256"] = snap.runner.digest(value)
        snap._write_new(receipt, value)
        return archive, receipt

    def extract(self, archive, receipt, output=None, **kwargs):
        defaults = {"expected_receipt_sha256": snap.runner.file_sha(receipt),
                    "expected_receipt_bytes": receipt.stat().st_size}
        return snap.safe_extract(archive, receipt, output or self.root / "out", **{**defaults, **kwargs})

    def test_extract_preserves_original_bytes_and_requires_fresh_output(self):
        archive, receipt = self.archive()
        result = self.extract(archive, receipt)
        self.assertTrue(result["archive_bytes_verified"])
        self.assertEqual((self.root / "out/reports/public.json").read_bytes(), b'{"public_final":"synthetic"}\n')
        with self.assertRaisesRegex(RuntimeError, "previously nonexistent"):
            self.extract(archive, receipt)

    def test_unsafe_paths_links_directory_and_duplicates_rejected_before_output(self):
        cases = [("../escape", tarfile.REGTYPE, False), ("/absolute", tarfile.REGTYPE, False),
                 ("a//b", tarfile.REGTYPE, False), ("a/../b", tarfile.REGTYPE, False),
                 ("link", tarfile.SYMTYPE, False), ("link", tarfile.LNKTYPE, False),
                 ("dir", tarfile.DIRTYPE, False), ("file", tarfile.REGTYPE, True)]
        for name, kind, duplicate in cases:
            with self.subTest(name=name, kind=kind, duplicate=duplicate):
                for filename in ("snapshot.tar.gz", "receipt.json"):
                    (self.root / filename).unlink(missing_ok=True)
                archive, receipt = self.archive(name=name, member_type=kind, duplicate=duplicate)
                with self.assertRaises(RuntimeError):
                    self.extract(archive, receipt)
                self.assertFalse((self.root / "out").exists())

    def test_receipt_sha_bytes_and_archive_sha_bytes_are_independent(self):
        archive, receipt = self.archive()
        for kwargs in ({"expected_receipt_sha256": "0" * 64}, {"expected_receipt_bytes": 1}):
            with self.assertRaisesRegex(RuntimeError, "independent receipt"):
                self.extract(archive, receipt, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "required"):
            snap.safe_extract(archive, receipt, self.root / "out")
        archive.write_bytes(archive.read_bytes() + b"corruption")
        with self.assertRaisesRegex(RuntimeError, "archive SHA/byte"):
            self.extract(archive, receipt)

    def test_safety_and_count_conflation_rejected(self):
        _, receipt = self.archive()
        original = snap.runner.load(receipt)
        for change in ({"counts": {**snap.COUNTS, "new_no_bash_rows": 19}}, {"active_questions": 1},
                       {"pipeline_phase": "running_no_bash80"}, {"worker_locks_acquired": 1},
                       {"other_arm_model_requests": 1}, {"control_model_requests": 1}, {"natural_error_reruns": 1}):
            value = {**original, **change}
            value["receipt_sha256"] = snap.runner.digest({k: v for k, v in value.items() if k != "receipt_sha256"})
            with self.assertRaises(RuntimeError):
                snap._validate_receipt(value)

    def test_raw_archive_members_are_checked_against_pins(self):
        archive, receipt = self.archive()
        proof = snap.runner.load(receipt)
        proof["files"]["reports/public.json"]["sha256"] = "0" * 64
        proof["receipt_sha256"] = snap.runner.digest({k: v for k, v in proof.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(RuntimeError, "member checksum"):
            snap._check_archive(archive, proof)
        proof["files"]["extra"] = {"sha256": "0" * 64, "bytes": 0}
        with self.assertRaisesRegex(RuntimeError, "membership"):
            snap._check_archive(archive, proof)

    def test_no_ready_receipt_before_full_validation(self):
        receipt = self.root / "snapshot_receipt.json"
        with patch.object(snap, "_extract_with_proof"), \
                patch.object(snap, "_verify_snapshot", side_effect=RuntimeError("synthetic verification failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic verification failure"):
                snap._publish_verified_receipt(self.root, receipt, {}, self.root / "archive.tar.gz")
        self.assertFalse(receipt.exists())
        with patch.object(snap, "_extract_with_proof"), patch.object(snap, "_verify_snapshot", return_value={"verified": False}):
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                snap._publish_verified_receipt(self.root, receipt, {}, self.root / "archive.tar.gz")
        self.assertFalse(receipt.exists())

    def process(self, *, state="R", ticks="456", boot="fixture-boot"):
        proc = self.root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
        (proc / "sys/kernel/random/boot_id").write_text(boot)
        (proc / "123").mkdir(exist_ok=True)
        (proc / "123/stat").write_text("123 (worker (nested)) " + " ".join([state] + ["0"] * 18 + [ticks]))
        return proc

    def test_worker_exit_identity_distinguishes_live_reused_pid_and_boot(self):
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "fixture-boot"}
        with self.assertRaisesRegex(RuntimeError, "still running"):
            snap.verify_worker_exit(identity, proc_root=self.process())
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=self.process(ticks="999"))["reason"], "pid_reused")
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=self.process(boot="another-boot"))["reason"], "different_boot")
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=self.process(state="Z"))["reason"], "exited_zombie")
        (self.root / "proc/123/stat").unlink()
        self.assertEqual(snap.verify_worker_exit(identity, proc_root=self.root / "proc")["reason"], "pid_absent")

    def test_report_selection_excludes_recursive_archives_and_pools(self):
        report = self.root / "reports" / snap.DIRECTORY
        for name in ("phase_state.json", "dispatches/qid_70.json", "snapshots/old/public_finals.json",
                     "downloaded/old/receipt.json", "workspace_backups/qid_70/workspace_manifest.json",
                     "archives/old/receipt.json"):
            path = report / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}")
        selected = {p.relative_to(report).as_posix() for p in snap._report_files(self.root, report / "snapshots/new")}
        self.assertEqual(selected, {"phase_state.json", "dispatches/qid_70.json"})

    def fixture(self):
        """Real pinned control plus synthetic twenty valid new artifact chains."""
        repo = snap.SOURCE_ROOT
        old_root = repo / "reports/strict_auto20_v1/downloaded/completed_20261023_1600/artifacts"
        if not old_root.is_dir():
            self.skipTest("Local immutable control/epoch fixture is unavailable")
        server = self.root / "server"
        server.mkdir()
        parent_source=repo / "reports/strict_no_bash20_v1/downloaded/complete_20261024_1832/extracted"
        parent_pins=snap.runner.load(repo / "operations/NO_BASH80_PARENT20_PINS.json")
        snap._copy_pinned(parent_source,server,list(parent_pins["files"]))
        suite = snap.job.NoBash80Suite(server)
        control_root = server / "control_bundle"
        snap._copy_pinned(repo, control_root, snap.CONTROL_FILES)
        manifest = snap.runner.load(old_root / "reports/strict_auto20_v1/run_manifest.json")
        manifest.pop("manifest_sha256")
        manifest.update(suite.execution_scope(snap.runner.shared.validate_queries()))
        manifest.update(control_root=str(control_root), analysis_contract=suite.analysis_contract())
        manifest["manifest_sha256"] = snap.runner.digest(manifest)
        snap._write_new(suite.reports / "run_manifest.json", manifest)
        epoch = f"service_epochs/{manifest['initial_service_epoch_sha256']}.json"
        snap._copy_pinned(old_root / "reports/strict_auto20_v1", suite.reports, [epoch])
        snap._copy_pinned(repo, server, [*manifest["source_sha256"], *manifest["administrative_source_sha256"], *snap.CHECKER_FILES,
            "operations/no_bash20_suite.py", "operations/test_no_bash20_suite.py",
            "operations/no_bash80_workspace_maintenance.py", "operations/test_no_bash80_workspace_maintenance.py"])
        fixture_locks = [server / "reports" / snap.DIRECTORY / "worker.lock",
                         server / "legacy_auto20/reports/strict_auto20_v1/worker.lock",
                         server / "reports/strict_ablations_v1/worker.lock"]
        for path in fixture_locks:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in snap.RUN_OPERATION_FILES:
            target = suite.reports / "operations" / name
            if name.endswith(".json"):
                snap._write_new(target, {"synthetic_offline_operation_fixture": True})
            else:
                target.write_bytes(b"opaque synthetic operation log\n")
        snap._write_new(suite.reports / "preflight_live.json", {"passed": True,
            "manifest_sha256": manifest["manifest_sha256"], "cohort_questions_used": 0,
            "tools_and_native_calls": {snap.ARM: {"passed": True}},
            "service": {"epoch_sha256": manifest["initial_service_epoch_sha256"]}})
        from strict_ablations.test_runner import StrictAblationRunnerTests
        fixture = StrictAblationRunnerTests()
        fixture.root, fixture.suite, fixture.manifest, fixture.config = server, suite, manifest, manifest["config"]
        identity = {"pid": 123, "proc_start_ticks": 456, "boot_id": "fixture-boot"}
        rows = []
        for query in snap.job.selected_queries():
            suite._dispatch(manifest, query, identity)
            row = fixture.write_terminal(snap.ARM, query, final="Synthetic offline no-bash backup answer")
            (suite.runs / snap.ARM / "_traces" / f"qid_{query['query_id']}" / "single.api.jsonl").write_bytes(b"opaque diagnostic fixture\n")
            rows.append(row)
        complete = {"version": snap.job.VERSION, "arm": snap.ARM, "passed": True,
            "manifest_sha256": manifest["manifest_sha256"], "selected_query_ids": manifest["selected_query_ids"],
            "terminal_rows_checked": 80, "question_runs": 80, "validated_natural_terminals": 80,
            "target_coverage": 80, "full_three_arm_suite_complete": False,
            "reused_no_bash_questions": 20, "merged_no_bash_target_coverage": 100,
            "parent20_reuse":snap.job.verify_parent20(server),
            "control_model_requests": 0, "other_arm_model_requests": 0, "natural_error_reruns": 0,
            "terminal_rows": [{"query_id": row["query_id"], "row_sha256": row["row_sha256"],
                "terminal_sha256": snap.runner.file_sha(suite.terminal_path(snap.ARM, row["query_id"])),
                "execution_artifacts": row["execution_artifacts"]} for row in rows]}
        snap._write_new(suite.reports / "no_bash80_completion.json", complete)
        snap._write_new(suite.reports / "phase_state.json", {"phase": "complete", "worker_identity": identity,
            "active_query_ids": [], "pending_query_ids": [], "manifest_sha256": manifest["manifest_sha256"],
            "receipt_sha256": snap.runner.file_sha(suite.reports / "no_bash80_completion.json")})
        proc = self.process()
        (proc / "123/stat").unlink()
        return server, control_root, proc, manifest

    def lock_paths(self, server):
        return [server / "reports" / snap.DIRECTORY / "worker.lock",
                server / "legacy_auto20/reports/strict_auto20_v1/worker.lock",
                server / "reports/strict_ablations_v1/worker.lock"]

    def export_fixture(self):
        server, control_root, proc, manifest = self.fixture()
        verify_exit = snap.verify_worker_exit
        with patch.object(snap, "verify_worker_exit", side_effect=lambda identity: verify_exit(identity, proc_root=proc)), \
                patch.object(snap.job.NoBash80Suite, "lock_paths", return_value=self.lock_paths(server)):
            result = snap.export_snapshot(server, server / f"reports/{snap.DIRECTORY}/snapshots/final", control_root=control_root)
        return server, manifest, result

    def test_full_export_delete_server_restore_and_verify_without_server_paths(self):
        server, manifest, result = self.export_fixture()
        original_manifest = (server / f"reports/{snap.DIRECTORY}/run_manifest.json").read_bytes()
        download = self.root / "download"
        download.mkdir()
        archive, receipt = download / "snapshot.tar.gz", download / "snapshot_receipt.json"
        shutil.copyfile(result["archive"], archive)
        shutil.copyfile(result["receipt"], receipt)
        shutil.rmtree(server)
        self.assertFalse(server.exists())
        extracted = download / "extracted"
        self.extract(archive, receipt, extracted)
        open_file = Path.open
        def forbid_server(path, *args, **kwargs):
            self.assertFalse(str(path).startswith("/root/data/"), "Offline verification attempted server resource access")
            self.assertFalse(path.is_relative_to(server), "Offline verification attempted deleted server access")
            return open_file(path, *args, **kwargs)
        with patch.object(Path, "open", new=forbid_server), patch("socket.create_connection", side_effect=AssertionError("No network")):
            verified = snap.load_verified_snapshot(extracted, receipt=receipt, archive=archive,
                expected_receipt_sha256=result["receipt_file_sha256"], expected_receipt_bytes=result["receipt_bytes"])
        self.assertTrue(verified["verified"])
        self.assertEqual({k: verified[k] for k in snap.COUNTS}, snap.COUNTS)
        self.assertEqual(verified["source_files_verified"], 135)
        self.assertEqual((extracted / f"reports/{snap.DIRECTORY}/run_manifest.json").read_bytes(), original_manifest)
        proof = snap.runner.load(receipt)
        self.assertEqual(proof["source_sha256"], manifest["source_sha256"])
        self.assertEqual(proof["worker_lock_paths"], list(snap.LOCK_FILES))
        self.assertFalse(any(snap.EXCLUDED_PARTS.intersection(Path(name).parts) for name in proof["files"]))
        # Execute the archived checker and its archived source tree in a fresh
        # interpreter, still denying both server roots. No validator is mocked.
        script = extracted / "operations/no_bash80_snapshot.py"
        guard = """from pathlib import Path
import runpy, sys
original = Path.open
deleted = Path(sys.argv.pop(1))
def offline_open(path, *args, **kwargs):
    if str(path).startswith('/root/data/') or path.is_relative_to(deleted):
        raise AssertionError('Server path used by standalone archived checker')
    return original(path, *args, **kwargs)
Path.open = offline_open
entry = sys.argv.pop(1)
sys.argv[0] = entry
runpy.run_path(entry, run_name='__main__')
"""
        standalone = subprocess.run([sys.executable, "-c", guard, str(server), str(script),
            "--root", str(extracted), "verify", "--archive", str(archive), "--receipt", str(receipt),
            "--expected-receipt-sha256", result["receipt_file_sha256"],
            "--expected-receipt-bytes", str(result["receipt_bytes"])],
            cwd=download, capture_output=True, text=True, timeout=120, check=True)
        self.assertIn('"verified": true', standalone.stdout)

    def test_changed_contract_missing_terminal_wrong_scope_and_dispatch_rejected(self):
        server, _, _, manifest = self.fixture()
        first = snap.job.selected_queries()[0]["query_id"]
        contract = server / f"runs/{snap.DIRECTORY}/{snap.ARM}/request_contracts/qid_{first}.json"
        original = contract.read_bytes()
        changed = snap.runner.load(contract)
        changed["system_prompt"] += " Synthetic drift"
        contract.unlink()
        snap._write_new(contract, changed)
        with self.assertRaisesRegex(RuntimeError, "prompt/tools differ"):
            snap._verify_rows(server, manifest)
        contract.write_bytes(original)
        dispatch = server / f"reports/{snap.DIRECTORY}/dispatches/qid_{first}.json"
        original_dispatch = dispatch.read_bytes()
        changed = snap.runner.load(dispatch)
        changed["attempt_id"] = 2
        changed["dispatch_sha256"] = snap.runner.digest({k: v for k, v in changed.items() if k != "dispatch_sha256"})
        dispatch.unlink(); snap._write_new(dispatch, changed)
        with self.assertRaisesRegex(RuntimeError, "Dispatch identity"):
            snap._verify_rows(server, manifest)
        dispatch.write_bytes(original_dispatch)
        wrong = server / f"runs/{snap.DIRECTORY}/auto_rerank"
        wrong.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Unexpected arm"):
            snap._verify_rows(server, manifest)
        wrong.rmdir()
        (server / snap.compact_paths(first)[0]).unlink()
        with self.assertRaisesRegex(RuntimeError, "Exactly fixed remaining80"):
            snap._verify_rows(server, manifest)

    def test_explicit_operation_allowlist_and_live_worker_block_export(self):
        server, control_root, proc, manifest = self.fixture()
        output = server / f"reports/{snap.DIRECTORY}/snapshots/final"
        verify_exit = snap.verify_worker_exit
        with patch.object(snap, "verify_worker_exit", side_effect=lambda identity: verify_exit(identity, proc_root=proc)), \
                patch.object(snap.job.NoBash80Suite, "lock_paths", return_value=self.lock_paths(server)):
            with self.assertRaisesRegex(RuntimeError, "allowlist"):
                snap.export_snapshot(server, output, control_root=control_root, operation_files=["operations/arbitrary.json"])
        self.assertFalse(output.exists())
        self.process()
        with patch.object(snap, "verify_worker_exit", side_effect=lambda identity: verify_exit(identity, proc_root=proc)), \
                patch.object(snap.job.NoBash80Suite, "lock_paths", return_value=self.lock_paths(server)):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                snap.export_snapshot(server, output, control_root=control_root)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
