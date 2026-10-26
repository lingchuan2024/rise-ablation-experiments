"""Real filesystem backup/verification/deletion tests; synthetic opaque bodies only."""
from pathlib import Path
import copy
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operations import terminal_workspace_maintenance as m


class WorkspaceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "remote"
        self.root.mkdir()
        self.arm, self.qid = "auto_rerank", 293
        self.work, self.runtime = m.targets(self.root, self.arm, self.qid).values()
        (self.work / "nested").mkdir(parents=True)
        self.runtime.mkdir(parents=True)
        self.corpus = self.root / "corpus.bin"
        self.corpus.write_bytes(b"opaque source bytes\x00" * 100)
        os.link(self.corpus, self.work / "nested" / "first.bin")
        os.link(self.corpus, self.work / "nested" / "second.bin")
        (self.work / "one.txt").write_bytes(b"opaque independent bytes")
        (self.runtime / "runtime.bin").write_bytes(b"opaque runtime bytes")
        self.other = self.work.parent / "qid_317"
        self.other.mkdir()
        (self.other / "active.bin").write_bytes(b"active question must remain untouched")
        self.compact = self.root / "runs" / m.NAMESPACE / self.arm / "_per_query" / "qid_293.json"
        self.compact.parent.mkdir()
        self.compact.write_bytes(b"immutable synthetic terminal")
        self.binding = {"source_root": str(self.root), "namespace": m.NAMESPACE,
                        "arm": self.arm, "query_id": self.qid, "terminal_sha256": m.file_sha(self.compact)}
        self.audit = patch.object(m, "audit_terminal", side_effect=self.audit_fixture)
        self.audit.start()
        self.addCleanup(self.audit.stop)
        self.addCleanup(self.temp.cleanup)

    def audit_fixture(self, root, arm, qid):
        if (Path(root).resolve(), arm, qid) != (self.root, self.arm, self.qid):
            raise RuntimeError("Wrong natural terminal")
        if m.file_sha(self.compact) != self.binding["terminal_sha256"]:
            raise RuntimeError("Compact evidence changed")
        return self.binding

    def export(self):
        output = self.root / "reports" / m.NAMESPACE / "workspace_backups" / "fresh"
        self.result = m.export_workspace(self.root, self.arm, self.qid, output)
        self.remote_receipt = output / "receipt.json"
        self.local = self.base / "local"
        self.local.mkdir()
        archive = self.local / "workspace.tar.gz"
        receipt = self.local / "receipt.json"
        shutil.copyfile(output / "workspace.tar.gz", archive)
        shutil.copyfile(self.remote_receipt, receipt)
        self.ack = m.verify_archive(archive, receipt,
            expected_receipt_sha256=self.result["receipt_file_sha256"],
            expected_receipt_bytes=self.result["receipt_bytes"])
        return self.result

    def delete(self, ack=None):
        return m.delete_workspace(self.root, self.arm, self.qid, self.remote_receipt, self.ack if ack is None else ack)

    def assert_preserved(self):
        self.assertTrue(self.work.parent.is_dir())
        self.assertTrue(self.runtime.parent.is_dir())
        self.assertEqual(self.corpus.read_bytes(), b"opaque source bytes\x00" * 100)
        self.assertEqual((self.other / "active.bin").read_bytes(), b"active question must remain untouched")
        self.assertEqual(m.file_sha(self.compact), self.binding["terminal_sha256"])

    def test_full_download_delete_and_offline_archive_verifies_after_remote_removed(self):
        result = self.export()
        self.assertEqual(result["files"], 4)
        self.assertEqual(result["directories"], 3)
        deleted = self.delete()
        self.assertTrue(deleted["deleted"])
        self.assertEqual((deleted["files_removed"], deleted["directories_removed"]), (4, 3))
        self.assertFalse(self.work.exists()); self.assertFalse(self.runtime.exists())
        self.assert_preserved()
        shutil.rmtree(self.root)
        ack = m.verify_archive(self.local / "workspace.tar.gz", self.local / "receipt.json",
            expected_receipt_sha256=result["receipt_file_sha256"], expected_receipt_bytes=result["receipt_bytes"])
        self.assertTrue(ack["verified"])

    def test_external_hardlink_add_and_remove_after_export_are_allowed(self):
        old_link = self.root / "external_old.bin"
        os.link(self.corpus, old_link)
        self.export()
        os.link(self.corpus, self.root / "external_new.bin")
        old_link.unlink()
        self.assertTrue(self.delete()["deleted"])
        self.assert_preserved()

    def test_external_hardlink_change_during_export_is_allowed(self):
        original = m.HashingReader.read
        triggered = []
        def read(reader, n=-1):
            if not triggered:
                triggered.append(True)
                os.link(self.corpus, self.root / "other_question_link.bin")
            return original(reader, n)
        with patch.object(m.HashingReader, "read", read):
            self.export()
        self.assertTrue(self.delete()["deleted"])
        self.assert_preserved()

    def test_content_change_even_with_restored_size_mtime_is_rejected_before_any_unlink(self):
        self.export()
        path = self.work / "one.txt"
        saved = path.stat()
        path.write_bytes(b"X" * saved.st_size)
        os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, "content no longer matches"):
            self.delete()
        self.assertTrue((self.work / "nested" / "first.bin").exists())
        self.assertTrue((self.runtime / "runtime.bin").exists())

    def test_replacement_inode_is_rejected_before_any_unlink(self):
        self.export()
        path = self.work / "one.txt"
        replacement = self.base / "replacement"
        replacement.write_bytes(path.read_bytes())
        saved = path.stat()
        os.utime(replacement, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        replacement.replace(path)
        with self.assertRaisesRegex(RuntimeError, "tree/stat differs"):
            self.delete()
        self.assertTrue((self.work / "nested" / "first.bin").exists())

    def test_extra_file_is_rejected_before_any_unlink(self):
        self.export()
        (self.work / "late.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(RuntimeError, "tree/stat differs"):
            self.delete()
        self.assertTrue((self.work / "nested" / "first.bin").exists())

    def test_missing_acknowledgement_is_rejected(self):
        self.export()
        with self.assertRaisesRegex(RuntimeError, "acknowledgement_sha256"):
            self.delete({})
        self.assertTrue(self.work.exists())

    def test_wrong_backup_ack_is_rejected(self):
        self.export()
        ack = copy.deepcopy(self.ack)
        ack["archive"]["sha256"] = "0" * 64
        ack["acknowledgement_sha256"] = m.digest({k: v for k, v in ack.items() if k != "acknowledgement_sha256"})
        with self.assertRaisesRegex(RuntimeError, "does not authorize"):
            self.delete(ack)
        self.assertTrue(self.work.exists())

    def test_changed_compact_outcome_refuses_delete(self):
        self.export()
        self.compact.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "Compact evidence changed"):
            self.delete()
        self.assertTrue(self.work.exists())

    def test_symlink_source_is_rejected(self):
        (self.work / "escape").symlink_to(self.corpus)
        with self.assertRaisesRegex(RuntimeError, "Symlink|escapes its root"):
            self.export()
        self.assertTrue(self.work.exists())

    def test_external_links_change_between_prehash_and_unlink_are_allowed(self):
        self.export()
        original = m.verify_open_file
        calls = []
        def verify(descriptor, row, cache):
            calls.append(row["member"])
            if len(calls) == 5:
                os.link(self.corpus, self.root / "active_link.bin")
            elif len(calls) == 6:
                (self.root / "active_link.bin").unlink()
            return original(descriptor, row, cache)
        with patch.object(m, "verify_open_file", verify):
            self.assertTrue(self.delete()["deleted"])
        self.assert_preserved()

    def test_invalid_arm_cannot_write_lock_outside_namespace(self):
        with self.assertRaisesRegex(RuntimeError, "authorized arm"):
            with m.operation_lock(self.root, "../../escape", self.qid):
                self.fail("invalid arm accepted")

    def test_corrupt_archive_and_wrong_receipt_bytes_are_rejected(self):
        self.export()
        archive, receipt = self.local / "workspace.tar.gz", self.local / "receipt.json"
        with self.assertRaisesRegex(RuntimeError, "receipt byte count"):
            m.verify_archive(archive, receipt, expected_receipt_bytes=1)
        with archive.open("ab") as out: out.write(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "archive SHA/bytes"):
            m.verify_archive(archive, receipt)

    def test_unsafe_archive_entry_rejected_even_with_matching_archive_receipt(self):
        self.export()
        archive, receipt_path = self.local / "unsafe.tar.gz", self.local / "unsafe_receipt.json"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        receipt = m.load(self.local / "receipt.json")
        receipt["archive"] = {"sha256": m.file_sha(archive), "bytes": archive.stat().st_size}
        receipt["receipt_sha256"] = m.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        m.write_new(receipt_path, receipt)
        with self.assertRaisesRegex(RuntimeError, "Unsafe relative path"):
            m.verify_archive(archive, receipt_path)


class TerminalAdmissionTests(unittest.TestCase):
    def setUp(self):
        from operations import recovery_suite
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.arm, self.qid = "auto_rerank", 293
        self.reports = self.root / "reports" / m.NAMESPACE
        self.row_path = self.root / "runs" / m.NAMESPACE / self.arm / "_per_query" / "qid_293.json"
        self.row = {"final_sha256": "final", "row_sha256": "row", "execution_artifacts": {"opaque": "sha"}}
        m.write_new(self.row_path, self.row)
        self.outcome_path = self.reports / "outcomes" / self.arm / "qid_293.json"
        self.manifest = {"manifest_sha256": "manifest"}
        self.effective = {"config_sha256": "residual-budget-config"}
        self.outcome = {"kind": "natural_terminal", "arm": self.arm, "query_id": self.qid,
                        "manifest_sha256": "manifest", "terminal_sha256": m.file_sha(self.row_path), "final_sha256": "final"}
        self.save_outcome()
        self.suite = SimpleNamespace(directory=m.NAMESPACE, reports=self.reports,
            verify_manifest=Mock(return_value=self.manifest), terminal_path=Mock(return_value=self.row_path),
            effective_manifest=Mock(return_value=self.effective), validate_terminal=Mock(),
            verify_execution_artifacts=Mock(), validate_outcome=Mock(side_effect=lambda *args: self.outcome))
        self.factory = patch.object(recovery_suite, "RecoverySuite", return_value=self.suite)
        self.queries = patch.object(recovery_suite.runner.shared, "validate_queries", return_value=[{"query_id":293,"query":"synthetic"}])
        self.factory.start(); self.queries.start()
        self.addCleanup(self.factory.stop); self.addCleanup(self.queries.stop)

    def save_outcome(self):
        self.outcome.pop("outcome_sha256", None)
        self.outcome["outcome_sha256"] = m.digest(self.outcome)
        if self.outcome_path.exists(): self.outcome_path.unlink()
        m.write_new(self.outcome_path, self.outcome)

    def test_admission_uses_effective_config_and_validates_actual_artifacts_and_outcome(self):
        binding = m.audit_terminal(self.root, self.arm, self.qid)
        self.assertEqual(binding["effective_config_sha256"], "residual-budget-config")
        self.suite.validate_terminal.assert_called_once_with(self.row, self.effective, self.arm, {"query_id":293,"query":"synthetic"})
        self.suite.verify_execution_artifacts.assert_called_once()
        self.suite.validate_outcome.assert_called_once()

    def test_observed_final_or_infrastructure_exhaustion_cannot_authorize_cleanup(self):
        for kind in ("materialized_observed_final", "infra_budget_exhausted"):
            self.outcome["kind"] = kind
            self.save_outcome()
            with self.assertRaisesRegex(RuntimeError, "Only a verified natural terminal"):
                m.audit_terminal(self.root, self.arm, self.qid)
        self.suite.validate_terminal.assert_not_called()

    def test_historical_terminal_path_cannot_authorize_cleanup(self):
        self.suite.terminal_path.return_value = self.root / "old" / "qid_293.json"
        with self.assertRaisesRegex(RuntimeError, "Historical/reused"):
            m.audit_terminal(self.root, self.arm, self.qid)

    def test_actual_artifact_validator_failure_blocks_admission(self):
        self.suite.verify_execution_artifacts.side_effect = RuntimeError("Actual artifact failed")
        with self.assertRaisesRegex(RuntimeError, "Actual artifact failed"):
            m.audit_terminal(self.root, self.arm, self.qid)


if __name__ == "__main__":
    unittest.main()
