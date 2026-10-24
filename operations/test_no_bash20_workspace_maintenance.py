"""Safety gates and real temporary-filesystem lifecycle; synthetic opaque data only."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operations import no_bash20_workspace_maintenance as m
from operations import terminal_workspace_maintenance as old


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "remote"; self.root.mkdir()
        self.c = m.core()
        self.arm, self.qid = "no_bash", 70
        self.work, self.runtime = self.c.targets(self.root, self.arm, self.qid).values()
        self.work.mkdir(parents=True); self.runtime.mkdir(parents=True)
        self.corpus = self.root / "shared_source.bin"; self.corpus.write_bytes(b"opaque corpus")
        os.link(self.corpus, self.work / "a.bin")
        (self.work / "b.bin").write_bytes(b"opaque independent")
        (self.runtime / "c.bin").write_bytes(b"opaque runtime")
        self.other = self.work.parent / "qid_81"; self.other.mkdir()
        (self.other / "active.bin").write_bytes(b"active must remain")
        self.compact = self.work.parent.parent / "_per_query/qid_70.json"
        self.compact.parent.mkdir(); self.compact.write_bytes(b"opaque immutable terminal")
        self.binding = {"source_root": str(self.root), "namespace": m.NAMESPACE, "arm": self.arm,
                        "query_id": self.qid, "manifest_sha256": m.MANIFEST_SHA256,
                        "effective_config_sha256": m.CONFIG_SHA256, "worker_identity": {"pid": 123},
                        "terminal_sha256": self.c.file_sha(self.compact)}
        self.audit = patch.object(m, "audit_terminal", side_effect=self.audit_fixture)
        self.audit.start(); self.addCleanup(self.audit.stop)
        self.gate = patch.object(m, "live_gate")
        self.gate_mock = self.gate.start(); self.addCleanup(self.gate.stop)
        admit = patch.object(m, "admission_gate", side_effect=lambda root, binding: m.live_gate(root, binding))
        admit.start(); self.addCleanup(admit.stop)

    def audit_fixture(self, root, arm, qid):
        if (Path(root), arm, qid) != (self.root, self.arm, self.qid):
            raise RuntimeError("Wrong question")
        if self.c.file_sha(self.compact) != self.binding["terminal_sha256"]:
            raise RuntimeError("Terminal changed")
        return self.binding

    def export(self):
        self.output = self.root / "reports" / m.NAMESPACE / "workspace_backups" / "fresh"
        self.result = m.export_workspace(self.root, self.arm, self.qid, self.output)
        self.local = self.base / "local"; self.local.mkdir()
        for name in ("workspace.tar.gz", "receipt.json"):
            shutil.copyfile(self.output / name, self.local / name)
        self.ack = m.verify_archive(self.local / "workspace.tar.gz", self.local / "receipt.json",
            expected_receipt_sha256=self.result["receipt_file_sha256"], expected_receipt_bytes=self.result["receipt_bytes"])

    def delete(self, ack=None):
        return m.delete_workspace(self.root, self.arm, self.qid, self.output / "receipt.json", self.ack if ack is None else ack)

    def test_complete_lifecycle_preserves_other_question_shared_source_and_compact(self):
        self.export(); result = self.delete()
        self.assertTrue(result["deleted"])
        self.assertEqual((result["files_removed"], result["directories_removed"]), (3, 2))
        self.assertFalse(self.work.exists()); self.assertFalse(self.runtime.exists())
        self.assertEqual(self.corpus.read_bytes(), b"opaque corpus")
        self.assertEqual((self.other / "active.bin").read_bytes(), b"active must remain")
        self.assertEqual(self.c.file_sha(self.compact), self.binding["terminal_sha256"])
        shutil.rmtree(self.root)
        self.assertTrue(m.verify_archive(self.local / "workspace.tar.gz", self.local / "receipt.json")["verified"])

    def test_legacy_module_namespace_and_global_os_are_not_mutated(self):
        self.export(); self.delete()
        self.assertEqual(old.NAMESPACE, "strict_recovery_v1")
        self.assertEqual(old.VERSION, "terminal-workspace-maintenance-v1")
        self.assertIs(old.os, os)

    def test_gate_closes_before_first_unlink_preserves_all_files(self):
        self.export()
        self.gate_mock.side_effect = [None, RuntimeError("tail batch")]
        with self.assertRaisesRegex(RuntimeError, "tail batch"): self.delete()
        self.assertTrue((self.work / "a.bin").exists()); self.assertTrue((self.work / "b.bin").exists())
        self.assertFalse((self.output / "deletion_receipt.json").exists())

    def test_gate_closes_mid_delete_records_partial_never_success(self):
        self.export()
        self.gate_mock.side_effect = [None, None, RuntimeError("tail batch")]
        with self.assertRaisesRegex(RuntimeError, "tail batch"): self.delete()
        records = list(self.output.glob("maintenance_interrupted_*.json"))
        self.assertEqual(len(records), 1)
        record = self.c.load(records[0])
        self.assertFalse(record["deleted"]); self.assertTrue(record["partial_maintenance"])
        self.assertEqual(record["files_removed"], 1)
        self.assertFalse((self.output / "deletion_receipt.json").exists())
        self.assertTrue((self.work / "b.bin").exists()); self.assertTrue(self.corpus.exists())
        self.assertTrue(m.verify_archive(self.local / "workspace.tar.gz", self.local / "receipt.json")["verified"])

    def test_old_same_qid_ack_is_rejected(self):
        self.export(); ack = copy.deepcopy(self.ack)
        ack["version"] = old.VERSION
        ack["binding"]["namespace"] = "strict_recovery_v1"
        ack["acknowledgement_sha256"] = self.c.digest({k:v for k,v in ack.items() if k != "acknowledgement_sha256"})
        with self.assertRaisesRegex(RuntimeError, "Other maintenance versions"): self.delete(ack)
        self.assertTrue((self.work / "a.bin").exists())

    def test_keyboard_interrupt_after_unlink_records_partial(self):
        self.export()
        self.gate_mock.side_effect = [None, None, KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt): self.delete()
        records = list(self.output.glob("maintenance_interrupted_*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(self.c.load(records[0])["error_type"], "KeyboardInterrupt")
        self.assertFalse((self.output / "deletion_receipt.json").exists())

    def test_drift_before_unlink_refuses_delete(self):
        self.export(); p = self.work / "b.bin"; saved = p.stat()
        p.write_bytes(b"x" * saved.st_size); os.utime(p, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, "content no longer matches"): self.delete()
        self.assertTrue((self.work / "a.bin").exists())

    def test_replacement_tree_refuses_delete(self):
        self.export(); (self.work / "extra.bin").write_bytes(b"late")
        with self.assertRaisesRegex(RuntimeError, "tree/stat differs"): self.delete()
        self.assertTrue((self.work / "a.bin").exists())

    def test_symlink_source_cannot_export(self):
        (self.work / "escape").symlink_to(self.corpus)
        with self.assertRaisesRegex(RuntimeError, "Symlink|escapes its root"): self.export()

    def test_wrong_arm_and_qid_fail_before_creating_output(self):
        for arm, qid in (("auto_rerank",70), ("no_bash",293), ("no_bash", True), ("../../escape",70)):
            with self.assertRaisesRegex(RuntimeError, "fixed-first20"):
                m.export_workspace(self.root, arm, qid, self.root / "forbidden")
        self.assertFalse((self.root / "forbidden").exists())


class LiveGateTests(unittest.TestCase):
    def setUp(self):
        from operations import no_bash20_suite as ns
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.reports = self.root / "reports" / m.NAMESPACE
        self.reports.mkdir(parents=True)
        self.binding = {"manifest_sha256": m.MANIFEST_SHA256, "worker_identity": {"pid":123}, "query_id":70}
        self.state = {"phase":"running_no_bash20", "manifest_sha256":m.MANIFEST_SHA256,
                      "worker_identity":{"pid":123}, "checked_unix":time.time(),
                      "active_query_ids":[81,96,116,122], "pending_query_ids":[127], "errors":[]}
        self.suite = SimpleNamespace(reports=self.reports, evidence_present=Mock(return_value=False))
        for p in (patch.object(m,"suite_for",return_value=self.suite), patch.object(ns,"process_matches",return_value=True)):
            p.start(); self.addCleanup(p.stop)

    def gate(self):
        (self.reports / "phase_state.json").write_text(json.dumps(self.state))
        m.live_gate(self.root,self.binding)

    def test_running_and_waiting_storage_with_real_pending_are_allowed(self):
        self.gate(); self.state["phase"]="waiting_storage"; self.gate()

    def test_pending_zero_or_stale_dispatch_state_are_rejected(self):
        self.state["pending_query_ids"]=[]
        with self.assertRaisesRegex(RuntimeError,"No undispatched"): self.gate()
        self.state["pending_query_ids"]=[127]; self.suite.evidence_present.return_value=True
        with self.assertRaisesRegex(RuntimeError,"Pending state is stale"): self.gate()

    def test_active_target_final_phase_completion_and_old_state_are_rejected(self):
        initial=copy.deepcopy(self.state)
        for modification in ({"active_query_ids":[70]}, {"phase":"complete"}, {"phase":"interrupted"},
                             {"checked_unix":time.time()-100}, {"errors":[{"kind":"failure"}]}):
            self.state={**initial,**modification}
            with self.assertRaises(RuntimeError): self.gate()
        self.state=initial
        (self.reports / "no_bash20_completion.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError,"completion barrier"): self.gate()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        from operations import no_bash20_suite as ns
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.c=m.core()
        self.path=self.root / "runs" / m.NAMESPACE / m.ARM / "_per_query/qid_70.json"
        self.row={"row_sha256":"row", "execution_artifacts":{"opaque":"hash"}}
        self.c.write_new(self.path,self.row)
        self.dispatch_path=self.root / "reports" / m.NAMESPACE / "dispatches/qid_70.json"
        self.dispatch={"dispatch_sha256":"dispatch","worker_identity":{"pid":123}}
        self.c.write_new(self.dispatch_path,self.dispatch)
        self.manifest={"manifest_sha256":m.MANIFEST_SHA256,"config_sha256":m.CONFIG_SHA256,
                       "contracts":{m.ARM:{"contract_sha256":"contract"}}}
        self.suite=SimpleNamespace(verify_manifest=Mock(return_value=self.manifest),terminal_path=Mock(return_value=self.path),
            dispatch_path=Mock(return_value=self.dispatch_path),validate_dispatch=Mock(return_value=self.dispatch),
            validate_terminal=Mock(),verify_execution_artifacts=Mock())
        for p in (patch.object(m,"suite_for",return_value=self.suite),
                  patch.object(ns,"selected_queries",return_value=[{"query_id":70,"query":"synthetic"}])):
            p.start();self.addCleanup(p.stop)

    def test_admission_validates_dispatch_terminal_and_actual_artifacts(self):
        b=m.audit_terminal(self.root,m.ARM,70)
        self.assertEqual(b["dispatch_file_sha256"],self.c.file_sha(self.dispatch_path))
        self.suite.verify_manifest.assert_called_once_with(resources=False)
        self.suite.validate_dispatch.assert_called_once();self.suite.validate_terminal.assert_called_once()
        self.suite.verify_execution_artifacts.assert_called_once()

    def test_missing_terminal_and_old_same_qid_terminal_are_rejected(self):
        self.path.unlink()
        with self.assertRaisesRegex(RuntimeError,"No natural terminal"): m.audit_terminal(self.root,m.ARM,70)
        self.suite.terminal_path.return_value=self.root / "runs/strict_recovery_v1/search_paging/_per_query/qid_70.json"
        with self.assertRaisesRegex(RuntimeError,"other namespace"): m.audit_terminal(self.root,m.ARM,70)

    def test_infrastructure_and_temp_publications_are_rejected(self):
        for sub,name in (("_infrastructure","qid_70.json"),("_checkpoints","qid_70.json.abcdefgh.tmp"),
                         ("request_contracts","qid_70.json.tmp")):
            p=self.path.parent.parent / sub / name;p.parent.mkdir(exist_ok=True);p.write_bytes(b"opaque")
            with self.assertRaisesRegex(RuntimeError,"evidence forbids"):m.audit_terminal(self.root,m.ARM,70)
            p.unlink()

    def test_any_validator_failure_denies_admission(self):
        for name in ("validate_dispatch","validate_terminal","verify_execution_artifacts"):
            getattr(self.suite,name).side_effect=RuntimeError("bad evidence")
            with self.assertRaisesRegex(RuntimeError,"bad evidence"):m.audit_terminal(self.root,m.ARM,70)
            getattr(self.suite,name).side_effect=None

    def test_manifest_config_mismatch_denies_admission(self):
        self.manifest["config_sha256"]="wrong"
        with self.assertRaisesRegex(RuntimeError,"manifest/configuration"):m.audit_terminal(self.root,m.ARM,70)


class FullStatusGateTests(unittest.TestCase):
    def setUp(self):
        self.binding={"query_id":70,"worker_identity":{"pid":123}}
        self.status={"prepared":True,"worker_identity_live":True,"worker_identity":{"pid":123},
                     "phase":"waiting_storage","arms":{"no_bash":{"terminal":[70],"partial_or_active":[81],
                                                                      "active":[81],"pending":[96]}}}
        self.suite=SimpleNamespace(status=Mock(side_effect=lambda: self.status))
        for p in (patch.object(m,"suite_for",return_value=self.suite), patch.object(m,"live_gate")):
            p.start();self.addCleanup(p.stop)

    def test_full_validated_status_admits_waiting_storage(self):
        m.admission_gate(Path("/synthetic"), self.binding)

    def test_terminal_active_contradiction_unknown_partial_and_no_pending_fail(self):
        base=copy.deepcopy(self.status)
        for patch_fields in ({"active":[70],"partial_or_active":[70]}, {"partial_or_active":[81,116]},
                             {"pending":[]}, {"terminal":[70,70]}, {"pending":[70]}):
            self.status=copy.deepcopy(base)
            self.status["arms"][m.ARM].update(patch_fields)
            with self.assertRaises(RuntimeError):m.admission_gate(Path("/synthetic"),self.binding)

    def test_nonlive_worker_never_admitted(self):
        self.status["worker_identity_live"]=False
        with self.assertRaisesRegex(RuntimeError,"Full status"):m.admission_gate(Path("/synthetic"),self.binding)

    def test_frozen_core_hash_required(self):
        with patch.object(m,"CORE_SHA256","0"*64):
            with self.assertRaisesRegex(RuntimeError,"algorithm changed"):m.core()


if __name__ == "__main__": unittest.main()
