"""Offline remaining80, C4, no-resampling, lifecycle and storage admission tests."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import fcntl
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from operations import no_bash80_suite as mod


class FakeSuite(mod.NoBash80Suite):
    """Real scheduler/dispatch files/locks; no model or corpus access."""
    def __init__(self, root, storage=None, fail_qid=None):
        super().__init__(root)
        self.reports.mkdir(parents=True)
        self.legacy_auto20_root = self.root / "separate_legacy_auto20_root"
        old = self.legacy_auto20_root / "reports/strict_auto20_v1"
        old.mkdir(parents=True); (old / "worker.lock").write_bytes(b"preserve old lock bytes")
        self.manifest = {"manifest_sha256": "manifest", "config_sha256": "config",
            "config": {"concurrency": 4, "runtime_environment": {}},
            "contracts": {mod.ARM: {"contract_sha256": "contract"}},
            "administrative_source_sha256": mod.admin_hashes(),
            "selected_query_ids": [q["query_id"] for q in mod.selected_queries()]}
        mod.runner.atomic_json(self.reports / "run_manifest.json", self.manifest)
        self.started = []; self.finished = []; self.states = []; self.capacity_calls = 0
        self.current = 0; self.peak = 0; self.mutex = threading.Lock()
        self.storage = storage; self.fail_qid = fail_qid

    def verify_parent(self):
        return {"parent_namespace": "strict_no_bash20_v1", "reused_no_bash_questions": 20,
            "parent_manifest_sha256": "fixture", "parent_file_pins_sha256": "fixture",
            "config_sha256": "config", "files_verified": 143, "parent_model_requests": 0}

    def verify_manifest(self, *, resources=True):
        self.assert_run_scope()
        return self.manifest

    def _stage_ready(self, manifest):
        pass

    def _state(self, identity, phase, **extra):
        self.states.append((phase, extra))
        super()._state(identity, phase, **extra)

    def storage_status(self):
        self.capacity_calls += 1
        return {"ready": self.storage(self.capacity_calls) if self.storage else True}

    def validate_terminal(self, row, manifest, arm, query):
        if row.get("query_id") != query["query_id"]:
            raise RuntimeError("fixture identity changed")
        return row

    def verify_execution_artifacts(self, manifest, arm, query, terminal):
        return {"verified": True}

    def save_terminal(self, qid):
        if not self.dispatch_path(qid).exists():
            query = next(q for q in mod.selected_queries() if q["query_id"] == qid)
            self._dispatch(self.manifest, query, getattr(self, "_worker_identity", mod.process_identity()))
        mod.runner.atomic_json(self.terminal_path(mod.ARM, qid), {
            "query_id": qid, "row_sha256": str(qid), "execution_artifacts": {"trace": "opaque"}}, immutable=True)

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        self.assertEqualArm(arm)
        qid = query["query_id"]
        dispatch = mod.runner.load(self.dispatch_path(qid))
        assert dispatch["worker_identity"] == self._worker_identity
        with self.mutex:
            self.started.append(qid); self.current += 1; self.peak = max(self.peak, self.current)
        # A failing question returns first so dispatch must stop while peers drain.
        threading.Event().wait(0.005 if qid == self.fail_qid else 0.02)
        with self.mutex:
            self.current -= 1
            if qid == self.fail_qid:
                raise RuntimeError("simulated infrastructure stop")
            self.save_terminal(qid); self.finished.append(qid)
        return {"query_id": qid}

    @staticmethod
    def assertEqualArm(arm):
        if arm != mod.ARM:
            raise AssertionError("unexpected arm")


class NoBash80Tests(unittest.TestCase):
    def test_fixed_remaining80_and_original_contract(self):
        rows = mod.selected_queries()
        self.assertEqual(rows, mod.runner.shared.validate_queries()[20:])
        self.assertEqual(len(rows), 80)
        self.assertEqual(mod.NoBash80Suite.probe_arms, ("no_bash",))
        contract = mod.runner.build_contract(mod.ARM)
        self.assertEqual(contract["contract_sha256"], "e9ed046be4d2203d4a248caecc9f2f95060612ca49deb320fee05342d2ab2cff")
        self.assertEqual({t["function"]["name"] for t in contract["tool_specs"]}, {"search", "read", "rerank"})

    def test_original135_sources_are_frozen(self):
        self.assertEqual(len(mod.verify_frozen_sources()), 135)
        with patch.object(mod.runner, "source_hashes", return_value={"changed.py": "x"}):
            with self.assertRaisesRegex(RuntimeError, "original135"):
                mod.verify_frozen_sources()

    def test_dispatch_rejects_recomputed_checksum_with_changed_budget_or_query(self):
        with tempfile.TemporaryDirectory() as root:
            suite = FakeSuite(root); query = mod.selected_queries()[0]
            suite._dispatch(suite.manifest, query, mod.process_identity())
            path = suite.dispatch_path(query["query_id"]); original = mod.runner.load(path)
            for key, value in (("attempt_id", 2), ("config_sha256", "changed"), ("query_id", 999999), ("contract_sha256", "other")):
                changed = {**original, key: value}
                changed["dispatch_sha256"] = mod.runner.digest({k: v for k, v in changed.items() if k != "dispatch_sha256"})
                mod.runner.atomic_json(path, changed)
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "dispatch identity"):
                    suite.validate_dispatch(suite.manifest, query)

    def test_outside_arm_and_cohort_rejected(self):
        suite = mod.NoBash80Suite("/tmp/not-created-no-bash-test")
        for arm, qid in (("auto_rerank", mod.selected_queries()[0]["query_id"]),
                         ("no_bash", mod.runner.shared.validate_queries()[0]["query_id"]),
                         ("no_bash", True)):
            with self.subTest(arm=arm, qid=qid), self.assertRaises(ValueError):
                suite.terminal_path(arm, qid)

    def test_c4_fixed_order_completion_and_immutable_dispatch(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            suite = FakeSuite(root); result = suite.run80()
            qids = [q["query_id"] for q in mod.selected_queries()]
            self.assertEqual(set(suite.started), set(qids))
            self.assertEqual(len(suite.started), 80); self.assertEqual(suite.peak, 4)
            self.assertEqual(result["validated_natural_terminals"], 80)
            self.assertEqual([r["query_id"] for r in result["terminal_rows"]], qids)
            self.assertEqual(result["other_arm_model_requests"], 0)
            for query in mod.selected_queries():
                rec = mod.runner.load(suite.dispatch_path(query["query_id"]))
                self.assertEqual(rec["attempt_id"], 1)
                self.assertEqual(rec["manifest_sha256"], "manifest")
                self.assertEqual(rec["dispatch_sha256"], mod.runner.digest({k: v for k, v in rec.items() if k != "dispatch_sha256"}))
            phase = mod.runner.load(suite.reports / "phase_state.json")
            self.assertEqual(phase["phase"], "complete")
            self.assertEqual(phase["receipt_sha256"], mod.runner.file_sha(suite.reports / "no_bash80_completion.json"))

    def test_rerun_validates_and_reuses_all_natural_terminals_without_requests(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            suite = FakeSuite(root); first = suite.run80()
            pins = {p.name: mod.runner.file_sha(p) for p in (suite.reports / "dispatches").iterdir()}
            suite.started.clear(); suite.finished.clear()
            second = suite.run80()
            self.assertEqual(first, second); self.assertFalse(suite.started)
            self.assertEqual(pins, {p.name: mod.runner.file_sha(p) for p in (suite.reports / "dispatches").iterdir()})

    def test_existing_terminal_not_dispatched_even_below_storage_gate(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            suite = FakeSuite(root, storage=lambda _: False)
            for query in mod.selected_queries(): suite.save_terminal(query["query_id"])
            self.assertTrue(suite.run80()["passed"])
            self.assertEqual(suite.started, [])

    def test_unknown_partial_and_working_only_and_dispatch_only_all_refuse_restart(self):
        for source in ("working", "ledger", "checkpoint", "dispatch", "atomic_temporary"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
                suite = FakeSuite(root); qid = mod.selected_queries()[0]["query_id"]
                paths = {"working": suite.runs / mod.ARM / "_working" / f"qid_{qid}",
                    "ledger": suite.runs / mod.ARM / "api_ledgers" / f"qid_{qid}.jsonl",
                    "checkpoint": suite.runs / mod.ARM / "_checkpoints" / f"qid_{qid}.json",
                    "dispatch": suite.dispatch_path(qid),
                    "atomic_temporary": suite.runs / mod.ARM / "_per_query" / f"qid_{qid}.json.abcdefgh"}
                path = paths[source]; path.parent.mkdir(parents=True, exist_ok=True)
                if source == "working": path.mkdir()
                else: path.write_text("{}")
                status = suite.status()["arms"][mod.ARM]
                self.assertIn(qid, status["partial_or_active"])
                self.assertNotIn("interrupted", status)
                with self.assertRaisesRegex(RuntimeError, "prior dispatch/partial"):
                    suite.run80()
                self.assertFalse(suite.started)

    def test_no_automatic_resume_entrypoints(self):
        with tempfile.TemporaryDirectory() as root:
            suite = mod.NoBash80Suite(root)
            with self.assertRaises(ValueError): suite.run(resume_infrastructure=True)
            with self.assertRaises(ValueError): suite.run(stage="full")
            with self.assertRaises(RuntimeError): suite.live_preflight(resume_infrastructure=True)
            with self.assertRaises(RuntimeError): suite.live_preflight(service_pid=123)

    def test_storage_blocks_new_dispatch_only_and_preserves_all_jobs(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            suite = FakeSuite(root, storage=lambda n: n > 2)
            with patch.object(mod.time, "sleep", wraps=lambda seconds: None) as sleeping:
                result = suite.run80()
            self.assertTrue(result["passed"])
            self.assertEqual(len(suite.started), 80)
            self.assertTrue(any(p == "waiting_storage" for p, _ in suite.states))
            sleeping.assert_called_with(30)

    def test_storage_drops_while_four_active_peers_finish_then_admission_resumes(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            suite = FakeSuite(root, storage=lambda n: not (5 <= n <= 7))
            with patch.object(mod.time, "sleep", wraps=lambda seconds: None): suite.run80()
            waits = [r for p, r in suite.states if p == "waiting_storage"]
            self.assertTrue(waits); self.assertEqual(len(waits[0]["active_query_ids"]), 4)
            self.assertEqual(len(suite.finished), 80); self.assertEqual(suite.peak, 4)

    def test_infrastructure_failure_stops_dispatch_and_drains_peers(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()):
            qid = mod.selected_queries()[0]["query_id"]
            suite = FakeSuite(root, fail_qid=qid)
            with self.assertRaisesRegex(RuntimeError, "active questions drained"):
                suite.run80()
            self.assertEqual(len(suite.started), 4); self.assertEqual(len(suite.finished), 3)
            self.assertEqual(suite.current, 0)
            self.assertTrue(any(p == "draining_after_error" for p, _ in suite.states))
            self.assertFalse((suite.reports / "no_bash80_completion.json").exists())
            self.assertEqual(mod.runner.load(suite.reports / "phase_state.json")["phase"], "interrupted")
            with self.assertRaisesRegex(RuntimeError, "prior dispatch/partial"):
                suite.run80()

    def test_three_locks_exclude_legacy_and_new_workers_without_editing_old_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            suite = FakeSuite(root)
            old = suite.lock_paths()[1]; before = old.read_bytes()
            with suite.worker_locks():
                for path in suite.lock_paths():
                    with path.open("rb") as handle, self.assertRaises(BlockingIOError):
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(old.read_bytes(), before)
            with old.open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError), suite.worker_locks(): pass
            self.assertEqual(old.read_bytes(), before)

    def test_legacy_lock_uses_real_separate_root_and_does_not_create_wrong_local_lock(self):
        suite = mod.NoBash80Suite("/tmp/new-no-bash-namespace")
        self.assertEqual(suite.lock_paths()[1], Path("/root/data/rise-strict-auto20-20261023/reports/strict_auto20_v1/worker.lock"))
        with patch.object(suite, "verify_parent", return_value=FakeSuite.verify_parent(suite)):
            scope = suite.execution_scope(mod.runner.shared.validate_queries())
        self.assertEqual(scope["worker_lock_policy"]["legacy_auto20_lock"], str(suite.lock_paths()[1]))
        with tempfile.TemporaryDirectory() as root:
            fixture = FakeSuite(root)
            wrong = fixture.root / "reports/strict_auto20_v1/worker.lock"
            with fixture.worker_locks():
                self.assertFalse(wrong.exists())
                with fixture.lock_paths()[1].open("rb") as handle, self.assertRaises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_scope_rejects_other_arm_and_qid_and_symlink(self):
        for kind in ("arm", "qid", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                suite = FakeSuite(root)
                if kind == "arm": (suite.runs / "search_paging").mkdir(parents=True)
                else:
                    folder = suite.runs / mod.ARM / "_working"; folder.mkdir(parents=True)
                    if kind == "qid": (folder / "qid_999999").mkdir()
                    else: (folder / f"qid_{mod.selected_queries()[0]['query_id']}").symlink_to(root)
                with self.assertRaises(RuntimeError): suite.assert_run_scope()

    def test_storage_threshold_boundaries(self):
        suite = mod.NoBash80Suite("/tmp/no-storage-created")
        for inodes, size, expected in ((40000, 10 * 1024 ** 3, True),
                (39999, 10 * 1024 ** 3, False), (40000, 10 * 1024 ** 3 - 1, False)):
            with patch.object(mod.os, "statvfs", return_value=SimpleNamespace(f_bavail=size, f_frsize=1, f_favail=inodes)):
                self.assertEqual(suite.storage_status()["ready"], expected)

    def test_active_status_needs_live_identity_and_preserves_partial_union(self):
        with tempfile.TemporaryDirectory() as root:
            suite = FakeSuite(root); qid = mod.selected_queries()[0]["query_id"]
            path = suite.dispatch_path(qid); path.parent.mkdir(parents=True); path.write_text("{}")
            mod.runner.atomic_json(suite.reports / "phase_state.json", {"worker_identity": {"pid": 123}, "active_query_ids": [qid], "phase": "running_no_bash80"})
            for live in (True, False):
                with patch.object(mod, "process_matches", return_value=live): result = suite.status()
                self.assertEqual(result["arms"][mod.ARM]["partial_or_active"], [qid])
                self.assertEqual(result["arms"][mod.ARM]["active"], [qid] if live else [])

    def test_live_gate_rejects_other_surface_or_service_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            suite = FakeSuite(root)
            ready = {"passed": True, "manifest_sha256": "manifest", "service": {"epoch_sha256": "epoch"},
                "tools_and_native_calls": {"no_bash": {"passed": True}}}
            path = suite.reports / "preflight_live.json"
            with patch.object(suite, "verify_service", return_value={"epoch_sha256": "epoch"}):
                mod.runner.atomic_json(path, ready)
                mod.NoBash80Suite._stage_ready(suite, suite.manifest)
                ready["tools_and_native_calls"] = {"auto_rerank": {"passed": True}}
                mod.runner.atomic_json(path, ready)
                with self.assertRaises(RuntimeError): mod.NoBash80Suite._stage_ready(suite, suite.manifest)
                ready["tools_and_native_calls"] = {"no_bash": {"passed": True}}
                ready["service"]["epoch_sha256"] = "old"
                mod.runner.atomic_json(path, ready)
                with self.assertRaisesRegex(RuntimeError, "epoch"): mod.NoBash80Suite._stage_ready(suite, suite.manifest)


    def test_remaining80_and_reused20_partition_exact_fixed100(self):
        first = {q["query_id"] for q in mod.runner.shared.validate_queries()[:20]}
        rest = {q["query_id"] for q in mod.selected_queries()}
        self.assertFalse(first & rest)
        self.assertEqual(first | rest, {q["query_id"] for q in mod.runner.shared.validate_queries()})
        self.assertEqual(len(rest), 80)

    def test_parent20_actual_pinned_snapshot_hashes_without_model_or_checkpoint_decoding(self):
        source = mod.ROOT / "reports/strict_no_bash20_v1/downloaded/complete_20261024_1832/extracted"
        result = mod.verify_parent20(source)
        self.assertEqual(result["files_verified"], 143)
        self.assertEqual(result["reused_no_bash_questions"], 20)
        self.assertEqual(result["parent_model_requests"], 0)
        self.assertEqual(result["config_sha256"], "4066e6ae83c5427924a3eb3577eb5c5621a4372b5be673852667b64fe8433c1a")

    def test_parent_changed_hash_is_rejected_without_editing_originals(self):
        source = mod.ROOT / "reports/strict_no_bash20_v1/downloaded/complete_20261024_1832/extracted"
        with patch.object(mod.runner, "file_sha", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "first20 artifact changed"):
                mod.verify_parent20(source)

    def test_prior_execution_in_other_namespace_rejected_even_before_new_runs_created(self):
        with tempfile.TemporaryDirectory() as root:
            suite = mod.NoBash80Suite(root)
            qid = mod.selected_queries()[0]["query_id"]
            path = Path(root) / "runs/strict_ablations_v1/no_bash/_working" / f"qid_{qid}"
            path.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "prior execution evidence"):
                suite.assert_run_scope()

    def test_same_first20_config_required_before_offline_preflight_or_any_requests(self):
        suite = mod.NoBash80Suite("/tmp/no-created-remaining80-test")
        with patch.object(suite, "verify_parent", return_value={"config_sha256":"original"}):
            with self.assertRaisesRegex(RuntimeError, "exactly the frozen first20"):
                suite.offline_preflight({"common_config":{"concurrency":4}})


if __name__ == "__main__": unittest.main()
