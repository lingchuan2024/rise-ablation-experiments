"""Recovery budget, no-resampling, accounting and ordered C4 admission tests."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import copy
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from operations import recovery_suite as mod


def base_config():
    return {"max_turns": 100, "wall_clock_seconds": 3600, "concurrency": 4,
            "model": "Qwen3.5-35B-A3B", "per_call_max_tokens": 32768, "fixed_other": "unchanged"}


def item(qid=293, strategy="restart_remaining_budget"):
    base = base_config(); effective = dict(base)
    debit = {"recorded_attempts": 22, "possible_inflight_calls": 1, "calls_charged": 23,
        "elapsed_seconds_charged": 1713, "known_input_tokens": 574007, "known_output_tokens": 5740,
        "unknown_usage_attempts": 1, "time_bound_method": "pre-dispatch logged time to independently observed dead worker"}
    if strategy == "restart_remaining_budget": effective.update(max_turns=77, wall_clock_seconds=1887)
    if strategy == "infra_budget_exhausted": debit["elapsed_seconds_charged"] = 3600
    return {"query_id": qid, "strategy": strategy, "attempt_id": 2 if strategy == "restart_remaining_budget" else 1,
        "effective_config": effective, "effective_config_sha256": mod.runner.digest(effective), "debit": debit,
        "evidence_files": ["evidence.json"], "observed_final_file": "observed.json"}


class FakeSuite(mod.RecoverySuite):
    def __init__(self, root, storage=None):
        super().__init__(root)
        self.started = []; self.finished = []; self.active_count = 0; self.peak = 0
        self.mutex = threading.Lock(); self.states = []; self.capacity_calls = 0
        self.storage = storage
        self.reports.mkdir(parents=True)
        old = self.root / "old"
        for name in (mod.original.DIRECTORY, "strict_auto20_v1"):
            directory = old / "reports" / name; directory.mkdir(parents=True, exist_ok=True)
            (directory / "worker.lock").write_bytes(b"old immutable")
        self.manifest = {"manifest_sha256": "test", "config": base_config(),
            "recovery_plan": {"source_root": str(old), "recovery_questions": [item(310, "materialized_observed_final")]},
            "reused_auto20_reference": {"root": str(old)}}
        mod.runner.atomic_json(self.reports / "run_manifest.json", self.manifest)
    def verify_manifest(self, *, resources=True): return self.manifest
    def _stage_ready(self, manifest): pass
    def _state(self, identity, phase, **extra): self.states.append((phase, extra))
    def storage_status(self):
        self.capacity_calls += 1
        return {"ready": self.storage(self.capacity_calls) if self.storage else True}
    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        with self.mutex:
            if arm == "search_paging":
                expected = {q["query_id"] for q in mod.plan_stages(manifest["recovery_plan"])[0]["queries"]}
                actual = {qid for a, qid in self.finished if a == "auto_rerank"}
                if expected != actual: raise AssertionError("paging before all automatic dispositions")
            self.started.append((arm, query["query_id"]))
            self.active_count += 1; self.peak = max(self.peak, self.active_count)
        time.sleep(0.002)
        with self.mutex:
            self.finished.append((arm, query["query_id"]))
            self.active_count -= 1
        return self.validate_outcome(manifest, arm, query)
    def validate_outcome(self, manifest, arm, query):
        qid = query["query_id"]
        return {"query_id": qid, "kind": "materialized_observed_final" if qid == 310 and arm == "auto_rerank" else "natural_terminal", "outcome_sha256": str(qid)}
    def status(self):
        kinds = {"natural_terminal": 96, "materialized_observed_final": 1, "infra_budget_exhausted": 0}
        return {"target_coverage": 3 + len(self.finished), "new_outcome_counts": kinds, "validated_natural_terminals": 99, "target_count": 100}


class RecoveryTests(unittest.TestCase):
    def test_only_two_reduced_config_fields_are_permitted(self):
        candidate = item()
        self.assertEqual(mod.validate_budget(candidate, base_config())["max_turns"], 77)
        for key, value in (("per_call_max_tokens", 16000), ("model", "other"), ("max_turns", 100), ("wall_clock_seconds", 3600)):
            altered = copy.deepcopy(candidate); altered["effective_config"][key] = value
            altered["effective_config_sha256"] = mod.runner.digest(altered["effective_config"])
            with self.subTest(key=key), self.assertRaises(ValueError): mod.validate_budget(altered, base_config())

    def test_no_fresh_budget_when_debit_exhausted(self):
        candidate = item(); candidate["debit"]["elapsed_seconds_charged"] = 3600
        candidate["effective_config"]["wall_clock_seconds"] = 0
        candidate["effective_config_sha256"] = mod.runner.digest(candidate["effective_config"])
        with self.assertRaisesRegex(ValueError, "positive remaining"): mod.validate_budget(candidate, base_config())
        exhausted = item(strategy="infra_budget_exhausted")
        self.assertEqual(mod.validate_budget(exhausted, base_config()), base_config())

    def test_310_cannot_be_restarted_even_with_available_budget(self):
        with self.assertRaisesRegex(ValueError, "310"): mod.validate_budget(item(310), base_config())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "310"): mod.RecoverySuite(d).terminal_path("auto_rerank", 310)

    def test_unproven_checkpoint_cannot_bypass_restart_debit(self):
        candidate = item(strategy="resume_checkpoint")
        with self.assertRaisesRegex(ValueError, "exact verified"): mod.validate_budget(candidate, base_config())

    def test_observed_final_materializes_without_solver_or_fabricated_cp(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d); q = next(q for q in mod.runner.shared.validate_queries() if q["query_id"] == 310)
            candidate = item(310, "materialized_observed_final")
            manifest = {"manifest_sha256": "m", "administrative_source_sha256": mod.admin_hashes(),
                "recovery_plan": {"source_root": d, "recovery_questions": [candidate]}}
            observed = {"final_sha256": "public-sha", "verified": True}
            with patch.object(mod, "validate_observed_final", return_value=observed), patch.object(mod.runner.Suite, "run_one") as solver:
                result = suite.run_one(manifest, "auto_rerank", q)
                solver.assert_not_called()
            self.assertEqual(result["kind"], "materialized_observed_final")
            self.assertEqual(result["new_model_requests"], 0)
            self.assertFalse(result["complete_trace_checkpoint"])
            self.assertFalse(suite.runs.exists())

    def test_budget_exhaustion_is_not_a_natural_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d); q = next(q for q in mod.runner.shared.validate_queries() if q["query_id"] == 293)
            manifest = {"manifest_sha256": "m", "administrative_source_sha256": mod.admin_hashes(),
                "recovery_plan": {"recovery_questions": [item(strategy="infra_budget_exhausted")]}}
            with patch.object(mod.runner.Suite, "run_one") as solver:
                result = suite.run_one(manifest, "auto_rerank", q); solver.assert_not_called()
            self.assertEqual(result["kind"], "infra_budget_exhausted")
            self.assertIsNone(result["final_text"]); self.assertFalse(result["natural_terminal"])

    def test_effective_config_has_separate_identity_and_parent_manifest_binding(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d)
            manifest = {"manifest_sha256": "parent", "config": base_config(), "config_sha256": "base",
                "recovery_plan": {"recovery_questions": [item()]}}
            effective = suite.effective_manifest(manifest, "auto_rerank", 293)
            self.assertEqual(effective["manifest_sha256"], "parent")
            self.assertEqual(effective["config"]["max_turns"], 77)
            self.assertNotEqual(effective["config_sha256"], "base")
            self.assertEqual(manifest["config"]["max_turns"], 100)
            self.assertEqual(suite.effective_manifest(manifest, "search_paging", 70)["config"], base_config())

    def test_known_usage_accumulates_unknown_not_zeroed(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d)
            row = {"input_tokens": 100, "output_tokens": 10, "api_ledger": {"attempts": 2, "attempts_without_usage": 0}}
            result = suite.accounting(row, item())
            self.assertEqual(result["known_total_input_tokens"], 574107)
            self.assertEqual(result["known_total_output_tokens"], 5750)
            self.assertEqual(result["unknown_usage_attempts"], 1)
            self.assertIsNone(result["total_token_cap"])

    def test_exact_resume_accounting_does_not_double_count_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d)
            row = {"input_tokens": 600000, "output_tokens": 6000, "api_ledger": {"attempts": 30, "attempts_without_usage": 0}}
            result = suite.accounting(row, item(strategy="resume_checkpoint"))
            self.assertEqual(result["known_total_input_tokens"], 600000)
            self.assertTrue(result["checkpoint_totals_already_include_prior"])

    def test_c4_stage_order_and_99_plus_one_coverage(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d); result = suite.run()
            self.assertEqual(len(suite.started), 97)
            self.assertEqual(len(set(suite.started)), 97)
            self.assertFalse(any(a == "auto_rerank" and q in mod.REUSED_QIDS for a, q in suite.started))
            self.assertEqual(suite.peak, 4)
            self.assertEqual(result["validated_natural_terminals"], 99)
            self.assertEqual(result["new_outcome_counts"]["materialized_observed_final"], 1)
            self.assertEqual(result["target_coverage"], 100)

    def test_low_storage_waits_and_then_admits_without_dropping_jobs(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            suite = FakeSuite(d, storage=lambda count: count > 2)
            with patch.object(mod.time, "sleep") as sleeping:
                result = suite.run()
            self.assertTrue(any(phase == "waiting_storage" for phase, _ in suite.states))
            self.assertTrue(any(c.args == (30,) for c in sleeping.call_args_list))
            self.assertEqual(result["target_coverage"], 100)
            self.assertEqual(len(suite.started), 97)

    def test_observed_outcome_cannot_be_selected_twice_with_different_final(self):
        with tempfile.TemporaryDirectory() as d:
            suite = mod.RecoverySuite(d)
            payload = {"arm": "auto_rerank", "query_id": 310, "kind": "materialized_observed_final", "final_sha256": "first"}
            suite._save_outcome(payload)
            with self.assertRaisesRegex(RuntimeError, "Immutable"):
                suite._save_outcome({**payload, "final_sha256": "second"})


    def test_worker_identity_pins_actual_recovery_entrypoint(self):
        identity = mod.process_identity()
        self.assertEqual(identity["entrypoint"], "operations.recovery_suite")
        self.assertEqual(identity["driver_sha256"], mod.runner.file_sha(Path(mod.__file__)))

    def test_observed_final_receipt_checksum_and_no_resampling_attestation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evidence = root / "evidence.json"; evidence.write_text('{}')
            candidate = item(310, "materialized_observed_final")
            candidate["debit"].update(recorded_attempts=15, possible_inflight_calls=0,
                calls_charged=15, unknown_usage_attempts=0)
            sources = {"evidence.json": mod.original._pin(evidence)}
            final = "Public answer"
            receipt = {"query_id": 310, "arm": "auto_rerank", "source_manifest_sha256": mod.SOURCE_MANIFEST_SHA,
                "verified": True, "finish_reason": "stop", "tool_calls_count": 0, "category": "content",
                "final_text": final, "final_sha256": mod.hashlib.sha256(final.encode()).hexdigest(),
                "response_id": "observed-id", "source_files": sources,
                "known_input_tokens": candidate["debit"]["known_input_tokens"],
                "known_output_tokens": candidate["debit"]["known_output_tokens"],
                "full_natural_terminal_reconstructed": False, "private_message_or_reasoning_values_decoded": False,
                "original_files_modified": False, "new_model_requests": 0, "new_tool_executions": 0,
                "recorded_attempts": 15, "unknown_usage_attempts": 0}
            receipt["receipt_sha256"] = mod.runner.digest(receipt)
            target = root / "observed.json"; target.write_text(json.dumps(receipt))
            plan = {"source_pins": {**sources, "observed.json": mod.original._pin(target)}}
            self.assertEqual(mod.validate_observed_final(root, candidate, plan)["response_id"], "observed-id")
            receipt["receipt_sha256"] = "wrong"; target.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "checksum"): mod.validate_observed_final(root, candidate, plan)
            receipt["new_model_requests"] = 1
            receipt["receipt_sha256"] = mod.runner.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            target.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "no-resampling"): mod.validate_observed_final(root, candidate, plan)



if __name__ == "__main__": unittest.main()
