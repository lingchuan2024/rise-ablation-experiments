"""Offline frozen-control verification; synthetic changes never touch results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from strict_ablations import control
from strict_experiment.strict_prompt import build_contract

ROOT = Path(__file__).resolve().parents[1]


class ControlBundleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cohort = [{"query_id": qid, "query": f"Question {qid}", "answer": f"Gold {qid}"}
                  for qid in range(100)]
        self.cohort_bytes = "".join(json.dumps(row) + "\n" for row in cohort).encode()
        path = self.root / control.COHORT_PATH
        path.parent.mkdir(parents=True)
        path.write_bytes(self.cohort_bytes)
        self.config = {"model": control.MODEL, "max_turns": 100, "concurrency": 4,
                       "runtime_environment": {"REASONING_EFFORT": "medium"}, "corpus": "/frozen/corpus"}
        self.contract = build_contract("B")
        self.manifest = {"config": self.config, "config_sha256": control._digest(self.config),
            "contracts": {"B": self.contract}, "cohort_sha256": control._sha(self.cohort_bytes),
            "query_ids": list(range(100))}
        self.manifest["manifest_sha256"] = control._digest(self.manifest)
        self.public = {"version": "strict-B100-public-finals-metrics-v1",
            "manifest_sha256": self.manifest["manifest_sha256"], "model": control.MODEL,
            "cohort_sha256": self.manifest["cohort_sha256"], "epoch_sha256": "e" * 64,
            "records": [self.record(row) for row in cohort]}
        self.receipt = {"version": "strict-B100-shutdown-integrity-v1", "arm": "B",
            "manifest_sha256": self.manifest["manifest_sha256"], "config_sha256": self.manifest["config_sha256"],
            "epoch_sha256": "e" * 64, "a_artifacts_unchanged": True,
            "a_before_sha256": "a" * 64, "a_after_sha256": "a" * 64,
            "items": [{"query_id": qid, "files": {p: "b" * 64 for p in control._expected_receipt_paths(qid)}}
                      for qid in range(100)]}
        self.metadata = {"manifest_sha256": self.manifest["manifest_sha256"], "model": control.MODEL,
            "cohort_sha256": self.manifest["cohort_sha256"], "epoch_sha256": "e" * 64,
            "original_config": copy.deepcopy(self.config), "B_contract": copy.deepcopy(self.contract)}
        self.completion = {"ready_for_shutdown": True, "phase": "B100_verified_public_export_saved",
            "B_terminal_files": 100, "a_artifacts_unchanged": True,
            "manifest_sha256": self.manifest["manifest_sha256"], "epoch_sha256": "e" * 64}
        self.pins = {}
        self.seal()
        patcher = mock.patch.multiple(control, **self.pins)
        patcher.start()
        self.addCleanup(patcher.stop)
        contract_patch = mock.patch.object(control, "_control_contract", return_value=copy.deepcopy(self.contract))
        contract_patch.start()
        self.addCleanup(contract_patch.stop)

    @staticmethod
    def record(query):
        final = "" if query["query_id"] == 0 else "A public final"
        metrics = {key: 0 for key in control._METRIC_KEYS}
        metrics.update(n_turns=2, elapsed_seconds=3.0, input_tokens=12, output_tokens=8,
            model_total_tokens=20, combined_model_and_reranker_input_tokens=12,
            api_attempts=2, api_successful_calls=2)
        return {"query_id": query["query_id"], "question": query["query"], "gold_answer": query["answer"],
            "final_text": final, "final_sha256": control._sha(final.encode()), "terminal_row_sha256": "c" * 64,
            "terminated_by": "done", "metrics": metrics,
            "reranker_summary": {"rerank_calls": 0, "scoring_calls": 0, "scored_passages": 0,
                "input_tokens": 0, "scoring_seconds": 0.0, "preparation_seconds": 0.0}}

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def seal(self):
        self.receipt.pop("receipt_sha256", None)
        self.receipt["receipt_sha256"] = control._digest(self.receipt)
        self.write(control.MANIFEST_PATH, self.manifest)
        self.pins.update(MANIFEST_SHA256=self.manifest["manifest_sha256"],
            COHORT_SHA256=self.manifest["cohort_sha256"],
            PUBLIC_EXPORT_SHA256=self.write(control.PUBLIC_PATH, self.public),
            INTEGRITY_RECEIPT_SHA256=self.write(control.RECEIPT_PATH, self.receipt),
            FROZEN_METADATA_SHA256=self.write(control.METADATA_PATH, self.metadata))
        self.completion["files"] = {"/original/" + p.name: self.pins[pin] for p, pin in (
            (control.PUBLIC_PATH, "PUBLIC_EXPORT_SHA256"), (control.RECEIPT_PATH, "INTEGRITY_RECEIPT_SHA256"),
            (control.METADATA_PATH, "FROZEN_METADATA_SHA256"))}
        self.pins["COMPLETION_CHECK_SHA256"] = self.write(control.COMPLETION_PATH, self.completion)

    def resealed_verify(self):
        # Re-pin only the synthetic fixture so deeper semantic gates are
        # exercised independently of the first byte-hash gate.
        self.seal()
        with mock.patch.multiple(control, **self.pins):
            return control.verify_control_bundle(self.root)

    def test_valid_bundle_preserves_all_public_records_and_empty_finals(self):
        result = control.verify_control_bundle(self.root, common_config=copy.deepcopy(self.config))
        self.assertEqual(result["records"], self.public["records"])
        self.assertEqual(result["records"][0]["final_text"], "")
        self.assertEqual(result["contract_sha256"], self.contract["contract_sha256"])
        self.assertEqual(set(result), {"version", "manifest_sha256", "public_export_sha256",
            "integrity_receipt_sha256", "model", "cohort_sha256", "contract_sha256", "config_sha256",
            "common_config", "records"})
        result["common_config"]["runtime_environment"]["REASONING_EFFORT"] = "other"
        self.assertEqual(control.verify_control_bundle(self.root)["common_config"], self.config)

    def test_file_tamper_rejected_without_repinning(self):
        for relative in (control.PUBLIC_PATH, control.RECEIPT_PATH, control.METADATA_PATH, control.COMPLETION_PATH):
            with self.subTest(path=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    control.verify_control_bundle(self.root)
                path.write_bytes(original)

    def test_manifest_payload_and_cohort_bytes_are_pinned(self):
        self.manifest["config"]["max_turns"] = 101
        self.write(control.MANIFEST_PATH, self.manifest)
        with self.assertRaisesRegex(ValueError, "checksum"):
            control.verify_control_bundle(self.root)
        self.manifest["config"]["max_turns"] = 100
        self.write(control.MANIFEST_PATH, self.manifest)
        (self.root / control.COHORT_PATH).write_bytes(self.cohort_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "cohort SHA256"):
            control.verify_control_bundle(self.root)

    def test_common_configuration_cannot_change_budget_sampling_paths_or_types(self):
        for key, value in (("max_turns", 101), ("concurrency", 8), ("concurrency", 4.0),
                           ("corpus", "/migrated/corpus"), ("temperature", 0.7)):
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "common_config"):
                control.verify_control_bundle(self.root, common_config=changed)
        changed = copy.deepcopy(self.config)
        changed["runtime_environment"]["REASONING_EFFORT"] = "high"
        with self.assertRaisesRegex(ValueError, "common_config"):
            control.verify_control_bundle(self.root, common_config=changed)

    def test_contract_inputs_must_match_even_when_wrapper_label_differs(self):
        for field, value in (("system_prompt", "legacy v14"), ("tool_specs", []),
                             ("user_prompt_template", "Changed {question}")):
            candidate = {**self.contract, field: value}
            with mock.patch.object(control, "_control_contract", return_value=candidate):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "New control input"):
                    control.verify_control_bundle(self.root)
        candidate = {**self.contract, "condition": "control", "version": "new-wrapper"}
        with mock.patch.object(control, "_control_contract", return_value=candidate):
            control.verify_control_bundle(self.root)

    def test_order_question_gold_and_final_hash_fail_independently(self):
        original = copy.deepcopy(self.public["records"])
        for mutate in (lambda rows: rows.reverse(), lambda rows: rows[0].update(question="Changed"),
                       lambda rows: rows[0].update(gold_answer="Changed"),
                       lambda rows: rows[0].update(final_text="Changed"),
                       lambda rows: rows.pop()):
            self.public["records"] = copy.deepcopy(original)
            mutate(self.public["records"])
            with self.subTest(change=mutate), self.assertRaises(ValueError):
                self.resealed_verify()

    def test_missing_or_invalid_numeric_ledger_rejected(self):
        original = copy.deepcopy(self.public["records"])
        for value in (None, True, -1, "12"):
            self.public["records"] = copy.deepcopy(original)
            self.public["records"][1]["metrics"]["input_tokens"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resealed_verify()
        for mutate in (lambda r: r["metrics"].pop("api_attempts_without_usage"),
                       lambda r: r["metrics"].update(model_total_tokens=21),
                       lambda r: r["metrics"].update(combined_model_and_reranker_input_tokens=13),
                       lambda r: r["reranker_summary"].update(scored_passages=1)):
            self.public["records"] = copy.deepcopy(original)
            mutate(self.public["records"][1])
            with self.subTest(change=mutate), self.assertRaises(ValueError):
                self.resealed_verify()

    def test_receipt_must_cover_each_contract_and_numeric_api_ledger(self):
        self.receipt["items"][0]["files"].pop("runs/strict_qwen35/B/api_ledgers/qid_0.jsonl")
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.resealed_verify()

    def test_judge_labels_rejected_instead_of_reused(self):
        self.public["records"][0]["judge"] = {"correct": True}
        with self.assertRaisesRegex(ValueError, "judge labels"):
            self.resealed_verify()


class LocalFrozenControlTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / control.PUBLIC_PATH).exists(), "Local ignored B100 export is unavailable")
    def test_real_frozen_b100_is_read_only_and_matches_new_control(self):
        paths = (control.MANIFEST_PATH, control.PUBLIC_PATH, control.METADATA_PATH,
                 control.RECEIPT_PATH, control.COMPLETION_PATH, control.COHORT_PATH)
        before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
        result = control.verify_control_bundle(ROOT)
        self.assertEqual(len(result["records"]), 100)
        self.assertTrue(any(r["final_text"] == "" for r in result["records"]))
        self.assertEqual(result["manifest_sha256"], control.MANIFEST_SHA256)
        self.assertEqual(result["model"], "Qwen3.5-35B-A3B")
        self.assertEqual(before, {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
        control.verify_control_bundle(ROOT, common_config=result["common_config"])


if __name__ == "__main__":
    unittest.main()
