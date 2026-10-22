"""Synthetic-only checks; no benchmark answer or model is evaluated here."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from operations import compare_b_historical as comparison


def metrics(*, new=False):
    result = {key: None for key in comparison.METRICS}
    result.update(n_turns=2 if new else 3, elapsed_seconds=3 if new else 6,
        input_tokens=200 if new else 100, raw_non_cached_input_tokens=195 if new else 95,
        cached_input_tokens=5, output_tokens=20, reasoning_tokens=0,
        model_total_tokens=220 if new else 120,
        rerank_calls=1 if new else 0, scoring_calls=1 if new else 0,
        scored_passages=10 if new else 0, reranker_input_tokens=50 if new else 0,
        scoring_seconds=.25 if new else 0, preparation_seconds=.5 if new else 0,
        combined_model_and_reranker_input_tokens=250 if new else 100)
    if new:
        result.update(api_attempts=2, api_successful_calls=2, api_attempts_without_usage=0)
    return result


def synthetic_records():
    records = []
    for group in comparison.GROUPS:
        for qid in (101, 202, 303):
            text = f"Synthetic public final variant {comparison.GROUPS.index(group)} for question {qid}"
            records.append({"source_kind": group, "query_id": qid,
                "question": f"Synthetic question {qid}", "gold_answer": f"Synthetic gold {qid}",
                "final_text": text, "final_sha256": comparison.analyze.text_sha(text),
                "terminated_by": "done", "metrics": metrics(new=group == "new_B"),
                "origin": {"path": "PRIVATE_SOURCE_DIRECTORY", "sha256": "f" * 64},
                "judge": "DO_NOT_INHERIT_JUDGE", "is_correct": "DO_NOT_INHERIT_VERDICT"})
    return records


class HistoricalComparisonTests(unittest.TestCase):
    def bundle(self):
        return comparison.make_bundle(synthetic_records(),
            {"source": "PRIVATE_SOURCE_BINDING", "epoch": "e" * 64},
            {"archived_correct": 61, "archived_n": 100})

    def review(self, public, private):
        # Fixture labels are generated only for synthetic records.
        mapping = {row["case_id"]: row for row in private["records"]}
        return {"packet_sha256": public["packet_sha256"],
            "rubric_sha256": public["rubric_sha256"], "reviewer": "synthetic-test-fixture",
            "records": [{"case_id": row["case_id"], "final_sha256": row["final_sha256"],
                "correct": mapping[row["case_id"]]["source_kind"] == "new_B"
                    or mapping[row["case_id"]]["query_id"] == 101,
                "reason": "Synthetic fixture judgement; not a benchmark evaluation."}
                for row in public["records"]]}

    def test_blind_packet_has_only_public_fields_and_no_metadata_or_old_verdicts(self):
        public, private, material = self.bundle()
        self.assertEqual(len(public["records"]), 6)
        for row in public["records"]:
            self.assertEqual(set(row), {"case_id", "question", "gold_answer", "final_text", "final_sha256"})
            self.assertEqual(len(row["case_id"]), 24)
        import json
        packet = json.dumps(public)
        for forbidden in ("PRIVATE_SOURCE", "query_id", "metrics", "shuffle_salt", "DO_NOT_INHERIT", *comparison.GROUPS):
            self.assertNotIn(forbidden, packet)
        self.assertNotIn("DO_NOT_INHERIT", json.dumps((private, material)))
        self.assertEqual(len(private["shuffle_salt_hex"]), 64)
        comparison.validate_bundle(public, private, material)

    def test_independent_builds_have_fresh_private_random_salts_and_ids(self):
        a, map_a, _ = self.bundle(); b, map_b, _ = self.bundle()
        self.assertNotEqual(map_a["shuffle_salt_hex"], map_b["shuffle_salt_hex"])
        self.assertNotEqual({row["case_id"] for row in a["records"]}, {row["case_id"] for row in b["records"]})

    def test_missing_wrong_sha_duplicate_or_nonboolean_reviews_are_rejected(self):
        public, private, material = self.bundle()
        valid = self.review(public, private)
        variants = []
        missing = copy.deepcopy(valid); missing["records"].pop(); variants.append(missing)
        sha = copy.deepcopy(valid); sha["records"][0]["final_sha256"] = "0" * 64; variants.append(sha)
        duplicate = copy.deepcopy(valid); duplicate["records"][-1] = copy.deepcopy(duplicate["records"][0]); variants.append(duplicate)
        integer = copy.deepcopy(valid); integer["records"][0]["correct"] = 1; variants.append(integer)
        blank = copy.deepcopy(valid); blank["records"][0]["reason"] = " "; variants.append(blank)
        stale = copy.deepcopy(valid); stale["packet_sha256"] = "0" * 64; variants.append(stale)
        for submission in variants:
            with self.subTest(submission=submission), self.assertRaises(ValueError):
                comparison.validate_reviews(submission, public, private, material)

    def test_pair_aggregation_and_exact_mcnemar_are_correct(self):
        public, private, material = self.bundle()
        summary = comparison.summarize_reviews(self.review(public, private), public, private, material)
        self.assertEqual(summary["groups"]["historical_baseline"]["correct"], 1)
        self.assertEqual(summary["groups"]["new_B"]["correct"], 3)
        paired = summary["paired_comparison"]
        self.assertEqual({key: paired[key] for key in ("both_correct", "baseline_only", "B_only", "both_incorrect")},
                         {"both_correct": 1, "baseline_only": 0, "B_only": 2, "both_incorrect": 0})
        self.assertEqual(paired["delta_correct"], 2)
        self.assertAlmostEqual(paired["delta_accuracy"], 2 / 3)
        self.assertEqual(paired["mcnemar_exact_two_sided"], .5)
        old = summary["groups"]["historical_baseline"]["metrics"]
        new = summary["groups"]["new_B"]["metrics"]
        self.assertIsNone(old["api_attempts"]["sum"])
        self.assertEqual(old["api_attempts"]["missing"], 3)
        self.assertEqual(new["input_tokens"]["sum"], 600)
        self.assertEqual(new["output_tokens"]["sum"], 60)
        self.assertEqual(new["model_total_tokens"]["sum"], 660)
        self.assertEqual(new["reranker_input_tokens"]["sum"], 150)
        self.assertEqual(new["elapsed_seconds"]["sum"], 9)
        self.assertEqual(new["scoring_seconds"]["sum"], .75)

    def test_empty_natural_final_cannot_be_marked_correct(self):
        rows = synthetic_records()
        rows[0].update(final_text="", final_sha256=comparison.analyze.text_sha(""), terminated_by="max_turns")
        public, private, material = comparison.make_bundle(rows, {"source": "fixture"}, {})
        review = self.review(public, private)
        empty = next(row for row in public["records"] if not row["final_text"])
        next(row for row in review["records"] if row["case_id"] == empty["case_id"])["correct"] = True
        with self.assertRaises(ValueError):
            comparison.validate_reviews(review, public, private, material)

    def test_saved_input_is_normalized_without_double_counting_reasoning(self):
        row = {"n_turns": 5, "elapsed_seconds": 10,
               "agent_usage": {"input_tokens": 10, "cached_input_tokens": 4,
                               "output_tokens": 20, "reasoning_tokens": 5}}
        result = comparison.legacy_metrics(row)
        self.assertEqual(result["input_tokens"], 14)
        self.assertEqual(result["raw_non_cached_input_tokens"], 10)
        self.assertEqual(result["model_total_tokens"], 34)
        self.assertEqual(result["reasoning_tokens"], 5)

    def test_pinned_sources_and_existing_exports_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            comparison.safe_write(path, {"public": "fixture"}, private=True)
            sha = comparison.analyze.file_sha(path)
            self.assertEqual(comparison.checked_file(path, sha), {"public": "fixture"})
            comparison.safe_write(path, {"public": "fixture"}, private=True)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                comparison.safe_write(path, {"public": "changed"})
            path.write_text('{"public":"changed"}')
            with self.assertRaises(ValueError):
                comparison.checked_file(path, sha)

    def test_fresh_mapping_mutation_is_rejected_even_with_recomputed_digest(self):
        public, private, material = self.bundle()
        changed = copy.deepcopy(private)
        changed["records"][0]["query_id"] += 1
        changed.pop("mapping_sha256")
        changed = comparison.seal(changed, "mapping_sha256")
        material = copy.deepcopy(material); material.pop("materials_sha256")
        material["mapping_sha256"] = changed["mapping_sha256"]
        material = comparison.seal(material, "materials_sha256")
        with self.assertRaises(ValueError):
            comparison.validate_bundle(public, changed, material)

    def test_B_export_rejects_epoch_final_hash_or_question_drift(self):
        rows = [row for row in synthetic_records() if row["source_kind"] == "new_B"]
        cohort = {row["query_id"]: {"query_id": row["query_id"], "query": row["question"], "answer": row["gold_answer"]} for row in rows}
        manifest = {"config": {"model": comparison.analyze.MODEL}, "contracts": {"B": {"fixture": "contract"}}}
        binding = {"manifest_sha256": comparison.MANIFEST_SHA, "epoch_sha256": comparison.EPOCH_SHA,
                   "cohort_sha256": comparison.analyze.COHORT_SHA256, "model": comparison.analyze.MODEL}
        metadata = {**binding, "original_config": manifest["config"], "B_contract": manifest["contracts"]["B"],
                    "rubric": comparison.analyze.rubric_document(),
                    "rubric_sha256": comparison.analyze.digest(comparison.analyze.rubric_document())}
        public = {**binding, "version": "strict-B100-public-finals-metrics-v1",
                  "records": [{**row, "terminal_row_sha256": "f" * 64} for row in rows]}
        self.assertEqual(len(comparison.validate_b_export(public, metadata, manifest, cohort)), 3)
        variants = []
        epoch = copy.deepcopy(public); epoch["epoch_sha256"] = "0" * 64; variants.append(epoch)
        sha = copy.deepcopy(public); sha["records"][0]["final_sha256"] = "0" * 64; variants.append(sha)
        question = copy.deepcopy(public); question["records"][0]["question"] += " changed"; variants.append(question)
        repeated = copy.deepcopy(public); repeated["records"][-1] = copy.deepcopy(repeated["records"][0]); variants.append(repeated)
        interrupted = copy.deepcopy(public); interrupted["records"][0]["terminated_by"] = "api_error"; variants.append(interrupted)
        for altered in variants:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                comparison.validate_b_export(altered, metadata, manifest, cohort)

    def proof_fixture(self, directory):
        root = Path(directory)
        finals, metadata, integrity_path = (root / name for name in ("finals.json", "metadata.json", "integrity.json"))
        text = "Synthetic public final"
        public = {"epoch_sha256": comparison.EPOCH_SHA, "records": [
            {"final_text": text, "final_sha256": comparison.analyze.text_sha(text)}]}
        comparison.safe_write(finals, public)
        comparison.safe_write(metadata, {"epoch_sha256": comparison.EPOCH_SHA})
        receipt = comparison.seal({"version": "strict-B100-shutdown-integrity-v1", "arm": "B",
            "manifest_sha256": comparison.MANIFEST_SHA, "epoch_sha256": comparison.EPOCH_SHA,
            "checker_sha256": comparison.B_CHECKER_SHA, "config_sha256": "c" * 64,
            "a_artifacts_unchanged": True, "a_before_sha256": "a" * 64, "a_after_sha256": "a" * 64,
            "items": [{"query_id": qid, "files": {"fixture": "f" * 64}} for qid in range(100)]}, "receipt_sha256")
        comparison.safe_write(integrity_path, receipt)
        proof = {"ready_for_shutdown": True, "manifest_sha256": comparison.MANIFEST_SHA,
            "epoch_sha256": comparison.EPOCH_SHA, "a_artifacts_unchanged": True, "B_terminal_files": 100,
            "files": {"/remote/reports/strict_qwen35/scoring_B_only/public_finals_metrics.json": comparison.analyze.file_sha(finals),
                "/remote/reports/strict_qwen35/scoring_B_only/frozen_metadata.json": comparison.analyze.file_sha(metadata),
                "/remote/reports/strict_qwen35/operations/epoch_20261022/b_shutdown_integrity_receipt.json": comparison.analyze.file_sha(integrity_path)}}
        proof_path = root / "remote_check.json"
        comparison.safe_write(proof_path, proof)
        return proof_path, finals, metadata, integrity_path

    def test_external_proof_rejects_self_rehashed_final_or_changed_both_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.proof_fixture(directory)
            verified = comparison.verify_completion_proof(*paths, config_sha256="c" * 64)
            self.assertEqual(verified["expected"]["epoch_sha256"], comparison.EPOCH_SHA)
            self.assertEqual(verified["proof"]["sha256"], comparison.analyze.file_sha(paths[0]))
            public = comparison.analyze.load(paths[1])
            public["records"][0]["final_text"] = "Replaced final with a newly self-consistent SHA"
            public["records"][0]["final_sha256"] = comparison.analyze.text_sha(public["records"][0]["final_text"])
            import json
            paths[1].write_text(json.dumps(public))
            with self.assertRaises(ValueError):
                comparison.verify_completion_proof(*paths, config_sha256="c" * 64)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.proof_fixture(directory)
            for path in paths[1:3]:
                document = comparison.analyze.load(path); document["epoch_sha256"] = "0" * 64
                path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                comparison.verify_completion_proof(*paths, config_sha256="c" * 64)

    def test_external_proof_is_required_and_must_be_ready_and_complete(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            paths = self.proof_fixture(directory)
            proof = comparison.analyze.load(paths[0]); proof["ready_for_shutdown"] = False
            paths[0].write_text(json.dumps(proof))
            with self.assertRaises(ValueError):
                comparison.verify_completion_proof(*paths)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.proof_fixture(directory)
            proof = comparison.analyze.load(paths[0])
            proof["files"] = {key: value for key, value in proof["files"].items() if not key.endswith("frozen_metadata.json")}
            paths[0].write_text(json.dumps(proof))
            with self.assertRaises(ValueError):
                comparison.verify_completion_proof(*paths)
            paths[0].unlink()
            with self.assertRaises(FileNotFoundError):
                comparison.verify_completion_proof(*paths)


if __name__ == "__main__":
    unittest.main()
