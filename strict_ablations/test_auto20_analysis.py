"""Forty-final paired blinding, integrity, ledger and coverage checks; no network."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from strict_ablations import auto20_analysis as analysis
from strict_ablations.prompt import build_contract


class AutoTwentyAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cohort = analysis.fixed_cohort()
        self.qids = list(self.cohort)
        config = {"model": analysis.common.MODEL, "max_turns": 100}
        self.reference = {"version": "fixture-control", "manifest_sha256": "11" * 32,
            "public_export_sha256": "22" * 32, "integrity_receipt_sha256": "33" * 32,
            "contract_sha256": "44" * 32, "config_sha256": analysis.digest(config),
            "model": analysis.common.MODEL, "cohort_sha256": analysis.common.COHORT_SHA256}
        manifest = {"config": config, "config_sha256": analysis.digest(config),
            "cohort_sha256": analysis.common.COHORT_SHA256, "question_runs": 20,
            "selected_query_ids": self.qids, "reused_control_questions": 20,
            "control_reference": self.reference, "control_root": "/fixture/pinned-control",
            "contracts": {analysis.ARM: build_contract(analysis.ARM)},
            "analysis_contract": {"version": analysis.VERSION,
                                  "rubric_sha256": analysis.digest(analysis.rubric_document())},
            "execution_plan": {"version": "strict-auto20-pilot-v1", "arms": [analysis.ARM],
                "selection": "first20_in_frozen_fixed100_order", "question_runs": 20,
                "control_policy": "reuse_matching_B100_public_finals; no control requests",
                "stage": "exploratory_pilot"}}
        self.manifest = self.seal(manifest, "manifest_sha256")
        self.new = [self.new_row(qid, index) for index, qid in enumerate(self.qids)]
        self.controls = []
        for index, (qid, query) in enumerate(self.cohort.items()):
            row = self.new_row(qid, index)
            row["reranker_summary"].update(rerank_calls=1, scoring_calls=1,
                scored_passages=10, input_tokens=100, scoring_seconds=1)
            self.controls.append({"query_id": qid, "question": query["query"],
                "gold_answer": query["answer"], "final_text": row["final_text"],
                "final_sha256": row["final_sha256"], "terminal_row_sha256": analysis.digest(["control", qid]),
                "terminated_by": row["terminated_by"],
                "metrics": analysis.common.shared.row_metrics(row, "B"),
                "reranker_summary": row["reranker_summary"],
                "is_correct": False, "judge": {"is_correct": False}})

    @staticmethod
    def seal(value, field="row_sha256"):
        value = copy.deepcopy(value)
        value.pop(field, None)
        value[field] = analysis.digest(value)
        return value

    def new_row(self, qid, index):
        final = "" if index == 0 else "Exact Answer: synthetic fixture final"
        row = {"arm": analysis.ARM, "model": analysis.common.MODEL, "query_id": qid,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "config_sha256": self.manifest["config_sha256"],
            "contract_sha256": build_contract(analysis.ARM)["contract_sha256"],
            "final_text": final, "final_sha256": analysis.text_sha(final),
            "terminated_by": "wall_clock_timeout" if not final else "done", "n_turns": 3,
            "elapsed_seconds": 20 + index, "input_tokens": 12,
            "raw_non_cached_input_tokens": 10, "cached_input_tokens": 2,
            "output_tokens": 8, "reasoning_tokens": 4,
            "api_ledger": {"attempts": 3, "successful_calls": 3, "attempts_without_usage": 0,
                "input_tokens": 12, "output_tokens": 8, "agent_totals_match": True},
            "reranker_summary": {"rerank_calls": 2, "scoring_calls": 2,
                "scored_passages": 20, "input_tokens": 200,
                "scoring_seconds": 2, "preparation_seconds": 3, "page_calls": 0},
            "ablation_summary": {"arm": analysis.ARM, "independent_rerank_calls": 0,
                "independent_rerank_page_calls": 0, "automatic_rerank_calls": 2,
                "search_page_calls": 0, "retrieval_calls": 1, "retrieval_queries": 2,
                "search_invocations": 1, "automatic_rerank_errors": 0, "automatic_empty_scopes": 0},
            "is_correct": True, "judge": {"is_correct": True}}
        return self.seal(row)

    def rows(self):
        return analysis.normalize_records(self.new, self.controls, self.cohort, self.manifest)

    def packet(self):
        rows = self.rows()
        packet, private = analysis.export_blind(rows, self.cohort, self.manifest, seed="01" * 32)
        return rows, packet, private

    def submission(self, packet, private):
        mappings = {row["case_id"]: row for row in private["records"]}
        submission = {key: copy.deepcopy(value) for key, value in packet.items()
            if key not in {"records", "rubric", "review_template"}}
        submission["reviewer"] = "fresh blinded synthetic reviewer"
        submission["records"] = []
        for row in packet["records"]:
            mapped = mappings[row["case_id"]]
            index = self.qids.index(mapped["query_id"])
            threshold = 10 if mapped["arm"] == analysis.CONTROL else 12
            submission["records"].append({"case_id": row["case_id"],
                "final_sha256": row["final_sha256"], "correct": bool(row["final_text"]) and index < threshold,
                "reason": "Fresh comparison of public final to frozen gold."})
        return submission

    def scored(self):
        rows, packet, private = self.packet()
        reviews = analysis.import_reviews(self.submission(packet, private), private, rows, self.manifest)
        labels = analysis.load_labels([reviews], rows, self.manifest)
        return rows, packet, private, reviews, labels

    def test_frozen_first20_selection_and_scope(self):
        self.assertEqual(self.qids, [70, 81, 96, 116, 122, 127, 132, 153, 156, 161,
                                    193, 209, 228, 229, 239, 245, 246, 251, 256, 270])
        for change in ({"question_runs": 300}, {"selected_query_ids": self.qids[::-1]},
                       {"analysis_contract": {"version": "old-scoring"}}):
            manifest = self.seal({**self.manifest, **change}, "manifest_sha256")
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.check_manifest(manifest)

    def test_complete40_normalization_preserves_empty_and_ignores_old_labels(self):
        rows = self.rows()
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(not row["final_text"] for row in rows.values()), 2)
        self.assertTrue(all("is_correct" not in row and "judge" not in row for row in rows.values()))
        self.assertEqual(rows[(analysis.CONTROL, self.qids[0])]["metrics"]["manual_rerank_calls"], 1)
        self.assertEqual(rows[(analysis.ARM, self.qids[0])]["metrics"]["automatic_rerank_calls"], 2)
        self.assertIsNone(rows[(analysis.CONTROL, self.qids[0])]["metrics"]["retrieval_calls"])

    def test_missing_duplicate_extra_or_other_question_is_rejected(self):
        for new, control in ((self.new[:-1], self.controls), (self.new, self.controls[:-1]),
                             (self.new + [self.new[0]], self.controls),
                             ([self.new[0]] * 20, self.controls), (self.new, [self.controls[0]] * 20)):
            with self.subTest(n=len(new), c=len(control)), self.assertRaises(ValueError):
                analysis.normalize_records(new, control, self.cohort, self.manifest)

    def test_model_final_manifest_and_metric_tamper_are_rejected(self):
        changes = ({"model": "Qwen3.8-flash"}, {"final_text": "changed"},
                   {"manifest_sha256": "00" * 32}, {"terminated_by": "api_error"})
        for change in changes:
            new = copy.deepcopy(self.new)
            new[0] = self.seal({**new[0], **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.normalize_records(new, self.controls, self.cohort, self.manifest)
        new = copy.deepcopy(self.new)
        new[0]["ablation_summary"]["automatic_rerank_calls"] = 3
        new[0] = self.seal(new[0])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            analysis.normalize_records(new, self.controls, self.cohort, self.manifest)

    def test_packet_is_anonymous_but_contains_question_gold_final_and_hash(self):
        rows, packet, private = self.packet()
        self.assertEqual(len(packet["records"]), 40)
        self.assertEqual(set(packet["records"][0]), {"case_id", "question", "gold_answer", "final_text", "final_sha256"})
        self.assertTrue(all("arm" not in row and "query_id" not in row and "row_sha256" not in row
                            for row in packet["records"]))
        self.assertEqual(len(private["records"]), 40)
        self.assertNotIn("shuffle_seed_secret", packet)
        self.assertEqual(packet, analysis.export_blind(rows, self.cohort, self.manifest, seed="01" * 32)[0])

    def test_fresh_review_requires_40_boolean_reason_hash_bound_cases(self):
        rows, packet, private = self.packet()
        original = self.submission(packet, private)
        mutations = (lambda s: s["records"].pop(),
            lambda s: s["records"].append(copy.deepcopy(s["records"][0])),
            lambda s: s["records"][0].update(correct=1),
            lambda s: s["records"][0].update(reason=""),
            lambda s: s["records"][0].update(final_sha256="00" * 32),
            lambda s: s.update(packet_sha256="00" * 32), lambda s: s.update(reviewer=""))
        for mutate in mutations:
            submission = copy.deepcopy(original); mutate(submission)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                analysis.import_reviews(submission, private, rows, self.manifest)
        empty = copy.deepcopy(original)
        empty_id = next(row["case_id"] for row in packet["records"] if not row["final_text"])
        next(row for row in empty["records"] if row["case_id"] == empty_id)["correct"] = True
        with self.assertRaisesRegex(ValueError, "empty"):
            analysis.import_reviews(empty, private, rows, self.manifest)

    def test_mapping_and_public_payload_changes_cannot_reuse_reviews(self):
        rows, packet, private = self.packet()
        changed = copy.deepcopy(private)
        changed["records"][0]["row_sha256"] = "00" * 32
        changed = self.seal(changed, "mapping_sha256")
        with self.assertRaisesRegex(ValueError, "mapped"):
            analysis.import_reviews(self.submission(packet, private), changed, rows, self.manifest)
        rows = copy.deepcopy(rows)
        rows[(analysis.ARM, self.qids[0])]["question"] += " changed"
        with self.assertRaisesRegex(ValueError, "payload"):
            analysis.import_reviews(self.submission(packet, private), private, rows, self.manifest)

    def test_summary_counts_20_denominators_and_real_scoring_resources(self):
        rows, _, _, reviews, labels = self.scored()
        summary = analysis.summarize(rows, self.qids, labels, self.manifest)
        self.assertEqual(summary["freshly_reviewed_finals"], 40)
        self.assertEqual(summary["arms"][analysis.CONTROL]["correct"], 9)
        self.assertEqual(summary["arms"][analysis.ARM]["correct"], 11)
        self.assertEqual(summary["arms"][analysis.ARM]["accuracy_percent"], 55)
        self.assertEqual(summary["arms"][analysis.ARM]["n"], 20)
        self.assertEqual(summary["paired_vs_control"]["auto_only"], 2)
        self.assertEqual(summary["paired_vs_control"]["accuracy_difference_percentage_points"], 10)
        metrics = summary["arms"][analysis.ARM]["metrics"]
        self.assertEqual(metrics["automatic_rerank_calls"]["known_sum"], 40)
        self.assertEqual(metrics["scoring_calls"]["known_sum"], 40)
        self.assertEqual(metrics["scored_passages"]["known_sum"], 400)
        self.assertEqual(metrics["model_total_tokens"]["known_sum"], 400)
        self.assertEqual(metrics["reranker_input_tokens"]["known_sum"], 4000)
        self.assertEqual(summary["arms"][analysis.CONTROL]["metrics"]["retrieval_calls"]["missing_rows"], 20)

    def test_partial_or_historical_review_cannot_fill_coverage(self):
        rows, _, _, reviews, labels = self.scored()
        for mutation in (lambda d: d.update(source="historical_judge"), lambda d: d["records"].pop()):
            document = copy.deepcopy(reviews); mutation(document)
            document = self.seal(document, "reviews_sha256")
            with self.assertRaises(ValueError):
                analysis.load_labels([document], rows, self.manifest)
        labels[(analysis.ARM, self.qids[0])]["final_sha256"] = "00" * 32
        with self.assertRaisesRegex(ValueError, "Stale summary"):
            analysis.summarize(rows, self.qids, labels, self.manifest)

    def test_collect_delegates_20_integrity_checks_and_uses_pinned_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = self
            class Suite:
                def verify_manifest(self, *, resources):
                    test.assertIs(resources, False)
                    return copy.deepcopy(test.manifest)
                def terminal_path(self, arm, qid):
                    test.assertEqual(arm, analysis.ARM)
                    return root / "runs" / arm / "_per_query" / f"qid_{qid}.json"
                def validate_terminal(self, *args):
                    calls["terminal"] += 1
                def verify_execution_artifacts(self, *args):
                    calls["artifacts"] += 1
            suite, calls = Suite(), {"terminal": 0, "artifacts": 0}
            for row in self.new:
                analysis.write_new(suite.terminal_path(analysis.ARM, row["query_id"]), row)
            bundle = {**self.reference, "records": self.controls}
            with patch.object(analysis, "verify_control_bundle", return_value=bundle) as verify:
                rows, cohort, manifest, provenance = analysis.collect(root, suite=suite)
            self.assertEqual(len(rows), 40)
            self.assertEqual(calls, {"terminal": 20, "artifacts": 20})
            verify.assert_called_once_with(self.manifest["control_root"], common_config=self.manifest["config"])
            self.assertEqual(len(provenance), 21)
            suite.terminal_path(analysis.ARM, self.qids[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "coverage"):
                analysis.collect(root, suite=suite)

    def test_cli_export_is_exclusive_and_private_mapping_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet, mapping = root / "packet.json", root / "mapping.json"
            with patch.object(analysis, "collect", return_value=(self.rows(), self.cohort, self.manifest, {})):
                analysis.main(["--root", str(root), "export", "--out", str(packet), "--private-map", str(mapping)])
                self.assertEqual(mapping.stat().st_mode & 0o777, 0o600)
                with self.assertRaisesRegex(ValueError, "new output"):
                    analysis.main(["export", "--out", str(packet), "--private-map", str(mapping)])


if __name__ == "__main__":
    unittest.main()
