"""Fresh review identity, historical pairing and separate-cost accounting."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ablation_analysis_fixture", ROOT / "analyze_ablations.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AblationAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "analysis"
        self.cohort = {i: {"query_id": i, "query": f"question {i}", "answer": f"answer {i}"}
                       for i in (1, 2)}

    def row(self, variant="search_paging", identity=1):
        return {"query_id": identity, "question": self.cohort[identity]["query"],
            "gold_answer": self.cohort[identity]["answer"], "final_text": "Exact Answer: answer 1",
            "terminated_by": "done", "model": "Qwen3.5-35B-A3B", "n_turns": 5,
            "elapsed_seconds": 12, "is_correct": True,
            "judge": {"correct": "no", "reasoning": ""},
            "agent_usage": {"input_tokens": 101, "output_tokens": 19},
            "tool_usage": {"search": 4, "rerank": 0},
            "run_config": {"extras": {"ablation_variant": variant}},
            "runner_specific": {"n_bm25_searches": 2, "n_bm25_queries": 3,
                "reranker_summary": {"scoring_calls": 2, "scored_passages": 2000,
                    "input_tokens": 999, "ablation_search": {
                        "search_page_calls": 2, "auto_rerank_calls": 3}}}}

    def save(self, path, value):
        analysis.write_json(path, value)
        return path

    def historical(self):
        records = []
        for identity in self.cohort:
            row = self.row(identity=identity)
            row["correct"] = False
            records.append(row)
        return self.save(self.root / "historical.json", {"records": records, "strict": {"correct": 0}})

    def cases(self):
        return {(variant, identity): self.row(variant, identity)
                for variant in analysis.VARIANTS for identity in self.cohort}

    def reviewed(self, packet, correct=False):
        return {"version": analysis.VERSION, "packet_sha256": packet["packet_sha256"],
            "records": [{"review_id": row["review_id"], "content_sha256": row["content_sha256"],
                         "correct": correct, "reason": "fresh public-final review"}
                        for row in packet["records"]]}

    def test_blind_ids_are_stable_and_reviews_bind_exact_answer_content(self):
        cases = self.cases()
        packet, private = analysis.review_packets(cases, self.cohort, self.output)
        self.assertEqual(len(packet["records"]), 6)
        for row in packet["records"]:
            self.assertNotIn("variant", row)
            self.assertNotIn("query_id", row)
        repeated, mapping = analysis.review_packets(cases, self.cohort, self.output)
        self.assertEqual(packet, repeated)
        self.assertEqual(private, mapping)
        path = self.save(self.root / "reviews.json", self.reviewed(packet))
        labels = analysis.apply_reviews(path, packet, private)
        self.assertEqual(labels, {key: False for key in cases})
        changed = copy.deepcopy(cases)
        changed[("no_bash", 1)]["final_text"] += " changed"
        with self.assertRaisesRegex(ValueError, "result changed"):
            analysis.review_packets(changed, self.cohort, self.output)

    def test_missing_duplicate_stale_wrong_query_and_unfinished_reviews_rejected(self):
        packet, private = analysis.review_packets(self.cases(), self.cohort, self.output)
        valid = self.reviewed(packet)
        versions = []
        missing = copy.deepcopy(valid); missing["records"].pop(); versions.append(missing)
        duplicate = copy.deepcopy(valid); duplicate["records"].append(duplicate["records"][0]); versions.append(duplicate)
        stale = copy.deepcopy(valid); stale["records"][0]["content_sha256"] = "0" * 64; versions.append(stale)
        wrong_query = copy.deepcopy(valid); wrong_query["records"][0]["query_id"] = 900; versions.append(wrong_query)
        blank = copy.deepcopy(valid); blank["records"][0]["correct"] = None; versions.append(blank)
        old_packet = copy.deepcopy(valid); old_packet["packet_sha256"] = "0" * 64; versions.append(old_packet)
        for number, review in enumerate(versions):
            with self.subTest(number=number), self.assertRaises(ValueError):
                analysis.apply_reviews(self.save(self.root / f"invalid{number}.json", review), packet, private)

    def test_analyze_preserves_filled_review_template_when_used_as_input(self):
        cases = self.cases()
        for (variant, identity), row in cases.items():
            self.save(self.root / "runs" / variant / "_per_query" / f"qid_{identity}.json", row)
        history = self.historical()
        with patch.object(analysis, "load_cohort", return_value=self.cohort):
            first = analysis.analyze(self.root, history, self.output)
            self.assertTrue(all(arm["accuracy"] is None for arm in first["arms"].values()))
            packet = analysis.load(self.output / "review_packet.json")
            template = self.save(self.output / "review_template.json", self.reviewed(packet, True))
            second = analysis.analyze(self.root, history, self.output, template)
        self.assertTrue(all(arm["accuracy"] == 1.0 for arm in second["arms"].values()))
        self.assertTrue(analysis.load(template)["records"][0]["correct"])
        self.assertEqual(second["arms"]["no_bash"]["paired_vs_historical_v14"]["wrong_to_correct"], 2)

    def test_unjudged_partial_outputs_do_not_infer_correctness_from_placeholders(self):
        cases = {("auto_rerank", 1): self.row("auto_rerank")}
        summary = analysis.summarize(cases, self.cohort, {1: False, 2: False},
            {"n": 2, "correct": 0})
        arm = summary["arms"]["auto_rerank"]
        self.assertEqual(arm["completed"], 1)
        self.assertEqual(arm["expected"], 2)
        self.assertIsNone(arm["correct"])
        self.assertIsNone(arm["accuracy"])
        self.assertNotIn("paired_vs_historical_v14", arm)
        totals = arm["totals_for_completed_rows"]
        self.assertEqual(totals["retrieval_calls"], 2)
        self.assertEqual(totals["model_search_calls"], 4)
        self.assertEqual(totals["search_page_calls"], 2)
        self.assertEqual(totals["model_rerank_calls"], 0)
        self.assertEqual(totals["auto_rerank_calls"], 3)
        self.assertEqual(totals["backend_scoring_calls"], 2)
        self.assertEqual(totals["agent_input_tokens"], 101)
        self.assertEqual(totals["backend_input_tokens"], 999)

    def test_result_identity_duplicates_and_wrong_arm_cannot_be_silently_used(self):
        path = self.root / "runs/search_paging/_per_query/qid_1.json"
        row = self.row()
        self.save(path, row)
        self.assertEqual(len(analysis.collect_rows(self.root, self.cohort)), 1)
        row["question"] = "another question"
        self.save(path, row)
        with self.assertRaisesRegex(ValueError, "question/gold"):
            analysis.collect_rows(self.root, self.cohort)
        row = self.row("no_bash")
        self.save(path, row)
        with self.assertRaisesRegex(ValueError, "wrong ablation"):
            analysis.collect_rows(self.root, self.cohort)
        self.save(path, self.row())
        self.save(path.with_name("duplicate.json"), self.row())
        with self.assertRaises(ValueError):
            analysis.collect_rows(self.root, self.cohort)

    def test_absent_public_final_cannot_be_marked_correct(self):
        row = self.row(); row["final_text"] = ""; row["terminated_by"] = "max_turns"
        packet, private = analysis.review_packets({("search_paging", 1): row}, self.cohort, self.output)
        path = self.save(self.root / "reviews.json", self.reviewed(packet, True))
        with self.assertRaisesRegex(ValueError, "absent public final"):
            analysis.apply_reviews(path, packet, private)

    def test_no_completed_runs_produces_empty_review_packet_and_no_accuracy(self):
        with patch.object(analysis, "load_cohort", return_value=self.cohort):
            summary = analysis.analyze(self.root, self.historical(), self.output)
        self.assertEqual(summary["completed_question_runs"], 0)
        self.assertEqual(analysis.load(self.output / "review_packet.json")["records"], [])
        self.assertTrue(all(arm["accuracy"] is None for arm in summary["arms"].values()))

    def test_cohort_and_historical_records_are_validated(self):
        path = self.root / "questions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.cohort.values()))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(analysis.load_cohort(path, expected_sha=sha, expected_count=2), self.cohort)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            analysis.load_cohort(path, expected_sha="0" * 64, expected_count=2)
        history = self.historical()
        values = analysis.load(history)
        values["records"][0]["correct"] = True
        self.save(history, values)
        with self.assertRaisesRegex(ValueError, "strict count"):
            analysis.historical_reference(history, self.cohort)


if __name__ == "__main__":
    unittest.main()
