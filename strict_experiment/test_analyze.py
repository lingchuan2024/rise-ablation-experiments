import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("strict_analysis_under_test", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class StrictAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        self.cohort_bytes = (MODULE_PATH.parents[1] / "data/queries_100.jsonl").read_bytes()
        (self.root / "data/queries_100.jsonl").write_bytes(self.cohort_bytes)
        self.cohort = {row["query_id"]: row for row in [json.loads(line) for line in self.cohort_bytes.decode().splitlines()]}
        self.qids = list(self.cohort)[:2]
        config = {"model": analysis.MODEL, "max_turns": 100}
        self.manifest = {"version": analysis.EXPERIMENT_VERSION, "config": config,
            "config_sha256": analysis.digest(config), "cohort_sha256": analysis.COHORT_SHA256,
            "query_ids": list(self.cohort), "contracts": {arm: {"contract_sha256": arm.lower() * 64} for arm in analysis.ARMS},
            "analysis_contract": {"version": analysis.VERSION, "rubric_sha256": analysis.digest(analysis.rubric_document())}}
        self.manifest["manifest_sha256"] = analysis.digest(self.manifest)
        analysis.write_new(self.root / "reports/strict_qwen35/run_manifest.json", self.manifest)
        self.rows = {(arm, qid): self.make_row(arm, qid) for arm in analysis.ARMS for qid in self.qids}

    def make_row(self, arm, qid):
        row = {"version": analysis.EXPERIMENT_VERSION, "model": analysis.MODEL,
            "arm": arm, "query_id": qid, "manifest_sha256": self.manifest["manifest_sha256"],
            "config_sha256": self.manifest["config_sha256"], "contract_sha256": analysis.contract_sha(self.manifest, arm),
            "question_sha256": analysis.text_sha(self.cohort[qid]["query"]),
            "final_text": f"New final for {arm}/{qid}", "terminated_by": "done", "n_turns": 3,
            "elapsed_seconds": 20, "input_tokens": 12, "raw_non_cached_input_tokens": 10,
            "cached_input_tokens": 2, "output_tokens": 8, "reasoning_tokens": 4,
            "api_ledger": {"attempts": 3, "successful_calls": 3, "attempts_without_usage": 0,
                "input_tokens": 12, "output_tokens": 8, "agent_totals_match": True},
            "reranker_summary": {} if arm == "A" else {"rerank_calls": 1, "scoring_calls": 1,
                "scored_passages": 10, "input_tokens": 100, "scoring_seconds": 2, "preparation_seconds": 3},
            "is_correct": True, "judge": {"is_correct": True}}
        return self.seal(row)

    def seal(self, row):
        row = copy.deepcopy(row)
        row.pop("row_sha256", None)
        row["final_sha256"] = analysis.text_sha(row["final_text"])
        row["row_sha256"] = analysis.digest(row)
        return row

    def submit(self, public, *, correct=True):
        return {key: value for key, value in {**public, "reviewer": "new independent reviewer",
            "records": [{"case_id": row["case_id"], "final_sha256": row["final_sha256"],
                "correct": correct, "reason": "Compared the new public final to the frozen gold."} for row in public["records"]]}.items()
            if key not in {"review_template", "rubric"}}

    def test_manifest_and_rows_bind_actual_frozen_cohort(self):
        manifest, cohort, provenance = analysis.load_experiment(self.root)
        self.assertEqual(len(cohort), 100)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(provenance["cohort_file_sha256"], analysis.COHORT_SHA256)
        for (arm, qid), row in self.rows.items():
            analysis.write_new(self.root / "runs/strict_qwen35" / arm / "_per_query" / f"qid_{qid}.json", row)
        raw = self.root / "runs/strict_qwen35/A/_traces/never-read.json"
        raw.parent.mkdir(parents=True)
        raw.write_text("not JSON, must not open")
        rows, qids, provenance = analysis.collect_rows(self.root, manifest, cohort, scope="first10")
        self.assertEqual(rows, self.rows)
        self.assertEqual(len(qids), 10)
        self.assertEqual(len(provenance), 4)

    def test_row_hash_config_final_and_infrastructure_rejected(self):
        row = copy.deepcopy(self.rows[("A", self.qids[0])])
        row["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            analysis.validate_row(row, "A", self.qids[0], self.manifest, self.cohort)
        for key, value in (("config_sha256", "other"), ("question_sha256", "other"), ("terminated_by", "api_error:502")):
            changed = self.seal({**row, key: value, "input_tokens": 12})
            with self.assertRaises(ValueError):
                analysis.validate_row(changed, "A", self.qids[0], self.manifest, self.cohort)

    def test_normalized_tokens_count_cache_and_reasoning_once(self):
        metrics = analysis.row_metrics(self.rows[("B", self.qids[0])], "B")
        self.assertEqual(metrics["model_total_tokens"], 20)
        self.assertEqual(metrics["reranker_input_tokens"], 100)
        self.assertEqual(metrics["combined_model_and_reranker_input_tokens"], 112)
        changed = copy.deepcopy(self.rows[("B", self.qids[0])])
        changed["input_tokens"] = 10
        with self.assertRaisesRegex(ValueError, "noncached"):
            analysis.row_metrics(changed, "B")
        changed = copy.deepcopy(self.rows[("A", self.qids[0])])
        changed["reranker_summary"] = {"scoring_calls": 1}
        with self.assertRaisesRegex(ValueError, "must not incur"):
            analysis.row_metrics(changed, "A")

    def test_reranker_alias_and_unknown_counters(self):
        row = copy.deepcopy(self.rows[("B", self.qids[0])])
        row["reranker_summary"]["scoring_input_tokens"] = row["reranker_summary"].pop("input_tokens")
        self.assertEqual(analysis.row_metrics(row, "B")["reranker_input_tokens"], 100)
        row["reranker_summary"].pop("scored_passages")
        with self.assertRaises(ValueError):
            analysis.row_metrics(row, "B")

    def test_blinding_is_stable_and_has_no_arm_or_source_paths(self):
        public, private = analysis.export_blind(self.rows, self.cohort, self.manifest, seed="01" * 32)
        repeat, _ = analysis.export_blind(self.rows, self.cohort, self.manifest, seed="01" * 32)
        self.assertEqual(public, repeat)
        self.assertEqual(public["mapping_sha256"], private["mapping_sha256"])
        for row in public["records"]:
            self.assertEqual(set(row), {"case_id", "question", "gold_answer", "final_text", "final_sha256"})
        self.assertNotIn("shuffle_seed_secret", public)
        # The solver's prose can reveal its own behavior; artifact metadata must not.
        self.assertEqual(len({row["case_id"] for row in public["records"]}), 4)

    def test_only_fresh_complete_hash_bound_reviews_accepted(self):
        public, private = analysis.export_blind(self.rows, self.cohort, self.manifest, seed="02" * 32)
        submission = self.submit(public)
        reviews = analysis.import_reviews(submission, private, self.rows, self.manifest)
        self.assertEqual(len(analysis.load_labels([reviews], self.rows, self.manifest)), 4)
        for changed in ({**submission, "rubric_sha256": "old"}, {**submission, "records": submission["records"][:-1]}):
            with self.assertRaises(ValueError):
                analysis.import_reviews(changed, private, self.rows, self.manifest)
        changed = copy.deepcopy(submission)
        changed["records"][0]["final_sha256"] = "stale"
        with self.assertRaises(ValueError):
            analysis.import_reviews(changed, private, self.rows, self.manifest)
        with self.assertRaises(ValueError):
            analysis.load_labels([reviews, reviews], self.rows, self.manifest)

    def test_empty_natural_failure_is_retained_and_never_correct(self):
        key = ("A", self.qids[0])
        self.rows[key] = self.seal({**self.rows[key], "final_text": "", "terminated_by": "max_turns", "n_turns": 100})
        public, private = analysis.export_blind(self.rows, self.cohort, self.manifest, seed="03" * 32)
        with self.assertRaisesRegex(ValueError, "empty"):
            analysis.import_reviews(self.submit(public), private, self.rows, self.manifest)
        labels = analysis.load_labels([analysis.import_reviews(self.submit(public, correct=False), private, self.rows, self.manifest)], self.rows, self.manifest)
        summary = analysis.summarize(self.rows, self.qids, labels, self.manifest)
        self.assertEqual(summary["arms"]["A"]["empty_public_finals"], 1)
        self.assertEqual(summary["arms"]["A"]["terminal_questions"], 2)
        self.assertEqual(summary["arms"]["A"]["accuracy_percent"], 0)

    def test_placeholders_ignored_and_partial_accuracy_not_presented(self):
        summary = analysis.summarize(self.rows, self.qids, {}, self.manifest)
        self.assertIsNone(summary["arms"]["A"]["accuracy_percent"])
        self.assertIsNone(summary["arms"]["A"]["correct"])
        self.assertEqual(summary["arms"]["A"]["known_correct_among_reviewed"], 0)
        self.assertIsNone(summary["paired"]["net_correct_b_minus_a"])
        self.assertIsNone(summary["paired"]["mcnemar_exact_two_sided_p"])
        self.assertEqual(summary["arms"]["B"]["metrics"]["model_total_tokens"]["known_sum"], 40)
        self.assertEqual(summary["arms"]["B"]["metrics"]["reranker_input_tokens"]["known_sum"], 200)

    def test_paired_difference_exact_mcnemar(self):
        labels = {key: {"correct": key[0] == "B"} for key in self.rows}
        summary = analysis.summarize(self.rows, self.qids, labels, self.manifest)
        self.assertEqual(summary["paired"]["accuracy_difference_percentage_points"], 100)
        self.assertEqual(summary["paired"]["a_only_correct"], 0)
        self.assertEqual(summary["paired"]["b_only_correct"], 2)
        self.assertEqual(summary["paired"]["mcnemar_exact_two_sided_p"], .5)
        self.assertEqual(analysis.mcnemar_exact(0, 4), .125)
        self.assertEqual(analysis.mcnemar_exact(0, 0), 1)

    def test_api_unknown_usage_is_explicit_and_mismatched_totals_rejected(self):
        row = copy.deepcopy(self.rows[("B", self.qids[0])])
        row["api_ledger"]["attempts_without_usage"] = 1
        row["api_ledger"]["attempts"] = 4
        self.rows[("B", self.qids[0])] = row
        summary = analysis.summarize(self.rows, self.qids, {}, self.manifest)
        self.assertFalse(summary["arms"]["B"]["api_usage_complete_for_observed_rows"])
        self.assertEqual(summary["arms"]["B"]["metrics"]["api_attempts_without_usage"]["known_sum"], 1)
        row["api_ledger"]["input_tokens"] = 999
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            analysis.row_metrics(row, "B")

    def test_fresh_first10_and_remaining_reviews_can_be_merged_without_duplicates(self):
        first = {key: row for key, row in self.rows.items() if key[1] == self.qids[0]}
        rest = {key: row for key, row in self.rows.items() if key not in first}
        documents = []
        for selected, seed in ((first, "04" * 32), (rest, "05" * 32)):
            public, private = analysis.export_blind(selected, self.cohort, self.manifest, seed=seed)
            documents.append(analysis.import_reviews(self.submit(public, correct=False), private, self.rows, self.manifest))
        labels = analysis.load_labels(documents, self.rows, self.manifest)
        self.assertEqual(len(labels), 4)
        summary = analysis.summarize(self.rows, self.qids, labels, self.manifest)
        self.assertEqual(summary["paired"]["mcnemar_exact_two_sided_p"], 1)
        self.assertEqual(summary["paired"]["both_wrong"], 2)

    def test_tampered_manifest_and_legacy_rubric_are_rejected(self):
        path = self.root / "reports/strict_qwen35/run_manifest.json"
        for change in ({"version": "v14-hosted-qwen38-first10-v1"}, {"analysis_contract": {}}):
            manifest = {**self.manifest, **change}
            manifest.pop("manifest_sha256")
            manifest["manifest_sha256"] = analysis.digest(manifest)
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                analysis.load_experiment(self.root)

    def test_outputs_are_exclusive_and_mapping_is_private(self):
        path = self.root / "private.json"
        analysis.write_new(path, {"hello": "world"}, private=True)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            analysis.write_new(path, {})


if __name__ == "__main__":
    unittest.main()
