from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hosted_analysis_test", ROOT / "analyze_hosted_pilot.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
COHORT = [json.loads(line) for line in (ROOT / "data/queries_100.jsonl").read_text().splitlines() if line.strip()][:10]


class HostedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "snapshot"
        self.output = self.root / "analysis"
        query_path = self.snapshot / "server/hosted_pilot/queries_10.jsonl"
        query_path.parent.mkdir(parents=True)
        query_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in COHORT))
        cfg = {"model": analysis.MODEL, "base_url": analysis.ENDPOINT, "concurrency": 3,
            "question_count_per_variant": 10, "question_runs": 30, "source_cohort_sha256": analysis.COHORT_SHA,
            "max_turns": 100, "wall_clock_seconds_per_question": 3600, "read_default_limit": 2000,
            "per_call_max_completion_tokens": 32000, "thinking_mode": "enabled", "reasoning_effort": "medium",
            "preserve_thinking": True, "allowed_reported_models": [analysis.MODEL],
            "bm25_k": 1000, "bm25_top_n_preview": 10, "bm25_snippet_chars": 100,
            "document_index": "/index", "filename_map": "/map", "corpus": "/corpus", "passage_index": "/passages",
            "reranker_model": "Qwen/Qwen3-Reranker-0.6B", "reranker_revision": analysis.RERANKER_PIN,
            "reranker_device": "cuda:0", "reranker_dtype": "bfloat16", "reranker_sharing_policy": "shared-fixture"}
        payload = {"version": "v14-hosted-qwen38-first10-v1", "config": cfg,
            "query_ids": list(analysis.QIDS), "variant_order": list(analysis.VARIANTS),
            "question_runs": 30, "queries_sha256": analysis.sha(query_path)}
        self.manifest = {**payload, "manifest_sha256": analysis.digest(payload)}
        analysis.write_json(self.snapshot / "server/hosted_pilot/reports/run_manifest.json", self.manifest)

    def row(self, variant="search_paging", qid=70, *, termination="done"):
        question = next(row for row in COHORT if row["query_id"] == qid)
        extras = {"ablation_variant": variant, "hosted_pilot_version": "v14-hosted-qwen38-first10-v1",
            "hosted_endpoint": analysis.ENDPOINT, "hosted_enable_thinking": True,
            "hosted_preserve_thinking": True, "hosted_reasoning_effort": "medium",
            "hosted_completion_budget_kind": "max_completion_tokens", "hosted_total_completion_tokens": 32000,
            "hosted_allowed_reported_models": [analysis.MODEL],
            "bm25_k": 1000, "bm25_top_n_preview": 10, "bm25_snippet_chars": 100, "read_default_limit": 2000,
            "structured_doc_mode": True, "no_bash": variant == "no_bash", "enable_sandbox": True,
            "reranker_passage_tokens": 128, "reranker_passage_overlap_tokens": 32,
            "index_dir": "/index", "filename_map": "/map", "bc_plus_docs": "/corpus", "space_passage_index": "/passages",
            "reranker_model": "Qwen/Qwen3-Reranker-0.6B", "reranker_revision": analysis.RERANKER_PIN,
            "reranker_device": "cuda:0", "reranker_dtype": "bfloat16", "hosted_reranker_sharing_policy": "shared-fixture"}
        result = {"query_id": qid, "model": analysis.MODEL, "question": question["query"],
            "gold_answer": question["answer"], "final_text": f"Public answer {qid}", "terminated_by": termination,
            "is_correct": False, "agent_cost_usd": 0.0, "elapsed_seconds": 20.0, "n_turns": 2,
            "agent_usage": {"input_tokens": 20, "cached_input_tokens": 80, "output_tokens": 20, "reasoning_tokens": 10},
            "run_config": {"agent_model": analysis.MODEL, "max_model_calls": 100,
                "per_call_max_tokens": 32000, "wall_clock_timeout_sec": 3600, "api_max_retries": 3, "extras": extras},
            "tool_usage": {"search": 1, "read": 1}, "runner_specific": {"n_bm25_searches": 1,
                "n_bm25_queries": 2, "reranker_summary": {"rerank_calls": 0, "page_calls": 0,
                    "scoring_calls": 0, "scored_passages": 0, "input_tokens": 0,
                    "scoring_seconds": 0.0, "preparation_seconds": 0.0,
                    "ablation_search": {"retrieval_calls": 1, "retrieval_queries": 2,
                        "search_page_calls": 0, "tool_errors": 0, "auto_rerank_calls": 0,
                        "auto_rerank_errors": 0, "empty_scopes": 2 if variant == "auto_rerank" else 0}}}}
        result["run_config_hash"] = analysis.run_config_hash(result["run_config"])
        return result

    def save_row(self, row, variant):
        analysis.write_json(self.snapshot / "runs/hosted_qwen38_first10" / variant / "_per_query" / f"qid_{row['query_id']}.json", row)

    def transport(self, qid=70, response_id="response-1", **changes):
        value = {"query_id": qid, "requested_model": analysis.MODEL, "reported_model": analysis.MODEL,
            "thinking_mode": "enabled", "preserve_thinking": True, "reasoning_effort": "medium",
            "requested_max_completion_tokens": 32000, "status": "ok", "response_id": response_id,
            "response_delivered": True, "unix_time": 100.0, "request_seconds": 3.0,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 80}, "completion_tokens_details": {"reasoning_tokens": 10}}}
        value.update(changes)
        return value

    def save_transport(self, rows, variant="search_paging"):
        path = self.snapshot / "runs/hosted_qwen38_first10" / variant / "api_transport.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def reviews(self, records):
        path = self.root / "fresh_reviews.json"
        analysis.write_json(path, {"version": analysis.VERSION, "manifest_sha256": self.manifest["manifest_sha256"], "records": records})
        return path

    def test_partial_snapshot_cost_includes_active_and_undelivered_usage_without_double_counting(self):
        row = self.row()
        row["is_correct"] = True  # Still a placeholder; must never become a label.
        row["tool_usage"]["search"] = 4  # retrieval +two pages +one failed/unknown page.
        row["tool_usage"]["rerank"] = 3
        row["runner_specific"]["reranker_summary"].update(page_calls=1, rerank_calls=3,
            ablation_search={"retrieval_calls": 1, "retrieval_queries": 2,
                "search_page_calls": 2, "auto_rerank_calls": 0, "tool_errors": 1})
        self.save_row(row, "search_paging")
        self.save_transport([self.transport(), self.transport(81, "response-2", response_delivered=False)])
        result = analysis.analyze(self.snapshot, self.output)
        arm = result["arms"]["search_paging"]
        self.assertEqual(result["completed_question_runs"], 1)
        self.assertIsNone(arm["accuracy"])
        self.assertEqual(arm["accuracy_status"], "pending")
        self.assertIsNone(arm["correct_reviewed"])
        self.assertEqual(arm["observed_unfinished_query_ids"], [81])
        all_tokens = arm["api_all_observed"]["known_token_totals"]
        self.assertEqual(all_tokens, {"prompt_tokens": 200, "cached_input_tokens": 160,
            "uncached_input_tokens": 40, "output_tokens": 40, "reasoning_tokens": 20, "total_tokens": 240})
        self.assertEqual(arm["api_for_completed_queries"]["known_token_totals"]["total_tokens"], 120)
        self.assertEqual(arm["completed_row_totals"]["manual_rerank_calls"], 3)
        self.assertEqual(arm["completed_row_totals"]["rerank_page_calls"], 1)
        self.assertEqual(arm["completed_row_totals"]["search_page_calls"], 2)
        self.assertEqual(arm["completed_row_totals"]["search_failed_or_unregistered_calls"], 1)
        self.assertEqual(arm["cost"]["status"], "not_configured")
        self.assertIsNone(arm["cost"]["amount"])
        self.assertNotIn("Public answer", (self.output / "metrics.json").read_text())

    def test_exact30_new_hash_bound_reviews_and_sensitivity_are_separate(self):
        reviews = []
        for variant in analysis.VARIANTS:
            for qid in analysis.QIDS:
                row = self.row(variant, qid)
                if variant == "auto_rerank":
                    row["runner_specific"]["reranker_summary"].update(rerank_calls=2,
                        ablation_search={"retrieval_calls": 1, "retrieval_queries": 2,
                            "search_page_calls": 0, "auto_rerank_calls": 2, "tool_errors": 0,
                            "auto_rerank_errors": 0, "empty_scopes": 0})
                self.save_row(row, variant)
                review = {"variant": variant, "query_id": qid, "final_text_sha256": analysis.final_sha(row["final_text"]),
                    "correct": qid != 116, "reason": "fresh strict review"}
                if qid == 116:
                    review["sensitivity_correct"] = True
                reviews.append(review)
        result = analysis.analyze(self.snapshot, self.output, reviews=self.reviews(reviews))
        self.assertEqual(result["completed_question_runs"], 30)
        for variant in analysis.VARIANTS:
            arm = result["arms"][variant]
            self.assertEqual(arm["accuracy"], 0.9)
            self.assertEqual(arm["sensitivity"]["accuracy"], 1.0)
            self.assertEqual(arm["sensitivity"]["provided_labels"], 1)
            self.assertEqual(arm["api_all_observed"]["status"], "not_downloaded_or_not_started")
        self.assertEqual(result["arms"]["auto_rerank"]["completed_row_totals"]["auto_rerank_calls"], 20)
        self.assertNotIn("historical_v14", result)

    def test_stale_answer_sha_and_wrong_cohort_or_variant_are_rejected(self):
        row = self.row()
        self.save_row(row, "search_paging")
        reviews = self.reviews([{"variant": "search_paging", "query_id": 70,
            "final_text_sha256": "stale", "correct": True, "reason": "review"}])
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            analysis.analyze(self.snapshot, self.output, reviews=reviews)
        row["run_config"]["extras"]["ablation_variant"] = "auto_rerank"
        self.save_row(row, "search_paging")
        with self.assertRaisesRegex(ValueError, "run_config_hash differs"):
            analysis.analyze(self.snapshot, self.output)

    def test_missing_usage_and_failed_api_are_not_free_or_completed_answers(self):
        row = self.row(termination="api_error: unavailable")
        self.save_row(row, "search_paging")
        self.save_transport([self.transport(status="error", response_id=None, usage=None, response_delivered=False)])
        pricing = self.root / "pricing.json"
        analysis.write_json(pricing, {"currency": "TEST", "uncached_input_per_million": 1,
            "cached_input_per_million": 0.5, "output_per_million": 2})
        result = analysis.analyze(self.snapshot, self.output, pricing=pricing)
        arm = result["arms"]["search_paging"]
        self.assertEqual(arm["completed"], 0)
        self.assertEqual(arm["infrastructure_query_ids"], [70])
        self.assertEqual(arm["api_all_observed"]["requests_without_usage"], 1)
        self.assertIsNone(arm["cost"]["amount"])
        self.assertEqual(arm["cost"]["status"], "partial_known_usage")
        self.assertIsNone(arm["accuracy"])

    def test_duplicate_response_ids_across_arms_fail_instead_of_double_billing(self):
        self.save_transport([self.transport()])
        self.save_transport([self.transport()], "auto_rerank")
        with self.assertRaisesRegex(ValueError, "Duplicate API response"):
            analysis.analyze(self.snapshot, self.output)

    def test_pricing_uses_uncached_cached_and_output_only(self):
        self.save_transport([self.transport()])
        pricing = self.root / "pricing.json"
        analysis.write_json(pricing, {"currency": "TEST", "uncached_input_per_million": 1,
            "cached_input_per_million": 0.5, "output_per_million": 2})
        result = analysis.analyze(self.snapshot, self.output, pricing=pricing)
        # 20 uncached +80 cached*0.5 +20 output*2; reasoning is already output.
        self.assertAlmostEqual(result["arms"]["search_paging"]["cost"]["amount"], 100 / 1_000_000)

    def test_resource_counter_and_partial_tail_failures_are_explicit(self):
        original = self.row()
        for change, message in (("hash", "run_config_hash"), ("bm25", "wrong frozen"),
                                ("resource", "resource identity"), ("missing_counter", "counter is absent"),
                                ("counter_mismatch", "counters disagree")):
            with self.subTest(change=change):
                row = copy.deepcopy(original)
                if change == "hash":
                    row["run_config_hash"] = "wrong"
                elif change == "bm25":
                    row["run_config"]["extras"]["bm25_k"] = 100
                    row["run_config_hash"] = analysis.run_config_hash(row["run_config"])
                elif change == "resource":
                    row["run_config"]["extras"]["index_dir"] = "/other-index"
                    row["run_config_hash"] = analysis.run_config_hash(row["run_config"])
                elif change == "missing_counter":
                    del row["runner_specific"]["reranker_summary"]["scoring_calls"]
                else:
                    row["runner_specific"]["n_bm25_searches"] = 2
                self.save_row(row, "search_paging")
                with self.assertRaisesRegex(ValueError, message):
                    analysis.analyze(self.snapshot, self.output)
        self.save_row(original, "search_paging")
        self.save_transport([self.transport()])
        path = self.snapshot / "runs/hosted_qwen38_first10/search_paging/api_transport.jsonl"
        with path.open("a") as stream:
            stream.write('{"query_id":')
        with self.assertRaisesRegex(ValueError, "尾行不完整，请重新同步"):
            analysis.analyze(self.snapshot, self.output)
        no_bash = self.row("no_bash")
        no_bash["runner_specific"]["reranker_summary"]["ablation_search"] = {}
        no_bash["tool_usage"]["bash"] = 1  # Unknown-tool attempt is retained, not executed.
        analysis.validate_row(no_bash, "no_bash", {row["query_id"]: row for row in COHORT}, self.manifest["config"])
        self.assertEqual(analysis.row_metrics(no_bash)["forbidden_tool_name_attempts"], 1)
        different_queries = self.row()
        different_queries["runner_specific"]["n_bm25_queries"] = 1
        with self.assertRaisesRegex(ValueError, "counters disagree"):
            analysis.row_metrics(different_queries)
        automatic = self.row("auto_rerank")
        automatic["tool_usage"]["rerank"] = 1  # Rejected guess of an unregistered tool.
        analysis.validate_row(automatic, "auto_rerank", {row["query_id"]: row for row in COHORT}, self.manifest["config"])
        self.assertEqual(analysis.row_metrics(automatic)["forbidden_tool_name_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
