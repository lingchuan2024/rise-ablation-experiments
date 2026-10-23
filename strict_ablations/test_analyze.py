"""Synthetic 400-final coverage, blinding, ledger and exclusive-output tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from strict_ablations import analyze as analysis


class FakeSuite:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.validated, self.artifacts = [], []

    def verify_manifest(self, *, resources):
        if resources is not False:
            raise AssertionError("Analysis must not inspect live resources")
        return copy.deepcopy(self.manifest)

    def terminal_path(self, arm, qid):
        return self.root / "runs" / arm / "_per_query" / f"qid_{qid}.json"

    def validate_terminal(self, row, manifest, arm, query):
        analysis.shared.checked_digest(row, "row_sha256", "Terminal row")
        if (row["arm"] != arm or row["query_id"] != query["query_id"]
                or row["manifest_sha256"] != manifest["manifest_sha256"]
                or row["config_sha256"] != manifest["config_sha256"]
                or row["contract_sha256"] != manifest["contracts"][arm]["contract_sha256"]):
            raise ValueError("Terminal identity differs")
        if row["terminated_by"] not in {"done", "max_turns", "wall_clock_timeout"}:
            raise ValueError("Not a natural terminal")
        self.validated.append((arm, query["query_id"]))

    def verify_execution_artifacts(self, manifest, arm, query, row):
        if row.get("fixture_artifacts_match") is not True:
            raise ValueError("Execution artifact receipts differ")
        self.artifacts.append((arm, query["query_id"]))


class FourArmAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        source = analysis.ROOT / "data/queries_100.jsonl"
        (self.root / "data/queries_100.jsonl").write_bytes(source.read_bytes())
        self.cohort = {q["query_id"]: q for q in
            [json.loads(line) for line in source.read_text().splitlines() if line.strip()]}
        self.qids = list(self.cohort)
        config = {"model": analysis.MODEL, "max_turns": 100}
        reference = {"version": "pinned-public-control-fixture", "manifest_sha256": "11" * 32,
            "public_export_sha256": "22" * 32, "integrity_receipt_sha256": "33" * 32,
            "contract_sha256": "44" * 32, "config_sha256": analysis.digest(config),
            "model": analysis.MODEL, "cohort_sha256": analysis.COHORT_SHA256}
        self.manifest = {"config": config, "config_sha256": analysis.digest(config),
            "contracts": {arm: {"contract_sha256": analysis.digest(arm)} for arm in analysis.ARMS},
            "control_reference": reference, "control_root": ".",
            "cohort_sha256": analysis.COHORT_SHA256, "query_ids": self.qids,
            "analysis_contract": {"version": analysis.VERSION,
                                  "rubric_sha256": analysis.digest(analysis.rubric_document())}}
        self.manifest["manifest_sha256"] = analysis.digest(self.manifest)
        self.suite = FakeSuite(self.root, self.manifest)
        self.suite.cohort_path = self.root / "data/queries_100.jsonl"
        self.bundle = {**reference, "records": [], "common_config": copy.deepcopy(config)}
        for index, (qid, query) in enumerate(self.cohort.items()):
            control = self.row(analysis.CONTROL, qid, index)
            self.bundle["records"].append({"query_id": qid, "question": query["query"],
                "gold_answer": query["answer"], "final_text": control["final_text"],
                "final_sha256": control["final_sha256"], "terminal_row_sha256": control["row_sha256"],
                "terminated_by": control["terminated_by"],
                "metrics": analysis.shared.row_metrics(control, "B"),
                "reranker_summary": control["reranker_summary"]})
            for arm in analysis.ARMS:
                analysis.write_new(self.suite.terminal_path(arm, qid), self.row(arm, qid, index))
        self.gate = patch.object(analysis, "verify_control_bundle", return_value=self.bundle)
        self.mock_gate = self.gate.start()
        self.addCleanup(self.gate.stop)

    def row(self, arm, qid, index):
        auto, paging = arm == "auto_rerank", arm == "search_paging"
        calls = 1 if arm == analysis.CONTROL else 2
        scoring = 2 if auto else 1
        text = "" if index == 0 else "Exact Answer: fixture public final"
        row = {"arm": arm, "query_id": qid, "model": analysis.MODEL,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "config_sha256": self.manifest["config_sha256"],
            "contract_sha256": self.manifest["control_reference"]["contract_sha256"] if arm == analysis.CONTROL
                                else self.manifest["contracts"][arm]["contract_sha256"],
            "final_text": text, "terminated_by": "wall_clock_timeout" if not text else "done",
            "n_turns": 3, "elapsed_seconds": 20 + index, "input_tokens": 12,
            "raw_non_cached_input_tokens": 10, "cached_input_tokens": 2,
            "output_tokens": 8, "reasoning_tokens": 4,
            "api_ledger": {"attempts": 3, "successful_calls": 3, "attempts_without_usage": 0,
                "input_tokens": 12, "output_tokens": 8, "agent_totals_match": True},
            "reranker_summary": {"rerank_calls": calls, "scoring_calls": scoring,
                "scored_passages": 10 * scoring, "input_tokens": 100 * scoring,
                "scoring_seconds": 2 * scoring, "preparation_seconds": 3,
                "page_calls": 0 if auto or arm == analysis.CONTROL else 1},
            "ablation_summary": {"arm": arm, "independent_rerank_calls": 0 if auto else calls,
                "independent_rerank_page_calls": 0 if auto or arm == analysis.CONTROL else 1,
                "automatic_rerank_calls": calls if auto else 0, "search_page_calls": int(paging),
                "retrieval_calls": 1, "retrieval_queries": 2, "search_invocations": 1 + int(paging),
                "automatic_rerank_errors": 0, "automatic_empty_scopes": 0},
            "fixture_artifacts_match": True,
            # These deliberately contradictory placeholders must never be read.
            "is_correct": True, "judge": {"is_correct": True}}
        return self.seal(row)

    def seal(self, row):
        row = copy.deepcopy(row)
        row.pop("row_sha256", None)
        row["final_sha256"] = analysis.text_sha(row["final_text"])
        row["row_sha256"] = analysis.digest(row)
        return row

    def collected(self):
        return analysis.collect(self.root, suite=self.suite)

    def submission(self, packet, private):
        mapping = {r["case_id"]: r for r in private["records"]}
        result = {k: copy.deepcopy(v) for k, v in packet.items()
                  if k not in {"rubric", "review_template", "records"}}
        result["reviewer"] = "fresh independent fixture reviewer"
        result["records"] = []
        for case in packet["records"]:
            mapped = mapping[case["case_id"]]
            arm, index = mapped["arm"], self.qids.index(mapped["query_id"])
            threshold = {analysis.CONTROL: 50, "search_paging": 55, "auto_rerank": 45, "no_bash": 0}[arm]
            correct = bool(case["final_text"]) and (index < threshold if arm != "no_bash" else 25 <= index < 75)
            result["records"].append({"case_id": case["case_id"], "final_sha256": case["final_sha256"],
                "correct": correct, "reason": "Fresh comparison of full final to frozen gold."})
        return result

    def scored(self):
        rows, cohort, manifest, provenance = self.collected()
        packet, private = analysis.export_blind(rows, cohort, manifest, seed="01" * 32)
        reviews = analysis.import_reviews(self.submission(packet, private), private, rows, manifest)
        labels = analysis.load_labels([reviews], rows, manifest)
        return rows, cohort, manifest, packet, private, reviews, labels

    def test_collects_declared_finals_and_delegates_300_integrity_checks(self):
        poison = self.root / "runs/search_paging/_traces/do-not-open.json"
        poison.parent.mkdir(parents=True)
        poison.write_text("not JSON; this untracked trajectory must not be discovered")
        original_read = Path.read_text
        def guard(path, *args, **kwargs):
            if path == poison:
                raise AssertionError("Analysis scanned an untracked trajectory")
            return original_read(path, *args, **kwargs)
        with patch.object(Path, "read_text", guard):
            rows, cohort, manifest, provenance = self.collected()
        self.assertEqual(len(rows), 400)
        self.assertEqual(len(self.suite.validated), 300)
        self.assertEqual(len(self.suite.artifacts), 300)
        self.assertEqual(self.mock_gate.call_args.kwargs["common_config"], manifest["config"])
        self.assertEqual(self.mock_gate.call_args.args[0], self.root)
        self.assertEqual(len(cohort), 100)
        self.assertEqual(provenance["control_reference_sha256"], analysis.digest(manifest["control_reference"]))
        for row in rows.values():
            self.assertNotIn("judge", row)
            self.assertNotIn("is_correct", row)
        first = self.bundle["records"][0]
        self.assertEqual(rows[(analysis.CONTROL, first["query_id"])]["row_sha256"], first["terminal_row_sha256"])

    def test_artifact_root_does_not_need_a_dataset_copy(self):
        del self.suite.cohort_path
        (self.root / "data/queries_100.jsonl").unlink()
        (self.root / "data").rmdir()
        rows, cohort, _, provenance = self.collected()
        self.assertEqual(len(rows), 400)
        self.assertEqual(list(cohort), self.qids)
        self.assertEqual(provenance["cohort_file_sha256"], analysis.COHORT_SHA256)

    def test_injected_cohort_source_still_requires_exact_frozen_bytes(self):
        fixture = self.root / "fixture_cohort.jsonl"
        fixture.write_bytes((self.root / "data/queries_100.jsonl").read_bytes())
        manifest, cohort, _ = analysis.load_experiment(self.root, suite=self.suite, cohort_path=fixture)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(list(cohort), self.qids)
        fixture.write_text(fixture.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            analysis.load_experiment(self.root, suite=self.suite, cohort_path=fixture)

    def test_missing_extra_and_duplicate_control_coverage_rejected(self):
        path = self.suite.terminal_path("no_bash", self.qids[-1])
        contents = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.collected()
        path.write_bytes(contents)
        extra = path.parent / "qid_duplicate.json"
        extra.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.collected()
        extra.unlink()
        self.bundle["records"][-1] = copy.deepcopy(self.bundle["records"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.collected()

    def test_bad_terminal_artifact_and_stale_control_binding_fail(self):
        path = self.suite.terminal_path("auto_rerank", self.qids[1])
        row = analysis.load(path)
        row["fixture_artifacts_match"] = False
        path.write_text(json.dumps(self.seal(row)))
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.collected()
        row["fixture_artifacts_match"] = True
        path.write_text(json.dumps(self.seal(row)))
        self.bundle["public_export_sha256"] = "ff" * 32
        with self.assertRaisesRegex(ValueError, "control reference"):
            self.collected()

    def test_rubric_freeze_and_fixed100_are_required(self):
        self.assertNotEqual(analysis.VERSION, analysis.shared.VERSION)
        self.assertIn("all anonymous answers", " ".join(analysis.rubric_document()["rules"]))
        changed = copy.deepcopy(self.manifest)
        changed["analysis_contract"]["rubric_sha256"] = "ff" * 32
        changed.pop("manifest_sha256")
        changed["manifest_sha256"] = analysis.digest(changed)
        self.suite.manifest = changed
        with self.assertRaisesRegex(ValueError, "rubric"):
            self.collected()

    def test_anonymous_export_has_all_finals_no_group_or_judge_fields(self):
        rows, cohort, manifest, _ = self.collected()
        packet, private = analysis.export_blind(rows, cohort, manifest, seed="02" * 32)
        repeat, _ = analysis.export_blind(rows, cohort, manifest, seed="02" * 32)
        self.assertEqual(packet, repeat)
        self.assertEqual(len(packet["records"]), 400)
        self.assertEqual(len({r["case_id"] for r in packet["records"]}), 400)
        for case in packet["records"]:
            self.assertEqual(set(case), {"case_id", "question", "gold_answer", "final_text", "final_sha256"})
        self.assertNotIn("shuffle_seed_secret", packet)
        self.assertEqual(sum(not r["final_text"] for r in packet["records"]), 4)
        self.assertEqual(len(private["records"]), 400)
        rows.pop(("no_bash", self.qids[-1]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            analysis.export_blind(rows, cohort, manifest)

    def test_import_requires_complete_fresh_boolean_hash_bound_reviews(self):
        rows, cohort, manifest, _ = self.collected()
        packet, private = analysis.export_blind(rows, cohort, manifest, seed="03" * 32)
        submission = self.submission(packet, private)
        for field, value in (("version", analysis.shared.VERSION), ("packet_sha256", "ff" * 32),
                             ("rubric_sha256", "ff" * 32), ("reviewer", "")):
            changed = {**submission, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                analysis.import_reviews(changed, private, rows, manifest)
        for change in (lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
                       lambda r: r[0].update(correct=1), lambda r: r[0].update(reason=""),
                       lambda r: r[0].update(final_sha256="ff" * 32)):
            changed = copy.deepcopy(submission)
            change(changed["records"])
            with self.assertRaises(ValueError):
                analysis.import_reviews(changed, private, rows, manifest)

    def test_mapping_binds_terminal_sha_and_public_question_payload(self):
        rows, cohort, manifest, _ = self.collected()
        packet, private = analysis.export_blind(rows, cohort, manifest, seed="04" * 32)
        submission = self.submission(packet, private)
        first_key = next(iter(rows))
        changed = copy.deepcopy(rows)
        changed[first_key]["row_sha256"] = "ee" * 32
        with self.assertRaisesRegex(ValueError, "stale"):
            analysis.import_reviews(submission, private, changed, manifest)
        changed = copy.deepcopy(rows)
        changed[first_key]["question"] += "changed"
        with self.assertRaisesRegex(ValueError, "payload"):
            analysis.import_reviews(submission, private, changed, manifest)
        changed_private = copy.deepcopy(private)
        changed_private["records"][-1] = copy.deepcopy(changed_private["records"][0])
        changed_private.pop("mapping_sha256")
        changed_private["mapping_sha256"] = analysis.digest(changed_private)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.import_reviews(submission, changed_private, rows, manifest)

    def test_empty_natural_failures_stay_in_denominator(self):
        rows, cohort, manifest, _ = self.collected()
        packet, private = analysis.export_blind(rows, cohort, manifest, seed="05" * 32)
        submission = self.submission(packet, private)
        empty_id = next(r["case_id"] for r in packet["records"] if not r["final_text"])
        next(r for r in submission["records"] if r["case_id"] == empty_id)["correct"] = True
        with self.assertRaisesRegex(ValueError, "empty"):
            analysis.import_reviews(submission, private, rows, manifest)
        rows, cohort, manifest, packet, private, reviews, labels = self.scored()
        summary = analysis.summarize(rows, list(cohort), labels, manifest)
        for arm in analysis.GROUPS:
            self.assertEqual(summary["arms"][arm]["n"], 100)
            self.assertEqual(summary["arms"][arm]["empty_public_finals"], 1)
        self.assertEqual(summary["arms"][analysis.CONTROL]["correct"], 49)

    def test_summary_reconciles_calls_tokens_and_unadjusted_pairwise_tests(self):
        rows, cohort, manifest, packet, private, reviews, labels = self.scored()
        summary = analysis.summarize(rows, list(cohort), labels, manifest)
        paging = summary["paired_vs_control"]["search_paging"]
        self.assertEqual(paging["net_correct_arm_minus_control"], 5)
        self.assertEqual(paging["arm_only_correct"], 5)
        self.assertEqual(paging["control_only_correct"], 0)
        self.assertEqual(paging["mcnemar_exact_two_sided_p_unadjusted"], .0625)
        auto = summary["paired_vs_control"]["auto_rerank"]
        self.assertEqual(auto["net_correct_arm_minus_control"], -5)
        self.assertEqual(auto["mcnemar_exact_two_sided_p_unadjusted"], .0625)
        self.assertEqual(len(summary["paired_vs_control"]), 3)
        for arm in analysis.GROUPS:
            metrics = summary["arms"][arm]["metrics"]
            self.assertEqual(metrics["model_total_tokens"]["known_sum"], 2000)
            self.assertEqual(metrics["combined_model_and_reranker_input_tokens"]["known_sum"],
                             21200 if arm == "auto_rerank" else 11200)
        auto_metrics = summary["arms"]["auto_rerank"]["metrics"]
        self.assertEqual(auto_metrics["manual_rerank_calls"]["known_sum"], 0)
        self.assertEqual(auto_metrics["automatic_rerank_calls"]["known_sum"], 200)
        self.assertIsNone(summary["arms"][analysis.CONTROL]["metrics"]["retrieval_calls"]["known_sum"])
        self.assertIn("historical", " ".join(summary["limitations"]))
        self.assertIn("strategy package", " ".join(summary["limitations"]))
        self.assertIn("without multiplicity adjustment", " ".join(summary["limitations"]))
        analysis.shared.checked_digest(summary, "summary_sha256", "Summary")

    def test_old_or_partial_reviews_cannot_fill_missing_fresh_labels(self):
        rows, cohort, manifest, packet, private, reviews, labels = self.scored()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analysis.load_labels([reviews, reviews], rows, manifest)
        old = copy.deepcopy(reviews)
        old["version"] = analysis.shared.VERSION
        old.pop("reviews_sha256")
        old["reviews_sha256"] = analysis.digest(old)
        with self.assertRaisesRegex(ValueError, "fresh reviews"):
            analysis.load_labels([old], rows, manifest)
        partial = copy.deepcopy(reviews)
        partial["records"].pop()
        partial.pop("reviews_sha256")
        partial["reviews_sha256"] = analysis.digest(partial)
        with self.assertRaisesRegex(ValueError, "400"):
            analysis.load_labels([partial], rows, manifest)
        labels.pop(next(iter(labels)))
        with self.assertRaisesRegex(ValueError, "400"):
            analysis.summarize(rows, list(cohort), labels, manifest)

    def test_malformed_counter_or_ledger_never_becomes_zero(self):
        row = self.row("auto_rerank", self.qids[1], 1)
        row["ablation_summary"]["automatic_rerank_calls"] = None
        with self.assertRaisesRegex(ValueError, "metric"):
            analysis._new_metrics(row, "auto_rerank")
        row = self.row("no_bash", self.qids[1], 1)
        row["api_ledger"]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            analysis._new_metrics(row, "no_bash")
        row = self.row("search_paging", self.qids[1], 1)
        row["api_ledger"]["attempts"] += 1
        row["api_ledger"]["attempts_without_usage"] = 1
        self.assertEqual(analysis._new_metrics(row, "search_paging")["api_attempts_without_usage"], 1)

    def test_cli_outputs_are_exclusive_and_mapping_private(self):
        collected = self.collected()
        packet_path, private_path = self.root / "packet.json", self.root / "map.json"
        argv = ["export", "--out", str(packet_path), "--private-map", str(private_path)]
        with patch.object(analysis, "collect", return_value=collected):
            analysis.main(argv)
        self.assertEqual(stat.S_IMODE(private_path.stat().st_mode), 0o600)
        before = packet_path.read_bytes()
        with patch.object(analysis, "collect", side_effect=AssertionError("Must reject before collection")):
            with self.assertRaisesRegex(ValueError, "overwriting"):
                analysis.main(argv)
        self.assertEqual(packet_path.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            analysis.write_new(packet_path, {"should": "never overwrite"})


if __name__ == "__main__":
    unittest.main()
