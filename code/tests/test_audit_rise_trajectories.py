"""Offline checks for the read-only RISE trajectory audit."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_rise_trajectories import audit_run, main


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "run"
    queries = root / "queries.jsonl"
    queries.write_text(
        json.dumps({
            "query_id": 1, "query": "Where?", "answer": "Paris",
            "gold_doc_ids": ["42"],
        }) + "\n", encoding="utf-8",
    )
    filename_map = root / "map.json"
    _write_json(filename_map, {"relpath_to_docid": {"a.txt": "42"}})
    _write_json(run_dir / "_per_query" / "qid_1.json", {
        "query_id": 1, "question": "Where?", "gold_answer": "Paris",
        "gold_doc_ids": ["42"], "final_text": "Exact Answer: Paris",
        "terminated_by": "done", "is_correct": True,
        "judge": {
            "correct": "yes", "model": "judge-model",
            "reasoning": "Short-circuit: exact match",
        },
    })
    _write_json(run_dir / "_traces" / "qid_1" / "single.json", {
        "final_text": "Exact Answer: Paris",
        "turns": [{"finish_reason": "stop", "tool_calls": [{
            "name": "read", "args": {"file_path": "a.txt"},
            "result_paths": ["a.txt"], "result_unresolved_paths": [],
            "result_chars": 11, "result_text": "Useful text",
            "result_text_preview": "Useful text", "error": "",
        }]}],
    })
    return run_dir, queries, filename_map


class AuditRiseTrajectoriesTest(unittest.TestCase):
    def test_clean_shortcut_counts_without_findings_or_run_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))
            report = audit_run(run_dir, queries, filename_map)
            after = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))

            self.assertEqual(report["summary"]["errors"], 0)
            self.assertEqual(report["summary"]["warnings"], 0)
            self.assertEqual(report["summary"]["missing_full_tool_outputs"], 0)
            self.assertEqual(report["summary"]["exact_answer_shortcuts"], 1)
            self.assertEqual(before, after)

    def test_flags_fixed_query_judge_and_trace_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            row_path = run_dir / "_per_query" / "qid_1.json"
            row = json.loads(row_path.read_text())
            row.update({
                "question": "Changed", "gold_answer": "London",
                "gold_doc_ids": ["99"], "is_correct": False,
            })
            row["judge"]["reasoning"] = ""
            row["judge"]["model"] = ""
            _write_json(row_path, row)
            report = audit_run(run_dir, queries, filename_map)
            codes = {finding["code"] for finding in report["per_query"][0]["findings"]}

            self.assertGreater(report["summary"]["errors"], 0)
            self.assertTrue({
                "question_mismatch", "gold_answer_mismatch", "gold_doc_ids_mismatch",
                "judge_correct_mismatch", "judge_reasoning_missing", "judge_model_missing",
            }.issubset(codes))

    def test_flags_preview_error_unresolved_path_repeat_read_and_done_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            trace = json.loads(trace_path.read_text())
            bad_call = {
                "name": "read", "args": {"file_path": "missing.txt"},
                "result_paths": ["missing.txt"], "result_unresolved_paths": [],
                "result_text_preview": "Error: file not found", "error": "",
            }
            trace["turns"] = [
                {"finish_reason": "tool_calls", "tool_calls": [bad_call]},
                {"finish_reason": "tool_calls", "tool_calls": [bad_call]},
                {"finish_reason": "length", "tool_calls": [bad_call]},
            ]
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            item = report["per_query"][0]
            codes = {finding["code"] for finding in item["findings"]}

            self.assertEqual(item["error_count"], 0)
            self.assertEqual(item["stats"]["tool_error_outputs"], 3)
            self.assertEqual(item["stats"]["unresolved_paths"], 3)
            self.assertEqual(item["stats"]["repeated_read_paths"], {"missing.txt": 3})
            self.assertEqual(item["stats"]["repeated_read_requests"][0]["count"], 3)
            self.assertTrue({
                "tool_error_output", "unresolved_paths",
                "repeated_read_request", "done_after_length",
            }.issubset(codes))
            self.assertEqual(main([
                "--run-dir", str(run_dir), "--queries", str(queries),
                "--filename-map", str(filename_map), "--out", str(Path(tmp) / "audit.json"),
            ]), 0)

    def test_missing_row_and_trace_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            (run_dir / "_per_query" / "qid_1.json").unlink()
            (run_dir / "_traces" / "qid_1" / "single.json").unlink()
            report = audit_run(run_dir, queries, filename_map)
            self.assertEqual(report["summary"]["queries_with_errors"], 1)
            self.assertEqual(report["summary"]["errors"], 2)
            self.assertEqual(main([
                "--run-dir", str(run_dir), "--queries", str(queries),
                "--out", str(Path(tmp) / "audit.json"),
            ]), 1)

    def test_counts_historical_tool_result_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            trace = json.loads(trace_path.read_text())
            call = trace["turns"][0]["tool_calls"][0]
            call["result_text"] = ""
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            self.assertEqual(report["summary"]["missing_full_tool_outputs"], 1)
            self.assertEqual(report["per_query"][0]["stats"]["missing_full_tool_outputs"], 1)

    def test_distinct_read_offsets_do_not_warn_as_duplicate_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            trace = json.loads(trace_path.read_text())
            base_call = trace["turns"][0]["tool_calls"][0]
            trace["turns"][0]["tool_calls"] = [
                {**base_call, "args": {"file_path": "a.txt", "offset": offset, "limit": 10}}
                for offset in (0, 10, 20)
            ]
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            item = report["per_query"][0]
            self.assertEqual(item["stats"]["repeated_read_paths"], {"a.txt": 3})
            self.assertEqual(item["stats"]["repeated_read_requests"], [])
            self.assertEqual(item["warning_count"], 0)

    def test_repeated_failed_bash_request_is_reported_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            trace = json.loads(trace_path.read_text())
            repeated = {
                "name": "bash", "args": {"command": "grep missing a.txt"},
                "result_chars": 18, "result_text": "(no matches found)",
                "result_text_preview": "(no matches found)", "error": "",
            }
            trace["turns"] = [
                {"finish_reason": "tool_calls", "tool_calls": [repeated]}
                for _ in range(4)
            ]
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            item = report["per_query"][0]
            self.assertEqual(item["stats"]["repeated_tool_requests"], [
                {"tool": "bash", "args": {"command": "grep missing a.txt"}, "count": 4},
            ])
            self.assertEqual(report["summary"]["queries_with_repeated_tool_requests"], 1)
            self.assertEqual(report["summary"]["repeated_tool_call_excess"], 3)
            self.assertEqual(
                [f["code"] for f in item["findings"]], ["repeated_tool_request"]
            )

    def test_large_single_turn_tool_call_burst_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            trace = json.loads(trace_path.read_text())
            trace["turns"][0]["tool_calls"] = [
                {
                    "name": "bash", "args": {"command": f"grep word-{i} a.txt"},
                    "result_text": "(no matches found)", "result_chars": 18,
                    "result_text_preview": "(no matches found)", "error": "",
                }
                for i in range(20)
            ]
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            item = report["per_query"][0]
            self.assertEqual(item["stats"]["max_tool_calls_in_turn"], 20)
            self.assertEqual(report["summary"]["queries_with_tool_call_bursts"], 1)
            self.assertEqual(item["findings"][0]["code"], "tool_call_burst")
            self.assertEqual(item["findings"][0]["turn"], 0)

    def test_done_without_required_answer_line_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            row_path = run_dir / "_per_query" / "qid_1.json"
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            row = json.loads(row_path.read_text())
            trace = json.loads(trace_path.read_text())
            row["final_text"] = trace["final_text"] = "Still thinking about the answer."
            _write_json(row_path, row)
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            self.assertEqual(report["summary"]["done_without_exact_answer"], 1)
            self.assertIn(
                "done_without_exact_answer",
                {f["code"] for f in report["per_query"][0]["findings"]},
            )

    def test_length_limited_turn_with_tool_calls_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, queries, filename_map = _fixture(Path(tmp))
            row_path = run_dir / "_per_query" / "qid_1.json"
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            row = json.loads(row_path.read_text())
            trace = json.loads(trace_path.read_text())
            row["terminated_by"] = "max_turns"
            trace["turns"][0]["finish_reason"] = "length"
            _write_json(row_path, row)
            _write_json(trace_path, trace)
            report = audit_run(run_dir, queries, filename_map)
            self.assertEqual(report["summary"]["length_with_tool_calls"], 1)
            finding = next(
                f for f in report["per_query"][0]["findings"]
                if f["code"] == "length_with_tool_calls"
            )
            self.assertEqual(finding["turn"], 0)


if __name__ == "__main__":
    unittest.main()
