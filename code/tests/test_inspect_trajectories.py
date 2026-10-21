"""Synthetic, offline fixtures for the run_rise trajectory inspector."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.inspect_trajectories import inspect_run, main


def _write_pair(run_dir: Path, qid: int, *, mismatch: bool = False) -> None:
    question = f"What happened for question {qid}?"
    call = {
        "name": "search", "args": {"queries": ["history archive"]}, "args_raw": "",
        "result_docids": ["42"], "result_paths": ["history/example.txt"],
        "result_unresolved_paths": [], "result_summary": "search (1 query): 1 doc added",
        "result_chars": 77, "result_text": "", "result_text_preview": "1. [4.2] history/example.txt",
        "result_truncated": False, "duration_seconds": 0.25, "error": "",
    }
    turns = [
        {
            "turn": 0, "finish_reason": "tool_calls", "content": "I will search.",
            "reasoning_content": "Need one document.",
            "tokens": {"prompt": 9, "cached_prompt": 1, "completion": 3, "reasoning": 1},
            "cost_usd": 0.01, "tool_calls": [call],
        },
        {
            "turn": 1, "finish_reason": "stop", "content": "Exact Answer: yes",
            "reasoning_content": "", "tokens": {"prompt": 12, "cached_prompt": 2,
                                                "completion": 4, "reasoning": 0},
            "cost_usd": 0.02, "tool_calls": [],
        },
    ]
    trace = {
        "schema_version": "1.1", "model": "test-model", "system_prompt": "system",
        "user_prompt": f"QUESTION:\n{question}", "final_text": "Exact Answer: yes",
        "terminated_by": "done", "error": "", "elapsed_seconds": 1.2,
        "totals": {
            "n_turns": 2, "tool_call_count": 1, "tool_call_breakdown": {"search": 1},
            "prompt_tokens": 21, "cached_prompt_tokens": 3,
            "completion_tokens": 7, "reasoning_tokens": 1, "agent_cost_usd": 0.03,
        },
        "turns": turns,
    }
    row = {
        "schema_version": "1.1", "protocol_version": "1", "runner": "rise",
        "runner_mode": "doc", "model": "test-model", "run_config": {},
        "run_config_hash": "deadbeef1234", "query_id": qid, "question": question,
        "gold_answer": "yes", "gold_doc_ids": ["42"], "evidence_doc_ids": ["42"],
        "n_gold": 1, "n_evidence": 1, "final_text": "Exact Answer: yes",
        "judge": {"model": ""}, "is_correct": False,
        "n_surfaced": 1, "surfaced_relpaths": ["history/example.txt"],
        "all_retrieved": ["42"], "coverage_any": 1.0, "coverage_mean": 1.0,
        "coverage_all": 1.0, "evidence_coverage_mean": 1.0,
        "terminated_by": "done", "elapsed_seconds": 1.2, "n_turns": 2,
        "n_attempts": 1, "max_calls_hit": False, "wall_clock_hit": False,
        "tool_usage": {"search": 1}, "runner_specific": {},
        "agent_usage": {"input_tokens": 21, "cached_input_tokens": 3,
                        "output_tokens": 7, "reasoning_tokens": 1},
        "agent_cost_usd": 0.031 if mismatch else 0.03,
        "discarded_attempt_audit": {"tokens": 0, "cost_usd": 0},
        "judge_usage": {}, "judge_cost_usd": 0.0,
        "total_cost_usd": 0.031 if mismatch else 0.03,
        "trace_path": f"_traces/qid_{qid}/single.json",
    }
    row_path = run_dir / "_per_query" / f"qid_{qid}.json"
    trace_path = run_dir / "_traces" / f"qid_{qid}" / "single.json"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(row), encoding="utf-8")
    trace_path.write_text(json.dumps(trace), encoding="utf-8")


class InspectTrajectoriesTest(unittest.TestCase):
    def test_context_limit_with_trace_error_is_a_valid_terminal_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_pair(run_dir, 1)
            row_path = run_dir / "_per_query" / "qid_1.json"
            trace_path = run_dir / "_traces" / "qid_1" / "single.json"
            row = json.loads(row_path.read_text(encoding="utf-8"))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            row["terminated_by"] = trace["terminated_by"] = "context_limit"
            row["final_text"] = trace["final_text"] = ""
            trace["error"] = "BadRequestError: maximum context length exceeded"
            row_path.write_text(json.dumps(row), encoding="utf-8")
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            report = inspect_run(run_dir, root / "report", expected_count=1)

            self.assertTrue(report["complete"])
            self.assertEqual(report["issues"], [])

    def test_complete_run_writes_timelines_and_integrity_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_pair(run_dir, 1)
            _write_pair(run_dir, 2)
            queries = root / "queries.jsonl"
            queries.write_text('{"query_id": 1}\n{"query_id": 2}\n', encoding="utf-8")
            out_dir = root / "report"

            report = inspect_run(run_dir, out_dir, queries=queries, expected_count=2)

            self.assertTrue(report["complete"])
            self.assertEqual(report["paired_count"], 2)
            self.assertEqual(report["judge_pending_count"], 2)
            timelines = [json.loads(line) for line in (out_dir / "timelines.jsonl").read_text().splitlines()]
            self.assertEqual(len(timelines), 2)
            self.assertEqual(timelines[0]["timeline"][0]["tool_calls"][0]["result_paths"]["items"],
                             ["history/example.txt"])
            md = (out_dir / "queries" / "qid_1.md").read_text()
            self.assertIn("Need one document.", md)
            self.assertIn("history/example.txt", md)
            self.assertIn("Exact Answer: yes", md)
            self.assertEqual(main(["--run-dir", str(run_dir), "--queries", str(queries),
                                   "--expected-count", "2", "--out-dir", str(out_dir)]), 0)

    def test_missing_trace_and_query_absent_from_both_sets_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_pair(run_dir, 1)
            _write_pair(run_dir, 2)
            (run_dir / "_traces" / "qid_2" / "single.json").unlink()
            queries = root / "queries.jsonl"
            queries.write_text('{"query_id": 1}\n{"query_id": 2}\n{"query_id": 3}\n', encoding="utf-8")
            out_dir = root / "report"

            report = inspect_run(run_dir, out_dir, queries=queries, expected_count=3)

            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_trace_ids"], ["2", "3"])
            self.assertEqual(report["missing_per_query_ids"], ["3"])
            self.assertTrue((out_dir / "queries" / "qid_3.md").exists())
            self.assertEqual(main(["--run-dir", str(run_dir), "--queries", str(queries),
                                   "--expected-count", "3", "--out-dir", str(out_dir)]), 1)

    def test_row_trace_disagreement_fails_without_queries_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_pair(run_dir, 4, mismatch=True)

            report = inspect_run(run_dir, root / "report", expected_count=1)

            self.assertFalse(report["complete"])
            self.assertFalse(report["expected_ids_known"])
            self.assertTrue(any("agent cost differs" in item["message"] for item in report["issues"]))

    def test_malformed_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_pair(run_dir, 1)
            (run_dir / "_traces" / "qid_1" / "single.json").write_text("{broken", encoding="utf-8")

            report = inspect_run(run_dir, root / "report", expected_count=1)

            self.assertFalse(report["complete"])
            self.assertTrue(any("cannot parse" in item["message"] for item in report["issues"]))
            self.assertEqual(json.loads((root / "report" / "timelines.jsonl").read_text())["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
