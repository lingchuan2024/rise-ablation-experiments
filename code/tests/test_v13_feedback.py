"""Offline checks for observational feedback and its v13-only wiring."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.tools import bash_tool_spec, read_tool_spec
from rise.v13_feedback import ProgressFeedback, PROGRESS_FEEDBACK_VERSION, v13_bash_tool_spec
import test_v13_agent as fixtures
from test_indexed_accumulated_space import CheapBackend
from test_v10_agent import response


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = 1
        self.feedback = ProgressFeedback(pool_version_provider=lambda: self.pool,
                                         log_path=self.root / "trace" / "tool_feedback.jsonl")

    def observe(self, name="bash", args=None, result="same matched paragraph"):
        return self.feedback.observe(name, args or {"command": "rg term"}, result)

    def records(self):
        return [json.loads(line) for line in self.feedback.log_path.read_text().splitlines()]

    def test_repeat_threshold_throttle_execution_and_hash_only_logging(self):
        calls = []
        wrapped = self.feedback.wrap("bash", lambda args: calls.append(args) or "same matched paragraph")
        outputs = [wrapped({"command": "rg term"}) for _ in range(6)]
        self.assertEqual(len(calls), 6)
        self.assertEqual(["Tool feedback:" in value for value in outputs],
                         [False, False, True, False, False, True])
        self.assertTrue(all(output.startswith("same matched paragraph") for output in outputs))
        records = self.records()
        self.assertEqual([record["same_result_count"] for record in records], list(range(1, 7)))
        self.assertEqual(len({record["visible_result_sha256"] for record in records}), 1)
        self.assertNotIn("rg term", self.feedback.log_path.read_text())
        self.assertNotIn("same matched paragraph", self.feedback.log_path.read_text())
        self.assertLessEqual(max(len(record["feedback"] or "") for record in records), 300)

    def test_pool_growth_resets_but_same_pool_does_not(self):
        for _ in range(2):
            self.observe()
        self.pool += 1
        output = self.observe()
        self.assertNotIn("feedback", output)
        self.observe()
        self.assertIn("Tool feedback:", self.observe())
        self.assertEqual([record["same_result_count"] for record in self.records()], [1, 2, 1, 2, 3])

    def test_different_args_or_visible_text_do_not_infer_repetition_from_same_path(self):
        for args, result in [({"command": "cat doc.txt"}, "first fact"),
                             ({"command": "cat doc.txt"}, "second fact"),
                             ({"command": "head doc.txt"}, "second fact")]:
            self.assertNotIn("feedback", self.observe(args=args, result=result))
        self.assertEqual([record["same_result_count"] for record in self.records()], [1, 1, 1])

    def test_search_record_normalization_preserves_document_content_and_match_counts(self):
        def search_result(record, *, path="source.txt", new=0):
            return (f'Query 1: "fact" — 1 matched ({new} new, {1 - new} already in working dir); showing top 1:\n'
                    f'  Search record: {record}\n'
                    f'   1. [1.00] {path}\n       "Source fact S999.q9"\n'
                    'Accumulated original-document pool: {"pool_version":1,"documents":1}')
        args = {"queries": ["fact"]}
        outputs = [self.observe("search", args, search_result(f"S{i:03d}.q1")) for i in range(1, 5)]
        self.assertEqual(["Tool feedback:" in output for output in outputs], [False, False, True, False])
        self.assertEqual(outputs[0], search_result("S001.q1"))
        self.assertTrue(outputs[2].startswith(search_result("S003.q1") + "\n"))
        self.assertEqual(len({record["visible_result_sha256"] for record in self.records()}), 4)
        self.assertEqual(len({record["result_fingerprint_sha256"] for record in self.records()}), 1)
        self.assertNotIn("Tool feedback:", self.observe("search", args, search_result("S005.q1", path="different.txt")))
        self.assertNotIn("Tool feedback:", self.observe("search", args, search_result("S006.q1", new=1)))
        self.pool += 1
        self.assertNotIn("Tool feedback:", self.observe("search", args, search_result("S007.q1")))
        self.assertEqual(self.records()[-1]["same_result_count"], 1)

    def test_empty_output_and_errors_are_separate(self):
        for args, result in [({"command": "a"}, "Error: subprocess timed out"),
                             ({"command": "b"}, "(no output; exit=2)"),
                             ({"command": "c"}, "[exit=1]\n[stderr]\nmissing file")]:
            self.assertNotIn("Tool feedback:", self.observe(args=args, result=result))
        self.assertEqual(self.feedback.summary["bash_empty_calls"], 0)
        self.assertEqual(self.feedback.summary["bash_error_calls"], 3)
        self.observe(args={"command": "d"}, result="(no matches found)")
        second = self.observe(args={"command": "e"}, result="(command succeeded, no output)")
        self.assertIn("Repeated empty bash output", second)
        self.assertEqual(self.feedback.summary["empty_feedback"], 1)

    def test_truncation_feedback_does_not_cut_original_long_line(self):
        raw = "a" * 4000 + "\n...[truncated; 1682499 chars cut from output]"
        self.assertEqual(self.observe(args={"command": "rg a"}, result=raw), raw)
        second = self.observe(args={"command": "rg b"}, result=raw)
        self.assertTrue(second.startswith(raw + "\nTool feedback:"))
        self.assertIn("head limits lines", second)
        self.assertEqual(self.feedback.summary["bash_truncated_calls"], 2)
        self.observe(args={"command": "rg c"}, result="short result")
        self.assertEqual(self.feedback.export_state()["truncated_streak"], 0)

    def test_rerank_json_preserved_and_page_scope_changes_are_distinct(self):
        raw = {"status": "ok", "result_id": "R-1-a", "items": [
            {"file_path": "source.txt", "text": "complete fact", "read_args": {"file_path": "source.txt"}}],
            "pool_version": 1, "offset": 0, "top_k": 3, "next_offset": 1,
            "cache_hit": False, "preparation_seconds": 1.0, "scoring_seconds": 0.9}
        args = {"query": "fact", "file_paths": ["source.txt"]}
        for index in range(3):
            current = copy.deepcopy(raw)
            current["cache_hit"] = index > 0
            current["preparation_seconds"] = index / 100
            current["scoring_seconds"] = 0 if index else 0.9
            output = json.loads(self.observe("rerank", args, json.dumps(current)))
            self.assertEqual(output["items"], raw["items"])
            self.assertEqual(output["cache_hit"], current["cache_hit"])
            self.assertEqual("feedback" in output, index == 2)
        changed_scope = {"query": "fact", "file_paths": ["other.txt"]}
        self.assertNotIn("feedback", json.loads(self.observe("rerank", changed_scope, json.dumps(raw))))
        page = {**raw, "offset": 1, "next_offset": 2}
        self.assertNotIn("feedback", json.loads(self.observe("rerank", {"result_id": "R-1-a", "offset": 1}, json.dumps(page))))
        records = self.records()
        self.assertEqual([record["same_result_count"] for record in records], [1, 2, 3, 1, 1])
        self.assertEqual(len({record["result_fingerprint_sha256"] for record in records[:3]}), 1)
        self.assertGreater(len({record["visible_result_sha256"] for record in records[:3]}), 1)

    def test_counter_restore_rejects_version_or_wrong_pool(self):
        self.observe()
        self.observe()
        state = self.feedback.export_state()
        restored = ProgressFeedback(pool_version_provider=lambda: self.pool,
                                    log_path=self.root / "restored.jsonl")
        restored.restore_state(state)
        self.assertIn("Tool feedback:", restored.observe("bash", {"command": "rg term"}, "same matched paragraph"))
        self.assertEqual(restored.summary["observed_calls"], 3)
        changed = copy.deepcopy(state)
        changed["version"] = "old"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            restored.restore_state(changed)
        self.pool += 1
        with self.assertRaisesRegex(ValueError, "restored original pool"):
            restored.restore_state(state)

    def test_logging_failure_does_not_replace_tool_evidence(self):
        with patch.object(Path, "open", side_effect=OSError("fixture filesystem outage")):
            self.assertEqual(self.observe(), "same matched paragraph")
        self.assertEqual(self.feedback.summary["logging_errors"], 1)

    def test_timeout_schema_is_v13_only_and_read_is_not_observed(self):
        original = bash_tool_spec()
        before = copy.deepcopy(original)
        v13 = v13_bash_tool_spec(original)
        self.assertEqual(original, before)
        self.assertNotIn("timeout", bash_tool_spec()["function"]["parameters"]["properties"])
        self.assertEqual(v13["function"]["parameters"]["properties"]["timeout"]["maximum"], 60)
        with self.assertRaisesRegex(ValueError, "only"):
            self.feedback.wrap("read", lambda _args: "baseline content")


class FeedbackAgentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.V13AgentIntegrationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_checkpoint_restores_repetition_counts(self):
        fixture = self.fixture
        checkpoint = fixture.root / "checkpoint.json"
        config = fixture.config()
        first = iter([response("bash", {"command": "rg aurora"}),
                      response("bash", {"command": "rg aurora"}), RuntimeError("transport outage")])
        def interrupted(*_args, **_kwargs):
            value = next(first)
            if isinstance(value, Exception):
                raise value
            return value
        shell = lambda *_args, **_kwargs: lambda _args: "same complete evidence"
        initial = fixture.run_agent(config, CheapBackend(), interrupted, checkpoint=checkpoint, shell=shell)
        self.assertTrue(initial.terminated_by.startswith("api_error"))
        saved = json.loads(checkpoint.read_text())
        self.assertEqual(saved["tool_state"]["version"], PROGRESS_FEEDBACK_VERSION)
        self.assertEqual(saved["tool_state"]["feedback"]["usage"]["observed_calls"], 2)
        replies = iter([response("bash", {"command": "rg aurora"}), response()])
        requests = []
        def continued(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return next(replies)
        resumed = fixture.run_agent(config, CheapBackend(), continued, checkpoint=checkpoint, shell=shell)
        self.assertIn("Same arguments and returned content 3 times", requests[-1]["messages"][-1]["content"])
        self.assertEqual(resumed.reranker_summary["progress_feedback"]["observed_calls"], 3)
        self.assertEqual(resumed.reranker_summary["progress_feedback"]["feedback_calls"], 1)
        self.assertEqual(len((fixture.root / "trace/tool_feedback.jsonl").read_text().splitlines()), 3)

    def test_real_search_wrapper_repeated_records_do_not_hide_duplicate_results(self):
        fixture = self.fixture
        responses = iter([*[response("search", {"queries": ["history"]}) for _ in range(4)], response()])
        requests = []
        def chat(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return next(responses)
        run = fixture.run_agent(fixture.config(), CheapBackend(), chat)
        outputs = [request["messages"][-1]["content"] for request in requests[1:]]
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(f"Search record: S{i:03d}.q1" in output for i, output in enumerate(outputs, 1)))
        self.assertEqual(["Tool feedback:" in output for output in outputs], [False, False, False, True])
        records = [json.loads(line) for line in (fixture.root / "trace/tool_feedback.jsonl").read_text().splitlines()]
        self.assertEqual([record["pool_version"] for record in records], [1, 1, 1, 1])
        # First import reports 12 new documents; that observation remains
        # different from the three subsequent searches of already imported docs.
        self.assertEqual([record["same_result_count"] for record in records], [1, 1, 2, 3])
        self.assertEqual(run.reranker_summary["progress_feedback"]["feedback_calls"], 1)
        self.assertEqual(run.reranker_summary["pool_documents"], 12)

    def test_baseline_read_results_and_parameters_stay_identical(self):
        fixture = self.fixture
        responses = iter([response("search", {"queries": ["history"]}),
                          response("read", {"file_path": "docs/doc00.txt"}),
                          response("read", {"file_path": "docs/doc00.txt"}),
                          response("read", {"file_path": "docs/doc00.txt"}), response()])
        requests = []
        def chat(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return next(responses)
        run = fixture.run_agent(fixture.config(), CheapBackend(), chat)
        reads = [request["messages"][-1]["content"] for request in requests[2:]]
        self.assertEqual(len(set(reads)), 1)
        self.assertNotIn("Tool feedback:", reads[-1])
        read_spec = next(tool for tool in requests[0]["tools"]
                         if tool["function"]["name"] == "read")
        baseline_spec = read_tool_spec()
        restored_descriptions = copy.deepcopy(read_spec)
        restored_descriptions["function"]["description"] = baseline_spec["function"]["description"]
        restored_descriptions["function"]["parameters"]["properties"]["limit"]["description"] = (
            baseline_spec["function"]["parameters"]["properties"]["limit"]["description"])
        self.assertEqual(restored_descriptions, baseline_spec)
        self.assertIn("Set limit explicitly", read_spec["function"]["description"])
        self.assertNotIn("prefer the default", read_spec["function"]["parameters"]["properties"]["limit"]["description"])
        self.assertEqual(run.reranker_summary["progress_feedback"]["observed_calls"], 1)
        self.assertEqual(run.n_read_calls, 3)


if __name__ == "__main__":
    unittest.main()
