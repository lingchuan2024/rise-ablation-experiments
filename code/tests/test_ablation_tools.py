"""Ablation contracts: stable search pages, isolated auto scopes, absent shell."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rise.ablation_tools import (AutoRerankSearchTool, SearchPagingTool,
                                WithoutBashResponse)
from rise.ablation_prompt import (ablation_read_tool_spec,
    ablation_rerank_tool_spec, ablation_search_tool_spec, build_ablation_prompt)
from rise.rerank_v13 import rerank_tool_spec
from rise.tools import read_tool_spec


class FakeService:
    def __init__(self):
        self.manager = SimpleNamespace(_search_sets={})
        self.calls = []
        self.failure_query = None
        self.raise_error = None

    def __call__(self, args):
        self.calls.append(copy.deepcopy(args))
        if self.raise_error is not None:
            raise self.raise_error
        if args["query"] == self.failure_query:
            return json.dumps({"status": "error", "error": "scoring unavailable"})
        return json.dumps({"status": "ok", "result_id": "R-private",
            "next_offset": 10, "cache_hit": False, "scoring_seconds": 1.23,
            "items": [{"file_path": path, "text": f"ranked-{path}",
                       "read_args": {"file_path": path, "offset": 4, "limit": 8}}
                      for path in reversed(args["file_paths"])]})


class AblationToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = []
        for i in range(25):
            path = f"docs/doc{i:02d}.txt"
            target = self.root / path
            target.parent.mkdir(exist_ok=True)
            target.write_text(f"original snippet {i} " + "x" * 140, encoding="utf-8")
            self.docs.append({"file_path": path, "source_sha256": hashlib.sha256(
                target.read_bytes()).hexdigest(), "search_rank": i + 1,
                "search_score": float(25 - i)})
        self.service = FakeService()
        self.retrievals = []
        self.scopes = [self.docs]

    def underlying(self, args):
        self.retrievals.append(copy.deepcopy(args))
        queries = args.get("queries", args.get("query"))
        notice = "legacy normalization notice\n" if "query" in args else ""
        if isinstance(queries, str):
            value = queries.strip()
            queries = json.loads(value) if value.startswith("[") and value.endswith("]") else [value]
        if not isinstance(queries, (list, tuple)):
            queries = []
        queries = [str(query).strip() for query in queries if str(query).strip()]
        if not queries:
            return notice + "Error: search called without any queries."
        count = len(self.retrievals)
        for i, query in enumerate(queries, 1):
            scope = self.scopes[min(i - 1, len(self.scopes) - 1)]
            self.service.manager._search_sets[f"S{count:03d}.q{i}"] = {
                "query": query, "docs": copy.deepcopy(scope)}
        return notice + "baseline first preview; ORIGINAL_NOT_RETURNED_BY_AUTO"

    def paging(self):
        return SearchPagingTool(self.underlying, self.service, working_dir=self.root)

    def test_pages_preserve_original_order_without_retrieval_or_scoring(self):
        tool = self.paging()
        first = tool({"queries": ["alpha"]})
        self.assertTrue(first.startswith("baseline first preview;"))
        self.assertIn('"next_offset":10', first)
        second = tool({"search_id": "S001.q1", "offset": 10})
        self.assertIn("11. [15.00] docs/doc10.txt", second)
        self.assertIn("20. [6.00] docs/doc19.txt", second)
        self.assertNotIn("docs/doc20.txt", second)
        last = tool({"search_id": "S001.q1", "offset": 20})
        self.assertIn('"next_offset":null', last)
        self.assertIn('"returned_documents":5', last)
        self.assertEqual(len(self.retrievals), 1)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(tool.summary["retrieval_calls"], 1)
        self.assertEqual(tool.summary["search_page_calls"], 2)

    def test_old_page_survives_new_search_and_manager_checkpoint_replacement(self):
        tool = self.paging()
        tool({"queries": ["alpha"]})
        expected = tool({"search_id": "S001.q1", "offset": 10})
        tool({"queries": ["beta"]})
        state = tool.export_state()
        self.service.manager = copy.deepcopy(self.service.manager)
        restored = self.paging()
        restored.restore_state(state)
        self.assertEqual(restored({"search_id": "S001.q1", "offset": 10}), expected)
        self.assertEqual(restored.summary["retrieval_calls"], 2)
        self.assertEqual(restored.summary["search_page_calls"], 2)

    def test_invalid_page_never_imports_and_snapshot_change_fails(self):
        tool = self.paging()
        tool({"queries": ["alpha"]})
        for args in ({"search_id": "missing"}, {"search_id": "S001.q1", "offset": -1},
                     {"search_id": "S001.q1", "offset": True},
                     {"search_id": "S001.q1", "queries": ["beta"]},
                     {"queries": ["beta"], "offset": 10}):
            self.assertTrue(tool(args).startswith("Error:"))
        (self.root / "docs/doc10.txt").write_text("changed", encoding="utf-8")
        self.assertIn("source changed since search", tool({"search_id": "S001.q1", "offset": 10}))
        self.assertEqual(len(self.retrievals), 1)
        self.assertEqual(self.service.calls, [])

    def test_page_beyond_end_is_empty_and_has_no_next_page(self):
        tool = self.paging()
        tool({"queries": ["alpha"]})
        page = tool({"search_id": "S001.q1", "offset": 1000})
        self.assertIn('"returned_documents":0', page)
        self.assertIn('"next_offset":null', page)

    def test_new_search_preserves_v14_argument_recovery_and_original_notice(self):
        for args, expected in (({"queries": " alpha "}, ["alpha"]),
                               ({"queries": '["alpha","beta"]'}, ["alpha", "beta"]),
                               ({"query": "alpha"}, ["alpha"]),
                               ({"queries": ["alpha", 2]}, ["alpha", "2"])):
            paging = self.paging()
            output = paging(args)
            self.assertEqual(self.retrievals[-1], args)
            self.assertIn("Search pages:", output)
            if "query" in args:
                self.assertTrue(output.startswith("legacy normalization notice"))
            auto = AutoRerankSearchTool(self.underlying, self.service)
            result = json.loads(auto(args))
            self.assertEqual([entry["query"] for entry in result["queries"]], expected)
            self.assertEqual(self.retrievals[-1], args)
        invalid = self.paging()
        self.assertIn("Error:", invalid({"queries": []}))
        self.assertEqual(invalid.summary["retrieval_calls"], 0)

    def test_auto_ranks_each_queries_own_candidates_with_fixed_parameters(self):
        self.scopes = [self.docs[:12], self.docs[12:]]
        tool = AutoRerankSearchTool(self.underlying, self.service)
        encoded = tool({"queries": ["alpha", "beta"]})
        result = json.loads(encoded)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(self.retrievals), 1)
        self.assertEqual(len(self.service.calls), 2)
        for call, scope, query in zip(self.service.calls, self.scopes, ["alpha", "beta"]):
            self.assertEqual(call, {"query": query, "file_paths": [d["file_path"] for d in scope],
                "bm25_candidates": 1000, "top_k": 100, "offset": 0, "limit": 10})
        self.assertEqual(len(result["queries"][0]["items"]), 10)
        self.assertEqual(result["queries"][0]["items"][0]["file_path"], "docs/doc11.txt")
        self.assertIn("read_args", result["queries"][0]["items"][0])
        for private in ("ORIGINAL_NOT_RETURNED", "result_id", "next_offset", "scoring_seconds", "cache_hit"):
            self.assertNotIn(private, encoded)
        state = tool.export_state()
        restored = AutoRerankSearchTool(self.underlying, self.service)
        restored.restore_state(state)
        self.assertEqual(restored.summary, tool.summary)

    def test_auto_empty_scope_skips_backend_and_partial_failure_keeps_success(self):
        self.scopes = [[], self.docs[:2], self.docs[2:4]]
        self.service.failure_query = "failed"
        tool = AutoRerankSearchTool(self.underlying, self.service)
        result = json.loads(tool({"queries": ["empty", "failed", "good"]}))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["queries"][0]["status"], "ok")
        self.assertEqual(result["queries"][0]["items"], [])
        self.assertEqual(result["queries"][1]["status"], "error")
        self.assertEqual(result["queries"][1]["items"], [])
        self.assertEqual(result["queries"][2]["status"], "ok")
        self.assertEqual(len(result["queries"][2]["items"]), 2)
        self.assertEqual(len(self.service.calls), 2)
        self.assertEqual(tool.summary["empty_scopes"], 1)
        self.assertEqual(tool.summary["auto_rerank_errors"], 1)

    def test_auto_does_not_swallow_infrastructure_exception_or_add_paging(self):
        tool = AutoRerankSearchTool(self.underlying, self.service)
        self.assertTrue(tool({"search_id": "S001.q1", "offset": 10}).startswith("Error:"))
        self.assertEqual(self.retrievals, [])
        self.service.raise_error = RuntimeError("infrastructure unavailable")
        with self.assertRaisesRegex(RuntimeError, "infrastructure unavailable"):
            tool({"queries": ["alpha"]})

    def test_without_bash_removes_nested_arguments_and_preserves_truncation(self):
        payload = {"items": [{"read_may_truncate": True,
            "bash_args": {"command": "private"}, "nested": [{"bash_args": {}, "text": "evidence"}],
            "read_args": {"file_path": "a.txt"}}]}
        output = WithoutBashResponse(lambda _: json.dumps(payload))({})
        self.assertNotIn("bash_args", output)
        self.assertTrue(json.loads(output)["items"][0]["read_may_truncate"])
        self.assertIn("read_args", output)

    def test_prompt_and_specs_only_describe_available_capabilities(self):
        prompt = build_ablation_prompt("no_bash")
        read_spec = ablation_read_tool_spec("no_bash", read_tool_spec())
        rank_spec = ablation_rerank_tool_spec("no_bash", rerank_tool_spec())
        combined = prompt + json.dumps([read_spec, rank_spec])
        self.assertNotIn("bash", combined.lower())
        self.assertIn("2000", combined)
        self.assertIn("read_may_truncate", combined)
        from rise.independent_rerank_prompt import build_independent_rerank_prompt, v13_read_tool_spec
        baseline = build_independent_rerank_prompt()
        unchanged_suffix = baseline[baseline.index("You need not call every tool."):]
        self.assertTrue(prompt.endswith(unchanged_suffix))
        self.assertIn("100–200 lines", read_spec["function"]["description"])
        baseline_read = v13_read_tool_spec(read_tool_spec())
        self.assertEqual(read_spec["function"]["parameters"], baseline_read["function"]["parameters"])
        auto = build_ablation_prompt("auto_rerank") + json.dumps(
            ablation_read_tool_spec("auto_rerank", read_tool_spec()))
        self.assertNotIn("rerank(", auto)
        self.assertNotIn("rerank item's", auto)
        start = baseline.index("- bash(command, timeout?):")
        end = baseline.index("- read(file_path, offset?, limit?):", start)
        self.assertIn(baseline[start:end], auto)
        self.assertIn(baseline_read["function"]["description"].replace(
            "rerank item's", "search item's"), auto)
        self.assertIn("search_id", json.dumps(ablation_search_tool_spec("search_paging")))
        self.assertNotIn("search_id", json.dumps(ablation_search_tool_spec("auto_rerank")))
        with self.assertRaises(ValueError):
            ablation_rerank_tool_spec("auto_rerank", rerank_tool_spec())


if __name__ == "__main__":
    unittest.main()
