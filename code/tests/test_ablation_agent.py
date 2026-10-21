"""Three ablations through the real Agent loop, with offline source evidence.

The fixtures exercise the genuine source pool, passage postings and checkpoint
format. Chat, neural inference and shell execution are mocked; no service,
weights or network access are needed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.trajectory import compute_run_config_hash, cached_row_is_reusable, SCHEMA_VERSION
from rise.ablation_prompt import bind_search_paging_prompt_version, SEARCH_PAGING_PROMPT_VERSION
from test_indexed_accumulated_space import build_raw_index, CheapBackend
from test_reranker_tool import structured
from test_v10_agent import response
import test_v13_agent as v13_fixture


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


class AblationAgentIntegrationTests(unittest.TestCase):
    setUp = v13_fixture.V13AgentIntegrationTests.setUp
    run_agent = v13_fixture.V13AgentIntegrationTests.run_agent

    def config(self, variant="search_paging"):
        config = v13_fixture.V13AgentIntegrationTests.config(self)
        config.extras["ablation_variant"] = variant
        return config

    def test_search_pages_keep_original_ranking_after_new_search_without_scoring(self):
        backend, requests, retrievals = CheapBackend(), [], []
        search_id = None

        def retrieve(_searcher, _doc_ids, query, **_kwargs):
            retrievals.append(query)
            return self.results

        def chat(_client, **kwargs):
            nonlocal search_id
            requests.append(copy.deepcopy(kwargs))
            step = len(requests)
            if step == 1:
                return response("search", {"queries": ["aurora"]})
            visible = kwargs["messages"][-1]["content"]
            if step == 2:
                self.assertNotIn("docs/doc11.txt", visible)
                search_id = re.search(r"S\d+\.q\d+", visible).group(0)
                return response("search", {"search_id": search_id, "offset": 10})
            if step == 3:
                self.assertIn("docs/doc10.txt", visible)
                self.assertIn("docs/doc11.txt", visible)
                return response("search", {"queries": ["history"]})
            if step == 4:
                return response("search", {"search_id": search_id, "offset": 10})
            self.assertEqual(step, 5)
            self.assertIn("docs/doc11.txt", visible)
            return response()

        run = self.run_agent(self.config(), backend, chat, results=retrieve)
        self.assertEqual(run.terminated_by, "done")
        self.assertEqual([t["function"]["name"] for t in requests[0]["tools"]],
                         ["search", "bash", "read", "rerank"])
        self.assertEqual(retrievals, ["aurora", "history"])
        self.assertEqual(run.tool_call_breakdown["search"], 4)
        self.assertEqual(run.n_bm25_searches, 2)
        self.assertEqual(backend.calls, [])
        self.assertIn("docs/doc11.txt", run.surfaced_relpaths)

    def test_search_paging_checkpoint_restores_without_retrieval_or_scoring(self):
        config, backend = self.config(), CheapBackend()
        checkpoint = self.root / "checkpoint.json"
        replies = iter([response("search", {"queries": ["aurora"]}),
                        RuntimeError("fixture transport outage")])

        def interrupted(*_args, **_kwargs):
            result = next(replies)
            if isinstance(result, Exception):
                raise result
            return result

        initial = self.run_agent(config, backend, interrupted, checkpoint=checkpoint)
        self.assertTrue(initial.terminated_by.startswith("api_error"))
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        visible = next(m["content"] for m in state["messages"]
                       if m.get("tool_call_id") == "call-search")
        search_id = re.search(r"S\d+\.q\d+", visible).group(0)
        resumed_backend, requests = CheapBackend(), []
        replies = iter([response("search", {"search_id": search_id, "offset": 10}), response()])

        def continued(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return next(replies)

        def forbidden_retrieval(*_args, **_kwargs):
            self.fail("resuming a saved search page must not retrieve")

        resumed = self.run_agent(config, resumed_backend, continued,
                                 checkpoint=checkpoint, results=forbidden_retrieval)
        self.assertEqual(resumed.terminated_by, "done")
        self.assertIn("docs/doc11.txt", requests[-1]["messages"][-1]["content"])
        self.assertEqual(resumed.n_bm25_searches, 1)
        self.assertEqual(resumed.tool_call_breakdown["search"], 2)
        self.assertEqual(resumed_backend.calls, [])

    def test_auto_rerank_scores_each_query_scope_and_hidden_tool_is_not_callable(self):
        backend, requests = CheapBackend(), []
        groups = {"history": self.results[:6], "aurora calibration": self.results[6:]}

        def retrieve(_searcher, _doc_ids, query, **_kwargs):
            return groups[query]

        def chat(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            step = len(requests)
            if step == 1:
                return response("search", {"queries": list(groups)})
            visible = kwargs["messages"][-1]["content"]
            if step == 2:
                payload = json.loads(visible)
                self.assertEqual(payload["status"], "ok")
                nodes = list(dictionaries(payload))
                self.assertFalse(any("result_id" in node or "next_offset" in node for node in nodes))
                shown = {node["file_path"] for node in nodes if "file_path" in node}
                self.assertIn("docs/doc11.txt", shown)
                return response("rerank", {"query": "aurora"})
            self.assertEqual(step, 3)
            self.assertIn("unknown tool", visible)
            return response()

        run = self.run_agent(self.config("auto_rerank"), backend, chat, results=retrieve)
        self.assertEqual(run.terminated_by, "done")
        self.assertEqual([t["function"]["name"] for t in requests[0]["tools"]],
                         ["search", "bash", "read"])
        self.assertNotIn("rerank(", requests[0]["messages"][0]["content"])
        self.assertEqual([query for query, _texts in backend.calls], list(groups))
        self.assertEqual([len(texts) for _query, texts in backend.calls], [6, 6])
        self.assertTrue(all("aurora calibration evidence" not in text for text in backend.calls[0][1]))
        self.assertTrue(any("aurora calibration evidence" in text for text in backend.calls[1][1]))
        self.assertEqual(run.n_bm25_searches, 1)
        self.assertEqual(run.reranker_summary["scoring_calls"], 2)
        self.assertIn("docs/doc11.txt", run.surfaced_relpaths)

    def test_auto_rerank_checkpoint_keeps_completed_scoring_without_repeat(self):
        config, backend = self.config("auto_rerank"), CheapBackend()
        checkpoint = self.root / "checkpoint.json"
        replies = iter([response("search", {"queries": ["aurora"]}),
                        RuntimeError("fixture transport outage")])

        def interrupted(*_args, **_kwargs):
            result = next(replies)
            if isinstance(result, Exception):
                raise result
            return result

        initial = self.run_agent(config, backend, interrupted, checkpoint=checkpoint)
        self.assertTrue(initial.terminated_by.startswith("api_error"))
        self.assertEqual(len(backend.calls), 1)
        resumed_backend = CheapBackend()
        resumed = self.run_agent(config, resumed_backend, lambda *_args, **_kwargs: response(),
                                 checkpoint=checkpoint)
        self.assertEqual(resumed.terminated_by, "done")
        self.assertEqual(resumed_backend.calls, [])
        self.assertEqual(resumed.reranker_summary["scoring_calls"], 1)
        self.assertEqual(resumed.n_bm25_searches, 1)

    def test_no_bash_never_constructs_shell_and_preserves_long_line_warning(self):
        self.corpus = self.root / "long-corpus"
        self.mapping = {"0": "docs/short.txt", "1": "docs/long.txt"}
        self.documents = {
            "docs/short.txt": structured("history education"),
            "docs/long.txt": structured("navigation " * 700 + "aurora calibration evidence"),
        }
        self.index = build_raw_index(self.root / "long-index", self.corpus, self.documents)
        self.results = [("0", 2.0), ("1", 1.0)]
        backend = CheapBackend(lambda _query, texts: [9.0 if "aurora" in text else 0.0 for text in texts])
        requests = []

        def forbidden_shell(*_args, **_kwargs):
            self.fail("no_bash must not construct a shell tool")

        def chat(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            step = len(requests)
            if step == 1:
                return response("search", {"queries": ["aurora"]})
            if step == 2:
                return response("rerank", {"query": "aurora calibration", "bm25_candidates": 12,
                                           "top_k": 1, "limit": 1})
            visible = kwargs["messages"][-1]["content"]
            if step == 3:
                payload = json.loads(visible)
                self.assertEqual(payload["status"], "ok")
                self.assertTrue(payload["items"][0]["read_may_truncate"])
                self.assertIn("read_args", payload["items"][0])
                self.assertFalse(any("bash_args" in node for node in dictionaries(payload)))
                return response("bash", {"command": "ls"})
            self.assertEqual(step, 4)
            self.assertIn("unknown tool", visible)
            return response()

        config = self.config("no_bash")
        config.extras["structured_doc_mode"] = True
        run = self.run_agent(config, backend, chat, shell=forbidden_shell)
        self.assertEqual(run.terminated_by, "done")
        self.assertEqual([t["function"]["name"] for t in requests[0]["tools"]],
                         ["search", "read", "rerank"])
        guidance = requests[0]["messages"][0]["content"] + json.dumps(requests[0]["tools"])
        self.assertIsNone(re.search(r"\bbash\b", guidance, re.IGNORECASE))
        system_prompt = requests[0]["messages"][0]["content"]
        for preserved in ("For additional body context, 100–200 lines",
                          "Line counts are not token budgets",
                          "Split multi-condition questions into focused checks",
                          "Progress feedback describes call repetition",
                          "Every document begins with a YAML frontmatter block"):
            self.assertIn(preserved, system_prompt)
        self.assertEqual(len(backend.calls), 1)

    def test_variants_have_distinct_checkpoint_identities(self):
        configs = [self.config(variant) for variant in ("search_paging", "auto_rerank", "no_bash")]
        hashes = {compute_run_config_hash(config.to_hashable_dict()) for config in configs}
        self.assertEqual(len(hashes), 3)

    def test_paging_guidance_version_blocks_old_rows_and_stale_direct_calls(self):
        config = self.config()
        old_hash = compute_run_config_hash(config.to_hashable_dict())
        bind_search_paging_prompt_version(config.extras)
        new_hash = compute_run_config_hash(config.to_hashable_dict())
        self.assertNotEqual(old_hash, new_hash)
        old_row = {"schema_version": SCHEMA_VERSION, "run_config_hash": old_hash}
        self.assertFalse(cached_row_is_reusable(
            old_row, expected_run_config_hash=new_hash)[0])
        current_row = {**old_row, "run_config_hash": new_hash}
        self.assertTrue(cached_row_is_reusable(
            current_row, expected_run_config_hash=new_hash)[0])
        self.assertEqual(config.extras["search_paging_prompt_version"], SEARCH_PAGING_PROMPT_VERSION)

        config.extras["search_paging_prompt_version"] = "v14-search-paging-guidance-v1"
        with self.assertRaisesRegex(ValueError, "search_paging_prompt_version"):
            self.run_agent(config, CheapBackend(),
                           lambda *_args, **_kwargs: self.fail("stale prompt must not reach the API"))

        for variant in ("auto_rerank", "no_bash"):
            other = self.config(variant)
            before = copy.deepcopy(other.to_hashable_dict())
            bind_search_paging_prompt_version(other.extras)
            self.assertEqual(other.to_hashable_dict(), before)


if __name__ == "__main__":
    unittest.main()
