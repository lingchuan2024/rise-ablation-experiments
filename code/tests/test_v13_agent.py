"""Exercise v13's real Agent loop with offline postings and mocked inference.

The fixture uses the actual passage-index and local BM25 path. Neural scores,
chat responses, and sandbox execution are mocked; no remote service or model
download is required.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stderr
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.protocol import for_rise
from rise.rise_agent import run_rise_agent
from rise.trajectory import compute_run_config_hash
from test_indexed_accumulated_space import build_raw_index, CheapBackend
from test_reranker_tool import structured
from test_v10_agent import response


class V13AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "corpus"
        self.mapping = {str(i): f"docs/doc{i:02d}.txt" for i in range(12)}
        self.documents = {
            rel: structured("aurora calibration evidence" if did == "11"
                            else "history education document " + did)
            for did, rel in self.mapping.items()
        }
        self.index = build_raw_index(self.root / "index", self.corpus, self.documents)
        self.working = self.root / "_working" / "1"
        self.results = [(did, float(12 - int(did))) for did in self.mapping]

    def config(self):
        config = for_rise(agent_model="test-model", enable_sandbox=True,
                          max_model_calls=12, api_max_retries=1)
        config.extras.update({
            "independent_rerank_enabled": True,
            "space_passage_index": str(self.root / "index"),
            "space_index_identity": {
                "format_version": self.index.manifest["format_version"],
                "configuration_sha256": self.index.manifest["configuration"]["sha256"],
                "corpus_content_sha256": self.index.manifest["corpus_content_sha256"],
            },
        })
        return config

    def run_agent(self, config, backend, chat, *, checkpoint=None, results=None,
                  shell=None):
        with patch("rise.tools.retrieve", side_effect=results if results is not None
                   else lambda *_args, **_kwargs: self.results), \
             patch("rise.reranker_backend.QwenReranker", return_value=backend), \
             patch("rise.agent.chat_with_retry", side_effect=chat), \
             patch("rise.rise_agent.make_bash_tool", side_effect=shell or
                   (lambda *_args, **_kwargs: lambda _arguments: "(command succeeded, no output)")):
            return run_rise_agent(
                question="find aurora calibration", query_id=1, run_config=config,
                searcher=object(), doc_ids=list(self.mapping),
                docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                working_dir=self.working, client=object(),
                trace_path=self.root / "trace" / "single.json",
                turn_checkpoint_path=checkpoint, map_keys=set(self.mapping.values()),
                enable_sandbox=True, passage_index=self.index)

    def test_four_tools_surface_rerank_source_and_keep_full_pool_for_bash_read(self):
        backend = CheapBackend(lambda _query, texts: [
            9.0 if "aurora" in text else 0.0 for text in texts])
        requests, shell_roots, shell_calls = [], [], []
        def shell(root, **kwargs):
            shell_roots.append((root, kwargs))
            def execute(args):
                shell_calls.append(args)
                self.assertEqual(len(list(root.rglob("*.txt"))), 12)
                self.assertTrue((root / "docs/doc00.txt").is_file())
                self.assertTrue((root / "docs/doc11.txt").is_file())
                return "all 12 original documents are accessible"
            return execute
        step = 0
        def chat(_client, **kwargs):
            nonlocal step
            step += 1
            requests.append(copy.deepcopy(kwargs))
            if step == 1:
                return response("search", {"queries": ["aurora"]})
            if step == 2:
                self.assertNotIn("docs/doc11.txt", kwargs["messages"][-1]["content"])
                return response("rerank", {"query": "aurora calibration", "bm25_candidates": 12,
                                           "top_k": 1, "limit": 1})
            if step == 3:
                ranking = json.loads(kwargs["messages"][-1]["content"])
                self.assertEqual(ranking["status"], "ok")
                self.assertEqual(ranking["items"][0]["file_path"], "docs/doc11.txt")
                return response("bash", {"command": "ls docs"})
            if step == 4:
                return response("read", {"file_path": "docs/doc00.txt"})
            if step == 5:
                self.assertIn("history education document 0", kwargs["messages"][-1]["content"])
                return response()
            self.fail("unexpected extra model turn")
        run = self.run_agent(self.config(), backend, chat, shell=shell)
        self.assertEqual(run.terminated_by, "done")
        self.assertEqual(run.tool_call_breakdown, {"search": 1, "rerank": 1, "bash": 1, "read": 1})
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]],
                         ["search", "bash", "read", "rerank"])
        self.assertIn("docs/doc11.txt", run.surfaced_relpaths)
        self.assertEqual(run.n_bash_calls, 1)
        self.assertEqual(run.n_read_calls, 1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 12)
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(shell_roots[0][0], self.working)
        self.assertTrue(shell_roots[0][1]["enable_sandbox"])
        self.assertEqual(len(shell_calls), 1)
        self.assertEqual(len(list(self.working.rglob("*.txt"))), 12)

    def test_search_then_direct_original_read_does_not_score(self):
        backend = CheapBackend()
        replies = iter([response("search", {"queries": ["history"]}),
                        response("read", {"file_path": "docs/doc00.txt"}), response()])
        run = self.run_agent(self.config(), backend, lambda *_args, **_kwargs: next(replies))
        self.assertEqual(run.terminated_by, "done")
        self.assertEqual(run.tool_call_breakdown, {"search": 1, "read": 1})
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)

    def test_result_checkpoint_resumes_paging_without_scoring_again(self):
        checkpoint = self.root / "checkpoint.json"
        config, backend = self.config(), CheapBackend()
        first = iter([response("search", {"queries": ["aurora"]}),
                      response("rerank", {"query": "aurora calibration", "bm25_candidates": 12,
                                         "top_k": 2, "limit": 1}),
                      RuntimeError("fixture transport outage")])
        def interrupted(*_args, **_kwargs):
            value = next(first)
            if isinstance(value, Exception):
                raise value
            return value
        initial = self.run_agent(config, backend, interrupted, checkpoint=checkpoint)
        self.assertTrue(initial.terminated_by.startswith("api_error"))
        self.assertEqual(len(backend.calls), 1)
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        rerank_messages = [message for message in state["messages"]
                           if message.get("tool_call_id") == "call-rerank"]
        result_id = json.loads(rerank_messages[-1]["content"])["result_id"]
        self.assertEqual(state["identity"]["run_config_hash"],
                         compute_run_config_hash(config.to_hashable_dict()))
        resumed_backend = CheapBackend()
        page_replies = iter([response("rerank", {"result_id": result_id, "top_k": 3,
                                                 "offset": 1, "limit": 2}), response()])
        requests = []
        def continued(_client, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return next(page_replies)
        resumed = self.run_agent(config, resumed_backend, continued, checkpoint=checkpoint)
        page = json.loads(requests[-1]["messages"][-1]["content"])
        self.assertEqual(page["status"], "ok")
        self.assertEqual(len(page["items"]), 2)
        self.assertEqual(page["result_id"], result_id)
        self.assertEqual(resumed.terminated_by, "done")
        self.assertEqual(resumed_backend.calls, [])
        self.assertEqual(resumed_backend.tokenizer_calls, 0)
        self.assertEqual(resumed.n_bm25_searches, 1)
        self.assertEqual(resumed.tool_call_breakdown["rerank"], 2)
        self.assertEqual(resumed.reranker_summary["scoring_calls"], 1)

    def test_v13_config_identity_is_distinct_and_rejects_incompatible_checkpoint(self):
        checkpoint = self.root / "checkpoint.json"
        config = self.config()
        self.run_agent(config, CheapBackend(), lambda *_args, **_kwargs: response(), checkpoint=checkpoint)
        before = compute_run_config_hash(config.to_hashable_dict())
        changed = copy.deepcopy(config)
        changed.extras["independent_rerank_enabled"] = False
        self.assertNotEqual(before, compute_run_config_hash(changed.to_hashable_dict()))
        with patch("rise.agent.chat_with_retry") as chat:
            with self.assertRaisesRegex(ValueError, "Checkpoint identity changed"):
                run_rise_agent(question="find aurora calibration", query_id=1,
                    run_config=changed, searcher=object(), doc_ids=list(self.mapping),
                    docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                    working_dir=self.working, client=object(),
                    turn_checkpoint_path=checkpoint, enable_sandbox=True)
            chat.assert_not_called()

    def test_invalid_direct_agent_modes_are_rejected_before_model_calls(self):
        invalid = ({"no_bash": True}, {"reranker_enabled": True},
                   {"space_reranker_enabled": True}, {"bm25_k": 100},
                   {"passage_mode": True}, {"reranker_passage_overlap_tokens": 0},
                   {"independent_rerank_prompt_version": "old-prompt"},
                   {"independent_rerank_response_version": "old-response"})
        for extras in invalid:
            with self.subTest(extras=extras):
                config = self.config()
                config.extras.update(extras)
                with patch("rise.agent.chat_with_retry") as chat, \
                     patch("rise.reranker_backend.QwenReranker") as backend:
                    with self.assertRaises(ValueError):
                        run_rise_agent(question="question", query_id=1, run_config=config,
                            searcher=object(), doc_ids=list(self.mapping),
                            docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                            working_dir=self.working, client=object(), passage_index=self.index)
                    chat.assert_not_called()
                    backend.assert_not_called()


class V13CLIIntegrationTests(unittest.TestCase):
    def test_invalid_cli_modes_fail_before_corpus_or_api_access(self):
        spec = importlib.util.spec_from_file_location("v13_agent_cli_test", ROOT / "scripts/run_rise.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        invalid = (("--no-bash",), ("--no-sandbox",), ("--reranker",),
                   ("--space-reranker",), ("--passage-mode",), ("--bm25-k", "100"))
        for extra in invalid:
            with self.subTest(extra=extra), \
                 patch.object(sys, "argv", ["run", "--independent-rerank",
                                            "--space-passage-index", "/unused/index", *extra]), \
                 patch.object(runner, "_load_records") as records, \
                 patch.object(runner, "make_client") as client, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                records.assert_not_called()
                client.assert_not_called()
        with patch.object(sys, "argv", ["run", "--independent-rerank"]), \
             patch.object(runner, "_load_records") as records, \
             redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            records.assert_not_called()


if __name__ == "__main__":
    unittest.main()
