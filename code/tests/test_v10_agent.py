"""Exercise the actual v10 agent loop, search import, trace and resume."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.protocol import for_rise
from rise.rise_agent import run_rise_agent
from rise.reranker_prompt import RERANKER_PROMPT_VERSION
from rise.tools import make_bm25_search_tool
from test_reranker_tool import FakeBackend, structured


class Call:
    def __init__(self, name, args):
        self.id = "call-" + name
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self):
        return {"id": self.id, "type": "function", "function": {
            "name": self.function.name, "arguments": self.function.arguments}}


def response(name=None, args=None):
    calls = [Call(name, args)] if name else []
    message = SimpleNamespace(content="" if calls else "Exact Answer: aurora",
                              tool_calls=calls, reasoning_content="")
    return SimpleNamespace(choices=[SimpleNamespace(message=message,
        finish_reason="tool_calls" if calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class V10AgentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.mapping = {str(i): f"doc{i:04d}.txt" for i in range(1000)}
        for i, rel in self.mapping.items():
            body = "unrelated filler history" if i != "998" else "aurora calibration evidence"
            (self.corpus / rel).write_text(structured(body), encoding="utf-8")
        self.results = [(did, float(1000 - int(did))) for did in self.mapping]

    def config(self, enabled=True):
        config = for_rise(agent_model="test-model", max_model_calls=6, api_max_retries=1)
        if enabled:
            config.extras["reranker_enabled"] = True
        return config

    def run_agent(self, config, backend, chat, checkpoint=None):
        with patch("rise.tools.retrieve", return_value=self.results), \
             patch("rise.reranker_backend.QwenReranker", return_value=backend), \
             patch("rise.agent.chat_with_retry", side_effect=chat):
            return run_rise_agent(
                question="find aurora calibration", query_id=1, run_config=config,
                searcher=object(), doc_ids=list(self.mapping), docid_to_relpath=self.mapping,
                bc_plus_docs_root=self.corpus, working_dir=self.root / "working",
                client=object(), trace_path=self.root / "trace" / "single.json",
                turn_checkpoint_path=checkpoint, map_keys=set(self.mapping.values()))

    def test_search_import_1000_and_preview_text_match_frozen_baseline(self):
        import importlib.util
        path = ROOT.parent / "baseline_snapshot" / "src" / "rise" / "tools.py"
        spec = importlib.util.spec_from_file_location("rise.frozen_tools_v10_test", path)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        outputs = []
        for factory in (frozen.make_bm25_search_tool, make_bm25_search_tool):
            wd = self.root / ("frozen" if not outputs else "v10off")
            fn = factory(searcher=object(), doc_ids=list(self.mapping),
                docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                working_dir=wd, k=1000, top_n_preview=10, snippet_chars=100)
            with patch.dict(fn.__globals__, {"retrieve": lambda *_args, **_kw: self.results}):
                outputs.append(fn({"queries": ["aurora calibration"]}))
            self.assertEqual(len(list(wd.rglob("*.txt"))), 1000)
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn("doc0998.txt", outputs[1])

    def test_actual_agent_registers_optional_tool_and_traces_previewed_sources(self):
        backend = FakeBackend(lambda _q, texts: [5. if "aurora" in text else 0. for text in texts])
        requests = []
        replies = iter([response("search", {"queries": ["aurora calibration"]}),
                        response("reranker", {"set_id": "S001.q1"}), response()])
        def chat(_client, **kwargs):
            requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
            return next(replies)
        run = self.run_agent(self.config(), backend, chat)
        self.assertEqual(len(list((self.root / "working").rglob("*.txt"))), 1000)
        self.assertEqual(run.tool_call_breakdown, {"search": 1, "reranker": 1})
        self.assertEqual(run.reranker_summary["scored_passages"], 100)
        self.assertIn("doc0998.txt", run.surfaced_relpaths)
        trace = json.loads((self.root / "trace" / "single.json").read_text())
        ranking = trace["turns"][1]["tool_calls"][0]
        self.assertEqual(ranking["result_docids"][0], "998")
        self.assertEqual(ranking["result_paths"][0], "doc0998.txt")
        self.assertIn("reranker", [tool["function"]["name"] for tool in requests[0]["tools"]])
        self.assertIn("top 1000", requests[0]["messages"][0]["content"])
        received = requests[2]["messages"][-1]["content"]
        self.assertEqual(received, ranking["result_text"])
        self.assertEqual(backend.calls[0][0], "aurora calibration")

    def test_api_failure_restores_sets_cached_ranking_and_removes_future_state(self):
        backend = FakeBackend()
        replies = iter([response("search", {"queries": ["aurora calibration"]}),
                        response("reranker", {"set_id": "S001.q1"}),
                        RuntimeError("transient 500")])
        def first_chat(_client, **_kwargs):
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item
        checkpoint = self.root / "checkpoint.json"
        run = self.run_agent(self.config(), backend, first_chat, checkpoint)
        self.assertTrue(run.terminated_by.startswith("api_error"), run.terminated_by)
        state = json.loads(checkpoint.read_text())
        self.assertIn("S001.q1", state["tool_state"]["sets"])
        self.assertEqual(len(state["tool_state"]["views"]["S001.q1"]["items"]), 100)
        self.assertEqual(len(backend.calls), 1)
        (self.root / "working" / "future.txt").write_text("future state")
        second_backend = FakeBackend()
        replies = iter([response("reranker", {"set_id": "S001.q1", "offset": 10}), response()])
        resumed = self.run_agent(self.config(), second_backend,
                                lambda _client, **_kwargs: next(replies), checkpoint)
        self.assertFalse((self.root / "working" / "future.txt").exists())
        self.assertEqual(second_backend.calls, [])
        self.assertEqual(second_backend.tokenizer_calls, 0)
        self.assertEqual(resumed.reranker_summary["cache_hits"], 1)
        self.assertEqual(resumed.tool_call_breakdown, {"search": 1, "reranker": 2})

    def test_disabled_agent_never_constructs_backend_or_advertises_reranker(self):
        requests = []
        config = self.config(False)
        before = config.to_hashable_dict()
        def chat(_client, **kwargs):
            requests.append(kwargs)
            return response()
        with patch("rise.reranker_backend.QwenReranker") as backend, \
             patch("rise.agent.chat_with_retry", side_effect=chat):
            run = run_rise_agent(question="question", query_id=1, run_config=config,
                searcher=object(), doc_ids=list(self.mapping), docid_to_relpath=self.mapping,
                bc_plus_docs_root=self.corpus, working_dir=self.root / "working", client=object())
        backend.assert_not_called()
        self.assertEqual(config.to_hashable_dict(), before)
        self.assertEqual(run.reranker_summary, {})
        self.assertEqual([tool["function"]["name"] for tool in requests[0]["tools"]], ["search", "bash", "read"])

    def test_direct_agent_fills_missing_overlap_identity_before_checkpoint_hash(self):
        config = self.config()
        checkpoint = self.root / "checkpoint.json"
        run = self.run_agent(config, FakeBackend(),
                             lambda _client, **_kwargs: response(), checkpoint)
        self.assertEqual(run.terminated_by, "done")
        expected = {"reranker_passage_overlap_tokens": 32,
                    "reranker_passage_stride_tokens": 96,
                    "reranker_chunking_version": "token-overlap-v2"}
        for key, value in expected.items():
            self.assertEqual(config.extras[key], value)
        from rise.trajectory import compute_run_config_hash
        state = json.loads(checkpoint.read_text())
        self.assertEqual(state["identity"]["run_config_hash"],
                         compute_run_config_hash(config.to_hashable_dict()))

    def test_direct_agent_rejects_old_chunk_or_prompt_contract_before_model_calls(self):
        old_settings = ({"reranker_passage_overlap_tokens": 0},
                        {"reranker_passage_stride_tokens": 128},
                        {"reranker_chunking_version": "token-disjoint-v1"},
                        {"reranker_prompt_version": "v10-reranker-first-v2"})
        for settings in old_settings:
            with self.subTest(settings=settings):
                config = self.config()
                config.extras.update(settings)
                with patch("rise.reranker_backend.QwenReranker") as backend, \
                     patch("rise.agent.chat_with_retry") as chat:
                    with self.assertRaises(ValueError):
                        run_rise_agent(question="question", query_id=1, run_config=config,
                            searcher=object(), doc_ids=list(self.mapping),
                            docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                            working_dir=self.root / "working", client=object())
                    backend.assert_not_called()
                    chat.assert_not_called()

    def test_emitted_prompt_matches_available_tools_in_all_document_modes(self):
        for structured_docs in (False, True):
            for no_bash in (False, True):
                with self.subTest(structured=structured_docs, no_bash=no_bash):
                    config = self.config()
                    config.extras.update(structured_doc_mode=structured_docs, no_bash=no_bash)
                    requests = []
                    def chat(_client, **kwargs):
                        requests.append(kwargs)
                        return response()
                    backend = FakeBackend()
                    self.run_agent(config, backend, chat)
                    prompt = requests[0]["messages"][0]["content"]
                    names = {tool["function"]["name"] for tool in requests[0]["tools"]}
                    self.assertEqual(names, {"search", "read", "reranker"} |
                                     (set() if no_bash else {"bash"}))
                    self.assertIn(f"You have {'three' if no_bash else 'four'} tools:", prompt)
                    self.assertIn("prefer reranker", prompt)
                    self.assertIn("may read that source directly", prompt)
                    self.assertIn("do not need to rerank every search", prompt)
                    self.assertIn("After two actions with no new evidence", prompt)
                    self.assertNotIn("repeat until confident", prompt)
                    self.assertNotIn("An additional tool", prompt)
                    self.assertEqual("- bash(command)" in prompt, not no_bash)
                    self.assertEqual("The TOC is a navigation index" in prompt, structured_docs)
                    self.assertIn("top 1000", prompt)
                    self.assertIn("top 10", prompt)
                    self.assertEqual(config.extras["reranker_prompt_version"], RERANKER_PROMPT_VERSION)
                    self.assertEqual(backend.calls, [])

    def test_disabled_emitted_prompts_equal_frozen_baseline_in_all_document_modes(self):
        source = ROOT.parent / "baseline_snapshot/src/rise/rise_agent.py"
        frozen = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id.startswith("RISE_SYSTEM_PROMPT")}
        for structured_docs in (False, True):
            for no_bash in (False, True):
                with self.subTest(structured=structured_docs, no_bash=no_bash):
                    config = self.config(False)
                    config.extras.update(structured_doc_mode=structured_docs, no_bash=no_bash)
                    requests = []
                    def chat(_client, **kwargs):
                        requests.append(kwargs)
                        return response()
                    self.run_agent(config, FakeBackend(), chat)
                    name = "RISE_SYSTEM_PROMPT" + ("_STRUCTURED" if structured_docs else "")
                    name += "_NOBASH" if no_bash else ""
                    self.assertEqual(requests[0]["messages"][0]["content"], frozen[name])
                    self.assertNotIn("reranker_prompt_version", config.extras)


if __name__ == "__main__":
    unittest.main()
