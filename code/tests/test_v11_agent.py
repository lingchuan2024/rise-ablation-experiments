"""Exercise real accumulated-pool ranking and the v11 Agent/tool contract."""
from __future__ import annotations

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

from rise.accumulated_space import AccumulatedSpaceManager
from rise.protocol import for_rise
from rise.rise_agent import run_rise_agent
from rise.space_tools import SpaceTools, space_tool_specs
from test_reranker_tool import FakeBackend, structured
from test_v10_agent import response


class V11ToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        (self.pool / "a.txt").write_text(structured("aurora evidence first entity"))
        (self.pool / "b.txt").write_text(structured("second entity education university"))
        self.backend = FakeBackend()
        self.manager = AccumulatedSpaceManager(self.pool, self.root / "views",
                                              self.backend, candidates=1000, top_k=100)
        self.manager.register_search(["last search unrelated university"], [[("a.txt", 2.)]])
        self.tools = SpaceTools(self.manager)

    def call(self, name, args):
        return json.loads(getattr(self.tools, name)(args))

    def test_explicit_goal_first_refresh_then_cached_shell_only_selected_view(self):
        missing = self.call("dci", {"command": "ls"})
        self.assertEqual(missing["status"], "error")
        self.assertFalse(self.backend.calls)
        factory_calls = []
        def factory(root, **kwargs):
            factory_calls.append((root, kwargs))
            return lambda args: "(command succeeded, no output)"
        with patch("rise.space_tools.make_bash_tool", side_effect=factory):
            prepared = self.call("dci", {"goal_query": "aurora first entity", "command": "ls"})
            self.assertFalse(prepared["command_executed"])
            self.assertFalse(factory_calls)
            self.assertEqual(self.backend.calls[0][0], "aurora first entity")
            vid = prepared["view_id"]
            command = self.call("dci", {"view_id": vid, "command": "ls"})
            self.assertTrue(command["command_executed"])
            self.assertTrue(command["ranking_cache_hit"])
            self.assertEqual(len(self.backend.calls), 1)
            self.assertEqual(factory_calls[0][0], self.manager.view_root(vid))
            self.assertNotEqual(factory_calls[0][0], self.pool)
            self.assertTrue(factory_calls[0][1]["enable_sandbox"])
            self.manager.register_search(["second"], [[("b.txt", 1.)]])
            refreshed = self.call("dci", {"view_id": vid, "command": "ls"})
            self.assertFalse(refreshed["command_executed"])
            self.assertEqual(refreshed["trigger_reason"], "pool_changed")
            self.assertEqual(refreshed["pool_documents"], 2)
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(len(self.backend.calls), 2)

    def test_goal_and_k_switch_reuses_rank_but_requires_new_view_acknowledgement(self):
        first = self.call("dci", {"goal_query": "aurora", "top_k": 1})
        self.manager.register_search(["duplicate"], [[("a.txt", 1.)]])
        repeated = self.call("dci", {})
        self.assertEqual(repeated["view_id"], first["view_id"])
        self.assertEqual(len(self.backend.calls), 1)
        larger = self.call("dci", {"top_k": 2, "view_id": first["view_id"], "command": "ls"})
        self.assertFalse(larger["command_executed"])
        self.assertEqual(larger["trigger_reason"], "top_k_changed")
        self.assertEqual(len(self.backend.calls), 1)
        goal = self.call("dci", {"goal_query": "education university", "command": "ls"})
        self.assertEqual(goal["trigger_reason"], "goal_changed")
        self.assertFalse(goal["command_executed"])
        self.assertEqual(len(self.backend.calls), 2)

    def test_read_source_expansion_and_persistent_receipts(self):
        view = self.call("dci", {"goal_query": "aurora"})
        passage = view["items"][0]
        read = self.call("read", {"view_id": view["view_id"], "passage_id": passage["passage_id"]})
        self.assertEqual(read["passage"]["text"], passage["text"])
        expanded = self.call("expand_source", {"source_id": passage["source_id"],
            "char_start": passage["char_start"], "char_end": passage["char_end"]})
        self.assertEqual(expanded["text"], passage["text"])
        unsafe = self.call("expand_source", {"file_path": "../outside.txt"})
        self.assertEqual(unsafe["status"], "error")
        unknown = self.call("read", {"view_id": view["view_id"], "passage_id": "unknown"})
        self.assertEqual(unknown["status"], "error")
        self.call("dci", {"goal_query": "different goal"})
        self.assertEqual(self.manager.summary["receipts"], 2)

    def test_long_single_line_character_ranges_are_complete_and_bounded(self):
        body = "longline " + "X" * 21000 + " final_fact"
        (self.pool / "c.txt").write_text(body)
        self.manager.register_search(["longline"], [[("c.txt", 1.)]])
        first = self.call("expand_source", {"file_path": "c.txt"})
        self.assertEqual(first["text"], body[:8000])
        second = self.call("expand_source", {"file_path": "c.txt", "char_start": first["next_char_start"]})
        third = self.call("expand_source", {"file_path": "c.txt", "char_start": second["next_char_start"]})
        self.assertEqual(first["text"] + second["text"] + third["text"], body)
        too_big = self.call("expand_source", {"file_path": "c.txt", "char_end": 17000})
        self.assertEqual(too_big["status"], "error")

    def test_stale_view_checkpoint_restores_goal_and_refreshes_safely(self):
        prior = self.call("dci", {"goal_query": "aurora"})
        self.manager.register_search(["new"], [[("b.txt", 1.)]])
        state = self.tools.export_state()
        other_backend = FakeBackend()
        other_manager = AccumulatedSpaceManager(self.pool, self.root / "other_views", other_backend)
        other = SpaceTools(other_manager)
        other.restore_state(state)
        self.assertEqual(other.active_goal, "aurora")
        self.assertIsNone(other.active_view_id)
        refreshed = json.loads(other.dci({"view_id": prior["view_id"], "command": "ls"}))
        self.assertFalse(refreshed["command_executed"])
        self.assertEqual(refreshed["pool_documents"], 2)

    def test_no_bash_disables_commands_without_disabling_view_preparation(self):
        self.tools.no_bash = True
        denied = self.call("dci", {"goal_query": "aurora", "command": "ls"})
        self.assertEqual(denied["status"], "error")
        self.assertFalse(self.backend.calls)
        self.assertEqual(self.call("dci", {"goal_query": "aurora"})["status"], "ok")
        self.assertNotIn("command", space_tool_specs(no_bash=True)[0]["function"]["parameters"]["properties"])


class V11AgentIntegrationTests(unittest.TestCase):
    def test_direct_source_validation_does_not_load_or_score_reranker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "a.txt").write_text(structured("aurora exact answer"))
            config = for_rise(agent_model="test-model", enable_sandbox=True, max_model_calls=5, api_max_retries=1)
            config.extras["space_reranker_enabled"] = True
            backend = FakeBackend()
            replies = [response("search", {"queries": ["aurora"]}),
                       response("expand_source", {"file_path": "a.txt"}), response()]
            with patch("rise.tools.retrieve", return_value=[("1", 1.)]), \
                 patch("rise.reranker_backend.QwenReranker", return_value=backend), \
                 patch("rise.agent.chat_with_retry", side_effect=replies):
                run = run_rise_agent(question="aurora", query_id=1, run_config=config,
                    searcher=object(), doc_ids=["1"], docid_to_relpath={"1": "a.txt"},
                    bc_plus_docs_root=corpus, working_dir=root / "_working" / "1",
                    client=object())
            self.assertEqual(run.terminated_by, "done")
            self.assertEqual(run.tool_call_breakdown, {"search": 1, "expand_source": 1})
            self.assertEqual(backend.calls, [])
            self.assertEqual(backend.tokenizer_calls, 0)

    def test_real_turn_checkpoint_resumes_view_without_neural_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "a.txt").write_text(structured("aurora first debut"))
            config = for_rise(agent_model="test-model", enable_sandbox=True, max_model_calls=8, api_max_retries=1)
            config.extras["space_reranker_enabled"] = True
            checkpoint = root / "checkpoint.json"
            common = dict(question="aurora", query_id=1, run_config=config,
                searcher=object(), doc_ids=["1"], docid_to_relpath={"1": "a.txt"},
                bc_plus_docs_root=corpus, working_dir=root / "_working" / "1",
                client=object(), turn_checkpoint_path=checkpoint)
            initial_backend = FakeBackend()
            initial = [response("search", {"queries": ["aurora"]}),
                       response("dci", {"goal_query": "aurora first debut"}),
                       RuntimeError("fixture transport outage")]
            with patch("rise.tools.retrieve", return_value=[("1", 1.)]), \
                 patch("rise.reranker_backend.QwenReranker", return_value=initial_backend), \
                 patch("rise.agent.chat_with_retry", side_effect=initial):
                first = run_rise_agent(**common)
            self.assertTrue(first.terminated_by.startswith("api_error"))
            self.assertEqual(len(initial_backend.calls), 1)
            state = json.loads(checkpoint.read_text())
            vid = state["tool_state"]["active_view_id"]
            second_backend = FakeBackend()
            continued = [response("dci", {"view_id": vid, "command": "ls"}), response()]
            with patch("rise.reranker_backend.QwenReranker", return_value=second_backend), \
                 patch("rise.agent.chat_with_retry", side_effect=continued), \
                 patch("rise.space_tools.make_bash_tool", return_value=lambda args: "(command succeeded, no output)"):
                resumed = run_rise_agent(**common)
            self.assertEqual(resumed.terminated_by, "done")
            self.assertEqual(second_backend.calls, [])
            self.assertEqual(second_backend.tokenizer_calls, 0)
            self.assertEqual(resumed.reranker_summary["scoring_calls"], 1)
            self.assertEqual(resumed.reranker_summary["dci_command_calls"], 1)
            self.assertEqual(resumed.n_bm25_searches, 1)

    def test_actual_loop_imports_union_and_refreshes_without_executing_old_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            mapping = {str(i): f"doc{i:02d}.txt" for i in range(12)}
            for i, rel in mapping.items():
                (corpus / rel).write_text(structured("aurora first debut" if i == "0" else "history education document " + i))
            backend = FakeBackend(lambda q, texts: [5. if "aurora" in t else 0. for t in texts])
            config = for_rise(agent_model="test-model", max_model_calls=10, api_max_retries=1)
            config.extras["space_reranker_enabled"] = True
            requests, step = [], 0
            def chat(_client, **kwargs):
                nonlocal step
                step += 1
                requests.append(kwargs)
                if step == 1:
                    return response("search", {"queries": ["aurora"]})
                if step == 2:
                    return response("dci", {"goal_query": "aurora television debut"})
                if step == 3:
                    result = json.loads(kwargs["messages"][-1]["content"])
                    return response("read", result["items"][0]["read_args"])
                if step == 4:
                    return response("search", {"queries": ["new education evidence"]})
                if step == 5:
                    return response("dci", {"command": "ls", "view_id": "obsolete"})
                if step == 6:
                    result = json.loads(kwargs["messages"][-1]["content"])
                    self.assertFalse(result["command_executed"])
                    self.assertEqual(result["pool_documents"], 12)
                    return response("expand_source", {"file_path": "doc00.txt"})
                return response()
            results = [[(str(i), 12. - i) for i in range(6)],
                       [(str(i), 12. - i) for i in range(6, 12)]]
            with patch("rise.tools.retrieve", side_effect=results), \
                 patch("rise.reranker_backend.QwenReranker", return_value=backend), \
                 patch("rise.agent.chat_with_retry", side_effect=chat), \
                 patch("rise.space_tools.make_bash_tool") as shell:
                run = run_rise_agent(question="find first debut", query_id=1,
                    run_config=config, searcher=object(), doc_ids=list(mapping),
                    docid_to_relpath=mapping, bc_plus_docs_root=corpus,
                    working_dir=root / "_working" / "1", client=object(),
                    trace_path=root / "trace" / "single.json", map_keys=set(mapping.values()))
                shell.assert_not_called()
            self.assertEqual(run.terminated_by, "done")
            self.assertEqual(run.reranker_summary["pool_documents"], 12)
            self.assertEqual(run.reranker_summary["scoring_calls"], 2)
            self.assertEqual(run.n_bash_calls, 0)
            self.assertEqual(run.n_bm25_searches, 2)
            self.assertIn("doc00.txt", run.surfaced_relpaths)
            names = [s["function"]["name"] for s in requests[0]["tools"]]
            self.assertEqual(names, ["search", "dci", "read", "expand_source"])
            self.assertNotIn("reranker", names)
            self.assertEqual(len(list((root / "_working" / "1").rglob("*.txt"))), 12)
            self.assertTrue((root / "_views" / "1").exists())


if __name__ == "__main__":
    unittest.main()
