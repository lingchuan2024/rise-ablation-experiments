"""A failed second API turn resumes without repeating completed tool work."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import BadRequestError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.agent import Agent, AgentRun, _is_context_limit_error
from rise.turn_checkpoint import TurnCheckpoint


class FakeToolCall:
    id = "call-1"
    function = SimpleNamespace(name="search", arguments='{"queries":["test"]}')

    def model_dump(self):
        return {
            "id": self.id, "type": "function",
            "function": {"name": "search", "arguments": self.function.arguments},
        }


def response(*, text: str = "", tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls or [], reasoning_content="")
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = next(self.replies)
        if isinstance(item, Exception):
            raise item
        return item


class TurnCheckpointTests(unittest.TestCase):
    def test_context_limit_after_checkpoint_is_terminal_and_traced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            workspace = root / "working"
            workspace.mkdir()
            checkpoint = TurnCheckpoint(
                root / "checkpoint.json",
                identity={"model": "nvidia/test", "query_id": "1", "run_config_hash": "abc"},
                working_dir=workspace, corpus_root=corpus, relpath_to_docid={},
            )
            tool_calls = []

            def search(_args):
                tool_calls.append(True)
                return "no matches"

            def agent(client):
                return Agent(
                    client=client, model="nvidia/test", tools={"search": search},
                    tool_specs=[], max_turns=4, max_retries=1,
                )

            # One completed tool turn is saved; a transient error leaves it
            # available to resume without repeating that tool invocation.
            first = agent(FakeClient([
                response(tool_calls=[FakeToolCall()]), RuntimeError("transient 500"),
            ])).run(system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint)
            self.assertEqual(first.terminated_reason, "api_error")
            self.assertEqual(len(tool_calls), 1)

            http_response = httpx.Response(
                400, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
            )
            context_error = BadRequestError(
                "messages are 1,083,103 tokens; model limit is 1,000,000",
                response=http_response,
                body={"error": {"code": "context_length_exceeded"}},
            )
            self.assertTrue(_is_context_limit_error(context_error))
            context_message_error = BadRequestError(
                "Maximum context length is 1,000,000 tokens",
                response=http_response, body={"error": {"code": "invalid_request_error"}},
            )
            self.assertTrue(_is_context_limit_error(context_message_error))
            trace_path = root / "trace.json"
            resumed_client = FakeClient([context_error])
            terminal = agent(resumed_client).run(
                system_prompt="system", user_prompt="question",
                trace_path=trace_path, turn_checkpoint=checkpoint,
            )
            self.assertEqual(terminal.terminated_reason, "context_limit")
            self.assertEqual(terminal.final_answer, "")
            self.assertEqual(len(terminal.turns), 1)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(len(resumed_client.calls[0]["messages"]), 4)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["terminated_by"], "context_limit")
            self.assertEqual(trace["final_text"], "")
            self.assertEqual(len(trace["turns"]), 1)
            self.assertEqual(json.loads(checkpoint.path.read_text(encoding="utf-8"))["run"]["terminated_reason"], "context_limit")

            # A process interrupted before committing its per-query row may
            # load this terminal checkpoint without making another API call.
            again = agent(FakeClient([])).run(
                system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
            )
            self.assertEqual(again.terminated_reason, "context_limit")
            self.assertEqual(len(tool_calls), 1)

            other_bad_request = BadRequestError(
                "Invalid tool schema", response=http_response,
                body={"error": {"code": "invalid_request_error", "message": "Invalid tool schema"}},
            )
            self.assertFalse(_is_context_limit_error(other_bad_request))
            self.assertFalse(_is_context_limit_error(RuntimeError("maximum context length")))

    def test_completed_coerce_is_checkpointed_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            workspace = root / "working"
            workspace.mkdir()
            checkpoint = TurnCheckpoint(
                root / "checkpoint.json",
                identity={"model": "nvidia/test", "query_id": "1", "run_config_hash": "abc"},
                working_dir=workspace, corpus_root=corpus, relpath_to_docid={},
            )

            def agent(client):
                return Agent(
                    client=client, model="nvidia/test", tools={}, tool_specs=[],
                    max_turns=0, max_retries=1, coerce_final_on_max_turns=True,
                )

            first_client = FakeClient([response(text="Exact Answer: A")])
            with patch("rise.agent.estimate_cost", return_value=0.0):
                first = agent(first_client).run(
                    system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
                )
            self.assertEqual(first.terminated_reason, "coerced_max_turns")
            self.assertEqual(len(first_client.calls), 1)
            saved = json.loads(checkpoint.path.read_text(encoding="utf-8"))
            self.assertEqual(saved["run"]["terminated_reason"], "coerced_max_turns")

            resumed_client = FakeClient([])
            resumed = agent(resumed_client).run(
                system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
            )
            self.assertEqual(resumed.final_answer, "Exact Answer: A")
            self.assertEqual(resumed.terminated_reason, "coerced_max_turns")
            self.assertEqual(resumed_client.calls, [])

    def test_resumed_run_keeps_the_original_wall_clock_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            workspace = root / "working"
            workspace.mkdir()
            checkpoint = TurnCheckpoint(
                root / "checkpoint.json",
                identity={"model": "nvidia/test", "query_id": "1", "run_config_hash": "abc"},
                working_dir=workspace, corpus_root=corpus, relpath_to_docid={},
            )
            checkpoint.save(
                AgentRun(model="nvidia/test", system_prompt="system", user_prompt="question"),
                [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
                elapsed_seconds=61.0,
            )
            client = FakeClient([response(text="should not be called")])
            run = Agent(
                client=client, model="nvidia/test", tools={}, tool_specs=[],
                max_turns=4, max_retries=1, wall_clock_timeout_sec=60,
                coerce_final_on_max_turns=False,
            ).run(system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint)
            self.assertEqual(run.terminated_reason, "wall_clock_timeout")
            self.assertEqual(client.calls, [])

    def test_api_failure_resumes_after_completed_tool_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            source = corpus / "site" / "doc.txt"
            source.parent.mkdir(parents=True)
            source.write_text("evidence", encoding="utf-8")
            workspace = root / "working"
            workspace.mkdir()
            checkpoint_path = root / "checkpoint.json"
            trace_path = root / "single.json"
            identity = {"model": "nvidia/test", "query_id": "1", "run_config_hash": "abc"}
            checkpoint = TurnCheckpoint(
                checkpoint_path, identity=identity, working_dir=workspace,
                corpus_root=corpus, relpath_to_docid={"site/doc.txt": "doc-1"},
            )
            tool_calls = []

            def search(_args):
                tool_calls.append(True)
                target = workspace / "site" / "doc.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                return "found site/doc.txt"

            def agent(client):
                return Agent(
                    client=client, model="nvidia/test", tools={"search": search},
                    tool_specs=[], max_turns=4, max_retries=1,
                )

            failed_client = FakeClient([response(tool_calls=[FakeToolCall()]), RuntimeError("transient 500")])
            with patch("rise.agent.estimate_cost", return_value=0.0):
                first = agent(failed_client).run(
                    system_prompt="system", user_prompt="question", trace_path=trace_path,
                    turn_checkpoint=checkpoint,
                )
            self.assertEqual(first.terminated_reason, "api_error")
            self.assertEqual(len(first.turns), 1)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(stat.S_IMODE(checkpoint_path.stat().st_mode), 0o600)
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["workspace_docs"], [{"relpath": "site/doc.txt", "docid": "doc-1"}])

            extra = workspace / "uncheckpointed.txt"
            extra.write_text("discard me", encoding="utf-8")
            wrong_identity = TurnCheckpoint(
                checkpoint_path, identity={**identity, "run_config_hash": "different"},
                working_dir=workspace, corpus_root=corpus,
                relpath_to_docid={"site/doc.txt": "doc-1"},
            )
            with self.assertRaisesRegex(ValueError, "identity changed"):
                wrong_identity.load()
            self.assertTrue(extra.exists())

            corrupt = dict(saved)
            corrupt["run"] = {**saved["run"], "system_prompt": "different"}
            checkpoint_path.write_text(json.dumps(corrupt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model or prompts"):
                agent(FakeClient([])).run(
                    system_prompt="system", user_prompt="question", turn_checkpoint=checkpoint,
                )
            self.assertTrue(extra.exists())
            checkpoint_path.write_text(json.dumps(saved), encoding="utf-8")

            resumed_client = FakeClient([response(text="Exact Answer: evidence")])
            with patch("rise.agent.estimate_cost", return_value=0.0):
                resumed = agent(resumed_client).run(
                    system_prompt="system", user_prompt="question", trace_path=trace_path,
                    turn_checkpoint=checkpoint,
                )
            self.assertEqual(resumed.terminated_reason, "done")
            self.assertEqual(resumed.final_answer, "Exact Answer: evidence")
            self.assertEqual(len(resumed.turns), 2)
            self.assertEqual(len(tool_calls), 1)
            self.assertFalse(extra.exists())
            self.assertTrue((workspace / "site" / "doc.txt").samefile(source))
            self.assertEqual(len(resumed_client.calls[0]["messages"]), 4)
            self.assertEqual(len(json.loads(trace_path.read_text(encoding="utf-8"))["turns"]), 2)


if __name__ == "__main__":
    unittest.main()
