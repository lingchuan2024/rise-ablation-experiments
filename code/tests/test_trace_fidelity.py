"""Tool responses in a saved trace match the text the agent received."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.agent import Agent, _tool_returned_error


class FakeToolCall:
    def __init__(self, call_id: str, name: str, args: dict):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def response(*, text: str = "", tool_calls=None):
    calls = tool_calls or []
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=calls, reasoning_content=""),
            finish_reason="tool_calls" if calls else "stop",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class TraceFidelityTests(unittest.TestCase):
    def test_search_and_read_results_are_full_and_tool_errors_are_marked(self):
        search_text = (
            "search (1 query): 1 new doc added; working directory now contains 1 doc.\n"
            "  1. [12.34] history/example.txt\n"
            "long preview: " + "x" * 600
        )
        read_text = "1\t" + "文" * 600 + "\n2\tlast line"
        missing_text = "Error: file not found: 'history/missing.txt'"
        tool_calls = [
            FakeToolCall("search-1", "search", {"queries": ["example"]}),
            FakeToolCall("read-1", "read", {"file_path": "history/example.txt"}),
            FakeToolCall("read-2", "read", {"file_path": "history/missing.txt"}),
        ]

        def read_tool(args):
            return read_text if args["file_path"] == "history/example.txt" else missing_text

        replies = iter([response(tool_calls=tool_calls), response(text="answer")])
        requests = []

        def fake_chat(_client, **kwargs):
            requests.append([dict(message) for message in kwargs["messages"]])
            return next(replies)

        agent = Agent(
            client=object(), model="test-model",
            tools={"search": lambda _args: search_text, "read": read_tool},
            tool_specs=[], max_turns=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text("old trace", encoding="utf-8")
            os.chmod(trace_path, 0o644)
            with patch("rise.agent.chat_with_retry", side_effect=fake_chat):
                run = agent.run(
                    system_prompt="system", user_prompt="question",
                    trace_path=trace_path,
                    trace_relpath_to_docid={"history/example.txt": "42"},
                )

            calls = json.loads(trace_path.read_text(encoding="utf-8"))["turns"][0]["tool_calls"]
            self.assertEqual([call["result_text"] for call in calls],
                             [search_text, read_text, missing_text])
            self.assertEqual([call["result_text_preview"] for call in calls],
                             [text[:200] for text in (search_text, read_text, missing_text)])
            self.assertEqual([call["result_chars"] for call in calls],
                             [len(text) for text in (search_text, read_text, missing_text)])
            self.assertEqual(calls[0]["result_paths"], ["history/example.txt"])
            self.assertEqual(calls[0]["result_docids"], ["42"])
            self.assertEqual(calls[1]["result_paths"], ["history/example.txt"])
            self.assertEqual(calls[1]["result_docids"], ["42"])
            self.assertEqual(calls[2]["result_paths"], ["history/missing.txt"])
            self.assertEqual(calls[2]["result_docids"], [])
            self.assertEqual([call["error"] for call in calls],
                             ["", "", "tool_returned_error"])
            self.assertEqual(run.turns[0].tool_calls[2].error, "tool_returned_error")
            self.assertEqual([message["content"] for message in requests[1][-3:]],
                             [search_text, read_text, missing_text])
            self.assertEqual(stat.S_IMODE(trace_path.stat().st_mode), 0o600)

    def test_error_marker_after_tool_notice(self):
        self.assertEqual(
            _tool_returned_error("Note: legacy query format\nError: search called without queries."),
            "tool_returned_error",
        )
        self.assertEqual(_tool_returned_error("1\tThis text discusses Error: codes."), "")


if __name__ == "__main__":
    unittest.main()
