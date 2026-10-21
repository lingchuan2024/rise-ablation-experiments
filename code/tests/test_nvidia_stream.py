"""Offline checks that NVIDIA SSE turns reconstruct full chat responses."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rise.api_retry import TerminatedResponseError, chat_with_retry


def chunk(*, delta=None, finish=None, usage=None):
    return ChatCompletionChunk.model_validate({
        "id": "response-1",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "choices": ([{"index": 0, "delta": delta or {}, "finish_reason": finish}]
                    if delta is not None or finish is not None else []),
        "usage": usage,
    })


class FakeStream:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class NvidiaStreamTests(unittest.TestCase):
    def client(self, responses):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return responses.pop(0)

        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))), requests

    def request(self, client, *, stream=True, retries=1):
        return chat_with_retry(
            client,
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[{"role": "user", "content": "Find a source"}],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
            max_tokens=32_000,
            temperature=0.7,
            stream_nvidia=stream,
            max_retries=retries,
        )

    def test_reassembles_reasoning_tool_calls_and_usage(self):
        stream = FakeStream([
            chunk(delta={"role": "assistant", "reasoning": "Need "}),
            chunk(delta={"reasoning_content": "sources.", "tool_calls": [
                {"index": 0, "id": "call-a", "type": "function",
                 "function": {"name": "search", "arguments": '{"q":'}},
                {"index": 1, "id": "call-b", "type": "function",
                 "function": {"name": "read", "arguments": '{"p":'}},
            ]}),
            chunk(delta={"content": "Searching", "tool_calls": [
                {"index": 0, "function": {"arguments": '"abc"}'}},
                {"index": 1, "function": {"arguments": '"file"}'}},
            ]}),
            chunk(delta={}, finish="tool_calls"),
            chunk(usage={"prompt_tokens": 100, "completion_tokens": 20,
                         "total_tokens": 120,
                         "prompt_tokens_details": {"cached_tokens": 5},
                         "completion_tokens_details": {"reasoning_tokens": 9}}),
        ])
        client, requests = self.client([stream])
        response = self.request(client)
        self.assertTrue(stream.closed)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual(response.choices[0].message.content, "Searching")
        self.assertEqual(response.choices[0].message.reasoning_content, "Need sources.")
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        calls = response.choices[0].message.tool_calls
        self.assertEqual([(call.id, call.function.name, call.function.arguments)
                          for call in calls], [
            ("call-a", "search", '{"q":"abc"}'),
            ("call-b", "read", '{"p":"file"}'),
        ])
        self.assertEqual(response.usage.prompt_tokens, 100)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 5)
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 9)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["stream_options"], {"include_usage": True})
        self.assertEqual(requests[0]["max_tokens"], 32_000)
        self.assertEqual(requests[0]["temperature"], 0.7)

    def test_interrupted_stream_retries_whole_turn(self):
        broken = FakeStream([chunk(delta={"content": "partial"}), RuntimeError("lost connection")])
        complete = FakeStream([
            chunk(delta={"content": "answer"}, finish="stop"),
            chunk(usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}),
        ])
        client, requests = self.client([broken, complete])
        with patch("rise.api_retry.time.sleep"):
            response = self.request(client, retries=2)
        self.assertEqual(response.choices[0].message.content, "answer")
        self.assertEqual(len(requests), 2)
        self.assertTrue(broken.closed)
        self.assertTrue(complete.closed)

    def test_missing_terminal_usage_is_rejected(self):
        stream = FakeStream([chunk(delta={"content": "answer"}, finish="stop")])
        client, _ = self.client([stream])
        with self.assertRaises(TerminatedResponseError):
            self.request(client)

    def test_default_json_transport_is_unchanged(self):
        answer = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="answer", tool_calls=[]))])
        client, requests = self.client([answer])
        self.assertIs(self.request(client, stream=False), answer)
        self.assertNotIn("stream", requests[0])
        self.assertNotIn("stream_options", requests[0])


if __name__ == "__main__":
    unittest.main()
