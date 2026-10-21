"""Offline checks for the scoped Ark Agent Plan adapter; no real credentials."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rise.api_retry import chat_with_retry
from rise.decompose import make_client


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
MODEL = "deepseek-v4.1-flash"


class ArkAdapterTests(unittest.TestCase):
    def _request(self, *, base_url=ARK_BASE_URL, model=MODEL, **kwargs):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="OK", tool_calls=[]),
        )])
        endpoint = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint)))
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": base_url}, clear=True):
            result = chat_with_retry(
                client, model=model, messages=kwargs.pop("messages", []),
                max_tokens=32000, max_retries=1, **kwargs,
            )
        self.assertIs(result, response)
        endpoint.assert_called_once()
        return endpoint.call_args.kwargs

    def test_ark_defaults_enable_thinking_and_high_effort(self):
        request = self._request()
        self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["max_tokens"], 32000)
        self.assertNotIn("max_completion_tokens", request)

    def test_explicit_thinking_and_effort_override_are_preserved(self):
        extra_body = {"thinking": {"type": "disabled"}, "metadata": {"test": True}}
        request = self._request(extra_body=extra_body, reasoning_effort="none")
        self.assertEqual(request["extra_body"], extra_body)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertIsNot(request["extra_body"], extra_body)
        self.assertEqual(extra_body, {"thinking": {"type": "disabled"}, "metadata": {"test": True}})

    def test_ark_does_not_mutate_caller_extra_body_or_history(self):
        extra_body = {"metadata": {"experiment": "offline"}}
        messages = [{"role": "assistant", "content": "", "reasoning_content": "fixture-reasoning",
                     "tool_calls": [{"id": "call_test", "type": "function", "function": {
                         "name": "lookup", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call_test", "content": "fixture-result"}]
        request = self._request(extra_body=extra_body, messages=messages)
        self.assertEqual(extra_body, {"metadata": {"experiment": "offline"}})
        self.assertEqual(request["extra_body"]["metadata"], extra_body["metadata"])
        self.assertEqual(request["messages"], messages)
        self.assertEqual(request["messages"][0]["reasoning_content"], "fixture-reasoning")

    def test_other_providers_do_not_receive_ark_parameters(self):
        for base_url in ("https://api.deepseek.com/v1", "https://example.invalid/v1", ""):
            with self.subTest(base_url=base_url):
                request = self._request(base_url=base_url)
                self.assertNotIn("extra_body", request)
                self.assertNotIn("reasoning_effort", request)

    def test_other_models_at_ark_do_not_receive_v41_defaults(self):
        for model in ("deepseek-v4-pro", "mimo-v2.5", "qwen3.8-flash"):
            with self.subTest(model=model):
                request = self._request(model=model)
                self.assertNotIn("extra_body", request)
                self.assertNotIn("reasoning_effort", request)

    def test_alibaba_retains_its_own_thinking_parameter(self):
        request = self._request(base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(request["extra_body"], {"enable_thinking": True})
        self.assertNotIn("reasoning_effort", request)

    def test_ark_client_disables_nested_retry_and_sets_timeout(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-fixture-key",
                                    "DEEPSEEK_BASE_URL": ARK_BASE_URL}, clear=True), \
                patch("rise.decompose.OpenAI") as sdk:
            client = make_client(MODEL)
        sdk.assert_called_once_with(api_key="offline-fixture-key", base_url=ARK_BASE_URL,
                                    max_retries=0, timeout=180)
        self.assertIs(client, sdk.return_value)

    def test_direct_deepseek_client_remains_unchanged(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-fixture-key"}, clear=True), \
                patch("rise.decompose.OpenAI") as sdk:
            make_client(MODEL)
        sdk.assert_called_once_with(api_key="offline-fixture-key", base_url="https://api.deepseek.com/v1")

    def test_ark_missing_credential_fails_without_creating_client(self):
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": ARK_BASE_URL}, clear=True), \
                patch("rise.decompose.OpenAI") as sdk:
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                make_client(MODEL)
        sdk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
