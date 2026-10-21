"""Token Plan quota exhaustion must stop retries, unlike transient throttling."""
from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rise.api_retry import chat_with_retry


def _error(message: str) -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))
    return RateLimitError(message, response=response, body={"code": message.split(":", 1)[0]})


class QwenQuotaTests(unittest.TestCase):
    def _pipeline(self):
        path = Path(__file__).resolve().parents[1] / "scripts/complete_qwen_rise.py"
        spec = importlib.util.spec_from_file_location("complete_qwen_rise", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_pipeline_distinguishes_token_throttle_from_monthly_exhaustion(self):
        module = self._pipeline()
        self.assertFalse(module._has_quota_error(
            "429 insufficient_quota: Allocated quota exceeded, please increase your quota limit."
        ))
        self.assertTrue(module._has_quota_error(
            "429 insufficient_quota: Your token-plan quota has been exhausted."
        ))

    def test_moderation_failure_is_recorded_without_blocking_other_queries(self):
        module = self._pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "_checkpoints"
            checkpoints.mkdir()
            (checkpoints / "qid_1.json").write_text(json.dumps({
                "run": {"error": "BadRequestError: data_inspection_failed"}
            }))
            (checkpoints / "qid_2.json").write_text(json.dumps({
                "run": {"error": "RateLimitError: Requests rate limit exceeded"}
            }))
            self.assertEqual(module._record_moderation_blocks(root, ["1", "2"]), {"1"})
            self.assertEqual(module._blocked_ids(root), {"1"})

    def test_exhausted_quota_stops_without_retry(self):
        endpoint = Mock(side_effect=_error("insufficient_quota: Your token-plan quota has been exhausted."))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint)))
        with patch("rise.api_retry.time.sleep") as sleep:
            with self.assertRaises(RateLimitError):
                chat_with_retry(client, model="qwen3.8-flash", messages=[], max_retries=3)
        self.assertEqual(endpoint.call_count, 1)
        sleep.assert_not_called()

    def test_transient_rate_limit_still_retries(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="OK", tool_calls=[]),
        )])
        endpoint = Mock(side_effect=[_error("Requests rate limit exceeded"), response])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint)))
        with patch("rise.api_retry.time.sleep"):
            result = chat_with_retry(client, model="qwen3.8-flash", messages=[], max_retries=3)
        self.assertIs(result, response)
        self.assertEqual(endpoint.call_count, 2)

    def test_allocated_quota_with_insufficient_quota_code_is_transient(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="OK", tool_calls=[]),
        )])
        endpoint = Mock(side_effect=[_error("insufficient_quota: Allocated quota exceeded, please increase your quota limit."), response])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint)))
        with patch("rise.api_retry.time.sleep"):
            result = chat_with_retry(client, model="qwen3.8-flash", messages=[], max_retries=3)
        self.assertIs(result, response)
        self.assertEqual(endpoint.call_count, 2)

    def test_local_qwen35_requests_visible_thinking(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="OK", tool_calls=[]),
        )])
        endpoint = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint)))
        client.base_url = "http://127.0.0.1:8000/v1"
        client.post = Mock(return_value={"count": 12, "max_model_len": 100000})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"RISE_LOCAL_VLLM_BASE_URL": "http://127.0.0.1:8000/v1", "RISE_API_DIAGNOSTICS_DIR": tmp}):
            chat_with_retry(client, model="qwen3.5-27b", messages=[], max_retries=1)
        self.assertEqual(
            endpoint.call_args.kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )


if __name__ == "__main__":
    unittest.main()
