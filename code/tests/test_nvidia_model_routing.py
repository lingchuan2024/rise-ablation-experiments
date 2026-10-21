"""Offline NVIDIA Catalog routing and judge parameter checks."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class NvidiaRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("openai") is not None:
            return
        # Routing tests never issue API requests. A minimal SDK surface lets
        # them run even in the system Python without project dependencies.
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = type("OpenAI", (), {})
        for name in (
            "APIConnectionError", "APIError", "APITimeoutError", "BadRequestError",
            "InternalServerError", "NotFoundError", "PermissionDeniedError",
            "RateLimitError", "AuthenticationError",
        ):
            setattr(openai_stub, name, type(name, (Exception,), {}))
        sdk_patch = patch.dict(sys.modules, {"openai": openai_stub})
        sdk_patch.start()
        cls.addClassCleanup(sdk_patch.stop)

    def test_client_routes_catalog_ids_to_nvidia(self):
        decompose = importlib.import_module("rise.decompose")
        models = (
            "nvidia/nemotron-3-super-120b-a12b",
            "z-ai/glm-5.3",
            "moonshotai/kimi-k2.5",
            "deepseek-ai/deepseek-v3.2",
        )
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nv-key"}, clear=True):
            with patch.object(decompose, "OpenAI") as client:
                for model in models:
                    with self.subTest(model=model):
                        decompose.make_client(model)
                        client.assert_called_with(
                            api_key="test-nv-key",
                            base_url="https://integrate.api.nvidia.com/v1",
                        )
                self.assertEqual(client.call_count, len(models))

    def test_qwen_uses_token_plan_credential_and_endpoint(self):
        decompose = importlib.import_module("rise.decompose")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-plan-key", "NVIDIA_API_KEY": "wrong-provider"}, clear=True):
            with patch.object(decompose, "OpenAI") as client:
                decompose.make_client("qwen3.8-flash")
                client.assert_called_once_with(
                    api_key="test-plan-key",
                    base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                    max_retries=0,
                )
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-plan-key", "DASHSCOPE_BASE_URL": "https://example.invalid/v1"}, clear=True):
            with patch.object(decompose, "OpenAI") as client:
                decompose.make_client("qwen3.8-flash")
                client.assert_called_once_with(api_key="test-plan-key", base_url="https://example.invalid/v1", max_retries=0)
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
                decompose.make_client("qwen3.8-flash")

    def test_local_qwen35_models_use_local_endpoint(self):
        decompose = importlib.import_module("rise.decompose")
        with patch.dict(os.environ, {"RISE_LOCAL_VLLM_BASE_URL": "http://127.0.0.1:8000/v1"}, clear=True):
            for model in ("qwen3.5-27b", "Qwen3.5-35B-A3B"):
                with self.subTest(model=model), patch.object(decompose, "OpenAI") as client:
                    decompose.make_client(model)
                    client.assert_called_once_with(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1", timeout=1200.0, max_retries=0)

    def test_catalog_base_url_override_and_missing_key(self):
        decompose = importlib.import_module("rise.decompose")
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nv-key", "NVIDIA_BASE_URL": "https://example.invalid/v1"}, clear=True):
            with patch.object(decompose, "OpenAI") as client:
                decompose.make_client("nvidia/nemotron-3-super-120b-a12b")
                client.assert_called_once_with(api_key="test-nv-key", base_url="https://example.invalid/v1")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA_API_KEY"):
                decompose.make_client("z-ai/glm-5.3")

    def test_catalog_multiple_keys_constructs_ten_primary_four_reserve_pool(self):
        decompose = importlib.import_module("rise.decompose")
        env = {"NVIDIA_API_KEYS": ",".join(f"fake-key-{i}" for i in range(19))}
        with patch.dict(os.environ, env, clear=True), patch.object(decompose, "OpenAI") as sdk:
            client = decompose.make_client("nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(client.primary_count, 10)
        self.assertEqual(client.spare_count, 4)
        self.assertEqual(sdk.call_count, 14)
        self.assertEqual(
            sdk.call_args_list[0].kwargs,
            {"api_key": "fake-key-0", "base_url": "https://integrate.api.nvidia.com/v1", "max_retries": 0},
        )

    def test_optional_tcp_maxseg_is_applied_to_each_nvidia_client(self):
        if importlib.util.find_spec("httpx") is None:
            self.skipTest("httpx is needed to inspect the optional NVIDIA transport")
        decompose = importlib.import_module("rise.decompose")
        env = {
            "NVIDIA_API_KEYS": ",".join(f"fake-key-{i}" for i in range(14)),
            "NVIDIA_TCP_MAXSEG": "1200",
        }
        with patch.dict(os.environ, env, clear=True), \
             patch.object(decompose, "OpenAI") as sdk, \
             patch("httpx.HTTPTransport") as transport_type, \
             patch("httpx.Client", side_effect=lambda **kwargs: object()) as client_type:
            decompose.make_client("nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(sdk.call_count, 14)
        self.assertEqual(transport_type.call_count, 14)
        self.assertEqual(client_type.call_count, 14)
        for call in client_type.call_args_list:
            timeout = call.kwargs["timeout"]
            self.assertEqual((timeout.connect, timeout.write, timeout.read, timeout.pool),
                             (15.0, 30.0, 600.0, 15.0))
        for call in transport_type.call_args_list:
            self.assertEqual(
                call.kwargs["socket_options"],
                [(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1200)],
            )
        clients = [call.kwargs["http_client"] for call in sdk.call_args_list]
        self.assertEqual(len({id(client) for client in clients}), 14)
        self.assertTrue(all("http_client" in call.kwargs for call in sdk.call_args_list))

    def test_invalid_tcp_maxseg_is_rejected_without_creating_clients(self):
        decompose = importlib.import_module("rise.decompose")
        for value in ("abc", "0", "65536"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nv-key", "NVIDIA_TCP_MAXSEG": value}, clear=True), \
                 patch.object(decompose, "OpenAI") as sdk:
                with self.assertRaisesRegex(ValueError, "NVIDIA_TCP_MAXSEG"):
                    decompose.make_client("nvidia/nemotron-3-super-120b-a12b")
                sdk.assert_not_called()

    def test_existing_provider_routes_remain_unchanged(self):
        decompose = importlib.import_module("rise.decompose")
        env = {
            "XIAOMI_API_KEY": "test-xiaomi",
            "DEEPSEEK_API_KEY": "test-deepseek",
            "OPENAI_DIRECT_API_KEY": "test-openai",
            "NVIDIA_API_KEY": "test-nvidia",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(decompose, "OpenAI") as client:
            for model, key, url in (
                ("mimo-v2.5-pro", "test-xiaomi", "https://api.xiaomimimo.com/v1"),
                ("deepseek-chat", "test-deepseek", "https://api.deepseek.com/v1"),
                ("gpt-5.4-mini", "test-openai", "https://api.openai.com/v1"),
            ):
                with self.subTest(model=model):
                    decompose.make_client(model)
                    client.assert_called_with(api_key=key, base_url=url)

    def test_nvidia_and_glm_reasoners_are_classified(self):
        retry = importlib.import_module("rise.api_retry")
        for model in (
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "z-ai/glm-5.3",
        ):
            with self.subTest(model=model):
                self.assertTrue(retry.is_reasoner_model(model))
                self.assertFalse(retry.is_responses_api_model(model))

    def test_glm_judge_uses_low_reasoning_and_reasoner_output_budget(self):
        # The judge function does not need BM25; stub its optional retrieval
        # imports so this isolated check works without a local corpus setup.
        retrieval_stub = types.ModuleType("rise.retrieval")
        retrieval_stub.retrieve = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {
            "bm25s": types.ModuleType("bm25s"),
            "Stemmer": types.ModuleType("Stemmer"),
            "rise.retrieval": retrieval_stub,
        }):
            judge_module = importlib.import_module("rise.bcp_retrieval_agent")
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content='{"extracted_final_answer":"A","reasoning":"match","correct":"yes","confidence":100}'
            ))],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )
        with patch.object(judge_module, "chat_with_retry", return_value=response) as chat:
            result = judge_module.bcp_judge("Q", "A", "A", client=object(), model="z-ai/glm-5.3")
        self.assertEqual(result["correct"], "yes")
        self.assertEqual(chat.call_args.kwargs["reasoning_effort"], "low")
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 4096)
        self.assertNotIn("temperature", chat.call_args.kwargs)
        self.assertEqual(chat.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_qwen_judge_normalizes_boolean_correct_label(self):
        retrieval_stub = types.ModuleType("rise.retrieval")
        retrieval_stub.retrieve = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {
            "bm25s": types.ModuleType("bm25s"),
            "Stemmer": types.ModuleType("Stemmer"),
            "rise.retrieval": retrieval_stub,
        }):
            judge_module = importlib.import_module("rise.bcp_retrieval_agent")
        for raw_label, expected in (("true", "yes"), (False, "no")):
            with self.subTest(raw_label=raw_label):
                response = types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                        content=json.dumps({"extracted_final_answer": "A", "reasoning": "checked", "correct": raw_label, "confidence": 95})
                    ))],
                    usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                )
                with patch.object(judge_module, "chat_with_retry", return_value=response):
                    result = judge_module.bcp_judge("Q", "A", "A", client=object(), model="qwen3.8-flash")
                self.assertEqual(result["correct"], expected)


if __name__ == "__main__":
    unittest.main()
