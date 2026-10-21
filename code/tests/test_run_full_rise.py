"""Synthetic checks for the local full-RISE entry point; no corpus or API."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_full_rise.py"
SPEC = importlib.util.spec_from_file_location("run_full_rise", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
from rise.protocol import for_rise
from rise.decompose import make_client
from rise.trajectory import compute_run_config_hash


class FullRiseLocalTests(unittest.TestCase):
    def test_retry_budget_is_recorded_in_run_identity(self):
        base = for_rise(agent_model="nvidia/nemotron-3-super-120b-a12b")
        robust = for_rise(agent_model="nvidia/nemotron-3-super-120b-a12b", api_max_retries=5)
        self.assertEqual(base.api_max_retries, 3)
        self.assertEqual(robust.api_max_retries, 5)
        self.assertNotEqual(
            compute_run_config_hash(base.to_hashable_dict()),
            compute_run_config_hash(robust.to_hashable_dict()),
        )

    def test_local_qwen_context_budget_and_endpoint(self):
        base = for_rise(agent_model="qwen3.5-27b")
        local = for_rise(agent_model="qwen3.5-27b", per_call_max_tokens=8192)
        self.assertEqual(local.per_call_max_tokens, 8192)
        self.assertNotEqual(
            compute_run_config_hash(base.to_hashable_dict()),
            compute_run_config_hash(local.to_hashable_dict()),
        )
        with patch.dict(os.environ, {"RISE_LOCAL_VLLM_BASE_URL": "http://127.0.0.1:8000/v1"}, clear=True), \
             patch("rise.decompose.OpenAI") as openai_client, patch.object(runner, "_load_env_file"):
            make_client("qwen3.5-27b")
            self.assertEqual(openai_client.call_args.kwargs["base_url"], "http://127.0.0.1:8000/v1")
            runner._require_key("qwen3.5-27b")

    def test_nvidia_catalog_model_uses_nvidia_key(self):
        models = (
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "z-ai/glm-5.3",
            "moonshotai/kimi-k2.5",
            "deepseek-ai/deepseek-v3.2",
        )
        with patch.object(runner, "_load_env_file"), patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
            for model in models:
                with self.subTest(model=model):
                    runner._require_key(model)
        with patch.object(runner, "_load_env_file"), patch.dict(os.environ, {"OPENAI_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(SystemExit, "NVIDIA_API_KEY"):
                runner._require_key(models[0])

    def test_qwen_requires_token_plan_key(self):
        with patch.object(runner, "_load_env_file"), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            runner._require_key("qwen3.8-flash")
        with patch.object(runner, "_load_env_file"), patch.dict(os.environ, {"NVIDIA_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(SystemExit, "DASHSCOPE_API_KEY"):
                runner._require_key("qwen3.8-flash")

    def test_provider_model_id_has_single_default_output_directory(self):
        model = "nvidia/nemotron-3-super-120b-a12b"
        output = runner._default_run_root(model)
        self.assertEqual(output.parent, runner.ROOT / "runs")
        self.assertEqual(output, runner._default_run_root(model))
        self.assertNotIn("/", output.name)
        self.assertNotEqual(output, runner._default_run_root(model.replace("/", "_")))

    def test_existing_model_default_output_directory_is_stable(self):
        self.assertEqual(
            runner._default_run_root("gpt-5.4-mini"),
            runner.ROOT / "runs" / "rise_gpt-5.4-mini_100k",
        )

    def test_input_manifest_blocks_reuse_with_changed_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            runner._pin_inputs(run, {"queries_sha256": "one"})
            runner._pin_inputs(run, {"queries_sha256": "one"})
            with self.assertRaisesRegex(SystemExit, "inputs changed"):
                runner._pin_inputs(run, {"queries_sha256": "two"})

    def test_complete_trajectory_set_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = root / "queries.jsonl"
            queries.write_text("\n".join(
                json.dumps({"query_id": qid}) for qid in (10, 20)
            ) + "\n", encoding="utf-8")
            run = root / "run"
            (run / "_per_query").mkdir(parents=True)
            for qid in (10, 20):
                (run / "_per_query" / f"qid_{qid}.json").write_text(
                    json.dumps({"query_id": qid}), encoding="utf-8"
                )
                trace = run / "_traces" / f"qid_{qid}" / "single.json"
                trace.parent.mkdir(parents=True)
                trace.write_text("{}", encoding="utf-8")
            runner._check_trajectories(run, queries, 2)
            (run / "_traces" / "qid_20" / "single.json").unlink()
            with self.assertRaisesRegex(SystemExit, "missing traces"):
                runner._check_trajectories(run, queries, 2)


if __name__ == "__main__":
    unittest.main()
