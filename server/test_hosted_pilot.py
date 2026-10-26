from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hosted_pilot as hosted
sys.path.insert(0, str(hosted.ROOT / "code/src"))


def config():
    return {"allowed_reported_models": [hosted.MODEL], "thinking_mode": "enabled",
        "preserve_thinking": True, "reasoning_effort": "medium", "request_timeout_seconds": 600,
        "wall_clock_seconds_per_question": 3600,
        "runtime_environment": {"DASHSCOPE_BASE_URL": hosted.ENDPOINT, "REASONING_EFFORT": "medium"}}


def response(model=hosted.MODEL, *, content="OK", calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls or [], reasoning_content=None)
    return SimpleNamespace(model=model, id="mock-response", usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")])


class FakeClient:
    base_url = hosted.ENDPOINT
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item
    def with_options(self, **kwargs):
        return self


class HostedClientTests(unittest.TestCase):
    def make(self, root, client, settings=None):
        state = hosted.guards.StopState(root / "pause.json", root / "events.jsonl")
        adapter = hosted.bound_client_class(settings or config())(client, hosted.MODEL, hosted.ENDPOINT, root / "transport.jsonl", state)
        return adapter, state

    def test_exact_api_shape_total_budget_thinking_and_no_message_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([response()])
            adapter, state = self.make(root, client)
            adapter.chat.completions.create(model=hosted.MODEL, messages=[{"role": "user", "content": "private prompt"}], max_tokens=32000)
            request = client.requests[0]
            self.assertNotIn("max_tokens", request)
            self.assertEqual(request["max_completion_tokens"], 32000)
            self.assertEqual(request["reasoning_effort"], "medium")
            self.assertEqual(request["extra_body"], {"enable_thinking": True, "preserve_thinking": True})
            ledger = (root / "transport.jsonl").read_text()
            self.assertNotIn("private prompt", ledger)
            self.assertNotIn("messages", ledger)
            self.assertEqual(json.loads(ledger)["requested_max_completion_tokens"], 32000)
            self.assertFalse(state.paused)

    def test_unallowed_model_pauses_and_explicit_snapshot_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, state = self.make(root, FakeClient([response("unexpected-model")]))
            with self.assertRaises(hosted.guards.SandboxPaused):
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            self.assertTrue(state.paused)
            settings = config()
            settings["allowed_reported_models"].append("qwen3.8-flash-snapshot")
            adapter, state = self.make(root / "snapshot", FakeClient([response("qwen3.8-flash-snapshot")]), settings)
            self.assertEqual(adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32).model, "qwen3.8-flash-snapshot")
            self.assertFalse(state.paused)

    def test_credential_reflection_is_suppressed_and_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(hosted.os.environ, {hosted.SECRET_ENV: "mock-secret-value"}):
            root = Path(tmp)
            adapter, _ = self.make(root, FakeClient([RuntimeError("mock-secret-value reflected by server")]))
            with self.assertRaisesRegex(RuntimeError, "details suppressed") as failure:
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            self.assertNotIn("mock-secret-value", str(failure.exception))
            self.assertNotIn("mock-secret-value", (root / "transport.jsonl").read_text())

    def test_http_error_diagnostics_are_small_sanitized_fields(self):
        error = RuntimeError("SDK provider error")
        error.status_code = 400
        error.body = {"error": {"code": "BadArgument", "message": "token mock-secret-value unsupported flag"}, "headers": {"Authorization": "mock-secret-value"}, "private_request": "private prompt"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(hosted.os.environ, {hosted.SECRET_ENV: "mock-secret-value"}):
            root = Path(tmp)
            adapter, _ = self.make(root, FakeClient([error]))
            with self.assertRaises(RuntimeError):
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            ledger = (root / "transport.jsonl").read_text()
            row = json.loads(ledger)
            self.assertEqual(row["http_status"], 400)
            self.assertEqual(row["provider_error_code"], "BadArgument")
            self.assertIn("[REDACTED]", row["provider_error_message"])
            self.assertNotIn("mock-secret-value", ledger)
            self.assertNotIn("private prompt", ledger)
            self.assertNotIn("headers", ledger)

    def test_deadline_caps_timeout_and_prevents_expired_request(self):
        from rise.api_retry import RequestDeadlineExceeded
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeClient([response()])
            adapter, state = self.make(root, client)
            state.local.query_deadline = 110.0
            with patch.object(hosted.time, "monotonic", return_value=100.0):
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            self.assertEqual(client.requests[0]["timeout"], 10.0)
            state.local.query_deadline = 99.0
            with patch.object(hosted.time, "monotonic", return_value=100.0), self.assertRaises(RequestDeadlineExceeded):
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            self.assertEqual(len(client.requests), 1)
            self.assertFalse(state.paused)

    def test_completion_after_deadline_keeps_usage_but_is_not_delivered(self):
        from rise.api_retry import RequestDeadlineExceeded
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter, state = self.make(root, FakeClient([response()]))
            state.local.query_deadline = 110.0
            with patch.object(hosted.time, "monotonic", side_effect=[100.0, 100.0, 111.0, 111.0]), self.assertRaises(RequestDeadlineExceeded):
                adapter.chat.completions.create(model=hosted.MODEL, messages=[], max_tokens=32)
            row = json.loads((root / "transport.jsonl").read_text())
            self.assertEqual(row["usage"]["total_tokens"], 7)
            self.assertFalse(row["response_delivered"])
            self.assertFalse(state.paused)


class HostedPilotTests(unittest.TestCase):
    def test_prepare_freezes_first10_resources_and_config_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = root / "server/hosted_pilot"
            args = hosted.parser().parse_args(["prepare", "--index-dir", "/index", "--filename-map", "/map", "--corpus", "/corpus", "--passage-index", "/passages", "--reranker-device", "cpu", "--reranker-dtype", "float32"])
            with patch.object(hosted, "ROOT", root), patch.object(hosted, "PILOT", pilot), patch.object(hosted, "REPORTS", pilot / "reports"), patch.object(hosted, "source_hashes", return_value={"code.py": "frozen"}) as sources, patch.object(hosted, "resource_identity", return_value={"index": "frozen"}), patch.dict(hosted.os.environ, {hosted.SECRET_ENV: "mock-persist-secret"}):
                result = hosted.prepare(args)
                manifest, cfg = hosted.verify_manifest()
                self.assertEqual(result["question_runs"], 30)
                self.assertEqual(cfg["concurrency"], 3)
                self.assertEqual(len(manifest["query_ids"]), 10)
                self.assertEqual(cfg["reranker_sharing_policy"], hosted.SHARING_POLICY)
                self.assertNotIn("mock-persist-secret", json.dumps(manifest))
                self.assertNotIn(hosted.SECRET_ENV, cfg["runtime_environment"])
                sources.return_value = {"code.py": "changed"}
                with self.assertRaisesRegex(RuntimeError, "source identity changed"):
                    hosted.verify_manifest()

    def test_exact_first10_and_three_independent_variants_c3(self):
        self.assertEqual([row["query_id"] for row in hosted.validate_queries()[:10]], [70, 81, 96, 116, 122, 127, 132, 153, 156, 161])
        cfg = config()
        cfg.update(document_index="/index", filename_map="/map", corpus="/corpus", passage_index="/passages",
            reranker_model="Qwen/Qwen3-Reranker-0.6B", reranker_device="cpu", reranker_dtype="float32",
            arm_outputs={variant: "/out/" + variant for variant in hosted.VARIANTS})
        for variant in hosted.VARIANTS:
            argv = hosted.build_argv(cfg, {"queries": "/first10.jsonl"}, variant)
            self.assertEqual(argv[argv.index("--model") + 1], hosted.MODEL)
            self.assertEqual(argv[argv.index("--concurrency") + 1], "3")
            self.assertEqual(argv[argv.index("--ablation-variant") + 1], variant)
            names = {spec["function"]["name"] for spec in hosted.tool_specs(variant)}
            expected = {"search", "read"} | ({"bash"} if variant != "no_bash" else set()) | ({"rerank"} if variant != "auto_rerank" else set())
            self.assertEqual(names, expected)

    def test_stdin_secret_only_in_environment_and_no_unrelated_credentials_survive(self):
        with patch.dict(hosted.os.environ, {"OPENAI_API_KEY": "unrelated", "RISE_LOCAL_VLLM_BASE_URL": "http://wrong"}, clear=True):
            with patch.object(hosted.sys, "stdin", io.StringIO('{"api_key":"mock-secret"}\n')):
                hosted.secret_from_stdin()
            environment = hosted.runtime_environment(config())
            self.assertEqual(environment[hosted.SECRET_ENV], "mock-secret")
            self.assertNotIn("OPENAI_API_KEY", environment)
            self.assertNotIn("RISE_LOCAL_VLLM_BASE_URL", environment)

    def test_native_tool_roundtrip_each_variant_with_frozen_flags(self):
        call = SimpleNamespace(id="mock-call", function=SimpleNamespace(name="search", arguments='{"queries":["compatibility probe"]}'))
        call.model_dump = lambda: {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
        for variant in hosted.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp, patch.object(hosted, "REPORTS", Path(tmp)):
                client = FakeClient([response(content=None, calls=[call]), response(content="Compatibility OK")])
                result = hosted.native_probe(config(), variant, client)
                self.assertTrue(result["passed"])
                self.assertEqual(result["requests"], 2)
                self.assertTrue(all(request["max_completion_tokens"] == 8192 for request in client.requests))
                self.assertTrue(all(request["extra_body"]["preserve_thinking"] for request in client.requests))
                self.assertEqual(client.requests[0]["tool_choice"], "auto")
                self.assertEqual(client.requests[1]["tool_choice"], "none")

    def test_missing_live_preflight_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(hosted, "REPORTS", Path(tmp)), patch.object(hosted, "verify_manifest", return_value=({"manifest_sha256": "bound"}, config())), patch.object(hosted.subprocess, "run") as launch:
            with self.assertRaises(FileNotFoundError):
                hosted.run()
            launch.assert_not_called()

    def test_checkpoint_elapsed_budget_propagates_to_each_hosted_retry(self):
        import rise.agent as agent
        import rise.rise_agent as rise_agent
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            checkpoint.write_text('{"elapsed_seconds":3590}')
            state = hosted.guards.StopState(Path(tmp) / "pause.json", Path(tmp) / "events.jsonl")
            observed = {}
            def chat(*args, **kwargs):
                observed.update(kwargs)
                return "mock response"
            def run(*args, **kwargs):
                return agent.chat_with_retry(None, model=hosted.MODEL)
            with patch.object(agent, "chat_with_retry", chat), patch.object(rise_agent, "run_rise_agent", run), patch.object(hosted.time, "monotonic", return_value=100.0):
                hosted.install_deadline_hooks(state, config())
                self.assertEqual(rise_agent.run_rise_agent(turn_checkpoint_path=checkpoint), "mock response")
                self.assertEqual(observed["deadline"], 110.0)
                checkpoint.write_text('{"elapsed_seconds":-1}')
                with self.assertRaises(ValueError):
                    rise_agent.run_rise_agent(turn_checkpoint_path=checkpoint)


if __name__ == "__main__":
    unittest.main()
