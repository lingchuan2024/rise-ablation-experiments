"""Offline integration checks for the strict experimental intervention.

Real source hardlinks, raw passage postings, BM25, Agent, SDK serialization,
local-Qwen adapter, checkpoints and RequestRecorder run here. Only neural
scores and the HTTP transport are fixtures; no network or GPU is used.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/tests"))

from strict_experiment import runtime
from strict_experiment.strict_prompt import (
    build_contract, build_prompt, official_tool_specs, official_user_prompt,
)
from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.rerank_v13 import RerankService
from rise.tools import make_bm25_search_tool, make_read_tool
import rise.tools as original_tools
from rise.trajectory import compute_run_config_hash
from test_indexed_accumulated_space import build_raw_index, CheapBackend
from test_reranker_tool import structured
from runtime_guards import SandboxPaused, StopState


def reply(name=None, arguments=None, *, text="Exact Answer: aurora", model=runtime.MODEL):
    calls = ([{"id": "call-" + name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}] if name else [])
    return {"id": "offline", "object": "chat.completion", "created": 1,
            "model": model, "choices": [{"index": 0,
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"role": "assistant", "content": "" if calls else text,
                            "tool_calls": calls or None}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10,
                      "total_tokens": 60,
                      "prompt_tokens_details": {"cached_tokens": 20},
                      "completion_tokens_details": {"reasoning_tokens": 4}}}


class StrictRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "corpus"
        self.mapping = {str(i): f"docs/doc{i:02d}.txt" for i in range(12)}
        self.documents = {
            path: structured("aurora calibration evidence" if did == "11"
                             else "history education document " + did)
            for did, path in self.mapping.items()}
        self.documents[self.mapping["0"]] += "\n" + "X" * 2400 + "CUTTAIL_SECRET\n"
        self.index = build_raw_index(self.root / "index", self.corpus, self.documents)
        self.results = [(did, float(12 - int(did))) for did in self.mapping]
        self.query = {"query_id": 1, "question": "find aurora calibration"}
        self.env = {
            "RISE_LOCAL_VLLM_BASE_URL": "http://strict-fixture/v1",
            "RISE_LOCAL_VLLM_CONTEXT_LENGTH": "262144",
            "RISE_LOCAL_VLLM_TOKEN_MARGIN": "256",
            "RISE_LOCAL_VLLM_MIN_OUTPUT_TOKENS": "1024",
            "RISE_LOCAL_VLLM_TIMEOUT": "1200",
            "RISE_QWEN35_PROFILE": "official-thinking-v1",
            "RISE_LOCAL_VLLM_SERVICE_SHA256": "offline-model-service-sha",
            "RISE_API_DIAGNOSTICS_DIR": str(self.root / "api-private")}
        environment = patch.dict(os.environ, self.env)
        environment.start()
        self.addCleanup(environment.stop)
        self.config = {
            "model": runtime.MODEL, "max_turns": 12, "per_call_max_tokens": 32000,
            "wall_clock_seconds": 3600, "api_max_retries": 1,
            "bm25_k": 1000, "bm25_top_n_preview": 10, "bm25_snippet_chars": 100,
            "read_default_limit": 2000, "runtime_environment": self.env,
            "base_url": self.env["RISE_LOCAL_VLLM_BASE_URL"],
            "corpus": str(self.corpus), "document_index": str(self.root / "doc-index"),
            "filename_map": str(self.root / "mapping.json"),
            "passage_index": str(self.root / "index")}
        self.requests = []
        self.health = StopState(self.root / "PAUSED.json", self.root / "events.jsonl")

    def service(self, working, backend=None, views="views", health=None):
        working.mkdir(parents=True, exist_ok=True)
        backend = backend or CheapBackend()
        if health is not None:
            backend = runtime.GuardedBackend(backend, health)
        manager = IndexedAccumulatedSpaceManager(
            working_dir=working, views_dir=self.root / views,
            backend=backend, candidates=1000, top_k=100,
            passage_index=self.index,
            runtime_to_corpus={path: path for path in self.mapping.values()})
        return RerankService(manager)

    def tools(self, arm, working, service=None):
        working.mkdir(parents=True, exist_ok=True)
        return runtime.build_tools(
            arm, config=self.config, searcher=object(), doc_ids=list(self.mapping),
            mapping=self.mapping, corpus=self.corpus, working_dir=working,
            rerank_service=service, enable_sandbox=True)

    def client(self, responses):
        def handle(request):
            body = json.loads(request.content)
            self.requests.append((request.url.path, copy.deepcopy(body)))
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 500, "max_model_len": 262144,
                                                 "tokens": []})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            item = responses(body) if callable(responses) else responses.pop(0)
            if isinstance(item, tuple):
                return httpx.Response(item[0], json=item[1])
            return httpx.Response(200, json=item)
        client = OpenAI(api_key="offline-fixture-key", base_url=self.config["base_url"],
                        max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close)
        return client

    def run_query(self, arm, client, *, config=None, output=None, backend=None,
                  expected=None):
        def make_service(_config, working, _runtime, _mapping, health=None):
            return self.service(working, backend=backend,
                                views="run-views-" + str(len(self.requests)), health=health)
        with patch.object(runtime, "resources", return_value=(
                object(), list(self.mapping), self.mapping)), \
             patch.object(runtime, "make_service", side_effect=make_service), \
             patch.object(runtime, "health_state", return_value=self.health), \
             patch("rise.decompose.make_client", return_value=client), \
             patch("rise.tools.retrieve", return_value=self.results):
            return runtime.run_strict_query(
                arm=arm, query=self.query, config=config or self.config,
                output_dir=output or self.root / "run", expected_contract=expected or build_contract(arm))

    def test_common_search_and_read_returns_equal_original_bytes(self):
        a_root, b_root = self.root / "A", self.root / "B"
        backend = CheapBackend()
        service = self.service(b_root, backend)
        tools_a, tools_b = self.tools("A", a_root), self.tools("B", b_root, service)
        reference = make_bm25_search_tool(
            searcher=object(), doc_ids=list(self.mapping), docid_to_relpath=self.mapping,
            bc_plus_docs_root=self.corpus, working_dir=self.root / "official",
            k=1000, top_n_preview=10, snippet_chars=100)
        with patch("rise.tools.retrieve", return_value=self.results):
            for _ in range(3):
                args = {"queries": ["aurora", "history"]}
                original = reference(args)
                self.assertEqual(tools_a["search"](args), original)
                self.assertEqual(tools_b["search"](args), original)
                for unexpected in ("Reranker set", "Search record", "Tool feedback",
                                   "Accumulated original-document pool", "S001.q1"):
                    self.assertNotIn(unexpected, original)
        self.assertEqual(backend.calls, [])
        for tool in (tools_a["read"], tools_b["read"]):
            output = tool({"file_path": self.mapping["0"]})
            self.assertEqual(output, make_read_tool(a_root)({"file_path": self.mapping["0"]}))
            self.assertIn("[line truncated", output)
            self.assertNotIn("CUTTAIL_SECRET", output)
        for working in (a_root, b_root):
            self.assertEqual(len(list(working.rglob("*.txt"))), 12)
            self.assertTrue(all(os.path.samefile(working / path, self.corpus / path)
                                for path in self.mapping.values()))

    def test_safe_shared_outputs_match_actual_pinned_official_implementation(self):
        # Importing this reference executes only ordinary module definitions;
        # prompt/schema extraction remains AST-only in strict_prompt.py.
        spec = importlib.util.spec_from_file_location(
            "rise.strict_frozen_official_tools", ROOT / "strict_experiment/upstream/tools.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        baseline_working = self.root / "frozen-official"
        original = official.make_bm25_search_tool(
            searcher=object(), doc_ids=list(self.mapping), docid_to_relpath=self.mapping,
            bc_plus_docs_root=self.corpus, working_dir=baseline_working,
            k=1000, top_n_preview=10, snippet_chars=100)
        a = self.tools("A", self.root / "A")
        b = self.tools("B", self.root / "B", self.service(self.root / "B"))
        with patch("rise.tools.retrieve", return_value=self.results), \
             patch.object(official, "retrieve", return_value=self.results):
            for queries in (["aurora"], ["history", "aurora"], ["history"]):
                arguments = {"queries": queries}
                expected = original(arguments)
                self.assertEqual(a["search"](arguments), expected)
                self.assertEqual(b["search"](arguments), expected)
        for tools, working in ((a, self.root / "A"), (b, self.root / "B")):
            official_reader = official.make_read_tool(working)
            for arguments in ({"file_path": "docs/doc00.txt"},
                              {"file_path": "docs/doc11.txt", "offset": 0, "limit": 60},
                              {"file_path": "docs/doc11.txt", "offset": 10, "limit": 1},
                              {"file_path": "docs/missing.txt"}):
                self.assertEqual(tools["read"](arguments), official_reader(arguments))

    def test_b_search_registration_callback_returns_none(self):
        working = self.root / "B"
        service = self.service(working)
        with patch.object(runtime, "make_bm25_search_tool", wraps=make_bm25_search_tool) as factory:
            tools = self.tools("B", working, service)
        callback = factory.call_args.kwargs["on_search_results"]
        with patch("rise.tools.retrieve", return_value=self.results):
            tools["search"]({"queries": ["aurora"]})
        imported = [[(self.mapping[did], score) for did, score in self.results]]
        self.assertIsNone(callback(["aurora"], imported))

    def test_b_rerank_can_find_unpreviewed_doc_without_mutating_pool(self):
        working = self.root / "B"
        backend = CheapBackend(lambda _query, texts: [9.0 if "aurora" in t else 0.0 for t in texts])
        service = self.service(working, backend)
        tools = self.tools("B", working, service)
        with patch("rise.tools.retrieve", return_value=self.results):
            output = tools["search"]({"queries": ["aurora"]})
        self.assertNotIn("docs/doc11.txt", output)
        before = {p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in working.rglob("*.txt")}
        result = json.loads(tools["rerank"]({"query": "aurora calibration",
                                              "bm25_candidates": 12, "top_k": 1, "limit": 1}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["items"][0]["file_path"], "docs/doc11.txt")
        self.assertIn("aurora calibration evidence", tools["read"](result["items"][0]["read_args"]))
        after = {p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in working.rglob("*.txt")}
        self.assertEqual(before, after)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.tokenizer_calls, 0)

    def test_a_has_no_rerank_and_common_schemas_are_exactly_official(self):
        tools = self.tools("A", self.root / "A")
        self.assertEqual(list(tools), ["search", "bash", "read"])
        with self.assertRaises(KeyError):
            tools["rerank"]
        self.assertEqual(build_contract("A")["tool_specs"], official_tool_specs())
        self.assertEqual(build_contract("B")["tool_specs"][:3], official_tool_specs())
        for arm, service in (("A", object()), ("B", None)):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.tools(arm, self.root / arm, service)

    def test_shell_is_original_factory_and_preserves_rg_command_and_private_tmpfs(self):
        self.assertIs(runtime.make_bash_tool, original_tools.make_bash_tool)
        command = "rg -r 'replacement' -n pattern . > /tmp/persisted"
        commands = []
        def spawn(cmd, **kwargs):
            commands.append((cmd, kwargs))
            return SimpleNamespace(communicate=lambda timeout: ("", ""), returncode=0, pid=123456)
        with patch.object(original_tools.sys, "platform", "linux"), \
             patch.object(original_tools.subprocess, "Popen", side_effect=spawn), \
             patch.object(original_tools.os, "killpg"):
            for arm in ("A", "B"):
                working = self.root / arm
                service = self.service(working, views="shell-views") if arm == "B" else None
                tools = self.tools(arm, working, service)
                self.assertEqual(tools["bash"]({"command": command}), "(command succeeded, no output)")
        self.assertEqual(len(commands), 2)
        for cmd, kwargs in commands:
            self.assertEqual(cmd[-3:], ["/bin/bash", "-c", command])
            self.assertIn(["--tmpfs", "/tmp"], [cmd[i:i+2] for i in range(len(cmd)-1)])
            self.assertIn(["--tmpfs", "/"], [cmd[i:i+2] for i in range(len(cmd)-1)])
            self.assertIn("--unshare-net", cmd)
            self.assertTrue(kwargs["start_new_session"])
        # Only the per-arm root differs; no argument or tool behavior changes.
        left = [part.replace(str(self.root / "A"), "<working>") for part in commands[0][0]]
        right = [part.replace(str(self.root / "B"), "<working>") for part in commands[1][0]]
        self.assertEqual(left, right)

    def test_real_sdk_agent_adapter_and_recorder_preserve_contracts_and_count_usage(self):
        for arm in ("A", "B"):
            with self.subTest(arm=arm):
                step = 0
                def respond(body):
                    nonlocal step
                    step += 1
                    self.assertEqual(body["messages"][0]["content"], build_prompt(arm))
                    self.assertEqual(body["tools"], build_contract(arm)["tool_specs"])
                    self.assertEqual(body["temperature"], 1.0)
                    self.assertEqual(body["top_p"], .95)
                    self.assertEqual(body["presence_penalty"], 1.5)
                    self.assertEqual(body["max_tokens"], 32000)
                    self.assertEqual(body["tool_choice"], "auto")
                    self.assertEqual(body["top_k"], 20)
                    self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
                    if step == 1:
                        return reply("search", {"queries": ["aurora"]})
                    if arm == "B" and step == 2:
                        self.assertNotIn("docs/doc11.txt", body["messages"][-1]["content"])
                        self.assertNotIn("Tool feedback", body["messages"][-1]["content"])
                        return reply("rerank", {"query": "aurora calibration",
                                                "bm25_candidates": 12, "top_k": 1, "limit": 1})
                    if arm == "B" and step == 3:
                        result = json.loads(body["messages"][-1]["content"])
                        self.assertEqual(result["items"][0]["file_path"], "docs/doc11.txt")
                        return reply("read", result["items"][0]["read_args"])
                    if arm == "A" and step == 2:
                        return reply("read", {"file_path": "docs/doc11.txt"})
                    self.assertIn("aurora calibration evidence", body["messages"][-1]["content"])
                    return reply()
                out = self.root / ("run-" + arm)
                backend = CheapBackend(lambda _q, texts: [9. if "aurora" in t else 0. for t in texts])
                result = self.run_query(arm, self.client(respond), output=out, backend=backend)
                expected_turns = 3 if arm == "A" else 4
                self.assertEqual(result["terminated_by"], "done")
                self.assertEqual(result["n_turns"], expected_turns)
                self.assertEqual(result["input_tokens"], 50 * expected_turns)
                self.assertEqual(result["cached_input_tokens"], 20 * expected_turns)
                self.assertEqual(result["output_tokens"], 10 * expected_turns)
                self.assertEqual(result["reasoning_tokens"], 4 * expected_turns)
                self.assertTrue(result["execution_checks"]["contract_verified"])
                contract = json.loads((out / "request_contracts/qid_1.json").read_text())
                self.assertEqual(contract["system_prompt"], build_prompt(arm))
                self.assertEqual(contract["tool_specs"], build_contract(arm)["tool_specs"])
                self.assertEqual(contract["user_prompt"], official_user_prompt().format(question=self.query["question"]))
                rows = [json.loads(line) for line in (out / "api_ledgers/qid_1.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), expected_turns)
                self.assertTrue(all(r["status"] == "ok" and r["reported_model"] == runtime.MODEL for r in rows))
                self.assertNotIn("offline-fixture-key", json.dumps(rows))
                self.assertEqual(len(backend.calls), 0 if arm == "A" else 1)

    def test_recorder_rejects_modified_model_prompt_tools_or_question_before_request(self):
        client = self.client([])
        recorder = runtime.RequestRecorder(client, arm="A", query=self.query,
            config=self.config, contract=build_contract("A"), output_dir=self.root / "recorder")
        valid = {"model": runtime.MODEL, "messages": [
            {"role": "system", "content": build_prompt("A")},
            {"role": "user", "content": official_user_prompt().format(question=self.query["question"])}],
            "tools": build_contract("A")["tool_specs"]}
        variants = []
        bad = copy.deepcopy(valid); bad["model"] = "qwen3.8-flash"; variants.append(bad)
        bad = copy.deepcopy(valid); bad["messages"][0]["content"] += "extra"; variants.append(bad)
        bad = copy.deepcopy(valid); bad["tools"][0]["function"]["description"] += "extra"; variants.append(bad)
        bad = copy.deepcopy(valid); bad["messages"][1]["content"] = "other question"; variants.append(bad)
        for request in variants:
            with self.subTest(request=request), self.assertRaises(ValueError):
                recorder.chat.completions.create(**request)
        self.assertEqual(self.requests, [])
        self.assertEqual(recorder._shared["requests"], 0)

    def test_reported_model_mismatch_fails_closed_and_is_audited(self):
        with self.assertRaises(SandboxPaused):
            self.run_query("A", self.client([reply(model="qwen3.8-flash")]))
        self.assertTrue(self.health.paused)
        checkpoint = json.loads((self.root / "run/_checkpoints/qid_1.json").read_text())
        self.assertEqual(checkpoint["run"]["terminated_reason"], "api_error")
        self.assertEqual(checkpoint["run"]["final_answer"], "")
        rows = [json.loads(line) for line in (self.root / "run/api_ledgers/qid_1.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["reported_model"], "qwen3.8-flash")

    def test_infrastructure_wrapper_preserves_ordinary_tool_errors_without_pause(self):
        arguments = {"command": "rg missing ."}
        cases = (("bash", "(no matches found)"),
                 ("bash", "[exit=1]\n[stderr]\nrg: regex parse error: unclosed group"),
                 ("bash", "Error: command must be a nonempty string"),
                 ("read", "Error: file not found: 'docs/missing.txt'."),
                 ("rerank", '{"status":"error","code":"invalid_arguments","error":"file_paths must be an array"}'))
        for name, expected in cases:
            with self.subTest(name=name, expected=expected):
                seen = []
                def original(value):
                    seen.append(value)
                    return expected
                wrapped = runtime.wrap_infrastructure_tools({name: original}, self.health)
                self.assertEqual(wrapped[name](arguments), expected)
                self.assertIs(seen[0], arguments)
                self.assertFalse(self.health.paused)
                self.assertFalse(self.health.pause_path.exists())

    def test_bwrap_and_spawn_failures_pause_but_completed_return_is_unchanged(self):
        cases = ("[exit=1]\n[stderr]\nbwrap: Creating new namespace failed: Operation not permitted",
                 "Error: failed to spawn bash: executable is unavailable")
        for i, expected in enumerate(cases):
            with self.subTest(expected=expected):
                health = StopState(self.root / f"pause-{i}.json", self.root / f"events-{i}.jsonl")
                calls = []
                def original(arguments):
                    calls.append(arguments)
                    return expected
                wrapped = runtime.wrap_infrastructure_tools({"bash": original}, health)
                self.assertEqual(wrapped["bash"]({"command": "rg needle ."}), expected)
                self.assertTrue(health.paused)
                self.assertEqual(json.loads(health.pause_path.read_text())["first_failure"]["kind"],
                                 "sandbox_infrastructure_failed")
                with self.assertRaises(SandboxPaused):
                    wrapped["bash"]({"command": "second command"})
                self.assertEqual(len(calls), 1)

    def test_guarded_backend_pauses_on_oom_and_tokenizer_failure_not_input_errors(self):
        from rise.reranker_backend import InputTooLongError
        errors = (RuntimeError("CUDA out of memory"),
                  ValueError("invalid scoring parameter"),
                  InputTooLongError("input exceeds model budget"))
        for i, error in enumerate(errors):
            with self.subTest(error=type(error).__name__):
                health = StopState(self.root / f"backend-pause-{i}.json", self.root / f"backend-events-{i}.jsonl")
                def score(*_args, **_kwargs):
                    raise error
                backend = runtime.GuardedBackend(SimpleNamespace(score=score), health)
                with self.assertRaises(type(error)) as caught:
                    backend.score("query", ["passage"])
                self.assertIs(caught.exception, error)
                self.assertEqual(health.paused, isinstance(error, RuntimeError) and not isinstance(error, InputTooLongError))
        error = ValueError("tokenizer initialization failed")
        def tokenizer():
            raise error
        guarded = runtime.GuardedBackend(SimpleNamespace(get_tokenizer=tokenizer), self.health)
        with self.assertRaises(ValueError):
            guarded.get_tokenizer()
        self.assertTrue(self.health.paused)

    def test_diagnostic_storage_error_pauses_without_replacing_tool_return(self):
        expected = '{"status":"error","code":"diagnostic_log_error"}'
        tools = runtime.wrap_infrastructure_tools({"rerank": lambda args: expected}, self.health)
        self.assertEqual(tools["rerank"]({"query": "aurora"}), expected)
        self.assertTrue(self.health.paused)
        self.assertEqual(json.loads(self.health.pause_path.read_text())["first_failure"]["kind"],
                         "reranker_diagnostic_storage_failed")

    def test_request_recorder_model_drift_pauses_and_blocks_next_request(self):
        client = self.client([reply(model="qwen3.8-flash")])
        recorder = runtime.RequestRecorder(client, arm="A", query=self.query,
            config=self.config, contract=build_contract("A"), output_dir=self.root / "guarded-recorder",
            health=self.health)
        request = {"model": runtime.MODEL, "messages": [
            {"role": "system", "content": build_prompt("A")},
            {"role": "user", "content": official_user_prompt().format(question=self.query["question"])}],
            "tools": build_contract("A")["tool_specs"]}
        with self.assertRaisesRegex(ValueError, "Reported model"):
            recorder.chat.completions.create(**request)
        self.assertTrue(self.health.paused)
        self.assertEqual(recorder._shared["verified"], 0)
        with self.assertRaises(SandboxPaused):
            recorder.with_options(max_retries=0).chat.completions.create(**request)
        self.assertEqual(len(self.requests), 1)

    def test_resume_changed_config_or_arm_rejects_before_another_request(self):
        self.run_query("A", self.client([reply()]))
        count = len(self.requests)
        changed = copy.deepcopy(self.config); changed["max_turns"] += 1
        self.assertNotEqual(compute_run_config_hash(runtime.run_config_for("A", self.config).to_hashable_dict()),
                            compute_run_config_hash(runtime.run_config_for("A", changed).to_hashable_dict()))
        for arm, config in (("A", changed), ("B", self.config)):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "Checkpoint identity changed"):
                self.run_query(arm, self.client([]), config=config)
        self.assertEqual(len(self.requests), count)

    def test_b_checkpoint_resume_pages_cached_ranking_without_rescoring(self):
        backend = CheapBackend()
        first = [reply("search", {"queries": ["aurora"]}),
                 reply("rerank", {"query": "aurora calibration", "bm25_candidates": 12,
                                  "top_k": 2, "limit": 1}),
                 (500, {"error": {"message": "fixture outage", "type": "server_error"}})]
        initial = self.run_query("B", self.client(first), backend=backend)
        self.assertEqual(initial["terminated_by"], "api_error")
        checkpoint = self.root / "run/_checkpoints/qid_1.json"
        state = json.loads(checkpoint.read_text())
        rank = json.loads(next(m["content"] for m in state["messages"]
                               if m.get("tool_call_id") == "call-rerank"))
        fresh_backend = CheapBackend()
        second = [reply("rerank", {"result_id": rank["result_id"], "offset": 1, "limit": 1}), reply()]
        resumed = self.run_query("B", self.client(second), backend=fresh_backend)
        self.assertEqual(resumed["terminated_by"], "done")
        self.assertEqual(resumed["tool_call_breakdown"], {"search": 1, "rerank": 2})
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(fresh_backend.calls, [])
        self.assertEqual(resumed["reranker_summary"]["scoring_calls"], 1)

    def test_changed_expected_contract_rejected_before_model_or_resource_access(self):
        bad = build_contract("A"); bad["system_prompt"] += "extra"
        with patch.object(runtime, "resources") as resources, \
             patch("rise.decompose.make_client") as client:
            with self.assertRaisesRegex(ValueError, "Frozen system_prompt"):
                runtime.run_strict_query(arm="A", query=self.query, config=self.config,
                    output_dir=self.root / "bad", expected_contract=bad)
        resources.assert_not_called()
        client.assert_not_called()

    def test_native_preflight_forces_only_probe_read_then_none_and_uses_no_cohort_question(self):
        def shell(_root, **_kwargs):
            def execute(arguments):
                command = arguments["command"]
                if command == "printf RISE_STRICT_SANDBOX_READY":
                    return "RISE_STRICT_SANDBOX_READY"
                if command == "printf RISE_TMP_PROBE >/tmp/rise-strict-probe":
                    return "(command succeeded, no output)"
                if command == "test ! -e /tmp/rise-strict-probe && printf FRESH_TMP":
                    return "FRESH_TMP"
                if command == "rg --version | head -n 1":
                    return "ripgrep fixture-version"
                self.fail("Unexpected preflight shell command")
            return execute
        def respond(body):
            choice = body["tool_choice"]
            self.assertEqual(body["max_tokens"], 2048)
            question = body["messages"][1]["content"]
            self.assertIn("Compatibility probe only:", question)
            self.assertNotEqual(question, official_user_prompt().format(question=self.query["question"]))
            if isinstance(choice, dict):
                self.assertEqual(choice, {"type": "function", "function": {"name": "read"}})
                return reply("read", {"file_path": "docs/doc00.txt", "offset": 0, "limit": 1})
            self.assertEqual(choice, "none")
            self.assertEqual(body["messages"][-1]["role"], "tool")
            self.assertIn("=== DOCUMENT HEADER ===", body["messages"][-1]["content"])
            return reply(text="RISE_QWEN35_READY")
        client = self.client(respond)
        backend = CheapBackend()
        def make_service(_config, working, _views, _mapping, health=None):
            return self.service(working, backend=backend, views="preflight-views", health=health)
        with patch.object(runtime, "ROOT", self.root), \
             patch.object(runtime, "resources", return_value=(object(), list(self.mapping), self.mapping)), \
             patch.object(runtime, "make_service", side_effect=make_service), \
             patch.object(runtime, "make_bash_tool", side_effect=shell), \
             patch("rise.decompose.make_client", return_value=client), \
             patch("rise.tools.retrieve", return_value=self.results):
            result = runtime.preflight(config=self.config,
                expected_contracts={arm: build_contract(arm) for arm in ("A", "B")})
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["shared_tools"].values()))
        self.assertEqual(result["cohort_questions_used"], 0)
        self.assertEqual(len(backend.calls), 1)
        completion_requests = [body for path, body in self.requests if path == "/v1/chat/completions"]
        self.assertEqual(len(completion_requests), 4)
        self.assertEqual([body["tool_choice"] for body in completion_requests], [
            {"type": "function", "function": {"name": "read"}}, "none",
            {"type": "function", "function": {"name": "read"}}, "none"])
        for arm in ("A", "B"):
            self.assertTrue(result["native_model"][arm]["model_verified"])
            self.assertTrue(result["native_model"][arm]["tool_call_round_trip"])
            self.assertTrue(result["native_model"][arm]["final_complete"])


if __name__ == "__main__":
    unittest.main()
