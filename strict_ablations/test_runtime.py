"""Offline integration tests: genuine BM25/search/read/index and mocked inference.

No service, credentials, neural weights or benchmark questions are used.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import bm25s
import httpx
from openai import OpenAI
import Stemmer

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "code/src", ROOT / "code/tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from strict_ablations import runtime
from strict_ablations.prompt import ARMS, CONTROL, build_contract
from strict_experiment.strict_prompt import official_user_prompt
from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.rerank_v13 import RerankService
from rise.tools import make_bm25_search_tool, make_read_tool
from rise.trajectory import compute_run_config_hash
from runtime_guards import SandboxPaused, StopState
from test_indexed_accumulated_space import build_raw_index, CheapBackend
from test_reranker_tool import structured


def reply(name=None, arguments=None, *, text="Exact Answer: aurora", model=runtime.MODEL):
    calls = ([{"id": "call-" + name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}] if name else [])
    return {"id": "offline", "object": "chat.completion", "created": 1,
        "model": model, "choices": [{"index": 0,
            "finish_reason": "tool_calls" if calls else "stop",
            "message": {"role": "assistant", "content": "" if calls else text,
                        "tool_calls": calls or None}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 4}}}


class StrictAblationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "corpus"
        self.mapping = {str(i): f"docs/doc{i:02d}.txt" for i in range(12)}
        documents = {path: structured("aurora calibration evidence" if did == "11"
            else "aurora history education document " + did)
            for did, path in self.mapping.items()}
        documents[self.mapping["0"]] += "\nneedle " + "X" * 2400 + "\n"
        self.index = build_raw_index(self.root / "index", self.corpus, documents)
        self.searcher = bm25s.BM25()
        tokens = bm25s.tokenize(list(documents.values()), stopwords="en",
            stemmer=Stemmer.Stemmer("english"), show_progress=False)
        self.searcher.index(tokens, show_progress=False)
        self.env = {"RISE_LOCAL_VLLM_BASE_URL": "http://strict-ablation-fixture/v1",
            "RISE_LOCAL_VLLM_CONTEXT_LENGTH": "262144",
            "RISE_LOCAL_VLLM_TOKEN_MARGIN": "256",
            "RISE_LOCAL_VLLM_MIN_OUTPUT_TOKENS": "1024",
            "RISE_LOCAL_VLLM_TIMEOUT": "1200",
            "RISE_QWEN35_PROFILE": "official-thinking-v1",
            "RISE_LOCAL_VLLM_SERVICE_SHA256": "offline-service-sha",
            "RISE_API_DIAGNOSTICS_DIR": str(self.root / "private-api")}
        environment = patch.dict(os.environ, self.env)
        environment.start(); self.addCleanup(environment.stop)
        # Tiny cohort-local limits allow a real small document index; the
        # formal runner separately validates the frozen common config.
        self.config = {"model": runtime.MODEL, "max_turns": 12,
            "per_call_max_tokens": 32768, "wall_clock_seconds": 3600,
            "api_max_retries": 1, "bm25_k": 12, "bm25_top_n_preview": 10,
            "bm25_snippet_chars": 100, "read_default_limit": 2000,
            "runtime_environment": self.env, "base_url": self.env["RISE_LOCAL_VLLM_BASE_URL"],
            "corpus": str(self.corpus), "document_index": str(self.root / "doc-index"),
            "filename_map": str(self.root / "mapping.json"), "passage_index": str(self.root / "index")}
        self.query = {"query_id": 1, "question": "find aurora calibration"}
        self.health = StopState(self.root / "pause.json", self.root / "events.jsonl")
        self.requests = []

    def service(self, working, backend=None, views="views", health=None):
        working.mkdir(parents=True, exist_ok=True)
        backend = backend or CheapBackend()
        if health is not None:
            backend = runtime.GuardedBackend(backend, health)
        return RerankService(IndexedAccumulatedSpaceManager(
            working_dir=working, views_dir=self.root / views, backend=backend,
            candidates=1000, top_k=100, passage_index=self.index,
            runtime_to_corpus={path: path for path in self.mapping.values()}))

    def tools(self, arm, *, working=None, service=None, health=None, config=None):
        working = working or self.root / arm
        working.mkdir(parents=True, exist_ok=True)
        return runtime.build_tools(arm, config=config or self.config,
            searcher=self.searcher, doc_ids=list(self.mapping), mapping=self.mapping,
            corpus=self.corpus, working_dir=working,
            rerank_service=service or self.service(working, views="views-" + arm),
            health=health)

    def client(self, responses):
        def handle(request):
            body = json.loads(request.content)
            self.requests.append((request.url.path, copy.deepcopy(body)))
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 500, "max_model_len": 262144,
                                                 "tokens": []})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            value = responses(body) if callable(responses) else responses.pop(0)
            if isinstance(value, tuple):
                return httpx.Response(value[0], json=value[1])
            return httpx.Response(200, json=value)
        client = OpenAI(api_key="offline-fixture-key", base_url=self.config["base_url"],
            max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close)
        return client

    def run_query(self, arm, responses, *, config=None, backend=None, output=None,
                  expected=None, health=None):
        client = self.client(responses)
        def make_service(_config, working, _runtime, _mapping, health=None):
            return self.service(working, backend=backend, views="run-views-" + arm,
                                health=health)
        with patch.object(runtime, "resources", return_value=(self.searcher, list(self.mapping), self.mapping)), \
             patch.object(runtime, "make_service", side_effect=make_service), \
             patch.object(runtime, "health_state", return_value=health or self.health), \
             patch("rise.decompose.make_client", return_value=client):
            return runtime.run_query(arm=arm, query=self.query, config=config or self.config,
                output_dir=output or self.root / "run", expected_contract=expected or build_contract(arm),
                report_dir=self.root / "new-reports")

    def test_common_original_search_read_and_normalization_are_unchanged(self):
        for arguments in ({"queries": ["aurora"]}, {"queries": "aurora"},
                          {"query": "aurora"}, {"queries": '["aurora"]'}):
            reference_work = self.root / ("official-" + str(len(list(self.root.glob("official-*")))))
            reference = make_bm25_search_tool(searcher=self.searcher, doc_ids=list(self.mapping),
                docid_to_relpath=self.mapping, bc_plus_docs_root=self.corpus,
                working_dir=reference_work, k=12, top_n_preview=10, snippet_chars=100)
            expected = reference(arguments)
            for arm in (CONTROL, "search_paging", "no_bash"):
                work = self.root / (arm + "-" + reference_work.name)
                tools = self.tools(arm, working=work)
                actual = tools["search"](arguments)
                if arm == "search_paging":
                    actual = actual.split("\nSearch pages: ")[0]
                self.assertEqual(actual, expected)
                self.assertEqual(tools["read"]({"file_path": self.mapping["0"]}),
                    make_read_tool(reference_work)({"file_path": self.mapping["0"]}))
                self.assertTrue(all(os.path.samefile(work / path, self.corpus / path)
                                    for path in self.mapping.values()))
                self.assertEqual(tools.summary["retrieval_calls"], 1)
                self.assertEqual(tools.summary["retrieval_queries"], 1)
                self.assertNotIn("Tool feedback", actual)
                self.assertNotIn("Accumulated original-document pool", actual)

    def test_removed_tools_are_absent_in_registry_and_schema_and_never_constructed(self):
        with patch.object(runtime, "make_bash_tool", side_effect=AssertionError("must not construct bash")):
            no_bash = self.tools("no_bash")
        self.assertNotIn("bash", no_bash)
        self.assertNotIn("bash", [x["function"]["name"] for x in build_contract("no_bash")["tool_specs"]])
        auto = self.tools("auto_rerank")
        self.assertNotIn("rerank", auto)
        self.assertNotIn("rerank", [x["function"]["name"] for x in build_contract("auto_rerank")["tool_specs"]])
        for arm in (*ARMS, CONTROL):
            tools = self.tools(arm, working=self.root / ("registry-" + arm))
            self.assertEqual(list(tools), [x["function"]["name"] for x in build_contract(arm)["tool_specs"]])

    def test_paging_reuses_fixed_record_without_import_or_scoring(self):
        backend = CheapBackend()
        work = self.root / "paging"
        tools = self.tools("search_paging", working=work,
                           service=self.service(work, backend=backend))
        original = tools["search"]({"queries": ["aurora"]})
        metadata = json.loads(original.split("\nSearch pages: ")[1])[0]
        before = sorted(path.relative_to(work).as_posix() for path in work.rglob("*.txt"))
        state = tools.export_state()
        page = tools["search"]({"search_id": metadata["search_id"], "offset": 10})
        self.assertIn('"returned_documents":2', page)
        self.assertEqual(before, sorted(path.relative_to(work).as_posix() for path in work.rglob("*.txt")))
        self.assertEqual(backend.calls, [])
        self.assertEqual(tools.summary["retrieval_calls"], 1)
        self.assertEqual(tools.summary["search_page_calls"], 1)
        restored = self.tools("search_paging", working=work,
            service=self.service(work, backend=CheapBackend(), views="restored"))
        restored.restore_state(state)
        self.assertEqual(restored["search"]({"search_id": metadata["search_id"], "offset": 10}), page)
        self.assertEqual(restored.summary["search_page_calls"], 1)

    def test_auto_rerank_uses_each_own_document_scope_and_separate_counters(self):
        work = self.root / "auto"
        backend = CheapBackend()
        service = self.service(work, backend=backend)
        seen = []
        class ObservedService:
            def __getattr__(self, name):
                return getattr(service, name)
            def __call__(self, args):
                seen.append(copy.deepcopy(args))
                return service(args)
        config = copy.deepcopy(self.config); config["bm25_k"] = 4
        tools = self.tools("auto_rerank", working=work, service=ObservedService(), config=config)
        result = json.loads(tools["search"]({"queries": ["aurora", "calibration"]}))
        self.assertEqual(len(result["queries"]), 2)
        records = list(service.manager._search_sets.values())
        self.assertEqual([entry["file_paths"] for entry in seen],
                         [[doc["file_path"] for doc in record["docs"]] for record in records])
        for entry in seen:
            self.assertEqual((entry["bm25_candidates"], entry["top_k"], entry["limit"]), (1000, 100, 10))
        self.assertNotIn("result_id", json.dumps(result))
        self.assertEqual(tools.summary["independent_rerank_calls"], 0)
        self.assertEqual(tools.summary["automatic_rerank_calls"], 2)
        self.assertEqual(tools.summary["search_page_calls"], 0)

    def test_no_bash_removes_only_unavailable_response_coordinates(self):
        work = self.root / "no-bash-response"
        service = self.service(work)
        tools = self.tools("no_bash", working=work, service=service)
        tools["search"]({"queries": ["aurora"]})
        args = {"query": "needle", "file_paths": [self.mapping["0"]],
                "bm25_candidates": 12, "top_k": 2, "limit": 2}
        original = json.loads(service(args))
        self.assertIn("bash_args", json.dumps(original))
        result = json.loads(tools["rerank"](args))
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("bash_args", json.dumps(result))
        self.assertTrue(all("read_args" in item for item in result["items"]))
        self.assertEqual(tools.summary["independent_rerank_calls"], 1)
        self.assertEqual(tools.summary["automatic_rerank_calls"], 0)

    def test_automatic_scoring_or_diagnostic_infrastructure_failure_pauses(self):
        def fail(_query, _texts):
            raise RuntimeError("fixture backend unavailable")
        for mode in ("backend", "diagnostic"):
            with self.subTest(mode=mode):
                health = StopState(self.root / (mode + "-pause.json"), self.root / (mode + "-events.jsonl"))
                work = self.root / ("auto-infra-" + mode)
                service = self.service(work, backend=CheapBackend(fail if mode == "backend" else None),
                                       views="infra-" + mode)
                tools = self.tools("auto_rerank", working=work, service=service, health=health)
                context = (patch.object(service, "_append_diagnostic", side_effect=OSError("storage full"))
                           if mode == "diagnostic" else patch.dict({}, {}))
                with context, self.assertRaises(SandboxPaused):
                    tools["search"]({"queries": ["aurora"]})
                self.assertTrue(health.paused)
                self.assertTrue(health.pause_path.exists())

    def test_automatic_ordinary_scoring_input_error_does_not_pause(self):
        def fail(_query, _texts):
            raise ValueError("ordinary unsupported input")
        work = self.root / "auto-input-error"
        service = self.service(work, backend=CheapBackend(fail), health=self.health)
        tools = self.tools("auto_rerank", working=work, service=service, health=self.health)
        result = json.loads(tools["search"]({"queries": ["aurora"]}))
        self.assertEqual(result["status"], "error")
        self.assertFalse(self.health.paused)
        self.assertEqual(tools.summary["automatic_rerank_errors"], 1)

    def test_real_agent_sdk_recorder_uses_new_contract_and_preserves_usage(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                step = 0
                def respond(body):
                    nonlocal step
                    step += 1
                    self.assertEqual(body["messages"][0]["content"], build_contract(arm)["system_prompt"])
                    self.assertEqual(body["tools"], build_contract(arm)["tool_specs"])
                    self.assertEqual(body["max_tokens"], 32768)
                    self.assertEqual((body["temperature"], body["top_p"], body["presence_penalty"]), (1.0, .95, 1.5))
                    self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
                    if step == 1:
                        return reply("search", {"queries": ["aurora"]})
                    if step == 2:
                        return reply("read", {"file_path": self.mapping["11"]})
                    self.assertIn("aurora calibration evidence", body["messages"][-1]["content"])
                    return reply()
                result = self.run_query(arm, respond, output=self.root / ("http-" + arm))
                self.assertEqual(result["terminated_by"], "done")
                self.assertEqual((result["input_tokens"], result["cached_input_tokens"],
                                  result["output_tokens"], result["reasoning_tokens"]), (150, 60, 30, 12))
                self.assertTrue(all(result["execution_checks"][name] for name in
                                    ("contract_verified", "model_verified", "token_ledger_verified")))
                self.assertEqual(result["ablation_summary"]["independent_rerank_calls"], 0)
                self.assertEqual(result["ablation_summary"]["automatic_rerank_calls"], int(arm == "auto_rerank"))
                self.assertEqual(result["common_config_sha256"], runtime.object_sha(self.config))

    def test_changed_expected_contract_rejected_before_resource_or_model_access(self):
        contract = copy.deepcopy(build_contract("search_paging")); contract["system_prompt"] += "extra"
        with patch.object(runtime, "resources") as resources, patch("rise.decompose.make_client") as client:
            with self.assertRaisesRegex(ValueError, "Frozen strict ablation contract"):
                runtime.run_query(arm="search_paging", query=self.query, config=self.config,
                    output_dir=self.root / "bad", expected_contract=contract, report_dir=self.root / "reports")
        resources.assert_not_called(); client.assert_not_called()

    def test_request_recorder_checks_prompt_tools_question_and_reported_model(self):
        arm = "no_bash"
        client = self.client([])
        recorder = runtime.RequestRecorder(client, arm=arm, query=self.query, config=self.config,
            contract=build_contract(arm), output_dir=self.root / "recorder", health=self.health)
        valid = {"model": runtime.MODEL, "messages": [
            {"role": "system", "content": build_contract(arm)["system_prompt"]},
            {"role": "user", "content": official_user_prompt().format(question=self.query["question"])}],
            "tools": build_contract(arm)["tool_specs"]}
        for kind in ("model", "system", "tools", "question"):
            bad = copy.deepcopy(valid)
            if kind == "model": bad["model"] = "wrong-model"
            elif kind == "system": bad["messages"][0]["content"] += "extra"
            elif kind == "tools": bad["tools"] = []
            else: bad["messages"][1]["content"] = "another question"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                recorder.chat.completions.create(**bad)
        self.assertEqual(self.requests, [])
        with self.assertRaises(SandboxPaused):
            self.run_query(arm, [reply(model="wrong-model")])
        ledger = [json.loads(line) for line in (self.root / "run/api_ledgers/qid_1.jsonl").read_text().splitlines()]
        self.assertEqual(ledger[0]["status"], "error")
        self.assertEqual(ledger[0]["reported_model"], "wrong-model")

    def test_paging_and_rerank_cache_resume_with_original_budget_and_counters(self):
        backend = CheapBackend()
        step = 0
        def first(body):
            nonlocal step
            step += 1
            if step == 1: return reply("search", {"queries": ["aurora"]})
            if step == 2:
                sid = json.loads(body["messages"][-1]["content"].split("\nSearch pages: ")[1])[0]["search_id"]
                return reply("search", {"search_id": sid, "offset": 10})
            if step == 3: return reply("rerank", {"query": "aurora", "bm25_candidates": 12, "top_k": 2, "limit": 1})
            return (500, {"error": {"message": "offline outage", "type": "server_error"}})
        initial = self.run_query("search_paging", first, backend=backend)
        self.assertEqual(initial["terminated_by"], "api_error")
        path = self.root / "run/_checkpoints/qid_1.json"
        state = json.loads(path.read_text())
        state["elapsed_seconds"] += 100
        path.write_text(json.dumps(state))
        rank = next(json.loads(message["content"]) for message in state["messages"]
                    if message.get("tool_call_id") == "call-rerank")
        sid = next(iter(state["tool_state"]["service"]["manager"]["search_sets"]))
        fresh = CheapBackend()
        resumed = self.run_query("search_paging", [
            reply("rerank", {"result_id": rank["result_id"], "offset": 1, "limit": 1}),
            reply("search", {"search_id": sid, "offset": 0}), reply()], backend=fresh)
        self.assertEqual(resumed["terminated_by"], "done")
        self.assertGreaterEqual(resumed["elapsed_seconds"], 100)
        self.assertEqual(resumed["tool_call_breakdown"], {"search": 3, "rerank": 2})
        self.assertEqual(resumed["ablation_summary"]["retrieval_calls"], 1)
        self.assertEqual(resumed["ablation_summary"]["search_page_calls"], 2)
        self.assertEqual(resumed["ablation_summary"]["independent_rerank_calls"], 2)
        self.assertEqual(resumed["ablation_summary"]["independent_rerank_page_calls"], 1)
        self.assertEqual(len(backend.calls), 1); self.assertEqual(fresh.calls, [])
        self.assertTrue(resumed["api_ledger"]["agent_totals_match"])

    def test_auto_state_resume_does_not_reset_auto_or_backend_usage(self):
        backend = CheapBackend()
        initial = self.run_query("auto_rerank", [reply("search", {"queries": ["aurora"]}),
            (500, {"error": {"message": "offline outage", "type": "server_error"}})], backend=backend)
        self.assertEqual(initial["terminated_by"], "api_error")
        fresh = CheapBackend()
        result = self.run_query("auto_rerank", [reply("search", {"queries": ["aurora"]}), reply()], backend=fresh)
        self.assertEqual(result["ablation_summary"]["automatic_rerank_calls"], 2)
        self.assertEqual(result["ablation_summary"]["independent_rerank_calls"], 0)
        self.assertEqual(result["reranker_summary"]["scoring_calls"], 1)
        self.assertEqual(fresh.calls, [])

    def test_variant_config_and_missing_checkpoint_cannot_reset_execution(self):
        self.run_query("no_bash", [reply()])
        changed = copy.deepcopy(self.config); changed["max_turns"] += 1
        self.assertNotEqual(compute_run_config_hash(runtime.run_config_for("no_bash", self.config).to_hashable_dict()),
                            compute_run_config_hash(runtime.run_config_for("no_bash", changed).to_hashable_dict()))
        for arm, config in (("search_paging", self.config), ("no_bash", changed)):
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "Checkpoint identity changed"):
                self.run_query(arm, [], config=config)
        checkpoint = self.root / "run/_checkpoints/qid_1.json"
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "fresh budget is forbidden"):
            self.run_query("no_bash", [])

    def test_consumed_wall_clock_is_not_reset_on_resume(self):
        self.run_query("auto_rerank", [reply("search", {"queries": ["aurora"]}),
            (500, {"error": {"message": "offline outage", "type": "server_error"}})])
        path = self.root / "run/_checkpoints/qid_1.json"
        state = json.loads(path.read_text()); state["elapsed_seconds"] = 3601
        path.write_text(json.dumps(state))
        before = len(self.requests)
        result = self.run_query("auto_rerank", [])
        self.assertEqual(result["terminated_by"], "wall_clock_timeout")
        self.assertEqual(len(self.requests), before)
        self.assertGreaterEqual(result["elapsed_seconds"], 3601)

    def test_health_is_shared_only_within_new_experiment_and_persisted_pause_blocks(self):
        left = runtime.health_state(self.config, self.root / "reports-left")
        right = runtime.health_state(self.config, self.root / "reports-right")
        self.assertIs(left, runtime.health_state(self.config, self.root / "reports-left"))
        self.assertIsNot(left, right)
        left.pause("offline_fixture", {})
        with self.assertRaises(SandboxPaused):
            runtime.health_state(self.config, self.root / "reports-left")
        self.assertFalse(right.paused)


if __name__ == "__main__":
    unittest.main()
