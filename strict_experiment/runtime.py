"""Strict experiment runtime: original shared tools, optional independent rerank.

No v14 prompt, ProgressFeedback, search-pool notice, or shell tutorial is used.
The common Agent and Linux sandbox adapters are frozen and identical in A/B.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "code/src", ROOT / "server"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from strict_experiment.strict_prompt import build_contract, official_user_prompt
from rise.agent import Agent
from rise.protocol import RunConfig
from rise.tools import make_bm25_search_tool, make_bash_tool, make_read_tool
from rise.trajectory import atomic_write_json, compute_run_config_hash
from rise.turn_checkpoint import TurnCheckpoint

RUNTIME_VERSION = "official-rise-rerank-strict-runtime-v1"
MODEL = "Qwen3.5-35B-A3B"
_RESOURCE_LOCK = threading.RLock()
_RESOURCES = {}
_HEALTH = {}


def health_state(config):
    from runtime_guards import StopState
    key = object_sha(config)
    with _RESOURCE_LOCK:
        if key not in _HEALTH:
            report = ROOT / "reports/strict_qwen35"
            _HEALTH[key] = StopState(report / "infrastructure_pause.json",
                                     report / "infrastructure_events.jsonl")
        return _HEALTH[key]


def wrap_infrastructure_tools(tools, health):
    """Stop broken infrastructure without changing ordinary tool feedback."""
    from runtime_guards import is_docker_infrastructure_failure
    wrapped = {}
    for name, tool in tools.items():
        def execute(args, _name=name, _tool=tool):
            health.raise_if_paused()
            try:
                result = _tool(args)
            except OSError as exc:
                health.pause("tool_storage_or_spawn_failed", {
                    "tool": _name, "error_type": type(exc).__name__})
                raise
            if _name == "bash" and isinstance(result, str):
                lower = result.lower()
                broken = (lower.startswith(("error: failed to spawn bash:",
                                             "error: bash subprocess exception:"))
                          or ("[stderr]" in lower and
                              ("bwrap:" in lower or is_docker_infrastructure_failure(1, result))))
                if broken:
                    health.pause("sandbox_infrastructure_failed", {"tool": "bash"})
            if _name == "rerank":
                try:
                    if json.loads(result).get("code") == "diagnostic_log_error":
                        health.pause("reranker_diagnostic_storage_failed", {"tool": "rerank"})
                except (ValueError, TypeError, AttributeError):
                    pass
            return result
        wrapped[name] = execute
    return wrapped


class GuardedBackend:
    """Keep infrastructure failures distinct from ordinary tool input errors."""
    def __init__(self, backend, health):
        self._backend, self._health = backend, health

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _invoke(self, name, *args, **kwargs):
        from rise.reranker_backend import InputTooLongError
        self._health.raise_if_paused()
        try:
            return getattr(self._backend, name)(*args, **kwargs)
        except Exception as exc:
            if name == "score" and isinstance(exc, (ValueError, InputTooLongError)):
                raise
            self._health.pause("reranker_backend_failed", {
                "operation": name, "error_type": type(exc).__name__})
            raise

    def score(self, *args, **kwargs):
        return self._invoke("score", *args, **kwargs)

    def get_tokenizer(self, *args, **kwargs):
        return self._invoke("get_tokenizer", *args, **kwargs)


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_sha(value):
    return text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False))


def run_config_for(arm, config):
    contract = build_contract(arm)
    return RunConfig(
        runner="strict_official_rise_rerank", runner_mode="doc",
        agent_model=config["model"], max_model_calls=config["max_turns"],
        per_call_max_tokens=config["per_call_max_tokens"],
        wall_clock_timeout_sec=config["wall_clock_seconds"],
        api_max_retries=config["api_max_retries"], run_max_attempts=1,
        coerce_final_on_max_turns=False,
        extras={"strict_runtime_version": RUNTIME_VERSION, "strict_arm": arm,
                "common_config_sha256": object_sha(config),
                "contract_sha256": contract["contract_sha256"],
                "bm25_k": config["bm25_k"],
                "bm25_top_n_preview": config["bm25_top_n_preview"],
                "bm25_snippet_chars": config["bm25_snippet_chars"],
                "bash_truncate_chars": 4000,
                "read_default_limit": config["read_default_limit"],
                "structured_doc_mode": True, "context_level": "level0"})


def build_tools(arm, *, config, searcher, doc_ids, mapping, corpus,
                working_dir, rerank_service=None, enable_sandbox=True):
    """B's callback has side effects only: it returns no search-visible IDs."""
    if arm not in {"A", "B"} or (arm == "B") != (rerank_service is not None):
        raise ValueError("Only B may have a rerank service, and B requires one")

    def register(queries, results):
        rerank_service.register_search(queries, results)
        return None

    tools = {
        "search": make_bm25_search_tool(
            searcher=searcher, doc_ids=doc_ids, docid_to_relpath=mapping,
            bc_plus_docs_root=corpus, working_dir=working_dir,
            k=config["bm25_k"], top_n_preview=config["bm25_top_n_preview"],
            snippet_chars=config["bm25_snippet_chars"],
            on_search_results=register if arm == "B" else None),
        "bash": make_bash_tool(working_dir, truncate_chars=4000,
                               enable_sandbox=enable_sandbox),
        "read": make_read_tool(working_dir,
                               default_limit=config["read_default_limit"]),
    }
    if rerank_service is not None:
        tools["rerank"] = rerank_service
    if list(tools) != [spec["function"]["name"]
                       for spec in build_contract(arm)["tool_specs"]]:
        raise ValueError("Callable registry does not match the frozen tool schema")
    return tools


class RequestRecorder:
    """Verify each actual request; save public contracts and numeric metadata.

    No auth header, API key, request history, or private reasoning is written
    by this recorder. The unchanged Agent owns private server-side traces.
    """
    def __init__(self, client, *, arm, query, config, contract, output_dir,
                 shared=None, health=None):
        self._client = client
        self.arm, self.config = arm, config
        self.query = {"query_id": query["query_id"],
                      "question": query.get("question", query.get("query"))}
        self.contract, self.output_dir = contract, Path(output_dir)
        self.health = health
        self._shared = shared if shared is not None else {
            "lock": threading.RLock(), "requests": 0, "verified": 0}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name):
        return getattr(self._client, name)

    def with_options(self, **kwargs):
        return type(self)(self._client.with_options(**kwargs), arm=self.arm,
                          query=self.query, config=self.config,
                          contract=self.contract, output_dir=self.output_dir,
                          shared=self._shared, health=self.health)

    def _create(self, *args, **kwargs):
        if self.health is not None:
            self.health.raise_if_paused()
        if args or kwargs.get("model") != MODEL:
            raise ValueError("Only the frozen named Qwen3.5 model is permitted")
        messages = kwargs.get("messages") or []
        if not messages or messages[0] != {
                "role": "system", "content": self.contract["system_prompt"]}:
            raise ValueError("Actual system prompt differs from the frozen contract")
        if kwargs.get("tools") != self.contract["tool_specs"]:
            raise ValueError("Actual tools differ from the frozen contract")
        user_prompt = official_user_prompt().format(question=self.query["question"])
        if not any(message == {"role": "user", "content": user_prompt}
                   for message in messages):
            raise ValueError("Actual request does not contain the original question")
        # Contract is created at the point of an actual request, not inferred
        # from a prepared config. Resumed calls must agree with the first one.
        qid = str(self.query["query_id"])
        payload = {"arm": self.arm, "query_id": self.query["query_id"],
                   "model": MODEL, "question_sha256": text_sha(self.query["question"]),
                   "user_prompt": user_prompt, "system_prompt": messages[0]["content"],
                   "tool_specs": copy.deepcopy(kwargs["tools"]),
                   "system_prompt_sha256": self.contract["system_prompt_sha256"],
                   "tool_specs_sha256": self.contract["tool_specs_sha256"],
                   "contract_sha256": self.contract["contract_sha256"],
                   "common_config_sha256": object_sha(self.config)}
        path = self.output_dir / "request_contracts" / f"qid_{qid}.json"
        with self._shared["lock"]:
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError("Previously sent request contract changed")
            else:
                atomic_write_json(path, payload)
            self._shared["requests"] += 1
        row = {"query_id": self.query["query_id"], "arm": self.arm,
               "requested_model": MODEL, "status": "error",
               "request_messages_sha256": object_sha(messages),
               "contract_sha256": self.contract["contract_sha256"],
               "request_parameters": {key: copy.deepcopy(kwargs[key]) for key in
                   ("max_tokens", "max_completion_tokens", "temperature", "top_p",
                    "presence_penalty", "frequency_penalty", "tool_choice", "extra_body")
                   if key in kwargs}, "unix_time": time.time()}
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(**kwargs)
            reported = getattr(response, "model", None)
            row["reported_model"] = reported
            if reported != MODEL:
                if self.health is not None:
                    self.health.pause("solver_model_identity_mismatch", {})
                raise ValueError("Reported model does not match the frozen model")
            choice = response.choices[0] if response.choices else None
            usage = getattr(response, "usage", None)
            row.update(status="ok", finish_reason=choice.finish_reason if choice else None,
                       usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
                       response_id=getattr(response, "id", None))
            with self._shared["lock"]:
                self._shared["verified"] += 1
            return response
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["request_seconds"] = time.monotonic() - started
            ledger = self.output_dir / "api_ledgers" / f"qid_{qid}.jsonl"
            with self._shared["lock"]:
                ledger.parent.mkdir(parents=True, exist_ok=True)
                with ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())


def resources(config):
    key = tuple(config[k] for k in ("document_index", "filename_map", "corpus"))
    with _RESOURCE_LOCK:
        if key not in _RESOURCES:
            from rise.retrieval import load_index
            searcher, doc_ids = load_index(Path(config["document_index"]))
            paths = json.loads(Path(config["filename_map"]).read_text())["relpath_to_docid"]
            mapping = {str(doc_id): path for path, doc_id in paths.items()}
            _RESOURCES[key] = (searcher, doc_ids, mapping)
        return _RESOURCES[key]


def make_service(config, working_dir, runtime_dir, mapping, health=None):
    from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
    from rise.passage_index import PassageIndex
    from rise.rerank_v13 import RerankService
    from shared_reranker import install_shared_reranker_factory
    with _RESOURCE_LOCK:
        install_shared_reranker_factory()
        from rise.reranker_backend import QwenReranker
        key = (config["passage_index"], config["corpus"])
        if key not in _RESOURCES:
            _RESOURCES[key] = PassageIndex(Path(config["passage_index"]),
                                          corpus_root=Path(config["corpus"]))
        index = _RESOURCES[key]
        if set(mapping.values()) - set(index._path_to_doc):
            raise ValueError("Passage index does not cover the original document corpus")
        backend = QwenReranker(
            model=config["reranker_model"], revision=config["reranker_revision"],
            device=config["reranker_device"], dtype=config["reranker_dtype"],
            batch_size=8, max_length=32768, local_files_only=True)
        if health is not None:
            backend = GuardedBackend(backend, health)
    manager = IndexedAccumulatedSpaceManager(
        working_dir=working_dir, views_dir=runtime_dir, backend=backend,
        candidates=1000, top_k=100, passage_index=index,
        runtime_to_corpus={path: path for path in mapping.values()})
    return RerankService(manager)


def run_strict_query(*, arm, query, config, output_dir, expected_contract):
    query_started = time.monotonic()
    query = {"query_id": query["query_id"],
             "question": query.get("question", query.get("query"))}
    if not isinstance(query["question"], str) or not query["question"]:
        raise ValueError("Question text is missing")
    contract = build_contract(arm)
    for key in ("system_prompt", "tool_specs", "system_prompt_sha256",
                "tool_specs_sha256", "contract_sha256"):
        if expected_contract.get(key) != contract[key]:
            raise ValueError(f"Frozen {key} does not match the actual runtime")
    if config["model"] != MODEL:
        raise ValueError("Strict experiment is fixed to Qwen3.5-35B-A3B")
    for key, value in config["runtime_environment"].items():
        os.environ[key] = str(value)
    if os.environ.get("RISE_LOCAL_VLLM_BASE_URL", "").rstrip("/") != config["base_url"].rstrip("/"):
        raise ValueError("Common model endpoint does not match runtime environment")
    if os.environ.get("RISE_QWEN35_PROFILE") != "official-thinking-v1":
        raise ValueError("Strict Qwen3.5 sampling profile is missing")
    qid = str(query["query_id"])
    if not qid.isdigit():
        raise ValueError("Unexpected query ID")
    output_dir = Path(output_dir)
    health = health_state(config)
    health.local.qid = query["query_id"]
    health.raise_if_paused()
    checkpoint_path = output_dir / "_checkpoints" / f"qid_{qid}.json"
    trace_path = output_dir / "_traces" / f"qid_{qid}" / "single.json"
    working_dir = output_dir / "_working" / f"qid_{qid}"
    runtime_dir = output_dir / "_rerank_runtime" / f"qid_{qid}"
    if not checkpoint_path.exists():
        for directory in (working_dir, runtime_dir):
            if directory.exists():
                shutil.rmtree(directory)
    working_dir.mkdir(parents=True, exist_ok=True)
    searcher, doc_ids, mapping = resources(config)
    service = make_service(config, working_dir, runtime_dir, mapping, health=health) if arm == "B" else None
    tools = build_tools(arm, config=config, searcher=searcher, doc_ids=doc_ids,
                        mapping=mapping, corpus=Path(config["corpus"]),
                        working_dir=working_dir, rerank_service=service)
    tools = wrap_infrastructure_tools(tools, health)
    run_config = run_config_for(arm, config)
    from rise.decompose import make_client
    client = RequestRecorder(make_client(MODEL), arm=arm, query=query, config=config,
                             contract=contract, output_dir=output_dir, health=health)
    checkpoint = TurnCheckpoint(
        checkpoint_path, identity={"model": MODEL, "query_id": qid,
            "question_sha256": text_sha(query["question"]),
            "run_config_hash": compute_run_config_hash(run_config.to_hashable_dict()),
            "system_prompt_sha256": contract["system_prompt_sha256"],
            "tool_specs_sha256": contract["tool_specs_sha256"]},
        working_dir=working_dir, corpus_root=Path(config["corpus"]),
        relpath_to_docid={path: did for did, path in mapping.items()},
        tool_state_provider=service.export_state if service else None,
        tool_state_restorer=service.restore_state if service else None)
    setup_seconds = time.monotonic() - query_started
    run = Agent(
        client=client, model=MODEL, tools=tools, tool_specs=contract["tool_specs"],
        max_turns=run_config.max_model_calls,
        per_call_max_tokens=run_config.per_call_max_tokens,
        max_retries=run_config.api_max_retries,
        wall_clock_timeout_sec=run_config.wall_clock_timeout_sec,
        coerce_final_on_max_turns=False, context_level="level0").run(
            system_prompt=contract["system_prompt"],
            user_prompt=official_user_prompt().format(question=query["question"]),
            trace_path=trace_path,
            trace_relpath_to_docid={path: did for did, path in mapping.items()},
            turn_checkpoint=checkpoint)
    if health.paused:
        # The unmodified Agent has saved the completed turn and the same
        # remaining budget. Do not turn a broken sandbox/backend into a score.
        from runtime_guards import SandboxPaused
        raise SandboxPaused("Infrastructure paused; preserve the original checkpoint")
    ledger_path = output_dir / "api_ledgers" / f"qid_{qid}.jsonl"
    ledger = ([json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
              if ledger_path.exists() else [])
    successful = [record for record in ledger if record.get("status") == "ok"]
    recorded_path = output_dir / "request_contracts" / f"qid_{qid}.json"
    recorded = json.loads(recorded_path.read_text()) if recorded_path.exists() else {}
    contract_verified = bool(successful) and all(
        record.get("contract_sha256") == contract["contract_sha256"] for record in ledger)
    contract_verified = contract_verified and all(recorded.get(key) == contract[key]
        for key in ("system_prompt", "tool_specs", "system_prompt_sha256",
                    "tool_specs_sha256", "contract_sha256"))
    model_verified = bool(successful) and all(
        record.get("reported_model") == MODEL for record in successful)
    ledger_input = sum((record.get("usage") or {}).get("prompt_tokens", 0) for record in successful)
    ledger_output = sum((record.get("usage") or {}).get("completion_tokens", 0) for record in successful)
    ledger_matches = (ledger_input == run.total_prompt_tokens + run.total_cached_prompt_tokens
                      and ledger_output == run.total_completion_tokens)
    # Input includes cached input; reasoning is a subset of output. Keeping
    # the original non-cached counter separately makes the ledger auditable.
    return {"arm": arm, "query_id": query["query_id"], "model": MODEL,
            "final_text": run.final_answer or "", "terminated_by": run.terminated_reason,
            "error": run.error, "n_turns": len(run.turns),
            "n_tool_calls": run.tool_call_count, "tool_call_breakdown": dict(run.tool_call_breakdown),
            "elapsed_seconds": run.elapsed_seconds,
            "invocation_setup_seconds": setup_seconds,
            "invocation_wall_seconds": time.monotonic() - query_started,
            "input_tokens": run.total_prompt_tokens + run.total_cached_prompt_tokens,
            "raw_non_cached_input_tokens": run.total_prompt_tokens,
            "cached_input_tokens": run.total_cached_prompt_tokens,
            "output_tokens": run.total_completion_tokens,
            "reasoning_tokens": run.total_reasoning_tokens,
            "cost_usd": None, "cost_status": "local_model_no_monetary_estimate",
            "reranker_summary": service.summary if service else {},
            "common_config_sha256": object_sha(config),
            "run_config_hash": compute_run_config_hash(run_config.to_hashable_dict()),
            "contract_sha256": contract["contract_sha256"],
            "api_ledger": {"attempts": len(ledger), "successful_calls": len(successful),
                           "input_tokens": ledger_input, "output_tokens": ledger_output,
                           "attempts_without_usage": sum(not isinstance(record.get("usage"), dict)
                                                         for record in ledger),
                           "agent_totals_match": ledger_matches},
            "execution_checks": {
                "contract_verified": contract_verified,
                "model_verified": model_verified,
                "token_ledger_verified": ledger_matches,
                "runtime_feedback": "original_shared_tools_no_progress_feedback"}}


def preflight(*, config, expected_contracts):
    """Live checks on non-cohort probes; no experiment answer is generated."""
    import math
    import tempfile
    from rise.api_retry import chat_with_retry
    from rise.decompose import make_client

    for arm in ("A", "B"):
        if build_contract(arm) != expected_contracts[arm]:
            raise ValueError("Preflight prompt/tool contract differs")
    os.environ.update(config["runtime_environment"])
    probe_root = ROOT / "reports/strict_qwen35/preflight_probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    shared = {"search": False, "read": False, "bash": False,
              "source_pool_unchanged": False}
    searcher, doc_ids, mapping = resources(config)
    with tempfile.TemporaryDirectory(prefix="tools-", dir=probe_root) as temporary:
        temporary = Path(temporary)
        work_a, work_b = temporary / "A", temporary / "B"
        work_a.mkdir(); work_b.mkdir()
        service = make_service(config, work_b, temporary / "rerank", mapping)
        surfaces = {arm: build_tools(
            arm, config=config, searcher=searcher, doc_ids=doc_ids,
            mapping=mapping, corpus=Path(config["corpus"]),
            working_dir=work_a if arm == "A" else work_b,
            rerank_service=service if arm == "B" else None)
            for arm in ("A", "B")}
        outputs = [surfaces[arm]["search"]({"queries": ["strict execution compatibility probe"]})
                   for arm in ("A", "B")]
        shared["search"] = outputs[0] == outputs[1] and "Reranker set:" not in outputs[1]
        imported_a = sorted(str(p.relative_to(work_a)) for p in work_a.rglob("*.txt"))
        imported_b = sorted(str(p.relative_to(work_b)) for p in work_b.rglob("*.txt"))
        shared["source_pool_unchanged"] = imported_a == imported_b and bool(imported_a)
        if not imported_a:
            raise RuntimeError("Shared search probe did not import any documents")
        args = {"file_path": imported_a[0], "offset": 0, "limit": 8}
        shared["read"] = surfaces["A"]["read"](args) == surfaces["B"]["read"](args)
        # Equal successful and ordinary failing command returns. Keep the
        # original binary and /tmp lifecycle; never rewrite model commands.
        shell = []
        for arm in ("A", "B"):
            tool = surfaces[arm]["bash"]
            results = [tool({"command": "printf RISE_STRICT_SANDBOX_READY"}),
                       tool({"command": "printf RISE_TMP_PROBE >/tmp/rise-strict-probe"}),
                       tool({"command": "test ! -e /tmp/rise-strict-probe && printf FRESH_TMP"}),
                       tool({"command": "rg --version | head -n 1"})]
            shell.append(results)
        shared["bash"] = (shell[0] == shell[1]
                          and shell[0][0] == "RISE_STRICT_SANDBOX_READY"
                          and "FRESH_TMP" in shell[0][2]
                          and "ripgrep" in shell[0][3])
        ranked = json.loads(service({"query": "strict execution compatibility probe",
                                     "file_paths": [imported_b[0]],
                                     "bm25_candidates": 2, "top_k": 2, "limit": 2}))
        # No fallback scores or a public parameter error may pass readiness.
        neural_ready = (ranked.get("status") == "ok"
                        and ranked.get("scored_passages", 0) > 0
                        and all(math.isfinite(item["reranker_score"])
                                for item in ranked.get("items", [])))
        if imported_b != sorted(str(p.relative_to(work_b)) for p in work_b.rglob("*.txt")):
            raise RuntimeError("Rerank changed the source pool")
        native = {}
        for arm in ("A", "B"):
            contract = build_contract(arm)
            query = {"query_id": 0, "question": (
                "Compatibility probe only: call read with file_path="
                + json.dumps(imported_a[0]) + ", offset=0, limit=1; "
                "then finish with RISE_QWEN35_READY.")}
            client = RequestRecorder(make_client(MODEL), arm=arm, query=query,
                                     config=config, contract=contract,
                                     output_dir=probe_root / arm)
            history = [{"role": "system", "content": contract["system_prompt"]},
                       {"role": "user", "content": official_user_prompt().format(question=query["question"])}]
            response = chat_with_retry(
                client, model=MODEL, messages=history, tools=contract["tool_specs"],
                tool_choice={"type": "function", "function": {"name": "read"}},
                max_tokens=2048, max_retries=1)
            calls = response.choices[0].message.tool_calls or []
            if len(calls) != 1 or calls[0].function.name != "read":
                raise RuntimeError("Native tool-call probe did not return a complete read call")
            arguments = json.loads(calls[0].function.arguments)
            if arguments.get("file_path") != imported_a[0]:
                raise RuntimeError("Native tool-call probe changed the requested source")
            result = surfaces[arm]["read"](arguments)
            if result.startswith("Error:"):
                raise RuntimeError("Native read probe failed")
            from rise.local_vllm import extract_reasoning
            assistant = {"role": "assistant", "content": response.choices[0].message.content or "",
                         "tool_calls": [call.model_dump() for call in calls]}
            reasoning = extract_reasoning(response.choices[0].message)
            if reasoning:
                assistant["reasoning_content"] = reasoning
            history += [assistant, {"role": "tool", "tool_call_id": calls[0].id,
                                     "content": result}]
            final = chat_with_retry(
                client, model=MODEL, messages=history, tools=contract["tool_specs"],
                tool_choice="none", max_tokens=2048, max_retries=1)
            choice = final.choices[0]
            native[arm] = {"model_verified": client._shared["verified"] == 2,
                           "tool_call_round_trip": True,
                           "final_complete": choice.finish_reason == "stop" and bool(choice.message.content),
                           "finish_reason": choice.finish_reason,
                           "usage": [item.usage.model_dump() if item.usage else None
                                     for item in (response, final)]}
        return {"passed": (all(shared.values()) and neural_ready
                            and all(row["model_verified"] and row["final_complete"]
                                    for row in native.values())),
                "shared_tools": shared, "native_model": native,
                "reranker_ready": neural_ready,
                "reranker_summary": service.summary,
                "cohort_questions_used": 0}
