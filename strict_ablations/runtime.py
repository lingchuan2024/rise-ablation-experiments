"""Strict ablation execution through composition of the frozen shared runtime.

This module never enters the v14 agent assembler or installs runtime hooks.
The common configuration is passed through unchanged; each surface has its
own contract, checkpoint identity and experiment-scoped infrastructure state.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import shutil
import threading
import time

from strict_experiment import runtime as shared
from strict_experiment.strict_prompt import official_user_prompt
from strict_ablations.prompt import ARMS, CONTROL, VERSION, build_contract
from rise.ablation_tools import AutoRerankSearchTool, SearchPagingTool, WithoutBashResponse
from rise.agent import Agent
from rise.protocol import RunConfig
from rise.tools import make_bm25_search_tool, make_bash_tool, make_read_tool
from rise.trajectory import compute_run_config_hash
from rise.turn_checkpoint import TurnCheckpoint
from runtime_guards import SandboxPaused, StopState

MODEL = shared.MODEL
RUNTIME_VERSION = "strict-ablations-runtime-v1"
RequestRecorder = shared.RequestRecorder
GuardedBackend = shared.GuardedBackend
wrap_infrastructure_tools = shared.wrap_infrastructure_tools
resources = shared.resources
make_service = shared.make_service
object_sha = shared.object_sha
text_sha = shared.text_sha
_HEALTH_LOCK = threading.RLock()
_HEALTH = {}


def health_state(config, report_dir):
    """Share a stop signal across this experiment, never the old B100 paths."""
    root = Path(report_dir).resolve()
    key = (str(root), object_sha(config))
    with _HEALTH_LOCK:
        pause = root / "infrastructure_pause.json"
        if pause.exists():
            raise SandboxPaused("Persisted infrastructure pause requires explicit recovery")
        state = _HEALTH.get(key)
        if state is None:
            state = StopState(pause, root / "infrastructure_events.jsonl")
            _HEALTH[key] = state
        # A missing pause file must not clear an in-process infrastructure
        # failure (its persistence may itself have failed). Resume uses a fresh
        # worker after the runner explicitly archives the pause and rechecks.
        state.raise_if_paused()
        return state


def run_config_for(arm, config):
    contract = build_contract(arm)
    return RunConfig(
        runner="strict_official_rise_ablations", runner_mode="doc",
        agent_model=config["model"], max_model_calls=config["max_turns"],
        per_call_max_tokens=config["per_call_max_tokens"],
        wall_clock_timeout_sec=config["wall_clock_seconds"],
        api_max_retries=config["api_max_retries"], run_max_attempts=1,
        coerce_final_on_max_turns=False,
        extras={"ablation_runtime_version": RUNTIME_VERSION,
                "ablation_contract_version": VERSION, "ablation_arm": arm,
                "common_config_sha256": object_sha(config),
                "contract_sha256": contract["contract_sha256"],
                "bm25_k": config["bm25_k"],
                "bm25_top_n_preview": config["bm25_top_n_preview"],
                "bm25_snippet_chars": config["bm25_snippet_chars"],
                "bash_truncate_chars": 4000,
                "read_default_limit": config["read_default_limit"],
                "structured_doc_mode": True, "context_level": "level0"})


class _InfrastructureService:
    """Guard the service even when it is called inside automatic search.

    RerankService turns scoring/storage exceptions into ordinary JSON errors.
    The backend and diagnostic guard retain the infrastructure classification;
    check it before AutoRerankSearchTool can convert that result to search JSON.
    """
    def __init__(self, service, health):
        self._service, self._health = service, health
        self._call = (wrap_infrastructure_tools({"rerank": service}, health)["rerank"]
                      if health is not None else service)

    def __getattr__(self, name):
        return getattr(self._service, name)

    def __call__(self, args):
        result = self._call(args)
        if self._health is not None:
            self._health.raise_if_paused()
        return result


class ToolSurface(Mapping):
    """Callable registry plus composite service/search checkpoint state."""
    def __init__(self, arm, *, config, searcher, doc_ids, mapping, corpus,
                 working_dir, rerank_service, health=None, enable_sandbox=True):
        if arm not in (*ARMS, CONTROL) or rerank_service is None:
            raise ValueError("A known strict arm and its own rerank service are required")
        self.arm, self.service = arm, rerank_service
        # Also protect explicitly supplied/probe services; make_service already
        # supplies this guard for formal queries. This changes only this query's
        # backend instance, never a backend class or another module's globals.
        if health is not None and not isinstance(self.service.manager.backend, GuardedBackend):
            self.service.manager.backend = GuardedBackend(self.service.manager.backend, health)
        self._usage = {"search_invocations": 0, "retrieval_calls": 0,
                       "retrieval_queries": 0, "independent_rerank_calls": 0,
                       "independent_rerank_page_calls": 0}
        self._search = None

        def register(queries, results):
            self.service.register_search(queries, results)
            self._usage["retrieval_calls"] += 1
            self._usage["retrieval_queries"] += len(queries)
            return None  # Preserve the original search feedback byte-for-byte.

        original_search = make_bm25_search_tool(
            searcher=searcher, doc_ids=doc_ids, docid_to_relpath=mapping,
            bc_plus_docs_root=corpus, working_dir=working_dir,
            k=config["bm25_k"], top_n_preview=config["bm25_top_n_preview"],
            snippet_chars=config["bm25_snippet_chars"], on_search_results=register)
        service = _InfrastructureService(self.service, health)
        search = original_search
        if arm == "search_paging":
            self._search = SearchPagingTool(original_search, service,
                working_dir=working_dir, snippet_chars=config["bm25_snippet_chars"])
            search = self._search
        elif arm == "auto_rerank":
            self._search = AutoRerankSearchTool(original_search, service)
            search = self._search

        def search_call(args):
            self._usage["search_invocations"] += 1
            return search(args)

        tools = {"search": search_call}
        # Do not even construct the removed capability in the no_bash arm.
        if arm != "no_bash":
            tools["bash"] = make_bash_tool(working_dir, truncate_chars=4000,
                                             enable_sandbox=enable_sandbox)
        tools["read"] = make_read_tool(working_dir,
                                       default_limit=config["read_default_limit"])
        if arm != "auto_rerank":
            rerank = WithoutBashResponse(service) if arm == "no_bash" else service

            def rerank_call(args):
                self._usage["independent_rerank_calls"] += 1
                if isinstance(args, dict) and "result_id" in args:
                    self._usage["independent_rerank_page_calls"] += 1
                return rerank(args)

            tools["rerank"] = rerank_call
        names = [spec["function"]["name"] for spec in build_contract(arm)["tool_specs"]]
        if list(tools) != names:
            raise ValueError("Callable registry differs from the strict arm contract")
        self.tools = wrap_infrastructure_tools(tools, health) if health is not None else tools

    def __getitem__(self, name):
        return self.tools[name]

    def __iter__(self):
        return iter(self.tools)

    def __len__(self):
        return len(self.tools)

    @property
    def summary(self):
        search = self._search.summary if self._search is not None else {}
        return {"version": RUNTIME_VERSION, "arm": self.arm,
                **copy.deepcopy(self._usage),
                "search_page_calls": search.get("search_page_calls", 0),
                "automatic_rerank_calls": search.get("auto_rerank_calls", 0),
                "automatic_rerank_errors": search.get("auto_rerank_errors", 0),
                "automatic_empty_scopes": search.get("empty_scopes", 0)}

    def export_state(self):
        return {"version": RUNTIME_VERSION, "arm": self.arm,
                "service": self.service.export_state(),
                "search": self._search.export_state() if self._search is not None else None,
                "usage": copy.deepcopy(self._usage)}

    def restore_state(self, state):
        if (not isinstance(state, dict)
                or set(state) != {"version", "arm", "service", "search", "usage"}
                or state.get("version") != RUNTIME_VERSION or state.get("arm") != self.arm):
            raise ValueError("Strict ablation checkpoint variant changed")
        usage = state.get("usage")
        if (not isinstance(usage, dict) or set(usage) != set(self._usage)
                or any(type(value) is not int or value < 0 for value in usage.values())):
            raise ValueError("Invalid strict ablation usage counters")
        if (self._search is None) != (state.get("search") is None):
            raise ValueError("Strict ablation search state differs from its surface")
        self.service.restore_state(state["service"])
        if self._search is not None:
            self._search.restore_state(state["search"])
        self._usage = copy.deepcopy(usage)


def build_tools(arm, *, config, searcher, doc_ids, mapping, corpus, working_dir,
                rerank_service, health=None, enable_sandbox=True):
    return ToolSurface(arm, config=config, searcher=searcher, doc_ids=doc_ids,
        mapping=mapping, corpus=corpus, working_dir=working_dir,
        rerank_service=rerank_service, health=health, enable_sandbox=enable_sandbox)


def run_query(*, arm, query, config, output_dir, expected_contract, report_dir):
    started = time.monotonic()
    query = {"query_id": query["query_id"],
             "question": query.get("question", query.get("query"))}
    if not isinstance(query["question"], str) or not query["question"]:
        raise ValueError("Question text is missing")
    contract = build_contract(arm)
    if expected_contract != contract:
        raise ValueError("Frozen strict ablation contract differs from actual runtime")
    if contract["user_prompt_template"] != official_user_prompt():
        raise ValueError("The official shared user prompt changed")
    if config["model"] != MODEL:
        raise ValueError("Strict ablations require the exact Qwen3.5 model")
    qid = str(query["query_id"])
    if not qid.isdigit():
        raise ValueError("Unexpected query ID")
    os.environ.update(config["runtime_environment"])
    if os.environ.get("RISE_LOCAL_VLLM_BASE_URL", "").rstrip("/") != config["base_url"].rstrip("/"):
        raise ValueError("Common model endpoint differs from runtime environment")
    if os.environ.get("RISE_QWEN35_PROFILE") != "official-thinking-v1":
        raise ValueError("Strict Qwen3.5 sampling profile is missing")
    output_dir = Path(output_dir)
    health = health_state(config, report_dir)
    health.local.qid = query["query_id"]
    health.raise_if_paused()
    checkpoint_path = output_dir / "_checkpoints" / f"qid_{qid}.json"
    ledger_path = output_dir / "api_ledgers" / f"qid_{qid}.jsonl"
    recorded_path = output_dir / "request_contracts" / f"qid_{qid}.json"
    if not checkpoint_path.exists() and (ledger_path.exists() or recorded_path.exists()):
        raise ValueError("Partial execution has no checkpoint; fresh budget is forbidden")
    working_dir = output_dir / "_working" / f"qid_{qid}"
    runtime_dir = output_dir / "_rerank_runtime" / f"qid_{qid}"
    for directory in (working_dir, runtime_dir):
        if directory.is_symlink():
            raise ValueError("Query directories must not be symlinks")
        if not checkpoint_path.exists() and directory.exists():
            shutil.rmtree(directory)
    working_dir.mkdir(parents=True, exist_ok=True)
    searcher, doc_ids, mapping = resources(config)
    service = make_service(config, working_dir, runtime_dir, mapping, health=health)
    surface = build_tools(arm, config=config, searcher=searcher, doc_ids=doc_ids,
        mapping=mapping, corpus=Path(config["corpus"]), working_dir=working_dir,
        rerank_service=service, health=health)
    run_config = run_config_for(arm, config)
    run_hash = compute_run_config_hash(run_config.to_hashable_dict())
    from rise.decompose import make_client
    client = RequestRecorder(make_client(MODEL), arm=arm, query=query, config=config,
        contract=contract, output_dir=output_dir, health=health)
    checkpoint = TurnCheckpoint(checkpoint_path,
        identity={"model": MODEL, "query_id": qid, "ablation_arm": arm,
            "ablation_runtime_version": RUNTIME_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "common_config_sha256": object_sha(config),
            "question_sha256": text_sha(query["question"]), "run_config_hash": run_hash,
            "system_prompt_sha256": contract["system_prompt_sha256"],
            "tool_specs_sha256": contract["tool_specs_sha256"]},
        working_dir=working_dir, corpus_root=Path(config["corpus"]),
        relpath_to_docid={path: did for did, path in mapping.items()},
        tool_state_provider=surface.export_state, tool_state_restorer=surface.restore_state)
    setup_seconds = time.monotonic() - started
    run = Agent(client=client, model=MODEL, tools=surface.tools,
        tool_specs=contract["tool_specs"], max_turns=run_config.max_model_calls,
        per_call_max_tokens=run_config.per_call_max_tokens,
        max_retries=run_config.api_max_retries,
        wall_clock_timeout_sec=run_config.wall_clock_timeout_sec,
        coerce_final_on_max_turns=False, context_level="level0").run(
            system_prompt=contract["system_prompt"],
            user_prompt=official_user_prompt().format(question=query["question"]),
            trace_path=output_dir / "_traces" / f"qid_{qid}" / "single.json",
            trace_relpath_to_docid={path: did for did, path in mapping.items()},
            turn_checkpoint=checkpoint)
    if health.paused:
        raise SandboxPaused("Infrastructure paused; preserve the consumed-budget checkpoint")
    ledger = ([json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
              if ledger_path.exists() else [])
    successful = [row for row in ledger if row.get("status") == "ok"]
    recorded = json.loads(recorded_path.read_text()) if recorded_path.exists() else {}
    contract_verified = bool(successful) and all(
        row.get("contract_sha256") == contract["contract_sha256"] for row in ledger)
    contract_verified = contract_verified and all(recorded.get(key) == contract[key]
        for key in ("system_prompt", "tool_specs", "system_prompt_sha256",
                    "tool_specs_sha256", "contract_sha256"))
    model_verified = bool(successful) and all(row.get("reported_model") == MODEL for row in successful)
    ledger_input = sum((row.get("usage") or {}).get("prompt_tokens", 0) for row in successful)
    ledger_output = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in successful)
    ledger_matches = (ledger_input == run.total_prompt_tokens + run.total_cached_prompt_tokens
                      and ledger_output == run.total_completion_tokens)
    return {"arm": arm, "query_id": query["query_id"], "model": MODEL,
        "final_text": run.final_answer or "", "terminated_by": run.terminated_reason,
        "error": run.error, "n_turns": len(run.turns),
        "n_tool_calls": run.tool_call_count, "tool_call_breakdown": dict(run.tool_call_breakdown),
        "elapsed_seconds": run.elapsed_seconds, "invocation_setup_seconds": setup_seconds,
        "invocation_wall_seconds": time.monotonic() - started,
        "input_tokens": run.total_prompt_tokens + run.total_cached_prompt_tokens,
        "raw_non_cached_input_tokens": run.total_prompt_tokens,
        "cached_input_tokens": run.total_cached_prompt_tokens,
        "output_tokens": run.total_completion_tokens, "reasoning_tokens": run.total_reasoning_tokens,
        "cost_usd": None, "cost_status": "local_model_no_monetary_estimate",
        "reranker_summary": service.summary, "ablation_summary": surface.summary,
        "common_config_sha256": object_sha(config), "run_config_hash": run_hash,
        "contract_sha256": contract["contract_sha256"],
        "api_ledger": {"attempts": len(ledger), "successful_calls": len(successful),
            "input_tokens": ledger_input, "output_tokens": ledger_output,
            "attempts_without_usage": sum(not isinstance(row.get("usage"), dict) for row in ledger),
            "agent_totals_match": ledger_matches},
        "execution_checks": {"contract_verified": contract_verified,
            "model_verified": model_verified, "token_ledger_verified": ledger_matches,
            "runtime_feedback": "original_shared_tools_only_declared_ablation_delta"}}
