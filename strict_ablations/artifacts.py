"""Validate private execution metadata without examining reasoning text.

Reasoning strings are discarded while decoding. Only identities, public final
bytes, numeric usage, elapsed budgets and tool-state metadata are inspected;
the result contains file hashes, never trajectory or checkpoint contents.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from strict_ablations import runtime
from strict_ablations.prompt import ARMS, build_contract
from rise.ablation_tools import ABLATION_VERSION
from rise.accumulated_space import STATE_VERSION
from rise.rerank_v13 import RERANK_TOOL_VERSION
from rise.trajectory import SCHEMA_VERSION, compute_run_config_hash


def _require(condition, message):
    if not condition:
        raise ValueError("Execution artifact validation failed: " + message)


def _object(pairs):
    result, seen = {}, set()
    for key, value in pairs:
        _require(key not in seen, "duplicate JSON key")
        seen.add(key)
        # Never compare, summarize, print or retain private reasoning strings.
        if key in {"reasoning_content", "thinking_content"}:
            continue
        result[key] = value
    return result


def _load(path):
    _require(path.is_file() and not path.is_symlink(), "missing or aliased artifact")
    data = path.read_bytes()
    return json.loads(data, object_pairs_hook=_object), hashlib.sha256(data).hexdigest()


def _number(value, label, *, integer=False):
    _require(type(value) in ((int,) if integer else (int, float))
             and math.isfinite(value) and value >= 0, "invalid numeric " + label)
    return value


def _counts(value, label):
    _require(isinstance(value, dict), "missing " + label)
    for key, number in value.items():
        _number(number, label + "." + key, integer=True)
    return value


def _tool_state(state, arm, terminal):
    _require(isinstance(state, dict)
             and set(state) == {"version", "arm", "service", "search", "usage"}
             and state["version"] == runtime.RUNTIME_VERSION and state["arm"] == arm,
             "tool-state version/arm")
    summary = terminal.get("ablation_summary")
    _require(isinstance(summary, dict) and summary.get("version") == runtime.RUNTIME_VERSION
             and summary.get("arm") == arm, "terminal ablation summary")
    usage = _counts(state["usage"], "tool-state usage")
    expected_usage = {"search_invocations", "retrieval_calls", "retrieval_queries",
                      "independent_rerank_calls", "independent_rerank_page_calls"}
    _require(set(usage) == expected_usage and all(summary.get(k) == v for k, v in usage.items()),
             "checkpoint/terminal usage differs")
    _require(usage["independent_rerank_page_calls"] <= usage["independent_rerank_calls"],
             "independent page count")
    search = state["search"]
    if arm == "no_bash":
        _require(search is None, "unexpected search wrapper state")
        expected = {"search_page_calls": 0, "automatic_rerank_calls": 0,
                    "automatic_rerank_errors": 0, "automatic_empty_scopes": 0}
    else:
        _require(isinstance(search, dict) and search.get("version") == ABLATION_VERSION
                 and search.get("variant") == arm, "search-state version/arm")
        counters = _counts({k: v for k, v in search.items() if k not in {"version", "variant"}},
                           "search-state usage")
        _require(all(counters.get(k) == usage[k] for k in ("retrieval_calls", "retrieval_queries")),
                 "search registration counters differ")
        expected = {"search_page_calls": counters.get("search_page_calls"),
            "automatic_rerank_calls": counters.get("auto_rerank_calls", 0),
            "automatic_rerank_errors": counters.get("auto_rerank_errors", 0),
            "automatic_empty_scopes": counters.get("empty_scopes", 0)}
    _require(all(summary.get(k) == v for k, v in expected.items()), "search summary differs")
    for key, value in expected.items():
        _number(value, key, integer=True)
    if arm == "auto_rerank":
        _require(usage["independent_rerank_calls"] == usage["independent_rerank_page_calls"] == 0
                 and expected["search_page_calls"] == 0, "automatic surface exposed manual/page calls")
    else:
        _require(expected["automatic_rerank_calls"] == expected["automatic_rerank_errors"]
                 == expected["automatic_empty_scopes"] == 0, "nonautomatic surface used automatic ranking")

    service = state["service"]
    _require(isinstance(service, dict) and service.get("version") == RERANK_TOOL_VERSION,
             "rerank service version")
    manager = service.get("manager")
    _require(isinstance(manager, dict) and manager.get("version") == STATE_VERSION,
             "accumulated manager version")
    _number(manager.get("space_version"), "pool version", integer=True)
    _require(service.get("cache_version") == manager["space_version"], "cached pool version")
    manager_usage = manager.get("usage")
    _require(isinstance(manager_usage, dict), "missing manager usage")
    physical = terminal.get("reranker_summary")
    _require(isinstance(physical, dict), "missing terminal reranker summary")
    binding = manager.get("passage_index")
    _require(isinstance(binding, dict) and binding.get("binding_version") == 1
             and binding.get("identity") == physical.get("passage_index_identity")
             and manager.get("backend_identity") == physical.get("model_identity")
             and manager.get("chunking_identity") == physical.get("chunking_identity")
             and manager.get("candidates") == physical.get("bm25_candidates")
             and manager.get("top_k") == physical.get("default_top_k"),
             "index/backend/chunking state differs")
    for key, value in manager_usage.items():
        _number(value, "manager usage." + key, integer=not key.endswith("seconds"))
        # RerankService deliberately shadows the manager cache-hit counter.
        if key != "ranking_cache_hits":
            _require(physical.get(key) == value, "manager/terminal usage differs")
    service_usage = _counts(service.get("usage"), "service usage")
    _require(all(physical.get(k) == v for k, v in service_usage.items()),
             "service/terminal usage differs")
    _require(service_usage.get("rerank_calls") == usage["independent_rerank_calls"]
             + expected["automatic_rerank_calls"], "manual/automatic service calls differ")
    _require(isinstance(service.get("history"), list)
             and len(service["history"]) == service_usage["rerank_calls"]
             and isinstance(service.get("rankings"), dict) and isinstance(service.get("results"), dict)
             and len(service["rankings"]) == physical.get("cached_rankings")
             and len(service["results"]) == physical.get("results"), "service cache/history binding differs")
    sources, sets = manager.get("sources"), manager.get("search_sets")
    _require(isinstance(sources, dict) and isinstance(sets, dict), "missing registered source/search sets")
    _require(physical.get("pool_documents") == len(sources)
             and physical.get("search_sets") == len(sets)
             and physical.get("pool_version") == manager["space_version"], "pool summary differs")
    _require(manager.get("counter") == usage["retrieval_calls"], "retrieval call budget differs")
    _require(usage["retrieval_calls"] <= usage["search_invocations"]
             and usage["retrieval_queries"] >= usage["retrieval_calls"], "retrieval invocation budget differs")
    return sources


def verify_trace_checkpoint(output, *, arm, query, config, contract, terminal):
    """Bind trace/CP to one immutable terminal and return only SHA summaries."""
    _require(arm in ARMS and contract == build_contract(arm), "unexpected arm/contract")
    qid = str(query["query_id"])
    question = query.get("question", query.get("query"))
    _require(qid.isdigit() and isinstance(question, str) and bool(question), "question identity")
    output = Path(output)
    trace, trace_sha = _load(output / "_traces" / f"qid_{qid}" / "single.json")
    checkpoint, checkpoint_sha = _load(output / "_checkpoints" / f"qid_{qid}.json")
    anchors = terminal.get("execution_artifacts")
    _require(isinstance(anchors, dict) and anchors.get("trace") == trace_sha
             and anchors.get("checkpoint") == checkpoint_sha, "immutable trace/checkpoint SHA anchors")
    _require(isinstance(trace, dict) and trace.get("schema_version") == SCHEMA_VERSION
             and isinstance(checkpoint, dict) and type(checkpoint.get("version")) is int
             and checkpoint["version"] == 1, "trace/checkpoint schema")
    saved = checkpoint.get("run")
    _require(isinstance(saved, dict), "missing saved run")
    run_hash = compute_run_config_hash(runtime.run_config_for(arm, config).to_hashable_dict())
    identity = {"model": runtime.MODEL, "query_id": qid, "ablation_arm": arm,
        "ablation_runtime_version": runtime.RUNTIME_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "common_config_sha256": runtime.object_sha(config),
        "question_sha256": runtime.text_sha(question), "run_config_hash": run_hash,
        "system_prompt_sha256": contract["system_prompt_sha256"],
        "tool_specs_sha256": contract["tool_specs_sha256"]}
    _require(checkpoint.get("identity") == identity and terminal.get("run_config_hash") == run_hash
             and terminal.get("common_config_sha256") == runtime.object_sha(config), "checkpoint run identity")
    user = contract["user_prompt_template"].format(question=question)
    for key, expected in (("model", runtime.MODEL), ("system_prompt", contract["system_prompt"]),
                          ("user_prompt", user)):
        _require(trace.get(key) == saved.get(key) == expected, "run input identity")
    finals = (trace.get("final_text"), saved.get("final_answer"), terminal.get("final_text"))
    _require(all(value is None or isinstance(value, str) for value in finals)
             and len({value or "" for value in finals}) == 1,
             "public final differs")
    reason = terminal.get("terminated_by")
    _require(isinstance(reason, str) and bool(reason)
             and not reason.startswith(("api_error", "infrastructure", "solver_api_failed"))
             and trace.get("terminated_by") == reason, "natural termination differs")
    turns, saved_turns, totals = trace.get("turns"), saved.get("turns"), trace.get("totals")
    _require(isinstance(turns, list) and isinstance(saved_turns, list) and isinstance(totals, dict),
             "missing turn metadata")
    n = _number(terminal.get("n_turns"), "terminal turns", integer=True)
    _require(n == totals.get("n_turns") == len(turns) == len(saved_turns)
             and n <= config["max_turns"], "turn budget differs")
    _require(saved.get("terminated_reason") == reason or (
        reason == "max_turns" and saved.get("terminated_reason") == "" and n == config["max_turns"]),
        "checkpoint termination differs")
    if reason == "max_turns":
        _require(n == config["max_turns"], "max-turn terminal has unconsumed turns")
    elapsed = _number(terminal.get("elapsed_seconds"), "terminal elapsed")
    cp_elapsed = _number(checkpoint.get("elapsed_seconds"), "checkpoint elapsed")
    _require(trace.get("elapsed_seconds") == elapsed and cp_elapsed <= elapsed + 1e-6
             and (elapsed == 0 or cp_elapsed > 0), "cumulative elapsed budget differs")
    _require(_number(saved.get("elapsed_seconds"), "saved run elapsed") <= cp_elapsed + 1e-6,
             "saved elapsed exceeds checkpoint budget")
    messages = checkpoint.get("messages")
    _require(isinstance(messages, list) and len(messages) >= 2
             and messages[:2] == [{"role": "system", "content": contract["system_prompt"]},
                                  {"role": "user", "content": user}], "checkpoint initial messages")
    token_fields = (("prompt_tokens", "total_prompt_tokens", "prompt"),
                    ("cached_prompt_tokens", "total_cached_prompt_tokens", "cached_prompt"),
                    ("completion_tokens", "total_completion_tokens", "completion"),
                    ("reasoning_tokens", "total_reasoning_tokens", "reasoning"))
    for trace_key, saved_key, turn_key in token_fields:
        total = _number(totals.get(trace_key), "trace " + trace_key, integer=True)
        _require(total == _number(saved.get(saved_key), "checkpoint " + saved_key, integer=True),
                 "trace/checkpoint token totals differ")
        accumulated = 0
        for index, (left, right) in enumerate(zip(turns, saved_turns)):
            _require(isinstance(left, dict) and isinstance(right, dict)
                     and left.get("turn") == right.get("turn") == index, "turn numbering")
            value = _number(left.get("tokens", {}).get(turn_key), "turn token", integer=True)
            _require(value == right.get(trace_key), "turn tokens differ")
            accumulated += value
        _require(total == accumulated, "turn/token sum differs")
    _require(totals["prompt_tokens"] + totals["cached_prompt_tokens"] == terminal.get("input_tokens")
             and totals["completion_tokens"] == terminal.get("output_tokens")
             and totals["cached_prompt_tokens"] == terminal.get("cached_input_tokens")
             and totals["reasoning_tokens"] == terminal.get("reasoning_tokens")
             and totals["prompt_tokens"] == terminal.get("raw_non_cached_input_tokens"),
             "terminal token totals differ")
    counts = Counter()
    for left, right in zip(turns, saved_turns):
        calls, saved_calls = left.get("tool_calls"), right.get("tool_calls")
        _require(isinstance(calls, list) and isinstance(saved_calls, list)
                 and len(calls) == len(saved_calls), "tool call counts differ")
        for call, saved_call in zip(calls, saved_calls):
            _require(isinstance(call.get("name"), str) and call.get("name") == saved_call.get("name")
                     and call.get("error") == saved_call.get("error"), "changed tool call metadata")
            if call["name"] not in {s["function"]["name"] for s in contract["tool_specs"]}:
                # A hallucinated removed capability is an ordinary, scored
                # model error. Invalid JSON is rejected before tool lookup.
                _require(call.get("error") in {"unknown_tool", "json_decode_error"},
                         "unavailable capability was executed")
            counts[call["name"]] += 1
    _require(dict(counts) == totals.get("tool_call_breakdown") == saved.get("tool_call_breakdown")
             == terminal.get("tool_call_breakdown"), "tool call breakdown differs")
    _require(sum(counts.values()) == totals.get("tool_call_count") == saved.get("tool_call_count")
             == terminal.get("n_tool_calls"), "tool invocation totals differ")
    sources = _tool_state(checkpoint.get("tool_state"), arm, terminal)
    docs = checkpoint.get("workspace_docs")
    _require(isinstance(docs, list) and all(isinstance(d, dict) for d in docs), "workspace metadata")
    paths = [doc.get("relpath") for doc in docs]
    _require(all(isinstance(p, str) for p in paths) and len(set(paths)) == len(paths)
             and set(paths) == set(sources), "workspace/pool binding differs")
    return {"trace_sha256": trace_sha, "checkpoint_sha256": checkpoint_sha}
