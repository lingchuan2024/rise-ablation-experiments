"""BrowseComp-Plus paper-aligned retrieval agent.

Paper-faithful settings (BCP §4.3, §E, and `texttron/BrowseComp-Plus`
`search_agent/{openai_client,prompts}.py`):
  - Two tools (paper's main setting): `search(query)` and `get_document(docid)`.
    `search` returns top-k=5 docs with first-512-token snippets; `get_document`
    returns the full text of one docid. Get-doc can be disabled for the
    ablation in §4.8.3.
  - Prompts: BCP `QUERY_TEMPLATE` (with both tools) or
    `QUERY_TEMPLATE_NO_GET_DOCUMENT` (search only), both verbatim.
  - `max_iterations=100` (BCP code default; counts model API calls, not
    individual tool calls — matches paper's outer-loop semantics).
  - Citation format in agent output: `[docid]`, e.g. `[20]`.

Tracks per-run: token usage (input / cached_input / output / reasoning),
tool calls (count + per-query log), retrieved docid set, elapsed time.
"""
from __future__ import annotations

from .local_vllm import extract_reasoning

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

import bm25s
from openai import OpenAI

from .api_retry import chat_with_retry
from .retrieval import retrieve


# ----- BCP Appendix E prompts (verbatim from texttron/BrowseComp-Plus
# `search_agent/prompts.py`) — used by retrieval-agent baselines -----
#
# With both `search` and `get_document` tools (paper's main setting):
BCP_E_PROMPT_WITH_GET_DOC = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: {question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

# Search-only ablation (BCP `QUERY_TEMPLATE_NO_GET_DOCUMENT`):
BCP_E_PROMPT_SEARCH_ONLY = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

# Back-compat alias for older runners that import `BCP_E_PROMPT`.
BCP_E_PROMPT = BCP_E_PROMPT_SEARCH_ONLY


# ----- DCI paper §C.1 prompt (verbatim) — used by DCI baseline -----
# Reproduces the paper's "DCI-Agent Prompt" appendix entry exactly. `@corpus`
# in the paper is substituted with the absolute corpus directory at runtime
# (matches the convention in the DCI-Agent-Lite codebase). Replace-substituted
# (not Python .format) at call time, so the `{{...}}` placeholders survive
# verbatim into the LLM-facing string, matching the paper's rendering.
DCI_PAPER_C1_PROMPT = """You are a careful research assistant. Answer the question below using ONLY documents in @{corpus_dir}. Do not use online search or any external tools beyond ripgrep and Bash.

Question: {question}

SEARCH STRATEGY (follow exactly):
1. Search directly using ripgrep/Bash — do NOT use the Agent tool, spawn subagents, or browse the web.
2. Run multiple ripgrep/Bash searches IN PARALLEL within a single response to save time.
3. Use diverse, targeted keywords to maximize recall before drawing conclusions.

INSTRUCTIONS:
• Search @{corpus_dir} thoroughly with multiple relevant keyword combinations.
• Identify and rule out competing candidate answers before committing to one.
• Cite every supporting finding inline using the document's path, e.g. [@{corpus_dir}/relative_path].

Your response MUST follow this exact format:
Explanation: {{step-by-step reasoning with inline, e.g. [@{corpus_dir}/relative_path]}}
Exact Answer: {{concise final answer only}}
Confidence: {{0–100%; use below 50% if evidence is weak, ambiguous, or missing}}"""

# Back-compat alias for older runners.
BCP_E_PROMPT_DCI_FILESYSTEM = DCI_PAPER_C1_PROMPT


# ----- Simplified BCP §E variant for the BM25-DCI single-agent setup -----
# Drops the citation requirement (no [path/to/file.txt] inline citations), and
# explicitly allows "Unknown" / "Unable to solve" when the mini-corpus doesn't
# contain enough information. The Explanation/Exact Answer/Confidence three-
# section structure is preserved so BCP §F judge can still extract cleanly.
# Anti-escape directive: agent must stay within cwd; absolute paths forbidden.
BCP_E_PROMPT_NO_CITATIONS = """Answer the following question using only the documents in the corpus directory at @{corpus_dir}. **Do Not use web search!** Use ripgrep (rg) instead of grep for fast searching. **Do not use absolute paths or `cd` to directories outside the current working directory.** All needed evidence has been pre-staged here; if it is not in the cwd, the corpus does not have the answer.

QUESTION:
{question}

Your response should be in the following format:
Explanation: {{your detailed reasoning and the supporting evidence you found in the documents}}
Exact Answer: {{your succinct, final answer, or "Unknown" if the corpus does not contain enough information to solve the question}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


@dataclass
class BcpRunResult:
    query_id: Any
    question: str
    final_text: str = ""
    rounds: list[dict] = field(default_factory=list)
    all_retrieved: list[str] = field(default_factory=list)
    search_calls: int = 0
    get_doc_calls: int = 0
    # token usage
    input_tokens: int = 0           # uncached prompt tokens
    cached_input_tokens: int = 0    # prefix-cache hits
    output_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0              # number of chat.completions roundtrips (= iterations)
    terminated_by: str = ""
    elapsed_seconds: float = 0.0
    # Full per-turn trajectory — list of TurnTrace dicts (see trajectory.py).
    turns: list[dict] = field(default_factory=list)
    # Per-turn cost rollup (= sum of TurnTrace.cost_usd).
    agent_cost_usd: float = 0.0


def _search_tool_def(k: int = 5) -> dict:
    """Byte-faithful port of `texttron/BrowseComp-Plus`:
    - `search_agent/openai_client.py:42-58` (schema shape, strict mode)
    - `searcher/searchers/base.py:57-61` (description template)
    The chat.completions wrapper (`type: function`, `function: {...}`) is
    our adapter — the inner `function` body is byte-identical to upstream's
    Responses-API tool dict (sans the wrapper)."""
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                f"Perform a search on a knowledge source. Returns top-{k} "
                f"hits with docid, score, and snippet. The snippet contains "
                f"the document's contents (may be truncated based on token "
                f"limits)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _get_document_tool_def() -> dict:
    """Byte-faithful port of:
    - `texttron/BrowseComp-Plus` `search_agent/openai_client.py:62-79`
    - `searcher/searchers/base.py:63-67`"""
    return {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve a full document by its docid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": "Document ID to retrieve",
                    },
                },
                "required": ["docid"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _truncate_words(text: str, max_words: int = 512) -> str:
    """Word-based approximation of 512-token truncation.

    The BCP paper says 'first 512 tokens'. Without DeepSeek's exact tokenizer,
    we use whitespace-split word count, which for English is close enough.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


import threading

_QWEN_TOKENIZER_SINGLETON = None
_QWEN_TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
_QWEN_TOKENIZER_LOCK = threading.Lock()


def _get_qwen_tokenizer():
    """Lazy-load Qwen3-0.6B tokenizer (byte-faithful to upstream BCP).
    `transformers.AutoTokenizer.from_pretrained` doesn't require torch.
    First call downloads ~50MB to `~/.cache/huggingface/`; cached after.
    Process-singleton so concurrent threads share the same instance.

    Thread-safety: `transformers` does its own lazy attribute resolution
    inside `from transformers import AutoTokenizer`, so concurrent first
    calls (e.g. from a ThreadPoolExecutor) can race and see a partial
    module state — manifests as `ImportError: cannot import name
    'AutoTokenizer'`. We serialize the first load with a lock; the
    second-and-later callers take the fast path with no contention.
    Callers can also pre-warm via `warmup_qwen_tokenizer()` from the
    main thread before spawning workers.
    """
    global _QWEN_TOKENIZER_SINGLETON
    if _QWEN_TOKENIZER_SINGLETON is None:
        with _QWEN_TOKENIZER_LOCK:
            if _QWEN_TOKENIZER_SINGLETON is None:
                from transformers import AutoTokenizer
                _QWEN_TOKENIZER_SINGLETON = AutoTokenizer.from_pretrained(_QWEN_TOKENIZER_NAME)
    return _QWEN_TOKENIZER_SINGLETON


def warmup_qwen_tokenizer() -> None:
    """Force-load the tokenizer on the calling thread. Use this from the
    runner main thread before any ThreadPoolExecutor work to avoid the
    `transformers` lazy-import race."""
    _get_qwen_tokenizer()


def _truncate_to_qwen_tokens(text: str, max_tokens: int) -> str:
    """Byte-faithful port of upstream BCP `_search` snippet truncation
    (texttron/BrowseComp-Plus `search_agent/openai_client.py:96-108`):

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > snippet_max_tokens:
            truncated = tokens[:snippet_max_tokens]
            snippet = tokenizer.decode(truncated, skip_special_tokens=True)
        else:
            snippet = text
    """
    if not text or max_tokens <= 0:
        return text
    tok = _get_qwen_tokenizer()
    tokens = tok.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tok.decode(tokens[:max_tokens], skip_special_tokens=True)


def _format_search_results(
    retrieved: list[tuple[str, float]],
    corpus_by_docid: Mapping[str, str],
    snippet_max_tokens: int = 512,
) -> str:
    """Byte-faithful port of `texttron/BrowseComp-Plus`
    `search_agent/openai_client.py::_search` (lines 92-123):
    returns `json.dumps(results, indent=2)` where each result is
    `{"docid": ..., "score": ..., "snippet": ...}` (or no `score` when
    the searcher didn't return one).

    Snippet truncation uses the upstream-faithful `Qwen/Qwen3-0.6B`
    tokenizer — token-level slice, matches upstream byte-for-byte.
    """
    import json as _json
    if not retrieved:
        return _json.dumps([], indent=2)
    results: list[dict] = []
    for docid, score in retrieved:
        text = corpus_by_docid.get(docid, "") or ""
        snippet = _truncate_to_qwen_tokens(text, snippet_max_tokens) if snippet_max_tokens and snippet_max_tokens > 0 else text
        item: dict = {"docid": docid}
        if score is not None:
            item["score"] = float(score)
        item["snippet"] = snippet
        results.append(item)
    return _json.dumps(results, indent=2)


def _accumulate_usage(run: BcpRunResult, usage: Any) -> None:
    """Pull tokens out of an openai-compatible usage object (DeepSeek or MiMo)."""
    if not usage:
        return
    from .api_retry import extract_cached_tokens, extract_reasoning_tokens
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    cached = extract_cached_tokens(usage)
    # prompt_tokens already includes cached; uncached = prompt - cached
    run.cached_input_tokens += cached
    run.input_tokens += max(0, prompt_tokens - cached)
    run.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
    run.reasoning_tokens += extract_reasoning_tokens(usage)
    run.api_calls += 1


def run_bcp_retrieval_agent(
    question: str,
    query_id: Any,
    *,
    run_config: "RunConfig",
    searcher: bm25s.BM25,
    doc_ids: Sequence[str],
    corpus_by_docid: Mapping[str, str],
    client: OpenAI,
    is_reasoner: bool = True,
) -> BcpRunResult:
    """Run one query through the BCP paper-aligned retrieval agent.

    All budget/timeout/retry knobs come from `run_config`. Method-specific
    knobs come from `run_config.extras`:

      - `per_search_k` (default 5): top-k passed to the search tool
      - `snippet_tokens` (default 512): Qwen-tokenizer snippet cap
      - `enable_get_document` (default True): include the get_document tool

    `run_config.max_model_calls` maps to upstream BCP's `max_iterations`
    (`openai_client.py:175` — counts model API calls, not tool calls).
    `run_config.per_call_max_tokens` overrides the legacy hardcoded
    8192-for-reasoners / 1024-for-others fallback (still used when this is
    unset, e.g. on legacy callers that bypass run_config).
    `is_reasoner` stays a runtime flag (depends on model id semantics, not
    config).

    For gpt-5 / o-series models, this routes to the Responses API path
    (`_run_bcp_via_responses_api`). Other providers (DeepSeek, Mimo) take
    the chat.completions path below.
    """
    extras = run_config.extras
    model = run_config.agent_model
    per_search_k = int(extras.get("per_search_k", 5))
    snippet_max_tokens = int(extras.get("snippet_tokens", 512))
    enable_get_document = bool(extras.get("enable_get_document", True))
    max_iterations = run_config.max_model_calls

    from .api_retry import is_responses_api_model
    if is_responses_api_model(model):
        return _run_bcp_via_responses_api(
            question=question, query_id=query_id,
            run_config=run_config,
            searcher=searcher, doc_ids=doc_ids, corpus_by_docid=corpus_by_docid,
            client=client,
        )

    from .trajectory import make_turn_from_response, ToolCallTrace, search_result_summary, doc_result_summary

    run = BcpRunResult(query_id=query_id, question=question)
    t0 = time.time()

    prompt_template = (
        BCP_E_PROMPT_WITH_GET_DOC if enable_get_document else BCP_E_PROMPT_SEARCH_ONLY
    )
    user_prompt = prompt_template.format(question=question)
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    tools = [_search_tool_def(k=per_search_k)]
    if enable_get_document:
        tools.append(_get_document_tool_def())

    # Per-call output token cap from RunConfig. Legacy reasoner/non-
    # reasoner split is only used when RunConfig leaves per_call_max_tokens
    # unset (zero-value sentinel).
    per_call_max_tokens = run_config.per_call_max_tokens if run_config.per_call_max_tokens else (8192 if is_reasoner else 1024)
    wall_clock_timeout_sec = run_config.wall_clock_timeout_sec

    for _ in range(max_iterations):
        # Wall-clock budget enforcement (Step A: parity with Agent class).
        # Set `terminated_by="wall_clock_timeout"` so downstream coverage /
        # cost accounting can distinguish "agent ran out of time" from
        # "agent finished cleanly" / "agent hit max_iterations".
        if (time.time() - t0) > wall_clock_timeout_sec:
            run.terminated_by = "wall_clock_timeout"
            break
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": per_call_max_tokens,
            }
            if not is_reasoner:
                kwargs["temperature"] = 0.3
            response = chat_with_retry(
                client, max_retries=run_config.api_max_retries, **kwargs
            )
        except Exception as e:
            run.terminated_by = f"api_error: {type(e).__name__}: {str(e)[:200]}"
            break

        _accumulate_usage(run, response.usage)

        # Per-turn trace (tokens + cost computed from live usage).
        turn_trace = make_turn_from_response(turn_idx=len(run.turns), response=response, model=model)
        run.agent_cost_usd += turn_trace.cost_usd

        msg = response.choices[0].message

        # Build assistant message for history; preserve reasoning_content for v4-pro
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        rc = extract_reasoning(msg)
        if rc:
            assistant_msg["reasoning_content"] = rc
        messages.append(assistant_msg)

        if not msg.tool_calls:
            # Agent stopped calling tools — final answer in msg.content
            run.final_text = msg.content or ""
            run.terminated_by = "no_tool_call"
            run.turns.append(turn_trace.to_dict())
            break

        # Execute every tool call this turn
        for tc in msg.tool_calls:
            name = tc.function.name
            args_raw = tc.function.arguments or ""
            try:
                args = json.loads(args_raw or "{}")
            except Exception:
                args = {}

            if name == "search":
                q = (args.get("query") or "").strip()
                t_tool = time.time()
                results = retrieve(searcher, doc_ids, q, k=per_search_k) if q else []
                tool_duration = time.time() - t_tool
                for did, _ in results:
                    if did not in run.all_retrieved:
                        run.all_retrieved.append(did)
                run.search_calls += 1
                snippet_text = _format_search_results(results, corpus_by_docid, snippet_max_tokens=snippet_max_tokens)
                result_docids = [d for d, _ in results]
                run.rounds.append({
                    "call_idx": run.search_calls + run.get_doc_calls,
                    "tool": "search",
                    "query": q,
                    "result_docids": result_docids,
                })
                turn_trace.tool_calls.append(ToolCallTrace(
                    id=tc.id,
                    name="search",
                    args=args,
                    args_raw=args_raw,
                    result_docids=result_docids,
                    result_summary=search_result_summary(q, len(result_docids)),
                    result_chars=len(snippet_text),
                    result_text_preview=snippet_text[:200],
                    result_truncated=False,
                    duration_seconds=tool_duration,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": snippet_text,
                })
            elif name == "get_document" and enable_get_document:
                docid = str(args.get("docid") or "").strip()
                t_tool = time.time()
                text = corpus_by_docid.get(docid)
                # Byte-faithful to texttron `_get_document` (openai_client.py:125-129):
                # error case = compact json; hit case = indent=2 pretty JSON.
                # Default `ensure_ascii=True` matches upstream (non-ASCII → \uNNNN).
                if text is None:
                    payload = json.dumps({"error": f"Document with docid '{docid}' not found"})
                else:
                    payload = json.dumps({"docid": docid, "text": text}, indent=2)
                tool_duration = time.time() - t_tool
                if docid and docid not in run.all_retrieved:
                    run.all_retrieved.append(docid)
                run.get_doc_calls += 1
                run.rounds.append({
                    "call_idx": run.search_calls + run.get_doc_calls,
                    "tool": "get_document",
                    "docid": docid,
                    "found": text is not None,
                })
                turn_trace.tool_calls.append(ToolCallTrace(
                    id=tc.id,
                    name="get_document",
                    args=args,
                    args_raw=args_raw,
                    result_docids=[docid] if text is not None else [],
                    result_summary=doc_result_summary(docid, len(text) if text is not None else None),
                    result_chars=len(payload),
                    result_text_preview=payload[:200],
                    result_truncated=False,
                    duration_seconds=tool_duration,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": payload,
                })
            else:
                unknown_msg = f"Unknown tool: {name}"
                turn_trace.tool_calls.append(ToolCallTrace(
                    id=tc.id,
                    name=name,
                    args=args,
                    args_raw=args_raw,
                    result_summary=unknown_msg,
                    result_chars=len(unknown_msg),
                    result_text_preview=unknown_msg[:200],
                    error="unknown_tool",
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": unknown_msg,
                })

        run.turns.append(turn_trace.to_dict())
    else:
        # for-else: loop completed without `break` → max_iterations exhausted
        run.terminated_by = "max_iterations"

    run.elapsed_seconds = time.time() - t0
    return run


# ----- Responses API path (gpt-5 / o-series) -----


def _run_bcp_via_responses_api(
    *,
    question: str,
    query_id: Any,
    run_config: "RunConfig",
    searcher: bm25s.BM25,
    doc_ids: Sequence[str],
    corpus_by_docid: Mapping[str, str],
    client: OpenAI,
) -> BcpRunResult:
    """BCP retrieval-agent loop on the Responses API.

    Same tool surface as the chat.completions path (`search` + optional
    `get_document`) but driven via `client.responses.create()` with
    `previous_response_id` chaining and `reasoning.summary="auto"`.
    """
    extras = run_config.extras
    model = run_config.agent_model
    per_search_k = int(extras.get("per_search_k", 5))
    snippet_max_tokens = int(extras.get("snippet_tokens", 512))
    enable_get_document = bool(extras.get("enable_get_document", True))
    max_iterations = run_config.max_model_calls
    per_call_max_tokens = run_config.per_call_max_tokens
    wall_clock_timeout_sec = run_config.wall_clock_timeout_sec

    import hashlib
    from .api_retry import responses_with_retry, extract_responses_usage, effective_reasoning_effort
    from .dci_artifacts import estimate_cost
    from .trajectory import ToolCallTrace, search_result_summary, doc_result_summary

    prompt_template = BCP_E_PROMPT_WITH_GET_DOC if enable_get_document else BCP_E_PROMPT_SEARCH_ONLY
    user_prompt = prompt_template.format(question=question)

    # Tool specs in Responses-API shape (no `function` wrapper).
    # Upstream (texttron/BrowseComp-Plus openai_client.py:57) sets
    # `"strict": True` + `additionalProperties: false`; we preserve those
    # from `_search_tool_def`/`_get_document_tool_def` rather than
    # overriding them here.
    def _tool_specs() -> list[dict]:
        chat_specs = [_search_tool_def(k=per_search_k)]
        if enable_get_document:
            chat_specs.append(_get_document_tool_def())
        out = []
        for spec in chat_specs:
            fn = spec.get("function") or spec
            entry: dict = {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
            # Honor `strict` from the inner spec (upstream sets it True).
            if "strict" in fn:
                entry["strict"] = fn["strict"]
            out.append(entry)
        return out
    tools_responses = _tool_specs()

    cache_key = hashlib.sha256(
        (model + "::bcp_retrieval::" + (BCP_E_PROMPT_WITH_GET_DOC if enable_get_document else BCP_E_PROMPT_SEARCH_ONLY)[:512]).encode("utf-8")
    ).hexdigest()[:48]
    reasoning_effort = effective_reasoning_effort()

    def _one_attempt(active_client: OpenAI) -> BcpRunResult:
        run = BcpRunResult(query_id=query_id, question=question)
        t0 = time.time()
        next_input: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        previous_response_id: str | None = None

        for _ in range(max_iterations):
            # Wall-clock budget (Step A: parity with Agent class).
            if (time.time() - t0) > wall_clock_timeout_sec:
                run.terminated_by = "wall_clock_timeout"
                break
            kwargs: dict[str, Any] = {
                "model": model,
                "input": next_input,
                "tools": tools_responses,
                "tool_choice": "auto",
                # Documented divergence from upstream BCP (`build_request`
                # at openai_client.py:153 sets `truncation: "auto"`). We
                # omit it for cross-baseline consistency: RISE, Pi-Serini,
                # and DCI native all run with truncation=disabled (the
                # Responses API default). When context overflow happens,
                # the API returns 400 and the query terminates with
                # `terminated_by="api_error: ..."` instead of running
                # many expensive cache-miss turns triggered by
                # truncation's middle-message drop. Mini_dev_10 audit
                # showed 4 hard queries spending $20+ on truncation
                # cache-thrash without ever reaching the gold doc; with
                # truncation=disabled those would 400 earlier — same
                # ✗ outcome at <$1/query.
                "reasoning": {"effort": reasoning_effort, "summary": "auto"},
                "prompt_cache_key": cache_key,
                "max_output_tokens": per_call_max_tokens,
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id
            try:
                response = responses_with_retry(
                    active_client, max_retries=run_config.api_max_retries, **kwargs
                )
            except Exception as e:
                run.terminated_by = f"api_error: {type(e).__name__}: {str(e)[:200]}"
                break

            previous_response_id = getattr(response, "id", None)
            status = getattr(response, "status", "") or ""

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            function_calls: list[Any] = []
            for blk in (getattr(response, "output", None) or []):
                btype = getattr(blk, "type", "")
                if btype == "reasoning":
                    for s in (getattr(blk, "summary", None) or []):
                        st = getattr(s, "text", None) or ""
                        if st:
                            reasoning_parts.append(st)
                elif btype == "message":
                    for c in (getattr(blk, "content", None) or []):
                        if getattr(c, "type", "") == "output_text":
                            ct = getattr(c, "text", None) or ""
                            if ct:
                                text_parts.append(ct)
                elif btype == "function_call":
                    function_calls.append(blk)
            content = "\n".join(text_parts)
            reasoning_content = "\n".join(reasoning_parts)

            u = extract_responses_usage(getattr(response, "usage", None))
            turn_cost = estimate_cost(
                {"input_tokens": u["input_tokens"], "cached_input_tokens": u["cached_input_tokens"], "output_tokens": u["output_tokens"]},
                model,
            )
            run.input_tokens += u["input_tokens"]
            run.cached_input_tokens += u["cached_input_tokens"]
            run.output_tokens += u["output_tokens"]
            run.reasoning_tokens += u["reasoning_tokens"]
            run.api_calls += 1
            run.agent_cost_usd += turn_cost

            turn_record: dict[str, Any] = {
                "turn": len(run.turns),
                "finish_reason": status,
                "content": content,
                "reasoning_content": reasoning_content,
                "tokens": {
                    "prompt": u["input_tokens"], "cached_prompt": u["cached_input_tokens"],
                    "completion": u["output_tokens"], "reasoning": u["reasoning_tokens"],
                },
                "cost_usd": turn_cost,
                "tool_calls": [],
            }

            if not function_calls:
                run.final_text = content
                run.terminated_by = "no_tool_call"
                run.turns.append(turn_record)
                break

            next_input = []
            for fc in function_calls:
                name = getattr(fc, "name", "") or ""
                call_id = getattr(fc, "call_id", None) or getattr(fc, "id", "") or ""
                args_raw = getattr(fc, "arguments", "") or ""
                try:
                    args = json.loads(args_raw or "{}")
                except Exception:
                    args = {}

                if name == "search":
                    q = (args.get("query") or "").strip()
                    t_tool = time.time()
                    results = retrieve(searcher, doc_ids, q, k=per_search_k) if q else []
                    tool_duration = time.time() - t_tool
                    for did, _score in results:
                        if did not in run.all_retrieved:
                            run.all_retrieved.append(did)
                    run.search_calls += 1
                    snippet_text = _format_search_results(results, corpus_by_docid, snippet_max_tokens=snippet_max_tokens)
                    result_docids = [d for d, _ in results]
                    run.rounds.append({
                        "call_idx": run.search_calls + run.get_doc_calls,
                        "tool": "search", "query": q, "result_docids": result_docids,
                    })
                    turn_record["tool_calls"].append(ToolCallTrace(
                        id=call_id, name="search", args=args, args_raw=args_raw,
                        result_docids=result_docids,
                        result_summary=search_result_summary(q, len(result_docids)),
                        result_chars=len(snippet_text),
                        result_text_preview=snippet_text[:200],
                        duration_seconds=tool_duration,
                    ).to_dict())
                    next_input.append({"type": "function_call_output", "call_id": call_id, "output": snippet_text})
                elif name == "get_document" and enable_get_document:
                    docid = str(args.get("docid") or "").strip()
                    t_tool = time.time()
                    text = corpus_by_docid.get(docid)
                    # Byte-faithful to texttron `_get_document` (see above note).
                    if text is None:
                        payload = json.dumps({"error": f"Document with docid '{docid}' not found"})
                    else:
                        payload = json.dumps({"docid": docid, "text": text}, indent=2)
                    tool_duration = time.time() - t_tool
                    if docid and docid not in run.all_retrieved:
                        run.all_retrieved.append(docid)
                    run.get_doc_calls += 1
                    run.rounds.append({
                        "call_idx": run.search_calls + run.get_doc_calls,
                        "tool": "get_document", "docid": docid, "found": text is not None,
                    })
                    turn_record["tool_calls"].append(ToolCallTrace(
                        id=call_id, name="get_document", args=args, args_raw=args_raw,
                        result_docids=[docid] if text is not None else [],
                        result_summary=doc_result_summary(docid, len(text) if text is not None else None),
                        result_chars=len(payload),
                        result_text_preview=payload[:200],
                        duration_seconds=tool_duration,
                    ).to_dict())
                    next_input.append({"type": "function_call_output", "call_id": call_id, "output": payload})
                else:
                    unknown = f"Unknown tool: {name}"
                    turn_record["tool_calls"].append(ToolCallTrace(
                        id=call_id, name=name, args=args, args_raw=args_raw,
                        result_summary=unknown, result_chars=len(unknown),
                        result_text_preview=unknown[:200],
                        error="unknown_tool",
                    ).to_dict())
                    next_input.append({"type": "function_call_output", "call_id": call_id, "output": unknown})

            run.turns.append(turn_record)
        else:
            run.terminated_by = "max_iterations"

        run.elapsed_seconds = time.time() - t0
        return run

    return _one_attempt(client)


# ----- BCP Appendix F judge (verbatim) -----
BCP_F_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Return ONLY a compact JSON object with keys: extracted_final_answer, reasoning, correct, confidence."""


def bcp_judge(
    question: str,
    correct_answer: str,
    response: str,
    *,
    client: OpenAI,
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    """Judge one (question, correct_answer, response) tuple per BCP §F.

    Returns dict {extracted_final_answer, reasoning, correct (yes/no),
    confidence, _usage}.
    """
    user_prompt = BCP_F_JUDGE_PROMPT.format(
        question=question,
        response=response or "[empty]",
        correct_answer=correct_answer,
    )
    from .api_retry import extract_cached_tokens, extract_reasoning_tokens, is_reasoner_model
    is_reasoner = is_reasoner_model(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096 if is_reasoner else 512,
    }
    if not is_reasoner:
        kwargs["temperature"] = 0.0
    if model.lower() == "z-ai/glm-5.3":
        # GLM-5.3 defaults to maximum reasoning; a short judging task needs
        # enough visible JSON output within the 4096-token reasoner budget.
        kwargs["reasoning_effort"] = "low"

    response_obj = chat_with_retry(client, **kwargs)
    raw = response_obj.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    raw_correct = data.get("correct")
    if isinstance(raw_correct, bool):
        correct = "yes" if raw_correct else "no"
    else:
        correct = str(raw_correct).strip().lower()
        correct = {"true": "yes", "false": "no"}.get(correct, correct)
    if correct not in ("yes", "no"):
        raise ValueError("Judge response has no valid correct label")

    usage = response_obj.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    cached = extract_cached_tokens(usage)
    return {
        "extracted_final_answer": data.get("extracted_final_answer", "None"),
        "reasoning": data.get("reasoning", ""),
        "correct": correct,
        "confidence": data.get("confidence", 100),
        "_usage": {
            "input_tokens": max(0, prompt_tokens - cached),
            "cached_input_tokens": cached,
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": extract_reasoning_tokens(usage) if is_reasoner else 0,
        },
    }
