"""Retry wrapper for DeepSeek (OpenAI-compatible) API calls.

Handles:
- Transient HTTP errors: 429 (rate limit), 5xx (server), connection / timeout
- DeepSeek-specific 'terminated' / partial-response cases (reasoning mid-stream
  is sometimes cut off — the SDK surfaces this as APIError or as a response
  with empty content + suspicious finish_reason)
- Empty content with `finish_reason='stop'` (treated as transient, retried)

Does NOT retry:
- 400 BadRequest (prompt is bad — deterministic)
- 401/403 (auth) — deterministic
- 404 (model not found) — deterministic
- JSON parse errors from the model — the caller should sanitize/fall back

Usage:
    response = chat_with_retry(
        client,
        model="deepseek-v4-pro",
        messages=[...],
        tools=[...],
        max_tokens=8192,
    )
"""
from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
import random
import time
from contextlib import closing
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from openai import AuthenticationError as OpenAIAuthenticationError
from .local_vllm import (
    RequestDeadlineExceeded, apply_budget, complete_response_category,
    extract_reasoning, is_local_qwen, prepare_request as prepare_local_request,
    request_timeout, settings as local_settings,
)

from openai.types.chat import ChatCompletion


# Errors that are deterministic failures — DO NOT retry.
NON_RETRIABLE_ERRORS = (
    BadRequestError,
    OpenAIAuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

# Errors that may be transient — retry with backoff.
RETRIABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class TerminatedResponseError(Exception):
    """Raised when DeepSeek returns a response that looks truncated/terminated."""


# Provider-aware helpers. Both DeepSeek and Xiaomi MiMo are OpenAI-compatible,
# but their token-usage payload differs:
#   - DeepSeek: usage.prompt_cache_hit_tokens (custom top-level field)
#   - MiMo / OpenAI standard: usage.prompt_tokens_details.cached_tokens
# `reasoning_tokens` is consistent across both: completion_tokens_details.reasoning_tokens.

REASONER_MODELS: frozenset[str] = frozenset({
    "deepseek-v4-pro", "deepseek-v4.1-flash", "deepseek-reasoner",
    "mimo-v2.5-pro", "mimo-v2.5",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "z-ai/glm-5.3",
    "qwen3.8-flash", "qwen3.5-27b", "qwen3.5-35b-a3b",
})


def is_reasoner_model(name: str) -> bool:
    """Return True if the given model id is a reasoning model (emits
    `reasoning_content` and benefits from higher `max_tokens`)."""
    if not name:
        return False
    n = name.lower()
    if n in REASONER_MODELS:
        return True
    # gpt-5 series + o-series (o3, o4, o5) are reasoning models.
    if n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5"):
        return True
    return False


def uses_max_completion_tokens(name: str) -> bool:
    """gpt-5 / o-series models reject `max_tokens` and require `max_completion_tokens`."""
    if not name:
        return False
    n = name.lower()
    return n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5")


def is_responses_api_model(name: str) -> bool:
    """Return True for models that should be driven via the Responses API
    (`client.responses.create(...)`) instead of chat-completions.

    For gpt-5 / o-series the Responses API exposes reasoning summary
    text and supports `previous_response_id` for server-side reasoning-state
    continuity across turns — both unavailable via chat.completions.
    """
    if not name:
        return False
    n = name.lower()
    return n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5")


def effective_reasoning_effort(default: str = "medium") -> str:
    """Return the effective reasoning effort for gpt/o-series runs.

    The experiment default is medium; callers can still override with
    REASONING_EFFORT for deliberate ablations.
    """
    import os as _os
    return _os.environ.get("REASONING_EFFORT") or default


def extract_cached_tokens(usage: Any) -> int:
    """Read the cached prompt-token count regardless of provider.

    DeepSeek puts it at `usage.prompt_cache_hit_tokens`. OpenAI/MiMo put it
    at `usage.prompt_tokens_details.cached_tokens`. Try both; return 0 if
    neither field is present or readable.
    """
    if usage is None:
        return 0
    n = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    if n:
        return n
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        try:
            return int(getattr(details, "cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def extract_reasoning_tokens(usage: Any) -> int:
    """Read the reasoning-token count (completion_tokens_details.reasoning_tokens),
    consistent across DeepSeek / OpenAI / MiMo."""
    if usage is None:
        return 0
    d = getattr(usage, "completion_tokens_details", None)
    if d is None:
        return 0
    try:
        return int(getattr(d, "reasoning_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def response_category(response: Any) -> str:
    if not response.choices:
        return "missing_choices"
    choice = response.choices[0]
    msg = choice.message
    if choice.finish_reason == "length":
        return "output_truncated"
    if getattr(msg, "tool_calls", None):
        return "tool_calls"
    if (getattr(msg, "content", None) or "").strip():
        return "content"
    return "reasoning_only" if extract_reasoning(msg) else "empty_response"


def _looks_terminated(response: Any) -> bool:
    """Detect a DeepSeek 'terminated' response.

    Indicators (any of):
    - message.content is empty AND no tool_calls AND finish_reason == 'stop'
      (model returned nothing usable mid-stream)
    - finish_reason explicitly 'error' or contains 'terminated'
    """
    if not response.choices:
        return True
    choice = response.choices[0]
    finish = (choice.finish_reason or "").lower()
    if finish in ("error", "terminated") or "terminated" in finish:
        return True
    msg = choice.message
    content = (msg.content or "").strip()
    has_tool_calls = bool(getattr(msg, "tool_calls", None))
    if not content and not has_tool_calls and finish == "stop":
        return True
    return False


def _stream_chat_completion(client: OpenAI, kwargs: dict[str, Any]) -> ChatCompletion:
    """Reassemble one complete NVIDIA SSE response in chat-completions shape.

    A partial stream is never returned to the agent: the next retry starts the
    same model turn with the same messages.  The terminal usage block is
    required so streamed turns keep the same token accounting as JSON turns.
    """
    stream = client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True},
    )
    first: Any = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: Any = None
    with closing(stream):
        for chunk in stream:
            if first is None:
                first = chunk
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in chunk.choices or []:
                if choice.index != 0:
                    raise TerminatedResponseError("stream returned an unexpected choice index")
                delta = choice.delta
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                reasoning = extract_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                for piece in getattr(delta, "tool_calls", None) or []:
                    index = piece.index
                    call = calls.setdefault(index, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if getattr(piece, "id", None):
                        if call["id"] and call["id"] != piece.id:
                            raise TerminatedResponseError("stream changed a tool-call ID")
                        call["id"] = piece.id
                    if getattr(piece, "type", None) not in (None, "function"):
                        raise TerminatedResponseError("stream returned an unsupported tool type")
                    function = getattr(piece, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            if call["function"]["name"] and call["function"]["name"] != function.name:
                                raise TerminatedResponseError("stream changed a tool name")
                            call["function"]["name"] = function.name
                        if getattr(function, "arguments", None):
                            call["function"]["arguments"] += function.arguments
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

    if first is None or finish_reason is None or usage is None:
        raise TerminatedResponseError("stream ended without a finish reason or usage")
    if calls and (sorted(calls) != list(range(len(calls))) or any(
        not item["id"] or not item["function"]["name"] for item in calls.values()
    )):
        raise TerminatedResponseError("stream ended with incomplete tool calls")
    payload = {
        "id": first.id,
        "object": "chat.completion",
        "created": first.created,
        "model": first.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
                "tool_calls": [calls[index] for index in sorted(calls)] or None,
            },
            "finish_reason": finish_reason,
        }],
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }
    return ChatCompletion.model_validate(payload)


def chat_with_retry(
    client: OpenAI,
    *,
    stream_nvidia: bool = False,
    diagnostic_path: str | Path | None = None,
    max_retries: int = 5,
    initial_backoff: float = 1.5,
    backoff_factor: float = 2.0,
    backoff_cap: float = 30.0,
    jitter: float = 0.25,
    deadline: float | None = None,
    **kwargs: Any,
) -> Any:
    """Call client.chat.completions.create with retry on transient errors.

    Backoff schedule (with jitter): 1.5s, 3s, 6s, 12s, 24s (capped at 30).

    Returns the response object on first success; raises the last exception (or
    a TerminatedResponseError) after `max_retries` failed attempts.
    """
    # gpt-5 / o-series reject `max_tokens`; translate to `max_completion_tokens`.
    model_name = kwargs.get("model", "") or ""
    if is_local_qwen(model_name):
        kwargs = prepare_local_request(kwargs)
    if model_name.lower() == "deepseek-v4.1-flash" and "token-plan.cn-beijing.maas.aliyuncs.com" in os.getenv("DEEPSEEK_BASE_URL", ""):
        # Alibaba's DeepSeek V4.1 endpoint defaults to thinking mode, but
        # request it explicitly so the trajectory includes reasoning_content.
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", True)
        kwargs["extra_body"] = extra_body
    if model_name.lower() == "deepseek-v4.1-flash" and "ark.cn-beijing.volces.com" in os.getenv("DEEPSEEK_BASE_URL", ""):
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("thinking", {"type": "enabled"})
        kwargs["extra_body"] = extra_body
        kwargs.setdefault("reasoning_effort", "high")
    if stream_nvidia and model_name != "nvidia/nemotron-3-super-120b-a12b":
        raise ValueError("stream_nvidia requires nvidia/nemotron-3-super-120b-a12b")
    if uses_max_completion_tokens(model_name) and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    # gpt-5 / o-series don't accept temperature/top_p tweaks either, and need
    # an explicit `reasoning_effort` — the API default is `minimal` which disables
    # CoT (reasoning_tokens=0). Default to `medium` for these experiments, override
    # with REASONING_EFFORT env var (one of minimal/low/medium/high/xhigh).
    if uses_max_completion_tokens(model_name):
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs.setdefault("reasoning_effort", effective_reasoning_effort())

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    local = is_local_qwen(model_name)
    budget = None
    if local:
        if diagnostic_path is None:
            diagnostic_path = Path(os.getenv("RISE_API_DIAGNOSTICS_DIR", "runs/api_diagnostics")) / (uuid.uuid4().hex + ".jsonl")

    def audit(attempt: int, **event: Any) -> None:
        if diagnostic_path is None:
            return
        path = Path(diagnostic_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only allowlisted request fields; no client headers, keys or env dump.
        request = {k: kwargs[k] for k in ("model", "messages", "tools", "tool_choice",
                   "max_tokens", "max_completion_tokens", "temperature", "top_p",
                   "presence_penalty") if k in kwargs}
        if "chat_template_kwargs" in (kwargs.get("extra_body") or {}):
            request["chat_template_kwargs"] = kwargs["extra_body"]["chat_template_kwargs"]
        if local:
            request["extra_body"] = {k: v for k, v in (kwargs.get("extra_body") or {}).items()
                                     if k in {"chat_template_kwargs", "top_k", "min_p", "repetition_penalty"}}
        record = {"time": time.time(), "attempt": attempt + 1, "budget": budget,
                  "adapter_settings": local_settings() if local else None,
                  "request": request, **event}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    if local:
        try:
            budget = apply_budget(client, kwargs, deadline=deadline)
        except Exception as e:
            audit(0, category="token_budget_preflight_failed", error=str(e))
            raise

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if deadline is not None and time.monotonic() >= deadline:
            raise RequestDeadlineExceeded("Agent deadline exhausted before API request")
        try:
            started = time.monotonic()
            call_kwargs = dict(kwargs)
            request_client = client
            if local:
                call_kwargs["timeout"] = request_timeout(deadline)
                if hasattr(client, "with_options"):
                    request_client = client.with_options(max_retries=0)
            response = (
                _stream_chat_completion(request_client, call_kwargs) if stream_nvidia
                else request_client.chat.completions.create(**call_kwargs)
            )
            if local:
                metadata = {"attempt": attempt + 1, "elapsed_seconds": time.monotonic() - started,
                            "request_timeout_seconds": call_kwargs["timeout"],
                            "deadline_exceeded": deadline is not None and time.monotonic() >= deadline}
                object.__setattr__(response, "_rise_api_metadata", metadata)
            category = complete_response_category(response) if local else response_category(response)
            audit(attempt, category=category, elapsed_seconds=time.monotonic() - started,
                  response=response.model_dump(mode="json") if hasattr(response, "model_dump") else str(response))
            if local:
                # Semantic incompleteness is not a transport retry. Agent
                # records the actual usage and terminal failure exactly once.
                return response
            if _looks_terminated(response) or (local and response.choices and not (response.choices[0].message.content or "").strip() and not response.choices[0].message.tool_calls):
                last_exc = TerminatedResponseError(
                    f"{category} response on attempt {attempt + 1}; "
                    f"finish_reason={response.choices[0].finish_reason if response.choices else None!r}; diagnostics={diagnostic_path}"
                )
            else:
                return response
        except NON_RETRIABLE_ERRORS as e:
            # Deterministic failure — propagate immediately
            audit(attempt, category=type(e).__name__, error=str(e), body=getattr(e, "body", None))
            raise
        except RateLimitError as e:
            # Token Plan monthly exhaustion and per-minute token throttling
            # can both carry insufficient_quota. Distinguish by message.
            detail = str(e).lower()
            if "token-plan quota has been exhausted" in detail:
                raise
            last_exc = e
        except RETRIABLE_ERRORS as e:
            last_exc = e
        except APIError as e:
            # Other API errors — treat as retriable but be cautious
            last_exc = e
        except OSError:
            # A failed audit write must not trigger another paid generation.
            raise
        except Exception as e:
            # Unknown error — log and retry once or twice
            if local:
                audit(attempt, category=type(e).__name__, error=str(e))
                raise
            last_exc = e

        if last_exc is not None and not isinstance(last_exc, TerminatedResponseError):
            audit(attempt, category=type(last_exc).__name__, error=str(last_exc), body=getattr(last_exc, "body", None))
        # backoff with multiplicative + jitter
        if deadline is not None and time.monotonic() >= deadline:
            raise RequestDeadlineExceeded("Agent deadline exhausted during API request") from last_exc
        if attempt < max_retries - 1:
            backoff = min(backoff_cap, initial_backoff * (backoff_factor ** attempt))
            backoff *= 1.0 + random.uniform(-jitter, jitter)
            delay = max(0.5, backoff)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise RequestDeadlineExceeded("Agent deadline cannot fund API retry backoff") from last_exc
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"chat_with_retry: {max_retries} attempts failed (no exception captured)")


# ----- Responses API path (gpt-5 / o-series, OpenAI) -----


def _looks_terminated_responses(response: Any) -> bool:
    """Detect a Responses-API truncation/empty response. Returns True if the
    `output` list is empty AND there's no message content / function_calls."""
    out = getattr(response, "output", None) or []
    if not out:
        # No output blocks at all → likely an aborted stream
        return True
    # Accept any non-empty output as valid (reasoning-only blocks are OK on
    # turns where the model immediately calls a tool with no visible content).
    return False


def responses_with_retry(
    client: Any,
    *,
    max_retries: int = 5,
    initial_backoff: float = 1.5,
    backoff_factor: float = 2.0,
    backoff_cap: float = 30.0,
    jitter: float = 0.25,
    **kwargs: Any,
) -> Any:
    """Call `client.responses.create(...)` with retry on transient errors.

    Mirrors `chat_with_retry()` but for the Responses API. Forwards all kwargs
    verbatim; caller is responsible for the request shape (`input`, `tools`,
    `reasoning`, `prompt_cache_key`, `previous_response_id`, etc.).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**kwargs)
            if _looks_terminated_responses(response):
                last_exc = TerminatedResponseError(
                    f"responses output looked empty on attempt {attempt + 1}"
                )
            else:
                return response
        except NON_RETRIABLE_ERRORS:
            raise
        except RETRIABLE_ERRORS as e:
            last_exc = e
        except APIError as e:
            last_exc = e
        except Exception as e:
            last_exc = e

        if attempt < max_retries - 1:
            backoff = min(backoff_cap, initial_backoff * (backoff_factor ** attempt))
            backoff *= 1.0 + random.uniform(-jitter, jitter)
            time.sleep(max(0.5, backoff))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        f"responses_with_retry: {max_retries} attempts failed (no exception captured)"
    )


def extract_responses_usage(usage: Any) -> dict[str, int]:
    """Pull token counts out of a Responses-API `usage` object.

    Responses API: `input_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens`,
    `input_tokens_details.cached_tokens`, `total_tokens`. Returns a dict with the
    same keys used elsewhere in the codebase so cost estimation flows
    unchanged (input_tokens=uncached, cached_input_tokens, output_tokens
    [includes reasoning per OpenAI convention], reasoning_tokens).
    """
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
    input_total = int(getattr(usage, "input_tokens", 0) or 0)
    output_total = int(getattr(usage, "output_tokens", 0) or 0)
    cached = 0
    in_details = getattr(usage, "input_tokens_details", None)
    if in_details is not None:
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    reasoning = 0
    out_details = getattr(usage, "output_tokens_details", None)
    if out_details is not None:
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": max(0, input_total - cached),  # UNCACHED prompt tokens
        "cached_input_tokens": cached,
        "output_tokens": output_total,  # includes reasoning per OpenAI convention
        "reasoning_tokens": reasoning,
    }
