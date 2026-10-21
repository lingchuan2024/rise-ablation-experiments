"""Local Qwen/vLLM compatibility and exact server-side token budgeting."""
from __future__ import annotations

import os
import math
import json
import time
from typing import Any

LOCAL_MODELS = {"qwen3.5-27b", "qwen3.5-35b-a3b"}
ADAPTER_VERSION = "qwen-vllm-v4-native-history"


class ContextBudgetError(ValueError):
    """The rendered history leaves no usable generation budget."""


class RequestDeadlineExceeded(TimeoutError):
    """The remaining agent wall-clock budget cannot fund another API attempt."""


def request_timeout(deadline: float | None = None) -> float:
    timeout = settings()["timeout_seconds"]
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestDeadlineExceeded("Agent deadline exhausted before API request")
        timeout = min(timeout, remaining)
    return timeout


def is_local_qwen(model: str) -> bool:
    return model.lower() in LOCAL_MODELS and bool(os.getenv("RISE_LOCAL_VLLM_BASE_URL"))


def extract_reasoning(message: Any) -> str:
    """Support both legacy providers and recent vLLM without duplicating text."""
    for field in ("reasoning_content", "reasoning"):
        value = message.get(field) if isinstance(message, dict) else getattr(message, field, None)
        if isinstance(value, str) and value:
            return value
    return ""


def settings() -> dict[str, Any]:
    margin = int(os.getenv("RISE_LOCAL_VLLM_TOKEN_MARGIN", "256"))
    minimum = int(os.getenv("RISE_LOCAL_VLLM_MIN_OUTPUT_TOKENS", "1024"))
    expected = int(os.getenv("RISE_LOCAL_VLLM_CONTEXT_LENGTH", "0"))
    timeout = float(os.getenv("RISE_LOCAL_VLLM_TIMEOUT", "1200"))
    if margin < 0 or minimum < 1 or expected < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid local vLLM budget/timeout settings")
    return dict(adapter=ADAPTER_VERSION, profile=os.getenv("RISE_QWEN35_PROFILE", "legacy"),
                service_sha256=os.getenv("RISE_LOCAL_VLLM_SERVICE_SHA256", ""),
                token_margin=margin, min_output_tokens=minimum,
                expected_context_length=expected, timeout_seconds=timeout)


def apply_budget(client: Any, kwargs: dict[str, Any], *, deadline: float | None = None) -> dict[str, int]:
    """Count the full rendered chat (including tools) at the serving endpoint.

    Never truncate history or guess from character counts. Tokenization failure
    is a preflight failure; no unchecked completion is sent.
    """
    cfg = settings()
    extra = kwargs.get("extra_body") or {}
    body = {"model": kwargs["model"], "messages": kwargs["messages"],
            "add_generation_prompt": True}
    if kwargs.get("tools") and kwargs.get("tool_choice") != "none":
        body["tools"] = kwargs["tools"]
    for key in ("chat_template_kwargs", "chat_template", "add_generation_prompt",
                "continue_final_message", "add_special_tokens", "mm_processor_kwargs"):
        if key in extra:
            body[key] = extra[key]
    url = str(client.base_url).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    timeout = request_timeout(deadline)
    # The preflight is part of this turn's deadline too. Disable hidden SDK
    # retries so the explicit wrapper owns request/backoff admission.
    budget_client = client.with_options(timeout=timeout, max_retries=0) if hasattr(client, "with_options") else client
    try:
        data = budget_client.post(url + "/tokenize", body=body, cast_to=dict[str, Any])
    except Exception as exc:
        if deadline is not None and time.monotonic() >= deadline:
            raise RequestDeadlineExceeded("Agent deadline exhausted during token budget preflight") from exc
        raise
    if deadline is not None and time.monotonic() >= deadline:
        raise RequestDeadlineExceeded("Agent deadline exhausted during token budget preflight")
    count, limit = data.get("count"), data.get("max_model_len")
    if type(count) is not int or type(limit) is not int or count < 0 or limit < 1:
        raise ValueError("vLLM /tokenize must return integer count and max_model_len")
    if cfg["expected_context_length"] and limit != cfg["expected_context_length"]:
        raise ValueError(f"vLLM context configuration mismatch: expected {cfg['expected_context_length']}, server {limit}")
    key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    requested = int(kwargs.get(key, 32000))
    if requested < 1:
        raise ValueError("Output token budget must be positive")
    available = limit - count - cfg["token_margin"]
    if available < min(requested, cfg["min_output_tokens"]):
        raise ContextBudgetError(f"context budget exhausted: input={count}, server_limit={limit}, remaining={available}")
    kwargs[key] = min(requested, available)
    return dict(input_tokens=count, server_context_length=limit,
                requested_output_tokens=requested, effective_output_tokens=kwargs[key],
                token_margin=cfg["token_margin"])


def prepare_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply an explicit, versioned Qwen3.5 general-thinking profile.

    Opt-in keeps other experiments/providers unchanged. No soft /think flags,
    forced end tokens, history truncation, or reasoning-to-answer promotion.
    """
    kwargs = dict(kwargs)
    # vLLM 0.17 reads only the native `reasoning` input field, then supplies
    # both aliases to the model template. See upstream issue #38488.
    # Normalize at the wire boundary, preserving the stored Agent history.
    messages = []
    for original in kwargs["messages"]:
        message = dict(original)
        if message.get("role") == "assistant":
            reasoning = extract_reasoning(message)
            message.pop("reasoning_content", None)
            if reasoning:
                message["reasoning"] = reasoning
        messages.append(message)
    kwargs["messages"] = messages
    extra = dict(kwargs.get("extra_body") or {})
    template = dict(extra.get("chat_template_kwargs") or {})
    template.setdefault("enable_thinking", True)
    profile = settings()["profile"]
    if profile not in {"legacy", "official-thinking-v1"}:
        raise ValueError(f"Unknown Qwen3.5 profile: {profile}")
    if profile == "official-thinking-v1":
        if kwargs["model"].lower() != "qwen3.5-35b-a3b":
            raise ValueError("official-thinking-v1 requires Qwen3.5-35B-A3B")
        if template["enable_thinking"] is not True:
            raise ValueError("official-thinking-v1 requires enable_thinking=True")
        kwargs.update(temperature=1.0, top_p=0.95, presence_penalty=1.5)
        extra.update(top_k=20, min_p=0.0, repetition_penalty=1.0)
    extra["chat_template_kwargs"] = template
    kwargs["extra_body"] = extra
    return kwargs


def complete_response_category(response: Any) -> str:
    """Accept a complete final answer or fully assembled function calls."""
    metadata = getattr(response, "_rise_api_metadata", {}) or {}
    if metadata.get("deadline_exceeded"):
        return "wall_clock_timeout"
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "missing_choices"
    choice = choices[0]
    finish = str(getattr(choice, "finish_reason", None) or "").strip().lower()
    message = getattr(choice, "message", None)
    if message is None:
        return "invalid_response"
    if finish == "length":
        return "output_truncated"
    if finish == "content_filter":
        return "content_filter"
    if finish not in {"stop", "tool_calls"}:
        return "incomplete_response"
    calls = getattr(message, "tool_calls", None) or []
    if not isinstance(calls, list):
        return "invalid_tool_calls"
    if calls:
        identities = set()
        for call in calls:
            function = getattr(call, "function", None)
            identity = getattr(call, "id", None)
            if (not isinstance(identity, str) or not identity.strip() or identity in identities
                    or getattr(call, "type", None) != "function" or function is None
                    or not isinstance(getattr(function, "name", None), str) or not function.name.strip()):
                return "invalid_tool_calls"
            identities.add(identity)
            try:
                arguments = json.loads(function.arguments)
            except (ValueError, TypeError, AttributeError):
                return "invalid_tool_calls"
            if not isinstance(arguments, dict):
                return "invalid_tool_calls"
        return "tool_calls"
    if finish == "tool_calls":
        return "invalid_tool_calls"
    if (getattr(message, "content", None) or "").strip():
        return "content"
    return "reasoning_only" if extract_reasoning(message) else "empty_response"
