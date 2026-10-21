"""Small observational feedback for v13, without changing tool execution.

Fingerprints describe visible results, never model reasoning or an assessment
of whether a document contains new evidence. Logs retain hashes and counters,
not commands, queries, or document text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time

PROGRESS_FEEDBACK_VERSION = "v13-observational-feedback-v1"
_TRUNCATED = re.compile(r"\n\.\.\.\[truncated; \d+ chars cut from output\]$")
_EMPTY = {"", "(command succeeded, no output)", "(no matches found)"}
_VOLATILE_RERANK_FIELDS = {"feedback", "cache_hit", "preparation_seconds", "scoring_seconds"}
_SEARCH_RECORD = re.compile(r"^([ \t]*Search record: )S\d+\.q\d+([ \t]*)$", re.MULTILINE)


def v13_bash_tool_spec(baseline_spec: dict) -> dict:
    """Expose the timeout already accepted by bash; leave baseline untouched."""
    spec = copy.deepcopy(baseline_spec)
    spec["function"]["parameters"]["properties"]["timeout"] = {
        "type": "number", "exclusiveMinimum": 0, "maximum": 60,
        "description": "Optional timeout in seconds; the backend clamps it to 1–60 seconds (default 60).",
    }
    return spec


def _digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProgressFeedback:
    def __init__(self, *, pool_version_provider, log_path: Path) -> None:
        self.pool_version_provider = pool_version_provider
        self.log_path = log_path
        self._pool_version = None
        self._repeats: dict[str, int] = {}
        self._truncated_streak = 0
        self._empty_streak = 0
        self._usage = {"observed_calls": 0, "feedback_calls": 0,
                       "repeated_result_feedback": 0, "truncated_feedback": 0,
                       "empty_feedback": 0, "bash_truncated_calls": 0,
                       "bash_empty_calls": 0, "bash_error_calls": 0,
                       "logging_errors": 0}
        self._tool_calls: dict[str, int] = {}

    def wrap(self, name, tool):
        if name not in {"search", "bash", "rerank"}:
            raise ValueError("v13 feedback observes search/bash/rerank only")
        def observed(args):
            # Run every requested call normally. A raised exception remains the
            # Agent's ordinary tool error; no synthesized result replaces it.
            result = tool(args)
            return self.observe(name, args, result)
        return observed

    def observe(self, name: str, args: dict, result: str) -> str:
        pool_version = self.pool_version_provider()
        if pool_version != self._pool_version:
            self._pool_version = pool_version
            self._repeats = {}
            self._truncated_streak = self._empty_streak = 0
        parsed = None
        if name == "rerank":
            try:
                parsed = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                pass
        # Only the known provenance ID line changes on every otherwise
        # identical search. Keep document paths/text, match/new counts, and
        # pool version intact, and hash the exact raw result separately.
        evidence = (_SEARCH_RECORD.sub(r"\1<search-record-id>\2", result)
                    if name == "search" else result)
        if isinstance(parsed, dict):
            evidence = {key: value for key, value in parsed.items()
                        if key not in _VOLATILE_RERANK_FIELDS}
        try:
            arguments_hash = _digest(args)
        except (TypeError, ValueError):
            # Invalid JSON values remain for the underlying tool to diagnose.
            arguments_hash = _digest({"invalid_type": type(args).__name__,
                                      "representation_sha256": hashlib.sha256(repr(args).encode()).hexdigest()})
        visible_hash = _digest(result)
        result_hash = _digest(evidence)
        key = _digest([name, pool_version, arguments_hash, result_hash])
        count = self._repeats.get(key, 0) + 1
        self._repeats[key] = count
        self._usage["observed_calls"] += 1
        self._tool_calls[name] = self._tool_calls.get(name, 0) + 1
        flags, reasons = [], []
        if count >= 3 and count % 3 == 0:
            flags.append("repeated_result")
            self._usage["repeated_result_feedback"] += 1
            reasons.append(f"Same arguments and returned content {count} times in this pool; reconsider the missing fact and query/scope.")
        truncated = empty = error = False
        if name == "bash":
            stripped = result.strip()
            error = (stripped.startswith("Error:") or stripped.startswith("[exit=")
                     or stripped.startswith("(no output; exit="))
            truncated = bool(_TRUNCATED.search(result))
            empty = not error and stripped in _EMPTY
            self._usage["bash_error_calls"] += int(error)
            self._usage["bash_truncated_calls"] += int(truncated)
            self._usage["bash_empty_calls"] += int(empty)
            self._truncated_streak = self._truncated_streak + 1 if truncated else 0
            self._empty_streak = self._empty_streak + 1 if empty else 0
            if self._truncated_streak >= 2 and (self._truncated_streak - 2) % 3 == 0:
                flags.append("consecutive_truncation")
                self._usage["truncated_feedback"] += 1
                reasons.append("Repeated bash truncation: narrow output with rg -l/-o; head limits lines, not long-line characters.")
            if self._empty_streak >= 2 and (self._empty_streak - 2) % 3 == 0:
                flags.append("consecutive_empty")
                self._usage["empty_feedback"] += 1
                reasons.append("Repeated empty bash output: reconsider the query/scope; search can add sources outside the current pool.")
        feedback = " ".join(reasons)
        # At most two observations fire together. Keep the model-side notice
        # short; all observed counts stay in the independent diagnostic log.
        if feedback:
            feedback = (feedback + " These are output observations; judge evidence yourself.")[:300]
            self._usage["feedback_calls"] += 1
        self._append_log({
            "schema_version": 1, "feedback_version": PROGRESS_FEEDBACK_VERSION,
            "call_index": self._usage["observed_calls"], "recorded_unix": time.time(),
            "tool": name, "pool_version": pool_version, "arguments_sha256": arguments_hash,
            "visible_result_sha256": visible_hash, "result_fingerprint_sha256": result_hash,
            "result_chars": len(result), "same_result_count": count,
            "bash_truncated": truncated, "bash_empty": empty, "bash_error": error,
            "bash_truncated_streak": self._truncated_streak,
            "bash_empty_streak": self._empty_streak, "feedback_reasons": flags,
            "feedback": feedback or None,
        })
        if not feedback:
            return result
        if isinstance(parsed, dict):
            parsed["feedback"] = feedback
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return result + "\nTool feedback: " + feedback

    def _append_log(self, record):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            # Logging must not erase evidence after an otherwise successful
            # tool execution. The summary/checkpoint make failures inspectable.
            self._usage["logging_errors"] += 1

    @property
    def summary(self):
        return {"version": PROGRESS_FEEDBACK_VERSION, **copy.deepcopy(self._usage),
                "tool_calls": copy.deepcopy(self._tool_calls), "log_path": str(self.log_path)}

    def export_state(self):
        return copy.deepcopy({"version": PROGRESS_FEEDBACK_VERSION,
            "pool_version": self._pool_version, "repeats": self._repeats,
            "truncated_streak": self._truncated_streak, "empty_streak": self._empty_streak,
            "usage": self._usage, "tool_calls": self._tool_calls})

    def restore_state(self, state):
        if not isinstance(state, dict) or state.get("version") != PROGRESS_FEEDBACK_VERSION:
            raise ValueError("Incompatible v13 progress feedback state")
        pool_version = state.get("pool_version")
        if pool_version is not None and pool_version != self.pool_version_provider():
            raise ValueError("Feedback state differs from restored original pool")
        repeats, usage, tool_calls = state.get("repeats"), state.get("usage"), state.get("tool_calls")
        if (not isinstance(repeats, dict) or not isinstance(usage, dict)
                or set(usage) != set(self._usage) or not isinstance(tool_calls, dict)
                or not set(tool_calls).issubset({"search", "bash", "rerank"})):
            raise ValueError("Invalid v13 feedback counters")
        values = [*repeats.values(), *usage.values(), *tool_calls.values(),
                  state.get("truncated_streak"), state.get("empty_streak")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("Invalid v13 feedback counters")
        if any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) for key in repeats):
            raise ValueError("Invalid v13 feedback fingerprints")
        if sum(tool_calls.values()) != usage["observed_calls"]:
            raise ValueError("Inconsistent v13 feedback counters")
        self._pool_version = pool_version
        self._repeats, self._usage = copy.deepcopy(repeats), copy.deepcopy(usage)
        self._tool_calls = copy.deepcopy(tool_calls)
        self._truncated_streak, self._empty_streak = state["truncated_streak"], state["empty_streak"]
