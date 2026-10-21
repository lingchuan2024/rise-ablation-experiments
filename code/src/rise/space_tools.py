"""v11 tools: versioned DCI views, bounded passage reads, explicit source expansion.

Only the materialized top-K directory is ever supplied to the shell sandbox.
The accumulated full-text pool remains a separate trusted-tool resource.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .tools import make_bash_tool

SPACE_TOOL_VERSION = "v11-space-tools-v1"
SOURCE_DEFAULT_CHARS = 8000
SOURCE_MAX_CHARS = 16000


def _spec(name: str, description: str, properties: dict,
          required: tuple[str, ...] = ()) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(required), "additionalProperties": False}}}


def space_tool_specs(*, no_bash: bool = False) -> list[dict]:
    properties = {
        "goal_query": {"type": "string", "description": "Current missing fact plus established entity names. Required for first exploration; never inherited from the last search."},
        "top_k": {"type": "integer", "minimum": 1, "description": "Number of passages retained, up to the BM25 candidate cap. Changing K reuses cached scores."},
        "view_id": {"type": "string", "description": "Previously returned view ID. Required to execute a command; stale views are refreshed first."}}
    if not no_bash:
        properties["command"] = {"type": "string", "description": "Shell command on ONLY this view's passage files. First/refresh call never executes an accompanying command."}
    return [
        _spec("dci", "Prepare or explore a goal-ranked view of the union of all imported search documents. Cached ranking is reused while pool, goal, and ranking configuration are unchanged.", properties),
        _spec("read", "Read one selected passage, identified by a stable passage ID within a view.", {
            "view_id": {"type": "string"}, "passage_id": {"type": "string"}},
            ("view_id", "passage_id")),
        _spec("expand_source", "Read a bounded original-source character range explicitly, outside the shell. Supply a source_id or an imported file_path; offsets are zero-based Unicode characters and the end is exclusive.", {
            "source_id": {"type": "string"}, "file_path": {"type": "string"},
            "char_start": {"type": "integer", "minimum": 0},
            "char_end": {"type": "integer", "minimum": 0}})]


class SpaceTools:
    def __init__(self, manager: Any, *, no_bash: bool = False,
                 truncate_chars: int = 4000):
        self.manager = manager
        self.no_bash = no_bash
        self.truncate_chars = truncate_chars
        self.active_goal: str | None = None
        self.active_view_id: str | None = None
        self.active_top_k: int | None = None
        self.active_space_version: int | None = None
        self._usage = {"dci_calls": 0, "dci_command_calls": 0,
                       "dci_refreshes": 0, "dci_view_reuses": 0,
                       "dci_command_errors": 0,
                       "read_calls": 0, "expand_source_calls": 0,
                       "tool_errors": 0}

    @staticmethod
    def _json(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def _error(self, tool: str, exc: Exception) -> str:
        self._usage["tool_errors"] += 1
        return self._json({"status": "error", "tool": tool, "error": str(exc)})

    def dci(self, args: dict) -> str:
        self._usage["dci_calls"] += 1
        try:
            if not isinstance(args, dict) or set(args) - {"goal_query", "command", "top_k", "view_id"}:
                raise ValueError("use only goal_query, command, top_k and view_id")
            goal = args.get("goal_query", self.active_goal)
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("first DCI exploration requires an explicit nonempty goal_query")
            command = args.get("command")
            if command is not None and (not isinstance(command, str) or not command.strip()):
                raise ValueError("command must be a nonempty string")
            if command is not None and self.no_bash:
                raise ValueError("DCI commands are disabled; prepare a view and use read")
            top_k = args.get("top_k", self.active_top_k)
            result = self.manager.prepare(goal, top_k=top_k)
            old_view = self.active_view_id
            old_goal, old_k, old_version = self.active_goal, self.active_top_k, self.active_space_version
            view_id = result["view_id"]
            refreshed = old_view != view_id
            self.active_goal = result["goal_query"]
            self.active_view_id = view_id
            self.active_top_k = result["top_k"]
            self.active_space_version = result["space_version"]
            if refreshed:
                self._usage["dci_refreshes"] += 1
            else:
                self._usage["dci_view_reuses"] += 1
            result.update({"command_executed": False,
                           "trigger_reason": "first_view" if old_view is None else
                               "pool_changed" if old_version != result["space_version"] else
                               "goal_changed" if old_goal != result["goal_query"] else
                               "top_k_changed" if old_k != result["top_k"] else
                               "view_changed" if refreshed else "cached_view"})
            if refreshed:
                result["notice"] = "View prepared/refreshed. Review its ID and preview, then issue a separate command with this view_id."
                return self._json(result)
            if command is None:
                return self._json(result)
            if args.get("view_id") != view_id:
                raise ValueError(f"command requires current view_id={view_id}; it was not executed")
            # Fail closed: v11 never offers an unsandboxed shell fallback.
            fn = make_bash_tool(self.manager.view_root(view_id),
                                truncate_chars=self.truncate_chars,
                                enable_sandbox=True)
            output = fn({"command": command})
            self._usage["dci_command_calls"] += 1
            surfaced = []
            for item in self.manager.get_view(view_id)["items"]:
                # Return source mapping for selected passage paths/IDs present
                # in shell results without exposing unrelated passages.
                path = item.get("snippet_file") or item.get("file_path", "")
                if (item["passage_id"] in output or path and (path in output or path in command)):
                    surfaced.append({key: item[key] for key in (
                        "passage_id", "source_id", "source_file_path",
                        "source_sha256", "char_start", "char_end")})
            command_error = (output.startswith("Error:") or
                             bool(re.match(r"\[exit=\d+\]", output)) or
                             output.startswith("(no output; exit="))
            if command_error:
                self._usage["dci_command_errors"] += 1
            result.update({"command_executed": True, "command_output": output,
                           "command_status": "error" if command_error else "ok",
                           "items": [], "surfaced_passages": surfaced})
            self.manager.record_receipt({"kind": "dci_command", "view_id": view_id,
                "goal_query": self.active_goal, "command": command,
                "result_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()})
            return self._json(result)
        except Exception as exc:
            return self._error("dci", exc)

    def read(self, args: dict) -> str:
        self._usage["read_calls"] += 1
        try:
            if not isinstance(args, dict) or set(args) != {"view_id", "passage_id"}:
                raise ValueError("read requires only view_id and passage_id")
            item = self.manager.get_passage(args["view_id"], args["passage_id"])
            receipt = {"kind": "read_passage", "view_id": args["view_id"],
                       "passage_id": args["passage_id"],
                       "source_id": item["source_id"],
                       "source_file_path": item["source_file_path"],
                       "source_sha256": item["source_sha256"],
                       "char_start": item["char_start"], "char_end": item["char_end"]}
            self.manager.record_receipt(receipt)
            return self._json({"status": "ok", "view_id": args["view_id"],
                               "passage": item, "receipt": receipt})
        except Exception as exc:
            return self._error("read", exc)

    def expand_source(self, args: dict) -> str:
        self._usage["expand_source_calls"] += 1
        try:
            if not isinstance(args, dict) or set(args) - {"source_id", "file_path", "char_start", "char_end"}:
                raise ValueError("use source_id or file_path, and optional char_start/char_end")
            sid, path = args.get("source_id"), args.get("file_path")
            if not sid and not path:
                raise ValueError("provide source_id or an imported file_path")
            if sid is not None and not isinstance(sid, str) or path is not None and not isinstance(path, str):
                raise ValueError("source_id and file_path must be strings")
            text, source = self.manager.source_text(sid or path)
            if sid and path and self.manager.get_source(path)["source_id"] != source["source_id"]:
                raise ValueError("source_id and file_path refer to different sources")
            start = args.get("char_start", 0)
            end = args.get("char_end", min(len(text), start + SOURCE_DEFAULT_CHARS)
                           if isinstance(start, int) else None)
            if (isinstance(start, bool) or not isinstance(start, int) or start < 0
                    or isinstance(end, bool) or not isinstance(end, int) or end < start
                    or start > len(text) or end > len(text)):
                raise ValueError("invalid source character range")
            if end - start > SOURCE_MAX_CHARS:
                raise ValueError(f"source range cannot exceed {SOURCE_MAX_CHARS} characters; read consecutive ranges")
            receipt = {"kind": "expand_source", "source_id": source["source_id"],
                       "source_file_path": source.get("source_file_path", source.get("file_path")),
                       "source_sha256": source.get("source_sha256", source.get("sha256")),
                       "char_start": start, "char_end": end}
            self.manager.record_receipt(receipt)
            return self._json({"status": "ok", "source": source,
                "char_start": start, "char_end": end, "total_chars": len(text),
                "text": text[start:end], "receipt": receipt,
                "next_char_start": end if end < len(text) else None})
        except Exception as exc:
            return self._error("expand_source", exc)

    @property
    def summary(self) -> dict:
        return {**self.manager.summary, **copy.deepcopy(self._usage),
                "active_view_id": self.active_view_id,
                "calls": self._usage["dci_calls"],
                "cache_hits": self.manager.summary.get("ranking_cache_hits", 0),
                "space_tool_version": SPACE_TOOL_VERSION}

    def export_state(self) -> dict:
        return copy.deepcopy({"version": SPACE_TOOL_VERSION,
            "manager": self.manager.export_state(), "active_goal": self.active_goal,
            "active_view_id": self.active_view_id, "active_top_k": self.active_top_k,
            "active_space_version": self.active_space_version,
            "usage": self._usage})

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != SPACE_TOOL_VERSION:
            raise ValueError("invalid v11 tool checkpoint version")
        self.manager.restore_state(state["manager"])
        self.active_goal = state.get("active_goal")
        self.active_view_id = state.get("active_view_id")
        self.active_top_k = state.get("active_top_k")
        self.active_space_version = state.get("active_space_version")
        self._usage = copy.deepcopy(state["usage"])
        if self.active_view_id is not None:
            try:
                view = self.manager.get_view(self.active_view_id)
            except ValueError:
                # A search can invalidate the preceding view immediately before
                # a checkpoint. Preserve the goal, and prepare a fresh view next.
                if self.active_space_version != self.manager.space_version:
                    self.active_view_id = None
                    return
                raise
            if view["goal_query"] != self.active_goal or view["top_k"] != self.active_top_k:
                raise ValueError("checkpoint active view does not match goal or K")
