"""Search surfaces for the three independent, frozen-v14 ablations."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .tools import _read_snippet


ABLATION_VERSION = "v14-ablations-v1"
PAGE_SIZE = 10


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _queries(args):
    if not isinstance(args, dict) or set(args) - {"queries", "query"}:
        raise ValueError("new search accepts queries only")
    # Keep v14's recovery for legacy/stringified model arguments. The public
    # schema still advertises queries:string[]. Pass the original arguments
    # downstream as well so v14's explanatory normalization notices survive.
    queries = args.get("queries")
    if queries is None:
        queries = args.get("query")
    if isinstance(queries, str):
        value = queries.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(q).strip() for q in parsed if str(q).strip()]
            except ValueError:
                pass
        return [value] if value else []
    if isinstance(queries, (list, tuple)):
        return [str(q).strip() for q in queries if str(q).strip()]
    return []


class _SearchTool:
    variant = ""

    def __init__(self, underlying_search, rerank_service):
        self.underlying_search = underlying_search
        self.service = rerank_service
        self._usage = {"retrieval_calls": 0, "retrieval_queries": 0,
                       "search_page_calls": 0, "tool_errors": 0}

    @property
    def manager(self):
        # RerankService.restore_state replaces its manager after validation.
        return self.service.manager

    def _retrieve(self, args, queries):
        previous = set(self.manager._search_sets)
        output = self.underlying_search(args)
        records = [(sid, record) for sid, record in self.manager._search_sets.items()
                   if sid not in previous]
        if (not records and isinstance(output, str)
                and any(line.lstrip().startswith("Error:") for line in output.splitlines())):
            return output, []
        if len(records) != len(queries):
            # Preserve an ordinary underlying error; never reuse an older set.
            raise RuntimeError("search did not register one record per subquery")
        if [record["query"] for _, record in records] != queries:
            raise RuntimeError("registered search queries differ from request")
        self._usage["retrieval_calls"] += 1
        self._usage["retrieval_queries"] += len(queries)
        return output, records

    @property
    def summary(self):
        return {"version": ABLATION_VERSION, "variant": self.variant,
                **copy.deepcopy(self._usage)}

    def export_state(self):
        return self.summary

    def restore_state(self, state):
        if (not isinstance(state, dict) or state.get("version") != ABLATION_VERSION
                or state.get("variant") != self.variant
                or set(state) != {"version", "variant", *self._usage}):
            raise ValueError("incompatible ablation search checkpoint")
        usage = {key: state[key] for key in self._usage}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in usage.values()):
            raise ValueError("invalid ablation search counters")
        self._usage = copy.deepcopy(usage)

    def _error(self, exc):
        self._usage["tool_errors"] += 1
        return "Error: " + str(exc)


class SearchPagingTool(_SearchTool):
    """Page immutable document-search records; the first preview is unchanged."""
    variant = "search_paging"

    def __init__(self, underlying_search, rerank_service, *, working_dir: Path,
                 snippet_chars: int = 100):
        super().__init__(underlying_search, rerank_service)
        self.root = Path(working_dir).resolve()
        self.snippet_chars = snippet_chars

    @staticmethod
    def _page_metadata(search_id, count, offset):
        returned = max(0, min(PAGE_SIZE, count - offset))
        return {"search_id": search_id, "offset": offset,
                "returned_documents": returned, "total_documents": count,
                "next_offset": offset + returned if offset + returned < count else None}

    def __call__(self, args):
        if not isinstance(args, dict):
            return self._error(ValueError("search arguments must be an object"))
        if "search_id" not in args:
            try:
                queries = _queries(args)
            except ValueError as exc:
                return self._error(exc)
            output, records = self._retrieve(args, queries)
            if not records:
                return output
            pages = [self._page_metadata(sid, len(record["docs"]), 0)
                     for sid, record in records]
            return output + "\nSearch pages: " + _json(pages)
        try:
            if set(args) - {"search_id", "offset"}:
                raise ValueError("a search page accepts only search_id and offset")
            sid, offset = args["search_id"], args.get("offset", 0)
            if not isinstance(sid, str) or sid not in self.manager._search_sets:
                raise ValueError("unknown search_id; use an ID returned by search")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("offset must be a nonnegative integer")
            record = self.manager._search_sets[sid]
            page = record["docs"][offset:offset + PAGE_SIZE]
            # Validate against the original search's hash, not the current pool
            # manifest; this record remains valid across later searches.
            for doc in page:
                target = (self.root / doc["file_path"]).resolve()
                if not target.is_relative_to(self.root) or not target.is_file():
                    raise ValueError("search snapshot source is unavailable")
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest != doc["source_sha256"]:
                    raise ValueError(f"source changed since search: {doc['file_path']}")
            self._usage["search_page_calls"] += 1
            metadata = self._page_metadata(sid, len(record["docs"]), offset)
            lines = [f'Search page: {_json(metadata)}', f'Query: "{record["query"]}"']
            if not page:
                lines.append("  (no more results in this search record)")
            for doc in page:
                rel = doc["file_path"]
                snippet = _read_snippet(self.root / rel, self.snippet_chars)
                lines.append(f'  {doc["search_rank"]:>2}. [{doc["search_score"]:.2f}] {rel}')
                lines.append(f'       "{snippet}"')
            return "\n".join(lines)
        except ValueError as exc:
            return self._error(exc)


class AutoRerankSearchTool(_SearchTool):
    """One automatic fixed-parameter passage ranking per original subquery."""
    variant = "auto_rerank"

    def __init__(self, underlying_search, rerank_service):
        super().__init__(underlying_search, rerank_service)
        self._usage.update({"auto_rerank_calls": 0, "auto_rerank_errors": 0,
                            "empty_scopes": 0})

    def __call__(self, args):
        try:
            queries = _queries(args)
        except ValueError as exc:
            return self._error(exc)
        output, records = self._retrieve(args, queries)
        if not records:
            return output
        results = []
        for _, record in records:
            paths = [doc["file_path"] for doc in record["docs"]]
            result = {"query": record["query"], "matched_documents": len(paths),
                      "status": "ok", "items": []}
            if not paths:
                self._usage["empty_scopes"] += 1
                results.append(result)
                continue
            self._usage["auto_rerank_calls"] += 1
            # Deliberately bypass the model-facing rerank feedback wrapper.
            ranked = json.loads(self.service({"query": record["query"],
                "file_paths": paths, "bm25_candidates": 1000,
                "top_k": 100, "offset": 0, "limit": PAGE_SIZE}))
            if ranked.get("status") != "ok":
                self._usage["auto_rerank_errors"] += 1
                self._usage["tool_errors"] += 1
                result.update({"status": "error", "error": "automatic rerank failed: "
                               + str(ranked.get("error", "unknown scoring error"))})
            else:
                # Keep complete passages and source coordinates from v14, but
                # never expose a hidden ranking's paging handles or metadata.
                result["items"] = copy.deepcopy(ranked["items"][:PAGE_SIZE])
            results.append(result)
        failed = any(result["status"] == "error" for result in results)
        return _json({"status": "error" if failed else "ok", "queries": results})


def _without_bash(value):
    if isinstance(value, dict):
        return {key: _without_bash(child) for key, child in value.items()
                if key != "bash_args"}
    if isinstance(value, list):
        return [_without_bash(child) for child in value]
    return value


class WithoutBashResponse:
    """Remove unavailable command-recovery arguments from public responses."""
    def __init__(self, tool):
        self.tool = tool

    def __call__(self, args):
        output = self.tool(args)
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            return output
        return _json(_without_bash(payload))
