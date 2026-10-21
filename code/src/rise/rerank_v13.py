"""Independent v13 passage reranking; no shell/view or original-pool mutation.

The lexical index remains immutable. Each call selects its own document scope
and BM25 candidate count; all scored candidates are cached independently from
the requested retained count and output page. No neural model is loaded merely
by constructing this service, registering search results or reading a page.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import time

from .accumulated_space import _MODEL_KEYS, _digest, _positive
from .indexed_accumulated_space import IndexedAccumulatedSpaceManager
from .reranker_tool import chunking_identity

RERANK_TOOL_VERSION = "v13-independent-rerank-tool-v1"
RERANK_RESPONSE_VERSION = "v13-compact-response-v1"
_RESULT = re.compile(r"R-(\d+)-([0-9a-f]{32})")

_RESPONSE_FIELDS = (
    "status", "result_id", "query", "scope_kind", "scope_documents",
    "pool_documents", "pool_version", "requested_candidates", "bm25_candidates",
    "scored_passages", "total_passages", "top_k", "retained_passages", "offset",
    "limit", "returned_passages", "next_offset", "cache_hit",
    "preparation_seconds", "scoring_seconds",
)
_ITEM_FIELDS = (
    "text", "char_start", "char_end", "token_count", "overlap_tokens",
    "bm25_rank", "bm25_score", "reranker_rank", "reranker_score",
)


def compact_rerank_response(response: dict) -> dict:
    """Project a rich diagnostic response without altering passages or caches."""
    if response.get("status") != "ok":
        return copy.deepcopy(response)
    public = {key: copy.deepcopy(response[key]) for key in _RESPONSE_FIELDS}
    public["items"] = []
    for item in response["items"]:
        fragment = {"file_path": item["source_file_path"]
                    if "source_file_path" in item else item["file_path"]}
        fragment.update({key: copy.deepcopy(item[key]) for key in _ITEM_FIELDS})
        fragment["read_args"] = copy.deepcopy(
            item["read_args"] if "read_args" in item else item["source_read_args"])
        # Baseline read clips long lines. Keep exact recovery only for sources
        # where that clipping can affect this fragment's source context.
        if item.get("read_may_truncate", False):
            fragment["read_may_truncate"] = True
            for key in ("bash_args", "context_char_start", "context_char_end"):
                fragment[key] = copy.deepcopy(item[key])
        public["items"].append(fragment)
    return public


def rerank_tool_spec() -> dict:
    return {"type": "function", "function": {
        "name": "rerank",
        "description": (
            "Find relevant original-body passages for your explicit evidence goal. "
            "Choose query, imported file subset, BM25 candidate count, retained "
            "count and output size yourself. Omitted files means the whole "
            "accumulated pool; [] means an empty scope. Returns source paths and "
            "read/bash coordinates, without restricting bash/read. For paging, "
            "supply result_id instead of query/file_paths/bm25_candidates; "
            "top_k, offset and limit can change without rescoring. Scores are "
            "relevance rankings, not verification of facts."),
        "parameters": {"type": "object", "additionalProperties": False,
            "properties": {
                "query": {"type": "string", "description": "Explicit fact or relationship to verify, including known entity names; required for a new ranking."},
                "file_paths": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Optional list of already imported relative source paths, such as files selected with bash; omitted or null means the accumulated pool."},
                "bm25_candidates": {"type": "integer", "minimum": 1, "description": "M: number of scope-local BM25 passages sent to the neural reranker; default 1000, adjustable per call."},
                "top_k": {"type": "integer", "minimum": 1, "description": "K: retain up to this many ranked passages; K <= M. Default min(100,M), or the first request's K on a result page."},
                "offset": {"type": "integer", "minimum": 0, "description": "Zero-based offset into the retained ranking; default 0."},
                "limit": {"type": "integer", "minimum": 1, "description": "Number of complete passages to return on this page; default 10, adjustable per call."},
                "result_id": {"type": "string", "description": "Tool-returned current ranking ID for pagination; mutually exclusive with query/file_paths/bm25_candidates."},
            }}}}


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class RerankService:
    def __init__(self, manager: IndexedAccumulatedSpaceManager):
        if not isinstance(manager, IndexedAccumulatedSpaceManager):
            raise ValueError("v13 requires an indexed accumulated-source manager")
        self.manager = manager
        self._rankings = {}
        self._results = {}
        self._cache_version = manager.space_version
        self._usage = {"rerank_calls": 0, "ranking_cache_hits": 0,
                       "page_calls": 0, "tool_errors": 0}
        self._history = []

    @property
    def diagnostics_path(self):
        """Outside the original pool; never exposed as model evidence."""
        return self.manager.views_root / "rerank_diagnostics.jsonl"

    def _append_diagnostic(self, record):
        with self.diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _sync_pool(self):
        if self._cache_version != self.manager.space_version:
            self._rankings.clear()
            self._results.clear()
            self._cache_version = self.manager.space_version

    def register_search(self, queries, per_query_results):
        result = self.manager.register_search(queries, per_query_results)
        self._sync_pool()
        return result

    def _paths(self, file_paths):
        if file_paths is None:
            return sorted(self.manager._sources)
        if not isinstance(file_paths, list) or any(not isinstance(p, str) for p in file_paths):
            raise ValueError("file_paths must be an array of imported relative file paths")
        normalized = []
        for rel in file_paths:
            # rg/grep invoked on '.' commonly return './docs/file.txt'. Keep
            # normalization deliberately narrow: never collapse '..', interpret
            # glob patterns or widen the imported-source boundary.
            while rel.startswith("./"):
                rel = rel[2:]
            normalized.append(rel)
        paths = sorted(set(normalized))
        for rel in paths:
            if rel not in self.manager._sources:
                raise ValueError(f"unknown imported source: {rel}")
            self.manager._chunker._path(rel)
        return paths

    def _scope_identity(self, paths, kind):
        return _digest({"kind": kind, "sources": [
            [p, self.manager._sources[p]["source_sha256"]] for p in paths]})

    def _key(self, query, scope_identity, candidates):
        return _digest({"version": RERANK_TOOL_VERSION,
                        "pool_version": self.manager.space_version,
                        "query": query, "scope_identity": scope_identity,
                        "bm25_candidates": candidates,
                        "index_identity": self.manager._passage_index_identity,
                        "chunking": chunking_identity(),
                        "model_identity": self.manager._identity()})

    def _lexical_candidates(self, query, paths, candidates):
        """Reuse v12 numeric lookup, with scope-local DF/N/avgdl and dynamic M."""
        started = time.monotonic()
        scope, doc_order, by_doc = self.manager._indexed_scope(paths)
        self.manager._usage["index_scope_seconds"] += time.monotonic() - started
        started = time.monotonic()
        scores = scope.scores(query)
        positions = self.manager._canonical_top(scope, scores, doc_order, candidates=candidates)
        self.manager._usage["index_lookup_seconds"] += time.monotonic() - started
        started = time.monotonic()
        items = self.manager._candidate_records(scope, positions, by_doc, scores)
        self.manager._usage["candidate_materialization_seconds"] += time.monotonic() - started
        return items, scope.N

    def _new_ranking(self, args):
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be an explicit nonempty evidence goal")
        query = query.strip()
        # Absence/null and [] deliberately have different semantics.
        paths = self._paths(args.get("file_paths"))
        kind = "files" if args.get("file_paths") is not None else "accumulated"
        m = _positive(args.get("bm25_candidates", 1000), "bm25_candidates")
        k = _positive(args.get("top_k", min(100, m)), "top_k")
        if k > m:
            raise ValueError("top_k must not exceed bm25_candidates")
        sid = self._scope_identity(paths, kind)
        validation_started = time.monotonic()
        self.manager._check_source_snapshots(paths)
        self.manager._usage["source_validation_seconds"] += time.monotonic() - validation_started
        key = self._key(query, sid, m)
        hit = key in self._rankings
        if not hit:
            items, total = self._lexical_candidates(query, paths, m)
            ranked = self.manager._score_candidates(query, items, total)
            # The first neural call may resolve the backend commit. Cache its
            # concrete identity rather than an earlier moving-ref identity.
            key = self._key(query, sid, m)
            ranked.update({"query": query, "file_paths": paths, "scope_kind": kind,
                           "scope_identity": sid, "requested_candidates": m,
                           "pool_version": self.manager.space_version,
                           "ranking_identity": self.manager._identity()})
            ranked["payload_sha256"] = _digest(ranked)
            self._rankings[key] = ranked
        else:
            self._usage["ranking_cache_hits"] += 1
        rid = f"R-{self.manager.space_version}-" + _digest([key, k])[:32]
        self._results[rid] = {"ranking_key": key, "top_k": k}
        return rid, self._rankings[key], k, hit

    def _existing_ranking(self, args):
        if any(name in args for name in ("query", "file_paths", "bm25_candidates")):
            raise ValueError("result_id cannot be combined with query/file_paths/bm25_candidates")
        rid = args["result_id"]
        match = _RESULT.fullmatch(rid) if isinstance(rid, str) else None
        if match and int(match[1]) != self.manager.space_version:
            raise ValueError("stale result: the accumulated pool changed; rerank with query again")
        if rid not in self._results:
            raise ValueError("unknown result_id; use an ID returned by rerank")
        result = self._results[rid]
        ranking = self._rankings[result["ranking_key"]]
        validation_started = time.monotonic()
        self.manager._check_source_snapshots(ranking["file_paths"])
        self.manager._usage["source_validation_seconds"] += time.monotonic() - validation_started
        if result["ranking_key"] != self._key(ranking["query"], ranking["scope_identity"], ranking["requested_candidates"]):
            raise ValueError("stale result: ranking configuration changed; rerank with query again")
        k = _positive(args.get("top_k", result["top_k"]), "top_k")
        if k > ranking["requested_candidates"]:
            raise ValueError("top_k must not exceed bm25_candidates")
        self._usage["page_calls"] += 1
        self._usage["ranking_cache_hits"] += 1
        return rid, ranking, k, True

    def _diagnostic_item(self, item):
        public = copy.deepcopy(item)
        rel = item["source_file_path"]
        public["file_path"] = rel
        public.pop("snippet_file", None)
        public["read_args"] = copy.deepcopy(item["source_read_args"])
        # read's existing line truncation is preserved. A byte-decoded source
        # window provides exact Unicode coordinates even for CRLF/long lines.
        _, doc = self.manager._indexed_doc(self.manager._sources[rel])
        start, end = max(0, item["char_start"] - 128), min(doc["chars"], item["char_end"] + 128)
        command = ("import pathlib; "
                   f"s=pathlib.Path({rel!r}).read_bytes().decode('utf-8',errors='replace'); "
                   f"print(s[{start}:{end}])")
        public["bash_args"] = {"command": "python3 -c " + shlex.quote(command)}
        public["context_char_start"] = start
        public["context_char_end"] = end
        # Membership provenance can grow when search reimports existing files.
        public["search_memberships"] = copy.deepcopy(self.manager._sources[rel]["search_memberships"])
        return public

    def __call__(self, args):
        started = time.monotonic()
        self._usage["rerank_calls"] += 1
        before = dict(self.manager._usage)
        response = None
        try:
            self._sync_pool()
            allowed = {"query", "file_paths", "bm25_candidates", "top_k", "offset", "limit", "result_id"}
            if not isinstance(args, dict) or set(args) - allowed:
                raise ValueError("use only query, file_paths, bm25_candidates, top_k, offset, limit or result_id")
            offset = _nonnegative(args.get("offset", 0), "offset")
            limit = _positive(args.get("limit", 10), "limit")
            rid, ranking, k, hit = (self._existing_ranking(args) if "result_id" in args
                                     else self._new_ranking(args))
            retained = min(k, len(ranking["items"]))
            page = ranking["items"][offset:min(retained, offset + limit)]
            next_offset = offset + len(page) if offset + len(page) < retained else None
            response = {"status": "ok", "result_id": rid,
                "query": ranking["query"], "scope_kind": ranking["scope_kind"],
                "scope_identity": ranking["scope_identity"],
                "scope_documents": len(ranking["file_paths"]),
                "pool_documents": len(self.manager._sources),
                "pool_version": self.manager.space_version,
                "requested_candidates": ranking["requested_candidates"],
                "bm25_candidates": len(ranking["items"]),
                "scored_passages": len(ranking["items"]),
                "total_passages": ranking["total_passages"], "top_k": k,
                "retained_passages": retained, "offset": offset, "limit": limit,
                "returned_passages": len(page), "next_offset": next_offset,
                "cache_hit": hit, "items": [self._diagnostic_item(p) for p in page],
                "chunking_identity": chunking_identity(),
                "model_identity": copy.deepcopy(ranking["model_identity"]),
                "backend_usage": {} if hit else copy.deepcopy(ranking["backend_usage"]),
                "preparation_seconds": time.monotonic() - started,
                "scoring_seconds": self.manager._usage["scoring_seconds"] - before["scoring_seconds"]}
        except Exception as exc:
            self._usage["tool_errors"] += 1
            message = str(exc)
            response = {"status": "error", "tool": "rerank", "error": message,
                        "code": "stale_result" if "stale result" in message else "rerank_error",
                        "pool_version": self.manager.space_version,
                        "preparation_seconds": time.monotonic() - started}
        finally:
            self.manager._usage["preparation_seconds"] += time.monotonic() - started
            # Keep requested knobs and actual costs for each attempt, including
            # failures, separately from original-source and shell statistics.
            try:
                json.dumps(args, allow_nan=False)
                history_args = copy.deepcopy(args)
            except (TypeError, ValueError):
                history_args = {"invalid_arguments": repr(args)}
        public = compact_rerank_response(response)
        encoded = json.dumps(public, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        delta = {key: value - before[key] for key, value in self.manager._usage.items()}
        record = {
            "schema_version": 1, "response_version": RERANK_RESPONSE_VERSION,
            "call_index": self._usage["rerank_calls"], "recorded_unix": time.time(),
            "args": history_args, "response": response, "usage_delta": delta,
            # A failed scoring call can have partial forward/preparation usage.
            # Parameter/scope errors and cache pages must not borrow the last
            # successful call's backend statistics.
            "backend_attempt_usage": copy.deepcopy(getattr(self.manager.backend, "last_usage", {}))
                if delta["scoring_calls"] else {},
            "public_response_chars": len(encoded),
        }
        try:
            self._append_diagnostic(record)
        except (OSError, TypeError, ValueError):
            self._usage["tool_errors"] += 1
            public = {"status": "error", "tool": "rerank", "code": "diagnostic_log_error",
                      "error": "Cannot persist rerank diagnostics; completed scoring remains cached.",
                      "pool_version": self.manager.space_version,
                      "preparation_seconds": time.monotonic() - started}
            encoded = json.dumps(public, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._history.append({"args": history_args,
            **{name: public[name] for name in (
                "status", "result_id", "requested_candidates", "bm25_candidates", "top_k",
                "offset", "limit", "returned_passages", "cache_hit", "preparation_seconds",
                "scoring_seconds", "error") if name in public}})
        return encoded

    @property
    def summary(self):
        return {**self.manager.summary, **copy.deepcopy(self._usage),
                "calls": self._usage["rerank_calls"],
                "pool_version": self.manager.space_version,
                "rerank_tool_version": RERANK_TOOL_VERSION,
                "cached_rankings": len(self._rankings),
                "results": len(self._results), "call_history": copy.deepcopy(self._history)}

    def export_state(self):
        self._sync_pool()
        return copy.deepcopy({"version": RERANK_TOOL_VERSION,
            "manager": self.manager.export_state(), "cache_version": self._cache_version,
            "rankings": self._rankings, "results": self._results,
            "usage": self._usage, "history": self._history})

    def restore_state(self, state):
        """Verify lexical candidates and source coordinates, never rescore."""
        if not isinstance(state, dict) or state.get("version") != RERANK_TOOL_VERSION:
            raise ValueError("invalid v13 checkpoint version")
        if not isinstance(state.get("rankings"), dict) or not isinstance(state.get("results"), dict):
            raise ValueError("invalid v13 cached rankings/results")
        # Validate using a private manager so a failed restore leaves this
        # service and its existing cumulative pool registrations untouched.
        original = self.manager
        validation = IndexedAccumulatedSpaceManager(
            working_dir=original.root, views_dir=original.views_root,
            backend=original.backend, candidates=original.candidates, top_k=original.top_k,
            passage_index=original.passage_index, runtime_to_corpus=original.runtime_to_corpus)
        validation.restore_state(state.get("manager"))
        if state.get("cache_version") != validation.space_version:
            raise ValueError("invalid v13 checkpoint pool version")
        self.manager = validation
        try:
            for key, ranking in state["rankings"].items():
                if not isinstance(ranking, dict):
                    raise ValueError("invalid v13 ranking")
                payload = {k: v for k, v in ranking.items() if k != "payload_sha256"}
                if ranking.get("payload_sha256") != _digest(payload):
                    raise ValueError("v13 ranking checksum mismatch")
                paths = self._paths(ranking.get("file_paths"))
                kind, query = ranking.get("scope_kind"), ranking.get("query")
                m = _positive(ranking.get("requested_candidates"), "checkpoint candidates")
                if (kind not in {"files", "accumulated"} or paths != ranking.get("file_paths")
                        or kind == "accumulated" and paths != sorted(validation._sources)
                        or not isinstance(query, str) or not query or query != query.strip()
                        or ranking.get("pool_version") != validation.space_version
                        or ranking.get("scope_identity") != self._scope_identity(paths, kind)
                        or ranking.get("ranking_identity") != validation._identity()
                        or not isinstance(ranking.get("model_identity"), dict)
                        or {name: ranking["model_identity"][name] for name in _MODEL_KEYS
                            if name in ranking["model_identity"]} != validation._identity()
                        or key != self._key(query, ranking["scope_identity"], m)):
                    raise ValueError("invalid v13 ranking identity/scope")
                candidates, total = self._lexical_candidates(query, paths, m)
                items = ranking.get("items")
                if (not isinstance(items, list) or len(items) != len(candidates)
                        or ranking.get("total_passages") != total):
                    raise ValueError("invalid v13 candidate count")
                expected = {p["passage_id"]: p for p in candidates}
                seen = set()
                for rank, item in enumerate(items, 1):
                    if not isinstance(item, dict) or item.get("passage_id") not in expected:
                        raise ValueError("invalid v13 checkpoint passage")
                    pid = item["passage_id"]
                    lexical = expected[pid]
                    # Search memberships may have grown after scoring. They
                    # are refreshed from the source record on every response.
                    fields = set(lexical) - {"search_memberships"}
                    if (pid in seen or item.get("reranker_rank") != rank
                            or any(item.get(f) != lexical[f] for f in fields)):
                        raise ValueError("invalid v13 source coordinates/BM25 passage")
                    score = item.get("reranker_score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("invalid v13 neural score")
                    seen.add(pid)
                if items != sorted(items, key=lambda p: (-p["reranker_score"], p["bm25_rank"])):
                    raise ValueError("invalid v13 rank order")
            for rid, result in state["results"].items():
                if not isinstance(result, dict) or result.get("ranking_key") not in state["rankings"]:
                    raise ValueError("invalid v13 result reference")
                key = result["ranking_key"]
                k = _positive(result.get("top_k"), "checkpoint top_k")
                expected_id = f"R-{validation.space_version}-" + _digest([key, k])[:32]
                if k > state["rankings"][key]["requested_candidates"] or rid != expected_id:
                    raise ValueError("invalid v13 result ID/count")
            usage = state.get("usage")
            if not isinstance(usage, dict) or set(usage) != set(self._usage):
                raise ValueError("invalid v13 checkpoint usage")
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in usage.values()):
                raise ValueError("invalid v13 usage count")
            history = state.get("history")
            if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
                raise ValueError("invalid v13 call history")
            # Restore counters from the saved state after the lexical validation
            # above, rather than charging replay as new preparation work.
            validation._usage = copy.deepcopy(state["manager"]["usage"])
            self._cache_version = validation.space_version
            self._rankings = copy.deepcopy(state["rankings"])
            self._results = copy.deepcopy(state["results"])
            self._usage = copy.deepcopy(usage)
            self._history = copy.deepcopy(history)
        except Exception:
            self.manager = original
            raise
