"""v10: on-demand BM25 passage selection followed by neural reranking.

Each scope is ONE search subquery's frozen document top-1000, never the
preview top-10 or the accumulated union. Neither registering search results
nor constructing this tool loads a tokenizer or neural weights.
"""
from __future__ import annotations

import copy
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import time
from typing import Any

import bm25s

from .retrieval import _stemmer

PASSAGE_TOKENS = 128
PASSAGE_OVERLAP_TOKENS = 32
PASSAGE_STRIDE_TOKENS = PASSAGE_TOKENS - PASSAGE_OVERLAP_TOKENS
CHUNKING_VERSION = "token-overlap-v2"
BM25_CANDIDATES = 100
PREVIEW_PASSAGES = 10
BODY_BOUNDARY = re.compile(r"(?m)^=== DOCUMENT BODY ===[ \t]*\r?$")


def chunking_identity() -> dict:
    return {"version": CHUNKING_VERSION, "passage_tokens": PASSAGE_TOKENS,
            "overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "stride_tokens": PASSAGE_STRIDE_TOKENS}


class RerankerTool:
    def __init__(self, working_dir: Path, backend: Any, tokenizer: Any = None):
        if working_dir.is_symlink():
            raise ValueError("reranker workspace must not be a symlink")
        self.root = working_dir.resolve()
        self.backend = backend
        self.tokenizer = tokenizer
        self._counter = 0
        self._sets: dict[str, dict] = {}
        self._views: dict[str, dict] = {}
        self._usage = {"calls": 0, "scoring_calls": 0, "cache_hits": 0,
                       "scored_passages": 0, "forward_calls": 0,
                       "completed_forward_calls": 0, "input_tokens": 0,
                       "input_tokens_attempted": 0, "padded_input_tokens": 0,
                       "scoring_seconds": 0., "errors": 0}

    def _path(self, rel: str) -> Path:
        if not isinstance(rel, str):
            raise ValueError("file_path must be a relative path")
        parts = PurePosixPath(rel)
        if (not rel or parts.is_absolute() or ".." in parts.parts
                or parts.as_posix() != rel or parts.suffix != ".txt"):
            raise ValueError(f"unsafe document path: {rel!r}")
        path = self.root / rel
        for parent in (path, *path.parents):
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError(f"symlink document path: {rel!r}")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError(f"missing or unsafe source: {rel!r}")
        return resolved

    def _read(self, rel: str) -> tuple[str, str]:
        data = self._path(rel).read_bytes()
        # Preserve CRLF and exact Unicode character coordinates for readback.
        return data.decode("utf-8", errors="replace"), hashlib.sha256(data).hexdigest()

    def register_search(self, queries: list[str], per_query_results: list[list[tuple[str, float]]]) -> list[str]:
        if len(queries) != len(per_query_results):
            raise ValueError("query/result count mismatch")
        staged: dict[str, dict] = {}
        counter = self._counter + 1
        for i, (query, results) in enumerate(zip(queries, per_query_results), 1):
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be nonempty")
            if len(results) > 1000:
                raise ValueError("v10 requires search top-1000")
            docs, seen = [], set()
            for rank, (rel, score) in enumerate(results, 1):
                if rel in seen:
                    continue
                _, digest = self._read(rel)
                if not math.isfinite(float(score)):
                    raise ValueError("search score must be finite")
                docs.append({"file_path": rel, "search_rank": rank,
                             "search_score": float(score), "sha256": digest})
                seen.add(rel)
            staged[f"S{counter:03d}.q{i}"] = {"query": query, "docs": docs}
        self._sets.update(staged)
        self._counter = counter
        return list(staged)

    def _sources(self, scope: dict) -> list[tuple[dict, str]]:
        sources = []
        for doc in scope["docs"]:
            text, digest = self._read(doc["file_path"])
            if digest != doc["sha256"]:
                raise ValueError(f"source changed since search: {doc['file_path']}")
            sources.append((doc, text))
        return sources

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _chunks(self, sources: list[tuple[dict, str]]) -> list[dict]:
        if self.tokenizer is None:
            self.tokenizer = self.backend.get_tokenizer()
        passages = []
        for doc, source in sources:
            # Match baseline read's splitlines(), including standalone CR and
            # Unicode line separators, while retaining raw char coordinates.
            source_lines = source.splitlines()
            line_ends, line_end = [], 0
            for line in source.splitlines(keepends=True):
                line_end += len(line)
                line_ends.append(line_end)
            boundary = BODY_BOUNDARY.search(source)
            body_start = boundary.end() if boundary else 0
            body = source[body_start:]
            encoded = self.tokenizer(body, add_special_tokens=False,
                                     return_offsets_mapping=True, truncation=False)
            offsets = encoded["offset_mapping"]
            # Fast tokenizers can give several byte tokens the same character
            # range. Cut only at character boundaries, then re-count the exact
            # substring which is actually sent to the reranker.
            cursor, token_index, overlap_tokens = 0, 0, 0
            while token_index < len(offsets):
                while token_index < len(offsets) and offsets[token_index][1] <= cursor:
                    token_index += 1
                if token_index >= len(offsets):
                    break
                last = min(len(offsets), token_index + PASSAGE_TOKENS) - 1
                end = int(offsets[last][1])
                while end > cursor and self._tokens(body[cursor:end]) > PASSAGE_TOKENS:
                    last -= 1
                    if last < token_index:
                        raise ValueError("one Unicode character exceeds the passage token budget")
                    end = int(offsets[last][1])
                if end <= cursor or end > len(body):
                    raise ValueError("tokenizer returned invalid character offsets")
                text = body[cursor:end]
                count = self._tokens(text)
                if text.strip() and count:
                    start_abs, end_abs = body_start + cursor, body_start + end
                    start_line = bisect_right(line_ends, start_abs)
                    end_line = bisect_right(line_ends, max(start_abs, end_abs - 1))
                    read_may_truncate = any(len(line) > 2000
                                            for line in source_lines[start_line:end_line + 1])
                    passages.append({
                        "file_path": doc["file_path"], "text": text,
                        "token_count": count, "overlap_tokens": overlap_tokens,
                        "char_start": start_abs,
                        "char_end": end_abs, "search_rank": doc["search_rank"],
                        "search_score": doc["search_score"],
                        "source_sha256": doc["sha256"],
                        "read_may_truncate": read_may_truncate,
                        "read_args": {"file_path": doc["file_path"],
                                      "offset": start_line, "limit": end_line - start_line + 1},
                    })
                    if read_may_truncate:
                        # Keep baseline read's line truncation untouched. Its
                        # existing bash tool can retrieve a later character
                        # window safely, without rewriting source files.
                        command = (
                            "import pathlib; "
                            f"s=pathlib.Path({doc['file_path']!r}).read_bytes().decode('utf-8',errors='replace'); "
                            f"print(s[{max(0, start_abs - 128)}:{min(len(source), end_abs + 128)}])"
                        )
                        passages[-1]["bash_args"] = {"command": "python3 -c " + shlex.quote(command)}
                # Stop as soon as the final token is covered; do not emit a
                # redundant short window consisting only of the overlap.
                if end >= int(offsets[-1][1]):
                    break
                # Re-tokenize the actual substring, rather than assuming its
                # token IDs equal a slice of the whole-document encoding.
                # Byte tokens may share a Unicode character's offsets. Move
                # to the next safe character boundary if exactly 32 tokens
                # would split that character; the overlap then stays <=32.
                local_offsets = self.tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=False)["offset_mapping"]
                cursor_before = cursor
                overlap_tokens = 0
                cursor = end
                for first in range(max(0, len(local_offsets) - PASSAGE_OVERLAP_TOKENS),
                                   len(local_offsets)):
                    overlap_start = cursor_before + int(local_offsets[first][0])
                    if not cursor_before < overlap_start < end:
                        continue
                    suffix_tokens = self._tokens(body[overlap_start:end])
                    if 0 < suffix_tokens <= PASSAGE_OVERLAP_TOKENS:
                        cursor, overlap_tokens = overlap_start, suffix_tokens
                        break
        return passages

    def _rank(self, scope: dict, sources: list[tuple[dict, str]]) -> dict:
        passages = self._chunks(sources)
        if not passages:
            return {"total_passages": 0, "items": [], "backend_usage": {}}
        texts = [p["text"] for p in passages]
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=False)
        # Retrieve all local scores before a stable top-100 cut so BM25 ties
        # at the boundary cannot drop arbitrary documents. bm25s scores are
        # sparse; the ranked payload, not the full pool, is checkpointed.
        if any(tokens.ids):
            index = bm25s.BM25(k1=1.5, b=0.75)
            index.index(tokens, show_progress=False)
            query_tokens = bm25s.tokenize([scope["query"]], stopwords="en",
                                         stemmer=_stemmer(), show_progress=False)
            indices, values = index.retrieve(query_tokens, k=len(passages), show_progress=False)
            order = sorted(zip(indices[0], values[0]), key=lambda pair: (-float(pair[1]), int(pair[0])))
        else:
            # A stopword-only corpus has no lexical terms: every BM25 score
            # is zero. bm25s cannot build that empty vocabulary itself.
            order = [(idx, 0.) for idx in range(len(passages))]
        candidates = []
        for rank, (idx, score) in enumerate(order[:BM25_CANDIDATES], 1):
            if not math.isfinite(float(score)):
                raise ValueError("BM25 returned a nonfinite score")
            candidates.append({**passages[int(idx)], "bm25_rank": rank, "bm25_score": float(score)})
        started = time.monotonic()
        self._usage["scoring_calls"] += 1
        try:
            scores = [float(s) for s in self.backend.score(scope["query"], [p["text"] for p in candidates])]
        finally:
            self._usage["scoring_seconds"] += time.monotonic() - started
            usage = dict(getattr(self.backend, "last_usage", {}) or {})
            for key in ("forward_calls", "completed_forward_calls", "input_tokens",
                        "input_tokens_attempted", "padded_input_tokens"):
                self._usage[key] += int(usage.get(key, 0))
        if len(scores) != len(candidates) or any(not math.isfinite(s) for s in scores):
            raise ValueError("reranker must return one finite score per BM25 candidate")
        for passage, score in zip(candidates, scores):
            passage["reranker_score"] = score
        ranked = sorted(candidates, key=lambda p: (-p["reranker_score"], p["bm25_rank"]))
        for rank, passage in enumerate(ranked, 1):
            passage["reranker_rank"] = rank
        self._usage["scored_passages"] += len(candidates)
        return {"total_passages": len(passages), "items": ranked,
                "backend_usage": usage, "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}

    def __call__(self, args: dict) -> str:
        self._usage["calls"] += 1
        try:
            if not isinstance(args, dict) or set(args) - {"set_id", "offset", "limit"}:
                raise ValueError("use only set_id, offset and limit; the original search query is reused automatically")
            sid = args.get("set_id")
            if not isinstance(sid, str) or sid not in self._sets:
                raise ValueError("unknown set_id; use a reranker set ID returned by search")
            offset, limit = args.get("offset", 0), args.get("limit", PREVIEW_PASSAGES)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("offset must be a nonnegative integer")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= PREVIEW_PASSAGES:
                raise ValueError("limit must be between 1 and 10")
            scope = self._sets[sid]
            sources = self._sources(scope)
            cache_hit = sid in self._views
            if cache_hit:
                self._usage["cache_hits"] += 1
            else:
                self._views[sid] = self._rank(scope, sources) if sources else {
                    "total_passages": 0, "items": [], "backend_usage": {}}
            view = self._views[sid]
            if offset > len(view["items"]):
                raise ValueError("offset exceeds the ranked result count")
            page = view["items"][offset:offset + limit]
            next_offset = offset + len(page)
            result = {"status": "ok", "set_id": sid, "query": scope["query"],
                      "scope_documents": len(scope["docs"]), "passage_tokens": PASSAGE_TOKENS,
                      "passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
                      "passage_stride_tokens": PASSAGE_STRIDE_TOKENS,
                      "chunking_version": CHUNKING_VERSION,
                      "total_passages": view["total_passages"],
                      "bm25_candidates": len(view["items"]), "scored_passages": len(view["items"]),
                      "cache_hit": cache_hit, "items": page,
                      "next_offset": next_offset if next_offset < len(view["items"]) else None,
                      "model_identity": view.get("model_identity", {}),
                      "backend_usage": {} if cache_hit else view["backend_usage"]}
            return json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            self._usage["errors"] += 1
            return "Error: reranker: " + str(exc)

    @property
    def summary(self) -> dict:
        return {**self._usage, "search_sets": len(self._sets), "ranked_sets": len(self._views),
                "chunking_identity": chunking_identity(),
                "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}

    def export_state(self) -> dict:
        return copy.deepcopy({"version": 2, "counter": self._counter,
                              "chunking_identity": chunking_identity(),
                              "sets": self._sets, "views": self._views, "usage": self._usage,
                              "backend_identity": getattr(self.backend, "identity", {})})

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != 2:
            raise ValueError("invalid reranker checkpoint version")
        if state.get("chunking_identity") != chunking_identity():
            raise ValueError("reranker checkpoint chunking identity changed")
        counter, sets = state.get("counter"), state.get("sets")
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0 or not isinstance(sets, dict):
            raise ValueError("invalid reranker checkpoint state")
        for sid, scope in sets.items():
            match = re.fullmatch(r"S(\d+)\.q([1-9]\d*)", sid)
            if not match or int(match[1]) > counter or not isinstance(scope, dict):
                raise ValueError("invalid reranker checkpoint set")
            if not isinstance(scope.get("query"), str) or not scope["query"].strip() or not isinstance(scope.get("docs"), list):
                raise ValueError("invalid reranker checkpoint scope")
            if len(scope["docs"]) > 1000:
                raise ValueError("reranker checkpoint exceeds top-1000")
            self._sources(scope)
        # Ranked results can be rebuilt; retain validated successful cached
        # views to avoid another neural forward after normal resume.
        views = state.get("views", {})
        if not isinstance(views, dict) or set(views) - set(sets):
            raise ValueError("invalid reranker checkpoint views")
        for sid, view in views.items():
            if not isinstance(view, dict) or not isinstance(view.get("items"), list) or len(view["items"]) > BM25_CANDIDATES:
                raise ValueError("invalid reranker checkpoint ranking")
            if view.get("items"):
                saved_identity = state.get("backend_identity", {})
                view_identity = view.get("model_identity", {})
                keys = ("model", "requested_revision", "resolved_revision", "template", "score_kind")
                if any(saved_identity.get(key) != view_identity.get(key) for key in keys):
                    raise ValueError("reranker checkpoint view model identity changed")
            sources = {doc["file_path"]: text for doc, text in self._sources(sets[sid])}
            for item in view["items"]:
                text = sources.get(item.get("file_path"))
                if (text is None or text[item["char_start"]:item["char_end"]] != item["text"]
                        or not 0 < item["token_count"] <= PASSAGE_TOKENS
                        or not math.isfinite(item["reranker_score"])):
                    raise ValueError("invalid reranker checkpoint passage")
        usage = state.get("usage", {})
        if not isinstance(usage, dict) or set(usage) != set(self._usage):
            raise ValueError("invalid reranker checkpoint usage")
        restore_identity = getattr(self.backend, "restore_identity", None)
        if restore_identity is not None:
            restore_identity(state.get("backend_identity", {}))
        self._counter, self._sets = counter, copy.deepcopy(sets)
        self._views = copy.deepcopy(views)
        self._usage = copy.deepcopy(usage)


def reranker_tool_spec() -> dict:
    return {"type": "function", "function": {
        "name": "reranker",
        "description": (
            "Use when a search query's top-10 preview does not provide enough relevant information. "
            "Within that query's original top-1000 documents, split bodies into passages of at most "
            "128 tokens with 32-token overlap (96-token stride; adjusted at Unicode boundaries), "
            "reuse the SAME original search query to select the top-100 passages by BM25, then "
            "rerank those passages with a neural relevance model. Returns 10 passages with source "
            "paths and read_args. Choose the set_id supplied by search. offset paginates the "
            "same cached ranking without another neural inference. Scores indicate relevance; "
            "read surrounding source lines to verify facts."
        ),
        "parameters": {"type": "object", "properties": {
            "set_id": {"type": "string", "description": "Original search subquery's ID, e.g. S001.q1."},
            "offset": {"type": "integer", "minimum": 0, "description": "Use next_offset from the previous result; default 0."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Passages shown on this page; default 10."}},
            "required": ["set_id"], "additionalProperties": False}}}
