"""v11 accumulated retrieval pool and goal-specific, materialized DCI views.

Search registration is lazy.  A view is ranked only when requested, using the
explicit evidence goal over the union of imported source documents.  The
complete neural ranking is cached independently of K; shell access belongs to
the separate v11 wrapper and is restricted to a materialized snippet directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import bm25s

from .retrieval import _stemmer
from .reranker_tool import RerankerTool, chunking_identity

STATE_VERSION = 1
PREVIEW_PASSAGES = 10
_SHA = re.compile(r"[0-9a-f]{64}")
_MODEL_KEYS = ("kind", "model", "revision", "requested_revision", "resolved_revision",
               "instruction", "template", "score_kind", "token_count_kind", "max_length",
               "device", "dtype", "batch_size", "local_files_only", "loading_policy")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class AccumulatedSpaceManager:
    """Manage immutable goal rankings and actual K-passage working spaces."""

    def __init__(self, working_dir: Path, views_dir: Path, backend: Any,
                 candidates: int = 1000, top_k: int = 100, tokenizer: Any = None):
        working_dir, views_dir = Path(working_dir), Path(views_dir)
        if working_dir.is_symlink() or views_dir.is_symlink():
            raise ValueError("source and view roots must not be symlinks")
        self.root, self.views_root = working_dir.resolve(), views_dir.resolve()
        if (self.views_root == self.root or self.views_root.is_relative_to(self.root)
                or self.root.is_relative_to(self.views_root)):
            raise ValueError("source and view roots must be separate directories")
        self.root.mkdir(parents=True, exist_ok=True)
        self.views_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.candidates = _positive(candidates, "candidates")
        self.top_k = _positive(top_k, "top_k")
        if self.top_k > self.candidates:
            raise ValueError("top_k cannot exceed BM25 candidates")
        self._chunker = RerankerTool(self.root, backend, tokenizer=tokenizer)
        self._counter = 0
        self.space_version = 0
        self._sources: dict[str, dict] = {}
        self._search_sets: dict[str, dict] = {}
        self._chunk_cache: dict[str, list[dict]] = {}
        self._rankings: dict[str, dict] = {}
        self._views: dict[str, dict] = {}
        self._receipts: list[dict] = []
        self._usage = {"prepare_calls": 0, "scoring_calls": 0, "ranking_cache_hits": 0,
                       "view_cache_hits": 0, "scored_passages": 0, "forward_calls": 0,
                       "completed_forward_calls": 0, "input_tokens": 0,
                       "input_tokens_attempted": 0, "padded_input_tokens": 0,
                       "scoring_seconds": 0., "preparation_seconds": 0.,
                       "source_validation_seconds": 0., "chunking_seconds": 0.,
                       "bm25_seconds": 0., "materialization_seconds": 0.,
                       "chunk_cache_hits": 0, "chunk_cache_misses": 0, "errors": 0}

    def _identity(self) -> dict:
        identity = dict(getattr(self.backend, "identity", {}) or {})
        return {key: identity[key] for key in _MODEL_KEYS if key in identity}

    @staticmethod
    def _source_id(rel: str) -> str:
        return "SRC-" + hashlib.sha256(rel.encode()).hexdigest()[:24]

    @staticmethod
    def _passage_id(rel: str, digest: str, start: int, end: int) -> str:
        return "P-" + _digest([rel, digest, start, end])[:32]

    def _source_data(self, source: dict) -> str:
        text, digest = self._chunker._read(source["file_path"])
        if digest != source["source_sha256"]:
            raise ValueError(f"source changed since search: {source['file_path']}")
        return text

    def register_search(self, queries: list[str],
                        per_query_results: list[list[tuple[str, float]]]) -> list[str]:
        if not isinstance(queries, list) or not queries or len(queries) != len(per_query_results):
            raise ValueError("nonempty queries and matching per-query results are required")
        counter = self._counter + 1
        sources, sets = copy.deepcopy(self._sources), {}
        changed = False
        for index, (query, results) in enumerate(zip(queries, per_query_results), 1):
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search query must be nonempty")
            if len(results) > 1000:
                raise ValueError("each search subquery may import at most 1000 documents")
            sid, docs, seen = f"S{counter:03d}.q{index}", [], set()
            for rank, (rel, score) in enumerate(results, 1):
                if rel in seen:
                    continue
                _, sha = self._chunker._read(rel)
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("search score must be finite")
                prior = sources.get(rel)
                if prior is None or prior["source_sha256"] != sha:
                    changed = True
                    sources[rel] = {"source_id": self._source_id(rel), "file_path": rel,
                                    "source_sha256": sha, "search_memberships": []}
                membership = {"set_id": sid, "query": query, "search_rank": rank,
                              "search_score": score}
                sources[rel]["search_memberships"].append(membership)
                docs.append({"file_path": rel, "source_id": self._source_id(rel),
                             "source_sha256": sha, "search_rank": rank, "search_score": score})
                seen.add(rel)
            sets[sid] = {"query": query, "docs": docs}
        self._sources, self._counter = sources, counter
        self._search_sets.update(sets)
        if changed:
            self.space_version += 1
            # Older rankings cannot be used for the updated union.  Evidence
            # receipts remain independently available after view invalidation.
            self._rankings.clear()
            self._views.clear()
        return list(sets)

    def list_sources(self) -> list[dict]:
        return copy.deepcopy(list(self._sources.values()))

    def source_id_for_path(self, file_path: str) -> str:
        return self.get_source(file_path)["source_id"]

    def get_source(self, source_id: str) -> dict:
        if source_id in self._sources:
            return copy.deepcopy(self._sources[source_id])
        for source in self._sources.values():
            if source["source_id"] == source_id:
                return copy.deepcopy(source)
        raise ValueError("unknown imported source")

    def source_text(self, source_id: str) -> tuple[str, dict]:
        source = self.get_source(source_id)
        return self._source_data(source), source

    def _passages(self, texts: dict[str, str]) -> list[dict]:
        started = time.monotonic()
        if self._chunker.tokenizer is None:
            self._chunker.tokenizer = self.backend.get_tokenizer()
        tokenizer_key = self._identity()
        passages = []
        for rel in sorted(self._sources):
            source = self._sources[rel]
            cache_key = _digest([source["source_sha256"], chunking_identity(), tokenizer_key])
            if cache_key not in self._chunk_cache:
                self._usage["chunk_cache_misses"] += 1
                dummy = {"file_path": rel, "sha256": source["source_sha256"],
                         "search_rank": 1, "search_score": 0.}
                chunks = self._chunker._chunks([(dummy, texts[rel])])
                # Cache only content coordinates.  Source path and search
                # memberships are attached anew for each canonical document.
                self._chunk_cache[cache_key] = [{key: value for key, value in chunk.items()
                                                if key not in {"file_path", "search_rank",
                                                               "search_score", "read_args", "bash_args"}}
                                               | {"line_offset": chunk["read_args"]["offset"],
                                                  "line_limit": chunk["read_args"]["limit"]}
                                               for chunk in chunks]
            else:
                self._usage["chunk_cache_hits"] += 1
            for cached in self._chunk_cache[cache_key]:
                chunk = copy.deepcopy(cached)
                start, end = chunk["char_start"], chunk["char_end"]
                pid = self._passage_id(rel, source["source_sha256"], start, end)
                offset, limit = chunk.pop("line_offset"), chunk.pop("line_limit")
                chunk.update(passage_id=pid, source_id=source["source_id"],
                             source_file_path=rel, file_path=pid + ".txt", snippet_file=pid + ".txt",
                             source_read_args={"file_path": rel, "offset": offset, "limit": limit},
                             search_memberships=copy.deepcopy(source["search_memberships"]))
                passages.append(chunk)
        self._usage["chunking_seconds"] += time.monotonic() - started
        return passages

    def _rank(self, query: str, passages: list[dict]) -> dict:
        if not passages:
            return {"total_passages": 0, "items": [], "backend_usage": {},
                    "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}
        bm25_started = time.monotonic()
        tokens = bm25s.tokenize([p["text"] for p in passages], stopwords="en",
                               stemmer=_stemmer(), show_progress=False)
        if any(tokens.ids):
            index = bm25s.BM25(k1=1.5, b=0.75)
            index.index(tokens, show_progress=False)
            query_tokens = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer(), show_progress=False)
            indices, values = index.retrieve(query_tokens, k=len(passages), show_progress=False)
            order = sorted(zip(indices[0], values[0]), key=lambda pair: (-float(pair[1]), int(pair[0])))
        else:
            order = [(i, 0.) for i in range(len(passages))]
        candidates = []
        for rank, (idx, score) in enumerate(order[:self.candidates], 1):
            if not math.isfinite(float(score)):
                raise ValueError("BM25 returned a nonfinite score")
            candidates.append({**passages[int(idx)], "bm25_rank": rank, "bm25_score": float(score)})
        self._usage["bm25_seconds"] += time.monotonic() - bm25_started
        started = time.monotonic()
        self._usage["scoring_calls"] += 1
        try:
            scores = [float(value) for value in self.backend.score(query, [p["text"] for p in candidates])]
        finally:
            self._usage["scoring_seconds"] += time.monotonic() - started
            usage = copy.deepcopy(getattr(self.backend, "last_usage", {}) or {})
            for key in ("forward_calls", "completed_forward_calls", "input_tokens",
                        "input_tokens_attempted", "padded_input_tokens"):
                self._usage[key] += int(usage.get(key, 0))
        if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker must return one finite score per BM25 candidate")
        for passage, score in zip(candidates, scores):
            passage["reranker_score"] = score
        ranked = sorted(candidates, key=lambda p: (-p["reranker_score"], p["bm25_rank"]))
        for rank, passage in enumerate(ranked, 1):
            passage["reranker_rank"] = rank
        self._usage["scored_passages"] += len(candidates)
        return {"total_passages": len(passages), "items": ranked, "backend_usage": usage,
                "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}

    def _ranking_key(self, query: str) -> str:
        return _digest({"space_version": self.space_version, "goal_query": query,
                        "candidates": self.candidates, "chunking": chunking_identity(),
                        "model_identity": self._identity()})

    def _materialize(self, view: dict) -> None:
        started = time.monotonic()
        directory = self.views_root / view["view_id"]
        if directory.is_symlink():
            raise ValueError("view directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        expected = {item["snippet_file"] for item in view["items"]}
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise ValueError("view directory must not contain symlinks")
            if entry.name not in expected or not entry.is_file():
                raise ValueError("view directory contains unexpected files; only retained snippets are allowed")
        for item in view["items"]:
            path = directory / item["snippet_file"]
            if path.is_symlink():
                raise ValueError("snippet file must not be a symlink")
            data = item["text"].encode("utf-8")
            if not path.exists() or path.read_bytes() != data:
                if path.exists():
                    path.chmod(0o600)
                path.write_bytes(data)
                path.chmod(0o444)
        self._usage["materialization_seconds"] += time.monotonic() - started

    @staticmethod
    def _coverage(items: list[dict]) -> dict:
        intervals: dict[str, list[tuple[int, int]]] = {}
        for item in items:
            intervals.setdefault(item["source_id"], []).append((item["char_start"], item["char_end"]))
        coverage = 0
        for spans in intervals.values():
            merged_end = -1
            for start, end in sorted(spans):
                coverage += max(0, end - max(start, merged_end))
                merged_end = max(merged_end, end)
        return {"retained_parent_documents": len(intervals), "retained_unique_characters": coverage}

    def prepare(self, goal_query: str, top_k: int | None = None) -> dict:
        started = time.monotonic()
        self._usage["prepare_calls"] += 1
        try:
            if not isinstance(goal_query, str) or not goal_query.strip():
                raise ValueError("explicit nonempty goal_query is required; no last-search default")
            query = goal_query.strip()
            k = self.top_k if top_k is None else _positive(top_k, "top_k")
            if k > self.candidates:
                raise ValueError("top_k cannot exceed BM25 candidates")
            if not self._sources:
                raise ValueError("search must import documents before preparing a DCI space")
            validation_started = time.monotonic()
            texts = {rel: self._source_data(source) for rel, source in self._sources.items()}
            self._usage["source_validation_seconds"] += time.monotonic() - validation_started
            ranking_key = self._ranking_key(query)
            if ranking_key not in self._rankings and self._chunker.tokenizer is None:
                self._chunker.tokenizer = self.backend.get_tokenizer()
                ranking_key = self._ranking_key(query)
            cache_hit = ranking_key in self._rankings
            if cache_hit:
                self._usage["ranking_cache_hits"] += 1
            else:
                ranked = self._rank(query, self._passages(texts))
                # Resolving tokenizer pins the revision before the key is
                # made. Runtime device/dtype fields do not change this key.
                self._rankings[ranking_key] = {**ranked, "goal_query": query,
                                              "space_version": self.space_version,
                                              "ranking_identity": self._identity()}
            ranking = self._rankings[ranking_key]
            view_id = "V-" + _digest([ranking_key, k])[:32]
            reused = view_id in self._views
            if reused:
                self._usage["view_cache_hits"] += 1
            else:
                items = copy.deepcopy(ranking["items"][:k])
                for item in items:
                    item["read_args"] = {"view_id": view_id, "passage_id": item["passage_id"]}
                self._views[view_id] = {"view_id": view_id, "space_version": self.space_version,
                                       "goal_query": query, "top_k": k, "ranking_key": ranking_key,
                                       "items": items, **self._coverage(items)}
            view = self._views[view_id]
            self._materialize(view)
            preview = copy.deepcopy(view["items"][:PREVIEW_PASSAGES])
            return {"status": "ok", "view_status": "reused" if reused else "created",
                    "view_id": view_id, "space_version": self.space_version, "goal_query": query,
                    "requested_candidates": self.candidates, "top_k": k,
                    "scope_documents": len(self._sources), "pool_documents": len(self._sources),
                    "total_passages": ranking["total_passages"],
                    "bm25_candidates": len(ranking["items"]), "scored_passages": len(ranking["items"]),
                    "retained_passages": len(view["items"]), "cache_hit": cache_hit,
                    "retained_parent_documents": view["retained_parent_documents"],
                    "retained_unique_characters": view["retained_unique_characters"],
                    "ranking_cache_hit": cache_hit, "items": preview,
                    "next_offset": len(preview) if len(preview) < len(view["items"]) else None,
                    "chunking_identity": chunking_identity(),
                    "model_identity": copy.deepcopy(ranking["model_identity"]),
                    "preparation_seconds": time.monotonic() - started,
                    "backend_usage": {} if cache_hit else copy.deepcopy(ranking["backend_usage"])}
        except Exception:
            self._usage["errors"] += 1
            raise
        finally:
            self._usage["preparation_seconds"] += time.monotonic() - started

    def get_view(self, view_id: str) -> dict:
        if view_id not in self._views:
            raise ValueError("unknown or outdated DCI view")
        view = copy.deepcopy(self._views[view_id])
        view["root"] = str(self.view_root(view_id))
        return view

    def view_root(self, view_id: str) -> Path:
        if view_id not in self._views:
            raise ValueError("unknown or outdated DCI view")
        directory = self.views_root / view_id
        if directory.is_symlink():
            raise ValueError("view directory must not be a symlink")
        return directory

    def get_passage(self, view_id: str, passage_id: str) -> dict:
        view = self.get_view(view_id)
        for item in view["items"]:
            if item["passage_id"] == passage_id:
                source = self.get_source(item["source_id"])
                if source["source_sha256"] != item["source_sha256"]:
                    raise ValueError("passage source hash is outdated")
                self._source_data(source)
                return item
        raise ValueError("passage is not retained in this DCI view")

    def record_receipt(self, receipt: dict) -> None:
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be a dictionary")
        json.dumps(receipt, ensure_ascii=False, allow_nan=False)
        self._receipts.append(copy.deepcopy(receipt))

    @property
    def summary(self) -> dict:
        return {**self._usage, "space_version": self.space_version,
                "pool_documents": len(self._sources), "search_sets": len(self._search_sets),
                "ranked_goals": len(self._rankings), "materialized_views": len(self._views),
                "receipts": len(self._receipts), "bm25_candidates": self.candidates,
                "default_top_k": self.top_k, "chunking_identity": chunking_identity(),
                "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}

    def export_state(self) -> dict:
        return copy.deepcopy({"version": STATE_VERSION, "counter": self._counter,
                              "space_version": self.space_version, "candidates": self.candidates,
                              "top_k": self.top_k, "chunking_identity": chunking_identity(),
                              "sources": self._sources, "search_sets": self._search_sets,
                              "rankings": self._rankings, "views": self._views,
                              "receipts": self._receipts, "usage": self._usage,
                              "backend_identity": getattr(self.backend, "identity", {})})

    def restore_state(self, state: dict) -> None:
        """Validate raw source coordinates and identity, then rebuild snippets.

        Tokenizer/model loading and neural forwards are unnecessary for a valid
        checkpoint. Chunk bodies are rebuilt lazily only if a new goal is used.
        """
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise ValueError("invalid accumulated-space checkpoint version")
        if (state.get("chunking_identity") != chunking_identity()
                or state.get("candidates") != self.candidates or state.get("top_k") != self.top_k):
            raise ValueError("accumulated-space checkpoint configuration changed")
        saved_identity = state.get("backend_identity", {})
        current_identity = dict(getattr(self.backend, "identity", {}) or {})
        for key in _MODEL_KEYS:
            if key == "resolved_revision" and current_identity.get(key) is None:
                continue
            if current_identity.get(key) != saved_identity.get(key):
                raise ValueError(f"accumulated-space checkpoint model identity changed: {key}")
        counter, version = state.get("counter"), state.get("space_version")
        if (isinstance(counter, bool) or not isinstance(counter, int) or counter < 0
                or isinstance(version, bool) or not isinstance(version, int) or version < 0):
            raise ValueError("invalid accumulated-space checkpoint counters")
        sources, sets = state.get("sources"), state.get("search_sets")
        if not isinstance(sources, dict) or not isinstance(sets, dict):
            raise ValueError("invalid accumulated-space source manifest")
        texts = {}
        for rel, source in sources.items():
            if (not isinstance(source, dict) or source.get("file_path") != rel
                    or source.get("source_id") != self._source_id(rel)
                    or not isinstance(source.get("source_sha256"), str)
                    or not _SHA.fullmatch(source["source_sha256"])
                    or not isinstance(source.get("search_memberships"), list)):
                raise ValueError("invalid accumulated-space source record")
            texts[rel] = self._source_data(source)
        for sid, scope in sets.items():
            match = re.fullmatch(r"S(\d+)\.q([1-9]\d*)", sid)
            if (not match or int(match[1]) > counter or not isinstance(scope, dict)
                    or not isinstance(scope.get("query"), str) or not scope["query"].strip()
                    or not isinstance(scope.get("docs"), list) or len(scope["docs"]) > 1000):
                raise ValueError("invalid accumulated-space search record")
            if any(not isinstance(doc, dict) or doc.get("file_path") not in sources
                   for doc in scope["docs"]):
                raise ValueError("search record references an unknown source")
        rankings, views = state.get("rankings"), state.get("views")
        if not isinstance(rankings, dict) or not isinstance(views, dict):
            raise ValueError("invalid accumulated-space cached rankings")
        identity = {key: saved_identity[key] for key in _MODEL_KEYS if key in saved_identity}
        for key, ranking in rankings.items():
            model_identity = ranking.get("model_identity", {}) if isinstance(ranking, dict) else {}
            if (not isinstance(ranking, dict) or ranking.get("space_version") != version
                    or not isinstance(ranking.get("goal_query"), str) or not ranking["goal_query"].strip()
                    or ranking.get("ranking_identity") != identity
                    or not isinstance(model_identity, dict)
                    or {key: model_identity[key] for key in _MODEL_KEYS if key in model_identity} != identity
                    or not isinstance(ranking.get("items"), list)
                    or len(ranking["items"]) > self.candidates):
                raise ValueError("invalid accumulated-space ranking")
            expected_key = _digest({"space_version": version, "goal_query": ranking["goal_query"],
                                    "candidates": self.candidates, "chunking": chunking_identity(),
                                    "model_identity": identity})
            if key != expected_key:
                raise ValueError("invalid accumulated-space ranking cache key")
            total = ranking.get("total_passages")
            if (isinstance(total, bool) or not isinstance(total, int)
                    or total < len(ranking["items"])):
                raise ValueError("invalid accumulated-space passage pool count")
            seen = set()
            for rank, item in enumerate(ranking["items"], 1):
                rel = item.get("source_file_path")
                source = sources.get(rel)
                start, end = item.get("char_start"), item.get("char_end")
                if (source is None or isinstance(start, bool) or not isinstance(start, int)
                        or isinstance(end, bool) or not isinstance(end, int)
                        or not 0 <= start < end <= len(texts[rel])):
                    raise ValueError("invalid checkpoint passage coordinates")
                pid = self._passage_id(rel, source["source_sha256"], start, end)
                if (pid in seen or item.get("passage_id") != pid
                        or item.get("source_id") != source["source_id"]
                        or item.get("source_sha256") != source["source_sha256"]
                        or item.get("snippet_file") != pid + ".txt"
                        or item.get("file_path") != pid + ".txt"
                        or item.get("text") != texts[rel][start:end]
                        or isinstance(item.get("token_count"), bool)
                        or not isinstance(item.get("token_count"), int)
                        or not 0 < item["token_count"] <= 128
                        or item.get("reranker_rank") != rank
                        or not math.isfinite(float(item.get("reranker_score", float("nan"))))
                        or not math.isfinite(float(item.get("bm25_score", float("nan"))))):
                    raise ValueError("invalid checkpoint passage")
                seen.add(pid)
        for vid, view in views.items():
            if not isinstance(view, dict) or view.get("ranking_key") not in rankings:
                raise ValueError("view references an unknown ranking")
            key, k = view["ranking_key"], view.get("top_k")
            _positive(k, "checkpoint top_k")
            ranking = rankings[key]
            expected_items = copy.deepcopy(ranking["items"][:k])
            for item in expected_items:
                item["read_args"] = {"view_id": vid, "passage_id": item["passage_id"]}
            if (k > self.candidates or vid != "V-" + _digest([key, k])[:32]
                    or view.get("view_id") != vid or view.get("space_version") != version
                    or view.get("goal_query") != ranking["goal_query"]
                    or view.get("items") != expected_items
                    or any(view.get(metric) != value for metric, value in self._coverage(expected_items).items())):
                raise ValueError("invalid accumulated-space view")
        usage, receipts = state.get("usage"), state.get("receipts")
        if not isinstance(usage, dict) or set(usage) != set(self._usage) or not isinstance(receipts, list):
            raise ValueError("invalid accumulated-space usage or receipts")
        if any(not isinstance(receipt, dict) for receipt in receipts):
            raise ValueError("invalid accumulated-space receipt")
        for value in usage.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid accumulated-space usage value")
        restore_identity = getattr(self.backend, "restore_identity", None)
        if restore_identity is not None:
            restore_identity(copy.deepcopy(saved_identity))
        self._counter, self.space_version = counter, version
        self._sources, self._search_sets = copy.deepcopy(sources), copy.deepcopy(sets)
        self._rankings, self._views = copy.deepcopy(rankings), copy.deepcopy(views)
        self._usage, self._receipts = copy.deepcopy(usage), copy.deepcopy(receipts)
        self._chunk_cache.clear()
        for view in self._views.values():
            self._materialize(view)
