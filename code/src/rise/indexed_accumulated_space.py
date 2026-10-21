"""v12: v11 DCI semantics backed by a published raw-TF passage index.

Each manager has private lexical query/scope state while the large immutable
memmaps are shared.  Ordinary preparation validates source stat fingerprints,
scores the imported union and reads only BM25 candidates.  Checkpoint restore
still SHA-checks every source, but keeps full source text only for cached hits.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import re
import time

import numpy as np
import Stemmer

from .accumulated_space import AccumulatedSpaceManager, PREVIEW_PASSAGES, _digest, _positive
from .passage_index import PassageIndex, stable_top_k
from .reranker_tool import BODY_BOUNDARY, chunking_identity


class _CheckpointSourceText:
    """A source-length/slicing proxy for the unmodified v11 restore validator."""

    def __init__(self, manager, source, chars):
        self.manager, self.source, self.chars = manager, source, chars
        self._text = None

    def __len__(self):
        return self.chars

    def __getitem__(self, key):
        if self._text is None:
            self._text = self.manager._read_runtime_source(self.source)
        return self._text[key]


class IndexedAccumulatedSpaceManager(AccumulatedSpaceManager):
    """Use offline coordinates/TF without changing the model's tool contract."""

    def __init__(self, working_dir: Path, views_dir: Path, backend,
                 candidates: int = 1000, top_k: int = 100, tokenizer=None,
                 *, passage_index: PassageIndex, runtime_to_corpus: dict | None = None):
        super().__init__(working_dir, views_dir, backend, candidates, top_k, tokenizer)
        if not isinstance(passage_index, PassageIndex):
            raise ValueError("passage_index must be a loaded PassageIndex")
        if passage_index.manifest.get("status") != "ready":
            raise ValueError("only a published ready passage index can back DCI")
        # Scope caches and PyStemmer objects are mutable and must not be shared
        # by the two concurrently running question managers.
        self.passage_index = copy.copy(passage_index)
        self.passage_index._scopes = OrderedDict()
        self.passage_index._stemmer = Stemmer.Stemmer("english")
        if runtime_to_corpus is not None and not isinstance(runtime_to_corpus, dict):
            raise ValueError("runtime_to_corpus must be a dictionary or None")
        self.runtime_to_corpus = None if runtime_to_corpus is None else dict(runtime_to_corpus)
        self._source_snapshots: dict[str, tuple] = {}
        self._restoring_indexed = False
        self._last_index_prepare: dict = {}
        self._index_prepare_history: list[dict] = []
        self._validate_index_configuration()
        manifest = self.passage_index.manifest
        self._passage_index_identity = {
            "format_version": manifest["format_version"],
            "configuration": copy.deepcopy(manifest.get("configuration", manifest.get("identity", {}))),
            "corpus_content_sha256": manifest.get("corpus_content_sha256"),
            "counts": {key: getattr(self.passage_index, key) for key in
                       ("document_count", "passage_count", "vocabulary_count", "posting_count")},
            "files": copy.deepcopy({name: metadata for name, metadata in manifest.get("files", {}).items()
                                    if name != "validation.json"}),
        }
        self._passage_index_identity["sha256"] = _digest(self._passage_index_identity)
        self._usage.update(index_scope_seconds=0., index_lookup_seconds=0.,
                           candidate_materialization_seconds=0., source_stat_checks=0,
                           source_rehashes=0, index_scope_cache_hits=0,
                           candidate_sources_read=0, candidate_passages_materialized=0)

    def _validate_index_configuration(self):
        configuration = self.passage_index.manifest.get("configuration",
                             self.passage_index.manifest.get("identity", {}))
        actual = configuration.get("chunking", {})
        expected = chunking_identity()
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("passage index chunking identity differs from v11 128/32 rules")
        checks = {"body_boundary_pattern": BODY_BOUNDARY.pattern,
                  "source_encoding": "utf-8", "source_encoding_errors": "replace"}
        if any(key in actual and actual[key] != value for key, value in checks.items()):
            raise ValueError("passage index source extraction/encoding differs from v11")
        identity = dict(getattr(self.backend, "identity", {}) or {})
        revision = configuration.get("tokenizer", {}).get("revision")
        if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            revision = revision.lower()
            for key in ("resolved_revision", "requested_revision", "revision"):
                current = identity.get(key)
                if current and re.fullmatch(r"[0-9a-fA-F]{40}", str(current)) and current.lower() != revision:
                    raise ValueError("backend tokenizer revision differs from passage index")
        if identity.get("kind") == "qwen_reranker":
            # The published corpus was tokenized with this concrete model.
            if identity.get("model") != "Qwen/Qwen3-Reranker-0.6B":
                raise ValueError("published passage tokenizer requires Qwen/Qwen3-Reranker-0.6B")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("passage index must pin an immutable Qwen tokenizer revision")
            restore = getattr(self.backend, "restore_identity", None)
            if restore is None:
                raise ValueError("Qwen backend cannot pin the passage tokenizer revision")
            restore({**identity, "resolved_revision": revision})

    def _corpus_path(self, rel: str) -> str:
        if self.runtime_to_corpus is None:
            result = rel
        else:
            if rel not in self.runtime_to_corpus:
                raise ValueError(f"runtime source is not mapped into passage index: {rel}")
            result = self.runtime_to_corpus[rel]
        # Reader performs the relative-path safety and membership checks.
        self.passage_index._doc_id(result)
        return result

    def _indexed_doc(self, source: dict) -> tuple[int, dict]:
        corpus = self._corpus_path(source["file_path"])
        did = self.passage_index._doc_id({"path": corpus,
                                         "source_sha256": source["source_sha256"]})
        return did, self.passage_index.docs[did]

    def _stat_source(self, rel: str) -> tuple:
        path = self._chunker._path(rel)  # includes ancestor symlink rejection
        info = path.stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _stream_verify_source(self, source: dict) -> tuple:
        rel = source["file_path"]
        self._indexed_doc(source)
        # Creating the second question's hard link can change inode ctime.
        # Retry once, still requiring the expected content and a stable read.
        for _ in range(2):
            before = self._stat_source(rel)
            digest = hashlib.sha256()
            with self._chunker._path(rel).open("rb") as stream:
                while block := stream.read(1 << 20):
                    digest.update(block)
            after = self._stat_source(rel)
            self._usage["source_rehashes"] += 1
            if digest.hexdigest() != source["source_sha256"]:
                raise ValueError(f"source changed since search: {rel}")
            if before == after:
                return after
        raise ValueError(f"source changed while validating: {rel}")

    def _read_runtime_source(self, source: dict) -> str:
        _, doc = self._indexed_doc(source)
        rel = source["file_path"]
        for _ in range(2):
            before = self._stat_source(rel)
            text = super()._source_data(source)
            after = self._stat_source(rel)
            if len(text) != doc["chars"]:
                raise ValueError(f"source character count differs from passage index: {rel}")
            if before == after:
                self._source_snapshots[rel] = after
                return text
        raise ValueError(f"source changed while reading a candidate: {rel}")

    def _source_data(self, source: dict):
        if self._restoring_indexed:
            _, doc = self._indexed_doc(source)
            self._source_snapshots[source["file_path"]] = self._stream_verify_source(source)
            return _CheckpointSourceText(self, source, doc["chars"])
        return super()._source_data(source)

    def _check_source_snapshots(self, file_paths=None):
        seen = set()
        for rel in sorted(self._sources if file_paths is None else file_paths):
            source = self._sources[rel]
            did, _ = self._indexed_doc(source)
            if did in seen:
                raise ValueError("distinct runtime sources map to one corpus document; local BM25 would change")
            seen.add(did)
            current = self._stat_source(rel)
            self._usage["source_stat_checks"] += 1
            if self._source_snapshots.get(rel) != current:
                current = self._stream_verify_source(source)
                self._source_snapshots[rel] = current

    def register_search(self, queries, per_query_results):
        rels = {rel for results in per_query_results for rel, _ in results}
        for rel in rels:
            self._corpus_path(rel)
        before = {rel: self._stat_source(rel) for rel in rels}
        # Parent registration stages sources, but updates search sets and
        # clears caches in place.  Preserve these references for rollback if
        # the new source bytes disagree with the frozen corpus generation.
        previous = (self._sources, self._counter, self.space_version,
                    dict(self._search_sets), dict(self._rankings), dict(self._views),
                    dict(self._source_snapshots))
        try:
            result = super().register_search(queries, per_query_results)
            seen = set()
            for rel, source in self._sources.items():
                did, _ = self._indexed_doc(source)
                if did in seen:
                    raise ValueError("distinct runtime sources map to one corpus document")
                seen.add(did)
            for rel in rels:
                source = self._sources[rel]
                current = self._stat_source(rel)
                if current != before[rel]:
                    current = self._stream_verify_source(source)
                self._source_snapshots[rel] = current
            return result
        except Exception:
            (self._sources, self._counter, self.space_version, self._search_sets,
             self._rankings, self._views, self._source_snapshots) = previous
            raise

    def _indexed_scope(self, file_paths=None):
        selectors, runtime_doc_order, by_doc = [], [], {}
        for rel in sorted(self._sources if file_paths is None else file_paths):
            source = self._sources[rel]
            did, doc = self._indexed_doc(source)
            if did in by_doc:
                raise ValueError("runtime-to-corpus mapping must be one-to-one")
            selectors.append({"doc_id": did, "sha256": source["source_sha256"]})
            runtime_doc_order.append(did)
            by_doc[did] = rel
        key = tuple(sorted(runtime_doc_order))
        if key in self.passage_index._scopes:
            self._usage["index_scope_cache_hits"] += 1
        return self.passage_index.scope(selectors), runtime_doc_order, by_doc

    def _canonical_top(self, scope, scores, runtime_doc_order, candidates=None):
        count = self.candidates if candidates is None else candidates
        if tuple(runtime_doc_order) == scope.doc_ids:
            return stable_top_k(scores, count)
        # Global corpus order can differ from sorted runtime-source order.
        # Reorder only numeric local positions; do not construct all passage
        # dictionaries.  Scoring still uses the identical selected set.
        blocks = []
        for did in runtime_doc_order:
            start, end = map(int, self.passage_index.doc_offsets[did:did + 2])
            local_start = int(np.searchsorted(scope.passage_ids, start))
            blocks.append(np.arange(local_start, local_start + end - start, dtype=np.int64))
        canonical = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        if len(canonical) != scope.N:
            raise ValueError("canonical runtime passage order does not cover the scope")
        return canonical[stable_top_k(scores[canonical], count)]

    def _candidate_records(self, scope, positions, by_doc, scores):
        candidates, texts, lines = [], {}, {}
        for rank, local_pid in enumerate(positions, 1):
            global_pid = int(scope.passage_ids[local_pid])
            row = self.passage_index.passage(global_pid)
            rel = by_doc[row["doc_id"]]
            source = self._sources[rel]
            if rel not in texts:
                texts[rel] = self._read_runtime_source(source)
                lines[rel] = texts[rel].splitlines()
                self._usage["candidate_sources_read"] += 1
            text = texts[rel]
            start, end = row["char_start"], row["char_end"]
            if not 0 <= start < end <= len(text):
                raise ValueError("indexed candidate coordinates are outside source")
            pid = self._passage_id(rel, source["source_sha256"], start, end)
            offset, limit = row["line_offset"], row["line_limit"]
            may_truncate = any(len(line) > 2000 for line in lines[rel][offset:offset + limit])
            item = {
                "passage_id": pid, "source_id": source["source_id"],
                "source_file_path": rel, "file_path": pid + ".txt", "snippet_file": pid + ".txt",
                "text": text[start:end], "token_count": row["token_count"],
                "overlap_tokens": row["overlap_tokens"], "char_start": start, "char_end": end,
                "source_sha256": source["source_sha256"], "read_may_truncate": may_truncate,
                "source_read_args": {"file_path": rel, "offset": offset, "limit": limit},
                "search_memberships": copy.deepcopy(source["search_memberships"]),
                "bm25_rank": rank, "bm25_score": float(scores[local_pid]),
            }
            # Frozen v11's accumulated _passages drops the v10 bash_args field;
            # retain its exact existing public payload for a same-prompt run.
            candidates.append(item)
        self._usage["candidate_passages_materialized"] += len(candidates)
        return candidates

    def _score_candidates(self, query, candidates, total):
        if not candidates:
            return {"total_passages": total, "items": [], "backend_usage": {},
                    "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}
        started = time.monotonic()
        self._usage["scoring_calls"] += 1
        try:
            scores = [float(value) for value in self.backend.score(query, [item["text"] for item in candidates])]
        finally:
            self._usage["scoring_seconds"] += time.monotonic() - started
            usage = dict(getattr(self.backend, "last_usage", {}) or {})
            for key in ("forward_calls", "completed_forward_calls", "input_tokens",
                        "input_tokens_attempted", "padded_input_tokens"):
                self._usage[key] += int(usage.get(key, 0))
        if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker must return one finite score per BM25 candidate")
        for passage, score in zip(candidates, scores):
            passage["reranker_score"] = score
        ranked = sorted(candidates, key=lambda item: (-item["reranker_score"], item["bm25_rank"]))
        for rank, item in enumerate(ranked, 1):
            item["reranker_rank"] = rank
        self._usage["scored_passages"] += len(candidates)
        return {"total_passages": total, "items": ranked, "backend_usage": usage,
                "model_identity": copy.deepcopy(getattr(self.backend, "identity", {}))}

    def prepare(self, goal_query: str, top_k: int | None = None) -> dict:
        started = time.monotonic()
        before_usage = dict(self._usage)
        cache_hit, succeeded = None, False
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
            stage = time.monotonic()
            self._check_source_snapshots()
            self._usage["source_validation_seconds"] += time.monotonic() - stage
            ranking_key = self._ranking_key(query)
            cache_hit = ranking_key in self._rankings
            if cache_hit:
                self._usage["ranking_cache_hits"] += 1
            else:
                stage = time.monotonic()
                scope, doc_order, by_doc = self._indexed_scope()
                scope_seconds = time.monotonic() - stage
                self._usage["index_scope_seconds"] += scope_seconds
                stage = time.monotonic()
                scores = scope.scores(query)
                positions = self._canonical_top(scope, scores, doc_order)
                lookup_seconds = time.monotonic() - stage
                self._usage["index_lookup_seconds"] += lookup_seconds
                self._usage["bm25_seconds"] += scope_seconds + lookup_seconds
                stage = time.monotonic()
                candidates = self._candidate_records(scope, positions, by_doc, scores)
                self._usage["candidate_materialization_seconds"] += time.monotonic() - stage
                ranked = self._score_candidates(query, candidates, scope.N)
                # Score may resolve backend identity; keep the base formula so
                # v11's unchanged static restore validator accepts this key.
                ranking_key = self._ranking_key(query)
                self._rankings[ranking_key] = {**ranked, "goal_query": query,
                    "space_version": self.space_version, "ranking_identity": self._identity()}
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
            succeeded = True
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
            self._last_index_prepare = {key: self._usage[key] - before_usage.get(key, 0)
                                        for key in self._usage}
            self._index_prepare_history.append({
                "goal_query": goal_query.strip() if isinstance(goal_query, str) else None,
                "space_version": self.space_version, "cache_hit": cache_hit,
                "status": "ok" if succeeded else "error",
                "delta_usage": copy.deepcopy(self._last_index_prepare),
            })
            self._index_prepare_history = self._index_prepare_history[-100:]

    @property
    def summary(self):
        return {**super().summary,
                "passage_index_identity": copy.deepcopy(self._passage_index_identity),
                "last_index_prepare": copy.deepcopy(self._last_index_prepare),
                "index_prepare_history": copy.deepcopy(self._index_prepare_history)}

    def export_state(self):
        state = super().export_state()
        state["passage_index"] = {
            "binding_version": 1, "identity": copy.deepcopy(self._passage_index_identity),
            "runtime_to_corpus": {rel: self._corpus_path(rel) for rel in self._sources},
            "index_prepare_history": copy.deepcopy(self._index_prepare_history),
        }
        return state

    def restore_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("invalid indexed accumulated-space checkpoint")
        binding = state.get("passage_index")
        if (not isinstance(binding, dict) or binding.get("binding_version") != 1
                or binding.get("identity") != self._passage_index_identity):
            raise ValueError("passage index generation/configuration changed since checkpoint")
        sources = state.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("invalid indexed accumulated-space sources")
        expected_mapping = {rel: self._corpus_path(rel) for rel in sources}
        if binding.get("runtime_to_corpus") != expected_mapping:
            raise ValueError("runtime-to-corpus mapping changed since checkpoint")
        history = binding.get("index_prepare_history", [])
        if not isinstance(history, list) or len(history) > 100:
            raise ValueError("invalid indexed prepare history")
        for entry in history:
            if (not isinstance(entry, dict) or entry.get("status") not in {"ok", "error"}
                    or entry.get("cache_hit") is not None and not isinstance(entry["cache_hit"], bool)
                    or entry.get("goal_query") is not None and not isinstance(entry["goal_query"], str)
                    or isinstance(entry.get("space_version"), bool)
                    or not isinstance(entry.get("space_version"), int) or entry["space_version"] < 0
                    or not isinstance(entry.get("delta_usage"), dict)
                    or set(entry["delta_usage"]) != set(self._usage)):
                raise ValueError("invalid indexed prepare history entry")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in entry["delta_usage"].values()):
                raise ValueError("invalid indexed prepare history metrics")
        old_snapshots = self._source_snapshots
        self._source_snapshots = {}
        self._restoring_indexed = True
        try:
            # Full original schema/ranking/view validation runs on lazy source
            # proxies: streaming SHA for every source, text only for hits.
            super().restore_state(state)
        except Exception:
            self._source_snapshots = old_snapshots
            raise
        finally:
            self._restoring_indexed = False
        self._index_prepare_history = copy.deepcopy(history)
        self._last_index_prepare = copy.deepcopy(history[-1]["delta_usage"]) if history else {}
