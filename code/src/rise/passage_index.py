#!/usr/bin/env python3
"""Read reusable passage/term postings and score a search-union locally.

This module only loads lexical data.  It never loads a neural tokenizer or a
reranker model.  Global postings store *raw* TF; N, DF and average length are
derived from the selected document union, rather than from the whole corpus.

The arrays are memory mapped.  A Scope keeps only its ordered passage IDs,
lexical lengths and bounded caches of postings for terms queried so far.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import OrderedDict
import hashlib
from itertools import pairwise
import json
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Iterable

import bm25s
from bm25s.scoring import _score_idf_lucene, _score_tfc_lucene
import numpy as np
import Stemmer

FORMAT_VERSION = "passage-lexical-postings-v1"
PASSAGE_DTYPE = np.dtype([
    ("doc_id", "<u4"), ("char_start", "<u8"), ("char_end", "<u8"),
    ("token_count", "<u2"), ("overlap_tokens", "<u2"),
    ("line_offset", "<u4"), ("line_limit", "<u4"),
])
ARRAY_DTYPES = {
    "passages": PASSAGE_DTYPE,
    "doc_offsets": np.dtype("<u8"),
    "lengths": np.dtype("<u4"),
    "term_indptr": np.dtype("<u8"),
    "posting_pids": np.dtype("<u8"),
    "posting_tfs": np.dtype("<u4"),
}


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("document path must be a string")
    parsed = PurePosixPath(value)
    if (not value or parsed.is_absolute() or ".." in parsed.parts
            or parsed.as_posix() != value):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def stable_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Exact score-descending, original-local-ID-ascending top K.

    argpartition alone would arbitrarily discard equal-score boundary items.
    We find the threshold, take all strictly higher scores, and fill ties in
    their original order before sorting just the selected candidates.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    count = len(scores)
    k = min(int(k), count)
    if not k:
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite BM25 score")
    if k == count:
        chosen = np.arange(count, dtype=np.int64)
    else:
        threshold = np.partition(scores, count - k)[count - k]
        higher = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)[:k - len(higher)]
        chosen = np.concatenate((higher, ties))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


class PassageIndex:
    """One immutable, published passage index.

    ``allow_unpublished`` is exclusively for pre-publication validation.  A
    normal consumer refuses any generation without manifest.status=ready.
    Source text is read by the stored UTF-8-replacement character coordinates
    and its complete source SHA is verified whenever text is requested.
    """

    def __init__(self, path: str | Path, allow_unpublished: bool = False,
                 *, corpus_root: str | Path | None = None):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text("utf-8"))
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported passage index format_version")
        if not allow_unpublished and self.manifest.get("status") != "ready":
            raise ValueError("index has not passed publication validation")
        cfg = self.manifest.get("bm25_config", {})
        if (cfg.get("stopwords", "en") != "en"
                or cfg.get("stemmer", "english") != "english"
                or cfg.get("method", "lucene") != "lucene"
                or float(cfg.get("k1", 1.5)) != 1.5
                or float(cfg.get("b", 0.75)) != 0.75):
            raise ValueError("index lexical configuration differs from frozen v11")
        expected_bm25s = cfg.get("version", cfg.get("bm25s_version", "0.3.11"))
        if bm25s.__version__ != expected_bm25s:
            raise ValueError(f"bm25s {expected_bm25s} required; got {bm25s.__version__}")
        self.corpus_root = Path(corpus_root or self.manifest["corpus_root"]).resolve()
        self.docs = []
        with (self.path / "docs.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    self.docs.append(json.loads(line))
        self._path_to_doc = {}
        for ordinal, doc in enumerate(self.docs):
            if doc["doc_id"] != ordinal:
                raise ValueError("doc_id must be contiguous in docs.jsonl order")
            rel = _relative_path(doc["path"])
            if rel in self._path_to_doc:
                raise ValueError("duplicate source path")
            self._path_to_doc[rel] = ordinal
        if [doc["path"] for doc in self.docs] != sorted(self._path_to_doc):
            raise ValueError("documents must be ordered by Python sorted(relative path)")
        self.vocab = json.loads((self.path / "vocab.json").read_text("utf-8"))
        if (not isinstance(self.vocab, list)
                or any(not isinstance(term, str) for term in self.vocab)
                or any(a >= b for a, b in pairwise(self.vocab))):
            raise ValueError("vocab must be a unique lexically sorted list")
        for name, dtype in ARRAY_DTYPES.items():
            arr = np.load(self.path / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            if arr.ndim != 1 or arr.dtype != dtype:
                raise ValueError(f"invalid dtype/shape for {name}: {arr.dtype}/{arr.shape}")
            setattr(self, name, arr)
        self.document_count = len(self.docs)
        self.passage_count = len(self.passages)
        self.vocabulary_count = len(self.vocab)
        self.posting_count = len(self.posting_pids)
        expected = {
            "document_count": self.document_count, "passage_count": self.passage_count,
            "vocabulary_count": self.vocabulary_count, "posting_count": self.posting_count,
        }
        for name, value in expected.items():
            if self.manifest.get(name, value) != value:
                raise ValueError(f"manifest {name} does not match index")
        if (len(self.lengths) != self.passage_count
                or len(self.doc_offsets) != self.document_count + 1
                or int(self.doc_offsets[0]) != 0
                or int(self.doc_offsets[-1]) != self.passage_count
                or len(self.term_indptr) != self.vocabulary_count + 1
                or int(self.term_indptr[0]) != 0
                or int(self.term_indptr[-1]) != self.posting_count
                or len(self.posting_tfs) != self.posting_count):
            raise ValueError("inconsistent index array sizes")
        self._stemmer = Stemmer.Stemmer("english")
        self._scopes: OrderedDict[tuple[int, ...], Scope] = OrderedDict()

    def info(self) -> dict:
        return {"index": str(self.path), "status": self.manifest.get("status"),
                "format_version": FORMAT_VERSION, "corpus_root": str(self.corpus_root),
                "document_count": self.document_count, "passage_count": self.passage_count,
                "vocabulary_count": self.vocabulary_count, "posting_count": self.posting_count,
                "bm25_config": self.manifest.get("bm25_config", {}),
                "identity": self.manifest.get("identity", self.manifest.get("configuration", {})),
                "disk_bytes": sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())}

    def _doc_id(self, selector) -> int:
        expected_sha = None
        if isinstance(selector, dict):
            expected_sha = selector.get("sha256", selector.get("source_sha256"))
            selector = selector.get("doc_id", selector.get("path", selector.get("file_path")))
        if isinstance(selector, bool):
            raise ValueError("boolean is not a document ID")
        if isinstance(selector, (int, np.integer)):
            doc_id = int(selector)
            if not 0 <= doc_id < self.document_count:
                raise ValueError(f"unknown doc_id: {doc_id}")
        elif isinstance(selector, str):
            rel = _relative_path(selector)
            if rel not in self._path_to_doc:
                raise ValueError(f"document is not in index: {rel}")
            doc_id = self._path_to_doc[rel]
        else:
            raise ValueError("document selectors must be relative paths or integer IDs")
        if expected_sha is not None and expected_sha != self.docs[doc_id]["sha256"]:
            raise ValueError(f"source hash differs from index: {self.docs[doc_id]['path']}")
        return doc_id

    def scope(self, documents: Iterable, *, cache: bool = True) -> "Scope":
        if isinstance(documents, (str, bytes)):
            raise ValueError("documents must be an iterable of source paths or IDs")
        key = tuple(sorted({self._doc_id(selector) for selector in documents}))
        if cache and key in self._scopes:
            self._scopes.move_to_end(key)
            return self._scopes[key]
        result = Scope(self, key)
        if cache:
            self._scopes[key] = result
            while len(self._scopes) > 2:
                self._scopes.popitem(last=False)
        return result

    def passage(self, pid: int, include_text: bool = False) -> dict:
        if isinstance(pid, bool) or not isinstance(pid, (int, np.integer)) or not 0 <= pid < self.passage_count:
            raise ValueError("invalid global passage ID")
        pid = int(pid)
        row = self.passages[pid]
        doc_id = int(row["doc_id"])
        doc = self.docs[doc_id]
        item = {"passage_id": pid, "doc_id": doc_id, "path": doc["path"],
                "source_sha256": doc["sha256"],
                **{field: int(row[field]) for field in PASSAGE_DTYPE.names if field != "doc_id"},
                "bm25_length": int(self.lengths[pid])}
        if include_text:
            item["text"] = self.text(pid)
        return item

    def _source_text(self, doc_id: int) -> str:
        doc = self.docs[doc_id]
        path = self.corpus_root / doc["path"]
        for component in (path, *path.parents):
            if component == self.corpus_root:
                break
            if component.is_symlink():
                raise ValueError(f"symlink source is not permitted: {doc['path']}")
        if not path.resolve().is_relative_to(self.corpus_root) or not path.is_file():
            raise ValueError(f"source missing or outside corpus: {doc['path']}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != doc["sha256"]:
            raise ValueError(f"source changed since index construction: {doc['path']}")
        text = data.decode("utf-8", errors="replace")
        if len(text) != doc["chars"]:
            raise ValueError(f"source character count differs: {doc['path']}")
        return text

    def text(self, pid: int) -> str:
        item = self.passage(pid)
        text = self._source_text(item["doc_id"])
        start, end = item["char_start"], item["char_end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("invalid passage character range")
        return text[start:end]

    def query_terms(self, query: str) -> list[str]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        return bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer,
                              return_ids=False, show_progress=False)[0]

    def term_id(self, term: str) -> int | None:
        pos = bisect_left(self.vocab, term)
        return pos if pos < self.vocabulary_count and self.vocab[pos] == term else None

    def verify(self, *, sources: bool = False) -> dict:
        started = time.monotonic()
        files = self.manifest.get("files", {})
        required = {name + ".npy" for name in ARRAY_DTYPES} | {"docs.jsonl", "vocab.json"}
        if not required.issubset(files):
            raise ValueError("manifest is missing required index file checksums")
        checked = []
        for rel, metadata in files.items():
            path = self.path / _relative_path(rel)
            if isinstance(metadata, str):
                digest, size = metadata, None
            else:
                digest, size = metadata["sha256"], metadata.get("bytes")
            if (path.is_symlink() or not path.is_file()
                    or (size is not None and path.stat().st_size != size)
                    or sha256_file(path) != digest):
                raise ValueError(f"index checksum mismatch: {rel}")
            checked.append(rel)
        if not files:
            raise ValueError("manifest has no index file checksums")
        if np.any(self.doc_offsets[1:] < self.doc_offsets[:-1]):
            raise ValueError("doc_offsets is not monotonic")
        if np.any(self.term_indptr[1:] < self.term_indptr[:-1]):
            raise ValueError("term_indptr is not monotonic")
        for doc_id, doc in enumerate(self.docs):
            start, end = map(int, self.doc_offsets[doc_id:doc_id + 2])
            rows = self.passages[start:end]
            if (end - start != doc["passage_count"]
                    or np.any(rows["doc_id"] != doc_id)
                    or np.any(rows["char_start"] >= rows["char_end"])
                    or np.any(rows["char_end"] > doc["chars"])
                    or np.any(rows["token_count"] > 128)
                    or np.any(rows["token_count"] == 0)
                    or np.any(rows["overlap_tokens"] > 32)):
                raise ValueError(f"invalid passage/document mapping: {doc_id}")
            if sources:
                self._source_text(doc_id)
        # Bound temporary allocation while validating all raw posting records.
        for start in range(0, self.posting_count, 1_000_000):
            end = min(start + 1_000_000, self.posting_count)
            if np.any(self.posting_pids[start:end] >= self.passage_count) or np.any(self.posting_tfs[start:end] == 0):
                raise ValueError("invalid posting PID or term frequency")
        # Each term is ordered, has one TF per passage, and does not straddle a
        # duplicate entry.  A chunk boundary check is performed term by term.
        for term_id in range(self.vocabulary_count):
            start, end = map(int, self.term_indptr[term_id:term_id + 2])
            if end == start:
                raise ValueError("empty term in vocabulary")
            previous = None
            for offset in range(start, end, 1_000_000):
                pids = self.posting_pids[offset:min(offset + 1_000_000, end)]
                if (previous is not None and int(pids[0]) <= previous) or np.any(pids[1:] <= pids[:-1]):
                    raise ValueError(f"term postings are not unique/ordered: {term_id}")
                previous = int(pids[-1])
        return {"passed": True, "files_checked": checked, "documents": self.document_count,
                "passages": self.passage_count, "postings": self.posting_count,
                "source_hashes_checked": self.document_count if sources else 0,
                "seconds": time.monotonic() - started}


class Scope:
    """Reusable exact BM25 statistics for one accumulated document union."""

    def __init__(self, index: PassageIndex, doc_ids: tuple[int, ...]):
        self.index = index
        self.doc_ids = doc_ids
        pieces = [np.arange(int(index.doc_offsets[d]), int(index.doc_offsets[d + 1]),
                            dtype=np.uint64) for d in doc_ids]
        self.passage_ids = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint64)
        self.pids = self.passage_ids
        self.N = len(self.passage_ids)
        self.lengths = np.asarray(index.lengths[self.passage_ids], dtype=np.uint32)
        # Identical integer-to-float64 mean to bm25s's np.array(doc lengths).mean().
        self.avgdl = np.asarray(self.lengths, dtype=np.int64).mean() if self.N else np.float64(0)
        self.scope_identity = hashlib.sha256(json.dumps(
            {"index_identity": index.manifest.get("identity", index.manifest.get("configuration", {})),
             "format_version": FORMAT_VERSION,
             "documents": [[d, index.docs[d]["sha256"]] for d in doc_ids]},
            sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        self.df: dict[int, int] = {}
        self._term_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 64 << 20
        self.last_timings: dict = {}

    def _term_weights(self, term_id: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._term_cache.get(term_id)
        if cached is not None:
            self._term_cache.move_to_end(term_id)
            return cached
        first, last = map(int, self.index.term_indptr[term_id:term_id + 2])
        selected_positions, selected_tfs = [], []
        for start in range(first, last, 1_000_000):
            end = min(start + 1_000_000, last)
            global_pids = self.index.posting_pids[start:end]
            positions = np.searchsorted(self.passage_ids, global_pids)
            valid = positions < self.N
            safe = np.minimum(positions, max(0, self.N - 1))
            if self.N:
                valid &= self.passage_ids[safe] == global_pids
            if np.any(valid):
                selected_positions.append(positions[valid].astype(np.int64, copy=False))
                selected_tfs.append(np.asarray(self.index.posting_tfs[start:end][valid], dtype=np.float32))
        positions = np.concatenate(selected_positions) if selected_positions else np.empty(0, dtype=np.int64)
        tfs = np.concatenate(selected_tfs) if selected_tfs else np.empty(0, dtype=np.float32)
        local_df = len(positions)
        self.df[term_id] = local_df
        weights = np.empty(local_df, dtype=np.float32)
        if local_df:
            idf = np.float32(_score_idf_lucene(local_df, self.N))
            lengths = self.lengths[positions]
            # A vector of dl changes NumPy broadcasting/promotions relative to
            # bm25s's scalar-doc-len loop.  Group by dl and execute the *same*
            # original scalar scorer on float32 TF arrays to preserve bits.
            for length in np.unique(lengths):
                mask = lengths == length
                tfc = _score_tfc_lucene(tfs[mask], l_d=int(length),
                                         l_avg=self.avgdl, k1=1.5, b=0.75)
                weights[mask] = idf * tfc
        result = positions, weights
        size = positions.nbytes + weights.nbytes
        if size <= self._cache_limit:
            while self._term_cache and self._cache_bytes + size > self._cache_limit:
                _, (old_positions, old_weights) = self._term_cache.popitem(last=False)
                self._cache_bytes -= old_positions.nbytes + old_weights.nbytes
            self._term_cache[term_id] = result
            self._cache_bytes += size
        return result

    def scores(self, query: str) -> np.ndarray:
        """Float32 complete score vector in passage_ids order; preserves duplicates."""
        started = time.monotonic()
        terms = self.index.query_terms(query)
        tokenize_end = time.monotonic()
        result = np.zeros(self.N, dtype=np.float32)
        if self.N:
            for term in terms:  # No deduplication/reordering of query terms.
                term_id = self.index.term_id(term)
                if term_id is not None:
                    positions, weights = self._term_weights(term_id)
                    np.add.at(result, positions, weights)
        self.last_timings = {"query_tokenize_seconds": tokenize_end - started,
                             "score_seconds": time.monotonic() - tokenize_end,
                             "query_terms": terms, "term_cache_bytes": self._cache_bytes}
        return result

    all_scores = scores

    def search(self, query: str, top_k: int = 1000, include_text: bool = False) -> dict:
        started = time.monotonic()
        scores = self.scores(query)
        scored = time.monotonic()
        local_order = stable_top_k(scores, top_k)
        selected = time.monotonic()
        items = []
        # Source bodies exist once per doc in this temporary retrieval cache.
        # Every body is fully SHA-checked on this search invocation, including
        # multiple hits from a single source; no stale persistent text cache.
        source_texts = {} if include_text else None
        for rank, local_pid in enumerate(local_order, 1):
            pid = int(self.passage_ids[local_pid])
            item = self.index.passage(pid)
            item.update(rank=rank, local_pid=int(local_pid), score=float(scores[local_pid]),
                        bm25_score=float(scores[local_pid]))
            if include_text:
                doc_id = item["doc_id"]
                if doc_id not in source_texts:
                    source_texts[doc_id] = self.index._source_text(doc_id)
                item["text"] = source_texts[doc_id][item["char_start"]:item["char_end"]]
            items.append(item)
        return {"query": query, "scope_identity": self.scope_identity,
                "scope_documents": len(self.doc_ids), "total_passages": self.N,
                "avgdl": float(self.avgdl), "top_k": len(items),
                "query_terms": self.last_timings["query_terms"], "items": items,
                "timings": {**self.last_timings, "top_k_seconds": selected - scored,
                            "materialize_seconds": time.monotonic() - selected,
                            "total_seconds": time.monotonic() - started}}


def _write_result(value: dict, destination: str | None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination:
        Path(destination).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("info", "search", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--index", required=True)
        sub.add_argument("--corpus-root", help="relocated immutable source corpus")
        sub.add_argument("--output")
        if command == "search":
            sub.add_argument("--documents", "--documents-file", required=True,
                             help="JSON list of relative source paths or integer doc IDs")
            sub.add_argument("--query", required=True)
            sub.add_argument("--top-k", type=int, default=1000)
            sub.add_argument("--include-text", action="store_true")
        if command == "verify":
            sub.add_argument("--sources", action="store_true", help="also SHA-check every source")
    args = parser.parse_args(argv)
    started = time.monotonic()
    index = PassageIndex(args.index, corpus_root=args.corpus_root)
    loaded = time.monotonic()
    if args.command == "info":
        result = index.info()
    elif args.command == "verify":
        result = index.verify(sources=args.sources)
    else:
        documents = json.loads(Path(args.documents).read_text("utf-8"))
        if isinstance(documents, dict):
            documents = documents.get("documents", documents.get("paths"))
        if not isinstance(documents, list):
            raise ValueError("documents file must contain a JSON list")
        scope = index.scope(documents)
        scoped = time.monotonic()
        result = scope.search(args.query, top_k=args.top_k, include_text=args.include_text)
        result["timings"]["scope_prepare_seconds"] = scoped - loaded
    result.setdefault("timings", {})["load_seconds"] = loaded - started
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(2)
