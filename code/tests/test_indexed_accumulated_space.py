"""Offline v12 adapter contracts; execute this suite only on the server.

Fixtures contain genuine NumPy raw-TF postings and use the same inexpensive
offset tokenizer as the frozen-manager reference. No neural weights, remote
model API or question evaluation are used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import bm25s
import numpy as np
import Stemmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.accumulated_space import AccumulatedSpaceManager
from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.passage_index import FORMAT_VERSION, PASSAGE_DTYPE, PassageIndex
from rise.reranker_tool import RerankerTool, chunking_identity
from test_reranker_tool import FakeFastTokenizer, structured

PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


class CheapBackend:
    def __init__(self, score_fn=None, revision=PIN):
        self.identity = {"kind": "test", "model": "test/fake-reranker",
                         "requested_revision": revision, "resolved_revision": revision,
                         "template": "test", "score_kind": "test"}
        self.calls, self.tokenizer_calls, self.last_usage = [], 0, {}
        self.score_fn = score_fn or (lambda query, texts: [float(text.count(query)) for text in texts])

    def get_tokenizer(self):
        self.tokenizer_calls += 1
        return FakeFastTokenizer()

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        self.last_usage = {"forward_calls": 1, "completed_forward_calls": 1,
                           "input_tokens": sum(len(FakeFastTokenizer.encode(text)) for text in texts)}
        return self.score_fn(query, texts)

    def restore_identity(self, identity):
        if identity != self.identity:
            raise ValueError("fake backend identity mismatch")


def build_raw_index(directory: Path, corpus: Path, documents: dict[str, str]) -> PassageIndex:
    """Build a small true raw-postings fixture without invoking the builder."""
    directory.mkdir(parents=True)
    corpus.mkdir(parents=True)
    chunker = RerankerTool.__new__(RerankerTool)
    chunker.tokenizer = FakeFastTokenizer()
    docs, rows, texts, offsets = [], [], [], [0]
    for did, rel in enumerate(sorted(documents)):
        source = documents[rel]
        path = corpus / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = source.encode("utf-8")
        path.write_bytes(data)
        metadata = {"file_path": rel, "sha256": sha(data), "search_rank": 1, "search_score": 0.0}
        chunks = chunker._chunks([(metadata, source)])
        docs.append({"doc_id": did, "path": rel, "sha256": sha(data),
                     "bytes": len(data), "chars": len(source), "passage_count": len(chunks)})
        for item in chunks:
            rows.append((did, item["char_start"], item["char_end"], item["token_count"],
                         item["overlap_tokens"], item["read_args"]["offset"], item["read_args"]["limit"]))
            texts.append(item["text"])
        offsets.append(len(rows))
    words = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"),
                           return_ids=False, show_progress=False)
    postings = defaultdict(list)
    for pid, terms in enumerate(words):
        for term, tf in Counter(terms).items():
            postings[term].append((pid, tf))
    vocab = sorted(postings)
    indptr, pids, tfs = [0], [], []
    for term in vocab:
        for pid, tf in postings[term]:
            pids.append(pid)
            tfs.append(tf)
        indptr.append(len(pids))
    arrays = {"passages": np.asarray(rows, dtype=PASSAGE_DTYPE),
              "doc_offsets": np.asarray(offsets, dtype="<u8"),
              "lengths": np.asarray([len(tokens) for tokens in words], dtype="<u4"),
              "term_indptr": np.asarray(indptr, dtype="<u8"),
              "posting_pids": np.asarray(pids, dtype="<u8"),
              "posting_tfs": np.asarray(tfs, dtype="<u4")}
    for name, array in arrays.items():
        np.save(directory / (name + ".npy"), array)
    (directory / "docs.jsonl").write_text("".join(json.dumps(doc) + "\n" for doc in docs), "utf-8")
    (directory / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), "utf-8")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
             for path in directory.iterdir() if path.is_file()}
    config = {"index_format_version": FORMAT_VERSION, "chunking": chunking_identity(),
              "tokenizer": {"revision": PIN, "files": {}},
              "fixture": "same-FakeFastTokenizer-as-frozen-reference"}
    config["sha256"] = json_sha(config)
    manifest = {"status": "ready", "format_version": FORMAT_VERSION,
                "corpus_root": str(corpus), "document_count": len(docs),
                "passage_count": len(rows), "posting_count": len(pids),
                "vocabulary_count": len(vocab), "configuration": config,
                "corpus_content_sha256": json_sha([(d["path"], d["sha256"]) for d in docs]),
                "bm25_config": {"version": bm25s.__version__, "stopwords": "en", "stemmer": "english",
                                "method": "lucene", "k1": 1.5, "b": .75, "dtype": "float32"},
                "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return PassageIndex(directory)


class IndexedAccumulatedSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool, self.corpus = self.root / "pool", self.root / "corpus"
        self.pool.mkdir()
        self.documents = {
            "docs/a.txt": structured("needle river village evidence " * 160),
            "docs/b.txt": structured("actor biography theatre university " * 130),
            "docs/c.txt": structured("short novel café e\u0301 中文 😀"),
        }
        self.index = build_raw_index(self.root / "index", self.corpus, self.documents)
        self.mapping = {rel: rel for rel in self.documents}
        self.copy_sources(self.documents, self.mapping)

    def copy_sources(self, documents, mapping):
        for runtime, corpus_rel in mapping.items():
            path = self.pool / runtime
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(documents[corpus_rel].encode("utf-8"))

    def manager(self, backend=None, *, index=None, mapping=None, views="indexed-views", **kwargs):
        return IndexedAccumulatedSpaceManager(
            self.pool, self.root / views, backend or CheapBackend(),
            passage_index=index or self.index,
            runtime_to_corpus=self.mapping if mapping is None else mapping, **kwargs)

    def frozen_manager(self, backend=None, **kwargs):
        return AccumulatedSpaceManager(self.pool, self.root / "frozen-views", backend or CheapBackend(),
                                       tokenizer=FakeFastTokenizer(), **kwargs)

    @staticmethod
    def candidate_items(manager):
        rankings = manager.export_state()["rankings"]
        ranking = next(iter(rankings.values()))
        return sorted(ranking["items"], key=lambda item: item["bm25_rank"])

    def assert_items_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        fields = ("passage_id", "source_id", "source_file_path", "source_sha256",
                  "char_start", "char_end", "token_count", "overlap_tokens", "text",
                  "source_read_args", "bm25_rank", "reranker_rank", "reranker_score")
        for a, b in zip(left, right):
            for name in fields:
                self.assertEqual(a[name], b[name], name)
            self.assertEqual(np.float32(a["bm25_score"]).view(np.uint32),
                             np.float32(b["bm25_score"]).view(np.uint32))

    def test_real_raw_index_matches_old_candidates_scores_and_view(self):
        for query in ("needle river", "actor biography", "zzzxqneverpresent"):
            with self.subTest(query=query):
                old = self.frozen_manager()
                new = self.manager(views="view-" + sha(query.encode())[:8])
                results = [[("docs/b.txt", 99.0)], [("docs/a.txt", 1.0), ("docs/c.txt", 0.0)]]
                old.register_search(["first", "second"], results)
                new.register_search(["first", "second"], results)
                left, right = old.prepare(query), new.prepare(query)
                self.assert_items_equal(self.candidate_items(old), self.candidate_items(new))
                self.assert_items_equal(old.get_view(left["view_id"])["items"], new.get_view(right["view_id"])["items"])
                self.assertEqual(left["total_passages"], right["total_passages"])
                self.assertEqual(left["retained_passages"], right["retained_passages"])

    def test_real_top1000_and_ties_match_old_runtime_order_with_inverse_mapping(self):
        documents = {f"docs/{i:04d}.txt": structured("calibration observatory evidence") for i in range(1007)}
        index = build_raw_index(self.root / "large-index", self.root / "large-corpus", documents)
        mapping = {f"reverse/{1006-i:04d}.txt": f"docs/{i:04d}.txt" for i in range(1007)}
        self.copy_sources(documents, mapping)
        results = [(runtime, float(i)) for i, runtime in enumerate(reversed(sorted(mapping)))]
        old_backend = CheapBackend(score_fn=lambda query, texts: [0.0] * len(texts))
        new_backend = CheapBackend(score_fn=lambda query, texts: [0.0] * len(texts))
        old = self.frozen_manager(old_backend)
        new = self.manager(new_backend, index=index, mapping=mapping, views="large-views")
        old.register_search(["first", "second"], [results[:1000], results[1000:]])
        new.register_search(["first", "second"], [results[:1000], results[1000:]])
        for query in ("calibration observatory", "zzzxqneverpresent"):
            left, right = old.prepare(query), new.prepare(query)
            left_candidates = next(r["items"] for r in old.export_state()["rankings"].values() if r["goal_query"] == query)
            right_candidates = next(r["items"] for r in new.export_state()["rankings"].values() if r["goal_query"] == query)
            self.assert_items_equal(sorted(left_candidates, key=lambda p: p["bm25_rank"]),
                                    sorted(right_candidates, key=lambda p: p["bm25_rank"]))
            self.assertEqual([p["source_file_path"] for p in sorted(right_candidates, key=lambda p: p["bm25_rank"])],
                             sorted(mapping)[:1000])
            self.assertEqual(right["bm25_candidates"], 1000)
            self.assertEqual(right["retained_passages"], 100)
            self.assertEqual(len(new_backend.calls[-1][1]), 1000)
            self.assert_items_equal(old.get_view(left["view_id"])["items"], new.get_view(right["view_id"])["items"])

    def test_search_union_goal_and_k_caches_remain_v11_compatible(self):
        backend = CheapBackend()
        manager = self.manager(backend, candidates=20, top_k=2)
        manager.register_search(["a"], [[("docs/a.txt", 1)]])
        first = manager.prepare("needle")
        manager.register_search(["duplicate"], [[("docs/a.txt", 99)]])
        repeated = manager.prepare("needle")
        self.assertEqual(manager.space_version, 1)
        self.assertEqual(first["view_id"], repeated["view_id"])
        self.assertTrue(repeated["cache_hit"])
        bigger = manager.prepare("needle", top_k=4)
        self.assertTrue(bigger["cache_hit"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(bigger["retained_passages"], 4)
        manager.prepare("river")
        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(manager.prepare("needle")["cache_hit"])
        manager.register_search(["b"], [[("docs/b.txt", 1)]])
        self.assertEqual(manager.space_version, 2)
        refreshed = manager.prepare("needle")
        self.assertFalse(refreshed["cache_hit"])
        self.assertEqual(refreshed["pool_documents"], 2)
        self.assertEqual(len(backend.calls), 3)
        with self.assertRaises(ValueError):
            manager.get_view(first["view_id"])
        self.assertEqual(backend.tokenizer_calls, 0)

    def test_prepare_reads_only_selected_source_and_never_calls_old_chunker(self):
        backend = CheapBackend()
        manager = self.manager(backend, candidates=1, top_k=1)
        manager.register_search(["both"], [[("docs/a.txt", 1), ("docs/b.txt", 2)]])
        counts = Counter()
        original_open = Path.open
        tracked = {self.pool / rel for rel in self.documents} | {self.corpus / rel for rel in self.documents}

        def counted_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path in tracked and "r" in mode:
                counts[path] += 1
            return original_open(path, *args, **kwargs)

        with patch.object(manager._chunker, "_chunks", side_effect=AssertionError("full pool must never be recut")), \
                patch.object(Path, "open", new=counted_open):
            first = manager.prepare("needle")
            self.assertEqual(counts[self.pool / "docs/b.txt"] + counts[self.corpus / "docs/b.txt"], 0)
            self.assertLessEqual(counts[self.pool / "docs/a.txt"] + counts[self.corpus / "docs/a.txt"], 1)
            self.assertTrue(first["items"])
            counts.clear()
            repeated = manager.prepare("needle")
            self.assertTrue(repeated["cache_hit"])
            self.assertEqual(sum(counts.values()), 0)
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(len(backend.calls), 1)

    def test_stat_change_rechecks_sha_and_rejects_changed_source(self):
        backend = CheapBackend()
        manager = self.manager(backend)
        manager.register_search(["a"], [[("docs/a.txt", 1)]])
        first = manager.prepare("needle")
        path = self.pool / "docs/a.txt"
        path.write_bytes(path.read_bytes() + b" changed")
        with self.assertRaises(ValueError):
            manager.prepare("needle")
        with self.assertRaises(ValueError):
            manager.get_passage(first["view_id"], first["items"][0]["passage_id"])
        with self.assertRaises(ValueError):
            manager.register_search(["changed"], [[("docs/a.txt", 1)]])
        self.assertEqual(len(backend.calls), 1)

    def test_missing_index_source_and_runtime_mapping_reject_atomically(self):
        rel = "docs/not-indexed.txt"
        (self.pool / rel).write_text(structured("not indexed"), "utf-8")
        for mapping, results in (({**self.mapping, rel: rel}, [("docs/a.txt", 1), (rel, 0)]),
                                 ({"docs/a.txt": "docs/a.txt"}, [("docs/a.txt", 1), ("docs/b.txt", 0)])):
            with self.subTest(mapping=mapping):
                backend = CheapBackend()
                manager = self.manager(backend, mapping=mapping)
                with self.assertRaises(ValueError):
                    manager.register_search(["invalid"], [results])
                self.assertEqual(manager.summary["pool_documents"], 0)
                self.assertEqual(manager.space_version, 0)
                self.assertEqual(backend.calls, [])

    def test_checkpoint_restore_reuses_ranking_and_requires_same_generation(self):
        manager = self.manager(candidates=20, top_k=2)
        manager.register_search(["one"], [[("docs/a.txt", 1), ("docs/b.txt", 0)]])
        original = manager.prepare("needle")
        state = json.loads(json.dumps(manager.export_state()))
        backend = CheapBackend()
        restored = self.manager(backend, candidates=20, top_k=2, views="restored-views")
        restored.restore_state(state)
        result = restored.prepare("needle", top_k=4)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(restored.get_view(original["view_id"])["view_id"], original["view_id"])
        clone = self.root / "different-generation"
        shutil.copytree(self.root / "index", clone)
        manifest_path = clone / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["configuration"]["sha256"] = "9" * 64
        manifest["corpus_content_sha256"] = "8" * 64
        manifest_path.write_text(json.dumps(manifest), "utf-8")
        different = self.manager(index=PassageIndex(clone), candidates=20, top_k=2, views="different-views")
        with self.assertRaisesRegex(ValueError, "(?i)(generation|index|identity)"):
            different.restore_state(state)

    def test_index_chunking_and_tokenizer_revision_mismatch_reject(self):
        for label, mutate in (
            ("wrong-window", lambda manifest: manifest["configuration"]["chunking"].update(passage_tokens=64)),
            ("wrong-overlap", lambda manifest: manifest["configuration"]["chunking"].update(overlap_tokens=0)),
            ("wrong-revision", lambda manifest: manifest["configuration"]["tokenizer"].update(revision="f" * 40)),
        ):
            with self.subTest(label=label):
                clone = self.root / label
                shutil.copytree(self.root / "index", clone)
                path = clone / "manifest.json"
                manifest = json.loads(path.read_text())
                mutate(manifest)
                path.write_text(json.dumps(manifest), "utf-8")
                with self.assertRaises(ValueError):
                    self.manager(index=PassageIndex(clone), views=label + "-views")

    def test_two_managers_share_immutable_arrays_but_not_mutable_scope_caches(self):
        backend_a, backend_b = CheapBackend(), CheapBackend()
        a = self.manager(backend_a, views="a-views")
        b = self.manager(backend_b, views="b-views")
        a.register_search(["a"], [[("docs/a.txt", 1)]])
        b.register_search(["b"], [[("docs/b.txt", 1)]])

        def explore(manager, queries):
            return [manager.prepare(query) for query in queries]

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(explore, a, ["needle", "river", "needle"])
            fb = pool.submit(explore, b, ["actor", "university", "actor"])
            results_a, results_b = fa.result(), fb.result()
        for result in results_a:
            self.assertEqual({item["source_file_path"] for item in result["items"]}, {"docs/a.txt"})
        for result in results_b:
            self.assertEqual({item["source_file_path"] for item in result["items"]}, {"docs/b.txt"})
        self.assertEqual(len(backend_a.calls), 2)
        self.assertEqual(len(backend_b.calls), 2)
        self.assertTrue(results_a[-1]["cache_hit"])
        self.assertTrue(results_b[-1]["cache_hit"])
        self.assertEqual(len(self.index._scopes), 0, "Adapter must isolate mutable Scope caches")
        self.assertEqual(backend_a.tokenizer_calls + backend_b.tokenizer_calls, 0)

    def test_prepare_history_records_bounded_stages_and_restored_cache_deltas(self):
        backend = CheapBackend()
        manager = self.manager(backend, candidates=20, top_k=2)
        manager.register_search(["one"], [[("docs/a.txt", 1)]])
        manager.prepare("needle")
        manager.prepare("river")
        manager.prepare("needle", top_k=4)
        summary = manager.summary
        history = summary["index_prepare_history"]
        self.assertEqual([entry["goal_query"] for entry in history], ["needle", "river", "needle"])
        self.assertEqual([entry["cache_hit"] for entry in history], [False, False, True])
        self.assertEqual([entry["status"] for entry in history], ["ok"] * 3)
        self.assertEqual(history[-1]["delta_usage"], summary["last_index_prepare"])
        self.assertEqual(history[0]["delta_usage"]["scoring_calls"], 1)
        self.assertEqual(history[0]["delta_usage"]["candidate_sources_read"], 1)
        self.assertEqual(history[-1]["delta_usage"]["prepare_calls"], 1)
        self.assertEqual(history[-1]["delta_usage"]["ranking_cache_hits"], 1)
        for key in ("scoring_calls", "index_scope_seconds", "index_lookup_seconds",
                    "candidate_sources_read", "candidate_passages_materialized", "chunking_seconds"):
            self.assertEqual(history[-1]["delta_usage"][key], 0, key)
        # Returning summary must not expose the manager's mutable history.
        history[-1]["delta_usage"]["prepare_calls"] = 999
        self.assertEqual(manager.summary["index_prepare_history"][-1]["delta_usage"]["prepare_calls"], 1)
        with self.assertRaises(ValueError):
            manager.prepare("")
        failed = manager.summary["index_prepare_history"][-1]
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["cache_hit"])
        self.assertEqual(failed["delta_usage"]["errors"], 1)
        state = json.loads(json.dumps(manager.export_state()))
        self.assertEqual(state["passage_index"]["index_prepare_history"],
                         manager.summary["index_prepare_history"])
        restored_backend = CheapBackend()
        restored = self.manager(restored_backend, candidates=20, top_k=2, views="history-restored-views")
        restored.restore_state(state)
        self.assertEqual(restored.summary["index_prepare_history"], state["passage_index"]["index_prepare_history"])
        self.assertEqual(restored.summary["last_index_prepare"], state["passage_index"]["index_prepare_history"][-1]["delta_usage"])
        before = restored.summary
        result = restored.prepare("needle")
        after = restored.summary
        entry = after["index_prepare_history"][-1]
        self.assertTrue(result["cache_hit"])
        self.assertEqual(restored_backend.calls, [])
        self.assertEqual(entry["status"], "ok")
        self.assertTrue(entry["cache_hit"])
        for key, delta in entry["delta_usage"].items():
            self.assertEqual(delta, after[key] - before[key], key)
            self.assertGreaterEqual(delta, 0, key)
        for key in ("scoring_calls", "index_scope_seconds", "index_lookup_seconds",
                    "candidate_sources_read", "candidate_passages_materialized", "source_rehashes"):
            self.assertEqual(entry["delta_usage"][key], 0, key)
        self.assertEqual(entry["delta_usage"]["ranking_cache_hits"], 1)
        self.assertEqual(entry["delta_usage"]["prepare_calls"], 1)
        for _ in range(103):
            restored.prepare("needle")
        bounded = restored.summary["index_prepare_history"]
        self.assertEqual(len(bounded), 100)
        self.assertTrue(all(entry["goal_query"] == "needle" and entry["cache_hit"] for entry in bounded))
        self.assertEqual(len(restored.export_state()["passage_index"]["index_prepare_history"]), 100)

    def test_checkpoint_rejects_malformed_prepare_history_schema_and_metrics(self):
        manager = self.manager(candidates=20, top_k=2)
        manager.register_search(["one"], [[("docs/a.txt", 1)]])
        manager.prepare("needle")
        original = manager.export_state()

        def change_entry(field, value):
            return lambda state: state["passage_index"]["index_prepare_history"][0].update({field: value})

        def change_metric(field, value):
            return lambda state: state["passage_index"]["index_prepare_history"][0]["delta_usage"].update({field: value})

        mutations = [
            ("not-list", lambda state: state["passage_index"].update(index_prepare_history={})),
            ("over-limit", lambda state: state["passage_index"].update(
                index_prepare_history=state["passage_index"]["index_prepare_history"] * 101)),
            ("bad-status", change_entry("status", "finished")),
            ("bad-cache-hit", change_entry("cache_hit", "yes")),
            ("bad-goal", change_entry("goal_query", 42)),
            ("bool-version", change_entry("space_version", True)),
            ("negative-version", change_entry("space_version", -1)),
            ("missing-metric", lambda state: state["passage_index"]["index_prepare_history"][0]["delta_usage"].pop("index_lookup_seconds")),
            ("bool-metric", change_metric("prepare_calls", True)),
            ("negative-metric", change_metric("index_lookup_seconds", -1)),
            ("nan-metric", change_metric("index_lookup_seconds", float("nan"))),
            ("infinite-metric", change_metric("index_lookup_seconds", float("inf"))),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                state = copy.deepcopy(original)
                mutate(state)
                backend = CheapBackend()
                restored = self.manager(backend, candidates=20, top_k=2, views="bad-history-" + label)
                with self.assertRaisesRegex(ValueError, "history"):
                    restored.restore_state(state)
                self.assertEqual(backend.calls, [])
                self.assertEqual(restored.summary["pool_documents"], 0)


if __name__ == "__main__":
    unittest.main()
