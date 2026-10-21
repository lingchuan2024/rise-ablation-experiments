"""Offline acceptance checks for v13's independent, adjustable rerank tool.

Use genuine raw-TF NumPy postings and the existing deterministic offset
tokenizer/scorer. No neural model, external API, or online evaluation is used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.accumulated_space import _digest
from rise.passage_index import PassageIndex
from rise.rerank_v13 import RerankService
from rise.tools import make_read_tool
from test_indexed_accumulated_space import CheapBackend, build_raw_index
from test_reranker_tool import FakeFastTokenizer, structured


class RerankV13Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.documents = {
            "docs/a.txt": structured("needle river village evidence " * 160),
            "docs/b.txt": structured("actor biography theatre university " * 130),
            "docs/c.txt": structured("short novel café e\u0301 中文 😀"),
        }
        self.index = self.make_index(self.documents, "default")
        self.backend = CheapBackend()
        self.manager, self.service = self.make_service(self.backend)

    def make_index(self, documents, name):
        index = build_raw_index(self.root / (name + "-index"),
                                self.root / (name + "-corpus"), documents)
        for rel, content in documents.items():
            target = self.pool / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        return index

    def make_service(self, backend=None, *, index=None, views="views", mapping=None):
        chosen = index or self.index
        manager = IndexedAccumulatedSpaceManager(
            self.pool, self.root / views, backend or CheapBackend(),
            passage_index=chosen,
            runtime_to_corpus=mapping or {doc["path"]: doc["path"] for doc in chosen.docs})
        return manager, RerankService(manager)

    def register(self, paths=None, *, service=None):
        paths = list(self.documents) if paths is None else paths
        (service or self.service).register_search(
            ["source retrieval"], [[(rel, float(len(paths) - i)) for i, rel in enumerate(paths)]])

    def call(self, service=None, **args):
        result = json.loads((service or self.service)(args))
        self.assertEqual(result["status"], "ok", result)
        return result

    def error(self, args, *, service=None, pattern=None):
        result = json.loads((service or self.service)(args))
        self.assertEqual(result["status"], "error", result)
        if pattern:
            self.assertRegex(json.dumps(result, ensure_ascii=False), pattern)
        return result

    def assert_item_coordinates(self, item):
        path = item["file_path"]
        original = (self.pool / path).read_bytes().decode("utf-8", errors="replace")
        self.assertEqual(original[item["char_start"]:item["char_end"]], item["text"])
        self.assertEqual(item["token_count"], len(FakeFastTokenizer.encode(item["text"])))
        self.assertGreater(item["token_count"], 0)
        self.assertLessEqual(item["token_count"], 128)
        self.assertGreaterEqual(item["overlap_tokens"], 0)
        self.assertLessEqual(item["overlap_tokens"], 32)
        args = item["read_args"]
        self.assertEqual(args["file_path"], path)
        self.assertIs(type(args["offset"]), int)
        self.assertIs(type(args["limit"]), int)
        self.assertGreaterEqual(args["offset"], 0)
        self.assertGreater(args["limit"], 0)
        self.assertNotIn("view_id", args)
        self.assertNotIn("passage_id", args)
        self.assertGreater(item["bm25_rank"], 0)
        self.assertGreater(item["reranker_rank"], 0)
        self.assertTrue(np.isfinite(item["bm25_score"]))
        self.assertTrue(np.isfinite(item["reranker_score"]))

    def test_defaults_full_pool_complete_fragments_and_readable_coordinates(self):
        self.register()
        result = self.call(query="needle")
        self.assertEqual(result["requested_candidates"], 1000)
        self.assertEqual(result["top_k"], 100)
        self.assertEqual(result["pool_documents"], 3)
        self.assertEqual(result["scope_documents"], 3)
        actual = min(1000, result["total_passages"])
        self.assertEqual(result["bm25_candidates"], actual)
        self.assertEqual(result["scored_passages"], actual)
        self.assertEqual(result["retained_passages"], min(100, actual))
        self.assertEqual(len(result["items"]), min(10, actual))
        self.assertFalse(result["cache_hit"])
        self.assertGreaterEqual(result["preparation_seconds"], 0)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(len(self.backend.calls[0][1]), actual)
        self.assertEqual(self.backend.tokenizer_calls, 0)
        for item in result["items"]:
            self.assert_item_coordinates(item)

        # Short Unicode evidence has an ordinary baseline read entry point.
        short = self.call(query="novel", file_paths=["docs/c.txt"], bm25_candidates=2, limit=2)
        item = short["items"][0]
        self.assertFalse(item.get("read_may_truncate", False))
        output = make_read_tool(self.pool)(item["read_args"])
        lines = [line.split("\t", 1)[1] for line in output.splitlines() if "\t" in line]
        self.assertIn(item["text"].strip(), "\n".join(lines))

    def test_subset_recomputes_bm25_instead_of_filtering_global_ranking(self):
        documents = {
            "local/a.txt": structured("alpha ground river valley"),
            "local/b.txt": structured("beta beta beta valley"),
            "local/c.txt": structured("beta ground river valley"),
            **{f"global/{i:02d}.txt": structured("alpha ground river valley") for i in range(12)},
        }
        index = self.make_index(documents, "local-bm25")
        backend = CheapBackend(score_fn=lambda query, texts: [0.] * len(texts))
        manager, service = self.make_service(backend, index=index, views="local-bm25-views")
        self.register(list(documents), service=service)
        subset = ["local/a.txt", "local/b.txt", "local/c.txt"]
        local = index.scope(subset, cache=False).search("alpha beta", 3)
        global_result = index.scope(list(documents), cache=False).search("alpha beta", len(documents))
        global_filtered = [item["path"] for item in global_result["items"] if item["path"] in subset]
        expected = [item["path"] for item in local["items"]]
        self.assertNotEqual(expected, global_filtered, "Fixture must distinguish local/global BM25")
        result = self.call(service, query="alpha beta", file_paths=list(reversed(subset)),
                           bm25_candidates=3, top_k=3, limit=3)
        self.assertEqual(result["scope_documents"], 3)
        self.assertEqual(result["pool_documents"], len(documents))
        self.assertEqual(result["total_passages"], 3)
        self.assertEqual([item["file_path"] for item in result["items"]], expected)
        for item, lexical in zip(result["items"], local["items"]):
            self.assertEqual(np.float32(item["bm25_score"]).view(np.uint32),
                             np.float32(lexical["bm25_score"]).view(np.uint32))
        self.assertEqual(len(backend.calls[0][1]), 3)
        self.assertEqual(len(manager._sources), len(documents))

    def test_model_may_score_more_than_1000_candidates_and_only_requested_m(self):
        documents = {f"large/{i:04d}.txt": structured("calibration observatory evidence")
                     for i in range(1007)}
        index = self.make_index(documents, "large")
        backend = CheapBackend(score_fn=lambda query, texts: [0.] * len(texts))
        manager, service = self.make_service(backend, index=index, views="large-views")
        paths = sorted(documents)
        service.register_search(["first", "second"],
                                [[(rel, 1.) for rel in paths[:1000]], [(rel, 1.) for rel in paths[1000:]]])
        small = self.call(service, query="calibration", bm25_candidates=3, top_k=3, limit=3)
        large = self.call(service, query="calibration", bm25_candidates=1005, top_k=1005, limit=1005)
        self.assertEqual([len(call[1]) for call in backend.calls], [3, 1005])
        self.assertEqual(len(small["items"]), 3)
        self.assertEqual(large["requested_candidates"], 1005)
        self.assertEqual(large["scored_passages"], 1005)
        self.assertEqual(large["retained_passages"], 1005)
        self.assertEqual(len(large["items"]), 1005)
        self.assertEqual([item["file_path"] for item in large["items"]], paths[:1005])
        exhausted = self.call(service, query="calibration", bm25_candidates=1010, top_k=1010, limit=1010)
        self.assertEqual(exhausted["requested_candidates"], 1010)
        self.assertEqual(exhausted["scored_passages"], 1007)
        self.assertEqual(exhausted["retained_passages"], 1007)
        self.assertEqual(len(manager._sources), 1007)

    def test_paging_and_changing_k_never_rescores_and_default_k_is_first_value(self):
        self.register()
        first = self.call(query="needle", bm25_candidates=9, top_k=4, limit=2)
        self.assertEqual(first["next_offset"], 2)
        second = self.call(result_id=first["result_id"], offset=2, limit=2)
        self.assertEqual(second["top_k"], 4)
        self.assertIsNone(second["next_offset"])
        bigger = self.call(result_id=first["result_id"], top_k=8, offset=4, limit=3)
        self.assertEqual(bigger["top_k"], 8)
        self.assertEqual(bigger["next_offset"], 7)
        default_again = self.call(result_id=first["result_id"], offset=0, limit=9)
        self.assertEqual(default_again["top_k"], 4)
        complete = self.call(query="needle", bm25_candidates=9, top_k=9, limit=9)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertTrue(all(result["cache_hit"] for result in (second, bigger, default_again, complete)))
        self.assertEqual(first["items"] + second["items"], complete["items"][:4])
        self.assertEqual(bigger["items"], complete["items"][4:7])
        beyond = self.call(result_id=first["result_id"], offset=100, limit=2)
        self.assertEqual(beyond["items"], [])
        self.assertIsNone(beyond["next_offset"])

    def test_cache_identity_covers_query_scope_and_m_with_canonical_scope_order(self):
        self.register()
        base = dict(query="needle", bm25_candidates=5, top_k=3, limit=2)
        first = self.call(**base)
        repeated = self.call(**{**base, "limit": 3, "top_k": 5})
        self.assertTrue(repeated["cache_hit"])
        self.call(**{**base, "query": "river"})
        self.call(**{**base, "bm25_candidates": 6})
        subset = self.call(**base, file_paths=["docs/b.txt", "docs/a.txt"])
        reverse = self.call(**base, file_paths=["docs/a.txt", "docs/b.txt"])
        self.assertTrue(reverse["cache_hit"])
        self.assertEqual(subset["items"], reverse["items"])
        repeated_paths = self.call(**base, file_paths=["docs/b.txt", "docs/a.txt", "docs/b.txt"])
        self.assertTrue(repeated_paths["cache_hit"])
        self.assertEqual(repeated_paths["scope_documents"], 2)
        self.assertEqual(repeated_paths["items"], subset["items"])
        self.assertEqual(len(self.backend.calls), 4)
        self.assertFalse(first["cache_hit"])

    def test_bash_dot_relative_paths_share_canonical_scope_without_parent_traversal(self):
        self.register()
        args = {"query": "needle", "bm25_candidates": 4, "top_k": 4, "limit": 4}
        canonical = self.call(**args, file_paths=["docs/a.txt"])
        shell_paths = self.call(**args, file_paths=["./docs/a.txt", "docs/a.txt"])
        self.assertTrue(shell_paths["cache_hit"])
        self.assertEqual(shell_paths["scope_documents"], 1)
        self.assertEqual(shell_paths["result_id"], canonical["result_id"])
        self.assertEqual(shell_paths["items"], canonical["items"])
        self.assertEqual(len(self.backend.calls), 1)
        self.error({**args, "file_paths": ["./../docs/a.txt"]})
        self.assertEqual(len(self.backend.calls), 1)

    def test_rerank_preserves_all_raw_sources_and_existing_dci_views(self):
        self.register()
        old_view = self.manager.prepare("actor", top_k=2)
        old_state = copy.deepcopy(self.manager.export_state())
        files = {path.relative_to(self.pool): path.read_bytes() for path in self.pool.rglob("*") if path.is_file()}
        with patch.object(self.manager, "prepare", side_effect=AssertionError("rerank must not prepare a DCI view")):
            result = self.call(query="needle", file_paths=["docs/a.txt"], bm25_candidates=2, top_k=1, limit=1)
        self.assertEqual(result["scope_documents"], 1)
        self.assertEqual(self.manager._sources, old_state["sources"])
        self.assertEqual(self.manager._rankings, old_state["rankings"])
        self.assertEqual(self.manager._views, old_state["views"])
        self.assertEqual(self.manager.get_view(old_view["view_id"])["view_id"], old_view["view_id"])
        self.assertEqual(files, {path.relative_to(self.pool): path.read_bytes()
                                 for path in self.pool.rglob("*") if path.is_file()})
        # bash/read can still inspect a source outside both requested scope and K.
        self.assertIn("theatre", make_read_tool(self.pool)({"file_path": "docs/b.txt"}))
        shell = subprocess.run(["sh", "-c", "cat docs/b.txt"], cwd=self.pool,
                               text=True, encoding="utf-8", capture_output=True, check=True)
        self.assertIn("theatre", shell.stdout)

    def test_empty_explicit_scope_is_distinct_from_omitted_scope(self):
        self.register()
        empty = self.call(query="needle", file_paths=[], bm25_candidates=4)
        self.assertEqual(empty["items"], [])
        self.assertEqual(empty["scope_documents"], 0)
        self.assertEqual(empty["total_passages"], 0)
        self.assertEqual(empty["scored_passages"], 0)
        self.assertEqual(self.backend.calls, [])
        full = self.call(query="needle", file_paths=None, bm25_candidates=4)
        omitted = self.call(query="needle", bm25_candidates=4)
        self.assertEqual(full["scope_documents"], 3)
        self.assertTrue(omitted["cache_hit"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_bad_parameters_return_errors_without_neural_calls(self):
        self.register(["docs/a.txt"])
        good = {"query": "needle", "bm25_candidates": 3, "top_k": 2}
        bad_args = [None, [], {}, {"query": ""}, {"query": "  "}, {"query": 1},
                    {**good, "unknown": True}, {**good, "file_paths": "docs/a.txt"},
                    {**good, "file_paths": [""]}, {**good, "file_paths": [True]},
                    {**good, "file_paths": ["docs/b.txt"]},
                    {**good, "file_paths": ["docs/not-found.txt"]},
                    {**good, "file_paths": ["../a.txt"]},
                    {**good, "top_k": 4}, {"result_id": "unknown"}]
        for field in ("bm25_candidates", "top_k", "limit"):
            bad_args.extend({**good, field: value} for value in (True, False, 0, -1, 1.5, "3"))
        bad_args.extend({**good, "offset": value} for value in (True, False, -1, 1.5, "3"))
        for args in bad_args:
            with self.subTest(args=args):
                self.error(args)
        self.assertEqual(self.backend.calls, [])

    def test_invalid_nonfinite_numeric_arguments_leave_strict_json_checkpoint(self):
        self.register()
        args = {"query": "needle", "bm25_candidates": 3, "top_k": 2,
                "offset": 0, "limit": 2}
        for field in ("bm25_candidates", "top_k", "offset", "limit"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    self.error({**args, field: value})
        self.assertEqual(self.backend.calls, [])
        encoded = json.dumps(self.service.export_state(), allow_nan=False)
        json.dumps(self.service.summary, allow_nan=False)
        state = json.loads(encoded)
        self.assertEqual(len(state["history"]), 12)
        self.assertEqual(self.service.summary["tool_errors"], 12)
        backend = CheapBackend()
        _, restored = self.make_service(backend, views="nonfinite-errors-restored-views")
        restored.restore_state(state)
        self.assertEqual(restored.summary["tool_errors"], 12)
        self.assertEqual(restored.summary["call_history"], state["history"])
        self.assertEqual(backend.calls, [])

    def test_new_search_expires_old_result_but_duplicate_search_does_not(self):
        self.register(["docs/a.txt"])
        first = self.call(query="needle", bm25_candidates=3, top_k=2)
        self.register(["docs/a.txt"])
        duplicate = self.call(result_id=first["result_id"])
        self.assertTrue(duplicate["cache_hit"])
        self.assertEqual(len(self.backend.calls), 1)
        self.register(["docs/b.txt"])
        self.error({"result_id": first["result_id"]}, pattern="(?i)(expired|stale|obsolete|过期)")
        fresh = self.call(query="needle", bm25_candidates=3, top_k=2)
        self.assertFalse(fresh["cache_hit"])
        self.assertEqual(fresh["pool_documents"], 2)
        self.assertEqual(len(self.backend.calls), 2)

    def test_source_change_rejects_new_query_cached_query_and_paging(self):
        self.register()
        first = self.call(query="needle", bm25_candidates=3, top_k=2)
        path = self.pool / "docs/a.txt"
        path.write_bytes(path.read_bytes() + " 中文变更".encode("utf-8"))
        self.error({"query": "actor", "bm25_candidates": 3, "top_k": 2})
        self.error({"query": "needle", "bm25_candidates": 3, "top_k": 2})
        self.error({"result_id": first["result_id"]})
        self.assertEqual(len(self.backend.calls), 1)

    def test_unicode_long_line_has_exact_character_bash_recovery(self):
        rel = "unicode/中文 'quoted' $.txt"
        documents = {rel: structured(("中文😀e\u0301长词" * 20 + " ") * 280 + "证据目标")}
        index = self.make_index(documents, "unicode-long")
        manager, service = self.make_service(index=index, views="unicode-long-views")
        self.register([rel], service=service)
        result = self.call(service, query="证据目标", bm25_candidates=8, top_k=8, limit=8)
        for item in result["items"]:
            self.assert_item_coordinates(item)
            self.assertTrue(item["read_may_truncate"])
            self.assertIsInstance(item["bash_args"]["command"], str)
            output = subprocess.run(["sh", "-c", item["bash_args"]["command"]], cwd=self.pool,
                                    text=True, encoding="utf-8", capture_output=True, check=True).stdout
            self.assertIn(item["text"], output)
        self.assertEqual(len(manager._sources), 1)

    def test_checkpoint_restores_all_m_scores_and_paging_without_neural_calls(self):
        self.register()
        first = self.call(query="needle", bm25_candidates=9, top_k=2, limit=1)
        expected = self.call(result_id=first["result_id"], top_k=9, limit=9)
        self.call(query="actor", file_paths=["docs/b.txt"], bm25_candidates=4, top_k=2)
        state = json.loads(json.dumps(self.service.export_state()))
        backend = CheapBackend()
        _, restored = self.make_service(backend, views="restored-views")
        restored.restore_state(state)
        page = self.call(restored, result_id=first["result_id"], top_k=9, limit=9)
        self.assertEqual(page["items"], expected["items"])
        default = self.call(restored, result_id=first["result_id"], limit=9)
        self.assertEqual(default["top_k"], 2)
        repeated = self.call(restored, query="needle", bm25_candidates=9, top_k=9, limit=9)
        subset = self.call(restored, query="actor", file_paths=["docs/b.txt"], bm25_candidates=4, top_k=4)
        self.assertTrue(all(result["cache_hit"] for result in (page, default, repeated, subset)))
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)

    def test_summary_records_requested_knobs_actual_scoring_cost_cache_and_errors(self):
        self.register()
        args = {"query": "needle", "bm25_candidates": 5, "top_k": 2, "limit": 1}
        first = self.call(**args)
        self.call(result_id=first["result_id"], offset=1, limit=1)
        self.call(query="needle", bm25_candidates=5, top_k=4, limit=4)
        self.error({"query": "needle", "limit": 0})
        summary = self.service.summary
        self.assertEqual(summary["rerank_calls"], 4)
        self.assertEqual(summary["calls"], 4)
        self.assertEqual(summary["scoring_calls"], 1)
        self.assertEqual(summary["scored_passages"], 5)
        self.assertEqual(summary["ranking_cache_hits"], 2)
        self.assertEqual(summary["page_calls"], 1)
        self.assertEqual(summary["tool_errors"], 1)
        self.assertEqual(summary["pool_documents"], 3)
        self.assertEqual(summary["cached_rankings"], 1)
        self.assertGreaterEqual(summary["scoring_seconds"], 0)
        self.assertGreaterEqual(summary["index_scope_seconds"], 0)
        self.assertGreaterEqual(summary["index_lookup_seconds"], 0)
        history = summary["call_history"]
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["args"], args)
        self.assertEqual([row["status"] for row in history], ["ok", "ok", "ok", "error"])
        self.assertEqual(history[0]["requested_candidates"], 5)
        self.assertEqual(history[0]["bm25_candidates"], 5)
        self.assertEqual([row["cache_hit"] for row in history[:3]], [False, True, True])
        self.assertEqual(history[1]["scoring_seconds"], 0)
        self.assertEqual(history[2]["scoring_seconds"], 0)
        self.assertTrue(all(row["preparation_seconds"] >= 0 for row in history))
        history[0]["args"]["query"] = "mutated externally"
        self.assertEqual(self.service.summary["call_history"][0]["args"], args)
        backend = CheapBackend()
        _, restored = self.make_service(backend, views="summary-restored-views")
        state = self.service.export_state()
        restored.restore_state(state)
        self.assertEqual(restored.summary["call_history"], state["history"])
        self.assertEqual(restored.summary["scoring_calls"], 1)
        self.assertEqual(restored.summary["scored_passages"], 5)
        self.assertEqual(backend.calls, [])

    def test_checkpoint_rejects_changed_index_and_cached_text_tampering_atomically(self):
        self.register()
        self.call(query="needle", bm25_candidates=5, top_k=2)
        state = self.service.export_state()
        clone = self.root / "other-generation"
        shutil.copytree(self.root / "default-index", clone)
        manifest_path = clone / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["configuration"]["sha256"] = "9" * 64
        manifest["corpus_content_sha256"] = "8" * 64
        manifest_path.write_text(json.dumps(manifest), "utf-8")
        _, incompatible = self.make_service(index=PassageIndex(clone), views="different-generation-views")
        with self.assertRaisesRegex(ValueError, "(?i)(generation|index|identity)"):
            incompatible.restore_state(copy.deepcopy(state))

        # Traverse by content rather than depending on an internal state key.
        damaged = copy.deepcopy(state)
        changed = False

        def corrupt(value):
            nonlocal changed
            if isinstance(value, dict):
                if not changed and "text" in value and "reranker_rank" in value:
                    value["text"] += " forged evidence"
                    changed = True
                    return
                for child in value.values():
                    corrupt(child)
            elif isinstance(value, list):
                for child in value:
                    corrupt(child)

        corrupt(damaged)
        self.assertTrue(changed, "Checkpoint must retain all scored candidate texts")
        backend = CheapBackend()
        manager, restored = self.make_service(backend, views="tampered-views")
        with self.assertRaises(ValueError):
            restored.restore_state(damaged)
        self.assertEqual(manager._sources, {})
        self.assertEqual(backend.calls, [])

    def test_checkpoint_revalidates_source_after_checksum_recomputed_and_rejects_v12(self):
        self.register()
        self.call(query="needle", bm25_candidates=5, top_k=2)
        original = self.service.export_state()
        damaged = copy.deepcopy(original)
        ranking = next(iter(damaged["rankings"].values()))
        ranking["items"][0]["char_end"] -= 1
        ranking["payload_sha256"] = _digest({k: v for k, v in ranking.items() if k != "payload_sha256"})
        backend = CheapBackend()
        _, restored = self.make_service(backend, views="rehashed-tampered-views")
        self.register(["docs/c.txt"], service=restored)
        before = restored.export_state()
        with self.assertRaises(ValueError):
            restored.restore_state(damaged)
        self.assertEqual(restored.export_state(), before)
        self.assertEqual(backend.calls, [])
        with self.assertRaisesRegex(ValueError, "(?i)(version|v13)"):
            restored.restore_state(original["manager"])


if __name__ == "__main__":
    unittest.main()
