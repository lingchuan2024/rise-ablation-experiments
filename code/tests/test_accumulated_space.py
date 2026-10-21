"""Offline contract checks for the v11 accumulated-pool manager."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.accumulated_space import AccumulatedSpaceManager
from test_reranker_tool import FakeFastTokenizer, structured


class Backend:
    def __init__(self, revision="a" * 40, fail=False, score_fn=None):
        self.identity = {"kind": "test", "model": "test/model", "requested_revision": revision,
                         "resolved_revision": revision, "template": "test", "score_kind": "test"}
        self.calls, self.tokenizer_calls, self.last_usage = [], 0, {}
        self.fail, self.score_fn = fail, score_fn

    def get_tokenizer(self):
        self.tokenizer_calls += 1
        return FakeFastTokenizer()

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        self.last_usage = {"forward_calls": 1, "completed_forward_calls": int(not self.fail),
                           "input_tokens": sum(len(FakeFastTokenizer.encode(text)) for text in texts)}
        if self.fail:
            raise RuntimeError("intentional neural failure")
        if self.score_fn is not None:
            return self.score_fn(query, texts)
        return [float(text.count(query)) for text in texts]

    def restore_identity(self, identity):
        if identity != self.identity:
            raise ValueError("identity mismatch")


class AccumulatedSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool, self.views = self.root / "sources", self.root / "views"
        self.pool.mkdir()

    def write(self, rel, text):
        path = self.pool / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return rel

    def manager(self, backend=None, **kwargs):
        return AccumulatedSpaceManager(self.pool, self.views, backend or Backend(), **kwargs)

    def test_union_is_lazy_and_duplicate_search_changes_no_version_or_cache(self):
        a, b = self.write("a.txt", "actor debut alpha"), self.write("b.txt", "novel author beta")
        backend = Backend()
        manager = self.manager(backend)
        self.assertEqual(manager.register_search(["first", "second"], [[(a, 1)], [(a, 2), (b, 1)]]),
                         ["S001.q1", "S001.q2"])
        self.assertEqual(manager.space_version, 1)
        self.assertEqual(manager.summary["pool_documents"], 2)
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(backend.calls, [])
        first = manager.prepare("actor debut")
        manager.register_search(["third"], [[(a, 10), (b, 1)]])
        second = manager.prepare("actor debut")
        self.assertEqual(second["view_id"], first["view_id"])
        self.assertTrue(second["ranking_cache_hit"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(manager.space_version, 1)
        self.assertEqual(manager.summary["search_sets"], 3)

    def test_goal_not_last_search_and_goal_switch_then_return_reuses_cache(self):
        a = self.write("a.txt", structured("actor debut alpha"))
        b = self.write("b.txt", structured("director university beta"))
        backend, manager = Backend(), None
        manager = self.manager(backend)
        manager.register_search(["irrelevant final search"], [[(a, 1), (b, 0)]])
        first = manager.prepare("actor debut")
        second = manager.prepare("director university")
        third = manager.prepare("actor debut")
        self.assertEqual([call[0] for call in backend.calls], ["actor debut", "director university"])
        self.assertNotEqual(first["view_id"], second["view_id"])
        self.assertEqual(first["view_id"], third["view_id"])
        self.assertTrue(third["cache_hit"])
        with self.assertRaisesRegex(ValueError, "goal_query"):
            manager.prepare("")

    def test_real_1000_candidate_cut_and_100_file_space_beyond_preview(self):
        # Two searches are needed: each still imports at most 1000 documents.
        results = []
        for i in range(1015):
            rel = self.write(f"docs/{i:04d}.txt", structured(f"calibration observatory evidence{i}"))
            results.append((rel, float(1015 - i)))
        backend = Backend(score_fn=lambda q, texts: [float(i) for i in range(len(texts))])
        manager = self.manager(backend)
        manager.register_search(["first", "second"], [results[:1000], results[1000:]])
        result = manager.prepare("calibration observatory")
        self.assertEqual(result["pool_documents"], 1015)
        self.assertEqual(result["total_passages"], 1015)
        self.assertEqual(result["bm25_candidates"], 1000)
        self.assertEqual(result["retained_passages"], 100)
        self.assertEqual(len(result["items"]), 10)
        self.assertEqual(result["next_offset"], 10)
        self.assertEqual(len(backend.calls[0][1]), 1000)
        actual = list(manager.view_root(result["view_id"]).iterdir())
        self.assertEqual(len(actual), 100)
        self.assertTrue(all(path.suffix == ".txt" for path in actual))
        self.assertEqual(len(list(self.pool.rglob("*.txt"))), 1015)
        self.assertEqual(len(list(self.pool.rglob("*.json"))), 0)
        retained = manager.get_view(result["view_id"])["items"]
        self.assertEqual(retained[-1]["reranker_rank"], 100)
        self.assertEqual(retained[0]["bm25_rank"], 1000)
        with self.assertRaisesRegex(ValueError, "not retained"):
            manager.get_passage(result["view_id"], "nonexistent")

    def test_expand_k_reuses_complete_ranking_and_snippet_sources_are_traceable(self):
        tokens = [f"word{i}" for i in range(900)]
        rel = self.write("long.txt", structured(" ".join(tokens)))
        backend = Backend()
        manager = self.manager(backend, candidates=20, top_k=2)
        manager.register_search(["search words"], [[(rel, 1)]])
        first, second = manager.prepare("word500"), manager.prepare("word500", top_k=5)
        self.assertNotEqual(first["view_id"], second["view_id"])
        self.assertTrue(second["ranking_cache_hit"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(list(manager.view_root(first["view_id"]).iterdir())), 2)
        self.assertEqual(len(list(manager.view_root(second["view_id"]).iterdir())), 5)
        for item in manager.get_view(second["view_id"])["items"]:
            text, source = manager.source_text(item["source_id"])
            self.assertEqual(text[item["char_start"]:item["char_end"]], item["text"])
            self.assertEqual(source["file_path"], rel)
            self.assertEqual(manager.source_id_for_path(rel), source["source_id"])
            self.assertEqual((manager.view_root(second["view_id"]) / item["file_path"]).read_text(), item["text"])
            self.assertLessEqual(item["token_count"], 128)
            self.assertEqual(item["read_args"]["view_id"], second["view_id"])
        with self.assertRaisesRegex(ValueError, "exceed"):
            manager.prepare("word500", top_k=21)

    def test_overlap_is_32_tokens_and_stable_passage_ids_across_goals(self):
        rel = self.write("long.txt", " ".join(f"term{i}" for i in range(260)))
        manager = self.manager(candidates=10, top_k=10)
        manager.register_search(["any"], [[(rel, 1)]])
        first, second = manager.prepare("term120"), manager.prepare("term121")
        ordered = sorted(manager.get_view(first["view_id"])["items"], key=lambda p: p["char_start"])
        self.assertEqual([p["token_count"] for p in ordered], [128, 128, 68])
        for left, right in zip(ordered, ordered[1:]):
            self.assertEqual(left["text"].split()[-32:], right["text"].split()[:32])
        self.assertEqual({p["passage_id"] for p in manager.get_view(first["view_id"])["items"]},
                         {p["passage_id"] for p in manager.get_view(second["view_id"])["items"]})

    def test_pool_addition_invalidates_views_and_receipts_survive(self):
        a, b = self.write("a.txt", "alpha"), self.write("b.txt", "beta")
        backend, manager = Backend(), None
        manager = self.manager(backend)
        manager.register_search(["one"], [[(a, 1)]])
        old = manager.prepare("alpha")
        manager.record_receipt({"kind": "read", "view_id": old["view_id"]})
        manager.register_search(["two"], [[(b, 1)]])
        new = manager.prepare("alpha")
        self.assertEqual(manager.space_version, 2)
        self.assertNotEqual(new["view_id"], old["view_id"])
        self.assertFalse(new["cache_hit"])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(manager.summary["receipts"], 1)
        with self.assertRaisesRegex(ValueError, "outdated"):
            manager.get_view(old["view_id"])

    def test_source_mutation_fails_until_explicitly_registered_then_changes_version(self):
        rel = self.write("a.txt", "alpha evidence")
        backend = Backend()
        manager = self.manager(backend)
        manager.register_search(["one"], [[(rel, 1)]])
        old = manager.prepare("alpha")
        self.write(rel, "beta changed evidence")
        with self.assertRaisesRegex(ValueError, "source changed"):
            manager.prepare("alpha")
        with self.assertRaisesRegex(ValueError, "source changed"):
            manager.get_passage(old["view_id"], old["items"][0]["passage_id"])
        self.assertEqual(len(backend.calls), 1)
        manager.register_search(["two"], [[(rel, 1)]])
        fresh = manager.prepare("beta")
        self.assertEqual(manager.space_version, 2)
        self.assertIn("beta", fresh["items"][0]["text"])

    def test_neural_failure_or_bad_scores_never_materialize_bm25_fallback(self):
        rel = self.write("a.txt", "alpha evidence")
        for backend in (Backend(fail=True), Backend(score_fn=lambda q, t: []),
                        Backend(score_fn=lambda q, t: [float("nan")] * len(t))):
            manager = self.manager(backend)
            manager.register_search(["one"], [[(rel, 1)]])
            with self.assertRaises((ValueError, RuntimeError)):
                manager.prepare("alpha")
            self.assertEqual(manager.summary["materialized_views"], 0)
            self.assertEqual(manager.summary["ranked_goals"], 0)
            self.assertEqual(manager.summary["errors"], 1)
            self.assertEqual(list(self.views.iterdir()), [])

    def test_checkpoint_restores_ranking_receipts_and_deleted_views_without_neural_forward(self):
        rel = self.write("a.txt", " ".join(f"alpha{i}" for i in range(300)))
        manager = self.manager(candidates=20, top_k=2)
        manager.register_search(["original search"], [[(rel, 1)]])
        original = manager.prepare("alpha150")
        manager.record_receipt({"kind": "read", "passage_id": original["items"][0]["passage_id"]})
        state = json.loads(json.dumps(manager.export_state()))
        shutil.rmtree(self.views)
        backend = Backend()
        restored = self.manager(backend, candidates=20, top_k=2)
        restored.restore_state(state)
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(restored.summary["receipts"], 1)
        self.assertEqual(len(list(restored.view_root(original["view_id"]).iterdir())), 2)
        result = restored.prepare("alpha150", top_k=3)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(result["retained_passages"], 3)
        self.assertEqual(backend.tokenizer_calls, 0, "restored cached goals need no tokenizer loading")

    def test_checkpoint_rejects_model_config_source_and_passage_tampering(self):
        rel = self.write("a.txt", "alpha evidence")
        manager = self.manager()
        manager.register_search(["one"], [[(rel, 1)]])
        manager.prepare("alpha")
        state = manager.export_state()
        with self.assertRaisesRegex(ValueError, "model identity"):
            self.manager(Backend("b" * 40)).restore_state(state)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.manager(candidates=100, top_k=100).restore_state(state)
        changed = copy.deepcopy(state)
        ranking = next(iter(changed["rankings"].values()))
        ranking["items"][0]["text"] = "fabricated evidence"
        with self.assertRaisesRegex(ValueError, "checkpoint passage"):
            self.manager().restore_state(changed)
        changed = copy.deepcopy(state)
        ranking = next(iter(changed["rankings"].values()))
        ranking["model_identity"]["resolved_revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "ranking"):
            self.manager().restore_state(changed)
        self.write(rel, "changed source")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.manager().restore_state(state)

    def test_search_registration_is_atomic_and_rejects_unsafe_sources(self):
        rel = self.write("a.txt", "alpha")
        manager = self.manager()
        for bad in ("../outside.txt", str(self.pool / rel), "missing.txt"):
            with self.assertRaises(ValueError):
                manager.register_search(["good", "bad"], [[(rel, 1)], [(bad, 1)]])
            self.assertEqual(manager.space_version, 0)
            self.assertEqual(manager.summary["pool_documents"], 0)
            self.assertEqual(manager.summary["search_sets"], 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            manager.register_search(["one"], [[(rel, float("nan"))]])

    def test_materialized_symlink_is_rejected_and_modified_snippet_repaired(self):
        rel = self.write("a.txt", "alpha evidence")
        manager = self.manager()
        manager.register_search(["one"], [[(rel, 1)]])
        result = manager.prepare("alpha")
        item = result["items"][0]
        path = manager.view_root(result["view_id"]) / item["file_path"]
        path.chmod(0o600)
        path.write_text("corruption")
        manager.prepare("alpha")
        self.assertEqual(path.read_text(), item["text"])
        path.unlink()
        path.symlink_to(self.pool / rel)
        with self.assertRaisesRegex(ValueError, "symlink"):
            manager.prepare("alpha")

    def test_extra_files_and_directories_are_rejected_on_reuse_and_restore(self):
        rel = self.write("a.txt", "alpha evidence")
        manager = self.manager()
        manager.register_search(["one"], [[(rel, 1)]])
        result = manager.prepare("alpha")
        state = manager.export_state()
        directory = manager.view_root(result["view_id"])
        unexpected = directory / "full-original.txt"
        unexpected.write_text("unretained corpus data")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            manager.prepare("alpha")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            self.manager().restore_state(state)
        unexpected.unlink()
        (directory / "unretained").mkdir()
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            manager.prepare("alpha")

    def test_coverage_merges_overlap_and_preparation_phases_are_measured(self):
        rel = self.write("a.txt", " ".join(f"alpha{i}" for i in range(260)))
        manager = self.manager(candidates=10, top_k=10)
        manager.register_search(["one"], [[(rel, 1)]])
        first = manager.prepare("alpha120")
        items = manager.get_view(first["view_id"])["items"]
        span = max(item["char_end"] for item in items) - min(item["char_start"] for item in items)
        self.assertEqual(first["retained_parent_documents"], 1)
        self.assertEqual(first["retained_unique_characters"], span)
        self.assertLess(span, sum(item["char_end"] - item["char_start"] for item in items))
        manager.prepare("alpha121")
        self.assertEqual(manager.summary["chunk_cache_misses"], 1)
        self.assertEqual(manager.summary["chunk_cache_hits"], 1)
        self.assertGreater(manager.summary["preparation_seconds"], manager.summary["scoring_seconds"])
        for metric in ("chunking_seconds", "bm25_seconds", "source_validation_seconds", "materialization_seconds"):
            self.assertGreater(manager.summary[metric], 0)


if __name__ == "__main__":
    unittest.main()
