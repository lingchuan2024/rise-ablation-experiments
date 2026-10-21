"""Offline contracts for compact v13 model output and durable diagnostics.

Exercise genuine passage/BM25 postings, without model downloads or API calls.
The baseline read source hashes were measured from the frozen v10 baseline.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.rerank_v13 import (
    RERANK_RESPONSE_VERSION, RerankService, compact_rerank_response,
)
from rise.tools import make_read_tool, read_tool_spec
from test_indexed_accumulated_space import CheapBackend, build_raw_index
from test_reranker_tool import FakeFastTokenizer, structured


ITEM_KEYS = {
    "file_path", "text", "char_start", "char_end", "read_args", "token_count",
    "overlap_tokens", "bm25_rank", "bm25_score", "reranker_rank", "reranker_score",
}
FALLBACK_KEYS = {
    "read_may_truncate", "bash_args", "context_char_start", "context_char_end",
}
PRIVATE_KEYS = {
    "backend_usage", "backend_attempt_usage", "model_identity", "chunking_identity",
    "scope_identity", "ranking_identity", "source_id", "source_sha256",
    "source_file_path", "source_read_args", "snippet_file", "passage_id",
    "search_memberships", "sequence_lengths",
}
BASELINE_READ_HASHES = {
    "make_read_tool": "cfab3e641a7bc48ee94f7e5dba7d4915d674e7e412de768b23356bc54b69795f",
    "read_tool_spec": "36da80e960153d2d6d9365474aa4b762cd7354c7bffbaa11f1b5488c678e4f3e",
}


class VerboseCheapBackend(CheapBackend):
    """Simulate the per-candidate diagnostics that inflated real tool output."""

    def score(self, query, texts):
        scores = super().score(query, texts)
        self.last_usage.update({
            "sequence_lengths": [len(FakeFastTokenizer.encode(text)) for text in texts],
            "diagnostic_marker": "PRIVATE-BACKEND-PREPARATION-DETAIL",
        })
        return scores


class RerankCompactResponseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.documents = {
            "docs/unicode.txt": structured("\r\n".join(
                f"needle café e\u0301 中文 😀 evidence row {i}" for i in range(90))),
            "docs/other.txt": structured("actor biography university theatre " * 160),
        }
        self.index = self.make_index(self.documents, "small")
        self.backend = VerboseCheapBackend()
        self.manager, self.service = self.make_service(self.backend)

    def make_index(self, documents, name):
        index = build_raw_index(self.root / (name + "-index"),
                                self.root / (name + "-corpus"), documents)
        for relative, text in documents.items():
            destination = self.pool / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(text.encode("utf-8"))
        return index

    def make_service(self, backend, *, index=None, views="views"):
        index = index or self.index
        manager = IndexedAccumulatedSpaceManager(
            self.pool, self.root / views, backend, passage_index=index,
            runtime_to_corpus={doc["path"]: doc["path"] for doc in index.docs})
        return manager, RerankService(manager)

    def register(self, service=None, paths=None, *, repetitions=1):
        service = service or self.service
        paths = list(self.documents) if paths is None else paths
        for i in range(repetitions):
            service.register_search([f"PRIVATE-SEARCH-PROVENANCE-{i}"],
                                    [[(path, float(len(paths) - j))
                                      for j, path in enumerate(paths)]])

    def call(self, service=None, **arguments):
        result = json.loads((service or self.service)(arguments))
        self.assertEqual(result["status"], "ok", result)
        self.assert_no_private_keys(result)
        return result

    def assert_no_private_keys(self, value):
        if isinstance(value, dict):
            self.assertFalse(PRIVATE_KEYS.intersection(value), value.keys())
            for child in value.values():
                self.assert_no_private_keys(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_no_private_keys(child)

    def diagnostics(self, service=None):
        service = service or self.service
        self.assertEqual(service.diagnostics_path,
                         service.manager.views_root / "rerank_diagnostics.jsonl")
        self.assertTrue(service.diagnostics_path.is_file())
        return [json.loads(line) for line in service.diagnostics_path.read_text("utf-8").splitlines()]

    def test_thousand_candidate_diagnostics_stay_on_disk_with_complete_page_metadata(self):
        # Real 128/32 passages: enough candidates for the requested M=1000.
        body = "\n".join("needle river village evidence " * 16 for _ in range(1600))
        documents = {"large/source.txt": structured(body)}
        index = self.make_index(documents, "large")
        backend = VerboseCheapBackend()
        _, service = self.make_service(backend, index=index, views="large-views")
        self.register(service, list(documents), repetitions=12)
        response = self.call(service, query="needle river", bm25_candidates=1000,
                             top_k=5, limit=1)
        self.assertEqual(response["requested_candidates"], 1000)
        self.assertEqual(response["scored_passages"], 1000)
        self.assertEqual(len(response["items"]), 1)
        self.assertEqual(set(response["items"][0]), ITEM_KEYS)
        public = json.dumps(response, ensure_ascii=False)
        self.assertNotIn("PRIVATE-BACKEND", public)
        self.assertNotIn("PRIVATE-SEARCH", public)
        # Bound the fixed envelope plus one genuine short fragment. The backend
        # preparation vector and twelve source imports cannot consume context.
        self.assertLess(len(public), 3500)
        records = self.diagnostics(service)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["response_version"], RERANK_RESPONSE_VERSION)
        self.assertEqual(len(record["response"]["backend_usage"]["sequence_lengths"]), 1000)
        self.assertEqual(len(record["backend_attempt_usage"]["sequence_lengths"]), 1000)
        full_item = record["response"]["items"][0]
        self.assertEqual(full_item["source_file_path"], "large/source.txt")
        self.assertEqual(len(full_item["search_memberships"]), 12)
        for name in ("passage_id", "source_sha256", "source_id", "source_read_args"):
            self.assertIn(name, full_item)
        self.assertIn("model_identity", record["response"])
        self.assertIn("scope_identity", record["response"])
        self.assertEqual(compact_rerank_response(record["response"]), response)

    def test_two_calls_append_full_diagnostics_but_paging_does_not_repeat_scoring(self):
        self.register(repetitions=3)
        first = self.call(query="needle", bm25_candidates=8, top_k=4, limit=2)
        before = copy.deepcopy(self.service.export_state()["rankings"])
        second = self.call(result_id=first["result_id"], top_k=6, offset=2, limit=3)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["top_k"], 6)
        self.assertEqual(second["next_offset"], 5)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.service.export_state()["rankings"], before)
        records = self.diagnostics()
        self.assertEqual([row["call_index"] for row in records], [1, 2])
        self.assertEqual(records[1]["args"], {
            "result_id": first["result_id"], "top_k": 6, "offset": 2, "limit": 3,
        })
        self.assertEqual(records[1]["backend_attempt_usage"], {})
        self.assertEqual(records[1]["usage_delta"]["scoring_calls"], 0)
        self.assertEqual(records[1]["usage_delta"]["scored_passages"], 0)
        self.assertEqual(records[1]["response"]["scoring_seconds"], 0)
        self.assertEqual(compact_rerank_response(records[1]["response"]), second)

    def test_projection_preserves_evidence_and_does_not_mutate_rich_payload(self):
        self.register()
        response = self.call(query="needle", bm25_candidates=6, top_k=4, limit=3)
        rich = self.diagnostics()[0]["response"]
        snapshot = copy.deepcopy(rich)
        projected = compact_rerank_response(rich)
        self.assertEqual(projected, response)
        self.assertEqual(rich, snapshot)
        for compact, original in zip(projected["items"], rich["items"]):
            for key in ITEM_KEYS - {"file_path", "read_args"}:
                self.assertEqual(compact[key], original[key])
            self.assertEqual(compact["file_path"], original["source_file_path"])
            self.assertEqual(compact["read_args"], original["source_read_args"])
        # A caller changing projected coordinates must not corrupt the cached
        # evidence or its checksum, even before JSON serialization.
        projected["items"][0]["read_args"]["offset"] = 98765
        self.assertEqual(rich, snapshot)

    def test_restored_scored_cache_returns_compact_pages_without_neural_calls(self):
        self.register()
        first = self.call(query="needle", bm25_candidates=8, top_k=3, limit=1)
        complete = self.call(result_id=first["result_id"], top_k=8, limit=8)
        state = json.loads(json.dumps(self.service.export_state()))
        backend = VerboseCheapBackend()
        _, restored = self.make_service(backend, views="restored-views")
        restored.restore_state(state)
        page = self.call(restored, result_id=first["result_id"], top_k=8, offset=2, limit=4)
        self.assertEqual(page["items"], complete["items"][2:6])
        self.assertTrue(page["cache_hit"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)
        record = self.diagnostics(restored)[0]
        self.assertEqual(record["backend_attempt_usage"], {})
        self.assertEqual(record["usage_delta"]["scoring_calls"], 0)
        self.assertEqual(compact_rerank_response(record["response"]), page)

    def test_logging_failure_preserves_scored_cache_and_recovers_without_rescoring(self):
        self.register()
        arguments = {"query": "needle", "bm25_candidates": 8, "top_k": 3, "limit": 2}
        with patch.object(self.service, "_append_diagnostic", side_effect=OSError("fixture full disk")):
            failed = json.loads(self.service(arguments))
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["code"], "diagnostic_log_error")
        self.assert_no_private_keys(failed)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.service.summary["cached_rankings"], 1)
        recovered = self.call(**arguments)
        self.assertTrue(recovered["cache_hit"])
        self.assertEqual(len(self.backend.calls), 1)
        records = self.diagnostics()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["call_index"], 2)
        self.assertEqual(records[0]["backend_attempt_usage"], {})
        self.assertEqual(records[0]["usage_delta"]["scoring_calls"], 0)
        self.assertEqual(compact_rerank_response(records[0]["response"]), recovered)
        self.assertEqual([row["status"] for row in self.service.summary["call_history"]],
                         ["error", "ok"])

    def test_error_diagnostics_distinguish_invalid_arguments_from_partial_scoring(self):
        self.register()
        self.call(query="needle", bm25_candidates=8, top_k=3, limit=2)
        # This error occurs before score(). The previous preparation vector
        # must not be attributed to a rejected parameter request.
        rejected = json.loads(self.service({"query": "actor", "limit": 0}))
        self.assertEqual(rejected["status"], "error")
        self.assert_no_private_keys(rejected)
        rejected_record = self.diagnostics()[-1]
        self.assertEqual(rejected_record["backend_attempt_usage"], {})
        self.assertEqual(rejected_record["usage_delta"]["scoring_calls"], 0)
        self.assertEqual(len(self.backend.calls), 1)

        def partial_forward_failure(query, texts):
            self.backend.calls.append((query, list(texts)))
            self.backend.last_usage = {
                "forward_calls": 1, "completed_forward_calls": 0,
                "input_tokens_attempted": 123,
                "sequence_lengths": [21] * len(texts),
                "diagnostic_marker": "PRIVATE-PARTIAL-FORWARD-DETAIL",
            }
            raise RuntimeError("fixture neural forward failed")

        with patch.object(self.backend, "score", side_effect=partial_forward_failure):
            failed = json.loads(self.service({"query": "actor", "bm25_candidates": 5,
                                              "top_k": 3, "limit": 2}))
        self.assertEqual(failed["status"], "error")
        self.assertIn("fixture neural forward failed", failed["error"])
        self.assert_no_private_keys(failed)
        self.assertNotIn("PRIVATE-PARTIAL", json.dumps(failed))
        record = self.diagnostics()[-1]
        self.assertEqual(record["backend_attempt_usage"], self.backend.last_usage)
        self.assertEqual(len(record["backend_attempt_usage"]["sequence_lengths"]), 5)
        self.assertEqual(record["usage_delta"]["scoring_calls"], 1)
        self.assertEqual(record["usage_delta"]["scored_passages"], 0)
        self.assertEqual(record["usage_delta"]["forward_calls"], 1)
        self.assertEqual(record["usage_delta"]["completed_forward_calls"], 0)
        self.assertEqual(record["usage_delta"]["input_tokens_attempted"], 123)

    def test_unicode_crlf_coordinates_and_baseline_read_recover_full_fragments(self):
        self.register()
        result = self.call(query="needle", file_paths=["docs/unicode.txt"],
                           bm25_candidates=9, top_k=9, limit=9)
        original = (self.pool / "docs/unicode.txt").read_bytes().decode("utf-8")
        self.assertIn("\r\n", original)
        self.assertIn("e\u0301 中文 😀", original)
        for item in result["items"]:
            self.assertEqual(set(item), ITEM_KEYS)
            self.assertEqual(original[item["char_start"]:item["char_end"]], item["text"])
            self.assertEqual(item["token_count"], len(FakeFastTokenizer.encode(item["text"])))
            self.assertLessEqual(item["token_count"], 128)
            self.assertLessEqual(item["overlap_tokens"], 32)
            output = make_read_tool(self.pool)(item["read_args"])
            numbered = [line.split("\t", 1)[1] for line in output.splitlines() if "\t" in line]
            # Baseline read normalizes source line endings. Raw character
            # coordinates above still refer to the original CRLF source.
            self.assertIn(item["text"].replace("\r\n", "\n").strip(), "\n".join(numbered))

    def test_long_unicode_lines_keep_exact_bash_fallback_without_source_mutation(self):
        relative = "unicode/中文 'quoted' $.txt"
        documents = {relative: structured(("中文😀e\u0301长词" * 20 + " ") * 280 + "证据目标")}
        index = self.make_index(documents, "long-line")
        _, service = self.make_service(VerboseCheapBackend(), index=index, views="long-views")
        self.register(service, [relative])
        original_bytes = (self.pool / relative).read_bytes()
        original = original_bytes.decode("utf-8")
        result = self.call(service, query="证据目标", bm25_candidates=8, top_k=8, limit=8)
        for item in result["items"]:
            self.assertEqual(set(item), ITEM_KEYS | FALLBACK_KEYS)
            self.assertIs(item["read_may_truncate"], True)
            self.assertEqual(original[item["char_start"]:item["char_end"]], item["text"])
            process = subprocess.run(["sh", "-c", item["bash_args"]["command"]],
                                     cwd=self.pool, capture_output=True, check=True)
            expected = original[item["context_char_start"]:item["context_char_end"]] + "\n"
            self.assertEqual(process.stdout.decode("utf-8"), expected)
            self.assertIn(item["text"], expected)
        self.assertEqual((self.pool / relative).read_bytes(), original_bytes)

    def test_read_function_and_tool_description_match_frozen_baseline_source_hashes(self):
        source = (ROOT / "src/rise/tools.py").read_text("utf-8")
        lines = source.splitlines(keepends=True)
        functions = {node.name: node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef)}
        for name, expected in BASELINE_READ_HASHES.items():
            with self.subTest(function=name):
                node = functions[name]
                actual = "".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")
                self.assertEqual(hashlib.sha256(actual).hexdigest(), expected)
        # Exercise the baseline defaults as well as preserving source identity.
        (self.pool / "many-lines.txt").write_text("\n".join(f"line {i}" for i in range(2001)), "utf-8")
        output = make_read_tool(self.pool)({"file_path": "many-lines.txt"})
        self.assertEqual(sum("\t" in line for line in output.splitlines()), 2000)
        self.assertIn("1 more lines remain; total 2001", output)
        (self.pool / "long-read.txt").write_text("x" * 2001, "utf-8")
        output = make_read_tool(self.pool)({"file_path": "long-read.txt"})
        self.assertIn("x" * 2000 + "...[line truncated; 1 chars]", output)
        self.assertIn("default limit (2000)", read_tool_spec()["function"]["description"])


if __name__ == "__main__":
    unittest.main()
