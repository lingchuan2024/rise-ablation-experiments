"""Local acceptance checks for v10's on-demand passage reranker.

These tests use the real English BM25 pipeline and a small offset-aware
tokenizer/scorer, so no model download, CUDA device, or API request is needed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.reranker_tool import RerankerTool, chunking_identity
from rise.tools import make_read_tool


BODY_BOUNDARY = "=== DOCUMENT BODY ==="


class FakeFastTokenizer:
    """A deterministic whitespace tokenizer exposing the fast-tokenizer API."""

    is_fast = True

    @staticmethod
    def encode(text, add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        return list(range(len(re.findall(r"\S+", text))))

    def __call__(self, text, *, return_offsets_mapping=False,
                 add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FakeBackend:
    identity = {"model": "test/fake-reranker", "revision": "local"}
    last_usage = {}

    def __init__(self, score_fn=None):
        self.tokenizer_calls = 0
        self.calls = []
        self.restored_identity = None
        self.score_fn = score_fn or (lambda _query, texts: [0.0] * len(texts))

    def get_tokenizer(self):
        self.tokenizer_calls += 1
        return FakeFastTokenizer()

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return self.score_fn(query, texts)

    def restore_identity(self, identity):
        self.identity = json.loads(json.dumps(identity))
        self.restored_identity = self.identity


class ByteOffsetTokenizer:
    """Exercise several byte tokens sharing one Unicode character's offsets."""

    is_fast = True

    @staticmethod
    def encode(text, add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        return list(range(len(text.encode("utf-8"))))

    def __call__(self, text, *, return_offsets_mapping=False,
                 add_special_tokens=False, **kwargs):
        assert not add_special_tokens
        offsets = [(index, index + 1) for index, char in enumerate(text)
                   for _ in char.encode("utf-8")]
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def structured(body, header="A header that is not evidence"):
    return f"=== DOCUMENT HEADER ===\n{header}\n{BODY_BOUNDARY}\n{body}\n"


def token_lines(tokens, width=16):
    return "\n".join(" ".join(tokens[start:start + width])
                     for start in range(0, len(tokens), width))


class RerankerToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)

    def write(self, relpath, text):
        destination = self.workspace / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return relpath

    def call(self, tool, **arguments):
        raw = tool(arguments)
        self.assertFalse(raw.startswith("Error:"), raw)
        response = json.loads(raw)
        self.assertEqual(response["status"], "ok")
        return response

    def assert_readable_item(self, item):
        original = (self.workspace / item["file_path"]).read_bytes().decode("utf-8", errors="replace")
        self.assertEqual(original[item["char_start"]:item["char_end"]], item["text"])
        self.assertEqual(item["token_count"], len(FakeFastTokenizer.encode(item["text"])))
        self.assertGreater(item["token_count"], 0)
        self.assertLessEqual(item["token_count"], 128)
        args = item["read_args"]
        self.assertEqual(args["file_path"], item["file_path"])
        self.assertIsInstance(args["offset"], int)
        self.assertIsInstance(args["limit"], int)
        self.assertGreaterEqual(args["offset"], 0)
        self.assertGreater(args["limit"], 0)
        self.assertFalse(item["read_may_truncate"])
        read_output = make_read_tool(self.workspace)(args)
        self.assertFalse(read_output.startswith("Error:"), read_output)
        actual_lines = [line.split("\t", 1)[1] for line in read_output.splitlines()
                        if "\t" in line]
        self.assertIn("\n".join(item["text"].strip().splitlines()), "\n".join(actual_lines))

    def test_search_1000_pool_finds_rank_999_body_passage_and_scores_only_100(self):
        query = "aurora calibration"
        candidates = []
        filler = [f"ordinarylongunrelatedtoken{index:03d}" for index in range(128)]
        target = "documents/doc_0998.txt"
        evidence = (["EVIDENCE_TARGET"] + ["aurora", "calibration"] * 16
                    + [f"evidencelongpreservedtoken{index:03d}" for index in range(95)])
        self.assertEqual(len(evidence), 128)
        for rank in range(1000):
            relpath = f"documents/doc_{rank:04d}.txt"
            body_tokens = filler + filler + evidence if relpath == target else filler
            # A misleading head in the first search result must never become a
            # body passage. All ten previewed bodies lack the query terms.
            header = ("aurora calibration " * 100) if rank == 0 else "No query evidence in head"
            self.write(relpath, structured(token_lines(body_tokens), header))
            candidates.append((relpath, float(1000 - rank)))

        scorer = FakeBackend(lambda _query, texts: [
            99.0 if "EVIDENCE_TARGET" in text else 0.0 for text in texts
        ])
        tool = RerankerTool(working_dir=self.workspace, backend=scorer)
        set_ids = tool.register_search([query], [candidates])
        self.assertEqual(set_ids, ["S001.q1"])
        self.assertEqual(scorer.calls, [])
        self.assertEqual(scorer.tokenizer_calls, 0, "register_search must not load a model")

        first = self.call(tool, set_id=set_ids[0])
        self.assertEqual(first["set_id"], set_ids[0])
        self.assertEqual(first["query"], query)
        self.assertEqual(first["total_passages"], 1003)
        self.assertEqual(first["bm25_candidates"], 100)
        self.assertEqual(first["scored_passages"], 100)
        self.assertEqual(len(first["items"]), 10)
        self.assertEqual(first["next_offset"], 10)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(scorer.calls[0][0], query)
        self.assertEqual(len(scorer.calls[0][1]), 100)
        self.assertTrue(all(0 < len(FakeFastTokenizer.encode(text)) <= 128
                            for text in scorer.calls[0][1]))
        self.assertTrue(all(BODY_BOUNDARY not in text for text in scorer.calls[0][1]))
        winner = first["items"][0]
        self.assertEqual(winner["file_path"], target)
        self.assertEqual(winner["reranker_rank"], 1)
        self.assertEqual(winner["reranker_score"], 99.0)
        self.assertEqual(winner["token_count"], 128)
        self.assertGreater(len(winner["text"]), 2000, "return a complete passage, not a preview")
        self.assertIn("EVIDENCE_TARGET", winner["text"])
        self.assertGreater(winner["char_start"],
                           (self.workspace / target).read_text().index(BODY_BOUNDARY))
        for item in first["items"]:
            self.assert_readable_item(item)

        second = self.call(tool, set_id=set_ids[0], offset=10, limit=10)
        self.assertEqual(len(scorer.calls), 1, "paging must reuse the existing model scores")
        self.assertEqual([item["reranker_rank"] for item in second["items"]], list(range(11, 21)))
        self.assertEqual(second["next_offset"], 20)
        first_positions = {(item["file_path"], item["char_start"]) for item in first["items"]}
        second_positions = {(item["file_path"], item["char_start"]) for item in second["items"]}
        self.assertTrue(first_positions.isdisjoint(second_positions))
        last = self.call(tool, set_id=set_ids[0], offset=90, limit=10)
        self.assertEqual(len(last["items"]), 10)
        self.assertIsNone(last["next_offset"])
        self.assertEqual(len(scorer.calls), 1)

    def test_small_pool_chunks_128_tokens_with_32_overlap_and_reverse_reranks(self):
        a_tokens = [f"token{index:03d}" for index in range(260)]
        b_tokens = ["中文🙂"] + [f"other{index:02d}" for index in range(9)]
        a = self.write("raw/a.txt", token_lines(a_tokens))
        b = self.write("raw/b.txt", token_lines(b_tokens))
        backend = FakeBackend(lambda _query, texts: [float(i) for i in range(len(texts))])
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["absent"], [[(a, 1.0), (b, 0.5)]])[0]
        result = self.call(tool, set_id=set_id)
        self.assertEqual(result["total_passages"], 4)
        self.assertEqual(result["bm25_candidates"], 4)
        self.assertEqual(result["scored_passages"], 4)
        self.assertEqual([item["bm25_rank"] for item in result["items"]], [4, 3, 2, 1])
        self.assertEqual([item["reranker_rank"] for item in result["items"]], [1, 2, 3, 4])
        self.assertIsNone(result["next_offset"])
        self.assertEqual(backend.tokenizer_calls, 0)
        self.assertEqual(result["passage_overlap_tokens"], 32)
        self.assertEqual(result["passage_stride_tokens"], 96)
        ordered = sorted(result["items"], key=lambda item: item["bm25_rank"])
        self.assertEqual([item["token_count"] for item in ordered], [128, 128, 68, 10])
        for relpath, expected_windows in [(a, [a_tokens[:128], a_tokens[96:224], a_tokens[192:]]),
                                          (b, [b_tokens])]:
            passages = sorted((item for item in result["items"] if item["file_path"] == relpath),
                              key=lambda item: item["char_start"])
            self.assertEqual([item["text"].split() for item in passages], expected_windows)
            self.assertEqual(passages[0]["overlap_tokens"], 0)
            for left, right in zip(passages, passages[1:]):
                self.assertLess(right["char_start"], left["char_end"])
                self.assertEqual(right["overlap_tokens"], 32)
        for item in result["items"]:
            self.assert_readable_item(item)

    def test_overlap_preserves_evidence_crossing_a_nonoverlapping_boundary(self):
        words = [f"w{i:03d}" for i in range(260)]
        words[125:131] = "rare evidence crosses the old boundary".split()
        phrase = "rare evidence crosses the old boundary"
        path = self.write("boundary.txt", " ".join(words))
        backend = FakeBackend(lambda _query, texts: [99.0 if phrase in text else 0.0
                                                    for text in texts])
        tool = RerankerTool(self.workspace, backend, FakeFastTokenizer())
        sid = tool.register_search([phrase], [[(path, 1.0)]])[0]
        result = self.call(tool, set_id=sid)
        self.assertEqual(result["items"][0]["reranker_score"], 99.0)
        self.assertIn(phrase, result["items"][0]["text"])
        self.assertEqual(result["items"][0]["overlap_tokens"], 32)
        self.assert_readable_item(result["items"][0])

    def test_complete_final_window_does_not_generate_overlap_only_tail(self):
        for length, counts in [(128, [128]), (224, [128, 128]), (225, [128, 128, 33])]:
            with self.subTest(length=length):
                path = self.write(f"tail_{length}.txt", " ".join(f"w{i}" for i in range(length)))
                tool = RerankerTool(self.workspace, FakeBackend(), FakeFastTokenizer())
                sid = tool.register_search(["absent"], [[(path, 1.0)]])[0]
                result = self.call(tool, set_id=sid)
                windows = sorted(result["items"], key=lambda p: p["char_start"])
                self.assertEqual([p["token_count"] for p in windows], counts)
                self.assertTrue(windows[-1]["text"].endswith(f"w{length - 1}"))

    def test_final_character_coverage_stops_when_standalone_encoding_merges_tokens(self):
        class ContextSensitiveTokenizer(ByteOffsetTokenizer):
            # Emulate a character using fewer tokens when the shorter window
            # is encoded independently from the whole document.
            @staticmethod
            def encode(text, add_special_tokens=False, **kwargs):
                assert not add_special_tokens
                weight = 4 if len(text) > 200 else 2
                return list(range(sum(weight if char == "🙂" else 1 for char in text)))

            def __call__(self, text, *, return_offsets_mapping=False,
                         add_special_tokens=False, **kwargs):
                weight = 4 if len(text) > 200 else 2
                offsets = [(i, i + 1) for i, char in enumerate(text)
                           for _ in range(weight if char == "🙂" else 1)]
                return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

        body = "a" * 222 + "🙂"
        path = self.write("context_boundary.txt", body)
        tool = RerankerTool(self.workspace, FakeBackend(), ContextSensitiveTokenizer())
        sid = tool.register_search(["absent"], [[(path, 1.0)]])[0]
        windows = tool._chunks(tool._sources(tool._sets[sid]))
        self.assertEqual(len(windows), 2, "the final character is already covered")
        self.assertEqual(windows[-1]["char_end"], len(body))
        self.assertEqual(windows[-1]["token_count"], 128)

    def test_unicode_byte_offsets_keep_safe_overlap_and_cover_the_entire_source(self):
        for label, body in [("three_bytes", "汉" * 100), ("four_bytes", "🙂" * 100),
                            ("mixed", "汉🙂abc" * 60)]:
            with self.subTest(label=label):
                path = self.write(f"{label}.txt", body)
                tool = RerankerTool(self.workspace, FakeBackend(), ByteOffsetTokenizer())
                sid = tool.register_search(["absent"], [[(path, 1.0)]])[0]
                sources = tool._sources(tool._sets[sid])
                windows = tool._chunks(sources)
                self.assertEqual(windows[0]["char_start"], 0)
                self.assertEqual(windows[-1]["char_end"], len(body))
                for passage in windows:
                    self.assertEqual(body[passage["char_start"]:passage["char_end"]], passage["text"])
                    self.assertEqual(passage["token_count"], len(passage["text"].encode("utf-8")))
                    self.assertLessEqual(passage["token_count"], 128)
                for left, right in zip(windows, windows[1:]):
                    self.assertLess(left["char_start"], right["char_start"])
                    self.assertLess(right["char_start"], left["char_end"])
                    self.assertLess(left["char_end"], right["char_end"])
                    actual = len(body[right["char_start"]:left["char_end"]].encode("utf-8"))
                    self.assertEqual(right["overlap_tokens"], actual)
                    self.assertLessEqual(actual, 32)
                    self.assertGreater(actual, 0)

    def test_english_stemming_and_stopwords_rank_body_using_original_query(self):
        query = "the RUNNING"
        misleading = self.write("head.txt", structured("orchard copper", "running " * 50))
        stopwords = self.write("stopwords.txt", "the " * 50 + "orchard")
        relevant = self.write("relevant.txt", "run marathon")
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search([query], [[(misleading, 3.0), (stopwords, 2.0), (relevant, 1.0)]])[0]
        result = self.call(tool, set_id=set_id)
        self.assertEqual(result["items"][0]["file_path"], relevant)
        self.assertGreater(result["items"][0]["bm25_score"], 0.0)
        self.assertEqual(backend.calls[0][0], query)
        self.assertTrue(all("running" not in text for text in backend.calls[0][1]))

    def test_query_sets_are_frozen_and_isolated_from_other_searches_and_workspace(self):
        a = self.write("pool_a.txt", "alpha evidence")
        b = self.write("pool_b.txt", "beta evidence")
        c = self.write("pool_c.txt", "gamma evidence")
        self.write("outside.txt", "alpha alpha beta beta gamma gamma")
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend)
        queries = ["Alpha Original?", "Beta Original?"]
        results = [[(a, 1.0)], [(b, 1.0)]]
        ids = tool.register_search(queries, results)
        self.assertEqual(ids, ["S001.q1", "S001.q2"])
        # Caller mutations and later searches must not change saved pools.
        queries[0] = "rewritten query"
        results[0].append((c, 10.0))
        later_id = tool.register_search(["Gamma Original?"], [[(c, 1.0)]])[0]
        self.assertEqual(later_id, "S002.q1")
        for set_id, expected_query, expected_file in [
            (ids[1], "Beta Original?", b), (later_id, "Gamma Original?", c), (ids[0], "Alpha Original?", a),
        ]:
            result = self.call(tool, set_id=set_id)
            self.assertEqual(result["query"], expected_query)
            self.assertEqual([item["file_path"] for item in result["items"]], [expected_file])
            self.assertEqual(backend.calls[-1][0], expected_query)
            self.assertEqual(backend.calls[-1][1], [(self.workspace / expected_file).read_text()])

    def test_ties_are_stable_by_search_rank_character_position_then_bm25_rank(self):
        a = self.write("z_rank_first.txt", token_lines([f"neutral{i:03d}" for i in range(129)]))
        b = self.write("a_rank_second.txt", "neutral end")
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["absent"], [[(a, 0.1), (b, 99.0)]])[0]
        result = self.call(tool, set_id=set_id)
        self.assertEqual([item["file_path"] for item in result["items"]], [a, a, b])
        self.assertEqual([item["bm25_rank"] for item in result["items"]], [1, 2, 3])
        self.assertEqual([item["reranker_rank"] for item in result["items"]], [1, 2, 3])
        self.assertLess(result["items"][0]["char_start"], result["items"][1]["char_start"])

    def test_body_marker_is_recognized_only_on_its_own_line(self):
        body = "正文🙂 evidence"
        proper = self.write("structured.txt", structured(body, "private header clue"))
        ordinary = self.write("raw.txt", "earlier BODY word\nordinary content")
        inline = self.write("inline.txt", f"prefix {BODY_BOUNDARY} inline evidence")
        tool = RerankerTool(working_dir=self.workspace, backend=FakeBackend(), tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["absent"], [[(proper, 3.0), (ordinary, 2.0), (inline, 1.0)]])[0]
        result = self.call(tool, set_id=set_id)
        items = {item["file_path"]: item for item in result["items"]}
        self.assertEqual(items[proper]["text"].strip(), body)
        self.assertIn("earlier BODY word", items[ordinary]["text"])
        self.assertIn("prefix", items[inline]["text"])
        for item in result["items"]:
            self.assert_readable_item(item)

    def test_invalid_arguments_return_error_without_loading_or_scoring(self):
        path = self.write("one.txt", "simple evidence")
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend)
        set_id = tool.register_search(["simple"], [[(path, 1.0)]])[0]
        for args in [
            {}, {"set_id": ""}, {"set_id": "unknown"},
            {"set_id": set_id, "offset": -1}, {"set_id": set_id, "offset": "abc"},
            {"set_id": set_id, "offset": 1.5}, {"set_id": set_id, "offset": True},
            {"set_id": set_id, "limit": 0}, {"set_id": set_id, "limit": -1},
            {"set_id": set_id, "limit": "two"}, {"set_id": set_id, "limit": True},
        ]:
            with self.subTest(args=args):
                self.assertTrue(tool(args).startswith("Error:"))
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)

    def test_missing_and_changed_sources_fail_instead_of_using_a_different_corpus(self):
        for failure in ["missing", "changed"]:
            with self.subTest(failure=failure):
                path = self.write(f"{failure}.txt", "original evidence")
                backend = FakeBackend()
                tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
                set_id = tool.register_search(["original"], [[(path, 1.0)]])[0]
                if failure == "missing":
                    (self.workspace / path).unlink()
                else:
                    (self.workspace / path).write_text("modified evidence", encoding="utf-8")
                self.assertTrue(tool({"set_id": set_id}).startswith("Error:"))
                self.assertEqual(backend.calls, [])

    def test_changed_source_is_checked_even_after_scoring_is_cached(self):
        path = self.write("cached.txt", "original evidence")
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["original"], [[(path, 1.0)]])[0]
        self.call(tool, set_id=set_id)
        (self.workspace / path).write_text("modified evidence", encoding="utf-8")
        self.assertTrue(tool({"set_id": set_id}).startswith("Error:"))
        self.assertEqual(len(backend.calls), 1)

    def test_scorer_errors_missing_scores_nan_and_infinity_never_produce_results(self):
        path = self.write("score.txt", token_lines([f"word{i:03d}" for i in range(260)]))

        def fail(_query, _texts):
            raise RuntimeError("the scorer is unavailable")

        for name, score_fn in [
            ("exception", fail), ("missing", lambda _q, _texts: [0.5]),
            ("nan", lambda _q, texts: [float("nan")] * len(texts)),
            ("infinity", lambda _q, texts: [float("inf")] * len(texts)),
        ]:
            with self.subTest(failure=name):
                backend = FakeBackend(score_fn)
                tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
                set_id = tool.register_search(["word"], [[(path, 1.0)]])[0]
                output = tool({"set_id": set_id})
                self.assertTrue(output.startswith("Error:"), output)
                self.assertEqual(len(backend.calls), 1)

    def test_json_checkpoint_restores_queries_frozen_pools_and_next_set_number(self):
        a = self.write("first.txt", "alpha evidence")
        b = self.write("second.txt", "beta evidence")
        backend = FakeBackend()
        original = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        ids = original.register_search(["Alpha?", "Beta?"], [[(a, 2.0)], [(b, 1.0)]])
        before = self.call(original, set_id=ids[0])
        state = json.loads(json.dumps(original.export_state()))
        restored_backend = FakeBackend()
        restored = RerankerTool(working_dir=self.workspace, backend=restored_backend, tokenizer=FakeFastTokenizer())
        restored.restore_state(state)
        after = self.call(restored, set_id=ids[0])
        self.assertEqual(after["query"], "Alpha?")
        self.assertEqual(after["items"], before["items"])
        beta = self.call(restored, set_id=ids[1])
        self.assertEqual(beta["query"], "Beta?")
        self.assertEqual([item["file_path"] for item in beta["items"]], [b])
        self.assertEqual(restored.register_search(["Later?"], [[(b, 1.0)]]), ["S002.q1"])
        # A persisted set keeps its source fingerprint across process restart.
        changed = RerankerTool(working_dir=self.workspace, backend=FakeBackend(), tokenizer=FakeFastTokenizer())
        changed.restore_state(state)
        (self.workspace / a).write_text("changed evidence", encoding="utf-8")
        self.assertTrue(changed({"set_id": ids[0]}).startswith("Error:"))

    def test_invalid_utf8_is_replaced_and_offsets_follow_the_decoded_source(self):
        path = "invalid_utf8.txt"
        prefix = "before evidence 中文🙂 ".encode("utf-8") + b"\xff\xfe "
        data = prefix + token_lines([f"following{i:03d}" for i in range(140)]).encode("utf-8")
        (self.workspace / path).write_bytes(data)
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["evidence"], [[(path, 1.0)]])[0]
        result = self.call(tool, set_id=set_id)
        self.assertEqual(result["total_passages"], 2)
        self.assertEqual(len(backend.calls), 1)
        decoded = data.decode("utf-8", errors="replace")
        self.assertIn("\ufffd", result["items"][0]["text"])
        for item in result["items"]:
            self.assertEqual(decoded[item["char_start"]:item["char_end"]], item["text"])
            self.assert_readable_item(item)

    def test_readback_line_ranges_follow_baseline_splitlines_for_unicode_and_cr(self):
        for label, separator in [("unicode_ls", "\u2028"), ("cr", "\r"),
                                 ("crlf", "\r\n"), ("unicode_nel", "\u0085")]:
            with self.subTest(separator=label):
                body = " ".join([f"ordinary{i:03d}" for i in range(128)]) + separator + "target evidence"
                path = f"{label}.txt"
                # Write bytes so CR and CRLF coordinates are preserved.
                (self.workspace / path).write_bytes(body.encode("utf-8"))
                backend = FakeBackend(lambda _q, texts: [
                    99.0 if "target evidence" in text else 0.0 for text in texts
                ])
                tool = RerankerTool(working_dir=self.workspace, backend=backend,
                                    tokenizer=FakeFastTokenizer())
                set_id = tool.register_search(["target evidence"], [[(path, 1.0)]])[0]
                result = self.call(tool, set_id=set_id)
                self.assertEqual(result["total_passages"], 2)
                winner = result["items"][0]
                self.assertEqual(winner["token_count"], 34)
                self.assertTrue(winner["text"].endswith("target evidence"))
                self.assertEqual(body[winner["char_start"]:winner["char_end"]], winner["text"])
                actual_read = make_read_tool(self.workspace)(winner["read_args"])
                self.assertIn("target evidence", actual_read)
                self.assert_readable_item(winner)

    def test_stopword_only_corpus_has_zero_bm25_scores_and_real_backend_scores(self):
        a = self.write("stop_a.txt", token_lines(["the"] * 260))
        b = self.write("stop_b.txt", "and and and and and and and")
        backend = FakeBackend(lambda _q, texts: [float(i) + 0.25 for i in range(len(texts))])
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["the and"], [[(a, 2.0), (b, 1.0)]])[0]
        result = self.call(tool, set_id=set_id)
        self.assertEqual(result["total_passages"], 4)
        self.assertEqual(result["scored_passages"], 4)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 4)
        self.assertEqual(backend.calls[0][0], "the and")
        self.assertEqual([item["bm25_score"] for item in result["items"]], [0.0] * 4)
        self.assertEqual([item["reranker_score"] for item in result["items"]], [3.25, 2.25, 1.25, 0.25])
        self.assertEqual([item["bm25_rank"] for item in result["items"]], [4, 3, 2, 1])

    def test_long_single_line_passage_supplies_executable_character_readback(self):
        tokens = [f"preserved_long_token{i:03d}" for i in range(512)]
        path = self.write("a long 'source'.txt", " ".join(tokens))
        backend = FakeBackend(lambda _q, texts: [float(i) for i in range(len(texts))])
        tool = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = tool.register_search(["absent"], [[(path, 1.0)]])[0]
        result = self.call(tool, set_id=set_id)
        winner = result["items"][0]
        self.assertEqual(winner["bm25_rank"], 5)
        self.assertEqual(winner["token_count"], 128)
        self.assertGreater(winner["char_start"], 2000)
        self.assertGreater(len(winner["text"]), 2000)
        original = (self.workspace / path).read_text(encoding="utf-8")
        self.assertEqual(original[winner["char_start"]:winner["char_end"]], winner["text"])
        self.assertTrue(winner["read_may_truncate"])
        ordinary_read = make_read_tool(self.workspace)(winner["read_args"])
        self.assertIn("line truncated", ordinary_read)
        self.assertNotIn(winner["text"].strip(), ordinary_read)
        self.assertIsInstance(winner["bash_args"], dict)
        command = winner["bash_args"]["command"]
        self.assertTrue(command.startswith("python3 -c "), command)
        execution = subprocess.run(["/bin/bash", "-c", command], cwd=self.workspace,
                                   capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        self.assertIn(winner["text"].strip(), execution.stdout)

    def test_search_scope_rejects_traversal_and_symlink_sources(self):
        legitimate = self.write("legitimate.txt", "ordinary evidence")
        self.write("folder/other.txt", "ordinary evidence")
        file_link = self.workspace / "file_link.txt"
        file_link.symlink_to(self.workspace / legitimate)
        directory_link = self.workspace / "directory_link"
        directory_link.symlink_to(self.workspace / "folder", target_is_directory=True)
        backend = FakeBackend()
        tool = RerankerTool(working_dir=self.workspace, backend=backend)
        for unsafe in [
            "../" + self.workspace.name + "/" + legitimate,
            "folder/../legitimate.txt", str(self.workspace / legitimate),
            "file_link.txt", "directory_link/other.txt",
        ]:
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                tool.register_search(["ordinary"], [[(unsafe, 1.0)]])
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.tokenizer_calls, 0)
        set_id = tool.register_search(["ordinary"], [[(legitimate, 1.0)]])[0]
        self.assertEqual(set_id, "S001.q1", "a rejected scope must not consume a set number")
        # A symlink introduced after the search must also fail closed.
        (self.workspace / legitimate).unlink()
        (self.workspace / legitimate).symlink_to(self.workspace / "folder/other.txt")
        self.assertTrue(tool({"set_id": set_id}).startswith("Error:"))
        self.assertEqual(backend.calls, [])
        with self.assertRaises(ValueError):
            RerankerTool(working_dir=directory_link, backend=FakeBackend())

    def test_checkpoint_rejects_model_identity_disagreement(self):
        path = self.write("identity.txt", "ordinary evidence")
        backend = FakeBackend()
        original = RerankerTool(working_dir=self.workspace, backend=backend, tokenizer=FakeFastTokenizer())
        set_id = original.register_search(["ordinary"], [[(path, 1.0)]])[0]
        self.call(original, set_id=set_id)
        state = json.loads(json.dumps(original.export_state()))
        self.assertEqual(state["backend_identity"], backend.identity)
        self.assertEqual(state["views"][set_id]["model_identity"], state["backend_identity"])
        restored_backend = FakeBackend()
        restored = RerankerTool(working_dir=self.workspace, backend=restored_backend,
                                tokenizer=FakeFastTokenizer())
        restored.restore_state(state)
        self.assertEqual(restored_backend.restored_identity, state["backend_identity"])
        self.call(restored, set_id=set_id)
        self.assertEqual(restored_backend.calls, [], "a valid cached identity preserves scores")
        for location in ["view", "backend"]:
            with self.subTest(location=location):
                bad_state = json.loads(json.dumps(state))
                changed_identity = {"model": "test/different-reranker", "revision": "other"}
                if location == "view":
                    bad_state["views"][set_id]["model_identity"] = changed_identity
                else:
                    bad_state["backend_identity"] = changed_identity
                rejected = RerankerTool(working_dir=self.workspace, backend=FakeBackend(),
                                        tokenizer=FakeFastTokenizer())
                with self.assertRaises(ValueError):
                    rejected.restore_state(bad_state)

    def test_checkpoint_rejects_old_or_changed_chunking_without_reusing_scores(self):
        path = self.write("cached_windows.txt", " ".join(f"word{i}" for i in range(260)))
        original = RerankerTool(self.workspace, FakeBackend(), FakeFastTokenizer())
        sid = original.register_search(["absent"], [[(path, 1.0)]])[0]
        self.call(original, set_id=sid)
        state = original.export_state()
        self.assertEqual(state["chunking_identity"], chunking_identity())
        for field, old_value in [("state_version", 1), ("overlap_tokens", 0),
                                 ("stride_tokens", 128), ("version", "nonoverlap-v1")]:
            with self.subTest(field=field):
                bad = json.loads(json.dumps(state))
                if field == "state_version":
                    bad["version"] = old_value
                    bad.pop("chunking_identity")
                else:
                    bad["chunking_identity"][field] = old_value
                backend = FakeBackend()
                restored = RerankerTool(self.workspace, backend, FakeFastTokenizer())
                with self.assertRaisesRegex(ValueError, "checkpoint (version|chunking)"):
                    restored.restore_state(bad)
                self.assertEqual(backend.calls, [])
                self.assertEqual(restored.summary["ranked_sets"], 0)


if __name__ == "__main__":
    unittest.main()
