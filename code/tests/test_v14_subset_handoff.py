"""Offline acceptance of an explicit bash -> rerank -> read path handoff.

The shell really runs rg through make_bash_tool in an isolated temporary
fixture. The existing raw-TF passage index and scope-local BM25 are real;
only neural relevance scores are deterministic. This suite does not change
or test the operating-system sandbox, and uses no question data or API.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rise.indexed_accumulated_space import IndexedAccumulatedSpaceManager
from rise.rerank_v13 import RerankService
from rise.tools import make_bash_tool, make_read_tool
from test_indexed_accumulated_space import CheapBackend, build_raw_index
from test_reranker_tool import structured


@unittest.skipUnless(shutil.which("rg"), "requires a real ripgrep executable")
class ExplicitSubsetHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.documents = {
            "docs/a.txt": structured(
                "SELECT_GROUP alpha\n"
                "calibration orbit EVIDENCE_A caf\u00e9 e\u0301 \u4e2d\u6587\n" + "context " * 60),
            "docs/b.txt": structured(
                "SELECT_GROUP beta\ncalibration EVIDENCE_B\n" + "context " * 80),
            # This source wins global lexical and neural ranking, but must
            # never enter scoring when the caller explicitly selects A/B.
            "docs/c.txt": structured(
                "calibration orbit " * 40 + "EVIDENCE_C OUTSIDE_SUBSET"),
        }
        self.index = build_raw_index(
            self.root / "index", self.root / "corpus", self.documents)
        for relative, content in self.documents.items():
            path = self.pool / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        self.backend = CheapBackend(score_fn=lambda _query, texts: [
            100.0 if "EVIDENCE_C" in text else
            10.0 if "EVIDENCE_A" in text else 1.0 for text in texts])
        self.manager = IndexedAccumulatedSpaceManager(
            self.pool, self.root / "views", self.backend,
            passage_index=self.index,
            runtime_to_corpus={relative: relative for relative in self.documents})
        self.rerank = RerankService(self.manager)
        self.rerank.register_search(
            ["synthetic source retrieval"],
            [[(relative, 1.0) for relative in self.documents]])
        self.bash = make_bash_tool(self.pool, enable_sandbox=False)
        self.read = make_read_tool(self.pool)
        self.query = "calibration orbit"

    def shell_paths(self, marker):
        # Do not manufacture file_paths: these exact lines become the next
        # tool's arguments, including rg's ordinary './' prefix.
        output = self.bash({
            "command": f"{shlex.quote(shutil.which('rg'))} -l -F "
                       f"{shlex.quote(marker)} .",
        })
        paths = output.splitlines()
        self.assertTrue(paths, output)
        self.assertTrue(all(path.startswith("./docs/") for path in paths), output)
        return paths

    def call(self, **args):
        result = json.loads(self.rerank(args))
        self.assertEqual(result["status"], "ok", result)
        return result

    def test_real_shell_paths_bound_local_scoring_and_read_preserves_full_pool(self):
        originals = {path: (self.pool / path).read_bytes() for path in self.documents}
        paths = self.shell_paths("SELECT_GROUP")
        self.assertEqual(set(paths), {"./docs/a.txt", "./docs/b.txt"})
        canonical = [path.removeprefix("./") for path in paths]

        local = self.index.scope(canonical, cache=False).search(
            self.query, 2, include_text=True)
        global_result = self.index.scope(self.documents, cache=False).search(self.query, 3)
        self.assertEqual(global_result["items"][0]["path"], "docs/c.txt")
        local_by_path = {item["path"]: item for item in local["items"]}
        global_by_path = {item["path"]: item for item in global_result["items"]}
        self.assertNotEqual(
            local_by_path["docs/a.txt"]["bm25_score"],
            global_by_path["docs/a.txt"]["bm25_score"],
            "Fixture must distinguish scope-local scoring from global filtering")

        result = self.call(query=self.query, file_paths=paths,
                           bm25_candidates=2, top_k=2, limit=2)
        self.assertEqual(result["scope_kind"], "files")
        self.assertEqual(result["scope_documents"], 2)
        self.assertEqual(result["pool_documents"], 3)
        self.assertEqual(result["total_passages"], 2)
        self.assertEqual(result["scored_passages"], 2)
        self.assertEqual([item["file_path"] for item in result["items"]],
                         ["docs/a.txt", "docs/b.txt"])
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][0], self.query)
        self.assertCountEqual(self.backend.calls[0][1],
                              [item["text"] for item in local["items"]])
        self.assertTrue(all("EVIDENCE_C" not in text for text in self.backend.calls[0][1]))
        self.assertEqual(self.backend.tokenizer_calls, 0)
        for item in result["items"]:
            expected = local_by_path[item["file_path"]]
            self.assertEqual(np.float32(item["bm25_score"]).view(np.uint32),
                             np.float32(expected["bm25_score"]).view(np.uint32))
            self.assertEqual(item["bm25_rank"], expected["rank"])
            source = originals[item["file_path"]].decode("utf-8")
            self.assertEqual(source[item["char_start"]:item["char_end"]], item["text"])
            read_args = item["read_args"]
            self.assertEqual(read_args["file_path"], item["file_path"])
            read_output = self.read(read_args)
            recovered = "\n".join(line.split("\t", 1)[1]
                                   for line in read_output.splitlines() if "\t" in line)
            expected_lines = source.splitlines()[
                read_args["offset"]:read_args["offset"] + read_args["limit"]]
            self.assertEqual(recovered, "\n".join(expected_lines))
            self.assertIn(item["text"].strip(), recovered)

        # A rerank subset restricts this call alone, never original tools.
        self.assertIn("EVIDENCE_C", self.read({"file_path": "docs/c.txt"}))
        self.assertIn("EVIDENCE_C", self.bash({"command": "cat docs/c.txt"}))
        self.assertEqual(set(self.manager._sources), set(self.documents))
        self.assertEqual(originals, {
            path: (self.pool / path).read_bytes() for path in self.documents})

    def test_omitted_file_paths_ignore_previous_shell_outputs(self):
        self.assertEqual(self.shell_paths("EVIDENCE_B"), ["./docs/b.txt"])
        whole_pool = self.call(query=self.query, bm25_candidates=3, top_k=3, limit=3)
        self.assertEqual(whole_pool["scope_kind"], "accumulated")
        self.assertEqual(whole_pool["scope_documents"], 3)
        self.assertEqual(whole_pool["scored_passages"], 3)
        self.assertEqual({item["file_path"] for item in whole_pool["items"]},
                         set(self.documents))
        self.assertEqual(whole_pool["items"][0]["file_path"], "docs/c.txt")
        self.assertEqual(len(self.backend.calls[0][1]), 3)

        self.assertEqual(self.shell_paths("EVIDENCE_A"), ["./docs/a.txt"])
        repeated = self.call(query=self.query, bm25_candidates=3, top_k=3, limit=3)
        self.assertEqual(repeated["scope_documents"], 3)
        self.assertEqual(repeated["result_id"], whole_pool["result_id"])
        self.assertEqual(repeated["items"], whole_pool["items"])
        self.assertTrue(repeated["cache_hit"])
        self.assertEqual(len(self.backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
