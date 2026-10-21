"""Query IDs cannot escape per-query output and working directories."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.query_ids import safe_query_id


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueryIdSafetyTests(unittest.TestCase):
    def test_fixed_numeric_ids_and_safe_names_are_accepted(self):
        self.assertEqual(safe_query_id(156), "156")
        self.assertEqual(safe_query_id("qid_156-A"), "qid_156-A")

    def test_preflight_and_direct_runner_reject_path_like_ids(self):
        preflight = load_script("preflight_rise")
        runner = load_script("run_rise")
        unsafe = ("../escape", "a/b", "a\\b", "..", ".", "", "é", "x" * 129, True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            for qid in unsafe:
                with self.subTest(qid=qid):
                    path.write_text(json.dumps({
                        "query_id": qid, "query": "Q", "answer": "A", "gold_doc_ids": []
                    }) + "\n", encoding="utf-8")
                    report = preflight.Report()
                    preflight.check_queries(path, report, expected_count=1)
                    self.assertFalse(report.ready)
                    self.assertTrue(any("invalid query_id" in message for _, message in report.lines))
                    with self.assertRaisesRegex(ValueError, "query_id"):
                        runner._load_records(path)


if __name__ == "__main__":
    unittest.main()
