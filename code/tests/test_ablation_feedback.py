"""Pagination metadata must not hide repeated unchanged search evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rise.v13_feedback import ProgressFeedback, normalize_search_page_ids


class PaginationFeedbackTests(unittest.TestCase):
    def test_new_ids_preserve_repetition_detection_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "feedback.jsonl"
            feedback = ProgressFeedback(pool_version_provider=lambda: 1, log_path=log,
                search_fingerprint_normalizer=normalize_search_page_ids)
            results = []
            for number in range(1, 4):
                result = (f"Search record: S{number:03}.q1\n"
                          "  1. [5.00] docs/a.txt\n"
                          f'Search pages: [{{"search_id":"S{number:03}.q1","next_offset":10}}]')
                results.append(feedback.observe("search", {"queries": ["aurora"]}, result))
            self.assertIn("Same arguments and returned content 3 times", results[-1])
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len({row["result_fingerprint_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["visible_result_sha256"] for row in rows}), 3)

    def test_document_text_and_page_offset_stay_in_fingerprint(self):
        evidence = 'Query: "S001.q1"\n  "S001.q1 quoted source"\n'
        first = evidence + 'Search page: {"search_id":"S001.q1","offset":10}'
        second = evidence + 'Search page: {"search_id":"S002.q1","offset":20}'
        self.assertNotEqual(normalize_search_page_ids(first), normalize_search_page_ids(second))
        self.assertIn(evidence, normalize_search_page_ids(first))
