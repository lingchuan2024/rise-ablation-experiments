"""Small, local-only tests; no dataset, API key, or pyarrow is needed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_local_corpus.py"
SPEC = importlib.util.spec_from_file_location("export_local_corpus", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def source(rows):
    return lambda columns: ({column: row[column] for column in columns} for row in rows)


class ExportLocalCorpusTests(unittest.TestCase):
    def test_pdf_viewer_url_uses_embedded_document_address(self):
        url = "chrome-extension://viewer-id/https://example.org/docs/paper.pdf"
        self.assertEqual(
            exporter._candidate_relpath(url, "10395"),
            "example.org/paper.pdf.txt",
        )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None,
                         "pyarrow is needed for the local Arrow cache round trip")
    def test_local_arrow_shards_create_bm25_parquet_and_aligned_toc_tree(self):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cache = base / "hf_cache"
            cache.mkdir()
            raw_rows = [
                {"docid": "one", "text": "raw one", "url": "https://example.org/page"},
                {"docid": "two", "text": "raw two", "url": "https://example.org/page"},
            ]
            for number, row in enumerate(raw_rows):
                shard = cache / f"browsecomp-plus-corpus-train-{number:05d}-of-00002.arrow"
                table = pa.Table.from_pylist([row])
                with shard.open("wb") as stream, ipc.new_stream(stream, table.schema) as writer:
                    writer.write_table(table)
            structured = base / "toc.parquet"
            pq.write_table(pa.Table.from_pylist([
                {**row, "text": "## Table of Contents\n=== DOCUMENT BODY ===\n" + row["text"]}
                for row in reversed(raw_rows)
            ]), structured)
            raw_parquet = base / "corpus" / "browsecomp_plus.parquet"
            command = [sys.executable, str(SCRIPT),
                       "--raw-arrow-dir", str(cache),
                       "--raw-parquet-out", str(raw_parquet),
                       "--structured-parquet", str(structured),
                       "--raw-out", str(base / "raw_tree"),
                       "--structured-out", str(base / "structured_tree"),
                       "--filename-map", str(base / "map.json")]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(pq.read_table(raw_parquet).to_pylist(), raw_rows)
            mapping = json.loads((base / "map.json").read_text())["relpath_to_docid"]
            self.assertEqual(set(mapping.values()), {"one", "two"})
            self.assertEqual(len(mapping), 2)  # colliding URLs receive distinct stable paths
            for relpath, did in mapping.items():
                raw_text = next(row["text"] for row in raw_rows if row["docid"] == did)
                self.assertEqual((base / "raw_tree" / relpath).read_text(), raw_text)
                self.assertTrue((base / "structured_tree" / relpath).read_text().endswith(raw_text))
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("Converted/reused 2 local Arrow rows", second.stdout)

            # A missing cache shard must be detected before reusing a parquet.
            (cache / "browsecomp-plus-corpus-train-00001-of-00002.arrow").unlink()
            third = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(third.returncode, 0)
            self.assertIn("expected 2 Arrow shards", third.stderr)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None,
                         "pyarrow is needed for the Parquet CLI round trip")
    def test_parquet_cli_round_trip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / "raw.parquet"
            structured = base / "structured.parquet"
            rows = [
                {"docid": "1", "text": "raw one", "url": "https://example.org/one"},
                {"docid": "2", "text": "raw two", "url": "https://example.org/two"},
            ]
            pq.write_table(pa.Table.from_pylist(rows), raw)
            pq.write_table(pa.Table.from_pylist([
                {**row, "text": "## Table of Contents\n=== DOCUMENT BODY ===\n" + row["text"]}
                for row in reversed(rows)
            ]), structured)
            command = [sys.executable, str(SCRIPT),
                       "--raw-parquet", str(raw),
                       "--structured-parquet", str(structured),
                       "--raw-out", str(base / "raw_tree"),
                       "--structured-out", str(base / "structured_tree"),
                       "--filename-map", str(base / "map.json")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            mapping = json.loads((base / "map.json").read_text())["relpath_to_docid"]
            self.assertEqual(set(mapping.values()), {"1", "2"})
            for relpath in mapping:
                self.assertTrue((base / "raw_tree" / relpath).is_file())
                self.assertIn("=== DOCUMENT BODY ===",
                              (base / "structured_tree" / relpath).read_text())

    def test_raw_and_structured_align_by_docid_and_paths_are_deterministic(self):
        raw = [
            {"docid": "b", "url": "https://example.org/a/page", "text": "raw B"},
            {"docid": "a", "url": "https://example.org/b/page", "text": "raw A"},
            {"docid": "c", "url": "https://other.org/%2E%2E/unsafe", "text": "raw C"},
        ]
        structured = [
            {"docid": "c", "url": raw[2]["url"], "text": "toc C"},
            {"docid": "a", "url": raw[1]["url"], "text": "toc A"},
            {"docid": "b", "url": raw[0]["url"], "text": "toc B"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            map_path = base / "runs" / "corpus_filename_docid_map.json"
            count = exporter.export_from_sources(
                source(raw), raw_out=base / "raw", map_path=map_path,
                structured_source=source(structured), structured_out=base / "structured",
            )
            self.assertEqual(count, 3)
            mapping = json.loads(map_path.read_text())["relpath_to_docid"]
            self.assertEqual(set(mapping.values()), {"a", "b", "c"})
            self.assertEqual(mapping, exporter._generated_map({r["docid"]: r["url"] for r in reversed(raw)}))
            by_id = {did: rel for rel, did in mapping.items()}
            self.assertNotEqual(by_id["a"], by_id["b"])
            for row in raw:
                rel = by_id[row["docid"]]
                exporter._validate_relpath(rel)
                self.assertEqual((base / "raw" / rel).read_text(), row["text"])
                structured_row = next(r for r in structured if r["docid"] == row["docid"])
                self.assertEqual((base / "structured" / rel).read_text(), structured_row["text"])
            # Re-running verifies existing files and map without rewriting them.
            self.assertEqual(exporter.export_from_sources(
                source(raw), raw_out=base / "raw", map_path=map_path,
                structured_source=source(structured), structured_out=base / "structured",
            ), 3)

    def test_existing_official_map_is_reused_verbatim(self):
        rows = [{"docid": "7", "url": "https://example.org/foo", "text": "doc"}]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            official = base / "official.json"
            official.write_text(json.dumps({"relpath_to_docid": {"special/site-title.txt": "7"}}))
            result = base / "runs" / "corpus_filename_docid_map.json"
            exporter.export_from_sources(
                source(rows), raw_out=base / "raw", map_path=result, existing_map=official,
            )
            self.assertEqual((base / "raw" / "special/site-title.txt").read_text(), "doc")
            self.assertEqual(json.loads(result.read_text())["relpath_to_docid"],
                             {"special/site-title.txt": "7"})

    def test_mismatch_or_unsafe_official_path_fails_before_writes(self):
        rows = [{"docid": "7", "url": "https://example.org/foo", "text": "doc"}]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            official = base / "official.json"
            official.write_text(json.dumps({"relpath_to_docid": {"../escape.txt": "7"}}))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                exporter.export_from_sources(source(rows), raw_out=base / "raw",
                                             map_path=base / "map.json", existing_map=official)
            self.assertFalse((base / "raw").exists())

            official.write_text(json.dumps({"relpath_to_docid": {"safe/x.txt": "8"}}))
            with self.assertRaisesRegex(ValueError, "docids do not exactly match"):
                exporter.export_from_sources(source(rows), raw_out=base / "raw",
                                             map_path=base / "map.json", existing_map=official)
            self.assertFalse((base / "raw").exists())

    def test_structured_mismatch_and_existing_file_conflict(self):
        rows = [{"docid": "7", "url": "https://example.org/foo", "text": "doc"}]
        wrong = [{"docid": "8", "url": rows[0]["url"], "text": "toc"}]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with self.assertRaisesRegex(ValueError, "same docid"):
                exporter.export_from_sources(source(rows), raw_out=base / "raw",
                                             map_path=base / "map.json",
                                             structured_source=source(wrong),
                                             structured_out=base / "structured")
            self.assertFalse((base / "raw").exists())
            exporter.export_from_sources(source(rows), raw_out=base / "raw", map_path=base / "map.json")
            bad = [{**rows[0], "text": "different"}]
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                exporter.export_from_sources(source(bad), raw_out=base / "raw", map_path=base / "map.json")
            self.assertEqual(next((base / "raw").rglob("*.txt")).read_text(), "doc")

    def test_symlink_inside_output_tree_is_rejected(self):
        rows = [{"docid": "7", "url": "https://example.org/foo", "text": "doc"}]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "raw").mkdir()
            (base / "outside").mkdir()
            (base / "raw" / "example.org").symlink_to(base / "outside", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                exporter.export_from_sources(source(rows), raw_out=base / "raw", map_path=base / "map.json")
            self.assertEqual(list((base / "outside").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
