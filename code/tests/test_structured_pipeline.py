"""Local-only regression checks for the structured RISE corpus pipeline."""

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "scripts" / "structured" / "run_full_corpus.py"
BUILD_TREE = REPO / "scripts" / "structured" / "build_structured_corpus.py"


def _run(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        capture_output=True, text=True, check=False,
    )


def _parquet_available() -> bool:
    if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None:
        return False
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


class StructuredPipelineLocalTests(unittest.TestCase):
    def test_request_keeps_paper_model_and_low_effort_without_api_dependencies(self):
        namespace = runpy.run_path(str(PIPELINE))
        request, truncation = namespace["build_request_line"]("doc-1", "A local document")
        self.assertEqual(request["custom_id"], "doc-1")
        self.assertEqual(request["body"]["model"], "gpt-5.4-nano")
        self.assertEqual(request["body"]["reasoning_effort"], "low")
        self.assertFalse(truncation["truncated"])

    def test_submit_is_blocked_without_api_permission_and_no_state_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "out"
            result = _run(PIPELINE, "--corpus", root / "absent.parquet", "--out-dir", out, "submit")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--allow-api", result.stderr)
            self.assertFalse(out.exists())

    def test_missing_local_output_never_triggers_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "raw.parquet"
            corpus.write_bytes(b"present; process checks batch output first")
            out = root / "out"
            out.mkdir()
            (out / "state.json").write_text(json.dumps({
                "stage": "packed", "corpus_path": str(corpus.resolve()),
                "batches": [{"idx": 0, "status": "pending", "batch_id": None}],
            }), encoding="utf-8")
            result = _run(PIPELINE, "--corpus", corpus, "--out-dir", out, "process")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no download was attempted", result.stderr)
            self.assertFalse((out / "audit.jsonl").exists())

    @unittest.skipUnless(_parquet_available(), "pandas and pyarrow are needed for parquet round trip")
    def test_local_pack_process_parquet_and_tree_export(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "raw.parquet"
            out = root / "processed"
            tree = root / "tree"
            mapping = root / "map.json"
            documents = [
                {"docid": "1", "text": "---\ntitle: One\n---\nFirst local body.", "url": "https://example.org/one"},
                {"docid": "2", "text": "---\ntitle: Two\n---\nSecond local body.", "url": "https://example.org/two"},
            ]
            pd.DataFrame(documents).to_parquet(corpus, index=False)
            mapping.write_text(json.dumps({"relpath_to_docid": {
                "example.org/one.txt": "1", "example.org/two.txt": "2",
            }}), encoding="utf-8")

            cli = ("--corpus", corpus, "--out-dir", out)
            packed = _run(PIPELINE, *cli, "pack")
            self.assertEqual(packed.returncode, 0, packed.stderr)
            state = json.loads((out / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "packed")
            requests = [json.loads(line) for line in
                        (out / "batches" / "batch_00_input.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(requests), 2)
            self.assertTrue(all(r["body"]["model"] == "gpt-5.4-nano" and
                                r["body"]["reasoning_effort"] == "low" for r in requests))
            again = _run(PIPELINE, *cli, "pack")
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("skipping pack", again.stdout)

            outputs = [
                {"custom_id": request["custom_id"], "response": {"body": {
                    "choices": [{"message": {"content": json.dumps({"sections": []})}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }}}
                for request in requests
            ]
            (out / "batches" / "batch_00_output.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in outputs), encoding="utf-8",
            )
            processed = _run(PIPELINE, *cli, "process,parquet")
            self.assertEqual(processed.returncode, 0, processed.stderr)
            result = pd.read_parquet(out / "data.parquet")
            self.assertEqual(len(result), 2)
            self.assertTrue({"docid", "text", "url", "original_text", "restructured_text"} <= set(result.columns))
            self.assertTrue(result["text"].str.contains("=== DOCUMENT BODY ===").all())
            self.assertEqual(list(result["original_text"]), [doc["text"] for doc in documents])

            exported = _run(BUILD_TREE, "--parquet", out / "data.parquet",
                            "--filename-map", mapping, "--out-dir", tree)
            self.assertEqual(exported.returncode, 0, exported.stderr)
            self.assertIn("=== DOCUMENT BODY ===", (tree / "example.org" / "one.txt").read_text(encoding="utf-8"))
            self.assertIn("=== DOCUMENT BODY ===", (tree / "example.org" / "two.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
