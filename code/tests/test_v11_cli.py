"""Validate v11 CLI defaults, exclusions, and config hashing without paid calls."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("v11_cli_runner_test", ROOT / "scripts" / "run_rise.py")
entry = load("v11_cli_entry_test", ROOT.parent / "run_v11.py")
from rise.trajectory import compute_run_config_hash


class Captured(Exception):
    pass


class V11CLITests(unittest.TestCase):
    def config(self, args=(), entry_point=False):
        result = {}
        def capture(config):
            result.update(config)
            raise Captured
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "_load_records", return_value=[]), \
             patch.object(runner, "compute_run_config_hash", side_effect=capture), \
             patch.object(runner, "effective_reasoning_effort", return_value="medium"), \
             patch.object(sys, "argv", ["run", "--out-root", directory, *args]), \
             redirect_stdout(StringIO()):
            with self.assertRaises(Captured):
                if entry_point:
                    with patch.object(entry, "_load_runner", return_value=runner):
                        entry.main(["--out-root", directory, *args])
                else:
                    runner.main()
        return result

    def test_v11_entry_defaults_m1000_k100_c2_and_overlap32(self):
        result = self.config(entry_point=True)
        extras = result["extras"]
        self.assertTrue(extras["space_reranker_enabled"])
        self.assertEqual(extras["space_bm25_candidates"], 1000)
        self.assertEqual(extras["space_top_k"], 100)
        self.assertEqual(extras["reranker_passage_tokens"], 128)
        self.assertEqual(extras["reranker_passage_overlap_tokens"], 32)
        self.assertEqual(extras["reranker_passage_stride_tokens"], 96)
        self.assertEqual(entry._default_arguments([])[-2:], ["--concurrency", "2"])

    def test_off_and_v10_modes_preserve_separate_hashes(self):
        base = self.config()
        off = self.config(("--no-space-reranker", "--space-bm25-candidates", "2000", "--space-top-k", "300"))
        self.assertEqual(compute_run_config_hash(base), compute_run_config_hash(off))
        v10 = self.config(("--reranker",))
        v11 = self.config(("--space-reranker",))
        self.assertNotEqual(compute_run_config_hash(v10), compute_run_config_hash(v11))
        self.assertNotIn("space_reranker_enabled", v10["extras"])
        self.assertNotIn("reranker_enabled", v11["extras"])

    def test_candidate_and_view_params_change_identity(self):
        first = self.config(("--space-reranker",))
        second = self.config(("--space-reranker", "--space-bm25-candidates", "1200", "--space-top-k", "150"))
        self.assertNotEqual(compute_run_config_hash(first), compute_run_config_hash(second))
        self.assertEqual(second["extras"]["space_top_k"], 150)

    def test_invalid_combinations_rejected_before_corpus_or_api(self):
        for args in (("--space-reranker", "--reranker"),
                     ("--space-reranker", "--no-sandbox"),
                     ("--space-reranker", "--passage-mode"),
                     ("--space-reranker", "--bm25-k", "100"),
                     ("--space-reranker", "--space-top-k", "1001"),
                     ("--space-reranker", "--space-bm25-candidates", "0")):
            with self.subTest(args=args), \
                 patch.object(sys, "argv", ["run", *args]), \
                 patch.object(runner, "_load_records") as records, \
                 patch.object(runner, "make_client") as client, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main()
                records.assert_not_called()
                client.assert_not_called()

    def test_ark_solver_route_is_recorded_without_a_key(self):
        with patch.dict("os.environ", {"DEEPSEEK_BASE_URL": "https://ark.cn-beijing.volces.com/api/plan/v3", "DEEPSEEK_API_KEY": "test-private-value"}):
            result = self.config(("--space-reranker", "--model", "deepseek-v4.1-flash"))
        self.assertEqual(result["extras"]["reasoning_effort"], "high")
        self.assertEqual(result["extras"]["solver_thinking"], {"type": "enabled"})
        self.assertEqual(result["extras"]["solver_reasoning_effort"], "high")
        self.assertNotIn("test-private-value", str(result))


if __name__ == "__main__":
    unittest.main()
