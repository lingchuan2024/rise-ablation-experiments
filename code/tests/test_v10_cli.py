"""Check v10 CLI identities and defaults without corpus or model requests."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = CODE_ROOT.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("v10_cli_test_runner", CODE_ROOT / "scripts" / "run_rise.py")
entry = load_module("v10_cli_test_entry", EXPERIMENT_ROOT / "run_v10.py")
from rise.trajectory import compute_run_config_hash


class ConfigCaptured(Exception):
    pass


class V10CLITests(unittest.TestCase):
    def capture_config(self, arguments=(), *, via_entry=False):
        captured = {}

        def record_config(config):
            captured.update(config)
            raise ConfigCaptured

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "REPO_ROOT", root), \
                 patch.object(runner, "_load_records", return_value=[]), \
                 patch.object(runner, "effective_reasoning_effort", return_value="medium"), \
                 patch.object(runner, "compute_run_config_hash", side_effect=record_config), \
                 patch.object(runner, "make_client") as make_client, \
                 patch.object(sys, "argv", ["run_rise.py", *arguments]), \
                 redirect_stdout(StringIO()):
                with self.assertRaises(ConfigCaptured):
                    if via_entry:
                        with patch.object(entry, "_load_runner", return_value=runner):
                            entry.main(arguments)
                    else:
                        runner.main()
                make_client.assert_not_called()
            outputs = [path.name for path in (root / "runs").iterdir()] if (root / "runs").exists() else []
        return captured, outputs

    def test_default_cli_preserves_baseline_search_and_config_identity(self):
        config, outputs = self.capture_config()
        self.assertEqual(config["extras"]["bm25_k"], 1000)
        self.assertEqual(config["extras"]["bm25_top_n_preview"], 10)
        self.assertEqual(config["extras"]["bm25_snippet_chars"], 100)
        self.assertFalse(any(key.startswith("reranker_") for key in config["extras"]))
        self.assertEqual(len(outputs), 1)
        self.assertFalse(outputs[0].endswith("_v10"))

    def test_disabled_reranker_options_do_not_change_baseline_hash(self):
        base, _ = self.capture_config()
        disabled, _ = self.capture_config(("--no-reranker", "--reranker-model", "unused/model", "--reranker-batch-size", "12"))
        self.assertEqual(compute_run_config_hash(base), compute_run_config_hash(disabled))

    def test_disabled_cli_config_matches_frozen_baseline(self):
        baseline = load_module("v10_frozen_cli_comparison", EXPERIMENT_ROOT.parent / "head_body_v1" / "baseline_snapshot" / "scripts" / "run_rise.py")
        configurations = []
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_rise.py", "--mini-dev", "/same/queries.jsonl", "--index-dir", "/same/index",
                "--filename-map", "/same/map.json", "--bc-plus-docs", "/same/corpus",
                "--out-root", directory,
            ]
            for selected in (baseline, runner):
                def record(config):
                    configurations.append(config)
                    raise ConfigCaptured

                with patch.object(selected, "_load_records", return_value=[]), \
                     patch.object(selected, "effective_reasoning_effort", return_value="medium"), \
                     patch.object(selected, "compute_run_config_hash", side_effect=record), \
                     patch.object(sys, "argv", arguments), redirect_stdout(StringIO()):
                    with self.assertRaises(ConfigCaptured):
                        selected.main()
        self.assertEqual(configurations[0], configurations[1])

    def test_enabled_defaults_are_fixed_and_isolate_cache_and_output(self):
        base, _ = self.capture_config()
        enabled, outputs = self.capture_config(("--reranker",))
        extras = enabled["extras"]
        self.assertEqual(extras["reranker_model"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(extras["reranker_revision"], "main")
        self.assertEqual(extras["reranker_device"], "auto")
        self.assertEqual(extras["reranker_dtype"], "auto")
        self.assertEqual(extras["reranker_batch_size"], 8)
        self.assertEqual(extras["reranker_max_length"], 32768)
        self.assertIs(extras["reranker_enabled"], True)
        self.assertIs(extras["reranker_local_files_only"], False)
        self.assertEqual(extras["reranker_passage_tokens"], 128)
        self.assertEqual(extras["reranker_passage_overlap_tokens"], 32)
        self.assertEqual(extras["reranker_passage_stride_tokens"], 96)
        self.assertEqual(extras["reranker_chunking_version"], "token-overlap-v2")
        self.assertEqual(extras["reranker_candidates"], 100)
        self.assertEqual(extras["reranker_preview"], 10)
        self.assertNotEqual(compute_run_config_hash(base), compute_run_config_hash(enabled))
        self.assertTrue(outputs[0].endswith("_v10"))

    def test_model_runtime_options_are_recorded_before_hashing(self):
        config, _ = self.capture_config((
            "--reranker", "--reranker-model", "test/relevance-model",
            "--reranker-revision", "0123456789abcdef",
            "--reranker-device", "cpu", "--reranker-dtype", "float32",
            "--reranker-batch-size", "3", "--reranker-max-length", "4096",
            "--reranker-local-files-only",
        ))
        expected = {
            "reranker_model": "test/relevance-model", "reranker_revision": "0123456789abcdef",
            "reranker_device": "cpu", "reranker_dtype": "float32", "reranker_batch_size": 3,
            "reranker_max_length": 4096, "reranker_local_files_only": True,
        }
        for key, value in expected.items():
            self.assertEqual(config["extras"][key], value)
        changed, _ = self.capture_config(("--reranker", "--reranker-batch-size", "4"))
        self.assertNotEqual(compute_run_config_hash(config), compute_run_config_hash(changed))

    def test_prompt_revision_separates_new_run_from_previous_v10_results(self):
        current, _ = self.capture_config(("--reranker",))
        self.assertEqual(current["extras"]["reranker_prompt_version"], runner.RERANKER_PROMPT_VERSION)
        previous = {**current, "extras": dict(current["extras"])}
        previous["extras"].pop("reranker_prompt_version")
        self.assertNotEqual(compute_run_config_hash(previous), compute_run_config_hash(current))

    def test_overlap_identity_separates_old_results_even_with_same_prompt_version(self):
        current, _ = self.capture_config(("--reranker",))
        current_hash = compute_run_config_hash(current)
        previous = {**current, "extras": dict(current["extras"])}
        for key in ("reranker_passage_overlap_tokens", "reranker_passage_stride_tokens",
                    "reranker_chunking_version"):
            previous["extras"].pop(key)
        self.assertNotEqual(compute_run_config_hash(previous), current_hash)
        # The chunk contract must affect identity independently of the prompt.
        for key, old_value in (("reranker_passage_overlap_tokens", 0),
                               ("reranker_passage_stride_tokens", 128),
                               ("reranker_chunking_version", "token-disjoint-v1")):
            with self.subTest(key=key):
                changed = {**current, "extras": {**current["extras"], key: old_value}}
                self.assertNotEqual(compute_run_config_hash(changed), current_hash)

    def test_explicit_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "custom-run"
            self.capture_config(("--reranker", "--out-root", str(output)))
            self.assertTrue(output.is_dir())
            self.assertFalse(output.with_name(output.name + "_v10").exists())

    def test_invalid_enabled_settings_are_rejected_before_inputs_or_clients(self):
        invalid = (
            ("--passage-mode",), ("--bm25-k", "100"), ("--bm25-top-n-preview", "11"),
            ("--reranker-batch-size", "0"), ("--reranker-max-length", "0"),
            ("--reranker-model", " "), ("--reranker-revision", ""),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), \
                 patch.object(sys, "argv", ["run_rise.py", "--reranker", *arguments]), \
                 patch.object(runner, "_load_records") as load_records, \
                 patch.object(runner, "make_client") as make_client, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)
                load_records.assert_not_called()
                make_client.assert_not_called()

    def test_disabled_baseline_allows_other_document_search_sizes(self):
        config, _ = self.capture_config(("--bm25-k", "100", "--bm25-top-n-preview", "5"))
        self.assertEqual(config["extras"]["bm25_k"], 100)
        self.assertEqual(config["extras"]["bm25_top_n_preview"], 5)

    def test_entry_defaults_enable_reranker_and_single_concurrency(self):
        self.assertEqual(entry._default_arguments([]), ["--reranker", "--concurrency", "1"])
        arguments = ["--no-reranker", "--concurrency=4", "--structured-docs"]
        self.assertEqual(entry._default_arguments(arguments), arguments)
        arguments = ["--reranker", "--concurrency", "2"]
        self.assertEqual(entry._default_arguments(arguments), arguments)

    def test_entry_points_input_defaults_at_original_checkout(self):
        config, _ = self.capture_config(via_entry=True)
        extras = config["extras"]
        self.assertEqual(extras["index_dir"], str((entry.BASELINE_ROOT / "runs" / "bm25_full").absolute()))
        self.assertEqual(extras["filename_map"], str((entry.BASELINE_ROOT / "runs" / "corpus_filename_docid_map.json").absolute()))
        self.assertEqual(extras["bc_plus_docs"], str((entry.BASELINE_ROOT / "corpus" / "bcp_plus").absolute()))
        self.assertIs(extras["structured_doc_mode"], False)
        self.assertIs(extras["reranker_enabled"], True)

    def test_entry_structured_default_and_explicit_paths(self):
        structured, _ = self.capture_config(("--structured-docs",), via_entry=True)
        self.assertEqual(structured["extras"]["bc_plus_docs"], str((entry.BASELINE_ROOT / "corpus" / "bcp_plus_structured").absolute()))
        overridden, _ = self.capture_config((
            "--structured-docs", "--bc-plus-docs", "/custom/corpus",
            "--index-dir", "/custom/index", "--filename-map", "/custom/map.json",
        ), via_entry=True)
        self.assertEqual(overridden["extras"]["bc_plus_docs"], "/custom/corpus")
        self.assertEqual(overridden["extras"]["index_dir"], "/custom/index")
        self.assertEqual(overridden["extras"]["filename_map"], "/custom/map.json")

    def test_entry_preserves_explicit_query_path_and_restores_argv(self):
        captured = []
        stub = SimpleNamespace(main=lambda: captured.extend(sys.argv[1:]))
        original_argv = sys.argv[:]
        with patch.object(entry, "_load_runner", return_value=stub):
            entry.main(("--mini-dev", "/custom/queries.jsonl", "--no-reranker", "--concurrency", "5"))
        self.assertEqual(captured, ["--mini-dev", "/custom/queries.jsonl", "--no-reranker", "--concurrency", "5"])
        self.assertEqual(sys.argv, original_argv)
        self.assertEqual(stub.DEFAULT_MINI_DEV, entry._default_query_path())

    def test_entry_prefers_original_query_file_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "rise-reproduction"
            experiment = Path(directory) / "experiments" / "v10"
            original = baseline / "data" / "queries_100.jsonl"
            fallback = experiment.parent / "head_body_v4_qwen35_20260930" / "queries_100.jsonl"
            for path in (original, fallback):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            with patch.object(entry, "BASELINE_ROOT", baseline), \
                 patch.object(entry, "EXPERIMENT_ROOT", experiment):
                self.assertEqual(entry._default_query_path(), original)

    def test_entry_uses_existing_fixed_query_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "rise-reproduction"
            experiment = Path(directory) / "experiments" / "v10"
            fallback = experiment.parent / "head_body_v4_qwen35_20260930" / "queries_100.jsonl"
            fallback.parent.mkdir(parents=True)
            fallback.write_text("", encoding="utf-8")
            stub = SimpleNamespace(main=lambda: None)
            with patch.object(entry, "BASELINE_ROOT", baseline), \
                 patch.object(entry, "EXPERIMENT_ROOT", experiment), \
                 patch.object(entry, "_load_runner", return_value=stub):
                entry.main([])
                self.assertEqual(stub.DEFAULT_MINI_DEV, fallback)

    def test_missing_query_defaults_retain_original_path_and_explain_override(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "rise-reproduction"
            experiment = Path(directory) / "experiments" / "v10"
            stub = SimpleNamespace(main=lambda: None)
            error_output = StringIO()
            with patch.object(entry, "BASELINE_ROOT", baseline), \
                 patch.object(entry, "EXPERIMENT_ROOT", experiment), \
                 patch.object(entry, "_load_runner", return_value=stub), \
                 redirect_stderr(error_output):
                entry.main([])
                self.assertEqual(stub.DEFAULT_MINI_DEV, baseline / "data" / "queries_100.jsonl")
                self.assertIn("--mini-dev", error_output.getvalue())
                error_output.seek(0)
                error_output.truncate(0)
                entry.main(["--mini-dev=/custom/queries.jsonl"])
                self.assertEqual(error_output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
