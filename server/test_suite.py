from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite
import preflight

ORIGINAL_ROOT = suite.ROOT
sys.path.insert(0, str(ORIGINAL_ROOT / "code/src"))
COHORT_BYTES = (ORIGINAL_ROOT / "data/queries_100.jsonl").read_bytes()


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pilot = self.root / "server"
        self.pilot.mkdir()
        (self.root / "code").mkdir()
        (self.root / "data").mkdir()
        (self.root / "data/queries_100.jsonl").write_bytes(COHORT_BYTES)
        (self.root / "data/historical_v14_evaluation.json").write_text('{"reference":"test"}\n')
        (self.root / "run_ablation.py").write_text("# frozen launcher\n")
        (self.root / "README.md").write_text("fixed300\n")
        self.settings = patch.multiple(suite, ROOT=self.root, PILOT=self.pilot, REPORTS=self.pilot / "reports")
        self.settings.start()
        self.resources = patch.object(suite, "resource_checks", return_value={"test_resource": "immutable"})
        self.capacity = patch.object(preflight, "capacity_evidence", return_value={"passed": True})
        self.resources.start()
        self.capacity.start()
        self.addCleanup(self.resources.stop)
        self.addCleanup(self.capacity.stop)
        self.addCleanup(self.settings.stop)
        self.addCleanup(self.tmp.cleanup)

    def prepared(self):
        args = suite.parser().parse_args(["prepare"])
        suite.prepare(args)
        return suite.verify_manifest()

    def test_schedule_rotates_three_arms_and_has_exact_300_runs(self):
        rows = suite.validate_queries()
        cells = suite.block_schedule(rows)
        self.assertEqual(len(cells), 30)
        self.assertEqual([row["variant"] for row in cells[:6]],
            ["search_paging", "auto_rerank", "no_bash", "auto_rerank", "no_bash", "search_paging"])
        for variant in suite.VARIANTS:
            ids = [qid for cell in cells if cell["variant"] == variant for qid in cell["query_ids"]]
            self.assertEqual(ids, [row["query_id"] for row in rows])
            self.assertEqual(len(set(ids)), 100)

    def test_prepare_freezes_sources_blocks_and_per_arm_outputs(self):
        manifest, config = self.prepared()
        self.assertEqual(manifest["question_runs"], 300)
        self.assertTrue(manifest["first_block_counts_toward_full100"])
        for variant in suite.VARIANTS:
            self.assertEqual(config["arm_outputs"][variant], str(self.root / "runs" / variant))
            arm = suite.load(suite.REPORTS / f"arm_{variant}.json")
            self.assertEqual(arm["manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(suite.status(manifest)["completed_runs"], 0)
        (self.root / "README.md").write_text("changed after freeze\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen source"):
            suite.verify_manifest()

    def test_dataset_and_block_mutations_rejected(self):
        manifest, _ = self.prepared()
        Path(manifest["blocks"]["1"]["path"]).write_text("{}\n")
        with self.assertRaisesRegex(RuntimeError, "block changed"):
            suite.verify_manifest()
        (self.root / "data/queries_100.jsonl").write_bytes(COHORT_BYTES + b"\n")
        with self.assertRaisesRegex(RuntimeError, "dataset bytes"):
            suite.validate_queries()

    def test_cannot_replace_prepared_identity(self):
        self.prepared()
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            suite.prepare(suite.parser().parse_args(["prepare"]))

    def test_no_live_report_means_no_model_or_worker_execution(self):
        self.prepared()
        with patch.object(suite.subprocess, "run") as launch:
            with self.assertRaises(FileNotFoundError):
                suite.execute()
            launch.assert_not_called()

    def test_each_cell_uses_only_one_arm_at_total_concurrency_four(self):
        manifest, config = self.prepared()
        argv = suite.build_argv(config, manifest["schedule"][3], manifest)
        self.assertEqual(argv[argv.index("--concurrency") + 1], "4")
        self.assertEqual(argv[argv.index("--ablation-variant") + 1], "auto_rerank")
        self.assertEqual(argv[argv.index("--out-root") + 1], config["arm_outputs"]["auto_rerank"])
        self.assertEqual(argv[argv.index("--max-turns") + 1], "100")
        self.assertEqual(argv[argv.index("--per-call-max-tokens") + 1], "32000")
        self.assertEqual(argv[argv.index("--api-max-retries") + 1], "3")
        self.assertNotIn("--coerce-final", argv)

    def test_passed_report_for_other_manifest_cannot_start(self):
        manifest, _ = self.prepared()
        suite.atomic_json(suite.REPORTS / "preflight_live.json", {"passed": True, "manifest_sha256": "other"})
        with patch.object(suite.subprocess, "run") as launch:
            with self.assertRaisesRegex(RuntimeError, "Exact-manifest"):
                suite.execute()
            launch.assert_not_called()

    def test_formal_worker_environment_drops_unrelated_credentials_and_overrides(self):
        _, config = self.prepared()
        with patch.dict(suite.os.environ, {"OPENAI_API_KEY": "test-secret", "REASONING_EFFORT": "high", "RISE_LOCAL_VLLM_BASE_URL": "http://wrong"}):
            environment = suite.runtime_environment(config)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(environment["REASONING_EFFORT"], "medium")
        self.assertEqual(environment["RISE_LOCAL_VLLM_BASE_URL"], config["base_url"])


class EntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("ablation_entry_test", ORIGINAL_ROOT / "run_ablation.py")
        cls.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)

    def test_requires_explicit_variant_and_ready_passage_index(self):
        with self.assertRaises(SystemExit):
            self.entry.default_arguments([])
        with self.assertRaises(SystemExit):
            self.entry.default_arguments(["--ablation-variant", "no_bash"])

    def test_defaults_match_v14_and_independent_rerank_cannot_be_disabled(self):
        args = self.entry.default_arguments(["--ablation-variant", "search_paging", "--space-passage-index", "/index"])
        self.assertIn("--independent-rerank", args)
        self.assertIn("--structured-docs", args)
        self.assertEqual(args[args.index("--model") + 1], suite.MODEL)
        self.assertEqual(args[args.index("--reranker-revision") + 1], suite.PIN)
        with self.assertRaises(SystemExit):
            self.entry.default_arguments(["--no-independent-rerank", "--help"])


class CapacityAndServiceTests(unittest.TestCase):
    def test_capacity_report_must_bind_current_service_sources_and_c4(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "coexistence.json"
            report.write_text(json.dumps({"model": suite.MODEL, "base_url": "http://127.0.0.1:8000/v1",
                "selected_concurrency": 4, "context_tokens": 262144, "reranker_revision": suite.PIN,
                "reranker_device": "cuda:0", "measured": True, "passed": True,
                "long_context_and_rerank_coexistence_passed": True, "formal_pilot_ready": True}))
            binding = root / "binding.json"
            sources = {"code.py": "source-hash"}
            binding.write_text(json.dumps({"source_sha256": sources, "service_config_sha256": "svc",
                "service_receipt_sha256": "launcher", "coexistence_report": str(report),
                "coexistence_report_sha256": suite.file_sha(report)}))
            config = {"capacity_report": str(binding), "capacity_report_sha256": suite.file_sha(binding),
                "service_config_sha256": "svc", "service_receipt_sha256": "launcher",
                "base_url": "http://127.0.0.1:8000/v1", "reranker_device": "cuda:0"}
            self.assertTrue(preflight.capacity_evidence(config, sources)["passed"])
            with self.assertRaisesRegex(RuntimeError, "exact sources"):
                preflight.capacity_evidence(config, {"code.py": "changed"})

    def test_service_configuration_rejects_nonmatching_budget_or_dtype(self):
        command = ["vllm", "serve", "/model", "--served-model-name", suite.MODEL,
            "--dtype", "bfloat16", "--max-model-len", "262144", "--max-num-seqs", "4",
            "--max-num-batched-tokens", "8192", "--tensor-parallel-size", "1", "--cpu-offload-gb", "0",
            "--reasoning-parser", "qwen35_official", "--tool-call-parser", "qwen3_coder",
            "--gpu-memory-utilization", "0.90", "--language-model-only", "--enable-auto-tool-choice", "--enforce-eager"]
        preflight.validate_service_command(command)
        changed = list(command)
        changed[changed.index("--max-num-seqs") + 1] = "8"
        with self.assertRaisesRegex(RuntimeError, "option differs"):
            preflight.validate_service_command(changed)


class RerankerGuardTests(unittest.TestCase):
    def test_native_errors_pause_even_if_service_turns_them_into_json(self):
        import runtime_guards
        from rise.reranker_backend import QwenReranker
        from rise.rerank_v13 import RerankService
        with tempfile.TemporaryDirectory() as tmp:
            state = runtime_guards.StopState(Path(tmp) / "pause.json", Path(tmp) / "events.jsonl")
            def score(self, query, passages):
                if query == "invalid":
                    raise ValueError("bad model argument")
                raise RuntimeError("CUDA out of memory")
            with patch.object(QwenReranker, "score", score), patch.object(QwenReranker, "get_tokenizer", lambda self: object()), patch.object(RerankService, "_append_diagnostic", lambda self, record: None):
                runtime_guards.install_reranker_hooks(state)
                backend = QwenReranker()
                with self.assertRaises(ValueError):
                    backend.score("invalid", ["text"])
                self.assertFalse(state.paused)
                try:
                    backend.score("valid", ["text"])
                except Exception:
                    public_result = '{"status":"error"}'
                self.assertEqual(json.loads(public_result)["status"], "error")
                self.assertTrue(state.paused)
                with self.assertRaises(runtime_guards.SandboxPaused):
                    state.raise_if_paused()
                self.assertEqual(suite.load(Path(tmp) / "pause.json")["first_failure"]["kind"], "reranker_backend_failed")


class SolverGuardTests(unittest.TestCase):
    def test_api_retry_exhaustion_pauses_before_returning_terminal_row(self):
        import runtime_guards
        import rise.agent as agent
        import rise.decompose as decompose
        import rise.rise_agent as rise_agent
        import rise.tools as tools
        for termination in ("api_error: transport unavailable", "wall_clock_timeout", "context_limit", "max_turns", "done"):
            with self.subTest(termination=termination), tempfile.TemporaryDirectory() as tmp:
                state = runtime_guards.StopState(Path(tmp) / "pause.json", Path(tmp) / "events.jsonl")
                def result(*args, **kwargs):
                    return SimpleNamespace(terminated_by=termination)
                with patch.object(agent, "chat_with_retry", lambda *args, **kwargs: None), patch.object(decompose, "make_client", lambda *args, **kwargs: None), patch.object(rise_agent, "run_rise_agent", result), patch.object(rise_agent, "make_bash_tool", lambda *args, **kwargs: None), patch.object(tools, "_run_docker_bash", lambda *args, **kwargs: None):
                    runtime_guards.install_runtime_hooks(state, suite.MODEL, "http://127.0.0.1:8000/v1", Path(tmp) / "transport.jsonl")
                    if termination.startswith("api_error"):
                        with self.assertRaises(runtime_guards.SandboxPaused):
                            rise_agent.run_rise_agent(question="probe", query_id=70)
                        self.assertTrue(state.paused)
                        failure = suite.load(Path(tmp) / "pause.json")["first_failure"]
                        self.assertEqual(failure["kind"], "solver_api_failed")
                        self.assertEqual(failure["termination_kind"], "api_error")
                        self.assertNotIn("transport unavailable", json.dumps(failure))
                    else:
                        self.assertEqual(rise_agent.run_rise_agent(question="probe", query_id=70).terminated_by, termination)
                        self.assertFalse(state.paused)
    def test_tokenizer_loading_value_error_is_infrastructure_not_parameter_error(self):
        import runtime_guards
        from rise.reranker_backend import QwenReranker
        from rise.rerank_v13 import RerankService
        with tempfile.TemporaryDirectory() as tmp:
            state = runtime_guards.StopState(Path(tmp) / "pause.json", Path(tmp) / "events.jsonl")
            def tokenizer(self):
                raise ValueError("invalid tokenizer cache")
            with patch.object(QwenReranker, "score", lambda self, query, passages: []), patch.object(QwenReranker, "get_tokenizer", tokenizer), patch.object(RerankService, "_append_diagnostic", lambda self, record: None):
                runtime_guards.install_reranker_hooks(state)
                with self.assertRaises(ValueError):
                    QwenReranker().get_tokenizer()
                self.assertTrue(state.paused)


if __name__ == "__main__":
    unittest.main()
