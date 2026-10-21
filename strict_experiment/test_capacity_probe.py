"""Capacity claims must remain bounded and depend on real usage/overlap."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from strict_experiment import capacity_probe as capacity
from strict_experiment import runner


class CapacityProbeTest(unittest.TestCase):
    def test_plan_covers_mixed_and_long_scheduler_without_claiming_full262k(self):
        plan = capacity.plan(runner.config_template())
        self.assertEqual(plan["max_generation_requests"], 6)
        self.assertEqual(plan["resident_reranker_instances"], 1)
        self.assertEqual(plan["reranker_leases"], 3)
        self.assertEqual(plan["benchmark_questions_used"], 0)
        self.assertFalse(plan["all_selected_concurrency_full262k_tested"])
        self.assertFalse(plan["formal32768_output_generation_tested"])
        for wave in plan["waves"]:
            self.assertEqual(len(wave["input_token_targets"]), 3)
            for target in wave["input_token_targets"]:
                self.assertLessEqual(target + 64 + 32768 + 256, 262144)
        self.assertEqual(plan["waves"][0]["input_token_targets"], [163840, 65536, 65536])
        self.assertEqual(plan["waves"][1]["input_token_targets"], [229056, 163840, 163840])

    def test_generation_requires_exact_model_input_usage_and_bounded_output(self):
        row = {"success": True, "reported_model": runner.MODEL, "prompt_tokens": 163840,
            "completion_tokens": 128, "finish_reason": "length"}
        self.assertTrue(capacity.validate_generation(row, 163840))
        for key, value in (("reported_model", "other"), ("prompt_tokens", 163839),
                ("completion_tokens", 129), ("completion_tokens", 0), ("finish_reason", "error")):
            with self.subTest(key=key, value=value):
                changed = {**row, key: value}
                self.assertFalse(capacity.validate_generation(changed, 163840))

    def test_wave_cannot_pass_merely_because_models_were_loaded(self):
        row = {"llm_requests": [{"success": True}] * 3,
            "reranker_request": {"success": True, "scores_count": 1000},
            "all_three_requests_admitted_concurrently": True,
            "overlap": {"real_rerank_llm_overlap_verified": True}}
        self.assertTrue(capacity.wave_passed(row))
        self.assertFalse(capacity.wave_passed({**row, "overlap": {"real_rerank_llm_overlap_verified": False}}))
        self.assertFalse(capacity.wave_passed({**row, "reranker_request": {"success": True, "scores_count": 2}}))
        self.assertFalse(capacity.wave_passed({**row, "all_three_requests_admitted_concurrently": False}))

    def test_overlap_requires_observed_running_llm_inside_actual_score_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            def record(at, running, waiting):
                return {"at_utc": datetime.fromtimestamp(at, timezone.utc).isoformat(),
                    "service_metrics": {"values": [
                        {"name": "vllm:num_requests_running", "value": running},
                        {"name": "vllm:num_requests_waiting", "value": waiting}]}}
            path.write_text("\n".join(json.dumps(row) for row in [record(10, 3, 0), record(20, 0, 3), record(30, 1, 2)]))
            self.assertFalse(capacity.overlap_samples(path, 15, 25)["real_rerank_llm_overlap_verified"])
            observed = capacity.overlap_samples(path, 25, 35)
            self.assertTrue(observed["real_rerank_llm_overlap_verified"])
            self.assertEqual(observed["observed_running_during_rerank"], 1)
            self.assertEqual(observed["observed_waiting_during_rerank"], 2)

    def capacity_fixture(self, folder):
        config = runner.config_template()
        config["capacity_report"] = str(folder / "capacity.json")
        service = folder / "service.json"
        service.write_text('{"pid":132,"gpu_memory_utilization":0.9}')
        config["model_identity"]["service_config"] = str(service)
        report = {**capacity.plan(config), "passed": True, "measured": True,
            "long_context_and_rerank_coexistence_passed": True,
            "source_sha256": runner.source_hashes(), "service_config_sha256": runner.file_sha(service),
            "telemetry": {"service_metrics_missing": False, "sampled_peak_gpu_used_mib": {"0": 77000}},
            "results": []}
        for name, targets in capacity.WAVES:
            report["results"].append({"name": name, "input_token_targets": list(targets),
                "all_three_requests_admitted_concurrently": True,
                "overlap": {"real_rerank_llm_overlap_verified": True},
                "reranker_request": {"success": True, "scores_count": 1000},
                "llm_requests": [{"success": True, "reported_model": runner.MODEL,
                    "actual_tokenized_input_tokens": target, "prompt_tokens": target,
                    "completion_tokens": 128, "finish_reason": "length"} for target in targets]})
        return config, report

    def save_capacity_fixture(self, config, report):
        report.pop("report_sha256", None)
        report["report_sha256"] = runner.digest(report)
        runner.atomic_json(config["capacity_report"], report)

    def test_prepare_capacity_gate_binds_source_service_config_and_actual_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config, report = self.capacity_fixture(folder)
            self.save_capacity_fixture(config, report)
            self.assertTrue(runner.verify_capacity_report(config)["passed"])
            report["results"][1]["llm_requests"][0]["prompt_tokens"] -= 1
            self.save_capacity_fixture(config, report)
            with self.assertRaisesRegex(RuntimeError, "actual token workload"):
                runner.verify_capacity_report(config)
            config, report = self.capacity_fixture(folder)
            report["source_sha256"]["server/concurrency_probe.py"] = "wrong-source"
            self.save_capacity_fixture(config, report)
            with self.assertRaisesRegex(RuntimeError, "another config, source, service"):
                runner.verify_capacity_report(config)

    def test_missing_or_stale_service_capacity_cannot_admit_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, report = self.capacity_fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "measured capacity report"):
                runner.verify_capacity_report(config)
            self.save_capacity_fixture(config, report)
            Path(config["model_identity"]["service_config"]).write_text('{"pid":999}')
            with self.assertRaisesRegex(RuntimeError, "another config, source, service"):
                runner.verify_capacity_report(config)

    def test_actual_overlap_and_vram_evidence_are_required_even_with_passed_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config, report = self.capacity_fixture(folder)
            report["results"][0]["overlap"]["real_rerank_llm_overlap_verified"] = False
            self.save_capacity_fixture(config, report)
            with self.assertRaisesRegex(RuntimeError, "capacity wave did not pass"):
                runner.verify_capacity_report(config)
            config, report = self.capacity_fixture(folder)
            report["telemetry"]["sampled_peak_gpu_used_mib"] = {}
            self.save_capacity_fixture(config, report)
            with self.assertRaisesRegex(RuntimeError, "VRAM telemetry"):
                runner.verify_capacity_report(config)


if __name__ == "__main__":
    unittest.main()
