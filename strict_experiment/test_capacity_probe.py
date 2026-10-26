"""Capacity claims must remain bounded and depend on real usage/overlap."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from strict_experiment import capacity_probe as capacity
from strict_experiment import runner


class CapacityProbeTest(unittest.TestCase):
    def test_plan_covers_mixed_and_long_scheduler_without_claiming_full262k(self):
        plan = capacity.plan(runner.config_template())
        self.assertEqual(plan["max_generation_requests"], 8)
        self.assertEqual(plan["resident_reranker_instances"], 1)
        self.assertEqual(plan["reranker_leases"], 4)
        self.assertEqual(plan["question_concurrency"], 4)
        self.assertEqual(plan["benchmark_questions_used"], 0)
        self.assertFalse(plan["all_selected_concurrency_full262k_tested"])
        self.assertFalse(plan["formal32768_output_generation_tested"])
        for wave in plan["waves"]:
            self.assertEqual(len(wave["input_token_targets"]), 4)
            for target in wave["input_token_targets"]:
                self.assertLessEqual(target + 64 + 32768 + 256, 262144)
        self.assertEqual(plan["waves"][0]["input_token_targets"], [163840, 65536, 65536, 65536])
        self.assertEqual(plan["waves"][1]["input_token_targets"], [229056, 163840, 163840, 163840])

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
        row = {"llm_requests": [{"success": True}] * 4,
            "reranker_request": {"success": True, "scores_count": 1000,
                "scheduler_admission_observed": True, "admitted_num_requests_running": 1},
            "all_four_requests_admitted_concurrently": True,
            "overlap": {"real_rerank_llm_overlap_verified": True}}
        self.assertTrue(capacity.wave_passed(row))
        self.assertFalse(capacity.wave_passed({**row, "overlap": {"real_rerank_llm_overlap_verified": False}}))
        self.assertFalse(capacity.wave_passed({**row, "reranker_request": {"success": True, "scores_count": 2}}))
        self.assertFalse(capacity.wave_passed({**row, "all_four_requests_admitted_concurrently": False}))
        self.assertFalse(capacity.wave_passed({**row, "llm_requests": [{"success": True}] * 3}))
        self.assertFalse(capacity.wave_passed({**row, "reranker_request": {"success": True,
            "scores_count": 1000, "scheduler_admission_observed": False}}))

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
                "all_four_requests_admitted_concurrently": True,
                "overlap": {"real_rerank_llm_overlap_verified": True},
                "reranker_request": {"success": True, "scores_count": 1000,
                    "scheduler_admission_observed": True, "admitted_num_requests_running": 1},
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

    def test_c3_reports_are_not_reinterpreted_as_c4_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, report = self.capacity_fixture(Path(temporary))
            report.update(version="strict-qwen35-c3-shared-reranker-capacity-v2",
                question_concurrency=3, reranker_leases=3)
            for wave in report["results"]:
                wave["llm_requests"] = wave["llm_requests"][:3]
                wave["input_token_targets"] = wave["input_token_targets"][:3]
                wave["all_three_requests_admitted_concurrently"] = wave.pop("all_four_requests_admitted_concurrently")
            self.save_capacity_fixture(config, report)
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

    def admission_fixture(self, samples):
        class Clock:
            elapsed = 0.0
            def monotonic(self): return self.elapsed
            def wall(self): return 1000.0 + self.elapsed
            def sleep(self, delay): self.elapsed += delay
        clock = Clock()
        class Transport:
            root_url = "http://127.0.0.1:8000"
            timeouts = []
            def request(self, url, *, timeout, raw):
                self.timeouts.append(timeout)
                return samples[min(len(self.timeouts) - 1, len(samples) - 1)]
        return clock, Transport()

    def wait_with_clock(self, clock, transport, timeout):
        with mock.patch.object(capacity.time, "monotonic", side_effect=clock.monotonic), \
                mock.patch.object(capacity.time, "time", side_effect=clock.wall), \
                mock.patch.object(capacity.time, "sleep", side_effect=clock.sleep):
            return capacity.wait_for_scheduler_admission(transport, timeout)

    def test_score_admission_waits_for_actual_running_not_http_or_queue(self):
        def metrics(running, waiting):
            return f'vllm:num_requests_running{{model_name="{runner.MODEL}"}} {running}\nvllm:num_requests_waiting {waiting}\n'
        clock, transport = self.admission_fixture([metrics(0, 0), metrics(0, 3), metrics(1, 2)])
        admitted = self.wait_with_clock(clock, transport, 60)
        self.assertTrue(admitted["scheduler_admission_observed"])
        self.assertEqual(admitted["admission_wait_started_unix"], 1000)
        self.assertEqual(admitted["scheduler_admitted_at_unix"], 1002)
        self.assertEqual(admitted["admitted_num_requests_running"], 1)
        self.assertEqual(admitted["admitted_num_requests_waiting"], 2)
        self.assertEqual(len(admitted["admission_observations"]), 3)

    def test_missing_admission_fails_closed_with_sixty_second_hard_cap(self):
        clock, transport = self.admission_fixture(['vllm:num_requests_running 0\nvllm:num_requests_waiting 3\n'])
        with self.assertRaises(capacity.AdmissionNotObserved) as raised:
            self.wait_with_clock(clock, transport, 1200)
        self.assertFalse(raised.exception.metadata["scheduler_admission_observed"])
        self.assertIsNone(raised.exception.metadata["scheduler_admitted_at_unix"])
        self.assertEqual(raised.exception.metadata["admission_wait_seconds"], 60)
        self.assertEqual(clock.elapsed, 60)
        self.assertTrue(all(0 < value <= 3 for value in transport.timeouts))

    def test_missing_metrics_cannot_pass_admission_and_short_timeout_is_respected(self):
        clock, transport = self.admission_fixture(['vllm:num_requests_waiting 3\n'])
        with self.assertRaises(capacity.AdmissionNotObserved) as raised:
            self.wait_with_clock(clock, transport, 2.5)
        self.assertEqual(clock.elapsed, 2.5)
        self.assertFalse(any(row["running_metric_present"] for row in raised.exception.metadata["admission_observations"]))
        self.assertTrue(all(value <= 2.5 for value in transport.timeouts))


if __name__ == "__main__":
    unittest.main()
