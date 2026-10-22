#!/usr/bin/env python3
"""Bounded C4 synthetic long-context / shared-reranker coexistence probe.

The two waves certify only their measured mixed workloads. They do not certify
four simultaneous262k histories or32768 generated tokens. No benchmark
question, solver response text, or private reasoning enters this report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT, ROOT / "server", ROOT / "code/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from strict_experiment import runner
import concurrency_probe as probe

VERSION = "strict-qwen35-c4-shared-reranker-capacity-v1"
CONCURRENCY = 4
TOLERANCE = 64
OUTPUT_CAP = 128
WAVES = (("mixed", (163840, 65536, 65536, 65536)),
         ("long_scheduler", (229056, 163840, 163840, 163840)))


def plan(config):
    runner.validate_config(config)
    if config["reranker_dtype"] != "bfloat16":
        raise ValueError("This real capacity probe fixes the v14 BF16 reranker")
    return {"version": VERSION, "model": runner.MODEL, "config_sha256": runner.digest(config),
        "base_url": config["base_url"], "context_tokens": config["context_length"],
        "question_concurrency": CONCURRENCY, "resident_reranker_instances": 1,
        "reranker_leases": CONCURRENCY, "reranker_revision": runner.PIN,
        "reranker_device": config["reranker_device"], "reranker_dtype": config["reranker_dtype"],
        "reranker_batch_size": 8, "rerank_passages_per_wave": 1000,
        "passage_tokens": 128, "output_tokens_per_request": OUTPUT_CAP,
        "formal_output_budget_not_generated_by_probe": config["per_call_max_tokens"],
        "token_calibration_tolerance": TOLERANCE,
        "waves": [{"name": name, "input_token_targets": list(targets)} for name, targets in WAVES],
        "max_generation_requests": 8, "max_calibration_tokenize_requests": 24,
        "scheduler_admission_wait_limit_seconds": 60,
        "rerank_start_rule": "Observe actual /metrics num_requests_running>0 within60seconds, then score; HTTP request start alone is insufficient.",
        "benchmark_questions_used": 0, "answer_quality_evaluated": False,
        "model_text_or_reasoning_persisted": False,
        "all_selected_concurrency_full262k_tested": False,
        "formal32768_output_generation_tested": False,
        "certification_scope": (
            "C4 admissions: one163840+three65536 inputs, then one229056+three163840 inputs; "
            "each wave overlaps one real shared BF16 reranker scoring1000 exact128-token passages. "
            "Requests may queue or be preempted. Four full262k resident histories and32768-token generation are NOT certified.")}


def calibrate(transport, body, nonce, target):
    chars, history = target * 3, []
    for attempt in range(8):
        chars = min(len(body), max(256, chars))
        count = transport.tokenize(probe.messages_for(body, chars, nonce))
        history.append({"attempt": attempt + 1, "fixture_chars": chars, "actual_input_tokens": count})
        if abs(count - target) <= TOLERANCE:
            return chars, history
        changed = int(chars * target / count)
        chars = changed if changed != chars else chars + (1 if count < target else -1)
    raise probe.ProbeError("capacity_fixture_token_calibration_exhausted")


def make_passages(tokenizer, body):
    tokens = tokenizer.encode(body, add_special_tokens=False, truncation=True, max_length=1000 * 96 + 160)
    passages = []
    for index in range(1000):
        start, take = index * 96, 128
        for _ in range(8):
            candidate = tokenizer.decode(tokens[start:start + take], skip_special_tokens=False)
            count = len(tokenizer.encode(candidate, add_special_tokens=False))
            if count == 128:
                passages.append(candidate); break
            take += 128 - count
        else:
            raise probe.ProbeError("cannot_construct_exact128_token_capacity_passage")
    if len({hashlib.sha256(passage.encode()).hexdigest() for passage in passages}) != 1000:
        raise probe.ProbeError("capacity_passages_not_unique")
    return passages


def validate_generation(row, expected_input):
    return (row.get("success") is True and row.get("reported_model") == runner.MODEL
        and row.get("prompt_tokens") == expected_input
        and type(row.get("completion_tokens")) is int
        and 0 < row["completion_tokens"] <= OUTPUT_CAP
        and row.get("finish_reason") in {"stop", "length"})


def generation(transport, messages, count, tag, barrier):
    barrier.wait(timeout=30)
    started = time.time()
    row = probe.generate(transport, messages, count, tag)
    row.update(started_unix=started, finished_unix=time.time())
    row["success"] = validate_generation(row, count)
    row["token_usage_matches_tokenize"] = row.get("prompt_tokens") == count
    return row


def overlap_samples(path, started, finished):
    samples, max_running, max_waiting = 0, 0, 0
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
            timestamp = datetime.fromisoformat(row["at_utc"]).timestamp()
            if not started <= timestamp <= finished:
                continue
            metrics = row.get("service_metrics", {}).get("values", [])
            running = sum(metric["value"] for metric in metrics if metric["name"].endswith("num_requests_running"))
            waiting = sum(metric["value"] for metric in metrics if metric["name"].endswith("num_requests_waiting"))
            samples += int(running > 0)
            max_running, max_waiting = max(max_running, running), max(max_waiting, waiting)
        except (ValueError, KeyError, TypeError):
            continue
    return {"llm_running_samples_during_rerank": samples,
        "observed_running_during_rerank": max_running, "observed_waiting_during_rerank": max_waiting,
        "real_rerank_llm_overlap_verified": samples > 0}


class AdmissionNotObserved(RuntimeError):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__("scheduler_admission_not_observed")


def scheduler_counts(transport, timeout):
    """A request reaching HTTP is not evidence that the engine admitted it."""
    raw = transport.request(transport.root_url + "/metrics", timeout=timeout, raw=True)
    running, waiting = [], []
    for line in raw.splitlines():
        match = probe.METRIC.match(line)
        if not match:
            continue
        name = match.group(1)
        if not name.endswith(("num_requests_running", "num_requests_waiting")):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        (running if name.endswith("num_requests_running") else waiting).append(value)
    return {"at_unix": time.time(), "running": sum(running) if running else None,
        "waiting": sum(waiting) if waiting else None, "running_metric_present": bool(running)}


def wait_for_scheduler_admission(transport, timeout):
    requested = float(timeout)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("Admission wait must be positive and finite")
    limit = min(60.0, requested)
    started, wall_started = time.monotonic(), time.time()
    deadline, observations = started + limit, []
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            row = scheduler_counts(transport, min(3.0, remaining))
        except Exception as exc:
            row = {"at_unix": time.time(), "running": None, "waiting": None,
                "running_metric_present": False, "metrics_error_type": type(exc).__name__}
        observations.append(row)
        elapsed = time.monotonic() - started
        if elapsed <= limit and row["running_metric_present"] and row["running"] > 0:
            return {"scheduler_admission_observed": True,
                "admission_wait_started_unix": wall_started,
                "scheduler_admitted_at_unix": row["at_unix"],
                "admission_wait_seconds": elapsed,
                "admitted_num_requests_running": row["running"],
                "admitted_num_requests_waiting": row["waiting"],
                "admission_observations": observations}
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(1.0, remaining))
    raise AdmissionNotObserved({"scheduler_admission_observed": False,
        "admission_wait_started_unix": wall_started, "scheduler_admitted_at_unix": None,
        "admission_wait_seconds": time.monotonic() - started,
        "admission_wait_limit_seconds": limit, "admission_observations": observations})


def wave_passed(row):
    requests = row.get("llm_requests", [])
    neural = row.get("reranker_request", {})
    return (len(requests) == CONCURRENCY and all(item.get("success") is True for item in requests)
        and neural.get("success") is True and neural.get("scores_count") == 1000
        and neural.get("scheduler_admission_observed") is True
        and isinstance(neural.get("admitted_num_requests_running"), (int, float))
        and neural["admitted_num_requests_running"] > 0
        and row.get("all_four_requests_admitted_concurrently") is True
        and row.get("overlap", {}).get("real_rerank_llm_overlap_verified") is True)


def execute(config, output, *, timeout=1200, rerank_delay=2):
    report = {**plan(config), "passed": False, "measured": False,
        "started_at_utc": probe.utc(), "source_sha256": runner.source_hashes(),
        "service_config_sha256": runner.file_sha(config["model_identity"]["service_config"]),
        "results": [], "failure_stage": "setup"}
    output = Path(output).resolve()
    if output != Path(config["capacity_report"]).resolve():
        raise ValueError("Probe output must equal the report path frozen in config")
    telemetry_path = output.with_name(output.stem + ".telemetry.jsonl")
    if output.exists() or telemetry_path.exists():
        raise ValueError("Preserve existing capacity artifacts; choose a new output")
    runner.atomic_json(output, report, immutable=True)
    telemetry = None
    try:
        os.environ.update(config["runtime_environment"])
        report["service"] = runner.verify_service(config)
        transport = probe.Transport(config["base_url"], timeout)
        telemetry = probe.Telemetry(transport, telemetry_path)
        telemetry.thread.start()
        report["failure_stage"] = "shared_reranker_initialization"
        import torch
        from shared_reranker import install_shared_reranker_factory
        install_shared_reranker_factory()
        from rise.reranker_backend import QwenReranker
        leases = [QwenReranker(model=config["reranker_model"], revision=runner.PIN,
            device=config["reranker_device"], dtype=config["reranker_dtype"],
            batch_size=8, max_length=32768, local_files_only=True) for _ in range(CONCURRENCY)]
        if len({id(lease._backend) for lease in leases}) != 1:
            raise probe.ProbeError("capacity_reranker_is_not_single_shared_backend")
        if not config["reranker_device"].startswith("cuda") or not torch.cuda.is_available():
            raise probe.ProbeError("capacity_requires_actual_cuda_reranker")
        body, nonce = probe.fixture_body(), uuid.uuid4().hex
        passages = make_passages(leases[0].get_tokenizer(), body)
        query = "Which synthetic inventory records describe blue and green counters and shelf locations?"
        leases[0].score(query, passages[:2])
        torch.cuda.synchronize(config["reranker_device"])
        report["reranker_identity"] = leases[0].identity
        report["reranker_torch_allocated_bytes_before_waves"] = torch.cuda.memory_allocated(config["reranker_device"])
        report["failure_stage"] = "fixture_calibration"
        calibrations = {}
        for target in sorted({target for _, targets in WAVES for target in targets}):
            chars, history = calibrate(transport, body, nonce + "-calibrate-" + str(target), target)
            calibrations[target] = chars
            report.setdefault("calibration", {})[str(target)] = history
        for name, targets in WAVES:
            report["failure_stage"] = name
            prepared = []
            for worker, target in enumerate(targets):
                tag = name + "-" + str(worker + 1)
                messages = probe.messages_for(body, calibrations[target], nonce + "-" + tag)
                count = transport.tokenize(messages)
                if abs(count - target) > TOLERANCE or count + config["per_call_max_tokens"] + 256 > config["context_length"]:
                    raise probe.ProbeError("capacity_input_or_formal_output_reservation_mismatch")
                prepared.append((messages, count, tag))
            barrier = threading.Barrier(CONCURRENCY + 1)
            torch.cuda.reset_peak_memory_stats(config["reranker_device"])
            def score():
                barrier.wait(timeout=30)
                neural = {"started_unix": None, "score_started": False, "success": False}
                try:
                    neural.update(wait_for_scheduler_admission(transport, timeout))
                    if rerank_delay:
                        time.sleep(rerank_delay)
                    neural["started_unix"] = time.time()
                    neural["score_started"] = True
                    scores = leases[0].score(query, passages)
                    torch.cuda.synchronize(config["reranker_device"])
                    neural.update(scores_count=len(scores),
                        success=len(scores) == 1000 and all(math.isfinite(float(value)) for value in scores),
                        usage={key: value for key, value in leases[0].last_usage.items() if key != "sequence_lengths"})
                except AdmissionNotObserved as exc:
                    neural.update(exc.metadata, error_type="scheduler_admission_not_observed")
                except Exception as exc:
                    neural.update(error_type=type(exc).__name__,
                        oom_detected="out of memory" in str(exc).lower() or "outofmemory" in str(exc).lower())
                neural["finished_unix"] = time.time()
                neural["elapsed_seconds"] = (neural["finished_unix"] - neural["started_unix"]
                    if neural["score_started"] else 0.0)
                return neural
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=CONCURRENCY + 1) as executor:
                futures = [executor.submit(generation, transport, messages, count, tag, barrier)
                           for messages, count, tag in prepared]
                neural_future = executor.submit(score)
                requests, neural = [future.result() for future in futures], neural_future.result()
            overlap = (overlap_samples(telemetry_path, neural["started_unix"], neural["finished_unix"])
                if neural["score_started"] else {"llm_running_samples_during_rerank": 0,
                    "real_rerank_llm_overlap_verified": False})
            row = {"name": name, "input_token_targets": list(targets), "llm_requests": requests,
                "reranker_request": neural, "overlap": overlap,
                "all_four_requests_admitted_concurrently": max(item["started_unix"] for item in requests) < min(item["finished_unix"] for item in requests),
                "wall_seconds": time.monotonic() - started,
                "reranker_torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(config["reranker_device"]),
                "reranker_torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(config["reranker_device"])}
            row["passed"] = wave_passed(row)
            report["results"].append(row)
            report["measured"] = True
            print(json.dumps({"stage": name, "passed": row["passed"], "wall_seconds": row["wall_seconds"]}), flush=True)
            if not row["passed"]:
                raise probe.ProbeError("capacity_wave_failed")
        report["passed"] = True
        report["failure_stage"] = None
    except probe.ProbeError as exc:
        report.update(exc.metadata)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,
            oom_detected="out of memory" in str(exc).lower() or "outofmemory" in str(exc).lower())
    finally:
        if telemetry is not None:
            report["telemetry"] = telemetry.finish()
            if report["telemetry"].get("service_metrics_missing") or not report["telemetry"].get("sampled_peak_gpu_used_mib"):
                report["passed"] = False
                report.setdefault("error_type", "capacity_telemetry_missing")
        report["finished_at_utc"] = probe.utc()
        report["long_context_and_rerank_coexistence_passed"] = report["passed"]
        report["report_sha256"] = runner.digest(report)
        runner.atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true"); mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", required=True); parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--rerank-delay", type=float, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.rerank_delay) or args.rerank_delay < 0:
        parser.error("timeout/delay must be finite, with timeout>0 and delay>=0")
    config = runner.load(args.config)
    if args.prepare_only:
        print(json.dumps({**plan(config), "network_requests": 0, "models_loaded": False}, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output report path")
    report = execute(config, args.output, timeout=args.timeout, rerank_delay=args.rerank_delay)
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
        "measured": report["measured"], "waves_completed": len(report["results"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
