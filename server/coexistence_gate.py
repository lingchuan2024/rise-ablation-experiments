#!/usr/bin/env python3
"""One mixed long-context wave with C resident real rerankers; metadata only.

Run only after concurrency_probe.py has finished. --prepare-only never accesses
the network or loads models. Exact BF16 model/context/tool implementation stays
unchanged. This does not certify C simultaneous full262k histories.
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

import concurrency_probe as probe

PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"
LONG_TOKENS = 163840


def calibrate_target(transport, body, nonce, target):
    chars, history = target * 3, []
    for attempt in range(8):
        chars = min(len(body), max(256, chars))
        count = transport.tokenize(probe.messages_for(body, chars, nonce))
        history.append({"attempt": attempt + 1, "actual_input_tokens": count, "fixture_chars": chars})
        if abs(count - target) <= probe.TOLERANCE:
            return chars, history
        changed = int(chars * target / count)
        chars = changed if changed != chars else chars + (1 if count < target else -1)
    raise probe.ProbeError("long_fixture_calibration_exhausted")


def make_passages(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
        max_length=1000 * 96 + 160)
    result = []
    for index in range(1000):
        start, take = index * 96, 128
        for _ in range(8):
            candidate = tokenizer.decode(tokens[start:start + take], skip_special_tokens=False)
            count = len(tokenizer.encode(candidate, add_special_tokens=False))
            if count == 128:
                result.append(candidate)
                break
            take += 128 - count
        else:
            raise probe.ProbeError("cannot_build_exact128_token_passage")
    if len({probe.digest(text) for text in result}) != 1000:
        raise probe.ProbeError("passage_fixture_not_unique")
    return result


def overlap_evidence(telemetry_path, scores):
    running_samples = []
    if telemetry_path.exists():
        for line in telemetry_path.read_text().splitlines():
            try:
                row = json.loads(line)
                active = sum(m["value"] for m in row.get("service_metrics", {}).get("values", [])
                    if m["name"].endswith("num_requests_running"))
                if active > 0:
                    running_samples.append(datetime.fromisoformat(row["at_utc"]).timestamp())
            except (ValueError, KeyError, TypeError):
                continue
    records = [{"worker": score["worker"], "running_llm_metric_samples_during_score":
        sum(score["started_unix"] <= timestamp <= score["finished_unix"] for timestamp in running_samples)} for score in scores]
    return {"per_reranker": records, "all_rerankers_overlap_observed_llm_running":
        bool(records) and all(row["running_llm_metric_samples_during_score"] > 0 for row in records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--rerank-start-delay-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports/concurrency_decision.json")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.rerank_start_delay_seconds < 0 or not math.isfinite(args.rerank_start_delay_seconds):
        parser.error("rerank start delay must be finite and nonnegative")
    transport = probe.Transport(args.base_url, args.timeout)
    config = {"schema_version": "qwen35-long-rerank-coexistence-gate-v1", "model": probe.MODEL,
        "base_url": transport.base_url, "context_tokens": probe.CONTEXT,
        "long_input_tokens_target": LONG_TOKENS, "other_input_tokens_target": probe.TARGET,
        "output_tokens_per_request": probe.OUTPUT_CAP, "reranker_revision": PIN,
        "reranker_device": args.reranker_device, "reranker_dtype": "bfloat16",
        "reranker_batch_size": 8, "rerank_candidates_each": 1000, "passage_tokens": 128,
        "all_selected_concurrency_full262k_tested": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.prepare_only:
        print(json.dumps({**config, "network_requests": 0, "models_loaded": False,
            "measured": False, "passed": False, "formal_pilot_ready": False}, indent=2))
        return 0
    performance = json.loads(args.probe_report.read_text())
    if (performance.get("passed") is not True or performance.get("measured") is not True
            or performance.get("model") != probe.MODEL or performance.get("context_tokens") != probe.CONTEXT
            or performance.get("base_url") != transport.base_url or not performance.get("finished_at_utc")):
        raise ValueError("A finished, measured matching performance report is required; never overlap probes")
    concurrency = args.concurrency or performance["selected_concurrency"]
    if not any(row.get("concurrency") == concurrency and row.get("passed") is True for row in performance.get("candidates", [])):
        raise ValueError("Requested concurrency did not pass the performance sweep")
    output = args.output.resolve()
    telemetry_path = output.with_name(output.stem + ".telemetry.jsonl")
    if output.exists() or telemetry_path.exists():
        raise ValueError("Refusing to overwrite prior capacity artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {**config, "selected_concurrency": concurrency, "performance_selected_concurrency": performance["selected_concurrency"],
        "performance_report": str(args.probe_report.resolve()), "performance_report_sha256": hashlib.sha256(args.probe_report.read_bytes()).hexdigest(),
        "started_at_utc": probe.utc(), "measured": False, "passed": False,
        "long_context_and_rerank_coexistence_passed": False, "formal_pilot_ready": False,
        "answer_quality_evaluated": False, "model_text_or_reasoning_persisted": False}
    with output.open("x") as stream:
        json.dump({**report, "state": "starting"}, stream)
    telemetry = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(args.code_root.resolve() / "src"))
        import torch
        from rise.reranker_backend import QwenReranker
        if not args.reranker_device.startswith("cuda") or not torch.cuda.is_available():
            raise probe.ProbeError("gpu_reranker_required")
        identity = transport.request(transport.base_url + "/models", timeout=15)
        if not any(row.get("id") == probe.MODEL and row.get("max_model_len") == probe.CONTEXT for row in identity.get("data", [])):
            raise probe.ProbeError("service_identity_mismatch")
        telemetry = probe.Telemetry(transport, telemetry_path)
        telemetry.thread.start()
        body, nonce = probe.fixture_body(), os.urandom(16).hex()
        long_chars, long_history = calibrate_target(transport, body, nonce + "-long-calibration", LONG_TOKENS)
        short_chars, short_history = calibrate_target(transport, body, nonce + "-short-calibration", probe.TARGET)
        report["calibration"] = {"long": long_history, "representative": short_history}
        requests = []
        for worker in range(concurrency):
            target, chars = (LONG_TOKENS, long_chars) if worker == 0 else (probe.TARGET, short_chars)
            tag = f"coexistence-{worker + 1}"
            messages = probe.messages_for(body, chars, nonce + "-" + tag)
            count = transport.tokenize(messages)
            if abs(count - target) > probe.TOLERANCE:
                raise probe.ProbeError("coexistence_fixture_token_mismatch")
            requests.append((messages, count, tag))
        backends = [QwenReranker(revision=PIN, device=args.reranker_device, dtype="bfloat16",
            batch_size=8, local_files_only=True) for _ in range(concurrency)]
        passages = make_passages(backends[0].get_tokenizer(), body)
        query = "Which synthetic inventory records describe blue and green counters and their shelf locations?"
        # Preload every model and keep every backend alive until ALL LLMs finish.
        for backend in backends:
            backend.score(query, passages[:2])
        torch.cuda.synchronize(args.reranker_device)
        report["resident_reranker_instances"] = len(backends)
        report["reranker_torch_allocated_bytes_before_wave"] = torch.cuda.memory_allocated(args.reranker_device)
        torch.cuda.reset_peak_memory_stats(args.reranker_device)
        barrier = threading.Barrier(2 * concurrency)
        def scorer(worker):
            barrier.wait(timeout=30)
            if args.rerank_start_delay_seconds:
                time.sleep(args.rerank_start_delay_seconds)
            started = time.time()
            result = {"worker": worker + 1, "started_unix": started, "success": False}
            try:
                scores = backends[worker].score(query, passages)
                torch.cuda.synchronize(args.reranker_device)
                result.update(success=len(scores) == 1000 and all(math.isfinite(float(v)) for v in scores),
                    scores_count=len(scores), usage={key: value for key, value in backends[worker].last_usage.items()
                        if key not in {"sequence_lengths"}})
            except Exception as exc:
                result.update(error_type=type(exc).__name__,
                    oom_detected="out of memory" in str(exc).lower() or "outofmemory" in str(exc).lower())
            result["finished_unix"] = time.time()
            result["elapsed_seconds"] = result["finished_unix"] - started
            return result
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2 * concurrency) as pool:
            llm_futures = [pool.submit(probe.generate, transport, messages, count, tag, barrier) for messages, count, tag in requests]
            score_futures = [pool.submit(scorer, worker) for worker in range(concurrency)]
            report["llm_requests"] = [future.result() for future in llm_futures]
            report["reranker_requests"] = [future.result() for future in score_futures]
        report["wall_seconds"] = time.perf_counter() - started
        report["measured"] = True
        report["reranker_torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.reranker_device)
        report["reranker_torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.reranker_device)
        report["all_requests_completed_with_reranker_models_resident"] = len(backends) == concurrency
    except probe.ProbeError as exc:
        report.update(exc.metadata)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,
            oom_detected="out of memory" in str(exc).lower() or "outofmemory" in str(exc).lower())
    finally:
        if telemetry:
            report["telemetry"] = telemetry.finish()
        if report.get("reranker_requests"):
            report["overlap_evidence"] = overlap_evidence(telemetry_path, report["reranker_requests"])
        passed = (report["measured"] and report.get("resident_reranker_instances") == concurrency
            and all(row["success"] for row in report.get("llm_requests", []))
            and all(row["success"] for row in report.get("reranker_requests", []))
            and report.get("overlap_evidence", {}).get("all_rerankers_overlap_observed_llm_running") is True)
        report.update(passed=passed, long_context_and_rerank_coexistence_passed=passed, formal_pilot_ready=passed,
            finished_at_utc=probe.utc(), certification_scope="One163840-token input plus C-1representative64k inputs, C resident real BF16 rerankers scoring1000 exact128token passages concurrently. Full262k simultaneous histories NOT certified.")
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, output)
    print(json.dumps({"output": str(output), "selected_concurrency": concurrency,
        "passed": report["passed"], "formal_pilot_ready": report["formal_pilot_ready"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
