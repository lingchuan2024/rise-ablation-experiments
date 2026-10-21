#!/usr/bin/env python3
"""Bounded, metadata-only local Qwen performance probe. No network without --execute.

Example:
  python concurrency_probe.py --prepare-only --base-url http://127.0.0.1:8000/v1
  python concurrency_probe.py --execute --base-url http://127.0.0.1:8000/v1 \
      --output reports/concurrency_probe.json
This certifies only the measured 64k workload, never long-context/rerank coexistence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

MODEL = "Qwen3.5-35B-A3B"
CONTEXT = 262144
TARGET = 65536
TOLERANCE = 64
OUTPUT_CAP = 128
VERSION = "qwen35-metadata-concurrency-probe-v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ProbeError(Exception):
    def __init__(self, kind, *, status=None, oom=False):
        self.metadata = {"error_type": kind, "http_status": status, "oom_detected": oom}
        super().__init__(kind)


class Transport:
    def __init__(self, base_url, timeout, api_key_env=None):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base-url must be an explicit HTTP(S) endpoint without credentials/query")
        self.base_url = base_url.rstrip("/")
        self.root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key_env:
            value = os.environ.get(api_key_env)
            if not value:
                raise ValueError("Requested API key environment variable is unavailable")
            self.headers["Authorization"] = "Bearer " + value

    def request(self, url, payload=None, *, timeout=None, raw=False):
        request = urllib.request.Request(url, headers=self.headers,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as stream:
                body = stream.read()
        except urllib.error.HTTPError as exc:
            # An error may echo the prompt. Never persist its text.
            body = exc.read()
            lower = body.lower()
            raise ProbeError("http_error", status=exc.code,
                oom=b"out of memory" in lower or b"outofmemory" in lower) from None
        except Exception as exc:
            raise ProbeError(type(exc).__name__) from None
        if raw:
            return body.decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            raise ProbeError("invalid_response_json") from None

    def tokenize(self, messages):
        result = self.request(self.root_url + "/tokenize", {
            "model": MODEL, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": True}})
        if result.get("max_model_len") != CONTEXT:
            raise ProbeError("context_mismatch")
        count = result.get("count")
        if type(count) is not int or count < 1:
            raise ProbeError("invalid_tokenize_count")
        return count


def fixture_body():
    # Unique neutral synthetic records, not repeated prose or benchmark answers.
    records = []
    for number in range(1, 16001):
        code = hashlib.sha256(f"neutral-record-{number}".encode()).hexdigest()[:16]
        records.append(f"Record {number:06d}: item code {code}; sector {number % 97}; "
            f"shelf {(number * 17) % 389}; blue counters {(number * 31) % 997}; "
            f"green counters {(number * 43) % 991}; route {(number * 11) % 199}; "
            f"inspection day {1 + number % 28}; sample mass {1 + (number * 7) % 400} grams.\n")
    return "".join(records)


def messages_for(body, chars, request_tag):
    return [{"role": "system", "content": "Read the neutral synthetic inventory. "
            "Return a brief summary of the inventory format. Do not call tools. "
            "This is a performance fixture, not a research question."},
        {"role": "user", "content": "Unique request nonce: " + request_tag + "\n" +
         body[:chars] + "\nSummarize the types of fields in this synthetic inventory briefly."}]


def calibrate(transport, body, run_nonce):
    chars = min(len(body), TARGET * 3)
    history = []
    for attempt in range(8):
        count = transport.tokenize(messages_for(body, chars, run_nonce + "-calibration"))
        history.append({"attempt": attempt + 1, "fixture_chars": chars, "actual_input_tokens": count})
        if abs(count - TARGET) <= TOLERANCE:
            return chars, history
        updated = int(chars * TARGET / count)
        if updated == chars:
            updated += 1 if count < TARGET else -1
        chars = max(256, min(len(body), updated))
    raise ProbeError("fixture_token_calibration_exhausted_8_attempts")


def response_metadata(response):
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    finish = choices[0].get("finish_reason") if choices else None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    passed = (response.get("model") == MODEL and type(prompt) is int and prompt > 0
        and type(completion) is int and completion > 0 and finish in {"stop", "length"})
    return {"success": passed, "reported_model": response.get("model"),
        "prompt_tokens": prompt, "completion_tokens": completion, "finish_reason": finish,
        "length_is_allowed_only_for_performance_probe": finish == "length",
        "answer_quality_evaluated": False,
        "error_type": None if passed else "response_metadata_contract_failed"}


def generate(transport, messages, actual_tokens, request_tag, barrier=None):
    if barrier:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    metadata = {"request_tag": request_tag, "actual_tokenized_input_tokens": actual_tokens,
        "fixture_sha256": digest(json.dumps(messages, ensure_ascii=False)), "requested_output_tokens": OUTPUT_CAP}
    try:
        response = transport.request(transport.base_url + "/chat/completions", {
            "model": MODEL, "messages": messages, "max_tokens": OUTPUT_CAP,
            "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5,
            "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}, "stream": False})
        metadata.update(response_metadata(response))
        # Never return or persist message content, tool arguments or reasoning.
        del response
    except ProbeError as exc:
        metadata.update(success=False, **exc.metadata)
    except Exception as exc:
        metadata.update(success=False, error_type=type(exc).__name__)
    elapsed = time.perf_counter() - started
    metadata["elapsed_seconds"] = elapsed
    count = metadata.get("completion_tokens")
    metadata["completion_tokens_per_second"] = count / elapsed if type(count) is int else None
    return metadata


METRIC = re.compile(r"^(vllm:[A-Za-z0-9_:]+)(\{[^\n]*\})?\s+([^\s]+)")
METRIC_PARTS = ("requests_running", "requests_waiting", "cache_usage", "preempt", "prompt_tokens_total",
    "generation_tokens_total", "time_to_first_token", "time_per_output_token", "inter_token_latency",
    "e2e_request_latency", "request_queue_time", "request_prefill_time", "request_decode_time")


class Telemetry:
    def __init__(self, transport, path):
        self.transport, self.path = transport, path
        self.stop_event = threading.Event()
        self.stream = path.open("x")
        self.samples = 0
        self.metrics_available_samples = 0
        self.peak_gpu_used_mib = {}
        self.max_running = self.max_waiting = None
        self.thread = threading.Thread(target=self.run, name="metadata-telemetry", daemon=True)

    @staticmethod
    def nvidia(query):
        try:
            result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=False)
            return {"available": result.returncode == 0, "rows": list(csv.reader(result.stdout.splitlines())) if result.returncode == 0 else [], "exit_code": result.returncode}
        except Exception as exc:
            return {"available": False, "error_type": type(exc).__name__, "rows": []}

    def sample(self):
        record = {"at_utc": utc()}
        metrics = []
        try:
            raw = self.transport.request(self.transport.root_url + "/metrics", timeout=3, raw=True)
            for line in raw.splitlines():
                match = METRIC.match(line)
                if match and any(part in match.group(1) for part in METRIC_PARTS):
                    try:
                        value = float(match.group(3))
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        metrics.append({"name": match.group(1), "labels": match.group(2) or "", "value": value})
            record["service_metrics"] = {"available": bool(metrics), "values": metrics}
        except ProbeError as exc:
            record["service_metrics"] = {"available": False, **exc.metadata, "values": []}
        record["gpu"] = self.nvidia("--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu")
        record["gpu_processes"] = self.nvidia("--query-compute-apps=pid,process_name,used_memory")
        return record

    def run(self):
        while not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                record = self.sample()
                self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.stream.flush()
                self.samples += 1
                self.metrics_available_samples += int(record["service_metrics"]["available"])
                for row in record["gpu"].get("rows", []):
                    try:
                        index, used = row[0].strip(), float(row[3])
                        self.peak_gpu_used_mib[index] = max(self.peak_gpu_used_mib.get(index, 0), used)
                    except (IndexError, ValueError):
                        pass
                for metric in record["service_metrics"]["values"]:
                    if metric["name"].endswith("num_requests_running"):
                        self.max_running = max(self.max_running or 0, metric["value"])
                    if metric["name"].endswith("num_requests_waiting"):
                        self.max_waiting = max(self.max_waiting or 0, metric["value"])
            except Exception as exc:
                self.stream.write(json.dumps({"at_utc": utc(), "telemetry_error_type": type(exc).__name__}) + "\n")
                self.stream.flush()
            self.stop_event.wait(max(0.0, 1.0 - (time.perf_counter() - started)))

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=12)
        if not self.thread.is_alive():
            self.stream.close()
        return {"path": str(self.path), "samples": self.samples,
            "service_metrics_available_samples": self.metrics_available_samples,
            "service_metrics_missing": self.metrics_available_samples == 0,
            "sampled_peak_gpu_used_mib": self.peak_gpu_used_mib,
            "observed_max_running": self.max_running, "observed_max_waiting": self.max_waiting,
            "sample_interval_target_seconds": 1, "peak_is_sampled_not_hardware_maximum": True}


def select(candidates):
    valid = [row for row in candidates if row.get("passed")]
    if not valid:
        return None
    best = max(row["completion_tokens_per_second"] for row in valid)
    return min(row["concurrency"] for row in valid if row["completion_tokens_per_second"] >= best * 0.95)


def execute(args, config):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path = output.with_name(output.stem + ".telemetry.jsonl")
    if output.exists() or telemetry_path.exists():
        raise ValueError("Refusing to overwrite existing probe artifacts")
    # Reserve output before any request so a second invocation cannot duplicate work.
    with output.open("x") as stream:
        json.dump({**config, "state": "starting", "measured": False, "passed": False}, stream)
    report = {**config, "started_at_utc": utc(), "measured": False, "passed": False,
        "selected_concurrency": None, "candidates": [], "performance_probe_only": True,
        "long_context_and_rerank_coexistence_passed": False, "formal_pilot_ready": False,
        "model_text_or_reasoning_persisted": False, "answer_quality_evaluated": False}
    telemetry = None
    try:
        transport = Transport(args.base_url, args.timeout, args.api_key_env)
        telemetry = Telemetry(transport, telemetry_path)
        telemetry.thread.start()
        identity = transport.request(transport.base_url + "/models", timeout=15)
        matches = [row for row in identity.get("data", []) if row.get("id") == MODEL]
        if len(matches) != 1 or matches[0].get("max_model_len", CONTEXT) != CONTEXT:
            raise ProbeError("model_or_context_identity_mismatch")
        body, nonce = fixture_body(), uuid.uuid4().hex
        chars, history = calibrate(transport, body, nonce)
        report["token_calibration"] = history
        report["tokenize_validation_requests"] = 0
        def fixture(tag):
            messages = messages_for(body, chars, nonce + "-" + tag)
            count = transport.tokenize(messages)
            report["tokenize_validation_requests"] += 1
            if abs(count - TARGET) > TOLERANCE or count + OUTPUT_CAP + 256 > CONTEXT:
                raise ProbeError("fixture_input_budget_mismatch")
            return messages, count
        warm_messages, warm_tokens = fixture("warmup")
        report["warmup"] = generate(transport, warm_messages, warm_tokens, "warmup")
        if not report["warmup"]["success"]:
            raise ProbeError("warmup_failed")
        for concurrency in config["candidate_concurrencies"]:
            row = {"concurrency": concurrency, "waves": [], "passed": False}
            report["candidates"].append(row)
            for wave_index in range(2):
                prepared = []
                for worker in range(concurrency):
                    tag = f"c{concurrency}-w{wave_index + 1}-r{worker + 1}"
                    messages, count = fixture(tag)
                    prepared.append((messages, count, tag))
                barrier = threading.Barrier(concurrency)
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(generate, transport, messages, count, tag, barrier) for messages, count, tag in prepared]
                    results = [future.result() for future in futures]
                elapsed = time.perf_counter() - started
                completion = sum(r.get("completion_tokens") or 0 for r in results)
                row["waves"].append({"wave": wave_index + 1, "wall_seconds": elapsed,
                    "completion_tokens": completion, "completion_tokens_per_second": completion / elapsed,
                    "all_requests_succeeded": all(r["success"] for r in results), "requests": results})
                print(json.dumps({"stage": "wave_finished", "concurrency": concurrency,
                    "wave": wave_index + 1, "wall_seconds": elapsed, "completion_tokens": completion,
                    "all_requests_succeeded": all(r["success"] for r in results)}), flush=True)
                report["measured"] = True
                if not all(r["success"] for r in results):
                    break
            requests = [r for wave in row["waves"] for r in wave["requests"]]
            row["passed"] = len(row["waves"]) == 2 and all(r["success"] for r in requests)
            wall = sum(w["wall_seconds"] for w in row["waves"])
            row["completion_tokens_per_second"] = sum(r.get("completion_tokens") or 0 for r in requests) / wall
            row["median_request_latency_seconds"] = statistics.median(r["elapsed_seconds"] for r in requests)
            row["max_request_latency_seconds"] = max(r["elapsed_seconds"] for r in requests)
            row["finish_reason_counts"] = {key: sum(str(r.get("finish_reason")) == key for r in requests) for key in sorted({str(r.get("finish_reason")) for r in requests})}
            row["failed_requests"] = sum(not r["success"] for r in requests)
            if not row["passed"]:
                report["larger_candidates_skipped_after_failure"] = True
                break
        report["selected_concurrency"] = select(report["candidates"])
        report["passed"] = report["selected_concurrency"] is not None
    except ProbeError as exc:
        report.update(exc.metadata)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        if telemetry:
            report["telemetry"] = telemetry.finish()
        report["finished_at_utc"] = utc()
        report["selection_rule"] = "Smallest passing C within95% of highest measured completion-token throughput; performance fixture only."
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--include-four", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).parent / "reports/concurrency_probe.json"))
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--api-key-env")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    # Constructor validates the URL/env without performing network I/O.
    transport = Transport(args.base_url, args.timeout, args.api_key_env)
    candidates = [1, 2, 3] + ([4] if args.include_four else [])
    config = {"schema_version": VERSION, "model": MODEL, "base_url": transport.base_url,
        "context_tokens": CONTEXT, "probe_input_tokens": TARGET, "input_token_tolerance": TOLERANCE,
        "output_tokens_per_request": OUTPUT_CAP, "candidate_concurrencies": candidates, "waves_per_candidate": 2,
        "maximum_measured_generation_requests": 2 * sum(candidates), "warmup_requests": 1,
        "maximum_calibration_tokenize_requests": 8,
        "maximum_validation_tokenize_requests": 1 + 2 * sum(candidates),
        "profile": "official-thinking-v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.prepare_only:
        print(json.dumps({**config, "mode": "prepare_only", "network_requests": 0,
            "measured": False, "passed": False, "formal_pilot_ready": False}, indent=2))
        return 0
    report = execute(args, config)
    print(json.dumps({"output": str(Path(args.output).resolve()), "passed": report["passed"],
        "selected_concurrency": report["selected_concurrency"], "formal_pilot_ready": False}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
