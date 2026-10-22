#!/usr/bin/env python3
"""Administrative B-only restart epoch; frozen experiment files stay unchanged.

The only serving identity differences permitted after a reboot are PID and
process start ticks (plus their derived identity digest). This module never
starts/stops a model, schedules A, edits a checkpoint by hand, or discards an
unaccounted successful response. Private backups remain in ignored reports/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code/src"))
from strict_experiment import runner, deployment_identity as identity

VERSION = "strict-B-restart-epoch-v1"
LIFECYCLE_FIELDS = {"pid", "process_start_ticks", "identity_sha256"}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def static_identity(receipt):
    require(receipt.get("identity_sha256") == identity.digest({k: v for k, v in receipt.items() if k != "identity_sha256"}), "Invalid serving identity digest")
    return {key: value for key, value in receipt.items() if key not in LIFECYCLE_FIELDS}


def assert_lifecycle_only(previous, actual):
    require(static_identity(previous) == static_identity(actual), "Restart changed static model/argv/parser/template/package/source identity")
    require(type(actual.get("pid")) is int and actual["pid"] > 0
            and type(actual.get("process_start_ticks")) is int and actual["process_start_ticks"] >= 0,
            "Invalid actual process lifecycle")
    return identity.digest(static_identity(actual))


def epoch_directory(path):
    path = Path(path).resolve()
    require(path.is_relative_to(runner.REPORTS.resolve() / "operations"), "Epoch reports/private backups must remain inside ignored experiment reports/operations")
    path.mkdir(parents=True, exist_ok=True)
    return path


def a_snapshot():
    """Read-only integrity map; no A prompt or question is evaluated."""
    root = runner.RUNS / "A"
    directories = ("_per_query", "_checkpoints", "api_ledgers", "request_contracts", "_traces")
    return {str(path.relative_to(root)): runner.file_sha(path)
        for directory in directories for path in sorted((root / directory).rglob("*")) if path.is_file()}


def host_capacity_evidence(config):
    """Reuse measured capacity only for the same observed GPU SKU/memory/image.

    Full userland serving software is checked separately by lifecycle identity.
    The old capacity telemetry did not record a kernel driver version: the new
    observed driver is recorded explicitly rather than inventing an old value.
    """
    capacity = runner.load(config["capacity_report"])
    telemetry = Path(capacity["telemetry"]["path"])
    old_gpu = None
    with telemetry.open() as stream:
        for line in stream:
            record = json.loads(line)
            rows = record.get("gpu", {}).get("rows", [])
            if rows:
                old_gpu = [(row[1].strip(), int(float(row[2]))) for row in rows]
                break
    require(old_gpu, "Original capacity GPU telemetry is missing")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    rows = list(csv.reader(result.stdout.splitlines()))
    measured = [{"name": row[0].strip(), "memory_total_mib": int(float(row[1])), "driver_version": row[2].strip()} for row in rows]
    require([(row["name"], row["memory_total_mib"]) for row in measured] == old_gpu,
            "Restart GPU SKU/memory differs from the measured C4 capacity host")
    require(config["sandbox_backend"] == "docker", "This epoch requires the original Docker sandbox")
    environment = {**os.environ, **config["runtime_environment"]}
    image = environment["RISE_DOCKER_IMAGE"]
    actual_image = subprocess.run(["/usr/bin/docker", "image", "inspect", image, "--format", "{{.Id}}"],
        env=environment, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    require(actual_image == image, "Restart Docker image differs from the frozen image")
    return {"passed": True, "original_capacity_sha256": runner.file_sha(config["capacity_report"]),
        "original_telemetry_sha256": runner.file_sha(telemetry), "gpu": measured,
        "docker_image_id": actual_image, "driver_history": "new driver recorded; kernel driver was not a field of old capacity telemetry",
        "capacity_scope": capacity["certification_scope"], "long_capacity_probe_repeated": False}


def create_epoch(manifest, epoch_dir, pid):
    epoch_dir = epoch_directory(epoch_dir)
    path = epoch_dir / "b_epoch.json"
    require(not path.exists(), "Preserve the existing restart epoch")
    config = manifest["config"]
    service_path = config["model_identity"]["service_config"]
    previous = runner.load(service_path)["actual_identity"]
    actual = identity.collect_service_identity(service_path, pid)
    static_sha = assert_lifecycle_only(previous, actual)
    payload = {"version": VERSION, "arm": "B", "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"], "service_config_sha256": runner.file_sha(service_path),
        "previous_identity_sha256": previous["identity_sha256"], "actual_identity": actual,
        "static_identity_sha256": static_sha, "allowed_lifecycle_fields": sorted(LIFECYCLE_FIELDS),
        "host_capacity_equivalence": host_capacity_evidence(config), "a_before": a_snapshot(),
        "operator_source_sha256": runner.file_sha(Path(__file__)), "created_unix": time.time()}
    payload["epoch_sha256"] = runner.digest(payload)
    runner.atomic_json(path, payload, immutable=True)
    return payload


def verify_epoch(manifest, epoch_dir, config=None):
    epoch = runner.load(Path(epoch_dir) / "b_epoch.json")
    require(epoch.get("epoch_sha256") == runner.digest({k: v for k, v in epoch.items() if k != "epoch_sha256"}), "Restart epoch digest changed")
    require(epoch.get("version") == VERSION and epoch.get("arm") == "B"
            and epoch.get("manifest_sha256") == manifest["manifest_sha256"]
            and epoch.get("config_sha256") == manifest["config_sha256"], "Restart epoch belongs to another experiment")
    require(config is None or config == manifest["config"], "Formal config cannot be replaced during resume")
    require(epoch["operator_source_sha256"] == runner.file_sha(Path(__file__)), "Restart administrative driver changed")
    service_path = manifest["config"]["model_identity"]["service_config"]
    require(runner.file_sha(service_path) == epoch["service_config_sha256"], "Frozen service configuration bytes changed")
    previous = runner.load(service_path)["actual_identity"]
    actual = identity.collect_service_identity(service_path, epoch["actual_identity"]["pid"])
    require(actual == epoch["actual_identity"], "New epoch process or software drifted after verification")
    require(assert_lifecycle_only(previous, actual) == epoch["static_identity_sha256"], "Restart static identity differs")
    return {"verified": True, "version": identity.VERSION, "identity_sha256": actual["identity_sha256"],
        "pid": actual["pid"], "process_start_ticks": actual["process_start_ticks"],
        "model_root": actual["model_root"], "packages": actual["packages"],
        "restart_epoch_sha256": epoch["epoch_sha256"]}


@contextmanager
def bind_epoch(manifest, epoch_dir):
    original = identity.verify_state
    identity.verify_state = lambda config: verify_epoch(manifest, epoch_dir, config)
    try:
        yield
    finally:
        identity.verify_state = original


def verify_service_b(manifest, epoch_dir):
    """The administrative tokenizer probe also uses B; no A contract is sent."""
    config, contract = manifest["config"], manifest["contracts"]["B"]
    serving = verify_epoch(manifest, epoch_dir, config)
    payload = {"model": runner.MODEL, "messages": [
        {"role": "system", "content": contract["system_prompt"]},
        {"role": "user", "content": "Compatibility check only."}],
        "tools": contract["tool_specs"], "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True}}
    url = config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
    with urlopen(Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=30) as stream:
        tokenized = json.load(stream)
    require(type(tokenized.get("count")) is int and tokenized["count"] > 0
        and tokenized.get("max_model_len") == config["context_length"], "B serving tokenizer/context differs")
    return {"deployment_identity": serving, "arm": "B", "probe_prompt_tokens": tokenized["count"],
        "context_length": tokenized["max_model_len"]}


def short_preflight(manifest, epoch_dir):
    """Two synthetic B requests plus unchanged B tools; zero A requests."""
    from strict_experiment import runtime
    from rise.api_retry import chat_with_retry
    from rise.decompose import make_client
    from rise.local_vllm import extract_reasoning
    config, contract = manifest["config"], manifest["contracts"]["B"]
    os.environ.update(config["runtime_environment"])
    epoch_dir = epoch_directory(epoch_dir)
    report_path = epoch_dir / "b_short_preflight.json"
    require(not report_path.exists(), "Preserve the existing short preflight report")
    with bind_epoch(manifest, epoch_dir):
        serving = verify_service_b(manifest, epoch_dir)
        searcher, doc_ids, mapping = runtime.resources(config)
        with tempfile.TemporaryDirectory(prefix="b-only-probe-", dir=epoch_dir) as temporary:
            folder = Path(temporary); work = folder / "work"; work.mkdir()
            service = runtime.make_service(config, work, folder / "rerank", mapping)
            tools = runtime.build_tools("B", config=config, searcher=searcher, doc_ids=doc_ids,
                mapping=mapping, corpus=Path(config["corpus"]), working_dir=work, rerank_service=service)
            tools["search"]({"queries": ["strict execution compatibility probe"]})
            imported = sorted(str(path.relative_to(work)) for path in work.rglob("*.txt"))
            require(imported, "B compatibility search imported no documents")
            require(tools["bash"]({"command": "printf B_RESTART_READY"}) == "B_RESTART_READY", "Original Docker bash failed")
            require("ripgrep" in tools["bash"]({"command": "rg --version | head -n 1"}), "Original rg is unavailable")
            ranked = json.loads(service({"query": "strict execution compatibility probe", "file_paths": [imported[0]],
                "bm25_candidates": 2, "top_k": 2, "limit": 2}))
            require(ranked.get("status") == "ok" and ranked.get("scored_passages", 0) > 0
                and all(math.isfinite(item["reranker_score"]) for item in ranked.get("items", [])), "Real original reranker failed")
            query = {"query_id": 0, "question": "Compatibility probe only: call read with file_path=" + json.dumps(imported[0]) + ", offset=0, limit=1; then finish with B_RESTART_READY."}
            client = runtime.RequestRecorder(make_client(runner.MODEL), arm="B", query=query,
                config=config, contract=contract, output_dir=epoch_dir / "synthetic_B")
            messages = [{"role": "system", "content": contract["system_prompt"]},
                {"role": "user", "content": contract["user_prompt_template"].format(question=query["question"])}]
            first = chat_with_retry(client, model=runner.MODEL, messages=messages, tools=contract["tool_specs"],
                tool_choice={"type": "function", "function": {"name": "read"}}, max_tokens=2048, max_retries=1)
            calls = first.choices[0].message.tool_calls or []
            require(len(calls) == 1 and calls[0].function.name == "read", "Native B read call is incomplete")
            args = json.loads(calls[0].function.arguments)
            require(args.get("file_path") == imported[0], "Native B probe changed source")
            result = tools["read"](args); require(not result.startswith("Error:"), "Native B read failed")
            assistant = {"role": "assistant", "content": first.choices[0].message.content or "", "tool_calls": [call.model_dump() for call in calls]}
            reasoning = extract_reasoning(first.choices[0].message)
            if reasoning: assistant["reasoning_content"] = reasoning
            messages += [assistant, {"role": "tool", "tool_call_id": calls[0].id, "content": result}]
            second = chat_with_retry(client, model=runner.MODEL, messages=messages, tools=contract["tool_specs"],
                tool_choice="none", max_tokens=2048, max_retries=1)
            require(second.choices[0].finish_reason == "stop" and bool(second.choices[0].message.content)
                and client._shared["verified"] == 2, "Native B result continuation failed")
    report = {"passed": True, "arm": "B", "a_requests": 0, "cohort_questions_used": 0,
        "manifest_sha256": manifest["manifest_sha256"], "epoch_sha256": runner.load(epoch_dir / "b_epoch.json")["epoch_sha256"],
        "service": serving, "native_calls": 2, "usage": [response.usage.model_dump() if response.usage else None for response in (first, second)],
        "reranker_summary": service.summary}
    runner.atomic_json(report_path, report, immutable=True)
    return report


def audit_entries(manifest, audit_path):
    audit = runner.load(audit_path)
    require(audit.get("passed") is True, "B restart requires the passed audit; preserve any prior blocked audit")
    require(audit.get("manifest_sha256") == manifest["manifest_sha256"]
            and audit.get("config_sha256") == manifest["config_sha256"], "B restart audit belongs to another config")
    return {item["query_id"]: item for item in audit.get("unfinished_admitted", [])}


def partial_paths(qid):
    root = runner.RUNS / "B"
    return {"checkpoint": root / "_checkpoints" / f"qid_{qid}.json",
        "ledger": root / "api_ledgers" / f"qid_{qid}.jsonl",
        "request_contract": root / "request_contracts" / f"qid_{qid}.json",
        "pause": root / "_infrastructure" / f"qid_{qid}.json"}


def verify_partial_audit(manifest, query, entries):
    qid = query["query_id"]; paths = partial_paths(qid)
    if not any(path.exists() for path in paths.values()):
        return None
    item = entries.get(qid)
    require(item is not None and item.get("decision") == "resume_same_checkpoint" and not item.get("problems"), "Partial B query lacks an approved, reconciled checkpoint audit")
    hashes = item.get("file_sha256", {})
    require(all(hashes.get(name) == (runner.file_sha(path) if path.exists() else None) for name, path in paths.items()), "Partial B artifacts changed after audit")
    require(paths["checkpoint"].is_file(), "Partial B query cannot start over without its checkpoint")
    require(item.get("successful_ledger_tail_not_in_checkpoint", {}).get("rows") == 0, "Unaccounted or unknown successful ledger tail must be reconciled before resume")
    charge = item.get("budget_charge_seconds")
    require(isinstance(charge, (int, float)) and math.isfinite(charge) and charge >= 0, "Partial B lost-window elapsed budget is unknown")
    return item


@contextmanager
def budget_overlays(epoch_dir, items):
    """Preserve original CP bytes, charge only verified lost pre-shutdown time."""
    from rise.turn_checkpoint import TurnCheckpoint
    original = TurnCheckpoint.load
    overlays = {}
    for qid, item in items.items():
        path = partial_paths(qid)["checkpoint"]
        target = Path(epoch_dir) / "private_checkpoint_backups" / f"qid_{qid}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Preserve the prior checkpoint backup; create a fresh audit/epoch for another interrupted recovery")
        shutil.copyfile(path, target); target.chmod(0o600)
        require(runner.file_sha(target) == item["file_sha256"]["checkpoint"], "Checkpoint backup differs from the audited bytes")
        overlays[str(path.resolve())] = item
    if not any(item["budget_charge_seconds"] > 0 for item in items.values()):
        # This reboot's api_error checkpoints were saved after the graceful
        # shutdown window. Their original elapsed time already includes it.
        # Preserve bytes and use the original loader without any overlay.
        yield
        return
    def load(checkpoint):
        state = original(checkpoint)
        item = overlays.get(str(checkpoint.path.resolve()))
        if item is not None and state is not None:
            require(runner.file_sha(checkpoint.path) == item["file_sha256"]["checkpoint"], "Checkpoint changed before applying its one-time elapsed overlay")
            state = copy.deepcopy(state)
            state["elapsed_seconds"] += item["budget_charge_seconds"]
        return state
    TurnCheckpoint.load = load
    try:
        yield
    finally:
        TurnCheckpoint.load = original


def run_b(manifest, epoch_dir, audit_path):
    epoch_dir = epoch_directory(epoch_dir)
    preflight = runner.load(epoch_dir / "b_short_preflight.json")
    epoch = runner.load(epoch_dir / "b_epoch.json")
    require(preflight.get("passed") is True and preflight.get("epoch_sha256") == epoch["epoch_sha256"], "B restart requires its matching short preflight")
    require(a_snapshot() == epoch["a_before"], "A artifacts changed since the B-only epoch was created")
    entries = audit_entries(manifest, audit_path)
    rows = runner.validate_queries(); pending, partial = [], {}
    require(len(rows) == 100 and manifest["config"]["concurrency"] == 4, "B-only resume requires the frozen100/C4 configuration")
    for query in rows:
        path = runner.terminal_path("B", query["query_id"])
        if path.exists():
            runner.validate_terminal(runner.load(path), manifest, "B", query)
        else:
            item = verify_partial_audit(manifest, query, entries)
            if item is not None: partial[query["query_id"]] = item
            pending.append(query)
    pending.sort(key=lambda query: query["query_id"] not in partial)
    with (runner.REPORTS / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with bind_epoch(manifest, epoch_dir), budget_overlays(epoch_dir, partial):
            verify_service_b(manifest, epoch_dir)
            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {}
                def submit(query):
                    # Recheck lifecycle cheaply at every dispatch. Complete
                    # source/package identity is checked before and after run.
                    process = identity._process(epoch["actual_identity"]["pid"])
                    require(all(process[key] == epoch["actual_identity"][key] for key in process), "Restart serving process changed during B-only resume")
                    return pool.submit(runner.run_one, manifest, "B", query, resume_infrastructure=True)
                for _ in range(4):
                    query = next(iterator, None)
                    if query is not None: active[submit(query)] = query["query_id"]
                while active:
                    future = next(as_completed(active)); qid = active.pop(future)
                    result = future.result()
                    print(json.dumps({"arm": "B", "query_id": qid, "status": result["status"]}), flush=True)
                    query = next(iterator, None)
                    if query is not None: active[submit(query)] = query["query_id"]
    require(a_snapshot() == epoch["a_before"], "A artifacts changed during B-only recovery")
    runner.verify_manifest()
    verify_epoch(manifest, epoch_dir)
    for query in rows:
        row = runner.validate_terminal(runner.load(runner.terminal_path("B", query["query_id"])), manifest, "B", query)
        runner.verify_execution_artifacts(manifest, "B", query, row)
    receipt = {"version": VERSION, "status": "B_completed", "b_terminal_count": 100,
        "a_started": 0, "a_artifacts_unchanged": True, "manifest_sha256": manifest["manifest_sha256"],
        "epoch_sha256": epoch["epoch_sha256"], "audit_sha256": runner.file_sha(audit_path),
        "new_b_dispatches": len(pending), "partial_b_resumed": list(partial),
        "unknown_inflight_usage": "retained in the separate audit; never assumed zero",
        "comparison": "B only; reuse of the existing baseline is handled separately"}
    runner.atomic_json(epoch_dir / "b_completion.json", receipt, immutable=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-dir", required=True); parser.add_argument("--manifest-sha256", required=True)
    modes = parser.add_subparsers(dest="command", required=True)
    create = modes.add_parser("create-epoch"); create.add_argument("--pid", type=int, required=True)
    modes.add_parser("preflight")
    run = modes.add_parser("run-b"); run.add_argument("--audit", required=True)
    args = parser.parse_args()
    manifest = runner.verify_manifest()
    require(manifest["manifest_sha256"] == args.manifest_sha256, "Wrong original experiment manifest")
    if args.command == "create-epoch": result = create_epoch(manifest, args.epoch_dir, args.pid)
    elif args.command == "preflight": result = short_preflight(manifest, args.epoch_dir)
    else: result = run_b(manifest, args.epoch_dir, args.audit)
    print(json.dumps({key: value for key, value in result.items() if key in {"passed", "status", "epoch_sha256", "b_terminal_count", "a_started", "a_requests"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
