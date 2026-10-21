"""Offline checks and explicit real-host gates; no service startup or cloud purchase."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
import urllib.request

from suite import MODEL, PIN, ROOT, PILOT, REPORTS, digest, file_sha, load, source_hashes, validate_queries, verify_manifest
from runtime_guards import GuardedLocalClient, StopState, atomic_json, instrument_docker


def focused_tests(python):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(PILOT), str(ROOT / "code/src"), str(ROOT / "code/tests")])
    commands = [[python, "-m", "unittest", "discover", "-s", str(PILOT), "-p", "test_*.py", "-v"],
        [python, "-m", "unittest", "discover", "-s", str(ROOT / "code/tests"), "-p", "test_ablation*.py", "-v"],
        [python, "-m", "unittest", "test_rerank_v13", "test_rerank_compact_response", "test_turn_checkpoint", "-v"]]
    for command in commands:
        result = subprocess.run(command, env=environment, cwd=ROOT / "code", capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Offline tests failed:\n" + (result.stdout + result.stderr)[-16000:])
    return {"passed": True, "test_commands": len(commands)}


def offline_check():
    rows = validate_queries()
    sources = source_hashes()
    for name in sources:
        if name.endswith(".py"):
            compile((ROOT / name).read_bytes(), name, "exec")
    tests = focused_tests(sys.executable)
    return {"status": "offline_passed_live_not_checked", "passed": True, "network_requests": 0,
        "models_loaded": False, "live_ready": False, "fixed_questions": len(rows),
        "question_runs": 300, "sources": len(sources), "tests": tests}


def service_identity(config):
    if not config.get("service_config") or not config.get("service_receipt"):
        raise RuntimeError("Provide the live service config and launcher receipt before prepare")
    if file_sha(config["service_config"]) != config["service_config_sha256"] or file_sha(config["service_receipt"]) != config["service_receipt_sha256"]:
        raise RuntimeError("Service identity bytes changed")
    service, receipt = load(config["service_config"]), load(config["service_receipt"])
    if receipt.get("config_sha256") != config["service_config_sha256"]:
        raise RuntimeError("Service launcher receipt belongs to different config")
    pid = receipt.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("A live service PID is required")
    command = [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    expected = service.get("command", [])
    validate_service_command(expected)
    if len(expected) < 2 or command[-len(expected[1:]):] != expected[1:]:
        raise RuntimeError("Live service command differs from the frozen launcher")
    with urllib.request.urlopen(config["base_url"] + "/models", timeout=15) as stream:
        models = json.load(stream)
    matches = [row for row in models.get("data", []) if row.get("id") == MODEL]
    if len(matches) != 1 or matches[0].get("max_model_len") != 262144:
        raise RuntimeError("Exact Qwen3.5 model/context identity check failed")
    return {"passed": True, "model": MODEL, "context_tokens": 262144,
        "pid": pid, "service_config_sha256": config["service_config_sha256"]}


def validate_service_command(command):
    required = {"--served-model-name": MODEL, "--dtype": "bfloat16", "--max-model-len": "262144",
        "--max-num-seqs": "4", "--max-num-batched-tokens": "8192", "--tensor-parallel-size": "1",
        "--cpu-offload-gb": "0", "--reasoning-parser": "qwen35_official", "--tool-call-parser": "qwen3_coder"}
    for flag, expected in required.items():
        if command.count(flag) != 1:
            raise RuntimeError("Missing or repeated frozen service option: " + flag)
        position = command.index(flag)
        if position + 1 >= len(command) or command[position + 1] != expected:
            raise RuntimeError("Frozen v14 service option differs: " + flag)
    if command.count("--gpu-memory-utilization") != 1:
        raise RuntimeError("A frozen GPU memory fraction is required")
    position = command.index("--gpu-memory-utilization")
    if position + 1 >= len(command) or float(command[position + 1]) != 0.90:
        raise RuntimeError("The v14 service GPU memory fraction is 0.90")
    if not all(flag in command for flag in ("--language-model-only", "--enable-auto-tool-choice", "--enforce-eager")):
        raise RuntimeError("Frozen official-thinking/tool service flags differ")
    if "--quantization" in command:
        raise RuntimeError("The experiment requires the inherited BF16 service")


def capacity_evidence(config, sources):
    if not config.get("capacity_report"):
        raise RuntimeError("Fresh measured C4/long-context/reranker capacity evidence is missing")
    binding = load(config["capacity_report"])
    if binding.get("source_sha256") != sources or binding.get("service_config_sha256") != config["service_config_sha256"] or binding.get("service_receipt_sha256") != config["service_receipt_sha256"]:
        raise RuntimeError("Capacity evidence does not bind these exact sources and live service")
    if file_sha(binding["coexistence_report"]) != binding["coexistence_report_sha256"]:
        raise RuntimeError("Measured capacity report bytes changed")
    report = load(binding["coexistence_report"])
    required = {"model": MODEL, "base_url": config["base_url"], "selected_concurrency": 4,
        "context_tokens": 262144, "reranker_revision": PIN, "reranker_device": config["reranker_device"],
        "measured": True, "passed": True, "long_context_and_rerank_coexistence_passed": True,
        "formal_pilot_ready": True}
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("Capacity report has not passed the matching C4 coexistence gate")
    return {"passed": True, "sha256": config["capacity_report_sha256"],
        "certification_scope": report.get("certification_scope")}


def measure_capacity(args):
    # Runs only against an already running, explicitly identified service.
    from suite import settings
    config = settings(args)
    service_identity(config)
    sources = source_hashes()
    folder = REPORTS / ("capacity_" + str(time.time_ns()))
    folder.mkdir(parents=True)
    performance, coexistence = folder / "performance.json", folder / "coexistence.json"
    from suite import runtime_environment
    environment = runtime_environment(config)
    commands = [[args.python, str(PILOT / "concurrency_probe.py"), "--execute", "--include-four",
        "--base-url", config["base_url"], "--output", str(performance)],
        [args.python, str(PILOT / "coexistence_gate.py"), "--execute", "--concurrency", "4",
        "--base-url", config["base_url"], "--probe-report", str(performance),
        "--code-root", str(ROOT / "code"), "--reranker-device", config["reranker_device"], "--output", str(coexistence)]]
    for command in commands:
        subprocess.run(command, env=environment, check=True)
    service_identity(config)
    if sources != source_hashes():
        raise RuntimeError("Source identity changed during capacity measurement")
    binding = {"version": "v14-ablation-capacity-binding-v1", "source_sha256": sources,
        "service_config_sha256": config["service_config_sha256"], "service_receipt_sha256": config["service_receipt_sha256"],
        "coexistence_report": str(coexistence), "coexistence_report_sha256": file_sha(coexistence),
        "finished_unix": time.time()}
    target = folder / "binding.json"
    atomic_json(target, binding)
    return {"status": "capacity_measured", "capacity_report": str(target), "passed": load(coexistence).get("passed") is True}


def sandbox_probe(config):
    import rise.tools as tools
    state = StopState(REPORTS / "preflight_sandbox_pause.json", REPORTS / "preflight_sandbox_events.jsonl")
    original = tools._run_docker_bash
    tools._run_docker_bash = instrument_docker(original, state)
    source = next(Path(config["corpus"]).rglob("*.txt"))
    def one(slot):
        with tempfile.TemporaryDirectory(prefix="ablation-sandbox-", dir=REPORTS) as temporary:
            folder = Path(temporary)
            (folder / "probe.txt").write_text("ABLATION_SANDBOX_READY\n")
            os.link(source, folder / "corpus_document.txt")
            bash = tools.make_bash_tool(folder, enable_sandbox=True)
            result = bash({"command": "rg -n 'ABLATION_SANDBOX_READY' probe.txt && test -r corpus_document.txt", "timeout": 120})
            if state.paused or "1:ABLATION_SANDBOX_READY" not in result or result.startswith("Error:"):
                raise RuntimeError("Actual Docker staging/copy/run path failed")
            return {"slot": slot, "passed": True}
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(one, range(12)))
        return {"passed": True, "concurrency": 4, "waves": 3, "calls": rows}
    finally:
        tools._run_docker_bash = original


def local_resources(config):
    from rise.passage_index import PassageIndex
    from rise.retrieval import load_index
    from rise.reranker_backend import QwenReranker
    index = PassageIndex(config["passage_index"], corpus_root=config["corpus"])
    _, docids = load_index(config["document_index"])
    mapping = load(config["filename_map"])["relpath_to_docid"]
    if len(docids) != 100195 or set(map(str, docids)) != set(map(str, mapping.values())) or not set(mapping).issubset(index._path_to_doc):
        raise RuntimeError("Frozen corpus/document/passage scopes differ")
    for rel in mapping:
        key, path = PurePosixPath(rel), Path(config["corpus"]) / rel
        if key.is_absolute() or ".." in key.parts or path.is_symlink() or not path.is_file():
            raise RuntimeError("Mapped corpus source missing or unsafe")
    backend = QwenReranker(revision=PIN, device=config["reranker_device"], dtype="bfloat16", batch_size=8, local_files_only=True)
    scores = backend.score("What color is the sky?", ["The clear daytime sky is blue.", "A bicycle has two wheels."])
    if len(scores) != 2 or not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError("Real reranker forward failed")
    return {"passed": True, "documents": len(docids), "passages": index.passage_count, "reranker_forward_passed": True}


def native_probe(config):
    from rise.api_retry import chat_with_retry
    from rise.decompose import make_client
    from rise.local_vllm import extract_reasoning
    from rise.tools import bm25_search_tool_spec
    state = StopState(REPORTS / "native_probe_pause.json", REPORTS / "native_probe_events.jsonl")
    client = GuardedLocalClient(make_client(MODEL), MODEL, config["base_url"], REPORTS / "preflight_transport.jsonl", state)
    specs = [bm25_search_tool_spec()]
    messages = [{"role": "system", "content": 'Compatibility check. Call search exactly once with queries ["compatibility probe"], then acknowledge the tool result.'},
        {"role": "user", "content": "Call search before answering."}]
    first = chat_with_retry(client, model=MODEL, messages=messages, tools=specs, tool_choice="required", max_tokens=8192, max_retries=1)
    message = first.choices[0].message
    calls = message.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != "search" or json.loads(calls[0].function.arguments) != {"queries": ["compatibility probe"]}:
        raise RuntimeError("Native tool call/JSON verification failed")
    assistant = {"role": "assistant", "content": message.content, "tool_calls": [call.model_dump() for call in calls]}
    reasoning = extract_reasoning(message)
    if reasoning:
        assistant["reasoning_content"] = reasoning
    messages.extend([assistant, {"role": "tool", "tool_call_id": calls[0].id, "content": '{"status":"ok","items":[]}'},
        {"role": "user", "content": "Reply exactly Compatibility OK without another tool call."}])
    second = chat_with_retry(client, model=MODEL, messages=messages, tools=specs, tool_choice="none", max_tokens=8192, max_retries=1)
    final = second.choices[0].message
    if final.tool_calls or (final.content or "").strip().rstrip(".") != "Compatibility OK":
        raise RuntimeError("Native tool-result continuation failed")
    return {"passed": True, "requests": 2, "part_of_fixed100": False,
        "usage": [value.usage.model_dump() if value.usage else None for value in (first, second)]}


def live_check():
    manifest, config = verify_manifest()
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "manifest_sha256": manifest["manifest_sha256"], "started_unix": time.time()}
    try:
        if Path(sys.executable).resolve() != Path(config["python"]).resolve():
            raise RuntimeError("Run preflight with the prepared Python runtime")
        from suite import runtime_environment
        environment = runtime_environment(config)
        os.environ.clear()
        os.environ.update(environment)
        sys.path.insert(0, str(ROOT / "code/src"))
        report["service"] = service_identity(config)
        report["capacity"] = capacity_evidence(config, manifest["source_sha256"])
        report["offline"] = offline_check()
        report["resources"] = local_resources(config)
        report["sandbox"] = sandbox_probe(config)
        report["native_probe"] = native_probe(config)
        verify_manifest()
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_unix"] = time.time()
        atomic_json(REPORTS / "preflight_live.json", report)
    return report
