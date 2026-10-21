#!/usr/bin/env python3
"""Freeze and run the official structured RISE / added-rerank comparison.

No command starts a model service or obtains credentials. ``offline-preflight``
never contacts a model. ``prepare`` freezes actual resources; ``live-preflight``
and ``run`` require an already provisioned exact Qwen3.5 service.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports/strict_qwen35"
RUNS = ROOT / "runs/strict_qwen35"
DATASET = ROOT / "data/queries_100.jsonl"
VERSION = "official-rise-plus-rerank-qwen35-v1"
MODEL = "Qwen3.5-35B-A3B"
PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"
DATASET_SHA = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"
ARMS = ("A", "B")
_RECEIPT_CACHE = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value, *, immutable=False):
    """An immutable terminal is created atomically and can never be replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)  # Fails with EEXIST; no correctness-based overwrite.
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_queries():
    if file_sha(DATASET) != DATASET_SHA:
        raise RuntimeError("The original fixed100 dataset bytes changed")
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(rows) != 100 or len({row["query_id"] for row in rows}) != 100:
        raise RuntimeError("The fixed100 cohort is incomplete or duplicated")
    if any(not isinstance(row.get("query"), str) or not row["query"] for row in rows):
        raise RuntimeError("Every fixed question must be present")
    return rows


def config_template():
    return {"version": VERSION, "model": MODEL, "base_url": "http://127.0.0.1:8000/v1",
        "concurrency": 3, "context_length": 262144, "max_turns": 100,
        "per_call_max_tokens": 32768, "wall_clock_seconds": 3600,
        "api_max_retries": 1, "run_max_attempts": 1,
        "coerce_final_on_max_turns": False, "context_level": "level0",
        "read_default_limit": 2000, "bm25_k": 1000,
        "bm25_top_n_preview": 10, "bm25_snippet_chars": 100,
        "sandbox_backend": "bubblewrap", "python": "/REQUIRED/python",
        "document_index": "/REQUIRED/document_index", "filename_map": "/REQUIRED/filename_map.json",
        "corpus": "/REQUIRED/structured_corpus", "passage_index": "/REQUIRED/passage_index_128_32",
        "corpus_manifest": "/REQUIRED/corpus_content_manifest.json",
        "document_index_manifest": "/REQUIRED/document_index_manifest.json",
        "rg_binary": "/REQUIRED/rg", "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
        "reranker_revision": PIN, "reranker_device": "cuda:0", "reranker_dtype": "bfloat16",
        "model_identity": {"repo_id": "Qwen/Qwen3.5-35B-A3B", "revision": "REQUIRED_IMMUTABLE_REVISION",
            "weights_manifest": "/REQUIRED/model_weight_manifest.json",
            "tokenizer_config": "/REQUIRED/tokenizer_config.json",
            "service_config": "/REQUIRED/service_config.json"},
        "runtime_environment": {"RISE_LOCAL_VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "RISE_LOCAL_VLLM_CONTEXT_LENGTH": "262144", "RISE_LOCAL_VLLM_TOKEN_MARGIN": "256",
            "RISE_LOCAL_VLLM_MIN_OUTPUT_TOKENS": "1024", "RISE_LOCAL_VLLM_TIMEOUT": "1200",
            "RISE_QWEN35_PROFILE": "official-thinking-v1", "REASONING_EFFORT": "medium",
            "RISE_BASH_SANDBOX": "bubblewrap", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3",
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}}


def validate_config(config, *, actual_resources=False):
    """Both arms receive the very same object, including all adaptations."""
    required = set(config_template())
    if set(config) != required:
        raise ValueError("Config keys must match the reviewed template exactly")
    fixed = ("version", "model", "concurrency", "context_length", "max_turns", "per_call_max_tokens",
        "wall_clock_seconds", "api_max_retries", "run_max_attempts", "coerce_final_on_max_turns",
        "context_level", "read_default_limit", "bm25_k", "bm25_top_n_preview", "bm25_snippet_chars",
        "reranker_model", "reranker_revision")
    defaults = config_template()
    for key in fixed:
        if type(config[key]) is not type(defaults[key]) or config[key] != defaults[key]:
            raise ValueError("Frozen common configuration changed: " + key)
    url = urlsplit(config["base_url"])
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment or url.path.rstrip("/") != "/v1":
        raise ValueError("Use a credential-free vLLM /v1 endpoint")
    if config["sandbox_backend"] not in {"bubblewrap", "docker"}:
        raise ValueError("The sandbox must be bwrap or Docker for both arms")
    if config["reranker_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Explicit reranker dtype required")
    environment = config["runtime_environment"]
    if not isinstance(environment, dict) or any(not isinstance(v, str) for v in environment.values()):
        raise ValueError("Runtime environment must be a mapping of strings")
    for key, value in defaults["runtime_environment"].items():
        expected = config["base_url"] if key == "RISE_LOCAL_VLLM_BASE_URL" else config["sandbox_backend"] if key == "RISE_BASH_SANDBOX" else value
        if environment.get(key) != expected:
            raise ValueError("Common Qwen/sandbox profile differs: " + key)
    permitted = set(defaults["runtime_environment"]) | {"PATH", "HF_HOME", "CUDA_VISIBLE_DEVICES", "RISE_DOCKER_IMAGE", "DOCKER_HOST", "LANG", "LC_ALL", "RISE_LOCAL_VLLM_SERVICE_SHA256"}
    if not set(environment).issubset(permitted):
        raise ValueError("Unreviewed environment key or credential in config")
    if config["sandbox_backend"] == "docker" and not environment.get("RISE_DOCKER_IMAGE", "").startswith("sha256:"):
        raise ValueError("Docker requires an immutable identical image")
    identity = config["model_identity"]
    if not isinstance(identity, dict) or set(identity) != set(defaults["model_identity"]) or identity["repo_id"] != "Qwen/Qwen3.5-35B-A3B":
        raise ValueError("Exact target model provenance is required")
    if actual_resources and not re.fullmatch(r"[0-9a-f]{40}", identity["revision"]):
        raise ValueError("The actual model must have an immutable source revision")
    return config


def source_hashes():
    files = list((ROOT / "code").rglob("*.py")) + list((ROOT / "strict_experiment").rglob("*.py"))
    files += [ROOT / "server/shared_reranker.py", ROOT / "server/runtime_guards.py"]
    files += list((ROOT / "strict_experiment/upstream").rglob("*.json"))
    files += [DATASET]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(set(files))}


def write_resource_receipt(kind, root, output, *, model_revision=None):
    """Hash actual bytes; this is preparation work, never a readiness assertion."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.is_relative_to(root):
        raise ValueError("Put the receipt outside an existing immutable resource directory")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size, "sha256": file_sha(path)})
    if not files:
        raise ValueError("Cannot attest an empty resource")
    receipt = {"version": "strict-resource-receipt-v1", "kind": kind,
        "root": str(root), "files": files, "content_sha256": digest(files)}
    if kind == "model_weights":
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision or ""):
            raise ValueError("The model receipt requires its exact immutable revision")
        receipt.update(repo_id="Qwen/Qwen3.5-35B-A3B", revision=model_revision)
    atomic_json(output, receipt, immutable=True)
    return {"receipt": str(output), "files": len(files), "content_sha256": receipt["content_sha256"]}


def verify_resource_receipt(path, kind, *, expected_root=None, expected_paths=None, model_revision=None):
    receipt = load(path)
    if receipt.get("version") != "strict-resource-receipt-v1" or receipt.get("kind") != kind:
        raise RuntimeError("A real resource content receipt is required: " + kind)
    root = Path(receipt.get("root", "")).resolve()
    if not root.is_dir() or (expected_root is not None and root != Path(expected_root).resolve()):
        raise RuntimeError("Resource receipt names another directory: " + kind)
    files = receipt.get("files")
    if not isinstance(files, list) or not files or receipt.get("content_sha256") != digest(files):
        raise RuntimeError("Invalid resource file manifest: " + kind)
    if kind == "model_weights" and (receipt.get("repo_id") != "Qwen/Qwen3.5-35B-A3B" or receipt.get("revision") != model_revision):
        raise RuntimeError("The model weight receipt has another model or revision")
    paths, metadata = [], []
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts or PurePosixPath(relative).as_posix() != relative:
            raise RuntimeError("Invalid resource relative path")
        if type(item.get("size")) is not int or item["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise RuntimeError("Invalid resource size/content hash")
        source = root / relative
        if not source.is_file() or source.stat().st_size != item["size"]:
            raise RuntimeError("Resource file missing or changed: " + kind)
        stat = source.stat()
        paths.append(relative)
        metadata.append((relative, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    actual = {source.relative_to(root).as_posix() for source in root.rglob("*") if source.is_file()}
    if len(paths) != len(set(paths)) or set(paths) != actual or (expected_paths is not None and set(paths) != set(expected_paths)):
        raise RuntimeError("Resource receipt does not cover the exact resource file set: " + kind)
    cache_key = (str(Path(path).resolve()), file_sha(path), digest(metadata))
    if cache_key not in _RECEIPT_CACHE:
        for item in files:
            if file_sha(root / item["path"]) != item["sha256"]:
                raise RuntimeError("Actual resource bytes differ from receipt: " + kind)
        after = []
        for relative in paths:
            stat = (root / relative).stat()
            after.append((relative, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        if after != metadata:
            raise RuntimeError("Resource changed while verifying its actual bytes")
        _RECEIPT_CACHE[cache_key] = True
    return {"files": len(files), "content_sha256": receipt["content_sha256"], "actual_bytes_verified": True}


def verify_rg_binary(config):
    """Freeze the binary the unchanged sandbox actually resolves as rg."""
    configured = Path(config["rg_binary"]).resolve()
    if config["sandbox_backend"] == "docker":
        environment = {**os.environ, **config["runtime_environment"]}
        result = subprocess.run(["/usr/bin/docker", "run", "--rm", "--pull=never",
            "--network=none", "--read-only", "--entrypoint", "/bin/bash",
            environment["RISE_DOCKER_IMAGE"], "-c", 'sha256sum "$(command -v rg)"'],
            env=environment, check=True, capture_output=True, text=True, timeout=120)
        if result.stdout.split()[0] != file_sha(configured):
            raise RuntimeError("Configured rg is not the actual Docker image binary")
    else:
        bundled = ROOT / "code/bin/rg"
        candidates = [bundled] if bundled.is_file() and os.access(bundled, os.X_OK) else [Path(directory) / "rg" for directory in ("/usr/local/bin", "/usr/bin", "/bin")]
        actual = next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
        if actual is None or not os.path.samefile(configured, actual):
            raise RuntimeError("Configured rg is not the binary resolved inside the unchanged sandbox")
    return file_sha(configured)


def contracts():
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code/src"))
    module = importlib.import_module("strict_experiment.strict_prompt")
    module.verify_contracts()
    return {arm: module.build_contract(arm) for arm in ARMS}


def resource_identity(config):
    validate_config(config, actual_resources=True)
    for key in ("python", "filename_map", "corpus_manifest", "document_index_manifest", "rg_binary"):
        if not Path(config[key]).is_file():
            raise RuntimeError("Required resource file missing: " + key)
    for key in ("document_index", "corpus", "passage_index"):
        if not Path(config[key]).is_dir():
            raise RuntimeError("Required resource directory missing: " + key)
    if not os.access(config["python"], os.X_OK) or not os.access(config["rg_binary"], os.X_OK):
        raise RuntimeError("Python and rg must be executable")
    model = config["model_identity"]
    for key in ("weights_manifest", "tokenizer_config", "service_config"):
        if not Path(model[key]).is_file():
            raise RuntimeError("Required model provenance file missing: " + key)
    passage_path = Path(config["passage_index"]) / "manifest.json"
    passage = load(passage_path)
    configuration = passage.get("configuration", passage.get("identity", {}))
    chunk = configuration.get("chunking", {})
    if passage.get("status") != "ready" or (chunk.get("passage_tokens"), chunk.get("overlap_tokens")) != (128, 32) or configuration.get("tokenizer", {}).get("revision") != PIN:
        raise RuntimeError("The fixed128/32 published passage index differs")
    mapping = load(config["filename_map"]).get("relpath_to_docid", {})
    if len(mapping) != 100195:
        raise RuntimeError("The structured corpus map must cover100195 documents")
    content = {"corpus": verify_resource_receipt(config["corpus_manifest"], "corpus",
            expected_root=config["corpus"], expected_paths=mapping),
        "document_index": verify_resource_receipt(config["document_index_manifest"], "document_index", expected_root=config["document_index"]),
        "model_weights": verify_resource_receipt(model["weights_manifest"], "model_weights", model_revision=model["revision"])}
    rg_sha256 = verify_rg_binary(config)
    return {"model_revision": model["revision"],
        "model_files": {key: file_sha(model[key]) for key in ("weights_manifest", "tokenizer_config", "service_config")},
        "resource_files": {key: file_sha(config[key]) for key in ("python", "filename_map", "corpus_manifest", "document_index_manifest", "rg_binary")},
        "passage_manifest_sha256": file_sha(passage_path), "mapped_documents": len(mapping),
        "passage_corpus_content_sha256": passage.get("corpus_content_sha256"),
        "passage_configuration_sha256": configuration.get("sha256"),
        "actual_content_receipts": content, "actual_sandbox_rg_sha256": rg_sha256}


def offline_preflight(config=None):
    rows = validate_queries()
    bundle = contracts()
    if config is not None:
        validate_config(config)
    for name in source_hashes():
        if name.endswith(".py"):
            compile((ROOT / name).read_bytes(), name, "exec")
    return {"passed": True, "live_ready": False, "network_requests": 0,
        "version": VERSION, "arms": list(ARMS), "fixed100": len(rows),
        "pilot_query_ids": [row["query_id"] for row in rows[:10]],
        "config_sha256": digest(config) if config is not None else None,
        "contract_sha256": {arm: bundle[arm]["contract_sha256"] for arm in ARMS}}


def prepare(config):
    if (REPORTS / "run_manifest.json").exists():
        raise RuntimeError("This experiment already has a frozen manifest; preserve it")
    offline_preflight(config)
    rows = validate_queries()
    bundle = contracts()
    resources = resource_identity(config)
    if any((RUNS / arm).exists() for arm in ARMS):
        raise RuntimeError("Existing arm output cannot be bound as a fresh run")
    from strict_experiment.analyze import rubric_document
    payload = {"version": VERSION, "config": config, "config_sha256": digest(config),
        "contracts": bundle, "cohort_sha256": DATASET_SHA,
        "query_ids": [row["query_id"] for row in rows],
        "pilot_query_ids": [row["query_id"] for row in rows[:10]],
        "source_sha256": source_hashes(), "resource_identity": resources,
        "question_runs": 200, "historical_results_reused": False,
        "selection": "all100 in original file order; pilot=first10; no outcome selection",
        "analysis_contract": {"version": "strict-qwen35-analysis-v1", "rubric_sha256": digest(rubric_document())},
        "scheduling": "paired original question order; A then B; global C3",
        "terminal_policy": "immutable; natural errors count; only interrupted infrastructure may resume same checkpoint"}
    manifest = {**payload, "manifest_sha256": digest(payload)}
    for arm in ARMS:
        atomic_json(REPORTS / "contracts" / (arm + ".json"), bundle[arm], immutable=True)
    atomic_json(REPORTS / "run_manifest.json", manifest, immutable=True)
    return manifest


def verify_manifest(*, resources=True):
    manifest = load(REPORTS / "run_manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("version") != VERSION or digest(payload) != manifest.get("manifest_sha256"):
        raise RuntimeError("Invalid frozen experiment manifest")
    config = validate_config(manifest["config"], actual_resources=resources)
    if digest(config) != manifest.get("config_sha256") or source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("Frozen code or common configuration changed")
    if contracts() != manifest["contracts"] or DATASET_SHA != manifest["cohort_sha256"]:
        raise RuntimeError("Prompt/tools/cohort changed after freeze")
    validate_queries()
    if resources and resource_identity(config) != manifest["resource_identity"]:
        raise RuntimeError("Frozen runtime/model/corpus/index identity changed")
    return manifest


def verify_service(config):
    """Read the actual endpoint; no model alias or missing service is accepted."""
    url = config["base_url"].rstrip("/")
    request = Request(url + "/models", headers={"Accept": "application/json"})
    with urlopen(request, timeout=15) as stream:
        models = json.load(stream)
    ids = [item.get("id") for item in models.get("data", [])]
    if MODEL not in ids:
        raise RuntimeError("The endpoint does not report the authorized exact Qwen3.5 model")
    contract = contracts()["A"]
    payload = {"model": MODEL, "messages": [{"role": "system", "content": contract["system_prompt"]},
        {"role": "user", "content": "Compatibility check only."}],
        "tools": contract["tool_specs"], "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True}}
    request = Request(url.removesuffix("/v1") + "/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as stream:
        tokenization = json.load(stream)
    if type(tokenization.get("count")) is not int or tokenization.get("max_model_len") != config["context_length"]:
        raise RuntimeError("The serving tokenizer/context budget does not match the frozen contract")
    return {"reported_models": ids, "context_length": tokenization["max_model_len"],
        "probe_prompt_tokens": tokenization["count"], "models_response_sha256": digest(models)}


def live_preflight():
    manifest = verify_manifest()
    config = manifest["config"]
    os.environ.update(config["runtime_environment"])
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code/src"))
    from strict_experiment import runtime
    report = {"passed": False, "manifest_sha256": manifest["manifest_sha256"], "started_unix": time.time()}
    try:
        report["service"] = verify_service(config)
        report["runtime"] = runtime.preflight(config=config, expected_contracts=manifest["contracts"])
        if report["runtime"].get("passed") is not True:
            raise RuntimeError("Shared sandbox/native-tool/reranker execution preflight failed")
        verify_manifest()
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_unix"] = time.time()
        atomic_json(REPORTS / "preflight_live.json", report)
    return report


def terminal_path(arm, qid):
    return RUNS / arm / "_per_query" / f"qid_{qid}.json"


def validate_terminal(row, manifest, arm, query):
    payload = {key: value for key, value in row.items() if key != "row_sha256"}
    expected = {"version": VERSION, "arm": arm, "query_id": query["query_id"],
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
        "contract_sha256": manifest["contracts"][arm]["contract_sha256"],
        "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest()}
    if any(row.get(key) != value for key, value in expected.items()) or digest(payload) != row.get("row_sha256"):
        raise RuntimeError("An immutable terminal row does not belong to this frozen experiment")
    if not isinstance(row.get("final_text"), str) or row.get("final_sha256") != hashlib.sha256(row["final_text"].encode()).hexdigest():
        raise RuntimeError("The immutable final answer bytes changed")
    if row.get("model") != MODEL:
        raise RuntimeError("A terminal row reports the wrong model")
    return row


def verify_execution_artifacts(manifest, arm, query, terminal):
    """Check what was actually sent, rather than trusting a completion flag."""
    output = RUNS / arm
    expected = manifest["contracts"][arm]
    record = load(output / "request_contracts" / f"qid_{query['query_id']}.json")
    required = {"arm": arm, "query_id": query["query_id"], "model": MODEL,
        "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
        "system_prompt": expected["system_prompt"], "tool_specs": expected["tool_specs"],
        "system_prompt_sha256": expected["system_prompt_sha256"],
        "tool_specs_sha256": expected["tool_specs_sha256"],
        "contract_sha256": expected["contract_sha256"],
        "common_config_sha256": manifest["config_sha256"]}
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError("The actual sent prompt/tools/config differs from the frozen contract")
    if record.get("user_prompt") != expected["user_prompt_template"].format(question=query["query"]):
        raise RuntimeError("The actual sent user prompt differs from the official template")
    ledger = output / "api_ledgers" / f"qid_{query['query_id']}.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    if not entries or not any(row.get("status") == "ok" for row in entries):
        raise RuntimeError("No actual successful model request is recorded")
    for row in entries:
        if row.get("requested_model") != MODEL or row.get("contract_sha256") != expected["contract_sha256"]:
            raise RuntimeError("An actual API request used another model or contract")
        if row.get("status") == "ok" and row.get("reported_model") != MODEL:
            raise RuntimeError("An actual API response reports another model")
    successful = [row for row in entries if row.get("status") == "ok"]
    if any(not isinstance(row.get("usage"), dict) for row in successful):
        raise RuntimeError("Successful request usage is missing")
    for row in successful:
        usage = row["usage"]
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens")):
            raise RuntimeError("Invalid actual API usage counters")
    input_tokens = sum(row["usage"]["prompt_tokens"] for row in successful)
    output_tokens = sum(row["usage"]["completion_tokens"] for row in successful)
    if input_tokens != terminal.get("input_tokens") or output_tokens != terminal.get("output_tokens"):
        raise RuntimeError("Actual API token ledger differs from terminal totals")
    return {"request_contract_sha256": file_sha(output / "request_contracts" / f"qid_{query['query_id']}.json"),
        "api_ledger_sha256": file_sha(ledger), "recorded_requests": len(entries),
        "input_tokens": input_tokens, "output_tokens": output_tokens}


def pilot_gate(manifest):
    """Only execution validity, never answer correctness, determines promotion."""
    checks = []
    for query in validate_queries()[:10]:
        for arm in ARMS:
            row = validate_terminal(load(terminal_path(arm, query["query_id"])), manifest, arm, query)
            execution = row.get("execution_checks", {})
            if any(execution.get(key) is not True for key in ("contract_verified", "model_verified", "token_ledger_verified")):
                raise RuntimeError("Pilot lacks actual request/model contract verification")
            if str(row.get("terminated_by", "")).startswith(("api_error", "infrastructure")):
                raise RuntimeError("Infrastructure interruptions are unresolved; pilot cannot promote")
            checks.append({"arm": arm, "query_id": query["query_id"], "row_sha256": row["row_sha256"],
                **verify_execution_artifacts(manifest, arm, query, row)})
    return {"passed": True, "manifest_sha256": manifest["manifest_sha256"],
        "checked_rows": checks, "correctness_used": False,
        "criteria": "20 terminal identities, actual prompt/tools/model verified, no unresolved infrastructure"}


def run_one(manifest, arm, query, *, resume_infrastructure=False):
    path = terminal_path(arm, query["query_id"])
    if path.exists():
        validate_terminal(load(path), manifest, arm, query)
        return {"arm": arm, "query_id": query["query_id"], "status": "reused_immutable_terminal"}
    output = RUNS / arm
    pause = output / "_infrastructure" / f"qid_{query['query_id']}.json"
    if pause.exists() and not resume_infrastructure:
        raise RuntimeError("Explicit infrastructure resume is required; natural terminals are never retried")
    if pause.exists() and load(pause).get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("An infrastructure interruption belongs to another experiment")
    checkpoint = output / "_checkpoints" / f"qid_{query['query_id']}.json"
    ledger = output / "api_ledgers" / f"qid_{query['query_id']}.jsonl"
    request_contract = output / "request_contracts" / f"qid_{query['query_id']}.json"
    if any(item.exists() for item in (checkpoint, ledger, request_contract)):
        if not resume_infrastructure:
            raise RuntimeError("A partial query requires explicit infrastructure resume")
        # A lost checkpoint after an admitted request cannot be repaired by
        # rerunning the question from its initial prompt with a fresh budget.
        if (ledger.exists() or request_contract.exists()) and not checkpoint.exists():
            raise RuntimeError("Partial model execution has no checkpoint; a fresh retry is forbidden")
        if checkpoint.exists():
            saved = load(checkpoint)
            expected = {"model": MODEL, "query_id": str(query["query_id"]),
                "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
                "system_prompt_sha256": manifest["contracts"][arm]["system_prompt_sha256"],
                "tool_specs_sha256": manifest["contracts"][arm]["tool_specs_sha256"]}
            if any(saved.get("identity", {}).get(key) != value for key, value in expected.items()):
                raise RuntimeError("Partial checkpoint identity differs from this experiment")
            if not isinstance(saved.get("elapsed_seconds"), (int, float)) or saved["elapsed_seconds"] < 0:
                raise RuntimeError("Partial checkpoint lacks the consumed wall-clock budget")
    from strict_experiment.runtime import run_strict_query
    try:
        # Answers and gold documents are intentionally not passed to the solver.
        solver_query = {"query_id": query["query_id"], "query": query["query"]}
        row = run_strict_query(arm=arm, query=solver_query, config=manifest["config"],
            output_dir=output, expected_contract=manifest["contracts"][arm])
        if str(row.get("terminated_by", "")).startswith(("api_error", "infrastructure")):
            raise RuntimeError("Runtime reported unresolved infrastructure; keep its checkpoint")
        row = {**row, "version": VERSION, "arm": arm, "query_id": query["query_id"],
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][arm]["contract_sha256"]}
        row["final_sha256"] = hashlib.sha256(row["final_text"].encode()).hexdigest()
        row["row_sha256"] = digest(row)
        validate_terminal(row, manifest, arm, query)
        atomic_json(path, row, immutable=True)
        return {"arm": arm, "query_id": query["query_id"], "status": "terminal_saved", "terminated_by": row["terminated_by"]}
    except Exception as exc:
        # No exception text/headers are persisted. A failed infrastructure launch
        # never writes a natural terminal and cannot silently restart a question.
        atomic_json(pause, {"status": "infrastructure_paused", "error_type": type(exc).__name__,
            "arm": arm, "query_id": query["query_id"], "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_required_for_partial_query": True, "unix_time": time.time()})
        raise


def run(stage, *, resume_infrastructure=False):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_manifest()
        preflight = load(REPORTS / "preflight_live.json")
        if preflight.get("passed") is not True or preflight.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("The frozen manifest must pass live preflight before execution")
        config = manifest["config"]
        os.environ.update(config["runtime_environment"])
        verify_service(config)
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code/src"))
        rows = validate_queries()
        if stage == "full":
            gate = pilot_gate(manifest)
            atomic_json(REPORTS / "pilot_gate.json", gate)
        selected = rows[:10] if stage == "pilot" else rows
        for query in selected:
            for arm in ARMS:
                path = terminal_path(arm, query["query_id"])
                if path.exists():
                    validate_terminal(load(path), manifest, arm, query)
        pending = [(arm, query) for query in selected for arm in ARMS if not terminal_path(arm, query["query_id"]).exists()]
        # Submit only C3 at once. On an infrastructure failure, no queued new
        # question starts; active queries may finish their current admitted turn.
        iterator = iter(pending)
        completions = []
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            active = {}
            for _ in range(config["concurrency"]):
                item = next(iterator, None)
                if item:
                    arm, query = item
                    active[executor.submit(run_one, manifest, arm, query, resume_infrastructure=resume_infrastructure)] = item
            while active:
                future = next(as_completed(active))
                active.pop(future)
                completions.append(future.result())
                item = next(iterator, None)
                if item:
                    arm, query = item
                    active[executor.submit(run_one, manifest, arm, query, resume_infrastructure=resume_infrastructure)] = item
        verify_manifest()
        if stage == "pilot":
            atomic_json(REPORTS / "pilot_gate.json", pilot_gate(manifest))
        receipt = {"stage": stage, "status": "completed", "manifest_sha256": manifest["manifest_sha256"],
            "finished_unix": time.time(), "new_terminal_rows": len(completions), "natural_error_reruns": 0}
        atomic_json(REPORTS / (stage + "_completion.json"), receipt)
        return receipt


def status():
    manifest = verify_manifest(resources=False)
    rows = validate_queries()
    report = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"], "arms": {}}
    for arm in ARMS:
        complete, invalid, interrupted = [], [], []
        for query in rows:
            qid = query["query_id"]
            path = terminal_path(arm, qid)
            if path.exists():
                try:
                    validate_terminal(load(path), manifest, arm, query); complete.append(qid)
                except (ValueError, RuntimeError, OSError):
                    invalid.append(qid)
            elif any((RUNS / arm / directory / f"qid_{qid}{extension}").exists()
                    for directory, extension in (("_infrastructure", ".json"), ("_checkpoints", ".json"),
                        ("request_contracts", ".json"), ("api_ledgers", ".jsonl"))):
                interrupted.append(qid)
        report["arms"][arm] = {"terminal": complete, "invalid": invalid, "infrastructure_paused": interrupted,
            "pending": [row["query_id"] for row in rows if row["query_id"] not in complete + invalid + interrupted]}
    report["completed_runs"] = sum(len(value["terminal"]) for value in report["arms"].values())
    return report


def run_all(*, resume_infrastructure=False):
    pilot = run("pilot", resume_infrastructure=resume_infrastructure)
    full = run("full", resume_infrastructure=resume_infrastructure)
    return {"version": VERSION, "status": "completed", "pilot": pilot, "full": full,
        "question_count_per_arm": 100, "question_runs": 200,
        "pilot_results_included_in_formal_total": True, "natural_error_reruns": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template")
    offline = sub.add_parser("offline-preflight"); offline.add_argument("--config")
    prepare_parser = sub.add_parser("prepare"); prepare_parser.add_argument("--config", required=True)
    sub.add_parser("live-preflight"); sub.add_parser("status")
    receipt = sub.add_parser("write-resource-receipt")
    receipt.add_argument("--kind", choices=("corpus", "document_index", "model_weights"), required=True)
    receipt.add_argument("--root", required=True); receipt.add_argument("--output", required=True)
    receipt.add_argument("--model-revision")
    launch = sub.add_parser("run"); launch.add_argument("--stage", choices=("pilot", "full"), required=True)
    launch.add_argument("--resume-infrastructure", action="store_true")
    all_stages = sub.add_parser("run-all"); all_stages.add_argument("--resume-infrastructure", action="store_true")
    args = parser.parse_args()
    if args.command == "template": result = config_template()
    elif args.command == "offline-preflight": result = offline_preflight(load(args.config) if args.config else None)
    elif args.command == "prepare":
        manifest = prepare(load(args.config))
        result = {"status": "prepared_live_preflight_required", "manifest_sha256": manifest["manifest_sha256"],
            "config_sha256": manifest["config_sha256"], "question_count_per_arm": 100,
            "question_runs": 200, "manifest": str(REPORTS / "run_manifest.json")}
    elif args.command == "live-preflight": result = live_preflight()
    elif args.command == "status": result = status()
    elif args.command == "write-resource-receipt": result = write_resource_receipt(args.kind, args.root, args.output, model_revision=args.model_revision)
    elif args.command == "run-all": result = run_all(resume_infrastructure=args.resume_infrastructure)
    else: result = run(args.stage, resume_infrastructure=args.resume_infrastructure)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
