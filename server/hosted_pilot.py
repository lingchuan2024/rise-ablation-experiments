#!/usr/bin/env python3
"""Independent hosted Qwen3.8 Flash pilot: three variants, first10 each, total C3.

This driver does not start a server, purchase resources or write credentials.
prepare is offline; preflight --live and run make explicit hosted API calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import runtime_guards as guards
from shared_reranker import SHARED_RERANKER_POLICY
from suite import DATASET_SHA, PIN, VARIANTS, digest, file_sha, load, source_hashes as suite_sources, validate_queries

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "server/hosted_pilot"
REPORTS = PILOT / "reports"
MODEL = "qwen3.8-flash"
ENDPOINT = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
VERSION = "v14-hosted-qwen38-first10-v1"
SECRET_ENV = "DASHSCOPE_API_KEY"
SHARING_POLICY = SHARED_RERANKER_POLICY


def source_hashes():
    sources = suite_sources()
    sources["HOSTED_PILOT.md"] = file_sha(ROOT / "HOSTED_PILOT.md")
    return sources


def secret_from_stdin():
    payload = json.loads(sys.stdin.readline())
    if not isinstance(payload, dict) or set(payload) != {"api_key"} or not isinstance(payload["api_key"], str) or not payload["api_key"].strip():
        raise ValueError("stdin must contain one JSON object with a nonempty api_key")
    os.environ[SECRET_ENV] = payload["api_key"]


def runtime_environment(config, *, require_secret=True):
    keys = ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "HOME")
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    environment.update(config["runtime_environment"])
    secret = os.environ.get(SECRET_ENV)
    if require_secret and not secret:
        raise RuntimeError("Hosted API credential is absent from the process environment")
    if secret:
        environment[SECRET_ENV] = secret
    return environment


def sanitized_text(value, limit=300):
    value = str(value)
    secret = os.environ.get(SECRET_ENV)
    if secret:
        value = value.replace(secret, "[REDACTED]")
    return "".join(char if char.isprintable() else " " for char in value)[:limit]


def safe_error_metadata(exc):
    row = {"error_type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        row["http_status"] = status
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    code = error.get("code", getattr(exc, "code", None))
    if code is not None:
        row["provider_error_code"] = sanitized_text(code, 100)
    row["provider_error_message"] = sanitized_text(error.get("message", getattr(exc, "message", str(exc))))
    return row


def settings(args):
    if args.base_url.rstrip("/") != ENDPOINT:
        raise ValueError("Only the user-specified Qwen token-plan endpoint is permitted")
    allowed = [MODEL, *(args.allow_reported_model or [])]
    if any(not value or value != value.strip() or any(char in value for char in "*?\n\r") for value in allowed):
        raise ValueError("Reported model identities must be explicit exact names")
    config = {"model": MODEL, "base_url": ENDPOINT, "allowed_reported_models": sorted(set(allowed)),
        "concurrency": 3, "question_count_per_variant": 10, "question_runs": 30,
        "max_turns": 100, "per_call_max_tokens": 32000, "api_max_retries": 3,
        "wall_clock_seconds_per_question": 3600, "read_default_limit": 2000,
        "bm25_k": 1000, "bm25_top_n_preview": 10, "bm25_snippet_chars": 100,
        "context_level": "level0", "thinking_mode": args.thinking_mode,
        "reasoning_effort": "medium", "preserve_thinking": True,
        "per_call_max_completion_tokens": 32000, "request_timeout_seconds": 600,
        "python": os.path.abspath(os.path.expanduser(args.python)),
        "document_index": str(Path(args.index_dir).resolve()),
        "filename_map": str(Path(args.filename_map).resolve()), "corpus": str(Path(args.corpus).resolve()),
        "passage_index": str(Path(args.passage_index).resolve()),
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B", "reranker_revision": PIN,
        "reranker_device": args.reranker_device, "reranker_dtype": args.reranker_dtype,
        "reranker_sharing_policy": SHARING_POLICY,
        "sandbox_backend": args.sandbox_backend,
        "source_cohort_sha256": DATASET_SHA,
        "arm_outputs": {variant: str(ROOT / "runs/hosted_qwen38_first10" / variant) for variant in VARIANTS}}
    environment = {"DASHSCOPE_BASE_URL": ENDPOINT, "REASONING_EFFORT": "medium",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "RISE_BASH_SANDBOX": args.sandbox_backend,
        "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "PYTHONUNBUFFERED": "1"}
    if args.hf_home:
        environment["HF_HOME"] = str(Path(args.hf_home).resolve())
    if args.visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    if args.sandbox_backend == "docker":
        if not args.docker_image:
            raise ValueError("The Docker backend requires an explicit verified image")
        environment["RISE_DOCKER_IMAGE"] = args.docker_image
        if args.docker_host:
            if "@" in args.docker_host:
                raise ValueError("Do not put credentials in Docker URLs")
            environment["DOCKER_HOST"] = args.docker_host
    config["runtime_environment"] = environment
    return config


def resource_identity(config):
    paths = {key: Path(config[key]) for key in ("python", "document_index", "filename_map", "corpus", "passage_index")}
    if any(not path.exists() for path in paths.values()):
        raise RuntimeError("Discover and provide all runtime/corpus/index resources before prepare")
    if not paths["python"].is_file() or not os.access(paths["python"], os.X_OK):
        raise RuntimeError("Configured Python is not executable")
    for key in ("document_index", "corpus", "passage_index"):
        if not paths[key].is_dir():
            raise RuntimeError("Configured resource directory is not a directory")
    manifest_path = paths["passage_index"] / "manifest.json"
    manifest = load(manifest_path)
    configuration = manifest.get("configuration", manifest.get("identity", {}))
    chunk = configuration.get("chunking", {})
    if manifest.get("status") != "ready" or (chunk.get("passage_tokens"), chunk.get("overlap_tokens")) != (128, 32) or configuration.get("tokenizer", {}).get("revision") != PIN:
        raise RuntimeError("A published fixed128/32 passage index with the frozen tokenizer revision is required")
    mapping = load(paths["filename_map"]).get("relpath_to_docid", {})
    if len(mapping) != 100195:
        raise RuntimeError("The original corpus map must contain100195 documents")
    return {"passage_manifest_sha256": file_sha(manifest_path),
        "filename_map_sha256": file_sha(paths["filename_map"]),
        "passage_corpus_content_sha256": manifest.get("corpus_content_sha256"),
        "passage_configuration_sha256": configuration.get("sha256"), "mapped_documents": len(mapping)}


def prepare(args):
    path = REPORTS / "run_manifest.json"
    if path.exists():
        raise RuntimeError("This hosted pilot already has a frozen manifest; preserve it")
    rows = validate_queries()[:10]
    config = settings(args)
    resources = resource_identity(config)
    if any(Path(out).exists() for out in config["arm_outputs"].values()):
        raise RuntimeError("Hosted output roots already exist; do not bind unreviewed prior results")
    queries = PILOT / "queries_10.jsonl"
    queries.parent.mkdir(parents=True, exist_ok=True)
    queries.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    payload = {"version": VERSION, "config": config, "source_sha256": source_hashes(),
        "resource_identity": resources, "queries": str(queries), "queries_sha256": file_sha(queries),
        "query_ids": [row["query_id"] for row in rows],
        "variant_order": list(VARIANTS), "question_runs": 30,
        "selection": "first10 records of original fixed100 in original order; no correctness selection"}
    manifest = {**payload, "manifest_sha256": digest(payload)}
    guards.atomic_json(path, manifest)
    return {"status": "prepared_live_preflight_required", "manifest_sha256": manifest["manifest_sha256"], "question_runs": 30}


def verify_manifest():
    manifest = load(REPORTS / "run_manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("version") != VERSION or digest(payload) != manifest.get("manifest_sha256"):
        raise RuntimeError("Hosted manifest identity is invalid")
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("Hosted pilot source identity changed after preparation")
    if file_sha(manifest["queries"]) != manifest["queries_sha256"] or [row["query_id"] for row in validate_queries()[:10]] != manifest["query_ids"]:
        raise RuntimeError("Frozen original first10 cohort changed")
    if resource_identity(manifest["config"]) != manifest["resource_identity"]:
        raise RuntimeError("Frozen corpus/index identity changed")
    return manifest, manifest["config"]


class HostedGuardedClient:
    allowed_models = (MODEL,)
    thinking_mode = "enabled"
    preserve_thinking = True
    reasoning_effort = "medium"
    request_timeout_seconds = 600

    def __init__(self, client, model, base_url, log_path, state):
        self._client, self._model, self._base_url = client, model, base_url
        self._log_path, self._state = Path(log_path), state
        if model != MODEL or str(client.base_url).rstrip("/") != ENDPOINT or base_url.rstrip("/") != ENDPOINT:
            raise RuntimeError("Hosted client routing differs from the frozen endpoint/model")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name):
        return getattr(self._client, name)

    def with_options(self, **kwargs):
        return type(self)(self._client.with_options(**kwargs), self._model, self._base_url, self._log_path, self._state)

    def _create(self, *args, **kwargs):
        from rise.api_retry import RequestDeadlineExceeded
        self._state.raise_if_paused()
        if kwargs.get("model") != MODEL:
            raise RuntimeError("Only the frozen requested hosted model is permitted")
        extra = dict(kwargs.get("extra_body") or {})
        for key, value in {"enable_thinking": True, "preserve_thinking": self.preserve_thinking}.items():
            if key in extra and extra[key] != value:
                raise RuntimeError("Thinking setting differs from the immutable hosted manifest")
            extra[key] = value
        kwargs["extra_body"] = extra
        if kwargs.get("reasoning_effort", self.reasoning_effort) != self.reasoning_effort:
            raise RuntimeError("Reasoning effort differs from the immutable hosted manifest")
        kwargs["reasoning_effort"] = self.reasoning_effort
        if "max_tokens" in kwargs:
            if "max_completion_tokens" in kwargs:
                raise RuntimeError("Send only one total completion-token budget")
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        deadline = getattr(self._state.local, "query_deadline", None)
        timeout = float(self.request_timeout_seconds)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestDeadlineExceeded("Agent deadline exhausted before hosted API request")
            timeout = min(timeout, remaining)
        supplied_timeout = kwargs.get("timeout")
        if isinstance(supplied_timeout, (int, float)) and math.isfinite(float(supplied_timeout)) and supplied_timeout > 0:
            timeout = min(timeout, float(supplied_timeout))
        kwargs["timeout"] = timeout
        started = time.monotonic()
        row = {"unix_time": time.time(), "query_id": getattr(self._state.local, "qid", None),
            "requested_model": MODEL, "requested_max_completion_tokens": kwargs.get("max_completion_tokens"),
            "thinking_mode": self.thinking_mode, "preserve_thinking": self.preserve_thinking,
            "reasoning_effort": self.reasoning_effort, "request_timeout_seconds": timeout,
            "transport": "hosted-Qwen-token-plan", "status": "error", "response_delivered": False}
        try:
            response = self._client.chat.completions.create(*args, **kwargs)
            reported_model = str(response.model)
            row["reported_model"] = sanitized_text(reported_model, 120)
            if reported_model not in self.allowed_models:
                self._state.pause("solver_model_identity_mismatch", {"reported_model": row["reported_model"]})
                raise guards.SandboxPaused("Hosted response model identity is not explicitly allowed")
            choice = response.choices[0] if response.choices else None
            row.update(status="ok", response_id=response.id, finish_reason=choice.finish_reason if choice else None,
                tool_calls=len(choice.message.tool_calls or []) if choice else 0,
                usage=response.usage.model_dump() if response.usage else None)
            if deadline is not None and time.monotonic() >= deadline:
                raise RequestDeadlineExceeded("Agent deadline exhausted during hosted API request")
            row["response_delivered"] = True
            return response
        except Exception as exc:
            row.update(safe_error_metadata(exc))
            secret = os.environ.get(SECRET_ENV)
            if secret and secret in str(exc):
                message = sanitized_text(str(exc))
                try:
                    sanitized = type(exc)(message, response=exc.response,
                        body={"code": row.get("provider_error_code"), "message": row["provider_error_message"]})
                except (AttributeError, TypeError):
                    sanitized = RuntimeError("Hosted API error details suppressed because they reflected credential material")
                raise sanitized from None
            raise
        finally:
            row["request_seconds"] = time.monotonic() - started
            with self._state._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def bound_client_class(config):
    class BoundHostedClient(HostedGuardedClient):
        allowed_models = tuple(config["allowed_reported_models"])
        thinking_mode = config["thinking_mode"]
        preserve_thinking = config["preserve_thinking"]
        reasoning_effort = config["reasoning_effort"]
        request_timeout_seconds = config["request_timeout_seconds"]
    return BoundHostedClient


def install_deadline_hooks(state, config):
    """Hosted-only deadline propagation; legacy/local Agent code is unchanged."""
    import rise.agent as agent
    import rise.rise_agent as rise_agent
    original_chat, original_run = agent.chat_with_retry, rise_agent.run_rise_agent

    @functools.wraps(original_run)
    def run_with_deadline(*args, **kwargs):
        elapsed = 0.0
        checkpoint = kwargs.get("turn_checkpoint_path")
        if checkpoint is not None and Path(checkpoint).is_file():
            elapsed = float(load(checkpoint).get("elapsed_seconds", 0.0))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Hosted checkpoint elapsed budget is invalid")
        budget = float(config["wall_clock_seconds_per_question"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Hosted wall-clock budget is invalid")
        state.local.query_deadline = time.monotonic() + max(0.0, budget - elapsed)
        try:
            return original_run(*args, **kwargs)
        finally:
            state.local.query_deadline = None

    @functools.wraps(original_chat)
    def chat_with_deadline(*args, **kwargs):
        deadline = getattr(state.local, "query_deadline", None)
        if deadline is not None:
            supplied = kwargs.get("deadline")
            kwargs["deadline"] = min(deadline, supplied) if supplied is not None else deadline
        return original_chat(*args, **kwargs)

    agent.chat_with_retry = chat_with_deadline
    rise_agent.run_rise_agent = run_with_deadline


def install_config_hook(config):
    import rise.protocol as protocol
    original = protocol.for_rise
    def hosted_config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.extras.update(hosted_pilot_version=VERSION, hosted_endpoint=ENDPOINT,
            hosted_allowed_reported_models=config["allowed_reported_models"],
            hosted_enable_thinking=True, hosted_preserve_thinking=True,
            hosted_reasoning_effort="medium", hosted_completion_budget_kind="max_completion_tokens",
            hosted_total_completion_tokens=32000, hosted_reranker_sharing_policy=SHARING_POLICY)
        return result
    protocol.for_rise = hosted_config


def tool_specs(variant):
    from rise.ablation_prompt import ablation_read_tool_spec, ablation_rerank_tool_spec, ablation_search_tool_spec
    from rise.rerank_v13 import rerank_tool_spec
    from rise.tools import bash_tool_spec, read_tool_spec
    from rise.v13_feedback import v13_bash_tool_spec
    specs = [ablation_search_tool_spec(variant)]
    if variant != "no_bash":
        specs.append(v13_bash_tool_spec(bash_tool_spec()))
    specs.append(ablation_read_tool_spec(variant, read_tool_spec()))
    if variant != "auto_rerank":
        specs.append(ablation_rerank_tool_spec(variant, rerank_tool_spec()))
    return specs


def build_argv(config, manifest, variant):
    return [str(ROOT / "run_ablation.py"), "--ablation-variant", variant, "--model", MODEL,
        "--mini-dev", manifest["queries"], "--index-dir", config["document_index"],
        "--filename-map", config["filename_map"], "--bc-plus-docs", config["corpus"],
        "--space-passage-index", config["passage_index"], "--structured-docs", "--concurrency", "3",
        "--reranker-model", config["reranker_model"], "--reranker-revision", PIN,
        "--reranker-device", config["reranker_device"], "--reranker-dtype", config["reranker_dtype"],
        "--reranker-batch-size", "8", "--reranker-local-files-only", "--max-turns", "100",
        "--per-call-max-tokens", "32000", "--api-max-retries", "3", "--read-default-limit", "2000",
        "--turn-checkpoint", "--out-root", config["arm_outputs"][variant]]


def status(manifest=None):
    manifest = manifest or verify_manifest()[0]
    result = {"question_runs": 30, "arms": {}}
    for variant in VARIANTS:
        out = Path(manifest["config"]["arm_outputs"][variant])
        completed, checkpointed, invalid = [], [], []
        for qid in manifest["query_ids"]:
            path = out / "_per_query" / f"qid_{qid}.json"
            if path.is_file():
                row = load(path)
                if str(row.get("query_id")) != str(qid) or str(row.get("terminated_by", "")).startswith("api_error"):
                    invalid.append(qid)
                else:
                    completed.append(qid)
            elif (out / "_checkpoints" / f"qid_{qid}.json").is_file():
                checkpointed.append(qid)
        result["arms"][variant] = {"completed": completed, "checkpointed": checkpointed, "invalid": invalid,
            "pending": [qid for qid in manifest["query_ids"] if qid not in completed and qid not in checkpointed and qid not in invalid]}
    result["completed_runs"] = sum(len(arm["completed"]) for arm in result["arms"].values())
    result["infrastructure_paused"] = (REPORTS / "infrastructure_pause.json").exists()
    return result


def native_probe(config, variant, client=None):
    from rise.api_retry import chat_with_retry
    from rise.decompose import make_client
    from rise.local_vllm import extract_reasoning
    state = guards.StopState(REPORTS / "probe_pause.json", REPORTS / "probe_events.jsonl")
    guarded = bound_client_class(config)(client or make_client(MODEL), MODEL, ENDPOINT,
        REPORTS / "preflight_transport.jsonl", state)
    specs = tool_specs(variant)
    messages = [{"role": "system", "content": 'Compatibility check. Call search exactly once with queries ["compatibility probe"], then acknowledge the tool result.'},
        {"role": "user", "content": "Call search before answering."}]
    first = chat_with_retry(guarded, model=MODEL, messages=messages, tools=specs,
        tool_choice="auto", max_tokens=8192, max_retries=3)
    message = first.choices[0].message
    calls = message.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != "search" or json.loads(calls[0].function.arguments) != {"queries": ["compatibility probe"]}:
        raise RuntimeError("Hosted native tool/JSON compatibility probe failed")
    assistant = {"role": "assistant", "content": message.content, "tool_calls": [call.model_dump() for call in calls]}
    reasoning = extract_reasoning(message)
    if reasoning:
        assistant["reasoning_content"] = reasoning
    messages.extend([assistant, {"role": "tool", "tool_call_id": calls[0].id, "content": '{"status":"ok","items":[]}'},
        {"role": "user", "content": "Reply exactly Compatibility OK without another tool call."}])
    second = chat_with_retry(guarded, model=MODEL, messages=messages, tools=specs,
        tool_choice="none", max_tokens=8192, max_retries=3)
    final = second.choices[0].message
    if final.tool_calls or (final.content or "").strip().rstrip(".") != "Compatibility OK":
        raise RuntimeError("Hosted native tool-result continuation failed")
    return {"passed": True, "variant": variant, "requests": 2,
        "tool_names": [spec["function"]["name"] for spec in specs],
        "reported_models": [first.model, second.model], "thinking_mode": config["thinking_mode"],
        "usage": [response.usage.model_dump() if response.usage else None for response in (first, second)]}


def resources_probe(config):
    from rise.passage_index import PassageIndex
    from rise.retrieval import load_index
    from rise.reranker_backend import QwenReranker
    index = PassageIndex(config["passage_index"], corpus_root=config["corpus"])
    _, docids = load_index(config["document_index"])
    mapping = load(config["filename_map"])["relpath_to_docid"]
    if set(map(str, docids)) != set(map(str, mapping.values())) or not set(mapping).issubset(index._path_to_doc):
        raise RuntimeError("Document/passage index scopes differ")
    backends = [QwenReranker(revision=PIN, device=config["reranker_device"], dtype=config["reranker_dtype"],
        batch_size=8, local_files_only=True) for _ in range(3)]
    def score(backend):
        scores = backend.score("What color is the sky?", ["The clear daytime sky is blue.", "A bicycle has two wheels."])
        if len(scores) != 2 or not all(math.isfinite(float(value)) for value in scores):
            raise RuntimeError("Real hosted-pilot reranker forward failed")
        return True
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(score, backends))
    return {"passed": all(results), "documents": len(docids), "passages": index.passage_count,
        "requested_reranker_leases": 3, "question_concurrency": 3, "device": config["reranker_device"],
        "dtype": config["reranker_dtype"], "reranker_sharing_policy": SHARING_POLICY,
        "test_scope": "three leases over shared serialized scorer, two passages each; no local solver required"}


def sandbox_probe(config):
    from rise.tools import make_bash_tool
    source = next(Path(config["corpus"]).rglob("*.txt"))
    def probe(slot):
        with tempfile.TemporaryDirectory(prefix="hosted-sandbox-", dir=REPORTS) as temporary:
            folder = Path(temporary)
            (folder / "probe.txt").write_text("HOSTED_SANDBOX_READY\n")
            os.link(source, folder / "corpus_document.txt")
            result = make_bash_tool(folder, enable_sandbox=True)({"command": "rg -n HOSTED_SANDBOX_READY probe.txt && test -r corpus_document.txt", "timeout": 120})
            if "1:HOSTED_SANDBOX_READY" not in result or result.startswith("Error:"):
                raise RuntimeError("Actual sandbox/ripgrep source path failed")
            return {"slot": slot, "passed": True}
    with ThreadPoolExecutor(max_workers=3) as executor:
        return {"passed": True, "backend": config["sandbox_backend"], "calls": list(executor.map(probe, range(3)))}


def offline_check():
    for name in source_hashes():
        if name.endswith(".py"):
            compile((ROOT / name).read_bytes(), name, "exec")
    return {"passed": True, "live_ready": False, "network_requests": 0,
        "question_runs": 30, "question_ids": [row["query_id"] for row in validate_queries()[:10]],
        "test_command": "python -m unittest discover -s server -p test_hosted_pilot.py -v"}


def live_preflight():
    manifest, config = verify_manifest()
    environment = runtime_environment(config)
    os.environ.clear()
    os.environ.update(environment)
    sys.path.insert(0, str(ROOT / "code/src"))
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "manifest_sha256": manifest["manifest_sha256"], "started_unix": time.time()}
    try:
        state = guards.StopState(REPORTS / "preflight_backend_pause.json", REPORTS / "preflight_backend_events.jsonl")
        guards.install_reranker_hooks(state)
        from shared_reranker import install_shared_reranker_factory
        install_shared_reranker_factory()
        report["resources"] = resources_probe(config)
        report["sandbox"] = sandbox_probe(config)
        with ThreadPoolExecutor(max_workers=3) as executor:
            report["native_probes"] = list(executor.map(
                lambda variant: native_probe(config, variant), VARIANTS))
        report["native_request_concurrency"] = 3
        verify_manifest()
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_unix"] = time.time()
        guards.atomic_json(REPORTS / "preflight_live.json", report)
    return report


def require_preflight(manifest):
    report = load(REPORTS / "preflight_live.json")
    if report.get("passed") is not True or report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("The exact frozen hosted manifest requires a passed live preflight")
    return report


def worker(variant):
    manifest, config = verify_manifest()
    require_preflight(manifest)
    environment = runtime_environment(config)
    os.environ.clear()
    os.environ.update(environment)
    sys.path.insert(0, str(ROOT / "code/src"))
    out = Path(config["arm_outputs"][variant])
    binding = out / "_hosted_manifest.json"
    if binding.exists():
        if load(binding).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("Hosted output belongs to a different frozen pilot")
    elif out.exists() and any(out.iterdir()):
        raise RuntimeError("Refusing unbound hosted output")
    else:
        guards.atomic_json(binding, {"manifest_sha256": manifest["manifest_sha256"], "variant": variant})
    state = guards.StopState(REPORTS / "infrastructure_pause.json", REPORTS / "infrastructure_events.jsonl")
    if state.pause_path.exists():
        raise RuntimeError("Resolve the existing infrastructure pause before starting hosted work")
    guards.GuardedLocalClient = bound_client_class(config)
    guards.install_runtime_hooks(state, MODEL, ENDPOINT, out / "api_transport.jsonl")
    guards.install_reranker_hooks(state)
    from shared_reranker import install_shared_reranker_factory
    install_shared_reranker_factory()
    install_deadline_hooks(state, config)
    install_config_hook(config)
    # Index/import/read storage failures use the same stop state as the API,
    # native reranker and sandbox; queued questions must not continue.
    import rise.rise_agent as agent
    for name in ("make_bm25_search_tool", "make_read_tool"):
        original = getattr(agent, name)
        def factory(*args, _original=original, **kwargs):
            tool = _original(*args, **kwargs)
            def guarded(params):
                state.raise_if_paused()
                try:
                    return tool(params)
                except OSError as exc:
                    if exc.errno in {28, 122} and not state.paused:
                        state.pause("filesystem_quota_or_full", {"errno": exc.errno})
                    raise
            return guarded
        setattr(agent, name, factory)
    sys.argv = build_argv(config, manifest, variant)
    runpy.run_path(str(ROOT / "run_ablation.py"), run_name="__main__")
    state.raise_if_paused()
    if len(status(manifest)["arms"][variant]["completed"]) != 10:
        raise RuntimeError("Hosted variant is incomplete; inspect retained checkpoints")
    guards.atomic_json(REPORTS / f"completed_{variant}.json", {"variant": variant,
        "manifest_sha256": manifest["manifest_sha256"], "finished_unix": time.time()})


def run(resume=False):
    manifest, config = verify_manifest()
    report = require_preflight(manifest)
    environment = runtime_environment(config)
    with (REPORTS / ".hosted.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pause = REPORTS / "infrastructure_pause.json"
        if pause.exists():
            if not resume or report["finished_unix"] <= load(pause)["first_failure"]["unix_time"]:
                raise RuntimeError("Infrastructure recovery requires fresh live preflight and explicit resume")
            pause.rename(REPORTS / f"infrastructure_pause.{time.time_ns()}.json")
        receipt = REPORTS / "worker_receipt.json"
        if receipt.exists() and not resume:
            raise RuntimeError("Existing hosted attempt: inspect status and explicitly resume")
        guards.atomic_json(receipt, {"pid": os.getpid(), "manifest_sha256": manifest["manifest_sha256"],
            "started_unix": time.time(), "credential_persisted": False, "concurrency": 3})
        for variant in manifest["variant_order"]:
            completion = REPORTS / f"completed_{variant}.json"
            if completion.exists():
                if load(completion).get("manifest_sha256") != manifest["manifest_sha256"]:
                    raise RuntimeError("Hosted completion identity differs")
                continue
            result = subprocess.run([config["python"], str(Path(__file__).resolve()), "_worker", "--variant", variant],
                env=environment, check=False)
            if result.returncode:
                raise RuntimeError("Hosted variant stopped; checkpoints and original budgets are retained")
        result = status(manifest)
        if result["completed_runs"] != 30:
            raise RuntimeError("The hosted pilot has not completed30 problem runs")
        guards.atomic_json(REPORTS / "completion.json", {**result,
            "manifest_sha256": manifest["manifest_sha256"], "finished_unix": time.time()})
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "preflight", "run", "status", "_worker"))
    p.add_argument("--secret-stdin", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--resume-after-preflight", action="store_true")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--base-url", default=ENDPOINT)
    p.add_argument("--allow-reported-model", action="append")
    p.add_argument("--thinking-mode", choices=("enabled",), default="enabled")
    for name in ("index-dir", "filename-map", "corpus", "passage-index", "reranker-device"):
        p.add_argument("--" + name)
    p.add_argument("--reranker-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--sandbox-backend", choices=("bubblewrap", "docker"), default="bubblewrap")
    for name in ("hf-home", "visible-devices", "docker-image", "docker-host"):
        p.add_argument("--" + name)
    return p


def main():
    os.umask(0o077)
    args = parser().parse_args()
    if args.secret_stdin:
        secret_from_stdin()
    if args.action == "prepare":
        required = ("index_dir", "filename_map", "corpus", "passage_index", "reranker_device")
        if any(not getattr(args, key) for key in required):
            raise ValueError("prepare requires discovered corpus/index/map/passage-index and reranker device paths")
        result = prepare(args)
    elif args.action == "verify":
        manifest, _ = verify_manifest()
        result = {"verified": True, "manifest_sha256": manifest["manifest_sha256"], "question_runs": 30}
    elif args.action == "preflight":
        result = live_preflight() if args.live else offline_check()
    elif args.action == "status":
        result = status()
    elif args.action == "run":
        result = run(args.resume_after_preflight)
    else:
        if not args.variant:
            raise ValueError("An internal worker requires a frozen variant")
        worker(args.variant)
        result = {"variant": args.variant, "completed": True}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never echo raw provider exception bodies or credentials to shell logs.
        print(json.dumps({"status": "hosted_pilot_failed", **safe_error_metadata(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None
