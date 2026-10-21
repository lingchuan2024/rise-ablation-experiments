#!/usr/bin/env python3
"""Freeze and schedule three v14 ablations in rotating ten-question blocks.

Preparation and offline checks never contact a model. Live preflight, capacity
measurement and execution are separate explicit commands.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from urllib.parse import urlsplit

from runtime_guards import StopState, atomic_json, install_runtime_hooks, install_reranker_hooks

PILOT = Path(__file__).resolve().parent
ROOT = PILOT.parent
REPORTS = PILOT / "reports"
MODEL = "Qwen3.5-35B-A3B"
PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"
VERSION = "v14-ablations-suite-v1"
VARIANTS = ("search_paging", "auto_rerank", "no_bash")
DATASET_SHA = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_queries():
    path = ROOT / "data/queries_100.jsonl"
    if file_sha(path) != DATASET_SHA:
        raise RuntimeError("The original fixed100 dataset bytes changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 100 or len({row["query_id"] for row in rows}) != 100:
        raise RuntimeError("The fixed100 cohort is incomplete or duplicated")
    return rows


def source_hashes():
    files = list((ROOT / "code").rglob("*.py")) + list(PILOT.glob("*.py")) + list(ROOT.glob("*.py"))
    files += list((ROOT / "baseline_snapshot").rglob("*.py"))
    files += [ROOT / "README.md", ROOT / "data/queries_100.jsonl"] + list((ROOT / "data").glob("*.json"))
    if (ROOT / "source_manifest.json").is_file():
        files.append(ROOT / "source_manifest.json")
    if (ROOT / "origin/source_manifest.json").is_file():
        files.append(ROOT / "origin/source_manifest.json")
    if (ROOT / "code/bin/rg").is_file():
        files.append(ROOT / "code/bin/rg")
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(files)}


def block_schedule(rows):
    cells = []
    for block in range(10):
        order = VARIANTS[block % 3:] + VARIANTS[:block % 3]
        for variant in order:
            cells.append({"cell": len(cells) + 1, "block": block + 1, "variant": variant,
                "query_ids": [row["query_id"] for row in rows[block * 10:(block + 1) * 10]]})
    return cells


def resource_checks(config):
    needed = {key: Path(config[key]) for key in ("python", "document_index", "filename_map", "corpus", "passage_index")}
    if any(not path.exists() for path in needed.values()):
        raise RuntimeError("A configured runtime, corpus, map or index is absent")
    if not needed["python"].is_file() or not os.access(needed["python"], os.X_OK):
        raise RuntimeError("The prepared Python runtime is not executable")
    for key in ("document_index", "corpus", "passage_index"):
        if not needed[key].is_dir():
            raise RuntimeError("A configured resource directory is not a directory")
    index_path = needed["passage_index"] / "manifest.json"
    index = load(index_path)
    configuration = index.get("configuration", index.get("identity", {}))
    chunk = configuration.get("chunking", {})
    if index.get("status") != "ready" or (chunk.get("passage_tokens"), chunk.get("overlap_tokens")) != (128, 32):
        raise RuntimeError("The published passage index/chunking identity differs")
    if configuration.get("tokenizer", {}).get("revision") != PIN:
        raise RuntimeError("The passage tokenizer revision differs")
    mapping = load(needed["filename_map"]).get("relpath_to_docid", {})
    if len(mapping) != 100195:
        raise RuntimeError("The frozen filename map must cover 100195 documents")
    sys.path.insert(0, str(ROOT / "code/src"))
    from rise.protocol import for_rise
    if for_rise(agent_model=MODEL).wall_clock_timeout_sec != 3600:
        raise RuntimeError("The shared single-question wall-clock budget changed")
    return {"passage_manifest_sha256": file_sha(index_path), "filename_map_sha256": file_sha(needed["filename_map"]),
        "passage_index_identity": {key: index.get(key) for key in ("format_version", "corpus_content_sha256")},
        "passage_configuration_sha256": configuration.get("sha256"), "mapped_documents": len(mapping),
        "python_executable": str(needed["python"])}


def endpoint(value):
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use a credential-free model endpoint")
    return value.rstrip("/")


def settings(args):
    data = Path(args.data_root).resolve()
    config = {"model": MODEL, "context_tokens": 262144, "concurrency": 4,
        "max_turns": 100, "per_call_max_tokens": 32000, "api_max_retries": 3,
        "wall_clock_seconds_per_question": 3600, "context_level": "level0",
        "read_default_limit": 2000, "bm25_k": 1000, "bm25_top_n_preview": 10,
        "bm25_snippet_chars": 100, "reranker_revision": PIN,
        "reranker_device": args.reranker_device,
        "python": os.path.abspath(os.path.expanduser(args.python)),
        "base_url": endpoint(args.base_url),
        "document_index": str((Path(args.document_index) if args.document_index else data / "runs/bm25_full").resolve()),
        "filename_map": str((Path(args.filename_map) if args.filename_map else data / "runs/corpus_filename_docid_map.json").resolve()),
        "corpus": str((Path(args.corpus) if args.corpus else data / "corpus/bcp_plus_structured").resolve()),
        "passage_index": str((Path(args.passage_index) if args.passage_index else data / "runs/passage_index_128_32").resolve()),
        "capacity_report": str(Path(args.capacity_report).resolve()) if args.capacity_report else None,
        "capacity_report_sha256": file_sha(args.capacity_report) if args.capacity_report else None,
        "service_config": str(Path(args.service_config).resolve()) if args.service_config else None,
        "service_receipt": str(Path(args.service_receipt).resolve()) if args.service_receipt else None,
        "source_cohort_sha256": DATASET_SHA, "historical_control_only": True,
        "historical_evaluation": str(ROOT / "data/historical_v14_evaluation.json"),
        "historical_evaluation_sha256": file_sha(ROOT / "data/historical_v14_evaluation.json"),
        "arm_outputs": {variant: str(ROOT / "runs" / variant) for variant in VARIANTS}}
    config["service_config_sha256"] = file_sha(config["service_config"]) if config["service_config"] else None
    config["service_receipt_sha256"] = file_sha(config["service_receipt"]) if config["service_receipt"] else None
    env = {"CUDA_VISIBLE_DEVICES": args.visible_devices, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "RISE_BASH_SANDBOX": "docker", "RISE_DOCKER_IMAGE": args.docker_image,
        "RISE_LOCAL_VLLM_BASE_URL": config["base_url"], "RISE_LOCAL_VLLM_CONTEXT_LENGTH": "262144",
        "RISE_LOCAL_VLLM_TIMEOUT": "1200", "RISE_QWEN35_PROFILE": "official-thinking-v1",
        "REASONING_EFFORT": "medium", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    if args.docker_host:
        parsed = urlsplit(args.docker_host)
        if parsed.username or parsed.password:
            raise ValueError("Docker host cannot contain credentials")
        env["DOCKER_HOST"] = args.docker_host
    if args.hf_home:
        env["HF_HOME"] = str(Path(args.hf_home).resolve())
    if config["service_config_sha256"]:
        env["RISE_LOCAL_VLLM_SERVICE_SHA256"] = config["service_config_sha256"]
    config["runtime_environment"] = env
    return config


def runtime_environment(config):
    # Match the historical local-vLLM launcher: no unrelated API credentials or
    # endpoint/tool-injection settings survive into formal workers.
    keys = ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "HOME")
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    environment.update(config["runtime_environment"])
    return environment


def prepare(args):
    rows = validate_queries()
    manifest_path = REPORTS / "run_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("A prepared suite already exists; do not replace its frozen identity")
    if any(Path(ROOT / "runs" / variant).exists() for variant in VARIANTS):
        raise RuntimeError("Prepare before creating arm output directories")
    config = settings(args)
    sources = source_hashes()
    from preflight import capacity_evidence
    capacity_evidence(config, sources)
    resources = resource_checks(config)
    schedule = block_schedule(rows)
    blocks = {}
    for block in range(10):
        target = PILOT / "prepared" / f"block_{block + 1:02}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[block * 10:(block + 1) * 10]))
        blocks[str(block + 1)] = {"path": str(target), "sha256": file_sha(target)}
    payload = {"version": VERSION, "config": config, "source_sha256": sources, "resource_checks": resources,
        "blocks": blocks, "schedule": schedule, "question_runs": 300,
        "first_block_counts_toward_full100": True}
    manifest = {**payload, "manifest_sha256": digest(payload)}
    atomic_json(manifest_path, manifest)
    for variant in VARIANTS:
        atomic_json(REPORTS / f"arm_{variant}.json", {"variant": variant,
            "manifest_sha256": manifest["manifest_sha256"], "output": config["arm_outputs"][variant],
            "query_ids": [row["query_id"] for row in rows]})
    return {"status": "prepared_not_live_ready", "manifest_sha256": manifest["manifest_sha256"],
            "question_runs": 300, "next": "preflight --live on the target host"}


def verify_manifest():
    manifest = load(REPORTS / "run_manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("version") != VERSION or digest(payload) != manifest.get("manifest_sha256"):
        raise RuntimeError("Manifest identity is invalid")
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("Frozen source bytes changed; preserve the old attempt")
    validate_queries()
    for block in manifest["blocks"].values():
        if file_sha(block["path"]) != block["sha256"]:
            raise RuntimeError("Frozen ten-question block changed")
    config = manifest["config"]
    for key in ("capacity_report", "service_config", "service_receipt"):
        if config.get(key) and file_sha(config[key]) != config[key + "_sha256"]:
            raise RuntimeError("Frozen runtime evidence changed: " + key)
    if resource_checks(config) != manifest["resource_checks"]:
        raise RuntimeError("Frozen resource identity changed")
    return manifest, config


def build_argv(config, cell, manifest):
    return [str(ROOT / "run_ablation.py"), "--ablation-variant", cell["variant"],
        "--model", MODEL, "--mini-dev", manifest["blocks"][str(cell["block"])]["path"],
        "--index-dir", config["document_index"], "--filename-map", config["filename_map"],
        "--bc-plus-docs", config["corpus"], "--space-passage-index", config["passage_index"],
        "--structured-docs", "--concurrency", "4", "--reranker-model", "Qwen/Qwen3-Reranker-0.6B",
        "--reranker-revision", PIN, "--reranker-device", config["reranker_device"],
        "--reranker-dtype", "bfloat16", "--reranker-batch-size", "8", "--reranker-local-files-only",
        "--max-turns", "100", "--per-call-max-tokens", "32000", "--api-max-retries", "3",
        "--turn-checkpoint", "--out-root", config["arm_outputs"][cell["variant"]]]


def status(manifest=None):
    manifest = manifest or verify_manifest()[0]
    result = {"question_runs": 300, "arms": {}}
    for variant in VARIANTS:
        out = Path(manifest["config"]["arm_outputs"][variant])
        qids = [row["query_id"] for row in validate_queries()]
        completed, checkpointed = [], []
        for qid in qids:
            if (out / "_per_query" / f"qid_{qid}.json").is_file():
                completed.append(qid)
            elif (out / "_checkpoints" / f"qid_{qid}.json").is_file():
                checkpointed.append(qid)
        result["arms"][variant] = {"completed": completed, "checkpointed": checkpointed,
            "pending": [qid for qid in qids if qid not in completed and qid not in checkpointed]}
    result["completed_runs"] = sum(len(row["completed"]) for row in result["arms"].values())
    result["infrastructure_paused"] = (REPORTS / "infrastructure_pause.json").exists()
    return result


def require_live_preflight(manifest):
    report = load(REPORTS / "preflight_live.json")
    if report.get("passed") is not True or report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("Exact-manifest live preflight must pass first")
    return report


def cell_worker(number):
    manifest, config = verify_manifest()
    require_live_preflight(manifest)
    cell = next(row for row in manifest["schedule"] if row["cell"] == number)
    out = Path(config["arm_outputs"][cell["variant"]])
    binding = out / "_suite_manifest.json"
    if binding.exists():
        if load(binding).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("Output belongs to a different frozen suite")
    elif out.exists() and any(out.iterdir()):
        raise RuntimeError("Refusing unbound existing output")
    else:
        atomic_json(binding, {"manifest_sha256": manifest["manifest_sha256"], "variant": cell["variant"]})
    if Path(sys.executable).resolve() != Path(config["python"]).resolve():
        raise RuntimeError("Execute with the prepared Python runtime")
    environment = runtime_environment(config)
    os.environ.clear()
    os.environ.update(environment)
    os.environ["RISE_API_DIAGNOSTICS_DIR"] = str(out / "api_diagnostics")
    sys.path.insert(0, str(ROOT / "code/src"))
    from preflight import service_identity
    service_identity(config)
    state = StopState(REPORTS / "infrastructure_pause.json", REPORTS / "infrastructure_events.jsonl")
    if state.pause_path.exists():
        raise RuntimeError("Infrastructure pause remains unresolved")
    install_runtime_hooks(state, MODEL, config["base_url"], out / "api_transport.jsonl")
    install_reranker_hooks(state)
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
                    if exc.errno in {28, 122}:
                        state.pause("filesystem_quota_or_full", {"errno": exc.errno})
                    raise
            return guarded
        setattr(agent, name, factory)
    sys.argv = build_argv(config, cell, manifest)
    runpy.run_path(str(ROOT / "run_ablation.py"), run_name="__main__")
    state.raise_if_paused()
    current = status(manifest)["arms"][cell["variant"]]
    if not set(cell["query_ids"]).issubset(current["completed"]):
        raise RuntimeError("Cell incomplete; inspect errors and checkpoints before recovery")
    atomic_json(REPORTS / "cells" / f"cell_{number:02}.json", {**cell,
        "manifest_sha256": manifest["manifest_sha256"], "completed": True, "finished_unix": time.time()})


def execute(limit_blocks=None, resume=False):
    manifest, config = verify_manifest()
    report = require_live_preflight(manifest)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / ".suite.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pause = REPORTS / "infrastructure_pause.json"
        if pause.exists():
            if not resume or report["finished_unix"] <= load(pause)["first_failure"]["unix_time"]:
                raise RuntimeError("Fresh live preflight and --resume-after-preflight are required")
            pause.rename(REPORTS / f"infrastructure_pause.{time.time_ns()}.json")
        receipt = REPORTS / "worker_receipt.json"
        if receipt.exists() and not resume:
            raise RuntimeError("Existing attempt: inspect status, then explicitly resume")
        atomic_json(receipt, {"pid": os.getpid(), "manifest_sha256": manifest["manifest_sha256"], "started_unix": time.time()})
        for cell in manifest["schedule"]:
            if limit_blocks and cell["block"] > limit_blocks:
                break
            done = REPORTS / "cells" / f"cell_{cell['cell']:02}.json"
            if done.exists():
                if load(done).get("manifest_sha256") != manifest["manifest_sha256"]:
                    raise RuntimeError("Cell completion identity mismatch")
                continue
            verify_manifest()
            result = subprocess.run([config["python"], str(Path(__file__).resolve()), "_worker",
                "--cell", str(cell["cell"])], env=runtime_environment(config), check=False)
            if result.returncode:
                raise RuntimeError(f"Cell {cell['cell']} stopped; inspect retained outputs before recovery")
        result = status(manifest)
        atomic_json(REPORTS / "suite_completion.json", {**result,
            "status": "complete" if result["completed_runs"] == 300 else "health_block_complete",
            "manifest_sha256": manifest["manifest_sha256"], "finished_unix": time.time()})
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "status", "preflight", "measure-capacity", "run", "_worker"))
    p.add_argument("--live", action="store_true")
    p.add_argument("--first-block-only", action="store_true")
    p.add_argument("--resume-after-preflight", action="store_true")
    p.add_argument("--cell", type=int)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data-root", default="/root/data/rise-reproduction")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--reranker-device", default="cuda:0")
    p.add_argument("--visible-devices", default="0")
    p.add_argument("--docker-image", default="sha256:4cca46f0b1841ae9708781b564f57bb0f296427b936cf5cd4af143329e0e06b1")
    for name in ("document-index", "filename-map", "corpus", "passage-index", "capacity-report", "service-config", "service-receipt", "docker-host", "hf-home"):
        p.add_argument("--" + name)
    return p


def main():
    args = parser().parse_args()
    if args.action == "prepare":
        value = prepare(args)
    elif args.action == "verify":
        m, _ = verify_manifest()
        value = {"status": "frozen_identity_verified", "manifest_sha256": m["manifest_sha256"], "question_runs": 300}
    elif args.action == "status":
        value = status()
    elif args.action == "preflight":
        from preflight import offline_check, live_check
        value = live_check() if args.live else offline_check()
    elif args.action == "measure-capacity":
        from preflight import measure_capacity
        value = measure_capacity(args)
    elif args.action == "run":
        value = execute(1 if args.first_block_only else None, args.resume_after_preflight)
    else:
        if args.cell not in range(1, 31):
            raise SystemExit("A valid prepared cell number is required")
        cell_worker(args.cell)
        value = {"cell": args.cell, "completed": True}
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
