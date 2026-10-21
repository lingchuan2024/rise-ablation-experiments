#!/usr/bin/env python3
"""Local-only entry point for the paper's full RISE, 100-query experiment.

This script never downloads a dataset.  `check` is read-only; `index` builds
BM25 from a local parquet; `run` performs the paid agent calls; `judge`
performs the paid LLM-as-judge calls; `report` reads saved trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rise.query_ids import safe_query_id


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries_100.jsonl")
    parser.add_argument("--raw-parquet", type=Path, default=ROOT / "corpus" / "browsecomp_plus.parquet")
    parser.add_argument("--raw-docs", type=Path, default=ROOT / "corpus" / "bcp_plus")
    parser.add_argument("--structured-docs", type=Path, default=ROOT / "corpus" / "bcp_plus_structured")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "runs" / "bm25_full")
    parser.add_argument("--filename-map", type=Path, default=ROOT / "runs" / "corpus_filename_docid_map.json")
    parser.add_argument("--expected-count", type=int, default=100)


def _call(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def _check_dependencies(*modules: str) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing Python packages: " + ", ".join(missing)
            + ". Install the project's pyproject.toml dependencies in a virtual environment."
        )


def _check_inputs(args: argparse.Namespace) -> None:
    cmd = [
        str(ROOT / "scripts" / "preflight_rise.py"),
        "--queries", str(args.queries),
        "--raw-parquet", str(args.raw_parquet),
        "--raw-docs", str(args.raw_docs),
        "--structured-docs", str(args.structured_docs),
        "--index-dir", str(args.index_dir),
        "--filename-map", str(args.filename_map),
        "--expected-queries", str(args.expected_count),
    ]
    _call(cmd)


def _load_env_file() -> None:
    # The official runners use python-dotenv too.  No value is ever printed.
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")


def _require_key(model: str) -> None:
    _load_env_file()
    normalized = model.strip().lower()
    if normalized.count("/") == 1 and all(normalized.split("/")):
        keys = ("NVIDIA_API_KEYS", "NVIDIA_API_KEY")
    elif normalized.startswith(("gpt-", "o3", "o4", "o5")):
        keys = ("OPENAI_DIRECT_API_KEY", "OPENAI_API_KEY")
    elif normalized.startswith("mimo"):
        keys = ("XIAOMI_API_KEY",)
    elif normalized.startswith("deepseek"):
        keys = ("DEEPSEEK_API_KEY",)
    elif normalized in {"qwen3.5-27b", "qwen3.5-35b-a3b"} and os.environ.get("RISE_LOCAL_VLLM_BASE_URL"):
        return
    elif normalized.startswith("qwen"):
        keys = ("DASHSCOPE_API_KEY",)
    else:
        raise SystemExit(
            f"Unsupported model routing for {model!r}; use a gpt-, mimo-, "
            "deepseek-, qwen-, or NVIDIA Catalog publisher/model ID."
        )
    if not any(os.environ.get(name) for name in keys):
        raise SystemExit("Missing model API credential. Set " + " or ".join(keys) + " in rise-reproduction/.env.")


def _default_run_root(model: str) -> Path:
    """Keep provider/model IDs in one directory under runs, without collisions."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    if not safe:
        raise SystemExit("--model must contain a non-empty model ID.")
    if safe != model:
        safe += "_" + hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return ROOT / "runs" / f"rise_{safe}_100k"


def _recorded_reasoning_effort(model: str) -> str:
    # The hosted NVIDIA Nemotron Super endpoint defaults to high reasoning.
    # We record its actual default without changing the API request.
    return "high" if model in ("gpt-5.4-nano", "nvidia/nemotron-3-super-120b-a12b") else "medium"


def _read_query_ids(queries: Path) -> set[str]:
    ids: list[str] = []
    with queries.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.append(safe_query_id(json.loads(line)["query_id"]))
    if len(ids) != len(set(ids)):
        raise SystemExit("Query file has duplicate query_id values.")
    return set(ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_tree_digest(root: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relpath in sorted(relative_paths):
        path = root / relpath
        stat = path.stat()
        digest.update(f"{relpath}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _input_manifest(args: argparse.Namespace) -> dict[str, object]:
    filename_map = json.loads(args.filename_map.read_text(encoding="utf-8"))
    relative_paths = list(filename_map["relpath_to_docid"])
    parquet_stat = args.raw_parquet.stat()
    index_files = sorted(path for path in (args.index_dir / "bm25_index").rglob("*") if path.is_file())
    index_meta = hashlib.sha256()
    for path in index_files:
        stat = path.stat()
        index_meta.update(f"{path.relative_to(args.index_dir)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "model": args.model,
        "reasoning_effort": _recorded_reasoning_effort(args.model),
        "queries": str(args.queries.resolve()),
        "queries_sha256": _sha256(args.queries),
        "raw_parquet": str(args.raw_parquet.resolve()),
        "raw_parquet_size": parquet_stat.st_size,
        "raw_parquet_mtime_ns": parquet_stat.st_mtime_ns,
        "structured_docs": str(args.structured_docs.resolve()),
        "structured_tree_stat_sha256": _stat_tree_digest(args.structured_docs, relative_paths),
        "filename_map_sha256": _sha256(args.filename_map),
        "index_docids_sha256": _sha256(args.index_dir / "doc_ids.json"),
        "index_files_stat_sha256": index_meta.hexdigest(),
    }


def _pin_inputs(run_dir: Path, manifest: dict[str, object]) -> None:
    path = run_dir / "_local_inputs.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise SystemExit(
                f"Local inputs changed for {run_dir}. Use a new --out-root "
                "to keep trajectories from different inputs separate."
            )
        return
    if (run_dir / "_per_query").exists() or (run_dir / "_traces").exists():
        raise SystemExit(
            f"{run_dir} already has trajectories but no _local_inputs.json; "
            "use a new --out-root to avoid mixing results."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _check_hardlink_volume(structured_docs: Path, run_dir: Path) -> None:
    existing = run_dir
    while not existing.exists():
        existing = existing.parent
    if structured_docs.stat().st_dev != existing.stat().st_dev:
        raise SystemExit(
            "RISE imports retrieved documents with hardlinks. "
            f"Put --out-root on the same filesystem as {structured_docs}."
        )


def _check_trajectories(run_dir: Path, queries: Path, expected_count: int) -> None:
    expected = _read_query_ids(queries)
    if len(expected) != expected_count:
        raise SystemExit(f"Expected {expected_count} query IDs, found {len(expected)} in {queries}.")
    rows: dict[str, dict] = {}
    for path in (run_dir / "_per_query").glob("qid_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        qid = str(row.get("query_id"))
        if qid in rows:
            raise SystemExit(f"Duplicate per-query result for {qid}.")
        rows[qid] = row
    missing = expected - rows.keys()
    extra = rows.keys() - expected
    missing_traces = [
        qid for qid in expected & rows.keys()
        if not (run_dir / "_traces" / f"qid_{qid}" / "single.json").is_file()
    ]
    if missing or extra or missing_traces:
        raise SystemExit(
            f"Incomplete run: {len(rows)}/{len(expected)} per-query rows; "
            f"missing IDs={sorted(missing)}; extra IDs={sorted(extra)}; "
            f"missing traces={sorted(missing_traces)}. Re-run the same command to resume."
        )
    print(f"Verified {expected_count} per-query rows and {expected_count} trajectory files in {run_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Read-only validation of local inputs; no API calls.")
    _add_inputs(check)
    index = sub.add_parser("index", help="Build the paper's BM25 index from local raw parquet.")
    index.add_argument("--raw-parquet", type=Path, default=ROOT / "corpus" / "browsecomp_plus.parquet")
    index.add_argument("--index-dir", type=Path, default=ROOT / "runs" / "bm25_full")
    run = sub.add_parser("run", help="Run full RISE on the local 100-query set; makes model API calls.")
    _add_inputs(run)
    run.add_argument("--model", default="gpt-5.4-mini")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--api-max-retries", type=int, default=3,
                     help="Retry attempts per transient agent API failure; default preserves existing runs.")
    run.add_argument("--turn-checkpoint", action="store_true",
                     help="Resume from the last completed agent turn after an API failure.")
    run.add_argument("--nvidia-stream", action="store_true",
                     help="Stream Nemotron chat turns while retaining complete turn traces.")
    run.add_argument("--out-root", type=Path, default=None,
                     help="Default: runs/rise_<safe-model-id>_100k")
    judge = sub.add_parser("judge", help="Judge saved answers with gpt-5.1; makes model API calls.")
    judge.add_argument("--queries", type=Path, default=ROOT / "data" / "queries_100.jsonl")
    judge.add_argument("--expected-count", type=int, default=100)
    judge.add_argument("--out-root", type=Path, default=ROOT / "runs" / "rise_gpt-5.4-mini_100k")
    judge.add_argument("--judge-model", default="gpt-5.1")
    judge.add_argument("--concurrency", type=int, default=4)
    report = sub.add_parser("report", help="Export human-readable saved trajectory timelines.")
    report.add_argument("--queries", type=Path, default=ROOT / "data" / "queries_100.jsonl")
    report.add_argument("--expected-count", type=int, default=100)
    report.add_argument("--out-root", type=Path, default=ROOT / "runs" / "rise_gpt-5.4-mini_100k")
    report.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ("queries", "raw_parquet", "raw_docs", "structured_docs",
                 "index_dir", "filename_map", "out_root", "report_dir"):
        value = getattr(args, name, None)
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command == "run" and args.out_root is None:
        args.out_root = _default_run_root(args.model)

    if args.command == "check":
        _check_inputs(args)
    elif args.command == "index":
        if not args.raw_parquet.is_file():
            raise SystemExit(f"Local raw parquet does not exist: {args.raw_parquet}")
        if args.index_dir.exists() and any(args.index_dir.iterdir()):
            raise SystemExit(f"Index directory already contains files: {args.index_dir}")
        _check_dependencies("bm25s", "Stemmer", "pyarrow", "tqdm")
        _call([str(ROOT / "scripts" / "build_bm25_index.py"),
               "--corpus", str(args.raw_parquet), "--out", str(args.index_dir)])
    elif args.command == "run":
        if args.concurrency < 1:
            raise SystemExit("--concurrency must be at least 1")
        if args.api_max_retries < 1:
            raise SystemExit("--api-max-retries must be at least 1")
        _check_inputs(args)
        _check_dependencies("bm25s", "Stemmer", "pyarrow", "openai", "dotenv")
        _require_key(args.model)
        _check_hardlink_volume(args.structured_docs, args.out_root)
        _pin_inputs(args.out_root, _input_manifest(args))
        env = dict(os.environ)
        env["REASONING_EFFORT"] = _recorded_reasoning_effort(args.model)
        command = [str(ROOT / "scripts" / "run_rise.py"),
               "--mini-dev", str(args.queries),
               "--index-dir", str(args.index_dir),
               "--filename-map", str(args.filename_map),
               "--bc-plus-docs", str(args.structured_docs),
               "--structured-docs", "--model", args.model,
               "--bm25-k", "1000", "--max-turns", "100",
               "--concurrency", str(args.concurrency),
               "--api-max-retries", str(args.api_max_retries),
               "--out-root", str(args.out_root)]
        if args.turn_checkpoint:
            command.append("--turn-checkpoint")
        if args.nvidia_stream:
            command.append("--nvidia-stream")
        _call(command, env=env)
        _check_trajectories(args.out_root, args.queries, args.expected_count)
    elif args.command == "judge":
        _check_trajectories(args.out_root, args.queries, args.expected_count)
        _check_dependencies("openai", "dotenv")
        _require_key(args.judge_model)
        _call([str(ROOT / "scripts" / "judge.py"), "--run-dir", str(args.out_root),
               "--mode", "online", "--judge-model", args.judge_model,
               "--concurrency", str(args.concurrency)])
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in
                (args.out_root / "_per_query").glob("qid_*.json")]
        pending = [str(r.get("query_id")) for r in rows if not (r.get("judge") or {}).get("model")]
        if pending:
            raise SystemExit(f"Judging incomplete for query IDs: {pending}")
        print(f"Verified {len(rows)} judged rows.")
    else:
        report_dir = args.report_dir or args.out_root / "trajectory_reports"
        command = [str(ROOT / "scripts" / "inspect_trajectories.py"),
                   "--run-dir", str(args.out_root),
                   "--expected-count", str(args.expected_count), "--out-dir", str(report_dir)]
        if args.queries.is_file():
            command.extend(("--queries", str(args.queries)))
        _call(command)


if __name__ == "__main__":
    main()
