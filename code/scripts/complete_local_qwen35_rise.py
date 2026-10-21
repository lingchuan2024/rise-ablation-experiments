#!/usr/bin/env python3
"""Resume the fixed 100-question RISE run against the local Qwen3.5 vLLM server.

The server URL is supplied in RISE_LOCAL_VLLM_BASE_URL. Inputs are already
present on this machine; this script neither downloads nor builds a corpus.
Each two-question batch is checkpointed, judged, and counted before moving on.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.5-27b"
TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qid(line: str) -> str:
    return str(json.loads(line)["query_id"])


def _row(out_root: Path, qid: str) -> dict | None:
    path = out_root / "_per_query" / f"qid_{qid}.json"
    return _read(path) if path.is_file() else None


def _complete(out_root: Path, qid: str) -> bool:
    row = _row(out_root, qid)
    trace = out_root / "_traces" / f"qid_{qid}" / "single.json"
    return bool(
        row and row.get("model") == MODEL
        and row.get("terminated_by") not in {"api_error", "context_limit"}
        and trace.is_file() and trace.stat().st_size
    )


def _status(path: Path, out_root: Path, stage: str, batch: int) -> None:
    rows = [_read(p) for p in (out_root / "_per_query").glob("qid_*.json")]
    usable = [row for row in rows if _complete(out_root, str(row.get("query_id")))]
    judged = [row for row in usable if (row.get("judge") or {}).get("reasoning")]
    payload = {
        "stage": stage, "batch": batch, "expected": 100,
        "rows": len(usable),
        "traces": sum(bool((out_root / "_traces" / f"qid_{row['query_id']}" / "single.json").is_file()) for row in usable),
        "judged": len(judged), "correct": sum(bool(row.get("is_correct")) for row in judged),
        "reasoning_tokens": sum(int(row.get("agent_usage", {}).get("reasoning_tokens") or 0) for row in usable),
        "input_tokens": sum(int(row.get("agent_usage", {}).get("input_tokens") or 0) for row in usable),
        "output_tokens": sum(int(row.get("agent_usage", {}).get("output_tokens") or 0) for row in usable),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print("STATUS:", json.dumps(payload), flush=True)


def _call(argv: list[str]) -> int:
    print("COMMAND:", " ".join(argv), flush=True)
    rc = subprocess.run([sys.executable, *argv], cwd=ROOT, check=False).returncode
    print("EXIT:", rc, flush=True)
    return rc


def _checkpoint_error(out_root: Path, qids: list[str]) -> str:
    parts = []
    for qid in qids:
        path = out_root / "_checkpoints" / f"qid_{qid}.json"
        if path.is_file():
            parts.append(str((_read(path).get("run") or {}).get("error") or ""))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/rise_qwen35_27b_100")
    parser.add_argument("--status-out", type=Path, default=ROOT / "logs/qwen35_full100_status.json")
    parser.add_argument("--audit-out", type=Path, default=ROOT / "logs/qwen35_full100_audit.json")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--per-call-max-tokens", type=int, default=8192)
    parser.add_argument("--transient-retries", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("RISE_LOCAL_VLLM_BASE_URL"):
        parser.error("RISE_LOCAL_VLLM_BASE_URL is required")
    if args.concurrency < 1 or args.per_call_max_tokens < 1 or args.transient_retries < 1:
        parser.error("concurrency, token budget, and retry count must be positive")
    for name in ("queries", "out_root", "status_out", "audit_out"):
        setattr(args, name, getattr(args, name).resolve())
    lines = [line for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 100 or len({_qid(line) for line in lines}) != 100:
        parser.error("Expected the 100 distinct fixed queries")
    args.out_root.mkdir(parents=True, exist_ok=True)
    batch_root = args.out_root / "_batch_inputs"
    batch_root.mkdir(exist_ok=True)
    for start in range(0, 100, 2):
        batch = start // 2 + 1
        batch_lines = lines[start:start + 2]
        qids = [_qid(line) for line in batch_lines]
        pending = [line for line in batch_lines if not _complete(args.out_root, _qid(line))]
        if pending:
            batch_path = batch_root / f"batch_{batch:02d}.jsonl"
            for attempt in range(1, args.transient_retries + 1):
                batch_path.write_text("\n".join(pending) + "\n", encoding="utf-8")
                _status(args.status_out, args.out_root, "agent", batch)
                rc = _call([
                    str(ROOT / "scripts/run_rise.py"), "--mini-dev", str(batch_path),
                    "--index-dir", str(ROOT / "runs/bm25_full"),
                    "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
                    "--bc-plus-docs", str(ROOT / "corpus/bcp_plus_structured"),
                    "--structured-docs", "--model", MODEL, "--bm25-k", "1000",
                    "--max-turns", "100", "--per-call-max-tokens", str(args.per_call_max_tokens),
                    "--concurrency", str(args.concurrency), "--api-max-retries", "3",
                    "--turn-checkpoint", "--out-root", str(args.out_root),
                ])
                pending = [line for line in pending if not _complete(args.out_root, _qid(line))]
                if not pending:
                    break
                if any((_row(args.out_root, _qid(line)) or {}).get("terminated_by") == "context_limit" for line in pending):
                    _status(args.status_out, args.out_root, "context_limit", batch)
                    return 2
                error = _checkpoint_error(args.out_root, [_qid(line) for line in pending])
                if rc == 0 or not any(marker in error.lower() for marker in TRANSIENT_MARKERS):
                    print("AGENT_FAILURE:", error[:500], flush=True)
                    _status(args.status_out, args.out_root, "agent_failed", batch)
                    return 2
                if attempt == args.transient_retries:
                    _status(args.status_out, args.out_root, "agent_failed", batch)
                    return 2
                _status(args.status_out, args.out_root, "retrying", batch)
                time.sleep(min(30 * attempt, 120))
        if _call([
            str(ROOT / "scripts/judge.py"), "--run-dir", str(args.out_root),
            "--mode", "online", "--judge-model", MODEL,
            "--concurrency", str(args.concurrency),
        ]):
            _status(args.status_out, args.out_root, "judge_failed", batch)
            return 3
        _status(args.status_out, args.out_root, "running", batch)

    for command in (
        [str(ROOT / "scripts/run_full_rise.py"), "report", "--queries", str(args.queries),
         "--expected-count", "100", "--out-root", str(args.out_root)],
        [str(ROOT / "scripts/audit_rise_trajectories.py"), "--run-dir", str(args.out_root),
         "--queries", str(args.queries), "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
         "--out", str(args.audit_out)],
    ):
        if _call(command):
            _status(args.status_out, args.out_root, "finalize_failed", 50)
            return 4
    _status(args.status_out, args.out_root, "complete", 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
