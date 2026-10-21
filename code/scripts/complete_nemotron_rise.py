#!/usr/bin/env python3
"""Resume a Nemotron RISE run through judging and trajectory audit.

Run with ``python -u`` and redirect stdout/stderr to a private log. The first
agent pass uses ten workers by default; later passes reduce concurrency when
the hosted endpoint is overloaded. Saved rows and turn checkpoints make each pass safe
to resume. This script never reads or prints API credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _run(args: list[str]) -> int:
    print("COMMAND:", " ".join(args), flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT, check=False)
    print("EXIT:", result.returncode, flush=True)
    return result.returncode


def _snapshot(out_root: Path, expected: int) -> dict:
    rows: list[dict] = []
    for path in (out_root / "_per_query").glob("qid_*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    return {
        "expected": expected,
        "rows": len(rows),
        "traces": len(list((out_root / "_traces").glob("qid_*/single.json"))),
        "judged": sum(bool((row.get("judge") or {}).get("model")) for row in rows),
        "termination_counts": dict(Counter(str(row.get("terminated_by") or "") for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/rise_nemotron_super_100")
    parser.add_argument("--audit-out", type=Path, default=ROOT / "logs/full100_audit.json")
    parser.add_argument("--status-out", type=Path, default=ROOT / "logs/full100_pipeline_status.json")
    parser.add_argument("--first-concurrency", type=int, default=10)
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=30)
    parser.add_argument("--max-run-rounds", type=int, default=10)
    parser.add_argument("--max-judge-rounds", type=int, default=8)
    args = parser.parse_args()
    if (args.expected_count < 1 or args.first_concurrency < 1 or args.api_max_retries < 1
            or args.max_run_rounds < 1
            or args.max_judge_rounds < 1 or args.retry_delay < 0):
        parser.error("counts and round limits must be positive")
    for name in ("queries", "out_root", "audit_out", "status_out"):
        setattr(args, name, getattr(args, name).resolve())

    def write_status(stage: str, round_number: int = 0) -> dict:
        status = {"stage": stage, "round": round_number, **_snapshot(args.out_root, args.expected_count)}
        args.status_out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.status_out.with_suffix(args.status_out.suffix + ".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.status_out)
        print("STATUS:", json.dumps(status, ensure_ascii=False), flush=True)
        return status

    run_base = [
        str(ROOT / "scripts/run_full_rise.py"), "run",
        "--queries", str(args.queries), "--expected-count", str(args.expected_count),
        "--model", MODEL, "--turn-checkpoint", "--nvidia-stream",
        "--api-max-retries", str(args.api_max_retries),
        "--out-root", str(args.out_root),
    ]
    for number in range(1, args.max_run_rounds + 1):
        concurrency = (args.first_concurrency if number == 1
                       else min(4, args.first_concurrency) if number == 2 else 1)
        print(f"RUN ROUND {number}, concurrency={concurrency}", flush=True)
        exit_code = _run([*run_base, "--concurrency", str(concurrency)])
        status = write_status("agent", number)
        if exit_code == 0 and status["rows"] == args.expected_count \
                and status["traces"] == args.expected_count:
            break
        if number < args.max_run_rounds:
            time.sleep(args.retry_delay)
    else:
        write_status("agent_incomplete", args.max_run_rounds)
        return 2

    judge_base = [
        str(ROOT / "scripts/run_full_rise.py"), "judge",
        "--queries", str(args.queries), "--expected-count", str(args.expected_count),
        "--judge-model", MODEL, "--out-root", str(args.out_root),
    ]
    for number in range(1, args.max_judge_rounds + 1):
        concurrency = (args.first_concurrency if number == 1
                       else min(4, args.first_concurrency) if number == 2 else 1)
        print(f"JUDGE ROUND {number}, concurrency={concurrency}", flush=True)
        exit_code = _run([*judge_base, "--concurrency", str(concurrency)])
        status = write_status("judge", number)
        if exit_code == 0 and status["judged"] == args.expected_count:
            break
        if number < args.max_judge_rounds:
            time.sleep(args.retry_delay)
    else:
        write_status("judge_incomplete", args.max_judge_rounds)
        return 3

    if _run([
        str(ROOT / "scripts/run_full_rise.py"), "report",
        "--queries", str(args.queries), "--expected-count", str(args.expected_count),
        "--out-root", str(args.out_root),
    ]):
        write_status("report_failed")
        return 4
    if _run([
        str(ROOT / "scripts/audit_rise_trajectories.py"),
        "--run-dir", str(args.out_root), "--queries", str(args.queries),
        "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
        "--out", str(args.audit_out),
    ]):
        write_status("audit_failed")
        return 5
    write_status("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
