#!/usr/bin/env python3
"""Run fixed RISE queries two at a time with Qwen Token Plan, then judge.

The batch boundary limits wasted requests when the plan reports
the explicit monthly-plan exhaustion message. Completed rows, traces, and turn checkpoints are
reused on restart. Credentials come only from the process environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.8-flash"
QUOTA_MARKERS = ("token-plan quota has been exhausted",)
TRANSIENT_MARKERS = ("rate limit", "429", "timeout", "timed out", "connection", "server error", "502", "503", "504")
MODERATION_MARKER = "data_inspection_failed"


def _safe(text: str) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"sk-sp-[A-Za-z0-9._-]+", "[REDACTED]", text)


def _call(argv: list[str], *, capture: bool = False) -> tuple[int, str]:
    print("COMMAND:", " ".join(argv), flush=True)
    if capture:
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output = _safe(result.stdout or "")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        return result.returncode, output
    result = subprocess.run([sys.executable, *argv], cwd=ROOT, check=False)
    print("EXIT:", result.returncode, flush=True)
    return result.returncode, ""


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(out_root: Path, expected: int) -> dict:
    rows = [_read(path) for path in sorted((out_root / "_per_query").glob("qid_*.json"))]
    return {
        "expected": expected,
        "rows": len(rows),
        "traces": len(list((out_root / "_traces").glob("qid_*/single.json"))),
        "judged": sum(bool((row.get("judge") or {}).get("reasoning")) for row in rows),
        "correct": sum(bool(row.get("is_correct")) for row in rows if (row.get("judge") or {}).get("reasoning")),
        "input_tokens": sum(int((row.get("agent_usage") or {}).get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int((row.get("agent_usage") or {}).get("output_tokens") or 0) for row in rows),
    }


def _status(path: Path, out_root: Path, expected: int, stage: str, batch: int) -> dict:
    payload = {"stage": stage, "batch": batch, **_snapshot(out_root, expected),
               "blocked_ids": sorted(_blocked_ids(out_root))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print("STATUS:", json.dumps(payload), flush=True)
    return payload


def _qid(line: str) -> str:
    return str(json.loads(line)["query_id"])


def _complete(out_root: Path, qid: str) -> bool:
    row = out_root / "_per_query" / f"qid_{qid}.json"
    trace = out_root / "_traces" / f"qid_{qid}" / "single.json"
    if not (row.is_file() and trace.is_file() and trace.stat().st_size):
        return False
    data = _read(row)
    return data.get("model") == MODEL and data.get("terminated_by") not in {"api_error", "context_limit"}


def _checkpoint_errors(out_root: Path, qids: list[str]) -> str:
    errors = []
    for qid in qids:
        path = out_root / "_checkpoints" / f"qid_{qid}.json"
        if path.is_file():
            try:
                errors.append(str((_read(path).get("run") or {}).get("error") or ""))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return _safe("\n".join(errors))


def _has_quota_error(text: str) -> bool:
    return any(marker in text.lower() for marker in QUOTA_MARKERS)


def _blocked_ids(out_root: Path) -> set[str]:
    path = out_root / "_blocked_queries.json"
    return set(_read(path)) if path.is_file() else set()


def _record_moderation_blocks(out_root: Path, qids: list[str]) -> set[str]:
    blocked = _blocked_ids(out_root)
    for qid in qids:
        if MODERATION_MARKER in _checkpoint_errors(out_root, [qid]).lower():
            blocked.add(qid)
    path = out_root / "_blocked_queries.json"
    if blocked:
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps({qid: MODERATION_MARKER for qid in sorted(blocked)}, indent=2) + "\n")
        temp.replace(path)
    return blocked


def _judge(out_root: Path) -> tuple[bool, str]:
    _, output = _call([
        str(ROOT / "scripts/judge.py"), "--run-dir", str(out_root),
        "--mode", "online", "--judge-model", MODEL, "--concurrency", "2",
    ], capture=True)
    rows = [_read(p) for p in (out_root / "_per_query").glob("qid_*.json")]
    return all(bool((r.get("judge") or {}).get("reasoning")) for r in rows), output


def _partial_artifacts(out_root: Path, lines: list[str], audit_out: Path) -> None:
    completed = [line for line in lines if _complete(out_root, _qid(line))]
    if not completed:
        return
    view = out_root / "_completed_view"
    (view / "_per_query").mkdir(parents=True, exist_ok=True)
    (view / "_traces").mkdir(exist_ok=True)
    for line in completed:
        qid = _qid(line)
        row = out_root / "_per_query" / f"qid_{qid}.json"
        trace = out_root / "_traces" / f"qid_{qid}" / "single.json"
        row_link = view / "_per_query" / row.name
        trace_link = view / "_traces" / f"qid_{qid}" / "single.json"
        trace_link.parent.mkdir(parents=True, exist_ok=True)
        if not row_link.exists():
            row_link.symlink_to(row)
        if not trace_link.exists():
            trace_link.symlink_to(trace)
    subset = view / "queries_completed.jsonl"
    subset.write_text("\n".join(completed) + "\n", encoding="utf-8")
    _call([
        str(ROOT / "scripts/inspect_trajectories.py"), "--run-dir", str(view),
        "--queries", str(subset), "--expected-count", str(len(completed)),
        "--out-dir", str(out_root / "trajectory_reports_completed"),
    ])
    _call([
        str(ROOT / "scripts/audit_rise_trajectories.py"), "--run-dir", str(view),
        "--queries", str(subset), "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
        "--out", str(audit_out),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/rise_qwen38_flash_100")
    parser.add_argument("--status-out", type=Path, default=ROOT / "logs/qwen_full100_status.json")
    parser.add_argument("--audit-out", type=Path, default=ROOT / "logs/qwen_full100_audit.json")
    parser.add_argument("--retry-delay", type=int, default=60)
    parser.add_argument("--max-transient-retries", type=int, default=12)
    parser.add_argument("--per-call-max-tokens", type=int, default=32000)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()
    if args.retry_delay < 0 or args.max_transient_retries < 1 or args.per_call_max_tokens < 1 or args.expected_count < 1:
        parser.error("retry delay must be nonnegative; retries, token budget, and expected count must be positive")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error("DASHSCOPE_API_KEY is required in the process environment")
    for name in ("queries", "out_root", "status_out", "audit_out"):
        setattr(args, name, getattr(args, name).resolve())
    lines = [line for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [_qid(line) for line in lines]
    if len(lines) != args.expected_count or len(set(qids)) != args.expected_count:
        parser.error(f"Expected {args.expected_count} distinct fixed queries")
    args.out_root.mkdir(parents=True, exist_ok=True)
    batch_root = args.out_root / "_batch_inputs"
    batch_root.mkdir(exist_ok=True)
    blocked = _blocked_ids(args.out_root)

    for start in range(0, len(lines), 2):
        batch = start // 2 + 1
        batch_lines = lines[start:start + 2]
        batch_path = batch_root / f"batch_{batch:02d}.jsonl"
        pending = [line for line in batch_lines
                   if not _complete(args.out_root, _qid(line)) and _qid(line) not in blocked]
        if pending:
            for attempt in range(1, args.max_transient_retries + 1):
                batch_path.write_text("\n".join(pending) + "\n", encoding="utf-8")
                _status(args.status_out, args.out_root, args.expected_count, "agent", batch)
                rc, _ = _call([
                    str(ROOT / "scripts/run_rise.py"),
                    "--mini-dev", str(batch_path),
                    "--index-dir", str(ROOT / "runs/bm25_full"),
                    "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
                    "--bc-plus-docs", str(ROOT / "corpus/bcp_plus_structured"),
                    "--structured-docs", "--model", MODEL,
                    "--bm25-k", "1000", "--max-turns", "100",
                    "--per-call-max-tokens", str(args.per_call_max_tokens),
                    "--concurrency", "2", "--api-max-retries", "3",
                    "--turn-checkpoint", "--out-root", str(args.out_root),
                ])
                blocked = _record_moderation_blocks(args.out_root, [_qid(line) for line in pending])
                pending = [line for line in pending
                           if not _complete(args.out_root, _qid(line)) and _qid(line) not in blocked]
                if not pending:
                    break
                error = _checkpoint_errors(args.out_root, [_qid(line) for line in pending])
                if _has_quota_error(error):
                    _partial_artifacts(args.out_root, lines, args.audit_out)
                    _status(args.status_out, args.out_root, args.expected_count, "quota_exhausted", batch)
                    return 0
                if not error or not any(marker in error.lower() for marker in TRANSIENT_MARKERS):
                    print("AGENT_FAILURE:", error[:500], flush=True)
                    _status(args.status_out, args.out_root, args.expected_count, "agent_failed", batch)
                    return 2
                if attempt == args.max_transient_retries:
                    _status(args.status_out, args.out_root, args.expected_count, "agent_failed", batch)
                    return 2
                _status(args.status_out, args.out_root, args.expected_count, "rate_limited", batch)
                time.sleep(min(args.retry_delay * attempt, 300))

        for attempt in range(1, args.max_transient_retries + 1):
            done, judge_output = _judge(args.out_root)
            if done:
                break
            if _has_quota_error(judge_output):
                _partial_artifacts(args.out_root, lines, args.audit_out)
                _status(args.status_out, args.out_root, args.expected_count, "quota_exhausted", batch)
                return 0
            if attempt == args.max_transient_retries:
                _status(args.status_out, args.out_root, args.expected_count, "judge_failed", batch)
                return 3
            _status(args.status_out, args.out_root, args.expected_count, "rate_limited", batch)
            time.sleep(min(args.retry_delay * attempt, 300))
        _status(args.status_out, args.out_root, args.expected_count, "running", batch)

    if blocked:
        _partial_artifacts(args.out_root, lines, args.audit_out)
        _status(args.status_out, args.out_root, args.expected_count, "complete_with_blocked_questions", batch)
        return 0

    rc, _ = _call([
        str(ROOT / "scripts/run_full_rise.py"), "run", "--queries", str(args.queries),
        "--expected-count", str(args.expected_count), "--model", MODEL, "--turn-checkpoint",
        "--concurrency", "2", "--api-max-retries", "3", "--out-root", str(args.out_root),
    ])
    if rc:
        _status(args.status_out, args.out_root, args.expected_count, "finalize_failed", batch)
        return 4
    for argv in (
        [str(ROOT / "scripts/run_full_rise.py"), "report", "--queries", str(args.queries),
         "--expected-count", str(args.expected_count), "--out-root", str(args.out_root)],
        [str(ROOT / "scripts/audit_rise_trajectories.py"), "--run-dir", str(args.out_root),
         "--queries", str(args.queries), "--filename-map", str(ROOT / "runs/corpus_filename_docid_map.json"),
         "--out", str(args.audit_out)],
    ):
        rc, _ = _call(argv)
        if rc:
            _status(args.status_out, args.out_root, args.expected_count, "finalize_failed", batch)
            return 4
    _status(args.status_out, args.out_root, args.expected_count, "complete", batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
