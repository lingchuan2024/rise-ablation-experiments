#!/usr/bin/env python3
"""Combine a fixed 100-query DeepSeek run with successful retry results.

Keep the original and retry directories intact. A retry replaces an original
result only when it has a complete judged row and a nonempty trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-v4.1-flash"


def _valid_row(run_dir: Path, qid: str) -> tuple[Path, Path] | None:
    row = run_dir / "_per_query" / f"qid_{qid}.json"
    trace = run_dir / "_traces" / f"qid_{qid}" / "single.json"
    if not (row.is_file() and trace.is_file() and trace.stat().st_size > 0):
        return None
    data = json.loads(row.read_text(encoding="utf-8"))
    if data.get("model") != MODEL or data.get("terminated_by") in {"api_error", "context_limit"}:
        return None
    if not (data.get("judge") or {}).get("reasoning"):
        return None
    return row, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "runs/rise_deepseek_v41_flash_100")
    parser.add_argument("--retry", type=Path, default=ROOT / "runs/rise_deepseek_v41_flash_retry14")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/rise_deepseek_v41_flash_merged100")
    args = parser.parse_args()
    ids = [str(json.loads(line)["query_id"]) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 100 or len(set(ids)) != 100:
        parser.error("Expected 100 distinct fixed queries")
    retry_status = args.retry / "_per_query"
    if not retry_status.is_dir():
        parser.error("Retry directory has no results yet")
    args.out_root.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, str] = {}
    blocked: list[str] = []
    correct = 0
    for qid in ids:
        source = "retry" if _valid_row(args.retry, qid) else "original"
        source_dir = args.retry if source == "retry" else args.original
        selected = _valid_row(source_dir, qid)
        original_trace = args.original / "_traces" / f"qid_{qid}" / "single.json"
        trace_source = selected[1] if selected else original_trace
        if not trace_source.is_file():
            raise FileNotFoundError(f"Missing trajectory for query {qid}")
        trace_target = args.out_root / "_traces" / f"qid_{qid}" / "single.json"
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_source, trace_target)
        if selected:
            row_target = args.out_root / "_per_query" / f"qid_{qid}.json"
            row_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected[0], row_target)
            correct += bool(json.loads(selected[0].read_text(encoding="utf-8")).get("is_correct"))
            provenance[qid] = source
        else:
            blocked.append(qid)
            provenance[qid] = "blocked"
    summary = {
        "expected": 100,
        "traces": 100,
        "judged": 100 - len(blocked),
        "correct": correct,
        "restored_by_retry": sum(source == "retry" for source in provenance.values()),
        "blocked_ids": blocked,
        "provenance": provenance,
    }
    (args.out_root / "_merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "provenance"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
