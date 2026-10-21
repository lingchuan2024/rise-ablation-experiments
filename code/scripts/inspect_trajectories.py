#!/usr/bin/env python3
"""Inspect local RISE trajectories without running agents or loading a corpus.

Reads the files produced by ``scripts/run_rise.py`` under ``--run-dir``:
``_per_query/qid_<id>.json`` and ``_traces/qid_<id>/single.json``. Writes one
Markdown timeline per query, a JSONL summary, and an integrity report to a
separate output directory. A nonzero exit status means the expected query set
is incomplete or the paired files disagree.

Example::

    python scripts/inspect_trajectories.py --run-dir runs/rise_example \
        --queries data/queries_100.jsonl --expected-count 100 \
        --out-dir runs/rise_example/_trajectory_report

``--queries`` is optional. Without it, the script verifies the count and
that the per-query and trace ID sets agree; it cannot name a query missing
from *both* sets. No model API, index, corpus, or package dependencies are
used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


# Mirrors the v1.1 row builder in src/rise/trajectory.py. Kept local so this
# read-only inspector works with Python's standard library alone.
_ROW_REQUIRED = (
    "schema_version", "protocol_version", "runner", "runner_mode", "model",
    "run_config_hash", "run_config", "query_id", "question", "gold_answer",
    "gold_doc_ids", "evidence_doc_ids", "n_gold", "n_evidence", "final_text",
    "judge", "is_correct", "n_surfaced", "surfaced_relpaths", "all_retrieved",
    "coverage_any", "coverage_mean", "coverage_all", "evidence_coverage_mean",
    "terminated_by", "elapsed_seconds", "n_turns", "n_attempts", "max_calls_hit",
    "wall_clock_hit", "agent_usage", "agent_cost_usd", "discarded_attempt_audit",
    "judge_usage", "judge_cost_usd", "total_cost_usd", "tool_usage",
    "runner_specific", "trace_path",
)
_TRACE_REQUIRED = (
    "schema_version", "model", "system_prompt", "user_prompt", "final_text",
    "terminated_by", "error", "elapsed_seconds", "totals", "turns",
)


def _sort_key(qid: str) -> tuple[int, int | str, str]:
    return (0, int(qid), qid) if qid.isdecimal() else (1, qid, qid)


def _issue(level: str, qid: str | None, message: str) -> dict[str, str | None]:
    return {"level": level, "query_id": qid, "message": message}


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot parse {path}: {type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return None, f"{path} must contain a JSON object"
    return obj, None


def _index_inputs(run_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    rows = {
        path.stem[len("qid_"):]: path
        for path in (run_dir / "_per_query").glob("qid_*.json")
        if path.is_file()
    }
    traces = {
        path.parent.name[len("qid_"):]: path
        for path in (run_dir / "_traces").glob("qid_*/single.json")
        if path.is_file()
    }
    return rows, traces


def _query_ids(path: Path) -> tuple[set[str], list[dict[str, str | None]]]:
    ids: set[str] = set()
    issues: list[dict[str, str | None]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(_issue("error", None, f"{path}:{line_no}: invalid JSON: {exc}"))
                    continue
                if not isinstance(rec, dict) or "query_id" not in rec:
                    issues.append(_issue("error", None, f"{path}:{line_no}: missing query_id"))
                    continue
                qid = str(rec["query_id"])
                if qid in ids:
                    issues.append(_issue("error", qid, f"{path}:{line_no}: duplicate query_id"))
                ids.add(qid)
    except (OSError, UnicodeError) as exc:
        issues.append(_issue("error", None, f"cannot read queries file {path}: {exc}"))
    return ids, issues


def _same_number(a: Any, b: Any, *, abs_tol: float = 1e-8) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=abs_tol)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_pair(qid: str, row: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, str | None]]:
    issues: list[dict[str, str | None]] = []

    def fail(message: str) -> None:
        issues.append(_issue("error", qid, message))

    missing_row = [key for key in _ROW_REQUIRED if key not in row]
    missing_trace = [key for key in _TRACE_REQUIRED if key not in trace]
    if missing_row:
        fail(f"row missing required v1.1 fields: {', '.join(missing_row)}")
    if missing_trace:
        fail(f"trace missing required fields: {', '.join(missing_trace)}")
    if str(row.get("query_id")) != qid:
        fail(f"row query_id {row.get('query_id')!r} differs from filename ID {qid!r}")
    if row.get("runner") != "rise":
        fail(f"row runner is {row.get('runner')!r}, expected 'rise'")
    if not isinstance(row.get("question"), str):
        fail("row question is not a string")
    elif trace.get("user_prompt") != f"QUESTION:\n{row['question']}":
        fail("trace user_prompt does not match row question")
    if not isinstance(row.get("run_config_hash"), str) or not row["run_config_hash"]:
        fail("row run_config_hash is missing")
    expected_pointer = f"_traces/qid_{qid}/single.json"
    if row.get("trace_path") != expected_pointer:
        fail(f"row trace_path is {row.get('trace_path')!r}, expected {expected_pointer!r}")
    if row.get("schema_version") != trace.get("schema_version"):
        fail("schema_version differs between row and trace")
    if row.get("model") != trace.get("model"):
        fail("model differs between row and trace")
    if row.get("final_text") != trace.get("final_text"):
        fail("final_text differs between row and trace")
    trace_termination = trace.get("terminated_by")
    row_termination = row.get("terminated_by")
    # A context window overflow is a terminal model outcome. Its detailed
    # provider error stays in the trace while the per-query status is stable.
    expected_termination = (
        trace_termination if trace_termination == "context_limit"
        else f"{trace_termination}: {trace['error']}"
        if trace.get("error") else trace_termination
    )
    if row_termination != expected_termination:
        fail("terminated_by differs between row and trace")

    turns = trace.get("turns")
    totals = trace.get("totals")
    if not isinstance(turns, list):
        fail("trace turns is not a list")
        return issues
    if not isinstance(totals, dict):
        fail("trace totals is not an object")
        return issues
    if not all(isinstance(turn, dict) for turn in turns):
        fail("trace contains a non-object turn")
        return issues

    n_turns = len(turns)
    if row.get("n_turns") != n_turns or totals.get("n_turns") != n_turns:
        fail(f"n_turns disagree: row={row.get('n_turns')!r}, trace totals={totals.get('n_turns')!r}, turns={n_turns}")
    for index, turn in enumerate(turns):
        if turn.get("turn") != index:
            fail(f"turn index {index} stores turn={turn.get('turn')!r}")
        if not isinstance(turn.get("tool_calls"), list):
            fail(f"turn {index} tool_calls is not a list")
    for per_turn_key, total_key in (
        ("prompt", "prompt_tokens"), ("cached_prompt", "cached_prompt_tokens"),
        ("completion", "completion_tokens"), ("reasoning", "reasoning_tokens"),
    ):
        numbers = []
        for turn in turns:
            tokens = turn.get("tokens")
            if (not isinstance(tokens, dict) or not isinstance(tokens.get(per_turn_key), int)
                    or isinstance(tokens.get(per_turn_key), bool)):
                fail(f"turn {turn.get('turn')} has invalid tokens.{per_turn_key}")
                break
            numbers.append(tokens[per_turn_key])
        else:
            if totals.get(total_key) != sum(numbers):
                fail(f"trace totals.{total_key} differs from turn token sum")
    if not all(_is_finite_number(turn.get("cost_usd")) for turn in turns):
        fail("a turn has invalid cost_usd")
    elif not _same_number(totals.get("agent_cost_usd"), sum(turn["cost_usd"] for turn in turns)):
        fail("trace agent cost differs from turn cost sum")

    tool_names: list[str] = []
    for turn in turns:
        calls = turn.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                fail(f"turn {turn.get('turn')} contains an invalid tool call")
                continue
            tool_names.append(call["name"])
    breakdown = dict(Counter(tool_names))
    if totals.get("tool_call_count") != len(tool_names):
        fail("tool_call_count differs from recorded tool calls")
    if totals.get("tool_call_breakdown") != breakdown:
        fail("trace tool_call_breakdown differs from recorded tool calls")
    if row.get("tool_usage") != breakdown:
        fail("row tool_usage differs from recorded tool calls")

    token_map = {
        "input_tokens": "prompt_tokens",
        "cached_input_tokens": "cached_prompt_tokens",
        "output_tokens": "completion_tokens",
        "reasoning_tokens": "reasoning_tokens",
    }
    usage = row.get("agent_usage")
    if not isinstance(usage, dict):
        fail("row agent_usage is not an object")
    else:
        for row_key, trace_key in token_map.items():
            if usage.get(row_key) != totals.get(trace_key):
                fail(f"token count differs: agent_usage.{row_key} versus trace totals.{trace_key}")
    if not _same_number(row.get("agent_cost_usd"), totals.get("agent_cost_usd")):
        fail("agent cost differs between row and trace totals")
    return issues


def _excerpt(value: Any, max_chars: int) -> dict[str, Any]:
    text = value if isinstance(value, str) else ""
    return {"text": text[:max_chars], "chars": len(text), "truncated": len(text) > max_chars}


def _sample(values: Any, maximum: int) -> dict[str, Any]:
    seq = values if isinstance(values, list) else []
    return {"items": seq[:maximum], "count": len(seq), "truncated": len(seq) > maximum}


def _timeline(trace: dict[str, Any], max_chars: int, max_paths: int) -> list[dict[str, Any]]:
    turns = trace.get("turns")
    if not isinstance(turns, list):
        return []
    timeline: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        calls: list[dict[str, Any]] = []
        raw_calls = turn.get("tool_calls")
        for call in raw_calls if isinstance(raw_calls, list) else []:
            if not isinstance(call, dict):
                continue
            calls.append({
                "name": call.get("name"),
                "args": call.get("args"),
                "result_summary": _excerpt(call.get("result_summary"), max_chars),
                "result_paths": _sample(call.get("result_paths"), max_paths),
                "result_docids": _sample(call.get("result_docids"), max_paths),
                "result_text_preview": _excerpt(call.get("result_text_preview"), max_chars),
                "result_text": _excerpt(call.get("result_text"), max_chars),
                "result_chars": call.get("result_chars"),
                "result_truncated": call.get("result_truncated"),
                "duration_seconds": call.get("duration_seconds"),
                "error": call.get("error"),
            })
        timeline.append({
            "turn": turn.get("turn"),
            "finish_reason": turn.get("finish_reason"),
            "content": _excerpt(turn.get("content"), max_chars),
            "reasoning_content": _excerpt(turn.get("reasoning_content"), max_chars),
            "tokens": turn.get("tokens"),
            "cost_usd": turn.get("cost_usd"),
            "tool_calls": calls,
        })
    return timeline


def _markdown_code(value: Any) -> str:
    body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    run_lengths = [len(match.group()) for match in re.finditer(r"`+", body)]
    fence = "`" * max(3, max(run_lengths, default=0) + 1)
    return f"{fence}text\n{body}\n{fence}\n"


def _render_markdown(summary: dict[str, Any]) -> str:
    qid = str(summary["query_id"])
    lines = [f"# Query {qid}", "", f"Status: **{summary['status']}**", ""]
    for key, label in (("question", "Question"), ("gold_answer", "Gold answer"), ("final_text", "Final response")):
        value = summary.get(key)
        if value is not None:
            lines.extend([f"## {label}", "", _markdown_code(value), ""])
    lines.extend([
        "## Run", "",
        f"- Model: {summary.get('model') or 'unknown'}",
        f"- Terminated by: {summary.get('terminated_by') or 'unknown'}",
        f"- Turns: {summary.get('n_turns') if summary.get('n_turns') is not None else 'unknown'}",
        f"- Agent cost (USD): {summary.get('agent_cost_usd') if summary.get('agent_cost_usd') is not None else 'unknown'}",
        f"- Trace: {summary.get('trace_path') or 'missing'}",
        "",
    ])
    if summary["issues"]:
        lines.extend(["## Integrity issues", ""])
        lines.extend(f"- {item['level']}: {item['message']}" for item in summary["issues"])
        lines.append("")
    for turn in summary["timeline"]:
        lines.extend([
            f"## Turn {turn['turn']}", "",
            f"Finish reason: {turn['finish_reason'] or 'unknown'}; tokens: {turn['tokens']}; cost: {turn['cost_usd']} USD", "",
        ])
        for key, label in (("content", "Assistant content"), ("reasoning_content", "Reasoning content")):
            excerpt = turn[key]
            if excerpt["chars"]:
                suffix = f" (first {len(excerpt['text'])} of {excerpt['chars']} characters)" if excerpt["truncated"] else ""
                lines.extend([f"### {label}{suffix}", "", _markdown_code(excerpt["text"]), ""])
        for index, call in enumerate(turn["tool_calls"], 1):
            lines.extend([f"### Tool {index}: {call['name']}", "", "Arguments:", "", _markdown_code(call["args"]), ""])
            for key, label in (("result_summary", "Result summary"), ("result_text_preview", "Result preview"), ("result_text", "Stored result text")):
                excerpt = call[key]
                if excerpt["chars"]:
                    suffix = f" (first {len(excerpt['text'])} of {excerpt['chars']} characters)" if excerpt["truncated"] else ""
                    lines.extend([f"{label}{suffix}:", "", _markdown_code(excerpt["text"]), ""])
            for key, label in (("result_paths", "Paths"), ("result_docids", "Doc IDs")):
                sample = call[key]
                if sample["count"]:
                    suffix = f" (first {len(sample['items'])} of {sample['count']})" if sample["truncated"] else ""
                    lines.extend([f"{label}{suffix}:", "", _markdown_code(sample["items"]), ""])
            if call["error"]:
                lines.extend(["Tool error:", "", _markdown_code(call["error"]), ""])
            lines.extend([
                f"Result characters: {call['result_chars']}; source truncated: {call['result_truncated']}; "
                f"duration: {call['duration_seconds']} s", "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def _safe_filename(qid: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", qid):
        return f"qid_{qid}.md"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", qid)[:60].strip("_") or "id"
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:10]
    return f"qid_{slug}_{digest}.md"


def inspect_run(
    run_dir: Path,
    out_dir: Path,
    *,
    queries: Path | None = None,
    expected_count: int = 100,
    max_text_chars: int = 1000,
    max_paths: int = 20,
) -> dict[str, Any]:
    """Generate reports and return the integrity report. Inputs are read only."""
    if expected_count < 0 or max_text_chars < 1 or max_paths < 1:
        raise ValueError("expected_count must be >= 0; max_text_chars and max_paths must be >= 1")
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    input_dirs = (run_dir / "_per_query", run_dir / "_traces")
    if out_dir == run_dir or any(out_dir == root or root in out_dir.parents for root in input_dirs):
        raise ValueError("out_dir must be separate from run_dir and its input directories")
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")

    row_paths, trace_paths = _index_inputs(run_dir)
    row_ids, trace_ids = set(row_paths), set(trace_paths)
    expected_ids: set[str] | None = None
    issues: list[dict[str, str | None]] = []
    if queries is not None:
        expected_ids, query_issues = _query_ids(queries)
        issues.extend(query_issues)
        if expected_count and len(expected_ids) != expected_count:
            issues.append(_issue("error", None, f"queries file contains {len(expected_ids)} unique IDs; expected {expected_count}"))

    if expected_count and len(row_ids) != expected_count:
        issues.append(_issue("error", None, f"per-query file count is {len(row_ids)}; expected {expected_count}"))
    if expected_count and len(trace_ids) != expected_count:
        issues.append(_issue("error", None, f"trace file count is {len(trace_ids)}; expected {expected_count}"))

    missing_row_ids = (expected_ids - row_ids) if expected_ids is not None else (trace_ids - row_ids)
    missing_trace_ids = (expected_ids - trace_ids) if expected_ids is not None else (row_ids - trace_ids)
    extra_row_ids = (row_ids - expected_ids) if expected_ids is not None else set()
    extra_trace_ids = (trace_ids - expected_ids) if expected_ids is not None else set()
    for label, ids in (("missing per-query file", missing_row_ids), ("missing trace file", missing_trace_ids),
                       ("unexpected per-query file", extra_row_ids), ("unexpected trace file", extra_trace_ids)):
        for qid in sorted(ids, key=_sort_key):
            issues.append(_issue("error", qid, label))

    # Include empty qid directories so interrupted trace writes are visible.
    for path in (run_dir / "_traces").glob("qid_*"):
        if path.is_dir() and not (path / "single.json").is_file():
            qid = path.name[len("qid_"):]
            if qid not in missing_trace_ids:
                issues.append(_issue("error", qid, "trace directory has no single.json"))

    all_ids = row_ids | trace_ids | (expected_ids or set())
    summaries: list[dict[str, Any]] = []
    for qid in sorted(all_ids, key=_sort_key):
        row, row_error = _read_json(row_paths[qid]) if qid in row_paths else (None, None)
        trace, trace_error = _read_json(trace_paths[qid]) if qid in trace_paths else (None, None)
        local_issues: list[dict[str, str | None]] = []
        for error in (row_error, trace_error):
            if error:
                local_issues.append(_issue("error", qid, error))
        if row is not None and trace is not None:
            local_issues.extend(_check_pair(qid, row, trace))
        issues.extend(local_issues)
        status = "paired" if row is not None and trace is not None else (
            "row_only" if row is not None else "trace_only" if trace is not None else "missing"
        )
        if local_issues:
            status = "invalid"
        summaries.append({
            "query_id": qid,
            "status": status,
            "question": row.get("question") if row else None,
            "gold_answer": row.get("gold_answer") if row else None,
            "final_text": row.get("final_text") if row else trace.get("final_text") if trace else None,
            "model": row.get("model") if row else trace.get("model") if trace else None,
            "terminated_by": row.get("terminated_by") if row else trace.get("terminated_by") if trace else None,
            "n_turns": row.get("n_turns") if row else (trace["totals"].get("n_turns") if isinstance(trace.get("totals"), dict) else None) if trace else None,
            "agent_cost_usd": row.get("agent_cost_usd") if row else (trace["totals"].get("agent_cost_usd") if isinstance(trace.get("totals"), dict) else None) if trace else None,
            "trace_path": str(trace_paths[qid].relative_to(run_dir)) if qid in trace_paths else None,
            "judge_pending": bool(row and isinstance(row.get("judge"), dict) and not row["judge"].get("model")),
            "issues": local_issues + [item for item in issues if item["query_id"] == qid and item not in local_issues],
            "timeline": _timeline(trace, max_text_chars, max_paths) if trace else [],
        })

    report = {
        "run_dir": str(run_dir),
        "queries": str(queries.resolve()) if queries is not None else None,
        "expected_count": expected_count,
        "expected_ids_known": expected_ids is not None,
        "per_query_count": len(row_ids),
        "trace_count": len(trace_ids),
        "paired_count": len(row_ids & trace_ids),
        "missing_per_query_ids": sorted(missing_row_ids, key=_sort_key),
        "missing_trace_ids": sorted(missing_trace_ids, key=_sort_key),
        "unexpected_per_query_ids": sorted(extra_row_ids, key=_sort_key),
        "unexpected_trace_ids": sorted(extra_trace_ids, key=_sort_key),
        "judge_pending_count": sum(s["judge_pending"] for s in summaries),
        "issues": issues,
        "complete": not any(item["level"] == "error" for item in issues),
    }
    report["note"] = (
        "Without --queries, IDs absent from both input sets cannot be identified; count is checked."
        if expected_ids is None else "Expected IDs were read from --queries."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    md_dir = out_dir / "queries"
    md_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        (md_dir / _safe_filename(summary["query_id"])).write_text(
            _render_markdown(summary), encoding="utf-8"
        )
    (out_dir / "timelines.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in summaries), encoding="utf-8"
    )
    (out_dir / "integrity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_lines = [
        "# RISE trajectory integrity", "",
        f"- Run directory: `{run_dir}`",
        f"- Expected queries: {expected_count if expected_count else 'unspecified'}",
        f"- Per-query files: {len(row_ids)}",
        f"- Trace files: {len(trace_ids)}",
        f"- Paired files: {len(row_ids & trace_ids)}",
        f"- Judge pending: {report['judge_pending_count']}",
        f"- Complete: **{'yes' if report['complete'] else 'no'}**", "",
        report["note"], "",
        "## Issues", "",
    ]
    md_lines.extend(
        f"- {item['level']} ({item['query_id'] if item['query_id'] is not None else 'run'}): {item['message']}"
        for item in issues
    )
    if not issues:
        md_lines.append("- None")
    md_lines.extend(["", "## Query timelines", ""])
    md_lines.extend(
        f"- [{s['query_id']}](queries/{_safe_filename(s['query_id'])}) — {s['status']}"
        for s in summaries
    )
    (out_dir / "integrity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=None, help="Optional local query JSONL for exact ID checks")
    ap.add_argument("--expected-count", type=int, default=100, help="Expected number of queries; 0 disables count check")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-text-chars", type=int, default=1000, help="Per-field excerpt limit in summaries")
    ap.add_argument("--max-paths", type=int, default=20, help="Per-tool path/docid sample limit")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.run_dir / "_trajectory_report"
    try:
        report = inspect_run(
            args.run_dir, out_dir, queries=args.queries, expected_count=args.expected_count,
            max_text_chars=args.max_text_chars, max_paths=args.max_paths,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{'complete' if report['complete'] else 'incomplete'}: "
        f"{report['paired_count']} paired / {report['expected_count'] or 'unspecified'} expected; "
        f"{len(report['issues'])} issues; report: {out_dir.resolve()}"
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
