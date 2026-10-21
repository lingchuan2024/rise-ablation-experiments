#!/usr/bin/env python3
"""Read-only semantic checks for a completed RISE run.

Historical traces may store only a preview of search/read tool output. This
audit can identify suspicious calls, but cannot by itself establish the
factual correctness of the final answer.

Example:
    python scripts/audit_rise_trajectories.py \
        --run-dir runs/rise_nemotron_super_pilot10 \
        --queries data/queries_10.jsonl \
        --filename-map runs/corpus_filename_docid_map.json \
        --out logs/pilot10_audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


def _sort_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdecimal() else (1, value, value)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_queries(path: Path) -> dict[str, dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("query_id") is None:
            raise ValueError(f"{path}:{line_no}: missing query_id")
        qid = str(value["query_id"])
        if qid in queries:
            raise ValueError(f"{path}:{line_no}: duplicate query_id {qid!r}")
        queries[qid] = value
    return queries


def _load_filename_map(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = _read_object(path)
    paths = value.get("relpath_to_docid")
    if not isinstance(paths, dict):
        raise ValueError(f"{path}: missing relpath_to_docid object")
    return {str(key): str(docid) for key, docid in paths.items()}


def _ids(value: Any) -> set[str]:
    return {str(item) for item in value} if isinstance(value, list) else set()


def _normalize_path(path: str) -> str:
    path = path.strip()
    for prefix in ("./", "bc_plus_docs/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def _finding(level: str, code: str, detail: str, *, turn: int | None = None) -> dict[str, Any]:
    finding: dict[str, Any] = {"level": level, "code": code, "detail": detail}
    if turn is not None:
        finding["turn"] = turn
    return finding


def _audit_one(
    qid: str,
    expected: dict[str, Any] | None,
    row_path: Path | None,
    trace_path: Path | None,
    filename_map: dict[str, str] | None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "query_id": qid,
        "findings": findings,
        "stats": {
            "turns": 0,
            "tool_calls": 0,
            "max_tool_calls_in_turn": 0,
            "length_with_tool_calls": 0,
            "tool_error_outputs": 0,
            "missing_full_tool_outputs": 0,
            "unresolved_paths": 0,
            "repeated_read_paths": {},
            "repeated_tool_requests": [],
            "exact_answer_shortcut": False,
            "done_without_exact_answer": False,
        },
    }

    if expected is None:
        findings.append(_finding("error", "unexpected_query", "No matching fixed query"))
    if row_path is None:
        findings.append(_finding("error", "missing_row", "Per-query row is missing"))
    if trace_path is None:
        findings.append(_finding("error", "missing_trace", "Trajectory trace is missing"))
    row: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    for path, kind in ((row_path, "row"), (trace_path, "trace")):
        if path is None:
            continue
        try:
            value = _read_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            findings.append(_finding("error", f"invalid_{kind}", f"{type(exc).__name__}: {exc}"))
            continue
        if kind == "row":
            row = value
        else:
            trace = value

    if row is not None:
        if str(row.get("query_id")) != qid:
            findings.append(_finding("error", "row_query_id_mismatch", "Row query_id differs from filename"))
        if expected is not None:
            for field, source in (("question", "query"), ("gold_answer", "answer")):
                if row.get(field) != expected.get(source):
                    findings.append(_finding("error", f"{field}_mismatch",
                                             f"Row {field} differs from fixed query {source}"))
            if _ids(row.get("gold_doc_ids")) != _ids(expected.get("gold_doc_ids")):
                findings.append(_finding("error", "gold_doc_ids_mismatch",
                                         "Row gold_doc_ids differ from fixed query"))
        if row.get("terminated_by") == "done" and not re.search(
            r"^\s*Exact Answer\s*:", str(row.get("final_text") or ""), re.IGNORECASE | re.MULTILINE
        ):
            result["stats"]["done_without_exact_answer"] = True
            findings.append(_finding(
                "warning", "done_without_exact_answer",
                "Run ended naturally but final_text lacks the required Exact Answer line",
            ))

        judge = row.get("judge")
        if not isinstance(judge, dict):
            findings.append(_finding("error", "judge_missing", "Judge result is missing or malformed"))
        else:
            correct = str(judge.get("correct", "")).strip().lower()
            if correct not in ("yes", "no"):
                findings.append(_finding("error", "judge_correct_missing",
                                         "judge.correct must be yes or no"))
            elif not isinstance(row.get("is_correct"), bool) or row["is_correct"] != (correct == "yes"):
                findings.append(_finding("error", "judge_correct_mismatch",
                                         "judge.correct disagrees with is_correct"))
            if not str(judge.get("reasoning") or "").strip():
                findings.append(_finding("error", "judge_reasoning_missing", "Judge reasoning is empty"))
            if not str(judge.get("model") or "").strip():
                findings.append(_finding("error", "judge_model_missing", "Judge model is empty"))
            result["stats"]["exact_answer_shortcut"] = (
                str(judge.get("reasoning") or "").startswith("Short-circuit:")
            )

    if trace is not None:
        turns = trace.get("turns")
        if not isinstance(turns, list):
            findings.append(_finding("error", "turns_invalid", "Trace turns must be a list"))
            turns = []
        result["stats"]["turns"] = len(turns)
        if row is not None:
            if row.get("final_text") != trace.get("final_text"):
                findings.append(_finding("error", "final_text_mismatch",
                                         "Row and trace final_text differ"))
            if row.get("terminated_by") == "done" and turns and isinstance(turns[-1], dict) \
                    and turns[-1].get("finish_reason") == "length":
                findings.append(_finding("warning", "done_after_length",
                                         "Run is marked done but last model response ended by length"))

        read_counts: Counter[str] = Counter()
        read_requests: Counter[tuple[str, str, str]] = Counter()
        other_requests: Counter[tuple[str, str]] = Counter()
        max_tool_calls_turn = 0
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                findings.append(_finding("error", "turn_invalid", "Turn is not an object", turn=index))
                continue
            calls = turn.get("tool_calls")
            if not isinstance(calls, list):
                findings.append(_finding("error", "tool_calls_invalid",
                                         "Tool calls are not a list", turn=index))
                continue
            if len(calls) > result["stats"]["max_tool_calls_in_turn"]:
                result["stats"]["max_tool_calls_in_turn"] = len(calls)
                max_tool_calls_turn = index
            if str(turn.get("finish_reason") or "").lower() == "length" and calls:
                result["stats"]["length_with_tool_calls"] += 1
                findings.append(_finding(
                    "warning", "length_with_tool_calls",
                    f"Length-limited model turn still emitted {len(calls)} tool calls",
                    turn=index,
                ))
            for call in calls:
                result["stats"]["tool_calls"] += 1
                if not isinstance(call, dict):
                    findings.append(_finding("error", "tool_call_invalid",
                                             "Tool call is not an object", turn=index))
                    continue
                name = str(call.get("name") or "")
                preview = str(call.get("result_text_preview") or "")
                result_chars = call.get("result_chars")
                if isinstance(result_chars, (int, float)) and result_chars > 0 \
                        and not call.get("result_text"):
                    result["stats"]["missing_full_tool_outputs"] += 1
                if preview.lstrip().startswith("Error:") or str(call.get("error") or "").strip():
                    result["stats"]["tool_error_outputs"] += 1
                    findings.append(_finding("warning", "tool_error_output",
                                             f"{name} returned an error (preview or call.error)",
                                             turn=index))
                paths = call.get("result_unresolved_paths")
                unresolved = {str(path) for path in paths} if isinstance(paths, list) else set()
                if filename_map is not None and name in ("search", "read", "read_doc"):
                    result_paths = call.get("result_paths")
                    if isinstance(result_paths, list):
                        unresolved.update(
                            str(path) for path in result_paths
                            if _normalize_path(str(path)) not in filename_map
                        )
                if unresolved:
                    result["stats"]["unresolved_paths"] += len(unresolved)
                    findings.append(_finding("warning", "unresolved_paths",
                                             f"{name}: {sorted(unresolved)!r}", turn=index))
                if name in ("read", "read_doc", "read_passage"):
                    args = call.get("args")
                    if isinstance(args, dict):
                        target = args.get("file_path") or args.get("path") or args.get("target") \
                            or args.get("passage_id")
                        if isinstance(target, str) and target.strip():
                            normalized = _normalize_path(target)
                            read_counts[normalized] += 1
                            read_requests[(name, normalized, json.dumps(args, sort_keys=True, ensure_ascii=False))] += 1
                elif name in ("bash", "search") and isinstance(call.get("args"), dict):
                    other_requests[(name, json.dumps(call["args"], sort_keys=True, ensure_ascii=False))] += 1
        repeated = {path: count for path, count in sorted(read_counts.items()) if count >= 3}
        result["stats"]["repeated_read_paths"] = repeated
        exact_repeats = [
            {"tool": name, "path": path, "args": json.loads(args_json), "count": count}
            for (name, path, args_json), count in sorted(read_requests.items()) if count >= 3
        ]
        result["stats"]["repeated_read_requests"] = exact_repeats
        for item in exact_repeats:
            findings.append(_finding("warning", "repeated_read_request",
                                     f"{item['tool']} {item['path']!r} with the same args was called {item['count']} times"))
        repeated_other = [
            {"tool": name, "args": json.loads(args_json), "count": count}
            for (name, args_json), count in sorted(other_requests.items()) if count >= 3
        ]
        result["stats"]["repeated_tool_requests"] = repeated_other
        for item in repeated_other:
            findings.append(_finding("warning", "repeated_tool_request",
                                     f"{item['tool']} with the same args was called {item['count']} times"))
        if result["stats"]["max_tool_calls_in_turn"] >= 20:
            findings.append(_finding(
                "warning", "tool_call_burst",
                f"One model turn emitted {result['stats']['max_tool_calls_in_turn']} tool calls",
                turn=max_tool_calls_turn,
            ))

    result["error_count"] = sum(f["level"] == "error" for f in findings)
    result["warning_count"] = sum(f["level"] == "warning" for f in findings)
    return result


def audit_run(
    run_dir: Path,
    queries_path: Path,
    filename_map_path: Path | None = None,
) -> dict[str, Any]:
    queries = _load_queries(queries_path)
    filename_map = _load_filename_map(filename_map_path)
    rows = {p.stem[4:]: p for p in (run_dir / "_per_query").glob("qid_*.json")}
    traces = {p.parent.name[4:]: p for p in (run_dir / "_traces").glob("qid_*/single.json")}
    qids = sorted(set(queries) | set(rows) | set(traces), key=_sort_key)
    per_query = [
        _audit_one(qid, queries.get(qid), rows.get(qid), traces.get(qid), filename_map)
        for qid in qids
    ]
    summary = {
        "expected_queries": len(queries),
        "rows": len(rows),
        "traces": len(traces),
        "queries_audited": len(per_query),
        "errors": sum(item["error_count"] for item in per_query),
        "warnings": sum(item["warning_count"] for item in per_query),
        "queries_with_errors": sum(item["error_count"] > 0 for item in per_query),
        "queries_with_warnings": sum(item["warning_count"] > 0 for item in per_query),
        "tool_error_outputs": sum(item["stats"]["tool_error_outputs"] for item in per_query),
        "missing_full_tool_outputs": sum(
            item["stats"]["missing_full_tool_outputs"] for item in per_query
        ),
        "unresolved_paths": sum(item["stats"]["unresolved_paths"] for item in per_query),
        "exact_answer_shortcuts": sum(item["stats"]["exact_answer_shortcut"] for item in per_query),
        "queries_with_repeated_tool_requests": sum(
            bool(item["stats"]["repeated_tool_requests"]) for item in per_query
        ),
        "repeated_tool_call_excess": sum(
            request["count"] - 1 for item in per_query
            for request in item["stats"]["repeated_tool_requests"]
        ),
        "queries_with_tool_call_bursts": sum(
            item["stats"]["max_tool_calls_in_turn"] >= 20 for item in per_query
        ),
        "length_with_tool_calls": sum(
            item["stats"]["length_with_tool_calls"] for item in per_query
        ),
        "done_without_exact_answer": sum(
            item["stats"]["done_without_exact_answer"] for item in per_query
        ),
    }
    return {
        "scope": (
            "Read-only checks of fixed queries, saved rows, and trace metadata/previews. "
            "Some historical traces omit full tool outputs. This audit reports those omissions "
            "but does not establish factual answer correctness."
        ),
        "run_dir": str(run_dir),
        "queries": str(queries_path),
        "filename_map": str(filename_map_path) if filename_map_path else None,
        "summary": summary,
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--filename-map", type=Path)
    parser.add_argument("--out", type=Path, help="Write JSON outside the run directory; default: stdout")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.out is not None and args.out.resolve().is_relative_to(run_dir):
        parser.error("--out must be outside --run-dir to keep run outputs unchanged")
    try:
        report = audit_run(run_dir, args.queries, args.filename_map)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"Audit: {report['summary']['errors']} errors, "
              f"{report['summary']['warnings']} warnings; {args.out}")
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
