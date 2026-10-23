"""Read-only automatic20 tool-metadata audit; never inspect reasoning text.

May run locally against downloaded artifacts or over SSH via Python stdin.
No experiment code is imported and no files or processes are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time


def public_object(pairs):
    # Private reasoning/message content is discarded during JSON decoding.
    return {key: value for key, value in pairs if key not in {
        "reasoning_content", "thinking_content", "content", "messages"}}


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=public_object)


def inspect_query(output, qid):
    terminal = output / "_per_query" / f"qid_{qid}.json"
    checkpoint = output / "_checkpoints" / f"qid_{qid}.json"
    trace = output / "_traces" / f"qid_{qid}" / "single.json"
    if not any(path.exists() for path in (terminal, checkpoint, trace)):
        return {"query_id": qid, "status": "not_started"}
    row = load(terminal) if terminal.exists() else {}
    cp = load(checkpoint) if checkpoint.exists() else {}
    saved = cp.get("run", {})
    # The checkpoint is updated atomically per completed turn. Active in-flight
    # calls are deliberately not inferred from incomplete bytes or raw logs.
    run = saved if saved else (load(trace) if trace.exists() else {})
    turns = run.get("turns", [])
    counts, truncated, lengths, errors, searches = Counter(), Counter(), defaultdict(list), [], []
    for index, turn in enumerate(turns):
        for call in turn.get("tool_calls", []):
            name = call.get("name", "unknown")
            counts[name] += 1
            value = call.get("result", call.get("result_text", ""))
            if not isinstance(value, str):
                value = ""
            chars = call.get("result_chars", len(value))
            lengths[name].append(chars)
            if call.get("truncated"):
                truncated[name] += 1
            if call.get("error"):
                errors.append({"turn": turn.get("turn", index), "tool": name,
                    "error": str(call["error"])[:160],
                    "public_error_prefix": value[:240] if value.lstrip().startswith("Error:") else None})
            if name == "search":
                try:
                    payload = json.loads(value)
                except (ValueError, TypeError):
                    searches.append({"turn": turn.get("turn", index),
                        "json_result": False, "result_chars": chars,
                        "error_prefix": value[:240] if value.lstrip().startswith("Error:") else None})
                    continue
                if not isinstance(payload, dict):
                    searches.append({"turn": turn.get("turn", index), "json_result": True,
                        "result_chars": chars, "unexpected_json_type": type(payload).__name__})
                    continue
                records = payload.get("queries", [])
                searches.append({"turn": turn.get("turn", index),
                    "json_result": True, "status": payload.get("status"),
                    "result_chars": chars, "subqueries": [{
                        "status": item.get("status"), "matched_documents": item.get("matched_documents"),
                        "returned_items": len(item.get("items", [])),
                        "error": str(item.get("error", ""))[:240] or None}
                        for item in records if isinstance(item, dict)]})
    state = cp.get("tool_state", {})
    wrapper = state.get("search") or {}
    service = state.get("service") or {}
    manager = service.get("manager") or {}
    scoring = manager.get("usage") or {}
    ablation = row.get("ablation_summary", {})
    physical = row.get("reranker_summary", {})
    return {"query_id": qid, "status": "terminal" if row else "checkpointed_active_or_waiting",
        "termination": row.get("terminated_by"), "saved_turns": len(turns),
        "checkpoint_age_seconds": round(time.time() - checkpoint.stat().st_mtime, 1) if checkpoint.exists() else None,
        "checkpoint_elapsed_seconds": cp.get("elapsed_seconds"),
        "tool_calls": dict(counts), "truncated_tool_calls": dict(truncated), "ordinary_tool_errors": errors,
        "automatic_rerank_calls": ablation.get("automatic_rerank_calls", wrapper.get("auto_rerank_calls")),
        "automatic_rerank_errors": ablation.get("automatic_rerank_errors", wrapper.get("auto_rerank_errors")),
        "automatic_empty_scopes": ablation.get("automatic_empty_scopes", wrapper.get("empty_scopes")),
        "scoring_calls": physical.get("scoring_calls", scoring.get("scoring_calls")),
        "scored_passages": physical.get("scored_passages", scoring.get("scored_passages")),
        "reranker_errors": physical.get("errors", scoring.get("errors")),
        "returned_chars": {name: {"calls": len(values), "sum": sum(values),
            "min": min(values), "max": max(values)} for name, values in lengths.items()},
        "search_return_metadata": searches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/data/rise-strict-auto20-20261023"))
    parser.add_argument("--qids", type=int, nargs="*")
    args = parser.parse_args()
    reports = args.root / "reports/strict_auto20_v1"
    manifest = load(reports / "run_manifest.json")
    qids = args.qids if args.qids is not None else manifest["selected_query_ids"]
    if any(qid not in manifest["selected_query_ids"] for qid in qids):
        raise ValueError("Only the frozen selected20 may be audited")
    output = args.root / "runs/strict_auto20_v1/auto_rerank"
    result = {"read_only": True, "generated_unix": time.time(),
        "manifest_sha256": manifest["manifest_sha256"],
        "infrastructure_pause_present": (reports / "infrastructure_pause.json").exists(),
        "private_reasoning_inspected_or_emitted": False,
        "records": [inspect_query(output, qid) for qid in qids]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
