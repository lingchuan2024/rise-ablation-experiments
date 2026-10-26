"""Export only public finals and actual tool calls from downloaded auto20 traces.

Administrative, read-only input processing. No experiment imports, inference,
historical labels or checkpoint/message-history reads are needed. The output
is an explicit allowlist and must use a fresh path with mode0600.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "auto20-public-tool-trace-v1"
PRIVATE_KEYS = {"reasoning_content", "thinking_content", "thinking", "reasoning", "messages"}
ITEM_KEYS = ("file_path", "text", "char_start", "char_end", "token_count", "overlap_tokens",
             "bm25_rank", "bm25_score", "reranker_rank", "reranker_score", "read_args",
             "read_may_truncate", "bash_args", "context_char_start", "context_char_end")
USAGE_KEYS = ("scoring_calls", "scored_passages", "forward_calls", "completed_forward_calls",
              "input_tokens", "errors", "ranking_cache_hits", "scoring_seconds")


def safe_object(pairs):
    # Reasoning strings are discarded during decoding, never inspected or
    # emitted. We also never copy turns wholesale or inspect message history.
    return {key: value for key, value in pairs if key not in PRIVATE_KEYS}


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw, object_pairs_hook=safe_object), hashlib.sha256(raw).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_digest(value, field):
    payload = {key: item for key, item in value.items() if key != field}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != value.get(field):
        raise ValueError("Invalid " + field)


def rank_query(record):
    items = record.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Search items are not an array")
    public, changed, from_outside = [], 0, 0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Search item is not an object")
        entry = {key: item[key] for key in ITEM_KEYS if key in item}
        before, after = item.get("bm25_rank"), item.get("reranker_rank")
        valid = type(before) is int and type(after) is int and before > 0 and after > 0
        entry["rank_delta_bm25_minus_reranker"] = before - after if valid else None
        changed += int(valid and before != after)
        from_outside += int(valid and before > 10)
        public.append(entry)
    scores = [item.get("reranker_score") for item in items]
    ranks = [item.get("reranker_rank") for item in items]
    comparable = bool(items) and all(number(value) for value in scores)
    return {"query": record.get("query"), "status": record.get("status"),
        "matched_documents": record.get("matched_documents"), "error": record.get("error"),
        "returned_items": len(items), "items": public,
        "returned_slice_statistics": {
            "items_with_changed_rank": changed,
            "items_with_bm25_rank_beyond10": from_outside,
            "items_with_bm25_rank_at_most10": sum(type(item.get("bm25_rank")) is int
                and 1 <= item["bm25_rank"] <= 10 for item in items),
            "reranker_ranks_are_first_contiguous_slice": ranks == list(range(1, len(items) + 1)),
            "reranker_scores_nonincreasing": all(left >= right for left, right in zip(scores, scores[1:]))
                if comparable else None,
            "scope": "Only model-visible returned passages; not the full M1000 ranking or a relevance judgement"}}


def public_query(root, question):
    qid = question["query_id"]
    output = root / "runs/strict_auto20_v1/auto_rerank"
    trace, trace_sha = load(output / "_traces" / f"qid_{qid}" / "single.json")
    terminal, terminal_sha = load(output / "_per_query" / f"qid_{qid}.json")
    checked_digest(terminal, "row_sha256")
    if terminal.get("query_id") != qid or terminal.get("question_sha256") != hashlib.sha256(question["query"].encode()).hexdigest():
        raise ValueError("Question and terminal identity differ")
    if trace.get("final_text") is not None and not isinstance(trace["final_text"], str):
        raise ValueError("Invalid trace final type")
    final = trace.get("final_text") or ""
    if not isinstance(final, str) or terminal.get("final_text") != final:
        raise ValueError("Trace and public terminal finals differ")
    if terminal.get("final_sha256") != hashlib.sha256(final.encode()).hexdigest():
        raise ValueError("Public final SHA differs")
    if terminal.get("execution_artifacts", {}).get("trace") != trace_sha:
        raise ValueError("Terminal does not bind these exact trace bytes")
    calls, counts, chars = [], Counter(), Counter()
    for index, turn in enumerate(trace.get("turns", [])):
        for position, call in enumerate(turn.get("tool_calls", [])):
            name = call.get("name")
            result = call.get("result_text")
            if not isinstance(name, str) or not isinstance(result, str):
                raise ValueError("Actual tool name/result_text is missing")
            if call.get("result_chars") != len(result):
                raise ValueError("Actual tool return character count differs")
            item = {"turn": turn.get("turn", index), "call_in_turn": position, "name": name,
                "args": call.get("args"), "result_text": result, "result_chars": len(result),
                "result_truncated": call.get("result_truncated"),
                "duration_seconds": call.get("duration_seconds"), "error": call.get("error")}
            if name == "search":
                try:
                    payload = json.loads(result)
                except (ValueError, TypeError):
                    item["automatic_search_metadata"] = {"json_result": False}
                else:
                    if isinstance(payload, dict):
                        item["automatic_search_metadata"] = {"json_result": True,
                            "status": payload.get("status"),
                            "queries": [rank_query(query) for query in payload.get("queries", [])]}
                    else:
                        item["automatic_search_metadata"] = {"json_result": True,
                            "unexpected_type": type(payload).__name__}
            calls.append(item)
            counts[name] += 1
            chars[name] += len(result)
    if dict(counts) != terminal.get("tool_call_breakdown"):
        raise ValueError("Trace and terminal tool-call counters differ")
    physical = terminal.get("reranker_summary", {})
    usage = {key: physical.get(key) for key in USAGE_KEYS}
    positive = lambda key: number(usage[key]) and usage[key] > 0
    return {"query_id": qid, "question": question["query"], "public_final": final,
        "terminated_by": terminal.get("terminated_by"), "source_trace_sha256": trace_sha,
        "source_terminal_sha256": terminal_sha, "tool_calls": calls,
        "counts": dict(counts), "returned_chars_by_tool": dict(chars),
        "recorded_reranker_usage": usage,
        "recorded_completed_neural_compute": None if any(usage[key] is None for key in
            ("scoring_calls", "scored_passages", "completed_forward_calls")) else
            all(positive(key) for key in ("scoring_calls", "scored_passages", "completed_forward_calls")),
        "answer_evidence_used": "undetermined; inspect public final and tool results with explicit evidence linkage",
        "limitations": "Observed scoring and changed returned ranks do not establish relevance, answer support, or a causal accuracy gain"}


def export(root, cohort, qids=None):
    root = Path(root)
    manifest, manifest_file_sha = load(root / "reports/strict_auto20_v1/run_manifest.json")
    checked_digest(manifest, "manifest_sha256")
    selected = manifest["selected_query_ids"]
    qids = selected if qids is None else qids
    if not qids or len(set(qids)) != len(qids) or any(qid not in selected for qid in qids):
        raise ValueError("Review only unique members of the fixed auto20 selection")
    if hashlib.sha256(Path(cohort).read_bytes()).hexdigest() != manifest.get("cohort_sha256"):
        raise ValueError("Frozen question file differs")
    questions = {row["query_id"]: {"query_id": row["query_id"], "query": row["query"]}
        for row in (json.loads(line) for line in Path(cohort).read_text().splitlines() if line.strip())}
    return {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_file_sha256": manifest_file_sha, "selected_query_ids": qids,
        "input_processing": "Only downloaded traces, terminals, manifest and question text; no checkpoints or historical labels",
        "excluded": ["system_prompt", "user_prompt", "turn.content", "reasoning_content",
                     "thinking_content", "thinking", "messages", "raw_tool_calls", "gold_answer", "historical_judgements"],
        "rank_delta_sign": "Positive=bm25_rank-reranker_rank: promoted; negative: demoted",
        "evidence_levels": [
            "Tool status=ok/items plus recorded completed_forward_calls/scored_passages support successful execution; invocation alone is insufficient",
            "Returned item rank differences demonstrate an order change within the model-visible slice; they do not measure relevance or complete-ranking quality",
            "Answer evidence use requires a reviewable passage-to-public-final connection; a later read of the same source only shows revisiting, not semantic use"],
        "records": [public_query(root, questions[qid]) for qid in qids]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded artifact root")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--qids", nargs="+", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.root, args.cohort, args.qids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "questions": len(payload["records"]),
        "private_reasoning_or_historical_labels_emitted": False}))


if __name__ == "__main__":
    main()
