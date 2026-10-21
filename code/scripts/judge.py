#!/usr/bin/env python3
"""Decoupled judge — runs BCP Appendix F judging over a run directory's
per_query rows, atomically updating the placeholder judge fields with
real values.

Two modes:
  --mode batch   (default)  Submits one OpenAI batch per run-dir,
                            polls until completed, downloads results, and
                            writes them back. 50% cheaper than online,
                            ~minutes-to-hours latency. Resumable via
                            `--resume-batch <id>` or via the per-run-dir
                            `_judge_batch_state.json` checkpoint.
  --mode online              Synchronous chat.completions calls (one per
                            row), full concurrency = `--concurrency`.
                            Use when you need accuracy NOW or the judge
                            provider doesn't support batch (mimo, deepseek).

Empty-final-text rows are NOT sent to the batch — they get a deterministic
placeholder judge inline (any-judge would judge them "no" anyway).

Workflow:
  1. Run any of the 4 agent runners — they write per_query.json files
     with placeholder judge fields (`judge.reasoning == ""`).
  2. Run this script over the run directory:
       uv run python scripts/judge.py \\
           --run-dir outputs/<run_dir> \\
           --judge-model gpt-5.1
     For each row whose judge is "pending", fills the fields in-place,
     updates `is_correct` and `total_cost_usd`, and writes
     `_judge_summary.json` next to `_summary.json` with the accuracy +
     judge cost rollup.

`--force` re-judges rows that already have non-empty `judge.reasoning`
(useful when swapping judge models for ablation).

Multiple `--run-dir` flags are supported; each gets its own batch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from rise.api_retry import is_reasoner_model
from rise.bcp_retrieval_agent import BCP_F_JUDGE_PROMPT, bcp_judge
from rise.decompose import make_client
from rise.dci_artifacts import PRICE_TABLE, estimate_cost
from rise.nvidia_key_pool import redact_nvidia_credentials
from rise.trajectory import atomic_write_json, is_judge_pending


# ===== Shared helpers ========================================================


def _empty_text_judge(judge_model: str) -> tuple[dict, dict, float]:
    """Deterministic placeholder for rows where the agent produced no final
    text — no need to spend a judge call to say 'no'."""
    judge_out = {
        "extracted_final_answer": "None",
        "reasoning": "Agent produced no final text.",
        "correct": "no",
        "confidence": 100,
        "model": judge_model,
    }
    judge_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    return judge_out, judge_usage, 0.0


_EXACT_ANSWER_RE = re.compile(r"Exact Answer:\s*([^\n]+)", re.IGNORECASE)


def _extract_exact_answer(final_text: str) -> str:
    """Pull the value following 'Exact Answer:' (case-insensitive). Returns
    '' if not present."""
    m = _EXACT_ANSWER_RE.search(final_text or "")
    return m.group(1).strip() if m else ""


def _normalize_for_exact_match(s: str) -> str:
    """Conservative normalization for short-circuit equality check:
    - strip leading/trailing whitespace
    - strip surrounding markdown bold (`**…**`) or italic (`*…*`)
    - strip trailing punctuation (`.`, `,`, `;`, `:`)
    - casefold (case-insensitive)
    Does NOT strip parentheticals, articles, or interior characters — those
    cases get escalated to the LLM judge."""
    s = (s or "").strip()
    # Alternate markdown-strip + trailing-punct-strip until stable, so
    # `**Foo**.` and `**Foo.**` both reduce to `foo`.
    prev: str | None = None
    while prev != s:
        prev = s
        if len(s) >= 4 and s.startswith("**") and s.endswith("**"):
            s = s[2:-2].strip()
        elif len(s) >= 2 and s.startswith("*") and s.endswith("*"):
            s = s[1:-1].strip()
        while s and s[-1] in ".,;:":
            s = s[:-1].rstrip()
    return s.casefold()


def _try_short_circuit_judge(row: dict, judge_model: str) -> tuple[dict, dict, float] | None:
    """If the agent's `Exact Answer:` line equals the gold answer (after
    conservative normalization), return a deterministic 'yes' judge tuple
    so we can skip the LLM call. Returns None if no match — let the LLM
    handle it."""
    final_text = row.get("final_text") or ""
    gold = row.get("gold_answer") or ""
    agent_ans = _extract_exact_answer(final_text)
    if not agent_ans or not gold:
        return None
    if _normalize_for_exact_match(agent_ans) != _normalize_for_exact_match(gold):
        return None
    judge_out = {
        "extracted_final_answer": agent_ans,
        "reasoning": "Short-circuit: agent Exact Answer matches gold byte-exact (after whitespace/markdown/case normalization). No LLM judge call.",
        "correct": "yes",
        "confidence": 100,
        "model": judge_model,
    }
    judge_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    return judge_out, judge_usage, 0.0


def _apply_judge_to_row(
    row: dict,
    judge_out: dict,
    judge_usage: dict,
    judge_cost: float,
) -> dict:
    new_row = dict(row)
    new_row["judge"] = judge_out
    new_row["judge_usage"] = judge_usage
    new_row["judge_cost_usd"] = judge_cost
    new_row["total_cost_usd"] = (row.get("agent_cost_usd") or 0.0) + judge_cost
    new_row["is_correct"] = str(judge_out.get("correct", "no")).strip().lower() == "yes"
    return new_row


def _load_rows(run_dir: Path) -> list[tuple[Path, dict]]:
    per_query_dir = run_dir / "_per_query"
    out: list[tuple[Path, dict]] = []
    for p in sorted(per_query_dir.glob("qid_*.json")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except Exception as e:
            print(f"  skip {p.name}: parse error: {e}", file=sys.stderr)
    return out


def _select_pending(rows: list[tuple[Path, dict]], *, force: bool) -> list[tuple[Path, dict]]:
    return [(p, r) for (p, r) in rows if force or is_judge_pending(r)]


# ===== Online mode ===========================================================


def _judge_one_online(row: dict, *, judge_client, judge_model: str) -> dict:
    final_text = row.get("final_text") or ""
    if not final_text.strip():
        judge_out, judge_usage, judge_cost = _empty_text_judge(judge_model)
        return _apply_judge_to_row(row, judge_out, judge_usage, judge_cost)
    # Short-circuit: if agent's `Exact Answer:` matches gold (normalized),
    # skip the LLM call.
    sc = _try_short_circuit_judge(row, judge_model)
    if sc is not None:
        judge_out, judge_usage, judge_cost = sc
        return _apply_judge_to_row(row, judge_out, judge_usage, judge_cost)
    judge_out = bcp_judge(
        question=row.get("question") or "",
        correct_answer=row.get("gold_answer") or "",
        response=final_text,
        client=judge_client,
        model=judge_model,
    )
    judge_usage = judge_out.pop("_usage", {})
    judge_out["model"] = judge_model
    judge_cost = estimate_cost(judge_usage, judge_model)
    return _apply_judge_to_row(row, judge_out, judge_usage, judge_cost)


def _process_online(
    run_dir: Path,
    *,
    judge_client,
    judge_model: str,
    force: bool,
    concurrency: int,
) -> int:
    rows = _load_rows(run_dir)
    to_judge = _select_pending(rows, force=force)
    if not to_judge:
        print(f"  ({run_dir.name}: all {len(rows)} rows already judged; use --force to re-judge)")
        _write_judge_summary(run_dir, rows, judge_model, mode="online")
        return 0
    print(f"  {run_dir.name}: online-judging {len(to_judge)}/{len(rows)} rows  c={concurrency}")
    print_lock = Lock()
    t0 = time.time()

    def _one(item: tuple[Path, dict]) -> None:
        p, row = item
        try:
            new_row = _judge_one_online(row, judge_client=judge_client, judge_model=judge_model)
        except Exception as e:
            safe_error = redact_nvidia_credentials(str(e))
            with print_lock:
                print(
                    f"  qid={row.get('query_id')} JUDGE_FAILED: "
                    f"{type(e).__name__}: {safe_error[:200]}",
                    file=sys.stderr,
                )
            return
        atomic_write_json(p, new_row)
        with print_lock:
            mark = "✓" if new_row["is_correct"] else "✗"
            ja = (new_row.get("judge") or {}).get("extracted_final_answer", "")
            print(
                f"  qid={new_row.get('query_id'):>5}  {mark:>4}  "
                f"${new_row['judge_cost_usd']:.4f}  "
                f"{str(new_row.get('gold_answer'))[:30]} -> {str(ja)[:35]}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        list(ex.map(_one, to_judge))

    print(f"  elapsed {time.time() - t0:.1f}s")
    rows_after = _load_rows(run_dir)
    _write_judge_summary(run_dir, rows_after, judge_model, mode="online")
    return 0


# ===== Batch mode ============================================================
#
# Per OpenAI batch API: JSONL of `{custom_id, method, url, body}` lines,
# uploaded via `files.create(purpose='batch')`, then `batches.create(...)`,
# polled until `status == 'completed'`. Output file is JSONL of
# `{custom_id, response: {status_code, body}, error}`.
#
# We apply the 50% batch discount to `judge_cost_usd` at write-time so the
# `_judge_summary.json` totals are correct without extra plumbing.


BATCH_PRICE_MULTIPLIER = 0.5
BATCH_STATE_FILENAME = "_judge_batch_state.json"
BATCH_INPUT_FILENAME = "_judge_batch_input.jsonl"


def _build_batch_line(qid: Any, question: str, gold: str, final_text: str, judge_model: str) -> dict:
    user_prompt = BCP_F_JUDGE_PROMPT.format(
        question=question,
        response=final_text or "[empty]",
        correct_answer=gold,
    )
    is_reasoner = is_reasoner_model(judge_model)
    body: dict[str, Any] = {
        "model": judge_model,
        "messages": [{"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"},
    }
    # batch (gpt-5/o-series) rejects `max_tokens` ("Unsupported
    # parameter: 'max_tokens' is not supported with this model. Use
    # 'max_completion_tokens' instead.") — online chat.completions is
    # more lenient, but the batch endpoint
    # enforces the new naming. Use `max_completion_tokens` for reasoner
    # models; `max_tokens` for legacy non-reasoner models.
    if is_reasoner:
        body["max_completion_tokens"] = 4096
    else:
        body["max_tokens"] = 512
        body["temperature"] = 0.0
    return {
        "custom_id": f"qid_{qid}",
        "method": "POST",
        "url": "/chat/completions",
        "body": body,
    }


def _parse_batch_response_line(result_line: dict, judge_model: str) -> tuple[dict, dict, float] | None:
    """Returns (judge_out, judge_usage, judge_cost) or None on error.

    Note: cost already includes the 50% batch discount.
    """
    if result_line.get("error"):
        print(f"  batch error for {result_line.get('custom_id')}: {result_line['error']}",
              file=sys.stderr)
        return None
    response = result_line.get("response") or {}
    if response.get("status_code") and response["status_code"] != 200:
        print(
            f"  non-200 for {result_line.get('custom_id')}: status={response.get('status_code')}",
            file=sys.stderr,
        )
        return None
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    raw_content = choices[0].get("message", {}).get("content") or "{}"
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    comp_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(comp_details.get("reasoning_tokens", 0) or 0)
    judge_usage = {
        "input_tokens": max(0, prompt_tokens - cached),
        "cached_input_tokens": cached,
        "output_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    judge_cost = estimate_cost(judge_usage, judge_model) * BATCH_PRICE_MULTIPLIER
    judge_out = {
        "extracted_final_answer": data.get("extracted_final_answer", "None"),
        "reasoning": data.get("reasoning", ""),
        "correct": str(data.get("correct", "no")).strip().lower(),
        "confidence": data.get("confidence", 100),
        "model": judge_model,
    }
    return judge_out, judge_usage, judge_cost


def _submit_batch(
    judge_client,
    run_dir: Path,
    pending: list[tuple[Path, dict]],
    judge_model: str,
) -> dict:
    """Build JSONL, upload, create batch. Returns the state dict."""
    # Empty-text rows: handle inline so we don't waste a batch slot.
    submit_rows: list[tuple[Path, dict]] = []
    empty_text_paths: list[Path] = []
    short_circuit_paths: list[Path] = []
    for p, r in pending:
        if not (r.get("final_text") or "").strip():
            empty_text_paths.append(p)
            continue
        sc = _try_short_circuit_judge(r, judge_model)
        if sc is not None:
            short_circuit_paths.append((p, sc))
            continue
        submit_rows.append((p, r))

    for p in empty_text_paths:
        row = json.loads(p.read_text(encoding="utf-8"))
        judge_out, judge_usage, judge_cost = _empty_text_judge(judge_model)
        atomic_write_json(p, _apply_judge_to_row(row, judge_out, judge_usage, judge_cost))

    for p, sc in short_circuit_paths:
        row = json.loads(p.read_text(encoding="utf-8"))
        judge_out, judge_usage, judge_cost = sc
        atomic_write_json(p, _apply_judge_to_row(row, judge_out, judge_usage, judge_cost))

    if empty_text_paths:
        print(f"  handled {len(empty_text_paths)} empty-text row(s) inline (skipped batch)")
    if short_circuit_paths:
        print(f"  short-circuit {len(short_circuit_paths)} row(s) (exact-match, skipped batch)")

    if not submit_rows:
        # Nothing to batch (all rows handled inline by empty-text + short-circuit)
        return {
            "batch_id": None,
            "input_file_id": None,
            "judge_model": judge_model,
            "submitted_at_epoch": int(time.time()),
            "status": "completed",
            "n_submitted": 0,
            "n_inline_empty": len(empty_text_paths),
            "n_inline_short_circuit": len(short_circuit_paths),
        }

    # Build JSONL
    jsonl_path = run_dir / BATCH_INPUT_FILENAME
    with jsonl_path.open("w", encoding="utf-8") as f:
        for p, r in submit_rows:
            line = _build_batch_line(
                qid=r["query_id"],
                question=r.get("question") or "",
                gold=r.get("gold_answer") or "",
                final_text=r.get("final_text") or "",
                judge_model=judge_model,
            )
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Upload
    with jsonl_path.open("rb") as f:
        upload = judge_client.files.create(file=f, purpose="batch")

    # Create batch
    batch = judge_client.batches.create(
        input_file_id=upload.id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": upload.id,
        "judge_model": judge_model,
        "submitted_at_epoch": int(time.time()),
        "n_submitted": len(submit_rows),
        "n_inline_empty": len(empty_text_paths),
        "status": "submitted",
        "input_jsonl_relpath": str(jsonl_path.relative_to(run_dir)),
    }
    (run_dir / BATCH_STATE_FILENAME).write_text(json.dumps(state, indent=2))
    print(f"  submitted batch {batch.id} ({len(submit_rows)} rows, file={upload.id})")
    return state


def _poll_batch(judge_client, batch_id: str, *, poll_interval: int) -> Any:
    """Poll until terminal status. Returns the final Batch object."""
    last_print = 0.0
    while True:
        batch = judge_client.batches.retrieve(batch_id)
        rc = getattr(batch, "request_counts", None)
        rc_str = ""
        if rc is not None:
            done = getattr(rc, "completed", 0)
            total = getattr(rc, "total", 0)
            failed = getattr(rc, "failed", 0)
            rc_str = f"  ({done}/{total} done"
            if failed:
                rc_str += f", {failed} failed"
            rc_str += ")"
        now = time.time()
        if now - last_print > 1:  # at most one print per poll
            print(f"  batch {batch_id}: status={batch.status}{rc_str}", flush=True)
            last_print = now
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch
        time.sleep(poll_interval)


def _collect_batch(judge_client, run_dir: Path, state: dict, judge_model: str) -> int:
    """Download batch output, apply to per_query.json files. Returns count
    of rows updated."""
    batch_id = state["batch_id"]
    batch = judge_client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"  ERROR: batch {batch_id} status={batch.status}; cannot collect", file=sys.stderr)
        return 0
    if not batch.output_file_id:
        print(f"  ERROR: batch {batch_id} has no output_file_id", file=sys.stderr)
        return 0
    raw = judge_client.files.content(batch.output_file_id).text
    n_updated = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        parsed = _parse_batch_response_line(result, judge_model)
        if parsed is None:
            continue
        judge_out, judge_usage, judge_cost = parsed
        custom_id = result["custom_id"]
        qid = custom_id.removeprefix("qid_")
        per_query_path = run_dir / "_per_query" / f"qid_{qid}.json"
        if not per_query_path.exists():
            print(f"  WARN: {custom_id}: per_query.json not found", file=sys.stderr)
            continue
        row = json.loads(per_query_path.read_text(encoding="utf-8"))
        new_row = _apply_judge_to_row(row, judge_out, judge_usage, judge_cost)
        atomic_write_json(per_query_path, new_row)
        mark = "✓" if new_row["is_correct"] else "✗"
        print(
            f"  qid={qid:>5}  {mark:>4}  "
            f"${new_row['judge_cost_usd']:.4f}  "
            f"{str(new_row.get('gold_answer'))[:30]} -> "
            f"{str(new_row['judge'].get('extracted_final_answer'))[:35]}",
            flush=True,
        )
        n_updated += 1
    state["status"] = "completed"
    (run_dir / BATCH_STATE_FILENAME).write_text(json.dumps(state, indent=2))
    return n_updated


def _process_batch(
    run_dir: Path,
    *,
    judge_client,
    judge_model: str,
    force: bool,
    poll_interval: int,
    no_wait: bool,
    resume_batch_id: str | None,
) -> int:
    state_file = run_dir / BATCH_STATE_FILENAME

    # Decide whether to resume or submit fresh.
    if resume_batch_id is not None:
        if not state_file.exists():
            print(f"  ERROR: --resume-batch needs {state_file} (missing)", file=sys.stderr)
            return 1
        state = json.loads(state_file.read_text())
        if state["batch_id"] != resume_batch_id:
            print(
                f"  ERROR: {state_file} has batch_id={state['batch_id']!r}, "
                f"not {resume_batch_id!r}",
                file=sys.stderr,
            )
            return 1
        print(f"  resuming batch {state['batch_id']} from state file")
    elif state_file.exists():
        existing = json.loads(state_file.read_text())
        if existing.get("status") == "submitted":
            print(f"  resuming in-flight batch {existing['batch_id']} from state file")
            state = existing
        elif existing.get("status") == "completed" and not force:
            print(f"  ({run_dir.name}: previous batch completed; use --force to re-judge)")
            rows = _load_rows(run_dir)
            _write_judge_summary(run_dir, rows, judge_model, mode="batch")
            return 0
        else:
            # `--force` over completed batch: treat as new submission
            state = None
    else:
        state = None

    if state is None:
        rows = _load_rows(run_dir)
        pending = _select_pending(rows, force=force)
        if not pending:
            print(f"  ({run_dir.name}: all {len(rows)} rows already judged; use --force to re-judge)")
            _write_judge_summary(run_dir, rows, judge_model, mode="batch")
            return 0
        print(f"  {run_dir.name}: batch-judging {len(pending)}/{len(rows)} rows")
        state = _submit_batch(judge_client, run_dir, pending, judge_model)
        if state.get("batch_id") is None:
            # Pure empty-text case — nothing actually got batched
            rows_after = _load_rows(run_dir)
            _write_judge_summary(run_dir, rows_after, judge_model, mode="batch")
            return 0

    if no_wait:
        print(
            f"  --no-wait: exiting. Resume later with: "
            f"--resume-batch {state['batch_id']}"
        )
        return 0

    batch_id = state["batch_id"]
    batch = _poll_batch(judge_client, batch_id, poll_interval=poll_interval)
    if batch.status != "completed":
        print(f"  batch {batch_id} ended with status={batch.status}", file=sys.stderr)
        state["status"] = batch.status
        state_file.write_text(json.dumps(state, indent=2))
        return 1
    n_updated = _collect_batch(judge_client, run_dir, state, judge_model)
    print(f"  collected {n_updated} judged row(s)")
    rows_after = _load_rows(run_dir)
    _write_judge_summary(run_dir, rows_after, judge_model, mode="batch")
    return 0


# ===== Summary ===============================================================


def _write_judge_summary(
    run_dir: Path,
    rows: list[tuple[Path, dict]],
    judge_model: str,
    *,
    mode: str,
) -> None:
    if not rows:
        return
    judged = [r for (_, r) in rows if not is_judge_pending(r)]
    pending = [r for (_, r) in rows if is_judge_pending(r)]
    n = len(rows)
    n_judged = len(judged)
    n_correct = sum(1 for r in judged if r.get("is_correct"))
    judge_cost = sum((r.get("judge_cost_usd") or 0.0) for r in judged)
    agent_cost = sum((r.get("agent_cost_usd") or 0.0) for (_, r) in rows)
    total_cost = sum((r.get("total_cost_usd") or 0.0) for r in judged) + sum(
        (r.get("agent_cost_usd") or 0.0) for r in pending
    )
    summary = {
        "judge_model": judge_model,
        "judge_prompt": "BCP Appendix F (verbatim)",
        "judge_mode": mode,  # "batch" | "online"
        "n": n,
        "n_judged": n_judged,
        "n_pending": n - n_judged,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_judged) if n_judged else 0.0,
        "total_agent_cost_usd": agent_cost,
        "total_judge_cost_usd": judge_cost,
        "total_cost_usd": total_cost,
        "mean_cost_per_query_usd": (total_cost / n) if n else 0,
        "judge_price_table_per_1m": PRICE_TABLE.get(judge_model, {}),
        "batch_discount_applied": (mode == "batch"),
    }
    out_path = run_dir / "_judge_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"  -> {out_path.name}: "
        f"accuracy {summary['accuracy']*100:.1f}% "
        f"({n_correct}/{n_judged}), "
        f"agent ${agent_cost:.4f}, judge ${judge_cost:.4f} ({mode}), total ${total_cost:.4f}"
    )


# ===== Main ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Path to a runner's output dir (containing `_per_query/`). Repeat to judge multiple runs.",
    )
    ap.add_argument("--judge-model", required=True)
    ap.add_argument(
        "--mode",
        choices=["batch", "online"],
        default="batch",
        help="batch (default, 50%% off, async polling) or online (synchronous).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-judge rows that already have a real judge.reasoning.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Online mode only. Number of parallel chat.completions calls.",
    )
    ap.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Batch mode: seconds between status polls.",
    )
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="Batch mode: submit + exit. Re-run with --resume-batch <id> later to collect.",
    )
    ap.add_argument(
        "--resume-batch",
        default=None,
        help="Batch mode: skip submission, collect this batch id (state file must exist).",
    )
    args = ap.parse_args()

    judge_client = make_client(args.judge_model)
    print(
        f"judge_model = {args.judge_model}  mode = {args.mode}  "
        f"force = {args.force}  c = {args.concurrency}"
    )
    rc = 0
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        print()
        if args.mode == "online":
            rc = _process_online(
                run_dir,
                judge_client=judge_client,
                judge_model=args.judge_model,
                force=args.force,
                concurrency=args.concurrency,
            ) or rc
        else:
            rc = _process_batch(
                run_dir,
                judge_client=judge_client,
                judge_model=args.judge_model,
                force=args.force,
                poll_interval=args.poll_interval,
                no_wait=args.no_wait,
                resume_batch_id=args.resume_batch,
            ) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
