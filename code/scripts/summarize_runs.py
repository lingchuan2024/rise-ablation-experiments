"""Uniform aggregator + table renderer for per-query result dirs.

Reads `outputs/<run_dir>/_per_query/*.json` (schema in
`src/rise/trajectory.py:build_per_query_row`) and produces a single
markdown table row per run. All cross-baseline columns come from
`PER_QUERY_REQUIRED_KEYS` so the schema is the source of truth. Things
that only exist for one architecture (BCP `gold_recall`, RISE
`n_bm25_searches`, DCI `bash_kinds`) are pulled from `runner_specific`
when present and emitted as N/A otherwise.

Usage:

    uv run python scripts/summarize_runs.py \
        outputs/rise_gpt-5.4-mini_t100_k1000_bm25_full \
        outputs/rise_gpt-5.4-mini_t100_k1000_bm25_full_structured \
        outputs/bcp_retrieval_agent_mini_dev_gpt-5.4-mini_k5_gd_mi100_bm25_full
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def _mean(xs: Iterable[float], default: float = 0.0) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else default


def _load_per_query(run_dir: Path) -> list[dict[str, Any]]:
    pq_dir = run_dir / "_per_query"
    if not pq_dir.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(pq_dir.glob("qid_*.json"))]


def _load_judge_summary(run_dir: Path) -> dict[str, Any]:
    f = run_dir / "_judge_summary.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def _bm25_recall_from_traces(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    """Compute (mean_gold_recall, mean_evidence_recall, n) over BM25 candidates
    by reading `_traces/qid_<id>/single.json` (RISE-style trace). Only counts
    bm25_search calls. Returns (0, 0, 0) if no traces are found.
    """
    gold_R, ev_R = [], []
    for row in rows:
        qid = row["query_id"]
        trace = run_dir / "_traces" / f"qid_{qid}" / "single.json"
        if not trace.is_file():
            continue
        try:
            t = json.loads(trace.read_text())
        except Exception:
            continue
        bm25_dids: set[str] = set()
        for turn in t.get("turns", []):
            for tc in turn.get("tool_calls", []):
                if tc.get("name") in ("bm25_search", "search"):
                    for did in tc.get("result_docids") or []:
                        bm25_dids.add(str(did))
        if not bm25_dids:
            continue
        gold = set(row["gold_doc_ids"])
        evi = set(row["evidence_doc_ids"])
        if gold:
            gold_R.append(len(gold & bm25_dids) / len(gold))
        if evi:
            ev_R.append(len(evi & bm25_dids) / len(evi))
    return _mean(gold_R), _mean(ev_R), len(gold_R)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    rows = _load_per_query(run_dir)
    if not rows:
        return {"run_dir": run_dir.name, "n": 0}
    js = _load_judge_summary(run_dir)

    # --- common fields ---
    runner = rows[0].get("runner", "?")
    model = rows[0].get("model", "?")
    n = len(rows)

    # Pending = placeholder judge row (judge.model empty)
    n_pending = sum(1 for r in rows if not (r.get("judge") or {}).get("model"))
    n_judged = n - n_pending
    n_correct = sum(1 for r in rows if r.get("is_correct"))
    acc = (n_correct / n_judged) if n_judged else None

    mean_turns = _mean(r["n_turns"] for r in rows)
    mean_elapsed = _mean(r["elapsed_seconds"] for r in rows)

    cov_mean = _mean(r["coverage_mean"] for r in rows)
    ev_cov = _mean(r["evidence_coverage_mean"] for r in rows)
    cov_any = _mean(r["coverage_any"] for r in rows)
    cov_all = _mean(r["coverage_all"] for r in rows)

    # Tool usage map — sum across all rows
    tool_totals: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("tool_usage") or {}).items():
            tool_totals[k] = tool_totals.get(k, 0) + int(v)
    tool_means = {k: v / n for k, v in tool_totals.items()}

    # Termination counts
    term_counts: dict[str, int] = {}
    for r in rows:
        k = r.get("terminated_by", "?") or "?"
        term_counts[k] = term_counts.get(k, 0) + 1
    n_no_final = sum(1 for r in rows if not (r.get("final_text") or "").strip())
    n_max_calls = sum(1 for r in rows if r.get("max_calls_hit"))
    n_wall_clock = sum(1 for r in rows if r.get("wall_clock_hit"))

    # Tokens (agent_usage)
    tok_in = tok_c = tok_out = tok_r = 0
    for r in rows:
        u = r.get("agent_usage") or {}
        tok_in += u.get("input_tokens", 0)
        tok_c += u.get("cached_input_tokens", 0)
        tok_out += u.get("output_tokens", 0)
        tok_r += u.get("reasoning_tokens", 0)
    cache_ratio = (tok_c / (tok_c + tok_in)) if (tok_c + tok_in) else 0.0

    agent_cost = sum(r.get("agent_cost_usd", 0.0) for r in rows)
    judge_cost = sum(r.get("judge_cost_usd", 0.0) for r in rows)
    total_cost = agent_cost + judge_cost

    # --- runner_specific extras (BCP gold_recall etc.) ---
    rs_keys = set()
    for r in rows:
        rs_keys.update((r.get("runner_specific") or {}).keys())
    bm25_gold_R = bm25_ev_R = None
    if "gold_recall" in rs_keys:
        bm25_gold_R = _mean(
            (r.get("runner_specific") or {}).get("gold_recall", 0.0) for r in rows
        )
    if "evidence_recall" in rs_keys:
        bm25_ev_R = _mean(
            (r.get("runner_specific") or {}).get("evidence_recall", 0.0) for r in rows
        )
    # If runner doesn't track recall in runner_specific, derive from traces
    if bm25_gold_R is None:
        g, e, ntr = _bm25_recall_from_traces(run_dir, rows)
        if ntr > 0:
            bm25_gold_R, bm25_ev_R = g, e

    # RISE-specific: n_bm25_searches, n_bm25_queries
    n_bm25_searches = _mean(
        (r.get("runner_specific") or {}).get("n_bm25_searches", 0) for r in rows
    ) if "n_bm25_searches" in rs_keys else None
    n_bm25_queries = _mean(
        (r.get("runner_specific") or {}).get("n_bm25_queries", 0) for r in rows
    ) if "n_bm25_queries" in rs_keys else None

    return {
        "run_dir": run_dir.name,
        "runner": runner,
        "model": model,
        "n": n,
        "n_correct": n_correct,
        "n_pending": n_pending,
        "accuracy": acc,
        "mean_turns": mean_turns,
        "mean_elapsed_seconds": mean_elapsed,
        "tool_means": tool_means,
        "term_counts": term_counts,
        "n_no_final": n_no_final,
        "n_max_calls": n_max_calls,
        "n_wall_clock": n_wall_clock,
        "tokens": {
            "input": tok_in,
            "cached": tok_c,
            "output": tok_out,
            "reasoning": tok_r,
            "cache_ratio": cache_ratio,
        },
        "agent_cost_usd": agent_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": total_cost,
        "bm25_gold_recall": bm25_gold_R,
        "bm25_evidence_recall": bm25_ev_R,
        "coverage_mean": cov_mean,
        "evidence_coverage_mean": ev_cov,
        "coverage_any": cov_any,
        "coverage_all": cov_all,
        "n_bm25_searches": n_bm25_searches,
        "n_bm25_queries": n_bm25_queries,
        "judge_mode": js.get("judge_mode"),
        "judge_model": js.get("judge_model"),
    }


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x*100:.1f}%"


def _fmt_num(x: float | None, fmt: str = ".1f") -> str:
    return format(x, fmt) if x is not None else "—"


def _fmt_money(x: float) -> str:
    return f"${x:.2f}"


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    """Render a uniform markdown table across all runs."""
    headers = [
        "run_dir", "runner", "model", "n", "Acc",
        "Turns", "Search", "Bash", "Read", "get_doc",
        "BM25 gold_R", "BM25 ev_R", "cov_mean", "ev_cov",
        "Tokens (in / cached / out / reason)", "Cache%",
        "Agent $", "Judge $", "Total $",
        "no_final", "max_calls", "wall_clock",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for s in summaries:
        if s.get("n", 0) == 0:
            lines.append(f"| {s['run_dir']} | — | — | 0 | — |" + " |" * (len(headers) - 5))
            continue
        tm = s["tool_means"]
        search = tm.get("search", tm.get("bm25_search"))
        bash = tm.get("bash")
        read = tm.get("read")
        get_doc = tm.get("get_document")
        tok = s["tokens"]
        row = [
            s["run_dir"],
            s["runner"],
            s["model"],
            str(s["n"]) + (f" (judge_pending {s['n_pending']})" if s["n_pending"] else ""),
            _fmt_pct(s["accuracy"]) if s["accuracy"] is not None else "judge pending",
            _fmt_num(s["mean_turns"]),
            _fmt_num(search),
            _fmt_num(bash),
            _fmt_num(read),
            _fmt_num(get_doc),
            _fmt_pct(s["bm25_gold_recall"]),
            _fmt_pct(s["bm25_evidence_recall"]),
            _fmt_pct(s["coverage_mean"]),
            _fmt_pct(s["evidence_coverage_mean"]),
            f"{tok['input']/1e6:.2f}M / {tok['cached']/1e6:.2f}M / {tok['output']/1e6:.3f}M / {tok['reasoning']/1e6:.3f}M",
            _fmt_pct(tok["cache_ratio"]),
            _fmt_money(s["agent_cost_usd"]),
            _fmt_money(s["judge_cost_usd"]),
            _fmt_money(s["total_cost_usd"]),
            str(s["n_no_final"]),
            str(s["n_max_calls"]),
            str(s["n_wall_clock"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path, help="One or more outputs/<run_dir>")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON list instead of markdown table")
    args = ap.parse_args()

    summaries = [summarize_run(d) for d in args.run_dirs]
    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
    else:
        print(render_markdown(summaries))


if __name__ == "__main__":
    main()
