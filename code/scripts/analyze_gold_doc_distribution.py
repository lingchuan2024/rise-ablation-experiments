#!/usr/bin/env python3
"""Analyze where annotated gold docs rank inside the agent's BM25 search results.

For every query in a run directory this script replays each `search` tool call in
the stored trajectory and locates the query's gold_doc_ids within:

  1. the per-query top-10 preview that the model actually sees
     (parsed from `result_text`, ranks + BM25 scores), and
  2. the list of newly added working-directory docs stored in
     `result_paths` (top-N new docs by score; invisible to the model but
     accessible afterwards via bash/read).

Outputs (default: <run_dir>/_analysis/gold_dist/):
  gold_dist_per_call.csv   one row per (qid, search call)
  gold_dist_per_query.csv  one row per qid with first-appearance / best-rank summary
  gold_dist_heatmap.png    rows = qids, cols = search call, color = where the gold doc sits
  gold_dist_first_rank.png stacked histogram of first-appearance category by correctness
  gold_dist_score_ratio.png gold-score/top-score scatter + first-appearance call histogram

Example:
  python3 scripts/analyze_gold_doc_distribution.py \
      --run-dir downloads/qwen_full_results_20260926/runs/rise_qwen38_flash_100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

RANK_LINE_RE = re.compile(
    r"^\s*(\d+)\.\s+\[([\d.]+)\]\s+(\S+?)(?:\s+\[already in wd\])?\s*$", re.M
)
QUERY_HEADER_RE = re.compile(r'^Query (\d+): "(.*)"\s+[-—–]\s+(\d+) matched')
RANK_BUCKETS = ["1", "2", "3", "4-5", "6-10", "11-30 (added to wd)", "retrieved, rank>30/unseen", "never retrieved"]


def load_docid_map(map_path: Path) -> dict[str, str]:
    raw = json.loads(map_path.read_text())
    if "relpath_to_docid" in raw:
        raw = raw["relpath_to_docid"]
    return {docid: relpath for relpath, docid in raw.items()}


def parse_search_preview(result_text: str) -> list[dict]:
    """Parse the per-query top-10 preview out of a search call's result_text."""
    entries = []
    query_idx = None
    for line in result_text.splitlines():
        m = QUERY_HEADER_RE.match(line)
        if m:
            query_idx = int(m.group(1))
            continue
        r = RANK_LINE_RE.match(line)
        if r and query_idx is not None:
            entries.append(
                {
                    "query": query_idx,
                    "rank": int(r.group(1)),
                    "score": float(r.group(2)),
                    "path": r.group(3),
                }
            )
    return entries


def iter_search_calls(trace_path: Path):
    data = json.loads(trace_path.read_text())
    call_idx = 0
    for turn in data.get("turns", []):
        for call in turn.get("tool_calls", []):
            if call.get("name") != "search":
                continue
            yield call_idx, call
            call_idx += 1


def analyze_run(run_dir: Path, docid_to_path: dict[str, str], out_dir: Path) -> None:
    per_query_dir = run_dir / "_per_query"
    traces_dir = run_dir / "_traces"
    qid_files = sorted(
        per_query_dir.glob("qid_*.json"),
        key=lambda p: int(re.search(r"qid_(\d+)", p.name).group(1)),
    )
    if not qid_files:
        sys.exit(f"no _per_query/qid_*.json under {run_dir}")

    per_call_rows = []
    per_query_rows = []

    for qf in qid_files:
        qid = re.search(r"qid_(\d+)", qf.name).group(1)
        meta = json.loads(qf.read_text())
        gold_ids = [str(g) for g in meta.get("gold_doc_ids") or []]
        gold_paths = sorted({docid_to_path[g] for g in gold_ids if g in docid_to_path})
        gold_path_set = set(gold_paths)
        is_correct = bool(meta.get("is_correct"))
        all_retrieved = set(str(d) for d in meta.get("all_retrieved") or [])
        ever_retrieved = any(g in all_retrieved for g in gold_ids)

        trace_path = traces_dir / f"qid_{qid}" / "single.json"
        best_rank = None
        best_rank_call = None
        best_rank_ratio = None
        best_newdocs_pos = None
        first_preview_call = None
        first_any_call = None
        n_search_calls = 0

        if trace_path.exists():
            for call_idx, call in iter_search_calls(trace_path):
                n_search_calls += 1
                preview = parse_search_preview(call.get("result_text") or "")
                top_score = max((e["score"] for e in preview), default=None)

                gold_preview = [e for e in preview if e["path"] in gold_path_set]
                gp_rank = min((e["rank"] for e in gold_preview), default=None)
                gp_score = None
                gp_query = None
                if gold_preview:
                    best_e = min(gold_preview, key=lambda e: e["rank"])
                    gp_score = best_e["score"]
                    gp_query = best_e["query"]
                    if first_preview_call is None:
                        first_preview_call = call_idx
                    if best_rank is None or gp_rank < best_rank:
                        best_rank = gp_rank
                        best_rank_call = call_idx
                        best_rank_ratio = gp_score / top_score if top_score else None

                newdocs = call.get("result_paths") or []
                nd_positions = [i + 1 for i, p in enumerate(newdocs) if p in gold_path_set]
                nd_pos = min(nd_positions) if nd_positions else None
                if nd_pos is not None:
                    if best_newdocs_pos is None or nd_pos < best_newdocs_pos:
                        best_newdocs_pos = nd_pos

                if gp_rank is not None or nd_pos is not None:
                    if first_any_call is None:
                        first_any_call = call_idx

                per_call_rows.append(
                    {
                        "qid": qid,
                        "call_idx": call_idx,
                        "n_preview_entries": len(preview),
                        "gold_in_preview_rank": gp_rank if gp_rank is not None else "",
                        "gold_in_preview_score": gp_score if gp_score is not None else "",
                        "gold_in_preview_query": gp_query if gp_query is not None else "",
                        "gold_in_newdocs_pos": nd_pos if nd_pos is not None else "",
                        "top_preview_score": top_score if top_score is not None else "",
                        "score_ratio": round(gp_score / top_score, 4) if (gp_score and top_score) else "",
                        "gold_paths": ";".join(gold_paths),
                    }
                )
        else:
            print(f"  [warn] missing trace for qid_{qid}", file=sys.stderr)

        if best_rank is not None:
            category = {1: "1", 2: "2", 3: "3"}.get(
                best_rank, "4-5" if best_rank <= 5 else "6-10"
            )
        elif best_newdocs_pos is not None:
            category = "11-30 (added to wd)"
        elif ever_retrieved:
            category = "retrieved, rank>30/unseen"
        else:
            category = "never retrieved"

        per_query_rows.append(
            {
                "qid": qid,
                "is_correct": is_correct,
                "n_search_calls": n_search_calls,
                "best_preview_rank": best_rank if best_rank is not None else "",
                "best_preview_call": best_rank_call if best_rank_call is not None else "",
                "best_preview_score_ratio": best_rank_ratio if best_rank_ratio is not None else "",
                "best_newdocs_pos": best_newdocs_pos if best_newdocs_pos is not None else "",
                "first_any_call": first_any_call if first_any_call is not None else "",
                "ever_retrieved": ever_retrieved,
                "visible_in_preview": best_rank is not None,
                "category": category,
                "gold_paths": ";".join(gold_paths),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "gold_dist_per_call.csv", per_call_rows)
    _write_csv(out_dir / "gold_dist_per_query.csv", per_query_rows)

    print_summary(per_query_rows, run_dir)
    plot_heatmap(per_query_rows, per_call_rows, run_dir, out_dir)
    plot_first_rank(per_query_rows, run_dir, out_dir)
    plot_score_ratio(per_query_rows, run_dir, out_dir)
    print(f"\noutputs written to {out_dir}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(per_query_rows: list[dict], run_dir: Path) -> None:
    n = len(per_query_rows)
    n_correct = sum(r["is_correct"] for r in per_query_rows)
    n_preview = sum(r["visible_in_preview"] for r in per_query_rows)
    n_ever = sum(r["ever_retrieved"] for r in per_query_rows)
    print(f"run: {run_dir.name}")
    print(f"queries: {n}  answered-correct: {n_correct} ({n_correct / n:.1%})")
    print(f"gold doc visible in a top-10 preview at least once: {n_preview} ({n_preview / n:.1%})")
    print(f"gold doc ever added to working dir: {n_ever} ({n_ever / n:.1%})")
    print("\nbest-appearance category x correctness (row % correct):")
    for cat in RANK_BUCKETS:
        rows = [r for r in per_query_rows if r["category"] == cat]
        if not rows:
            continue
        c = sum(r["is_correct"] for r in rows)
        print(f"  {cat:<28} n={len(rows):>3}  correct={c:>3} ({c / len(rows):.0%})")


def _category_code(row: dict, call_idx: int, calls_by_qid: dict[str, list[dict]]) -> float:
    """Cell code for the heatmap: 1-10 preview rank, 20 = in new-docs list, 30 = ever retrieved only, NaN = absent."""
    calls = calls_by_qid[row["qid"]]
    if call_idx >= len(calls):
        return math.nan
    call = calls[call_idx]
    if call["gold_in_preview_rank"] != "":
        return float(call["gold_in_preview_rank"])
    if call["gold_in_newdocs_pos"] != "":
        return 20.0
    return math.nan


def plot_heatmap(per_query_rows, per_call_rows, run_dir, out_dir) -> None:
    calls_by_qid: dict[str, list[dict]] = {}
    for r in per_call_rows:
        calls_by_qid.setdefault(r["qid"], []).append(r)
    for qid in calls_by_qid:
        calls_by_qid[qid].sort(key=lambda r: r["call_idx"])

    max_calls = max((len(v) for v in calls_by_qid.values()), default=0)
    order = sorted(
        per_query_rows,
        key=lambda r: (
            r["first_any_call"] if r["first_any_call"] != "" else 999,
            r["best_preview_rank"] if r["best_preview_rank"] != "" else 99,
            int(r["qid"]),
        ),
    )
    matrix = np.full((len(order), max_calls), math.nan)
    for i, row in enumerate(order):
        for j in range(max_calls):
            matrix[i, j] = _category_code(row, j, calls_by_qid)

    # color: rank1 -> green ... rank10 -> red; 20 -> orange; NaN -> light gray
    rank_colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.05, 0.95, 10))
    cmap = ListedColormap(list(rank_colors) + ["#fdb462", "#b3b3b3"])
    cmap.set_bad("#f0f0f0")
    norm = BoundaryNorm(np.arange(0.5, 11.5, 1.0).tolist() + [25.0, 35.0], cmap.N)

    height = max(6, len(order) * 0.24)
    fig, ax = plt.subplots(figsize=(min(2 + max_calls * 0.55, 14), height))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(max_calls))
    ax.set_xticklabels(range(max_calls), fontsize=8)
    ax.set_xlabel("search call index (chronological)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"qid_{r['qid']}" for r in order], fontsize=6)
    for tick, r in zip(ax.get_yticklabels(), order):
        tick.set_color("#1a9850" if r["is_correct"] else "#d73027")
    ax.set_title(
        f"Where gold docs sit in search results — {run_dir.name}\n"
        "(green row label = answered correctly; red = incorrect)"
    )
    cb = fig.colorbar(im, ax=ax, ticks=list(range(1, 11)) + [20, 30], pad=0.01)
    cb.ax.set_yticklabels([str(i) for i in range(1, 11)] + ["in wd (11-30)", "retrieved, not in top-30"], fontsize=7)
    cb.set_label("best gold-doc rank in that call's top-10 preview", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gold_dist_heatmap.png", dpi=150)
    plt.close(fig)


def plot_first_rank(per_query_rows, run_dir, out_dir) -> None:
    cats = RANK_BUCKETS
    correct_counts, incorrect_counts = [], []
    for cat in cats:
        rows = [r for r in per_query_rows if r["category"] == cat]
        correct_counts.append(sum(r["is_correct"] for r in rows))
        incorrect_counts.append(len(rows) - correct_counts[-1])

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, correct_counts, color="#1a9850", label="correct")
    ax.bar(x, incorrect_counts, bottom=correct_counts, color="#d73027", label="incorrect")
    for i, (c, w) in enumerate(zip(correct_counts, incorrect_counts)):
        total = c + w
        if total:
            ax.text(i, total + 0.4, f"{total}\n({c / total:.0%} correct)", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("number of queries")
    ax.set_title(
        f"Best rank at which the gold doc appears — {run_dir.name}\n"
        "(best top-10 preview rank across all search calls of a query)"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gold_dist_first_rank.png", dpi=150)
    plt.close(fig)


def plot_score_ratio(per_query_rows, run_dir, out_dir) -> None:
    rng = random.Random(0)
    rows_preview = [r for r in per_query_rows if r["best_preview_rank"] != "" and r["best_preview_score_ratio"] != ""]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for r in rows_preview:
        x = float(r["best_preview_rank"]) + rng.uniform(-0.3, 0.3)
        y = float(r["best_preview_score_ratio"])
        ax1.scatter(x, y, c="#1a9850" if r["is_correct"] else "#d73027", alpha=0.65, s=28, edgecolors="none")
    ax1.axhline(1.0, color="#666666", lw=0.8, ls="--")
    ax1.set_xlabel("best gold-doc rank in top-10 preview (jittered)")
    ax1.set_ylabel("gold score / top-1 score in that call")
    ax1.set_title("How close the gold doc scores to the top hit")
    ax1.set_xticks(range(1, 11))
    ax1.set_ylim(0, 1.05)
    ax1.spines[["top", "right"]].set_visible(False)

    first_calls = [int(r["first_any_call"]) for r in per_query_rows if r["first_any_call"] != ""]
    never = sum(1 for r in per_query_rows if r["first_any_call"] == "")
    bins = np.arange(-0.5, max(first_calls, default=0) + 1.5, 1.0)
    ax2.hist(
        [c for c, r in zip(first_calls, [r for r in per_query_rows if r["first_any_call"] != ""]) if r["is_correct"]],
        bins=bins, color="#1a9850", alpha=0.85, label="correct",
    )
    ax2.hist(
        [c for c, r in zip(first_calls, [r for r in per_query_rows if r["first_any_call"] != ""]) if not r["is_correct"]],
        bins=bins, color="#d73027", alpha=0.7, label="incorrect",
    )
    ax2.set_xlabel("search call index of first gold-doc appearance")
    ax2.set_ylabel("number of queries")
    ax2.set_title(f"How deep agents must dig ({never} queries: gold never appears)")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Gold-doc accessibility vs outcome — {run_dir.name}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "gold_dist_score_ratio.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="run directory containing _per_query/ and _traces/")
    ap.add_argument(
        "--map",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "corpus_filename_docid_map.json",
        help="corpus_filename_docid_map.json (default: <repo>/runs/corpus_filename_docid_map.json)",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (default: <run_dir>/_analysis/gold_dist)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "_analysis" / "gold_dist")).resolve()
    if not (run_dir / "_per_query").is_dir():
        sys.exit(f"not a run directory: {run_dir}")
    docid_to_path = load_docid_map(args.map.resolve())
    print(f"loaded docid map: {len(docid_to_path)} docs")
    analyze_run(run_dir, docid_to_path, out_dir)


if __name__ == "__main__":
    main()
