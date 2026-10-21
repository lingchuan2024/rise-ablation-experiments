#!/usr/bin/env python3
"""Run native DCI baseline (Python `Agent` class, no Pi subprocess).

DCI = Direct Corpus Interaction: the agent has `bash` + `read` over the
full corpus filesystem, no `search` / retriever. Faithful to
`chen2026dci` §C.1 prompt + §4 max_turns=300. Replaces the legacy
Pi-based runner at `scripts/legacy/run_dci_pi_runtime.py` for headline
numbers.

Per-query output: `_traces/qid_<id>/dci_native.json` with the full
schema-v1.1 trajectory. Aggregate: `_summary.json` (judged with BCP §F).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from rise.api_retry import effective_reasoning_effort
from rise.dci_artifacts import PRICE_TABLE, estimate_cost
from rise.dci_native import run_dci_native
from rise.decompose import make_client
from rise.protocol import for_dci_native
from rise.trajectory import (
    PLACEHOLDER_JUDGE_OUT,
    PLACEHOLDER_JUDGE_USAGE,
    atomic_write_json,
    build_per_query_row,
    cached_row_is_reusable,
    compute_run_config_hash,
    validate_per_query_row,
)

DEFAULT_MINI_DEV = REPO_ROOT / "data" / "queries_100.jsonl"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus" / "bcp_plus"
DEFAULT_MAP = REPO_ROOT / "runs" / "corpus_filename_docid_map.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mini-dev", type=Path, default=DEFAULT_MINI_DEV,
                    help="JSONL query set (default: data/queries_100.jsonl).")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Full corpus filesystem root (read/bash operate over this).")
    ap.add_argument("--filename-map", type=Path, default=DEFAULT_MAP,
                    help="JSON mapping relpath ↔ docid for coverage computation.")
    ap.add_argument("--model", default="gpt-5.4-mini",
                    help="Agent model. gpt-5.4-mini / gpt-5.4-nano / mimo-v2.5-pro / ...")
    ap.add_argument("--max-model-calls", type=int, default=300,
                    help="DCI paper §4: max_turns=300. Counts LLM API calls.")
    # Pi-faithful bash tail-truncate (matches truncateTail defaults in
    # pi-mono/.../tools/truncate.ts). Tuned to match Pi-based DCI byte-for-byte.
    ap.add_argument("--bash-max-output-lines", type=int, default=2000,
                    help="Pi truncateTail max_lines (default 2000).")
    ap.add_argument("--bash-max-output-bytes", type=int, default=50 * 1024,
                    help="Pi truncateTail max_bytes (default 50KB).")
    ap.add_argument("--read-default-limit", type=int, default=2000)
    ap.add_argument("--no-sandbox", action="store_true",
                    help="Disable sandbox-exec on bash (debug only on macOS; required on Linux).")
    ap.add_argument("--coerce-final", action="store_true",
                    help="Inject 'commit now' turn at max_model_calls. "
                         "Default off — paper-faithful, fair comparison.")
    ap.add_argument("--context-level", default="level3",
                    choices=["level0", "level1", "level2", "level3", "level4", "level5"],
                    help="Pi-faithful runtime context-management level. "
                         "DCI paper §4 uses level3 (truncate tool results "
                         "to 20k chars, micro-compact at 240k accumulated).")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    if args.out_root is None:
        tag = args.model.replace("deepseek-", "")
        cd_tag = args.corpus_dir.name
        args.out_root = REPO_ROOT / "runs" / f"dci_native_{tag}_t{args.max_model_calls}_{cd_tag}"
    args.out_root = args.out_root.absolute()
    args.corpus_dir = args.corpus_dir.absolute()
    args.out_root.mkdir(parents=True, exist_ok=True)

    reasoning_effort = effective_reasoning_effort()
    coerce = bool(args.coerce_final)

    # Build the shared RunConfig and compute its hash for cache resume.
    run_config = for_dci_native(
        agent_model=args.model,
        corpus_dir=args.corpus_dir,
        filename_map=args.filename_map,
        bash_max_output_lines=args.bash_max_output_lines,
        bash_max_output_bytes=args.bash_max_output_bytes,
        read_default_limit=args.read_default_limit,
        max_model_calls=args.max_model_calls,
        coerce_final_on_max_turns=coerce,
        reasoning_effort=reasoning_effort,
        context_management=args.context_level,
    )
    run_config_hash = compute_run_config_hash(run_config.to_hashable_dict())
    print(f"run_config_hash: {run_config_hash}")

    trace_root = (args.out_root / "_traces").absolute()
    per_query_dir = (args.out_root / "_per_query").absolute()
    trace_root.mkdir(parents=True, exist_ok=True)
    per_query_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading filename↔docid map ({args.filename_map}) ...")
    map_blob = json.loads(args.filename_map.read_text(encoding="utf-8"))
    relpath_to_docid: dict[str, str] = map_blob["relpath_to_docid"]
    docid_to_relpath: dict[str, str] = {v: k for k, v in relpath_to_docid.items()}
    map_keys = set(relpath_to_docid.keys())
    print(f"  {len(relpath_to_docid):,} entries")

    records = [
        json.loads(l) for l in args.mini_dev.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"running native DCI on {len(records)} queries")
    print(f"  model={args.model}  (judge: decoupled — run scripts/judge.py after)")
    print(f"  max_model_calls={args.max_model_calls}  bash_tail={args.bash_max_output_lines}L/{args.bash_max_output_bytes // 1024}KB")
    print(f"  coerce={'YES' if coerce else 'no'}  c={args.concurrency}")
    print(f"  corpus_dir={args.corpus_dir}")
    print(f"  out_root={args.out_root}")

    agent_client = make_client(args.model)
    print()

    print_lock = threading.Lock()
    header = f"{'qid':>5}  {'turns':>5}  {'bash':>4}  {'read':>4}  {'covA':>4}  {'covM':>5}  {'cost':>8}  expected -> final_text"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    def _process(rec: dict) -> dict:
        qid = rec["query_id"]
        per_query_path = per_query_dir / f"qid_{qid}.json"
        if per_query_path.exists():
            try:
                cached = json.loads(per_query_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if cached is not None:
                ok, reason = cached_row_is_reusable(
                    cached,
                    expected_runner="dci_native",
                    expected_model=args.model,
                    expected_run_config_hash=run_config_hash,
                )
                if ok:
                    return cached
                print(f"qid={qid} cache invalid ({reason}); re-running", flush=True)
        trace_path = trace_root / f"qid_{qid}" / "dci_native.json"
        run = run_dci_native(
            question=rec["query"], query_id=qid,
            run_config=run_config,
            client=agent_client,
            trace_path=trace_path,
            trace_relpath_to_docid=relpath_to_docid,
            enable_sandbox=(not args.no_sandbox),
        )

        agent_cost = run.cost_usd

        # Judge — decoupled. We write placeholder fields here; the
        # `scripts/judge.py` script reads per_query.json later and
        # atomically overwrites the judge / is_correct / total_cost_usd
        # fields. Signal of "unjudged": judge.reasoning == "".
        judge_out = dict(PLACEHOLDER_JUDGE_OUT)
        judge_usage = dict(PLACEHOLDER_JUDGE_USAGE)
        judge_cost = 0.0

        # Coverage: walk read_paths (direct reads) + bash output (rg/grep hits)
        # for relpaths that map to corpus docids.
        gold = set(rec.get("gold_doc_ids", []))
        evi = set(rec.get("evidence_doc_ids", []))
        # `args.corpus_dir` is `.absolute()` (symlink path); the agent's
        # `read` tool resolves symlinks, so read_paths come back rooted
        # at the symlink TARGET. Strip both prefixes to be safe.
        corpus_roots = [str(args.corpus_dir), str(args.corpus_dir.resolve())]
        surfaced = set(run.surfaced_relpaths) | _surfaced_relpaths(run, map_keys, corpus_roots=corpus_roots)
        surfaced_docids = {relpath_to_docid[p] for p in surfaced if p in relpath_to_docid}

        row = build_per_query_row(
            run_config=run_config, run_config_hash=run_config_hash,
            query_id=qid, question=rec["query"], gold_answer=rec["answer"],
            gold_doc_ids=list(gold), evidence_doc_ids=list(evi),
            final_text=run.final_text,
            judge_out=judge_out,
            judge_usage=judge_usage, judge_cost_usd=judge_cost,
            terminated_by=run.terminated_by, elapsed_seconds=run.elapsed_seconds,
            n_turns=run.n_turns, n_attempts=1,
            agent_usage={
                "input_tokens": run.input_tokens,
                "cached_input_tokens": run.cached_input_tokens,
                "output_tokens": run.output_tokens,
                "reasoning_tokens": run.reasoning_tokens,
            },
            agent_cost_usd=agent_cost,
            discarded_attempt_audit={
                "tokens": run.discarded_attempt_tokens,
                "input_tokens": run.discarded_attempt_input_tokens,
                "output_tokens": run.discarded_attempt_output_tokens,
                "reasoning_tokens": run.discarded_attempt_reasoning_tokens,
                "cost_usd": run.discarded_attempt_cost_usd,
            },
            surfaced_relpaths=surfaced,
            surfaced_docids=surfaced_docids,
            all_retrieved=sorted(surfaced_docids),
            tool_usage=dict(run.tool_call_breakdown),
            runner_specific={
                "bash_kinds": run.bash_kinds,
                "read_paths": run.read_paths,
                "n_micro_compactions": run.n_micro_compactions,
                "n_tool_results_truncated": run.n_tool_results_truncated,
            },
            trace_path=str(trace_path.relative_to(args.out_root)),
        )
        atomic_write_json(per_query_path, row)
        return row

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        future_to_rec = {ex.submit(_process, rec): rec for rec in records}
        for fut in as_completed(future_to_rec):
            rec = future_to_rec[fut]
            try:
                row = fut.result()
            except Exception as e:
                with print_lock:
                    print(f"qid={rec['query_id']} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            results.append(row)
            with print_lock:
                tu = row.get("tool_usage") or {}
                # No `corr` column — judge is decoupled; row contains
                # placeholder judge fields until scripts/judge.py runs.
                print(
                    f"{row['query_id']:>5}  {row['n_turns']:>5}  "
                    f"{tu.get('bash', 0):>4}  {tu.get('read', 0):>4}  "
                    f"{row['coverage_any']*100:>3.0f}  {row['coverage_mean']*100:>4.1f}  "
                    f"${row['agent_cost_usd']:>7.4f}  "
                    f"{str(row['gold_answer'])[:30]} -> {str(row['final_text'])[:35].splitlines()[0] if row['final_text'] else ''}",
                    flush=True,
                )

    results.sort(key=lambda r: int(r["query_id"]) if str(r["query_id"]).isdigit() else 0)
    n = len(results)
    judge_pending = all((r.get("judge") or {}).get("reasoning", "") == "" for r in results)

    summary = {
        "config": {
            "model": args.model,
            "max_model_calls": args.max_model_calls,
            "bash_max_output_lines": args.bash_max_output_lines,
            "bash_max_output_bytes": args.bash_max_output_bytes,
            "read_default_limit": args.read_default_limit,
            "coerce_final_on_max_turns": coerce,
            "concurrency": args.concurrency,
            "price_table": PRICE_TABLE.get(args.model, {}),
            "run_config_hash": run_config_hash,
        },
        "aggregate": {
            "n": n,
            "judge_pending": judge_pending,
            "mean_n_turns": mean(r["n_turns"] for r in results) if n else 0,
            "mean_n_bash_calls": mean((r.get("tool_usage") or {}).get("bash", 0) for r in results) if n else 0,
            "mean_n_read_calls": mean((r.get("tool_usage") or {}).get("read", 0) for r in results) if n else 0,
            "mean_coverage_any": mean(r["coverage_any"] for r in results) if n else 0,
            "mean_coverage_mean": mean(r["coverage_mean"] for r in results) if n else 0,
            "mean_coverage_all": mean(r["coverage_all"] for r in results) if n else 0,
            "mean_evidence_coverage_mean": mean(r["evidence_coverage_mean"] for r in results) if n else 0,
            "mean_n_surfaced": mean(r["n_surfaced"] for r in results) if n else 0,
            "n_committed_natural": sum(1 for r in results if (r.get("terminated_by") or "").split(":")[0].strip() == "done"),
            "n_committed_coerced": sum(1 for r in results if "coerced_max_turns" in (r.get("terminated_by") or "") or "wall_clock_timeout" in (r.get("terminated_by") or "")),
            "n_no_final": sum(1 for r in results if not (r.get("final_text", "") or "").strip()),
            "total_agent_cost_usd": sum(r["agent_cost_usd"] for r in results),
            "mean_agent_cost_per_query_usd": (sum(r["agent_cost_usd"] for r in results) / n) if n else 0,
            "mean_elapsed_seconds": mean(r["elapsed_seconds"] for r in results) if n else 0,
        },
        "per_query": results,
    }
    out_path = args.out_root / "_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    a = summary["aggregate"]
    print()
    print(f"(judge pending — run: scripts/judge.py --run-dir {args.out_root} --judge-model <MODEL>)")
    print(f"n: {a['n']}  mean turns: {a['mean_n_turns']:.1f}")
    print(f"mean bash / read:   {a['mean_n_bash_calls']:.1f} / {a['mean_n_read_calls']:.1f}")
    print(f"commit:             natural={a['n_committed_natural']}/{a['n']}, coerced={a['n_committed_coerced']}/{a['n']}, no_final={a['n_no_final']}/{a['n']}")
    print(f"mean covA / covM:   {a['mean_coverage_any']*100:.1f} / {a['mean_coverage_mean']*100:.1f}")
    print(f"agent cost total / mean: ${a['total_agent_cost_usd']:.4f} / ${a['mean_agent_cost_per_query_usd']:.4f}")
    print(f"\nwrote -> {out_path}")


def _surfaced_relpaths(run, map_keys: set[str], *, corpus_roots: list[str] | None = None) -> set[str]:
    """Walk read.path/read.file_path args + (later) bash output strings for
    corpus relpaths that map back to docids.

    The DCI prompt explicitly tells the agent to cite paths as
    `@/abs/path/to/bc_plus_docs/<rel>` — so most reads come in as ABSOLUTE
    paths under `corpus_root`. We strip that prefix to get the
    corpus-relative key that lives in `map_keys`. Tolerates multiple
    candidate roots (typically: the .absolute() symlink path AND its
    .resolve() target), plus the older `./` and `bc_plus_docs/`
    relative prefixes.
    """
    surfaced: set[str] = set()
    roots = list(corpus_roots or [])
    prefixes = [r.rstrip("/") + "/" for r in roots if r]

    def _norm(p: str) -> str:
        s = p.strip().lstrip("@").strip()
        for pre in prefixes:
            if s.startswith(pre):
                s = s[len(pre):]
                break
        marker = "/bc_plus_docs/"
        if marker in s:
            s = s.split(marker, 1)[1]
        for pre in ("./", "bc_plus_docs/"):
            if s.startswith(pre):
                s = s[len(pre):]
        return s

    # Read args
    for fp in run.read_paths:
        if not fp:
            continue
        s = _norm(fp)
        if s in map_keys:
            surfaced.add(s)
    # Bash output walking is intentionally deferred — the AgentRun object
    # doesn't keep per-turn bash result blobs in memory after the loop;
    # those live in the trajectory JSON on disk. The coverage we report
    # here is the conservative lower bound (only direct reads + agent's
    # cited paths). A post-hoc trace walker can add rg-hit extraction
    # later without breaking the runner.
    return surfaced


if __name__ == "__main__":
    main()
