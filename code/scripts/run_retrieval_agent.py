#!/usr/bin/env python3
"""Run the BrowseComp-Plus paper-aligned retrieval agent on the dev set.

Setup (BCP §4.3 + §E + §F + texttron/BrowseComp-Plus code, verbatim where
it matters):
  - Agent: `search(query)` + `get_document(docid)` tools (paper main setting;
    pass `--no-get-document` to reproduce the §4.8.3 ablation), top-k=5 search,
    first-512-token snippet (word-split approximation; uniform across all our
    baselines for fairness).
  - System/user prompt: BCP `QUERY_TEMPLATE` (with both tools) or
    `QUERY_TEMPLATE_NO_GET_DOCUMENT` (search-only), both verbatim.
  - max_iterations=100 (matches BCP code's outer-loop default — counts model
    API calls, not individual tool calls).
  - Judge: BCP Appendix F prompt verbatim.

Tracks per-query AND aggregate: tokens (input / cached_input / output /
reasoning), tool call count + per-call log (search + get_document separately),
cost estimate, judge result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from rise.api_retry import effective_reasoning_effort

from rise.bcp_retrieval_agent import run_bcp_retrieval_agent, warmup_qwen_tokenizer
from rise.corpus_fs_lookup import DocidFsLookup, load_docid_to_relpath_map
from rise.decompose import make_client
from rise.dci_artifacts import PRICE_TABLE, estimate_cost
from rise.protocol import for_bcp_retrieval
from rise.retrieval import load_index
from rise.trajectory import (
    PLACEHOLDER_JUDGE_OUT, PLACEHOLDER_JUDGE_USAGE,
    SCHEMA_VERSION, atomic_write_json, build_per_query_row,
    cached_row_is_reusable, compute_run_config_hash,
)

DEFAULT_MINI_DEV = REPO_ROOT / "data" / "queries_100.jsonl"
DEFAULT_INDEX = REPO_ROOT / "runs" / "bm25_full"
DEFAULT_BC_PLUS_DOCS = REPO_ROOT / "corpus" / "bcp_plus"
DEFAULT_FILENAME_MAP = REPO_ROOT / "runs" / "corpus_filename_docid_map.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mini-dev", type=Path, default=DEFAULT_MINI_DEV)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--bc-plus-docs", type=Path, default=DEFAULT_BC_PLUS_DOCS,
                    help="Filesystem root of `.txt` corpus files; docs read on demand via DocidFsLookup.")
    ap.add_argument("--filename-map", type=Path, default=DEFAULT_FILENAME_MAP,
                    help="JSON {relpath: docid} for the BCP+ corpus; inverted at load to {docid: relpath}.")
    ap.add_argument("--agent-model", default="deepseek-v4-pro")
    ap.add_argument("--per-search-k", type=int, default=5)
    ap.add_argument("--snippet-tokens", type=int, default=512)
    ap.add_argument("--max-iterations", type=int, default=100,
                    help="BCP code default (openai_client.py:175). Counts model API calls, not individual tool calls.")
    ap.add_argument("--no-get-document", action="store_true",
                    help="Disable the `get_document` tool (BCP §4.8.3 ablation; default is to enable it, matching the paper's main setting).")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    enable_get_document = not args.no_get_document

    if args.out is None:
        agent_tag = args.agent_model.replace("deepseek-", "")
        gd_tag = "gd" if enable_get_document else "search"
        # Encode corpus identity (index dir name) in the default path so 100k and
        # 1M runs land in different output dirs by default.
        idx_tag = args.index_dir.name
        args.out = REPO_ROOT / "runs" / f"bcp_retrieval_agent_mini_dev_{agent_tag}_k{args.per_search_k}_{gd_tag}_mi{args.max_iterations}_{idx_tag}.json"

    run_root = args.out.with_suffix("")
    per_query_dir = run_root / "_per_query"
    trace_root = run_root / "_traces"
    per_query_dir.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    reasoning_effort = effective_reasoning_effort()

    # Full run-config hash — covers every knob that changes agent behavior
    # OR the corpus the agent sees. A cached per-query row is only reusable
    # when this hash matches.
    # Unified RunConfig (budget/wall_clock/retries plumb into the run loop).
    run_config = for_bcp_retrieval(
        agent_model=args.agent_model,
        index_dir=args.index_dir,
        bc_plus_docs=args.bc_plus_docs,
        filename_map=args.filename_map,
        per_search_k=args.per_search_k,
        snippet_tokens=args.snippet_tokens,
        enable_get_document=enable_get_document,
        max_model_calls=args.max_iterations,
        reasoning_effort=reasoning_effort,
    )
    run_config_hash = compute_run_config_hash(run_config.to_hashable_dict())
    print(f"run_config_hash: {run_config_hash}")

    print(f"loading BM25 index ...")
    retriever, doc_ids = load_index(args.index_dir)
    print(f"  {len(doc_ids):,} docs")
    docid_to_relpath = load_docid_to_relpath_map(args.filename_map)
    corpus_by_docid = DocidFsLookup(args.bc_plus_docs, docid_to_relpath)
    print(f"  {len(corpus_by_docid):,} corpus texts via disk-on-demand (DocidFsLookup)")
    print(f"warming up Qwen3 tokenizer ...")
    warmup_qwen_tokenizer()
    print(f"  tokenizer ready")

    records = [json.loads(l) for l in args.mini_dev.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"running BCP retrieval agent on {len(records)} queries")
    print(f"  agent: {args.agent_model}  (judge: decoupled — run scripts/judge.py after)")
    print(f"  per_search_k={args.per_search_k}, snippet_tokens={args.snippet_tokens}, max_iterations={args.max_iterations}, enable_get_document={enable_get_document}, concurrency={args.concurrency}")
    print()

    agent_client = make_client(args.agent_model)
    from rise.api_retry import is_reasoner_model
    is_reasoner_agent = is_reasoner_model(args.agent_model)

    def _process(rec: dict) -> dict:
        qid = rec["query_id"]
        # Resume: only reuse a saved row when its schema/runner/model/judge_model
        # all match this run's config — otherwise an old (pre-schema-change)
        # artifact would pollute the aggregate.
        per_query_path = per_query_dir / f"qid_{qid}.json"
        if per_query_path.exists():
            try:
                cached = json.loads(per_query_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if cached is not None:
                ok, reason = cached_row_is_reusable(
                    cached,
                    expected_runner="bcp_retrieval_agent",
                    expected_model=args.agent_model,
                    expected_run_config_hash=run_config_hash,
                )
                if ok:
                    return cached
                print(f"qid={qid} cache invalid ({reason}); re-running", flush=True)
        # 1) Run the agent
        run = run_bcp_retrieval_agent(
            question=rec["query"],
            query_id=qid,
            run_config=run_config,
            searcher=retriever,
            doc_ids=doc_ids,
            corpus_by_docid=corpus_by_docid,
            client=agent_client,
            is_reasoner=is_reasoner_agent,
        )
        agent_usage = {
            "input_tokens": run.input_tokens,
            "cached_input_tokens": run.cached_input_tokens,
            "output_tokens": run.output_tokens,
            "reasoning_tokens": run.reasoning_tokens,
        }
        # Prefer per-turn cost (matches what the live API charged each call);
        # fall back to whole-run estimate if turn trace was empty.
        agent_cost = run.agent_cost_usd if run.agent_cost_usd > 0 else estimate_cost(agent_usage, args.agent_model)

        # 2) Judge — decoupled. We write placeholder fields here; the
        # `scripts/judge.py` script reads per_query.json later and
        # atomically overwrites the judge / is_correct / total_cost_usd
        # fields. Signal of "unjudged": judge.reasoning == "".
        judge_out = dict(PLACEHOLDER_JUDGE_OUT)
        judge_usage = dict(PLACEHOLDER_JUDGE_USAGE)
        judge_cost = 0.0

        # 3) Compute recall from the retrieved set
        gold = set(rec.get("gold_doc_ids", []))
        evi = set(rec.get("evidence_doc_ids", []))
        retrieved_set = set(run.all_retrieved)
        gR = len(retrieved_set & gold) / max(1, len(gold))
        eR = len(retrieved_set & evi) / max(1, len(evi))

        # Step D: trajectory + tool-call rounds move out of per_query.json
        # into a separate `_traces/qid_X/bcp_retrieval.json` file. Per-query
        # JSON keeps only the `trace_path` pointer.
        trace_dir = trace_root / f"qid_{qid}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "bcp_retrieval.json"
        atomic_write_json(trace_path, {
            "schema_version": SCHEMA_VERSION,
            "runner": "bcp_retrieval_agent",
            "query_id": qid,
            "turns": run.turns,
            "rounds": run.rounds,
            "all_retrieved": run.all_retrieved,
        })

        # `surfaced_docids` for BCP = the set of docids the retrieval tools
        # ever surfaced (== retrieved_set). Pass empty surfaced_relpaths
        # because BCP retrieves by docid, not relpath.
        row = build_per_query_row(
            run_config=run_config, run_config_hash=run_config_hash,
            query_id=qid, question=rec["query"], gold_answer=rec["answer"],
            gold_doc_ids=list(gold), evidence_doc_ids=list(evi),
            final_text=run.final_text,
            judge_out=judge_out,
            judge_usage=judge_usage, judge_cost_usd=judge_cost,
            terminated_by=run.terminated_by, elapsed_seconds=run.elapsed_seconds,
            n_turns=len(run.turns or []), n_attempts=1,
            agent_usage=agent_usage, agent_cost_usd=agent_cost,
            surfaced_relpaths=[],
            surfaced_docids=retrieved_set,
            all_retrieved=run.all_retrieved,
            tool_usage={
                "search": run.search_calls,
                "get_document": run.get_doc_calls,
            },
            runner_specific={
                "gold_recall": gR,
                "evidence_recall": eR,
                "api_calls": run.api_calls,
            },
            trace_path=str(trace_path.relative_to(run_root)),
        )
        atomic_write_json(per_query_path, row)
        return row

    results: list[dict] = []
    print_lock = threading.Lock()
    header = f"{'qid':>5}  {'sc':>3}  {'gd':>3}  {'ndoc':>5}  {'gR':>5}  {'eR':>5}  {'cost':>7}  expected -> final_text"
    print(header, flush=True)
    print("-" * len(header), flush=True)

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
                rs = row.get("runner_specific") or {}
                n_retrieved = len(row.get("all_retrieved") or [])
                # No `corr` column — judge is decoupled; row contains
                # placeholder judge fields until scripts/judge.py runs.
                print(
                    f"{row['query_id']:>5}  {tu.get('search', 0):>3}  {tu.get('get_document', 0):>3}  {n_retrieved:>5}  "
                    f"{rs.get('gold_recall', 0.0)*100:>5.1f}  {rs.get('evidence_recall', 0.0)*100:>5.1f}  "
                    f"${row['agent_cost_usd']:>6.4f}  "
                    f"{str(row['gold_answer'])[:30]} -> {str(row['final_text'])[:35].splitlines()[0] if row['final_text'] else ''}",
                    flush=True,
                )

    results.sort(key=lambda r: int(r["query_id"]) if str(r["query_id"]).isdigit() else 0)

    if not results:
        print("no results")
        return

    n = len(results)
    judge_pending = all((r.get("judge") or {}).get("reasoning", "") == "" for r in results)
    agg_tokens = {k: sum(r["agent_usage"][k] for r in results) for k in ["input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"]}
    total_agent_cost = sum(r["agent_cost_usd"] for r in results)

    summary = {
        "config": {
            "agent_model": args.agent_model,
            "per_search_k": args.per_search_k,
            "snippet_tokens": args.snippet_tokens,
            "max_iterations": args.max_iterations,
            "enable_get_document": enable_get_document,
            "concurrency": args.concurrency,
            "system_prompt": "BCP QUERY_TEMPLATE (verbatim)" if enable_get_document else "BCP QUERY_TEMPLATE_NO_GET_DOCUMENT (verbatim)",
            "price_table_per_1m": PRICE_TABLE.get(args.agent_model, {}),
        },
        "aggregate": {
            "n": n,
            "judge_pending": judge_pending,
            "mean_search_calls": mean((r.get("tool_usage") or {}).get("search", 0) for r in results),
            "mean_get_doc_calls": mean((r.get("tool_usage") or {}).get("get_document", 0) for r in results),
            "mean_n_retrieved": mean(len(r.get("all_retrieved") or []) for r in results),
            "mean_gold_recall": mean((r.get("runner_specific") or {}).get("gold_recall", 0.0) for r in results),
            "mean_evidence_recall": mean((r.get("runner_specific") or {}).get("evidence_recall", 0.0) for r in results),
            "agent_tokens_total": agg_tokens,
            "total_agent_cost_usd": total_agent_cost,
            "mean_agent_cost_per_query_usd": total_agent_cost / n,
            "mean_elapsed_seconds": mean(r["elapsed_seconds"] for r in results),
        },
        "per_query": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    a = summary["aggregate"]
    run_root = args.out.with_suffix("")
    print()
    print(f"(judge pending — run: scripts/judge.py --run-dir {run_root} --judge-model <MODEL>)")
    print(f"n: {a['n']}")
    print(f"mean search calls:  {a['mean_search_calls']:.1f}")
    print(f"mean retrieved:     {a['mean_n_retrieved']:.1f}")
    print(f"mean gold recall:   {a['mean_gold_recall']*100:.1f}")
    print(f"mean evidence recall: {a['mean_evidence_recall']*100:.1f}")
    print(f"agent tokens:       in={agg_tokens['input_tokens']:,}  cached={agg_tokens['cached_input_tokens']:,}  out={agg_tokens['output_tokens']:,}  reasoning={agg_tokens['reasoning_tokens']:,}")
    print(f"agent cost total / mean: ${total_agent_cost:.4f} / ${a['mean_agent_cost_per_query_usd']:.4f}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
