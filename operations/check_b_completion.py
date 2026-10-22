#!/usr/bin/env python3
"""B100-only administrative completion gate; never starts a model or A run.

Only readiness, paths, hashes and counts reach stdout. Private trace/CP values
are compared locally. The export contains public finals and numeric metrics;
blind pairing with the reused historical baseline happens later, locally.
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_experiment import runner, analyze
from operations.strict_qwen35_check_completion import require, write_or_verify

ORIGINAL_MANIFEST_SHA256 = "6d64352b47ae0ec09ea2af6763be74d94b451754caf5e096dc98b5c920c67736"


def a_snapshot():
    root = runner.RUNS / "A"
    return {str(path.relative_to(root)): runner.file_sha(path)
        for name in ("_per_query", "_checkpoints", "api_ledgers", "request_contracts", "_traces")
        for path in sorted((root / name).rglob("*")) if path.is_file()}


def verify_b(manifest, cohort):
    proof, records = [], []
    corpus_receipt_sha = runner.file_sha(manifest["config"]["corpus_manifest"])
    for query in runner.validate_queries():
        qid = query["query_id"]
        output = runner.RUNS / "B"
        terminal = runner.terminal_path("B", qid)
        row = runner.validate_terminal(runner.load(terminal), manifest, "B", query)
        analyze.validate_row(row, "B", qid, manifest, cohort)
        require(all(row.get("execution_checks", {}).get(key) is True for key in
                    ("contract_verified", "model_verified", "token_ledger_verified")))
        runner.verify_execution_artifacts(manifest, "B", query, row)
        paths = [terminal, output / "request_contracts" / f"qid_{qid}.json",
            output / "api_ledgers" / f"qid_{qid}.jsonl",
            output / "_traces" / f"qid_{qid}" / "single.json",
            output / "_checkpoints" / f"qid_{qid}.json",
            output / "_maintenance/completed" / f"qid_{qid}.json",
            output / "_maintenance/source_manifests" / f"qid_{qid}.json"]
        trace, cp = runner.load(paths[3]), runner.load(paths[4])
        saved, contract = cp["run"], manifest["contracts"]["B"]
        user = contract["user_prompt_template"].format(question=query["query"])
        require(trace["model"] == saved["model"] == runner.MODEL)
        require(trace["system_prompt"] == saved["system_prompt"] == contract["system_prompt"])
        require(trace["user_prompt"] == saved["user_prompt"] == user)
        require(trace["final_text"] == saved["final_answer"] == row["final_text"])
        require(trace["terminated_by"] == row["terminated_by"])
        require(saved["terminated_reason"] == row["terminated_by"] or (
            row["terminated_by"] == "max_turns" and saved["terminated_reason"] == ""
            and row["n_turns"] == manifest["config"]["max_turns"]))
        require(trace["totals"]["n_turns"] == len(trace["turns"])
                == len(saved["turns"]) == row["n_turns"])
        require(cp["identity"] == {"model": runner.MODEL, "query_id": str(qid),
            "question_sha256": row["question_sha256"], "run_config_hash": row["run_config_hash"],
            "system_prompt_sha256": contract["system_prompt_sha256"],
            "tool_specs_sha256": contract["tool_specs_sha256"]})
        require(trace["totals"]["prompt_tokens"] + trace["totals"]["cached_prompt_tokens"]
                == saved["total_prompt_tokens"] + saved["total_cached_prompt_tokens"] == row["input_tokens"])
        require(trace["totals"]["completion_tokens"] == saved["total_completion_tokens"] == row["output_tokens"])
        maintenance, recovery = runner.load(paths[5]), runner.load(paths[6])
        expected = {"arm": "B", "query_id": qid, "manifest_sha256": manifest["manifest_sha256"],
                    "terminal_row_sha256": row["row_sha256"]}
        require(all(maintenance.get(key) == recovery.get(key) == value for key, value in expected.items()))
        require(maintenance["status"] == "completed"
                and maintenance["recovery_map_sha256"] == runner.file_sha(paths[6]))
        require(recovery["source_config_sha256"] == manifest["config_sha256"])
        require(recovery["corpus_receipt_sha256"] == corpus_receipt_sha)
        require(not (output / "_working" / f"qid_{qid}").exists()
                and not (output / "_rerank_runtime" / f"qid_{qid}").exists())
        proof.append({"query_id": qid, "files": {
            str(path.relative_to(ROOT)): runner.file_sha(path) for path in paths}})
        # Never export rerank call_history, index_prepare_history, tool payloads
        # or source evidence. Its numeric counters are sufficient for costs.
        numeric_rerank = {key: value for key, value in row["reranker_summary"].items()
            if type(value) in (int, float) and math.isfinite(value)}
        records.append({"query_id": qid, "question": query["query"], "gold_answer": query["answer"],
            "final_text": row["final_text"], "final_sha256": row["final_sha256"],
            "terminal_row_sha256": row["row_sha256"], "terminated_by": row["terminated_by"],
            "metrics": analyze.row_metrics(row, "B"), "reranker_summary": numeric_rerank})
        del trace, cp, saved, recovery, maintenance
    require(len(proof) == len(records) == 100)
    return proof, records


def main():
    epoch_dir = runner.REPORTS / "operations/epoch_20261022"
    count = len(list((runner.RUNS / "B/_per_query").glob("qid_*.json")))
    output = {"ready_for_shutdown": False, "phase": "waiting_for_B100", "B_terminal_files": count}
    if count != 100 or not (epoch_dir / "b_completion.json").exists():
        print(json.dumps(output)); return
    try:
        output["phase"] = "worker_active"
        with (runner.REPORTS / "worker.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output["phase"] = "validating_B100"
            manifest = runner.verify_manifest(resources=False)
            require(manifest["manifest_sha256"] == ORIGINAL_MANIFEST_SHA256)
            epoch = runner.load(epoch_dir / "b_epoch.json")
            require(epoch["epoch_sha256"] == runner.digest({k: v for k, v in epoch.items() if k != "epoch_sha256"}))
            require(epoch["arm"] == "B" and epoch["manifest_sha256"] == ORIGINAL_MANIFEST_SHA256
                    and epoch["config_sha256"] == manifest["config_sha256"])
            completion = runner.load(epoch_dir / "b_completion.json")
            require(completion.get("status") == "B_completed" and completion.get("b_terminal_count") == 100
                    and completion.get("a_started") == 0 and completion.get("a_artifacts_unchanged") is True
                    and completion.get("manifest_sha256") == ORIGINAL_MANIFEST_SHA256
                    and completion.get("epoch_sha256") == epoch["epoch_sha256"])
            require(a_snapshot() == epoch["a_before"])
            loaded, cohort, _ = analyze.load_experiment(ROOT)
            require(loaded == manifest)
            proof, records = verify_b(manifest, cohort)
            require(a_snapshot() == epoch["a_before"])
            payload = {"version": "strict-B100-shutdown-integrity-v1", "arm": "B",
                "manifest_sha256": ORIGINAL_MANIFEST_SHA256, "config_sha256": manifest["config_sha256"],
                "epoch_sha256": epoch["epoch_sha256"], "a_before_sha256": runner.digest(epoch["a_before"]),
                "a_after_sha256": runner.digest(a_snapshot()), "a_artifacts_unchanged": True,
                "checker_sha256": runner.file_sha(Path(__file__)),
                "b_completion_sha256": runner.file_sha(epoch_dir / "b_completion.json"), "items": proof}
            payload["receipt_sha256"] = runner.digest(payload)
            integrity = epoch_dir / "b_shutdown_integrity_receipt.json"
            write_or_verify(integrity, payload)
            scoring = runner.REPORTS / "scoring_B_only"
            public_path, metadata_path = scoring / "public_finals_metrics.json", scoring / "frozen_metadata.json"
            binding = {"manifest_sha256": ORIGINAL_MANIFEST_SHA256, "epoch_sha256": epoch["epoch_sha256"],
                "model": runner.MODEL, "cohort_sha256": runner.DATASET_SHA,
                "baseline_policy": "reuse historical baseline; interrupted paired-run A artifacts are unused"}
            write_or_verify(public_path, {"version": "strict-B100-public-finals-metrics-v1", **binding,
                "usage_scope": "Recorded API usage only; unlogged shutdown-window usage remains unknown per the restart audit.",
                "records": records}, private=True)
            write_or_verify(metadata_path, {**binding, "original_config": manifest["config"],
                "B_contract": manifest["contracts"]["B"], "rubric": analyze.rubric_document(),
                "rubric_sha256": analyze.digest(analyze.rubric_document()),
                "recovery_audit_sha256": completion.get("audit_sha256"),
                "partial_B_resumed": completion.get("partial_b_resumed", []),
                "blind_review": "Pair these public finals with the reused baseline, then blind-shuffle locally before fresh scoring."}, private=True)
            require(a_snapshot() == epoch["a_before"])
            output = {"ready_for_shutdown": True, "phase": "B100_verified_public_export_saved",
                "B_terminal_files": 100, "a_artifacts_unchanged": True,
                "unused_A_terminal_files": sum(path.startswith("_per_query/") for path in epoch["a_before"]),
                "manifest_sha256": ORIGINAL_MANIFEST_SHA256, "epoch_sha256": epoch["epoch_sha256"],
                "files": {str(path): runner.file_sha(path) for path in (integrity, public_path, metadata_path)}}
    except Exception as exc:
        output["error_type"] = type(exc).__name__
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
