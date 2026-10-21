#!/usr/bin/env python3
"""Administrative completion gate; never runs a question or shuts down a host.

Output is limited to readiness, counts, paths and hashes. Private checkpoint
and trace JSON is compared locally; no document text, hidden reasoning, final
answer or private shuffle seed is printed or sent to a model/service.
This top-level operations file is outside the frozen runner's source_hashes.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_experiment import runner


def require(condition):
    if not condition:
        raise ValueError("Administrative completion validation failed")


def counts():
    return {arm: len(list((runner.RUNS / arm / "_per_query").glob("qid_*.json")))
            for arm in runner.ARMS}


def write_or_verify(path, payload, *, private=False):
    if path.exists():
        require(runner.load(path) == payload)
    else:
        runner.atomic_json(path, payload, immutable=True)
    if private:
        path.chmod(0o600)


def verify_all(manifest):
    entries = []
    rows = runner.validate_queries()
    require(len(rows) == 100)
    corpus_receipt_sha = runner.file_sha(manifest["config"]["corpus_manifest"])
    for query in rows:
        qid = query["query_id"]
        for arm in runner.ARMS:
            output = runner.RUNS / arm
            row_path = runner.terminal_path(arm, qid)
            row = runner.validate_terminal(runner.load(row_path), manifest, arm, query)
            termination = row.get("terminated_by")
            require(isinstance(termination, str) and bool(termination))
            require(not termination.startswith(("api_error", "infrastructure", "solver_api_failed")))
            require(all(row.get("execution_checks", {}).get(key) is True for key in
                        ("contract_verified", "model_verified", "token_ledger_verified")))
            runner.verify_execution_artifacts(manifest, arm, query, row)
            paths = {
                "terminal": row_path,
                "request_contract": output / "request_contracts" / f"qid_{qid}.json",
                "api_ledger": output / "api_ledgers" / f"qid_{qid}.jsonl",
                "trace": output / "_traces" / f"qid_{qid}" / "single.json",
                "checkpoint": output / "_checkpoints" / f"qid_{qid}.json",
                "maintenance": output / "_maintenance/completed" / f"qid_{qid}.json",
                "recovery_map": output / "_maintenance/source_manifests" / f"qid_{qid}.json",
            }
            trace = runner.load(paths["trace"])
            checkpoint = runner.load(paths["checkpoint"])
            saved = checkpoint["run"]
            contract = manifest["contracts"][arm]
            user_prompt = contract["user_prompt_template"].format(question=query["query"])
            require(trace["model"] == saved["model"] == runner.MODEL)
            require(trace["system_prompt"] == saved["system_prompt"] == contract["system_prompt"])
            require(trace["user_prompt"] == saved["user_prompt"] == user_prompt)
            require(trace["final_text"] == saved["final_answer"] == row["final_text"])
            require(trace["terminated_by"] == termination)
            require(trace["totals"]["n_turns"] == len(trace["turns"])
                    == len(saved["turns"]) == row["n_turns"])
            # The unchanged Agent saves its final tool turn before assigning
            # max_turns. That legitimate checkpoint may have an empty reason.
            require(saved["terminated_reason"] == termination or (
                termination == "max_turns" and saved["terminated_reason"] == ""
                and row["n_turns"] == manifest["config"]["max_turns"]))
            expected_identity = {
                "model": runner.MODEL, "query_id": str(qid),
                "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
                "run_config_hash": row["run_config_hash"],
                "system_prompt_sha256": contract["system_prompt_sha256"],
                "tool_specs_sha256": contract["tool_specs_sha256"],
            }
            require(checkpoint["identity"] == expected_identity)
            require(trace["totals"]["prompt_tokens"] + trace["totals"]["cached_prompt_tokens"]
                    == saved["total_prompt_tokens"] + saved["total_cached_prompt_tokens"]
                    == row["input_tokens"])
            require(trace["totals"]["completion_tokens"]
                    == saved["total_completion_tokens"] == row["output_tokens"])
            maintenance = runner.load(paths["maintenance"])
            recovery = runner.load(paths["recovery_map"])
            identity = {"arm": arm, "query_id": qid,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "terminal_row_sha256": row["row_sha256"]}
            require(all(maintenance.get(key) == recovery.get(key) == value
                        for key, value in identity.items()))
            require(maintenance["status"] == "completed")
            require(maintenance["recovery_map_sha256"] == runner.file_sha(paths["recovery_map"]))
            require(recovery["source_config_sha256"] == manifest["config_sha256"])
            require(recovery["corpus_receipt_sha256"] == corpus_receipt_sha)
            require(not (output / "_working" / f"qid_{qid}").exists())
            require(not (output / "_rerank_runtime" / f"qid_{qid}").exists())
            entries.append({"arm": arm, "query_id": qid,
                "files": {str(path.relative_to(ROOT)): runner.file_sha(path)
                          for path in paths.values()}})
            del trace, checkpoint, saved, recovery, maintenance
    require(len(entries) == 200)
    require({arm: sum(item["arm"] == arm for item in entries)
             for arm in runner.ARMS} == {"A": 100, "B": 100})
    return entries


def export_scoring(manifest):
    """Use the same API as analyze.py export-blind; no trajectory is read."""
    from strict_experiment import analyze
    frozen, cohort, _ = analyze.load_experiment(ROOT)
    require(frozen == manifest)
    rows, qids, _ = analyze.collect_rows(ROOT, manifest, cohort, scope="all100")
    require(len(rows) == 200 and len(qids) == 100)
    directory = runner.REPORTS / "scoring"
    public_path = directory / "blind_finals.json"
    private_path = directory / "blind_map.private.json"
    rubric_path = directory / "rubric.json"
    require(not public_path.exists() or private_path.exists())
    seed = analyze.load(private_path)["shuffle_seed_secret"] if private_path.exists() else None
    public, private = analyze.export_blind(rows, cohort, manifest, seed=seed)
    # Save the map first. An interrupted export can recreate its matching
    # public file using the same private seed without overwriting anything.
    write_or_verify(private_path, private, private=True)
    write_or_verify(public_path, public, private=True)
    write_or_verify(rubric_path, analyze.rubric_document())
    return {"cases": len(public["records"]),
        "public_finals": {"path": str(public_path), "sha256": runner.file_sha(public_path)},
        "private_map": {"path": str(private_path), "sha256": runner.file_sha(private_path)},
        "rubric": {"path": str(rubric_path), "sha256": runner.file_sha(rubric_path)}}


def main():
    report = {"ready_for_shutdown": False,
              "phase": "waiting_for_full_completion", "terminal_files": counts()}
    if not (runner.REPORTS / "full_completion.json").exists():
        print(json.dumps(report, ensure_ascii=False))
        return
    try:
        report["phase"] = "worker_active"
        # This is an existing lock opened without truncation or modification.
        with (runner.REPORTS / "worker.lock").open("rb") as worker_lock:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["phase"] = "validating_completion"
            manifest = runner.verify_manifest(resources=False)
            for name in ("full_completion.json", "pilot_completion.json"):
                receipt = runner.load(runner.REPORTS / name)
                require(receipt.get("status") == "completed"
                        and receipt.get("manifest_sha256") == manifest["manifest_sha256"])
            for name in ("preflight_live.json", "pilot_gate.json"):
                receipt = runner.load(runner.REPORTS / name)
                require(receipt.get("passed") is True
                        and receipt.get("manifest_sha256") == manifest["manifest_sha256"])
            items = verify_all(manifest)
            payload = {"version": "strict-qwen35-shutdown-integrity-v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "config_sha256": manifest["config_sha256"],
                "checker_sha256": runner.file_sha(Path(__file__)),
                "full_completion_sha256": runner.file_sha(runner.REPORTS / "full_completion.json"),
                "items": items}
            payload["receipt_sha256"] = runner.digest(payload)
            receipt_path = runner.REPORTS / "operations/shutdown_integrity_receipt.json"
            write_or_verify(receipt_path, payload)
            report["phase"] = "exporting_public_finals"
            scoring = export_scoring(manifest)
            report = {"ready_for_shutdown": True, "phase": "artifacts_verified_and_public_finals_exported",
                "terminal_files": {"A": 100, "B": 100}, "verified_artifacts": 200,
                "manifest_sha256": manifest["manifest_sha256"],
                "integrity_receipt": {"path": str(receipt_path), "sha256": runner.file_sha(receipt_path)},
                "scoring": scoring}
    except Exception as exc:
        # Exception messages may quote file contents; emit only the type.
        report["error_type"] = type(exc).__name__
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
