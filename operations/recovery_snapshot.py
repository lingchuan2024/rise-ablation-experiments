#!/usr/bin/env python3
"""Opaque backup and path-independent validation of the amended recovery run.

The archive preserves original bytes and server paths. Physical roots are mapped
separately; no solver, network transport, tools, scoring, or private text output
is invoked. Exactly 99 natural terminals and one observed public final cover the
100 continuation targets. The observed final is never presented as a full trace.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from operations import continuation_snapshot as prior
from operations import recovery_suite as recovery
from strict_ablations import runner, control

VERSION = "strict-recovery-snapshot-v1"
DIRECTORY = recovery.DIRECTORY
COUNTS = {"target_coverage": 100, "validated_natural_terminals": 99,
          "new_natural_terminals": 96, "reused_continuation_natural_terminals": 3,
          "materialized_observed_finals": 1, "new_outcomes": 97,
          "reused_auto20_rows": 20, "merged_auto_rerank_rows": 100,
          "search_paging_rows": 20, "public_finals": 220}
KINDS = {"natural_terminal": 96, "materialized_observed_final": 1, "infra_budget_exhausted": 0}
require, _safe, _checked, _sha = prior.require, prior._safe, prior._checked, prior._sha
_write_new, _copy_pinned, _archive = prior._write_new, prior._copy_pinned, prior._archive
verify_worker_exit = prior.verify_worker_exit


def jobs(plan):
    return [(stage["arm"], query) for stage in recovery.plan_stages(plan) for query in stage["queries"]]


def compact_paths(arm, qid):
    base = f"runs/{DIRECTORY}/{arm}"
    return [f"{base}/_per_query/qid_{qid}.json", f"{base}/request_contracts/qid_{qid}.json",
            f"{base}/api_ledgers/qid_{qid}.jsonl", f"{base}/_traces/qid_{qid}/single.json",
            f"{base}/_checkpoints/qid_{qid}.json", f"{base}/_traces/qid_{qid}/single.api.jsonl"]


def _pins(root, pins):
    require(isinstance(pins, dict) and pins, "Missing immutable evidence pins")
    for name, pin in pins.items():
        path = _safe(root, name)
        require(path.is_file() and path.stat().st_size == pin["bytes"] and
                runner.file_sha(path) == pin["sha256"], "Pinned evidence bytes differ: " + name)


def _source_plan(plan, source_root, control_root):
    """Equivalent source admission using explicit physical roots, never plan paths."""
    require(plan.get("version") == recovery.PLAN_VERSION and
            plan.get("source_manifest_sha256") == recovery.SOURCE_MANIFEST_SHA and
            plan.get("protocol_amendment") is True and plan.get("first_additional_attempt_only") is True and
            plan.get("selection_rule") == recovery.SELECTION and plan.get("reused_terminal_qids") == recovery.REUSED_QIDS,
            "Recovery plan scope/selection differs")
    source, verified_control = prior._manifest(source_root, control_root=control_root)
    require(source["manifest_sha256"] == recovery.SOURCE_MANIFEST_SHA and
            source["config_sha256"] == plan.get("base_config_sha256"), "Original continuation identity differs")
    require(len(plan.get("source_pins", {})) == 53, "All 53 incident/source evidence files are required")
    _pins(source_root, plan["source_pins"])
    suite = recovery.original.ContinuationSuite(source_root)
    suite.assert_run_scope()
    actual = {p.name for p in (suite.runs / "auto_rerank/_per_query").glob("*.json")}
    require(actual == {f"qid_{qid}.json" for qid in recovery.REUSED_QIDS}, "Original natural terminal set changed")
    require(not list((suite.runs / "search_paging/_per_query").glob("*.json")), "Original paging runs are forbidden")
    required = {f"reports/{recovery.original.DIRECTORY}/run_manifest.json"}
    queries = {q["query_id"]: q for q in runner.shared.validate_queries()}
    rows = []
    for qid in recovery.REUSED_QIDS:
        row = suite.validate_terminal(runner.load(suite.terminal_path("auto_rerank", qid)), source, "auto_rerank", queries[qid])
        suite.verify_execution_artifacts(source, "auto_rerank", queries[qid], row)
        rows.append(row)
        required.update(recovery.source_artifact_paths("auto_rerank", qid).values())
    items = plan.get("recovery_questions")
    require(isinstance(items, list) and len(items) == 4 and {q.get("query_id") for q in items} == recovery.RECOVERY_QIDS,
            "All four interrupted questions require explicit dispositions")
    for item in items:
        qid = item["query_id"]
        require(item["strategy"] == ("materialized_observed_final" if qid == 310 else "restart_remaining_budget"),
                "This snapshot is restricted to the approved three restarts and one observed final")
        recovery.validate_budget(item, source["config"])
        require(item.get("evidence_files") and set(item["evidence_files"]) <= set(plan["source_pins"]),
                "Disposition evidence is not independently pinned")
        paths = recovery.source_artifact_paths("auto_rerank", qid)
        required.update(paths[k] for k in ("request_contract", "api_ledger", "api_diagnostic"))
        if qid != 366:
            required.add(paths["checkpoint"])
        # The numeric ledger contains no messages or private reasoning.
        entries = [json.loads(line) for line in _safe(source_root, paths["api_ledger"]).read_text().splitlines() if line.strip()]
        good = [entry for entry in entries if entry.get("status") == "ok"]
        contract = source["contracts"]["auto_rerank"]["contract_sha256"]
        require(all(entry.get("requested_model") == runner.MODEL and entry.get("contract_sha256") == contract
                    for entry in entries), "Preserved request identity differs")
        require(all(entry.get("reported_model") == runner.MODEL and
                    all(type((entry.get("usage") or {}).get(k)) is int and entry["usage"][k] >= 0
                        for k in ("prompt_tokens", "completion_tokens")) for entry in good), "Preserved numeric usage differs")
        measured = {"recorded_attempts": len(entries),
                    "known_input_tokens": sum(entry["usage"]["prompt_tokens"] for entry in good),
                    "known_output_tokens": sum(entry["usage"]["completion_tokens"] for entry in good)}
        require(all(item["debit"].get(key) == value for key, value in measured.items()), "Prior debit differs from its ledger")
        require(item["debit"]["unknown_usage_attempts"] >= sum(not isinstance(e.get("usage"), dict) for e in entries) +
                item["debit"]["possible_inflight_calls"], "Unknown prior usage was hidden")
        if qid == 310:
            required.add(item["observed_final_file"])
            recovery.validate_observed_final(source_root, item, plan)
    require(required <= set(plan["source_pins"]), "A required original artifact is outside source pins")
    return source, verified_control, rows


def _manifest(root, *, source_root, control_root):
    manifest = runner.load(Path(root) / "reports" / DIRECTORY / "run_manifest.json")
    _checked(manifest, "manifest_sha256")
    plan = manifest["recovery_plan"]
    source, verified_control, old_rows = _source_plan(plan, source_root, control_root)
    require(manifest.get("recovery_plan_sha256") == runner.digest(plan) and
            manifest.get("source_continuation_manifest_sha256") == source["manifest_sha256"], "Recovery plan binding differs")
    overridden = {"manifest_sha256", "initial_service_epoch_sha256", "administrative_source_sha256",
                  "question_runs", "execution_plan", "scheduling", "terminal_policy"}
    require(all(manifest.get(key) == value for key, value in source.items() if key not in overridden),
            "An inherited experiment setting changed")
    expected = {"recovery_version": recovery.VERSION, "directory": DIRECTORY, "question_runs": 100,
                "new_scheduled_outcomes": 97, "reused_continuation_terminals": recovery.REUSED_QIDS,
                "protocol_amendment": True, "original_single_attempt_policy_preserved": False,
                "storage_admission": recovery.STORAGE_ADMISSION,
                "administrative_source_sha256": recovery.admin_hashes()}
    require(all(manifest.get(key) == value for key, value in expected.items()), "Recovery admission/source scope differs")
    stages = [{"name": s["name"], "arm": s["arm"], "query_ids": [q["query_id"] for q in s["queries"]]}
              for s in recovery.plan_stages(plan)]
    require(manifest.get("execution_plan") == {"version": recovery.VERSION, "stages": stages,
            "selection_rule": recovery.SELECTION, "original_full_three_arm_suite_complete": False},
            "Recovery stage order/scope differs")
    epoch = runner.load(Path(root) / "reports" / DIRECTORY / "service_epochs" /
                        (manifest["initial_service_epoch_sha256"] + ".json"))
    _checked(epoch, "epoch_sha256")
    require(epoch["epoch_sha256"] == manifest["initial_service_epoch_sha256"], "Initial epoch differs")
    return manifest, verified_control, old_rows


def _outcomes(root, manifest, source_root):
    suite = recovery.RecoverySuite(root)
    suite.assert_run_scope()
    expected = {(arm, q["query_id"]) for arm, q in jobs(manifest["recovery_plan"])}
    actual = set()
    for path in (suite.reports / "outcomes").rglob("*"):
        if path.is_file() or path.is_symlink():
            relative = path.relative_to(suite.reports / "outcomes").as_posix()
            _safe(suite.reports / "outcomes", relative)
            require(len(path.relative_to(suite.reports / "outcomes").parts) == 2 and path.suffix == ".json",
                    "Unexpected selected-outcome path")
            value = runner.load(path)
            pair = (value.get("arm"), value.get("query_id"))
            require(path.name == f"qid_{pair[1]}.json" and path.parent.name == pair[0] and pair not in actual,
                    "Duplicate/renamed selected outcome")
            actual.add(pair)
    require(actual == expected and len(actual) == 97, "Exactly 97 new selected outcomes are required")
    natural_pairs = expected - {("auto_rerank", 310)}
    actual_terminals = {(arm, int(p.stem.removeprefix("qid_"))) for arm in suite.probe_arms
                        for p in (suite.runs / arm / "_per_query").glob("qid_*.json")}
    require(actual_terminals == natural_pairs, "Exactly 96 new solver terminals are required")
    rows, outcomes = [], []
    for arm, query in jobs(manifest["recovery_plan"]):
        qid = query["query_id"]
        outcome = runner.load(suite.outcome_path(arm, qid))
        item = recovery.question_plan(manifest["recovery_plan"], qid) if arm == "auto_rerank" else None
        _checked(outcome, "outcome_sha256")
        common = {"version": recovery.VERSION, "arm": arm, "query_id": qid,
                  "manifest_sha256": manifest["manifest_sha256"], "selection_rule": recovery.SELECTION,
                  "recovery_strategy": item["strategy"] if item else "original_first_attempt"}
        require(all(outcome.get(key) == value for key, value in common.items()), "Selected outcome identity differs")
        if arm == "auto_rerank" and qid == 310:
            observed = recovery.validate_observed_final(source_root, item, manifest["recovery_plan"])
            require(outcome.get("kind") == "materialized_observed_final" and outcome.get("observation") == observed and
                    outcome.get("natural_terminal") is False and outcome.get("observed_natural_response") is True and
                    outcome.get("new_model_requests") == 0 and outcome.get("complete_trace_checkpoint") is False and
                    outcome.get("final_sha256") == observed["final_sha256"], "Observed final was resampled or misrepresented")
        else:
            require(outcome.get("kind") == "natural_terminal" and outcome.get("natural_terminal") is True,
                    "A solver result is not a validated natural terminal")
            # Parent validation stays unchanged and uses only this physical run root.
            suite.validate_outcome(manifest, arm, query)
            view = suite.effective_manifest(manifest, arm, qid)
            require(outcome.get("effective_config_sha256") == view["config_sha256"] and
                    outcome.get("same_common_configuration") is (view["config"] == manifest["config"]) and
                    outcome.get("original_single_attempt_policy_preserved") is (item is None),
                    "Recovery budget/protocol disclosure differs")
            for relative in compact_paths(arm, qid):
                require(_safe(root, relative).is_file(), "Missing compact/raw API artifact: " + relative)
            rows.append(runner.load(suite.terminal_path(arm, qid)))
        outcomes.append(outcome)
    require(len(rows) == 96, "New natural terminal count differs")
    return rows, outcomes


def _completion(root, manifest, outcomes):
    reports = Path(root) / "reports" / DIRECTORY
    state, complete = runner.load(reports / "phase_state.json"), runner.load(reports / "recovery_completion.json")
    require(state.get("phase") == "complete" and state.get("manifest_sha256") == manifest["manifest_sha256"] and
            state.get("receipt_sha256") == runner.file_sha(reports / "recovery_completion.json"), "Pipeline is not complete")
    stages = []
    for stage in recovery.plan_stages(manifest["recovery_plan"]):
        selected = [o for o in outcomes if o["arm"] == stage["arm"]]
        expected = {"version": recovery.VERSION, "passed": True, "stage": stage["name"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "selected_outcomes": [{k: o[k] for k in ("query_id", "kind", "outcome_sha256")} for o in selected]}
        actual = runner.load(reports / (stage["name"] + "_completion.json"))
        require(actual == expected, "Stage completion does not bind its exact outcomes")
        stages.append(actual)
    expected = {"version": recovery.VERSION, "passed": True, "manifest_sha256": manifest["manifest_sha256"],
                "stages": stages, "new_outcome_counts": KINDS, "validated_natural_terminals": 99,
                "target_coverage": 100, "target_count": 100, "control_model_requests": 0,
                "old_auto20_model_requests": 0, "observed_final310_new_model_requests": 0,
                "original_full_three_arm_suite_complete": False}
    require(complete == expected, "Completion must distinguish 99 natural and one observed final")
    return state, complete


def _public_finals(new_rows, old_rows, reused_rows, outcomes, control_bundle):
    observed = next(o for o in outcomes if o["kind"] == "materialized_observed_final")
    projected = {"arm": "auto_rerank", "query_id": 310, "final_text": observed["observation"]["final_text"],
                 "final_sha256": observed["final_sha256"], "row_sha256": observed["outcome_sha256"],
                 "manifest_sha256": observed["manifest_sha256"], "terminated_by": "observed_final_only"}
    result = prior._public_finals([*old_rows, *new_rows, projected], reused_rows, control_bundle)
    for record in result:
        special = record["arm"] == "auto_rerank" and record["query_id"] == 310
        record.update(outcome_kind="materialized_observed_final" if special else "natural_terminal",
                      committed_source_terminal=not special,
                      archived_evidence=("only_public_control_bundle" if record["arm"] == "control" else
                                         "observed_public_final_and_original_interrupted_artifacts" if special else
                                         "complete_trace_checkpoint"))
        if special:
            record["natural_terminal"] = False
            record["complete_trace_checkpoint"] = False
            record["observation_receipt_sha256"] = observed["observation"]["receipt_sha256"]
    return result


def _required(root, manifest):
    required = {f"reports/{DIRECTORY}/{name}" for name in
                ("run_manifest.json", "phase_state.json", "recovery_completion.json", "auto80_completion.json",
                 "paging20_completion.json", "worker.lock")}
    required.add(f"reports/{DIRECTORY}/service_epochs/{manifest['initial_service_epoch_sha256']}.json")
    required.update(manifest["source_sha256"])
    required.update(manifest["administrative_source_sha256"])
    required.update(("operations/recovery_snapshot.py", "operations/continuation_snapshot.py"))
    for arm, query in jobs(manifest["recovery_plan"]):
        qid = query["query_id"]
        required.add(f"reports/{DIRECTORY}/outcomes/{arm}/qid_{qid}.json")
        if (arm, qid) != ("auto_rerank", 310): required.update(compact_paths(arm, qid))
    return required


def _report_files(root, output):
    selected = set()
    for path in (Path(root) / "reports" / DIRECTORY).rglob("*"):
        relative = path.relative_to(root)
        if (path.resolve().is_relative_to(output) or
                {"snapshots", "downloaded", "archives"}.intersection(relative.parts)):
            continue
        if "workspace_backups" in relative.parts:
            # Full workspace manifests and tar files are maintained separately.
            # Retain only small receipts that bind backup, local ACK, and deletion.
            if (path.is_file() and path.name in {"receipt.json", "deletion_receipt.json",
                    "acknowledgement.json", "local_acknowledgement.json", "local_verification.json"} and
                    path.stat().st_size <= 1024 * 1024):
                selected.add(_safe(root, relative.as_posix()))
            continue
        if path.is_file() and path.suffix in {".json", ".jsonl", ".lock"}:
            selected.add(_safe(root, relative.as_posix()))
    return selected


def _files(root, output, manifest, operation_files):
    selected = {_safe(root, relative) for relative in _required(root, manifest)}
    selected.update(_report_files(root, output))
    for name in operation_files:
        path = _safe(root, name)
        require(name.startswith("operations/") and path.suffix in {".json", ".jsonl", ".log", ".py", ".sh", ".txt"}
                and not {"credentials", "secrets", ".env"}.intersection(p.lower() for p in path.parts),
                "Extra administrative file path is unsafe")
        selected.add(path)
    for name, sha in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        require(runner.file_sha(_safe(root, name)) == sha, "Archived frozen/admin source differs")
    require(all(p.is_file() and not {"_working", "_rerank_runtime", "venv", "models", "index"}.intersection(p.parts)
                for p in selected), "Missing compact file or excluded runtime pool")
    return sorted(selected)


def export_snapshot(root, output, *, source_root=None, reused_auto20_root=None, control_root=None, operation_files=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and output != root and not output.exists(),
            "Snapshot output must be a previously nonexistent directory under the run root")
    header = runner.load(root / "reports" / DIRECTORY / "run_manifest.json")
    source_root = Path(source_root or header["recovery_plan"]["source_root"]).resolve()
    reused_root = Path(reused_auto20_root or header["reused_auto20_reference"]["root"]).resolve()
    control_root = Path(control_root or header["control_root"]).resolve()
    with ExitStack() as stack:
        lock_paths = [root / f"reports/{DIRECTORY}/worker.lock", source_root / "reports/strict_continue_v1/worker.lock",
                      reused_root / "reports/strict_auto20_v1/worker.lock"]
        for path in lock_paths:
            handle = stack.enter_context(path.open("rb"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, verified_control, old_rows = _manifest(root, source_root=source_root, control_root=control_root)
        rows, outcomes = _outcomes(root, manifest, source_root)
        state, complete = _completion(root, manifest, outcomes)
        exit_proof = verify_worker_exit(state.get("worker_identity"))
        reused = prior._verify_reused(reused_root, manifest)
        output.mkdir(parents=True, exist_ok=False)
        bundles = {"source_continuation": output / "bundles/source_continuation",
                   "reused_auto20": output / "bundles/auto20", "control": output / "bundles/control"}
        _copy_pinned(source_root, bundles["source_continuation"], manifest["recovery_plan"]["source_pins"])
        _copy_pinned(reused_root, bundles["reused_auto20"], manifest["reused_auto20_reference"]["files"])
        _copy_pinned(control_root, bundles["control"], prior.CONTROL_FILES)
        copied_manifest, copied_control, copied_old = _manifest(root, source_root=bundles["source_continuation"], control_root=bundles["control"])
        copied_reused = prior._verify_reused(bundles["reused_auto20"], manifest)
        finals = _public_finals(rows, old_rows, reused, outcomes, verified_control)
        require(finals == _public_finals(rows, copied_old, copied_reused, outcomes, copied_control), "Copied public finals differ")
        finals_path = output / "public_finals.json"
        value = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"], "records": finals,
                 "counts": COUNTS, "policy": "Fresh review required; infrastructure amendment and observed-only final remain explicit"}
        value["public_finals_sha256"] = runner.digest(value)
        _write_new(finals_path, value)
        plan_path = output / "recovery_plan.json"
        _write_new(plan_path, manifest["recovery_plan"])
        files = sorted({*_files(root, output, manifest, operation_files), finals_path, plan_path,
                        *(p for p in (output / "bundles").rglob("*") if p.is_file())})
        archive = output / "recovery_snapshot.tar.gz"
        pins = _archive(root, files, archive)
        # Recheck immutable evidence/terminal bindings after copying under all locks.
        _pins(source_root, manifest["recovery_plan"]["source_pins"])
        require(_outcomes(root, manifest, source_root) == (rows, outcomes), "Selected outcomes changed while backing up")
        require(_completion(root, manifest, outcomes) == (state, complete), "Completion changed while backing up")
        receipt = {"version": VERSION, "ready_for_shutdown": True, "counts": COUNTS,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
                   "manifest_sha256": manifest["manifest_sha256"],
                   "manifest_file_sha256": runner.file_sha(root / f"reports/{DIRECTORY}/run_manifest.json"),
                   "source_sha256": manifest["source_sha256"], "files": pins,
                   "bundles": {k: p.relative_to(root).as_posix() for k, p in bundles.items()},
                   "public_finals_path": finals_path.relative_to(root).as_posix(),
                   "recovery_plan_path": plan_path.relative_to(root).as_posix(),
                   "recovery_plan_serialization": "Canonical projection of the unchanged embedded plan; original file may be included explicitly",
                   "archive": {"path": archive.relative_to(root).as_posix(), "sha256": runner.file_sha(archive), "bytes": archive.stat().st_size},
                   "worker_exit_verification": exit_proof, "worker_locks_acquired": 3,
                   "pipeline_phase": "complete", "active_questions": 0,
                   "control_model_requests": 0, "reused_auto20_model_requests": 0, "observed_final310_new_model_requests": 0,
                   "original_full_three_arm_suite_complete": False, "checker_sha256": runner.file_sha(Path(__file__)),
                   "exclusions": ["working/runtime document pools (separately maintained)", "weights", "venv", "corpus", "index",
                                  "credentials", "prior archives", "archive itself", "receipt itself"]}
        receipt["receipt_sha256"] = runner.digest(receipt)
        receipt_path = output / "snapshot_receipt.json"
        # Verify the complete archive-backed member set through physical bundles.
        # Publish no ready receipt if this last independent verification fails.
        _publish_verified_receipt(root, receipt_path, receipt, archive)
    return {"ready_for_shutdown": True, **COUNTS, "archive": str(archive), "archive_sha256": receipt["archive"]["sha256"],
            "archive_bytes": receipt["archive"]["bytes"], "receipt": str(receipt_path),
            "receipt_file_sha256": runner.file_sha(receipt_path), "receipt_bytes": receipt_path.stat().st_size,
            "files": len(pins), "external_model_requests": 0}


def _receipt(path, *, expected_sha256=None, expected_bytes=None):
    path = Path(path)
    if expected_sha256 is not None: require(runner.file_sha(path) == expected_sha256, "Downloaded independent receipt SHA differs")
    if expected_bytes is not None: require(path.stat().st_size == expected_bytes, "Downloaded independent receipt byte size differs")
    proof = runner.load(path)
    _validate_receipt(proof)
    return proof


def _validate_receipt(proof):
    _checked(proof, "receipt_sha256")
    require(proof.get("version") == VERSION and proof.get("counts") == COUNTS and proof.get("ready_for_shutdown") is True and
            proof.get("pipeline_phase") == "complete" and proof.get("worker_locks_acquired") == 3 and proof.get("active_questions") == 0 and
            proof.get("worker_exit_verification", {}).get("worker_exited") is True and
            proof.get("original_full_three_arm_suite_complete") is False, "Recovery snapshot safety/count scope differs")
    require(all(proof.get(k) == 0 for k in ("control_model_requests", "reused_auto20_model_requests", "observed_final310_new_model_requests")),
            "Forbidden model requests recorded")
    require(isinstance(proof.get("files"), dict) and proof["files"], "Missing snapshot checksum map")


def safe_extract(archive, receipt, output, *, expected_receipt_sha256=None, expected_receipt_bytes=None):
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    prior._check_archive(archive, proof)
    output = Path(output).resolve()
    require(not output.exists(), "Extract into a new, previously nonexistent directory")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)) and set(names) == set(proof["files"]), "Unsafe or changed archive membership")
        for member in members:
            _safe(output, member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and
                    member.size == proof["files"][member.name]["bytes"], "Archive contains an unsafe member type or size")
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = _safe(output, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as original, target.open("xb") as stream:
                os.chmod(target, 0o600)
                shutil.copyfileobj(original, stream)
            require(runner.file_sha(target) == proof["files"][member.name]["sha256"], "Extracted member checksum differs")
    return {"safely_extracted": True, "root": str(output), "files": len(members), "archive_bytes_verified": True}


def _verify_snapshot(root, proof, archive):
    root = Path(root).resolve()
    _validate_receipt(proof)
    prior._check_archive(archive, proof)
    _pins(root, proof["files"])
    bundles = {k: _safe(root, value) for k, value in proof["bundles"].items()}
    require(set(bundles) == {"source_continuation", "reused_auto20", "control"}, "Physical bundle mapping differs")
    manifest, verified_control, old_rows = _manifest(root, source_root=bundles["source_continuation"], control_root=bundles["control"])
    require(manifest["manifest_sha256"] == proof["manifest_sha256"] and manifest["source_sha256"] == proof["source_sha256"] and
            runner.file_sha(root / f"reports/{DIRECTORY}/run_manifest.json") == proof["manifest_file_sha256"], "Manifest original bytes differ")
    required = _required(root, manifest) | {proof["public_finals_path"], proof["recovery_plan_path"]}
    required.update(f"{proof['bundles']['source_continuation']}/{name}" for name in manifest["recovery_plan"]["source_pins"])
    required.update(f"{proof['bundles']['reused_auto20']}/{name}" for name in manifest["reused_auto20_reference"]["files"])
    required.update(f"{proof['bundles']['control']}/{name}" for name in prior.CONTROL_FILES)
    require(required <= set(proof["files"]), "A required artifact is outside the independent receipt")
    for name, sha in {**manifest["source_sha256"], **manifest["administrative_source_sha256"]}.items():
        require(proof["files"].get(name, {}).get("sha256") == sha, "Archived source pin differs")
    require(proof["files"]["operations/recovery_snapshot.py"]["sha256"] == proof["checker_sha256"], "Archived checker differs")
    require(runner.load(_safe(root, proof["recovery_plan_path"])) == manifest["recovery_plan"], "Standalone plan differs")
    rows, outcomes = _outcomes(root, manifest, bundles["source_continuation"])
    state, _ = _completion(root, manifest, outcomes)
    require(state.get("worker_identity") == proof["worker_exit_verification"]["expected"], "Worker exit proof identity differs")
    reused = prior._verify_reused(bundles["reused_auto20"], manifest)
    finals = runner.load(_safe(root, proof["public_finals_path"]))
    _checked(finals, "public_finals_sha256")
    require(finals.get("version") == VERSION and finals.get("manifest_sha256") == manifest["manifest_sha256"] and
            finals.get("counts") == COUNTS and finals.get("records") == _public_finals(rows, old_rows, reused, outcomes, verified_control),
            "Public final coverage or observed-only labeling differs")
    return {"verified": True, **COUNTS, "archive_bytes_verified": True, "source_pins_verified": 53,
            "manifest_sha256": manifest["manifest_sha256"], "external_model_requests": 0,
            "original_full_three_arm_suite_complete": False}


def _publish_verified_receipt(root, receipt_path, proof, archive):
    verified = _verify_snapshot(root, proof, archive)
    require(verified.get("verified") is True, "Backup validation did not finish")
    _write_new(receipt_path, proof)
    return verified


def load_verified_snapshot(root, *, receipt, archive, expected_receipt_sha256=None, expected_receipt_bytes=None):
    proof = _receipt(receipt, expected_sha256=expected_receipt_sha256, expected_bytes=expected_receipt_bytes)
    return _verify_snapshot(root, proof, archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--source-root", type=Path)
    export.add_argument("--reused-auto20-root", type=Path)
    export.add_argument("--control-root", type=Path)
    export.add_argument("--operation-file", action="append", default=[])
    for name in ("extract", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--expected-receipt-sha256")
        command.add_argument("--expected-receipt-bytes", type=int)
        if name == "extract": command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_snapshot(args.root, args.output_dir, source_root=args.source_root,
            reused_auto20_root=args.reused_auto20_root, control_root=args.control_root, operation_files=args.operation_file)
    elif args.command == "extract":
        result = safe_extract(args.archive, args.receipt, args.output_dir,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    else:
        result = load_verified_snapshot(args.root, receipt=args.receipt, archive=args.archive,
            expected_receipt_sha256=args.expected_receipt_sha256, expected_receipt_bytes=args.expected_receipt_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
