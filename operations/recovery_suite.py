#!/usr/bin/env python3
"""Auditable infrastructure recovery in a new namespace; frozen solver unchanged.

A restart is explicitly a second attempt with conservative debits, never an
invented checkpoint or a silently reset budget. Observed finals do not resample.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_ablations import runner
from operations import continuation_suite as original

VERSION = "strict-infrastructure-recovery-v1"
PLAN_VERSION = "strict-continuation-recovery-plan-v1"
DIRECTORY = "strict_recovery_v1"
SOURCE_MANIFEST_SHA = "0d82c408352829635682ae80b4361f9de43b9b2bb8ef322439afdb50f56ab3da"
RECOVERY_QIDS = {293, 310, 317, 366}
REUSED_QIDS = [278, 280, 288]
STRATEGIES = {"restart_remaining_budget", "resume_checkpoint", "materialized_observed_final", "infra_budget_exhausted"}
ADMIN_FILES = ("operations/continuation_suite.py", "operations/recovery_suite.py", "operations/recover_observed_final.py")
SELECTION = "fixed_first_natural_terminal_no_correctness_selection"
STORAGE_ADMISSION = {"min_free_inodes": 40000, "min_free_bytes": 10 * 1024 ** 3, "poll_seconds": 30}


def admin_hashes():
    return {p: runner.file_sha(ROOT / p) for p in ADMIN_FILES}


def process_identity():
    identity = original.process_identity()
    identity.update(driver_sha256=runner.file_sha(Path(__file__)), entrypoint="operations.recovery_suite")
    return identity


def safe_relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or str(p) != value:
        raise ValueError("Expected a canonical relative evidence path")
    return p


def pin_files(root, pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Recovery needs immutable source evidence pins")
    for name, expected in pins.items():
        safe_relative(name)
        if original._pin(Path(root) / name) != expected:
            raise RuntimeError("Recovery source evidence bytes changed: " + name)


def question_plan(plan, qid):
    return next((q for q in plan["recovery_questions"] if q["query_id"] == qid), None)


def validate_budget(item, base):
    strategy = item.get("strategy")
    if strategy not in STRATEGIES:
        raise ValueError("Unsupported recovery strategy")
    debit = item["debit"]
    numeric = ("recorded_attempts", "possible_inflight_calls", "calls_charged", "elapsed_seconds_charged",
               "known_input_tokens", "known_output_tokens", "unknown_usage_attempts")
    if any(type(debit.get(k)) is not int or debit[k] < 0 for k in numeric):
        raise ValueError("Budget debits and known usage must be nonnegative integers")
    if debit["possible_inflight_calls"] not in {0, 1} or debit["calls_charged"] != debit["recorded_attempts"] + debit["possible_inflight_calls"]:
        raise ValueError("Charged calls must include every recorded attempt and the bounded in-flight uncertainty")
    if not isinstance(debit.get("time_bound_method"), str) or not debit["time_bound_method"]:
        raise ValueError("The elapsed debit needs an explicit evidence-backed bound")
    effective = item["effective_config"]
    if set(effective) != set(base) or any(effective[k] != base[k] for k in base if k not in {"max_turns", "wall_clock_seconds"}):
        raise ValueError("Recovery may only reduce max_turns and wall_clock_seconds")
    if item.get("effective_config_sha256") != runner.digest(effective):
        raise ValueError("Effective per-question config SHA differs")
    remaining_calls = base["max_turns"] - debit["calls_charged"]
    remaining_seconds = base["wall_clock_seconds"] - debit["elapsed_seconds_charged"]
    if strategy == "restart_remaining_budget":
        if item.get("attempt_id") != 2 or item["query_id"] == 310:
            raise ValueError("Only the first additional attempt is permitted; query310 cannot be resampled")
        if (type(effective["max_turns"]) is not int or type(effective["wall_clock_seconds"]) is not int or
                effective["max_turns"] != remaining_calls or effective["wall_clock_seconds"] != remaining_seconds or
                remaining_calls <= 0 or remaining_seconds <= 0):
            raise ValueError("Restart must use exactly the positive remaining conservative budget")
    elif strategy == "infra_budget_exhausted":
        if remaining_calls > 0 and remaining_seconds > 0:
            raise ValueError("A positive remaining budget cannot be declared exhausted")
        if effective != base:
            raise ValueError("A non-executed exhausted outcome keeps its baseline config reference")
    elif strategy == "resume_checkpoint":
        if (effective != base or item.get("attempt_id") != 1 or
                item.get("exact_checkpoint_and_budget_verified") is not True or
                debit["possible_inflight_calls"] or debit["unknown_usage_attempts"]):
            raise ValueError("Same-checkpoint resume requires exact verified state/budget without unresolved calls")
        if not item.get("resume_files"):
            raise ValueError("A checkpoint resume must pin its copied original artifacts")
    else:
        if effective != base or item.get("attempt_id") != 1 or not item.get("observed_final_file"):
            raise ValueError("An observed natural final must retain its original attempt identity")
    return effective


def plan_stages(plan):
    return [{**s, "queries": [q for q in s["queries"] if not (s["arm"] == "auto_rerank" and q["query_id"] in REUSED_QIDS)]}
            for s in original.stages()]


def source_artifact_paths(arm, qid):
    prefix = f"runs/{original.DIRECTORY}/{arm}"
    return {"terminal": f"{prefix}/_per_query/qid_{qid}.json",
        "request_contract": f"{prefix}/request_contracts/qid_{qid}.json",
        "api_ledger": f"{prefix}/api_ledgers/qid_{qid}.jsonl",
        "trace": f"{prefix}/_traces/qid_{qid}/single.json",
        "checkpoint": f"{prefix}/_checkpoints/qid_{qid}.json",
        "api_diagnostic": f"{prefix}/_traces/qid_{qid}/single.api.jsonl"}


def validate_observed_final(source_root, item, plan):
    name = item["observed_final_file"]
    if name not in plan["source_pins"]:
        raise ValueError("Observed final must be pinned independently")
    row = runner.load(Path(source_root) / name)  # This is a public projection, never raw API diagnostics.
    if row.get("receipt_sha256") != runner.digest({k: v for k, v in row.items() if k != "receipt_sha256"}):
        raise ValueError("Observed-final projection receipt checksum differs")
    if (row.get("full_natural_terminal_reconstructed") is not False or
            row.get("private_message_or_reasoning_values_decoded") is not False or
            row.get("original_files_modified") is not False or
            row.get("new_model_requests") != 0 or row.get("new_tool_executions") != 0 or
            row.get("recorded_attempts") != item["debit"]["recorded_attempts"] or
            row.get("unknown_usage_attempts") != item["debit"]["unknown_usage_attempts"]):
        raise ValueError("Observed-final projection privacy/no-resampling/accounting attestations differ")
    expected = {"query_id": item["query_id"], "arm": "auto_rerank", "source_manifest_sha256": SOURCE_MANIFEST_SHA,
                "verified": True, "finish_reason": "stop", "tool_calls_count": 0, "category": "content"}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Observed-final public attestation failed")
    final = row.get("final_text")
    if not isinstance(final, str) or not final.strip() or row.get("final_sha256") != hashlib.sha256(final.encode()).hexdigest():
        raise ValueError("Observed public final bytes differ")
    if not isinstance(row.get("response_id"), str) or not row["response_id"]:
        raise ValueError("Observed final needs its original response ID")
    sources = row.get("source_files")
    if not isinstance(sources, dict) or not sources or any(plan["source_pins"].get(k) != v for k, v in sources.items()):
        raise ValueError("Observed-final audit must bind its immutable source evidence")
    for key in ("known_input_tokens", "known_output_tokens"):
        if row.get(key) != item["debit"][key]:
            raise ValueError("Observed final accounting differs from the original ledger")
    return row


def validate_plan(plan):
    if (plan.get("version") != PLAN_VERSION or plan.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA or
            plan.get("protocol_amendment") is not True or plan.get("first_additional_attempt_only") is not True or
            plan.get("selection_rule") != SELECTION or plan.get("reused_terminal_qids") != REUSED_QIDS):
        raise ValueError("Recovery scope/selection policy must be explicitly frozen")
    source_root = Path(plan["source_root"]).resolve()
    source = original.ContinuationSuite(source_root)
    manifest = source.verify_manifest(resources=False)
    if manifest["manifest_sha256"] != SOURCE_MANIFEST_SHA or manifest["config_sha256"] != plan.get("base_config_sha256"):
        raise ValueError("Recovery references a different original execution")
    pin_files(source_root, plan["source_pins"])
    required = {f"reports/{original.DIRECTORY}/run_manifest.json"}
    items = plan.get("recovery_questions")
    if not isinstance(items, list) or len(items) != 4 or {q.get("query_id") for q in items} != RECOVERY_QIDS:
        raise ValueError("Exactly the four interrupted questions must have an audited disposition")
    if question_plan(plan, 310)["strategy"] != "materialized_observed_final":
        raise ValueError("Query310 has an observed natural final and must not be sampled again")
    for qid in REUSED_QIDS:
        query = next(q for q in runner.shared.validate_queries() if q["query_id"] == qid)
        path = source.terminal_path("auto_rerank", qid)
        row = source.validate_terminal(runner.load(path), manifest, "auto_rerank", query)
        source.verify_execution_artifacts(manifest, "auto_rerank", query, row)
        required.update(v for k, v in source_artifact_paths("auto_rerank", qid).items() if k != "api_diagnostic")
    existing = {int(p.stem.removeprefix("qid_")) for p in (source.runs / "auto_rerank" / "_per_query").glob("qid_*.json")}
    if existing != set(REUSED_QIDS):
        raise ValueError("The original natural terminal set changed")
    for item in items:
        qid = item["query_id"]
        validate_budget(item, manifest["config"])
        if not item.get("evidence_files") or any(name not in plan["source_pins"] for name in item["evidence_files"]):
            raise ValueError("Every disposition must reference pinned audit evidence")
        paths = source_artifact_paths("auto_rerank", qid)
        required.update(paths[k] for k in ("request_contract", "api_ledger", "api_diagnostic"))
        if (source_root / paths["checkpoint"]).exists(): required.add(paths["checkpoint"])
        ledger = [json.loads(line) for line in (source_root / paths["api_ledger"]).read_text().splitlines() if line.strip()]
        successes = [r for r in ledger if r.get("status") == "ok"]
        contract_sha = manifest["contracts"]["auto_rerank"]["contract_sha256"]
        if any(r.get("requested_model") != runner.MODEL or r.get("contract_sha256") != contract_sha for r in ledger):
            raise ValueError("Preserved requests differ from the frozen model/tool contract")
        if any(r.get("reported_model") != runner.MODEL or
               any(type((r.get("usage") or {}).get(k)) is not int or r["usage"][k] < 0 for k in ("prompt_tokens", "completion_tokens"))
               for r in successes):
            raise ValueError("Prior successful request model/usage is invalid")
        known = {"recorded_attempts": len(ledger),
            "known_input_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in successes),
            "known_output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in successes)}
        if any(item["debit"].get(k) != v for k, v in known.items()):
            raise ValueError("Prior measured usage differs from its preserved ledger")
        unknown = sum(not isinstance(r.get("usage"), dict) for r in ledger)
        if item["debit"]["unknown_usage_attempts"] < unknown + item["debit"]["possible_inflight_calls"]:
            raise ValueError("Unknown prior usage cannot be silently converted to zero")
        if item["strategy"] == "materialized_observed_final":
            validate_observed_final(source_root, item, plan)
        if item["strategy"] == "resume_checkpoint":
            names = set(item["resume_files"])
            if not {paths[k] for k in ("checkpoint", "request_contract", "api_ledger")}.issubset(names):
                raise ValueError("Checkpoint resume needs checkpoint, request contract and ledger")
            if not names.issubset(set(paths.values()) - {paths["terminal"]}) or not names.issubset(plan["source_pins"]):
                raise ValueError("Unexpected or unpinned checkpoint import")
    if not required.issubset(plan["source_pins"]):
        raise ValueError("Original execution artifacts are not all pinned")
    return manifest


class RecoverySuite(runner.Suite):
    directory = DIRECTORY
    probe_arms = ("auto_rerank", "search_paging")

    def prepare(self, plan, *, service_pid):
        old = validate_plan(plan)
        if (self.reports / "run_manifest.json").exists() or self.runs.exists():
            raise RuntimeError("An existing recovery manifest/output cannot be rebound")
        current = runner.shared.resource_identity(old["config"])
        runner.same_static_resources(current, old["resource_identity"])
        epoch = runner.deployment.prepare_epoch(old["config"], current, pid=service_pid)
        payload = copy.deepcopy(old)
        payload.pop("manifest_sha256")
        payload.update(recovery_version=VERSION, directory=DIRECTORY,
            recovery_plan=copy.deepcopy(plan), recovery_plan_sha256=runner.digest(plan),
            source_continuation_manifest_sha256=SOURCE_MANIFEST_SHA,
            initial_service_epoch_sha256=epoch["epoch_sha256"],
            administrative_source_sha256=admin_hashes(), question_runs=100,
            new_scheduled_outcomes=97, reused_continuation_terminals=REUSED_QIDS,
            execution_plan={"version": VERSION, "stages": [{"name": s["name"], "arm": s["arm"],
                "query_ids": [q["query_id"] for q in s["queries"]]} for s in plan_stages(plan)],
                "selection_rule": SELECTION, "original_full_three_arm_suite_complete": False},
            protocol_amendment=True, original_single_attempt_policy_preserved=False,
            storage_admission=STORAGE_ADMISSION,
            scheduling="all automatic80 target dispositions, including3 reused and1 observed final, then paging20; global C4",
            terminal_policy="Original natural terminals immutable; one selected residual-budget restart per approved interrupted question; observed final never resampled")
        manifest = {**payload, "manifest_sha256": runner.digest(payload)}
        self.save_epoch(epoch)
        runner.atomic_json(self.reports / "run_manifest.json", manifest, immutable=True)
        return manifest

    def verify_manifest(self, *, resources=True):
        manifest = super().verify_manifest(resources=resources)
        if (manifest.get("recovery_version") != VERSION or manifest.get("directory") != DIRECTORY or
                manifest.get("administrative_source_sha256") != admin_hashes() or
                manifest.get("recovery_plan_sha256") != runner.digest(manifest["recovery_plan"])):
            raise RuntimeError("Recovery protocol/source/plan changed")
        old = validate_plan(manifest["recovery_plan"])
        if manifest["config"] != old["config"] or manifest["control_reference"] != old["control_reference"]:
            raise RuntimeError("Recovery base resource/control identity changed")
        expected = [{"name": s["name"], "arm": s["arm"], "query_ids": [q["query_id"] for q in s["queries"]]}
                    for s in plan_stages(manifest["recovery_plan"])]
        if manifest.get("storage_admission") != STORAGE_ADMISSION:
            raise RuntimeError("Frozen storage-admission policy changed")
        if manifest["execution_plan"]["stages"] != expected:
            raise RuntimeError("Recovery scheduling scope changed")
        self.assert_run_scope()
        return manifest

    def effective_manifest(self, manifest, arm, qid):
        view = dict(manifest)
        item = question_plan(manifest["recovery_plan"], qid) if arm == "auto_rerank" else None
        if item:
            view["config"] = item["effective_config"]
            view["config_sha256"] = item["effective_config_sha256"]
        return view

    def terminal_path(self, arm, qid):
        allowed = {q["query_id"] for q in original.selected_queries(arm)} - (set(REUSED_QIDS) if arm == "auto_rerank" else set())
        if type(qid) is not int or qid not in allowed:
            raise ValueError("Only the unresolved authorized continuation questions are new outcomes")
        if arm == "auto_rerank" and qid == 310:
            raise ValueError("Observed final310 has no fabricated natural terminal artifact")
        return super().terminal_path(arm, qid)

    def assert_run_scope(self):
        proxy = original.ContinuationSuite(self.root)
        proxy.runs = self.runs
        proxy.assert_run_scope()
        for arm in self.probe_arms:
            forbidden = REUSED_QIDS + [310] if arm == "auto_rerank" else []
            for d in original.ALLOWED_RUN_DIRECTORIES:
                for qid in forbidden:
                    if any((self.runs / arm / d).glob(f"qid_{qid}*")):
                        raise RuntimeError("Reused/observed outcome must not have a new solver workspace")

    def outcome_path(self, arm, qid):
        return self.reports / "outcomes" / arm / f"qid_{qid}.json"

    def _save_outcome(self, payload):
        path = self.outcome_path(payload["arm"], payload["query_id"])
        value = {**payload, "outcome_sha256": runner.digest(payload)}
        if path.exists():
            if runner.load(path) != value: raise RuntimeError("Immutable selected recovery outcome changed")
        else: runner.atomic_json(path, value, immutable=True)
        return value

    def accounting(self, row, item):
        current = {"input_tokens": row.get("input_tokens", 0), "output_tokens": row.get("output_tokens", 0),
            "recorded_attempts": row.get("api_ledger", {}).get("attempts", 0),
            "unknown_usage_attempts": row.get("api_ledger", {}).get("attempts_without_usage", 0)}
        prior = item["debit"] if item else {"known_input_tokens": 0, "known_output_tokens": 0,
            "recorded_attempts": 0, "unknown_usage_attempts": 0, "calls_charged": 0, "elapsed_seconds_charged": 0}
        includes_prior = bool(item and item["strategy"] == "resume_checkpoint")
        return {"previous_attempt": prior, "selected_attempt": current,
            "known_total_input_tokens": current["input_tokens"] + (0 if includes_prior else prior["known_input_tokens"]),
            "known_total_output_tokens": current["output_tokens"] + (0 if includes_prior else prior["known_output_tokens"]),
            "unknown_usage_attempts": current["unknown_usage_attempts"] + (0 if includes_prior else prior["unknown_usage_attempts"]),
            "checkpoint_totals_already_include_prior": includes_prior,
            "total_token_cap": None, "token_policy": "report every known usage; unknown remains unknown; no invented total-token cap"}

    def validate_outcome(self, manifest, arm, query):
        qid = query["query_id"]
        outcome = runner.load(self.outcome_path(arm, qid))
        if outcome.get("outcome_sha256") != runner.digest({k: v for k, v in outcome.items() if k != "outcome_sha256"}):
            raise RuntimeError("Selected outcome checksum changed")
        if any(outcome.get(k) != v for k, v in {"arm": arm, "query_id": qid, "manifest_sha256": manifest["manifest_sha256"]}.items()):
            raise RuntimeError("Selected outcome belongs to another recovery")
        item = question_plan(manifest["recovery_plan"], qid) if arm == "auto_rerank" else None
        if outcome["kind"] == "natural_terminal":
            if item and item["strategy"] not in {"restart_remaining_budget", "resume_checkpoint"}:
                raise RuntimeError("A non-executed disposition became a fabricated terminal")
            view = self.effective_manifest(manifest, arm, qid)
            path = self.terminal_path(arm, qid)
            row = self.validate_terminal(runner.load(path), view, arm, query)
            self.verify_execution_artifacts(view, arm, query, row)
            if outcome["terminal_sha256"] != runner.file_sha(path) or outcome["final_sha256"] != row["final_sha256"] or outcome["accounting"] != self.accounting(row, item):
                raise RuntimeError("Outcome final/accounting no longer matches its selected terminal")
        elif outcome["kind"] == "materialized_observed_final":
            if not item or item["strategy"] != outcome["kind"]: raise RuntimeError("Unplanned observed final")
            observed = validate_observed_final(manifest["recovery_plan"]["source_root"], item, manifest["recovery_plan"])
            if outcome["observation"] != observed: raise RuntimeError("Observed-final attestation changed")
        elif outcome["kind"] == "infra_budget_exhausted":
            if not item or item["strategy"] != outcome["kind"]: raise RuntimeError("Unplanned infrastructure exhaustion")
            if outcome.get("final_text") is not None or outcome.get("natural_terminal") is not False:
                raise RuntimeError("Infrastructure exhaustion cannot be reported as a natural timeout")
        else: raise RuntimeError("Unknown selected recovery outcome")
        return outcome

    def _import_checkpoint(self, manifest, item):
        receipt_path = self.reports / "checkpoint_imports" / f"qid_{item['query_id']}.json"
        if receipt_path.exists():
            receipt = runner.load(receipt_path)
            if receipt.get("manifest_sha256") != manifest["manifest_sha256"]: raise RuntimeError("Checkpoint import identity changed")
            return
        source_root = Path(manifest["recovery_plan"]["source_root"])
        copied = {}
        old_prefix = f"runs/{original.DIRECTORY}/auto_rerank/"
        for name in item["resume_files"]:
            if not name.startswith(old_prefix): raise RuntimeError("Unexpected checkpoint import path")
            target = self.runs / "auto_rerank" / name.removeprefix(old_prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = manifest["recovery_plan"]["source_pins"][name]
            if target.exists():
                if original._pin(target) != expected: raise RuntimeError("Partial checkpoint import changed")
            else:
                with target.open("xb") as stream: stream.write((source_root / name).read_bytes())
                if original._pin(target) != expected: raise RuntimeError("Copied checkpoint bytes differ")
            copied[name] = expected
        runner.atomic_json(receipt_path, {"manifest_sha256": manifest["manifest_sha256"], "copied": copied}, immutable=True)

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        if manifest["administrative_source_sha256"] != admin_hashes(): raise RuntimeError("Recovery administrative source drift")
        qid = query["query_id"]
        if self.outcome_path(arm, qid).exists():
            return self.validate_outcome(manifest, arm, query)
        item = question_plan(manifest["recovery_plan"], qid) if arm == "auto_rerank" else None
        common = {"version": VERSION, "arm": arm, "query_id": qid, "manifest_sha256": manifest["manifest_sha256"],
            "selection_rule": SELECTION, "recovery_strategy": item["strategy"] if item else "original_first_attempt"}
        if item and item["strategy"] == "materialized_observed_final":
            observed = validate_observed_final(manifest["recovery_plan"]["source_root"], item, manifest["recovery_plan"])
            return self._save_outcome({**common, "kind": item["strategy"], "observation": observed,
                "final_sha256": observed["final_sha256"], "natural_terminal": False,
                "observed_natural_response": True, "new_model_requests": 0, "complete_trace_checkpoint": False})
        if item and item["strategy"] == "infra_budget_exhausted":
            return self._save_outcome({**common, "kind": item["strategy"], "debit": item["debit"],
                "final_text": None, "natural_terminal": False, "new_model_requests": 0})
        imported = bool(item and item["strategy"] == "resume_checkpoint")
        if imported: self._import_checkpoint(manifest, item)
        view = self.effective_manifest(manifest, arm, qid)
        runner.Suite.run_one(self, view, arm, query, resume_infrastructure=resume_infrastructure or imported)
        path = self.terminal_path(arm, qid); row = runner.load(path)
        if manifest["administrative_source_sha256"] != admin_hashes(): raise RuntimeError("Recovery administrative source drift")
        return self._save_outcome({**common, "kind": "natural_terminal", "natural_terminal": True,
            "terminal_sha256": runner.file_sha(path), "final_sha256": row["final_sha256"] ,
            "effective_config_sha256": view["config_sha256"], "same_common_configuration": view["config"] == manifest["config"],
            "original_single_attempt_policy_preserved": item is None or imported,
            "accounting": self.accounting(row, item)})

    def _stage_ready(self, manifest):
        self.verify_manifest()
        ready = runner.load(self.reports / "preflight_live.json")
        if ready.get("passed") is not True or ready.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("Recovery requires a fresh live preflight")
        if set(ready.get("tools_and_native_calls", {})) != set(self.probe_arms) or any(r.get("passed") is not True for r in ready["tools_and_native_calls"].values()):
            raise RuntimeError("Both unchanged tool surfaces must pass")
        service = self.verify_service(manifest)
        if ready["service"]["epoch_sha256"] != service["epoch_sha256"]: raise RuntimeError("Service epoch changed")
        os.environ.update(manifest["config"]["runtime_environment"])

    def _state(self, identity, phase, **extra):
        value = {"version": VERSION, "worker_identity": identity, "phase": phase, "checked_unix": time.time(), **extra}
        runner.atomic_json(self.reports / "phase_state.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def status(self):
        if not (self.reports / "run_manifest.json").exists(): return {"version": VERSION, "prepared": False}
        manifest = self.verify_manifest(resources=False)
        groups = {}; kinds = {"natural_terminal": 0, "materialized_observed_final": 0, "infra_budget_exhausted": 0}
        for stage in plan_stages(manifest["recovery_plan"]):
            resolved, pending, partial = [], [], []
            for query in stage["queries"]:
                qid, arm = query["query_id"], stage["arm"]
                if self.outcome_path(arm, qid).exists():
                    outcome = self.validate_outcome(manifest, arm, query); kinds[outcome["kind"]] += 1; resolved.append(qid)
                elif any((self.runs / arm / d / f"qid_{qid}{ext}").exists() for d, ext in (("_per_query", ".json"), ("_checkpoints", ".json"), ("request_contracts", ".json"), ("api_ledgers", ".jsonl"), ("_working", ""))): partial.append(qid)
                else: pending.append(qid)
            groups[stage["arm"]] = {"resolved": resolved, "partial_or_active": partial, "pending": pending}
        return {"version": VERSION, "prepared": True, "arms": groups, "new_outcome_counts": kinds,
            "resolved_new_outcomes": sum(kinds.values()), "reused_natural_terminals": 3,
            "validated_natural_terminals": 3 + kinds["natural_terminal"],
            "target_coverage": 3 + sum(kinds.values()), "target_count": 100,
            "original_full_three_arm_suite_complete": False}

    def storage_status(self):
        stats = os.statvfs(self.root)
        free_bytes = stats.f_bavail * stats.f_frsize
        free_inodes = stats.f_favail
        return {"ready": free_bytes >= STORAGE_ADMISSION["min_free_bytes"] and free_inodes >= STORAGE_ADMISSION["min_free_inodes"],
                "free_bytes": free_bytes, "free_inodes": free_inodes, **STORAGE_ADMISSION}

    def run(self, *, resume_infrastructure=False, worker_identity=None):
        identity = worker_identity or process_identity()
        header = runner.load(self.reports / "run_manifest.json")
        with ExitStack() as stack:
            for path, mode in ((self.reports / "worker.lock", "w"),
                    (Path(header["recovery_plan"]["source_root"]) / f"reports/{original.DIRECTORY}/worker.lock", "rb"),
                    (Path(header["reused_auto20_reference"]["root"]) / "reports/strict_auto20_v1/worker.lock", "rb")):
                handle = stack.enter_context(path.open(mode)); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.verify_manifest()
            try:
                receipts = []
                for stage in plan_stages(manifest["recovery_plan"]):
                    self._stage_ready(manifest)
                    self._state(identity, "running_" + stage["name"], manifest_sha256=manifest["manifest_sha256"])
                    jobs = deque(stage["queries"]); checked = 0
                    with ThreadPoolExecutor(max_workers=manifest["config"]["concurrency"]) as pool:
                        active = {}
                        while jobs or active:
                            while jobs and len(active) < manifest["config"]["concurrency"]:
                                query = jobs[0]
                                item = question_plan(manifest["recovery_plan"], query["query_id"]) if stage["arm"] == "auto_rerank" else None
                                no_solver = (self.outcome_path(stage["arm"], query["query_id"]).exists() or
                                             (item and item["strategy"] in {"materialized_observed_final", "infra_budget_exhausted"}))
                                capacity = self.storage_status()
                                if not no_solver and not capacity["ready"]:
                                    self._state(identity, "waiting_storage", stage=stage["name"], active_questions=len(active),
                                        pending_questions=len(jobs), storage=capacity, manifest_sha256=manifest["manifest_sha256"])
                                    break
                                jobs.popleft()
                                future = pool.submit(self.run_one, manifest, stage["arm"], query, resume_infrastructure=resume_infrastructure)
                                active[future] = query
                            if active:
                                future = next(as_completed(active)); active.pop(future); future.result(); checked += 1
                                self._state(identity, "running_" + stage["name"], stage_resolved=checked, stage_total=len(stage["queries"]), manifest_sha256=manifest["manifest_sha256"])
                            elif jobs:
                                time.sleep(STORAGE_ADMISSION["poll_seconds"])
                    self.verify_manifest()
                    outcomes = [self.validate_outcome(manifest, stage["arm"], q) for q in stage["queries"]]
                    receipt = {"version": VERSION, "passed": True, "stage": stage["name"], "manifest_sha256": manifest["manifest_sha256"],
                        "selected_outcomes": [{"query_id": o["query_id"], "kind": o["kind"], "outcome_sha256": o["outcome_sha256"]} for o in outcomes]}
                    path = self.reports / (stage["name"] + "_completion.json")
                    if path.exists():
                        if runner.load(path) != receipt: raise RuntimeError("Stage completion changed")
                    else: runner.atomic_json(path, receipt, immutable=True)
                    receipts.append(receipt)
                status = self.status()
                if status["target_coverage"] != 100: raise RuntimeError("All100 original targets need an explicit selected disposition")
                receipt = {"version": VERSION, "passed": True, "manifest_sha256": manifest["manifest_sha256"],
                    "stages": receipts, **{k: status[k] for k in ("new_outcome_counts", "validated_natural_terminals", "target_coverage", "target_count")},
                    "control_model_requests": 0, "old_auto20_model_requests": 0, "observed_final310_new_model_requests": 0,
                    "original_full_three_arm_suite_complete": False}
                path = self.reports / "recovery_completion.json"
                if path.exists():
                    if runner.load(path) != receipt: raise RuntimeError("Recovery completion changed")
                else: runner.atomic_json(path, receipt, immutable=True)
                self._state(identity, "complete", manifest_sha256=manifest["manifest_sha256"], receipt_sha256=runner.file_sha(path))
                return receipt
            except BaseException as exc:
                self._state(identity, "interrupted", error_type=type(exc).__name__, manifest_sha256=manifest["manifest_sha256"])
                raise

    def pipeline(self, plan, *, service_pid, resume_infrastructure=False):
        identity = process_identity()
        try:
            if not (self.reports / "run_manifest.json").exists():
                self._state(identity, "prepare"); self.prepare(plan, service_pid=service_pid)
            self._state(identity, "live_preflight")
            self.live_preflight(resume_infrastructure=resume_infrastructure, service_pid=service_pid if resume_infrastructure else None)
            return self.run(resume_infrastructure=resume_infrastructure, worker_identity=identity)
        except BaseException as exc:
            self._state(identity, "interrupted", error_type=type(exc).__name__); raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", default=str(ROOT))
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("offline-preflight", "prepare", "pipeline"):
        a = sub.add_parser(name); a.add_argument("--plan", required=True)
        if name in {"prepare", "pipeline"}: a.add_argument("--service-pid", type=int, required=True)
        if name == "pipeline": a.add_argument("--resume-infrastructure", action="store_true")
    a = sub.add_parser("live-preflight"); a.add_argument("--resume-infrastructure", action="store_true"); a.add_argument("--service-pid", type=int)
    a = sub.add_parser("run"); a.add_argument("--resume-infrastructure", action="store_true")
    sub.add_parser("status"); args = p.parse_args(argv); suite = RecoverySuite(args.root)
    if args.command == "offline-preflight":
        plan = runner.load(args.plan); old = validate_plan(plan)
        result = {"passed": True, "model_requests": 0, "plan_sha256": runner.digest(plan), "source_manifest_sha256": old["manifest_sha256"], "administrative_source_sha256": admin_hashes()}
    elif args.command == "prepare": result = suite.prepare(runner.load(args.plan), service_pid=args.service_pid)
    elif args.command == "pipeline": result = suite.pipeline(runner.load(args.plan), service_pid=args.service_pid, resume_infrastructure=args.resume_infrastructure)
    elif args.command == "live-preflight": result = suite.live_preflight(service_pid=args.service_pid, resume_infrastructure=args.resume_infrastructure)
    elif args.command == "run": result = suite.run(resume_infrastructure=args.resume_infrastructure)
    else: result = suite.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
