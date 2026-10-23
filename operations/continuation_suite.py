#!/usr/bin/env python3
"""Administrative continuation: frozen automatic-rerank remaining80, then paging20.

All solver/tool implementations remain the original 135 frozen source files.
The completed automatic20 is an immutable external reference, never a new job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_ablations import runner
from strict_ablations.auto20 import Auto20Suite
from strict_ablations.prompt import ARMS, CONTROL, VERSION as PROMPT_VERSION, build_contract

VERSION = "strict-continuation-auto80-paging20-v1"
DIRECTORY = "strict_continue_v1"
AUTO20_MANIFEST_SHA = "67b4ff0f8f1a9d25583ad77ab150f919dec078aa34c824126648d88433ee1384"
AUTO20_MANIFEST_FILE_SHA = "9bb748ffa0fca71ade546d47d8eb2c602f2d55e3398eaa7b9ec6c96cec568fc2"
AUTO20_COMPLETION_SHA = "4e664cd2f45abdfb1062eeb6acbd125e59a410a319c190cd9a9a7197886c0516"
ADMIN_FILES = ("operations/continuation_suite.py",)
ALLOWED_RUN_DIRECTORIES = {"_per_query", "_traces", "_checkpoints", "request_contracts",
    "api_ledgers", "_infrastructure", "_working", "_rerank_runtime"}


def stages():
    rows = runner.shared.validate_queries()
    if len(rows) != 100 or len({r["query_id"] for r in rows}) != 100:
        raise RuntimeError("Continuation requires the original unique fixed100 in frozen order")
    return [{"name": "auto80", "arm": "auto_rerank", "queries": rows[20:]},
            {"name": "paging20", "arm": "search_paging", "queries": rows[:20]}]


def selected_queries(arm):
    for stage in stages():
        if stage["arm"] == arm:
            return stage["queries"]
    raise ValueError("Only automatic remaining80 and snippet paging first20 are authorized")


def admin_hashes():
    return {name: runner.file_sha(ROOT / name) for name in ADMIN_FILES}


def process_identity():
    identity = {"pid": os.getpid(), "started_unix": time.time(),
                "driver_sha256": runner.file_sha(Path(__file__))}
    stat = Path("/proc/self/stat")
    identity["proc_start_ticks"] = int(stat.read_text().rsplit(") ", 1)[1].split()[19]) if stat.exists() else None
    boot = Path("/proc/sys/kernel/random/boot_id")
    identity["boot_id"] = boot.read_text().strip() if boot.exists() else None
    return identity



# Exact temporary names used by the unchanged writers. These are neither
# terminals nor evidence of a committed checkpoint/request. They may coexist
# briefly with their target or remain after an interrupted atomic write.
_ATOMIC_TEMP_PATTERNS = {
    "_per_query": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}",
    "_infrastructure": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}",
    "_checkpoints": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}\.tmp",
    "request_contracts": r"qid_([1-9][0-9]*)\.json\.tmp",
}


def atomic_temporary_qid(directory, name):
    pattern = _ATOMIC_TEMP_PATTERNS.get(directory)
    match = re.fullmatch(pattern, name) if pattern is not None else None
    return int(match[1]) if match else None


def _pin(path):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Referenced artifact must be a regular unaliased file: " + str(path))
    return {"sha256": runner.file_sha(path), "bytes": path.stat().st_size}


def verify_reused_auto20(root, *, common_config, control_reference):
    """Verify bytes/identities and metadata; never emit private trace content."""
    root = Path(root).resolve()
    old = Auto20Suite(root)
    manifest_path = old.reports / "run_manifest.json"
    completion_path = old.reports / "pilot20_completion.json"
    if (_pin(manifest_path)["sha256"] != AUTO20_MANIFEST_FILE_SHA or
            _pin(completion_path)["sha256"] != AUTO20_COMPLETION_SHA):
        raise RuntimeError("The completed automatic20 manifest/completion bytes changed")
    manifest = runner.load(manifest_path)
    if (manifest.get("manifest_sha256") != AUTO20_MANIFEST_SHA or
            runner.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != AUTO20_MANIFEST_SHA):
        raise RuntimeError("Unexpected completed automatic20 manifest")
    if (manifest["source_sha256"] != runner.source_hashes() or len(manifest["source_sha256"]) != 135 or
            manifest["config"] != common_config or manifest["config_sha256"] != runner.digest(common_config) or
            manifest["control_reference"] != control_reference or
            manifest["contracts"] != {arm: build_contract(arm) for arm in ARMS}):
        raise RuntimeError("Automatic20 source/config/control/contracts differ from continuation")
    rows = runner.shared.validate_queries()
    qids = [r["query_id"] for r in rows[:20]]
    if (manifest.get("cohort_sha256") != runner.shared.DATASET_SHA or
            manifest.get("query_ids") != [r["query_id"] for r in rows] or
            manifest.get("selected_query_ids") != qids):
        raise RuntimeError("Reused automatic20 is not the first20 of the same fixed100")
    completion = runner.load(completion_path)
    if (completion.get("passed") is not True or completion.get("manifest_sha256") != AUTO20_MANIFEST_SHA or
            completion.get("selected_query_ids") != qids or completion.get("terminal_rows_checked") != 20 or
            any(completion.get(k) != 0 for k in ("natural_error_reruns", "control_model_requests", "other_arm_model_requests"))):
        raise RuntimeError("Completed automatic20 receipt failed")
    if set((old.runs / "auto_rerank" / "_per_query").glob("*.json")) != {
            old.terminal_path("auto_rerank", qid) for qid in qids}:
        raise RuntimeError("Reused automatic20 terminal set is not exactly fixed first20")
    files = {str(p.relative_to(root)): _pin(p) for p in (manifest_path, completion_path)}
    for query in rows[:20]:
        qid = query["query_id"]
        terminal_path = old.terminal_path("auto_rerank", qid)
        terminal = old.validate_terminal(runner.load(terminal_path), manifest, "auto_rerank", query)
        old.verify_execution_artifacts(manifest, "auto_rerank", query, terminal)
        paths = [terminal_path,
            old.runs / "auto_rerank" / "request_contracts" / f"qid_{qid}.json",
            old.runs / "auto_rerank" / "api_ledgers" / f"qid_{qid}.jsonl",
            old.runs / "auto_rerank" / "_traces" / f"qid_{qid}" / "single.json",
            old.runs / "auto_rerank" / "_checkpoints" / f"qid_{qid}.json"]
        files.update({str(p.relative_to(root)): _pin(p) for p in paths})
    return {"root": str(root), "manifest_sha256": AUTO20_MANIFEST_SHA,
        "manifest_file_sha256": AUTO20_MANIFEST_FILE_SHA, "completion_sha256": AUTO20_COMPLETION_SHA,
        "source_sha256": manifest["source_sha256"], "selected_query_ids": qids,
        "terminal_rows": 20, "files": files, "new_model_requests": 0}


class ContinuationSuite(runner.Suite):
    directory = DIRECTORY
    probe_arms = ("auto_rerank", "search_paging")

    def execution_scope(self, rows):
        expected = stages()
        if rows != runner.shared.validate_queries():
            raise RuntimeError("Continuation cohort differs from the frozen fixed100")
        return {"question_runs": 100, "reused_control_questions": 100,
            "reused_auto20_questions": 20,
            "selected_query_ids": {s["arm"]: [q["query_id"] for q in s["queries"]] for s in expected},
            "scheduling": "all automatic remaining80 natural terminals, then snippet paging first20; shared global C4",
            "execution_plan": {"version": VERSION, "question_runs": 100,
                "stages": [{"name": s["name"], "arm": s["arm"],
                    "query_ids": [q["query_id"] for q in s["queries"]]} for s in expected],
                "control_policy": "reuse frozen B100 public finals; no control model requests",
                "automatic_first20_policy": "immutable external reference; no rescheduling",
                "full_three_arm_suite_complete": False},
            "administrative_source_sha256": admin_hashes()}

    def template(self, control_root, reuse_auto20_root):
        return {**super().template(control_root),
                "reuse_auto20_root": str(Path(reuse_auto20_root).resolve())}

    def offline_preflight(self, config=None, *, control_root=None, reuse_auto20_root=None):
        base = None
        if config is not None:
            if set(config) != {"version", "control_root", "common_config", "reuse_auto20_root"}:
                raise ValueError("Continuation config must explicitly bind control and completed automatic20")
            base = {k: v for k, v in config.items() if k != "reuse_auto20_root"}
            reuse_auto20_root = config["reuse_auto20_root"]
        report = super().offline_preflight(base, control_root=control_root)
        reference = None
        if reuse_auto20_root is not None:
            if base is None:
                if control_root is None:
                    raise ValueError("Verifying reused automatic20 requires a control root")
                base = super().template(control_root)
            if base["common_config"]["concurrency"] != 4:
                raise RuntimeError("Continuation must preserve question-level C4")
            bundle = runner.verify_control_bundle(base["control_root"], common_config=base["common_config"])
            reference = verify_reused_auto20(reuse_auto20_root, common_config=base["common_config"],
                control_reference=runner.control_reference(bundle))
        return {**report, "continuation_version": VERSION,
                "arms": list(self.probe_arms), "new_question_runs": 100,
                **self.execution_scope(runner.shared.validate_queries()),
                "reused_auto20_reference": reference}

    def prepare(self, config, *, service_pid=None):
        report = self.offline_preflight(config)
        if (self.reports / "run_manifest.json").exists() or self.runs.exists():
            raise RuntimeError("Existing continuation outputs/manifest cannot be rebound")
        common = config["common_config"]
        bundle = runner.verify_control_bundle(config["control_root"], common_config=common)
        control_manifest = runner.load(Path(config["control_root"]) / "reports/strict_qwen35/run_manifest.json")
        if runner.shared.source_hashes() != control_manifest["source_sha256"]:
            raise RuntimeError("Frozen shared source differs from B100")
        current = runner.shared.resource_identity(common)
        runner.same_static_resources(current, control_manifest["resource_identity"])
        epoch = runner.deployment.prepare_epoch(common, current, pid=service_pid)
        rows = runner.shared.validate_queries()
        payload = {"version": runner.SUITE_VERSION, "prompt_version": PROMPT_VERSION,
            "config": copy.deepcopy(common), "config_sha256": runner.digest(common),
            "control_root": str(Path(config["control_root"]).resolve()),
            "control_reference": runner.control_reference(bundle),
            "control_model_inputs": build_contract(CONTROL),
            "contracts": {arm: build_contract(arm) for arm in ARMS},
            "cohort_sha256": runner.shared.DATASET_SHA, "query_ids": [r["query_id"] for r in rows],
            "source_sha256": runner.source_hashes(), "resource_identity": current,
            "initial_service_epoch_sha256": epoch["epoch_sha256"],
            **self.execution_scope(rows), "reused_auto20_reference": report["reused_auto20_reference"],
            "analysis_contract": self.analysis_contract(),
            "terminal_policy": "immutable; all natural failures included; same-checkpoint infrastructure resume only",
            "causal_scope": "historical B100 and automatic20 references; matched static resources/config; new process epoch",
            "auto_rerank_scope": "coupled retrieval/rerank policy; not an isolated neural-ranking ablation"}
        manifest = {**payload, "manifest_sha256": runner.digest(payload)}
        self.save_epoch(epoch)
        runner.atomic_json(self.reports / "run_manifest.json", manifest, immutable=True)
        return manifest

    def verify_manifest(self, *, resources=True):
        manifest = super().verify_manifest(resources=resources)
        expected = self.execution_scope(runner.shared.validate_queries())
        if any(runner.digest(manifest.get(k)) != runner.digest(v) for k, v in expected.items()):
            raise RuntimeError("Continuation execution scope/administrative source changed")
        self.verify_reuse_reference(manifest)
        self.assert_run_scope()
        return manifest

    def verify_reuse_reference(self, manifest):
        reference = manifest.get("reused_auto20_reference", {})
        observed = verify_reused_auto20(reference.get("root", ""), common_config=manifest["config"],
                                      control_reference=manifest["control_reference"])
        if reference != observed:
            raise RuntimeError("The immutable automatic20 reference changed")
        return observed

    def assert_run_scope(self):
        if not self.runs.exists():
            return
        allowed = {s["arm"]: {q["query_id"] for q in s["queries"]} for s in stages()}
        for arm_path in self.runs.iterdir():
            if arm_path.is_symlink() or not arm_path.is_dir() or arm_path.name not in allowed:
                raise RuntimeError("Forbidden arm/file exists in continuation runs")
            for directory in arm_path.iterdir():
                if directory.is_symlink() or not directory.is_dir() or directory.name not in ALLOWED_RUN_DIRECTORIES:
                    raise RuntimeError("Unexpected continuation artifact directory")
                for path in directory.iterdir():
                    temporary_qid = atomic_temporary_qid(directory.name, path.name)
                    if temporary_qid is not None:
                        # A concurrent replace may remove this temporary between
                        # iterdir and stat. A still-existing entry must be regular.
                        if (temporary_qid not in allowed[arm_path.name] or path.is_symlink() or
                                (path.exists() and not path.is_file())):
                            raise RuntimeError("Unsafe or unauthorized atomic temporary artifact")
                        continue
                    match = re.fullmatch(r"qid_([1-9][0-9]*)(?:\.jsonl?)?", path.name)
                    if path.is_symlink() or match is None or int(match[1]) not in allowed[arm_path.name]:
                        raise RuntimeError("An artifact is outside the authorized arm/query scope")

    def terminal_path(self, arm, qid):
        if type(qid) is not int or qid not in {q["query_id"] for q in selected_queries(arm)}:
            raise ValueError("Only automatic remaining80 and snippet paging first20 may be scheduled")
        return super().terminal_path(arm, qid)

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        if manifest.get("administrative_source_sha256") != admin_hashes():
            raise RuntimeError("Administrative source drifted during continuation")
        self.terminal_path(arm, query["query_id"])
        result = super().run_one(manifest, arm, query, resume_infrastructure=resume_infrastructure)
        if manifest.get("administrative_source_sha256") != admin_hashes():
            raise RuntimeError("Administrative source drifted while a query ran")
        return result

    def pilot_gate(self, manifest):
        raise RuntimeError("The original three-arm pilot is outside this continuation")

    def _stage_ready(self, manifest):
        checked = self.verify_manifest()
        if checked["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("Continuation manifest changed between stages")
        ready = runner.load(self.reports / "preflight_live.json")
        if ready.get("passed") is not True or ready.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("Continuation requires its own successful live preflight")
        probes = ready.get("tools_and_native_calls", {})
        if set(probes) != set(self.probe_arms) or any(p.get("passed") is not True for p in probes.values()):
            raise RuntimeError("Both automatic and paging probe surfaces must pass")
        os.environ.update(manifest["config"]["runtime_environment"])
        service = self.verify_service(manifest)
        if ready.get("service", {}).get("epoch_sha256") != service["epoch_sha256"]:
            raise RuntimeError("The current process epoch requires its own live preflight")
        return service

    def _terminal_bindings(self, manifest, stage):
        bindings = []
        for query in stage["queries"]:
            path = self.terminal_path(stage["arm"], query["query_id"])
            row = self.validate_terminal(runner.load(path), manifest, stage["arm"], query)
            self.verify_execution_artifacts(manifest, stage["arm"], query, row)
            bindings.append({"arm": stage["arm"], "query_id": query["query_id"],
                "row_sha256": row["row_sha256"], "file_sha256": runner.file_sha(path),
                "execution_artifacts": row["execution_artifacts"]})
        return bindings

    def _stage_receipt(self, manifest, stage):
        bindings = self._terminal_bindings(manifest, stage)
        path = self.reports / (stage["name"] + "_completion.json")
        payload = {"version": VERSION, "passed": True, "stage": stage["name"],
            "arm": stage["arm"], "manifest_sha256": manifest["manifest_sha256"],
            "question_runs": len(bindings), "terminal_rows_checked": len(bindings),
            "terminal_rows": bindings, "natural_error_reruns": 0,
            "control_model_requests": 0, "other_arm_model_requests": 0,
            "reused_auto20_model_requests": 0}
        if path.exists():
            old = runner.load(path)
            if any(old.get(k) != v for k, v in payload.items()):
                raise RuntimeError("An immutable stage completion receipt changed")
            return old
        receipt = {**payload, "finished_unix": time.time()}
        runner.atomic_json(path, receipt, immutable=True)
        return receipt

    def _state(self, identity, phase, **extra):
        state = {"version": VERSION, "worker_identity": identity, "phase": phase,
                 "checked_unix": time.time(), **extra}
        runner.atomic_json(self.reports / "phase_state.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def run(self, stage="all", *, resume_infrastructure=False, worker_identity=None):
        if stage != "all":
            raise ValueError("Continuation always enforces automatic80 then paging20")
        self.reports.mkdir(parents=True, exist_ok=True)
        identity = worker_identity or process_identity()
        with ExitStack() as stack:
            lock = stack.enter_context((self.reports / "worker.lock").open("w"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Open read-only: coordinate with the old exclusive worker lock without
            # modifying any byte of its completed artifacts or lock file.
            header = runner.load(self.reports / "run_manifest.json")
            reuse_root = Path(header["reused_auto20_reference"]["root"])
            old_lock = stack.enter_context((reuse_root / "reports/strict_auto20_v1/worker.lock").open("rb"))
            fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.verify_manifest()
            receipts = []
            try:
                for stage in stages():
                    self._stage_ready(manifest)
                    self._state(identity, "running_" + stage["name"], manifest_sha256=manifest["manifest_sha256"])
                    jobs = iter(stage["queries"])
                    completed = 0
                    with ThreadPoolExecutor(max_workers=manifest["config"]["concurrency"]) as pool:
                        active = {}
                        for _ in range(manifest["config"]["concurrency"]):
                            query = next(jobs, None)
                            if query is not None:
                                active[pool.submit(self.run_one, manifest, stage["arm"], query,
                                    resume_infrastructure=resume_infrastructure)] = query["query_id"]
                        while active:
                            future = next(as_completed(active)); active.pop(future)
                            future.result(); completed += 1
                            self._state(identity, "running_" + stage["name"],
                                manifest_sha256=manifest["manifest_sha256"],
                                stage_terminals_checked=completed, stage_total=len(stage["queries"]))
                            query = next(jobs, None)
                            if query is not None:
                                active[pool.submit(self.run_one, manifest, stage["arm"], query,
                                    resume_infrastructure=resume_infrastructure)] = query["query_id"]
                    self.verify_manifest()
                    receipts.append(self._stage_receipt(manifest, stage))
                status = self.status()
                if status["completed_new_runs"] != 100:
                    raise RuntimeError("Continuation completion requires all100 new natural terminals")
                payload = {"version": VERSION, "passed": True, "stage": "all",
                    "manifest_sha256": manifest["manifest_sha256"], "question_runs": 100,
                    "terminal_rows_checked": 100, "stages": receipts,
                    "control_model_requests": 0, "other_arm_model_requests": 0,
                    "reused_auto20_model_requests": 0, "natural_error_reruns": 0,
                    "full_three_arm_suite_complete": False}
                path = self.reports / "continuation_completion.json"
                if path.exists():
                    receipt = runner.load(path)
                    if any(receipt.get(k) != v for k, v in payload.items()):
                        raise RuntimeError("Immutable final completion changed")
                else:
                    receipt = {**payload, "finished_unix": time.time()}
                    runner.atomic_json(path, receipt, immutable=True)
                self._state(identity, "complete", manifest_sha256=manifest["manifest_sha256"],
                            receipt_sha256=runner.file_sha(path), terminal_rows_checked=100)
                return receipt
            except BaseException as exc:
                self._state(identity, "interrupted", manifest_sha256=manifest["manifest_sha256"],
                            error_type=type(exc).__name__)
                raise

    def status(self):
        if not (self.reports / "run_manifest.json").exists():
            return {"version": VERSION, "prepared": False, "new_question_runs": 100,
                    "arms": list(self.probe_arms)}
        manifest = self.verify_manifest(resources=False)
        groups = {}
        for stage in stages():
            complete, interrupted, pending = [], [], []
            for query in stage["queries"]:
                arm, qid = stage["arm"], query["query_id"]
                path = self.terminal_path(arm, qid)
                if path.exists():
                    row = self.validate_terminal(runner.load(path), manifest, arm, query)
                    self.verify_execution_artifacts(manifest, arm, query, row)
                    complete.append(qid)
                elif any((self.runs / arm / d / f"qid_{qid}{e}").exists() for d, e in
                         (("_checkpoints", ".json"), ("request_contracts", ".json"),
                          ("api_ledgers", ".jsonl"), ("_infrastructure", ".json"),
                          ("_working", ""), ("_rerank_runtime", ""))):
                    interrupted.append(qid)
                else:
                    pending.append(qid)
            groups[stage["arm"]] = {"terminal": complete, "interrupted": interrupted, "pending": pending}
        return {"version": VERSION, "prepared": True, "arms": groups,
            "new_question_runs": 100, "reused_auto20_questions": 20, "reused_control_questions": 100,
            "completed_new_runs": sum(len(g["terminal"]) for g in groups.values()),
            "full_three_arm_suite_complete": False}

    def pipeline(self, config, *, service_pid, resume_infrastructure=False):
        identity = process_identity()
        try:
            if not (self.reports / "run_manifest.json").exists():
                self._state(identity, "prepare")
                self.prepare(config, service_pid=service_pid)
            self._state(identity, "live_preflight")
            # An existing frozen epoch may be refreshed only during explicit recovery.
            self.live_preflight(resume_infrastructure=resume_infrastructure,
                                service_pid=service_pid if resume_infrastructure else None)
            return self.run(resume_infrastructure=resume_infrastructure, worker_identity=identity)
        except BaseException as exc:
            self._state(identity, "interrupted", error_type=type(exc).__name__)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--control-root", required=True)
    template.add_argument("--reuse-auto20-root", required=True)
    offline = sub.add_parser("offline-preflight")
    offline.add_argument("--config"); offline.add_argument("--control-root"); offline.add_argument("--reuse-auto20-root")
    for command in ("prepare", "pipeline"):
        p = sub.add_parser(command); p.add_argument("--config", required=True)
        p.add_argument("--service-pid", required=True, type=int)
        if command == "pipeline": p.add_argument("--resume-infrastructure", action="store_true")
    live = sub.add_parser("live-preflight")
    live.add_argument("--resume-infrastructure", action="store_true"); live.add_argument("--service-pid", type=int)
    run = sub.add_parser("run"); run.add_argument("--resume-infrastructure", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv); suite = ContinuationSuite(args.root)
    if args.command == "template": result = suite.template(args.control_root, args.reuse_auto20_root)
    elif args.command == "offline-preflight":
        result = suite.offline_preflight(runner.load(args.config) if args.config else None,
            control_root=args.control_root, reuse_auto20_root=args.reuse_auto20_root)
    elif args.command == "prepare": result = suite.prepare(runner.load(args.config), service_pid=args.service_pid)
    elif args.command == "live-preflight": result = suite.live_preflight(
        resume_infrastructure=args.resume_infrastructure, service_pid=args.service_pid)
    elif args.command == "pipeline": result = suite.pipeline(runner.load(args.config),
        service_pid=args.service_pid, resume_infrastructure=args.resume_infrastructure)
    elif args.command == "run": result = suite.run(resume_infrastructure=args.resume_infrastructure)
    else: result = suite.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
