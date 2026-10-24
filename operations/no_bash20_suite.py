#!/usr/bin/env python3
"""Administrative fixed-first20 no-bash run; frozen strict B100 solver unchanged.

No service launcher, cleanup, scoring-based selection, or partial-budget restart
is provided. A dispatched question without its validated terminal is an explicit
operator recovery case, never a fresh-budget retry.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_ablations import runner

VERSION = "strict-no-bash20-suite-v1"
DIRECTORY = "strict_no_bash20_v1"
ARM = "no_bash"
COUNT = 20
OLD_AUTO20_ROOT = Path("/root/data/rise-strict-auto20-20261023")
FROZEN_SOURCE_DIGEST = "a5a3e172a1581ac95d24cec857a187e14a87b4e21af0cbce366916e6fa3cc5a1"
ADMIN_FILES = ("operations/no_bash20_suite.py", "operations/test_no_bash20_suite.py")
STORAGE_ADMISSION = {"min_free_inodes": 40000, "min_free_bytes": 10 * 1024 ** 3, "poll_seconds": 30}
ALLOWED_RUN_DIRECTORIES = {"_per_query", "_traces", "_checkpoints", "request_contracts",
    "api_ledgers", "_infrastructure", "_working", "_rerank_runtime"}
_TEMP = {
    "_per_query": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}",
    "_infrastructure": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}",
    "_checkpoints": r"qid_([1-9][0-9]*)\.json\.[a-z0-9_]{8}\.tmp",
    "request_contracts": r"qid_([1-9][0-9]*)\.json\.tmp",
}


def selected_queries():
    rows = runner.shared.validate_queries()
    if len(rows) != 100 or len({q["query_id"] for q in rows}) != 100:
        raise RuntimeError("The original unique fixed100 order is required")
    return rows[:COUNT]


def admin_hashes():
    return {name: runner.file_sha(ROOT / name) for name in ADMIN_FILES}


def verify_frozen_sources():
    sources = runner.source_hashes()
    if len(sources) != 135 or runner.digest(sources) != FROZEN_SOURCE_DIGEST:
        raise RuntimeError("The original135 experiment source files changed")
    return sources


def process_identity():
    stat, boot = Path("/proc/self/stat"), Path("/proc/sys/kernel/random/boot_id")
    return {"pid": os.getpid(), "started_unix": time.time(),
        "proc_start_ticks": int(stat.read_text().rsplit(") ", 1)[1].split()[19]) if stat.exists() else None,
        "boot_id": boot.read_text().strip() if boot.exists() else None,
        "driver_sha256": runner.file_sha(Path(__file__)), "entrypoint": "operations.no_bash20_suite"}


def process_matches(identity):
    """Read only public process identity; never environment or credentials."""
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int:
        return False
    try:
        proc = Path("/proc") / str(identity["pid"])
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        command = (proc / "cmdline").read_bytes().split(b"\0")
        entry = any(arg.endswith((b"operations/no_bash20_suite.py", b"operations.no_bash20_suite")) for arg in command)
        return (fields[0] != "Z" and entry and int(fields[19]) == identity.get("proc_start_ticks")
            and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity.get("boot_id")
            and identity.get("driver_sha256") == runner.file_sha(Path(__file__)))
    except (OSError, ValueError, IndexError):
        return False


class NoBash20Suite(runner.Suite):
    directory = DIRECTORY
    probe_arms = (ARM,)
    legacy_auto20_root = OLD_AUTO20_ROOT

    def execution_scope(self, rows):
        if rows != runner.shared.validate_queries():
            raise RuntimeError("No-bash20 must preserve the complete frozen cohort identity")
        qids = [q["query_id"] for q in selected_queries()]
        return {"no_bash20_version": VERSION, "directory": DIRECTORY,
            "question_runs": COUNT, "reused_control_questions": COUNT,
            "selected_query_ids": qids, "pilot_query_ids": qids,
            "administrative_source_sha256": admin_hashes(), "storage_admission": STORAGE_ADMISSION,
            "worker_lock_policy": {
                "namespace_lock": f"reports/{DIRECTORY}/worker.lock",
                "legacy_auto20_lock": str(self.legacy_auto20_root / "reports/strict_auto20_v1/worker.lock"),
                "standard_suite_lock": "reports/strict_ablations_v1/worker.lock",
                "legacy_lock_open_mode": "rb"},
            "scheduling": "original first20 order; no_bash only; global C4; immutable natural terminals",
            "execution_plan": {"version": VERSION, "arms": [ARM], "question_runs": COUNT,
                "selection": "first20_in_frozen_fixed100_order", "stage": "exploratory_pilot",
                "control_policy": "reuse_matching_B100_public_finals; no control requests",
                "partial_policy": "refuse_unknown_partial; no automatic resume or budget reset"}}

    def analysis_contract(self):
        from strict_ablations import auto20_analysis
        return {"version": auto20_analysis.VERSION,
                "rubric_sha256": runner.digest(auto20_analysis.rubric_document())}

    def lock_paths(self):
        # The auto20 lock is also acquired by both continuation workers.
        return [self.reports / "worker.lock",
                self.legacy_auto20_root / "reports/strict_auto20_v1/worker.lock",
                self.root / "reports/strict_ablations_v1/worker.lock"]

    @contextmanager
    def worker_locks(self):
        with ExitStack() as stack:
            for index, path in enumerate(self.lock_paths()):
                if index != 1:
                    path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    raise RuntimeError("Worker lock must not be a symlink")
                handle = stack.enter_context(path.open("rb" if index == 1 else "a+b"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def offline_preflight(self, config=None, *, control_root=None):
        verify_frozen_sources()
        report = super().offline_preflight(config, control_root=control_root)
        return {**report, **self.execution_scope(runner.shared.validate_queries()),
                "arms": [ARM], "new_question_runs": COUNT, "model_requests": 0}

    def prepare(self, config, *, service_pid=None):
        with self.worker_locks():
            return super().prepare(config, service_pid=service_pid)

    def verify_manifest(self, *, resources=True):
        verify_frozen_sources()
        manifest = super().verify_manifest(resources=resources)
        expected = self.execution_scope(runner.shared.validate_queries())
        if any(runner.digest(manifest.get(k)) != runner.digest(v) for k, v in expected.items()):
            raise RuntimeError("Frozen no-bash20 scope/administrative source changed")
        if manifest["config"]["concurrency"] != 4:
            raise RuntimeError("No-bash20 requires the original C4 configuration")
        self.assert_run_scope()
        return manifest

    def assert_run_scope(self):
        if not self.runs.exists():
            return
        qids = {q["query_id"] for q in selected_queries()}
        for arm in self.runs.iterdir():
            if arm.is_symlink() or not arm.is_dir() or arm.name != ARM:
                raise RuntimeError("Unexpected arm in no-bash20 output")
            for directory in arm.iterdir():
                if directory.is_symlink() or not directory.is_dir() or directory.name not in ALLOWED_RUN_DIRECTORIES:
                    raise RuntimeError("Unexpected no-bash20 artifact directory")
                for path in directory.iterdir():
                    match = re.fullmatch(_TEMP.get(directory.name, r"(?!)"), path.name)
                    if match is None:
                        match = re.fullmatch(r"qid_([1-9][0-9]*)(?:\.jsonl?)?", path.name)
                    if path.is_symlink() or match is None or int(match[1]) not in qids:
                        raise RuntimeError("Artifact outside the authorized no-bash first20")

    def terminal_path(self, arm, qid):
        if arm != ARM or type(qid) is not int or qid not in {q["query_id"] for q in selected_queries()}:
            raise ValueError("Only no_bash fixed first20 may be scheduled")
        return super().terminal_path(arm, qid)

    def dispatch_path(self, qid):
        self.terminal_path(ARM, qid)
        return self.reports / "dispatches" / f"qid_{qid}.json"

    def evidence_present(self, qid):
        if self.dispatch_path(qid).exists():
            return True
        output = self.runs / ARM
        return any((output / d / f"qid_{qid}{ext}").exists() for d, ext in (
            ("_per_query", ".json"), ("_checkpoints", ".json"), ("request_contracts", ".json"),
            ("api_ledgers", ".jsonl"), ("_infrastructure", ".json"), ("_working", ""),
            ("_rerank_runtime", ""), ("_traces", ""))) or any(
                list((output / d).glob(f"qid_{qid}.json.*")) for d in _TEMP)

    def _dispatch(self, manifest, query, identity):
        qid = query["query_id"]
        if self.evidence_present(qid):
            raise RuntimeError("A partial/dispatched question cannot receive a fresh budget")
        payload = {"version": VERSION, "arm": ARM, "query_id": qid,
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][ARM]["contract_sha256"],
            "worker_identity": identity, "dispatched_unix": time.time(), "attempt_id": 1}
        runner.atomic_json(self.dispatch_path(qid), {**payload, "dispatch_sha256": runner.digest(payload)}, immutable=True)

    def validate_dispatch(self, manifest, query):
        record = runner.load(self.dispatch_path(query["query_id"]))
        expected = {"version": VERSION, "arm": ARM, "query_id": query["query_id"], "attempt_id": 1,
            "question_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
            "contract_sha256": manifest["contracts"][ARM]["contract_sha256"]}
        if (record.get("dispatch_sha256") != runner.digest({k: v for k, v in record.items() if k != "dispatch_sha256"})
                or any(record.get(k) != v for k, v in expected.items())
                or not isinstance(record.get("worker_identity"), dict)):
            raise RuntimeError("Question dispatch identity/budget/contract changed")
        return record

    def run_one(self, manifest, arm, query, *, resume_infrastructure=False):
        path = self.terminal_path(arm, query["query_id"])
        if resume_infrastructure:
            raise RuntimeError("Automatic infrastructure resume is deliberately unavailable")
        if manifest["administrative_source_sha256"] != admin_hashes():
            raise RuntimeError("No-bash20 administrative source drift")
        if not path.exists():
            record = self.validate_dispatch(manifest, query)
            if record.get("worker_identity") != self._worker_identity:
                raise RuntimeError("Question dispatch belongs to a different worker/manifest")
        result = super().run_one(manifest, arm, query, resume_infrastructure=False)
        if manifest["administrative_source_sha256"] != admin_hashes():
            raise RuntimeError("No-bash20 administrative source drift")
        return result

    def pilot_gate(self, manifest):
        raise RuntimeError("The three-arm gate is outside this no-bash20 run")

    def live_preflight(self, *, resume_infrastructure=False, service_pid=None):
        if resume_infrastructure or service_pid is not None:
            raise RuntimeError("Prepare binds the fresh service epoch; rebinding/resume is unavailable")
        with self.worker_locks():
            return self._live_preflight()

    def _stage_ready(self, manifest):
        ready = runner.load(self.reports / "preflight_live.json")
        if (ready.get("passed") is not True or ready.get("manifest_sha256") != manifest["manifest_sha256"]
                or set(ready.get("tools_and_native_calls", {})) != {ARM}
                or ready["tools_and_native_calls"][ARM].get("passed") is not True):
            raise RuntimeError("This manifest requires its own passed no-bash live preflight")
        service = self.verify_service(manifest)
        if ready.get("service", {}).get("epoch_sha256") != service["epoch_sha256"]:
            raise RuntimeError("The current service epoch needs its own live preflight")
        os.environ.update(manifest["config"]["runtime_environment"])

    def storage_status(self):
        stat = os.statvfs(self.root)
        free_bytes, free_inodes = stat.f_bavail * stat.f_frsize, stat.f_favail
        return {"ready": free_bytes >= STORAGE_ADMISSION["min_free_bytes"] and free_inodes >= STORAGE_ADMISSION["min_free_inodes"],
                "free_bytes": free_bytes, "free_inodes": free_inodes, **STORAGE_ADMISSION}

    def _state(self, identity, phase, **extra):
        value = {"version": VERSION, "worker_identity": identity, "phase": phase,
                 "checked_unix": time.time(), **extra}
        runner.atomic_json(self.reports / "phase_state.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def status(self):
        if not (self.reports / "run_manifest.json").exists():
            return {"version": VERSION, "prepared": False, "new_question_runs": COUNT}
        manifest = self.verify_manifest(resources=False)
        phase_path = self.reports / "phase_state.json"
        state = runner.load(phase_path) if phase_path.exists() else {}
        owner_live = process_matches(state.get("worker_identity"))
        terminal, partial, pending, active = [], [], [], []
        for query in selected_queries():
            qid = query["query_id"]
            path = self.terminal_path(ARM, qid)
            if path.exists():
                self.validate_dispatch(manifest, query)
                row = self.validate_terminal(runner.load(path), manifest, ARM, query)
                self.verify_execution_artifacts(manifest, ARM, query, row)
                terminal.append(qid)
            elif self.evidence_present(qid):
                partial.append(qid)
                if owner_live and qid in state.get("active_query_ids", []):
                    active.append(qid)
            else:
                pending.append(qid)
        return {"version": VERSION, "prepared": True, "phase": state.get("phase"),
            "worker_identity": state.get("worker_identity"), "worker_identity_live": owner_live,
            "arms": {ARM: {"terminal": terminal, "partial_or_active": partial,
                "active": active, "pending": pending}}, "new_question_runs": COUNT,
            "reused_control_questions": COUNT, "completed_new_runs": len(terminal),
            "validated_natural_terminals": len(terminal), "target_coverage": len(terminal),
            "full_three_arm_suite_complete": False}

    def _run_locked(self, identity):
        self._worker_identity = identity
        manifest = self.verify_manifest()
        self._stage_ready(manifest)
        existing = self.status()
        if existing["arms"][ARM]["partial_or_active"]:
            raise RuntimeError("Unresolved prior dispatch/partial needs separate audited recovery; no fresh retry")
        complete = set(existing["arms"][ARM]["terminal"])
        jobs = deque(q for q in selected_queries() if q["query_id"] not in complete)
        failures = []
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {}
                while jobs or active:
                    capacity = self.storage_status()
                    while jobs and len(active) < 4 and capacity["ready"] and not failures:
                        query = jobs.popleft()
                        self._dispatch(manifest, query, identity)
                        active[pool.submit(self.run_one, manifest, ARM, query)] = query
                        capacity = self.storage_status()
                    phase = "draining_after_error" if failures else ("waiting_storage" if jobs and not capacity["ready"] else "running_no_bash20")
                    self._state(identity, phase, manifest_sha256=manifest["manifest_sha256"],
                        active_query_ids=sorted(q["query_id"] for q in active.values()),
                        pending_query_ids=[q["query_id"] for q in jobs], completed=len(complete),
                        storage=capacity, errors=failures)
                    if active:
                        done, _ = wait(active, timeout=STORAGE_ADMISSION["poll_seconds"], return_when=FIRST_COMPLETED)
                        for future in done:
                            query = active.pop(future)
                            try:
                                future.result(); complete.add(query["query_id"])
                            except Exception as exc:
                                failures.append({"query_id": query["query_id"], "error_type": type(exc).__name__})
                    elif failures:
                        break
                    elif jobs:
                        time.sleep(STORAGE_ADMISSION["poll_seconds"])
            if failures:
                raise RuntimeError("A question was interrupted; active questions drained without rescheduling")
            self.verify_manifest()
            result = self.status()
            if result["completed_new_runs"] != COUNT:
                raise RuntimeError("Completion requires exactly20 validated natural terminals")
            rows = []
            for query in selected_queries():
                path = self.terminal_path(ARM, query["query_id"]); row = runner.load(path)
                rows.append({"query_id": query["query_id"], "row_sha256": row["row_sha256"],
                    "terminal_sha256": runner.file_sha(path), "execution_artifacts": row["execution_artifacts"]})
            receipt = {"version": VERSION, "passed": True, "arm": ARM,
                "manifest_sha256": manifest["manifest_sha256"], "selected_query_ids": manifest["selected_query_ids"],
                "terminal_rows_checked": COUNT, "terminal_rows": rows, "question_runs": COUNT,
                "validated_natural_terminals": COUNT, "target_coverage": COUNT,
                "control_model_requests": 0, "other_arm_model_requests": 0, "natural_error_reruns": 0,
                "full_three_arm_suite_complete": False}
            path = self.reports / "no_bash20_completion.json"
            if path.exists():
                if runner.load(path) != receipt:
                    raise RuntimeError("Immutable no-bash20 completion changed")
            else:
                runner.atomic_json(path, receipt, immutable=True)
            self._state(identity, "complete", manifest_sha256=manifest["manifest_sha256"],
                receipt_sha256=runner.file_sha(path), active_query_ids=[], pending_query_ids=[])
            return receipt
        except BaseException as exc:
            self._state(identity, "interrupted", manifest_sha256=manifest["manifest_sha256"],
                        error_type=type(exc).__name__, errors=failures)
            raise

    def run20(self):
        with self.worker_locks():
            return self._run_locked(process_identity())

    def run(self, stage="pilot20", *, resume_infrastructure=False):
        if stage != "pilot20" or resume_infrastructure:
            raise ValueError("Only first-attempt no-bash20 is authorized by this runner")
        return self.run20()

    def pipeline(self, config, *, service_pid):
        identity = process_identity()
        with self.worker_locks():
            try:
                if not (self.reports / "run_manifest.json").exists():
                    self._state(identity, "prepare")
                    super().prepare(config, service_pid=service_pid)
                else:
                    raise RuntimeError("Pipeline requires a fresh namespace; use status/run20 for a prepared manifest")
                self._state(identity, "live_preflight")
                self._live_preflight()
                return self._run_locked(identity)
            except BaseException as exc:
                self._state(identity, "interrupted", error_type=type(exc).__name__)
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template"); template.add_argument("--control-root", default=str(ROOT))
    offline = sub.add_parser("offline-preflight"); offline.add_argument("--config"); offline.add_argument("--control-root")
    for name in ("prepare", "pipeline"):
        p = sub.add_parser(name); p.add_argument("--config", required=True); p.add_argument("--service-pid", type=int, required=True)
    for name in ("live-preflight", "run20", "status"):
        sub.add_parser(name)
    args = parser.parse_args(argv); suite = NoBash20Suite(args.root)
    if args.command == "template": result = suite.template(args.control_root)
    elif args.command == "offline-preflight": result = suite.offline_preflight(runner.load(args.config) if args.config else None, control_root=args.control_root)
    elif args.command == "prepare": result = suite.prepare(runner.load(args.config), service_pid=args.service_pid)
    elif args.command == "pipeline": result = suite.pipeline(runner.load(args.config), service_pid=args.service_pid)
    elif args.command == "live-preflight": result = suite.live_preflight()
    elif args.command == "run20": result = suite.run20()
    else: result = suite.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
