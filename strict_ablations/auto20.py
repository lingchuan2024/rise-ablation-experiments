"""Run only the fixed first20 automatic-rerank pilot with reused B100 control.

The pilot has its own namespace and manifest. It neither schedules the other
two variants nor claims completion of the separate three-arm fixed100 suite.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
import time

from strict_ablations import runner

VERSION = "strict-auto20-pilot-v1"
DIRECTORY = "strict_auto20_v1"
ARM = "auto_rerank"
COUNT = 20


def selected_queries():
    rows = runner.shared.validate_queries()
    if len(rows) != 100:
        raise RuntimeError("The pilot must select from the frozen fixed100 cohort")
    return rows[:COUNT]


class Auto20Suite(runner.Suite):
    directory = DIRECTORY
    probe_arms = (ARM,)

    def execution_scope(self, rows):
        qids = [row["query_id"] for row in rows[:COUNT]]
        if len(rows) != 100 or len(qids) != COUNT:
            raise RuntimeError("The pilot requires the first20 of the frozen100")
        return {"question_runs": COUNT, "reused_control_questions": COUNT,
            "selected_query_ids": qids, "pilot_query_ids": qids,
            "scheduling": "original first20 order; automatic-rerank only; shared global C4",
            "execution_plan": {"version": VERSION, "arms": [ARM],
                "selection": "first20_in_frozen_fixed100_order", "question_runs": COUNT,
                "control_policy": "reuse_matching_B100_public_finals; no control requests",
                "stage": "exploratory_pilot"}}

    def analysis_contract(self):
        from strict_ablations import auto20_analysis
        return {"version": auto20_analysis.VERSION,
                "rubric_sha256": runner.digest(auto20_analysis.rubric_document())}

    def offline_preflight(self, config=None, *, control_root=None):
        report = super().offline_preflight(config, control_root=control_root)
        return {**report, "pilot_version": VERSION, "arms": [ARM],
                "new_question_runs": COUNT, **self.execution_scope(runner.shared.validate_queries())}

    def verify_manifest(self, *, resources=True):
        manifest = super().verify_manifest(resources=resources)
        expected = self.execution_scope(runner.shared.validate_queries())
        if any(runner.digest(manifest.get(key)) != runner.digest(value)
               for key, value in expected.items()):
            raise RuntimeError("The frozen automatic20 pilot execution scope changed")
        return manifest

    def terminal_path(self, arm, qid):
        if arm != ARM or type(qid) is not int or qid not in {
                row["query_id"] for row in selected_queries()}:
            raise ValueError("Only the fixed20 automatic-rerank pilot may be scheduled")
        return super().terminal_path(arm, qid)

    def pilot_gate(self, manifest):
        raise RuntimeError("The three-arm pilot gate is outside this automatic20 pilot")

    def run(self, stage="pilot20", *, resume_infrastructure=False):
        if stage != "pilot20":
            raise ValueError("This entry point admits only the automatic20 pilot")
        return self.run20(resume_infrastructure=resume_infrastructure)

    def run20(self, *, resume_infrastructure=False):
        self.reports.mkdir(parents=True, exist_ok=True)
        with (self.reports / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.verify_manifest()
            ready = runner.load(self.reports / "preflight_live.json")
            if ready.get("passed") is not True or ready.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise RuntimeError("This automatic20 manifest needs its own live preflight")
            os.environ.update(manifest["config"]["runtime_environment"])
            service = self.verify_service(manifest)
            if ready.get("service", {}).get("epoch_sha256") != service["epoch_sha256"]:
                raise RuntimeError("The current process epoch needs its own live preflight")
            jobs = iter(selected_queries())
            completed = []
            with ThreadPoolExecutor(max_workers=manifest["config"]["concurrency"]) as pool:
                active = {}
                for _ in range(manifest["config"]["concurrency"]):
                    query = next(jobs, None)
                    if query is not None:
                        active[pool.submit(self.run_one, manifest, ARM, query,
                            resume_infrastructure=resume_infrastructure)] = query["query_id"]
                while active:
                    future = next(as_completed(active))
                    active.pop(future)
                    completed.append(future.result())
                    query = next(jobs, None)
                    if query is not None:
                        active[pool.submit(self.run_one, manifest, ARM, query,
                            resume_infrastructure=resume_infrastructure)] = query["query_id"]
            self.verify_manifest()
            status = self.status()
            if status["completed_new_runs"] != COUNT:
                raise RuntimeError("Automatic20 completion requires all20 validated terminals")
            receipt = {"version": VERSION, "passed": True, "stage": "pilot20",
                "manifest_sha256": manifest["manifest_sha256"], "arm": ARM,
                "selected_query_ids": manifest["selected_query_ids"],
                "finished_unix": time.time(), "natural_error_reruns": 0,
                "terminal_rows_checked": len(completed), "question_runs": COUNT,
                "control_model_requests": 0, "other_arm_model_requests": 0,
                "full_three_arm_suite_complete": False}
            runner.atomic_json(self.reports / "pilot20_completion.json", receipt)
            return receipt

    def status(self):
        if not (self.reports / "run_manifest.json").exists():
            return {"version": VERSION, "prepared": False, "live_ready": False,
                    "arms": [ARM], "new_question_runs": COUNT}
        manifest = self.verify_manifest(resources=False)
        complete, interrupted, pending = [], [], []
        for query in selected_queries():
            qid = query["query_id"]
            path = self.terminal_path(ARM, qid)
            if path.exists():
                row = self.validate_terminal(runner.load(path), manifest, ARM, query)
                self.verify_execution_artifacts(manifest, ARM, query, row)
                complete.append(qid)
            elif any((self.runs / ARM / directory / f"qid_{qid}{suffix}").exists()
                     for directory, suffix in (("_checkpoints", ".json"),
                         ("request_contracts", ".json"), ("api_ledgers", ".jsonl"),
                         ("_infrastructure", ".json"))):
                interrupted.append(qid)
            else:
                pending.append(qid)
        return {"version": VERSION, "prepared": True, "arms": {
                ARM: {"terminal": complete, "interrupted": interrupted, "pending": pending}},
                "new_question_runs": COUNT, "reused_control_questions": COUNT,
                "completed_new_runs": len(complete), "full_three_arm_suite_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(runner.ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--control-root", default=str(runner.ROOT))
    offline = sub.add_parser("offline-preflight")
    offline.add_argument("--config"); offline.add_argument("--control-root")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--service-pid", type=int, required=True)
    live = sub.add_parser("live-preflight")
    live.add_argument("--resume-infrastructure", action="store_true")
    live.add_argument("--service-pid", type=int)
    run = sub.add_parser("run20")
    run.add_argument("--resume-infrastructure", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    suite = Auto20Suite(args.root)
    if args.command == "template":
        result = suite.template(args.control_root)
    elif args.command == "offline-preflight":
        result = suite.offline_preflight(runner.load(args.config) if args.config else None,
                                        control_root=args.control_root)
    elif args.command == "prepare":
        result = suite.prepare(runner.load(args.config), service_pid=args.service_pid)
    elif args.command == "live-preflight":
        result = suite.live_preflight(resume_infrastructure=args.resume_infrastructure,
                                      service_pid=args.service_pid)
    elif args.command == "run20":
        result = suite.run20(resume_infrastructure=args.resume_infrastructure)
    else:
        result = suite.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
