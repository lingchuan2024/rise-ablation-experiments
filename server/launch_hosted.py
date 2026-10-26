#!/usr/bin/env python3
"""Start the frozen hosted pilot with an in-memory credential and detached logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import hosted_pilot as pilot
from runtime_guards import atomic_json


def launch(*, resume=False):
    manifest, config = pilot.verify_manifest()
    pilot.require_preflight(manifest)
    environment = pilot.runtime_environment(config)
    receipt_path = pilot.REPORTS / "launch_receipt.json"
    if receipt_path.exists():
        previous = pilot.load(receipt_path)
        pid = previous.get("pid")
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("The previous hosted launcher process is still running")
        if not resume:
            raise RuntimeError("Preserve the previous attempt and explicitly resume after inspection")
        receipt_path.rename(pilot.REPORTS / f"launch_receipt.{time.time_ns()}.json")
    pilot.REPORTS.mkdir(parents=True, exist_ok=True)
    command = [config["python"], str(Path(pilot.__file__).resolve()), "run"]
    if resume:
        command.append("--resume-after-preflight")
    log_path = pilot.REPORTS / "runner.log"
    with log_path.open("ab", buffering=0) as log:
        worker = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = {"pid": worker.pid, "manifest_sha256": manifest["manifest_sha256"],
        "started_unix": time.time(), "log": str(log_path), "concurrency": 3,
        "question_runs": 30, "credential_persisted": False,
        "credential_transport": "stdin_to_parent_then_child_environment"}
    atomic_json(receipt_path, receipt)
    return receipt


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret-stdin", action="store_true")
    parser.add_argument("--resume-after-preflight", action="store_true")
    args = parser.parse_args()
    if args.secret_stdin:
        pilot.secret_from_stdin()
    result = launch(resume=args.resume_after_preflight)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "hosted_launch_failed", "error_type": type(exc).__name__}),
              file=sys.stderr)
        raise SystemExit(1) from None
