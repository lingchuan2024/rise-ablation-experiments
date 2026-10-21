#!/usr/bin/env python3
"""Start the 100-query run after the follow-up pilot passes its audit gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FAILURE_STAGES = {"agent_incomplete", "judge_incomplete", "report_failed", "audit_failed"}


def pilot_is_ready(status: dict, audit: dict, expected: int = 10) -> tuple[bool, str]:
    summary = audit.get("summary") or {}
    if status.get("stage") != "complete":
        return False, "pilot pipeline has not completed"
    for key in ("rows", "traces", "judged"):
        if status.get(key) != expected:
            return False, f"pilot {key} count is {status.get(key)!r}, expected {expected}"
    if summary.get("errors") != 0:
        return False, f"pilot audit has {summary.get('errors')!r} structural errors"
    if summary.get("missing_full_tool_outputs") != 0:
        return False, "pilot trajectories omit full tool outputs"
    return True, "pilot passed: complete rows, judgments, traces, and audit fidelity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-status", type=Path,
                        default=ROOT / "logs/pilot_next10_pipeline_status.json")
    parser.add_argument("--pilot-audit", type=Path,
                        default=ROOT / "logs/pilot_next10_audit.json")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=float, default=8)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.max_wait_hours <= 0:
        parser.error("poll interval and maximum wait must be positive")
    deadline = time.monotonic() + args.max_wait_hours * 3600
    print("Waiting for the next-10 pilot to complete and pass its audit", flush=True)
    while time.monotonic() < deadline:
        if args.pilot_status.is_file():
            try:
                status = json.loads(args.pilot_status.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                status = {}
            stage = status.get("stage")
            if stage in FAILURE_STAGES:
                print(f"Pilot stopped at stage {stage}; full run was not started", flush=True)
                return 2
            if stage == "complete":
                if not args.pilot_audit.is_file():
                    print("Pilot audit file is missing; full run was not started", flush=True)
                    return 3
                try:
                    audit = json.loads(args.pilot_audit.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    print(f"Pilot audit cannot be read: {type(exc).__name__}", flush=True)
                    return 3
                ready, reason = pilot_is_ready(status, audit)
                print(reason, flush=True)
                if not ready:
                    return 3
                return subprocess.run(
                    [sys.executable, "-u", str(ROOT / "scripts/complete_nemotron_rise.py")],
                    cwd=ROOT, check=False,
                ).returncode
        time.sleep(args.poll_seconds)
    print("Pilot did not complete before the wait limit; full run was not started", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
