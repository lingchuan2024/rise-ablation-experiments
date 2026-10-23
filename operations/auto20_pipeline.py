#!/usr/bin/env python3
"""Run the frozen auto20 preflight and worker in one administrative process."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strict_ablations.auto20 import Auto20Suite
from strict_ablations.runner import atomic_json


def main():
    os.chdir(ROOT)
    suite = Auto20Suite(ROOT)
    state_path = ROOT / "operations/auto20_pipeline_state.json"
    identity = {"pid": os.getpid(), "started_unix": time.time(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def state(phase, **extra):
        value = {**identity, "phase": phase, "checked_unix": time.time(), **extra}
        atomic_json(state_path, value)
        print(json.dumps(value), flush=True)

    try:
        state("live_preflight")
        suite.live_preflight()
        state("running_auto20")
        receipt = suite.run20()
        state("complete", receipt=receipt)
    except BaseException as exc:
        state("interrupted", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
