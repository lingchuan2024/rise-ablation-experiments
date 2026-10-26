#!/usr/bin/env python3
"""Run v14: optional explicit bash-path handoff to independent rerank."""
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
if "--no-independent-rerank" in sys.argv:
    raise SystemExit("run_v14.py requires --independent-rerank")
if "--independent-rerank" not in sys.argv:
    sys.argv.append("--independent-rerank")
runpy.run_path(str(ROOT / "code/scripts/run_rise.py"), run_name="__main__")
