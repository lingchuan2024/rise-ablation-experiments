#!/usr/bin/env python3
"""Run one independent v14 ablation; server/suite.py schedules the fixed300 suite."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VARIANTS = ("search_paging", "auto_rerank", "no_bash")
MODEL = "Qwen3.5-35B-A3B"
PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"


def _has_option(arguments, name):
    return any(value == name or value.startswith(name + "=") for value in arguments)


def _option_value(arguments, name):
    for index, value in enumerate(arguments):
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def default_arguments(argv):
    arguments = list(argv)
    if "--no-independent-rerank" in arguments:
        raise SystemExit("The v14 ablations require --independent-rerank")
    if not any(value in {"-h", "--help"} for value in arguments):
        if not _has_option(arguments, "--ablation-variant"):
            raise SystemExit("Select --ablation-variant search_paging|auto_rerank|no_bash")
        if not _has_option(arguments, "--space-passage-index"):
            raise SystemExit("Provide the existing ready --space-passage-index")
    for flag in ("--independent-rerank", "--structured-docs", "--turn-checkpoint"):
        if flag not in arguments:
            arguments.append(flag)
    defaults = {"--model": MODEL, "--mini-dev": str(ROOT / "data/queries_100.jsonl"),
                "--max-turns": "100", "--per-call-max-tokens": "32000",
                "--api-max-retries": "3", "--concurrency": "4",
                "--reranker-model": "Qwen/Qwen3-Reranker-0.6B",
                "--reranker-revision": PIN, "--reranker-dtype": "bfloat16",
                "--reranker-batch-size": "8"}
    variant = _option_value(arguments, "--ablation-variant")
    if variant in VARIANTS:
        output_name = "search_paging_prompt_v2" if variant == "search_paging" else variant
        defaults["--out-root"] = str(ROOT / "runs" / output_name)
    for name, value in defaults.items():
        if not _has_option(arguments, name):
            arguments.extend([name, value])
    return arguments


def main(argv=None):
    sys.argv = [str(ROOT / "run_ablation.py"), *default_arguments(sys.argv[1:] if argv is None else argv)]
    spec = importlib.util.spec_from_file_location("v14_ablation_cli", ROOT / "code/scripts/run_rise.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
