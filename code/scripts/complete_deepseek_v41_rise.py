#!/usr/bin/env python3
"""Run the fixed 100 RISE questions with DeepSeek V4.1 Flash, two at a time.

Uses the same OpenAI-compatible Alibaba endpoint and credential as the Qwen
run. Credentials stay in the process environment and are never written here.
The Qwen runner supplies checkpoint, moderation, judging, and audit logic.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import complete_qwen_rise as runner


ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-v4.1-flash"
TOKEN_PLAN_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def main() -> int:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is required in the process environment")
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["DEEPSEEK_BASE_URL"] = TOKEN_PLAN_BASE_URL
    runner.MODEL = MODEL
    sys.argv = [
        sys.argv[0],
        "--out-root", str(ROOT / "runs/rise_deepseek_v41_flash_100"),
        "--status-out", str(ROOT / "logs/deepseek_v41_full100_status.json"),
        "--audit-out", str(ROOT / "logs/deepseek_v41_full100_audit.json"),
        "--per-call-max-tokens", "16000",
        *sys.argv[1:],
    ]
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
