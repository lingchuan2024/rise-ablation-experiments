#!/usr/bin/env python3
"""Check the v10 tokenizer offline by default; --score also loads weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rise.reranker_backend import QwenReranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    backend = QwenReranker(model=args.model, revision=args.revision, device=args.device,
                           local_files_only=not args.allow_download)
    result = {"status": "unavailable", "weights_requested": args.score}
    try:
        tokenizer = backend.get_tokenizer()
        result.update(status="ok", sample_tokens=len(tokenizer.encode(
            "An on-demand reranker retrieves evidence from document bodies.", add_special_tokens=False)))
        if args.score:
            result["scores"] = backend.score("aurora calibration", [
                "This source describes aurora instrument calibration.",
                "This source lists unrelated history facts."])
            result["usage"] = backend.last_usage
    except Exception as exc:
        result.update(status="unavailable", message=str(exc))
    result["identity"] = backend.identity
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
