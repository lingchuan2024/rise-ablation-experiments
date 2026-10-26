#!/usr/bin/env python3
"""Project an already returned public final, without replay or private-message parsing.

Only allowlisted JSON paths are decoded. Unknown values (including checkpoint
messages, all reasoning fields, requests' messages and tool arguments) are
lexically skipped as opaque bytes. No model/tool calls or original-file edits.
The derived outcome is explicitly NOT a reconstructed natural terminal/trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

VERSION = "strict-observed-public-final-recovery-v1"
MODEL = "Qwen3.5-35B-A3B"
COUNT = object()
SCALAR = object()
NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


class Projection:
    """Small JSON reader which never decodes values outside a whitelist.

    Unknown strings are scanned only for their closing delimiter/JSON syntax;
    their contents are never passed to json.loads or retained in the result.
    Error messages contain only byte positions, never surrounding source text.
    """
    def __init__(self, data):
        self.data, self.pos = data, 0

    def fail(self):
        raise ValueError("Malformed JSON at byte " + str(self.pos))

    def ws(self):
        while self.pos < len(self.data) and self.data[self.pos] in b" \r\n\t":
            self.pos += 1

    def take(self, char):
        self.ws()
        if self.pos >= len(self.data) or self.data[self.pos] != char:
            self.fail()
        self.pos += 1

    def string_end(self):
        if self.pos >= len(self.data) or self.data[self.pos] != 34:
            self.fail()
        self.pos += 1
        while self.pos < len(self.data):
            ch = self.data[self.pos]
            self.pos += 1
            if ch == 34:
                return
            if ch < 32:
                self.fail()
            if ch == 92:
                if self.pos >= len(self.data):
                    self.fail()
                escape = self.data[self.pos]
                self.pos += 1
                if escape == 117:
                    part = self.data[self.pos:self.pos + 4]
                    if len(part) != 4 or any(c not in b"0123456789abcdefABCDEF" for c in part):
                        self.fail()
                    self.pos += 4
                elif escape not in b'"\\/bfnrt':
                    self.fail()
        self.fail()

    def key(self):
        self.ws()
        start = self.pos
        self.string_end()
        try:
            return json.loads(self.data[start:self.pos])
        except Exception:
            self.fail()

    def value(self, spec=None, depth=0):
        if depth > 300:
            self.fail()
        self.ws()
        if self.pos >= len(self.data):
            self.fail()
        start, ch = self.pos, self.data[self.pos]
        if ch == 123:
            require(spec is None or isinstance(spec, dict), "Projected value has wrong JSON type")
            self.pos += 1
            result, seen = {}, set()
            self.ws()
            if self.pos < len(self.data) and self.data[self.pos] == 125:
                self.pos += 1
                return result if spec is not None else None
            while True:
                # Keys are structural metadata; unknown values remain opaque.
                key = self.key()
                require(key not in seen, "Duplicate JSON key at byte " + str(self.pos))
                seen.add(key)
                self.take(58)
                child = spec.get(key) if isinstance(spec, dict) else None
                value = self.value(child, depth + 1)
                if child is not None:
                    result[key] = value
                self.ws()
                if self.pos >= len(self.data):
                    self.fail()
                delim = self.data[self.pos]
                self.pos += 1
                if delim == 125:
                    return result if spec is not None else None
                if delim != 44:
                    self.fail()
        if ch == 91:
            require(spec is None or spec is COUNT or isinstance(spec, list), "Projected value has wrong JSON type")
            self.pos += 1
            result, count = [], 0
            self.ws()
            if self.pos < len(self.data) and self.data[self.pos] == 93:
                self.pos += 1
                return 0 if spec is COUNT else ([] if spec is not None else None)
            while True:
                value = self.value(spec[0] if isinstance(spec, list) else None, depth + 1)
                count += 1
                if isinstance(spec, list):
                    result.append(value)
                self.ws()
                if self.pos >= len(self.data):
                    self.fail()
                delim = self.data[self.pos]
                self.pos += 1
                if delim == 93:
                    return count if spec is COUNT else (result if spec is not None else None)
                if delim != 44:
                    self.fail()
        if ch == 34:
            self.string_end()
        else:
            literal = next((v for v in (b"true", b"false", b"null")
                            if self.data.startswith(v, self.pos)), None)
            if literal:
                self.pos += len(literal)
            else:
                match = NUMBER.match(self.data, self.pos)
                if not match:
                    self.fail()
                self.pos = match.end()
        if spec is None:
            return None
        if spec is COUNT:
            require(self.data[start:self.pos] == b"null", "Projected array must be array or null")
            return 0
        require(spec is SCALAR, "Projected value has wrong JSON type")
        try:
            return json.loads(self.data[start:self.pos])
        except Exception:
            self.fail()

    def project(self, spec):
        value = self.value(spec)
        self.ws()
        require(self.pos == len(self.data), "Trailing JSON bytes at " + str(self.pos))
        return value


USAGE = {key: SCALAR for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
API_SPEC = {"time": SCALAR, "category": SCALAR, "attempt": SCALAR,
    "elapsed_seconds": SCALAR, "request": {"model": SCALAR},
    "response": {"id": SCALAR, "model": SCALAR, "usage": USAGE,
                 "choices": [{"finish_reason": SCALAR, "message": {"tool_calls": COUNT}}]}}
# Keep the sentinel identities when constructing the final-only projection.
FINAL_API_SPEC = {**API_SPEC, "response": {**API_SPEC["response"], "choices": [
    {"finish_reason": SCALAR, "message": {"tool_calls": COUNT, "content": SCALAR}}]}}
CP_SPEC = {"version": SCALAR, "elapsed_seconds": SCALAR,
    "identity": {key: SCALAR for key in ("model", "query_id", "ablation_arm", "contract_sha256",
        "common_config_sha256", "question_sha256", "system_prompt_sha256", "tool_specs_sha256", "run_config_hash")},
    "run": {key: (COUNT if key == "turns" else SCALAR) for key in
        ("turns", "terminated_reason", "total_prompt_tokens", "total_cached_prompt_tokens", "total_completion_tokens")}}


def pin(root, path):
    require(path.is_file() and not path.is_symlink(), "Missing or aliased source file")
    require(path.resolve().is_relative_to(root), "Source file escapes its root")
    parent = path.parent
    while parent != root:
        require(not parent.is_symlink(), "Source ancestor is aliased")
        parent = parent.parent
    return {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def number(value, *, integer=False):
    require(type(value) in ((int,) if integer else (int, float))
            and math.isfinite(value) and value >= 0, "Missing or invalid numeric metadata")
    return value


def recover(root, output, *, expected_manifest_sha256, query_id=310):
    require(query_id == 310, "Only the observed final for qid 310 is authorized")
    root, output = Path(root).resolve(), Path(output).resolve()
    run = root / "runs/strict_continue_v1/auto_rerank"
    paths = {"manifest": root / "reports/strict_continue_v1/run_manifest.json",
        "contract": run / "request_contracts/qid_310.json",
        "ledger": run / "api_ledgers/qid_310.jsonl",
        "api": run / "_traces/qid_310/single.api.jsonl",
        "checkpoint": run / "_checkpoints/qid_310.json"}
    require(not output.exists() and output not in paths.values()
            and not output.is_relative_to(root / "runs"), "Derived output must be new and outside original runs")
    pins = {p.relative_to(root).as_posix(): pin(root, p) for p in paths.values()}
    # These three files contain public contracts/numeric metadata only.
    manifest = json.loads(paths["manifest"].read_bytes())
    require(manifest.get("manifest_sha256") == expected_manifest_sha256
            == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}), "Original manifest differs")
    require(manifest["config"]["model"] == MODEL, "Original model differs")
    require(manifest.get("config_sha256") == digest(manifest["config"]), "Common configuration digest differs")
    contract = json.loads(paths["contract"].read_bytes())
    frozen = manifest["contracts"]["auto_rerank"]
    require(contract.get("query_id") == query_id and contract.get("arm") == "auto_rerank"
            and contract.get("model") == MODEL and contract.get("common_config_sha256") == manifest["config_sha256"]
            and all(contract.get(k) == frozen[k] for k in
                ("contract_sha256", "system_prompt", "tool_specs", "system_prompt_sha256", "tool_specs_sha256")),
            "Actual recorded request contract differs")
    ledger = [json.loads(line) for line in paths["ledger"].read_bytes().splitlines() if line.strip()]
    lines = [line for line in paths["api"].read_bytes().splitlines() if line.strip()]
    require(ledger and len(ledger) == len(lines), "API events and ledger must reconcile exactly")
    totals = {key: 0 for key in ("prompt_tokens", "completion_tokens")}
    events = []
    for index, (row, line) in enumerate(zip(ledger, lines)):
        event = Projection(line).project(FINAL_API_SPEC if index == len(lines) - 1 else API_SPEC)
        response = event.get("response", {})
        require(row.get("status") == "ok" and row.get("requested_model") == row.get("reported_model") == MODEL
                and row.get("query_id") == query_id and row.get("arm") == "auto_rerank"
                and row.get("contract_sha256") == frozen["contract_sha256"], "Ledger request identity/status differs")
        require(response.get("id") == row.get("response_id") and response.get("model") == MODEL
                and event.get("request", {}).get("model") == MODEL, "Recorded response/request identity differs")
        require(isinstance(response.get("choices"), list) and len(response["choices"]) == 1,
                "Exactly one response choice is required")
        for key in totals:
            value = number(row.get("usage", {}).get(key), integer=True)
            require(response.get("usage", {}).get(key) == value, "API/ledger numeric usage differs")
            totals[key] += value
        require(response["choices"][0].get("finish_reason") == row.get("finish_reason"), "Finish reason differs")
        number(event.get("time")); number(row.get("unix_time")); number(row.get("request_seconds"))
        require(event["time"] >= row["unix_time"], "Response precedes its request")
        if index != len(lines) - 1:
            require(not (event.get("category") == "content"
                and response["choices"][0].get("finish_reason") == "stop"
                and response["choices"][0].get("message", {}).get("tool_calls", 0) == 0),
                "An earlier final exists; answer selection is forbidden")
        events.append(event)
    cp = Projection(paths["checkpoint"].read_bytes()).project(CP_SPEC)
    expected_identity = {"model": MODEL, "query_id": str(query_id), "ablation_arm": "auto_rerank",
        "contract_sha256": frozen["contract_sha256"], "common_config_sha256": manifest["config_sha256"],
        "question_sha256": contract["question_sha256"], "system_prompt_sha256": frozen["system_prompt_sha256"],
        "tool_specs_sha256": frozen["tool_specs_sha256"]}
    require(cp.get("version") == 1 and all(cp.get("identity", {}).get(k) == v for k, v in expected_identity.items()),
            "Checkpoint public identity differs")
    saved = cp.get("run", {}); n = number(saved.get("turns"), integer=True)
    require(n == len(ledger) - 1 and saved.get("terminated_reason") == "", "Only a one-response final commit gap is supported")
    cp_input = number(saved.get("total_prompt_tokens"), integer=True) + number(saved.get("total_cached_prompt_tokens"), integer=True)
    cp_output = number(saved.get("total_completion_tokens"), integer=True)
    require(cp_input == sum(row["usage"]["prompt_tokens"] for row in ledger[:n])
            and cp_output == sum(row["usage"]["completion_tokens"] for row in ledger[:n]),
            "Checkpoint does not cover the exact ledger prefix")
    cp_elapsed = number(cp.get("elapsed_seconds"))
    final_event = events[-1]; response = final_event["response"]; choice = response["choices"][0]
    message = choice.get("message", {}); final = message.get("content")
    require(final_event.get("category") == "content" and choice.get("finish_reason") == "stop"
            and message.get("tool_calls", 0) == 0 and isinstance(final, str) and bool(final.strip()),
            "The last observed response is not a complete public final")
    receipt = {"version": VERSION, "verified": True, "query_id": query_id, "arm": "auto_rerank",
        "source_manifest_sha256": manifest["manifest_sha256"], "source_root": str(root),
        "outcome_kind": "materialized_observed_final", "full_natural_terminal_reconstructed": False,
        "response_id": response["id"], "finish_reason": "stop", "category": "content", "tool_calls_count": 0,
        "final_text": final, "final_sha256": sha(final.encode()), "source_files": pins,
        "known_input_tokens": totals["prompt_tokens"], "known_output_tokens": totals["completion_tokens"],
        "recorded_attempts": len(ledger), "unknown_usage_attempts": 0,
        "elapsed_seconds": None, "elapsed_time_status": "not_reconstructed",
        "recorded_checkpoint_elapsed_seconds": cp_elapsed,
        "request_to_final_seconds_lower_bound": final_event["time"] - ledger[0]["unix_time"],
        "checkpoint_turns": n, "uncommitted_final_responses": 1,
        "new_model_requests": 0, "new_tool_executions": 0, "original_files_modified": False,
        "private_message_or_reasoning_values_decoded": False,
        "policy": "First already-returned final; no resampling, replay, invented trace or reconstructed checkpoint"}
    receipt["receipt_sha256"] = digest(receipt)
    require(pins == {p.relative_to(root).as_posix(): pin(root, p) for p in paths.values()}, "Source bytes changed while projecting")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        os.chmod(output, 0o600)
        stream.write((json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return {"verified": True, "query_id": query_id, "outcome_kind": receipt["outcome_kind"],
        "full_natural_terminal_reconstructed": False, "response_id": response["id"],
        "public_final_sha256": receipt["final_sha256"], "public_final_chars": len(final),
        "known_input_tokens": totals["prompt_tokens"], "known_output_tokens": totals["completion_tokens"],
        "checkpoint_turns": n, "recorded_attempts": len(ledger), "output": str(output),
        "output_sha256": sha(output.read_bytes()), "output_bytes": output.stat().st_size,
        "new_model_requests": 0, "new_tool_executions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.output_file,
        expected_manifest_sha256=args.expected_manifest_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
