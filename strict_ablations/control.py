"""Read-only reuse gate for the frozen minimal-prompt Qwen3.5 B100 control.

No solver, judge, network, GPU, or original private trajectory is loaded. The
pinned public export and shutdown receipt attest the recorded numeric ledger
for all 100 questions; this does not claim usage missing from the original
shutdown window is known. Empty finals are retained for fresh blind scoring.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re

VERSION = "strict-ablations-frozen-control-v1"
MODEL = "Qwen3.5-35B-A3B"
# The manifest identity is its canonical payload digest, not its pretty-JSON
# file byte digest. The remaining pins below are byte digests of saved files.
MANIFEST_SHA256 = "6d64352b47ae0ec09ea2af6763be74d94b451754caf5e096dc98b5c920c67736"
PUBLIC_EXPORT_SHA256 = "66cc83c7a0aacb4b11ea2b203ff47136c5c960ba9d7b33eb620ff5eac63b7723"
INTEGRITY_RECEIPT_SHA256 = "d7864bbe0e21748a160c6750f992bc9d211ea50691cc084cee2b0f1da05c152b"
FROZEN_METADATA_SHA256 = "46e58b41172c4231620982494db34773827871a182972990887d9f9e4854e4db"
COMPLETION_CHECK_SHA256 = "d5fb03a15b520b335f30cf274750a27039c61416b7d233d4d30761ec5cc9db99"
COHORT_SHA256 = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"

REPORT = Path("reports/strict_qwen35")
MANIFEST_PATH = REPORT / "run_manifest.json"
PUBLIC_PATH = REPORT / "scoring_B_only/public_finals_metrics.json"
METADATA_PATH = REPORT / "scoring_B_only/frozen_metadata.json"
RECEIPT_PATH = REPORT / "operations/b_shutdown_integrity_receipt.json"
COMPLETION_PATH = REPORT / "operations/b_completion_check.json"
COHORT_PATH = Path("data/queries_100.jsonl")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_METRIC_KEYS = {
    "n_turns", "elapsed_seconds", "input_tokens", "output_tokens",
    "cached_input_tokens", "reasoning_tokens", "model_total_tokens",
    "api_attempts", "api_successful_calls", "api_attempts_without_usage",
    "rerank_calls", "scoring_calls", "scored_passages", "reranker_input_tokens",
    "scoring_seconds", "preparation_seconds",
    "combined_model_and_reranker_input_tokens",
}
_RECORD_KEYS = {
    "query_id", "question", "gold_answer", "final_text", "final_sha256",
    "terminal_row_sha256", "terminated_by", "metrics", "reranker_summary",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _contract_digest(value):
    # strict_experiment.strict_prompt uses the default JSON separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _exact(left, right):
    """JSON equality including bool/int/float distinctions, recursively."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _json_bytes(data):
    def invalid_constant(_):
        raise ValueError("Non-finite JSON number")
    return json.loads(data, object_pairs_hook=_object_pairs, parse_constant=invalid_constant)


def _load(root, relative, expected_sha=None):
    path = root / relative
    _require(path.is_file(), f"Missing frozen control file: {relative}")
    data = path.read_bytes()
    if expected_sha is not None:
        _require(_sha(data) == expected_sha, f"Frozen file SHA256 mismatch: {relative}")
    value = _json_bytes(data)
    _require(type(value) is dict, f"Expected JSON object: {relative}")
    return value


def _check_digest(value, field, *, contract=False):
    _require(type(value) is dict, f"Missing object for {field}")
    claimed = value.get(field)
    _require(isinstance(claimed, str) and bool(_HEX.fullmatch(claimed)), f"Invalid {field}")
    payload = {k: v for k, v in value.items() if k != field}
    actual = _contract_digest(payload) if contract else _digest(payload)
    _require(actual == claimed, f"Canonical {field} checksum mismatch")


def _control_contract():
    try:
        module = importlib.import_module("strict_ablations.prompt")
    except ModuleNotFoundError as exc:
        if exc.name != "strict_ablations.prompt":
            raise
        # Temporary compatibility while the new prompt module is being added.
        module = importlib.import_module("strict_experiment.strict_prompt")
        return module.build_contract("B")
    return module.build_contract("control")


def _numeric(value, name, *, integer=True):
    _require(type(value) is int if integer else type(value) in (int, float),
             f"Missing/non-numeric control ledger field: {name}")
    _require(value >= 0 and math.isfinite(value), f"Invalid control ledger field: {name}")


def _validate_metrics(record, max_turns):
    metrics, rerank = record.get("metrics"), record.get("reranker_summary")
    _require(type(metrics) is dict and set(metrics) == _METRIC_KEYS,
             "Incomplete numeric ledger coverage")
    _require(type(rerank) is dict and bool(rerank), "Missing numeric reranker ledger")
    for key, value in metrics.items():
        _numeric(value, key, integer=not key.endswith("seconds"))
    for key, value in rerank.items():
        _require(type(key) is str, "Invalid reranker ledger key")
        _numeric(value, "reranker_summary." + key, integer=not key.endswith("seconds"))
    _require(0 < metrics["n_turns"] <= max_turns, "Control turn count exceeds frozen budget")
    _require(metrics["cached_input_tokens"] <= metrics["input_tokens"]
        and metrics["reasoning_tokens"] <= metrics["output_tokens"], "Invalid inclusive token subsets")
    _require(metrics["model_total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"],
             "Control model token total mismatch")
    _require(metrics["combined_model_and_reranker_input_tokens"] ==
             metrics["input_tokens"] + metrics["reranker_input_tokens"], "Combined input token total mismatch")
    _require(metrics["api_successful_calls"] <= metrics["api_attempts"]
        and metrics["api_attempts_without_usage"] <= metrics["api_attempts"], "Invalid API ledger counts")
    for target, source in (("rerank_calls", "rerank_calls"), ("scoring_calls", "scoring_calls"),
            ("scored_passages", "scored_passages"), ("reranker_input_tokens", "input_tokens"),
            ("scoring_seconds", "scoring_seconds"), ("preparation_seconds", "preparation_seconds")):
        _require(source in rerank and metrics[target] == rerank[source], "Reranker ledger counters disagree")
    _require(metrics["scoring_calls"] <= metrics["rerank_calls"], "Scoring calls exceed rerank calls")
    if record["terminated_by"] == "max_turns":
        _require(metrics["n_turns"] == max_turns, "max_turns termination differs from frozen budget")


def _expected_receipt_paths(qid):
    base = "runs/strict_qwen35/B/"
    return {
        base + f"_per_query/qid_{qid}.json",
        base + f"request_contracts/qid_{qid}.json",
        base + f"api_ledgers/qid_{qid}.jsonl",
        base + f"_traces/qid_{qid}/single.json",
        base + f"_checkpoints/qid_{qid}.json",
        base + f"_maintenance/completed/qid_{qid}.json",
        base + f"_maintenance/source_manifests/qid_{qid}.json",
    }


def verify_control_bundle(root, *, common_config=None) -> dict:
    """Verify and return the already-run B100, without reading judge labels.

    ``root`` is the original experiment directory containing ``data`` and
    ``reports/strict_qwen35``. Supplied common_config must equal *all* frozen
    config values and types, including paths; no migration or budget override
    is performed. ValueError rejects incomplete, stale, or changed bundles.
    The returned contract hash is the original strict B contract identity.
    """
    root = Path(root)
    manifest = _load(root, MANIFEST_PATH)
    _check_digest(manifest, "manifest_sha256")
    _require(manifest["manifest_sha256"] == MANIFEST_SHA256, "Not the frozen minimal-prompt B100 manifest")
    config = manifest.get("config")
    _require(type(config) is dict and config.get("model") == MODEL, "Control model differs")
    _require(_digest(config) == manifest.get("config_sha256"), "Frozen config checksum mismatch")
    _require(type(config.get("max_turns")) is int and config["max_turns"] > 0, "Missing frozen turn budget")
    if common_config is not None:
        _require(_exact(common_config, config), "common_config must exactly equal the frozen manifest config; path migration requires explicit review")

    public = _load(root, PUBLIC_PATH, PUBLIC_EXPORT_SHA256)
    metadata = _load(root, METADATA_PATH, FROZEN_METADATA_SHA256)
    receipt = _load(root, RECEIPT_PATH, INTEGRITY_RECEIPT_SHA256)
    completion = _load(root, COMPLETION_PATH, COMPLETION_CHECK_SHA256)
    _check_digest(receipt, "receipt_sha256")
    _require(public.get("version") == "strict-B100-public-finals-metrics-v1"
        and receipt.get("version") == "strict-B100-shutdown-integrity-v1"
        and receipt.get("arm") == "B", "Wrong control export/receipt version or arm")

    cohort_file = root / COHORT_PATH
    _require(cohort_file.is_file(), "Missing original fixed100 cohort")
    cohort_bytes = cohort_file.read_bytes()
    _require(_sha(cohort_bytes) == COHORT_SHA256 == manifest.get("cohort_sha256"), "Fixed100 cohort SHA256 mismatch")
    cohort = [_json_bytes(line) for line in cohort_bytes.splitlines() if line.strip()]
    _require(len(cohort) == 100 and all(type(row) is dict and type(row.get("query_id")) is int for row in cohort),
             "Fixed100 cohort coverage must be 100")
    qids = [row["query_id"] for row in cohort]
    _require(len(set(qids)) == 100 and _exact(manifest.get("query_ids"), qids), "Fixed100 order/coverage mismatch")

    binding = {"manifest_sha256": MANIFEST_SHA256, "model": MODEL, "cohort_sha256": COHORT_SHA256}
    _require(all(public.get(k) == metadata.get(k) == v for k, v in binding.items()), "Control metadata binding mismatch")
    epoch = public.get("epoch_sha256")
    _require(isinstance(epoch, str) and bool(_HEX.fullmatch(epoch))
        and metadata.get("epoch_sha256") == receipt.get("epoch_sha256") == completion.get("epoch_sha256") == epoch,
        "Control epoch mismatch")
    _require(receipt.get("manifest_sha256") == MANIFEST_SHA256
        and receipt.get("config_sha256") == manifest["config_sha256"], "Integrity receipt manifest/config mismatch")
    _require(receipt.get("a_artifacts_unchanged") is True
        and receipt.get("a_before_sha256") == receipt.get("a_after_sha256"), "B-only receipt does not preserve unused A")
    _require(_exact(metadata.get("original_config"), config), "Metadata original_config differs")

    contracts = manifest.get("contracts")
    _require(type(contracts) is dict and type(contracts.get("B")) is dict, "Missing strict B contract")
    frozen_contract = contracts["B"]
    _check_digest(frozen_contract, "contract_sha256", contract=True)
    _require(_exact(metadata.get("B_contract"), frozen_contract), "Metadata B contract differs")
    candidate = _control_contract()
    _require(type(candidate) is dict, "Missing new control contract")
    for field in ("system_prompt", "tool_specs", "user_prompt_template"):
        _require(field in frozen_contract and _exact(candidate.get(field), frozen_contract[field]),
                 "New control input differs from strict B: " + field)
    _require(_sha(frozen_contract["system_prompt"].encode()) == frozen_contract.get("system_prompt_sha256")
        and _contract_digest(frozen_contract["tool_specs"]) == frozen_contract.get("tool_specs_sha256"), "Strict B prompt/tool hashes differ")

    _require(completion.get("ready_for_shutdown") is True
        and completion.get("phase") == "B100_verified_public_export_saved"
        and type(completion.get("B_terminal_files")) is int and completion["B_terminal_files"] == 100
        and completion.get("a_artifacts_unchanged") is True
        and completion.get("manifest_sha256") == MANIFEST_SHA256, "Completion check does not attest B100")
    files = completion.get("files")
    _require(type(files) is dict, "Completion file bindings missing")
    for relative, pin in ((PUBLIC_PATH, PUBLIC_EXPORT_SHA256), (METADATA_PATH, FROZEN_METADATA_SHA256),
                          (RECEIPT_PATH, INTEGRITY_RECEIPT_SHA256)):
        # Original cloud receipt lived in operations/epoch_20261022; local
        # copies are in operations. Match its unique basename, never open it.
        matches = [v for k, v in files.items() if type(k) is str and PurePosixPath(k).name == relative.name]
        _require(matches == [pin], "Completion check frozen file binding differs")

    records, items = public.get("records"), receipt.get("items")
    _require(type(records) is list and type(items) is list and len(records) == len(items) == 100,
             "Public/receipt coverage must be 100")
    _require(all(type(item) is dict for item in items), "Invalid integrity receipt item")
    _require(_exact([item.get("query_id") for item in items], qids), "Integrity receipt fixed100 order/coverage mismatch")
    _require(all(type(record) is dict for record in records), "Invalid public record")
    _require(_exact([record.get("query_id") for record in records], qids), "Public fixed100 order/coverage mismatch")
    for query, record, item in zip(cohort, records, items):
        _require(set(record) == _RECORD_KEYS, "Unexpected public fields; judge labels must not be reused")
        _require(type(query.get("query")) is str and type(query.get("answer")) is str
            and record.get("question") == query["query"] and record.get("gold_answer") == query["answer"],
            "Fixed100 question/gold differs")
        final = record.get("final_text")
        _require(type(final) is str and record.get("final_sha256") == _sha(final.encode("utf-8")), "Per-question final SHA256 mismatch")
        _require(isinstance(record.get("terminal_row_sha256"), str)
            and bool(_HEX.fullmatch(record["terminal_row_sha256"])), "Missing terminal row hash")
        _require(record.get("terminated_by") in {"done", "max_turns", "wall_clock_timeout"}, "Non-natural terminal result")
        attested = item.get("files")
        _require(type(attested) is dict and set(attested) == _expected_receipt_paths(query["query_id"])
            and all(type(v) is str and bool(_HEX.fullmatch(v)) for v in attested.values()),
            "Integrity receipt lacks per-question contract/ledger/trace coverage")
        _validate_metrics(record, config["max_turns"])

    return {"version": VERSION, "manifest_sha256": MANIFEST_SHA256,
        "public_export_sha256": PUBLIC_EXPORT_SHA256,
        "integrity_receipt_sha256": INTEGRITY_RECEIPT_SHA256,
        "model": MODEL, "cohort_sha256": COHORT_SHA256,
        "contract_sha256": frozen_contract["contract_sha256"],
        "config_sha256": manifest["config_sha256"], "common_config": copy.deepcopy(config),
        "records": copy.deepcopy(records)}
