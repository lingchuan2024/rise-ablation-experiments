#!/usr/bin/env python3
"""Fresh blinded comparison of the reused baseline100 and completed new B100.

This administrative adapter never reads trajectories, imports historical
judgements, calls a model, runs a question, or modifies frozen experiment code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
from strict_experiment import analyze

VERSION = "reused-baseline-new-B-fresh-comparison-v1"
MANIFEST_SHA = "6d64352b47ae0ec09ea2af6763be74d94b451754caf5e096dc98b5c920c67736"
EPOCH_SHA = "9ec063fc1f79c27c7470a3ac19ff9b159fb28e0b331b54479795155aad5cef5d"
B_CHECKER_SHA = "3672d522ebc2372204834c1d40ff9e944975d85c0668e4e2a99592753c4e8ddf"
BASELINE_DIR = ROOT.parent / "baseline_qwen35_docker34_rerun_20261019"
OLD_ROWS = WORKSPACE / "analysis/latest_20261002/qwen/runs/baseline/_per_query"
V14_FILE = ROOT.parent / "independent_rerank_v14_20261014/analysis_qwen35_full100_20261019/reports/evaluation.json"
PINS = {
    "selection": "5bcb21e2f299204cd01f5606fbcf02b889e35d2ff9b1c2a93cb9350c175e3875",
    "fixed34": "61eb4f07621fc15ed4fb7cc61d2e3f973dd58d58eaa3a7cc26bd025e972ebb89",
    "retained66": "09a925ecee046c3833316e6828e336b75d05ee6115d8babde4af46c082cd4f1f",
    "v14": "4e458aa7d2ec2edfe15257c878ce4b4f143fce65a3a71ce6893e34427dfe02af",
}
GROUPS = ("historical_baseline", "new_B")
HEX = re.compile(r"[0-9a-f]{64}")
METRICS = ("n_turns", "elapsed_seconds", "input_tokens", "raw_non_cached_input_tokens",
    "cached_input_tokens", "output_tokens", "reasoning_tokens", "model_total_tokens",
    "api_attempts", "api_successful_calls", "api_attempts_without_usage", "rerank_calls",
    "scoring_calls", "scored_passages", "reranker_input_tokens", "scoring_seconds",
    "preparation_seconds", "combined_model_and_reranker_input_tokens")
LIMITATIONS = [
    "The baseline is reused from historical66 plus fixed34. New B ran later; this is not a concurrent, fully environment-matched causal experiment.",
    "One stochastic trajectory per question and100 questions do not establish a guaranteed gain; report discordant pairs and exact McNemar without selecting outcomes.",
    "Model input includes recorded cache. Reasoning is a subset of output, not an additional token total. A saved reasoning/cache zero is only a recorded zero, not proof of absent reasoning or cache use.",
    "New B usage includes recorded API responses only. Unlogged or undelivered requests around the restart remain unknown and must not be assigned zero.",
    "Elapsed question time includes service contention. Rerank preparation/scoring are elapsed-time components, not additional wall time to add again.",
    "Historical v14 is an overall enhanced system. Its archived61/100 is separate context and does not isolate rerank's incremental effect.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(path, expected):
    path = Path(path)
    require(analyze.file_sha(path) == expected, "A pinned source file changed")
    return analyze.load(path)


def checked_object(value, field):
    analyze.checked_digest(value, field, "Comparison artifact")
    return value


def seal(value, field):
    return {**value, field: analyze.digest(value)}


def safe_write(path, value, *, private=False):
    path = Path(path)
    if path.exists():
        require(analyze.load(path) == value, "Preserve the existing artifact; use a new output directory")
    else:
        analyze.write_new(path, value, private=private)
    if private:
        path.chmod(0o600)


def validate_final(row, query):
    require(type(row.get("query_id")) is int and row["query_id"] == query["query_id"], "Question identity differs")
    require(row.get("question") == query["query"], "Question text differs from the fixed cohort")
    require(row.get("gold_answer") == query["answer"], "Gold answer differs from the fixed cohort")
    require(isinstance(row.get("final_text"), str), "A public final string is required")
    return analyze.text_sha(row["final_text"])


def legacy_metrics(row):
    usage = row["agent_usage"]
    raw = analyze.nonnegative(usage.get("input_tokens"), "saved input", integer=True)
    cached = analyze.nonnegative(usage.get("cached_input_tokens"), "saved cached", integer=True)
    output = analyze.nonnegative(usage.get("output_tokens"), "saved output", integer=True)
    reasoning = analyze.nonnegative(usage.get("reasoning_tokens"), "saved reasoning", integer=True)
    require(reasoning <= output, "Reasoning is a subset of output")
    return {"n_turns": analyze.nonnegative(row.get("n_turns"), "turns", integer=True),
        "elapsed_seconds": analyze.nonnegative(row.get("elapsed_seconds"), "elapsed"),
        "input_tokens": raw + cached, "raw_non_cached_input_tokens": raw,
        "cached_input_tokens": cached, "output_tokens": output, "reasoning_tokens": reasoning,
        "model_total_tokens": raw + cached + output,
        "api_attempts": None, "api_successful_calls": None, "api_attempts_without_usage": None,
        "rerank_calls": 0, "scoring_calls": 0, "scored_passages": 0,
        "reranker_input_tokens": 0, "scoring_seconds": 0, "preparation_seconds": 0,
        "combined_model_and_reranker_input_tokens": raw + cached}


def load_baseline(cohort):
    selection_path = BASELINE_DIR / "updated_baseline100_score.json"
    fixed_path = BASELINE_DIR / "verified34_results.json"
    selection = checked_file(selection_path, PINS["selection"])
    fixed = checked_file(fixed_path, PINS["fixed34"])
    selected = {row["query_id"]: row["replaced_by_fixed34_rerun"] for row in selection["records"]}
    require(len(selected) == 100 and set(selected) == set(cohort), "Baseline selection must cover exactly fixed100")
    require(all(type(value) is bool for value in selected.values()), "Invalid baseline replacement selector")
    replacements = {row["query_id"]: (index, row) for index, row in enumerate(fixed["per_query"])}
    require(len(fixed["per_query"]) == len(replacements) == 34, "The fixed34 source has duplicates or missing rows")
    require(set(replacements) == {qid for qid, replace in selected.items() if replace}, "Do not select old/new outcomes")
    records, retained_files = [], []
    for qid, query in cohort.items():
        if qid in replacements:
            index, row = replacements[qid]
            source = {"path": str(fixed_path), "sha256": PINS["fixed34"], "json_pointer": f"/per_query/{index}"}
        else:
            path = OLD_ROWS / f"qid_{qid}.json"
            row = analyze.load(path)
            sha = analyze.file_sha(path)
            retained_files.append({"query_id": qid, "sha256": sha})
            source = {"path": str(path), "sha256": sha}
        final_sha = validate_final(row, query)
        require(row.get("model") == analyze.MODEL, "Historical solver model differs")
        # Copy only the public final and numerical metrics. Historical judge,
        # is_correct and selected-answer fields are deliberately never copied.
        records.append({"source_kind": GROUPS[0], "query_id": qid,
            "question": query["query"], "gold_answer": query["answer"],
            "final_text": row["final_text"], "final_sha256": final_sha,
            "terminated_by": row["terminated_by"], "metrics": legacy_metrics(row), "origin": source})
    retained_files.sort(key=lambda item: item["query_id"])
    require(len(retained_files) == 66 and analyze.digest(retained_files) == PINS["retained66"], "Retained66 source bytes changed")
    return records, {"selection": {"path": str(selection_path), "sha256": PINS["selection"]},
        "fixed34": {"path": str(fixed_path), "sha256": PINS["fixed34"]},
        "retained66_file_hashes": retained_files,
        "retained66_sha256": PINS["retained66"], "selection_rule": "fixed34 replaces all34; retain the remaining66; no best-of selection"}


def validate_b_export(public, metadata, manifest, cohort):
    require(public.get("version") == "strict-B100-public-finals-metrics-v1", "Use the completed B-only public export")
    require(public.get("manifest_sha256") == metadata.get("manifest_sha256") == MANIFEST_SHA,
            "New B does not bind the original experiment manifest")
    epoch = metadata.get("epoch_sha256")
    require(epoch == EPOCH_SHA and public.get("epoch_sha256") == epoch, "B export does not bind the independently verified restart epoch")
    require(public.get("model") == metadata.get("model") == analyze.MODEL, "B export solver model differs")
    require(public.get("cohort_sha256") == metadata.get("cohort_sha256") == analyze.COHORT_SHA256, "B export cohort differs")
    require(metadata.get("original_config") == manifest["config"] and metadata.get("B_contract") == manifest["contracts"]["B"], "B frozen configuration or tool contract differs")
    require(metadata.get("rubric") == analyze.rubric_document()
            and metadata.get("rubric_sha256") == analyze.digest(analyze.rubric_document()), "B rubric differs")
    rows = public.get("records", [])
    require(len(rows) == len(cohort), "B export must include every fixed question")
    records, seen = [], set()
    for row in rows:
        qid = row.get("query_id")
        require(type(qid) is int and qid in cohort and qid not in seen, "B question is unknown or duplicated")
        seen.add(qid)
        require(validate_final(row, cohort[qid]) == row.get("final_sha256"), "B final SHA256 differs")
        require(isinstance(row.get("terminal_row_sha256"), str) and HEX.fullmatch(row["terminal_row_sha256"]), "B terminal provenance is missing")
        require(isinstance(row.get("terminated_by"), str) and row["terminated_by"]
                and not row["terminated_by"].startswith(("api_error", "infrastructure", "solver_api_failed")), "B infrastructure interruptions are not natural terminals")
        require(isinstance(row.get("metrics"), dict), "B numerical metrics are missing")
        metrics = {key: row["metrics"].get(key) for key in METRICS}
        for key, value in metrics.items():
            if value is not None:
                analyze.nonnegative(value, key, integer=key not in {"elapsed_seconds", "scoring_seconds", "preparation_seconds"})
        require(all(metrics[key] is not None for key in ("n_turns", "elapsed_seconds", "input_tokens", "output_tokens", "rerank_calls", "scored_passages", "reranker_input_tokens", "scoring_seconds", "preparation_seconds")), "B required numerical metrics are missing")
        require(metrics["model_total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"], "B model token total differs")
        require(metrics["cached_input_tokens"] is None or metrics["cached_input_tokens"] <= metrics["input_tokens"], "Cached input is a subset of input")
        require(metrics["reasoning_tokens"] is None or metrics["reasoning_tokens"] <= metrics["output_tokens"], "Reasoning is a subset of output")
        if metrics["raw_non_cached_input_tokens"] is None and metrics["cached_input_tokens"] is not None:
            metrics["raw_non_cached_input_tokens"] = metrics["input_tokens"] - metrics["cached_input_tokens"]
        records.append({"source_kind": GROUPS[1], "query_id": qid,
            "question": row["question"], "gold_answer": row["gold_answer"],
            "final_text": row["final_text"], "final_sha256": row["final_sha256"],
            "terminated_by": row["terminated_by"], "metrics": metrics,
            "origin": {"terminal_row_sha256": row["terminal_row_sha256"]}})
    require(seen == set(cohort), "B question coverage differs")
    return records


def verify_completion_proof(proof_path, b_finals, b_metadata, integrity_path, *, config_sha256=None):
    """Independently bind local exports to the saved remote checker stdout.

    The export cannot authenticate itself by recomputing its own final/hash
    fields: its whole-file bytes must match the separately saved remote proof.
    """
    proof_path, integrity_path = Path(proof_path), Path(integrity_path)
    proof = analyze.load(proof_path)
    expected = {"ready_for_shutdown": True, "manifest_sha256": MANIFEST_SHA,
        "epoch_sha256": EPOCH_SHA, "a_artifacts_unchanged": True, "B_terminal_files": 100}
    require(all(proof.get(key) == value and type(proof.get(key)) is type(value)
                for key, value in expected.items()), "The independently saved completion check did not pass")
    files = proof.get("files")
    require(isinstance(files, dict), "Completion proof file hashes are missing")
    selected = {}
    targets = {
        "public_finals": ("/scoring_B_only/public_finals_metrics.json", Path(b_finals)),
        "metadata": ("/scoring_B_only/frozen_metadata.json", Path(b_metadata)),
        "integrity": ("/epoch_20261022/b_shutdown_integrity_receipt.json", integrity_path),
    }
    for name, (suffix, path) in targets.items():
        matches = [(remote, sha) for remote, sha in files.items()
                   if isinstance(remote, str) and remote.startswith("/") and remote.endswith(suffix)]
        require(len(matches) == 1, "A required remote completion-proof file hash is missing or ambiguous")
        remote, sha = matches[0]
        require(isinstance(sha, str) and HEX.fullmatch(sha)
                and analyze.file_sha(path) == sha, "A local B export differs from the independent remote proof")
        selected[name] = {"remote_path": remote, "local_path": str(path.resolve()), "expected_sha256": sha}
    integrity = checked_object(analyze.load(integrity_path), "receipt_sha256")
    require(integrity.get("version") == "strict-B100-shutdown-integrity-v1" and integrity.get("arm") == "B"
            and integrity.get("manifest_sha256") == MANIFEST_SHA and integrity.get("epoch_sha256") == EPOCH_SHA
            and integrity.get("checker_sha256") == B_CHECKER_SHA
            and integrity.get("a_artifacts_unchanged") is True
            and integrity.get("a_before_sha256") == integrity.get("a_after_sha256"), "The independent B integrity receipt differs")
    require(isinstance(integrity.get("a_before_sha256"), str) and HEX.fullmatch(integrity["a_before_sha256"]), "A integrity binding is missing")
    items = integrity.get("items", [])
    require(len(items) == 100 and len({row["query_id"] for row in items}) == 100, "The B integrity receipt must bind100 unique questions")
    if config_sha256 is not None:
        require(integrity.get("config_sha256") == config_sha256, "Integrity receipt configuration differs")
    return {"proof": {"path": str(proof_path.resolve()), "sha256": analyze.file_sha(proof_path)},
        "expected": {**expected, "checker_sha256": B_CHECKER_SHA, "config_sha256": config_sha256},
        "verified_files": selected, "integrity_receipt_sha256": integrity["receipt_sha256"]}


def historical_context():
    doc = checked_file(V14_FILE, PINS["v14"])
    require(doc.get("strict", {}).get("n") == 100 and doc["strict"]["correct"] == 61, "Archived v14 context differs")
    return {"source": str(V14_FILE), "source_sha256": PINS["v14"],
        "archived_correct": 61, "archived_n": 100,
        "interpretation": "Historical overall enhanced v14 system; this score is not a fresh label source or a rerank-only causal effect."}


def case_id(salt, binding_sha, source, qid, final_sha):
    message = json.dumps([binding_sha, source, qid, final_sha], separators=(",", ":")).encode()
    return hmac.new(salt, message, hashlib.sha256).hexdigest()[:24]


def make_bundle(records, provenance, context, *, salt_hex=None):
    salt_hex = secrets.token_hex(32) if salt_hex is None else salt_hex
    require(isinstance(salt_hex, str) and HEX.fullmatch(salt_hex), "Use a private random32-byte salt")
    salt, binding = bytes.fromhex(salt_hex), analyze.digest(provenance)
    qids = {group: {row["query_id"] for row in records if row["source_kind"] == group} for group in GROUPS}
    require(qids[GROUPS[0]] == qids[GROUPS[1]] and bool(qids[GROUPS[0]])
            and len(records) == 2 * len(qids[GROUPS[0]]), "Exactly one public final per group/question is required")
    public_rows, mapping, materials = [], [], []
    for row in records:
        require(row["source_kind"] in GROUPS and analyze.text_sha(row["final_text"]) == row["final_sha256"], "Public final bytes differ")
        cid = case_id(salt, binding, row["source_kind"], row["query_id"], row["final_sha256"])
        public_rows.append({"case_id": cid, "question": row["question"], "gold_answer": row["gold_answer"],
                            "final_text": row["final_text"], "final_sha256": row["final_sha256"]})
        mapping.append({"case_id": cid, "source_kind": row["source_kind"],
                        "query_id": row["query_id"], "final_sha256": row["final_sha256"]})
        materials.append({"case_id": cid, "terminated_by": row["terminated_by"],
                          "metrics": row["metrics"], "origin": row["origin"]})
    public_rows.sort(key=lambda row: row["case_id"])
    mapping.sort(key=lambda row: row["case_id"])
    materials.sort(key=lambda row: row["case_id"])
    require(len({row["case_id"] for row in public_rows}) == len(records), "Duplicate blind case ID")
    public = seal({"version": "public-final-blind-review-v1", "rubric": analyze.rubric_document(),
        "rubric_sha256": analyze.digest(analyze.rubric_document()), "records": public_rows}, "packet_sha256")
    private = seal({"version": VERSION, "packet_sha256": public["packet_sha256"],
        "source_binding_sha256": binding, "shuffle_salt_hex": salt_hex,
        "query_ids": sorted(qids[GROUPS[0]]), "records": mapping}, "mapping_sha256")
    material = seal({"version": VERSION, "packet_sha256": public["packet_sha256"],
        "mapping_sha256": private["mapping_sha256"], "source_binding": provenance,
        "historical_v14_context": context, "records": materials}, "materials_sha256")
    return public, private, material


def validate_bundle(public, private, material):
    checked_object(public, "packet_sha256"); checked_object(private, "mapping_sha256"); checked_object(material, "materials_sha256")
    require(public.get("version") == "public-final-blind-review-v1"
            and private.get("version") == material.get("version") == VERSION, "Comparison artifact version differs")
    analyze.validate_rubric(public.get("rubric"))
    require(public.get("rubric_sha256") == analyze.digest(analyze.rubric_document()), "Frozen rubric hash differs")
    require(public["packet_sha256"] == private["packet_sha256"] == material["packet_sha256"]
            and material["mapping_sha256"] == private["mapping_sha256"], "Comparison artifacts do not match")
    require(private["source_binding_sha256"] == analyze.digest(material["source_binding"]), "Source binding changed")
    salt = bytes.fromhex(private["shuffle_salt_hex"])
    require(len(salt) == 32, "Invalid private shuffle salt")
    p = {row["case_id"]: row for row in public["records"]}
    m = {row["case_id"]: row for row in private["records"]}
    numbers = {row["case_id"]: row for row in material["records"]}
    require(len(p) == len(public["records"]) == len(m) == len(private["records"])
            == len(numbers) == len(material["records"])
            and set(p) == set(m) == set(numbers), "Duplicate, missing or mixed comparison cases")
    pairs = {}
    for cid, row in m.items():
        require(row["source_kind"] in GROUPS and type(row["query_id"]) is int, "Invalid private mapping")
        require(row["final_sha256"] == p[cid]["final_sha256"] == analyze.text_sha(p[cid]["final_text"]), "Mapped final SHA differs")
        require(cid == case_id(salt, private["source_binding_sha256"], row["source_kind"], row["query_id"], row["final_sha256"]), "Blind case identity differs")
        key = (row["source_kind"], row["query_id"])
        require(key not in pairs, "Duplicate group/question")
        pairs[key] = cid
    expected = {(group, qid) for group in GROUPS for qid in private["query_ids"]}
    require(set(pairs) == expected, "Every question needs both anonymous finals")
    for qid in private["query_ids"]:
        x, y = (p[pairs[(group, qid)]] for group in GROUPS)
        require(x["question"] == y["question"] and x["gold_answer"] == y["gold_answer"], "Paired questions/gold differ")
    return p, m, numbers, pairs


def validate_reviews(submission, public, private, material):
    p, m, numbers, pairs = validate_bundle(public, private, material)
    require(submission.get("packet_sha256") == public["packet_sha256"]
            and submission.get("rubric_sha256") == public["rubric_sha256"], "Reviews bind another packet/rubric")
    require(isinstance(submission.get("reviewer"), str) and submission["reviewer"].strip(), "Identify the fresh blinded reviewer")
    labels = {}
    for row in submission.get("records", []):
        cid = row.get("case_id")
        require(cid in p and cid not in labels, "Unknown or duplicate reviewed ID")
        require(row.get("final_sha256") == p[cid]["final_sha256"], "Reviewed final SHA differs")
        require(type(row.get("correct")) is bool and isinstance(row.get("reason"), str)
                and row["reason"].strip(), "Each fresh judgement needs Boolean correct and a nonempty reason")
        require(not row["correct"] or bool(p[cid]["final_text"].strip()), "An empty final cannot be correct")
        labels[cid] = row["correct"]
    require(set(labels) == set(p), "A fresh judgement is required for every anonymous case")
    return labels, numbers, pairs


def numerical_summary(values):
    valid = [value for value in values if value is not None]
    for value in valid:
        analyze.nonnegative(value, "aggregate metric")
    return {"recorded": len(valid), "missing": len(values) - len(valid),
        "sum": sum(valid) if valid else None,
        "mean": statistics.mean(valid) if valid else None,
        "median": statistics.median(valid) if valid else None}


def summarize_reviews(submission, public, private, material):
    labels, numbers, pairs = validate_reviews(submission, public, private, material)
    qids, groups = private["query_ids"], {}
    for group in GROUPS:
        ids = [pairs[(group, qid)] for qid in qids]
        count = sum(labels[cid] for cid in ids)
        groups[group] = {"n": len(ids), "correct": count, "accuracy": count / len(ids),
            "termination_counts": dict(Counter(numbers[cid]["terminated_by"] for cid in ids)),
            "metrics": {key: numerical_summary([numbers[cid]["metrics"].get(key) for cid in ids]) for key in METRICS}}
    transitions = Counter()
    per_question = []
    for qid in qids:
        old, new = (labels[pairs[(group, qid)]] for group in GROUPS)
        label = "both_correct" if old and new else "baseline_only" if old else "B_only" if new else "both_incorrect"
        transitions[label] += 1
        per_question.append({"query_id": qid, "historical_baseline_correct": old, "new_B_correct": new})
    comparison = {key: transitions[key] for key in ("both_correct", "baseline_only", "B_only", "both_incorrect")}
    comparison.update(delta_correct=groups[GROUPS[1]]["correct"] - groups[GROUPS[0]]["correct"],
        delta_accuracy=groups[GROUPS[1]]["accuracy"] - groups[GROUPS[0]]["accuracy"],
        mcnemar_exact_two_sided=analyze.mcnemar_exact(transitions["baseline_only"], transitions["B_only"]))
    return seal({"version": VERSION, "source": "fresh_blinded_review",
        "reviewer": submission["reviewer"], "packet_sha256": public["packet_sha256"],
        "mapping_sha256": private["mapping_sha256"], "materials_sha256": material["materials_sha256"],
        "submission_sha256": analyze.digest(submission), "rubric_sha256": public["rubric_sha256"],
        "groups": groups, "paired_comparison": comparison, "paired_labels": per_question,
        "historical_v14_context": material["historical_v14_context"],
        "baseline_selection": "historical66 plus fixed34; interrupted new-run A artifacts are unused",
        "cost_usd": None, "cost_note": "No monetary tariff is applied; historical zero placeholders do not mean free.",
        "limitations": LIMITATIONS}, "summary_sha256")


def build(args):
    manifest, cohort, _ = analyze.load_experiment(ROOT)
    require(manifest["manifest_sha256"] == MANIFEST_SHA and len(cohort) == 100, "Use the original fixed100 experiment")
    completion_proof = verify_completion_proof(args.completion_proof, args.b_finals,
        args.b_metadata, args.b_integrity, config_sha256=manifest["config_sha256"])
    old, baseline_sources = load_baseline(cohort)
    public_b, metadata = analyze.load(args.b_finals), analyze.load(args.b_metadata)
    new = validate_b_export(public_b, metadata, manifest, cohort)
    context = historical_context()
    provenance = {"cohort_sha256": analyze.COHORT_SHA256, "B_manifest_sha256": MANIFEST_SHA,
        "B_epoch_sha256": metadata["epoch_sha256"], "baseline_sources": baseline_sources,
        "B_public_export": {"path": str(args.b_finals.resolve()), "sha256": analyze.file_sha(args.b_finals)},
        "B_metadata": {"path": str(args.b_metadata.resolve()), "sha256": analyze.file_sha(args.b_metadata)},
        "independent_remote_completion_proof": completion_proof,
        "B_usage_scope": public_b.get("usage_scope"),
        "recovery_audit_sha256": metadata.get("recovery_audit_sha256"),
        "historical_baseline_reported_score": {"correct": 52, "n": 100,
            "note": "Historical context only; no historical per-case verdict is imported."}}
    directory = args.out_dir
    private_path = directory / "private_map.json"
    require(not any((directory / name).exists() for name in ("blind_packet.json", "materials.private.json"))
            or private_path.exists(), "An existing packet/materials file has lost its private mapping; preserve it and use a new directory")
    salt = analyze.load(private_path)["shuffle_salt_hex"] if private_path.exists() else None
    public, private, materials = make_bundle(old + new, provenance, context, salt_hex=salt)
    require(len(public["records"]) == 200, "Build must export all200 cases")
    safe_write(private_path, private, private=True)
    safe_write(directory / "materials.private.json", materials, private=True)
    safe_write(directory / "historical_v14_context.json", context)
    safe_write(directory / "blind_packet.json", public, private=True)
    return {"status": "built", "cases": 200, "questions": 100,
            "public_packet": str(directory / "blind_packet.json"), "packet_sha256": public["packet_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/strict_qwen35/historical_comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--b-finals", type=Path, default=ROOT / "reports/strict_qwen35/scoring_B_only/public_finals_metrics.json")
    create.add_argument("--b-metadata", type=Path, default=ROOT / "reports/strict_qwen35/scoring_B_only/frozen_metadata.json")
    create.add_argument("--completion-proof", type=Path, default=ROOT / "reports/strict_qwen35/operations/b_completion_check.json")
    create.add_argument("--b-integrity", type=Path, default=ROOT / "reports/strict_qwen35/operations/b_shutdown_integrity_receipt.json")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--reviews", type=Path, required=True)
    summarize.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args)
    else:
        directory = args.out_dir
        public = analyze.load(directory / "blind_packet.json")
        require(len(public.get("records", [])) == 200, "Review the complete200-case packet")
        summary = summarize_reviews(analyze.load(args.reviews), public,
            analyze.load(directory / "private_map.json"), analyze.load(directory / "materials.private.json"))
        out = args.out or directory / "fresh_summary.json"
        safe_write(out, summary, private=True)
        result = {"status": "summarized", "output": str(out), "summary_sha256": summary["summary_sha256"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
