"""Fresh four-arm public-final scoring for three ablations and reused B100.

Only public finals are used for judgement. The suite may locally verify trace
and checkpoint integrity; hidden reasoning and historical judgements never enter
the review packet or scoring. Collection requires all 400 natural terminals.
Every output uses exclusive creation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import statistics

from strict_experiment import analyze as shared
from strict_ablations.control import verify_control_bundle
from strict_ablations.prompt import ARMS, CONTROL

VERSION = "strict-b100-ablations-analysis-v1"
ROOT = Path(__file__).resolve().parents[1]
MODEL = shared.MODEL
COHORT_SHA256 = shared.COHORT_SHA256
GROUPS = (CONTROL, *ARMS)
digest = shared.digest
text_sha = shared.text_sha
file_sha = shared.file_sha
load = shared.load
write_new = shared.write_new
mcnemar_exact = shared.mcnemar_exact
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_BINDINGS = {
    "manifest_sha256", "public_export_sha256", "integrity_receipt_sha256",
    "contract_sha256", "config_sha256", "model", "cohort_sha256",
}
_COUNT_METRICS = (
    "manual_rerank_calls", "independent_rerank_page_calls", "automatic_rerank_calls",
    "search_page_calls", "retrieval_calls", "retrieval_queries", "search_invocations",
    "automatic_rerank_errors", "automatic_empty_scopes",
)


def rubric_document():
    rubric = copy.deepcopy(shared.rubric_document())
    rubric["version"] = VERSION
    rubric["rules"] = [rule.replace("both anonymous answers", "all anonymous answers")
                       for rule in rubric["rules"]]
    return rubric


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value, label):
    _require(isinstance(value, str) and bool(_HEX.fullmatch(value)), label + " must be a SHA256")
    return value


def _suite(root, suite=None):
    if suite is None:
        from strict_ablations.runner import Suite
        suite = Suite(root=root)
    return suite


def _check_manifest(manifest):
    shared.checked_digest(manifest, "manifest_sha256", "Manifest")
    _require(manifest.get("config", {}).get("model") == MODEL, "Unexpected solver model")
    _require(manifest.get("config_sha256") == digest(manifest["config"]), "Config checksum differs")
    _require(manifest.get("cohort_sha256") == COHORT_SHA256, "Fixed100 cohort differs")
    _require(set(manifest.get("contracts", {})) == set(ARMS), "Exactly the three new-arm contracts are required")
    for contract in manifest["contracts"].values():
        _sha(contract.get("contract_sha256"), "Arm contract")
    _require(manifest.get("analysis_contract") == {
        "version": VERSION, "rubric_sha256": digest(rubric_document())},
        "Manifest must freeze this four-arm scoring rubric before execution")
    reference = manifest.get("control_reference")
    _require(isinstance(reference, dict) and _CONTROL_BINDINGS <= set(reference)
             and "records" not in reference and "common_config" not in reference,
             "Frozen public control reference bindings are required")


def load_experiment(root=None, *, suite=None, cohort_path=None):
    root = Path(root) if root is not None else ROOT
    suite = _suite(root, suite)
    manifest = suite.verify_manifest(resources=False)
    _check_manifest(manifest)
    # --root is the independent artifact directory, not a source checkout.
    # Use the same frozen cohort as the runner. A fixture/source supplied by
    # the suite or caller must still pass the exact pinned byte SHA below.
    source = cohort_path if cohort_path is not None else getattr(suite, "cohort_path", None)
    if source is None:
        from strict_experiment.runner import DATASET
        source = DATASET
    path = Path(source)
    _require(file_sha(path) == COHORT_SHA256, "Fixed100 cohort file SHA256 differs")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [query.get("query_id") for query in records]
    _require(len(qids) == 100 and all(type(qid) is int for qid in qids)
             and len(set(qids)) == 100, "Exactly 100 unique integer question IDs are required")
    if "query_ids" in manifest:
        _require(manifest["query_ids"] == qids, "Manifest fixed100 order differs")
    _require(all(isinstance(query.get("query"), str) and isinstance(query.get("answer"), str)
                 for query in records), "Frozen question/gold text is missing")
    return manifest, {query["query_id"]: query for query in records}, {
        "cohort_file_sha256": COHORT_SHA256,
        "verified_manifest_sha256": manifest["manifest_sha256"],
    }


def _count(value, name, *, optional=False):
    return shared.nonnegative(value, name, integer=True, optional=optional)


def _new_metrics(row, arm):
    metrics = shared.row_metrics(row, "B")
    usage = row.get("ablation_summary")
    _require(isinstance(usage, dict) and usage.get("arm") == arm,
             "New-arm usage counters are missing or belong to another arm")
    sources = {"manual_rerank_calls": "independent_rerank_calls",
               **{field: field for field in _COUNT_METRICS if field != "manual_rerank_calls"}}
    for field, source in sources.items():
        metrics[field] = _count(usage.get(source), "ablation_summary." + source)
    _require(metrics["rerank_calls"] == metrics["manual_rerank_calls"] + metrics["automatic_rerank_calls"],
             "Manual/automatic calls do not reconcile with the service ledger")
    _require(metrics["independent_rerank_page_calls"] <= metrics["manual_rerank_calls"],
             "Independent ranking page calls exceed manual calls")
    _require(metrics["scoring_calls"] <= metrics["rerank_calls"], "Scoring calls exceed rerank calls")
    if arm == "auto_rerank":
        _require(metrics["manual_rerank_calls"] == metrics["independent_rerank_page_calls"] == 0,
                 "Automatic arm cannot call independent rerank")
    else:
        _require(metrics["automatic_rerank_calls"] == metrics["automatic_rerank_errors"]
                 == metrics["automatic_empty_scopes"] == 0,
                 "Nonautomatic arm incurred automatic ranking")
    if arm != "search_paging":
        _require(metrics["search_page_calls"] == 0, "Document paging is exclusive to search_paging")
    return metrics


def _control_metrics(record):
    # verify_control_bundle already checks this complete numeric public ledger.
    # Repeat arithmetic checks so this normalization never silently changes it.
    metrics = copy.deepcopy(record["metrics"])
    for key, value in metrics.items():
        shared.nonnegative(value, "control.metrics." + key, integer=not key.endswith("seconds"),
                           optional=key in {"cached_input_tokens", "reasoning_tokens", "api_attempts_without_usage"})
    _require(metrics["model_total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"],
             "Control model token total differs")
    _require(metrics["combined_model_and_reranker_input_tokens"] ==
             metrics["input_tokens"] + metrics["reranker_input_tokens"], "Control input total differs")
    metrics.update({field: None for field in _COUNT_METRICS})
    metrics.update({"manual_rerank_calls": metrics["rerank_calls"],
                    "automatic_rerank_calls": 0, "search_page_calls": 0,
                    "automatic_rerank_errors": 0, "automatic_empty_scopes": 0,
                    "independent_rerank_page_calls": _count(
                        record.get("reranker_summary", {}).get("page_calls"),
                        "control.page_calls", optional=True)})
    return metrics


def _public_row(row, query, *, arm, metrics, row_sha256):
    final = row.get("final_text")
    _require(isinstance(final, str) and row.get("final_sha256") == text_sha(final),
             "Public final SHA256 differs")
    _sha(row_sha256, "Terminal row identity")
    _require(isinstance(row.get("terminated_by"), str) and bool(row["terminated_by"]),
             "A natural terminal reason is required")
    return {"arm": arm, "query_id": query["query_id"], "question": query["query"],
            "gold_answer": query["answer"], "final_text": final,
            "final_sha256": row["final_sha256"], "row_sha256": row_sha256,
            "terminated_by": row["terminated_by"], "metrics": metrics}


def _coverage(rows, cohort):
    _require(len(cohort) == 100, "Full fixed100 cohort is required")
    expected = {(arm, qid) for arm in GROUPS for qid in cohort}
    _require(set(rows) == expected, "Complete unique coverage requires 100 records per arm, 400 total")
    for (arm, qid), row in rows.items():
        _require(row.get("arm") == arm and row.get("query_id") == qid,
                 "Normalized row identity differs")
        _require(row.get("question") == cohort[qid]["query"]
                 and row.get("gold_answer") == cohort[qid]["answer"], "Question/gold differs")
        _require(isinstance(row.get("final_text"), str)
                 and row.get("final_sha256") == text_sha(row["final_text"]), "Final SHA256 differs")
        _sha(row.get("row_sha256"), "Terminal row identity")


def collect_rows(root, manifest, cohort, *, suite=None):
    """Collect 300 validated terminals and 100 pinned public control finals."""
    root, suite = Path(root), _suite(root, suite)
    _check_manifest(manifest)
    rows, provenance = {}, {}
    for arm in ARMS:
        paths = {qid: Path(suite.terminal_path(arm, qid)) for qid in cohort}
        expected_paths = {path.resolve() for path in paths.values()}
        actual_paths = {path.resolve() for parent in {path.parent for path in paths.values()}
                        for path in parent.glob("*.json")}
        _require(len(expected_paths) == 100 and actual_paths == expected_paths,
                 f"Complete unique terminal coverage is required for {arm}")
        for qid, query in cohort.items():
            path = paths[qid]
            row = load(path)
            suite.validate_terminal(row, manifest, arm, query)
            suite.verify_execution_artifacts(manifest, arm, query, row)
            rows[(arm, qid)] = _public_row(row, query, arm=arm,
                metrics=_new_metrics(row, arm), row_sha256=row.get("row_sha256"))
            provenance[str(path.resolve())] = file_sha(path)
    control_root = Path(manifest["control_root"])
    if not control_root.is_absolute():
        control_root = root / control_root
    verified = verify_control_bundle(control_root, common_config=manifest["config"])
    for key, value in manifest["control_reference"].items():
        _require(key in verified and digest(verified[key]) == digest(value),
                 "Reused control reference differs: " + key)
    records = verified.get("records")
    _require(isinstance(records, list) and len(records) == 100,
             "Public control bundle must cover 100 records")
    for record in records:
        qid = record.get("query_id")
        _require(type(qid) is int and qid in cohort and (CONTROL, qid) not in rows,
                 "Duplicate/unexpected public control question")
        query = cohort[qid]
        _require(record.get("question") == query["query"] and record.get("gold_answer") == query["answer"],
                 "Control question/gold text differs")
        rows[(CONTROL, qid)] = _public_row(record, query, arm=CONTROL,
            metrics=_control_metrics(record), row_sha256=record.get("terminal_row_sha256"))
    _coverage(rows, cohort)
    provenance["control_reference_sha256"] = digest(manifest["control_reference"])
    return rows, list(cohort), provenance


def collect(root=None, *, suite=None, cohort_path=None):
    root = Path(root) if root is not None else ROOT
    suite = _suite(root, suite)
    manifest, cohort, provenance = load_experiment(root, suite=suite, cohort_path=cohort_path)
    rows, qids, row_sources = collect_rows(root, manifest, cohort, suite=suite)
    return rows, cohort, manifest, {**provenance, **row_sources}


def _binding(manifest):
    return {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
            "config_sha256": manifest["config_sha256"], "cohort_sha256": manifest["cohort_sha256"],
            "control_reference_sha256": digest(manifest["control_reference"]),
            "rubric_sha256": digest(rubric_document())}


def _public_packet(private, rows, manifest):
    cases = []
    for mapped in private["records"]:
        row = rows[(mapped["arm"], mapped["query_id"])]
        cases.append({"case_id": mapped["case_id"], "question": row["question"],
                      "gold_answer": row["gold_answer"], "final_text": row["final_text"],
                      "final_sha256": row["final_sha256"]})
    cases.sort(key=lambda item: item["case_id"])
    _require(digest(cases) == private["public_records_sha256"], "Anonymous question/final payload differs")
    packet = {**_binding(manifest), "mapping_sha256": private["mapping_sha256"],
              "rubric": rubric_document(), "reviewer": None, "records": cases,
              "review_template": {"case_id": "copy unchanged", "final_sha256": "copy unchanged",
                                  "correct": None, "reason": "fresh strict judgement"}}
    return {**packet, "packet_sha256": digest(packet)}


def export_blind(rows, cohort, manifest, *, seed=None):
    _check_manifest(manifest)
    _coverage(rows, cohort)
    seed = secrets.token_hex(32) if seed is None else seed
    try:
        secret = bytes.fromhex(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Blind shuffle seed must be 32 secret random bytes") from exc
    _require(len(secret) == 32, "Blind shuffle seed must be 32 secret random bytes")
    mapped, public_records = [], []
    for (arm, qid), row in rows.items():
        case_id = hmac.new(secret, f"{manifest['manifest_sha256']}:{arm}:{qid}".encode(),
                           hashlib.sha256).hexdigest()[:24]
        mapped.append({"case_id": case_id, "arm": arm, "query_id": qid,
                       "final_sha256": row["final_sha256"], "row_sha256": row["row_sha256"]})
        public_records.append({"case_id": case_id, "question": row["question"],
            "gold_answer": row["gold_answer"], "final_text": row["final_text"],
            "final_sha256": row["final_sha256"]})
    mapped.sort(key=lambda item: item["case_id"])
    public_records.sort(key=lambda item: item["case_id"])
    _require(len({item["case_id"] for item in mapped}) == 400, "Duplicate anonymous case ID")
    private = {**_binding(manifest), "shuffle_seed_secret": seed, "records": mapped,
               "public_records_sha256": digest(public_records)}
    private["mapping_sha256"] = digest(private)
    return _public_packet(private, rows, manifest), private


def _check_mapping(private, rows, manifest):
    shared.checked_digest(private, "mapping_sha256", "Private blind mapping")
    _require(all(private.get(key) == value for key, value in _binding(manifest).items()),
             "Mapping does not bind this experiment/control/rubric")
    mapped, identities = {}, set()
    _require(isinstance(private.get("records"), list) and len(private["records"]) == 400,
             "Blind mapping must contain all 400 cases")
    for record in private["records"]:
        key = (record.get("arm"), record.get("query_id"))
        row = rows.get(key)
        _require(row is not None and key not in identities and record.get("case_id") not in mapped
                 and record.get("final_sha256") == row["final_sha256"]
                 and record.get("row_sha256") == row["row_sha256"],
                 "Blind mapping has duplicate/stale terminal rows")
        identities.add(key)
        mapped[record["case_id"]] = record
    _require(identities == set(rows) and len(rows) == 400, "Blind mapping must cover every terminal")
    return mapped


def _reviewed_label(reviewed, mapped, rows):
    key = (mapped["arm"], mapped["query_id"])
    correct, reason = reviewed.get("correct"), reviewed.get("reason")
    _require(reviewed.get("final_sha256") == mapped["final_sha256"], "Reviewed final SHA256 differs")
    _require(type(correct) is bool and isinstance(reason, str) and bool(reason.strip()),
             "Fresh review requires Boolean correct and a nonempty reason")
    _require(not correct or bool(rows[key]["final_text"].strip()), "An empty public final cannot be correct")
    return {**mapped, "correct": correct, "reason": reason}


def import_reviews(submission, private, rows, manifest):
    _check_manifest(manifest)
    mapped = _check_mapping(private, rows, manifest)
    expected_packet = _public_packet(private, rows, manifest)
    for key, value in {**_binding(manifest), "mapping_sha256": private["mapping_sha256"],
                       "packet_sha256": expected_packet["packet_sha256"]}.items():
        _require(submission.get(key) == value, "Review does not bind this exact four-arm packet/rubric")
    _require(isinstance(submission.get("reviewer"), str) and bool(submission["reviewer"].strip()),
             "Identify the fresh blinded reviewer")
    reviewed, seen = [], set()
    for item in submission.get("records", []):
        case_id = item.get("case_id")
        _require(case_id in mapped and case_id not in seen, "Unknown/duplicate reviewed case")
        reviewed.append(_reviewed_label(item, mapped[case_id], rows))
        seen.add(case_id)
    _require(seen == set(mapped), "Review all 400 anonymous cases, including empty finals")
    result = {**_binding(manifest), "mapping_sha256": private["mapping_sha256"],
              "packet_sha256": expected_packet["packet_sha256"],
              "source": "fresh_blinded_review", "reviewer": submission["reviewer"],
              "submission_sha256": digest(submission), "records": reviewed}
    return {**result, "reviews_sha256": digest(result)}


def load_labels(review_documents, rows, manifest):
    labels = {}
    for document in review_documents:
        shared.checked_digest(document, "reviews_sha256", "Fresh reviews")
        _require(document.get("source") == "fresh_blinded_review"
                 and isinstance(document.get("reviewer"), str) and bool(document["reviewer"].strip())
                 and all(document.get(k) == v for k, v in _binding(manifest).items()),
                 "Only fresh reviews of this four-arm packet are accepted")
        _sha(document.get("packet_sha256"), "Reviewed packet")
        for item in document.get("records", []):
            key = (item.get("arm"), item.get("query_id"))
            row = rows.get(key)
            _require(row is not None and key not in labels
                     and item.get("row_sha256") == row["row_sha256"]
                     and item.get("final_sha256") == row["final_sha256"],
                     "Duplicate/stale/unexpected fresh label")
            labels[key] = _reviewed_label(item, item, rows)
    _require(set(labels) == set(rows) and len(labels) == 400,
             "All 400 finals require fresh labels; historical scores cannot fill gaps")
    return labels


def _stats(values):
    known = [value for value in values if value is not None]
    return {"recorded": len(known), "missing_rows": len(values) - len(known),
            "known_sum": sum(known) if known else None,
            "mean": statistics.mean(known) if known else None,
            "median": statistics.median(known) if known else None}


def summarize(rows, qids, labels, manifest):
    _check_manifest(manifest)
    _require(len(qids) == len(set(qids)) == 100
             and set(rows) == {(arm, qid) for arm in GROUPS for qid in qids},
             "Summary requires complete 100-question coverage for all four groups")
    _require(set(labels) == set(rows), "Summary requires all 400 fresh labels")
    for key, row in rows.items():
        item = labels[key]
        _require((item.get("arm"), item.get("query_id")) == key
                 and item.get("row_sha256") == row["row_sha256"], "Stale summary label")
        _reviewed_label(item, item, rows)
        _require(item["final_sha256"] == row["final_sha256"], "Stale summary final SHA256")
    result = {**_binding(manifest), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "questions_per_arm": 100, "freshly_reviewed_finals": 400,
        "control_policy": "Reuse the frozen strict B100 public finals; freshly score all 400 anonymous finals under one rubric. No historical correctness labels are reused.",
        "arms": {}, "paired_vs_control": {},
        "token_semantics": "Input includes cached input; reasoning is a subset of output. Model total=input+output. Reranker input is reported separately. Missing usage remains unknown, so recorded token sums are known-usage lower bounds.",
        "call_semantics": "rerank_calls is all underlying service invocations, including automatic ranking and cached pages; manual_rerank_calls and automatic_rerank_calls separate the interfaces. search_page_calls counts document pages, independent_rerank_page_calls counts ranking pages. Unknown historical retrieval counters remain null.",
        "cost": {"amount": None, "status": "not_priced", "note": "No tariff is assumed; zero historical cost placeholders do not mean free."},
        "limitations": [
            "The reused B100 is a historical control, not a concurrent randomized control; shared frozen settings do not eliminate time, serving-contention or execution-epoch differences.",
            "One stochastic trajectory per question; observed paired differences do not establish that the changed interface caused every changed answer.",
            "The automatic arm is a strategy package: immediate per-subquery scoring, original-query reuse, fixed scope/parameters, changed search output and removal of independent rerank. It does not isolate the neural model's causal contribution.",
            "Three two-sided exact McNemar comparisons against the same control are reported without multiplicity adjustment; p-values are explicitly unadjusted, not a corrected familywise claim.",
            "Elapsed time includes resource contention. Reranker preparation/scoring times are components of elapsed time and must not be added again.",
            "API attempts without usage and any unrecorded interruption-window usage remain unknown. Monetary costs are not inferred.",
        ]}
    fields = sorted({field for row in rows.values() for field in row["metrics"]})
    for arm in GROUPS:
        group = [rows[(arm, qid)] for qid in qids]
        correct = sum(labels[(arm, qid)]["correct"] for qid in qids)
        result["arms"][arm] = {"n": 100, "correct": correct, "accuracy": correct / 100,
            "accuracy_percent": correct, "freshly_reviewed": 100,
            "termination_counts": dict(Counter(row["terminated_by"] for row in group)),
            "empty_public_finals": sum(not row["final_text"].strip() for row in group),
            "api_usage_complete_for_recorded_rows": all(row["metrics"].get("api_attempts_without_usage") == 0 for row in group),
            "token_totals_are_known_usage_only": True,
            "metrics": {field: _stats([row["metrics"].get(field) for row in group]) for field in fields}}
    for arm in ARMS:
        arm_only, control_only, both, neither = [], [], [], []
        for qid in qids:
            c, a = labels[(CONTROL, qid)]["correct"], labels[(arm, qid)]["correct"]
            (both if c and a else control_only if c else arm_only if a else neither).append(qid)
        result["paired_vs_control"][arm] = {
            "n_pairs": 100, "both_correct": len(both), "both_wrong": len(neither),
            "control_only_correct": len(control_only), "arm_only_correct": len(arm_only),
            "net_correct_arm_minus_control": len(arm_only) - len(control_only),
            "accuracy_difference_percentage_points": len(arm_only) - len(control_only),
            "arm_only_qids": arm_only, "control_only_qids": control_only,
            "mcnemar_exact_two_sided_p_unadjusted": mcnemar_exact(len(control_only), len(arm_only)),
            "p_value_adjustment": "none; three comparisons against one historical control",
            "matched_metric_differences_arm_minus_control": {
                field: _stats([rows[(arm, qid)]["metrics"].get(field) - rows[(CONTROL, qid)]["metrics"].get(field)
                    if rows[(arm, qid)]["metrics"].get(field) is not None
                    and rows[(CONTROL, qid)]["metrics"].get(field) is not None else None for qid in qids])
                for field in fields}}
    return {**result, "summary_sha256": digest(result)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "import", "summarize"):
        child = children.add_parser(command)
        child.add_argument("--out", type=Path, required=True)
        if command == "export":
            child.add_argument("--private-map", type=Path, required=True)
        elif command == "import":
            child.add_argument("--private-map", type=Path, required=True)
            child.add_argument("--submission", type=Path, required=True)
        else:
            child.add_argument("--reviews", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    _require(not args.out.exists(), "Use a new output path; overwriting is forbidden")
    if args.command == "export":
        _require(args.out.resolve() != args.private_map.resolve() and not args.private_map.exists(),
                 "Use new separate packet and private mapping paths")
    rows, cohort, manifest, provenance = collect(args.root)
    if args.command == "export":
        packet, private = export_blind(rows, cohort, manifest)
        write_new(args.private_map, private, private=True)
        write_new(args.out, packet, private=True)
    elif args.command == "import":
        result = import_reviews(load(args.submission), load(args.private_map), rows, manifest)
        write_new(args.out, result, private=True)
    else:
        labels = load_labels([load(path) for path in args.reviews], rows, manifest)
        result = summarize(rows, list(cohort), labels, manifest)
        result.pop("summary_sha256")
        result["source_file_sha256"] = {**provenance,
            **{str(path.resolve()): file_sha(path) for path in args.reviews}}
        result["summary_sha256"] = digest(result)
        write_new(args.out, result)
    print(json.dumps({"output": str(args.out), "terminal_finals": len(rows), "fresh_review_required": 400}))


if __name__ == "__main__":
    main()
