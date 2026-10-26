"""Fresh paired public-final review of the fixed auto-rerank twenty-question pilot.

The twenty historical control finals are reused as bytes, never as old labels.
Execution metadata are verified locally; only forty public finals enter review.
No model client, external judge, trajectory reasoning, or network is used here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import secrets
import json

from strict_ablations import analyze as common
from strict_ablations.control import verify_control_bundle
from strict_ablations.prompt import CONTROL, build_contract

VERSION = "strict-auto20-analysis-v1"
ARM = "auto_rerank"
GROUPS = (CONTROL, ARM)
COUNT = 20
ROOT = Path(__file__).resolve().parents[1]
digest, text_sha, file_sha = common.digest, common.text_sha, common.file_sha
load, write_new = common.load, common.write_new
require = common._require


def rubric_document():
    rubric = copy.deepcopy(common.shared.rubric_document())
    rubric["version"] = VERSION
    return rubric


def fixed_cohort():
    from strict_experiment import runner
    rows = runner.validate_queries()[:COUNT]
    return {row["query_id"]: row for row in rows}


def check_manifest(manifest):
    common.shared.checked_digest(manifest, "manifest_sha256", "Auto20 manifest")
    require(manifest.get("config", {}).get("model") == common.MODEL
        and manifest.get("config_sha256") == digest(manifest["config"]), "Model/config differs")
    require(manifest.get("cohort_sha256") == common.COHORT_SHA256, "Fixed100 cohort differs")
    require(manifest.get("question_runs") == COUNT and manifest.get("reused_control_questions") == COUNT,
            "This analysis requires the isolated twenty-question pilot")
    require(manifest.get("selected_query_ids") == list(fixed_cohort()), "Pilot selection differs")
    plan = manifest.get("execution_plan", {})
    require(plan.get("version") == "strict-auto20-pilot-v1" and plan.get("arms") == [ARM]
        and plan.get("question_runs") == COUNT and plan.get("selection") == "first20_in_frozen_fixed100_order"
        and plan.get("stage") == "exploratory_pilot", "Pilot execution scope differs")
    require(manifest.get("analysis_contract") == {"version": VERSION,
        "rubric_sha256": digest(rubric_document())}, "Fresh paired review contract was not frozen")
    require(manifest.get("contracts", {}).get(ARM) == build_contract(ARM), "Automatic contract differs")
    reference = manifest.get("control_reference")
    require(isinstance(reference, dict) and common._CONTROL_BINDINGS <= set(reference),
            "Pinned control reference is missing")


def _coverage(rows, cohort):
    expected = fixed_cohort()
    require(cohort == expected and list(cohort) == list(expected),
            "Use precisely the first twenty frozen questions in their original order")
    require(set(rows) == {(arm, qid) for arm in GROUPS for qid in cohort},
            "Complete unique coverage requires forty public finals")
    for (arm, qid), row in rows.items():
        require(row.get("arm") == arm and row.get("query_id") == qid
            and row.get("question") == cohort[qid]["query"]
            and row.get("gold_answer") == cohort[qid]["answer"], "Question/identity differs")
        require(isinstance(row.get("final_text"), str)
            and row.get("final_sha256") == text_sha(row["final_text"]), "Public final SHA differs")
        common._sha(row.get("row_sha256"), "Terminal SHA")


def normalize_records(new_rows, control_records, cohort, manifest):
    """Normalize already execution-verified terminals and pinned control records.

    The standalone interface also checks identity, hashes, coverage and numeric
    ledgers. Production callers use collect(), which verifies all private files
    and the complete frozen control bundle before reaching this function.
    """
    check_manifest(manifest)
    require(isinstance(new_rows, list) and isinstance(control_records, list)
        and len(new_rows) == len(control_records) == COUNT, "Exactly twenty records per group are required")
    rows = {}
    for row in new_rows:
        common.shared.checked_digest(row, "row_sha256", "Auto terminal")
        qid = row.get("query_id")
        require(type(qid) is int and qid in cohort and (ARM, qid) not in rows, "Duplicate/unexpected new terminal")
        require(row.get("arm") == ARM and row.get("model") == common.MODEL
            and row.get("manifest_sha256") == manifest["manifest_sha256"]
            and row.get("config_sha256") == manifest["config_sha256"]
            and row.get("contract_sha256") == manifest["contracts"][ARM]["contract_sha256"],
            "New terminal belongs to another execution")
        reason = row.get("terminated_by")
        require(isinstance(reason, str) and bool(reason)
            and not reason.startswith(("api_error", "infrastructure", "solver_api_failed")),
            "An infrastructure interruption cannot be scored as a terminal")
        rows[(ARM, qid)] = common._public_row(row, cohort[qid], arm=ARM,
            metrics=common._new_metrics(row, ARM), row_sha256=row["row_sha256"])
    for row in control_records:
        qid = row.get("query_id")
        require(type(qid) is int and qid in cohort and (CONTROL, qid) not in rows,
                "Duplicate/unexpected historical control")
        require(row.get("question") == cohort[qid]["query"]
            and row.get("gold_answer") == cohort[qid]["answer"], "Control question/gold differs")
        rows[(CONTROL, qid)] = common._public_row(row, cohort[qid], arm=CONTROL,
            metrics=common._control_metrics(row), row_sha256=row.get("terminal_row_sha256"))
    _coverage(rows, cohort)
    return rows


def collect(root=None, *, suite=None):
    if suite is None:
        from strict_ablations.auto20 import Auto20Suite
        suite = Auto20Suite(root)
    manifest = suite.verify_manifest(resources=False)
    check_manifest(manifest)
    cohort = fixed_cohort()
    paths = {qid: suite.terminal_path(ARM, qid) for qid in cohort}
    actual = {p.resolve() for parent in {p.parent for p in paths.values()} for p in parent.glob("*.json")}
    require(actual == {p.resolve() for p in paths.values()}, "Auto20 terminal coverage is incomplete or contains extras")
    terminals, provenance = [], {}
    for qid, query in cohort.items():
        path = paths[qid]
        row = load(path)
        suite.validate_terminal(row, manifest, ARM, query)
        suite.verify_execution_artifacts(manifest, ARM, query, row)
        terminals.append(row)
        provenance[str(path.resolve())] = file_sha(path)
    verified = verify_control_bundle(manifest["control_root"], common_config=manifest["config"])
    require(all(key in verified and digest(verified[key]) == digest(value)
        for key, value in manifest["control_reference"].items()), "Frozen B100 reference changed")
    control = [row for row in verified["records"] if row["query_id"] in cohort]
    rows = normalize_records(terminals, control, cohort, manifest)
    return rows, cohort, manifest, {**provenance,
        "control_reference_sha256": digest(manifest["control_reference"])}


def _binding(manifest):
    return {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"], "cohort_sha256": manifest["cohort_sha256"],
        "control_reference_sha256": digest(manifest["control_reference"]),
        "rubric_sha256": digest(rubric_document())}


def _packet(private, rows, manifest):
    public = [{"case_id": mapped["case_id"], **{key: rows[(mapped["arm"], mapped["query_id"])][key]
        for key in ("question", "gold_answer", "final_text", "final_sha256")}} for mapped in private["records"]]
    public.sort(key=lambda row: row["case_id"])
    require(digest(public) == private["public_records_sha256"], "Anonymous public payload changed")
    result = {**_binding(manifest), "mapping_sha256": private["mapping_sha256"],
        "rubric": rubric_document(), "reviewer": None, "records": public,
        "review_template": {"case_id": "copy unchanged", "final_sha256": "copy unchanged",
                            "correct": None, "reason": "fresh strict judgement"}}
    return {**result, "packet_sha256": digest(result)}


def export_blind(rows, cohort, manifest, *, seed=None):
    check_manifest(manifest)
    _coverage(rows, cohort)
    seed = secrets.token_hex(32) if seed is None else seed
    try:
        secret = bytes.fromhex(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Use a secret 32-byte blind shuffle seed") from exc
    require(len(secret) == 32, "Use a secret 32-byte blind shuffle seed")
    mapped, public = [], []
    for (arm, qid), row in rows.items():
        case_id = hmac.new(secret, f"{manifest['manifest_sha256']}:{arm}:{qid}".encode(), hashlib.sha256).hexdigest()[:24]
        mapped.append({"case_id": case_id, "arm": arm, "query_id": qid,
            "row_sha256": row["row_sha256"], "final_sha256": row["final_sha256"]})
        public.append({"case_id": case_id, **{key: row[key]
            for key in ("question", "gold_answer", "final_text", "final_sha256")}})
    mapped.sort(key=lambda row: row["case_id"])
    public.sort(key=lambda row: row["case_id"])
    require(len({row["case_id"] for row in mapped}) == 40, "Anonymous IDs collided")
    private = {**_binding(manifest), "shuffle_seed_secret": seed, "records": mapped,
        "public_records_sha256": digest(public)}
    private["mapping_sha256"] = digest(private)
    return _packet(private, rows, manifest), private


def _mapping(private, rows, manifest):
    common.shared.checked_digest(private, "mapping_sha256", "Private paired mapping")
    require(all(private.get(key) == value for key, value in _binding(manifest).items()), "Stale mapping identity")
    require(len(private.get("records", [])) == len(rows) == 40, "Mapping must cover forty cases")
    mapped, identities = {}, set()
    for item in private["records"]:
        key = (item.get("arm"), item.get("query_id"))
        row = rows.get(key)
        require(row is not None and key not in identities and item.get("case_id") not in mapped
            and item.get("row_sha256") == row["row_sha256"]
            and item.get("final_sha256") == row["final_sha256"], "Duplicate/stale mapped terminal")
        identities.add(key)
        mapped[item["case_id"]] = item
    require(identities == set(rows), "Mapping coverage differs")
    return mapped


def import_reviews(submission, private, rows, manifest):
    check_manifest(manifest)
    mapped = _mapping(private, rows, manifest)
    packet = _packet(private, rows, manifest)
    require(all(submission.get(key) == value for key, value in {**_binding(manifest),
        "mapping_sha256": private["mapping_sha256"], "packet_sha256": packet["packet_sha256"]}.items()),
        "Review is not bound to this exact paired packet")
    require(isinstance(submission.get("reviewer"), str) and submission["reviewer"].strip(),
            "Identify the fresh blinded reviewer")
    labels, seen = [], set()
    for item in submission.get("records", []):
        case = item.get("case_id")
        require(case in mapped and case not in seen, "Unknown/duplicate reviewed case")
        labels.append(common._reviewed_label(item, mapped[case], rows))
        seen.add(case)
    require(seen == set(mapped), "Freshly review all forty public finals, including empty answers")
    result = {**_binding(manifest), "mapping_sha256": private["mapping_sha256"],
        "packet_sha256": packet["packet_sha256"], "source": "fresh_blinded_review",
        "reviewer": submission["reviewer"], "submission_sha256": digest(submission), "records": labels}
    return {**result, "reviews_sha256": digest(result)}


def load_labels(documents, rows, manifest):
    labels = {}
    for document in documents:
        common.shared.checked_digest(document, "reviews_sha256", "Fresh paired reviews")
        require(document.get("source") == "fresh_blinded_review"
            and isinstance(document.get("reviewer"), str) and document["reviewer"].strip()
            and all(document.get(key) == value for key, value in _binding(manifest).items()),
            "Historical labels cannot replace fresh paired review")
        common._sha(document.get("packet_sha256"), "Reviewed packet")
        for item in document.get("records", []):
            key = (item.get("arm"), item.get("query_id"))
            require(key in rows and key not in labels and item.get("row_sha256") == rows[key]["row_sha256"]
                and item.get("final_sha256") == rows[key]["final_sha256"], "Duplicate/stale reviewed row")
            labels[key] = common._reviewed_label(item, item, rows)
    require(set(labels) == set(rows) and len(labels) == 40, "Exactly forty fresh labels are required")
    return labels


def summarize(rows, qids, labels, manifest):
    check_manifest(manifest)
    cohort = fixed_cohort()
    require(qids == list(cohort), "Summary question order differs")
    _coverage(rows, cohort)
    require(set(labels) == set(rows), "Summary requires forty fresh labels")
    for key, item in labels.items():
        require((item.get("arm"), item.get("query_id")) == key
            and item.get("row_sha256") == rows[key]["row_sha256"]
            and item.get("final_sha256") == rows[key]["final_sha256"], "Stale summary label")
        common._reviewed_label(item, item, rows)
    fields = sorted({field for row in rows.values() for field in row["metrics"]})
    result = {**_binding(manifest), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": common.MODEL, "questions_per_arm": COUNT, "freshly_reviewed_finals": 40,
        "control_policy": "Frozen B100 same-question finals reused; all forty finals freshly reviewed together",
        "arms": {}, "limitations": [
            "Exploratory first-twenty-question pilot; no full100 accuracy claim or prompt selection based on scores.",
            "Historical control and one stochastic trajectory per question do not establish every answer change was caused by the intervention.",
            "The automatic arm jointly changes timing, query/scope/parameter selection, search output and independent-tool access; it is not a neural-model-only ablation.",
            "Elapsed time includes contention; preparation/scoring times are components, not additional elapsed time.",
            "Unknown API usage stays unknown; no monetary tariff or free-cost assumption is introduced."],
        "token_semantics": "Input includes cached input; reasoning is part of output; total=input+output. Reranker input is separate.",
        "call_semantics": "Service calls, real scoring calls, scored passages, automatic calls and manual calls are separate; cache hits are not fresh neural forwards."}
    for arm in GROUPS:
        group = [rows[(arm, qid)] for qid in qids]
        correct = sum(labels[(arm, qid)]["correct"] for qid in qids)
        result["arms"][arm] = {"n": COUNT, "correct": correct, "accuracy": correct / COUNT,
            "accuracy_percent": 100 * correct / COUNT,
            "termination_counts": dict(Counter(row["terminated_by"] for row in group)),
            "empty_public_finals": sum(not row["final_text"].strip() for row in group),
            "metrics": {field: common._stats([row["metrics"].get(field) for row in group]) for field in fields}}
    buckets = {key: [] for key in ("both_correct", "both_wrong", "control_only", "auto_only")}
    for qid in qids:
        c, a = labels[(CONTROL, qid)]["correct"], labels[(ARM, qid)]["correct"]
        buckets["both_correct" if c and a else "control_only" if c else "auto_only" if a else "both_wrong"].append(qid)
    result["paired_vs_control"] = {"n_pairs": COUNT, **{key: len(value) for key, value in buckets.items()},
        "control_only_qids": buckets["control_only"], "auto_only_qids": buckets["auto_only"],
        "net_correct_auto_minus_control": len(buckets["auto_only"]) - len(buckets["control_only"]),
        "accuracy_difference_percentage_points": 100 * (len(buckets["auto_only"]) - len(buckets["control_only"])) / COUNT,
        "mcnemar_exact_two_sided_p_exploratory": common.mcnemar_exact(len(buckets["control_only"]), len(buckets["auto_only"])),
        "matched_metric_differences_auto_minus_control": {field: common._stats([
            rows[(ARM, qid)]["metrics"][field] - rows[(CONTROL, qid)]["metrics"][field]
            if rows[(ARM, qid)]["metrics"].get(field) is not None and rows[(CONTROL, qid)]["metrics"].get(field) is not None
            else None for qid in qids]) for field in fields}}
    return {**result, "summary_sha256": digest(result)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    children = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "import", "summarize"):
        child = children.add_parser(name)
        child.add_argument("--out", type=Path, required=True)
        if name in {"export", "import"}:
            child.add_argument("--private-map", type=Path, required=True)
        if name == "import":
            child.add_argument("--submission", type=Path, required=True)
        if name == "summarize":
            child.add_argument("--reviews", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a new output path")
    if args.command == "export":
        require(args.out.resolve() != args.private_map.resolve() and not args.private_map.exists(),
                "Use separate new packet and mapping paths")
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
        result["source_file_sha256"] = {**provenance, **{str(p.resolve()): file_sha(p) for p in args.reviews}}
        result["summary_sha256"] = digest(result)
        write_new(args.out, result)
    print(json.dumps({"output": str(args.out), "terminal_finals": len(rows), "fresh_review_required": 40}))


if __name__ == "__main__":
    main()
