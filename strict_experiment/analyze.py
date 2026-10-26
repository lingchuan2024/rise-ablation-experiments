#!/usr/bin/env python3
"""Hash-bound public-final scoring and metrics for the strict Qwen3.5 pair.

No raw trajectory, private reasoning or historical score is read. All artifacts
written here are new files; review imports cannot overwrite previous scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import statistics

VERSION = "strict-qwen35-analysis-v1"
EXPERIMENT_VERSION = "official-rise-plus-rerank-qwen35-v1"
MODEL = "Qwen3.5-35B-A3B"
COHORT_SHA256 = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"
ARMS = ("A", "B")
ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value, *, private=False):
    """Exclusive creation avoids silently replacing a previous judgement/map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def checked_digest(value, field, label):
    if not isinstance(value, dict) or value.get(field) != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError(label + " checksum differs")
    return value[field]


def nonnegative(value, field, *, integer=False, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("Invalid nonnegative metric: " + field)
    return int(value) if integer else value


def rubric_document():
    """Frozen before generating answers; no historical judgement imported."""
    return {
        "version": VERSION,
        "rubric_id": "strict-public-final-gold-v1",
        "primary_label": "correct",
        "rules": [
            "Judge only the new public final against this question and frozen gold answer; do not infer an answer from hidden reasoning or intermediate tools.",
            "A missing final, abstention, or an answer with the wrong entity/value is incorrect, including natural time/turn/context/output budget exhaustion.",
            "Accept unambiguous equivalent names or harmless formatting differences; require all components the question explicitly asks for, including a full birth name when requested.",
            "A different nontrivial spelling or name component is not automatically equivalent. Record ambiguous gold/source variants as a limitation; keep the declared strict label and do not choose rules after seeing an arm's score.",
            "If multiple incompatible answers are given without a clear final choice, mark incorrect. A clearly selected answer is judged as the submitted answer.",
            "Use the same rule for both anonymous answers to each question. Correct requires a Boolean label and a nonempty reason; final SHA256 must match.",
            "Perform a fresh blinded review. Do not import historical baseline, v14, pilot, placeholder judge or is_correct fields. No natural error may be rerun to improve its score.",
        ],
        "reviewer_instruction": "Review all exported cases independently of arm identity, return the exact case_id/final_sha256 with correct=true/false and reason, and identify the reviewer.",
    }


def validate_rubric(value):
    if value != rubric_document():
        raise ValueError("Only the predeclared strict rubric is supported")
    return digest(value)


def load_experiment(root, manifest_path=None, cohort_path=None):
    root = Path(root)
    manifest_path = Path(manifest_path) if manifest_path else root / "reports/strict_qwen35/run_manifest.json"
    cohort_path = Path(cohort_path) if cohort_path else root / "data/queries_100.jsonl"
    manifest = load(manifest_path)
    checked_digest(manifest, "manifest_sha256", "Manifest")
    if manifest.get("version") != EXPERIMENT_VERSION or manifest.get("config", {}).get("model") != MODEL:
        raise ValueError("Not the strict official-RISE plus rerank Qwen3.5 experiment")
    if manifest.get("config_sha256") != digest(manifest["config"]):
        raise ValueError("Config checksum differs")
    if file_sha(cohort_path) != COHORT_SHA256 or manifest.get("cohort_sha256") != COHORT_SHA256:
        raise ValueError("The fixed official 100-question cohort differs")
    rows = [json.loads(line) for line in cohort_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qids = [row["query_id"] for row in rows]
    if len(rows) != 100 or len(set(qids)) != 100 or manifest.get("query_ids") != qids:
        raise ValueError("Manifest must bind the exact ordered official 100 questions")
    if set(manifest.get("contracts", {})) != set(ARMS):
        raise ValueError("Exactly the new A and B prompt/tool contracts are required")
    analysis_contract = manifest.get("analysis_contract", {})
    if analysis_contract.get("version") != VERSION or analysis_contract.get("rubric_sha256") != digest(rubric_document()):
        raise ValueError("Manifest has not frozen this scoring rubric before the run")
    return manifest, {row["query_id"]: row for row in rows}, {
        "manifest_file_sha256": file_sha(manifest_path),
        "cohort_file_sha256": file_sha(cohort_path),
    }


def contract_sha(manifest, arm):
    contract = manifest["contracts"][arm]
    value = contract.get("contract_sha256") if isinstance(contract, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Missing prompt/tool contract SHA256")
    return value


def validate_row(row, arm, qid, manifest, cohort):
    checked_digest(row, "row_sha256", "Terminal row")
    expected = {"version": EXPERIMENT_VERSION, "arm": arm, "query_id": qid,
        "manifest_sha256": manifest["manifest_sha256"], "config_sha256": manifest["config_sha256"],
        "contract_sha256": contract_sha(manifest, arm),
        "question_sha256": text_sha(cohort[qid]["query"])}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Terminal row does not match its frozen experiment/question/arm")
    if "model" in row and row["model"] != MODEL:
        raise ValueError("Terminal model differs")
    if not isinstance(row.get("final_text"), str) or row.get("final_sha256") != text_sha(row["final_text"]):
        raise ValueError("Public final SHA256 differs")
    reason = row.get("terminated_by")
    if not isinstance(reason, str) or not reason:
        raise ValueError("Missing terminal reason")
    if reason.startswith(("api_error", "infrastructure", "solver_api_failed")):
        raise ValueError("An infrastructure/API interruption is not a natural terminal result")
    row_metrics(row, arm)


def collect_rows(root, manifest, cohort, *, scope="all100"):
    if scope not in {"first10", "all100"}:
        raise ValueError("Unknown report scope")
    target = list(cohort)[:10] if scope == "first10" else list(cohort)
    rows, provenance = {}, {}
    for arm in ARMS:
        directory = Path(root) / "runs/strict_qwen35" / arm / "_per_query"
        for path in sorted(directory.glob("*.json")):
            row = load(path)
            qid = row.get("query_id")
            if isinstance(qid, bool) or not isinstance(qid, int) or qid not in cohort or path.name != f"qid_{qid}.json":
                raise ValueError("Unexpected/misnamed terminal question")
            validate_row(row, arm, qid, manifest, cohort)
            if qid in target:
                key = (arm, qid)
                if key in rows:
                    raise ValueError("Duplicate terminal result")
                rows[key] = row
                provenance[str(path.relative_to(root))] = file_sha(path)
    return rows, target, provenance


def row_metrics(row, arm):
    metrics = {key: nonnegative(row.get(key), key, integer=key == "n_turns")
        for key in ("n_turns", "elapsed_seconds")}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
        metrics[key] = nonnegative(row.get(key), key, integer=True, optional=key in {"cached_input_tokens", "reasoning_tokens"})
    if metrics["cached_input_tokens"] is not None and metrics["cached_input_tokens"] > metrics["input_tokens"]:
        raise ValueError("Cached tokens are a subset of inclusive input")
    if metrics["reasoning_tokens"] is not None and metrics["reasoning_tokens"] > metrics["output_tokens"]:
        raise ValueError("Reasoning tokens are a subset of output")
    raw = row.get("raw_non_cached_input_tokens")
    if raw is not None:
        nonnegative(raw, "raw_non_cached_input_tokens", integer=True)
        if metrics["cached_input_tokens"] is None or raw + metrics["cached_input_tokens"] != metrics["input_tokens"]:
            raise ValueError("Inclusive input does not equal raw noncached plus cached")
    metrics["model_total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
    ledger = row.get("api_ledger")
    if not isinstance(ledger, dict) or ledger.get("agent_totals_match") is not True:
        raise ValueError("An independently matched numeric API ledger is required")
    for source, target in (("attempts", "api_attempts"), ("successful_calls", "api_successful_calls"),
            ("attempts_without_usage", "api_attempts_without_usage")):
        metrics[target] = nonnegative(ledger.get(source), "api_ledger." + source,
            integer=True, optional=source == "attempts_without_usage")
    if metrics["api_successful_calls"] > metrics["api_attempts"]:
        raise ValueError("API successful calls exceed attempts")
    if metrics["api_attempts_without_usage"] is not None and metrics["api_attempts_without_usage"] > metrics["api_attempts"]:
        raise ValueError("API unknown-usage count exceeds attempts")
    for key in ("input_tokens", "output_tokens"):
        if nonnegative(ledger.get(key), "api_ledger." + key, integer=True) != metrics[key]:
            raise ValueError("API ledger and normalized agent tokens differ")
    reranker = row.get("reranker_summary")
    if not isinstance(reranker, dict):
        raise ValueError("Reranker summary must be present, including an empty summary for A")
    counters = {"rerank_calls": "rerank_calls", "scoring_calls": "scoring_calls",
        "scored_passages": "scored_passages", "reranker_input_tokens": "input_tokens",
        "scoring_seconds": "scoring_seconds", "preparation_seconds": "preparation_seconds"}
    for target, source in counters.items():
        value = reranker.get(source)
        if source == "input_tokens" and value is None:
            value = reranker.get("scoring_input_tokens")
        if arm == "A" and value is None:
            value = 0
        metrics[target] = nonnegative(value, "reranker." + source, integer=not source.endswith("seconds"))
    if arm == "A" and any(metrics[key] for key in counters):
        raise ValueError("Original RISE A must not incur reranking")
    metrics["combined_model_and_reranker_input_tokens"] = metrics["input_tokens"] + metrics["reranker_input_tokens"]
    return metrics


def export_blind(rows, cohort, manifest, *, seed=None):
    if not rows:
        raise ValueError("No new terminal finals to review")
    seed = secrets.token_hex(32) if seed is None else seed
    try:
        secret = bytes.fromhex(seed)
    except (ValueError, TypeError) as exc:
        raise ValueError("Blind shuffle seed must be 32 secret random bytes") from exc
    if len(secret) != 32:
        raise ValueError("Blind shuffle seed must be 32 secret random bytes")
    public_records, mapping = [], []
    for (arm, qid), row in sorted(rows.items()):
        case_id = hmac.new(secret, f"{manifest['manifest_sha256']}:{arm}:{qid}".encode(), hashlib.sha256).hexdigest()[:24]
        public_records.append({"case_id": case_id, "question": cohort[qid]["query"],
            "gold_answer": cohort[qid]["answer"], "final_text": row["final_text"], "final_sha256": row["final_sha256"]})
        mapping.append({"case_id": case_id, "arm": arm, "query_id": qid, "final_sha256": row["final_sha256"], "row_sha256": row["row_sha256"]})
    public_records.sort(key=lambda row: row["case_id"])
    mapping.sort(key=lambda row: row["case_id"])
    binding = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "rubric_sha256": digest(rubric_document())}
    private = {**binding, "shuffle_seed_secret": seed, "records": mapping}
    private["mapping_sha256"] = digest(private)
    public = {**binding, "mapping_sha256": private["mapping_sha256"],
        "rubric": rubric_document(), "reviewer": None, "records": public_records,
        "review_template": {"case_id": "copy unchanged", "final_sha256": "copy unchanged", "correct": None, "reason": "fresh strict judgement"}}
    return public, private


def import_reviews(submission, private, rows, manifest):
    checked_digest(private, "mapping_sha256", "Private blind mapping")
    for key, value in {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
            "rubric_sha256": digest(rubric_document()), "mapping_sha256": private["mapping_sha256"]}.items():
        if submission.get(key) != value or private.get(key) != value:
            raise ValueError("Review/mapping does not bind this experiment/rubric")
    if not isinstance(submission.get("reviewer"), str) or not submission["reviewer"].strip():
        raise ValueError("Identify the fresh blinded reviewer")
    known = {}
    for mapped in private.get("records", []):
        key = (mapped["arm"], mapped["query_id"])
        row = rows.get(key)
        if mapped["case_id"] in known or row is None or mapped["final_sha256"] != row["final_sha256"] or mapped["row_sha256"] != row["row_sha256"]:
            raise ValueError("Blind mapping has duplicate or stale terminal rows")
        known[mapped["case_id"]] = mapped
    labels, seen = [], set()
    for reviewed in submission.get("records", []):
        case_id = reviewed.get("case_id")
        mapped = known.get(case_id)
        if mapped is None or case_id in seen or reviewed.get("final_sha256") != mapped["final_sha256"]:
            raise ValueError("Unknown/duplicate/stale reviewed final")
        correct, reason = reviewed.get("correct"), reviewed.get("reason")
        if type(correct) is not bool or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Fresh review needs Boolean correct and a reason")
        row = rows[(mapped["arm"], mapped["query_id"])]
        if correct and not row["final_text"].strip():
            raise ValueError("An empty public final cannot be correct")
        seen.add(case_id)
        labels.append({"arm": mapped["arm"], "query_id": mapped["query_id"],
            "case_id": case_id, "final_sha256": mapped["final_sha256"],
            "row_sha256": mapped["row_sha256"], "correct": correct, "reason": reason})
    if seen != set(known):
        raise ValueError("Review every case in this blinded export")
    result = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "rubric_sha256": digest(rubric_document()), "mapping_sha256": private["mapping_sha256"],
        "source": "fresh_blinded_review", "reviewer": submission["reviewer"],
        "submission_sha256": digest(submission), "records": labels}
    result["reviews_sha256"] = digest(result)
    return result


def load_labels(review_documents, rows, manifest):
    labels = {}
    for reviews in review_documents:
        checked_digest(reviews, "reviews_sha256", "Fresh reviews")
        if (reviews.get("version") != VERSION or reviews.get("manifest_sha256") != manifest["manifest_sha256"]
                or reviews.get("rubric_sha256") != digest(rubric_document()) or reviews.get("source") != "fresh_blinded_review"):
            raise ValueError("Only newly imported blinded reviews for this run are accepted")
        for label in reviews.get("records", []):
            key = (label.get("arm"), label.get("query_id"))
            row = rows.get(key)
            if row is None or key in labels or label.get("final_sha256") != row["final_sha256"] or label.get("row_sha256") != row["row_sha256"]:
                raise ValueError("Duplicate/stale/unexpected fresh review")
            if type(label.get("correct")) is not bool or not isinstance(label.get("reason"), str) or not label["reason"].strip():
                raise ValueError("Malformed strict label")
            if label["correct"] and not row["final_text"].strip():
                raise ValueError("An empty public final cannot be correct")
            labels[key] = label
    return labels


def mcnemar_exact(a_only, b_only):
    n = a_only + b_only
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2 ** n)) if n else 1.0


def summarize(rows, qids, labels, manifest, *, scope="all100"):
    result = {"version": VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": manifest["config_sha256"], "rubric_sha256": digest(rubric_document()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "scope": scope, "questions_per_arm": len(qids), "arms": {},
        "comparison": "A=official structured RISE; B=the same configuration plus independent rerank and necessary interface instructions. Historical v14 scores are excluded from this estimator.",
        "token_semantics": "input_tokens includes cached input; reasoning_tokens is a subset of output_tokens. Model total=input+output. Reranker pointwise input is reported separately, never counted as model output.",
        "cost": {"amount": None, "status": "not_priced", "note": "Token/GPU counters are measured; missing tariffs or historical $0 placeholders do not mean free."},
        "limitations": ["One stochastic trajectory per question; paired differences do not establish a guaranteed gain.",
            "First10 is an execution check, not a selection or score-based gate; unchanged-config answers are part of the final100.",
            "A 100-question cohort may have limited power. Report discordant pairs and the two-sided exact McNemar p-value without claiming significance from a positive point estimate.",
            "Elapsed wall time includes serving contention and shared resources; scoring/preparation time are components of elapsed time and must not be added again."]}
    metrics_by_key = {key: row_metrics(row, key[0]) for key, row in rows.items()}
    for arm in ARMS:
        keys = [(arm, qid) for qid in qids if (arm, qid) in rows]
        reviewed = [key for key in keys if key in labels]
        complete = len(keys) == len(qids) and len(reviewed) == len(qids)
        correct = sum(labels[key]["correct"] for key in reviewed)
        metrics = [metrics_by_key[key] for key in keys]
        stats = {}
        for field in metrics[0] if metrics else []:
            known = [row[field] for row in metrics if row[field] is not None]
            stats[field] = {"known_sum": sum(known), "missing_rows": len(metrics) - len(known),
                "mean": statistics.mean(known) if known else None,
                "median": statistics.median(known) if known else None}
        result["arms"][arm] = {"terminal_questions": len(keys), "reviewed_questions": len(reviewed),
            "correct": correct if complete else None, "known_correct_among_reviewed": correct,
            "accuracy_percent": 100 * correct / len(qids) if complete else None,
            "fully_scored": complete, "missing_terminal_qids": [qid for qid in qids if (arm, qid) not in rows],
            "missing_review_qids": [qid for qid in qids if (arm, qid) in rows and (arm, qid) not in labels],
            "termination_counts": dict(Counter(rows[key]["terminated_by"] for key in keys)),
            "empty_public_finals": sum(not rows[key]["final_text"].strip() for key in keys),
            "api_usage_complete_for_observed_rows": bool(keys) and all(metrics_by_key[key]["api_attempts_without_usage"] == 0 for key in keys),
            "token_totals_are_known_usage_only": True, "metrics": stats}
    paired = [qid for qid in qids if all((arm, qid) in labels for arm in ARMS)]
    a_only = sum(labels[("A", qid)]["correct"] and not labels[("B", qid)]["correct"] for qid in paired)
    b_only = sum(labels[("B", qid)]["correct"] and not labels[("A", qid)]["correct"] for qid in paired)
    both_correct = sum(all(labels[(arm, qid)]["correct"] for arm in ARMS) for qid in paired)
    paired_complete = len(paired) == len(qids)
    result["paired"] = {"reviewed_pairs": len(paired), "expected_pairs": len(qids),
        "a_only_correct": a_only, "b_only_correct": b_only, "both_correct": both_correct,
        "both_wrong": len(paired) - a_only - b_only - both_correct,
        "net_correct_b_minus_a": b_only - a_only if paired_complete else None,
        "observed_discordance_net_b_minus_a": b_only - a_only,
        "accuracy_difference_percentage_points": 100 * (b_only - a_only) / len(qids) if paired_complete else None,
        "mcnemar_exact_two_sided_p": mcnemar_exact(a_only, b_only) if paired_complete else None,
        "complete_for_scope": paired_complete,
        "matched_metric_differences_b_minus_a": {}}
    metric_pairs = [qid for qid in qids if all((arm, qid) in metrics_by_key for arm in ARMS)]
    for field in next(iter(metrics_by_key.values()), {}):
        differences = [metrics_by_key[("B", qid)][field] - metrics_by_key[("A", qid)][field] for qid in metric_pairs
            if all(metrics_by_key[(arm, qid)][field] is not None for arm in ARMS)]
        result["paired"]["matched_metric_differences_b_minus_a"][field] = {"pairs": len(differences),
            "sum": sum(differences), "mean": statistics.mean(differences) if differences else None}
    result["summary_sha256"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cohort", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    rubric = sub.add_parser("rubric")
    rubric.add_argument("--out", type=Path, required=True)
    for command in ("export-blind", "import-reviews", "summarize"):
        child = sub.add_parser(command)
        child.add_argument("--scope", choices=("first10", "all100"), default="all100")
        child.add_argument("--out", type=Path, required=True)
        if command == "export-blind":
            child.add_argument("--private-map", type=Path, required=True)
            child.add_argument("--exclude-reviews", type=Path, action="append", default=[],
                help="Previously imported new-run reviews; export only the remaining cases.")
        elif command == "import-reviews":
            child.add_argument("--submission", type=Path, required=True)
            child.add_argument("--private-map", type=Path, required=True)
        else:
            child.add_argument("--reviews", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "rubric":
        write_new(args.out, rubric_document())
        print(json.dumps({"rubric_sha256": digest(rubric_document())}))
        return
    manifest, cohort, provenance = load_experiment(args.root, args.manifest, args.cohort)
    rows, qids, row_provenance = collect_rows(args.root, manifest, cohort, scope=args.scope)
    if args.command == "export-blind":
        if args.out.exists() or args.private_map.exists() or args.out.resolve() == args.private_map.resolve():
            raise ValueError("Use new separate output and private mapping paths")
        excluded = load_labels([load(path) for path in args.exclude_reviews], rows, manifest)
        public, private = export_blind({key: value for key, value in rows.items() if key not in excluded}, cohort, manifest)
        write_new(args.private_map, private, private=True)
        write_new(args.out, public, private=True)
    elif args.command == "import-reviews":
        write_new(args.out, import_reviews(load(args.submission), load(args.private_map), rows, manifest), private=True)
    else:
        labels = load_labels([load(path) for path in args.reviews], rows, manifest)
        result = summarize(rows, qids, labels, manifest, scope=args.scope)
        result.pop("summary_sha256")
        result["source_file_sha256"] = {**provenance, **row_provenance,
            **{str(path.resolve()): file_sha(path) for path in args.reviews}}
        result["summary_sha256"] = digest(result)
        write_new(args.out, result)
    print(json.dumps({"output": str(args.out), "terminal_rows": len(rows), "scope": args.scope}))


if __name__ == "__main__":
    main()
