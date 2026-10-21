#!/usr/bin/env python3
"""Summarize saved ablations and prepare answer-bound, arm-blinded review.

No model is called and no correctness is inferred from placeholder judge fields.
The three new arms need fresh reviews; historical v14 labels are reference only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
VARIANTS = ("search_paging", "auto_rerank", "no_bash")
COHORT_SHA256 = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"
VERSION = "v14-ablations-analysis-v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def qid(value):
    if isinstance(value, bool):
        raise ValueError("Boolean query ID is invalid")
    number = int(value)
    if str(number) != str(value):
        raise ValueError("Query ID must be an integer")
    return number


def load_cohort(path, expected_sha=COHORT_SHA256, expected_count=100):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Frozen question cohort SHA256 differs")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    result = {}
    for row in records:
        identity = qid(row["query_id"])
        if identity in result:
            raise ValueError("Duplicate cohort query ID")
        if not isinstance(row.get("query"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("Question and gold answer must be strings")
        result[identity] = row
    if len(result) != expected_count:
        raise ValueError("Frozen question count differs")
    return result


def validate_answer_identity(row, cohort):
    identity = qid(row["query_id"])
    if identity not in cohort:
        raise ValueError("Unexpected query ID in results")
    question = cohort[identity]
    if row.get("question") != question["query"] or row.get("gold_answer") != question["answer"]:
        raise ValueError("Result question/gold does not match its query ID")
    if not isinstance(row.get("final_text"), str):
        raise ValueError("Saved public final must be a string")
    return identity


def collect_rows(suite_root, cohort):
    cases = {}
    for variant in VARIANTS:
        directory = Path(suite_root) / "runs" / variant / "_per_query"
        for path in sorted(directory.glob("*.json")):
            row = load(path)
            identity = validate_answer_identity(row, cohort)
            key = (variant, identity)
            if key in cases:
                raise ValueError("Duplicate result for one arm/query")
            if path.name != f"qid_{identity}.json":
                raise ValueError("Result filename does not match its query ID")
            extras = (row.get("run_config") or {}).get("extras") or {}
            if extras.get("ablation_variant") != variant:
                raise ValueError("Saved result has the wrong ablation variant")
            if row.get("model") != "Qwen3.5-35B-A3B":
                raise ValueError("Saved result has the wrong solver model")
            if str(row.get("terminated_by", "")).startswith("api_error"):
                raise ValueError("An infrastructure/API failure is not a completed experiment result")
            cases[key] = row
    return cases


def historical_reference(path, cohort):
    value = load(path)
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError("Historical evaluation must contain fresh-review records")
    result = {}
    for row in records:
        identity = validate_answer_identity(row, cohort)
        if identity in result or type(row.get("correct")) is not bool:
            raise ValueError("Historical IDs or correctness fields are invalid")
        result[identity] = row["correct"]
    if set(result) != set(cohort):
        raise ValueError("Historical evaluation must cover the exact frozen cohort")
    if value.get("strict", {}).get("correct") != sum(result.values()):
        raise ValueError("Historical strict count disagrees with its reviewed records")
    return result, {"path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "n": len(result), "correct": sum(result.values()),
                    "accuracy": sum(result.values()) / len(result),
                    "comparison_kind": "historical_control"}


def review_packets(cases, cohort, output):
    """Keep stable random IDs while a partial suite acquires more saved rows."""
    output = Path(output)
    private_path = output / "_private_review_map.json"
    previous = load(private_path).get("records", []) if private_path.exists() else []
    prior = {(row["variant"], qid(row["query_id"])): row for row in previous}
    if len(prior) != len(previous) or not set(prior).issubset(cases):
        raise ValueError("Saved review mapping is duplicated or completed results disappeared")
    public, private = [], []
    for (variant, identity), row in sorted(cases.items()):
        payload = {"question": cohort[identity]["query"], "gold_answer": cohort[identity]["answer"],
                   "final_text": row["final_text"], "terminated_by": row["terminated_by"]}
        content_sha, row_sha = digest(payload), digest(row)
        old = prior.get((variant, identity))
        if old and (old["content_sha256"] != content_sha or old["row_sha256"] != row_sha):
            raise ValueError("A saved reviewed result changed; preserve the earlier analysis and use a new output directory")
        review_id = old["review_id"] if old else "R-" + uuid.uuid4().hex
        public.append({"review_id": review_id, "content_sha256": content_sha, **payload})
        private.append({"review_id": review_id, "variant": variant, "query_id": identity,
                        "content_sha256": content_sha, "row_sha256": row_sha})
    public.sort(key=lambda row: row["review_id"])
    if len({row["review_id"] for row in public}) != len(public):
        raise ValueError("Duplicate blind review ID")
    packet = {"version": VERSION, "records": public}
    packet["packet_sha256"] = digest(packet)
    write_json(output / "review_packet.json", packet)
    write_json(private_path, {"version": VERSION, "packet_sha256": packet["packet_sha256"],
                              "records": private})
    private_path.chmod(0o600)
    template = {"version": VERSION,
        "packet_sha256": packet["packet_sha256"], "records": [
            {"review_id": row["review_id"], "content_sha256": row["content_sha256"],
             "correct": None, "reason": ""} for row in public]}
    template_path = output / "review_template.json"
    if template_path.exists() and load(template_path).get("packet_sha256") != packet["packet_sha256"]:
        template_path = output / f"review_template_{packet['packet_sha256'][:12]}.json"
    if not template_path.exists():
        write_json(template_path, template)
    return packet, private


def apply_reviews(path, packet, private):
    review = load(path)
    if review.get("version") != VERSION or review.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("Reviews are not bound to this exact public answer packet")
    expected = {row["review_id"]: row for row in private}
    public = {row["review_id"]: row for row in packet["records"]}
    labels = {}
    for row in review.get("records", []):
        if set(row) != {"review_id", "content_sha256", "correct", "reason"}:
            raise ValueError("Review fields differ; arm and query IDs belong only in the private mapping")
        identity = row["review_id"]
        if identity in labels or identity not in expected:
            raise ValueError("Duplicate or unexpected review ID")
        bound = expected[identity]
        if row["content_sha256"] != bound["content_sha256"]:
            raise ValueError("Review answer SHA256 differs")
        if type(row["correct"]) is not bool or not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Every answer needs a Boolean fresh judgment and review reason")
        if row["correct"] and not public[identity]["final_text"].strip():
            raise ValueError("An absent public final cannot be marked correct")
        labels[identity] = row["correct"]
    if set(labels) != set(expected):
        raise ValueError("Fresh reviews are missing for saved experiment results")
    return {(row["variant"], qid(row["query_id"])): labels[row["review_id"]] for row in private}


def metrics(row):
    runner = row.get("runner_specific") or {}
    ranking = runner.get("reranker_summary") or {}
    search = ranking.get("ablation_search") or {}
    usage = row.get("agent_usage") or {}
    tools = row.get("tool_usage") or {}
    return {"n_turns": row.get("n_turns", 0), "elapsed_seconds": row.get("elapsed_seconds", 0),
        "agent_input_tokens": usage.get("input_tokens", 0),
        "agent_output_tokens": usage.get("output_tokens", 0),
        "agent_cached_input_tokens": usage.get("cached_input_tokens", 0),
        "agent_reasoning_tokens": usage.get("reasoning_tokens", 0),
        "model_search_calls": tools.get("search", 0),
        "retrieval_calls": runner.get("n_bm25_searches", 0),
        "retrieval_queries": runner.get("n_bm25_queries", 0),
        "search_page_calls": search.get("search_page_calls", 0),
        "model_rerank_calls": tools.get("rerank", 0),
        "auto_rerank_calls": search.get("auto_rerank_calls", 0),
        "backend_scoring_calls": ranking.get("scoring_calls", 0),
        "backend_scored_passages": ranking.get("scored_passages", 0),
        "backend_input_tokens": ranking.get("input_tokens", 0),
        "backend_scoring_seconds": ranking.get("scoring_seconds", 0),
        "visible_source_count": row.get("n_surfaced", len(row.get("surfaced_relpaths") or [])),
        "gold_source_covered": int(row.get("coverage_any") is True)}


def summarize(cases, cohort, historical, reference, labels=None):
    result = {"version": VERSION, "expected_question_runs": len(cohort) * len(VARIANTS),
              "completed_question_runs": len(cases), "historical_v14": reference,
              "correctness_source": "fresh_answer_bound_reviews" if labels is not None else "unreviewed",
              "arms": {}}
    for variant in VARIANTS:
        rows = {identity: row for (arm, identity), row in cases.items() if arm == variant}
        totals = Counter()
        for row in rows.values():
            totals.update(metrics(row))
        reviewed = sum((variant, identity) in (labels or {}) for identity in rows)
        full = set(rows) == set(cohort) and reviewed == len(cohort)
        correct = sum(labels[(variant, identity)] for identity in rows) if reviewed == len(rows) and labels is not None else None
        arm = {"completed": len(rows), "expected": len(cohort), "reviewed": reviewed,
            "status": "complete_reviewed" if full else "partial_or_unreviewed",
            "no_public_final": sum(not row["final_text"].strip() for row in rows.values()),
            "termination_counts": dict(Counter(row["terminated_by"].split(":", 1)[0] for row in rows.values())),
            "totals_for_completed_rows": dict(totals),
            "correct": correct, "accuracy": correct / len(cohort) if full else None}
        if full:
            wins = sum(labels[(variant, identity)] and not historical[identity] for identity in cohort)
            losses = sum(not labels[(variant, identity)] and historical[identity] for identity in cohort)
            n = wins + losses
            pvalue = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)) if n else 1.0
            arm["paired_vs_historical_v14"] = {"wrong_to_correct": wins, "correct_to_wrong": losses,
                "accuracy_delta_percentage_points": 100 * (correct - sum(historical.values())) / len(cohort),
                "mcnemar_exact_p": pvalue, "limitation": "historical control; execution batches differ"}
        result["arms"][variant] = arm
    return result


def report_markdown(summary):
    lines = ["# v14 消融实验统计", "", "复用历史 v14 对照；新实验的占位 judge/is_correct 不用于判分。",
        "", "| 实验 | 已完成 | 已复核 | 正确率 | 无公开终答 |", "|---|---:|---:|---:|---:|"]
    for variant, arm in summary["arms"].items():
        accuracy = f"{arm['accuracy']:.1%}" if arm["accuracy"] is not None else "待完整运行及新复核"
        lines.append(f"| {variant} | {arm['completed']}/{arm['expected']} | {arm['reviewed']} | {accuracy} | {arm['no_public_final']} |")
    ref = summary["historical_v14"]
    lines += ["", f"历史 v14：{ref['correct']}/{ref['n']}。与新实验比较时保留运行批次差异。",
        "", "metrics.json/metrics.csv 分开记录真实检索、翻页、模型 rerank 与后台评分成本。",
        "review_packet.json 用于隐藏实验组别的终答复核；私有映射保留在 _private_review_map.json。",
        "填写 review_template.json 的 correct 和 reason 后，使用 --reviews 重新分析。",
        "片段相关性、证据命中、完成状态及中间推理均不能代替终答正确性。"]
    return "\n".join(lines) + "\n"


def analyze(suite_root, historical_path, output, reviews=None):
    suite_root, output = Path(suite_root), Path(output)
    cohort = load_cohort(suite_root / "data/queries_100.jsonl")
    historical, reference = historical_reference(historical_path, cohort)
    cases = collect_rows(suite_root, cohort)
    packet, private = review_packets(cases, cohort, output)
    labels = apply_reviews(reviews, packet, private) if reviews is not None else None
    summary = summarize(cases, cohort, historical, reference, labels)
    write_json(output / "metrics.json", summary)
    fields = ["variant", "query_id", "terminated_by", "has_public_final", "correct", *metrics({})]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (variant, identity), row in sorted(cases.items()):
            writer.writerow({"variant": variant, "query_id": identity,
                "terminated_by": row["terminated_by"], "has_public_final": bool(row["final_text"].strip()),
                "correct": labels.get((variant, identity)) if labels is not None else "", **metrics(row)})
    (output / "comparison.md").write_text(report_markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=ROOT)
    parser.add_argument("--historical-evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    historical = args.historical_evaluation or args.suite_root / "data/historical_v14_evaluation.json"
    output = args.output or args.suite_root / "reports/analysis"
    summary = analyze(args.suite_root, historical, output, args.reviews)
    print(json.dumps({"completed_question_runs": summary["completed_question_runs"],
        "expected_question_runs": summary["expected_question_runs"], "report": str(output / "comparison.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
