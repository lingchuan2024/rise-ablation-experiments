#!/usr/bin/env python3
"""Analyze downloaded public metadata from the frozen Qwen3.8 first10 pilot.

Reads the manifest, cohort, terminal rows and numeric API transport only. No
raw trace/checkpoint is opened; placeholder judge/is_correct/cost is ignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("search_paging", "auto_rerank", "no_bash")
QIDS = (70, 81, 96, 116, 122, 127, 132, 153, 156, 161)
MODEL = "qwen3.8-flash"
ENDPOINT = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
COHORT_SHA = "129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c"
RERANKER_PIN = "6e9e69830b95c52b5fd889b7690dda3329508de3"
VERSION = "hosted-qwen38-analysis-v1"
TOKEN_FIELDS = ("prompt_tokens", "cached_input_tokens", "uncached_input_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def final_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_config_hash(value):
    # Exact canonicalization used by the frozen trajectory.py implementation;
    # unlike manifest digests, its JSON separators keep the default spaces.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        default=str).encode()).hexdigest()[:12]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def query_id(value):
    if isinstance(value, bool):
        raise ValueError("Boolean query ID is invalid")
    number = int(value)
    if str(number) != str(value):
        raise ValueError("Query ID must be an integer")
    return number


def number(value, name, *, integer=False, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("Invalid nonnegative numeric field: " + name)
    return int(value) if integer else value


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_manifest(snapshot, cohort_path):
    snapshot = Path(snapshot)
    path = snapshot / "server/hosted_pilot/reports/run_manifest.json"
    manifest = load(path)
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("version") != "v14-hosted-qwen38-first10-v1" or digest(payload) != manifest.get("manifest_sha256"):
        raise ValueError("Downloaded hosted manifest checksum/version differs")
    cfg = manifest.get("config", {})
    expected = {"model": MODEL, "base_url": ENDPOINT, "concurrency": 3,
        "question_count_per_variant": 10, "question_runs": 30, "source_cohort_sha256": COHORT_SHA,
        "max_turns": 100, "wall_clock_seconds_per_question": 3600, "read_default_limit": 2000,
        "per_call_max_completion_tokens": 32000, "thinking_mode": "enabled",
        "reasoning_effort": "medium", "preserve_thinking": True,
        "bm25_k": 1000, "bm25_top_n_preview": 10, "bm25_snippet_chars": 100,
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B", "reranker_revision": RERANKER_PIN}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Downloaded manifest does not describe the authorized hosted pilot")
    allowed = cfg.get("allowed_reported_models")
    if not isinstance(allowed, list) or MODEL not in allowed or any(not isinstance(value, str) or not value for value in allowed):
        raise ValueError("Manifest needs explicit allowed response-model identities")
    if manifest.get("query_ids") != list(QIDS) or manifest.get("variant_order") != list(VARIANTS) or manifest.get("question_runs") != 30:
        raise ValueError("All three arms must use the exact original first10 cohort")
    full_path = Path(cohort_path)
    if sha(full_path) != COHORT_SHA:
        raise ValueError("Original fixed100 cohort bytes differ")
    full = [json.loads(line) for line in full_path.read_text().splitlines() if line.strip()]
    cohort = {query_id(row["query_id"]): row for row in full[:10]}
    if list(cohort) != list(QIDS):
        raise ValueError("Original first10 query identities differ")
    queries_path = snapshot / "server/hosted_pilot/queries_10.jsonl"
    if sha(queries_path) != manifest.get("queries_sha256"):
        raise ValueError("Downloaded pilot cohort checksum differs")
    queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
    if queries != full[:10]:
        raise ValueError("Downloaded pilot questions/gold differ from original first10")
    provenance = {str(path.relative_to(snapshot)): sha(path), str(queries_path.relative_to(snapshot)): sha(queries_path)}
    return manifest, cohort, provenance


def validate_row(row, variant, cohort, config):
    qid = query_id(row["query_id"])
    if qid not in cohort or row.get("model") != MODEL:
        raise ValueError("Saved result has an unexpected query ID/model")
    if row.get("question") != cohort[qid]["query"] or row.get("gold_answer") != cohort[qid]["answer"] or not isinstance(row.get("final_text"), str):
        raise ValueError("Saved public answer/question/gold identity differs")
    rc = row.get("run_config") or {}
    if row.get("run_config_hash") != run_config_hash(rc):
        raise ValueError("Saved run_config_hash differs from its canonical configuration")
    extras = rc.get("extras") or {}
    required = {"ablation_variant": variant, "hosted_pilot_version": "v14-hosted-qwen38-first10-v1",
        "hosted_endpoint": ENDPOINT, "hosted_enable_thinking": True, "hosted_preserve_thinking": True,
        "hosted_reasoning_effort": "medium", "hosted_completion_budget_kind": "max_completion_tokens",
        "hosted_total_completion_tokens": 32000}
    required.update(bm25_k=1000, bm25_top_n_preview=10, bm25_snippet_chars=100,
        read_default_limit=2000, structured_doc_mode=True, no_bash=variant == "no_bash",
        enable_sandbox=True, reranker_passage_tokens=128, reranker_passage_overlap_tokens=32,
        reranker_model="Qwen/Qwen3-Reranker-0.6B", reranker_revision=RERANKER_PIN)
    if any(extras.get(key) != value for key, value in required.items()) or rc.get("agent_model") != MODEL:
        raise ValueError("Saved result has the wrong frozen hosted configuration")
    for key, expected in {"max_model_calls": 100, "per_call_max_tokens": 32000, "wall_clock_timeout_sec": 3600, "api_max_retries": 3}.items():
        if rc.get(key) != expected:
            raise ValueError("Saved result budget differs: " + key)
    if extras.get("hosted_allowed_reported_models") != config.get("allowed_reported_models"):
        raise ValueError("Saved allowed response-model identities differ from manifest")
    resources = {"index_dir": "document_index", "filename_map": "filename_map", "bc_plus_docs": "corpus",
        "space_passage_index": "passage_index", "reranker_model": "reranker_model",
        "reranker_revision": "reranker_revision", "reranker_device": "reranker_device",
        "reranker_dtype": "reranker_dtype", "hosted_reranker_sharing_policy": "reranker_sharing_policy"}
    if any(key not in config or extras.get(field) != config[key] for field, key in resources.items()):
        raise ValueError("Saved result resource identity differs from the frozen manifest")
    if not isinstance(row.get("terminated_by"), str) or not row["terminated_by"]:
        raise ValueError("A terminal result needs its termination reason")
    if not isinstance(row.get("tool_usage"), dict):
        raise ValueError("Saved tool counters are absent")
    number(row.get("elapsed_seconds"), "elapsed_seconds")
    number(row.get("n_turns"), "n_turns", integer=True)
    row_metrics(row)  # Refuse absent/contradictory core counters before reporting.
    return qid


def collect_rows(snapshot, cohort, manifest, provenance):
    cases, infrastructure = {}, {}
    for variant in VARIANTS:
        directory = Path(snapshot) / "runs/hosted_qwen38_first10" / variant
        binding = directory / "_hosted_manifest.json"
        if binding.exists():
            value = load(binding)
            if value.get("manifest_sha256") != manifest["manifest_sha256"] or value.get("variant") != variant:
                raise ValueError("Downloaded output manifest binding differs")
            provenance[str(binding.relative_to(snapshot))] = sha(binding)
        for path in sorted((directory / "_per_query").glob("*.json")):
            row = load(path)
            qid = validate_row(row, variant, cohort, manifest["config"])
            key = (variant, qid)
            if key in cases or key in infrastructure or path.name != f"qid_{qid}.json":
                raise ValueError("Duplicate/misnamed result for one arm/query")
            provenance[str(path.relative_to(snapshot))] = sha(path)
            reason = row["terminated_by"].split(":", 1)[0]
            if reason.startswith(("api_error", "infrastructure", "solver_api_failed")):
                infrastructure[key] = row
            else:
                cases[key] = row
    return cases, infrastructure


def usage_tokens(usage):
    usage = usage if isinstance(usage, dict) else {}
    result = {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": nested(usage, "prompt_tokens_details", "cached_tokens"),
        "reasoning_tokens": nested(usage, "completion_tokens_details", "reasoning_tokens")}
    alternative = usage.get("prompt_cache_hit_tokens")
    if result["cached_input_tokens"] is None:
        result["cached_input_tokens"] = alternative
    elif alternative is not None and result["cached_input_tokens"] != alternative:
        raise ValueError("Conflicting cached-token counters in API usage")
    if result["reasoning_tokens"] is None:
        result["reasoning_tokens"] = usage.get("reasoning_tokens")
    result = {key: number(value, key, integer=True, nullable=True) for key, value in result.items()}
    prompt, cached, output, reasoning, total = (result[key] for key in ("prompt_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"))
    if prompt is not None and cached is not None and cached > prompt:
        raise ValueError("API cached tokens exceed inclusive prompt tokens")
    if reasoning is not None and output is not None and reasoning > output:
        raise ValueError("API reasoning tokens exceed output tokens")
    if None not in (prompt, output, total) and total != prompt + output:
        raise ValueError("API total tokens differ from prompt + completion")
    result["uncached_input_tokens"] = prompt - cached if prompt is not None and cached is not None else None
    return result


def collect_transport(snapshot, cohort, manifest, provenance):
    result = {}
    seen = set()
    for variant in VARIANTS:
        path = Path(snapshot) / "runs/hosted_qwen38_first10" / variant / "api_transport.jsonl"
        records = []
        if path.exists():
            raw = path.read_bytes()
            lines = raw.decode("utf-8").splitlines()
            provenance[str(path.relative_to(snapshot))] = hashlib.sha256(raw).hexdigest()
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if line_number == len(lines) and not raw.endswith(b"\n"):
                        raise ValueError("活动快照尾行不完整，请重新同步: " + str(path)) from exc
                    raise ValueError(f"Invalid transport JSON at {path}:{line_number}") from exc
                qid = query_id(row["query_id"])
                if qid not in cohort or row.get("requested_model") != MODEL:
                    raise ValueError("Formal transport has an unexpected model/query ID")
                if row.get("thinking_mode") != "enabled" or row.get("preserve_thinking") is not True or row.get("reasoning_effort") != "medium" or row.get("requested_max_completion_tokens") != 32000:
                    raise ValueError("Formal transport request shape differs from the frozen pilot")
                if row.get("status") not in {"ok", "error"}:
                    raise ValueError("Unknown formal transport status")
                if row.get("status") == "ok" and row.get("reported_model") not in manifest["config"]["allowed_reported_models"]:
                    raise ValueError("Formal response model is outside the exact allowlist")
                response_id = row.get("response_id")
                if response_id:
                    if response_id in seen:
                        raise ValueError("Duplicate API response ID: do not concatenate repeated snapshots")
                    seen.add(response_id)
                number(row.get("unix_time"), "unix_time")
                number(row.get("request_seconds"), "request_seconds")
                row["_tokens"] = usage_tokens(row.get("usage"))
                row["_qid"] = qid
                records.append(row)
        result[variant] = {"present": path.exists(), "records": records}
    return result


def apply_reviews(path, cases, manifest):
    if path is None:
        return {}, None
    value = load(path)
    if value.get("version") != VERSION or value.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Fresh reviews must bind this exact hosted manifest/version")
    labels = {}
    for row in value.get("records", []):
        key = (row.get("variant"), query_id(row["query_id"]))
        if key not in cases or key in labels:
            raise ValueError("Review has a duplicate/uncompleted/unexpected arm/query")
        if row.get("final_text_sha256") != final_sha(cases[key]["final_text"]):
            raise ValueError("Fresh review final_text SHA256 differs")
        if type(row.get("correct")) is not bool or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("Fresh review needs a Boolean strict label and nonempty reason")
        if "sensitivity_correct" in row and type(row["sensitivity_correct"]) is not bool:
            raise ValueError("Optional sensitivity label must be Boolean")
        if not cases[key]["final_text"].strip() and (row["correct"] or row.get("sensitivity_correct") is True):
            raise ValueError("An absent public final cannot be marked correct")
        labels[key] = row
    return labels, {"path": str(Path(path).resolve()), "sha256": sha(path), "records": len(labels)}


def row_metrics(row):
    runner = row.get("runner_specific")
    ranking = (runner or {}).get("reranker_summary")
    if not isinstance(runner, dict) or not isinstance(ranking, dict):
        raise ValueError("Core runner/reranker counters are absent")
    for key in ("n_bm25_searches", "n_bm25_queries"):
        if key not in runner:
            raise ValueError("Core retrieval counter is absent: " + key)
        number(runner[key], key, integer=True)
    for key in ("rerank_calls", "page_calls", "scoring_calls", "scored_passages", "input_tokens", "scoring_seconds", "preparation_seconds"):
        if key not in ranking:
            raise ValueError("Core reranker counter is absent: " + key)
    variant = row["run_config"]["extras"]["ablation_variant"]
    search = ranking.get("ablation_search") or {}
    if variant in {"search_paging", "auto_rerank"}:
        required = {"retrieval_calls", "retrieval_queries", "search_page_calls", "tool_errors"}
        if variant == "auto_rerank":
            required |= {"auto_rerank_calls", "auto_rerank_errors", "empty_scopes"}
        if not isinstance(search, dict) or not required.issubset(search):
            raise ValueError("Core ablation search counters are absent")
        for key in required:
            number(search[key], "ablation_search." + key, integer=True)
        if search["retrieval_calls"] != runner["n_bm25_searches"] or search["retrieval_queries"] != runner["n_bm25_queries"]:
            raise ValueError("Successful retrieval counters disagree")
    tools = row["tool_usage"]
    for key, value in tools.items():
        number(value, "tool_usage." + key, integer=True)
    metrics = {"turns": row["n_turns"], "elapsed_seconds": row["elapsed_seconds"],
        "model_search_calls": tools.get("search", 0), "model_bash_calls": tools.get("bash", 0),
        "model_read_calls": tools.get("read", 0), "manual_rerank_calls": tools.get("rerank", 0),
        "retrieval_calls": runner["n_bm25_searches"],
        "retrieval_queries": search["retrieval_queries"] if variant in {"search_paging", "auto_rerank"} else runner["n_bm25_queries"],
        "runner_retrieval_query_counter": runner["n_bm25_queries"],
        "search_page_calls": search.get("search_page_calls", 0), "auto_rerank_calls": search.get("auto_rerank_calls", 0),
        "ranking_service_calls": ranking.get("rerank_calls", 0),
        "rerank_page_calls": ranking.get("page_calls", 0), "backend_scoring_calls": ranking.get("scoring_calls", 0),
        "backend_scored_passages": ranking.get("scored_passages", 0), "backend_input_tokens": ranking.get("input_tokens", 0),
        "backend_scoring_seconds": ranking.get("scoring_seconds", 0), "backend_preparation_seconds": ranking.get("preparation_seconds", 0)}
    registered = metrics["retrieval_calls"] + metrics["search_page_calls"]
    if metrics["model_search_calls"] < registered:
        raise ValueError("Model search count is smaller than successful retrieval/page counts")
    if variant == "auto_rerank" and (metrics["search_page_calls"] or metrics["rerank_page_calls"]):
        raise ValueError("Automatic ranking has no model-visible result pagination")
    if variant == "auto_rerank" and metrics["auto_rerank_calls"] + search["empty_scopes"] != metrics["retrieval_queries"]:
        raise ValueError("Automatic ranking/empty-scope counters differ from subqueries")
    metrics["search_failed_or_unregistered_calls"] = metrics["model_search_calls"] - registered
    metrics["model_tool_calls_total"] = sum(tools.values())
    metrics["forbidden_tool_name_attempts"] = (tools.get("bash", 0) if variant == "no_bash" else
                                              tools.get("rerank", 0) if variant == "auto_rerank" else 0)
    for key, value in metrics.items():
        number(value, key, integer=not key.endswith("seconds"))
    metrics["runner_query_counter_delta"] = metrics["runner_retrieval_query_counter"] - metrics["retrieval_queries"]
    metrics["query_counter_mismatch_rows"] = int(metrics["runner_query_counter_delta"] != 0)
    return metrics


def summarize_transport(records, *, present=True):
    totals = {field: sum(row["_tokens"][field] for row in records if row["_tokens"][field] is not None) for field in TOKEN_FIELDS}
    missing = {field: sum(row["_tokens"][field] is None for row in records) for field in TOKEN_FIELDS}
    span = max((row["unix_time"] + row["request_seconds"] for row in records), default=0) - min((row["unix_time"] for row in records), default=0)
    return {"status": "observed" if present else "not_downloaded_or_not_started", "requests": len(records),
        "response_status_counts": dict(Counter(row["status"] for row in records)),
        "response_delivery_counts": dict(Counter(str(row.get("response_delivered")) for row in records)),
        "requests_without_usage": sum(not isinstance(row.get("usage"), dict) for row in records),
        "known_token_totals": totals, "token_fields_missing_records": missing,
        "request_seconds_sum": sum(row["request_seconds"] for row in records),
        "observed_request_span_seconds": span}


def cost_estimate(transport, pricing):
    if pricing is None:
        return {"status": "not_configured", "amount": None, "currency": None,
                "note": "Placeholder row costs are ignored; unconfigured pricing does not mean free."}
    required = ("uncached_input_per_million", "cached_input_per_million", "output_per_million")
    for key in required:
        number(pricing.get(key), key)
    if not isinstance(pricing.get("currency"), str) or not pricing["currency"].strip():
        raise ValueError("Pricing needs an explicit currency")
    tokens, missing = transport["known_token_totals"], transport["token_fields_missing_records"]
    fields = ("uncached_input_tokens", "cached_input_tokens", "output_tokens")
    known = sum(tokens[field] * pricing[rate] / 1_000_000 for field, rate in zip(fields, required))
    complete = transport["status"] == "observed" and not any(missing[field] for field in fields)
    return {"status": "estimated_from_complete_known_usage" if complete else "partial_known_usage",
        "amount": known if complete else None, "known_usage_amount": known, "currency": pricing["currency"],
        "note": "User-provided rates; reasoning is already included in output. Not an invoice."}


def summarize(cases, infrastructure, ledgers, labels, manifest, provenance, review_source, pricing=None):
    result = {"version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "endpoint": ENDPOINT, "manifest_sha256": manifest["manifest_sha256"],
        "expected_question_runs": 30, "completed_question_runs": len(cases),
        "reviewed_question_runs": len(labels), "review_source": review_source,
        "source_sha256": provenance, "raw_traces_opened": False,
        "comparison_policy": "Only these three same-model, same-first10 arms; no causal comparison with historical Qwen3.5 v14.",
        "token_semantics": "API prompt includes cached input; reasoning is a subset of completion. total=prompt+completion. Known totals may omit missing usage; inspect missing-record counts.",
        "counter_semantics": "Model tool counters include failed/unknown-tool attempts, not proof of execution; forbidden tool names are reported separately. Search calls include snippet pages and manual rerank calls include ranking pages. A/B successful normalized subqueries agree with the final rewritten runner counts; C retains baseline runner query counting. Automatic rerank counts subqueries. Turns and tool calls are separate counters.",
        "arms": {}}
    for variant in VARIANTS:
        rows = {qid: row for (arm, qid), row in cases.items() if arm == variant}
        reviewed = {qid: labels[(variant, qid)] for qid in rows if (variant, qid) in labels}
        full = set(rows) == set(QIDS) and set(reviewed) == set(QIDS)
        records = ledgers[variant]["records"]
        transport = summarize_transport(records, present=ledgers[variant]["present"])
        completed_transport = summarize_transport([row for row in records if row["_qid"] in rows], present=ledgers[variant]["present"])
        totals = Counter()
        for row in rows.values():
            totals.update(row_metrics(row))
        tool_totals = Counter()
        for row in rows.values():
            tool_totals.update(row["tool_usage"])
        strict_correct = sum(row["correct"] for row in reviewed.values()) if reviewed else None
        overrides = sum("sensitivity_correct" in row for row in reviewed.values())
        sensitivity_correct = sum(row.get("sensitivity_correct", row["correct"]) for row in reviewed.values()) if overrides else None
        result["arms"][variant] = {"expected_query_ids": list(QIDS), "completed_query_ids": sorted(rows),
            "completed": len(rows), "pending_query_ids": [qid for qid in QIDS if qid not in rows],
            "observed_unfinished_query_ids": sorted({row["_qid"] for row in records} - set(rows)),
            "infrastructure_query_ids": sorted(qid for arm, qid in infrastructure if arm == variant),
            "termination_counts": dict(Counter(row["terminated_by"].split(":", 1)[0] for row in rows.values())),
            "no_public_final": sum(not row["final_text"].strip() for row in rows.values()),
            "reviewed": len(reviewed), "correct_reviewed": strict_correct,
            "accuracy_status": "complete_fresh_reviewed" if full else "pending",
            "accuracy": strict_correct / 10 if full else None,
            "sensitivity": {"provided_labels": overrides, "correct_with_provided_overrides": sensitivity_correct,
                "changed_labels": sum(row.get("sensitivity_correct", row["correct"]) != row["correct"] for row in reviewed.values()),
                "accuracy": sensitivity_correct / 10 if full and overrides else None,
                "rule": "Use provided sensitivity_correct only where present; retain strict labels elsewhere."},
            "completed_row_totals": dict(totals),
            "tool_usage_totals": dict(tool_totals),
            "mean_completed_elapsed_seconds": totals["elapsed_seconds"] / len(rows) if rows else None,
            "api_all_observed": transport, "api_for_completed_queries": completed_transport,
            "cost": cost_estimate(transport, pricing)}
    return result


def markdown(summary):
    lines = ["# Qwen3.8 Flash 三组各十题试跑", "", "| 实验 | 已完成 | 严格正确/已审 | 敏感性正确/已审 | 完整严格正确率 | API总token（已知） |", "|---|---:|---:|---:|---:|---:|"]
    for variant, arm in summary["arms"].items():
        accuracy = f"{arm['accuracy']:.0%}" if arm["accuracy"] is not None else "pending"
        tokens = arm["api_all_observed"]["known_token_totals"]["total_tokens"]
        missing = arm["api_all_observed"]["token_fields_missing_records"]["total_tokens"]
        token_text = f"{tokens:,}" if arm["api_all_observed"]["status"] == "observed" else "未观察"
        reviewed = f"{arm['correct_reviewed']}/{arm['reviewed']}" if arm["reviewed"] else "pending"
        sensitivity = arm["sensitivity"]["correct_with_provided_overrides"]
        sensitivity_text = f"{sensitivity}/{arm['reviewed']}" if sensitivity is not None else "—"
        lines.append(f"| {variant} | {arm['completed']}/10 | {reviewed} | {sensitivity_text} | {accuracy} | {token_text}" + (f"（另{missing}条缺usage）" if missing else "") + " |")
    lines += ["", "正确率只来自绑定本次manifest和公开final_text SHA256的新人工评审。占位is_correct、judge和零费用字段均未用于结论。",
        "", "API汇总包含已观察到的活动题请求；另列已完成qid的请求。缓存属于prompt，推理属于output，不重复相加。缺失usage不是零消耗。",
        "", "单题耗时之和、API请求耗时之和及观察到的请求时间跨度分别记录，均不冒充正式运行墙钟。费用未配置时为unknown，不视作免费。",
        "", "只比较本次同模型、同十题的三组；不与不同模型的历史v14作因果比较。没有读取或发布raw trace/checkpoint。"]
    return "\n".join(lines) + "\n"


def analyze(snapshot, output, *, cohort_path=ROOT / "data/queries_100.jsonl", reviews=None, pricing=None):
    snapshot, output = Path(snapshot), Path(output)
    manifest, cohort, provenance = load_manifest(snapshot, cohort_path)
    cases, infrastructure = collect_rows(snapshot, cohort, manifest, provenance)
    ledgers = collect_transport(snapshot, cohort, manifest, provenance)
    labels, review_source = apply_reviews(reviews, cases, manifest)
    rates = load(pricing) if pricing is not None else None
    summary = summarize(cases, infrastructure, ledgers, labels, manifest, provenance, review_source, rates)
    if pricing is not None:
        summary["pricing_source"] = {"path": str(Path(pricing).resolve()), "sha256": sha(pricing)}
    write_json(output / "metrics.json", summary)
    per_query = []
    for (variant, qid), row in sorted(cases.items()):
        label = labels.get((variant, qid))
        api = summarize_transport([item for item in ledgers[variant]["records"] if item["_qid"] == qid], present=ledgers[variant]["present"])
        per_query.append({"variant": variant, "query_id": qid, "terminated_by": row["terminated_by"],
            "final_text_sha256": final_sha(row["final_text"]), "review_status": "reviewed" if label else "pending",
            "strict_correct": label["correct"] if label else None,
            "sensitivity_correct": label.get("sensitivity_correct") if label else None,
            **row_metrics(row), **{"api_" + key: value for key, value in api["known_token_totals"].items()},
            "api_requests": api["requests"], "api_requests_missing_usage": api["requests_without_usage"]})
    write_json(output / "per_query_metrics.json", per_query)
    if per_query:
        with (output / "per_query_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
            writer.writeheader()
            writer.writerows(per_query)
    (output / "report.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "reports/hosted_qwen38_first10")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohort", type=Path, default=ROOT / "data/queries_100.jsonl")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--pricing", type=Path)
    args = parser.parse_args()
    output = args.output or args.snapshot / "analysis"
    summary = analyze(args.snapshot, output, cohort_path=args.cohort, reviews=args.reviews, pricing=args.pricing)
    print(json.dumps({"completed_question_runs": summary["completed_question_runs"],
        "reviewed_question_runs": summary["reviewed_question_runs"], "expected_question_runs": 30,
        "report": str(output / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
