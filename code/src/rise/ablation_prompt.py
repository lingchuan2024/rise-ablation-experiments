"""Model-facing contracts for the independent v14 ablation arms."""
from __future__ import annotations

import copy

from .independent_rerank_prompt import (RERANK_FILE_SUBSET_EXAMPLE,
    _read_guidance as _v14_read_guidance, build_independent_rerank_prompt,
    v13_read_tool_spec, v14_rerank_tool_spec)
from .tools import bm25_search_tool_spec


ABLATION_PROMPT_VERSION = "v14-ablations-prompt-v1"
VARIANTS = {"search_paging", "auto_rerank", "no_bash"}
_FINAL = (
    "When required facts are supported, answer. If evidence remains insufficient, "
    "state the unresolved part rather than inventing it. Output exactly one final response:\n"
    "Explanation: {brief evidence-based explanation}\n"
    "Exact Answer: {succinct answer, or Unable to determine}\n"
    "Confidence: {0-100%}"
)


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")


def ablation_search_tool_spec(variant):
    _variant(variant)
    spec = copy.deepcopy(bm25_search_tool_spec())
    fn = spec["function"]
    if variant == "search_paging":
        fn["description"] += (
            " Each subquery returns a search_id and next_offset. To inspect later "
            "originally ranked document snippets, call search with only search_id "
            "and offset. Each page contains at most 10 documents from that fixed "
            "search record; paging does not retrieve or import documents. New "
            "searches do not invalidate these document-search records."
        )
        fn["parameters"]["properties"].update({
            "search_id": {"type": "string", "description": "Previously returned document-search ID. Use without queries."},
            "offset": {"type": "integer", "minimum": 0,
                       "description": "Zero-based document offset for an existing search_id; default 0, fixed page size 10."}})
        fn["parameters"]["required"] = []
    elif variant == "auto_rerank":
        fn["description"] = (
            "Search the full corpus with one or more complementary queries. Each "
            "query imports its own complete top-1000 document set into the "
            "accumulated original-document pool. The system automatically ranks "
            "body passages within that query's own candidates using the same query "
            "and returns at most ten complete passages with original-source "
            "file_path/read_args. This search has no result-pagination interface. "
            "Use read to verify relevant passages in their original context."
        )
    return spec


def ablation_read_tool_spec(variant, baseline_spec, *, read_default_limit=2000):
    _variant(variant)
    spec = v13_read_tool_spec(baseline_spec, read_default_limit=read_default_limit)
    if variant == "no_bash":
        spec["function"]["description"] = _remove_read_command_recovery(
            spec["function"]["description"])
    elif variant == "auto_rerank":
        spec["function"]["description"] = spec["function"]["description"].replace(
            "rerank item's", "search item's")
        prop = spec["function"]["parameters"]["properties"]["limit"]
        prop["description"] = prop["description"].replace(
            "rerank item's", "search item's"
        )
    return spec


def ablation_rerank_tool_spec(variant, baseline_spec):
    _variant(variant)
    if variant == "auto_rerank":
        raise ValueError("auto_rerank has no independently callable ranking tool")
    if variant == "search_paging":
        return v14_rerank_tool_spec(baseline_spec)
    spec = copy.deepcopy(baseline_spec)
    spec["function"]["description"] = spec["function"]["description"].replace(
        "\n\n" + RERANK_FILE_SUBSET_EXAMPLE, ""
    ).replace("read/bash coordinates, without restricting bash/read.",
              "read coordinates, without restricting read.")
    prop = spec["function"]["parameters"]["properties"]["file_paths"]
    prop["description"] = prop["description"].replace(
        ", such as files selected with bash", ""
    )
    return spec


def _remove_read_command_recovery(text):
    return text.replace(
        "If read_may_truncate is true, use the item's bash_args for exact passage recovery.",
        "If read_may_truncate is true, the returned source context may be clipped by this per-line limit."
    )


def _no_bash_prompt(*, document_guide, read_default_limit):
    """Preserve v14 verbatim outside instructions for the removed capability."""
    prompt = build_independent_rerank_prompt(
        document_guide=document_guide, read_default_limit=read_default_limit)
    start = prompt.index("- bash(command, timeout?):")
    end = prompt.index("- read(file_path, offset?, limit?):", start)
    prompt = prompt[:start] + prompt[end:]
    prompt = prompt.replace(RERANK_FILE_SUBSET_EXAMPLE + "\n\n", "")
    prompt = _remove_read_command_recovery(prompt)
    return prompt.replace("four independently available tools:", "three independently available tools:").replace(
        "imported source paths, such as files selected by bash, to restrict ",
        "imported source paths to restrict "
    ).replace("restricts bash/read.", "restricts read.").replace(
        "Choose search, bash, read or rerank according to the missing evidence. "
        "You may use bash/read directly after search, rerank the cumulative "
        "pool directly, or use bash to choose files and rerank that subset. "
        "Neither bash nor rerank is a prerequisite for the other. ",
        "Choose search, read or rerank according to the missing evidence. "
        "You may use read directly after search, rerank the cumulative pool "
        "directly, or explicitly select imported files and rerank that subset. "
        "Read and rerank are independently available. "
    )


def build_ablation_prompt(variant, *, document_guide="", read_default_limit=2000):
    _variant(variant)
    if variant == "search_paging":
        base = build_independent_rerank_prompt(
            document_guide=document_guide, read_default_limit=read_default_limit)
        return base + (
            "\n\nDocument-search pagination: search(queries) also returns each "
            "subquery's search_id and next_offset. To view ranks 11–20, call "
            "search with that search_id and offset=10, then follow next_offset. "
            "Each page shows at most ten original document-opening snippets "
            "in the original BM25 order; no new retrieval or import occurs. "
            "New searches do not invalidate these search_ids. A null next_offset "
            "means only that this search record is exhausted. Use only queries "
            "for a new search, or only search_id/offset for a page."
        )
    if variant == "auto_rerank":
        baseline = build_independent_rerank_prompt(read_default_limit=read_default_limit)
        start = baseline.index("- bash(command, timeout?):")
        end = baseline.index("- read(file_path, offset?, limit?):", start)
        bash_paragraph = baseline[start:end]
        read_guidance = _v14_read_guidance(read_default_limit).replace(
            "rerank item's", "search item's")
        return (
            "You answer research questions using retrieved source evidence in "
            "one conversation. You have three tools:\n"
            "- search(queries): each query imports its complete top-1000 document "
            "set into the accumulated original-document pool. The system "
            "automatically selects up to 1000 body passages using passage BM25 "
            "within that query's own candidates, then applies a neural relevance "
            "model and returns the first ten passages. The original search query "
            "is used for both stages. Passages use 128 tokens with 32-token overlap. "
            "Only search imports additional documents. There is no search-results "
            "pagination. Empty local results do not establish absence in the corpus.\n"
            + bash_paragraph +
            "- read(file_path, offset?, limit?): "
            + read_guidance + "\n"
            + document_guide +
            "Choose search, bash or read according to the missing evidence. "
            "Use specific known entity names and the remaining relationship "
            "when refining queries. Split multiple conditions into focused "
            "checks. Keep alternative candidates and source conflicts until "
            "conditions are verified. Do not repeat unchanged results or "
            "unsuccessful commands indefinitely; change the query, scope, source "
            "or investigation when no new evidence appears. " + _FINAL
        )
    return _no_bash_prompt(document_guide=document_guide,
                           read_default_limit=read_default_limit)
