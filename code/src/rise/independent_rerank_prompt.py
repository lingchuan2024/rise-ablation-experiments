"""v14 guidance for explicit, optional file-subset passage reranking."""
from __future__ import annotations

import copy


RERANK_PROMPT_VERSION = "v14-explicit-file-subset-prompt-v1"


RERANK_FILE_SUBSET_EXAMPLE = (
    "Optional file-subset example (the entity names and paths below are "
    "fictional placeholders):\n"
    "1. To locate candidate files for an already identified entity, call "
    "bash with {\"command\": \"rg -l -i 'EXAMPLE_ENTITY|EXAMPLE_ALIAS' .\"}. "
    "Suppose it returns only these paths:\n"
    "./docs/example_a.txt\n./docs/example_b.txt\n"
    "2. Explicitly pass those observed paths to rerank, for example: "
    "{\"query\": \"EXAMPLE_ENTITY specific missing relationship\", "
    "\"file_paths\": [\"docs/example_a.txt\", \"docs/example_b.txt\"], "
    "\"bm25_candidates\": 200, \"top_k\": 30, \"limit\": 5}. "
    "BM25 candidate selection and neural reranking then use only that file "
    "subset. The values 200/30/5 are adjustable examples, not required settings.\n"
    "3. Inspect useful passages in their original context by passing an "
    "item's actual returned read_args to read. For example, if read_args is "
    "{\"file_path\": \"docs/example_a.txt\", \"offset\": 40, \"limit\": 12}, "
    "call read with that object; use the returned coordinates rather than "
    "copying the example coordinates.\n"
    "You decide whether this sequence is useful; it is not required for "
    "every question. Do not automatically treat the preceding bash output "
    "as the rerank scope: it may contain body text, errors or truncated "
    "output. Check the returned paths and explicitly set file_paths yourself. "
    "Omitting file_paths still ranks the full accumulated pool. Selecting "
    "a subset affects only that rerank call; bash/read retain access to "
    "the full original-document pool."
)


def v14_rerank_tool_spec(baseline_spec: dict) -> dict:
    """Add the optional handoff example without changing rerank's contract."""
    spec = copy.deepcopy(baseline_spec)
    spec["function"]["description"] += "\n\n" + RERANK_FILE_SUBSET_EXAMPLE
    return spec


def _read_guidance(read_default_limit: int) -> str:
    return (
        "Read an imported original document. Offsets and limits are in LINES; "
        "offsets are zero-based. The default limit is "
        f"{read_default_limit} lines when limit is omitted; this is a fallback, "
        "not a recommended read size. Set limit explicitly for each read. "
        "Prefer a rerank item's read_args or a located section's line range "
        "to inspect the needed original context directly. For an unfamiliar "
        "document without a located range, start with offset=0, limit=60 "
        "to inspect its header and any table of contents, then read the "
        "relevant body sections. If navigation or a long table of contents "
        "fills that preview, advance offset or increase limit explicitly; "
        "do not treat the preview as proof that evidence is absent. "
        "For additional body context, 100–200 lines is a starting point, "
        "not a cap: expand or page as needed, and use a larger explicit "
        "limit when a complete section or document is necessary. These "
        "rules also apply to short or single-topic documents. Line counts "
        "are not token budgets; this tool has no total token/byte output cap. "
        "Individual long lines may be truncated at 2000 characters. "
        "Character coordinates differ from these line arguments. If "
        "read_may_truncate is true, use the item's bash_args for exact "
        "passage recovery."
    )


def v13_read_tool_spec(baseline_spec: dict, *,
                       read_default_limit: int = 2000) -> dict:
    """Align v13 reading guidance without changing the baseline read contract."""
    spec = copy.deepcopy(baseline_spec)
    spec["function"]["description"] = _read_guidance(read_default_limit)
    spec["function"]["parameters"]["properties"]["limit"]["description"] = (
        "Maximum number of lines to read. Set this explicitly: start with "
        "60 lines for an unfamiliar document, or use a known section's "
        "range or a rerank item's read_args. Expand or page when needed; "
        f"{read_default_limit} is only the fallback if omitted."
    )
    return spec


def build_independent_rerank_prompt(*, document_guide: str = "",
                                  read_default_limit: int = 2000) -> str:
    return (
        "You answer research questions using retrieved source evidence in one "
        "conversation. You have four independently available tools:\n\n"
        "- search(queries): search the full document corpus with distinctive "
        "natural-language terms. Each query imports its complete top-1000 "
        "document set into the working directory; only the first ten documents "
        "are previewed. All searches accumulate in the same original-document "
        "pool. A document-opening preview may contain navigation or metadata "
        "rather than evidence. Only search imports additional documents; an "
        "empty pool-local result does not mean the full corpus lacks evidence.\n"
        "- bash(command, timeout?): explore the accumulated original documents "
        "using rg/grep, listing, or focused reading. The working directory is "
        "the original-document pool and remains the same before and after "
        "rerank. Use returned relative source paths; do not guess absolute "
        "locations such as /tmp or /corpus. The documents are read-only "
        "and unrelated files are outside the sandbox. Output is truncated, so "
        "use focused commands. head limits lines, not characters: for long "
        "lines, first select files with rg -l, then extract short matches with "
        "rg -o. Use fixed-string matching for literal decimals and check "
        "numeric boundaries and surrounding meaning.\n"
        "- read(file_path, offset?, limit?): "
        + _read_guidance(read_default_limit) + "\n"
        "- rerank(query, file_paths?, bm25_candidates?, top_k?, offset?, limit?): "
        "find passages relevant to your explicit current question within the "
        "accumulated original-document pool. You choose query yourself. Omit "
        "file_paths, or pass null, for the full imported pool; pass an explicit list of "
        "imported source paths, such as files selected by bash, to restrict "
        "ranking to that subset. An empty list selects no documents. The tool "
        "computes passage BM25 statistics within the chosen scope, selects M "
        "candidates, reranks them with the neural relevance model, and retains "
        "K results. Defaults: bm25_candidates=M=1000, top_k=K=min(100,M), offset=0, "
        "limit=10. Adjust M, K and the page size yourself as useful, with "
        "1 <= K <= M. Passage segmentation is fixed at 128 tokens with "
        "32-token overlap. It returns source paths, complete passages and "
        "original-source read_args. It neither removes original documents nor "
        "restricts bash/read. Scope statistics show the number of selected "
        "documents and passages. A larger M sends more candidates to neural "
        "scoring and costs more time and compute. A larger K makes more of "
        "the already scored ranking available; limit controls only the "
        "number of complete passages returned on this page. Neither M nor K "
        "can find documents outside the imported pool.\n"
        "- rerank(result_id, top_k?, offset?, limit?): page or resize an existing "
        "current ranking using its returned result_id. Do not also supply "
        "query, file_paths or bm25_candidates on this form.\n\n"
        + RERANK_FILE_SUBSET_EXAMPLE + "\n\n"
        + document_guide +
        "Choose search, bash, read or rerank according to the missing evidence. "
        "You may use bash/read directly after search, rerank the cumulative "
        "pool directly, or use bash to choose files and rerank that subset. "
        "Neither bash nor rerank is a prerequisite for the other. You need "
        "not call every tool. When a candidate entity appears, prefer checking "
        "that entity plus the still-unverified relationship or fact. Split "
        "multi-condition questions into focused checks instead of broad lists "
        "of years or generic relationship words. Keep alternative candidates "
        "and source conflicts until the conditions are verified. Relevance "
        "rankings are navigation aids, not "
        "proof of identity, relationships or question conditions. Check useful "
        "passages against their original source context and retain conflicts.\n\n"
        "Use result_id and next_offset to page the same ranking. "
        "next_offset=null means only that the current K range is exhausted, "
        "not that all scored candidates or all source evidence were inspected. "
        "Changing K or page size reuses the existing scoring; changing the "
        "query, scope or M requires its corresponding ranking. New documents "
        "from search change the pool version and expire previous result IDs. "
        "If a result is stale, call rerank again with an explicit query and "
        "desired file scope/M, starting at offset=0. "
        "Do not repeat unchanged results or unsuccessful commands indefinitely; "
        "change the query, scope, source or investigation when no new evidence "
        "appears. On repeated output, empty results or truncation, identify "
        "the remaining evidence gap: refine local filtering or read context "
        "when the source is present; consider complementary search when "
        "coverage is uncertain. Progress feedback describes call repetition "
        "and output limits, not whether a fact is true or evidence is absent. "
        "You decide the next tool and rerank parameters. Once the required "
        "facts are supported, answer. If evidence "
        "is insufficient, state the unresolved part rather than inventing it.\n\n"
        "Output exactly one final response:\n"
        "Explanation: {brief evidence-based explanation}\n"
        "Exact Answer: {succinct answer, or Unable to determine}\n"
        "Confidence: {0-100%}"
    )
