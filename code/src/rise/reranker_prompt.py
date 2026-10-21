"""Tool-selection guidance for v10; baseline prompts remain independent."""
from __future__ import annotations


RERANKER_PROMPT_VERSION = "v10-reranker-first-v3-overlap32"


def build_reranker_prompt(*, no_bash: bool, document_guide: str = "",
                          read_default_limit: int = 2000) -> str:
    tools = [
        "- search(queries): retrieve and import the top 1000 documents for each "
        "query into your working directory. Only the top 10 documents are "
        "previewed. Each preview is a short document-opening excerpt, which may "
        "contain only metadata and need not contain relevant body evidence. "
        "The other retrieved documents are still available. Search matches "
        "bag-of-words terms, so use distinctive natural-language terms and "
        "complementary queries. Documents accumulate across searches. Each "
        "subquery returns its own Reranker set ID, such as S001.q1.\n",
        "- reranker(set_id, offset?, limit?): find relevant body passages within "
        "one search subquery's original top-1000 documents. Use the set_id "
        "returned by that search; do not invent an ID. The tool reuses the EXACT "
        "original search query, splits bodies into passages of at most 128 "
        "tokens with 32-token overlap (96-token stride, adjusted at Unicode "
        "boundaries), selects the top 100 passages by BM25, and reranks them with a "
        "neural relevance model. It returns up to 10 complete passages with "
        "source paths and read_args. Use next_offset for another page of the "
        "same cached ranking. You do not supply or rewrite its query.\n",
        "- read(file_path, offset?, limit?): inspect a retrieved source using "
        "its exact returned file path. offset is zero-based and limit is in "
        f"LINES; the default is {read_default_limit} lines. Use the reranker's "
        "read_args to check surrounding source text. Individual lines are "
        "capped at 2000 characters.\n",
    ]
    if not no_bash:
        tools.append(
            "- bash(command): use targeted rg/grep and other shell commands "
            "over retrieved documents in the working directory. Use relative "
            "paths; absolute paths cannot access corpus files. Inspect known "
            "paths or search for a specific fact rather than repeatedly listing "
            "the same files. If a reranker card has read_may_truncate=true, "
            "its bash_args extracts the complete source character window.\n"
        )
    else:
        tools[-1] += (
            "The reranker itself returns its complete passage even when "
            "read_may_truncate=true. Do not treat a truncated source line as "
            "complete surrounding context.\n"
        )

    routing = (
        "Choose the next tool according to the evidence you still need:\n"
        "(1) Search for the missing fact or relationship. A top-10 preview is "
        "a small sample, not a judgment about all 1000 retrieved documents.\n"
        "(2) If a preview identifies a concrete source likely to contain the "
        "needed fact, you may read that source directly. If the preview does "
        "not provide enough relevant information to locate the needed evidence, "
        "prefer reranker for that subquery's set_id as your next step. Choose "
        "the set whose original query best matches the missing fact. You do "
        "not need to rerank every search or an already-resolved fact.\n"
        "(3) Inspect useful reranked passages, then use read_args to verify "
        "their source context. Relevance scores are not proof that a fact is "
        "correct. If useful evidence is absent from the first page, use "
        "next_offset or a complementary search rather than reranking the same "
        "page again.\n"
        "(4) Track whether each action adds a new source, fact, or verified "
        "relationship. Do not repeat an identical failed call or a call whose "
        "unchanged result has already been inspected. After two actions with "
        "no new evidence, change strategy: use an uninspected reranker set/page, "
        "read a different relevant source/range, or search with a substantively "
        "different query. Use file paths returned by tools; do not repeatedly "
        "guess nonexistent filenames.\n"
        "(5) Verify the question's required facts in source bodies. Once they "
        "are supported, answer and stop calling tools. If the evidence remains "
        "insufficient after changing strategy, explicitly state that limitation "
        "instead of inventing an answer or continuing the same loop.\n\n"
        "Output exactly one final response in this format. Keep the explanation "
        "brief, summarize supporting evidence, and do not repeat it:\n"
        "Explanation: {brief evidence-based explanation}\n"
        "Exact Answer: {succinct answer, or Unable to determine}\n"
        "Confidence: {0-100%}"
    )
    count = "three" if no_bash else "four"
    return (
        "You answer research questions over a large document corpus you cannot "
        "see directly. Base your answer on retrieved source evidence.\n\n"
        f"You have {count} tools:\n" + "".join(tools) + "\n" + document_guide + routing
    )
