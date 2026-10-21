"""Single-agent variant: corpus tools plus an optional v10 reranker.

Architecture: a single Agent loop with all the corpus access in one
conversation context — no orchestrator / sub-agent split. Compared to the
hierarchical setup, this preserves full evidence (no prose-truncation
boundary), uses prompt cache more effectively (one monotonically growing
conversation), and matches the BCP retrieval-agent / DCI bash-only baselines
more closely.

Mini-corpus mode is ACCUMULATE: each bm25_search hardlinks the union of its
top-k matches into the working dir without clearing prior entries. The
`bash` and `read` tools are both scoped to the working dir, so the agent
can only inspect docs that some search has retrieved this session.

Output dataclass mirrors the field set used by the runner script (so the
existing aggregation/judging machinery works).
"""
from __future__ import annotations

import re
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import bm25s
from openai import OpenAI

from .agent import Agent, AgentRun
from .dci_artifacts import _classify_bash
from .protocol import RunConfig
from .independent_rerank_prompt import (
    RERANK_PROMPT_VERSION,
    build_independent_rerank_prompt,
    v13_read_tool_spec,
    v14_rerank_tool_spec,
)
from .reranker_prompt import RERANKER_PROMPT_VERSION, build_reranker_prompt
from .reranker_tool import CHUNKING_VERSION, PASSAGE_OVERLAP_TOKENS, PASSAGE_STRIDE_TOKENS
from .space_prompt import SPACE_PROMPT_VERSION, build_space_prompt
from .space_tools import SPACE_TOOL_VERSION, SpaceTools, space_tool_specs
from .trajectory import compute_run_config_hash
from .turn_checkpoint import TurnCheckpoint
from .tools import (
    bash_tool_spec,
    bm25_search_tool_spec,
    make_bash_tool,
    make_bm25_search_tool,
    make_read_doc_tool,
    make_read_passage_tool,
    make_read_tool,
    read_doc_tool_spec,
    read_passage_tool_spec,
    read_tool_spec,
)


RISE_SYSTEM_PROMPT = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have three tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use bash/read to "
    "explore beyond the preview.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory. Paths are relative to the working "
    "directory; absolute paths (anything starting with `/`) won't find "
    "anything — bash can't see outside the working directory. Use rg/grep "
    "to find phrases across retrieved docs. Files are plain text; run "
    "`ls` first to see the directory layout.\n"
    "- read(file_path): read a file by its full path.\n\n"
    "Use the tools iteratively: search to pull evidence into your working "
    "directory, bash/read to inspect it, and repeat until confident. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_NOBASH = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have two tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use read to "
    "explore beyond the preview.\n"
    "- read(file_path): read a file by its full path.\n\n"
    "Use the tools iteratively: search to pull candidate docs into your "
    "working directory, read to inspect them, and repeat until confident. "
    "When confident, output the final answer in this format and stop "
    "calling tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_STRUCTURED_NOBASH = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have two tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use read to "
    "explore beyond the preview.\n"
    "- read(file_path, offset?, limit?): read a file by its full path. "
    "Optional 0-indexed `offset` and `limit` (both in LINES, not chars) "
    "let you read a slice instead of the whole file; defaults to lines "
    "0..2000.\n\n"
    "Every document begins with a YAML frontmatter block (title / author "
    "/ date), followed by a `## Table of Contents` listing each section "
    "with a 1-indexed line range and a short description (e.g. `- L48–"
    "65: Early history — Founded 1874 as the Nautical School...`), then "
    "a literal `=== DOCUMENT BODY ===` line, then the body with `## "
    "<heading>` markers at the listed line ranges.\n\n"
    "How to read these docs:\n"
    "(1) Start with `read(path, offset=0, limit=60)` to see the YAML + "
    "TOC + sentinel. If the result does NOT contain the `=== DOCUMENT "
    "BODY ===` line, the TOC is longer — re-read with a larger limit "
    "(e.g. `offset=0, limit=200`).\n"
    "(2) The TOC is a navigation index, NOT an answer source. Each TOC "
    "entry is a short summary line, not the actual text — it can omit "
    "details, drop nuance, or be slightly off. Never extract a specific "
    "fact (number, name, date, quote) from a TOC summary line.\n"
    "(3) Pick the most relevant section(s) from the TOC and read each "
    "one in full: `read(path, offset=<start-1>, limit=<end-start+1>)`. "
    "Confirm the answer in the actual section body before committing. "
    "If the body doesn't contain the expected fact, the TOC summary was "
    "misleading — read a wider chunk or another section.\n"
    "(4) If multiple sections look plausible, read each. If no section "
    "looks right, read more of the doc (larger limit, or a different "
    "offset). Don't commit just because the TOC line sounds relevant.\n"
    "(5) If the TOC says `(no sections — short or single-topic document; "
    "full text follows)`, read the file normally.\n\n"
    "Use the tools iteratively: search to pull candidate docs into your "
    "working directory, read to inspect them, and repeat until confident. "
    "When confident, output the final answer in this format and stop "
    "calling tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_STRUCTURED = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have three tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use bash/read to "
    "explore beyond the preview.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory. Paths are relative to the working "
    "directory; absolute paths (anything starting with `/`) won't find "
    "anything — bash can't see outside the working directory. Use rg/grep "
    "to find phrases across retrieved docs. Files are plain text; run "
    "`ls` first to see the directory layout.\n"
    "- read(file_path, offset?, limit?): read a file by its full path. "
    "Optional 0-indexed `offset` and `limit` (both in LINES, not chars) "
    "let you read a slice instead of the whole file; defaults to lines "
    "0..2000.\n\n"
    "Every document begins with a YAML frontmatter block (title / author "
    "/ date), followed by a `## Table of Contents` listing each section "
    "with a 1-indexed line range and a short description (e.g. `- L48–"
    "65: Early history — Founded 1874 as the Nautical School...`), then "
    "a literal `=== DOCUMENT BODY ===` line, then the body with `## "
    "<heading>` markers at the listed line ranges.\n\n"
    "How to read these docs:\n"
    "(1) Start with `read(path, offset=0, limit=60)` to see the YAML + "
    "TOC + sentinel. If the result does NOT contain the `=== DOCUMENT "
    "BODY ===` line, the TOC is longer — re-read with a larger limit "
    "(e.g. `offset=0, limit=200`).\n"
    "(2) The TOC is a navigation index, NOT an answer source. Each TOC "
    "entry is a short summary line, not the actual text — it can omit "
    "details, drop nuance, or be slightly off. Never extract a specific "
    "fact (number, name, date, quote) from a TOC summary line.\n"
    "(3) Pick the most relevant section(s) from the TOC and read each "
    "one in full: `read(path, offset=<start-1>, limit=<end-start+1>)`. "
    "Confirm the answer in the actual section body before committing. "
    "If the body doesn't contain the expected fact, the TOC summary was "
    "misleading — read a wider chunk or another section.\n"
    "(4) If multiple sections look plausible, read each. If no section "
    "looks right, read more of the doc (larger limit, or a different "
    "offset). Don't commit just because the TOC line sounds relevant.\n"
    "(5) If the TOC says `(no sections — short or single-topic document; "
    "full text follows)`, read the file normally.\n\n"
    "Use the tools iteratively: search to pull evidence into your working "
    "directory, bash/read to inspect it, and repeat until confident. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_PASSAGE = (
    "You answer research questions over a large document corpus you can't "
    "see directly. The corpus is indexed at the PASSAGE level: each "
    "document is split into ~512-token chunks. Search returns the most "
    "relevant passages, not whole documents.\n\n"
    "You have four tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Returns top-matching passages; the union is ADDED to "
    "your working directory. Each passage file is `<domain>/<title>/"
    "passages/p<NNNN>.txt`, so the directory layout preserves the parent "
    "document for context. Per-query preview shows passage paths and "
    "short snippets. Write queries as natural-language descriptions with "
    "several distinctive terms.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory of passages. `rg -l` returns "
    "passage-level paths. Files are short (~3-5kB each), boilerplate-"
    "free by construction. Use `ls <domain>/<title>/passages` to see how "
    "a document was chunked.\n"
    "- read_passage(target): read ONE passage. Accepts either a "
    "passage_id (`5412_p0042`) or a passage relpath (the path search/rg "
    "shows you). No offset/limit needed — passages are bounded. Prefer "
    "this when one passage carries the evidence you need.\n"
    "- read_doc(file_path): read the FULL parent document, by parent "
    "relpath (`<domain>/<title>.txt`). Use this when one passage isn't "
    "enough and you need surrounding sections (e.g., cross-section "
    "claims). If you pass a passage relpath, it auto-coerces to the "
    "parent with a one-line warning.\n\n"
    "General strategy: search to pull relevant passages, scan them with "
    "read_passage (and bash for cross-passage rg), and only fall back to "
    "read_doc when one passage's context is genuinely insufficient. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_USER_PROMPT = "QUESTION:\n{question}"


@dataclass
class RISERun:
    query_id: Any
    question: str
    final_text: str = ""
    terminated_by: str = ""
    elapsed_seconds: float = 0.0
    # token / cost rollup
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    # tool stats
    n_turns: int = 0
    n_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    bash_kinds: dict[str, int] = field(default_factory=dict)
    n_bm25_searches: int = 0     # Actual retrieval calls; search pages are separate.
    n_bm25_queries: int = 0      # total # of QUERIES across all search calls
    n_bash_calls: int = 0
    n_read_calls: int = 0         # doc-mode `read` only
    n_read_passage_calls: int = 0  # passage-mode `read_passage`
    n_read_doc_calls: int = 0      # passage-mode `read_doc`
    # corpus coverage
    surfaced_relpaths: list[str] = field(default_factory=list)
    bm25_queries: list[str] = field(default_factory=list)  # flat list, all queries
    read_paths: list[str] = field(default_factory=list)
    reranker_summary: dict = field(default_factory=dict)


# bm25_search preview line: "  1. [12.34] path/to/file.txt"  (optionally with
# a trailing "[already in wd]" tag).
_BM25_PREVIEW_RE = re.compile(r"\]\s+(.+?\.txt)\s*(?:\[[^\]]*\])?\s*$")


_PASSAGE_ID_RE_SINGLE = re.compile(r"^\d+_p\d+$")


def _surfaced_relpaths_from_run(
    agent_run: AgentRun, map_keys: set[str], working_dir: Path,
    *,
    passage_id_to_relpath: dict[str, str] | None = None,
) -> set[str]:
    """Walk recorded tool-call args + results to find corpus relpaths that
    were touched (read tool target, bm25_search Top-20 preview, or rg/grep
    hits in bash output).

    In passage mode, `read_passage` may be called with a passage_id
    (`5412_p0042`) rather than a relpath. If `passage_id_to_relpath` is
    provided, we reverse-lookup such ids to their relpaths and add
    those, so coverage doesn't miss directly-read passages."""
    surfaced: set[str] = set()
    wd_str = str(working_dir.resolve())

    def _try_add(s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        for prefix in ("./", "bc_plus_docs/"):
            if s.startswith(prefix):
                s = s[len(prefix):]
        if s.startswith(wd_str):
            s = s[len(wd_str):].lstrip("/")
        if s in map_keys:
            surfaced.add(s)
            return True
        # Passage-id reverse lookup (passage mode only)
        if passage_id_to_relpath is not None and _PASSAGE_ID_RE_SINGLE.match(s):
            rel = passage_id_to_relpath.get(s)
            if rel and rel in map_keys:
                surfaced.add(rel)
                return True
        return False

    for t in agent_run.turns:
        for tc in t.tool_calls:
            if tc.name in {"dci", "expand_source", "read", "rerank", "search"}:
                try:
                    def visit(value):
                        if isinstance(value, dict):
                            for key, child in value.items():
                                if key in {"source_file_path", "file_path"} and isinstance(child, str):
                                    _try_add(child)
                                else:
                                    visit(child)
                        elif isinstance(value, list):
                            for child in value:
                                visit(child)
                    visit(json.loads(tc.result))
                except (ValueError, TypeError):
                    pass
            # `read` tool: file_path arg is a direct corpus relpath
            if tc.name == "read":
                _try_add((tc.args.get("file_path") or ""))
            elif tc.name == "reranker":
                try:
                    payload = json.loads(tc.result)
                    for item in payload.get("items", []):
                        _try_add(item.get("file_path", ""))
                except (ValueError, TypeError, AttributeError):
                    pass
            # `read_doc` (passage-mode): file_path arg may be a parent
            # relpath OR a passage relpath that gets coerced. Record the
            # raw arg; the runner's coverage step coerces passage→parent.
            elif tc.name == "read_doc":
                _try_add((tc.args.get("file_path") or ""))
            # `read_passage`: target is passage_id (`\d+_p\d+`) OR a
            # passage relpath. We add both forms to surfaced so coverage
            # can resolve via passage_id-as-docid or relpath lookups.
            elif tc.name == "read_passage":
                tgt = (tc.args.get("target") or tc.args.get("passage_id") or tc.args.get("file_path") or "").strip()
                if tgt:
                    _try_add(tgt)

            blob = tc.result or ""
            if ".txt" not in blob:
                continue
            for ln in blob.split("\n"):
                ls = ln.strip()
                if ".txt" not in ls:
                    continue
                # 1) direct full-line match (e.g. `ls` output)
                if _try_add(ls):
                    continue
                # 2) bm25_search preview line: "  N. [score] path.txt"
                m = _BM25_PREVIEW_RE.search(ls)
                if m and _try_add(m.group(1)):
                    continue
                # 3) generic: cut at first ".txt" (handles "path.txt:line:..." rg)
                ti = ls.find(".txt")
                if ti >= 0:
                    _try_add(ls[: ti + 4])
    return surfaced


def run_rise_agent(
    question: str,
    query_id: Any,
    *,
    run_config: "RunConfig",
    searcher: bm25s.BM25,
    doc_ids: Sequence[str],
    docid_to_relpath: dict[str, str],
    bc_plus_docs_root: Path,
    working_dir: Path,
    client: OpenAI,
    trace_path: Path | None = None,
    turn_checkpoint_path: Path | None = None,
    stream_nvidia: bool = False,
    map_keys: set[str] | None = None,
    enable_sandbox: bool = True,
    # Passage-mode plumbing. When `passage_mode=True`, `searcher / doc_ids
    # / docid_to_relpath / bc_plus_docs_root` are expected to be the
    # PASSAGE index, passage relpath map, and the passage-files root.
    # `parent_docs_root` + `relpath_to_parent_docid` are then used by the
    # new read_doc / read_passage tools. Toggled via run_config.runner_mode
    # ("passage" or "doc") — these params carry the runtime resources.
    parent_docs_root: Path | None = None,
    relpath_to_parent_docid: dict[str, str] | None = None,
    passage_index: Any | None = None,
) -> RISERun:
    """Run RISE on one query.

    All budget/timeout/retry knobs flow from `run_config`. Method-specific
    knobs come from `run_config.extras`:

      - `bm25_k`, `bm25_top_n_preview`, `bm25_snippet_chars`
      - `bash_truncate_chars`, `read_default_limit`
      - `passage_mode` (mirror of `run_config.runner_mode == "passage"`)

    `working_dir` is the per-query mini-corpus path. It will be (re)created
    on each bm25_search call. `searcher / doc_ids / docid_to_relpath /
    bc_plus_docs_root` are runtime resources passed by the caller.
    """
    extras = run_config.extras
    bm25_k = int(extras.get("bm25_k", 100))
    bm25_top_n_preview = int(extras.get("bm25_top_n_preview", 10))
    bm25_snippet_chars = int(extras.get("bm25_snippet_chars", 100))
    bash_truncate_chars = int(extras.get("bash_truncate_chars", 4000))
    read_default_limit = int(extras.get("read_default_limit", 2000))
    passage_mode = (run_config.runner_mode == "passage") or bool(extras.get("passage_mode", False))
    # When set, swap RISE_SYSTEM_PROMPT for RISE_SYSTEM_PROMPT_STRUCTURED — assumes
    # the corpus on disk has the doc_representation pipeline's YAML+TOC+heading
    # layout. Caller is responsible for pointing `bc_plus_docs_root` at the
    # restructured corpus tree.
    structured_doc_mode = bool(extras.get("structured_doc_mode", False))
    # When set, omit the `bash` tool from the registry — agent can only use
    # search and read. Ablation cell for "RISE minus mini-corpus exploration",
    # closest in spirit to a BCP retrieval-agent setup.
    no_bash = bool(extras.get("no_bash", False))
    reranker_enabled = bool(extras.get("reranker_enabled", False))
    space_enabled = bool(extras.get("space_reranker_enabled", False))
    independent_enabled = bool(extras.get("independent_rerank_enabled", False))
    ablation_variant = extras.get("ablation_variant")
    if ablation_variant is not None:
        if ablation_variant not in {"search_paging", "auto_rerank", "no_bash"}:
            raise ValueError("Unknown v14 ablation variant")
        if not independent_enabled:
            raise ValueError("v14 ablations require independent rerank")
        if no_bash and ablation_variant != "no_bash":
            raise ValueError("Only the no_bash variant may remove bash")
        no_bash = ablation_variant == "no_bash"
        extras["no_bash"] = no_bash
        if extras.setdefault("ablation_version", "v14-ablations-v1") != "v14-ablations-v1":
            raise ValueError("Unsupported ablation version")
        from .ablation_prompt import bind_search_paging_prompt_version
        bind_search_paging_prompt_version(extras)
        if bm25_snippet_chars != 100 or read_default_limit != 2000:
            raise ValueError("v14 ablations preserve snippet_chars=100 and read_default_limit=2000")
    if independent_enabled:
        from .rerank_v13 import RERANK_RESPONSE_VERSION, RERANK_TOOL_VERSION
        from .v13_feedback import PROGRESS_FEEDBACK_VERSION
        if reranker_enabled or space_enabled:
            raise ValueError("v13 independent rerank cannot be combined with other reranker modes")
        if (no_bash and ablation_variant != "no_bash") or not enable_sandbox:
            raise ValueError("v13 requires original-pool bash and its sandbox")
        if passage_mode or bm25_k != 1000 or bm25_top_n_preview != 10:
            raise ValueError("v13 requires document search top-1000 with top-10 preview")
        if not extras.get("space_passage_index") or passage_index is None:
            raise ValueError("v13 requires a loaded offline passage index")
        for key, expected in {
            "independent_rerank_prompt_version": RERANK_PROMPT_VERSION,
            "independent_rerank_tool_version": RERANK_TOOL_VERSION,
            "independent_rerank_response_version": RERANK_RESPONSE_VERSION,
            "progress_feedback_version": PROGRESS_FEEDBACK_VERSION,
            "independent_rerank_default_candidates": 1000,
            "independent_rerank_default_top_k": 100,
            "independent_rerank_default_limit": 10,
            "reranker_passage_tokens": 128,
            "reranker_passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "reranker_passage_stride_tokens": PASSAGE_STRIDE_TOKENS,
            "reranker_chunking_version": CHUNKING_VERSION,
        }.items():
            if extras.setdefault(key, expected) != expected:
                raise ValueError(f"v13 fixes {key}={expected}")
    if space_enabled and reranker_enabled:
        raise ValueError("v10 reranker and v11 accumulated-space mode are mutually exclusive")
    if space_enabled:
        if passage_mode or bm25_k != 1000 or bm25_top_n_preview != 10:
            raise ValueError("v11 requires document search top-1000 with top-10 preview")
        if not enable_sandbox:
            raise ValueError("v11 DCI requires the shell sandbox")
        candidates = extras.setdefault("space_bm25_candidates", 1000)
        top_k = extras.setdefault("space_top_k", 100)
        if (isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1
                or isinstance(top_k, bool) or not isinstance(top_k, int)
                or not 1 <= top_k <= candidates):
            raise ValueError("v11 requires candidates >= top_k >= 1")
        for key, expected in {
            "space_prompt_version": SPACE_PROMPT_VERSION,
            "space_tool_version": SPACE_TOOL_VERSION,
            "reranker_passage_tokens": 128,
            "reranker_passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "reranker_passage_stride_tokens": PASSAGE_STRIDE_TOKENS,
            "reranker_chunking_version": CHUNKING_VERSION,
        }.items():
            if extras.setdefault(key, expected) != expected:
                raise ValueError(f"v11 fixes {key}={expected}")
    if reranker_enabled and (passage_mode or bm25_k != 1000 or bm25_top_n_preview != 10):
        raise ValueError("v10 requires document search top-1000 with top-10 preview")
    if reranker_enabled and (extras.get("reranker_passage_tokens", 128) != 128
                            or extras.get("reranker_candidates", 100) != 100
                            or extras.get("reranker_preview", 10) != 10):
        raise ValueError("v10 fixes passage tokens=128, BM25 candidates=100, preview=10")
    if reranker_enabled:
        for key, expected in {
            "reranker_passage_overlap_tokens": PASSAGE_OVERLAP_TOKENS,
            "reranker_passage_stride_tokens": PASSAGE_STRIDE_TOKENS,
            "reranker_chunking_version": CHUNKING_VERSION,
        }.items():
            if extras.setdefault(key, expected) != expected:
                raise ValueError(f"v10 fixes {key}={expected}")
        version = extras.setdefault("reranker_prompt_version", RERANKER_PROMPT_VERSION)
        if version != RERANKER_PROMPT_VERSION:
            raise ValueError(f"Unsupported reranker prompt version: {version}")

    t0 = time.time()
    out = RISERun(query_id=query_id, question=question)

    # Ensure the working dir exists (empty) before the agent runs — the agent
    # might call bash before its first bm25_search, and we want bash's cwd
    # to be valid.
    if working_dir.exists() and (turn_checkpoint_path is None or not turn_checkpoint_path.is_file()):
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    reranker = None
    space_tools = None
    independent_rerank = None
    progress_feedback = None
    ablation_search = None
    if reranker_enabled or space_enabled or independent_enabled:
        from .reranker_backend import QwenReranker
        from .reranker_tool import RerankerTool, reranker_tool_spec
        backend = QwenReranker(
            model=extras.get("reranker_model", "Qwen/Qwen3-Reranker-0.6B"),
            revision=extras.get("reranker_revision", "main"),
            device=extras.get("reranker_device", "auto"),
            dtype=extras.get("reranker_dtype", "auto"),
            batch_size=int(extras.get("reranker_batch_size", 8)),
            max_length=int(extras.get("reranker_max_length", 32768)),
            local_files_only=bool(extras.get("reranker_local_files_only", False)),
        )
        if independent_enabled:
            from .indexed_accumulated_space import IndexedAccumulatedSpaceManager
            from .rerank_v13 import RerankService
            actual_index = {
                "format_version": passage_index.manifest["format_version"],
                "configuration_sha256": passage_index.manifest["configuration"]["sha256"],
                "corpus_content_sha256": passage_index.manifest["corpus_content_sha256"],
            }
            if extras.get("space_index_identity") != actual_index:
                raise ValueError("loaded passage index differs from v13 run configuration")
            runtime_dir = working_dir.parent.parent / "_rerank_runtime" / working_dir.name
            if runtime_dir.exists() and (turn_checkpoint_path is None or not turn_checkpoint_path.is_file()):
                shutil.rmtree(runtime_dir)
            manager = IndexedAccumulatedSpaceManager(
                working_dir=working_dir, views_dir=runtime_dir, backend=backend,
                candidates=1000, top_k=100, passage_index=passage_index,
                runtime_to_corpus={rel: rel for rel in docid_to_relpath.values()})
            independent_rerank = RerankService(manager)
        elif space_enabled:
            from .accumulated_space import AccumulatedSpaceManager
            views_dir = working_dir.parent.parent / "_views" / working_dir.name
            if views_dir.exists() and (turn_checkpoint_path is None or not turn_checkpoint_path.is_file()):
                shutil.rmtree(views_dir)
            manager_kwargs = dict(
                working_dir=working_dir, views_dir=views_dir, backend=backend,
                candidates=extras["space_bm25_candidates"], top_k=extras["space_top_k"])
            if extras.get("space_passage_index"):
                from .indexed_accumulated_space import IndexedAccumulatedSpaceManager
                if passage_index is None:
                    raise ValueError("v12 requires a loaded offline passage index")
                expected_index = extras.get("space_index_identity")
                actual_index = {
                    "format_version": passage_index.manifest["format_version"],
                    "configuration_sha256": passage_index.manifest["configuration"]["sha256"],
                    "corpus_content_sha256": passage_index.manifest["corpus_content_sha256"],
                }
                if expected_index != actual_index:
                    raise ValueError("loaded passage index differs from run configuration")
                manager = IndexedAccumulatedSpaceManager(
                    **manager_kwargs, passage_index=passage_index,
                    runtime_to_corpus={rel: rel for rel in docid_to_relpath.values()})
            else:
                manager = AccumulatedSpaceManager(**manager_kwargs)
            space_tools = SpaceTools(manager, no_bash=no_bash,
                                     truncate_chars=bash_truncate_chars)
        else:
            reranker = RerankerTool(working_dir=working_dir, backend=backend)

    bm25_fn = make_bm25_search_tool(
        searcher=searcher, doc_ids=doc_ids,
        docid_to_relpath=docid_to_relpath,
        bc_plus_docs_root=bc_plus_docs_root,
        working_dir=working_dir, k=bm25_k,
        top_n_preview=bm25_top_n_preview,
        snippet_chars=bm25_snippet_chars,
        on_search_results=(independent_rerank.register_search if independent_rerank is not None
                           else space_tools.manager.register_search if space_tools is not None
                           else reranker.register_search if reranker is not None else None),
    )
    bash_fn = None
    if ablation_variant != "no_bash":
        bash_fn = make_bash_tool(
            working_dir, truncate_chars=bash_truncate_chars,
            enable_sandbox=enable_sandbox,
        )

    if independent_rerank is not None:
        from .rerank_v13 import rerank_tool_spec
        from .v13_feedback import ProgressFeedback, v13_bash_tool_spec
        underlying_search = bm25_fn
        def independent_search(args):
            output = underlying_search(args).replace("  Reranker set:", "  Search record:")
            summary = independent_rerank.summary
            return output + "\nAccumulated original-document pool: " + json.dumps({
                "pool_version": summary.get("pool_version", summary.get("space_version")),
                "documents": summary.get("pool_documents"),
                "notice": ("read uses all imported original documents; rerank is independently available."
                           if ablation_variant == "no_bash" else
                           "bash/read use all imported original documents; rerank is independently available."),
            }, ensure_ascii=False)
        read_fn = make_read_tool(working_dir, default_limit=read_default_limit)
        feedback_path = (trace_path.parent / "tool_feedback.jsonl" if trace_path is not None
                         else working_dir.parent.parent / "_tool_feedback" / f"{working_dir.name}.jsonl")
        if turn_checkpoint_path is None or not turn_checkpoint_path.is_file():
            feedback_path.unlink(missing_ok=True)
        progress_feedback = ProgressFeedback(
            pool_version_provider=lambda: independent_rerank.manager.space_version,
            log_path=feedback_path)
        if ablation_variant == "search_paging":
            from .v13_feedback import normalize_search_page_ids
            progress_feedback.search_fingerprint_normalizer = normalize_search_page_ids
        tool_registry = {
            "search": progress_feedback.wrap("search", independent_search),
            "bash": progress_feedback.wrap("bash", bash_fn),
            "read": read_fn,
            "rerank": progress_feedback.wrap("rerank", independent_rerank),
        }
        tool_specs = [bm25_search_tool_spec(), v13_bash_tool_spec(bash_tool_spec()),
                      v13_read_tool_spec(read_tool_spec(),
                                         read_default_limit=read_default_limit),
                      v14_rerank_tool_spec(rerank_tool_spec())]
        document_guide = ""
        if structured_doc_mode:
            start = RISE_SYSTEM_PROMPT_STRUCTURED.index("Every document begins")
            end = RISE_SYSTEM_PROMPT_STRUCTURED.index("Use the tools iteratively:")
            document_guide = RISE_SYSTEM_PROMPT_STRUCTURED[start:end]
        system_prompt = build_independent_rerank_prompt(
            document_guide=document_guide, read_default_limit=read_default_limit)
        if ablation_variant is not None:
            from .ablation_tools import (
                SearchPagingTool, AutoRerankSearchTool, WithoutBashResponse,
            )
            from .ablation_prompt import (
                build_ablation_prompt, ablation_search_tool_spec,
                ablation_read_tool_spec, ablation_rerank_tool_spec,
            )
            search_callable = independent_search
            if ablation_variant == "search_paging":
                ablation_search = SearchPagingTool(
                    independent_search, independent_rerank,
                    working_dir=working_dir, snippet_chars=bm25_snippet_chars)
                search_callable = ablation_search
            elif ablation_variant == "auto_rerank":
                ablation_search = AutoRerankSearchTool(independent_search, independent_rerank)
                search_callable = ablation_search
            tool_registry = {"search": progress_feedback.wrap("search", search_callable)}
            tool_specs = [ablation_search_tool_spec(ablation_variant)]
            if not no_bash:
                tool_registry["bash"] = progress_feedback.wrap("bash", bash_fn)
                tool_specs.append(v13_bash_tool_spec(bash_tool_spec()))
            tool_registry["read"] = read_fn
            tool_specs.append(ablation_read_tool_spec(
                ablation_variant, read_tool_spec(), read_default_limit=read_default_limit))
            if ablation_variant != "auto_rerank":
                rerank_callable = (WithoutBashResponse(independent_rerank)
                                   if no_bash else independent_rerank)
                tool_registry["rerank"] = progress_feedback.wrap("rerank", rerank_callable)
                tool_specs.append(ablation_rerank_tool_spec(ablation_variant, rerank_tool_spec()))
            system_prompt = build_ablation_prompt(
                ablation_variant, document_guide=document_guide,
                read_default_limit=read_default_limit)
    elif space_tools is not None:
        underlying_search = bm25_fn
        def space_search(args):
            output = underlying_search(args)
            # The callback is shared with v10, but v11 search IDs are provenance
            # records rather than one-query reranker scopes.
            output = output.replace("  Reranker set:", "  Search record:")
            summary = space_tools.manager.summary
            return output + "\nAccumulated pool: " + json.dumps({
                "space_version": summary.get("space_version"),
                "documents": summary.get("pool_documents", summary.get("scope_documents")),
                "notice": "DCI uses this accumulated union with your explicit evidence goal."},
                ensure_ascii=False)
        tool_registry = {"search": space_search, "dci": space_tools.dci,
                         "read": space_tools.read, "expand_source": space_tools.expand_source}
        tool_specs = [bm25_search_tool_spec(), *space_tool_specs(no_bash=no_bash)]
        system_prompt = build_space_prompt(candidates=extras["space_bm25_candidates"],
                                          top_k=extras["space_top_k"], no_bash=no_bash)
    elif passage_mode:
        if parent_docs_root is None or relpath_to_parent_docid is None:
            raise ValueError(
                "passage_mode=True requires parent_docs_root and "
                "relpath_to_parent_docid (for read_doc fallback and "
                "[from <parent>] annotations)."
            )
        # In passage mode:
        # - `search` is over passages (caller passed passage index + map).
        # - `read_passage` reads a single passage from `bc_plus_docs_root`
        #   (which is the passage files root in passage mode).
        # - `read_doc` reads a parent doc from `parent_docs_root` (full corpus),
        #   and tolerates passage relpaths by stripping `/passages/p*.txt`.
        read_passage_fn = make_read_passage_tool(
            passage_files_root=bc_plus_docs_root,
            passage_id_to_relpath=docid_to_relpath,
            relpath_to_parent_docid=relpath_to_parent_docid,
        )
        read_doc_fn = make_read_doc_tool(parent_docs_root, default_limit=read_default_limit)
        tool_registry = {
            "search": bm25_fn,
            "bash": bash_fn,
            "read_passage": read_passage_fn,
            "read_doc": read_doc_fn,
        }
        tool_specs = [bm25_search_tool_spec(), bash_tool_spec(), read_passage_tool_spec(), read_doc_tool_spec()]
        system_prompt = RISE_SYSTEM_PROMPT_PASSAGE
    else:
        # read root = working_dir, same scope as bash. Agent can only read
        # docs that some search has retrieved into the working directory.
        read_fn = make_read_tool(working_dir, default_limit=read_default_limit)
        if no_bash:
            tool_registry = {"search": bm25_fn, "read": read_fn}
            tool_specs = [bm25_search_tool_spec(), read_tool_spec()]
            system_prompt = (
                RISE_SYSTEM_PROMPT_STRUCTURED_NOBASH if structured_doc_mode
                else RISE_SYSTEM_PROMPT_NOBASH
            )
        else:
            tool_registry = {"search": bm25_fn, "bash": bash_fn, "read": read_fn}
            tool_specs = [bm25_search_tool_spec(), bash_tool_spec(), read_tool_spec()]
            system_prompt = (
                RISE_SYSTEM_PROMPT_STRUCTURED if structured_doc_mode
                else RISE_SYSTEM_PROMPT
            )

    if reranker is not None:
        tool_registry["reranker"] = reranker
        tool_specs.append(reranker_tool_spec())
        # Retain baseline source-navigation rules, while replacing its old
        # tool list and search -> bash/read -> repeat decision flow as a whole.
        document_guide = ""
        if structured_doc_mode:
            start = system_prompt.index("Every document begins")
            end = system_prompt.index("Use the tools iteratively:")
            document_guide = system_prompt[start:end]
        system_prompt = build_reranker_prompt(
            no_bash=no_bash, document_guide=document_guide,
            read_default_limit=read_default_limit,
        )

    agent = Agent(
        client=client, model=run_config.agent_model,
        tools=tool_registry, tool_specs=tool_specs,
        max_turns=run_config.max_model_calls,
        per_call_max_tokens=run_config.per_call_max_tokens,
        max_retries=run_config.api_max_retries,
        stream_nvidia=stream_nvidia,
        wall_clock_timeout_sec=run_config.wall_clock_timeout_sec,
        coerce_final_on_max_turns=run_config.coerce_final_on_max_turns,
    )
    # Build relpath→docid map for the trace dump so search/read tool calls
    # store corpus docids (matching BCP / Pi-Serini / DCI) instead of paths.
    # In passage_mode: relpath here is a passage relpath; docid is a
    # passage_id. The runner separately resolves passage_id → parent_docid
    # for gold-doc coverage.
    trace_relpath_to_docid = {rel: did for did, rel in docid_to_relpath.items()}
    if independent_rerank is not None:
        def independent_tool_state():
            state = {"version": PROGRESS_FEEDBACK_VERSION,
                    "rerank": independent_rerank.export_state(),
                    "feedback": progress_feedback.export_state()}
            if ablation_variant is not None:
                state["ablation_variant"] = ablation_variant
                state["ablation_search"] = (ablation_search.export_state()
                                             if ablation_search is not None else None)
            return state
        def restore_independent_tool_state(state):
            if not isinstance(state, dict) or state.get("version") != PROGRESS_FEEDBACK_VERSION:
                raise ValueError("Incompatible v13 independent-tool checkpoint state")
            # TurnCheckpoint has already restored hardlinks. Restore the pool
            # before checking the feedback counters' pool identity.
            independent_rerank.restore_state(state.get("rerank"))
            progress_feedback.restore_state(state.get("feedback"))
            if ablation_variant is not None:
                if state.get("ablation_variant") != ablation_variant:
                    raise ValueError("Ablation checkpoint variant changed")
                if ablation_search is not None:
                    ablation_search.restore_state(state.get("ablation_search"))
    turn_checkpoint = None
    if turn_checkpoint_path is not None:
        digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        turn_checkpoint = TurnCheckpoint(
            turn_checkpoint_path,
            identity={
                "model": run_config.agent_model,
                "query_id": str(query_id),
                "question_sha256": digest(question),
                "run_config_hash": compute_run_config_hash(run_config.to_hashable_dict()),
                "system_prompt_sha256": digest(system_prompt),
                "tool_specs_sha256": digest(json.dumps(tool_specs, sort_keys=True, ensure_ascii=False)),
            },
            working_dir=working_dir,
            corpus_root=bc_plus_docs_root,
            relpath_to_docid=trace_relpath_to_docid,
            tool_state_provider=(independent_tool_state if independent_rerank is not None
                                 else space_tools.export_state if space_tools is not None
                                 else reranker.export_state if reranker is not None else None),
            tool_state_restorer=(restore_independent_tool_state if independent_rerank is not None
                                 else space_tools.restore_state if space_tools is not None
                                 else reranker.restore_state if reranker is not None else None),
        )
    agent_run: AgentRun = agent.run(
        system_prompt=system_prompt,
        user_prompt=RISE_USER_PROMPT.format(question=question),
        trace_path=trace_path,
        trace_relpath_to_docid=trace_relpath_to_docid,
        turn_checkpoint=turn_checkpoint,
    )

    out.final_text = agent_run.final_answer or ""
    out.terminated_by = agent_run.terminated_reason
    if agent_run.error and agent_run.terminated_reason != "context_limit":
        out.terminated_by = f"{agent_run.terminated_reason}: {agent_run.error}"
    out.elapsed_seconds = agent_run.elapsed_seconds
    out.input_tokens = agent_run.total_prompt_tokens
    out.cached_input_tokens = agent_run.total_cached_prompt_tokens
    out.output_tokens = agent_run.total_completion_tokens
    out.reasoning_tokens = agent_run.total_reasoning_tokens
    out.cost_usd = agent_run.total_cost_usd
    out.n_turns = len(agent_run.turns)
    out.n_tool_calls = agent_run.tool_call_count
    out.tool_call_breakdown = dict(agent_run.tool_call_breakdown)
    if reranker is not None or space_tools is not None or independent_rerank is not None:
        out.reranker_summary = (independent_rerank.summary if independent_rerank is not None
                               else space_tools.summary if space_tools is not None else reranker.summary)
        if progress_feedback is not None:
            out.reranker_summary["progress_feedback"] = progress_feedback.summary
        if ablation_variant is not None:
            out.reranker_summary["ablation_variant"] = ablation_variant
            out.reranker_summary["ablation_search"] = (
                ablation_search.summary if ablation_search is not None else {})
        if trace_path is not None:
            from .trajectory import atomic_write_json
            atomic_write_json(trace_path.parent / "reranker_summary.json", out.reranker_summary)
    out.n_bm25_searches = out.tool_call_breakdown.get("search", 0)
    if ablation_search is not None:
        out.n_bm25_searches = ablation_search.summary["retrieval_calls"]
    out.n_bash_calls = out.tool_call_breakdown.get("bash", 0)
    if space_tools is not None:
        out.n_bash_calls = out.reranker_summary["dci_command_calls"]
    out.n_read_calls = out.tool_call_breakdown.get("read", 0)
    out.n_read_passage_calls = out.tool_call_breakdown.get("read_passage", 0)
    out.n_read_doc_calls = out.tool_call_breakdown.get("read_doc", 0)

    bash_kinds: dict[str, int] = {}
    bm25_queries: list[str] = []
    read_paths: list[str] = []
    for turn in agent_run.turns:
        for tc in turn.tool_calls:
            if tc.name == "bash" or (tc.name == "dci" and tc.args.get("command")):
                if tc.name == "dci":
                    try:
                        if not json.loads(tc.result).get("command_executed"):
                            continue
                    except (ValueError, TypeError, AttributeError):
                        continue
                kind = _classify_bash(tc.args.get("command", ""))
                bash_kinds[kind] = bash_kinds.get(kind, 0) + 1
            elif tc.name == "search":
                # tolerant: queries list, or single query string
                qs = tc.args.get("queries")
                if qs is None:
                    qs = tc.args.get("query")
                if isinstance(qs, str):
                    qs = [qs]
                if isinstance(qs, list):
                    for q in qs:
                        if isinstance(q, str) and q.strip():
                            bm25_queries.append(q.strip())
            elif tc.name == "read":
                fp = tc.args.get("file_path", "")
                if fp:
                    read_paths.append(fp)
            elif tc.name == "expand_source":
                fp = tc.args.get("file_path") or tc.args.get("source_id", "")
                if fp:
                    read_paths.append(fp)
            elif tc.name == "read_doc":
                fp = tc.args.get("file_path", "")
                if fp:
                    read_paths.append(fp)
            elif tc.name == "read_passage":
                tgt = tc.args.get("target") or tc.args.get("passage_id") or tc.args.get("file_path", "")
                if tgt:
                    read_paths.append(tgt)
    out.bash_kinds = bash_kinds
    if ablation_search is not None:
        bm25_queries = [record["query"]
                        for record in independent_rerank.manager._search_sets.values()]
    out.bm25_queries = bm25_queries
    out.n_bm25_queries = len(bm25_queries)
    out.read_paths = read_paths

    if map_keys is not None:
        # In passage mode, `docid_to_relpath` is `passage_id_to_relpath`;
        # pass it so `_surfaced_relpaths_from_run` can reverse-resolve
        # passage_ids the agent passed to read_passage directly.
        pid_to_rel = docid_to_relpath if passage_mode else None
        out.surfaced_relpaths = sorted(
            _surfaced_relpaths_from_run(
                agent_run, map_keys, working_dir,
                passage_id_to_relpath=pid_to_rel,
            )
        )
    return out
