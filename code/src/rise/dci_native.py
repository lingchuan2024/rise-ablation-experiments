"""Native (Python) re-implementation of the DCI baseline.

Replaces the Pi subprocess (`dci-agent-lite`) pipeline with the same
`Agent` class used by RISE. The agent's tool
surface is **bash + read on the FULL corpus root**, with no
`search` tool — matching the DCI paper's "Direct Corpus Interaction"
design.

Key paper-faithful settings:
- Prompt: `DCI_PAPER_C1_PROMPT` from `bcp_retrieval_agent.py:60`
  (verbatim from the DCI paper §C.1; the legacy Pi-based runner
  already imports this constant).
- `max_model_calls = 300` (DCI paper §4 Implementation Details).
- `coerce_final_on_max_turns = False` (paper does not coerce final
  commit; same default as every other headline runner under the
  unified protocol).
- Pi-faithful `bash` tail-truncates at 2000 lines / 50KB and spills
  full output to `/tmp/pi-bash-*.log` on overflow.
- `context_level=level3` enables Pi-style per-tool-result truncation
  plus micro-compaction in the shared Agent loop.

Linux containment caveat: `enable_sandbox=False` ships unsandboxed
bash on Linux (macOS-only `sandbox-exec` would FileNotFoundError).
We rely on the DCI prompt's "do not use absolute paths or `cd`"
directive + post-hoc trace audit to verify the agent stays within
the corpus. Real Linux containment (bubblewrap/firejail) is out of
scope for Phase 1.

This module wires the Agent class with the DCI tool registry. The
runner script (`scripts/run_dci_native.py`) handles CLI args, BM25
loading (none — DCI has no retriever), per-query orchestration,
trajectory dumping, and judge.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from .agent import Agent, AgentRun
from .bcp_retrieval_agent import DCI_PAPER_C1_PROMPT
from .protocol import RunConfig
from .tools import (
    make_pi_bash_tool,
    make_pi_read_tool,
    pi_bash_tool_spec,
    pi_read_tool_spec,
)


@dataclass
class DciNativeRun:
    """Per-query result for the native DCI baseline. Mirrors the field
    set of RISERun / BcpRunResult / PiSeriniRunResult so downstream
    cost / coverage aggregation in the runner is uniform."""
    query_id: Any
    question: str
    final_text: str = ""
    terminated_by: str = ""
    elapsed_seconds: float = 0.0
    # Token / cost rollup (final-attempt-only, per RunConfig contract)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    # Audit fields for discarded refusal-retry attempts (Phase-1 required
    # per cost_attribution = "final_attempt_only"). Populated only when
    # the Agent's whole-loop refusal retry fires.
    discarded_attempt_tokens: int = 0
    discarded_attempt_input_tokens: int = 0
    discarded_attempt_output_tokens: int = 0
    discarded_attempt_reasoning_tokens: int = 0
    discarded_attempt_cost_usd: float = 0.0
    # Tool / turn stats
    n_turns: int = 0
    n_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    bash_kinds: dict[str, int] = field(default_factory=dict)
    n_bash_calls: int = 0
    n_read_calls: int = 0
    # L3 context-management instrumentation (set by Agent class)
    n_micro_compactions: int = 0
    n_tool_results_truncated: int = 0
    # Surfaced docs (relpaths from `read` args + bash output rg hits)
    surfaced_relpaths: list[str] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)


# ----- System prompt --------------------------------------------------------
#
# DCI paper §C.1 doesn't separate "system" from "user" — the entire prompt
# is one block delivered as the first user message. Pi adds a generated
# system prompt on top (containing tool guidelines + cwd + date).
#
# This template is a VERBATIM port of what Pi emits for `--tools read,bash`,
# captured via `external/.../pi_system_prompt.py --tools read,bash`. It
# embeds Pi's full template — including the (DCI-irrelevant) "Pi
# documentation" paragraph — so the model sees the same bytes Pi gave it.
# Two placeholders are substituted at runtime: `{cwd}` and `{current_date}`.
#
# Why include the Pi docs paragraph: this refactor's whole point is byte-
# faithful comparison to Pi. Stripping it would re-introduce a known
# divergence. The model treats it as flavor noise (it never reads Pi docs
# during DCI questions); leaving it in keeps the comparison clean.
DCI_NATIVE_SYSTEM_PROMPT_TEMPLATE = """You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Be concise in your responses
- Show file paths clearly when working with files

Current date: {current_date}
Current working directory: {cwd}"""


def build_dci_system_prompt(corpus_dir: Path) -> str:
    """Pi-faithful system prompt: substitute cwd + today's UTC date.
    Pi computes the date at session start; we do the same. Pi's `--cwd`
    resolves symlinks (verified by dumping Pi's prompt with the BCP+ corpus
    dir as `--cwd`), and `corpus_dir` is already `.resolve()`'d in
    `run_dci_native`, so the embedded path is byte-identical to Pi's."""
    import datetime as _dt
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    return DCI_NATIVE_SYSTEM_PROMPT_TEMPLATE.format(
        cwd=str(corpus_dir), current_date=today,
    )


def build_dci_user_prompt(question: str, corpus_dir: Path) -> str:
    """Apply DCI §C.1's template substitution. Uses `str.replace`, not
    `str.format`, because the template's `{{...}}` placeholders are
    intentional braces for the LLM to fill, not Python format slots."""
    return (
        DCI_PAPER_C1_PROMPT
        .replace("{corpus_dir}", str(corpus_dir))
        .replace("{question}", question)
    )


def _normalize_surface_path(path: str) -> str:
    s = (path or "").strip().strip("'\"").lstrip("@").strip()
    marker = "/bc_plus_docs/"
    if marker in s:
        s = s.split(marker, 1)[1]
    if s.startswith("bc_plus_docs/"):
        s = s[len("bc_plus_docs/"):]
    while s.startswith("./"):
        s = s[2:]
    while s.startswith("/"):
        s = s[1:]
    return s


def _surfaced_relpaths_from_text(blob: str, map_keys: set[str]) -> set[str]:
    """Extract corpus relpaths from DCI read/bash args or tool output.

    Line-oriented on purpose: rg/find output commonly emits one path per line,
    and slicing around `.txt` preserves filenames with spaces when the line is
    a path or an rg `path.txt:match` row.
    """
    surfaced: set[str] = set()
    if not blob or ".txt" not in blob:
        return surfaced
    for line in blob.split("\n"):
        ls = line.strip()
        if ".txt" not in ls:
            continue
        candidates: list[str] = []
        for marker in ("/bc_plus_docs/", "bc_plus_docs/"):
            idx = ls.find(marker)
            if idx >= 0:
                cand = ls[idx + len(marker):]
                ti = cand.find(".txt")
                if ti >= 0:
                    candidates.append(cand[: ti + 4])
        candidates.append(ls)
        ti = ls.find(".txt")
        if ti >= 0:
            candidates.append(ls[: ti + 4])
        for cand in candidates:
            rel = _normalize_surface_path(cand)
            if rel in map_keys:
                surfaced.add(rel)
    return surfaced


def run_dci_native(
    question: str,
    query_id: Any,
    *,
    run_config: "RunConfig",
    client: OpenAI,
    trace_path: Path | None = None,
    trace_relpath_to_docid: dict[str, str] | None = None,
    enable_sandbox: bool = False,
) -> DciNativeRun:
    """Run native DCI baseline on one query.

    All budget/timeout/retry knobs flow from `run_config` (the shared
    RunConfig protocol object). Method-specific knobs come from
    `run_config.extras`:

      - `corpus_dir` (REQUIRED): full corpus filesystem root.
      - `bash_max_output_lines` (default 2000): Pi truncateTail max lines.
      - `bash_max_output_bytes` (default 50KB): Pi truncateTail max bytes.
      - `context_management` (default "level3"): Pi runtime context level.

    The agent has two tools:
        bash(command)       — shell over corpus_dir (NOT sandboxed on Linux)
        read(path)          — Pi-faithful file read relative to corpus_dir

    No `search`, no BM25, no mini-corpus winnowing. The agent sees the
    full corpus filesystem and uses lexical tools to navigate it.

    `enable_sandbox` stays a runtime kwarg (not in RunConfig) because
    it's a host-environment property (macOS vs Linux), not a method
    choice.
    """
    extras = run_config.extras
    corpus_dir = Path(extras["corpus_dir"]).resolve()
    bash_max_output_lines = int(extras.get("bash_max_output_lines", 2000))
    bash_max_output_bytes = int(extras.get("bash_max_output_bytes", 50 * 1024))
    context_level = str(extras.get("context_management", "level3"))

    out = DciNativeRun(query_id=query_id, question=question)

    # Tool registry. Both tools are rooted at the FULL corpus dir.
    # Pi-faithful: bash + read implementations are byte-faithful ports of
    # `pi-mono/.../tools/{bash,read}.ts`. Tool specs use Pi's exact JSON
    # schema (incl. `path` not `file_path`, 1-indexed offset, optional
    # `timeout` on bash). Pi's read has no default limit; the agent reads
    # to EOF and truncateHead caps at 2000 lines / 50KB.
    bash_fn = make_pi_bash_tool(
        corpus_dir,
        max_lines=bash_max_output_lines,
        max_bytes=bash_max_output_bytes,
        enable_sandbox=enable_sandbox,
    )
    read_fn = make_pi_read_tool(corpus_dir)

    tool_registry = {"bash": bash_fn, "read": read_fn}
    # Tool list order matches Pi's `createAllToolDefinitions` (read first,
    # then bash). Pi's `--tools read,bash` filter preserves this order, so
    # the LLM sees the same tool listing.
    tool_specs = [pi_read_tool_spec(), pi_bash_tool_spec()]

    agent = Agent(
        client=client, model=run_config.agent_model,
        tools=tool_registry, tool_specs=tool_specs,
        max_turns=run_config.max_model_calls,
        per_call_max_tokens=run_config.per_call_max_tokens,
        max_retries=run_config.api_max_retries,
        wall_clock_timeout_sec=run_config.wall_clock_timeout_sec,
        coerce_final_on_max_turns=run_config.coerce_final_on_max_turns,
        context_level=context_level,
    )

    system_prompt = build_dci_system_prompt(corpus_dir)
    user_prompt = build_dci_user_prompt(question, corpus_dir)

    try:
        agent_run: AgentRun = agent.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_path=trace_path,
            trace_relpath_to_docid=trace_relpath_to_docid,
        )
    finally:
        # Per-query cleanup: remove pi-bash-*.log overflow files written
        # during this query. These are pi-faithful debug aids; the agent
        # references them by path during the query but no DCI trace has
        # ever shown an agent actually `cat`-ing one. Without this hook,
        # they accumulated to 246 GB across many runs (see PROGRESS.md
        # 2026-05-23 entry).
        cleanup = getattr(bash_fn, "cleanup_temp_files", None)
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass

    # Final-attempt-only cost (per RunConfig contract). Audit fields
    # for discarded refusal-restart attempts are filled if Agent
    # exposes them (Phase 1 plumbs this through; for now leave zero —
    # Agent.run returns only the final accepted AgentRun, discarded
    # attempts are dropped at agent.py:193).
    out.final_text = agent_run.final_answer or ""
    out.terminated_by = agent_run.terminated_reason
    if agent_run.error:
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
    out.n_bash_calls = out.tool_call_breakdown.get("bash", 0)
    out.n_read_calls = out.tool_call_breakdown.get("read", 0)
    out.n_micro_compactions = agent_run.n_micro_compactions
    out.n_tool_results_truncated = agent_run.n_tool_results_truncated

    # bash kinds + read paths
    from .dci_artifacts import _classify_bash
    bash_kinds: dict[str, int] = {}
    read_paths: list[str] = []
    surfaced_relpaths: set[str] = set()
    map_keys = set(trace_relpath_to_docid or {})
    for turn in agent_run.turns:
        for tc in turn.tool_calls:
            if tc.name == "bash":
                command = tc.args.get("command", "")
                kind = _classify_bash(command)
                bash_kinds[kind] = bash_kinds.get(kind, 0) + 1
                if map_keys:
                    surfaced_relpaths.update(_surfaced_relpaths_from_text(command, map_keys))
                    surfaced_relpaths.update(_surfaced_relpaths_from_text(tc.result, map_keys))
            elif tc.name == "read":
                fp = tc.args.get("path") or tc.args.get("file_path") or ""
                if fp:
                    read_paths.append(fp)
                    if map_keys:
                        surfaced_relpaths.update(_surfaced_relpaths_from_text(fp, map_keys))
    out.bash_kinds = bash_kinds
    out.read_paths = read_paths
    out.surfaced_relpaths = sorted(surfaced_relpaths)

    return out
