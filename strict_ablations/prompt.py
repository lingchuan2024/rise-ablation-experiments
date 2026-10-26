"""Minimal ablation contracts derived from the frozen strict B100 inputs.

Only the interface being ablated changes. No legacy v14 strategy, feedback,
argument recovery, or new reading capability is introduced by this module.
Importing it does not import an agent, retrieval backend, model, or corpus.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
from pathlib import Path

from strict_experiment import strict_prompt as frozen


CONTROL = "control"
ARMS = ("search_paging", "auto_rerank", "no_bash")
VERSION = "strict-b100-minimal-ablations-v1"
CONTROL_SYSTEM_PROMPT_SHA256 = "bb605017783d5c53aa53ba5ac2948842694c15b2f8a7f98f70a4d4d5757293fc"
CONTROL_TOOL_SPECS_SHA256 = "23b3e8c08214dec94a83f08fcf5f3e4968f4bafdc7c1922eaca231d72599347b"
CONTROL_USER_PROMPT_SHA256 = "6d1d40b7d38fa161ddaa08cf1196e9f529b1c82879c8ede6f20ec95556d61274"

PAGING_ADDITION = (
    "- search(search_id, offset?): view an earlier subquery's fixed document-search "
    "results. A new search also returns each subquery's search_id, offset and "
    "next_offset. For a page, supply only its actual search_id and an optional "
    "zero-based document offset (default 0), without queries. Each page contains "
    "at most ten original document-opening snippets in the original document "
    "BM25 order, using the same 100-character previews as the first page. Paging "
    "does not retrieve, import or score documents. New searches do not invalidate "
    "these document-search IDs. next_offset=null means that this fixed search "
    "record is exhausted. Document-search IDs are separate from rerank result IDs.\n"
)

# Reuse B100's exact passage/read_args explanation; only the enclosing tool
# changes in the automatic arm. The official read instruction remains literal.
PASSAGE_INTERFACE = frozen.RERANK_ADDITION[
    frozen.RERANK_ADDITION.index("Each item includes file_path"):
    frozen.RERANK_ADDITION.index("\n- rerank(result_id")
]
AUTO_RETURN = (
    "Each query imports its own top-1000 document match set into the working "
    "directory, which accumulates across turns. The system automatically selects "
    "up to M=1000 body passages by passage BM25 within that query's own document "
    "match set, applies the neural reranker to those candidates using that "
    "unchanged search query, retains up to K=100 passages and returns the first "
    "ten complete passages. Passages have 128 tokens with 32-token overlap. "
    "Documents accumulated from other queries are outside that query's ranking "
    "scope unless they also belong to its own match set. "
    + PASSAGE_INTERFACE + " "
    "This search has no result-pagination interface or independently callable "
    "rerank tool. Use bash/read to explore beyond the returned passages."
)

_SEARCH_RETURN = (
    "Returns a per-query top-10 preview with file paths and short snippets — "
    "the preview is only a sample; the full match set per query (hundreds to "
    "thousands of docs) is ADDED to your working directory and accumulates "
    "across turns, so use bash/read to explore beyond the preview."
)
_SPEC_SEARCH_RETURN = (
    "Returns a per-query top-10 preview with file paths and short snippets, "
    "and the new total working-directory size."
)
_PAGING_SPEC_ADDITION = (
    " New searches also return each subquery's search_id, offset and next_offset. "
    "A page call accepts only an actual returned search_id and optional "
    "zero-based offset (default 0), without queries. Each page contains at most "
    "ten original document-opening snippets in the original document BM25 "
    "order, with the same 100-character previews as the first page. Paging "
    "does not retrieve, import or score documents. Later searches do not "
    "invalidate these document-search IDs; next_offset=null means this fixed "
    "record is exhausted. These IDs are separate from rerank result IDs."
)
_NO_BASH_TEXT = (
    ("so use bash/read to explore beyond the preview.",
     "so use read to explore beyond the preview."),
    ("change the files accessible to bash/read.",
     "change the files accessible to read."),
    ("bash/read to inspect it, and repeat until confident.",
     "read to inspect it, and repeat until confident."),
)
_NO_BASH_SPEC = (
    ("read/bash coordinates, without restricting bash/read.",
     "read coordinates, without restricting read."),
    (", such as files selected with bash", ""),
)


def _arm(arm: str) -> str:
    if arm not in (CONTROL, *ARMS):
        raise ValueError(f"Unknown strict ablation arm: {arm!r}")
    return arm


def _replace_once(text: str, old: str, new: str) -> str:
    if not old or text.count(old) != 1:
        raise ValueError(f"Frozen prompt/spec anchor changed: {old!r}")
    return text.replace(old, new, 1)


def _bullet(prompt: str, prefix: str) -> str:
    matches = [line + "\n" for line in prompt.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen tool instruction: {prefix!r}")
    return matches[0]


def _base() -> dict:
    contract = frozen.build_contract("B")
    if (contract["system_prompt_sha256"] != CONTROL_SYSTEM_PROMPT_SHA256
            or contract["tool_specs_sha256"] != CONTROL_TOOL_SPECS_SHA256
            or frozen.sha256(contract["user_prompt_template"]) != CONTROL_USER_PROMPT_SHA256):
        raise ValueError("Frozen strict B100 model inputs changed")
    return contract


def build_prompt(arm: str) -> str:
    _arm(arm)
    original = _base()["system_prompt"]
    if arm == CONTROL:
        return original
    if arm == "search_paging":
        search = _bullet(original, "- search(queries):")
        return _replace_once(original, search, search + PAGING_ADDITION)
    if arm == "auto_rerank":
        # Removing the independently callable interface is exact and reversible.
        prompt = frozen.remove_rerank_addition(original)
        return _replace_once(prompt, _SEARCH_RETURN, AUTO_RETURN)
    bash = _bullet(original, "- bash(command):")
    prompt = _replace_once(original, bash, "")
    prompt = _replace_once(prompt, "You have four tools:", "You have three tools:")
    for old, new in _NO_BASH_TEXT:
        prompt = _replace_once(prompt, old, new)
    return prompt


def build_tool_specs(arm: str) -> list[dict]:
    _arm(arm)
    specs = copy.deepcopy(_base()["tool_specs"])
    if arm == CONTROL:
        return specs
    search = specs[0]["function"]
    if arm == "search_paging":
        search["description"] += _PAGING_SPEC_ADDITION
        properties = search["parameters"]["properties"]
        properties["search_id"] = {
            "type": "string",
            "description": "Actual document-search ID returned by search. Use without queries.",
        }
        properties["offset"] = {
            "type": "integer", "minimum": 0,
            "description": "Zero-based document offset for search_id; default 0, at most ten documents per page.",
        }
        # Like the existing independent rerank schema, alternative call forms
        # are explained here and validated by the runtime, not an extra tool.
        search["parameters"]["required"] = []
    elif arm == "auto_rerank":
        search["description"] = _replace_once(
            search["description"], _SPEC_SEARCH_RETURN, AUTO_RETURN)
        specs.pop(3)
    else:
        specs.pop(1)
        rerank = specs[-1]["function"]
        rerank["description"] = _replace_once(
            rerank["description"], *_NO_BASH_SPEC[0])
        paths = rerank["parameters"]["properties"]["file_paths"]
        paths["description"] = _replace_once(paths["description"], *_NO_BASH_SPEC[1])
    return specs


def build_contract(arm: str) -> dict:
    """Same public input/hash fields as the frozen strict experiment contract."""
    _arm(arm)
    prompt, specs = build_prompt(arm), build_tool_specs(arm)
    payload = {
        "condition": arm, "version": VERSION,
        "system_prompt": prompt, "tool_specs": specs,
        "user_prompt_template": _base()["user_prompt_template"],
        "system_prompt_sha256": frozen.sha256(prompt),
        "tool_specs_sha256": frozen.json_sha256(specs),
    }
    return {**payload, "contract_sha256": frozen.json_sha256(payload)}


def _restore_prompt(arm: str, prompt: str) -> str:
    """Reject unexpected extra text while reversing only the allowed edits."""
    if arm == CONTROL:
        return prompt
    if arm == "search_paging":
        return _replace_once(prompt, PAGING_ADDITION, "")
    if arm == "auto_rerank":
        prompt = _replace_once(prompt, AUTO_RETURN, _SEARCH_RETURN)
        prompt = _replace_once(prompt, "You have three tools:", "You have four tools:")
        return _replace_once(prompt, "Every document begins",
                             frozen.RERANK_ADDITION + "Every document begins")
    for old, new in reversed(_NO_BASH_TEXT):
        prompt = _replace_once(prompt, new, old)
    prompt = _replace_once(prompt, "You have three tools:", "You have four tools:")
    read = _bullet(prompt, "- read(file_path, offset?, limit?):")
    return _replace_once(prompt, read, _bullet(_base()["system_prompt"], "- bash(command):") + read)


def _restore_specs(arm: str, specs: list[dict]) -> list[dict]:
    specs = copy.deepcopy(specs)
    original = _base()["tool_specs"]
    if arm == "search_paging":
        search = specs[0]["function"]
        search["description"] = _replace_once(search["description"], _PAGING_SPEC_ADDITION, "")
        for name in ("search_id", "offset"):
            del search["parameters"]["properties"][name]
        search["parameters"]["required"] = ["queries"]
    elif arm == "auto_rerank":
        search = specs[0]["function"]
        search["description"] = _replace_once(search["description"], AUTO_RETURN, _SPEC_SEARCH_RETURN)
        specs.append(copy.deepcopy(original[-1]))
    elif arm == "no_bash":
        specs.insert(1, copy.deepcopy(original[1]))
        rerank = specs[-1]["function"]
        rerank["description"] = _replace_once(rerank["description"],
                                                _NO_BASH_SPEC[0][1], _NO_BASH_SPEC[0][0])
        paths = rerank["parameters"]["properties"]["file_paths"]
        # The removed subordinate example has no unique empty-string inverse.
        expected = original[-1]["function"]["parameters"]["properties"]["file_paths"]["description"]
        if paths["description"] != expected.replace(_NO_BASH_SPEC[1][0], "", 1):
            raise ValueError("Unexpected no-bash file_paths instruction change")
        paths["description"] = expected
    return specs


def verify_contracts() -> dict:
    """Verify pinned control, literal shared inputs and reversible interventions."""
    provenance = frozen.verify_contracts()
    base = _base()
    contracts = {arm: build_contract(arm) for arm in (CONTROL, *ARMS)}
    checks = {"frozen_B100_inputs_pinned": True,
              "control_inputs_exactly_frozen_B": True,
              "all_prompt_edits_reversible": True,
              "all_tool_edits_reversible": True,
              "common_read_spec_unchanged": True,
              "common_user_template_unchanged": True,
              "no_bash_unavailable_references_removed": True}
    for arm, contract in contracts.items():
        if _restore_prompt(arm, contract["system_prompt"]) != base["system_prompt"]:
            raise ValueError(f"Unexpected non-interface prompt changes: {arm}")
        if _restore_specs(arm, contract["tool_specs"]) != base["tool_specs"]:
            raise ValueError(f"Unexpected non-interface tool changes: {arm}")
        read = next(spec for spec in contract["tool_specs"] if spec["function"]["name"] == "read")
        if read != base["tool_specs"][2]:
            raise ValueError(f"Shared read API changed: {arm}")
        if contract["user_prompt_template"] != base["user_prompt_template"]:
            raise ValueError(f"User template changed: {arm}")
    c = contracts["no_bash"]
    if "bash" in c["system_prompt"] or "bash" in json.dumps(c["tool_specs"]):
        raise ValueError("Unavailable bash capability remains in no_bash instructions")
    return {
        "version": VERSION, "control": CONTROL, "arms": list(ARMS),
        "based_on": "strict_experiment.strict_prompt:B",
        "frozen_contract_version": provenance["version"],
        "frozen_B_contract_sha256": base["contract_sha256"],
        "checks": checks,
        "contracts": {
            arm: {key: contract[key] for key in (
                "system_prompt_sha256", "tool_specs_sha256", "contract_sha256")}
            | {"tool_names": [spec["function"]["name"] for spec in contract["tool_specs"]]}
            for arm, contract in contracts.items()
        },
        "user_prompt_template_sha256": CONTROL_USER_PROMPT_SHA256,
        "runtime_feedback_policy": "frozen strict B; no v13/v14 observational-feedback wrapper",
    }


def generate_report(output_dir: Path) -> dict:
    """Save exact contracts and control-relative diffs, with no model calls."""
    manifest = verify_contracts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = {arm: build_contract(arm) for arm in (CONTROL, *ARMS)}
    for arm, contract in contracts.items():
        (output_dir / f"{arm}.system_prompt.txt").write_text(contract["system_prompt"], encoding="utf-8")
        (output_dir / f"{arm}.tools.json").write_text(
            json.dumps(contract["tool_specs"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / f"{arm}.contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "user_prompt_template.txt").write_text(
        contracts[CONTROL]["user_prompt_template"], encoding="utf-8")
    links = []
    for arm in ARMS:
        for kind, key in (("system_prompt", "system_prompt"), ("tools", "tool_specs")):
            before, after = contracts[CONTROL][key], contracts[arm][key]
            if key == "tool_specs":
                before = json.dumps(before, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                after = json.dumps(after, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            diff = "".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"{CONTROL}.{kind}", tofile=f"{arm}.{kind}"))
            filename = f"{CONTROL}--{arm}.{kind}.diff"
            (output_dir / filename).write_text(diff, encoding="utf-8")
            links.append(f"- [{CONTROL} → {arm}: {kind}]({filename})")
    (output_dir / "prompt_contract_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join([
        "# Strict B100-derived ablation input contracts", "",
        "The control system prompt, tool schemas and user template exactly match the frozen B100 inputs.",
        "The historical control is a reference contract; this report does not schedule or rerun it.", "",
        "Only the interface under intervention changes: document-search paging; automatic per-subquery passage ranking without a separate rerank tool; or removal of bash.",
        "No pagination preference, ranking strategy, shell tutorial, feedback policy, argument recovery or new read capability is added.", "",
        "Automatic ranking uses each unchanged subquery's own top-1000 document match set, M=1000, K=100, ten complete 128-token passages with 32-token overlap.",
        "The shared official read schema and structured-document guide remain unchanged; no_bash removes only unavailable bash references.", "",
        "## Exact input differences", "", *links, "",
        "## Contract verification", "", "```json",
        json.dumps(manifest, ensure_ascii=False, indent=2), "```", "",
        "These checks cover model-facing inputs. Runtime tests must separately verify scope, paging, tool registration, unchanged read behavior and checkpoint isolation.", "",
    ])
    (output_dir / "PROMPT_DIFF.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
