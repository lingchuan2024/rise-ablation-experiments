"""Frozen official instructions, with an explicitly reversible rerank addition.

This module reads literal constants/functions with AST. It does not import an
agent, corpus, retrieval library, GPU backend, or mutable experimental prompt.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
OFFICIAL_REPOSITORY = "https://github.com/texttron/RISE"
OFFICIAL_COMMIT = "32672b59ead8381decae7d412f05bb53d399946a"
OFFICIAL_BLOBS = {
    "rise_agent.py": "99c24fba40002415c9d3fb4011dcebdbcc1fac86",
    "tools.py": "8efb217df87b0611c83ba79fd371101cf96e05ef",
}
CONTRACT_VERSION = "official-structured-plus-rerank-minimal-v1"
OFFICIAL_PROMPT_SHA256 = "98167bd65463d001e0819917363e765a22c9fb842ecf7909d0e9e8d7aef92bc6"
OFFICIAL_TOOL_SPECS_SHA256 = "026693b43a3b9392c153f24127159d0375df759a989d3fa54636cd23f9e5515f"

# This is one inserted API description. There are no shell examples, generic
# retrieval strategies, recovery policies, or changes to the original guide.
RERANK_ADDITION = (
    "- rerank(query, file_paths?, bm25_candidates?, top_k?, offset?, limit?): "
    "an independently available tool that ranks passages within the documents "
    "already imported by search. It does not import additional documents or "
    "change the files accessible to bash/read. For a new ranking, supply query. "
    "Omit file_paths or pass null for the accumulated document pool; an array "
    "of imported relative file paths restricts that call to those files, and "
    "[] selects no documents. Passage BM25 selects bm25_candidates=M passages "
    "within that scope, then the neural reranker scores them and retains "
    "top_k=K results. Defaults are M=1000, K=min(100,M), offset=0 and limit=10; "
    "1 <= K <= M. Passages have 128 tokens with 32-token overlap. The JSON "
    "response includes status, result_id, scope and pool document counts, "
    "candidate/scored/retained/returned passage counts, offset, limit, "
    "next_offset and items. Each item includes file_path, the complete text "
    "of the passage, BM25 and reranker ranks/scores, source coordinates and "
    "read_args. Pass an item's actual returned read_args object to read to "
    "inspect the passage in its original document context; read_args uses "
    "the existing zero-based line offset and line limit. Relevance scores "
    "are ranking signals, not verification of facts.\n"
    "- rerank(result_id, top_k?, offset?, limit?): page or resize a returned "
    "ranking without rescoring. Use its actual result_id and next_offset; "
    "do not also pass query, file_paths or bm25_candidates. next_offset=null "
    "means the retained K range is exhausted. A search that adds documents "
    "changes the pool version and expires previous result IDs.\n\n"
)


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal constant {name} in {path}")


def _literal_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            statements = [item for item in node.body
                          if not isinstance(item, ast.Expr)]
            if len(statements) != 1 or not isinstance(statements[0], ast.Return):
                raise ValueError(f"{name} must remain a literal return")
            return ast.literal_eval(statements[0].value)
    raise ValueError(f"Missing literal function {name} in {path}")


def official_prompt() -> str:
    return _literal_constant(UPSTREAM / "rise_agent.py", "RISE_SYSTEM_PROMPT_STRUCTURED")


def official_user_prompt() -> str:
    return _literal_constant(UPSTREAM / "rise_agent.py", "RISE_USER_PROMPT")


def official_tool_specs() -> list[dict]:
    return [_literal_function(UPSTREAM / "tools.py", name)
            for name in ("bm25_search_tool_spec", "bash_tool_spec", "read_tool_spec")]


def rerank_tool_spec() -> dict:
    # v14's independent tool contract, without its later instructional wrapper.
    return _literal_function(ROOT / "code/src/rise/rerank_v13.py", "rerank_tool_spec")


def _condition(value: str) -> str:
    if value not in {"A", "B"}:
        raise ValueError("condition must be A (official) or B (official plus rerank)")
    return value


def build_prompt(condition: str) -> str:
    if _condition(condition) == "A":
        return official_prompt()
    original = official_prompt()
    if original.count("You have three tools:") != 1:
        raise ValueError("The official tool-count anchor changed")
    if original.count("Every document begins") != 1:
        raise ValueError("The official document-guide anchor changed")
    return original.replace("You have three tools:", "You have four tools:", 1).replace(
        "Every document begins", RERANK_ADDITION + "Every document begins", 1)


def build_tool_specs(condition: str) -> list[dict]:
    specs = official_tool_specs()
    if _condition(condition) == "B":
        specs.append(rerank_tool_spec())
    return specs


def remove_rerank_addition(prompt: str) -> str:
    """The inverse is deliberately exact, rejecting extra prompt changes."""
    if prompt.count(RERANK_ADDITION) != 1 or prompt.count("You have four tools:") != 1:
        raise ValueError("Unexpected rerank addition or tool-count change")
    return prompt.replace(RERANK_ADDITION, "", 1).replace(
        "You have four tools:", "You have three tools:", 1)


def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def json_sha256(value) -> str:
    # Match the existing runner's system/tool checkpoint hashing convention.
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True))


def build_contract(condition: str) -> dict:
    """Exact model inputs and hashes suitable for freezing and API capture."""
    _condition(condition)
    prompt, specs = build_prompt(condition), build_tool_specs(condition)
    payload = {"condition": condition, "version": CONTRACT_VERSION,
               "system_prompt": prompt, "tool_specs": specs,
               "user_prompt_template": official_user_prompt(),
               "system_prompt_sha256": sha256(prompt),
               "tool_specs_sha256": json_sha256(specs)}
    return {**payload, "contract_sha256": json_sha256(payload)}


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def verify_contracts() -> dict:
    sources = {}
    for name, expected_blob in OFFICIAL_BLOBS.items():
        data = (UPSTREAM / name).read_bytes()
        if _git_blob_sha(data) != expected_blob:
            raise ValueError(f"Official source blob changed: {name}")
        sources[name] = {"git_blob_sha": expected_blob, "sha256": sha256(data)}
    a, b = build_prompt("A"), build_prompt("B")
    a_specs, b_specs = build_tool_specs("A"), build_tool_specs("B")
    if sha256(a) != OFFICIAL_PROMPT_SHA256:
        raise ValueError("Official structured prompt differs from its pinned source")
    if json_sha256(a_specs) != OFFICIAL_TOOL_SPECS_SHA256:
        raise ValueError("Official search/bash/read definitions changed")
    if remove_rerank_addition(b) != a or b_specs[:3] != a_specs:
        raise ValueError("A non-rerank instruction or common tool definition changed")
    return {
        "version": CONTRACT_VERSION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "sources": sources,
        "checks": {"official_source_blobs_match": True,
                   "A_prompt_exactly_official": True,
                   "B_inverse_exactly_A": True,
                   "common_tool_definitions_equal": True,
                   "common_user_template_equal": True},
        "A": {"system_prompt_sha256": sha256(a),
              "tool_specs_sha256": json_sha256(a_specs),
              "tool_names": [s["function"]["name"] for s in a_specs]},
        "B": {"system_prompt_sha256": sha256(b),
              "tool_specs_sha256": json_sha256(b_specs),
              "tool_names": [s["function"]["name"] for s in b_specs]},
        "user_prompt_template_sha256": sha256(official_user_prompt()),
        "rerank_addition_sha256": sha256(RERANK_ADDITION),
        "runtime_feedback_policy": "official; no v13/v14 observational-feedback wrapper",
    }


def _legacy_v14() -> tuple[str, list[dict]]:
    module = runpy.run_path(str(ROOT / "code/src/rise/independent_rerank_prompt.py"))
    feedback = runpy.run_path(str(ROOT / "code/src/rise/v13_feedback.py"))
    original = official_prompt()
    guide = original[original.index("Every document begins"):original.index("Use the tools iteratively:")]
    prompt = module["build_independent_rerank_prompt"](
        document_guide=guide, read_default_limit=2000)
    specs = official_tool_specs()
    specs[1] = feedback["v13_bash_tool_spec"](specs[1])
    specs[2] = module["v13_read_tool_spec"](specs[2], read_default_limit=2000)
    specs.append(module["v14_rerank_tool_spec"](rerank_tool_spec()))
    return prompt, specs


def generate_report(output_dir: Path) -> dict:
    """Save exact inputs and readable diffs before any model call."""
    manifest = verify_contracts()
    output_dir.mkdir(parents=True, exist_ok=True)
    v14_prompt, v14_specs = _legacy_v14()
    snapshots = {
        "official": (official_prompt(), official_tool_specs()),
        "existing_v14": (v14_prompt, v14_specs),
        "new_A": (build_prompt("A"), build_tool_specs("A")),
        "new_B": (build_prompt("B"), build_tool_specs("B")),
    }
    for name, (prompt, specs) in snapshots.items():
        (output_dir / f"{name}.system_prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / f"{name}.tools.json").write_text(
            json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "user_prompt_template.txt").write_text(official_user_prompt(), encoding="utf-8")
    manifest["existing_v14"] = {"system_prompt_sha256": sha256(v14_prompt),
                                 "tool_specs_sha256": json_sha256(v14_specs)}
    (output_dir / "prompt_contract_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    diffs = []
    for left, right in (("official", "existing_v14"), ("official", "new_A"),
                        ("official", "new_B"), ("new_A", "new_B")):
        for kind, index in (("system_prompt", 0), ("tools", 1)):
            first = snapshots[left][index]
            second = snapshots[right][index]
            if index:
                first = json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True)
                second = json.dumps(second, ensure_ascii=False, indent=2, sort_keys=True)
            diff = "".join(difflib.unified_diff(
                first.splitlines(keepends=True), second.splitlines(keepends=True),
                fromfile=f"{left}.{kind}", tofile=f"{right}.{kind}"))
            filename = f"{left}--{right}.{kind}.diff"
            (output_dir / filename).write_text(diff, encoding="utf-8")
            diffs.append(f"- [{left} → {right} {kind}]({filename}): "
                         + ("逐字相同" if not diff else "有差异，完整补丁已保存"))
    report = "\n".join([
        "# 严格提示词对照：官方原版、现有 v14、新实验 A/B", "",
        f"官方来源：[{OFFICIAL_REPOSITORY}@{OFFICIAL_COMMIT[:8]}]"
        f"({OFFICIAL_REPOSITORY}/tree/{OFFICIAL_COMMIT})。",
        "源文件以 GitHub Contents API 获取，Git blob SHA 已逐字节核验。", "",
        "| 维度 | 官方 / 新 A | 现有 v14 | 新 B |",
        "| --- | --- | --- | --- |",
        "| 开场与工具数 | 原文；三种工具 | 改写开场；四种工具 | 原文；仅 three 改为 four |",
        "| search 提示与定义 | 原文 | 系统说明重写 | 与官方逐字相同 |",
        "| bash 提示与定义 | 原文 | 增加长行、数值与路径提示；schema 增 timeout | 与官方逐字相同 |",
        "| read 提示与定义 | 原文 | 改写默认读取策略与长行恢复 | 与官方逐字相同 |",
        "| 结构化文档阅读说明 | 原文 | 复用原文 | 与官方逐字相同 |",
        "| 通用查询优化 / 失败恢复 | 仅原版现有内容 | 增加查询拆分、重复/空结果/截断恢复 | 不增加 |",
        "| bash → rerank 范例 | 无 | 有可选固定范例 | 无 |",
        "| rerank | 无 | 接口及额外范例/策略 | 仅独立接口、参数/返回、read_args 回读说明 |",
        "| 终答和停止说明 | 原文 | 重写 | 与官方逐字相同 |",
        "| 通用运行时反馈 | 官方工具行为 | ProgressFeedback 追加提示 | 与 A 相同；禁用 v14 wrapper |", "",
        "验证：删除 B 的唯一 rerank 说明块并把 four 还原 three 后，"
        "所得字符串与官方 A 完全相同；B 的前三个工具对象与 A 完全相同。",
        "未修正官方系统提示与工具说明中原有的读取偏好差异。", "",
        "## 完整差异", "", *diffs, "",
        "## 输入指纹", "", "```json",
        json.dumps(manifest, ensure_ascii=False, indent=2), "```", "",
        "这些文件只证明提示词和工具定义的控制；模型、运行配置、"
        "数据、工具实际返回与沙箱仍须由运行预检分别核验。", "",
    ])
    (output_dir / "PROMPT_DIFF.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
