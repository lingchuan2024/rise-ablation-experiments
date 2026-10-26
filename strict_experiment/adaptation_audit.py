"""Export exact source differences for the shared official/server adapters."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def functions(path):
    source = path.read_text(encoding="utf-8")
    return {node.name: ast.get_source_segment(source, node) + "\n"
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def generate(output=None):
    output = Path(output) if output else ROOT / "strict_experiment/prompt_audit/adaptations"
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "strict_experiment/upstream/tools.py"
    shared = ROOT / "code/src/rise/tools.py"
    before, after = functions(original), functions(shared)
    identity = {"policy": "identical existing server adapters in A and B; no tool behavior repair",
                "functions": {}, "shared_source_sha256": {}}
    for name in ("make_read_tool", "make_bm25_search_tool", "_read_snippet",
                 "make_bash_tool", "_sandboxed_bash_command"):
        diff = "".join(difflib.unified_diff(
            before.get(name, "").splitlines(True), after.get(name, "").splitlines(True),
            fromfile="official/tools.py::" + name,
            tofile="shared-server/tools.py::" + name))
        (output / (name + ".diff")).write_text(diff, encoding="utf-8")
        identity["functions"][name] = {"equal": not diff,
            "official_sha256": hashlib.sha256(before.get(name, "").encode()).hexdigest(),
            "shared_sha256": hashlib.sha256(after.get(name, "").encode()).hexdigest()}
    source_paths = ("code/src/rise/agent.py", "code/src/rise/tools.py",
                    "code/src/rise/api_retry.py", "code/src/rise/local_vllm.py",
                    "code/src/rise/turn_checkpoint.py", "code/src/rise/retrieval.py",
                    "server/shared_reranker.py", "strict_experiment/runtime.py")
    identity["shared_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}
    for name in ("agent",):
        a = (ROOT / f"strict_experiment/upstream/{name}.py").read_text()
        b = (ROOT / f"code/src/rise/{name}.py").read_text()
        (output / (name + ".diff")).write_text("".join(difflib.unified_diff(
            a.splitlines(True), b.splitlines(True),
            fromfile="official/" + name + ".py", tofile="shared-server/" + name + ".py")))
    (output / "source_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return identity


if __name__ == "__main__":
    generate()
