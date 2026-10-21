"""Check the v13 evidence guidance without inference or corpus-specific hints."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.independent_rerank_prompt import (
    RERANK_FILE_SUBSET_EXAMPLE,
    RERANK_PROMPT_VERSION,
    build_independent_rerank_prompt,
    v14_rerank_tool_spec,
)


def baseline_document_guide() -> str:
    """Read the literal guide used by agent wiring, without importing the runner."""
    source = (ROOT / "src/rise/rise_agent.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "RISE_SYSTEM_PROMPT_STRUCTURED"
                for target in node.targets):
            structured_prompt = ast.literal_eval(node.value)
            start = structured_prompt.index("Every document begins")
            end = structured_prompt.index("Use the tools iteratively:")
            return structured_prompt[start:end]
    raise AssertionError("Baseline structured-document guide not found")


class V13PromptGuidanceTests(unittest.TestCase):
    def prompts(self):
        yield "unstructured", build_independent_rerank_prompt()
        yield "structured", build_independent_rerank_prompt(
            document_guide=baseline_document_guide())

    def test_version_changes_checkpoint_identity(self):
        self.assertEqual(RERANK_PROMPT_VERSION, "v14-explicit-file-subset-prompt-v1")

    def test_optional_handoff_example_is_complete_in_both_prompt_modes(self):
        for mode, prompt in self.prompts():
            with self.subTest(mode=mode):
                self.assertEqual(prompt.count(RERANK_FILE_SUBSET_EXAMPLE), 1)
                self.assertIn("rg -l -i 'EXAMPLE_ENTITY|EXAMPLE_ALIAS' .", prompt)
                self.assertIn('"file_paths": ["docs/example_a.txt", "docs/example_b.txt"]', prompt)
                self.assertIn("actual returned read_args to read", prompt)
                self.assertIn("use the returned coordinates", prompt)
                self.assertIn("200/30/5 are adjustable examples", prompt)
                self.assertIn("it is not required for every question", prompt)
                self.assertIn("Do not automatically treat the preceding bash output", prompt)
                self.assertIn("explicitly set file_paths yourself", prompt)
                self.assertIn("Omitting file_paths still ranks the full accumulated pool", prompt)
                self.assertIn("bash/read retain access to the full original-document pool", prompt)

    def test_rerank_spec_wrapper_only_extends_description_without_mutation(self):
        # Keep this prompt-only contract independent from backend dependencies.
        source = (ROOT / "src/rise/rerank_v13.py").read_text(encoding="utf-8")
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "rerank_tool_spec")
        baseline = ast.literal_eval(function.body[0].value)
        untouched = copy.deepcopy(baseline)
        current = v14_rerank_tool_spec(baseline)
        self.assertEqual(baseline, untouched)
        self.assertEqual(current["function"]["description"],
                         baseline["function"]["description"] + "\n\n" + RERANK_FILE_SUBSET_EXAMPLE)
        current["function"]["description"] = baseline["function"]["description"]
        self.assertEqual(current, baseline)
        current["function"]["parameters"]["properties"]["file_paths"]["items"]["type"] = "changed"
        self.assertEqual(baseline, untouched)

    def test_agent_wires_example_to_rerank_tool_spec(self):
        source = (ROOT / "src/rise/rise_agent.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "v14_rerank_tool_spec"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), "v14_rerank_tool_spec(rerank_tool_spec())")

    def test_both_modes_distinguish_pool_coverage_from_ranking_and_page_size(self):
        for mode, prompt in self.prompts():
            with self.subTest(mode=mode):
                self.assertIn("Only search imports additional documents", prompt)
                self.assertIn("empty pool-local result does not mean", prompt)
                self.assertIn("larger M sends more candidates to neural scoring", prompt)
                self.assertIn("larger K makes more of the already scored ranking available", prompt)
                self.assertIn("limit controls only the number of complete passages", prompt)
                self.assertIn("Neither M nor K can find documents outside", prompt)
                self.assertIn("next_offset=null means only that the current K range", prompt)
                self.assertIn("not that all scored candidates", prompt)

    def test_both_modes_support_entity_checks_and_nonbinding_progress_feedback(self):
        for mode, prompt in self.prompts():
            with self.subTest(mode=mode):
                self.assertIn("candidate entity", prompt)
                self.assertIn("entity plus the still-unverified relationship or fact", prompt)
                self.assertIn("Split multi-condition questions", prompt)
                self.assertIn("Keep alternative candidates and source conflicts", prompt)
                self.assertIn("complementary search when coverage is uncertain", prompt)
                self.assertIn("not whether a fact is true or evidence is absent", prompt)
                self.assertIn("You decide the next tool and rerank parameters", prompt)
                self.assertIn("Neither bash nor rerank is a prerequisite", prompt)
                self.assertIn("You need not call every tool", prompt)

    def test_both_modes_explain_focused_reading_and_long_line_filtering(self):
        for mode, prompt in self.prompts():
            with self.subTest(mode=mode):
                self.assertIn("read_args", prompt)
                self.assertIn("read_may_truncate is true", prompt)
                self.assertIn("bash_args for exact passage recovery", prompt)
                self.assertIn("head limits lines, not characters", prompt)
                self.assertIn("select files with rg -l", prompt)
                self.assertIn("short matches with rg -o", prompt)
                self.assertIn("fixed-string matching for literal decimals", prompt)
                self.assertIn("numeric boundaries and surrounding meaning", prompt)
                self.assertIn("do not guess absolute locations", prompt)
                self.assertIn("Offsets and limits are in LINES", prompt)
                self.assertIn("offsets are zero-based", prompt)
                self.assertIn("default limit is 2000 lines", prompt)

    def test_structured_guide_is_kept_complete_once(self):
        guide = baseline_document_guide()
        prompt = build_independent_rerank_prompt(document_guide=guide)
        self.assertEqual(prompt.count(guide), 1)
        self.assertIn("Never extract a specific fact", guide)
        self.assertIn("read(path, offset=<start-1>, limit=<end-start+1>)", guide)
        self.assertNotIn("Every document begins", build_independent_rerank_prompt())
        custom = build_independent_rerank_prompt(read_default_limit=321)
        self.assertIn("default limit is 321 lines", custom)

    def test_general_guidance_has_no_pilot_specific_names_or_answers(self):
        forbidden = (
            "qid_70", "qid_81", "qid_96", "qid_116", "qid_122",
            "Claude Berri", "Thomas Langmann", "Sam O'Brien",
            "Glass FurnaceWay", "Juan Gabriel", "Nikhoj Sangbad",
            "Thomas Mapfumo", "Mukanya", "Kune Rima", "Bruce Chapman",
            "Jacek Rojowski", "Bożena Muszyńska",
        )
        for mode, prompt in self.prompts():
            with self.subTest(mode=mode):
                for item in forbidden:
                    self.assertNotIn(item.casefold(), prompt.casefold())


if __name__ == "__main__":
    unittest.main()
