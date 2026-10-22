"""Contracts for the experimental intervention, independent of model quality."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from strict_experiment.strict_prompt import (
    OFFICIAL_PROMPT_SHA256, RERANK_ADDITION, build_prompt, build_tool_specs,
    generate_report, official_prompt, official_tool_specs, remove_rerank_addition,
    sha256, verify_contracts,
)


class StrictPromptTests(unittest.TestCase):
    def test_official_sources_and_prompt_are_pinned(self):
        manifest = verify_contracts()
        self.assertTrue(all(manifest["checks"].values()))
        self.assertEqual(sha256(build_prompt("A")), OFFICIAL_PROMPT_SHA256)
        self.assertEqual(build_prompt("A"), official_prompt())

    def test_non_rerank_text_is_preserved_byte_for_byte(self):
        a, b = build_prompt("A"), build_prompt("B")
        self.assertEqual(remove_rerank_addition(b), a)
        self.assertEqual(b.count(RERANK_ADDITION), 1)
        self.assertIn("You have four tools:", b)
        # The original final-answer/stop block stays at the original end.
        self.assertTrue(b.endswith(a[a.index("Use the tools iteratively:"):]))

    def test_common_tool_objects_are_identical_and_unmodified(self):
        a, b = build_tool_specs("A"), build_tool_specs("B")
        self.assertEqual(b[:3], a)
        self.assertEqual(a, official_tool_specs())
        self.assertEqual([s["function"]["name"] for s in a], ["search", "bash", "read"])
        self.assertEqual([s["function"]["name"] for s in b], ["search", "bash", "read", "rerank"])
        self.assertNotIn("timeout", b[1]["function"]["parameters"]["properties"])
        b[2]["function"]["description"] = "mutated"
        self.assertEqual(build_tool_specs("B")[:3], a)

    def test_rerank_schema_has_real_array_and_both_call_forms(self):
        spec = build_tool_specs("B")[-1]["function"]
        props = spec["parameters"]["properties"]
        self.assertEqual(props["file_paths"]["type"], ["array", "null"])
        self.assertEqual(props["file_paths"]["items"], {"type": "string"})
        self.assertEqual(set(props), {"query", "file_paths", "bm25_candidates", "top_k", "offset", "limit", "result_id"})
        self.assertIn("read_args", RERANK_ADDITION)
        self.assertIn("next_offset=null", RERANK_ADDITION)

    def test_no_v14_extra_general_instructions_or_shell_example(self):
        b = build_prompt("B")
        for instruction in ("EXAMPLE_ENTITY", "numeric boundaries", "head limits lines",
                            "Do not repeat unchanged results", "Split multi-condition",
                            "read_may_truncate", "bash_args", "Progress feedback"):
            self.assertNotIn(instruction, b)
        self.assertNotIn("rg -l -i", build_tool_specs("B")[-1]["function"]["description"])

    def test_report_saves_exact_prompts_tools_and_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_report(root)
            self.assertEqual((root / "new_B.system_prompt.txt").read_text(), build_prompt("B"))
            self.assertEqual(json.loads((root / "new_B.tools.json").read_text()), build_tool_specs("B"))
            self.assertEqual((root / "official--new_A.system_prompt.diff").read_text(), "")
            self.assertEqual((root / "official--new_A.tools.diff").read_text(), "")
            self.assertIn("v14", (root / "PROMPT_DIFF.md").read_text())
            self.assertEqual(json.loads((root / "prompt_contract_manifest.json").read_text()), manifest)

    def test_unrecognized_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt("v14")
        with self.assertRaises(ValueError):
            build_tool_specs("v14")


if __name__ == "__main__":
    unittest.main()
