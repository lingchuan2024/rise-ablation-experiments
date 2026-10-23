"""Offline checks of the B100-derived interventions and their shared inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from strict_experiment import strict_prompt as frozen
from strict_ablations import prompt


def named(specs, name):
    return next(spec for spec in specs if spec["function"]["name"] == name)


class MinimalAblationPromptTests(unittest.TestCase):
    def test_control_is_literal_frozen_B100_and_remains_pinned(self):
        b = frozen.build_contract("B")
        c = prompt.build_contract(prompt.CONTROL)
        for key in ("system_prompt", "tool_specs", "user_prompt_template"):
            self.assertEqual(c[key], b[key])
        self.assertEqual(c["system_prompt_sha256"],
                         "bb605017783d5c53aa53ba5ac2948842694c15b2f8a7f98f70a4d4d5757293fc")
        self.assertEqual(c["tool_specs_sha256"],
                         "23b3e8c08214dec94a83f08fcf5f3e4968f4bafdc7c1922eaca231d72599347b")
        self.assertEqual(frozen.sha256(c["user_prompt_template"]),
                         "6d1d40b7d38fa161ddaa08cf1196e9f529b1c82879c8ede6f20ec95556d61274")
        self.assertNotIn(prompt.CONTROL, prompt.ARMS)
        self.assertEqual(set(prompt.ARMS), {"search_paging", "auto_rerank", "no_bash"})
        self.assertNotEqual(prompt.VERSION, b["version"])

    def test_paging_only_adds_the_two_search_call_forms(self):
        base = frozen.build_contract("B")
        c = prompt.build_contract("search_paging")
        self.assertEqual(c["system_prompt"].replace(prompt.PAGING_ADDITION, "", 1),
                         base["system_prompt"])
        self.assertEqual(c["tool_specs"][1:], base["tool_specs"][1:])
        original = base["tool_specs"][0]["function"]
        search = c["tool_specs"][0]["function"]
        self.assertTrue(search["description"].startswith(original["description"]))
        self.assertEqual(search["parameters"]["properties"]["queries"],
                         original["parameters"]["properties"]["queries"])
        self.assertEqual(set(search["parameters"]["properties"]),
                         {"queries", "search_id", "offset"})
        self.assertEqual(search["parameters"]["required"], [])
        self.assertEqual(search["parameters"]["properties"]["offset"]["minimum"], 0)
        for neutral_interface in ("at most ten", "100-character", "original document BM25 order",
                                  "does not retrieve, import or score", "separate from rerank result IDs"):
            self.assertIn(neutral_interface, prompt.PAGING_ADDITION)
        for strategy in ("prefer", "priorit", "must page", "insufficient", "before bash"):
            self.assertNotIn(strategy, prompt.PAGING_ADDITION.lower())

    def test_auto_keeps_official_query_advice_bash_read_and_ending(self):
        base = frozen.build_contract("B")
        c = prompt.build_contract("auto_rerank")
        text = c["system_prompt"]
        advice = ("Each query is matched as a bag of words against document text, so write "
                  "queries as natural-language descriptions with several distinctive terms; "
                  "pass multiple complementary queries together for broader coverage.")
        self.assertIn(advice, text)
        self.assertNotIn("- rerank(", text)
        self.assertNotIn(frozen.RERANK_ADDITION, text)
        self.assertIn("You have three tools:", text)
        for prefix in ("- bash(command):", "- read(file_path, offset?, limit?):"):
            line = next(line for line in base["system_prompt"].splitlines() if line.startswith(prefix))
            self.assertIn(line, text)
        ending = base["system_prompt"][base["system_prompt"].index("Every document begins"):]
        self.assertTrue(text.endswith(ending))
        self.assertEqual([s["function"]["name"] for s in c["tool_specs"]], ["search", "bash", "read"])
        self.assertEqual(c["tool_specs"][1:], base["tool_specs"][1:3])
        self.assertEqual(c["tool_specs"][0]["function"]["parameters"],
                         base["tool_specs"][0]["function"]["parameters"])
        description = c["tool_specs"][0]["function"]["description"]
        self.assertIn("Specific distinctive terms (rare phrases, proper nouns, distinctive number/date combinations) tend to give the best results.", description)
        for required in ("own top-1000", "M=1000", "K=100", "first ten complete passages",
                         "128 tokens with 32-token overlap", "unchanged search query",
                         "no result-pagination interface", prompt.PASSAGE_INTERFACE):
            self.assertIn(required, text)
            self.assertIn(required, description)

    def test_no_bash_removes_only_unavailable_capability(self):
        base = frozen.build_contract("B")
        c = prompt.build_contract("no_bash")
        self.assertNotIn("bash", c["system_prompt"])
        self.assertNotIn("bash", json.dumps(c["tool_specs"]))
        self.assertEqual([s["function"]["name"] for s in c["tool_specs"]], ["search", "read", "rerank"])
        self.assertEqual(named(c["tool_specs"], "search"), base["tool_specs"][0])
        self.assertEqual(named(c["tool_specs"], "read"), base["tool_specs"][2])
        expected_rerank = copy.deepcopy(base["tool_specs"][-1])
        fn = expected_rerank["function"]
        fn["description"] = fn["description"].replace(
            "read/bash coordinates, without restricting bash/read.",
            "read coordinates, without restricting read.")
        paths = fn["parameters"]["properties"]["file_paths"]
        paths["description"] = paths["description"].replace(
            ", such as files selected with bash", "")
        self.assertEqual(named(c["tool_specs"], "rerank"), expected_rerank)
        for api in ("file_paths", "[] selects no documents", "M=1000", "K=min(100,M)",
                    "read_args", "result_id", "next_offset=null"):
            self.assertIn(api, c["system_prompt"])
        for extra in ("read_may_truncate", "context_char_start", "read_chars", "EXAMPLE_ENTITY"):
            self.assertNotIn(extra, c["system_prompt"])

    def test_shared_guide_read_user_and_no_legacy_strategy(self):
        base = frozen.build_contract("B")
        start = base["system_prompt"].index("Every document begins")
        end = base["system_prompt"].index("Use the tools iteratively:")
        guide = base["system_prompt"][start:end]
        for arm in (prompt.CONTROL, *prompt.ARMS):
            with self.subTest(arm=arm):
                c = prompt.build_contract(arm)
                self.assertIn(guide, c["system_prompt"])
                self.assertEqual(named(c["tool_specs"], "read"), base["tool_specs"][2])
                self.assertEqual(c["user_prompt_template"], base["user_prompt_template"])
                for legacy in ("EXAMPLE_ENTITY", "head limits lines", "numeric boundaries",
                               "Split multi-condition", "Do not repeat unchanged results",
                               "Progress feedback", "prefer a rerank item's"):
                    self.assertNotIn(legacy, c["system_prompt"])
                if arm != "no_bash":
                    self.assertEqual(named(c["tool_specs"], "bash"), base["tool_specs"][1])

    def test_contract_hashes_are_recomputable_and_distinct(self):
        seen = set()
        for arm in (prompt.CONTROL, *prompt.ARMS):
            c = prompt.build_contract(arm)
            self.assertEqual(c["condition"], arm)
            self.assertEqual(c["version"], prompt.VERSION)
            self.assertEqual(c["system_prompt_sha256"], frozen.sha256(c["system_prompt"]))
            self.assertEqual(c["tool_specs_sha256"], frozen.json_sha256(c["tool_specs"]))
            payload = {k: v for k, v in c.items() if k != "contract_sha256"}
            self.assertEqual(c["contract_sha256"], frozen.json_sha256(payload))
            seen.add(c["contract_sha256"])
        self.assertEqual(len(seen), 4)

    def test_builds_are_fresh_and_do_not_mutate_control(self):
        before = frozen.build_contract("B")
        for arm in (prompt.CONTROL, *prompt.ARMS):
            c = prompt.build_contract(arm)
            c["tool_specs"][0]["function"]["description"] = "mutated"
            self.assertNotEqual(prompt.build_contract(arm)["tool_specs"][0]["function"]["description"], "mutated")
        self.assertEqual(frozen.build_contract("B"), before)

    def test_verify_and_report_save_exact_reviewable_inputs(self):
        result = prompt.verify_contracts()
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["contracts"]["control"]["system_prompt_sha256"],
                         prompt.CONTROL_SYSTEM_PROMPT_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prompt.generate_report(root)
            self.assertEqual(manifest, result)
            self.assertEqual(json.loads((root / "prompt_contract_manifest.json").read_text()), result)
            for arm in (prompt.CONTROL, *prompt.ARMS):
                c = prompt.build_contract(arm)
                self.assertEqual((root / f"{arm}.system_prompt.txt").read_text(), c["system_prompt"])
                self.assertEqual(json.loads((root / f"{arm}.tools.json").read_text()), c["tool_specs"])
                self.assertEqual(json.loads((root / f"{arm}.contract.json").read_text()), c)
            for arm in prompt.ARMS:
                for kind in ("system_prompt", "tools"):
                    diff = (root / f"control--{arm}.{kind}.diff").read_text()
                    self.assertIn(f"--- control.{kind}", diff)
                    self.assertIn(f"+++ {arm}.{kind}", diff)
            self.assertEqual(prompt.generate_report(root), result)

    def test_unknown_arm_does_not_silently_select_a_baseline(self):
        for arm in ("A", "B", "v14", "", "search_paging_prompt_v2"):
            for build in (prompt.build_prompt, prompt.build_tool_specs, prompt.build_contract):
                with self.subTest(arm=arm, build=build.__name__), self.assertRaises(ValueError):
                    build(arm)


if __name__ == "__main__":
    unittest.main()
