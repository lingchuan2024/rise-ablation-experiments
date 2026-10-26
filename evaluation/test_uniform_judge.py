"""Offline integrity tests: no model or network access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import uniform_judge as u


class UniformJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.group = {"group_id": "group", "question": "Name?", "gold_answer": "A",
            "answers": [{"case_id": "x", "final_text": "A",
                         "final_sha256": u.hashlib.sha256(b"A").hexdigest()}]}

    def content(self, correct=True, flag=False):
        return json.dumps({"reviews": [{"case_id": "x", "correct": correct,
                            "reason": "Same answer", "needs_review": flag}]})

    def bundle(self, third=False):
        u.write(self.root / "blind_packet.json", {"groups": [self.group]})
        u.write(self.root / "private_mapping.json", {"records": [{"case_id": "x", "arm": "v14",
            "query_id": 1, "final_sha256": self.group["answers"][0]["final_sha256"]}]})
        protocol = {"model": u.MODEL, "base_url": u.BASE_URL, "system": u.SYSTEM,
            "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": 4096,
            "concurrency": 2, "expected_counts": {"v14": 1}}
        u.write(self.root / "protocol.json", protocol)
        u.write(self.root / "preflight.json", {"packet_sha256": u.file_sha(self.root / "blind_packet.json"),
            "mapping_sha256": u.file_sha(self.root / "private_mapping.json"),
            "protocol_sha256": u.file_sha(self.root / "protocol.json")})
        pins = []
        for i in (1, 2, 3) if third else (1, 2):
            content = self.content(correct=i != 2 if third else True, flag=third)
            payload = u.payload_for(self.group, i, protocol)
            receipt = {"response_id": "response" + str(i), "returned_model": "deepseek-v4-1-flash-261017",
                "finish_reason": "stop", "content_sha256": u.hashlib.sha256(content.encode()).hexdigest(),
                "request_payload_sha256": u.hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()}
            f = self.root / f"pass_{i}/group.json"
            u.write(f, {"pass": i, "group_payload_sha256": u.digest(self.group),
                "labels": u.parse_labels(content, self.group), "raw_public_judge_content": content,
                "receipt": receipt})
            pins.append({"path": str(f.relative_to(self.root)), "sha256": u.file_sha(f)})
        u.write(self.root / "third_pass_selection.json", {"group_ids": ["group"] if third else []})
        u.write(self.root / "api_scoring_complete.json", {"complete": True})
        u.write(self.root / "response_receipt_pins.json", {"responses": pins})

    def test_format_cleanup_never_changes_string_literals(self):
        text = '{"a":<|endoftext|>true,"b":"literal <|endoftext|>"}'
        clean, count = u.normalize_content("```json\n" + text + "\n```")
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(clean), {"a": True, "b": "literal <|endoftext|>"})

    def test_valid_label_has_exact_final_hash(self):
        self.assertEqual(u.parse_labels(self.content(), self.group)[0]["final_sha256"],
                         self.group["answers"][0]["final_sha256"])

    def test_empty_final_must_be_incorrect(self):
        group = copy.deepcopy(self.group)
        group["answers"][0]["final_text"] = ""
        with self.assertRaisesRegex(ValueError, "Empty final"):
            u.parse_labels(self.content(), group)

    def test_bool_is_not_number_or_string(self):
        for value in (1, "true", None):
            obj = json.loads(self.content())
            obj["reviews"][0]["correct"] = value
            with self.assertRaisesRegex(ValueError, "Boolean"):
                u.parse_labels(json.dumps(obj), self.group)

    def test_duplicate_or_missing_case_rejected(self):
        obj = json.loads(self.content())
        obj["reviews"] *= 2
        with self.assertRaisesRegex(ValueError, "count"):
            u.parse_labels(json.dumps(obj), self.group)
        obj["reviews"] = []
        with self.assertRaises(ValueError):
            u.parse_labels(json.dumps(obj), self.group)

    def test_wrong_case_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Case IDs"):
            u.parse_labels(self.content().replace('"x"', '"other"'), self.group)

    def test_identical_finals_receive_identical_labels(self):
        group = copy.deepcopy(self.group)
        group["answers"].append({**group["answers"][0], "case_id": "y"})
        obj = json.loads(self.content())
        obj["reviews"].append({**obj["reviews"][0], "case_id": "y", "correct": False})
        with self.assertRaisesRegex(ValueError, "Identical finals"):
            u.parse_labels(json.dumps(obj), group)

    def test_immutable_output_cannot_replace_result(self):
        path = self.root / "result.json"
        u.write(path, {"correct": True}, immutable=True)
        with self.assertRaises(FileExistsError):
            u.write(path, {"correct": False}, immutable=True)
        self.assertTrue(u.load(path)["correct"])

    def test_empty_reason_rejected(self):
        with self.assertRaisesRegex(ValueError, "reason"):
            u.parse_labels(self.content().replace("Same answer", "  "), self.group)

    def test_modified_prepared_file_rejected(self):
        self.bundle()
        (self.root / "protocol.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            u.load_bundle(self.root)

    def test_third_pass_selected_from_flags_or_disagreement(self):
        self.bundle(third=True)
        self.assertEqual(u.third_pass_groups(self.root, [self.group]), [self.group])

    def test_missing_required_third_response_rejected(self):
        self.bundle(third=True)
        (self.root / "pass_3/group.json").unlink()
        with self.assertRaises(FileNotFoundError):
            u.validate_saved(self.root, self.group, 3)

    def test_changed_raw_judge_response_rejected(self):
        self.bundle()
        f = self.root / "pass_1/group.json"
        value = u.load(f)
        value["raw_public_judge_content"] = self.content(False)
        u.write(f, value)
        with self.assertRaisesRegex(ValueError, "Saved labels"):
            u.validate_saved(self.root, self.group, 1)

    def test_changed_labels_rejected(self):
        self.bundle()
        f = self.root / "pass_1/group.json"
        value = u.load(f)
        value["labels"][0]["correct"] = False
        u.write(f, value)
        with self.assertRaisesRegex(ValueError, "Saved labels"):
            u.validate_saved(self.root, self.group, 1)

    def test_resume_uses_validated_saved_result_without_network(self):
        self.bundle()
        grader = u.Grader(self.root, u.load(self.root / "protocol.json"), "test-only")
        with patch.object(grader, "request", side_effect=AssertionError("network prohibited")):
            value = grader.evaluate(self.group, 1)
        self.assertTrue(value["labels"][0]["correct"])

    def test_wrong_cohort_rejected_before_output_creation(self):
        input_path = self.root / "input.json"
        u.write(input_path, {"records": []})
        cohort = self.root / "cohort.jsonl"
        cohort.write_text('{"query_id":1}\n')
        with self.assertRaisesRegex(ValueError, "fixed100"):
            u.prepare(input_path, cohort, self.root / "new")
        self.assertFalse((self.root / "new").exists())

    def test_paired_exact_test_counts(self):
        first = {i: i < 68 for i in range(100)}
        # 54 both,14 first only,7 second only,25 both wrong.
        second = {i: i < 54 or 68 <= i < 75 for i in range(100)}
        r = u.paired(first, second)
        self.assertEqual((r["first_only_correct"], r["second_only_correct"]), (14, 7))
        self.assertAlmostEqual(r["mcnemar_exact_two_sided"], 0.18924713134765625)

    def test_collect_excludes_existing_labels_and_reasoning(self):
        base = {"query_id": 1, "question": "Q", "gold_answer": "A", "final_text": "A",
                "correct": False, "reasoning_content": "PRIVATE", "judge": "old"}
        files = [self.root / (name + ".json") for name in ("v14", "recovery", "nobash")]
        u.write(files[0], {"records": [base]})
        u.write(files[1], {"records": [{**base, "arm": "auto_rerank"}, {**base, "arm": "control"}]})
        u.write(files[2], {"records": [{**base, "arm": "no_bash"}]})
        output = self.root / "collected.json"
        self.assertEqual(u.collect(*files, output)["collected"], 3)
        raw = output.read_text()
        for forbidden in ('"correct"', "PRIVATE", '"judge"', '"reasoning_content"'):
            self.assertNotIn(forbidden, raw)


if __name__ == "__main__":
    unittest.main()
