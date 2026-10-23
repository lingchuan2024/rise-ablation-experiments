"""Offline tests for public-final projection; no model, SSH, or real trace reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from operations import recover_observed_final as recovery


class ObservedFinalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/strict_continue_v1/auto_rerank"
        self.output = self.root / "operations/recovery_evidence/qid_310_observed_final.json"
        self.secret = "PRIVATE_SENTINEL_SHOULD_NEVER_BE_DECODED"
        self.config = {"model": recovery.MODEL, "max_turns": 100}
        self.frozen = {"contract_sha256": "a" * 64, "system_prompt": "public system",
            "tool_specs": [], "system_prompt_sha256": "b" * 64, "tool_specs_sha256": "c" * 64}
        self.manifest = {"config": self.config, "config_sha256": recovery.digest(self.config),
                         "contracts": {"auto_rerank": self.frozen}}
        self.manifest["manifest_sha256"] = recovery.digest(self.manifest)
        self.contract = {"query_id": 310, "arm": "auto_rerank", "model": recovery.MODEL,
            "common_config_sha256": self.manifest["config_sha256"], "question_sha256": "d" * 64,
            **self.frozen}
        self.ledger, self.events = [], []
        for i in range(2):
            usage = {"prompt_tokens": 10 + i, "completion_tokens": 3 + i, "total_tokens": 13 + 2 * i}
            self.ledger.append({"query_id": 310, "arm": "auto_rerank", "requested_model": recovery.MODEL,
                "reported_model": recovery.MODEL, "status": "ok", "contract_sha256": "a" * 64,
                "response_id": "response-" + str(i), "usage": usage,
                "finish_reason": "tool_calls" if i == 0 else "stop", "unix_time": 10 + i,
                "request_seconds": 0.4})
            self.events.append({"time": 10.5 + i, "attempt": 1,
                "category": "tool_calls" if i == 0 else "content",
                "request": {"model": recovery.MODEL,
                    "messages": [{"role": "assistant", "reasoning_content": self.secret}]},
                "response": {"id": "response-" + str(i), "model": recovery.MODEL, "usage": dict(usage),
                    "choices": [{"finish_reason": "tool_calls" if i == 0 else "stop",
                        "message": {"content": "" if i == 0 else "Observed public answer.",
                            "reasoning_content": self.secret,
                            "tool_calls": [{"private_argument": self.secret}] if i == 0 else None}}]}})
        self.cp = {"version": 1, "elapsed_seconds": 0.7,
            "identity": {"model": recovery.MODEL, "query_id": "310", "ablation_arm": "auto_rerank",
                "contract_sha256": "a" * 64, "common_config_sha256": self.manifest["config_sha256"],
                "question_sha256": "d" * 64, "system_prompt_sha256": "b" * 64,
                "tool_specs_sha256": "c" * 64, "run_config_hash": "e" * 64},
            "messages": [{"role": "assistant", "reasoning_content": self.secret}],
            "run": {"turns": [{"reasoning_content": self.secret}], "terminated_reason": "",
                "total_prompt_tokens": 10, "total_cached_prompt_tokens": 0, "total_completion_tokens": 3}}
        self.paths = {"manifest": self.root / "reports/strict_continue_v1/run_manifest.json",
            "contract": self.run / "request_contracts/qid_310.json",
            "ledger": self.run / "api_ledgers/qid_310.jsonl",
            "api": self.run / "_traces/qid_310/single.api.jsonl",
            "checkpoint": self.run / "_checkpoints/qid_310.json"}
        self.write()

    def write(self):
        for key, value in (("manifest", self.manifest), ("contract", self.contract), ("checkpoint", self.cp)):
            self.paths[key].parent.mkdir(parents=True, exist_ok=True)
            self.paths[key].write_text(json.dumps(value))
        for key, rows in (("ledger", self.ledger), ("api", self.events)):
            self.paths[key].parent.mkdir(parents=True, exist_ok=True)
            self.paths[key].write_text("".join(json.dumps(row) + "\n" for row in rows))

    def recover(self):
        return recovery.recover(self.root, self.output,
            expected_manifest_sha256=self.manifest["manifest_sha256"])

    def test_hidden_values_never_reach_json_decoder(self):
        original = json.loads
        decoded = []
        def watched(value, *args, **kwargs):
            decoded.append(value)
            return original(value, *args, **kwargs)
        with mock.patch.object(recovery.json, "loads", side_effect=watched):
            result = recovery.Projection(json.dumps(self.events[-1]).encode()).project(recovery.FINAL_API_SPEC)
            cp = recovery.Projection(json.dumps(self.cp).encode()).project(recovery.CP_SPEC)
        self.assertEqual(result["response"]["choices"][0]["message"]["content"], "Observed public answer.")
        self.assertEqual(cp["run"]["turns"], 1)
        self.assertFalse(any(self.secret.encode() in (x if isinstance(x, bytes) else x.encode()) for x in decoded))
        self.assertNotIn("messages", cp)
        self.assertNotIn("reasoning_content", json.dumps(result))

    def test_derived_final_pins_sources_and_never_claims_natural_terminal(self):
        before = {key: path.read_bytes() for key, path in self.paths.items()}
        result = self.recover()
        receipt = json.loads(self.output.read_bytes())
        self.assertTrue(result["verified"])
        self.assertEqual(receipt["final_text"], "Observed public answer.")
        self.assertEqual((receipt["known_input_tokens"], receipt["known_output_tokens"]), (21, 7))
        self.assertIsNone(receipt["elapsed_seconds"])
        self.assertFalse(receipt["full_natural_terminal_reconstructed"])
        self.assertFalse(receipt["private_message_or_reasoning_values_decoded"])
        self.assertEqual(receipt["new_model_requests"], 0)
        self.assertNotIn(self.secret, self.output.read_text())
        self.assertEqual(before, {key: path.read_bytes() for key, path in self.paths.items()})

    def test_mismatched_usage_and_response_ids_rejected(self):
        for kind in ("usage", "id"):
            with self.subTest(kind=kind):
                old = copy.deepcopy(self.events)
                if kind == "usage": self.events[-1]["response"]["usage"]["completion_tokens"] += 1
                else: self.events[-1]["response"]["id"] = "different"
                self.write()
                with self.assertRaises(ValueError): self.recover()
                self.events = old

    def test_incomplete_or_tool_response_not_promoted(self):
        for kind in ("content", "tool", "finish"):
            with self.subTest(kind=kind):
                old = copy.deepcopy(self.events)
                choice = self.events[-1]["response"]["choices"][0]
                if kind == "content": choice["message"]["content"] = ""
                elif kind == "tool": choice["message"]["tool_calls"] = [{"name": "read"}]
                else: choice["finish_reason"] = "length"
                self.write()
                with self.assertRaises(ValueError): self.recover()
                self.events = old

    def test_checkpoint_gap_or_prefix_mismatch_rejected(self):
        self.cp["run"]["total_completion_tokens"] += 1
        self.write()
        with self.assertRaises(ValueError): self.recover()

    def test_earlier_final_cannot_be_replaced(self):
        self.ledger[0]["finish_reason"] = "stop"
        self.events[0]["category"] = "content"
        self.events[0]["response"]["choices"][0]["finish_reason"] = "stop"
        self.events[0]["response"]["choices"][0]["message"]["tool_calls"] = []
        self.write()
        with self.assertRaisesRegex(ValueError, "earlier final"): self.recover()

    def test_no_overwrite_or_other_query(self):
        self.recover()
        with self.assertRaises(ValueError): self.recover()
        with self.assertRaises(ValueError):
            recovery.recover(self.root, self.root / "new.json", expected_manifest_sha256="x", query_id=366)

    def test_malformed_hidden_value_rejected_without_echo(self):
        value = b'{"unread":"PRIVATE_SENTINEL\\q"}'
        with self.assertRaises(ValueError) as error:
            recovery.Projection(value).project({"visible": recovery.SCALAR})
        self.assertNotIn("PRIVATE_SENTINEL", str(error.exception))

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            recovery.Projection(b'{"time":1,"time":2}').project({"time": recovery.SCALAR})


if __name__ == "__main__":
    unittest.main()
