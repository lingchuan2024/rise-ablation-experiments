"""Qwen contract tests without neural packages, weights, or network access."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rise.reranker_backend import (
    DEFAULT_INSTRUCTION, InputTooLongError, QWEN_PREFIX, QWEN_SUFFIX,
    QwenReranker, RerankerError, RerankerUnavailableError,
)

COMMIT = "a" * 40


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, selectors):
        if not isinstance(selectors, tuple):
            selectors = (selectors,)

        def select(value, parts):
            if not parts:
                return value
            first, *rest = parts
            if isinstance(first, slice):
                return [select(item, rest) for item in value[first]]
            return select(value[first], rest)

        return FakeTensor(select(self.data, selectors))

    def __sub__(self, other):
        return FakeTensor([left - right for left, right in zip(self.data, other.data)])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        assert shape == (-1,)
        return self

    def tolist(self):
        return self.data


class FakeTokenizer:
    is_fast = True
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 999
    model_max_length = 32768

    def __init__(self):
        self.encoded = []

    def encode(self, text, *, add_special_tokens):
        # Deliberately accept no truncation option. Full middle text is retained.
        self.encoded.append((text, add_special_tokens))
        if text == QWEN_PREFIX:
            return [2, 3]
        if text == QWEN_SUFFIX:
            return [4, 5, 6]
        if text == "yes":
            return [7]
        if text == "no":
            return [8]
        return [ord(character) + 100 for character in text]

    def convert_tokens_to_ids(self, token):
        return {"yes": 7, "no": 8}[token]


class FakeModel:
    device = "cpu"
    dtype = "torch.float32"

    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=32768)
        self.calls = []
        self.raise_on_call = None
        self.bad_score = None
        self.bad_count = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_call == len(self.calls):
            raise RuntimeError("test forward failed")
        logits = []
        for row in kwargs["input_ids"].data:
            # The last passage character lies directly before the suffix.
            delta = float(row[-4] - 100 - ord("a") + 1)
            if self.bad_score is not None:
                delta = self.bad_score
            wrong_position = [0.0] * 10
            wrong_position[7] = -999.0
            final_position = [0.0] * 10
            final_position[7] = 10.0 + delta
            final_position[8] = 10.0
            logits.append([wrong_position, final_position])
        if self.bad_count:
            logits = logits[:1]
        return SimpleNamespace(logits=FakeTensor(logits))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.hub = ModuleType("huggingface_hub")
        self.hub.hf_hub_download = MagicMock(
            return_value=f"/fake/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/{COMMIT}/config.json",
        )
        self.transformers = ModuleType("transformers")
        self.transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=MagicMock(return_value=self.tokenizer),
        )
        self.transformers.AutoModelForCausalLM = SimpleNamespace(
            from_pretrained=MagicMock(return_value=self.model),
        )
        self.torch = ModuleType("torch")
        self.torch.long = "long"
        self.torch.float32 = "float32"
        self.torch.float16 = "float16"
        self.torch.bfloat16 = "bfloat16"
        self.torch.tensor = MagicMock(side_effect=lambda data, **kwargs: FakeTensor(data))
        self.torch.inference_mode = MagicMock(side_effect=nullcontext)
        self.torch.cuda = SimpleNamespace(
            is_available=lambda: False, is_bf16_supported=lambda: False,
        )
        self.torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        self.modules = patch.dict(sys.modules, {
            "huggingface_hub": self.hub, "transformers": self.transformers, "torch": self.torch,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_constructor_identity_and_empty_score_are_lazy(self):
        backend = QwenReranker()
        self.assertIsNone(backend.identity["resolved_revision"])
        json.dumps(backend.identity)
        self.assertEqual(backend.score("query", []), [])
        self.hub.hf_hub_download.assert_not_called()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.torch.tensor.assert_not_called()
        self.assertEqual(backend.last_usage["forward_calls"], 0)
        self.assertTrue(backend.last_usage["ok"])

    def test_tokenizer_then_weights_use_same_resolved_offline_snapshot(self):
        backend = QwenReranker(local_files_only=True)
        self.assertIs(backend.get_tokenizer(), self.tokenizer)
        self.assertIs(backend.get_tokenizer(), self.tokenizer)
        self.hub.hf_hub_download.assert_called_once_with(
            repo_id="Qwen/Qwen3-Reranker-0.6B", filename="config.json", revision="main",
            local_files_only=True,
        )
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-Reranker-0.6B", revision=COMMIT, use_fast=True,
            padding_side="left", trust_remote_code=False, local_files_only=True,
        )
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertEqual(backend.identity["resolved_revision"], COMMIT)
        self.assertEqual(backend.score("query", ["a"]), [1.0])
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-Reranker-0.6B", revision=COMMIT, torch_dtype="float32",
            trust_remote_code=False, local_files_only=True,
        )
        self.assertEqual(backend.identity["resolved_device"], "cpu")
        self.assertEqual(backend.identity["resolved_dtype"], "float32")
        json.dumps(backend.identity)

    def test_commit_revision_does_not_need_hub_config_fetch(self):
        backend = QwenReranker(revision=COMMIT.upper(), local_files_only=True)
        backend.get_tokenizer()
        self.hub.hf_hub_download.assert_not_called()
        self.assertEqual(backend.identity["resolved_revision"], COMMIT)

    def test_restore_identity_pins_checkpoint_commit_without_loading(self):
        backend = QwenReranker(local_files_only=True)
        saved = {**backend.identity, "resolved_revision": COMMIT}
        backend.restore_identity(saved)
        self.hub.hf_hub_download.assert_not_called()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertEqual(backend.identity["resolved_revision"], COMMIT)
        backend.get_tokenizer()
        self.hub.hf_hub_download.assert_not_called()
        self.assertEqual(self.transformers.AutoTokenizer.from_pretrained.call_args.kwargs["revision"], COMMIT)
        backend.score("q", ["a"])
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["revision"], COMMIT)

    def test_restore_identity_rejects_different_config_and_invalid_commits(self):
        backend = QwenReranker()
        for key, changed in (
            ("model", "Qwen/other"), ("requested_revision", "other"),
            ("max_length", 4096), ("batch_size", 4), ("device", "cuda"),
            ("dtype", "float16"), ("resolved_revision", "main"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                backend.restore_identity({**backend.identity, key: changed})
        saved = {**backend.identity, "resolved_revision": COMMIT}
        backend.restore_identity(saved)
        with self.assertRaisesRegex(ValueError, "already pinned"):
            backend.restore_identity({**saved, "resolved_revision": "b" * 40})
        with self.assertRaisesRegex(ValueError, "unresolved checkpoint"):
            backend.restore_identity({**saved, "resolved_revision": None})
        self.hub.hf_hub_download.assert_not_called()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()

    def test_restore_identity_respects_explicit_requested_commit(self):
        backend = QwenReranker(revision=COMMIT)
        with self.assertRaisesRegex(ValueError, "explicitly requested commit"):
            backend.restore_identity({**backend.identity, "resolved_revision": "b" * 40})

    def test_restore_unresolved_identity_preserves_lazy_construction(self):
        backend = QwenReranker()
        self.assertEqual(backend.max_length, 32768)
        backend.restore_identity(backend.identity)
        self.assertIsNone(backend.identity["resolved_revision"])
        self.hub.hf_hub_download.assert_not_called()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()

    def test_official_prompt_left_padding_final_logits_and_input_order(self):
        backend = QwenReranker(batch_size=2)
        self.assertEqual(backend.score("same query", ["a", "ccc", "bb"]), [1.0, 3.0, 2.0])
        self.assertEqual(self.tokenizer.padding_side, "left")
        self.assertIn((QWEN_PREFIX, False), self.tokenizer.encoded)
        self.assertIn((QWEN_SUFFIX, False), self.tokenizer.encoded)
        for passage in ("a", "ccc", "bb"):
            self.assertIn((
                f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: same query\n<Document>: {passage}", True,
            ), self.tokenizer.encoded)
        first = self.model.calls[0]
        self.assertEqual(first["input_ids"].data[0][:2], [0, 0])
        self.assertEqual(first["attention_mask"].data[0][:3], [0, 0, 1])
        self.assertEqual(first["attention_mask"].data[1], [1] * len(first["input_ids"].data[1]))
        self.assertEqual(first["input_ids"].data[0][-3:], [4, 5, 6])
        self.assertEqual(first["input_ids"].data[1][-3:], [4, 5, 6])
        self.assertFalse(first["use_cache"])
        self.assertEqual(first["logits_to_keep"], 1)
        usage = backend.last_usage
        self.assertEqual(usage["forward_calls"], 2)
        self.assertEqual(usage["completed_forward_calls"], 2)
        self.assertEqual(usage["input_tokens"], sum(usage["sequence_lengths"]))
        self.assertEqual(usage["input_tokens_attempted"], usage["input_tokens"])
        self.assertEqual(usage["padded_input_tokens"], usage["input_tokens"] + 2)
        self.assertTrue(usage["ok"])

    def test_usage_resets_between_scores_and_weights_load_only_once(self):
        backend = QwenReranker(batch_size=1)
        backend.score("q", ["a", "b"])
        self.assertEqual(backend.last_usage["forward_calls"], 2)
        backend.score("q", ["c"])
        self.assertEqual(backend.last_usage["forward_calls"], 1)
        self.assertEqual(backend.last_usage["passages"], 1)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_configured_limit_refuses_full_input_before_loading_weights(self):
        backend = QwenReranker(max_length=10)
        with self.assertRaises(InputTooLongError) as caught:
            backend.score("query", ["a" * 50])
        self.assertEqual(backend.last_usage["forward_calls"], 0)
        self.assertGreater(backend.last_usage["input_tokens"], 10)
        self.assertEqual(caught.exception.usage, backend.last_usage)
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertTrue(any("a" * 50 in text for text, _ in self.tokenizer.encoded))

    def test_tokenizer_native_limit_refuses_before_weights(self):
        self.tokenizer.model_max_length = 10
        backend = QwenReranker()
        with self.assertRaises(InputTooLongError):
            backend.score("q", ["a"])
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_model_native_limit_refuses_all_forwards(self):
        self.model.config.max_position_embeddings = 10
        backend = QwenReranker()
        with self.assertRaises(InputTooLongError):
            backend.score("q", ["a"])
        self.assertEqual(backend.last_usage["forward_calls"], 0)
        self.assertEqual(self.model.calls, [])

    def test_missing_or_slow_tokenizer_is_explicit(self):
        self.tokenizer.is_fast = False
        with self.assertRaisesRegex(RerankerUnavailableError, "fast tokenizer"):
            QwenReranker().get_tokenizer()
        self.transformers.AutoTokenizer.from_pretrained.side_effect = OSError("not cached")
        with self.assertRaisesRegex(RerankerUnavailableError, "not cached"):
            QwenReranker(local_files_only=True).get_tokenizer()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_unresolved_snapshot_is_explicit_and_never_loads_tokenizer(self):
        self.hub.hf_hub_download.side_effect = OSError("offline cache miss")
        with self.assertRaisesRegex(RerankerUnavailableError, "offline cache miss"):
            QwenReranker(local_files_only=True).get_tokenizer()
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_malformed_snapshot_path_is_rejected(self):
        self.hub.hf_hub_download.return_value = "/fake/cache/config.json"
        with self.assertRaises(RerankerUnavailableError):
            QwenReranker().get_tokenizer()

    def test_tokenizer_yes_no_must_be_distinct_single_tokens(self):
        self.tokenizer.convert_tokens_to_ids = lambda word: 7
        with self.assertRaisesRegex(RerankerError, "distinct single"):
            QwenReranker().score("q", ["a"])
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_padding_uses_eos_when_pad_is_missing(self):
        self.tokenizer.pad_token_id = None
        backend = QwenReranker()
        backend.score("q", ["a", "cc"])
        self.assertEqual(self.model.calls[0]["input_ids"].data[0][0], self.tokenizer.eos_token_id)

    def test_missing_padding_and_eos_is_explicit_before_weights(self):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(RerankerError, "padding or EOS"):
            QwenReranker().score("q", ["a"])
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_weight_loading_failure_has_no_fallback(self):
        self.transformers.AutoModelForCausalLM.from_pretrained.side_effect = OSError("weights missing")
        backend = QwenReranker(local_files_only=True)
        with self.assertRaisesRegex(RerankerUnavailableError, "weights missing"):
            backend.score("q", ["a"])
        self.assertFalse(backend.last_usage["ok"])
        self.assertEqual(backend.last_usage["forward_calls"], 0)

    def test_forward_failure_returns_no_partial_scores_and_retains_usage(self):
        self.model.raise_on_call = 2
        backend = QwenReranker(batch_size=1)
        with self.assertRaisesRegex(RerankerError, "no partial or fallback") as caught:
            backend.score("q", ["a", "b", "c"])
        self.assertEqual(backend.last_usage["forward_calls"], 2)
        self.assertEqual(backend.last_usage["completed_forward_calls"], 1)
        self.assertFalse(backend.last_usage["ok"])
        self.assertEqual(caught.exception.usage, backend.last_usage)

    def test_nonfinite_and_incorrect_score_counts_are_explicit(self):
        for bad_score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=bad_score):
                self.model.bad_score = bad_score
                with self.assertRaisesRegex(RerankerError, "finite scores"):
                    QwenReranker().score("q", ["a"])
        self.model.bad_score = None
        self.model.bad_count = True
        with self.assertRaisesRegex(RerankerError, "2 finite scores"):
            QwenReranker().score("q", ["a", "b"])

    def test_invalid_public_inputs_are_rejected_without_loading(self):
        for kwargs in (
            {"batch_size": 0}, {"max_length": True}, {"dtype": "int8"},
            {"revision": ""}, {"device": ""}, {"local_files_only": "true"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QwenReranker(**kwargs)
        backend = QwenReranker()
        for query, passages in (("", []), ("q", "passage"), ("q", [1]), ("q", iter(["a"]))):
            with self.subTest(query=query, passages=passages), self.assertRaises(ValueError):
                backend.score(query, passages)
        self.hub.hf_hub_download.assert_not_called()

    def test_auto_device_and_dtype_alias_are_recorded(self):
        backend = QwenReranker(device="auto", dtype="fp32")
        backend.score("q", ["a"])
        self.assertEqual(backend.identity["dtype"], "float32")
        self.assertEqual(backend.identity["resolved_device"], "cpu")
        self.assertEqual(backend.identity["resolved_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
