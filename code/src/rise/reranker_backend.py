"""Lazy, independent Qwen3 reranker for baseline v10.

The official yes/no prompt and final-position scoring were checked against
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B on 2026-10-09. Scores are the
yes-minus-no logit difference (the official CrossEncoder default), which ranks
identically to the official Transformers example's two-token probability.

Constructing this object imports no neural dependencies and loads no files.
get_tokenizer() resolves a Hub snapshot and loads only its fast tokenizer;
score() is the first operation that can load weights. Inputs are never silently
truncated and a neural failure never produces lexical or partial scores.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import re
import threading
from typing import Any


QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_LOADING_POLICY = "process_serialized_cold_load_v1"
# Transformers' cold-load contexts can mutate process-global initialization
# state. Distinct question backends must not overlap these contexts; forwards
# remain independent and concurrent after their models have been published.
_MODEL_LOAD_LOCK = threading.RLock()
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


class RerankerError(RuntimeError):
    """Reranking failed explicitly; no replacement scores were returned."""

    def __init__(self, message: str, *, usage: dict[str, Any] | None = None):
        super().__init__(message)
        self.usage = dict(usage or {})


class RerankerUnavailableError(RerankerError):
    """A dependency, tokenizer, model snapshot, or device is unavailable."""


class InputTooLongError(RerankerError):
    """The complete prompt exceeds a configured or native context limit."""

    code = "input_too_long"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_revision(model: str, revision: str, local_files_only: bool) -> str:
    """Resolve movable Hub refs from config.json, including an offline cache."""
    if Path(model).is_dir():
        raise RerankerUnavailableError(
            "Use a Hugging Face repository ID and commit revision; mutable local "
            "model directories cannot pin tokenizer and weights to one snapshot. "
            "Existing Hub snapshots support local_files_only=True."
        )
    if _COMMIT.fullmatch(revision):
        return revision.lower()
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(hf_hub_download(
            repo_id=model, filename="config.json", revision=revision,
            local_files_only=local_files_only,
        ))
        # Keep the cache's snapshots path: resolving its symlink would instead
        # lead into blobs and lose the immutable commit identity.
        index = cached.parts.index("snapshots")
        commit = cached.parts[index + 1]
        if not _COMMIT.fullmatch(commit):
            raise ValueError("Hub cache path does not identify a commit snapshot")
        return commit.lower()
    except Exception as exc:
        raise RerankerUnavailableError(
            f"Cannot resolve {model}@{revision} to a Hub commit: {exc}. "
            "Install requirements-v10.txt and ensure this snapshot is cached "
            "when local_files_only=True."
        ) from exc


def _load_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RerankerUnavailableError(
            "Qwen reranking requires torch; install requirements-v10.txt."
        ) from exc


def _finite_scores(value: Any, expected: int) -> list[float]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().reshape(-1).tolist()
        scores = [float(item) for item in value]
    except Exception as exc:
        raise RerankerError(f"Qwen returned malformed scores: {exc}") from exc
    if len(scores) != expected or any(not math.isfinite(item) for item in scores):
        raise RerankerError(f"Qwen must return {expected} finite scores in input order")
    return scores


class QwenReranker:
    """Pointwise Qwen3 scoring; batched inference preserves passage order."""

    def __init__(
        self, model: str = QWEN_MODEL, revision: str = "main", device: str = "cpu",
        dtype: str = "auto", batch_size: int = 8, max_length: int = 32768,
        local_files_only: bool = False,
    ):
        for name, value in (("model", model), ("revision", revision), ("device", device)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(dtype, str):
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        normalized_dtype = dtype.removeprefix("torch.")
        normalized_dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(
            normalized_dtype, normalized_dtype,
        )
        if normalized_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if not isinstance(local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        self.model_name = model
        self.requested_revision = revision
        self.resolved_revision: str | None = None
        self.device = device
        self.dtype = normalized_dtype
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.max_length = _positive_int(max_length, "max_length")
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._model: Any = None
        self._resolved_device: str | None = None
        self._resolved_dtype: str | None = None
        self.last_usage: dict[str, Any] = self._new_usage(0)

    @staticmethod
    def _new_usage(passages: int) -> dict[str, Any]:
        return {
            "ok": False, "passages": passages, "forward_calls": 0,
            "completed_forward_calls": 0, "input_tokens": 0,
            "input_tokens_attempted": 0, "padded_input_tokens": 0,
            "max_instance_tokens": 0, "sequence_lengths": [], "output_tokens": 0,
        }

    @property
    def identity(self) -> dict[str, Any]:
        """JSON-serializable configuration; reading it never loads a model."""
        return {
            "kind": "qwen_reranker", "model": self.model_name,
            "revision": self.requested_revision,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device, "dtype": self.dtype,
            "resolved_device": self._resolved_device,
            "resolved_dtype": self._resolved_dtype,
            "batch_size": self.batch_size, "max_length": self.max_length,
            "local_files_only": self.local_files_only,
            "loading_policy": RERANKER_LOADING_POLICY,
            "instruction": DEFAULT_INSTRUCTION,
            "template": "qwen3-reranker-official-transformers-v1",
            "score_kind": "yes_minus_no_logit",
            "token_count_kind": "sum_pointwise_contexts",
        }

    def get_tokenizer(self) -> Any:
        """Load a fast tokenizer only, pinning subsequent weights to its commit."""
        if self._tokenizer is not None:
            return self._tokenizer
        if self.resolved_revision is None:
            self.resolved_revision = _resolve_revision(
                self.model_name, self.requested_revision, self.local_files_only,
            )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.resolved_revision, use_fast=True,
                padding_side="left", trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            if not getattr(tokenizer, "is_fast", False):
                raise ValueError("Qwen reranking requires a fast tokenizer")
            tokenizer.padding_side = "left"
            self._tokenizer = tokenizer
            return tokenizer
        except Exception as exc:
            raise RerankerUnavailableError(
                f"Cannot load fast tokenizer {self.model_name}@{self.resolved_revision}: {exc}. "
                "Install requirements-v10.txt; no alternative tokenizer was used."
            ) from exc

    def restore_identity(self, identity: dict[str, Any]) -> None:
        """Lock a checkpoint's resolved commit without loading any resources.

        A saved candidate view uses token boundaries from its original model
        snapshot. Restoring its commit prevents a moving main ref from pairing
        those boundaries with newer tokenizer files or model weights.
        """
        if not isinstance(identity, dict):
            raise ValueError("saved reranker identity must be a dictionary")
        current = self.identity
        requested_keys = (
            "kind", "model", "revision", "requested_revision", "device", "dtype",
            "batch_size", "max_length", "local_files_only", "instruction", "template",
            "score_kind", "token_count_kind",
            "loading_policy",
        )
        for key in requested_keys:
            if key not in identity or identity[key] != current[key]:
                raise ValueError(f"Saved reranker identity does not match requested {key}")
        if "resolved_revision" not in identity:
            raise ValueError("Saved reranker identity must include resolved_revision")
        resolved = identity["resolved_revision"]
        if resolved is None:
            if self.resolved_revision is not None:
                raise ValueError("Cannot replace a pinned reranker identity with an unresolved checkpoint")
            return
        if not isinstance(resolved, str) or not _COMMIT.fullmatch(resolved):
            raise ValueError("Saved resolved_revision must be an immutable 40-character Hub commit")
        resolved = resolved.lower()
        if _COMMIT.fullmatch(self.requested_revision) and resolved != self.requested_revision.lower():
            raise ValueError("Saved resolved_revision does not match the explicitly requested commit")
        if self.resolved_revision is not None and self.resolved_revision != resolved:
            raise ValueError("Cannot change an already pinned reranker snapshot")
        self.resolved_revision = resolved

    def _runtime(self, torch: Any) -> tuple[str, str]:
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype = self.dtype
        if dtype == "auto":
            if device.startswith("cuda"):
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "float16" if device.startswith("mps") else "float32"
        return device, dtype

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with _MODEL_LOAD_LOCK:
            # Another caller of this same backend may have finished loading
            # while this thread was waiting for the process-wide lock.
            if self._model is not None:
                return self._model
            self.get_tokenizer()
            torch = _load_torch()
            try:
                device, dtype = self._runtime(torch)
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, revision=self.resolved_revision,
                    torch_dtype=getattr(torch, dtype), trust_remote_code=False,
                    local_files_only=self.local_files_only,
                ).to(device).eval()
                self._resolved_device = str(model.device)
                self._resolved_dtype = str(getattr(model, "dtype", dtype)).removeprefix("torch.")
                self._model = model
                return model
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"Cannot load Qwen weights {self.model_name}@{self.resolved_revision} "
                    f"on {self.device}/{self.dtype}: {exc}. No fallback scores were produced."
                ) from exc

    def _context_limit(self, tokenizer: Any, model: Any = None) -> int:
        limits = [self.max_length]
        # Hugging Face's enormous unspecified model_max_length sentinel is
        # harmless: min() retains the configured limit.
        for value in (
            getattr(tokenizer, "model_max_length", None),
            getattr(getattr(model, "config", None), "max_position_embeddings", None),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                limits.append(value)
        return min(limits)

    def _check_context(self, limit: int) -> None:
        self.last_usage["context_limit"] = limit
        if self.last_usage["max_instance_tokens"] > limit:
            raise InputTooLongError(
                f"Complete Qwen prompt requires {self.last_usage['max_instance_tokens']} tokens "
                f"but the context limit is {limit}; no input was truncated or forwarded."
            )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one finite raw relevance score per passage, in input order."""
        self.last_usage = self._new_usage(0)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence):
            raise ValueError("passages must be a sequence of strings")
        if any(not isinstance(passage, str) for passage in passages):
            raise ValueError("every passage must be a string")
        self.last_usage["passages"] = len(passages)
        if not passages:
            self.last_usage["ok"] = True
            return []
        try:
            tokenizer = self.get_tokenizer()
            prefix = list(tokenizer.encode(QWEN_PREFIX, add_special_tokens=False))
            suffix = list(tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False))
            sequences = [prefix + list(tokenizer.encode(
                f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: {passage}",
                add_special_tokens=True,
            )) + suffix for passage in passages]
            lengths = [len(sequence) for sequence in sequences]
            self.last_usage.update(
                input_tokens=sum(lengths), max_instance_tokens=max(lengths), sequence_lengths=lengths,
            )
            self._check_context(self._context_limit(tokenizer))
            yes_id, no_id = (
                tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no"),
            )
            if yes_id == no_id or any(
                not isinstance(index, int) or index < 0
                or index == getattr(tokenizer, "unk_token_id", None)
                or list(tokenizer.encode(word, add_special_tokens=False)) != [index]
                for word, index in (("yes", yes_id), ("no", no_id))
            ):
                raise RerankerError("Qwen tokenizer must encode yes and no as distinct single vocabulary tokens")
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if not isinstance(pad_id, int) or pad_id < 0:
                raise RerankerError("Qwen tokenizer requires a padding or EOS token")
            model = self._get_model()
            self._check_context(self._context_limit(tokenizer, model))
            torch = _load_torch()
            scores: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(sequences), self.batch_size):
                    batch = sequences[start:start + self.batch_size]
                    width = max(map(len, batch))
                    # Explicit left padding ensures the final array position is
                    # the actual final prompt token for every passage.
                    ids = [[pad_id] * (width - len(row)) + row for row in batch]
                    masks = [[0] * (width - len(row)) + [1] * len(row) for row in batch]
                    inputs = {
                        "input_ids": torch.tensor(ids, dtype=torch.long, device=model.device),
                        "attention_mask": torch.tensor(masks, dtype=torch.long, device=model.device),
                    }
                    self.last_usage["forward_calls"] += 1
                    self.last_usage["input_tokens_attempted"] += sum(map(len, batch))
                    self.last_usage["padded_input_tokens"] += width * len(batch)
                    # Only the final vocabulary projection is needed. Qwen3 in
                    # the supported Transformers version accepts logits_to_keep.
                    outputs = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                    self.last_usage["completed_forward_calls"] += 1
                    delta = outputs.logits[:, -1, yes_id].float() - outputs.logits[:, -1, no_id].float()
                    scores.extend(_finite_scores(delta, len(batch)))
            scores = _finite_scores(scores, len(passages))
            self.last_usage["ok"] = True
            return scores
        except RerankerError as exc:
            exc.usage = dict(self.last_usage)
            raise
        except Exception as exc:
            raise RerankerError(
                f"Qwen reranking failed; no partial or fallback scores returned: {exc}",
                usage=self.last_usage,
            ) from exc
