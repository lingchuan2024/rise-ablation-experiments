"""Observe loader contention without neural packages, weights, or network.

The fake dependency behaves like a process-global model loader: overlapping
initialization, transfer, or evaluation calls are observable. Tests also force
simultaneous forwards so a broad inference lock cannot pass unnoticed.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reranker_backend import COMMIT, FakeModel, FakeTensor, FakeTokenizer
from rise.reranker_backend import (
    QwenReranker, RERANKER_LOADING_POLICY, RerankerUnavailableError,
)


class LoadActivity:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = Counter()

    @contextmanager
    def observe(self, phase):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls[phase] += 1
        try:
            # Yield long enough for a competing worker to enter an unsafe
            # loader. No barrier is inside loading: serialization must work.
            time.sleep(0.03)
            yield
        finally:
            with self.lock:
                self.active -= 1


class LoadingConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.activity = LoadActivity()
        self.failure_stage = None
        self.forward_barrier = None
        self.forward_active = 0
        self.forward_peak = 0
        self.models = []
        owner = self

        class ObservedModel(FakeModel):
            def __init__(self, attempt):
                super().__init__()
                self.attempt = attempt

            def to(self, device):
                with owner.activity.observe("to"):
                    if self.attempt == 1 and owner.failure_stage == "to":
                        raise RuntimeError("simulated transfer failure")
                    return super().to(device)

            def eval(self):
                with owner.activity.observe("eval"):
                    if self.attempt == 1 and owner.failure_stage == "eval":
                        raise RuntimeError("simulated evaluation failure")
                    return super().eval()

            def __call__(self, **kwargs):
                with owner.activity.lock:
                    owner.forward_active += 1
                    owner.forward_peak = max(owner.forward_peak, owner.forward_active)
                try:
                    if owner.forward_barrier is not None:
                        owner.forward_barrier.wait(timeout=5)
                    return super().__call__(**kwargs)
                finally:
                    with owner.activity.lock:
                        owner.forward_active -= 1

        def load_tokenizer(*args, **kwargs):
            with self.activity.observe("tokenizer"):
                return FakeTokenizer()

        def load_weights(*args, **kwargs):
            with self.activity.observe("weights"):
                with self.activity.lock:
                    attempt = self.activity.calls["weights"]
                if attempt == 1 and self.failure_stage == "weights":
                    raise OSError("simulated weight loading failure")
                model = ObservedModel(attempt)
                self.models.append(model)
                return model

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_weights)
        torch = ModuleType("torch")
        torch.long = "long"
        torch.float32 = "float32"
        torch.tensor = lambda data, **kwargs: FakeTensor(data)
        torch.inference_mode = nullcontext
        self.modules = patch.dict(sys.modules, {"transformers": transformers, "torch": torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    @staticmethod
    def backend():
        return QwenReranker(revision=COMMIT, device="cpu", dtype="float32", local_files_only=True)

    @staticmethod
    def simultaneous_loads(backends):
        start = threading.Barrier(len(backends))

        def load(backend):
            start.wait(timeout=5)
            try:
                return backend._get_model()
            except RerankerUnavailableError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=len(backends)) as executor:
            futures = [executor.submit(load, backend) for backend in backends]
            return [future.result(timeout=10) for future in futures]

    def test_distinct_backends_never_overlap_cold_loading_or_transfer(self):
        first, second = self.backend(), self.backend()
        models = self.simultaneous_loads([first, second])
        self.assertIs(models[0], first._model)
        self.assertIs(models[1], second._model)
        self.assertIsNot(models[0], models[1])
        self.assertEqual(self.activity.peak, 1)
        self.assertEqual(self.activity.calls, {"tokenizer": 2, "weights": 2, "to": 2, "eval": 2})
        self.assertEqual(self.activity.active, 0)

    def test_same_backend_loads_once_when_two_callers_start_together(self):
        backend = self.backend()
        models = self.simultaneous_loads([backend, backend])
        self.assertIs(models[0], models[1])
        self.assertIs(models[0], backend._model)
        self.assertEqual(self.activity.calls, {"tokenizer": 1, "weights": 1, "to": 1, "eval": 1})
        self.assertEqual(self.activity.peak, 1)

    def test_failed_cold_load_releases_lock_and_publishes_only_complete_model(self):
        for stage in ("weights", "to", "eval"):
            with self.subTest(stage=stage):
                self.activity = LoadActivity()
                self.failure_stage = stage
                backend = self.backend()
                outcomes = self.simultaneous_loads([backend, backend])
                failures = [value for value in outcomes if isinstance(value, RerankerUnavailableError)]
                successes = [value for value in outcomes if not isinstance(value, RerankerUnavailableError)]
                self.assertEqual(len(failures), 1)
                self.assertEqual(len(successes), 1)
                self.assertIs(successes[0], backend._model)
                self.assertEqual(backend._model.attempt, 2)
                self.assertIs(backend._get_model(), successes[0])
                self.assertEqual(self.activity.calls["weights"], 2)
                self.assertEqual(self.activity.active, 0)
                self.assertEqual(self.activity.peak, 1)

    def test_warm_backends_can_forward_at_the_same_time(self):
        backends = [self.backend(), self.backend()]
        for backend in backends:
            backend._get_model()
        self.forward_barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(backend.score, "query", ["a"]) for backend in backends]
            scores = [future.result(timeout=10) for future in futures]
        self.assertEqual(scores, [[1.0], [1.0]])
        self.assertEqual(self.forward_peak, 2)
        self.assertEqual(self.activity.calls["weights"], 2)
        for backend in backends:
            self.assertTrue(backend.last_usage["ok"])
            self.assertEqual(backend.last_usage["completed_forward_calls"], 1)

    def test_loading_policy_is_serializable_and_required_for_restore(self):
        backend = self.backend()
        self.assertEqual(backend.identity["loading_policy"], RERANKER_LOADING_POLICY)
        backend.restore_identity(backend.identity)
        for changed in ({key: value for key, value in backend.identity.items() if key != "loading_policy"},
                        {**backend.identity, "loading_policy": "unserialized_cold_load"}):
            with self.assertRaisesRegex(ValueError, "loading_policy"):
                backend.restore_identity(changed)
        self.assertEqual(self.activity.calls, {})


if __name__ == "__main__":
    unittest.main()
