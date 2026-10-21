"""One shared model, serialized forwards, per-question usage and real guards."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "server"))

import rise.reranker_backend as backend_module
from shared_reranker import install_shared_reranker_factory, SHARED_RERANKER_POLICY


class FakeBackend:
    instances = []
    active = 0
    max_active = 0
    activity_lock = threading.Lock()

    def __init__(self, model="fixture/model", revision="fixed", device="cpu", dtype="auto",
                 batch_size=8, max_length=32768, local_files_only=False):
        self.model_name, self.device, self.revision = model, device, revision
        self.dtype, self.batch_size, self.max_length = dtype, batch_size, max_length
        self.local_files_only = local_files_only
        self._identity = {"model": model, "revision": revision, "device": device,
                          "requested_revision": revision, "resolved_revision": None,
                          "nested": {"commit": "fixed"}}
        self.tokenizer = object()
        self.tokenizer_calls = 0
        self.last_usage = self._new_usage(0)
        self.calls = []
        FakeBackend.instances.append(self)

    @staticmethod
    def _new_usage(passages):
        return {"input_tokens": 0, "forward_calls": 0, "passages": passages,
                "sequence_lengths": []}

    @property
    def identity(self):
        return self._identity

    def _enter(self):
        with FakeBackend.activity_lock:
            FakeBackend.active += 1
            FakeBackend.max_active = max(FakeBackend.max_active, FakeBackend.active)

    def _leave(self):
        with FakeBackend.activity_lock:
            FakeBackend.active -= 1

    def get_tokenizer(self):
        self._enter()
        try:
            self.tokenizer_calls += 1
            time.sleep(0.005)
            return self.tokenizer
        finally:
            self._leave()

    def restore_identity(self, identity):
        self._enter()
        try:
            time.sleep(0.005)
            if identity["model"] != self.model_name:
                raise ValueError("identity differs")
            self._identity = copy.deepcopy(identity)
        finally:
            self._leave()

    def score(self, query, passages):
        self._enter()
        try:
            self.last_usage = {"input_tokens": len(query) + sum(map(len, passages)),
                               "forward_calls": 1, "passages": len(passages),
                               "sequence_lengths": list(map(len, passages))}
            time.sleep(0.01)
            self.calls.append((query, list(passages)))
            if query == "fail":
                raise RuntimeError("fixture score failure")
            return [float(len(query) + len(text)) for text in passages]
        finally:
            self._leave()


class SharedRerankerTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances = []
        FakeBackend.active = FakeBackend.max_active = 0
        self.replace = patch.object(backend_module, "QwenReranker", FakeBackend)
        self.replace.start()
        self.addCleanup(self.replace.stop)

    def test_concurrent_factory_calls_share_one_model_with_private_usage(self):
        factory = install_shared_reranker_factory()
        with ThreadPoolExecutor(max_workers=3) as executor:
            leases = list(executor.map(lambda _: factory(model="same", device="cuda:0"), range(3)))
        self.assertEqual(len(FakeBackend.instances), 1)
        self.assertEqual(len({id(lease) for lease in leases}), 3)
        tasks = [(leases[i], "query" + str(i), ["passage" * (i + 1)]) for i in range(3)]
        with ThreadPoolExecutor(max_workers=3) as executor:
            scores = list(executor.map(lambda task: task[0].score(task[1], task[2]), tasks))
        for (lease, query, passages), result in zip(tasks, scores):
            self.assertEqual(result, [float(len(query) + len(text)) for text in passages])
            self.assertEqual(lease.last_usage["input_tokens"], len(query) + sum(map(len, passages)))
            self.assertEqual(lease.last_usage["sequence_lengths"], list(map(len, passages)))
        self.assertEqual(FakeBackend.max_active, 1)
        saved = leases[0].last_usage
        saved["sequence_lengths"].append(999)
        self.assertNotIn(999, leases[0].last_usage["sequence_lengths"])

    def test_tokenizer_identity_restore_and_scores_are_globally_serialized(self):
        factory = install_shared_reranker_factory()
        first, second = factory(device="cuda:0"), factory(device="cuda:1")
        identity = first.identity
        identity["nested"]["commit"] = "restored"
        with ThreadPoolExecutor(max_workers=3) as executor:
            operations = [executor.submit(first.get_tokenizer),
                          executor.submit(first.restore_identity, identity),
                          executor.submit(second.score, "query", ["passage"])]
            tokenizer = operations[0].result()
            for operation in operations[1:]:
                operation.result()
        self.assertIs(tokenizer, FakeBackend.instances[0].tokenizer)
        self.assertEqual(FakeBackend.max_active, 1)
        self.assertEqual(first.identity["nested"]["commit"], "restored")
        identity["nested"]["commit"] = "modified by caller"
        self.assertEqual(first.identity["nested"]["commit"], "restored")

    def test_exception_usage_is_per_lease_and_the_original_exception_propagates(self):
        factory = install_shared_reranker_factory()
        success, failure = factory(), factory()
        success.score("successful", ["evidence"])
        before = success.last_usage
        with self.assertRaisesRegex(RuntimeError, "fixture score failure"):
            failure.score("fail", ["partial", "usage"])
        self.assertEqual(failure.last_usage["input_tokens"], len("failpartialusage"))
        self.assertEqual(success.last_usage, before)
        fresh = factory()
        self.assertEqual(fresh.last_usage["input_tokens"], 0)
        self.assertEqual(fresh.last_usage["forward_calls"], 0)

    def test_original_class_guards_survive_and_pre_call_failure_has_zero_usage(self):
        original = FakeBackend.score
        guard_calls = []

        def guarded(self, query, passages):
            guard_calls.append(query)
            if query == "paused":
                raise RuntimeError("fixture infrastructure pause")
            return original(self, query, passages)

        with patch.object(FakeBackend, "score", guarded):
            factory = install_shared_reranker_factory()
            first, second = factory(), factory()
            first.score("success", ["evidence"])
            with self.assertRaisesRegex(RuntimeError, "infrastructure pause"):
                second.score("paused", ["never scored"])
        self.assertEqual(guard_calls, ["success", "paused"])
        self.assertGreater(first.last_usage["input_tokens"], 0)
        self.assertEqual(second.last_usage["input_tokens"], 0)
        self.assertEqual(second.last_usage["forward_calls"], 0)

    def test_exact_constructor_configuration_and_idempotent_installation(self):
        factory = install_shared_reranker_factory()
        same_a = factory()
        same_b = factory(model="fixture/model", revision="fixed", device="cpu", dtype="auto",
                         batch_size=8, max_length=32768, local_files_only=False)
        self.assertEqual(len(FakeBackend.instances), 1)
        self.assertIs(same_a.get_tokenizer(), same_b.get_tokenizer())
        factory(device="cuda:0")
        factory(revision="different")
        factory(batch_size=4)
        self.assertEqual(len(FakeBackend.instances), 4)
        self.assertIs(install_shared_reranker_factory(), factory)
        self.assertEqual(factory._rise_shared_reranker_policy, SHARED_RERANKER_POLICY)
        self.assertEqual(same_a.model_name, "fixture/model")

    def test_immutable_revision_is_pinned_before_any_lease_checkpoint(self):
        factory = install_shared_reranker_factory()
        revision = "6E9E69830B95C52B5FD889B7690DDA3329508DE3"
        first, second = factory(revision=revision), factory(revision=revision)
        self.assertEqual(len(FakeBackend.instances), 1)
        self.assertEqual(first.identity["resolved_revision"], revision.lower())
        self.assertEqual(second.identity["resolved_revision"], revision.lower())
        self.assertEqual(FakeBackend.instances[0].tokenizer_calls, 0)
        self.assertEqual(FakeBackend.instances[0].calls, [])
        first.restore_identity(first.identity)
        self.assertEqual(second.identity["resolved_revision"], revision.lower())
        mutable = factory(revision="main")
        self.assertIsNone(mutable.identity["resolved_revision"])


if __name__ == "__main__":
    unittest.main()
