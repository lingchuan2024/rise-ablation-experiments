"""Offline tests for concurrent key assignment and provider-wide backoff."""
from __future__ import annotations

import importlib
import os
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeAuth(Exception):
    pass


class FakeRate(Exception):
    def __init__(self, retry_after: str):
        self.response = types.SimpleNamespace(headers={"retry-after": retry_after})
        super().__init__("rate limited")


def fake_factory(callback, seen):
    def build(*, api_key, base_url, max_retries):
        seen.append((api_key, base_url, max_retries))

        def create(**kwargs):
            return callback(api_key, kwargs)

        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
            responses=types.SimpleNamespace(create=create),
        )
    return build


class NvidiaKeyPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The test can also run under a bare Python without the OpenAI SDK.
        try:
            importlib.import_module("openai")
        except ImportError:
            stub = types.ModuleType("openai")
            stub.AuthenticationError = FakeAuth
            stub.RateLimitError = FakeRate
            module_patch = patch.dict(sys.modules, {"openai": stub})
            module_patch.start()
            cls.addClassCleanup(module_patch.stop)

    def setUp(self):
        self.module = importlib.import_module("rise.nvidia_key_pool")

    def test_runtime_keys_preferred_and_reject_bad_lists_without_echoing_values(self):
        with patch.dict(os.environ, {
            "NVIDIA_API_KEYS": "  k0 , k1  ", "NVIDIA_API_KEY": "fallback"
        }, clear=True):
            self.assertEqual(self.module.credentials_from_environment(), ["k0", "k1"])
            self.assertEqual(
                self.module.redact_nvidia_credentials("failed k0, k1 and fallback"),
                "failed [REDACTED], [REDACTED] and [REDACTED]",
            )
        with patch.dict(os.environ, {"NVIDIA_API_KEYS": "private-token,"}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                self.module.credentials_from_environment()
            self.assertNotIn("private-token", str(caught.exception))
        with patch.dict(os.environ, {"NVIDIA_API_KEYS": "secret,secret"}, clear=True):
            with self.assertRaises(RuntimeError) as caught:
                self.module.credentials_from_environment()
            self.assertNotIn("secret", str(caught.exception))

    def test_token_plan_key_is_redacted_from_errors(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-private-qwen-key"}, clear=True):
            self.assertEqual(
                self.module.redact_nvidia_credentials("failed test-private-qwen-key"),
                "failed [REDACTED]",
            )

    def test_ten_threads_get_distinct_primary_keys_and_only_four_spares_exist(self):
        created = []
        pool = self.module.NvidiaKeyPool(
            [f"k{i}" for i in range(19)], base_url="https://example.invalid/v1",
            client_factory=fake_factory(lambda key, kwargs: key, created),
        )
        self.assertEqual(pool.primary_count, 10)
        self.assertEqual(pool.spare_count, 4)
        self.assertEqual([item[0] for item in created], [f"k{i}" for i in range(14)])
        self.assertTrue(all(item[2] == 0 for item in created))
        barrier = threading.Barrier(10)

        def worker(_):
            barrier.wait(timeout=3)
            return pool.chat.completions.create(model="nvidia/test")

        with ThreadPoolExecutor(max_workers=10) as executor:
            used = list(executor.map(worker, range(10)))
        self.assertEqual(set(used), {f"k{i}" for i in range(10)})
        self.assertEqual(pool.spare_count, 4)

    def test_auth_failure_replaces_only_failed_key_and_never_exposes_it(self):
        created = []

        def callback(key, kwargs):
            if key == "private-key-0":
                raise FakeAuth("private-key-0 authentication denied")
            return key

        with patch.object(self.module, "AuthenticationError", FakeAuth):
            pool = self.module.NvidiaKeyPool(
                ["private-key-0", "backup-key"], base_url="https://example.invalid/v1",
                client_factory=fake_factory(callback, created), primary_limit=1,
            )
            self.assertEqual(pool.chat.completions.create(model="nvidia/test"), "backup-key")
            self.assertEqual(pool.chat.completions.create(model="nvidia/test"), "backup-key")
            self.assertEqual(pool.spare_count, 0)
            no_backup = self.module.NvidiaKeyPool(
                ["private-key-0"], base_url="https://example.invalid/v1",
                client_factory=fake_factory(callback, []),
            )
            with self.assertRaises(RuntimeError) as caught:
                no_backup.chat.completions.create(model="nvidia/test")
            self.assertNotIn("private-key-0", str(caught.exception))

    def test_rate_limit_waits_retry_after_globally_without_rotating(self):
        created = []
        calls = []
        now = [100.0]

        def callback(key, kwargs):
            calls.append(key)
            if len(calls) == 1:
                raise FakeRate("7")
            return key

        def sleep(seconds):
            now[0] += seconds

        with patch.object(self.module, "RateLimitError", FakeRate), \
             patch.object(self.module.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(self.module.time, "sleep", side_effect=sleep):
            pool = self.module.NvidiaKeyPool(
                ["k0", "k1"], base_url="https://example.invalid/v1",
                client_factory=fake_factory(callback, created), primary_limit=1,
                default_rate_limit_delay=5,
            )
            with self.assertRaises(FakeRate):
                pool.chat.completions.create(model="nvidia/test")
            self.assertEqual(pool.chat.completions.create(model="nvidia/test"), "k0")
        self.assertEqual(calls, ["k0", "k0"])
        self.assertEqual(now[0], 107.0)
        self.assertEqual(pool.spare_count, 1)

    def test_429_from_one_thread_cools_down_another_primary_key(self):
        now = [100.0]
        calls = []

        def callback(key, kwargs):
            calls.append((key, now[0]))
            if key == "k0":
                raise FakeRate("7")
            return key

        def sleep(seconds):
            now[0] += seconds

        with patch.object(self.module, "RateLimitError", FakeRate), \
             patch.object(self.module.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(self.module.time, "sleep", side_effect=sleep):
            pool = self.module.NvidiaKeyPool(
                ["k0", "k1"], base_url="https://example.invalid/v1",
                client_factory=fake_factory(callback, []), primary_limit=2,
            )
            with self.assertRaises(FakeRate):
                pool.chat.completions.create(model="nvidia/test")
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(
                    executor.submit(pool.chat.completions.create, model="nvidia/test").result(),
                    "k1",
                )
        self.assertEqual(calls, [("k0", 100.0), ("k1", 107.0)])


if __name__ == "__main__":
    unittest.main()
