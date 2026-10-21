"""Opt-in process-shared reranking for the hosted-model pilot.

Install after runtime_guards.install_reranker_hooks, before creating any
question backend. Identical constructor configurations share one lazy backend;
each question receives a lease with private scoring-usage snapshots. A global
reentrant lock serializes tokenizer loading, scoring and identity restoration.
This changes resource scheduling only, not passage selection or scores.
"""
from __future__ import annotations

import copy
import inspect
import re
import threading


SHARED_RERANKER_POLICY = "process_shared_exact_config_serial_score_private_usage_v1"
_RESOURCE_LOCK = threading.RLock()


def _empty_usage(backend):
    initialize = getattr(backend, "_new_usage", None)
    return copy.deepcopy(initialize(0) if callable(initialize) else {})


class SharedRerankerLease:
    """One question's accounting over a serialized, shared model backend."""

    def __init__(self, backend):
        self._backend = backend
        with _RESOURCE_LOCK:
            self._last_usage = _empty_usage(backend)

    @property
    def identity(self):
        with _RESOURCE_LOCK:
            return copy.deepcopy(self._backend.identity)

    @property
    def last_usage(self):
        with _RESOURCE_LOCK:
            return copy.deepcopy(self._last_usage)

    def get_tokenizer(self):
        with _RESOURCE_LOCK:
            return self._backend.get_tokenizer()

    def restore_identity(self, identity):
        with _RESOURCE_LOCK:
            return self._backend.restore_identity(copy.deepcopy(identity))

    def score(self, query, passages):
        with _RESOURCE_LOCK:
            # A runtime guard can fail before the backend's own score method
            # resets usage. Such a failure must not inherit another lease's
            # last successful forward or token counts.
            self._backend.last_usage = _empty_usage(self._backend)
            try:
                return self._backend.score(query, passages)
            finally:
                self._last_usage = copy.deepcopy(self._backend.last_usage)

    def __getattr__(self, name):
        with _RESOURCE_LOCK:
            return getattr(self._backend, name)


def _configuration_key(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # Keep types as well as values: True/1 and 1/1.0 must not collide before
    # the real constructor gets a chance to validate its arguments.
    key = tuple((name, type(value), value) for name, value in bound.arguments.items())
    hash(key)
    return key


def install_shared_reranker_factory():
    """Replace QwenReranker with an idempotent exact-configuration factory.

    The returned callable has the original constructor signature and returns a
    distinct SharedRerankerLease on every invocation. All already-installed
    class-level runtime guards remain on the underlying backend.
    """
    import rise.reranker_backend as module

    with _RESOURCE_LOCK:
        original = module.QwenReranker
        if getattr(original, "_rise_shared_reranker_policy", None) == SHARED_RERANKER_POLICY:
            return original
        signature = inspect.signature(original)
        backends = {}

        def factory(*args, **kwargs):
            key = _configuration_key(signature, args, kwargs)
            with _RESOURCE_LOCK:
                if key not in backends:
                    backend = original(*args, **kwargs)
                    identity = copy.deepcopy(backend.identity)
                    requested = identity.get("requested_revision", identity.get("revision"))
                    if isinstance(requested, str) and re.fullmatch(r"[0-9a-fA-F]{40}", requested):
                        # The hosted pilot uses an immutable commit. Pin it
                        # before even an unscored question saves a checkpoint;
                        # another lease loading the model cannot then make an
                        # earlier unresolved checkpoint incompatible. This
                        # validates identity without loading neural resources.
                        identity["resolved_revision"] = requested.lower()
                        backend.restore_identity(identity)
                    backends[key] = backend
                return SharedRerankerLease(backends[key])

        factory.__name__ = "QwenReranker"
        factory.__doc__ = "Process-shared lazy reranker; returns a private-usage question lease."
        factory.__signature__ = signature
        factory.__wrapped__ = original
        factory._rise_shared_reranker_policy = SHARED_RERANKER_POLICY
        module.QwenReranker = factory
        return factory
