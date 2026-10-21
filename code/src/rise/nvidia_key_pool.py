"""Thread-safe NVIDIA Catalog credentials for OpenAI-compatible calls.

The first ten configured keys serve concurrent worker slots.  The next four
are cold spares for a key that fails authentication.  All keys belong to the
same provider, so a 429 pauses *every* slot and never triggers key rotation.
This prevents an account-wide rate limit from being evaded by switching keys.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from openai import AuthenticationError, RateLimitError


def credentials_from_environment() -> list[str]:
    """Read credentials at runtime, without including values in errors."""
    raw = os.getenv("NVIDIA_API_KEYS", "")
    if raw.strip():
        keys = [part.strip() for part in raw.split(",")]
        if not all(keys):
            raise RuntimeError("NVIDIA_API_KEYS contains an empty entry")
        if len(keys) != len(set(keys)):
            raise RuntimeError("NVIDIA_API_KEYS contains duplicate entries")
        return keys
    single = (os.getenv("NVIDIA_API_KEY") or "").strip()
    return [single] if single else []


def http_client_from_environment() -> Any | None:
    """Optionally cap TCP MSS for NVIDIA connections on affected networks.

    Each OpenAI SDK client receives its own transport. Without the setting,
    the SDK keeps its normal HTTP client and transport behavior.
    """
    raw = (os.getenv("NVIDIA_TCP_MAXSEG") or "").strip()
    if not raw:
        return None
    try:
        maxseg = int(raw)
    except ValueError as exc:
        raise ValueError("NVIDIA_TCP_MAXSEG must be an integer between 256 and 65535") from exc
    if not 256 <= maxseg <= 65535:
        raise ValueError("NVIDIA_TCP_MAXSEG must be an integer between 256 and 65535")
    if not hasattr(socket, "TCP_MAXSEG"):
        raise RuntimeError("NVIDIA_TCP_MAXSEG is not supported on this platform")
    import httpx

    transport = httpx.HTTPTransport(
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_MAXSEG, maxseg)],
    )
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(connect=15.0, write=30.0, read=600.0, pool=15.0),
        follow_redirects=True,
    )


def redact_nvidia_credentials(message: str) -> str:
    """Remove configured API credential values from error text before persistence."""
    raw_values = (os.getenv("NVIDIA_API_KEYS") or "").split(",")
    raw_values.append(os.getenv("NVIDIA_API_KEY") or "")
    raw_values.append(os.getenv("DASHSCOPE_API_KEY") or "")
    for value in raw_values:
        key = value.strip()
        if key:
            message = message.replace(key, "[REDACTED]")
    return message


def _retry_after_seconds(exc: RateLimitError) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return 0.0
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        pass
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, IndexError, OverflowError):
        return 0.0


class _CreateEndpoint:
    def __init__(self, pool: NvidiaKeyPool, path: tuple[str, ...]) -> None:
        self._pool = pool
        self._path = path

    def create(self, **kwargs: Any) -> Any:
        return self._pool._create(self._path, kwargs)


class NvidiaKeyPool:
    """OpenAI-compatible client facade with one stable credential per worker.

    The outer `chat_with_retry`/`responses_with_retry` wrappers still own the
    retry budget.  The SDK clients disable their own hidden retries so each
    429 reaches this pool, which enforces a provider-wide cooldown.
    """

    def __init__(
        self,
        keys: Sequence[str],
        *,
        base_url: str,
        client_factory: Callable[..., Any],
        primary_limit: int = 10,
        spare_limit: int = 4,
        default_rate_limit_delay: float = 5.0,
    ) -> None:
        if not keys:
            raise ValueError("At least one NVIDIA credential is required")
        if primary_limit < 1 or spare_limit < 0:
            raise ValueError("Invalid NVIDIA pool size")
        selected = list(keys[:primary_limit + spare_limit])
        # SDK retries would obscure the first 429 and make cooldown coordination
        # impossible. Existing api_retry wrappers provide the retry budget.
        self._clients = [
            client_factory(api_key=key, base_url=base_url, max_retries=0)
            for key in selected
        ]
        primary_count = min(primary_limit, len(selected))
        self._active = list(range(primary_count))
        self._spares = deque(range(primary_count, len(selected)))
        self._lock = threading.Lock()
        self._local = threading.local()
        self._next_slot = 0
        self._cooldown_until = 0.0
        self._default_rate_limit_delay = default_rate_limit_delay
        self.chat = SimpleNamespace(
            completions=_CreateEndpoint(self, ("chat", "completions"))
        )
        self.responses = _CreateEndpoint(self, ("responses",))

    @property
    def primary_count(self) -> int:
        return len(self._active)

    @property
    def spare_count(self) -> int:
        with self._lock:
            return len(self._spares)

    def _slot_for_thread(self) -> int:
        slot = getattr(self._local, "slot", None)
        if slot is None:
            with self._lock:
                slot = self._next_slot % len(self._active)
                self._next_slot += 1
            self._local.slot = slot
        return slot

    def _wait_for_cooldown(self) -> None:
        while True:
            with self._lock:
                delay = self._cooldown_until - time.monotonic()
            if delay <= 0:
                return
            time.sleep(delay)

    def _start_global_cooldown(self, exc: RateLimitError) -> None:
        delay = max(self._default_rate_limit_delay, _retry_after_seconds(exc))
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)

    def _replace_invalid_key(self, slot: int, failed_index: int) -> bool:
        with self._lock:
            if self._active[slot] != failed_index:
                return True  # another worker sharing this slot already replaced it
            if not self._spares:
                return False
            self._active[slot] = self._spares.popleft()
            return True

    def _create(self, path: tuple[str, ...], kwargs: dict[str, Any]) -> Any:
        slot = self._slot_for_thread()
        while True:
            self._wait_for_cooldown()
            with self._lock:
                index = self._active[slot]
            endpoint = self._clients[index]
            for name in path:
                endpoint = getattr(endpoint, name)
            try:
                return endpoint.create(**kwargs)
            except AuthenticationError:
                if not self._replace_invalid_key(slot, index):
                    # Do not propagate a provider error that could echo a key.
                    raise RuntimeError(
                        "NVIDIA authentication failed and no reserve credential remains"
                    ) from None
            except RateLimitError as exc:
                self._start_global_cooldown(exc)
                raise

    def close(self) -> None:
        for client in self._clients:
            close = getattr(client, "close", None)
            if close is not None:
                close()
