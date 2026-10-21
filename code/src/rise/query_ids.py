"""Safe query identifiers for per-query workspace and trace paths."""
from __future__ import annotations

import re

_SAFE_QUERY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def safe_query_id(value: object) -> str:
    """Return a filename-safe ID or reject it before any path is constructed."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("query_id must be a string or integer")
    text = str(value)
    if not _SAFE_QUERY_ID.fullmatch(text):
        raise ValueError("query_id must be 1-128 ASCII letters, digits, underscores or hyphens, starting with a letter or digit")
    return text
