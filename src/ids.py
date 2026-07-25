"""Identity helpers shared by HTTP, engine, and MCP."""

from __future__ import annotations


def canonicalize_user_id(user_id: str) -> str:
    """Stable partition key: trim + lowercase. Empty after strip is invalid."""
    uid = (user_id or "").strip().lower()
    if not uid:
        raise ValueError("user_id must be non-empty")
    return uid
