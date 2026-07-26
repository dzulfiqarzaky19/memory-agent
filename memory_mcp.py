"""MCP server exposing memory tools for AI agents.

Run: python memory_mcp.py
Connect via opencode.json as stdio MCP server.
"""
import os
import sys
from pathlib import Path

import httpx
from mcp.server import FastMCP

# Prefer package path used in Docker (PYTHONPATH=/app/src); fall back to repo layout.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from ids import canonicalize_user_id  # noqa: E402

MCP = FastMCP("memory-agent")
API_BASE = (os.getenv("MEMORY_AGENT_URL") or "http://127.0.0.1:8000").rstrip("/")
_API_SECRET = (os.getenv("MEMORY_API_SECRET") or "").strip()


def _headers() -> dict[str, str]:
    if not _API_SECRET:
        return {}
    return {"X-Memory-Key": _API_SECRET}


def _trust_banner(trust: dict | None) -> str:
    if not trust:
        return ""
    ok = trust.get("last_extract_ok")
    ok_s = "true" if ok is True else ("false" if ok is False else "unknown")
    return (
        f"[trust user={trust.get('user_id')} l0={trust.get('l0_count')} l1={trust.get('l1_count')} "
        f"extract_ok={ok_s} pending={trust.get('conversations_seen')} "
        f"behind_watermark={trust.get('behind_watermark')} "
        f"extraction_lag_exceeded={trust.get('extraction_lag_exceeded')} "
        f"extraction_due={trust.get('extraction_due')} "
        f"stale_seconds={trust.get('stale_seconds')} "
        f"recall_trusted={trust.get('recall_trusted')}]"
    )


def _stale_note(data: dict) -> str:
    """Loud staleness line — results are real but may lag the newest turns."""
    if not data.get("stale"):
        return ""
    secs = int(data.get("stale_seconds") or 0)
    if secs >= 3600:
        age = f"~{secs // 3600}h"
    elif secs >= 60:
        age = f"~{secs // 60}m"
    else:
        age = f"{secs}s"
    return (
        f"WARNING stale recall (behind by {age}) — extraction is pending or failed. "
        "Results below are real but may omit the most recent turns; "
        "do not treat absence as proof the user never said it."
    )


@MCP.tool()
async def search_memories(user_id: str, query: str) -> str:
    """Recall relevant memories before responding to the user."""
    uid = canonicalize_user_id(user_id)
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"{API_BASE}/search",
            json={"user_id": uid, "query": query, "top_k": 5},
            headers=_headers(),
        )
        r.raise_for_status()
        data = r.json()
    banner = _trust_banner(data.get("trust"))
    note = _stale_note(data)
    if not data["results"]:
        empty = (
            "No memories match — and recall is stale, so this is NOT proof the user has no prefs."
            if data.get("stale")
            else "No relevant memories found."
        )
        return f"{banner}\n{note}\n{empty}".strip()
    lines = [
        f"- [{m['score']:.4f}] ({m.get('type') or 'memory'}) {m['text']}"
        for m in data["results"]
    ]
    body = "Relevant memories:\n" + "\n".join(lines)
    return f"{banner}\n{note}\n{body}".strip()


@MCP.tool()
async def store_memories(user_id: str, messages: str) -> str:
    """Store the conversation after responding.

    Pass messages as alternating 'user: ...' and 'assistant: ...' lines.
    """
    uid = canonicalize_user_id(user_id)
    lines = [l.strip() for l in messages.strip().split("\n") if l.strip()]
    parsed = []
    for line in lines:
        if line.startswith("user:"):
            parsed.append({"role": "user", "content": line[5:].strip()})
        elif line.startswith("assistant:"):
            parsed.append({"role": "assistant", "content": line[10:].strip()})
    if not parsed:
        return "No valid messages found. Format: user: ... / assistant: ..."

    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{API_BASE}/add",
            json={"user_id": uid, "messages": parsed},
            headers=_headers(),
        )
        r.raise_for_status()
        data = r.json()
    return (
        f"Stored {len(parsed)} messages for {data.get('user_id', uid)}. "
        f"Memories extracted: {data['memories_added']} "
        f"(extract_status={data.get('extract_status', '?')})."
    )


@MCP.tool()
async def get_persona(user_id: str) -> str:
    """Get the user's persona summary."""
    uid = canonicalize_user_id(user_id)
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.get(f"{API_BASE}/persona/{uid}", headers=_headers())
        r.raise_for_status()
        data = r.json()
    banner = _trust_banner(data.get("trust"))
    note = _stale_note(data)
    if data["memory_count"] == 0:
        return f"{banner}\n{note}\nNo memories stored yet.".strip()
    return f"{banner}\n{note}\n[{data['memory_count']} memories] {data['summary']}".strip()


@MCP.tool()
async def reload_config(model: str, base_url: str = "") -> str:
    """Hot-swap the LLM model/config (server must set MEMORY_ALLOW_RELOAD=1)."""
    body = {"model": model}
    if base_url:
        body["base_url"] = base_url
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{API_BASE}/reload", json=body, headers=_headers())
        if r.status_code == 404:
            return "Reload disabled on server (MEMORY_ALLOW_RELOAD not set)."
        r.raise_for_status()
        data = r.json()
    return f"Switched to model={data['model']} at {data['base_url']}"


if __name__ == "__main__":
    MCP.run(transport="stdio")
