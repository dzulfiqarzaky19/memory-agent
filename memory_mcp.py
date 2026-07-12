"""MCP server exposing memory tools for AI agents.

Run: python memory_mcp.py
Connect via opencode.json as stdio MCP server.
"""
from mcp.server import FastMCP
import httpx

MCP = FastMCP("memory-agent")
API_BASE = "http://localhost:8000"


@MCP.tool()
async def search_memories(user_id: str, query: str) -> str:
    """Recall relevant memories before responding to the user."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{API_BASE}/search", json={"user_id": user_id, "query": query, "top_k": 5})
        r.raise_for_status()
        data = r.json()
    if not data["results"]:
        return "No relevant memories found."
    lines = [f"- [{m['score']:.4f}] {m['text']}" for m in data["results"]]
    return "Relevant memories:\n" + "\n".join(lines)


@MCP.tool()
async def store_memories(user_id: str, messages: str) -> str:
    """Store the conversation after responding.

    Pass messages as alternating 'user: ...' and 'assistant: ...' lines.
    """
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
        r = await c.post(f"{API_BASE}/add", json={"user_id": user_id, "messages": parsed})
        r.raise_for_status()
        data = r.json()
    return f"Stored {len(parsed)} messages. Memories extracted: {data['memories_added']}."


@MCP.tool()
async def get_persona(user_id: str) -> str:
    """Get the user's persona summary."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{API_BASE}/persona/{user_id}")
        r.raise_for_status()
        data = r.json()
    if data["memory_count"] == 0:
        return "No memories stored yet."
    return f"[{data['memory_count']} memories] {data['summary']}"


if __name__ == "__main__":
    MCP.run(transport="stdio")
