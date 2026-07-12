---
paths:
  - "memory_mcp.py"
  - "src/**"
  - "Dockerfile"
  - "docker-compose.yml"
---

# Docker & MCP server (memory-agent)

- After editing `memory_mcp.py` (or any `src/` file), run `docker compose up -d --build app` before retesting — it runs via `docker exec -i memory-agent-app python memory_mcp.py` off the baked image with NO bind mount, so host edits are invisible until rebuilt.
- Give LLM-backed MCP tools (e.g. `get_persona`) a long httpx timeout (120s, matching `store_memories`) — NOT the default short timeout (10s used for `search_memories`), which is fine for embedding-only calls but times out client-side on reasoning-model calls before the server (which keeps processing) responds.
