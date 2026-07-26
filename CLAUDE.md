# memory-agent

Local-first, agent-agnostic **work-partner** memory (Postgres + pgvector). HTTP sidecar + MCP.

Not a code graph → sibling **codescratch**. Not companion/brainify.

## Two modes (read the right doc)

| You are… | Read |
|----------|------|
| **Using** memory (recall/store in any session) | Protocol + MCP table below · `.claude/rules/memory.md` |
| **Implementing / reviewing this repo** | `.claude/rules/architecture.md` first · then `database.md` / `docker.md` · use `cs_*` on `src/` |

Root `.mcp.json` / `opencode.json` wire two **independent** MCPs: `memory` (this) + `codescratch` (no shared state).

## Memory protocol (clients)

**L0 durability** — host Stop hook → `POST /capture` when Claude plugin installed. Do not rely on the model for capture.

**Recall (MCP)** — still model/host driven:

- Session start: `get_persona` · `user_id=zaky` (lowercase)
- Before answers that need history: `search_memories`
- Manual store only if capture missing: `store_memories` with `user:\nassistant:` turns

Read trust/`stale` on replies — empty + untrusted ≠ “user has no prefs.”

| Tool | Purpose |
|------|---------|
| `search_memories` | Recall |
| `store_memories` | Fallback L0 write via `/add` path |
| `get_persona` | L3 summary |

HTTP: `POST /capture` · `POST /search` · `GET /persona/{id}` · `GET /health`.

## Code structure (sibling)

Prefer `cs_*` over blind grep for navigation in **this** tree when editing it.

| Tool | Purpose |
|------|---------|
| `cs_status` / `cs_search` / `cs_explore` | Map |
| `cs_callers` / `cs_callees` / `cs_impact` | Edges / blast radius |
| Host `codescratch ensure` | Freshness — not routine agent `cs_reindex` |

See `.claude/rules/codescratch.md`. Never put code-graph facts into memories or the reverse.

## Principles

- **Agent-agnostic** — HTTP or MCP; no vendor SDK lock-in
- **Local-first** — Postgres + pgvector; embeddings/LLM configurable
- **Fast writes** — capture/add enqueue extraction; worker + LLM off request path
- **Recall degrades** — serve stale L1 with banner rather than amnesia
- **Work partner** — prefs, decisions, instructions; thin craft continuity — not entertainment persona

## Layout (pointer)

```
src/server.py memory.py storage.py worker.py extraction.py
src/migrations.py embeddings.py models.py config.py ids.py
memory_mcp.py
tests/   # single-process pytest against shared DB
integrations/claude-code/   # Stop → /capture
```

Truth for paths/invariants: **`.claude/rules/architecture.md`**.

## Dev

```bash
docker compose up -d
curl -s http://127.0.0.1:8000/health
pytest tests/ -v          # one process
# MCP: python memory_mcp.py  (or docker exec app)
```

Config: `.env`. Door: `MEMORY_API_SECRET` → `X-Memory-Key`. Compose publishes `127.0.0.1:8000` / `5433`.
