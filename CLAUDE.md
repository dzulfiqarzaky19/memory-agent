# memory-agent

Local-first, agent-agnostic memory layer for AI agents. PostgreSQL + pgvector backend.

## For Other Agents Continuing This Work

This project is an **MCP-connected memory layer** for AI agents. PostgreSQL + pgvector backend.
The root `.mcp.json` / `opencode.json` connect **two independent** MCP servers:
- `memory` — user/session facts (this repo, Docker stdio)
- `codescratch` — code structure graph for the open workspace (sibling project `D:/dev/projects/codescratch`, no shared state)

You do NOT need to understand either internal codebase — use the MCP tools.

### Memory Protocol (FOLLOW THIS)

**L0 durability** is host-driven when the Claude Code plugin is installed
(`integrations/claude-code` Stop hook → `POST /capture`). Do not rely on the
model remembering to store turns for capture.

Still use MCP for **recall**:

**At session start**, recall the user persona:
```
Use the get_persona tool with the user's user_id
```

**Before responding to the user**, search for relevant memories:
```
Use the search_memories tool with the user's user_id and a brief query summary
```

**Manual store** (other hosts / plugin missing) via MCP:
```
Use the store_memories tool with the user's user_id and the conversation formatted as:
user: <message>
assistant: <response>
```

### Quick MCP Tool Reference

**Memory** (user/session — `user_id=zaky` lowercase):

| Tool | Purpose | Key params |
|------|---------|------------|
| `search_memories` | Recall before responding | `user_id`, `query` |
| `store_memories` | Manual store if auto-capture unavailable | `user_id`, `messages` |
| `get_persona` | User profile at session start | `user_id` |

HTTP auto-capture: `POST /capture` with `user_id`, `session_key`, `messages`.

**Code structure** (this workspace’s TS/JS/Python graph — separate process):

| Tool | Purpose |
|------|---------|
| `cs_status` | Trust / stale / exact reindex command |
| `cs_search` | Find symbol by name |
| `cs_explore` | One symbol/file: callers, calls, imports, bindings |
| `cs_callers` / `cs_callees` | Call edges |
| `cs_impact` | Blast radius (`direction=up\|down\|both`) |
| `cs_reindex` | Incremental reindex when trust=stale |

Prefer `cs_*` over blind grep for navigation. Memory ≠ code graph — never store code structure into memories or vice versa.

Prerequisite once per machine: `cd D:/dev/projects/codescratch && npm run build`.  
First open of a repo: `cs_reindex` with `full=true` or CLI `node dist/cli.js init <repo>`.

### Key Principles (do not violate)

- **ADD-only extraction** — memories accumulate, nothing is ever overwritten or deleted
- **Store wide, recall narrow** — capture everything, retrieve precisely
- **Agent-agnostic** — HTTP sidecar or MCP stdio, no SDK dependency, no vendor lock-in
- **Local-first** — PostgreSQL + pgvector, no cloud API required by default

## Architecture

```
src/
├── server.py          # FastAPI HTTP server (the sidecar)
├── memory.py          # Core memory engine (add/search/persona)
├── storage.py         # PostgreSQL + pgvector backend
├── extraction.py      # LLM fact extraction (ADD-only, no UPDATE)
├── embeddings.py      # Pluggable embedding providers
├── models.py          # Pydantic models / schemas
└── config.py          # Reads from .env via python-dotenv

memory_mcp.py          # MCP stdio server (for agent-native integration)
opencode.json          # OpenCode MCP server config
reset_db.py            # Wipe all data utility
tests/
├── conftest.py        # Fixtures (FakeEmbedding, db, engine, client)
├── test_memory.py     # Memory engine tests
└── test_storage.py    # Storage layer tests
```

## Memory Layers (L0 → L3)

Inspired by TencentDB Agent Memory's progressive disclosure:

- **L0 Conversation** — raw dialogue turns (stored in `conversations` table)
- **L1 Atom** — atomic facts extracted from conversations (stored in `memories` table with pgvector)
- **L2 Scenario** — grouped facts into contextual scenes (stored in `scenarios` table, rebuilt every 10 new memories)
- **L3 Persona** — user profile summary (computed on-demand from L1 atoms)

## Recall Strategy

When you `POST /search`, uses **hybrid retrieval** by default:

1. **Vector search** — cosine similarity on pgvector embeddings (semantic)
2. **Keyword search** — `pg_trgm` trigram similarity (exact term match)
3. **RRF fusion** — Reciprocal Rank Fusion combines both rankings
4. **Scenario / instruction fill** — unmatched L2-linked and instruction memories fill remaining `top_k` slots after matched results

Switch via `RECALL_STRATEGY` env var: `hybrid`, `vector`, or `keyword`.

## API

```
POST /add          — store conversation turns (greedy capture)
POST /search       — targeted recall (semantic + keyword fusion)
GET  /persona/{id} — user persona summary
GET  /scenarios/{id} — list L2 scenario groups
GET  /health
```

## Tech Stack

- Python 3.12+ + FastAPI
- PostgreSQL 16 + pgvector (via docker-compose)
- Configurable embeddings (local ONNX or OpenAI-compatible)
- Configurable LLM (OpenAI-compatible endpoint)
- MCP stdio server (via `mcp` package)

## Dev Commands

```bash
# Start everything
docker compose up -d

# Run MCP server (alternative to HTTP API)
python memory_mcp.py

# Run tests
pytest tests/ -v
python test_e2e.py

# Wipe all data
python reset_db.py
```

## Configuration

All settings in `.env` (single source of truth). Copy `.env` and edit for your setup.

### Key variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:localdev@localhost:5433/memory_agent` | PostgreSQL |
| `EMBEDDING_PROVIDER` | `openai` | `openai` (OpenAI-compat / TEI) or `local` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (compose TEI) |
| `EMBEDDING_DIMENSIONS` | `384` | Embedding vector dimensions (MiniLM) |
| `EMBEDDING_BASE_URL` | falls back to `OPENAI_BASE_URL` (`http://127.0.0.1:1234/v1`); compose: `http://embeddings:80/v1` | Embedding API endpoint |
| `LLM_PROVIDER` | (from `.env`) | LLM provider for extraction |
| `LLM_MODEL` | (from `.env`) | LLM for extraction |
| `LLM_BASE_URL` | (from `.env`) | LLM API endpoint |
| `EXTRACTION_EVERY_N_TURNS` | `5` | Extract every N user turns |
| `EXTRACTION_MAX_LAG_SECONDS` | `3600` | Wall-clock lag before recall untrusted (`0` off) |
| `RECALL_STRATEGY` | `hybrid` | `hybrid`, `vector`, or `keyword` |
| `RECALL_RRF_K` | `60` | RRF fusion constant |
| `RECALL_SIMILARITY_THRESHOLD` | `0.3` | Vector similarity minimum |
| `MEMORY_API_SECRET` | empty | Door: HTTP `X-Memory-Key`; empty = open |
