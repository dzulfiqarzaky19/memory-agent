# memory-agent

A local-first, agent-agnostic memory layer. Runs as an HTTP sidecar or MCP stdio server, uses PostgreSQL + pgvector.
Any AI agent — Claude Code, Codex CLI, Pi, OpenCode — can store and recall memories by calling a REST API or connecting via the MCP protocol.

## How it works

```
You / Agent  ──POST /add──▶  memory-agent  ──▶  PostgreSQL (pgvector)
                  (messages)      │
                                  ├─ L0: save raw conversation
                                  ├─ L1: LLM extracts atomic facts → embed → store
                                  ├─ L2: group facts into scenarios (every 10 memories)
                                  └─ L3: persona summary (on demand)

You / Agent  ──POST /search──▶  memory-agent  ──▶  return relevant memories
                  (query)         │
                                  ├─ vector similarity (pgvector)
                                  ├─ keyword match (pg_trgm)
                                  └─ RRF fusion → ranked results
```

**Key design: ADD-only.** Memories accumulate, nothing is ever overwritten or deleted. Dedup is by text hash per user.

### Memory Layers

| Layer | What | Where | When |
|-------|------|-------|------|
| L0 | Raw conversation turns | `conversations` table | Every `POST /add` |
| L1 | Extracted atomic facts (e.g. "User prefers dark mode") | `memories` table with pgvector index | Every N turns (default 5) |
| L2 | Grouped scenarios (e.g. "Development Preferences") | `scenarios` table | Every 10 new L1 memories |
| L3 | User persona summary | Computed on demand from L1 | `GET /persona/{id}` |

### Recall strategy (L1)

When you `POST /search`, it uses **hybrid retrieval** by default:

1. **Vector search** — cosine similarity on pgvector embeddings (semantic meaning)
2. **Keyword search** — `pg_trgm` trigram similarity (exact term match)
3. **RRF fusion** — Reciprocal Rank Fusion combines both rankings into one scored list
4. **Scenario boost** — matched L2 scenarios inject their L1 memories at score 0.7

You can switch to pure `vector` or `keyword` via `RECALL_STRATEGY` env var.

## Quick start

### With Docker (recommended)

```bash
# Start both PostgreSQL + memory-agent
docker compose up -d
```

This starts `db` (pgvector on port 5433) and `app` (memory-agent on port 8000).
The app container automatically connects to the database and your local LM Studio
via `host.docker.internal:1234`.

### Without Docker (local dev)

```bash
# 1. Start PostgreSQL + pgvector
docker compose up -d db

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit .env if needed (defaults point to LM Studio at localhost:1234)

# 4. Start the server
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

# 5. Or run the MCP stdio server (alternative to HTTP)
python memory_mcp.py
```

## MCP Integration

memory-agent exposes an MCP stdio server (`memory_mcp.py`) that agents can connect to directly.
It exposes three tools — `search_memories`, `store_memories`, `get_persona` — matching the HTTP API.

### OpenCode

The `opencode.json` in the project root registers the MCP server automatically.
Any agent using OpenCode gets the `memory` MCP tools out of the box.

```json
{
  "mcp": {
    "memory": {
      "type": "local",
      "command": ["python", "memory_mcp.py"],
      "enabled": true
    }
  }
}
```

### Any MCP-compatible agent

Add the server to the agent's MCP config to start using the memory tools directly.

## Configuration

All settings in `.env` (single source of truth). Copy `.env` and edit for your setup:

```bash
# LM Studio (default)
EMBEDDING_BASE_URL=http://127.0.0.1:1234/v1
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=google/gemma-4-e4b

# OpenAI API (faster)
# EMBEDDING_BASE_URL=https://api.openai.com/v1
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:localdev@localhost:5433/memory_agent` | PostgreSQL |
| `EMBEDDING_PROVIDER` | `openai` | `openai` (LM Studio) or `local` |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5@q8_0` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding vector dimensions |
| `EMBEDDING_BASE_URL` | `http://127.0.0.1:1234/v1` | Embedding API endpoint |
| `LLM_PROVIDER` | `openai` | LLM provider for extraction |
| `LLM_MODEL` | `google/gemma-4-e4b` | LLM for extraction |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | LLM API endpoint |
| `LLM_API_KEY` | `not-needed` | LLM API key |
| `LLM_MAX_TOKENS` | `4096` | Max tokens for LLM calls |
| `EXTRACTION_EVERY_N_TURNS` | `5` | Extract every N conversation turns |
| `EXTRACTION_MAX_MEMORIES` | `20` | Max memories per extraction |
| `RECALL_STRATEGY` | `hybrid` | `hybrid`, `vector`, or `keyword` |
| `RECALL_RRF_K` | `60` | RRF fusion constant |
| `RECALL_SIMILARITY_THRESHOLD` | `0.3` | Vector similarity minimum |

## API

```bash
# Store conversation
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"user_id": "zaky", "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'

# Search memories
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"user_id": "zaky", "query": "what tech stack?"}'

# Get persona
curl http://localhost:8000/persona/zaky

# List scenarios
curl http://localhost:8000/scenarios/zaky

# Health check
curl http://localhost:8000/health
```

### Response notes

- `POST /add` returns `memories_added: 0` most of the time — extraction only fires every N turns. Keep calling `/add` for every assistant response; extraction fires when the counter reaches the threshold. Returns `memory_ids` when extraction fires.
- `POST /search` returns ranked results from hybrid vector + keyword retrieval with RRF fusion, boosted by scenario matches.
- `GET /persona/{id}` returns `last_updated` timestamp of the most recent memory used.

## How to integrate with any AI agent

### Via MCP (recommended)

Add this server to the agent's MCP configuration:

```json
{
  "mcp": {
    "memory": {
      "type": "local",
      "command": ["python", "/path/to/memory_mcp.py"],
      "enabled": true
    }
  }
}
```

Then the agent uses `search_memories`, `store_memories`, and `get_persona` natively.

### Via HTTP (no MCP support)

Add this to the agent's CLAUDE.md or system prompt:

```
**Before responding**, recall relevant memories:
curl -s http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"user_id": "zaky", "query": "<summarize the user query>"}'

**After each response**, store the exchange:
curl -s -X POST http://localhost:8000/add -H "Content-Type: application/json" \
  -d '{"user_id": "zaky", "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'

**At session start**, fetch persona:
curl -s http://localhost:8000/persona/zaky
```

No SDK, no library import. The memory-agent is a sidecar — start it once and every agent
shares the same PostgreSQL database. Memories persist across agents, sessions, and contexts.

## Running tests

```bash
# Unit tests (fast — no LLM needed, uses FakeEmbedding)
pytest tests/ -v

# End-to-end test (hits real LM Studio, cleans up after)
python test_e2e.py
```

Tests live under `tests/`:
- `conftest.py` — fixtures: `FakeEmbedding`, `db` (live pg), `engine`, `client`
- `test_memory.py` — memory engine tests (add, search, scenarios, persona edge cases)
- `test_storage.py` — storage layer tests (CRUD, dedup, keyword, hybrid, scenarios)

## Utilities

```bash
# Wipe all data (conversations, memories, scenarios)
python reset_db.py
```

## Architecture

```
src/
├── server.py       # FastAPI HTTP server
├── memory.py       # Core engine: add, search, persona, scenarios
├── storage.py      # PostgreSQL + pgvector backend
├── extraction.py   # LLM fact extraction (supports markdown-wrapped JSON)
├── embeddings.py   # Embedding providers (OpenAI-compatible or local)
├── models.py       # Pydantic request/response schemas
└── config.py       # Reads from .env via python-dotenv

memory_mcp.py       # MCP stdio server
opencode.json       # OpenCode MCP config
reset_db.py         # Data wipe utility
tests/              # Test suite
```
