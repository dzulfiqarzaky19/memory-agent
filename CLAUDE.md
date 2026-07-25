# memory-agent

Local-first, agent-agnostic memory layer for AI agents. PostgreSQL + pgvector backend.

## For Other Agents Continuing This Work

This project is an **MCP-connected memory layer** for AI agents. PostgreSQL + pgvector backend.
The `opencode.json` in the project root connects the `memory` MCP server automatically.
You do NOT need to understand the internal codebase — just use the MCP tools.

### Memory Protocol (FOLLOW THIS)

Every agent working here **MUST** follow this protocol using MCP tools:

**At session start**, recall the user persona:
```
Use the get_persona tool with the user's user_id
```

**Before responding to the user**, search for relevant memories:
```
Use the search_memories tool with the user's user_id and a brief query summary
```

**After each exchange**, store the conversation:
```
Use the store_memories tool with the user's user_id and the conversation formatted as:
user: <message>
assistant: <response>
```

### Quick MCP Tool Reference

| Tool | Purpose | Key params |
|------|---------|------------|
| `search_memories` | Recall before responding | `user_id`, `query` |
| `store_memories` | Store after exchange | `user_id`, `messages` |
| `get_persona` | User profile at session start | `user_id` |

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
| `EMBEDDING_PROVIDER` | `openai` | `openai` (LM Studio) or `local` |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5@q8_0` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding vector dimensions |
| `EMBEDDING_BASE_URL` | `http://127.0.0.1:1234/v1` | Embedding API endpoint |
| `LLM_PROVIDER` | `openai` | LLM provider for extraction |
| `LLM_MODEL` | `google/gemma-4-e4b` | LLM for extraction |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | LLM API endpoint |
| `EXTRACTION_EVERY_N_TURNS` | `5` | Extract every N user turns |
| `RECALL_STRATEGY` | `hybrid` | `hybrid`, `vector`, or `keyword` |
| `RECALL_RRF_K` | `60` | RRF fusion constant |
| `RECALL_SIMILARITY_THRESHOLD` | `0.3` | Vector similarity minimum |
