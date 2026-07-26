# memory-agent — Agent's Guide

## What This Is

A local-first, agent-agnostic memory layer for AI agents. It takes raw conversation
turns, extracts atomic facts via an LLM, embeds them with a vector model, stores
them in PostgreSQL + pgvector, and retrieves them via hybrid (semantic + keyword)
search at recall time.

Runs as two surfaces:

- **HTTP sidecar** (`docker compose up` → `localhost:8000`) — FastAPI server
- **MCP stdio server** (`python memory_mcp.py`) — exposes tools for AI agent
  frameworks that speak MCP

## Memory Layers (L0 → L3)

```
Conversation Turn (L0)
    │
    ▼  [LLM extraction, every N turns]
Atomic Fact (L1) ─── stored in `memories` table with pgvector embedding
    │
    ├── [every 10 new L1s] → Scenario Group (L2)
    └── [on demand]        → Persona Summary (L3)
```

| Layer | Table | Purpose | Generated |
|-------|-------|---------|-----------|
| L0 | `conversations` | Raw dialogue turns | Every `POST /add` |
| L1 | `memories` | Extracted atomic facts + embedding | Every N user turns (`EXTRACTION_EVERY_N_TURNS`) |
| L2 | `scenarios` | Grouped related facts | Every 10 new L1 memories |
| L3 | (computed) | User personality summary | On-demand via `GET /persona/{id}` |

## Data Flow (ADD / CAPTURE) — queue, not sync extract

```
Agent/host → POST /add or POST /capture
  ├── 1. Save turns → conversations (L0)
  ├── 2. capture: session checkpoint + batch-hash dedupe (idempotent retry)
  ├── 3. Bump counter; if due → enqueue extraction_jobs (one live job/user)
  └── 4. Return immediately (memories_added=0 on hot path; extract_status=queued|skipped)

Worker (src/worker.py) leases job:
  ├── page L0 since watermark → LLM extract atoms → embed → memories (L1)
  ├── every 10 new L1 → rebuild scenarios (L2)
  └── mark success / failure+retry  (never hold a pool conn across LLM)
```

Host Stop hook should call **`/capture`**, not depend on the model. MCP `store_memories` is fallback (`/add` path).

Implementers: see `../.claude/rules/architecture.md` — **no LLM in request handlers**.

## Data Flow (SEARCH)

```
Agent → POST /search(query)
  ├── 1. memory_trust (stale/lag banner)
  ├── 2. Embed query → vector
  ├── 3. Vector cosine + keyword trgm → RRF fusion
  ├── 4. Priority tilt, recency/conflict demote
  ├── 5. Instruction + scenario fill for remaining top_k slots
  └── 6. Return results + stale + trust  (empty only if l1_count==0)
```

## MCP Tools

| Tool | What It Does | When To Call |
|------|--------------|-------------|
| `search_memories(user_id, query)` | Recall relevant memories | Before responding when history matters |
| `store_memories(user_id, messages)` | Fallback L0 write if host capture missing | Not every turn when Stop→/capture is live |
| `get_persona(user_id)` | User profile summary | Session start |
| `reload_config(model, base_url?)` | Hot-swap LLM (server must allow reload) | Ops only |

## Configuration — LLM & Embedding Providers

Everything is in `.env`. The system is **provider-agnostic** — it uses OpenAI-compatible
HTTP APIs, so any provider that exposes an OpenAI-compatible `/v1` endpoint works.

### Embedding Provider (for vectorizing memories)

```
EMBEDDING_PROVIDER=openai       # "openai" (any OpenAI-compat API) or "local" (ONNX)
EMBEDDING_MODEL=<model name>    # e.g. "text-embedding-3-small"
EMBEDDING_BASE_URL=<API URL>    # e.g. "https://api.openai.com/v1"
EMBEDDING_DIMENSIONS=<int>      # e.g. 1536 for text-embedding-3-small
```

For **local ONNX** embeddings (no external API):
```
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
```

### LLM Provider (for extracting atomic facts from conversations)

```
LLM_MODEL=<model name>          # e.g. "gpt-4o-mini", "claude-sonnet-4-20250514"
LLM_BASE_URL=<API URL>          # e.g. "https://api.openai.com/v1"
LLM_API_KEY=<key>               # API key
LLM_MAX_TOKENS=4096
```

### Example Configurations

| Provider | Embedding | LLM | `.env` changes |
|----------|-----------|-----|----------------|
| **OpenAI** | `text-embedding-3-small` (1536d) | `gpt-4o-mini` | Set `OPENAI_API_KEY`, update `EMBEDDING_BASE_URL`, `LLM_BASE_URL` → `https://api.openai.com/v1`, adjust `EMBEDDING_DIMENSIONS` |
| **Anthropic** | via OpenAI-compat proxy | `claude-sonnet-4-20250514` | Point `LLM_BASE_URL` to an Anthropic proxy (e.g. `https://api.anthropic.com/v1` if they expose OpenAI-compat, or a gateway like `litellm`/`openrouter`) |
| **Groq** | `nomic-embed-text-v1.5` (768d) | `llama-3.3-70b-versatile` | `EMBEDDING_BASE_URL=https://api.groq.com/openai/v1`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `GROQ_API_KEY` as `LLM_API_KEY` |
| **Google Gemini** | `text-embedding-004` (768d) | `gemini-2.0-flash` | Point to Google's OpenAI-compat endpoint or use a gateway |
| **OpenRouter** | varies | varies | `EMBEDDING_BASE_URL=https://openrouter.ai/api/v1`, `LLM_BASE_URL=https://openrouter.ai/api/v1` |
| **Compose default (TEI)** | `sentence-transformers/all-MiniLM-L6-v2` (384d) | from `.env` `LLM_*` | Compose wires `EMBEDDING_BASE_URL=http://embeddings:80/v1` — no host embedder required |
| **Local (LM Studio)** | e.g. nomic-embed-text-v1.5 (768d) | gemma / local chat model | **Optional override** — point `EMBEDDING_*` / `LLM_*` at `http://127.0.0.1:1234/v1` (host) or `http://host.docker.internal:1234/v1` (in-compose app). Dims must match the loaded embed model (often 768 ≠ compose 384) |
| **Ollama** | `nomic-embed-text` (768d) | `llama3.2` | `EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_BASE_URL=http://127.0.0.1:11434/v1` |
| **Local ONNX** | `all-MiniLM-L6-v2` (384d) | any OpenAI-compat | `EMBEDDING_PROVIDER=local`, LLM still via external endpoint |

### Container Considerations

Compose default: app uses the **TEI embeddings sidecar** (`embeddings:80`) with
`sentence-transformers/all-MiniLM-L6-v2` at **384-d**. LLM still comes from
`.env` (`LLM_BASE_URL` / host.docker.internal as needed). To point embeddings at
an external OpenAI-compat API instead:

```yaml
environment:
  EMBEDDING_BASE_URL: https://api.openai.com/v1
  EMBEDDING_MODEL: text-embedding-3-small
  EMBEDDING_DIMENSIONS: "1536"
  LLM_BASE_URL: https://api.openai.com/v1
```

### Docker Compose Override (custom providers)

Create a `docker-compose.override.yml` (gitignored by convention):

```yaml
services:
  app:
    environment:
      EMBEDDING_BASE_URL: https://api.openai.com/v1
      EMBEDDING_MODEL: text-embedding-3-small
      EMBEDDING_DIMENSIONS: 1536
      LLM_BASE_URL: https://api.openai.com/v1
      LLM_MODEL: gpt-4o-mini
```

## Key Constants (tunable in `.env`)

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `EXTRACTION_EVERY_N_TURNS` | `5` | Run LLM extraction every N user turns |
| `EXTRACTION_MAX_MEMORIES` | `20` | Max memories extracted per batch |
| `EXTRACTION_MAX_LAG_SECONDS` | `3600` | L0 newer than last extract beyond this ⇒ recall untrusted (`0` disables) |
| `PERSONA_EVERY_N_MEMORIES` | `50` | Regenerate persona every N new memories |
| `RECALL_STRATEGY` | `hybrid` | `hybrid`, `vector`, or `keyword` |
| `RECALL_RRF_K` | `60` | RRF fusion constant (higher = more rank democracy) |
| `RECALL_SIMILARITY_THRESHOLD` | `0.3` | Minimum vector similarity score |
| `MEMORY_API_SECRET` | empty | Door: HTTP `X-Memory-Key`. Empty = auth off (local tests) |

## MCP Server Connection

The MCP server is exposed inside the Docker container. Connect via:

**opencode.json** (OpenCode):
```json
{
  "mcp": {
    "memory": {
      "type": "local",
      "command": ["docker", "exec", "-i", "memory-agent-app", "python", "memory_mcp.py"],
      "enabled": true
    }
  }
}
```

**.mcp.json** (Cursor / Claude Desktop / other MCP clients):
```json
{
  "mcpServers": {
    "memory": {
      "command": "docker",
      "args": ["exec", "-i", "memory-agent-app", "python", "memory_mcp.py"]
    },
    "codescratch": {
      "command": "codescratch",
      "args": ["mcp"],
      "env": { "CODESCRATCH_ROOT": "${workspaceFolder}" }
    }
  }
}
```

Root `.mcp.json` also wires **codescratch** (sibling code-graph MCP; index at `<repo>/.codescratch/`). Memory ≠ code graph — never mix the two.

## Important Notes for Agents

1. **ADD-only** — memories are never overwritten or deleted. Extraction deduplicates
   by text hash, but existing memories are never modified.
2. **Extraction is batchy** — new conversation turns accumulate in a counter. Only when
   the counter passes `EXTRACTION_EVERY_N_TURNS` does the LLM extract facts. So
   short exchanges may report `memories_added=0` — that's normal.
3. **Trust / lag** — short behind-watermark alone stays trusted until cadence is
   overdue (`extraction_due`) or wall-clock lag exceeds `EXTRACTION_MAX_LAG_SECONDS`
   (`extraction_lag_exceeded`). Banner fields include `behind_watermark`,
   `extraction_lag_exceeded`, `extraction_due`. Untrusted empty ≠ “no prefs.”
4. **Door** — set `MEMORY_API_SECRET`; clients send `X-Memory-Key`. Empty secret = open.
5. **Docker networking** — embeddings default to compose TEI (`http://embeddings:80/v1`).
   LLM reaches the host via `host.docker.internal` when configured that way.
6. **No cloud lock-in** — every component can run fully local (TEI/ONNX + local LLM).
   No telemetry, no external service required.
