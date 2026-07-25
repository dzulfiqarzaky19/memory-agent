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

## Data Flow (ADD)

```
Agent → POST /add(messages)
  ├── 1. Save each turn → conversations (L0)
  ├── 2. Bump conversational counter
  ├── 3. Counter >= EXTRACTION_EVERY_N_TURNS?
  │     └── YES →
  │           a. LLM extracts atomic facts from pending conversations
  │           b. Embedder generates vector for each fact
  │           c. Store in memories table (L1)
  │           d. Every 10 L1s → rebuild scenarios (L2)
  └── 4. Return { memories_added, memory_ids }
```

## Data Flow (SEARCH)

```
Agent → POST /search(query)
  ├── 1. Embed query → vector
  ├── 2. Vector cosine search (semantic)
  ├── 3. Keyword trigram search (exact match)
  ├── 4. RRF fusion of both rankings
  ├── 5. Scenario / instruction fill: unmatched L2-linked + instructions fill remaining top_k
  ├── 6. (instructions ordered first among injects, then scenario-linked)
  └── 7. Return ranked results
```

## MCP Tools

| Tool | What It Does | When To Call |
|------|--------------|-------------|
| `search_memories(user_id, query)` | Recall relevant memories | Before responding to user |
| `store_memories(user_id, messages)` | Store exchange after replying | After each assistant response |
| `get_persona(user_id)` | Get user profile summary | At session start |
| `compact_context(user_id, messages, ...)` | Compress long context with memory injection | When approaching context limit |

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
| **Local (LM Studio)** | nomic-embed-text-v1.5 (768d) | gemma-4-e4b | `http://127.0.0.1:1234/v1` (default — works out of box) |
| **Ollama** | `nomic-embed-text` (768d) | `llama3.2` | `EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_BASE_URL=http://127.0.0.1:11434/v1` |
| **Local ONNX** | `all-MiniLM-L6-v2` (384d) | any OpenAI-compat | `EMBEDDING_PROVIDER=local`, LLM still via external endpoint |

### Container Considerations

The Docker container runs inside Docker networking. By default it connects to
`host.docker.internal:1234/v1` for both LLM and embeddings. If your provider
is **external** (OpenAI, Groq, etc.), override in `docker-compose.yml`:

```yaml
environment:
  EMBEDDING_BASE_URL: https://api.openai.com/v1
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
| `PERSONA_EVERY_N_MEMORIES` | `50` | Regenerate persona every N new memories |
| `RECALL_STRATEGY` | `hybrid` | `hybrid`, `vector`, or `keyword` |
| `RECALL_RRF_K` | `60` | RRF fusion constant (higher = more rank democracy) |
| `RECALL_SIMILARITY_THRESHOLD` | `0.3` | Minimum vector similarity score |

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
    }
  }
}
```

## Important Notes for Agents

1. **ADD-only** — memories are never overwritten or deleted. Extraction deduplicates
   by text hash, but existing memories are never modified.
2. **Extraction is batchy** — new conversation turns accumulate in a counter. Only when
   the counter passes `EXTRACTION_EVERY_N_TURNS` does the LLM extract facts. So
   short exchanges may report `memories_added=0` — that's normal.
3. **Docker networking** — the app container reaches the host via `host.docker.internal`.
   If you change LLM/embedding URLs, make sure the container can reach them.
4. **No cloud lock-in** — every component can run fully local (ONNX + local LLM).
   No telemetry, no external service required.
