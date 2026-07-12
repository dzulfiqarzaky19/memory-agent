# mem0 — Implementation Notes

Repo: [mem0ai/mem0](https://github.com/mem0ai/mem0) (Apache-2.0). Python. Verified against README + direct reads of `mem0/memory/main.py`, `mem0/configs/prompts.py`, `mem0/configs/base.py`, `mem0/vector_stores/pgvector.py`, `server/main.py`, and directory listings for `mem0/memory`, `mem0/`, `server/`, `mem0/embeddings`, `mem0/llms`, `mem0/vector_stores`. `docs.mem0.ai` was **not** fetched, so anything docs-only (vs. source-verified) is unconfirmed. Claims not confirmed against source are marked **[unverified]**.

## 1. Architecture

No progressive-disclosure layer concept (no L0–L3 equivalent). `Memory` / `AsyncMemory` classes expose a flat API: `add / search / get / get_all / update / delete / delete_all / history / reset`. Scoping is done via `user_id` / `agent_id` / `run_id` parameters rather than distinct memory-type classes. The `add()` pipeline internally runs as an 8-phase batch flow (fetch existing candidates → extract → dedup/decide → write → embed → index → history-log → return).

**Important gap vs. common perception**: no `graphs/` module and no `graph_store` field on `MemoryConfig` exist in the current source tree (`mem0/memory/graph_memory.py` returns 404). If mem0's marketing/README references Neo4j-backed "graph memory," treat that as **not present in current `main`** rather than a live feature.

## 2. Data model / storage schema

pgvector table is deliberately thin/schemaless:

```
id       UUID
vector   vector(dims)
payload  JSONB
```

Everything else — `user_id`, `agent_id`, `run_id`, timestamps, content hash, lemmatized text for keyword search — lives inside `payload`, not as typed columns. Indexes: optional DiskANN/HNSW on the vector column, plus an always-on GIN index over `to_tsvector('simple', payload->>'text_lemmatized')` for keyword/BM25-style search.

25 vector-store backends (pgvector, Qdrant, Chroma, Pinecone, Redis, Milvus, etc.) implement one common interface — schemaless JSONB is used across all of them, trading strict schema/queryability for backend portability.

## 3. Extraction / ingestion pipeline

`mem0/configs/prompts.py` contains three separate prompt systems:

- `USER_MEMORY_EXTRACTION_PROMPT` / `AGENT_MEMORY_EXTRACTION_PROMPT` — source-restricted fact extraction.
- `DEFAULT_UPDATE_MEMORY_PROMPT` — the classic ADD/UPDATE/DELETE/NONE tool-calling decision mem0 is widely known for.
- `ADDITIVE_EXTRACTION_PROMPT` — a newer, ADD-only prompt.

**Key finding**: the live `add()` code path currently calls `ADDITIVE_EXTRACTION_PROMPT`, not the ADD/UPDATE/DELETE tool-calling prompt. Dedup in the actual pipeline is **MD5 exact-hash matching**, not LLM-driven conflict resolution.

This is a real discrepancy between mem0's historically-advertised architecture (the ADD/UPDATE/DELETE tool-calling story cited widely in blog posts and comparisons) and what currently ships in `main`. Don't cite the tool-calling UPDATE/DELETE mechanism as mem0's current behavior without re-verifying against the latest source — it appears to have been superseded by simpler additive+hash-dedup logic, at least in the code path read here.

## 4. Retrieval / recall strategy

```
search(query, top_k, filters, threshold, rerank, explain, reference_date, show_expired)
```

- Hybrid retrieval = vector similarity + keyword (GIN lemmatized-text index) + entity-boost scoring, combined via `score_and_rank()`.
- Filtering by `user_id` / `agent_id` / `run_id` and arbitrary `filters`.
- Reranking is pluggable (`RerankerConfig`) but **off by default**.
- TTL/expiration support: `expiration_date`, `show_expired`, `reference_date` params exist in the search signature — not documented in the README **[worth independently re-confirming if relied upon]**.

## 5. Persona / profile generation

**Confirmed absent.** No analog to an L2 scenario or L3 persona/summary layer. The closest adjacent feature is `POST /generate-instructions` (server API), which returns `custom_instructions` for prompt tuning — not a computed, retrievable user profile. This is a genuine architectural differentiator vs. this project (`memory-agent` has both L2 scenarios and L3 persona).

## 6. API surface

**Python SDK**: `add / search / get / get_all / update / delete / delete_all / history / reset` on `Memory` / `AsyncMemory`.

**REST server** (`server/main.py`): a full FastAPI application — notably heavier than a bare sidecar:
- `alembic/` migrations (real schema versioning, unlike this project's simpler setup).
- `dashboard/` (a UI, not just an API).
- `auth.py` — API-key auth with admin-gated endpoints.
- `/generate-instructions` endpoint (see §5).

## 7. Tech stack, providers, config

- **Embedding providers**: 12 (openai, ollama, lmstudio, huggingface, gemini, etc.)
- **LLM providers**: 18 (anthropic, openai, ollama, lmstudio, vllm, groq, etc.)
- **Vector store backends**: 25

`MemoryConfig` fields confirmed in `mem0/configs/base.py`:
```
vector_store
llm
embedder
history_db_path
reranker
version              default "v1.1"
custom_instructions
```
No `graph_store` field exists. Default-model claims (specific GPT/embedding model defaults) were **not independently verified**.

## 8. Notable design decisions / tradeoffs vs. this project

- **Schemaless JSONB payload** across all 25 backends — flexibility/portability over strict schema and typed queries (this project uses typed Postgres tables via pgvector directly).
- **Entity-linking as a secondary vector-indexed boost**, not a confirmed true graph/Neo4j traversal — despite "graph memory" being a commonly cited mem0 feature, it is not present in the current source tree read here.
- **Batch-first, single-LLM-call extraction** trades cost/latency for loss of semantic conflict resolution: dedup is hash-only (MD5 exact match) in the current `add()` path, not the LLM ADD/UPDATE/DELETE tool-calling decision that's widely cited as mem0's signature mechanism.
- **TTL/expiration support** (`expiration_date`, `show_expired`, `reference_date`) is a feature this project's ADD-only design currently lacks — worth considering if memory staleness/expiry becomes a requirement.
- **No persona/L3 equivalent** — this project's L3 persona layer (and L2 scenario grouping) is a genuine differentiator, not something mem0 replicates.
- **Heavier server footprint**: mem0's REST server ships Alembic migrations, a dashboard, and admin-gated auth — more infrastructure than this project's FastAPI sidecar, at the cost of more moving parts to self-host.

**Caveats**: the two most commonly cited mem0 features in public discourse — graph memory (Neo4j) and the ADD/UPDATE/DELETE tool-calling extraction mechanism — were **not found live** in the current `main` branch source read here. Treat both as unconfirmed/possibly deprecated rather than current fact, and re-verify against `docs.mem0.ai` and the latest commit before relying on them in a public comparison.
