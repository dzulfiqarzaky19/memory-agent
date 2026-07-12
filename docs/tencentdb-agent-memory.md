# TencentDB Agent Memory — Implementation Notes

Repo: [TencentCloud/TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) (MIT). TypeScript, Node.js ≥22.16. Verified against README + direct reads of `src/core/store/sqlite.ts`, `src/core/store/types.ts`, `src/core/store/search-utils.ts`, `src/core/prompts/l1-extraction.ts`, `src/core/prompts/l1-dedup.ts`, `src/core/prompts/persona-generation.ts`, `src/config.ts`. Claims not confirmed against source are marked **[unverified]**.

## 1. Architecture

Four-tier "semantic pyramid," built bottom-up, read top-down:

- **L0 Conversation** — raw dialogue turns, one row per message.
- **L1 Atom** — atomic facts extracted from L0 (`src/core/record/l1-extractor.ts`).
- **L2 Scenario** — scene blocks grouping related L1 atoms, stored as Markdown (`src/core/scene/scene-extractor.ts`).
- **L3 Persona** — single synthesized profile document, `persona.md` (`src/core/persona/persona-generator.ts`).

Storage is deliberately heterogeneous: L0/L1 (evidence) live in a relational+vector DB for retrieval; L2/L3 (structure) persist as human-readable Markdown for inspection/traceability. Traceability chain: Persona → Scenario → Atom → Conversation via `node_id`/`result_ref` pointers — an agent reads the compact top layer and drills into raw evidence on demand ("progressive disclosure").

A separate subsystem, **symbolic memory / context offloading** (`src/offload/*`), externalizes verbose tool-call logs to `refs/*.md` and injects a compressed Mermaid graph into live context instead of raw text — claims up to 61.38% token reduction on WideSearch **[unverified, README claim only]**.

## 2. Data model / storage schema

Default backend: **`node:sqlite`** (`DatabaseSync`) + **`sqlite-vec`** + **FTS5**. Alternate: Tencent Cloud VectorDB (TCVDB), behind the same `IMemoryStore` interface.

| Table | Purpose | Key columns |
|---|---|---|
| `embedding_meta` | tracks embedding provider/model/dims for reindex detection | `key PK, value` |
| `l1_records` | L1 fact metadata | `record_id PK, content, type, priority INT DEFAULT 50, scene_name, session_key, session_id, timestamp_str, timestamp_start, timestamp_end, created_time, updated_time, metadata_json` |
| `l1_vec` | L1 vector index (vec0 virtual table, cosine) | `record_id PK, embedding float[N], updated_time` |
| `l0_conversations` | raw turns | `record_id PK, session_key, session_id, role, message_text, recorded_at, timestamp` |
| `l0_vec` | L0 vector index | same pattern as `l1_vec` |
| `l1_fts` / `l0_fts` | FTS5 keyword index (v2 schema) | `content` (jieba-segmented, indexed), `content_original` (UNINDEXED raw text), metadata columns UNINDEXED |

Notes:
- vec0 doesn't support `ON CONFLICT` → upserts are delete+insert in a manual transaction.
- WAL mode for thread safety.
- Provider/model/dimension change → vector tables dropped, requires `reindexAll()`.
- L2/L3 also tracked via a `ProfileRecord` sync type (`id = profile:v1:sha256(scope+type+filename)`, content, `contentMd5`, version, optimistic-locking `baselineVersion`) to keep Markdown files in sync with a remote store.

## 3. Extraction pipeline

**Trigger**: every N conversations (`pipeline.everyNConversations`, default 5) + idle timeout (`pipeline.l1IdleTimeoutSeconds`, default 600s **[unverified against config.ts]**).

**L1 extraction** is a single LLM call combining scene segmentation + fact extraction. The prompt instructs the model to:
- Detect scene continuation vs. scene switch.
- Extract only 3 memory types: `persona` (stable traits/preferences), `episodic` (objective events/decisions, with ISO 8601 `activity_start_time`/`activity_end_time` when derivable), `instruction` (long-term behavioral rules for the AI).
- Score each memory 0–100 `priority` per type-specific rubric.
- Exclude small talk, one-off requests, duplicates, the AI's own outputs, and purely subjective/emotional statements.
- Output strict JSON: `{scene_name, message_ids, memories:[{content, type, priority, source_message_ids, metadata}]}`.

**Dedup is NOT add-only** — this is the key architectural divergence from an append-only design. `l1-dedup.ts` runs a batch LLM "conflict detection" pass: new memories are compared against a candidate pool of similar existing memories (found via vector similarity; `extraction.enableDedup` default `true`), and each new memory is resolved as one of:

- `store` — no similar existing memory; add as new.
- `skip` — existing memory is equal/better; discard the new one.
- `update` — same fact/event; new memory supersedes old (`target_ids` deleted, replaced).
- `merge` — multiple memories (even across types) combined into one, with `merged_content`/`merged_type`/`merged_priority` (often boosted) and `merged_timestamps` = union of source timestamps.

Net effect: a mutable/curated fact store, not an append-only log. Raw history is preserved only at L0.

## 4. Retrieval / recall strategy

Config: `recall.strategy ∈ {"embedding","keyword","hybrid"}` (default `hybrid`), `recall.maxResults` default 5.

- **Vector**: cosine similarity via sqlite-vec KNN (`embedding MATCH ? AND k = ?`).
- **Keyword**: SQLite FTS5, BM25-derived score normalized 0–1. Chinese segmentation via `@node-rs/jieba` (`cutForSearch`), falling back to Unicode-regex splitting if jieba isn't installed. Query terms OR-joined as quoted phrases.
- **Fusion**: Reciprocal Rank Fusion (`src/core/store/search-utils.ts`, `rrfMerge()`), `RRF_K = 60` (canonical RRF-paper value). Score = Σ `1/(k + rank + 1)` across ranked lists; items appearing in multiple lists sum their scores.
- Backend capability negotiation: `IMemoryStore` exposes flags (`vectorSearch`, `ftsSearch`, `nativeHybridSearch`, `sparseVectors`) so a backend like TCVDB can bypass RRF with server-side hybrid search.
- Search applies to both L1 facts and L0 raw conversation (`searchL1Vector/Fts/Hybrid`, `searchL0Vector/Fts`).
- Exposed as tools: `tdai_memory_search`, `tdai_conversation_search`, plus automatic pre-turn injection (`auto-recall.ts`).

## 5. Persona / profile generation

Trigger: every N new memories, `persona.triggerEveryN` (default **50**, confirmed in `config.ts`). Also: `persona.maxScenes` (15), `persona.backupCount` (3), `persona.sceneBackupCount` (10), optional dedicated `persona.model`.

The LLM is given the existing `persona.md` plus changed/new L2 scene blocks and writes the updated persona via a file-write tool (`write` for full rewrite/first run, `edit` for incremental patches) — it does not read the file itself. Constraints: capped at ~2000 characters, must avoid hallucination (especially cold-start), must draw only from provided scene data, may only touch `persona.md`.

Content model — "four-layer deep scan":
1. Base & Facts (demographics, icebreakers)
2. Interest Graph (active hobby vs. passive consumption vs. dormant interest)
3. Interface (communication style/pet peeves — guides how the agent talks)
4. Core (decision logic, contradictions, drivers — lets the agent "co-pilot" decisions)

Output is narrative Markdown, language mirrors the dominant language of the source scene content. PersonaMem accuracy claim: 76% vs. 48% baseline **[unverified, README claim only]**.

## 6. API surface

Two deployment modes:

- **OpenClaw plugin** (`src/adapters/openclaw/*`) — in-process hooks `auto-capture.ts` (L0 write), `auto-recall.ts` (pre-turn injection), tools `tdai_memory_search` / `tdai_conversation_search`.
- **Hermes gateway** (`src/gateway/server.ts`, Python plugin under `hermes-plugin/memory/memory_tencentdb/`) — standalone HTTP service, default port **8420**, endpoints `/capture`, `/search`, `/recall`, `/health`. Bearer-token auth optional, CORS configurable. **Request/response payload shapes not verified** — only endpoint names/paths confirmed via README + file layout.

## 7. Tech stack, providers, config

- Runtime: Node.js ≥22.16, TypeScript, npm.
- Vector store: sqlite-vec (local default) or Tencent Cloud VectorDB (`src/core/store/tcvdb.ts`).
- Keyword store: SQLite FTS5 (degrades gracefully if unavailable) with jieba segmentation.
- Embeddings: pluggable `EmbeddingService`, local or OpenAI-compatible; dimension mismatch triggers automatic vector-table drop + reindex; `provider="none"` defers vector tables entirely (keyword-only).
- LLM: two runner adapters — piggyback on host agent's model (OpenClaw) or standalone LLM mode.

Confirmed config keys (`src/config.ts`):
```
extraction.enableDedup          default true
persona.triggerEveryN           default 50
persona.maxScenes               default 15
persona.backupCount             default 3
persona.sceneBackupCount        default 10
persona.model                   optional override
pipeline.everyNConversations    default 5
recall.maxResults               default 5
recall.strategy                 "embedding" | "keyword" | "hybrid", default "hybrid"
```
README-cited but **not directly source-verified**: `pipeline.l1IdleTimeoutSeconds` (600), `offload.enabled` (false), `offload.mmdMaxTokenRatio` (0.2), `recall.timeoutSeconds` (5).

## 8. Notable design decisions vs. this project

- **Not ADD-only**: L1 runs an explicit LLM `store/skip/update/merge` conflict-resolution pass that can delete/rewrite/merge prior memories across types — a curated store, not an append-only log (unlike `memory-agent`'s ADD-only invariant). Raw history is only guaranteed at L0.
- **Heterogeneous storage by layer** (DB for evidence, Markdown files for structure) is unusual — buys human-readability/debuggability at the cost of a separate sync/versioning mechanism (`ProfileRecord`) between DB and filesystem.
- **Symbolic memory / Mermaid graph offloading** targets short-term context-window compression during long agent sessions — orthogonal to the L0–L3 long-term pipeline; this project has no analog.
- **Single LLM call** combines scene segmentation + fact extraction (cost optimization) vs. two separate passes.
- **Backend abstraction** (`IMemoryStore`) cleanly separates local SQLite vs. remote TCVDB with capability-negotiation flags — conceptually similar to this project's `RECALL_STRATEGY` switch, but exposed as per-backend capability flags rather than one global setting.
- RRF constant `k=60` matches both the canonical RRF paper and this project's own `RECALL_RRF_K` default.

**Unverified / README-only claims**: 61.38% token reduction, 51.52% WideSearch success gain, 76% vs. 48% PersonaMem accuracy; exact gateway request/response schemas; most `offload.*` config keys; warmup-mode logic; idle-timeout/dedup-interval exact values beyond `l1IdleTimeoutSeconds`.
