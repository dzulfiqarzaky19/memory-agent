---
paths:
  - "src/**"
  - "memory_mcp.py"
  - "tests/**"
---

# Architecture (implementer + reviewer)

Work-partner memory sidecar. Not companion/brainify. Not a code graph (→ codescratch).

## Modules

| Path | Role |
|------|------|
| `src/server.py` | FastAPI routes + auth middleware + lifespan (start worker). **No LLM on request path.** |
| `src/memory.py` | Engine: `add` / `capture` / `search` / `get_persona` / `get_partner` / `run_extraction` / `memory_trust` |
| `src/storage.py` | Postgres pool, L0–L2 SQL, capture txn, `extraction_jobs`, hybrid search |
| `src/worker.py` | Lease jobs → `engine.run_extraction`. Lease not row-lock across LLM. |
| `src/extraction.py` | LLM → atoms `{content,type,priority}` types: `persona` \| `episodic` \| `instruction` |
| `src/migrations.py` | **Only** place for schema DDL. Numbered append-only `MIGRATIONS`. |
| `src/partner_seed.py` | Static thin craft spine (read-only, never written to DB). `.py` because Dockerfile copies only `src/`. |
| `src/embeddings.py` | Providers; use **`aembed` on the event loop**. Sync `embed` = boot probe only. |
| `src/models.py` | Pydantic request/response |
| `src/config.py` | `.env` |
| `src/ids.py` | `canonicalize_user_id` (lowercase) |
| `memory_mcp.py` | MCP stdio → HTTP sidecar (thin client) |

## Hot paths

```
POST /capture| /add
  → L0 + counter + capture_checkpoints (+ batch hash dedupe on capture)
  → enqueue extraction_jobs (coalesce one live job/user)
  → return ~ms   // never extract here

worker lease job
  → run_extraction: page L0 since watermark → extract → embed → L1
  → scenario rebuild cadence → mark success / failure+retry

POST /search
  → memory_trust → embed query → hybrid RRF (vector + trgm)
  → priority tilt → recency/conflict demote → instruction/scenario fill
  → stale L1 still served if any L1 exists (never false-empty)

GET /partner/{user_id}?agent_id=
  → compose only: persona CACHE (never generate) + instructions
  → seed spine + partner_facts(agent_self) + partner_facts(relation)
  → thin relation fills read-only from high-priority user episodic
  → no LLM, no embed  // SessionStart hook injects this
```

Relation compose is **suppress, not blend**: any stored `relation` row turns the
`user_episodic` fill off entirely. Deliberate — explicit scars beat inferred ones,
and mixing would dilute them with whatever episodic happens to score high. A cold
store still fills (no stored rows → episodic fill; `self` always has seed).
Top-up-to-N is a later product knob, not a bug.

Review anchors: `capture_atomic`, `enqueue_extraction_job`, `run_extraction`, `hybrid_search_memories`, `memory_trust`.

## HTTP surface

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | open; queued/dead job counts |
| POST | `/add` | greedy L0 + maybe enqueue |
| POST | `/capture` | host Stop; idempotent session_key+hash |
| POST | `/search` | + `stale` / `trust` |
| GET | `/persona/{user_id}` | L3 cache or generate |
| GET | `/partner/{user_id}` | pack: other + self + relation; cache-only |
| POST | `/partner/{user_id}` | explicit `agent_self` / `relation` delta |
| GET | `/scenarios/{user_id}` | L2 list |
| POST | `/reload` | 404 unless `MEMORY_ALLOW_RELOAD` |
| GET | `/config/llm` | non-secret wiring |

`X-Memory-Key` when `MEMORY_API_SECRET` set. `/health` always open.

## Invariants (reject diffs that break these)

1. **No LLM / blocking embed in request handlers** — enqueue only; worker drains.
2. **DDL only via new `migrations.py` entry** — never edit applied migration; never dump DDL into `Storage.initialize()`.
3. **`user_id` canonical lowercase** at trust boundary.
4. **Recall degrades, doesn’t blank** — untrusted/lagging still returns L1 + `stale`; empty only if `l1_count==0`.
5. **One live extraction job per user** — coalesce; no double-mine watermark.
6. **Memory ≠ codescratch** — no code-structure facts in L1; no shared DB.
7. **Product = work partner** — user prefs/decisions/instructions; not thick entertainment self.
8. **Agent facts stay out of `memories`** — `agent_self` / `relation` live in `partner_facts`. User read paths (`search_*`, `get_all_memories`, `count_memories`) filter nothing by type, so a row there would leak into recall, feed persona generation, and inflate `l1_count`/`recall_trusted`. Never "just add a `mem_type`".
9. **Partner reads never generate** — `/partner` is cache+compose only; a SessionStart hook must not trigger an LLM call. Generation stays in `GET /persona/{id}`.
10. **Partner facts are written only by explicit `POST /partner/{user_id}`** — never auto-mined from a session. **Enforced**, not just documented: `add_partner_fact` raises on any `agent_id` other than `PARTNER_AGENT_ID` (→400), so a spawn like `flow-review-a3f9` cannot create a spine. Reads accept any `agent_id`; only writes are pinned.

## Layers (data)

| L | Table / artifact | Written by |
|---|------------------|------------|
| L0 | `conversations` | `/add`, `/capture` |
| L1 | `memories` (+ embedding) | worker extraction |
| L2 | `scenarios` | worker every N new L1 |
| L3 | `personas` cache | `get_persona` on demand |
| — | `partner_facts` (agent_self / relation) | explicit `POST /partner/{user_id}` — **not** the worker, **not** user L1 |

L0/L1 are append-oriented today (hash dedupe). Score/decay/supersede = future; don’t invent it in handlers.

## Tests

`tests/` — single process only (shared DB). Extraction assertions must call `run_extraction` explicitly; `add`/`capture` only queue. See `database.md`.
