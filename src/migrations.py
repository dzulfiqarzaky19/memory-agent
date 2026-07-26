"""Versioned schema migrations.

Boot used to execute the entire SCHEMA_SQL — including a self-join DELETE over
`memories` — on every start. That made a normal restart issue DDL, which blocks
behind any `idle in transaction` connection and stalls startup. Now each version
runs once, recorded in `schema_migrations`; a steady-state boot executes no DDL.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Serializes concurrent boots (multiple uvicorn workers) so they cannot race DDL.
_ADVISORY_LOCK_KEY = 8_374_221_905_551_337

_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# 001 is the historical baseline: everything SCHEMA_SQL created up to the point
# versioning was introduced. On a database that already has `memories`, it is
# recorded as applied WITHOUT executing (see baseline handling below) — re-running
# its dedupe DELETE against live rows would be both slow and pointless.
MIGRATION_001_BASELINE = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    agent_id    TEXT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    embedding   vector({dims}),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);

-- One-time dedupe so the unique index below can be created. Runs on fresh
-- databases only; existing installs are baselined past it.
DELETE FROM memories a
 USING memories b
 WHERE a.user_id = b.user_id
   AND a.text_hash = b.text_hash
   AND (a.created_at > b.created_at
        OR (a.created_at = b.created_at AND a.id > b.id));
DROP INDEX IF EXISTS idx_mem_hash;
CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_user_hash ON memories(user_id, text_hash);

ALTER TABLE memories ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS idx_mem_tsv ON memories USING gin (text_tsv);
CREATE INDEX IF NOT EXISTS idx_mem_trgm ON memories USING gin (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_mem_embedding ON memories
    USING hnsw (embedding vector_cosine_ops);

ALTER TABLE memories ADD COLUMN IF NOT EXISTS mem_type TEXT NOT NULL DEFAULT 'episodic';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS priority INT NOT NULL DEFAULT 50;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS agent_id TEXT;
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(user_id, mem_type);

CREATE TABLE IF NOT EXISTS scenarios (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding   vector({dims}),
    memory_ids  JSONB DEFAULT '[]',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scen_user ON scenarios(user_id);
CREATE INDEX IF NOT EXISTS idx_scen_created ON scenarios(created_at);
CREATE INDEX IF NOT EXISTS idx_scen_embedding ON scenarios
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS extraction_state (
    user_id                 TEXT PRIMARY KEY,
    conversations_seen      INT NOT NULL DEFAULT 0,
    memories_since_scenario INT NOT NULL DEFAULT 0,
    updated_at              TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE extraction_state ADD COLUMN IF NOT EXISTS last_extraction_at TIMESTAMPTZ;
ALTER TABLE extraction_state ADD COLUMN IF NOT EXISTS last_extraction_id TEXT;
ALTER TABLE extraction_state ADD COLUMN IF NOT EXISTS last_extract_ok BOOLEAN;
ALTER TABLE extraction_state ADD COLUMN IF NOT EXISTS last_extract_error TEXT;
ALTER TABLE extraction_state ADD COLUMN IF NOT EXISTS last_extract_attempt_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS personas (
    user_id                TEXT PRIMARY KEY,
    summary                TEXT NOT NULL,
    memory_count           INT NOT NULL,
    memories_at_generation INT NOT NULL,
    generated_at           TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS capture_checkpoints (
    session_key     TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    messages_seen   INT NOT NULL DEFAULT 0,
    last_batch_hash TEXT,
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_capture_user ON capture_checkpoints(user_id);
"""

MIGRATION_002_EXTRACTION_JOBS = """
CREATE TABLE IF NOT EXISTS extraction_jobs (
    id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id      TEXT NOT NULL,
    agent_id     TEXT,
    status       TEXT NOT NULL DEFAULT 'queued',
    attempts     INT  NOT NULL DEFAULT 0,
    last_error   TEXT,
    lease_until  TIMESTAMPTZ,
    run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- At most ONE live job per user: extraction_state holds a single keyset cursor,
-- so two concurrent jobs for one user would double-mine it. Enqueue coalesces.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_live_user ON extraction_jobs(user_id)
    WHERE status IN ('queued','running');
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON extraction_jobs(status, run_after)
    WHERE status IN ('queued','running');
"""

MIGRATION_003_PARTNER_FACTS = """
-- Partner pack: agent thin self + relation scars. Deliberately NOT in `memories`.
-- User read paths (search_*, get_all_memories, count_memories) filter nothing by
-- type, so agent-owned rows there would leak into recall, feed persona generation,
-- and inflate l1_count/recall_trusted. A separate table keeps that isolation
-- structural instead of a filter every future query has to remember.
-- No embedding column: fetched by (user_id, agent_id, kind), never vector-searched.
CREATE TABLE IF NOT EXISTS partner_facts (
    id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id    TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    priority   INT  NOT NULL DEFAULT 50,
    metadata   JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_partner_dedupe
    ON partner_facts(user_id, agent_id, kind, text_hash);
CREATE INDEX IF NOT EXISTS idx_partner_lookup
    ON partner_facts(user_id, agent_id, kind, priority DESC, created_at DESC);
"""

MIGRATIONS: list[tuple[int, str]] = [
    (1, MIGRATION_001_BASELINE),
    (2, MIGRATION_002_EXTRACTION_JOBS),
    (3, MIGRATION_003_PARTNER_FACTS),
]


async def run_migrations(conn, dims: int) -> list[int]:
    """Apply pending migrations under an advisory lock. Returns versions applied."""
    await conn.execute(_VERSION_TABLE)

    # Pre-versioning database: `memories` already exists, so 001 describes what is
    # already there. Record it without executing — never re-run its dedupe DELETE
    # against live rows.
    if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM schema_migrations)"):
        has_memories = await conn.fetchval("SELECT to_regclass('public.memories') IS NOT NULL")
        if has_memories:
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (1) ON CONFLICT DO NOTHING"
            )
            logger.info("Baselined existing schema at migration 001 (not executed)")

    applied: list[int] = []
    await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)
    try:
        done = {
            r["version"]
            for r in await conn.fetch("SELECT version FROM schema_migrations")
        }
        for version, sql in MIGRATIONS:
            if version in done:
                continue
            async with conn.transaction():
                await conn.execute(sql.replace("{dims}", str(dims)))
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", version
                )
            applied.append(version)
            logger.info("Applied migration %03d", version)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
    return applied
