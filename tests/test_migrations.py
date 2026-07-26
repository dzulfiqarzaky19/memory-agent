"""Versioned migrations replace run-SCHEMA_SQL-every-boot (Chunk F)."""

from __future__ import annotations

import pytest

from config import EMBEDDING_DIMENSIONS
from migrations import MIGRATIONS, run_migrations


@pytest.mark.asyncio
async def test_all_migrations_recorded(db):
    async with db._pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
    recorded = [r["version"] for r in rows]
    assert recorded == [v for v, _ in MIGRATIONS]


@pytest.mark.asyncio
async def test_rerun_is_a_noop(db):
    """Steady-state boot must apply nothing — no DDL to block behind idle txns."""
    async with db._pool.acquire() as conn:
        applied = await run_migrations(conn, EMBEDDING_DIMENSIONS)
    assert applied == []


@pytest.mark.asyncio
async def test_boot_does_not_delete_existing_memories(db):
    """The old boot DELETE ran against live rows every start. It must not anymore."""
    uid = "test-mig-nodelete"
    emb = [0.1] * EMBEDDING_DIMENSIONS
    await db.save_memory(uid, "survives a reboot", emb)
    before = await db.count_memories(uid)
    assert before == 1

    async with db._pool.acquire() as conn:
        await run_migrations(conn, EMBEDDING_DIMENSIONS)
    assert await db.count_memories(uid) == before


class _Rollback(Exception):
    """Abort the transaction so the shared dev DB keeps its real migration rows."""


@pytest.mark.asyncio
async def test_baseline_skips_001_on_populated_db(db):
    """A pre-versioning DB records 001 without executing its dedupe DELETE."""
    seen: dict = {}
    async with db._pool.acquire() as conn:
        try:
            async with conn.transaction():
                await conn.execute("DELETE FROM schema_migrations")
                seen["applied"] = await run_migrations(conn, EMBEDDING_DIMENSIONS)
                seen["versions"] = {
                    r["version"]
                    for r in await conn.fetch("SELECT version FROM schema_migrations")
                }
                raise _Rollback()
        except _Rollback:
            pass

    # `memories` exists -> 001 baselined (recorded, never executed).
    assert 1 not in seen["applied"]
    assert seen["versions"] == {v for v, _ in MIGRATIONS}

    # Rollback restored the real rows.
    async with db._pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM schema_migrations")
    assert {r["version"] for r in rows} == {v for v, _ in MIGRATIONS}


@pytest.mark.asyncio
async def test_extraction_jobs_table_exists(db):
    async with db._pool.acquire() as conn:
        assert await conn.fetchval("SELECT to_regclass('public.extraction_jobs') IS NOT NULL")
        idx = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_jobs_live_user'"
        )
    # One live job per user is a DB invariant, not a convention.
    assert "UNIQUE" in idx
    assert "queued" in idx and "running" in idx
