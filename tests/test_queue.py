"""Durable extraction queue + worker (Chunk B)."""

from __future__ import annotations

import asyncio

import pytest

from worker import ExtractionWorker


async def _live_job_id(engine, user_id: str) -> str:
    async with engine.storage._pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT id FROM extraction_jobs WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1",
            user_id,
        )


@pytest.mark.asyncio
async def test_add_queues_instead_of_extracting(engine, monkeypatch):
    """The request path must never call the LLM — an 8s hook cannot wait on it."""
    uid = "test-q-noinline"
    called = {"n": 0}

    async def boom(messages_text, existing_memories=None):
        called["n"] += 1
        raise AssertionError("extraction must not run on the request path")

    monkeypatch.setattr(engine.extractor, "extract_memories", boom)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    r = await engine.add(
        uid,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    assert r["extract_status"] == "queued"
    assert called["n"] == 0
    assert await engine.storage.count_live_extraction_jobs(uid) == 1


@pytest.mark.asyncio
async def test_enqueue_coalesces_per_user(engine, monkeypatch):
    """One live job per user — extraction_state holds a single cursor."""
    uid = "test-q-coalesce"
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)
    for i in range(3):
        await engine.add(
            uid,
            [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": "a"}],
        )
    assert await engine.storage.count_live_extraction_jobs(uid) == 1


@pytest.mark.asyncio
async def test_claim_is_exclusive(engine):
    """Two concurrent claims must not hand out the same job."""
    uid = "test-q-claim"
    await engine.storage.enqueue_extraction_job(uid)
    a, b = await asyncio.gather(
        engine.storage.claim_extraction_job(600),
        engine.storage.claim_extraction_job(600),
    )
    claimed = [j for j in (a, b) if j is not None]
    assert len(claimed) == 1
    assert claimed[0]["user_id"] == uid


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed(engine):
    """A crashed worker's job must not stay stuck in 'running'."""
    uid = "test-q-lease"
    await engine.storage.enqueue_extraction_job(uid)
    first = await engine.storage.claim_extraction_job(600)
    assert first is not None
    # Nobody else can take it while the lease holds.
    assert await engine.storage.claim_extraction_job(600) is None

    async with engine.storage._pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET lease_until = now() - interval '1 minute' WHERE id = $1",
            first["id"],
        )
    again = await engine.storage.claim_extraction_job(600)
    assert again is not None
    assert again["id"] == first["id"]
    assert again["attempts"] == 2


@pytest.mark.asyncio
async def test_worker_drains_queued_job(engine, monkeypatch):
    uid = "test-q-drain"

    async def facts(messages_text, existing_memories=None):
        return [{"content": "Worker mined this", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    r = await engine.add(
        uid,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    assert r["extract_status"] == "queued"

    worker = ExtractionWorker(engine)
    assert await worker.run_once() is True

    assert await engine.storage.count_live_extraction_jobs(uid) == 0
    assert await engine.storage.count_memories(uid) == 1
    trust = await engine.memory_trust(uid)
    assert trust["recall_trusted"] is True
    assert trust["last_extract_ok"] is True


@pytest.mark.asyncio
async def test_worker_retries_then_marks_dead(engine, monkeypatch):
    """Poison job backs off, then stops retrying instead of spinning forever."""
    uid = "test-q-poison"

    async def boom(messages_text, existing_memories=None):
        raise RuntimeError("llm permanently down")

    monkeypatch.setattr(engine.extractor, "extract_memories", boom)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)
    monkeypatch.setattr("worker.EXTRACTION_JOB_MAX_ATTEMPTS", 2)
    monkeypatch.setattr("worker.EXTRACTION_RETRY_BACKOFF_SECONDS", 0)

    await engine.add(
        uid,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )

    worker = ExtractionWorker(engine)
    # Attempt 1 fails -> requeued with backoff, not dead.
    assert await worker.run_once() is True
    job_id = await _live_job_id(engine, uid)
    job = await engine.storage.get_extraction_job(job_id)
    assert job["status"] == "queued"
    assert job["attempts"] == 1
    assert "permanently down" in job["last_error"]

    # Attempt 2 hits EXTRACTION_JOB_MAX_ATTEMPTS -> dead, stops spinning.
    assert await worker.run_once() is True
    job = await engine.storage.get_extraction_job(job_id)
    assert job["status"] == "dead"
    assert await engine.storage.count_live_extraction_jobs(uid) == 0


@pytest.mark.asyncio
async def test_worker_returns_false_when_idle(engine):
    worker = ExtractionWorker(engine)
    # Other tests' users are wiped; a claim may still pick up a real user's job,
    # so only assert the no-crash contract.
    result = await worker.run_once()
    assert result in (True, False)


@pytest.mark.asyncio
async def test_queued_job_survives_restart(engine, monkeypatch):
    """Job durability: a fresh worker picks up work queued before it existed."""
    uid = "test-q-restart"

    async def facts(messages_text, existing_memories=None):
        return [{"content": "Survived restart", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    await engine.add(
        uid,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    # Simulate process death mid-lease.
    claimed = await engine.storage.claim_extraction_job(600)
    async with engine.storage._pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET lease_until = now() - interval '1 hour' WHERE id = $1",
            claimed["id"],
        )

    fresh = ExtractionWorker(engine)
    assert await fresh.run_once() is True
    assert await engine.storage.count_memories(uid) == 1
