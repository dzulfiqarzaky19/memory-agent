"""Transactional capture: concurrent hook retries must not duplicate L0 (Chunk C)."""

from __future__ import annotations

import asyncio

import pytest

BATCH = [
    {"role": "user", "content": "same exchange"},
    {"role": "assistant", "content": "same reply"},
]


@pytest.mark.asyncio
async def test_concurrent_identical_captures_write_one_batch(engine):
    """Two hooks racing the same exchange: exactly one writes, other sees duplicate."""
    uid = "test-race-dupe"
    key = "test-race-dupe-session"

    a, b = await asyncio.gather(
        engine.capture(uid, key, list(BATCH)),
        engine.capture(uid, key, list(BATCH)),
    )
    dupes = [r for r in (a, b) if r["duplicate"]]
    writes = [r for r in (a, b) if not r["duplicate"]]
    assert len(writes) == 1
    assert len(dupes) == 1

    convs = await engine.storage.get_recent_conversations(uid, limit=50)
    assert len(convs) == 2  # one batch, not two


@pytest.mark.asyncio
async def test_checkpoint_written_with_l0_not_after(engine, monkeypatch):
    """Checkpoint is committed with L0 — no window where L0 exists without it."""
    uid = "test-race-cp"
    key = "test-race-cp-session"
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    r = await engine.capture(uid, key, list(BATCH))
    assert r["duplicate"] is False
    assert r["extract_status"] == "queued"

    cp = await engine.storage.get_capture_checkpoint(key)
    assert cp is not None
    assert cp["user_id"] == uid
    assert cp["messages_seen"] == 2
    assert cp["last_batch_hash"]

    # Retry of the exact batch is a no-op even though extraction has not run.
    again = await engine.capture(uid, key, list(BATCH))
    assert again["duplicate"] is True
    convs = await engine.storage.get_recent_conversations(uid, limit=50)
    assert len(convs) == 2


@pytest.mark.asyncio
async def test_concurrent_distinct_batches_both_land(engine):
    """Different exchanges on one session must both persist, counter intact."""
    uid = "test-race-distinct"
    key = "test-race-distinct-session"

    first = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "1"}]
    second = [{"role": "user", "content": "second"}, {"role": "assistant", "content": "2"}]

    await asyncio.gather(
        engine.capture(uid, key, first),
        engine.capture(uid, key, second),
    )
    convs = await engine.storage.get_recent_conversations(uid, limit=50)
    contents = {c["content"] for c in convs}
    assert {"first", "second"} <= contents
    assert len(convs) == 4

    cp = await engine.storage.get_capture_checkpoint(key)
    assert cp["messages_seen"] == 4


@pytest.mark.asyncio
async def test_foreign_session_key_does_not_leak(engine):
    """A session_key reused by another user must not honor the foreign checkpoint."""
    key = "test-race-shared-session"
    await engine.capture("test-race-user-a", key, list(BATCH))
    r = await engine.capture("test-race-user-b", key, list(BATCH))
    assert r["duplicate"] is False

    b_convs = await engine.storage.get_recent_conversations("test-race-user-b", limit=50)
    assert len(b_convs) == 2
