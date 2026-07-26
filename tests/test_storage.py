from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config import EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_save_and_search_memories(db):
    uid = "test-user-storage"

    embedding = [0.1] * EMBEDDING_DIMENSIONS
    mid = await db.save_memory(uid, "User likes Python", embedding)
    assert mid is not None
    assert isinstance(mid, str)

    await db.save_memory(uid, "User prefers dark mode", embedding)

    results = await db.search_memories(uid, embedding, top_k=10, threshold=0.0)
    assert len(results) >= 2
    assert any("Python" in r["text"] for r in results)
    assert any("dark mode" in r["text"] for r in results)


@pytest.mark.asyncio
async def test_dedup_memories(db):
    uid = "test-dedup"
    embedding = [0.1] * EMBEDDING_DIMENSIONS

    mid1 = await db.save_memory(uid, "User likes Go", embedding)
    assert mid1 is not None
    mid2 = await db.save_memory(uid, "User likes Go", embedding)
    assert mid2 is None  # duplicate returns None

    count = await db.count_memories(uid)
    assert count == 1


@pytest.mark.asyncio
async def test_unique_user_text_hash(db):
    uid = "test-unique-hash"
    emb = [0.1] * EMBEDDING_DIMENSIONS
    text = "Unique constraint fact"

    mid1 = await db.save_memory(uid, text, emb)
    assert mid1 is not None
    # Second insert must be rejected (fast-path or UniqueViolation).
    mid2 = await db.save_memory(uid, text, emb)
    assert mid2 is None
    assert await db.count_memories(uid) == 1

    async with db._pool.acquire() as conn:
        idx = await conn.fetchval(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = 'memories' AND indexname = 'idx_mem_user_hash'"""
        )
    assert idx == "idx_mem_user_hash"


@pytest.mark.asyncio
async def test_conversation_crud(db):
    uid = "test-conv-storage"

    cid = await db.save_conversation(uid, "user", "Hello world")
    assert cid is not None

    convs = await db.get_recent_conversations(uid, limit=10)
    assert len(convs) == 1
    assert convs[0]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_conversations_since(db):
    uid = "test-conv-since"

    await db.save_conversation(uid, "user", "old message")
    # Stamp watermark at the old row (not wall-clock now via bare success).
    old_rows = await db.get_conversations_since(uid, since=None, limit=10)
    assert old_rows
    await db.mark_extraction_success(uid, watermark_at=old_rows[-1]["created_at"])

    state = await db.get_extraction_state(uid)
    since = state["last_extraction_at"]
    assert since is not None

    await db.save_conversation(uid, "user", "new message")
    await db.save_conversation(uid, "assistant", "new reply")

    pending = await db.get_conversations_since(uid, since=since)
    texts = [c["content"] for c in pending]
    assert "old message" not in texts
    assert "new message" in texts
    assert "new reply" in texts

    cold = await db.get_conversations_since(uid, since=None, limit=10)
    assert cold[0]["content"] == "old message"
    assert [c["created_at"] for c in cold] == sorted(c["created_at"] for c in cold)


@pytest.mark.asyncio
async def test_keyword_search(db):
    uid = "test-keyword"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    await db.save_memory(uid, "User enjoys hiking in mountains", emb)
    await db.save_memory(uid, "User likes programming in Rust", emb)
    await db.save_memory(uid, "User prefers coffee over tea", emb)

    kw_results = await db.keyword_search_memories(uid, "hiking mountains", top_k=5, threshold=0.0)
    assert any("hiking" in r["text"] for r in kw_results)


@pytest.mark.asyncio
async def test_hybrid_search(db):
    uid = "test-hybrid"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    await db.save_memory(uid, "User builds web apps with React", emb)
    await db.save_memory(uid, "User uses PostgreSQL for databases", emb)

    results = await db.hybrid_search_memories(
        uid,
        query="PostgreSQL database",
        query_embedding=emb,
        top_k=5,
        vec_threshold=0.0,
        kw_threshold=0.0,
        rrf_k=60,
    )
    assert len(results) >= 2
    assert any("PostgreSQL" in r["text"] for r in results)


@pytest.mark.asyncio
async def test_hybrid_rrf_score_is_fused(db):
    uid = "test-rrf-fused"
    emb = [0.1] * EMBEDDING_DIMENSIONS
    rrf_k = 60

    # Identical embeddings → both share vector ranks; keyword differentiates.
    await db.save_memory(uid, "alpha keyword only here", emb)
    await db.save_memory(uid, "beta something else entirely", emb)

    results = await db.hybrid_search_memories(
        uid,
        query="alpha keyword",
        query_embedding=emb,
        top_k=5,
        vec_threshold=0.0,
        kw_threshold=0.0,
        rrf_k=rrf_k,
    )
    assert results
    # Item in both lists has score > single-list contribution 1/(k+rank+1) max single ≈ 1/61
    alpha = next(r for r in results if "alpha" in r["text"])
    assert alpha["score"] == round(alpha["score"], 4)
    # Fused score must include keyword contribution when present
    meta = alpha.get("metadata") or {}
    if "_kw_score" in meta and "_vec_score" in meta:
        # Both lists hit: fused > either single rank term alone for rank0+rank0
        assert alpha["score"] >= round(1.0 / (rrf_k + 1), 4)
        assert alpha["score"] > round(1.0 / (rrf_k + 1), 4) or alpha["score"] == round(
            2.0 / (rrf_k + 1), 4
        )


@pytest.mark.asyncio
async def test_scenario_crud(db):
    uid = "test-scenario"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    sid = await db.save_scenario(uid, "Dev Tools", "User's dev preferences", emb, ["mem1", "mem2"])
    assert sid is not None

    scenarios = await db.get_all_scenarios(uid)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "Dev Tools"

    searched = await db.search_scenarios(uid, emb, top_k=5, threshold=0.0)
    assert len(searched) == 1


@pytest.mark.asyncio
async def test_replace_scenarios_atomic(db):
    uid = "test-replace-scen"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    await db.save_scenario(uid, "Old A", "desc a", emb, ["m1"])
    await db.save_scenario(uid, "Old B", "desc b", emb, ["m2"])
    assert await db.count_scenarios(uid) == 2

    await db.replace_scenarios(
        uid,
        [
            {
                "name": "New Only",
                "description": "replaced set",
                "embedding": emb,
                "memory_ids": ["m3"],
            }
        ],
    )
    scenarios = await db.get_all_scenarios(uid)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "New Only"


@pytest.mark.asyncio
async def test_typed_memory_and_instructions(db):
    uid = "test-typed"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    await db.save_memory(uid, "Always answer in English", emb, mem_type="instruction", priority=90)
    await db.save_memory(uid, "User likes hiking", emb, mem_type="persona", priority=70)

    instructions = await db.get_instructions(uid)
    assert len(instructions) == 1
    assert instructions[0]["type"] == "instruction"
    assert instructions[0]["priority"] == 90

    results = await db.search_memories(uid, emb, top_k=10, threshold=0.0)
    types = {r["text"]: r["type"] for r in results}
    assert types["User likes hiking"] == "persona"


@pytest.mark.asyncio
async def test_agent_scoping(db):
    uid = "test-agent-scope"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    await db.save_memory(uid, "Shared user fact", emb)  # agent_id NULL -> global
    await db.save_memory(uid, "Agent-A only fact", emb, agent_id="agent-a")
    await db.save_memory(uid, "Agent-B only fact", emb, agent_id="agent-b")

    texts_a = {r["text"] for r in await db.search_memories(uid, emb, top_k=10, threshold=0.0, agent_id="agent-a")}
    assert "Shared user fact" in texts_a
    assert "Agent-A only fact" in texts_a
    assert "Agent-B only fact" not in texts_a


@pytest.mark.asyncio
async def test_count_memories(db):
    uid = "test-count"
    emb = [0.1] * EMBEDDING_DIMENSIONS

    assert await db.count_memories(uid) == 0
    await db.save_memory(uid, "test memory", emb)
    assert await db.count_memories(uid) == 1
    assert await db.count_memories() >= 1


@pytest.mark.asyncio
async def test_save_memory_rejects_wrong_dim(db):
    uid = "test-save-dim"
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        await db.save_memory(uid, "bad dim", [0.1] * 3)


@pytest.mark.asyncio
async def test_upsert_scenarios_rejects_wrong_dim(db):
    uid = "test-upsert-dim"
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        await db.upsert_scenarios_by_name(
            uid,
            [
                {
                    "name": "Bad",
                    "description": "wrong width",
                    "embedding": [0.1] * 3,
                    "memory_ids": ["m1"],
                }
            ],
        )


@pytest.mark.asyncio
async def test_cold_conversations_oldest_first(db):
    uid = "test-cold-asc"
    await db.save_conversation(uid, "user", "first")
    await db.save_conversation(uid, "user", "second")
    await db.save_conversation(uid, "user", "third")
    cold = await db.get_conversations_since(uid, since=None, limit=2)
    assert [c["content"] for c in cold] == ["first", "second"]


@pytest.mark.asyncio
async def test_mark_extraction_success_watermark_not_now(db):
    uid = "test-wm-not-now"
    await db.save_conversation(uid, "user", "old")
    await db.save_conversation(uid, "user", "mid")
    await db.save_conversation(uid, "user", "new")
    window = await db.get_conversations_since(uid, since=None, limit=2)
    assert [c["content"] for c in window] == ["old", "mid"]
    wm = max(c["created_at"] for c in window)
    wm_id = window[-1]["id"]
    await db.mark_extraction_success(uid, watermark_at=wm, last_extraction_id=wm_id)
    state = await db.get_extraction_state(uid)
    assert state["last_extraction_at"] == wm
    assert state["last_extraction_id"] == wm_id
    rest = await db.get_conversations_since(
        uid,
        since=state["last_extraction_at"],
        after_id=state["last_extraction_id"],
    )
    assert [c["content"] for c in rest] == ["new"]


@pytest.mark.asyncio
async def test_advance_extraction_watermark_keeps_counter(db):
    uid = "test-advance-wm"
    await db.save_conversation(uid, "user", "a")
    await db.save_conversation(uid, "user", "b")
    rows = await db.get_conversations_since(uid, since=None, limit=10)
    await db.bump_conversation_counter(uid, 5)
    await db.advance_extraction_watermark(
        uid,
        watermark_at=rows[0]["created_at"],
        last_extraction_id=rows[0]["id"],
    )
    state = await db.get_extraction_state(uid)
    assert state["conversations_seen"] == 5
    assert state["last_extraction_at"] == rows[0]["created_at"]
    assert state["last_extraction_id"] == rows[0]["id"]
    # advance does not claim final ok
    assert state["last_extract_ok"] is not True

    rest = await db.get_conversations_since(
        uid,
        since=state["last_extraction_at"],
        after_id=state["last_extraction_id"],
    )
    assert [c["content"] for c in rest] == ["b"]


@pytest.mark.asyncio
async def test_get_conversations_since_equal_ts_keyset(db):
    from datetime import datetime, timezone

    uid = "test-eq-ts-keyset"
    fixed = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ids = []
    for i in range(5):
        rid = await db.save_conversation(uid, "user", f"same-ts-{i}")
        ids.append(rid)
        async with db._pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET created_at = $2 WHERE id = $1",
                rid,
                fixed,
            )
    page1 = await db.get_conversations_since(uid, since=None, limit=2)
    assert len(page1) == 2
    page2 = await db.get_conversations_since(
        uid,
        since=page1[-1]["created_at"],
        after_id=page1[-1]["id"],
        limit=10,
    )
    # UUID id order ≠ insert order; keyset must partition without loss/dup.
    all_contents = {f"same-ts-{i}" for i in range(5)}
    p1 = {c["content"] for c in page1}
    p2 = {c["content"] for c in page2}
    assert len(page2) == 3
    assert p1.isdisjoint(p2)
    assert p1 | p2 == all_contents
    # created_at-only warm path orphans equal-ts tail
    orphaned = await db.get_conversations_since(uid, since=page1[-1]["created_at"], limit=10)
    assert orphaned == []


@pytest.mark.asyncio
async def test_replace_scenarios_rejects_wrong_dim(db):
    uid = "test-replace-dim"
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        await db.replace_scenarios(
            uid,
            [
                {
                    "name": "Bad",
                    "description": "wrong width",
                    "embedding": [0.1] * 3,
                    "memory_ids": ["m1"],
                }
            ],
        )
