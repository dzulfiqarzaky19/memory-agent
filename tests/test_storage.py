from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_save_and_search_memories(db):
    uid = "test-user-storage"

    embedding = [0.1] * 768
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
    embedding = [0.1] * 768

    mid1 = await db.save_memory(uid, "User likes Go", embedding)
    assert mid1 is not None
    mid2 = await db.save_memory(uid, "User likes Go", embedding)
    assert mid2 is None  # duplicate returns None

    count = await db.count_memories(uid)
    assert count == 1


@pytest.mark.asyncio
async def test_conversation_crud(db):
    uid = "test-conv-storage"

    cid = await db.save_conversation(uid, "user", "Hello world")
    assert cid is not None

    convs = await db.get_recent_conversations(uid, limit=10)
    assert len(convs) == 1
    assert convs[0]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_keyword_search(db):
    uid = "test-keyword"
    emb = [0.1] * 768

    await db.save_memory(uid, "User enjoys hiking in mountains", emb)
    await db.save_memory(uid, "User likes programming in Rust", emb)
    await db.save_memory(uid, "User prefers coffee over tea", emb)

    kw_results = await db.keyword_search_memories(uid, "hiking mountains", top_k=5, threshold=0.0)
    assert any("hiking" in r["text"] for r in kw_results)


@pytest.mark.asyncio
async def test_hybrid_search(db):
    uid = "test-hybrid"
    emb = [0.1] * 768

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
async def test_scenario_crud(db):
    uid = "test-scenario"
    emb = [0.1] * 768

    sid = await db.save_scenario(uid, "Dev Tools", "User's dev preferences", emb, ["mem1", "mem2"])
    assert sid is not None

    scenarios = await db.get_all_scenarios(uid)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "Dev Tools"

    searched = await db.search_scenarios(uid, emb, top_k=5, threshold=0.0)
    assert len(searched) == 1


@pytest.mark.asyncio
async def test_typed_memory_and_instructions(db):
    uid = "test-typed"
    emb = [0.1] * 768

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
    emb = [0.1] * 768

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
    emb = [0.1] * 768

    assert await db.count_memories(uid) == 0
    await db.save_memory(uid, "test memory", emb)
    assert await db.count_memories(uid) == 1
    assert await db.count_memories() >= 1
