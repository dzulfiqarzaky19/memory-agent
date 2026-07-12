from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_add_conversation(engine):
    uid = "test-add-conv"
    messages = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    result = await engine.add(uid, messages)
    assert result["memories_added"] == 0
    assert result["memory_ids"] == []

    convs = await engine.storage.get_recent_conversations(uid)
    assert len(convs) == 2


@pytest.mark.asyncio
async def test_search_empty(engine):
    uid = "test-search-empty"
    results = await engine.search(uid, "anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_scenarios_empty(engine):
    uid = "test-scenarios-empty"
    result = await engine.get_scenarios(uid)
    assert result["total"] == 0
    assert result["scenarios"] == []


@pytest.mark.asyncio
async def test_priority_and_instruction_recall(engine):
    uid = "test-priority-recall"
    emb = engine.embedder.embed(["x"])[0]

    # Identical embeddings -> ranking decided by priority tilt.
    await engine.storage.save_memory(uid, "Low priority fact", emb, priority=10)
    await engine.storage.save_memory(uid, "High priority fact", emb, priority=95)
    await engine.storage.save_memory(uid, "Never use emojis", emb, mem_type="instruction", priority=90)

    results = await engine.search(uid, "fact", top_k=10)
    texts = [r["text"] for r in results]

    # Instruction is always surfaced even without matching the query.
    assert "Never use emojis" in texts
    # Higher-priority fact outranks the lower-priority one.
    assert texts.index("High priority fact") < texts.index("Low priority fact")


@pytest.mark.asyncio
async def test_persona_empty(engine):
    uid = "test-persona-empty"
    result = await engine.get_persona(uid)
    assert result["memory_count"] == 0
    assert "No memories" in result["summary"]
