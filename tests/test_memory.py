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
async def test_persona_empty(engine):
    uid = "test-persona-empty"
    result = await engine.get_persona(uid)
    assert result["memory_count"] == 0
    assert "No memories" in result["summary"]
