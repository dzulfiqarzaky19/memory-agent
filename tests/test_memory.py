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

    # Instruction fills remaining slots when top_k has room.
    assert "Never use emojis" in texts
    # Higher-priority fact outranks the lower-priority one among primary matches.
    assert texts.index("High priority fact") < texts.index("Low priority fact")


@pytest.mark.asyncio
async def test_search_injects_fill_remaining_only(engine, monkeypatch):
    uid = "test-inject-slots"
    emb = engine.embedder.embed(["x"])[0]

    # Keyword strategy so instruction text does not enter primary for query "widgets".
    monkeypatch.setattr("memory.RECALL_STRATEGY", "keyword")

    await engine.storage.save_memory(uid, "Matched primary fact about widgets", emb, priority=80)
    await engine.storage.save_memory(
        uid, "Always reply in English", emb, mem_type="instruction", priority=90
    )

    # top_k=1: only primary match, no room for instruction inject.
    tight = await engine.search(uid, "widgets", top_k=1)
    assert len(tight) == 1
    assert "widgets" in tight[0]["text"]

    # top_k large enough: instruction fills remaining slot below primary score.
    wide = await engine.search(uid, "widgets", top_k=5)
    texts = [r["text"] for r in wide]
    assert any("widgets" in t for t in texts)
    assert "Always reply in English" in texts
    primary = next(r for r in wide if "widgets" in r["text"])
    instruction = next(r for r in wide if "English" in r["text"])
    assert instruction["score"] < primary["score"]


@pytest.mark.asyncio
async def test_add_counts_user_turns(engine, monkeypatch):
    uid = "test-user-turns"
    calls = {"n": 0}

    async def fake_extract(messages_text, existing_memories=None):
        calls["n"] += 1
        return [{"content": f"fact-{calls['n']}", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", fake_extract)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 3)

    pair = lambda i: [
        {"role": "user", "content": f"user msg {i}"},
        {"role": "assistant", "content": f"assistant msg {i}"},
    ]

    # 2 user turns — below threshold
    await engine.add(uid, pair(1))
    await engine.add(uid, pair(2))
    assert calls["n"] == 0

    # 3rd user turn triggers extract once
    await engine.add(uid, pair(3))
    assert calls["n"] == 1

    # Assistant-only does not bump / extract
    await engine.add(uid, [{"role": "assistant", "content": "lonely assistant"}])
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_rebuild_keeps_old_on_empty_llm(engine, monkeypatch):
    uid = "test-rebuild-keep"
    emb = engine.embedder.embed(["x"])[0]

    mid = await engine.storage.save_memory(uid, "User likes Rust", emb)
    await engine.storage.save_scenario(
        uid, "Languages", "Programming languages", emb, [mid]
    )

    async def empty_group(memory_texts, existing_scenarios=None):
        return []

    monkeypatch.setattr(engine.extractor, "group_into_scenarios", empty_group)
    await engine._rebuild_scenarios(uid)

    scenarios = await engine.storage.get_all_scenarios(uid)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "Languages"


@pytest.mark.asyncio
async def test_group_prompt_omits_memory_uuids(engine, monkeypatch):
    captured = {}

    async def fake_llm(system, user):
        captured["user"] = user
        return '{"scenarios": [{"name": "X", "description": "Y", "memory_indices": [1]}]}'

    monkeypatch.setattr(engine.extractor, "_call_llm", fake_llm)

    fake_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    await engine.extractor.group_into_scenarios(
        memory_texts=["User likes Go"],
        existing_scenarios=[
            {
                "name": "Lang",
                "description": "langs",
                "memory_ids": [fake_uuid],
            }
        ],
    )
    assert fake_uuid not in captured["user"]
    assert "IDs:" not in captured["user"]
    assert "Lang: langs" in captured["user"]
    assert "memory_indices" in captured["user"]


@pytest.mark.asyncio
async def test_persona_empty(engine):
    uid = "test-persona-empty"
    result = await engine.get_persona(uid)
    assert result["memory_count"] == 0
    assert "No memories" in result["summary"]
