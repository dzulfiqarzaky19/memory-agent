from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_extract_watermark_is_window_max_not_now(engine, monkeypatch):
    """L0 written during extract must remain after watermark (C1)."""
    uid = "test-wm-window"
    times: list = []

    async def facts(messages_text, existing_memories=None):
        # Simulate concurrent L0 mid-extract (after window selected, before watermark).
        await engine.storage.save_conversation(
            uid, "user", "concurrent during extract"
        )
        return [{"content": "User said hello", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    r = await engine.add(
        uid,
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    )
    assert r["extract_status"] == "ok"
    state = await engine.storage.get_extraction_state(uid)
    wm = state["last_extraction_at"]
    # Concurrent row should be strictly after watermark (or equal edge) — still queryable.
    pending = await engine.storage.get_conversations_since(uid, since=wm, limit=50)
    assert any("concurrent during extract" in c["content"] for c in pending)


@pytest.mark.asyncio
async def test_extract_pages_cold_backlog(engine, monkeypatch):
    """Cold start with more L0 than one window must not orphan the tail (C2).

    cold_limit = max(EXTRACTION_EVERY_N_TURNS*4, 40) → 40 when N=1.
    Seed >40 user rows so the while-loop must page.
    """
    uid = "test-cold-page"
    calls = {"n": 0, "texts": []}

    async def facts(messages_text, existing_memories=None):
        calls["n"] += 1
        calls["texts"].append(messages_text)
        return [{"content": f"fact-batch-{calls['n']}", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    for i in range(45):
        await engine.storage.save_conversation(uid, "user", f"seed-user-{i:02d}")

    r = await engine.add(
        uid,
        [{"role": "user", "content": "trigger"}, {"role": "assistant", "content": "ok"}],
    )
    assert r["extract_status"] in ("ok", "ok_no_facts")
    assert calls["n"] >= 2  # first page 40, then remainder
    blob = "\n".join(calls["texts"])
    assert "seed-user-44" in blob
    assert "trigger" in blob
    state = await engine.storage.get_extraction_state(uid)
    leftover = await engine.storage.get_conversations_since(
        uid, since=state["last_extraction_at"], limit=50
    )
    assert leftover == []


@pytest.mark.asyncio
async def test_get_memories_by_ids_scoped_to_user(engine):
    uid_a = "test-ids-a"
    uid_b = "test-ids-b"
    emb = engine.embedder.embed(["x"])[0]
    mid_a = await engine.storage.save_memory(uid_a, "secret A", emb)
    mid_b = await engine.storage.save_memory(uid_b, "secret B", emb)
    rows = await engine.storage.get_memories_by_ids([mid_a, mid_b], user_id=uid_a)
    texts = {r["text"] for r in rows}
    assert "secret A" in texts
    assert "secret B" not in texts


@pytest.mark.asyncio
async def test_capture_ignores_foreign_session_checkpoint(engine, monkeypatch):
    sk = "sess-shared-key"
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 99)
    await engine.capture(
        "test-cap-a",
        sk,
        [{"role": "user", "content": "a1"}, {"role": "assistant", "content": "ok"}],
    )
    # Same session_key, different user — must not false-dedupe
    r = await engine.capture(
        "test-cap-b",
        sk,
        [{"role": "user", "content": "a1"}, {"role": "assistant", "content": "ok"}],
    )
    assert r["duplicate"] is False
    assert r["messages_captured"] == 2


@pytest.mark.asyncio
async def test_delete_user_data_clears_checkpoints(engine):
    # Do NOT call wipe_all here — shared live DB with real users.
    uid = "test-wipe-cp"
    await engine.capture(
        uid,
        "sess-wipe",
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    )
    assert await engine.storage.get_capture_checkpoint("sess-wipe") is not None
    await engine.storage.delete_user_data(uid)
    assert await engine.storage.get_capture_checkpoint("sess-wipe") is None


@pytest.mark.asyncio
async def test_wipe_all_sql_includes_checkpoints():
    import inspect
    from storage import Storage

    src = inspect.getsource(Storage.wipe_all)
    assert "capture_checkpoints" in src


@pytest.mark.asyncio
async def test_persona_failure_not_cached(engine, monkeypatch):
    uid = "test-persona-fail"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "User likes tea", emb)

    async def boom(memories):
        raise RuntimeError("llm down")

    monkeypatch.setattr(engine.extractor, "generate_persona", boom)
    r1 = await engine.get_persona(uid)
    assert "unavailable" in r1["summary"].lower() or "failed" in r1["summary"].lower()
    cached = await engine.storage.get_persona_cache(uid)
    assert cached is None


@pytest.mark.asyncio
async def test_trust_empty_user_is_trusted(engine):
    t = await engine.memory_trust("test-trust-empty")
    assert t["l0_count"] == 0
    assert t["l1_count"] == 0
    assert t["recall_trusted"] is True
    assert t["behind_watermark"] is False


@pytest.mark.asyncio
async def test_trust_l0_without_extract_untrusted(engine):
    uid = "test-trust-l0-only"
    await engine.add(
        uid,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
    )
    t = await engine.memory_trust(uid)
    assert t["l0_count"] >= 2
    assert t["last_extraction_at"] is None
    assert t["behind_watermark"] is True
    assert t["recall_trusted"] is False


@pytest.mark.asyncio
async def test_trust_after_successful_extract(engine, monkeypatch):
    uid = "test-trust-caught-up"

    async def facts(messages_text, existing_memories=None):
        return [{"content": "User said hello", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)
    r = await engine.add(
        uid,
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    )
    assert r["extract_status"] == "ok"
    t = await engine.memory_trust(uid)
    assert t["last_extract_ok"] is True
    assert t["l1_count"] >= 1
    assert t["recall_trusted"] is True
    assert t["behind_watermark"] is False


@pytest.mark.asyncio
async def test_trust_lag_exceeded_untrusted(engine, monkeypatch):
    uid = "test-trust-lag"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "User likes cats", emb)
    # Successful extract watermark in the past, then newer L0.
    await engine.storage.mark_extraction_success(uid)
    async with engine.storage._pool.acquire() as conn:
        await conn.execute(
            """UPDATE extraction_state
               SET last_extraction_at = $2
               WHERE user_id = $1""",
            uid,
            datetime.now(timezone.utc) - timedelta(hours=2),
        )
    await engine.storage.save_conversation(uid, "user", "new fact after lag")
    monkeypatch.setattr("memory.EXTRACTION_MAX_LAG_SECONDS", 3600)
    t = await engine.memory_trust(uid)
    assert t["behind_watermark"] is True
    assert t["extraction_lag_exceeded"] is True
    assert t["recall_trusted"] is False


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
async def test_capture_writes_l0_and_dedupes_batch(engine):
    uid = "test-capture-dedupe"
    sk = "sess-capture-1"
    messages = [
        {"role": "user", "content": "remember I like dark mode"},
        {"role": "assistant", "content": "got it"},
    ]
    first = await engine.capture(uid, sk, messages, agent_id="claude-code")
    assert first["messages_captured"] == 2
    assert first["duplicate"] is False
    assert first["messages_seen"] == 2

    again = await engine.capture(uid, sk, messages, agent_id="claude-code")
    assert again["messages_captured"] == 0
    assert again["duplicate"] is True
    assert again["messages_seen"] == 2

    convs = await engine.storage.get_recent_conversations(uid)
    assert len(convs) == 2

    nxt = await engine.capture(
        uid,
        sk,
        [
            {"role": "user", "content": "also prefer terse replies"},
            {"role": "assistant", "content": "noted"},
        ],
    )
    assert nxt["messages_captured"] == 2
    assert nxt["messages_seen"] == 4
    assert len(await engine.storage.get_recent_conversations(uid)) == 4


@pytest.mark.asyncio
async def test_search_empty(engine):
    uid = "test-search-empty"
    payload = await engine.search(uid, "anything", top_k=5)
    assert payload["results"] == []
    assert payload["trust"]["user_id"] == uid
    assert payload["trust"]["l0_count"] == 0


@pytest.mark.asyncio
async def test_scenarios_empty(engine):
    uid = "test-scenarios-empty"
    result = await engine.get_scenarios(uid)
    assert result["total"] == 0
    assert result["scenarios"] == []


@pytest.mark.asyncio
async def test_user_id_canonicalized(engine):
    await engine.add(
        "TeSt-Canon-User",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
    )
    convs = await engine.storage.get_recent_conversations("test-canon-user")
    assert len(convs) == 2
    payload = await engine.search("TEST-CANON-USER", "hi", top_k=3)
    assert payload["trust"]["user_id"] == "test-canon-user"


@pytest.mark.asyncio
async def test_priority_and_instruction_recall(engine):
    uid = "test-priority-recall"
    emb = engine.embedder.embed(["x"])[0]

    # Identical embeddings -> ranking decided by priority tilt.
    await engine.storage.save_memory(uid, "Low priority fact", emb, priority=10)
    await engine.storage.save_memory(uid, "High priority fact", emb, priority=95)
    await engine.storage.save_memory(uid, "Never use emojis", emb, mem_type="instruction", priority=90)

    results = (await engine.search(uid, "fact", top_k=10))["results"]
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
    tight = (await engine.search(uid, "widgets", top_k=1))["results"]
    assert len(tight) == 1
    assert "widgets" in tight[0]["text"]

    # top_k large enough: instruction fills remaining slot below primary score.
    wide = (await engine.search(uid, "widgets", top_k=5))["results"]
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
    r3 = await engine.add(uid, pair(3))
    assert calls["n"] == 1
    assert r3["extract_status"] == "ok"

    # Assistant-only does not bump / extract
    await engine.add(uid, [{"role": "assistant", "content": "lonely assistant"}])
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_extract_failure_does_not_reset_counter(engine, monkeypatch):
    uid = "test-extract-fail-open"
    calls = {"n": 0}

    async def boom(messages_text, existing_memories=None):
        calls["n"] += 1
        raise RuntimeError("llm down")

    monkeypatch.setattr(engine.extractor, "extract_memories", boom)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 2)

    pair = lambda i: [
        {"role": "user", "content": f"u{i}"},
        {"role": "assistant", "content": f"a{i}"},
    ]
    r1 = await engine.add(uid, pair(1))
    assert r1["extract_status"] == "skipped"
    r2 = await engine.add(uid, pair(2))
    assert calls["n"] == 1
    assert r2["extract_status"] == "failed"
    state = await engine.storage.get_extraction_state(uid)
    assert state["conversations_seen"] == 2  # not reset
    assert state["last_extract_ok"] is False

    # Next turn retries because counter still due
    await engine.add(uid, pair(3))
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_extract_empty_facts_resets_counter(engine, monkeypatch):
    uid = "test-extract-empty-ok"
    async def none_facts(messages_text, existing_memories=None):
        return []

    monkeypatch.setattr(engine.extractor, "extract_memories", none_facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)
    r = await engine.add(
        uid,
        [{"role": "user", "content": "hello only"}, {"role": "assistant", "content": "hi"}],
    )
    assert r["extract_status"] == "ok_no_facts"
    state = await engine.storage.get_extraction_state(uid)
    assert state["conversations_seen"] == 0
    assert state["last_extract_ok"] is True


@pytest.mark.asyncio
async def test_scenario_upsert_merges_not_wipes(engine, monkeypatch):
    uid = "test-scenario-merge"
    emb = engine.embedder.embed(["x"])[0]
    m1 = await engine.storage.save_memory(uid, "User likes Rust", emb)
    m2 = await engine.storage.save_memory(uid, "User likes Go", emb)
    await engine.storage.save_scenario(uid, "Languages", "Programming languages", emb, [m1])
    await engine.storage.save_scenario(uid, "KeepMe", "Should survive", emb, [m1])

    async def group(memory_texts, existing_scenarios=None):
        # Only emit Languages — KeepMe must remain under upsert-by-name.
        # Indices are 0-based (post group_into_scenarios normalization).
        return [
            {
                "name": "Languages",
                "description": "Updated langs",
                "memory_indices": list(range(len(memory_texts))),
            }
        ]

    monkeypatch.setattr(engine.extractor, "group_into_scenarios", group)
    await engine._rebuild_scenarios(uid)
    scenarios = await engine.storage.get_all_scenarios(uid)
    names = {s["name"] for s in scenarios}
    assert "KeepMe" in names
    lang = next(s for s in scenarios if s["name"] == "Languages")
    assert set(lang["memory_ids"]) >= {m1, m2}


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


@pytest.mark.asyncio
async def test_cold_extract_does_not_orphan_past_window(engine, monkeypatch):
    """Seeding > cold_limit L0 then forcing extract must mine all windows.

    Regression: mark_extraction_success(now) after ASC LIMIT left newer/unmined
    history with created_at <= now permanently outside created_at > since.
    """
    uid = "test-cold-no-orphan"
    windows: list[list[str]] = []

    async def track_extract(messages_text, existing_memories=None):
        # One marker per conversation line so we can count coverage.
        lines = [ln for ln in messages_text.splitlines() if ln.strip()]
        windows.append([ln.split(": ", 1)[-1] for ln in lines])
        return [
            {"content": f"fact-from-{lines[0][:40]}", "type": "episodic", "priority": 50}
        ]

    monkeypatch.setattr(engine.extractor, "extract_memories", track_extract)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    # cold_limit = max(1*4, 40) = 40. Seed 45 user turns (+ assistants) via L0
    # before any extract state, then one add that trips extraction.
    n_seed = 45
    for i in range(n_seed):
        await engine.storage.save_conversation(uid, "user", f"seed-user-{i:03d}")
        await engine.storage.save_conversation(uid, "assistant", f"seed-asst-{i:03d}")

    r = await engine.add(
        uid,
        [
            {"role": "user", "content": "trigger-extract"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    assert r["extract_status"] in ("ok", "ok_no_facts")
    assert len(windows) >= 2  # more than one page under cold_limit=40

    mined = {c for w in windows for c in w}
    for i in range(n_seed):
        assert f"seed-user-{i:03d}" in mined
        assert f"seed-asst-{i:03d}" in mined
    assert "trigger-extract" in mined

    state = await engine.storage.get_extraction_state(uid)
    assert state is not None
    assert state["last_extract_ok"] is True
    latest = await engine.storage.latest_conversation_at(uid)
    # Watermark is max mined created_at, not wall-clock-ahead of L0.
    assert state["last_extraction_at"] is not None
    assert latest is not None
    # Equal or slightly behind latest is fine; must not leave rows with
    # created_at > watermark unmined after a successful full catch-up.
    remaining = await engine.storage.get_conversations_since(
        uid, since=state["last_extraction_at"], limit=500
    )
    assert remaining == []
