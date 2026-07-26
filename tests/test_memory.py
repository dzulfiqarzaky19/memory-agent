from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


async def _add_and_drain(engine, uid: str, messages: list[dict]) -> dict:
    """add() now only queues. Drain synchronously so extraction is assertable."""
    r = await engine.add(uid, messages)
    if r["extract_status"] == "queued":
        return await engine.run_extraction(uid)
    return r


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

    r = await _add_and_drain(
        engine,
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

    r = await _add_and_drain(
        engine,
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
        uid,
        since=state["last_extraction_at"],
        limit=50,
        after_id=state.get("last_extraction_id"),
    )
    assert leftover == []


async def _trust_ready(engine, uid: str) -> None:
    """Mark extract OK so seeded L1 reads as caught-up (stale=false) recall."""
    await engine.storage.mark_extraction_success(uid)


def test_recency_multiplier_newer_higher():
    from memory import _recency_multiplier

    now = datetime.now(timezone.utc)
    old = now - timedelta(days=60)
    m_new = _recency_multiplier(now, half_life_days=30)
    m_old = _recency_multiplier(old, half_life_days=30)
    assert m_new > m_old
    assert _recency_multiplier(now, half_life_days=0) == 1.0


def test_conflict_demotes_older_duplicate():
    from memory import _apply_recency_and_conflict

    now = datetime.now(timezone.utc)
    rows = [
        {
            "id": "old",
            "text": "User prefers vim editor daily",
            "score": 1.0,
            "created_at": now - timedelta(days=90),
            "metadata": {},
        },
        {
            "id": "new",
            "text": "User prefers neovim editor daily",
            "score": 1.0,
            "created_at": now,
            "metadata": {},
        },
    ]
    out = _apply_recency_and_conflict(
        rows, half_life_days=0, jaccard_thresh=0.3, demote=0.5
    )
    by_id = {r["id"]: r for r in out}
    assert by_id["new"]["score"] > by_id["old"]["score"]
    assert by_id["old"]["metadata"].get("_conflict_demoted") is True
    assert by_id["old"]["metadata"].get("_superseded_by") == "new"
    assert out[0]["id"] == "new"


@pytest.mark.asyncio
async def test_search_prefers_newer_on_conflict(engine, monkeypatch):
    uid = "test-recency-conflict"
    emb = engine.embedder.embed(["x"])[0]
    monkeypatch.setattr("memory.RECALL_STRATEGY", "keyword")
    monkeypatch.setattr("memory.RECALL_RECENCY_HALF_LIFE_DAYS", 0)
    monkeypatch.setattr("memory.RECALL_CONFLICT_JACCARD", 0.3)
    monkeypatch.setattr("memory.RECALL_CONFLICT_DEMOTE", 0.5)

    old_id = await engine.storage.save_memory(
        uid, "User prefers the vim text editor", emb, priority=50
    )
    new_id = await engine.storage.save_memory(
        uid, "User prefers the neovim text editor", emb, priority=50
    )
    # Backdate older row so conflict demote can see age.
    async with engine.storage._pool.acquire() as conn:
        await conn.execute(
            "UPDATE memories SET created_at = $2 WHERE id = $1",
            old_id,
            datetime.now(timezone.utc) - timedelta(days=60),
        )
        await conn.execute(
            "UPDATE memories SET created_at = $2 WHERE id = $1",
            new_id,
            datetime.now(timezone.utc),
        )
    await _trust_ready(engine, uid)

    results = (await engine.search(uid, "editor prefers", top_k=5))["results"]
    texts = [r["text"] for r in results]
    assert any("neovim" in t for t in texts)
    if any("vim" in t and "neovim" not in t for t in texts):
        # Older near-dup still returned (ADD-only) but ranked below newer.
        vim_i = next(i for i, t in enumerate(texts) if "vim" in t and "neovim" not in t)
        neo_i = next(i for i, t in enumerate(texts) if "neovim" in t)
        assert neo_i < vim_i


@pytest.mark.asyncio
async def test_get_memories_by_ids_requires_user_id(engine):
    uid = "test-ids-req"
    emb = engine.embedder.embed(["x"])[0]
    mid = await engine.storage.save_memory(uid, "needs scope", emb)
    with pytest.raises(TypeError):
        await engine.storage.get_memories_by_ids([mid])  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        await engine.storage.get_memories_by_ids([mid], user_id="")


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
    r = await _add_and_drain(
        engine,
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
async def test_untrusted_search_serves_stale_not_empty(engine):
    """Degraded recall: untrusted serves real L1 with a loud banner, never amnesia."""
    uid = "test-untrusted-stale"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "Should still surface when stale", emb)
    # L1 present but no successful extract → untrusted, yet results must flow.
    payload = await engine.search(uid, "surface", top_k=5)
    assert payload["trust"]["recall_trusted"] is False
    assert payload["stale"] is True
    assert len(payload["results"]) == 1
    assert payload["results"][0]["text"] == "Should still surface when stale"


@pytest.mark.asyncio
async def test_failed_extract_still_serves_recall(engine):
    """A single LLM failure must not latch recall off."""
    uid = "test-stale-failed"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "Survives an extract failure", emb)
    await engine.storage.mark_extraction_success(uid)
    await engine.storage.mark_extraction_failure(uid, "llm exploded")

    payload = await engine.search(uid, "survives", top_k=5)
    assert payload["stale"] is True
    assert len(payload["results"]) == 1
    assert payload["trust"]["last_extract_error"] == "llm exploded"


@pytest.mark.asyncio
async def test_no_l1_is_hard_empty(engine):
    """Nothing to serve is still empty — degraded mode does not invent rows."""
    uid = "test-stale-nol1"
    await engine.storage.save_conversation(uid, "user", "hello")
    payload = await engine.search(uid, "hello", top_k=5)
    assert payload["results"] == []
    assert payload["stale"] is True


@pytest.mark.asyncio
async def test_stale_seconds_measures_unmined_l0(engine):
    uid = "test-stale-secs"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "anything", emb)
    # Watermark 2h behind the newest L0 row.
    await engine.storage.save_conversation(uid, "user", "recent turn")
    await engine.storage.mark_extraction_success(
        uid, watermark_at=datetime.now(timezone.utc) - timedelta(hours=2)
    )
    trust = await engine.memory_trust(uid)
    assert 7000 < trust["stale_seconds"] < 7400

    payload = await engine.search(uid, "anything", top_k=5)
    assert payload["stale_seconds"] == trust["stale_seconds"]


@pytest.mark.asyncio
async def test_caught_up_is_not_stale(engine):
    uid = "test-stale-caughtup"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "caught up fact", emb)
    await engine.storage.mark_extraction_success(uid)
    payload = await engine.search(uid, "caught", top_k=5)
    assert payload["stale"] is False
    assert payload["stale_seconds"] == 0


@pytest.mark.asyncio
async def test_empty_window_keeps_prior_watermark(engine, monkeypatch):
    uid = "test-empty-wm"
    prior = datetime.now(timezone.utc) - timedelta(days=1)
    await engine.storage.mark_extraction_success(uid, watermark_at=prior)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    async def no_pending(user_id, since=None, limit=200, after_id=None):
        return []

    monkeypatch.setattr(engine.storage, "get_conversations_since", no_pending)
    r = await _add_and_drain(
        engine,
        uid,
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
    )
    assert r["extract_status"] == "empty_window"
    state = await engine.storage.get_extraction_state(uid)
    # Must not jump to wall-clock now()
    assert state["last_extraction_at"] is not None
    got = state["last_extraction_at"]
    if got.tzinfo is None:
        got = got.replace(tzinfo=timezone.utc)
    assert abs((got - prior).total_seconds()) < 2


@pytest.mark.asyncio
async def test_priority_and_instruction_recall(engine):
    uid = "test-priority-recall"
    emb = engine.embedder.embed(["x"])[0]

    # Identical embeddings -> ranking decided by priority tilt.
    await engine.storage.save_memory(uid, "Low priority fact", emb, priority=10)
    await engine.storage.save_memory(uid, "High priority fact", emb, priority=95)
    await engine.storage.save_memory(uid, "Never use emojis", emb, mem_type="instruction", priority=90)
    await _trust_ready(engine, uid)

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
    await _trust_ready(engine, uid)

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
    await _add_and_drain(engine, uid, pair(1))
    await _add_and_drain(engine, uid, pair(2))
    assert calls["n"] == 0

    # 3rd user turn triggers extract once
    r3 = await _add_and_drain(engine, uid, pair(3))
    assert calls["n"] == 1
    assert r3["extract_status"] == "ok"

    # Assistant-only does not bump / extract
    await _add_and_drain(engine, uid, [{"role": "assistant", "content": "lonely assistant"}])
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
    assert r2["extract_status"] == "queued"
    # run_extraction raises so the queue can retry; state must stay due.
    with pytest.raises(RuntimeError):
        await engine.run_extraction(uid)
    assert calls["n"] == 1
    state = await engine.storage.get_extraction_state(uid)
    assert state["conversations_seen"] == 2  # not reset
    assert state["last_extract_ok"] is False

    # Next turn retries because counter still due
    await engine.add(uid, pair(3))
    with pytest.raises(RuntimeError):
        await engine.run_extraction(uid)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_extract_empty_facts_resets_counter(engine, monkeypatch):
    uid = "test-extract-empty-ok"
    async def none_facts(messages_text, existing_memories=None):
        return []

    monkeypatch.setattr(engine.extractor, "extract_memories", none_facts)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)
    r = await _add_and_drain(
        engine,
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

    r = await _add_and_drain(
        engine,
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
        uid,
        since=state["last_extraction_at"],
        limit=500,
        after_id=state.get("last_extraction_id"),
    )
    assert remaining == []


@pytest.mark.asyncio
async def test_extract_mid_pagination_failure_retries_remainder(engine, monkeypatch):
    """Page1 success must not zero cadence; page2 fail must re-mine remainder next add."""
    uid = "test-mid-page-fail"
    calls = {"n": 0, "texts": []}

    async def facts_then_boom(messages_text, existing_memories=None):
        calls["n"] += 1
        calls["texts"].append(messages_text)
        if calls["n"] == 1:
            return [{"content": "fact-page-1", "type": "episodic", "priority": 50}]
        if calls["n"] == 2:
            raise RuntimeError("llm down mid-pagination")
        return [{"content": f"fact-page-{calls['n']}", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts_then_boom)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    # cold_limit=40 → seed 50 user L0 so page2 is required.
    n_seed = 50
    for i in range(n_seed):
        await engine.storage.save_conversation(uid, "user", f"seed-user-{i:03d}")

    await engine.add(
        uid,
        [
            {"role": "user", "content": "trigger-1"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    with pytest.raises(RuntimeError):
        await engine.run_extraction(uid)
    assert calls["n"] == 2
    state1 = await engine.storage.get_extraction_state(uid)
    assert state1["last_extract_ok"] is False
    # Intermediate advance keeps counter due (not zeroed by page1 success).
    assert state1["conversations_seen"] >= 1
    assert state1["last_extraction_at"] is not None
    assert state1["last_extraction_id"] is not None

    r2 = await _add_and_drain(
        engine,
        uid,
        [
            {"role": "user", "content": "trigger-2"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    assert r2["extract_status"] in ("ok", "ok_no_facts")
    assert calls["n"] >= 3

    blob = "\n".join(calls["texts"])
    # Remainder of cold backlog must appear on retry (not only failure flags).
    assert "seed-user-049" in blob
    assert "trigger-2" in blob

    state2 = await engine.storage.get_extraction_state(uid)
    assert state2["last_extract_ok"] is True
    assert state2["conversations_seen"] == 0
    leftover = await engine.storage.get_conversations_since(
        uid,
        since=state2["last_extraction_at"],
        limit=500,
        after_id=state2.get("last_extraction_id"),
    )
    assert leftover == []


@pytest.mark.asyncio
async def test_extract_equal_ts_keyset_survives_mid_fail(engine, monkeypatch):
    """Durable last_extraction_id drains same-timestamp L0 across crash/resume."""
    uid = "test-equal-ts-keyset"
    calls = {"n": 0, "texts": []}
    fixed_ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    async def facts_then_boom(messages_text, existing_memories=None):
        calls["n"] += 1
        calls["texts"].append(messages_text)
        if calls["n"] == 1:
            return [{"content": "fact-eq-1", "type": "episodic", "priority": 50}]
        if calls["n"] == 2:
            raise RuntimeError("crash after page1")
        return [{"content": f"fact-eq-{calls['n']}", "type": "episodic", "priority": 50}]

    monkeypatch.setattr(engine.extractor, "extract_memories", facts_then_boom)
    monkeypatch.setattr("memory.EXTRACTION_EVERY_N_TURNS", 1)

    # 45 rows share one created_at so created_at-only resume would orphan the tail.
    n_seed = 45
    for i in range(n_seed):
        rid = await engine.storage.save_conversation(uid, "user", f"eq-user-{i:03d}")
        async with engine.storage._pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET created_at = $2 WHERE id = $1",
                rid,
                fixed_ts,
            )

    await engine.add(
        uid,
        [
            {"role": "user", "content": "eq-trigger-1"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    with pytest.raises(RuntimeError):
        await engine.run_extraction(uid)
    state1 = await engine.storage.get_extraction_state(uid)
    assert state1["last_extraction_id"] is not None
    # created_at-only resume skips same-ts siblings; keyset must still see them.
    bare = await engine.storage.get_conversations_since(
        uid, since=state1["last_extraction_at"], limit=500
    )
    keyed = await engine.storage.get_conversations_since(
        uid,
        since=state1["last_extraction_at"],
        limit=500,
        after_id=state1["last_extraction_id"],
    )
    bare_eq = [c for c in bare if c["content"].startswith("eq-user-")]
    keyed_eq = [c for c in keyed if c["content"].startswith("eq-user-")]
    assert bare_eq == []
    assert len(keyed_eq) > 0

    r2 = await _add_and_drain(
        engine,
        uid,
        [
            {"role": "user", "content": "eq-trigger-2"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    assert r2["extract_status"] in ("ok", "ok_no_facts")
    blob = "\n".join(calls["texts"])
    # UUID id order ≠ insert order; require full equal-ts set across pages.
    for i in range(n_seed):
        assert f"eq-user-{i:03d}" in blob
    # At least one equal-ts row only appears after the failed page (retry path).
    page1_blob = calls["texts"][0]
    rest_blob = "\n".join(calls["texts"][1:])
    only_after = [
        f"eq-user-{i:03d}"
        for i in range(n_seed)
        if f"eq-user-{i:03d}" not in page1_blob
    ]
    assert only_after, "expected some equal-ts rows beyond cold_limit page1"
    assert any(m in rest_blob for m in only_after)

    state2 = await engine.storage.get_extraction_state(uid)
    leftover = await engine.storage.get_conversations_since(
        uid,
        since=state2["last_extraction_at"],
        limit=500,
        after_id=state2.get("last_extraction_id"),
    )
    assert leftover == []
