"""Partner pack v1 — compose, isolation, and no-LLM-on-read (brief: partner-pack-v1)."""

from __future__ import annotations

import pytest

from config import EMBEDDING_DIMENSIONS, PARTNER_AGENT_ID


@pytest.mark.asyncio
async def test_cold_db_returns_structure_with_seed_self(engine):
    """Acceptance #2: `self` is non-empty even with nothing stored."""
    pack = await engine.get_partner("test-partner-cold")
    assert pack["user_id"] == "test-partner-cold"
    assert pack["agent_id"] == PARTNER_AGENT_ID
    assert pack["other"]["summary"] is None
    assert pack["other"]["instructions"] == []
    assert len(pack["self"]) > 0
    assert all(f["source"] == "seed" for f in pack["self"])
    assert "trust" in pack and "stale" in pack


@pytest.mark.asyncio
async def test_stored_self_delta_layers_on_seed(engine):
    uid = "test-partner-delta"
    r = await engine.add_partner_fact(
        uid, kind="agent_self", text="Prefers pgvector over pinecone", priority=88
    )
    assert r["duplicate"] is False

    pack = await engine.get_partner(uid)
    sources = {f["source"] for f in pack["self"]}
    assert sources == {"seed", "stored"}
    stored = [f for f in pack["self"] if f["source"] == "stored"]
    assert stored[0]["text"] == "Prefers pgvector over pinecone"
    # Sorted by priority regardless of origin.
    priorities = [f["priority"] for f in pack["self"]]
    assert priorities == sorted(priorities, reverse=True)


@pytest.mark.asyncio
async def test_duplicate_fact_is_noop(engine):
    uid = "test-partner-dupe"
    first = await engine.add_partner_fact(uid, kind="relation", text="same scar")
    second = await engine.add_partner_fact(uid, kind="relation", text="same scar")
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert await engine.storage.count_partner_facts(uid) == 1


@pytest.mark.asyncio
async def test_agent_id_selects_self_slice(engine):
    """Reads are scoped: one agent's spine never bleeds into another's pack.

    Rows are seeded via storage directly — the write API refuses non-main agents
    (see test_write_policy_rejects_non_main_agent).
    """
    uid = "test-partner-agents"
    await engine.storage.save_partner_fact(
        uid, "agent-a", "agent_self", "agent A private note"
    )
    await engine.storage.save_partner_fact(
        uid, "agent-b", "agent_self", "agent B private note"
    )

    a_texts = [f["text"] for f in (await engine.get_partner(uid, "agent-a"))["self"]]
    b_texts = [f["text"] for f in (await engine.get_partner(uid, "agent-b"))["self"]]
    assert "agent A private note" in a_texts
    assert "agent B private note" not in a_texts
    assert "agent B private note" in b_texts
    assert "agent A private note" not in b_texts


@pytest.mark.asyncio
async def test_write_policy_rejects_non_main_agent(engine):
    """Invariant 10: a spawn id must not be able to create its own spine."""
    uid = "test-partner-policy"
    with pytest.raises(ValueError, match="main shop agent"):
        await engine.add_partner_fact(
            uid, kind="agent_self", text="spawned spine", agent_id="flow-review-a3f9"
        )
    assert await engine.storage.count_partner_facts(uid) == 0

    # Omitted and explicit-main both land on the main shop agent.
    omitted = await engine.add_partner_fact(uid, kind="agent_self", text="from omit")
    explicit = await engine.add_partner_fact(
        uid, kind="relation", text="from explicit main", agent_id=PARTNER_AGENT_ID
    )
    assert omitted["agent_id"] == PARTNER_AGENT_ID
    assert explicit["agent_id"] == PARTNER_AGENT_ID
    assert await engine.storage.count_partner_facts(uid, PARTNER_AGENT_ID) == 2


@pytest.mark.asyncio
async def test_instructions_appear_under_other(engine):
    """Acceptance #3: a known instruction L1 shows up in the pack."""
    uid = "test-partner-instr"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(
        uid, "Always answer in English", emb, mem_type="instruction", priority=90
    )
    pack = await engine.get_partner(uid)
    texts = [i["text"] for i in pack["other"]["instructions"]]
    assert "Always answer in English" in texts


@pytest.mark.asyncio
async def test_persona_cache_surfaces_without_generating(engine):
    uid = "test-partner-persona"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "Some user fact", emb)
    await engine.storage.save_persona_cache(uid, "A senior engineer who ships.", 1)

    pack = await engine.get_partner(uid)
    assert pack["other"]["summary"] == "A senior engineer who ships."
    assert pack["other"]["memory_count"] == 1


@pytest.mark.asyncio
async def test_read_path_never_calls_llm(engine, monkeypatch):
    """Invariant 1: a SessionStart hook must not trigger persona generation."""
    uid = "test-partner-nollm"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "user fact without persona cache", emb)

    async def boom(*a, **kw):
        raise AssertionError("partner read must not call the LLM")

    monkeypatch.setattr(engine.extractor, "generate_persona", boom)
    monkeypatch.setattr(engine.extractor, "extract_memories", boom)

    pack = await engine.get_partner(uid)
    assert pack["other"]["summary"] is None  # cache miss, not generated
    assert len(pack["self"]) > 0


@pytest.mark.asyncio
async def test_partner_facts_do_not_pollute_user_l1(engine):
    """The regression this whole design exists to prevent.

    Partner facts must not reach user search, persona input, l1_count or trust.
    """
    uid = "test-partner-isolation"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(uid, "User likes strong coffee", emb)
    await engine.storage.mark_extraction_success(uid)

    before_count = await engine.storage.count_memories(uid)
    before_trust = await engine.memory_trust(uid)
    before_all = await engine.storage.get_all_memories(uid)

    await engine.add_partner_fact(
        uid, kind="agent_self", text="I verify before claiming done", priority=99
    )
    await engine.add_partner_fact(
        uid, kind="relation", text="We argued about ADD-only once", priority=99
    )

    assert await engine.storage.count_memories(uid) == before_count
    after_trust = await engine.memory_trust(uid)
    assert after_trust["l1_count"] == before_trust["l1_count"]
    assert after_trust["recall_trusted"] == before_trust["recall_trusted"]

    after_all = await engine.storage.get_all_memories(uid)
    assert len(after_all) == len(before_all)
    persona_input = " ".join(m["text"] for m in after_all)
    assert "verify before claiming" not in persona_input
    assert "ADD-only once" not in persona_input

    hits = (await engine.search(uid, "verify before claiming done", top_k=10))["results"]
    assert all("verify before claiming" not in h["text"] for h in hits)


@pytest.mark.asyncio
async def test_relation_fills_from_high_priority_episodic_read_only(engine):
    """Thin relation composes from user episodic — read-only, never dual-written."""
    uid = "test-partner-relfill"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(
        uid, "Decided to keep ADD-only storage", emb, mem_type="episodic", priority=90
    )
    await engine.storage.save_memory(
        uid, "Trivial aside nobody needs", emb, mem_type="episodic", priority=10
    )

    pack = await engine.get_partner(uid)
    filled = [f for f in pack["relation"] if f["source"] == "user_episodic"]
    assert any("ADD-only storage" in f["text"] for f in filled)
    assert all("Trivial aside" not in f["text"] for f in filled)
    # Composed at read time only — nothing copied into partner_facts.
    assert await engine.storage.count_partner_facts(uid) == 0


@pytest.mark.asyncio
async def test_stored_relation_suppresses_episodic_fill(engine):
    uid = "test-partner-relstored"
    emb = engine.embedder.embed(["x"])[0]
    await engine.storage.save_memory(
        uid, "High priority episodic thing", emb, mem_type="episodic", priority=95
    )
    await engine.add_partner_fact(uid, kind="relation", text="Real relation scar")

    pack = await engine.get_partner(uid)
    assert all(f["source"] != "user_episodic" for f in pack["relation"])


@pytest.mark.asyncio
async def test_bad_kind_rejected(engine):
    with pytest.raises(ValueError):
        await engine.add_partner_fact("test-partner-bad", kind="episodic", text="nope")
    with pytest.raises(ValueError):
        await engine.add_partner_fact("test-partner-bad", kind="agent_self", text="  ")


@pytest.mark.asyncio
async def test_user_id_canonicalized(engine):
    await engine.add_partner_fact("TEST-Partner-CANON", kind="relation", text="canon")
    pack = await engine.get_partner("test-partner-canon")
    assert pack["user_id"] == "test-partner-canon"
    assert any(f["source"] == "stored" for f in pack["relation"])


# -- HTTP surface --


def test_http_get_partner(client):
    r = client.get("/partner/test-partner-http")
    assert r.status_code == 200
    body = r.json()
    assert body["agent_id"] == PARTNER_AGENT_ID
    assert "self" in body and len(body["self"]) > 0  # wire name, not self_
    assert "other" in body and "relation" in body
    assert "trust" in body and "stale" in body


def test_http_post_then_get_roundtrip(client):
    uid = "test-partner-http-rt"
    r = client.post(
        f"/partner/{uid}",
        json={"kind": "relation", "text": "Ship small diffs", "priority": 80},
    )
    assert r.status_code == 200
    assert r.json()["duplicate"] is False

    body = client.get(f"/partner/{uid}").json()
    assert any(f["text"] == "Ship small diffs" for f in body["relation"])


def test_http_bad_kind_is_400(client):
    r = client.post(
        "/partner/test-partner-http-bad", json={"kind": "nonsense", "text": "x"}
    )
    assert r.status_code == 400


def test_http_oversized_text_is_422(client):
    r = client.post(
        "/partner/test-partner-http-big",
        json={"kind": "relation", "text": "x" * 40_000},
    )
    assert r.status_code == 422


def test_http_write_policy_rejects_spawn_agent(client):
    """Invariant 10 over HTTP: a spawn id cannot POST itself a spine."""
    uid = "test-partner-http-policy"
    r = client.post(
        f"/partner/{uid}",
        json={"kind": "agent_self", "text": "spawn spine", "agent_id": "flow-review-a3f9"},
    )
    assert r.status_code == 400
    assert "main shop agent" in r.json()["detail"]

    body = client.get(f"/partner/{uid}", params={"agent_id": "flow-review-a3f9"}).json()
    assert all(f["source"] != "stored" for f in body["self"])


def test_http_agent_id_query_param_scopes_reads(client):
    """Reads stay scoped by agent_id even though writes are pinned to main."""
    uid = "test-partner-http-agent"
    client.post(
        f"/partner/{uid}",
        json={"kind": "agent_self", "text": "written by main", "priority": 70},
    )
    main = client.get(f"/partner/{uid}", params={"agent_id": PARTNER_AGENT_ID}).json()
    other = client.get(f"/partner/{uid}", params={"agent_id": "some-other-agent"}).json()
    assert any(f["text"] == "written by main" for f in main["self"])
    assert all(f["text"] != "written by main" for f in other["self"])
