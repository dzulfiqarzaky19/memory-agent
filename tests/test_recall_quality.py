"""Recall ranking under real vector geometry (Chunk E).

FakeEmbedding gives every row an identical vector, so cosine order is constant and
none of this is exercised by the rest of the suite. SeededEmbedding provides
deterministic, distinct, L2-normalized vectors — model-free but geometrically real.
"""

from __future__ import annotations

import pytest


async def _ready(engine, uid: str) -> None:
    await engine.storage.mark_extraction_success(uid)


async def _save(engine, uid: str, text: str, **kw) -> str:
    emb = (await engine.embedder.aembed([text]))[0]
    return await engine.storage.save_memory(uid, text, emb, **kw)


@pytest.mark.asyncio
async def test_vector_search_ranks_by_similarity(seeded_engine, monkeypatch):
    """Closer vectors rank higher — the core claim FakeEmbedding cannot test."""
    eng = seeded_engine
    uid = "test-rq-vector"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "vector")
    query = "which editor do I use"

    eng.embedder.anchor("User edits code in Neovim", near=query, similarity=0.95)
    eng.embedder.anchor("User drinks oat milk lattes", near=query, similarity=0.35)

    await _save(eng, uid, "User edits code in Neovim")
    await _save(eng, uid, "User drinks oat milk lattes")
    await _ready(eng, uid)

    results = (await eng.search(uid, query, top_k=5))["results"]
    texts = [r["text"] for r in results]
    assert texts[0] == "User edits code in Neovim"
    assert results[0]["score"] > results[-1]["score"]


@pytest.mark.asyncio
async def test_similarity_threshold_excludes_distant_rows(seeded_engine, monkeypatch):
    eng = seeded_engine
    uid = "test-rq-threshold"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "vector")
    monkeypatch.setattr("memory.RECALL_SIMILARITY_THRESHOLD", 0.8)
    query = "deployment schedule"

    eng.embedder.anchor("Deploys happen on Tuesdays", near=query, similarity=0.93)
    eng.embedder.anchor("Cat is named Mochi", near=query, similarity=0.10)

    await _save(eng, uid, "Deploys happen on Tuesdays")
    await _save(eng, uid, "Cat is named Mochi")
    await _ready(eng, uid)

    texts = [r["text"] for r in (await eng.search(uid, query, top_k=5))["results"]]
    assert "Deploys happen on Tuesdays" in texts
    assert "Cat is named Mochi" not in texts


@pytest.mark.asyncio
async def test_hybrid_keeps_keyword_only_and_vector_only_hits(seeded_engine, monkeypatch):
    """RRF must not let one retriever starve the other."""
    eng = seeded_engine
    uid = "test-rq-hybrid"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "hybrid")
    query = "kubernetes"

    # Shares the query term but sits far away in vector space.
    eng.embedder.anchor("Runs kubernetes at work", near=query, similarity=0.05)
    # No shared term, but semantically adjacent.
    eng.embedder.anchor("Manages container orchestration", near=query, similarity=0.92)

    await _save(eng, uid, "Runs kubernetes at work")
    await _save(eng, uid, "Manages container orchestration")
    await _save(eng, uid, "Unrelated trivia about houseplants")
    await _ready(eng, uid)

    texts = [r["text"] for r in (await eng.search(uid, query, top_k=5))["results"]]
    assert "Runs kubernetes at work" in texts
    assert "Manages container orchestration" in texts


@pytest.mark.asyncio
async def test_priority_tilt_breaks_similarity_ties(seeded_engine, monkeypatch):
    eng = seeded_engine
    uid = "test-rq-priority"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "vector")
    query = "coffee preference"

    eng.embedder.anchor("Drinks espresso black", near=query, similarity=0.90)
    eng.embedder.anchor("Drinks filter coffee black", near=query, similarity=0.90)

    await _save(eng, uid, "Drinks espresso black", priority=95)
    await _save(eng, uid, "Drinks filter coffee black", priority=10)
    await _ready(eng, uid)

    texts = [r["text"] for r in (await eng.search(uid, query, top_k=5))["results"]]
    assert texts.index("Drinks espresso black") < texts.index("Drinks filter coffee black")


@pytest.mark.asyncio
async def test_conflict_demote_prefers_newer_under_real_geometry(seeded_engine, monkeypatch):
    """ADD-only newer-wins must survive distinct vectors, not just identical ones."""
    eng = seeded_engine
    uid = "test-rq-conflict"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "vector")
    query = "which editor"
    old, new = "User uses VS Code daily", "User uses Neovim daily"

    eng.embedder.anchor(old, near=query, similarity=0.91)
    eng.embedder.anchor(new, near=query, similarity=0.90)  # slightly worse raw match

    old_id = await _save(eng, uid, old)
    await _save(eng, uid, new)
    async with eng.storage._pool.acquire() as conn:
        await conn.execute(
            "UPDATE memories SET created_at = now() - interval '120 days' WHERE id = $1",
            old_id,
        )
    await _ready(eng, uid)

    results = (await eng.search(uid, query, top_k=5))["results"]
    texts = [r["text"] for r in results]
    assert texts.index(new) < texts.index(old)


@pytest.mark.asyncio
async def test_injects_never_displace_matched_primaries(seeded_engine, monkeypatch):
    eng = seeded_engine
    uid = "test-rq-injects"
    monkeypatch.setattr("memory.RECALL_STRATEGY", "vector")
    query = "billing question"

    eng.embedder.anchor("Invoices are sent monthly", near=query, similarity=0.94)
    await _save(eng, uid, "Invoices are sent monthly")
    await _save(eng, uid, "Never use emojis", mem_type="instruction", priority=99)
    await _ready(eng, uid)

    tight = (await eng.search(uid, query, top_k=1))["results"]
    assert len(tight) == 1
    assert tight[0]["text"] == "Invoices are sent monthly"

    wide = (await eng.search(uid, query, top_k=5))["results"]
    texts = [r["text"] for r in wide]
    assert texts[0] == "Invoices are sent monthly"
    assert "Never use emojis" in texts
    primary = wide[0]
    injected = next(r for r in wide if r["text"] == "Never use emojis")
    assert injected["score"] < primary["score"]


@pytest.mark.asyncio
async def test_seeded_embedding_is_deterministic_and_distinct(seeded_engine):
    eng = seeded_engine
    a1, b1 = await eng.embedder.aembed(["alpha", "beta"])
    a2, _ = await eng.embedder.aembed(["alpha", "beta"])
    assert a1 == a2  # deterministic
    assert a1 != b1  # distinct — the whole point vs FakeEmbedding
    assert abs(sum(x * x for x in a1) - 1.0) < 1e-9  # normalized
