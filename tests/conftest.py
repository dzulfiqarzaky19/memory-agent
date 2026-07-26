from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from collections.abc import AsyncGenerator
from typing import Any

import pytest_asyncio
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import DATABASE_URL, EMBEDDING_DIMENSIONS
from embeddings import EmbeddingProvider
from extraction import LLMExtractor
from memory import MemoryEngine
from server import app
from storage import Storage


class FakeEmbedding(EmbeddingProvider):
    def __init__(self, dims: int | None = None):
        self._dims = dims if dims is not None else EMBEDDING_DIMENSIONS

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dims for _ in texts]


class SeededEmbedding(EmbeddingProvider):
    """Distinct, deterministic, L2-normalized vectors keyed by text content.

    FakeEmbedding returns one identical vector for everything, so cosine distance
    is constant and vector/RRF ordering is untestable. This gives real geometry
    without a live model: same text always maps to the same point, and `anchor`
    lets a test place a memory a controlled distance from a query.
    """

    def __init__(self, dims: int | None = None):
        self._dims = dims if dims is not None else EMBEDDING_DIMENSIONS
        self._anchors: dict[str, tuple[str, float]] = {}

    @property
    def dimensions(self) -> int:
        return self._dims

    def anchor(self, text: str, near: str, similarity: float) -> None:
        """Place `text` at a chosen cosine similarity to `near`."""
        self._anchors[text] = (near, similarity)

    def _raw(self, text: str) -> list[float]:
        rng = random.Random(hashlib.sha256(text.encode("utf-8")).hexdigest())
        v = [rng.gauss(0.0, 1.0) for _ in range(self._dims)]
        return _normalize(v)

    def _vector(self, text: str) -> list[float]:
        if text not in self._anchors:
            return self._raw(text)
        near, sim = self._anchors[text]
        base = self._raw(near)
        noise = self._raw(f"__noise__{text}")
        # Gram-Schmidt: strip the base component so the mix hits `sim` exactly.
        dot = sum(b * n for b, n in zip(base, noise))
        perp = _normalize([n - dot * b for b, n in zip(base, noise)])
        w = math.sqrt(max(0.0, 1.0 - sim * sim))
        return _normalize([sim * b + w * p for b, p in zip(base, perp)])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# Live DB is shared with the sidecar. Tests MUST only touch user_id test-* and
# wipe it before AND after every test so persona/recall never sees fixture junk.
_TEST_TABLES = (
    "memories",
    "conversations",
    "scenarios",
    "extraction_state",
    "extraction_jobs",
    "personas",
    "capture_checkpoints",
)


async def _wipe_test_users(store: Storage) -> None:
    async with store._pool.acquire() as conn:
        for table in _TEST_TABLES:
            await conn.execute(f"DELETE FROM {table} WHERE user_id LIKE 'test-%'")


@pytest_asyncio.fixture(scope="function")
async def db():
    store = Storage(DATABASE_URL)
    await store.initialize()
    await _wipe_test_users(store)
    try:
        yield store
    finally:
        try:
            await _wipe_test_users(store)
        finally:
            await store.close()


@pytest_asyncio.fixture(scope="function")
async def engine(db: Storage) -> AsyncGenerator[Any, Any]:
    embedder = FakeEmbedding()
    extractor = LLMExtractor()
    eng = MemoryEngine(storage=db, embedder=embedder, extractor=extractor)
    try:
        yield eng
    finally:
        # Engine may have written via storage after last fixture assert.
        await _wipe_test_users(db)


@pytest_asyncio.fixture(scope="function")
async def seeded_engine(db: Storage) -> AsyncGenerator[Any, Any]:
    """Engine with real vector geometry — for recall ranking assertions."""
    eng = MemoryEngine(
        storage=db, embedder=SeededEmbedding(), extractor=LLMExtractor()
    )
    try:
        yield eng
    finally:
        await _wipe_test_users(db)


@pytest_asyncio.fixture(scope="function")
async def client(monkeypatch):
    # Lifespan runs a live embed dim-check; keep unit tests offline.
    monkeypatch.setattr(
        "server.create_embedding_provider",
        lambda: FakeEmbedding(),
    )
    # Door off for unit tests — test_auth.py covers the locked path explicitly.
    monkeypatch.setattr("server.MEMORY_API_SECRET", "")
    with TestClient(app) as c:
        try:
            yield c
        finally:
            # Own pool: server.storage belongs to TestClient's event loop, not ours.
            cleaner = Storage(DATABASE_URL)
            await cleaner.initialize()
            try:
                await _wipe_test_users(cleaner)
            finally:
                await cleaner.close()
