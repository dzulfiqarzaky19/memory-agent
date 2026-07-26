from __future__ import annotations

import os
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


# Live DB is shared with the sidecar. Tests MUST only touch user_id test-* and
# wipe it before AND after every test so persona/recall never sees fixture junk.
_TEST_TABLES = (
    "memories",
    "conversations",
    "scenarios",
    "extraction_state",
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
async def client(monkeypatch):
    # Lifespan runs a live embed dim-check; keep unit tests offline.
    monkeypatch.setattr(
        "server.create_embedding_provider",
        lambda: FakeEmbedding(),
    )
    with TestClient(app) as c:
        yield c
