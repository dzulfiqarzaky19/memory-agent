from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator
from typing import Any

import pytest_asyncio
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import DATABASE_URL
from embeddings import EmbeddingProvider
from extraction import LLMExtractor
from memory import MemoryEngine
from server import app
from storage import Storage


class FakeEmbedding(EmbeddingProvider):
    def __init__(self, dims: int = 768):
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dims for _ in texts]


@pytest_asyncio.fixture(scope="function")
async def db():
    store = Storage(DATABASE_URL)
    await store.initialize()
    async with store._pool.acquire() as conn:
        for table in ("memories", "conversations", "scenarios", "extraction_state", "personas"):
            await conn.execute(f"DELETE FROM {table} WHERE user_id LIKE 'test-%'")
    yield store
    await store.close()


@pytest_asyncio.fixture(scope="function")
async def engine(db: Storage) -> AsyncGenerator[Any, Any]:
    embedder = FakeEmbedding()
    extractor = LLMExtractor()
    eng = MemoryEngine(storage=db, embedder=embedder, extractor=extractor)
    yield eng


@pytest_asyncio.fixture(scope="function")
async def client():
    with TestClient(app) as c:
        yield c
