from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import asyncpg

from config import DATABASE_URL, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    agent_id    TEXT,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    embedding   vector({dims}),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(user_id, text_hash);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);

CREATE TABLE IF NOT EXISTS scenarios (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding   vector({dims}),
    memory_ids  JSONB DEFAULT '[]',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scen_user ON scenarios(user_id);
CREATE INDEX IF NOT EXISTS idx_scen_created ON scenarios(created_at);
"""


class Storage:
    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or DATABASE_URL
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL.replace("{dims}", str(EMBEDDING_DIMENSIONS)))
        logger.info("Storage initialized (pgvector schema ready)")

    async def close(self):
        if self._pool:
            await self._pool.close()

    # -- Conversations (L0) --

    async def save_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        row_id = str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversations (id, user_id, agent_id, role, content, metadata)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                row_id,
                user_id,
                agent_id,
                role,
                content,
                json.dumps(metadata or {}),
            )
        return row_id

    async def get_recent_conversations(
        self, user_id: str, limit: int = 20
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, role, content, metadata, created_at
                   FROM conversations
                   WHERE user_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                user_id,
                limit,
            )
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    # -- Memories (L1) --

    async def save_memory(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if await self.check_duplicate(user_id, text):
            return None
        row_id = str(uuid4())
        meta = metadata or {}
        meta["text_hash"] = text_hash
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memories (id, user_id, text, text_hash, embedding, metadata)
                   VALUES ($1, $2, $3, $4, $5::vector, $6)
                   """,
                row_id,
                user_id,
                text,
                text_hash,
                json.dumps(embedding),
                json.dumps(meta),
            )
        return row_id

    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, metadata, created_at,
                          1 - (embedding <=> $1::vector) AS similarity
                   FROM memories
                   WHERE user_id = $2
                     AND 1 - (embedding <=> $1::vector) >= $3
                   ORDER BY embedding <=> $1::vector
                   LIMIT $4""",
                json.dumps(query_embedding),
                user_id,
                threshold,
                top_k,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "score": round(float(r["similarity"]), 4),
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def keyword_search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.05,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, metadata, created_at,
                          similarity(text, $1) AS similarity
                   FROM memories
                   WHERE user_id = $2
                     AND similarity(text, $1) >= $3
                   ORDER BY similarity DESC
                   LIMIT $4""",
                query,
                user_id,
                threshold,
                top_k,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "score": round(float(r["similarity"]), 4),
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def hybrid_search_memories(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        vec_threshold: float = 0.3,
        kw_threshold: float = 0.05,
        rrf_k: int = 60,
    ) -> list[dict]:
        vector_results = await self.search_memories(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            threshold=vec_threshold,
        )
        keyword_results = await self.keyword_search_memories(
            user_id=user_id,
            query=query,
            top_k=top_k * 2,
            threshold=kw_threshold,
        )

        rrf_scores: dict[str, tuple[float, dict]] = {}
        for rank, r in enumerate(vector_results):
            rid = r["id"]
            score = 1.0 / (rrf_k + rank + 1)
            result = {
                "id": rid,
                "text": r["text"],
                "score": score,
                "metadata": r.get("metadata"),
                "created_at": r["created_at"],
            }
            result["metadata"] = dict(result["metadata"] or {})
            result["metadata"]["_vec_score"] = round(float(r["score"]), 4)
            rrf_scores[rid] = [score, result]

        for rank, r in enumerate(keyword_results):
            rid = r["id"]
            score = 1.0 / (rrf_k + rank + 1)
            if rid in rrf_scores:
                rrf_scores[rid][0] += score
                rrf_scores[rid][1]["metadata"]["_kw_score"] = round(float(r["score"]), 4)
            else:
                result = {
                    "id": rid,
                    "text": r["text"],
                    "score": score,
                    "metadata": dict(r.get("metadata") or {}),
                    "created_at": r["created_at"],
                }
                result["metadata"]["_kw_score"] = round(float(r["score"]), 4)
                rrf_scores[rid] = [score, result]

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [r[1] for r in sorted_results[:top_k]]

    async def count_memories(self, user_id: Optional[str] = None) -> int:
        async with self._pool.acquire() as conn:
            if user_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE user_id = $1", user_id
                )
            return await conn.fetchval("SELECT COUNT(*) FROM memories")

    async def get_all_memories(
        self, user_id: str, limit: int = 500
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, metadata, created_at
                   FROM memories
                   WHERE user_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                user_id,
                limit,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def check_duplicate(self, user_id: str, text: str) -> bool:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM memories WHERE user_id = $1 AND text_hash = $2)",
                user_id,
                text_hash,
            )

    # -- Scenarios (L2) --

    async def save_scenario(
        self,
        user_id: str,
        name: str,
        description: str,
        embedding: list[float],
        memory_ids: list[str],
        metadata: Optional[dict] = None,
    ) -> str:
        row_id = str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO scenarios (id, user_id, name, description, embedding, memory_ids, metadata)
                   VALUES ($1, $2, $3, $4, $5::vector, $6, $7)""",
                row_id,
                user_id,
                name,
                description,
                json.dumps(embedding),
                json.dumps(memory_ids),
                json.dumps(metadata or {}),
            )
        return row_id

    async def search_scenarios(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, description, memory_ids, metadata, created_at,
                          1 - (embedding <=> $1::vector) AS similarity
                   FROM scenarios
                   WHERE user_id = $2
                     AND 1 - (embedding <=> $1::vector) >= $3
                   ORDER BY embedding <=> $1::vector
                   LIMIT $4""",
                json.dumps(query_embedding),
                user_id,
                threshold,
                top_k,
            )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "memory_ids": json.loads(r["memory_ids"]) if isinstance(r["memory_ids"], str) else r["memory_ids"],
                "score": round(float(r["similarity"]), 4),
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def get_all_scenarios(self, user_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, description, memory_ids, metadata, created_at
                   FROM scenarios
                   WHERE user_id = $1
                   ORDER BY created_at DESC""",
                user_id,
            )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "memory_ids": json.loads(r["memory_ids"]) if isinstance(r["memory_ids"], str) else r["memory_ids"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def get_memories_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, metadata, created_at
                   FROM memories
                   WHERE id = ANY($1::text[])""",
                ids,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def count_scenarios(self, user_id: Optional[str] = None) -> int:
        async with self._pool.acquire() as conn:
            if user_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM scenarios WHERE user_id = $1", user_id
                )
            return await conn.fetchval("SELECT COUNT(*) FROM scenarios")

    # -- Cleanup --

    async def delete_user_data(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM conversations WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM scenarios WHERE user_id = $1", user_id)

    async def wipe_all(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE conversations, memories, scenarios")
