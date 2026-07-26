

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import asyncpg

from config import DATABASE_URL, EMBEDDING_DIMENSIONS
from migrations import run_migrations

logger = logging.getLogger(__name__)


def _memory_row(r, score: float) -> dict:
    meta = r["metadata"]
    return {
        "id": r["id"],
        "text": r["text"],
        "score": score,
        "type": r["mem_type"],
        "priority": r["priority"],
        "metadata": json.loads(meta) if isinstance(meta, str) else meta,
        "created_at": r["created_at"],
    }


# Schema lives in migrations.py (versioned). Boot no longer executes DDL
# unconditionally — see run_migrations().


class Storage:
    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or DATABASE_URL
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        dims = EMBEDDING_DIMENSIONS
        async with self._pool.acquire() as conn:
            applied = await run_migrations(conn, dims)
            await self._ensure_embedding_dimensions(conn, dims)
        logger.info(
            "Storage initialized (dims=%s, migrations applied=%s)",
            dims,
            applied or "none",
        )

    async def _ensure_embedding_dimensions(self, conn, dims: int) -> None:
        """If vector columns were created at a different width, rebuild them (data wipe)."""
        for table in ("memories", "scenarios"):
            row = await conn.fetchrow(
                """
                SELECT atttypmod AS dims
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname = $1
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """,
                table,
            )
            if row is None:
                continue
            # pgvector stores typmod as dimensions
            current = int(row["dims"]) if row["dims"] is not None else None
            if current == dims:
                continue
            logger.warning(
                "Rebuilding %s.embedding: %s-d -> %s-d "
                "(vector column wiped; L1/L2 text kept but unsearchable until re-embed)",
                table,
                current,
                dims,
            )
            await conn.execute(f"DROP INDEX IF EXISTS idx_mem_embedding")
            await conn.execute(f"DROP INDEX IF EXISTS idx_scen_embedding")
            await conn.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS embedding")
            await conn.execute(
                f"ALTER TABLE {table} ADD COLUMN embedding vector({dims})"
            )
            # Flag rows so ops/tools can detect amnesia after dim change.
            if table == "memories":
                await conn.execute(
                    """UPDATE memories
                       SET metadata = COALESCE(metadata, '{}'::jsonb)
                           || '{"_embed_stale": true}'::jsonb"""
                )
            else:
                await conn.execute(
                    """UPDATE scenarios
                       SET metadata = COALESCE(metadata, '{}'::jsonb)
                           || '{"_embed_stale": true}'::jsonb"""
                )
        # Recreate HNSW indexes if missing after rebuild
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mem_embedding ON memories
                USING hnsw (embedding vector_cosine_ops)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scen_embedding ON scenarios
                USING hnsw (embedding vector_cosine_ops)
            """
        )

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

    async def get_capture_checkpoint(self, session_key: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT session_key, user_id, messages_seen, last_batch_hash, updated_at
                   FROM capture_checkpoints WHERE session_key = $1""",
                session_key,
            )
        if row is None:
            return None
        return {
            "session_key": row["session_key"],
            "user_id": row["user_id"],
            "messages_seen": row["messages_seen"],
            "last_batch_hash": row["last_batch_hash"],
            "updated_at": row["updated_at"],
        }

    async def upsert_capture_checkpoint(
        self,
        session_key: str,
        user_id: str,
        messages_seen: int,
        last_batch_hash: Optional[str],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO capture_checkpoints
                       (session_key, user_id, messages_seen, last_batch_hash, updated_at)
                   VALUES ($1, $2, $3, $4, now())
                   ON CONFLICT (session_key) DO UPDATE SET
                       user_id = EXCLUDED.user_id,
                       messages_seen = EXCLUDED.messages_seen,
                       last_batch_hash = EXCLUDED.last_batch_hash,
                       updated_at = now()""",
                session_key,
                user_id,
                messages_seen,
                last_batch_hash,
            )

    async def capture_atomic(
        self,
        *,
        session_key: str,
        user_id: str,
        messages: list[dict],
        batch_hash: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        extract_every_n: int = 5,
    ) -> dict:
        """L0 + checkpoint + extraction job in ONE transaction.

        The checkpoint is written before any extraction exists to fail, so a host
        hook timing out after commit cannot cause duplicate L0 on retry. The row
        lock closes the check-then-act gap between reading and writing the
        checkpoint — safe only because no LLM call happens inside this transaction.
        """
        meta_json = json.dumps(metadata or {})
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Materialize the row so FOR UPDATE has something to lock; two
                # concurrent first-writes for one session serialize here.
                await conn.execute(
                    """INSERT INTO capture_checkpoints (session_key, user_id, messages_seen)
                       VALUES ($1, $2, 0)
                       ON CONFLICT (session_key) DO NOTHING""",
                    session_key,
                    user_id,
                )
                cp = await conn.fetchrow(
                    """SELECT user_id, messages_seen, last_batch_hash
                       FROM capture_checkpoints
                       WHERE session_key = $1
                       FOR UPDATE""",
                    session_key,
                )
                foreign = cp["user_id"] and cp["user_id"] != user_id
                if not foreign and cp["last_batch_hash"] == batch_hash:
                    return {
                        "duplicate": True,
                        "messages_seen": cp["messages_seen"],
                        "extract_status": "skipped",
                    }

                seen_before = 0 if foreign else cp["messages_seen"]

                for msg in messages:
                    await conn.execute(
                        """INSERT INTO conversations
                               (id, user_id, agent_id, role, content, metadata)
                           VALUES ($1, $2, $3, $4, $5, $6)""",
                        str(uuid4()),
                        user_id,
                        agent_id,
                        msg["role"],
                        msg["content"],
                        meta_json,
                    )

                extract_status = "skipped"
                user_turns = sum(1 for m in messages if m.get("role") == "user")
                if user_turns:
                    seen = await conn.fetchval(
                        """INSERT INTO extraction_state (user_id, conversations_seen)
                           VALUES ($1, $2)
                           ON CONFLICT (user_id) DO UPDATE
                             SET conversations_seen =
                                     extraction_state.conversations_seen + $2,
                                 updated_at = now()
                           RETURNING conversations_seen""",
                        user_id,
                        user_turns,
                    )
                    if seen >= extract_every_n:
                        await conn.execute(
                            """INSERT INTO extraction_jobs (user_id, agent_id)
                               VALUES ($1, $2)
                               ON CONFLICT DO NOTHING""",
                            user_id,
                            agent_id,
                        )
                        extract_status = "queued"

                messages_seen = seen_before + len(messages)
                await conn.execute(
                    """UPDATE capture_checkpoints
                       SET user_id = $2,
                           messages_seen = $3,
                           last_batch_hash = $4,
                           updated_at = now()
                       WHERE session_key = $1""",
                    session_key,
                    user_id,
                    messages_seen,
                    batch_hash,
                )
        return {
            "duplicate": False,
            "messages_seen": messages_seen,
            "extract_status": extract_status,
        }

    async def get_conversations_since(
        self,
        user_id: str,
        since: Optional[datetime] = None,
        limit: int = 200,
        after_id: Optional[str] = None,
    ) -> list[dict]:
        """L0 rows after the extraction cursor, oldest-first.

        Keyset on (created_at, id) so rows sharing a timestamp are not orphaned
        when the watermark advances to max(created_at) of a limited window.
        """
        async with self._pool.acquire() as conn:
            if since is None:
                # Cold start: mine oldest-first so early L0 is not dropped after success mark.
                rows = await conn.fetch(
                    """SELECT id, role, content, metadata, created_at
                       FROM conversations
                       WHERE user_id = $1
                       ORDER BY created_at ASC, id ASC
                       LIMIT $2""",
                    user_id,
                    limit,
                )
            elif after_id is not None:
                rows = await conn.fetch(
                    """SELECT id, role, content, metadata, created_at
                       FROM conversations
                       WHERE user_id = $1
                         AND (
                           created_at > $2
                           OR (created_at = $2 AND id > $3)
                         )
                       ORDER BY created_at ASC, id ASC
                       LIMIT $4""",
                    user_id,
                    since,
                    after_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, role, content, metadata, created_at
                       FROM conversations
                       WHERE user_id = $1 AND created_at > $2
                       ORDER BY created_at ASC, id ASC
                       LIMIT $3""",
                    user_id,
                    since,
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
            for r in rows
        ]

    # -- Memories (L1) --

    async def save_memory(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        mem_type: str = "episodic",
        priority: int = 50,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dim {len(embedding)} != EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
            )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if await self.check_duplicate(user_id, text):
            return None
        row_id = str(uuid4())
        meta = metadata or {}
        meta["text_hash"] = text_hash
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO memories
                           (id, user_id, text, text_hash, embedding, mem_type, priority, agent_id, metadata)
                       VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
                       """,
                    row_id,
                    user_id,
                    text,
                    text_hash,
                    json.dumps(embedding),
                    mem_type,
                    priority,
                    agent_id,
                    json.dumps(meta),
                )
        except asyncpg.UniqueViolationError:
            return None
        return row_id

    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.3,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, mem_type, priority, metadata, created_at,
                          1 - (embedding <=> $1::vector) AS similarity
                   FROM memories
                   WHERE user_id = $2
                     AND 1 - (embedding <=> $1::vector) >= $3
                     AND ($4::text IS NULL OR agent_id = $4 OR agent_id IS NULL)
                   ORDER BY embedding <=> $1::vector
                    LIMIT $5""",
                json.dumps(query_embedding),
                user_id,
                threshold,
                agent_id,
                top_k,
            )
        return [_memory_row(r, score=round(float(r["similarity"]), 4)) for r in rows]

    async def keyword_search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            # Primary: BM25-style full-text search (lexeme match, ranked by ts_rank).
            rows = await conn.fetch(
                """SELECT id, text, mem_type, priority, metadata, created_at,
                          ts_rank(text_tsv, plainto_tsquery('english', $1)) AS score
                   FROM memories
                   WHERE user_id = $2
                     AND text_tsv @@ plainto_tsquery('english', $1)
                     AND ($3::text IS NULL OR agent_id = $3 OR agent_id IS NULL)
                   ORDER BY score DESC
                   LIMIT $4""",
                query,
                user_id,
                agent_id,
                top_k,
            )
            # Fallback: trigram fuzzy match for short/typo queries with no lexeme hit.
            if not rows:
                rows = await conn.fetch(
                    """SELECT id, text, mem_type, priority, metadata, created_at,
                              similarity(text, $1) AS score
                       FROM memories
                       WHERE user_id = $2
                         AND similarity(text, $1) >= $3
                         AND ($4::text IS NULL OR agent_id = $4 OR agent_id IS NULL)
                       ORDER BY score DESC
                       LIMIT $5""",
                    query,
                    user_id,
                    max(threshold, 0.05),
                    agent_id,
                    top_k,
                )
        return [_memory_row(r, score=round(float(r["score"]), 4)) for r in rows]

    async def hybrid_search_memories(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        vec_threshold: float = 0.3,
        kw_threshold: float = 0.05,
        rrf_k: int = 60,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        vector_results = await self.search_memories(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            threshold=vec_threshold,
            agent_id=agent_id,
        )
        keyword_results = await self.keyword_search_memories(
            user_id=user_id,
            query=query,
            top_k=top_k * 2,
            threshold=kw_threshold,
            agent_id=agent_id,
        )

        rrf_scores: dict[str, tuple[float, dict]] = {}
        for rank, r in enumerate(vector_results):
            rid = r["id"]
            score = 1.0 / (rrf_k + rank + 1)
            result = {
                "id": rid,
                "text": r["text"],
                "score": score,
                "type": r.get("type"),
                "priority": r.get("priority"),
                "metadata": dict(r.get("metadata") or {}),
                "created_at": r["created_at"],
            }
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
                    "type": r.get("type"),
                    "priority": r.get("priority"),
                    "metadata": dict(r.get("metadata") or {}),
                    "created_at": r["created_at"],
                }
                result["metadata"]["_kw_score"] = round(float(r["score"]), 4)
                rrf_scores[rid] = [score, result]

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        out = []
        for fused, result in sorted_results[:top_k]:
            result["score"] = round(fused, 4)
            out.append(result)
        return out

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

    async def get_instructions(
        self, user_id: str, agent_id: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, mem_type, priority, metadata, created_at
                   FROM memories
                   WHERE user_id = $1
                     AND mem_type = 'instruction'
                     AND ($2::text IS NULL OR agent_id = $2 OR agent_id IS NULL)
                   ORDER BY priority DESC, created_at DESC
                   LIMIT $3""",
                user_id,
                agent_id,
                limit,
            )
        return [_memory_row(r, score=0.0) for r in rows]

    # -- Partner facts (agent self / relation; isolated from user L1) --

    async def get_partner_facts(
        self,
        user_id: str,
        agent_id: str,
        kind: str,
        limit: int = 20,
    ) -> list[dict]:
        """Facts for exactly one agent.

        Strict `agent_id = $2` on purpose — unlike the soft user-L1 filter
        (`IS NULL OR agent_id = $n`), a self-spine belongs to one agent and must
        not blend with another's.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, priority, metadata, created_at
                   FROM partner_facts
                   WHERE user_id = $1 AND agent_id = $2 AND kind = $3
                   ORDER BY priority DESC, created_at DESC
                   LIMIT $4""",
                user_id,
                agent_id,
                kind,
                limit,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "priority": r["priority"],
                "metadata": json.loads(r["metadata"])
                if isinstance(r["metadata"], str)
                else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def save_partner_fact(
        self,
        user_id: str,
        agent_id: str,
        kind: str,
        text: str,
        priority: int = 50,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Append a partner fact. Returns None on duplicate (same contract as save_memory)."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        row_id = str(uuid4())
        async with self._pool.acquire() as conn:
            got = await conn.fetchval(
                """INSERT INTO partner_facts
                       (id, user_id, agent_id, kind, text, text_hash, priority, metadata)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (user_id, agent_id, kind, text_hash) DO NOTHING
                   RETURNING id""",
                row_id,
                user_id,
                agent_id,
                kind,
                text,
                text_hash,
                priority,
                json.dumps(metadata or {}),
            )
        return got

    async def count_partner_facts(
        self, user_id: str, agent_id: Optional[str] = None
    ) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """SELECT COUNT(*) FROM partner_facts
                   WHERE user_id = $1 AND ($2::text IS NULL OR agent_id = $2)""",
                user_id,
                agent_id,
            )

    async def get_top_episodic(
        self,
        user_id: str,
        min_priority: int = 70,
        limit: int = 5,
    ) -> list[dict]:
        """High-priority user episodic, read-only fill for a thin relation slice.

        Never copied into partner_facts — composed at read time only.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, priority, metadata, created_at
                   FROM memories
                   WHERE user_id = $1
                     AND mem_type = 'episodic'
                     AND priority >= $2
                   ORDER BY priority DESC, created_at DESC
                   LIMIT $3""",
                user_id,
                min_priority,
                limit,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "priority": r["priority"],
                "metadata": json.loads(r["metadata"])
                if isinstance(r["metadata"], str)
                else r["metadata"],
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
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dim {len(embedding)} != EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
            )
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

    async def get_memories_by_ids(self, ids: list[str], user_id: str) -> list[dict]:
        """Fetch memories by id, always scoped to user_id (no cross-user leak)."""
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id is required")
        if not ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, text, mem_type, priority, metadata, created_at
                   FROM memories
                   WHERE user_id = $1 AND id = ANY($2::text[])""",
                user_id,
                ids,
            )
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "type": r["mem_type"],
                "priority": r["priority"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def count_stale_embeddings(self) -> int:
        """Rows flagged after a dim rebuild wipe (vectors empty / unusable)."""
        async with self._pool.acquire() as conn:
            mem = await conn.fetchval(
                """SELECT COUNT(*) FROM memories
                   WHERE metadata ? '_embed_stale'"""
            )
            scen = await conn.fetchval(
                """SELECT COUNT(*) FROM scenarios
                   WHERE metadata ? '_embed_stale'"""
            )
        return int(mem or 0) + int(scen or 0)

    async def count_scenarios(self, user_id: Optional[str] = None) -> int:
        async with self._pool.acquire() as conn:
            if user_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM scenarios WHERE user_id = $1", user_id
                )
            return await conn.fetchval("SELECT COUNT(*) FROM scenarios")

    async def delete_scenarios(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM scenarios WHERE user_id = $1", user_id)

    async def replace_scenarios(self, user_id: str, scenarios: list[dict]) -> None:
        """Legacy full wipe — prefer upsert_scenarios_by_name."""
        for s in scenarios:
            emb = s.get("embedding") or []
            if len(emb) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"embedding dim {len(emb)} != EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
                )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM scenarios WHERE user_id = $1", user_id)
                for s in scenarios:
                    await conn.execute(
                        """INSERT INTO scenarios
                               (id, user_id, name, description, embedding, memory_ids, metadata)
                           VALUES ($1, $2, $3, $4, $5::vector, $6, $7)""",
                        str(uuid4()),
                        user_id,
                        s["name"],
                        s["description"],
                        json.dumps(s["embedding"]),
                        json.dumps(s["memory_ids"]),
                        json.dumps(s.get("metadata") or {}),
                    )

    async def upsert_scenarios_by_name(self, user_id: str, scenarios: list[dict]) -> int:
        """Merge scenarios by case-insensitive name. Never deletes siblings."""
        if not scenarios:
            return 0
        for s in scenarios:
            emb = s.get("embedding") or []
            if len(emb) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"embedding dim {len(emb)} != EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
                )
        touched = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for s in scenarios:
                    name = (s.get("name") or "").strip()
                    if not name:
                        continue
                    existing = await conn.fetchrow(
                        """SELECT id, memory_ids FROM scenarios
                           WHERE user_id = $1 AND lower(name) = lower($2)
                           LIMIT 1""",
                        user_id,
                        name,
                    )
                    new_ids = list(s.get("memory_ids") or [])
                    if existing:
                        old_ids = existing["memory_ids"]
                        if isinstance(old_ids, str):
                            old_ids = json.loads(old_ids)
                        merged = list(dict.fromkeys(list(old_ids or []) + new_ids))
                        await conn.execute(
                            """UPDATE scenarios
                               SET name = $2,
                                   description = $3,
                                   embedding = $4::vector,
                                   memory_ids = $5,
                                   metadata = $6,
                                   updated_at = now()
                               WHERE id = $1""",
                            existing["id"],
                            name,
                            s["description"],
                            json.dumps(s["embedding"]),
                            json.dumps(merged),
                            json.dumps(s.get("metadata") or {}),
                        )
                    else:
                        await conn.execute(
                            """INSERT INTO scenarios
                                   (id, user_id, name, description, embedding, memory_ids, metadata)
                               VALUES ($1, $2, $3, $4, $5::vector, $6, $7)""",
                            str(uuid4()),
                            user_id,
                            name,
                            s["description"],
                            json.dumps(s["embedding"]),
                            json.dumps(new_ids),
                            json.dumps(s.get("metadata") or {}),
                        )
                    touched += 1
        return touched

    # -- Extraction cadence (persistent counters) --

    async def bump_conversation_counter(self, user_id: str, n: int) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO extraction_state (user_id, conversations_seen)
                   VALUES ($1, $2)
                   ON CONFLICT (user_id) DO UPDATE
                     SET conversations_seen = extraction_state.conversations_seen + $2,
                         updated_at = now()
                   RETURNING conversations_seen, last_extraction_at, last_extraction_id,
                             last_extract_ok, last_extract_error, last_extract_attempt_at""",
                user_id,
                n,
            )
        return {
            "conversations_seen": row["conversations_seen"],
            "last_extraction_at": row["last_extraction_at"],
            "last_extraction_id": row["last_extraction_id"],
            "last_extract_ok": row["last_extract_ok"],
            "last_extract_error": row["last_extract_error"],
            "last_extract_attempt_at": row["last_extract_attempt_at"],
        }

    async def get_extraction_state(self, user_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT conversations_seen, last_extraction_at, last_extraction_id,
                          last_extract_ok, last_extract_error, last_extract_attempt_at,
                          memories_since_scenario
                   FROM extraction_state WHERE user_id = $1""",
                user_id,
            )
        if not row:
            return None
        return {
            "conversations_seen": row["conversations_seen"],
            "last_extraction_at": row["last_extraction_at"],
            "last_extraction_id": row["last_extraction_id"],
            "last_extract_ok": row["last_extract_ok"],
            "last_extract_error": row["last_extract_error"],
            "last_extract_attempt_at": row["last_extract_attempt_at"],
            "memories_since_scenario": row["memories_since_scenario"],
        }

    async def count_conversations(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE user_id = $1", user_id
            )

    async def latest_conversation_at(self, user_id: str) -> Optional[datetime]:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT MAX(created_at) FROM conversations WHERE user_id = $1",
                user_id,
            )

    async def advance_extraction_watermark(
        self,
        user_id: str,
        watermark_at: datetime,
        last_extraction_id: str,
    ) -> None:
        """Partial page progress: durable (created_at, id) cursor only.

        Does not zero conversations_seen and does not claim final success.
        Mid-pagination failures must leave extract still due.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO extraction_state
                       (user_id, conversations_seen, last_extraction_at, last_extraction_id,
                        updated_at)
                   VALUES ($1, 0, $2, $3, now())
                   ON CONFLICT (user_id) DO UPDATE SET
                       last_extraction_at = $2,
                       last_extraction_id = $3,
                       last_extract_attempt_at = now(),
                       updated_at = now()""",
                user_id,
                watermark_at,
                last_extraction_id,
            )

    async def mark_extraction_success(
        self,
        user_id: str,
        watermark_at: Optional[datetime] = None,
        last_extraction_id: Optional[str] = None,
    ) -> None:
        """Full drain complete: clear counter + ok; set watermark only when provided.

        watermark_at = newest L0 *included* in a mined window (not wall-clock now).
        watermark_at=None keeps the prior cursor (empty_window must not jump to now()).
        last_extraction_id pairs with watermark for durable keyset resume.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO extraction_state
                       (user_id, conversations_seen, last_extraction_at, last_extraction_id,
                        last_extract_ok, last_extract_error, last_extract_attempt_at, updated_at)
                   VALUES ($1, 0, $2, $3, true, NULL, now(), now())
                   ON CONFLICT (user_id) DO UPDATE SET
                       conversations_seen = 0,
                       last_extraction_at = COALESCE($2, extraction_state.last_extraction_at),
                       last_extraction_id = COALESCE($3, extraction_state.last_extraction_id),
                       last_extract_ok = true,
                       last_extract_error = NULL,
                       last_extract_attempt_at = now(),
                       updated_at = now()""",
                user_id,
                watermark_at,
                last_extraction_id,
            )

    async def mark_extraction_failure(
        self,
        user_id: str,
        error: str,
        *,
        keep_due_at_least: int = 1,
    ) -> None:
        """Mark fail; never clear cadence. Force counter due so next add retries."""
        err = (error or "unknown")[:500]
        due = max(1, int(keep_due_at_least))
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE extraction_state
                   SET last_extract_ok = false,
                       last_extract_error = $2,
                       last_extract_attempt_at = now(),
                       conversations_seen = GREATEST(conversations_seen, $3),
                       updated_at = now()
                   WHERE user_id = $1""",
                user_id,
                err,
                due,
            )

    async def reset_conversation_counter(self, user_id: str) -> None:
        """Deprecated alias — success path only."""
        await self.mark_extraction_success(user_id)

    async def bump_scenario_counter(self, user_id: str, n: int) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """INSERT INTO extraction_state (user_id, memories_since_scenario)
                   VALUES ($1, $2)
                   ON CONFLICT (user_id) DO UPDATE
                     SET memories_since_scenario = extraction_state.memories_since_scenario + $2,
                         updated_at = now()
                   RETURNING memories_since_scenario""",
                user_id,
                n,
            )

    async def reset_scenario_counter(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE extraction_state SET memories_since_scenario = 0 WHERE user_id = $1",
                user_id,
            )

    # -- Extraction queue (durable, off the request path) --

    async def enqueue_extraction_job(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        conn=None,
    ) -> Optional[str]:
        """Queue a drain for this user. No-op if one is already live (partial unique index).

        Pass conn to enlist in the caller's transaction (capture_atomic).
        """
        sql = """INSERT INTO extraction_jobs (user_id, agent_id)
                 VALUES ($1, $2)
                 ON CONFLICT DO NOTHING
                 RETURNING id"""
        if conn is not None:
            return await conn.fetchval(sql, user_id, agent_id)
        async with self._pool.acquire() as c:
            return await c.fetchval(sql, user_id, agent_id)

    async def claim_extraction_job(self, lease_seconds: int = 600) -> Optional[dict]:
        """Lease one runnable job. Short transaction — never held across the LLM call."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE extraction_jobs SET
                       status = 'running',
                       attempts = attempts + 1,
                       lease_until = now() + make_interval(secs => $1::double precision),
                       updated_at = now()
                   WHERE id = (
                       SELECT id FROM extraction_jobs
                       WHERE run_after <= now()
                         AND (status = 'queued'
                              OR (status = 'running' AND lease_until < now()))
                       ORDER BY run_after
                       FOR UPDATE SKIP LOCKED
                       LIMIT 1
                   )
                   RETURNING id, user_id, agent_id, attempts""",
                float(lease_seconds),
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "agent_id": row["agent_id"],
            "attempts": row["attempts"],
        }

    async def finish_extraction_job(
        self,
        job_id: str,
        *,
        status: str,
        error: Optional[str] = None,
        retry_in_seconds: Optional[int] = None,
    ) -> None:
        """Terminal state, or reschedule when retry_in_seconds is given."""
        err = (error or None) and str(error)[:500]
        async with self._pool.acquire() as conn:
            if retry_in_seconds is not None:
                await conn.execute(
                    """UPDATE extraction_jobs SET
                           status = 'queued',
                           last_error = $2,
                           lease_until = NULL,
                           run_after = now() + make_interval(secs => $3::double precision),
                           updated_at = now()
                       WHERE id = $1""",
                    job_id,
                    err,
                    float(retry_in_seconds),
                )
                return
            await conn.execute(
                """UPDATE extraction_jobs SET
                       status = $2,
                       last_error = $3,
                       lease_until = NULL,
                       updated_at = now()
                   WHERE id = $1""",
                job_id,
                status,
                err,
            )

    async def get_extraction_job(self, job_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, user_id, agent_id, status, attempts, last_error,
                          lease_until, run_after
                   FROM extraction_jobs WHERE id = $1""",
                job_id,
            )
        return dict(row) if row else None

    async def count_extraction_jobs(self, status: str) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM extraction_jobs WHERE status = $1", status
            )

    async def count_live_extraction_jobs(self, user_id: str) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """SELECT COUNT(*) FROM extraction_jobs
                   WHERE user_id = $1 AND status IN ('queued','running')""",
                user_id,
            )

    # -- Persona cache (L3) --

    async def get_persona_cache(self, user_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT summary, memory_count, memories_at_generation, generated_at
                   FROM personas WHERE user_id = $1""",
                user_id,
            )
        if not row:
            return None
        return {
            "summary": row["summary"],
            "memory_count": row["memory_count"],
            "memories_at_generation": row["memories_at_generation"],
            "generated_at": row["generated_at"],
        }

    async def save_persona_cache(
        self, user_id: str, summary: str, memory_count: int
    ) -> datetime:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """INSERT INTO personas
                       (user_id, summary, memory_count, memories_at_generation)
                   VALUES ($1, $2, $3, $3)
                   ON CONFLICT (user_id) DO UPDATE
                     SET summary = $2,
                         memory_count = $3,
                         memories_at_generation = $3,
                         generated_at = now()
                   RETURNING generated_at""",
                user_id,
                summary,
                memory_count,
            )

    # -- Cleanup --

    async def delete_user_data(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM conversations WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM scenarios WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM extraction_state WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM personas WHERE user_id = $1", user_id)
            await conn.execute(
                "DELETE FROM capture_checkpoints WHERE user_id = $1", user_id
            )
            await conn.execute("DELETE FROM partner_facts WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM extraction_jobs WHERE user_id = $1", user_id)

    async def wipe_all(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """TRUNCATE conversations, memories, scenarios, extraction_state,
                          personas, capture_checkpoints, partner_facts,
                          extraction_jobs"""
            )
