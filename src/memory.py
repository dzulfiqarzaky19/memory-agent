from __future__ import annotations

import logging
from typing import Optional

from config import (
    EXTRACTION_EVERY_N_TURNS,
    RECALL_MAX_RESULTS,
    RECALL_RRF_K,
    RECALL_SIMILARITY_THRESHOLD,
    RECALL_STRATEGY,
)
from embeddings import EmbeddingProvider, create_embedding_provider
from extraction import LLMExtractor
from storage import Storage

logger = logging.getLogger(__name__)

SCENARIO_REBUILD_INTERVAL = 10


class MemoryEngine:
    def __init__(self, storage: Storage, embedder: EmbeddingProvider, extractor: LLMExtractor):
        self.storage = storage
        self.embedder = embedder
        self.extractor = extractor
        self._turn_counter: dict[str, int] = {}
        self._new_memory_counter: dict[str, int] = {}

    async def add(
        self,
        user_id: str,
        messages: list[dict],
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        for msg in messages:
            await self.storage.save_conversation(
                user_id=user_id,
                role=msg["role"],
                content=msg["content"],
                agent_id=agent_id,
                metadata=metadata,
            )

        self._turn_counter[user_id] = self._turn_counter.get(user_id, 0) + len(messages)

        memories_added = 0
        memory_ids = []
        if self._turn_counter[user_id] >= EXTRACTION_EVERY_N_TURNS:
            extracted = await self._extract_and_store(user_id, messages)
            memories_added = len(extracted)
            memory_ids = [m["id"] for m in extracted] if extracted else []
            self._turn_counter[user_id] = 0

            self._new_memory_counter[user_id] = self._new_memory_counter.get(user_id, 0) + memories_added
            if self._new_memory_counter[user_id] >= SCENARIO_REBUILD_INTERVAL:
                await self._rebuild_scenarios(user_id)
                self._new_memory_counter[user_id] = 0

        return {
            "memories_added": memories_added,
            "memory_ids": memory_ids,
        }

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        query_embedding = self.embedder.embed([query])[0]

        if RECALL_STRATEGY == "hybrid":
            memory_results = await self.storage.hybrid_search_memories(
                user_id=user_id,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                rrf_k=RECALL_RRF_K,
            )
        elif RECALL_STRATEGY == "keyword":
            memory_results = await self.storage.keyword_search_memories(
                user_id=user_id,
                query=query,
                top_k=top_k,
            )
        else:
            memory_results = await self.storage.search_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
            )

        scenario_results = await self.storage.search_scenarios(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=3,
        )

        if scenario_results:
            scenario_memory_ids = []
            for s in scenario_results:
                scenario_memory_ids.extend(s.get("memory_ids", []))

            seen = set()
            unique_ids = []
            for mid in scenario_memory_ids:
                if mid not in seen:
                    seen.add(mid)
                    unique_ids.append(mid)

            if unique_ids:
                scenario_memories = await self.storage.get_memories_by_ids(unique_ids[:20])
                existing_ids = {r["id"] for r in memory_results}
                for sm in scenario_memories:
                    if sm["id"] not in existing_ids:
                        sm["score"] = 0.7
                        sm["metadata"] = sm.get("metadata") or {}
                        sm["metadata"]["_from_scenario"] = True
                        memory_results.append(sm)

            memory_results.sort(key=lambda x: x["score"], reverse=True)

        return memory_results[:top_k]

    async def get_persona(self, user_id: str) -> dict:
        memories = await self.storage.get_all_memories(user_id, limit=200)
        if not memories:
            return {
                "user_id": user_id,
                "summary": "No memories stored yet.",
                "memory_count": 0,
                "last_updated": None,
            }

        memory_texts = [m["text"] for m in memories]
        summary = await self.extractor.generate_persona(memory_texts)

        return {
            "user_id": user_id,
            "summary": summary,
            "memory_count": len(memories),
            "last_updated": memories[0]["created_at"] if memories else None,
        }

    async def get_scenarios(self, user_id: str) -> dict:
        scenarios = await self.storage.get_all_scenarios(user_id)
        return {
            "user_id": user_id,
            "scenarios": scenarios,
            "total": len(scenarios),
        }

    async def _rebuild_scenarios(self, user_id: str) -> None:
        memories = await self.storage.get_all_memories(user_id, limit=200)
        if not memories:
            return

        existing_scenarios = await self.storage.get_all_scenarios(user_id)
        memory_texts = [m["text"] for m in memories]
        memory_ids = [m["id"] for m in memories]

        grouped = await self.extractor.group_into_scenarios(
            memory_texts=memory_texts,
            existing_scenarios=existing_scenarios,
        )

        if not grouped:
            return

        for group in grouped:
            indices = group.get("memory_indices", [])
            linked_ids = [memory_ids[i] for i in indices if i < len(memory_ids)]
            if not linked_ids:
                continue

            scenario_text = f"{group['name']}: {group['description']}"
            embedding = self.embedder.embed([scenario_text])[0]

            await self.storage.save_scenario(
                user_id=user_id,
                name=group["name"],
                description=group["description"],
                embedding=embedding,
                memory_ids=linked_ids,
            )

        logger.info(f"Rebuilt {len(grouped)} scenarios for {user_id}")

    async def _extract_and_store(
        self,
        user_id: str,
        new_messages: list[dict],
    ) -> list[dict]:
        messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in new_messages
        )

        existing = await self.storage.get_all_memories(user_id, limit=50)
        existing_texts = [m["text"] for m in existing]

        extracted_texts = await self.extractor.extract_memories(
            messages_text=messages_text,
            existing_memories=existing_texts,
        )

        if not extracted_texts:
            return []

        embeddings = self.embedder.embed(extracted_texts)
        stored = []
        for text, embedding in zip(extracted_texts, embeddings):
            row_id = await self.storage.save_memory(
                user_id=user_id,
                text=text,
                embedding=embedding,
            )
            if row_id is None:
                logger.debug(f"Skipping duplicate: {text[:50]}")
                continue
            stored.append({"id": row_id, "text": text})

        logger.info(f"Extracted {len(stored)} new memories for {user_id}")
        return stored
