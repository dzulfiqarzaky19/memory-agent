from __future__ import annotations

import logging
from typing import Optional

from config import (
    EXTRACTION_EVERY_N_TURNS,
    PERSONA_EVERY_N_MEMORIES,
    RECALL_MAX_RESULTS,
    RECALL_RRF_K,
    RECALL_SIMILARITY_THRESHOLD,
    RECALL_STRATEGY,
)
from embeddings import EmbeddingProvider
from extraction import LLMExtractor
from storage import Storage

logger = logging.getLogger(__name__)

SCENARIO_REBUILD_INTERVAL = 10


def _priority_weight(score: float, priority: Optional[int]) -> float:
    # Mild priority tilt: priority 50 is neutral (x1.0), 100 -> x1.25, 0 -> x0.75.
    p = 50 if priority is None else priority
    return score * (0.75 + p / 200.0)


class MemoryEngine:
    def __init__(self, storage: Storage, embedder: EmbeddingProvider, extractor: LLMExtractor):
        self.storage = storage
        self.embedder = embedder
        self.extractor = extractor

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

        user_turns = sum(1 for m in messages if m.get("role") == "user")
        memories_added = 0
        memory_ids = []

        if user_turns:
            state = await self.storage.bump_conversation_counter(user_id, user_turns)
            conversations_seen = state["conversations_seen"]
            last_extraction_at = state["last_extraction_at"]

            if conversations_seen >= EXTRACTION_EVERY_N_TURNS:
                cold_limit = max(EXTRACTION_EVERY_N_TURNS * 4, 40)
                pending = await self.storage.get_conversations_since(
                    user_id,
                    since=last_extraction_at,
                    limit=cold_limit if last_extraction_at is None else 200,
                )
                extracted = await self._extract_and_store(user_id, pending, agent_id)
                memories_added = len(extracted)
                memory_ids = [m["id"] for m in extracted] if extracted else []
                await self.storage.reset_conversation_counter(user_id)

                if memories_added:
                    memories_since_scenario = await self.storage.bump_scenario_counter(
                        user_id, memories_added
                    )
                    if memories_since_scenario >= SCENARIO_REBUILD_INTERVAL:
                        await self._rebuild_scenarios(user_id)
                        await self.storage.reset_scenario_counter(user_id)

        return {
            "memories_added": memories_added,
            "memory_ids": memory_ids,
        }

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = RECALL_MAX_RESULTS,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        query_embedding = self.embedder.embed([query])[0]
        threshold = RECALL_SIMILARITY_THRESHOLD

        if RECALL_STRATEGY == "hybrid":
            memory_results = await self.storage.hybrid_search_memories(
                user_id=user_id,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                vec_threshold=threshold,
                rrf_k=RECALL_RRF_K,
                agent_id=agent_id,
            )
        elif RECALL_STRATEGY == "keyword":
            memory_results = await self.storage.keyword_search_memories(
                user_id=user_id,
                query=query,
                top_k=top_k,
                agent_id=agent_id,
            )
        else:
            memory_results = await self.storage.search_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold,
                agent_id=agent_id,
            )

        # Priority tilt: high-priority atoms rank above equally-similar low-priority ones.
        for r in memory_results:
            r["score"] = round(_priority_weight(r["score"], r.get("priority")), 4)

        memory_results.sort(key=lambda x: x["score"], reverse=True)
        primary = memory_results[:top_k]
        existing_ids = {r["id"] for r in primary}

        # Option C: matched primary first; injects only fill remaining top_k slots.
        injects: list[dict] = []

        instructions = await self.storage.get_instructions(user_id, agent_id=agent_id, limit=5)
        for ins in instructions:
            if ins["id"] not in existing_ids:
                injects.append(ins)
                existing_ids.add(ins["id"])

        scenario_results = await self.storage.search_scenarios(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=3,
        )
        if scenario_results:
            unique_ids = []
            seen = set()
            for s in scenario_results:
                for mid in s.get("memory_ids", []):
                    if mid not in seen:
                        seen.add(mid)
                        unique_ids.append(mid)
            if unique_ids:
                scenario_memories = await self.storage.get_memories_by_ids(unique_ids[:20])
                for sm in scenario_memories:
                    if sm["id"] not in existing_ids:
                        sm["metadata"] = sm.get("metadata") or {}
                        sm["metadata"]["_from_scenario"] = True
                        injects.append(sm)
                        existing_ids.add(sm["id"])

        slots = top_k - len(primary)
        if slots > 0 and injects:
            floor = primary[-1]["score"] if primary else 0.0
            filled = []
            for i, item in enumerate(injects[:slots]):
                item = dict(item)
                item["score"] = round(floor - (i + 1) * 1e-4, 4)
                meta = dict(item.get("metadata") or {})
                if item.get("type") == "instruction" or meta.get("_instruction"):
                    meta["_instruction"] = True
                item["metadata"] = meta
                filled.append(item)
            return primary + filled

        return primary

    async def get_persona(self, user_id: str) -> dict:
        count = await self.storage.count_memories(user_id)
        if count == 0:
            return {
                "user_id": user_id,
                "summary": "No memories stored yet.",
                "memory_count": 0,
                "last_updated": None,
            }

        cached = await self.storage.get_persona_cache(user_id)
        if cached and (count - cached["memories_at_generation"]) < PERSONA_EVERY_N_MEMORIES:
            return {
                "user_id": user_id,
                "summary": cached["summary"],
                "memory_count": count,
                "last_updated": cached["generated_at"],
            }

        memories = await self.storage.get_all_memories(user_id, limit=200)
        summary = await self.extractor.generate_persona([m["text"] for m in memories])
        generated_at = await self.storage.save_persona_cache(user_id, summary, count)

        return {
            "user_id": user_id,
            "summary": summary,
            "memory_count": count,
            "last_updated": generated_at,
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

        built = []
        for group in grouped:
            indices = group.get("memory_indices", [])
            linked_ids = [memory_ids[i] for i in indices if i < len(memory_ids)]
            if not linked_ids:
                continue
            scenario_text = f"{group['name']}: {group['description']}"
            embedding = self.embedder.embed([scenario_text])[0]
            built.append(
                {
                    "name": group["name"],
                    "description": group["description"],
                    "embedding": embedding,
                    "memory_ids": linked_ids,
                }
            )

        if not built:
            return

        await self.storage.replace_scenarios(user_id, built)
        logger.info(f"Rebuilt {len(built)} scenarios for {user_id}")

    async def _extract_and_store(
        self,
        user_id: str,
        new_messages: list[dict],
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in new_messages
        )

        existing = await self.storage.get_all_memories(user_id, limit=50)
        existing_texts = [m["text"] for m in existing]

        atoms = await self.extractor.extract_memories(
            messages_text=messages_text,
            existing_memories=existing_texts,
        )

        if not atoms:
            return []

        embeddings = self.embedder.embed([a["content"] for a in atoms])
        stored = []
        for atom, embedding in zip(atoms, embeddings):
            row_id = await self.storage.save_memory(
                user_id=user_id,
                text=atom["content"],
                embedding=embedding,
                mem_type=atom["type"],
                priority=atom["priority"],
                agent_id=agent_id,
            )
            if row_id is None:
                logger.debug(f"Skipping duplicate: {atom['content'][:50]}")
                continue
            stored.append(
                {
                    "id": row_id,
                    "text": atom["content"],
                    "type": atom["type"],
                    "priority": atom["priority"],
                }
            )

        logger.info(f"Extracted {len(stored)} new memories for {user_id}")
        return stored
