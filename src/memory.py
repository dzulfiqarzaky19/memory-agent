from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from config import (
    EXTRACTION_EVERY_N_TURNS,
    EXTRACTION_MAX_LAG_SECONDS,
    PERSONA_EVERY_N_MEMORIES,
    RECALL_MAX_RESULTS,
    RECALL_RRF_K,
    RECALL_SIMILARITY_THRESHOLD,
    RECALL_STRATEGY,
)
from embeddings import EmbeddingProvider
from extraction import LLMExtractor
from ids import canonicalize_user_id
from storage import Storage

logger = logging.getLogger(__name__)

SCENARIO_REBUILD_INTERVAL = 10


def _priority_weight(score: float, priority: Optional[int]) -> float:
    # Mild priority tilt: priority 50 is neutral (x1.0), 100 -> x1.25, 0 -> x0.75.
    p = 50 if priority is None else priority
    return score * (0.75 + p / 200.0)


def _batch_hash(messages: list[dict]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(m.get("role", "").encode())
        h.update(b"\0")
        h.update(m.get("content", "").encode())
        h.update(b"\0")
    return h.hexdigest()


def _lag_seconds(ts) -> Optional[float]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


class MemoryEngine:
    def __init__(self, storage: Storage, embedder: EmbeddingProvider, extractor: LLMExtractor):
        self.storage = storage
        self.embedder = embedder
        self.extractor = extractor

    async def memory_trust(self, user_id: str) -> dict:
        """Recall is trusted only when extraction is caught up — not merely 'has any L1'."""
        uid = canonicalize_user_id(user_id)
        state = await self.storage.get_extraction_state(uid) or {}
        l0 = await self.storage.count_conversations(uid)
        l1 = await self.storage.count_memories(uid)
        seen = int(state.get("conversations_seen") or 0)
        last_ok = state.get("last_extract_ok")
        last_at = state.get("last_extraction_at")
        latest_l0 = await self.storage.latest_conversation_at(uid)
        lag = _lag_seconds(last_at)

        overdue = seen >= EXTRACTION_EVERY_N_TURNS
        never_extracted = l0 > 0 and last_at is None
        behind_watermark = False
        if last_at is not None and latest_l0 is not None:
            behind_watermark = _aware(latest_l0) > _aware(last_at)
        lag_exceeded = (
            behind_watermark
            and lag is not None
            and EXTRACTION_MAX_LAG_SECONDS > 0
            and lag > EXTRACTION_MAX_LAG_SECONDS
        )
        # Empty diary is honestly empty. Otherwise need a successful extract and no lag breach.
        if l0 == 0 and l1 == 0:
            trusted = last_ok is not False
        else:
            trusted = (
                last_ok is True
                and not overdue
                and not never_extracted
                and not lag_exceeded
            )
        pending = seen > 0 or behind_watermark or never_extracted
        return {
            "user_id": uid,
            "l0_count": l0,
            "l1_count": l1,
            "conversations_seen": seen,
            "extraction_pending": pending,
            "extraction_due": overdue,
            "behind_watermark": behind_watermark or never_extracted,
            "last_extract_ok": last_ok,
            "last_extract_error": state.get("last_extract_error"),
            "last_extraction_at": last_at,
            "last_extract_attempt_at": state.get("last_extract_attempt_at"),
            "extraction_lag_seconds": lag,
            "extraction_lag_exceeded": lag_exceeded,
            "recall_trusted": bool(trusted),
        }

    async def add(
        self,
        user_id: str,
        messages: list[dict],
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        uid = canonicalize_user_id(user_id)
        for msg in messages:
            await self.storage.save_conversation(
                user_id=uid,
                role=msg["role"],
                content=msg["content"],
                agent_id=agent_id,
                metadata=metadata,
            )

        user_turns = sum(1 for m in messages if m.get("role") == "user")
        memories_added = 0
        memory_ids: list[str] = []
        extract_status = "skipped"

        if user_turns:
            state = await self.storage.bump_conversation_counter(uid, user_turns)
            conversations_seen = state["conversations_seen"]
            last_extraction_at = state["last_extraction_at"]

            if conversations_seen >= EXTRACTION_EVERY_N_TURNS:
                # Page through pending L0 until empty. Watermark = last mined
                # created_at (not now()); keyset (created_at, id) drains ties.
                cold_limit = max(EXTRACTION_EVERY_N_TURNS * 4, 40)
                window_limit = cold_limit if last_extraction_at is None else 200
                cursor_at = last_extraction_at
                cursor_id: Optional[str] = None
                try:
                    mined_any = False
                    while True:
                        pending = await self.storage.get_conversations_since(
                            uid,
                            since=cursor_at,
                            limit=window_limit,
                            after_id=cursor_id,
                        )
                        if not pending:
                            if not mined_any:
                                await self.storage.mark_extraction_success(uid)
                                extract_status = "empty_window"
                            break

                        extracted = await self._extract_and_store(uid, pending, agent_id)
                        batch_n = len(extracted)
                        memories_added += batch_n
                        if extracted:
                            memory_ids.extend(m["id"] for m in extracted)

                        last_row = pending[-1]
                        watermark = last_row["created_at"]
                        await self.storage.mark_extraction_success(
                            uid, watermark_at=watermark
                        )
                        cursor_at = watermark
                        cursor_id = last_row["id"]
                        mined_any = True
                        extract_status = "ok" if memories_added else "ok_no_facts"

                        got = len(pending)
                        if got < window_limit:
                            break
                        window_limit = 200

                    if memories_added:
                        memories_since_scenario = await self.storage.bump_scenario_counter(
                            uid, memories_added
                        )
                        if memories_since_scenario >= SCENARIO_REBUILD_INTERVAL:
                            await self._rebuild_scenarios(uid)
                            await self.storage.reset_scenario_counter(uid)
                except Exception as e:
                    logger.error("Extraction failed for %s: %s", uid, e)
                    await self.storage.mark_extraction_failure(uid, str(e))
                    extract_status = "failed"

        return {
            "memories_added": memories_added,
            "memory_ids": memory_ids,
            "extract_status": extract_status,
            "user_id": uid,
        }

    async def capture(
        self,
        user_id: str,
        session_key: str,
        messages: list[dict],
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Auto-capture path: L0 write with per-session checkpoint (exact-batch dedupe)."""
        uid = canonicalize_user_id(user_id)
        cleaned = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant", "system") and (m.get("content") or "").strip()
        ]
        if not cleaned:
            cp = await self.storage.get_capture_checkpoint(session_key)
            return {
                "messages_captured": 0,
                "memories_added": 0,
                "memory_ids": [],
                "duplicate": False,
                "messages_seen": (cp or {}).get("messages_seen", 0),
                "user_id": uid,
                "extract_status": "skipped",
            }

        batch = _batch_hash(cleaned)
        cp = await self.storage.get_capture_checkpoint(session_key)
        if cp and cp.get("user_id") and cp["user_id"] != uid:
            # session_key collision across users — do not honor foreign checkpoint.
            cp = None
        if cp and cp.get("last_batch_hash") == batch:
            return {
                "messages_captured": 0,
                "memories_added": 0,
                "memory_ids": [],
                "duplicate": True,
                "messages_seen": cp["messages_seen"],
                "user_id": uid,
                "extract_status": "skipped",
            }

        meta = dict(metadata or {})
        meta["session_key"] = session_key
        meta["source"] = meta.get("source") or "auto-capture"
        result = await self.add(
            user_id=uid,
            messages=cleaned,
            agent_id=agent_id,
            metadata=meta,
        )
        seen = (cp["messages_seen"] if cp else 0) + len(cleaned)
        await self.storage.upsert_capture_checkpoint(
            session_key=session_key,
            user_id=uid,
            messages_seen=seen,
            last_batch_hash=batch,
        )
        return {
            "messages_captured": len(cleaned),
            "memories_added": result["memories_added"],
            "memory_ids": result["memory_ids"],
            "duplicate": False,
            "messages_seen": seen,
            "user_id": uid,
            "extract_status": result.get("extract_status", "skipped"),
        }

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = RECALL_MAX_RESULTS,
        agent_id: Optional[str] = None,
    ) -> dict:
        uid = canonicalize_user_id(user_id)
        trust = await self.memory_trust(uid)
        query_embedding = self.embedder.embed([query])[0]
        threshold = RECALL_SIMILARITY_THRESHOLD

        if RECALL_STRATEGY == "hybrid":
            memory_results = await self.storage.hybrid_search_memories(
                user_id=uid,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                vec_threshold=threshold,
                rrf_k=RECALL_RRF_K,
                agent_id=agent_id,
            )
        elif RECALL_STRATEGY == "keyword":
            memory_results = await self.storage.keyword_search_memories(
                user_id=uid,
                query=query,
                top_k=top_k,
                agent_id=agent_id,
            )
        else:
            memory_results = await self.storage.search_memories(
                user_id=uid,
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold,
                agent_id=agent_id,
            )

        # Priority tilt: high-priority atoms rank above equally-similar low-priority ones.
        # Keep full float for ordering; round only at the end so inject ranks don't collapse.
        for r in memory_results:
            r["score"] = _priority_weight(r["score"], r.get("priority"))

        memory_results.sort(key=lambda x: x["score"], reverse=True)
        primary = memory_results[:top_k]
        existing_ids = {r["id"] for r in primary}

        # Option C: matched primary first; injects only fill remaining top_k slots.
        injects: list[dict] = []

        instructions = await self.storage.get_instructions(uid, agent_id=agent_id, limit=5)
        for ins in instructions:
            if ins["id"] not in existing_ids:
                injects.append(ins)
                existing_ids.add(ins["id"])

        scenario_results = await self.storage.search_scenarios(
            user_id=uid,
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
                scenario_memories = await self.storage.get_memories_by_ids(
                    unique_ids[:20], user_id=uid
                )
                for sm in scenario_memories:
                    if sm["id"] not in existing_ids:
                        sm["metadata"] = sm.get("metadata") or {}
                        sm["metadata"]["_from_scenario"] = True
                        injects.append(sm)
                        existing_ids.add(sm["id"])

        slots = top_k - len(primary)
        results = primary
        if slots > 0 and injects:
            floor = primary[-1]["score"] if primary else 0.0
            filled = []
            for i, item in enumerate(injects[:slots]):
                item = dict(item)
                # Distinct sub-ranks without round(4) collisions.
                item["score"] = floor - (i + 1) * 1e-6
                meta = dict(item.get("metadata") or {})
                if item.get("type") == "instruction" or meta.get("_instruction"):
                    meta["_instruction"] = True
                item["metadata"] = meta
                filled.append(item)
            results = primary + filled

        for r in results:
            r["score"] = round(float(r["score"]), 6)

        return {
            "results": results,
            "total": len(results),
            "trust": trust,
        }

    async def get_persona(self, user_id: str) -> dict:
        uid = canonicalize_user_id(user_id)
        trust = await self.memory_trust(uid)
        count = trust["l1_count"]
        if count == 0:
            return {
                "user_id": uid,
                "summary": "No memories stored yet.",
                "memory_count": 0,
                "last_updated": None,
                "trust": trust,
            }

        cached = await self.storage.get_persona_cache(uid)
        if cached and (count - cached["memories_at_generation"]) < PERSONA_EVERY_N_MEMORIES:
            return {
                "user_id": uid,
                "summary": cached["summary"],
                "memory_count": count,
                "last_updated": cached["generated_at"],
                "trust": trust,
            }

        memories = await self.storage.get_all_memories(uid, limit=200)
        try:
            summary = await self.extractor.generate_persona([m["text"] for m in memories])
        except Exception as e:
            logger.error("Persona generation failed for %s: %s", uid, e)
            # Do not cache failure strings — keep prior cache or return ephemeral error.
            if cached:
                return {
                    "user_id": uid,
                    "summary": cached["summary"],
                    "memory_count": count,
                    "last_updated": cached["generated_at"],
                    "trust": trust,
                }
            return {
                "user_id": uid,
                "summary": "Persona unavailable (generation failed).",
                "memory_count": count,
                "last_updated": memories[0]["created_at"] if memories else None,
                "trust": trust,
            }
        bad = (
            not (summary or "").strip()
            or summary.strip().lower().startswith("error")
            or summary.strip() == "No persona available yet."
        )
        if bad:
            if cached:
                return {
                    "user_id": uid,
                    "summary": cached["summary"],
                    "memory_count": count,
                    "last_updated": cached["generated_at"],
                    "trust": trust,
                }
            return {
                "user_id": uid,
                "summary": "Persona unavailable (empty generation).",
                "memory_count": count,
                "last_updated": memories[0]["created_at"] if memories else None,
                "trust": trust,
            }
        generated_at = await self.storage.save_persona_cache(uid, summary, count)

        return {
            "user_id": uid,
            "summary": summary,
            "memory_count": count,
            "last_updated": generated_at,
            "trust": trust,
        }

    async def get_scenarios(self, user_id: str) -> dict:
        uid = canonicalize_user_id(user_id)
        scenarios = await self.storage.get_all_scenarios(uid)
        return {
            "user_id": uid,
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

        try:
            grouped = await self.extractor.group_into_scenarios(
                memory_texts=memory_texts,
                existing_scenarios=existing_scenarios,
            )
        except Exception as e:
            # Distinguish LLM failure from genuine empty grouping — do not spin forever silent.
            logger.error("Scenario grouping failed for %s: %s", user_id, e)
            return

        if not grouped:
            logger.info("Scenario group empty for %s — keeping existing L2", user_id)
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

        n = await self.storage.upsert_scenarios_by_name(user_id, built)
        logger.info("Upserted %s scenarios for %s (no full replace)", n, user_id)

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
