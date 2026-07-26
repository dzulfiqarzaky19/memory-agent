"""Durable extraction worker.

Leases one job at a time from `extraction_jobs` and drains that user's pending L0
into L1/L2. The lease — not a held row lock — is what prevents a double run, so no
pool connection is ever held across an LLM call.
"""

from __future__ import annotations

import asyncio
import logging

from config import (
    EXTRACTION_JOB_MAX_ATTEMPTS,
    EXTRACTION_LEASE_SECONDS,
    EXTRACTION_POLL_SECONDS,
    EXTRACTION_RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)


class ExtractionWorker:
    def __init__(self, engine, *, poll_seconds: float | None = None):
        self._engine = engine
        self._poll = EXTRACTION_POLL_SECONDS if poll_seconds is None else poll_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop(), name="extraction-worker")
            logger.info("Extraction worker started (poll=%ss)", self._poll)

    async def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        self._task = None
        logger.info("Extraction worker stopped")

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                did_work = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A crash here must not kill the loop — the job stays leased and
                # is reclaimed after expiry.
                logger.exception("Extraction worker iteration failed")
                did_work = False
            if not did_work:
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=self._poll
                    )
                except asyncio.TimeoutError:
                    pass

    async def run_once(self) -> bool:
        """Claim and run at most one job. Returns True if a job was claimed."""
        storage = self._engine.storage
        job = await storage.claim_extraction_job(EXTRACTION_LEASE_SECONDS)
        if job is None:
            return False

        try:
            result = await self._engine.run_extraction(job["user_id"], job["agent_id"])
        except Exception as e:
            if job["attempts"] >= EXTRACTION_JOB_MAX_ATTEMPTS:
                # Poison job: stop retrying, keep it visible for ops.
                logger.error(
                    "Extraction job %s dead after %s attempts: %s",
                    job["id"],
                    job["attempts"],
                    e,
                )
                await storage.finish_extraction_job(
                    job["id"], status="dead", error=str(e)
                )
            else:
                backoff = EXTRACTION_RETRY_BACKOFF_SECONDS * job["attempts"]
                logger.warning(
                    "Extraction job %s failed (attempt %s), retry in %ss: %s",
                    job["id"],
                    job["attempts"],
                    backoff,
                    e,
                )
                await storage.finish_extraction_job(
                    job["id"], status="queued", error=str(e), retry_in_seconds=backoff
                )
            return True

        await storage.finish_extraction_job(job["id"], status="done")
        logger.info(
            "Extraction job %s done for %s (+%s memories)",
            job["id"],
            job["user_id"],
            result["memories_added"],
        )
        return True
