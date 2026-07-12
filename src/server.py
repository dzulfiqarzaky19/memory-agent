from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from config import RECALL_MAX_RESULTS, RECALL_SIMILARITY_THRESHOLD
from embeddings import create_embedding_provider
from extraction import LLMExtractor
from memory import MemoryEngine
from models import (
    AddRequest,
    AddResponse,
    HealthResponse,
    PersonaResponse,
    ScenarioResult,
    ScenariosResponse,
    SearchRequest,
    SearchResponse,
    MemoryResult,
)
from storage import Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage = Storage()
engine: Optional[MemoryEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    await storage.initialize()
    embedder = create_embedding_provider()
    extractor = LLMExtractor()
    engine = MemoryEngine(storage=storage, embedder=embedder, extractor=extractor)
    logger.info("Memory agent started")
    yield
    await storage.close()
    logger.info("Memory agent stopped")


app = FastAPI(title="memory-agent", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    count = await storage.count_memories()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database="connected",
        memory_count=count,
    )


@app.post("/add", response_model=AddResponse)
async def add_memories(req: AddRequest):
    messages = [m.model_dump() for m in req.messages]
    result = await engine.add(
        user_id=req.user_id,
        messages=messages,
        agent_id=req.agent_id,
        metadata=req.metadata,
    )
    return AddResponse(
        memories_added=result["memories_added"],
        memory_ids=result["memory_ids"],
    )


@app.post("/search", response_model=SearchResponse)
async def search_memories(req: SearchRequest):
    results = await engine.search(
        user_id=req.user_id,
        query=req.query,
        top_k=req.top_k,
        agent_id=req.agent_id,
    )
    return SearchResponse(
        results=[
            MemoryResult(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                type=r.get("type"),
                priority=r.get("priority"),
                created_at=r["created_at"],
                metadata=r.get("metadata"),
            )
            for r in results
        ],
        total=len(results),
    )


@app.get("/persona/{user_id}", response_model=PersonaResponse)
async def get_persona(user_id: str):
    result = await engine.get_persona(user_id)
    return PersonaResponse(**result)


@app.get("/scenarios/{user_id}", response_model=ScenariosResponse)
async def get_scenarios(user_id: str):
    result = await engine.get_scenarios(user_id)
    return ScenariosResponse(
        user_id=result["user_id"],
        scenarios=[ScenarioResult(**s) for s in result["scenarios"]],
        total=result["total"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
