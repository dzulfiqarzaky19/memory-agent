from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config import (
    EMBEDDING_DIMENSIONS,
    EXTRACTION_WORKER_ENABLED,
    MEMORY_ALLOW_RELOAD,
    MEMORY_API_SECRET,
    MEMORY_BIND_HOST,
)
from embeddings import create_embedding_provider
from extraction import LLMExtractor
from memory import MemoryEngine
from models import (
    AddRequest,
    AddResponse,
    CaptureRequest,
    CaptureResponse,
    HealthResponse,
    MemoryTrust,
    PersonaResponse,
    ReloadConfig,
    ReloadResponse,
    ScenarioResult,
    ScenariosResponse,
    SearchRequest,
    SearchResponse,
    MemoryResult,
)
from storage import Storage
from worker import ExtractionWorker

# Open without secret (liveness only). Everything else needs X-Memory-Key when set.
_AUTH_OPEN_PATHS = frozenset({"/health"})


def _trust_model(trust: Optional[dict]) -> Optional[MemoryTrust]:
    if not trust:
        return None
    return MemoryTrust(**trust)


def _secret_ok(provided: str | None) -> bool:
    expected = MEMORY_API_SECRET
    if not expected:
        return True
    got = provided or ""
    # compare_digest raises on length mismatch — never 500 on bad keys.
    if len(got) != len(expected):
        return False
    return hmac.compare_digest(got, expected)


class MemoryKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if path in _AUTH_OPEN_PATHS:
            return await call_next(request)
        if not MEMORY_API_SECRET:
            return await call_next(request)
        key = request.headers.get("x-memory-key")
        if not _secret_ok(key):
            return Response(
                content='{"detail":"unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage = Storage()
engine: Optional[MemoryEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    await storage.initialize()
    embedder = create_embedding_provider()
    if embedder.dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"embedder.dimensions={embedder.dimensions} != EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
        )
    probe = embedder.embed(["dim-check"])
    if not probe or len(probe[0]) != EMBEDDING_DIMENSIONS:
        got = len(probe[0]) if probe else 0
        raise RuntimeError(
            f"live embed dim-check failed: got {got}, expected EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}"
        )
    stale = await storage.count_stale_embeddings()
    if stale:
        raise RuntimeError(
            f"{stale} memory/scenario row(s) have _embed_stale after a dim rebuild — "
            "re-embed or wipe before serving recall"
        )
    extractor = LLMExtractor()
    engine = MemoryEngine(storage=storage, embedder=embedder, extractor=extractor)
    worker = ExtractionWorker(engine)
    if EXTRACTION_WORKER_ENABLED:
        worker.start()
    else:
        logger.warning("Extraction worker disabled — jobs will queue but not run")
    if MEMORY_API_SECRET:
        logger.info(
            "Memory agent started (API key required, embed_dims=%s)",
            EMBEDDING_DIMENSIONS,
        )
    else:
        logger.warning(
            "Memory agent started with MEMORY_API_SECRET unset — HTTP routes open on bind host"
        )
    yield
    await worker.stop()
    await storage.close()
    logger.info("Memory agent stopped")


app = FastAPI(title="memory-agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(MemoryKeyMiddleware)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> Response:
    """Bad caller input (e.g. whitespace user_id on a path route) is 400, not 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
async def health():
    count = await storage.count_memories()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database="connected",
        memory_count=count,
        extraction_queued=await storage.count_extraction_jobs("queued"),
        extraction_dead=await storage.count_extraction_jobs("dead"),
    )


@app.get("/config/llm")
async def llm_config():
    """Non-secret LLM wiring check (values from env via config.py)."""
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER

    key = LLM_API_KEY or ""
    return {
        "LLM_PROVIDER": LLM_PROVIDER,
        "LLM_MODEL": LLM_MODEL,
        "LLM_BASE_URL": LLM_BASE_URL,
        "LLM_API_KEY_set": bool(key),
        "LLM_API_KEY_len": len(key),
        "MEMORY_API_SECRET_set": bool(MEMORY_API_SECRET),
        "MEMORY_ALLOW_RELOAD": MEMORY_ALLOW_RELOAD,
    }


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
        extract_status=result.get("extract_status", "skipped"),
        user_id=result.get("user_id"),
    )


@app.post("/capture", response_model=CaptureResponse)
async def capture_messages(req: CaptureRequest):
    """Host auto-capture (agent-end). Idempotent per session_key + batch hash."""
    messages = [m.model_dump() for m in req.messages]
    result = await engine.capture(
        user_id=req.user_id,
        session_key=req.session_key,
        messages=messages,
        agent_id=req.agent_id,
        metadata=req.metadata,
    )
    return CaptureResponse(**result)


@app.post("/search", response_model=SearchResponse)
async def search_memories(req: SearchRequest):
    payload = await engine.search(
        user_id=req.user_id,
        query=req.query,
        top_k=req.top_k,
        agent_id=req.agent_id,
    )
    results = payload["results"]
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
        total=payload["total"],
        stale=bool(payload.get("stale")),
        stale_seconds=int(payload.get("stale_seconds") or 0),
        trust=_trust_model(payload.get("trust")),
    )


@app.get("/persona/{user_id}", response_model=PersonaResponse)
async def get_persona(user_id: str):
    result = await engine.get_persona(user_id)
    return PersonaResponse(
        user_id=result["user_id"],
        summary=result["summary"],
        memory_count=result["memory_count"],
        last_updated=result.get("last_updated"),
        stale=bool(result.get("stale")),
        stale_seconds=int(result.get("stale_seconds") or 0),
        trust=_trust_model(result.get("trust")),
    )


@app.post("/reload", response_model=ReloadResponse)
async def reload_config(req: ReloadConfig):
    if not MEMORY_ALLOW_RELOAD:
        raise HTTPException(status_code=404, detail="not found")
    engine.extractor.reconfigure(
        model=req.model,
        base_url=req.base_url,
        api_key=req.api_key,
        max_tokens=req.max_tokens,
    )
    return ReloadResponse(
        status="ok",
        model=engine.extractor._model,
        base_url=engine.extractor._base_url,
    )


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

    uvicorn.run(app, host=MEMORY_BIND_HOST, port=8000)
