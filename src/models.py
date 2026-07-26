from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from config import RECALL_MAX_RESULTS


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


# Server caps sit well above the Claude Code hook's 2000/2500-char clip so
# auto-capture is never rejected; they exist to bound self-inflicted payloads.
MAX_CONTENT_CHARS = 32_000
MAX_MESSAGES = 200
MAX_ID_CHARS = 200
MAX_QUERY_CHARS = 4_000


class Message(BaseModel):
    role: Role
    content: str = Field(..., max_length=MAX_CONTENT_CHARS)
    name: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)


class AddRequest(BaseModel):
    messages: list[Message] = Field(..., max_length=MAX_MESSAGES)
    user_id: str = Field(..., min_length=1, max_length=MAX_ID_CHARS)
    agent_id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    metadata: Optional[dict] = None


class AddResponse(BaseModel):
    memories_added: int
    memory_ids: list[str]
    extract_status: str = "skipped"
    user_id: Optional[str] = None


class CaptureRequest(BaseModel):
    messages: list[Message] = Field(..., max_length=MAX_MESSAGES)
    user_id: str = Field(..., min_length=1, max_length=MAX_ID_CHARS)
    session_key: str = Field(..., min_length=1, max_length=MAX_ID_CHARS)
    agent_id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)
    metadata: Optional[dict] = None


class CaptureResponse(BaseModel):
    messages_captured: int
    memories_added: int
    memory_ids: list[str]
    duplicate: bool = False
    messages_seen: int = 0
    extract_status: str = "skipped"
    user_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    user_id: str = Field(..., min_length=1, max_length=MAX_ID_CHARS)
    top_k: int = Field(default=RECALL_MAX_RESULTS, ge=1, le=100)
    agent_id: Optional[str] = Field(default=None, max_length=MAX_ID_CHARS)


class MemoryResult(BaseModel):
    id: str
    text: str
    score: float
    type: Optional[str] = None
    priority: Optional[int] = None
    created_at: datetime
    metadata: Optional[dict] = None


class MemoryTrust(BaseModel):
    user_id: str
    l0_count: int = 0
    l1_count: int = 0
    conversations_seen: int = 0
    extraction_pending: bool = False
    extraction_due: bool = False
    behind_watermark: bool = False
    last_extract_ok: Optional[bool] = None
    last_extract_error: Optional[str] = None
    last_extraction_at: Optional[datetime] = None
    last_extract_attempt_at: Optional[datetime] = None
    extraction_lag_seconds: Optional[float] = None
    extraction_lag_exceeded: bool = False
    stale_seconds: int = 0
    recall_trusted: bool = False


class SearchResponse(BaseModel):
    results: list[MemoryResult]
    total: int
    stale: bool = False
    stale_seconds: int = 0
    trust: Optional[MemoryTrust] = None


class PersonaResponse(BaseModel):
    user_id: str
    summary: str
    memory_count: int
    last_updated: Optional[datetime] = None
    stale: bool = False
    stale_seconds: int = 0
    trust: Optional[MemoryTrust] = None


class ScenarioResult(BaseModel):
    id: str
    name: str
    description: str
    score: Optional[float] = None
    memory_ids: list[str]
    created_at: datetime
    metadata: Optional[dict] = None


class ScenariosResponse(BaseModel):
    user_id: str
    scenarios: list[ScenarioResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    memory_count: int
    extraction_queued: int = 0
    extraction_dead: int = 0


class ReloadConfig(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None


class ReloadResponse(BaseModel):
    status: str
    model: str
    base_url: str
