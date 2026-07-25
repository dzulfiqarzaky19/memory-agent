from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from config import RECALL_MAX_RESULTS


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None


class AddRequest(BaseModel):
    messages: list[Message]
    user_id: str = Field(..., min_length=1)
    agent_id: Optional[str] = None
    metadata: Optional[dict] = None


class AddResponse(BaseModel):
    memories_added: int
    memory_ids: list[str]
    extract_status: str = "skipped"
    user_id: Optional[str] = None


class CaptureRequest(BaseModel):
    messages: list[Message]
    user_id: str = Field(..., min_length=1)
    session_key: str = Field(..., min_length=1)
    agent_id: Optional[str] = None
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
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    top_k: int = Field(default=RECALL_MAX_RESULTS, ge=1, le=100)
    agent_id: Optional[str] = None


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
    last_extract_ok: Optional[bool] = None
    last_extract_error: Optional[str] = None
    last_extraction_at: Optional[datetime] = None
    last_extract_attempt_at: Optional[datetime] = None
    extraction_lag_seconds: Optional[float] = None
    recall_trusted: bool = False


class SearchResponse(BaseModel):
    results: list[MemoryResult]
    total: int
    trust: Optional[MemoryTrust] = None


class PersonaResponse(BaseModel):
    user_id: str
    summary: str
    memory_count: int
    last_updated: Optional[datetime] = None
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


class ReloadConfig(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None


class ReloadResponse(BaseModel):
    status: str
    model: str
    base_url: str
