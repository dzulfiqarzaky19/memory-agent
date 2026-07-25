from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    agent_id: Optional[str] = None


class MemoryResult(BaseModel):
    id: str
    text: str
    score: float
    type: Optional[str] = None
    priority: Optional[int] = None
    created_at: datetime
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    results: list[MemoryResult]
    total: int


class PersonaResponse(BaseModel):
    user_id: str
    summary: str
    memory_count: int
    last_updated: Optional[datetime] = None


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
