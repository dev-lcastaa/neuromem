"""Retrieval request/response models. The `explanation` object is mandatory, not optional."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from memory.domain.enums import MemoryType
from memory.storage.interface import Memory


class RankWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: float = Field(default=0.40, ge=0.0)
    keyword: float = Field(default=0.15, ge=0.0)
    importance: float = Field(default=0.20, ge=0.0)
    recency: float = Field(default=0.10, ge=0.0)
    activation: float = Field(default=0.10, ge=0.0)
    relationship: float = Field(default=0.05, ge=0.0)


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: float = 0.0
    keyword: float = 0.0
    importance: float = 0.0
    recency: float = 0.0
    activation: float = 0.0
    relationship: float = 0.0


class RecallQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    memory_types: list[MemoryType] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    candidate_pool: int = Field(default=50, ge=1, le=1000)
    include_relationships: bool = True
    weights: RankWeights | None = None
    include_content_vector: bool = False


class RecallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: MemoryType
    content: str
    score: float
    explanation: Explanation
    memory: Memory


class RecallResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    weights: RankWeights
    memories: list[RecallResult]
