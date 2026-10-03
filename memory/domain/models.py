"""Pydantic domain models for memories and relationships.

Models mirror the OpenSearch mappings in `opensearch/indexes/*.json`. `extra="forbid"`
enforces the same contract client-side that `dynamic: strict` enforces in OpenSearch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from memory.domain.enums import MemoryType, RelationshipType


class MemoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    agent_role: str | None = None
    source: str | None = None
    entities: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_dim: int | None = None


class MemoryBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    content: str
    content_vector: list[float] | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    activation: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)

    source_memory_ids: list[str] = Field(default_factory=list)
    related_memory_ids: list[str] = Field(default_factory=list)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)


class EpisodicMemory(MemoryBase):
    memory_type: Literal[MemoryType.EPISODIC] = MemoryType.EPISODIC
    event: str | None = None
    context: str | None = None
    event_timestamp: datetime | None = None
    participants: list[str] = Field(default_factory=list)
    actions: str | None = None
    outcome: str | None = None
    source_conversation: str | None = None


class SemanticMemory(MemoryBase):
    memory_type: Literal[MemoryType.SEMANTIC] = MemoryType.SEMANTIC
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    domain: str | None = None
    superseded_by: str | None = None


class ProceduralStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=0)
    action: str
    expected_result: str | None = None


class ProceduralMemory(MemoryBase):
    memory_type: Literal[MemoryType.PROCEDURAL] = MemoryType.PROCEDURAL
    name: str | None = None
    goal: str | None = None
    preconditions: str | None = None
    steps: list[ProceduralStep] = Field(default_factory=list)
    expected_outcome: str | None = None
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)


Memory = Annotated[
    EpisodicMemory | SemanticMemory | ProceduralMemory,
    Field(discriminator="memory_type"),
]

MemoryAdapter: TypeAdapter[EpisodicMemory | SemanticMemory | ProceduralMemory] = TypeAdapter(Memory)


class RelationshipMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = None
    source: str | None = None


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    from_id: str
    to_id: str
    from_type: MemoryType
    to_type: MemoryType
    relationship_type: RelationshipType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime | None = None
    created_by: str = "system"
    metadata: RelationshipMetadata = Field(default_factory=RelationshipMetadata)
