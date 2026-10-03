"""Scenario schema + per-query metrics + report row types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory.domain.enums import MemoryType, RelationshipType

MemoryTypeStr = Literal["episodic", "semantic", "procedural"]


class SeededMemorySpec(BaseModel):
    """One memory to insert during scenario setup. `{run_id}` in content is templated in."""

    model_config = ConfigDict(extra="forbid")

    key: str | None = None
    memory_type: MemoryTypeStr
    content: str
    subject: str | None = None
    event: str | None = None
    name: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    activation: float = Field(default=0.5, ge=0.0, le=1.0)


class SeededRelationshipSpec(BaseModel):
    """One relationship edge between two seeded memories, referenced by `key`."""

    model_config = ConfigDict(extra="forbid")

    from_key: str
    to_key: str
    relationship_type: RelationshipType
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    reason: str | None = None


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    expected_top_content_contains: list[str] = Field(default_factory=list)
    expected_recall_contains_any: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_answer_missing: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str = ""
    memories: list[SeededMemorySpec] = Field(default_factory=list)
    relationships: list[SeededRelationshipSpec] = Field(default_factory=list)
    queries: list[QuerySpec] = Field(default_factory=list)


class QueryMetrics(BaseModel):
    """One (scenario, query, mode) row of results."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    query_id: str
    mode: str
    precision_at_k: float
    recall_hit: float
    top1_match: bool
    answer_hit: bool
    answer_forbidden_present: bool
    irrelevant_retrievals: int
    retrieval_latency_ms: float
    answer_latency_ms: float
    retrieved_memory_ids: list[str]
    answer: str


def memory_type_to_enum(s: MemoryTypeStr) -> MemoryType:
    return MemoryType(s)
