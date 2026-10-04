"""Consolidation job envelope + outcome types."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from apps.agent.models import MemoryCandidate

ConflictDecision = Literal["DUPLICATE", "SUPERSEDES", "CONTRADICTS", "INDEPENDENT"]


class ConsolidationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    user_message: str
    assistant_reply: str = ""
    candidates: list[MemoryCandidate] = Field(default_factory=list)
    conversation_id: str = "adhoc"


class ConsolidationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    existing_id: str
    decision: ConflictDecision
    reason: str = ""


class ConsolidationOutcome(BaseModel):
    """Per-candidate outcome. `created_memory_id` is None when duplicate/ignored."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int
    classification: str
    action: Literal["created", "duplicate", "ignored", "superseded", "contradicted"]
    created_memory_id: str | None = None
    conflicts: list[ConsolidationDecision] = Field(default_factory=list)


class ConsolidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    source_memory_id: str
    outcomes: list[ConsolidationOutcome] = Field(default_factory=list)
