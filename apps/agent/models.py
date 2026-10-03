"""Public request/response models for the agent surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str


CandidateClassification = Literal["IGNORE", "EPISODIC", "SEMANTIC", "PROCEDURAL"]


class MemoryCandidate(BaseModel):
    """Proposed memory produced by `memory_candidate_extraction`. Never persisted here."""

    model_config = ConfigDict(extra="forbid")

    classification: CandidateClassification
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    source: str = "conversation"
    reason: str
