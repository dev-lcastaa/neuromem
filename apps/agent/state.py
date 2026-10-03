"""Working-memory shape for the LangGraph agent (plan §11)."""

from __future__ import annotations

from typing import Any, TypedDict

from apps.agent.models import ChatMessage, MemoryCandidate
from memory.retrieval.models import RecallResult


class AgentState(TypedDict, total=False):
    messages: list[ChatMessage]
    current_goal: str
    current_plan: str
    observations: list[str]
    recalled_memories: list[RecallResult]
    tool_results: list[dict[str, Any]]
    candidate_memories: list[MemoryCandidate]
    final_response: str
