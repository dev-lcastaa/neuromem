"""`POST /agent/chat` — single-turn agent invocation using the compiled LangGraph."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from apps.agent.models import ChatMessage, MemoryCandidate
from apps.agent.state import AgentState
from memory.consolidation.models import ConsolidationJob
from memory.consolidation.queue import ConsolidationQueue
from memory.retrieval.models import RecallResult


def get_agent_graph(request: Request) -> Any:
    return request.app.state.agent_graph


def get_consolidation_queue(request: Request) -> ConsolidationQueue | None:
    return getattr(request.app.state, "consolidation_queue", None)


AgentGraphDep = Annotated[Any, Depends(get_agent_graph)]
ConsolidationQueueDep = Annotated[ConsolidationQueue | None, Depends(get_consolidation_queue)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    conversation_id: str = "adhoc"


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    goal: str | None = None
    recalled_memories: list[RecallResult] = Field(default_factory=list)
    candidate_memories: list[MemoryCandidate] = Field(default_factory=list)
    consolidation_queued: bool = False


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    graph: AgentGraphDep,
    consolidation_queue: ConsolidationQueueDep,
) -> ChatResponse:
    initial_state: AgentState = {
        "messages": [*req.history, ChatMessage(role="user", content=req.message)],
    }
    final_state = await graph.ainvoke(initial_state)

    candidates = list(final_state.get("candidate_memories", []))
    response_text = str(final_state.get("final_response", ""))

    queued = False
    if consolidation_queue is not None and candidates:
        await consolidation_queue.enqueue(
            ConsolidationJob(
                user_message=req.message,
                assistant_reply=response_text,
                candidates=candidates,
                conversation_id=req.conversation_id,
            )
        )
        queued = True

    return ChatResponse(
        response=response_text,
        goal=final_state.get("current_goal"),
        recalled_memories=list(final_state.get("recalled_memories", [])),
        candidate_memories=candidates,
        consolidation_queued=queued,
    )
