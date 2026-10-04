"""`POST /agent/chat` — single-turn agent invocation using the compiled LangGraph."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from apps.agent.models import ChatMessage, MemoryCandidate
from apps.agent.state import AgentState
from ingestion.jobs import IngestionJobStatus, OpenSearchIngestionJobs
from memory.consolidation.models import ConsolidationJob
from memory.retrieval.models import RecallResult
from messaging.rabbitmq import RabbitMQBroker


def get_agent_graph(request: Request) -> Any:
    return request.app.state.agent_graph


def get_ingestion_broker(request: Request) -> RabbitMQBroker | None:
    return getattr(request.app.state, "ingestion_broker", None)


def get_ingestion_jobs(request: Request) -> OpenSearchIngestionJobs:
    return request.app.state.ingestion_jobs  # type: ignore[no-any-return]


AgentGraphDep = Annotated[Any, Depends(get_agent_graph)]
IngestionBrokerDep = Annotated[RabbitMQBroker | None, Depends(get_ingestion_broker)]
IngestionJobsDep = Annotated[OpenSearchIngestionJobs, Depends(get_ingestion_jobs)]


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
    consolidation_job_id: str | None = None


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    graph: AgentGraphDep,
    broker: IngestionBrokerDep,
    jobs: IngestionJobsDep,
) -> ChatResponse:
    initial_state: AgentState = {
        "messages": [*req.history, ChatMessage(role="user", content=req.message)],
    }
    final_state = await graph.ainvoke(initial_state)

    candidates = list(final_state.get("candidate_memories", []))
    response_text = str(final_state.get("final_response", ""))

    queued = False
    job_id = None
    if broker is not None and candidates:
        job = ConsolidationJob(
            user_message=req.message,
            assistant_reply=response_text,
            candidates=candidates,
            conversation_id=req.conversation_id,
        )
        await jobs.create(job)
        try:
            await broker.enqueue(job)
        except Exception as error:
            await jobs.update(job.job_id, status="failed", error=str(error))
            raise HTTPException(status_code=503, detail="Ingestion queue is unavailable") from error
        queued = True
        job_id = job.job_id

    return ChatResponse(
        response=response_text,
        goal=final_state.get("current_goal"),
        recalled_memories=list(final_state.get("recalled_memories", [])),
        candidate_memories=candidates,
        consolidation_queued=queued,
        consolidation_job_id=job_id,
    )


@router.get("/jobs/{job_id}", response_model=IngestionJobStatus)
async def get_job(job_id: str, jobs: IngestionJobsDep) -> IngestionJobStatus:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
