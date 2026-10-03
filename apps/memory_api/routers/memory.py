"""Memory CRUD + relationships + recall + lifecycle + dashboard reads."""

from __future__ import annotations

from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from memory.consolidation.models import ConsolidationJob, ConsolidationResult
from memory.consolidation.service import ConsolidationService
from memory.domain.enums import MemoryType
from memory.domain.models import Relationship
from memory.retrieval.models import RecallQuery, RecallResponse
from memory.service import MemoryService
from memory.storage.interface import Memory


def get_service(request: Request) -> MemoryService:
    return request.app.state.memory_service  # type: ignore[no-any-return]


def get_consolidation_service(request: Request) -> ConsolidationService:
    return request.app.state.consolidation_service  # type: ignore[no-any-return]


ServiceDep = Annotated[MemoryService, Depends(get_service)]
ConsolidationDep = Annotated[ConsolidationService, Depends(get_consolidation_service)]


def _stub(phase: int) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"Not implemented until Phase {phase}.",
    )


class AccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_ids: list[str] = Field(min_length=1)
    bump: float | None = Field(default=None, ge=0.0, le=1.0)


class AccessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updated: int


class DecayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_ids: list[str] = Field(min_length=1)


class DecayReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    found: bool
    activation: float | None = None
    access_count: int | None = None
    last_accessed_at: str | None = None
    age_days: float | None = None
    decay_factor: float | None = None
    effective_activation: float | None = None


class DecayResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: list[DecayReport]


class MemoryHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_type: str
    created_at: str | None
    updated_at: str | None
    last_accessed_at: str | None
    access_count: int
    activation: float
    effective_activation: float
    relationships: list[Relationship]


class StatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int]
    totals: dict[str, int]
    averages: dict[str, float]


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    score: float
    importance: float
    activation: float
    created_at: str | None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[SearchResult]


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    created_at: str | None
    importance: float
    confidence: float
    activation: float
    access_count: int


class TimelineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[TimelineEntry]


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    importance: float
    activation: float


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    from_id: str
    to_id: str
    relationship_type: str
    confidence: float


class GraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center_id: str
    depth: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]


router = APIRouter(prefix="/memory")


@router.post(
    "/store",
    tags=["store"],
    response_model=Memory,
    status_code=status.HTTP_201_CREATED,
)
async def store(memory: Memory, service: ServiceDep) -> Memory:
    return await service.create_memory(memory)


@router.post(
    "/recall",
    tags=["recall"],
    response_model=RecallResponse,
)
async def recall(query: RecallQuery, service: ServiceDep) -> RecallResponse:
    return await service.recall(query)


@router.post(
    "/consolidate",
    tags=["consolidate"],
    response_model=ConsolidationResult,
)
async def consolidate(job: ConsolidationJob, service: ConsolidationDep) -> ConsolidationResult:
    return await service.process(job)


@router.post("/access", tags=["lifecycle"], response_model=AccessResponse)
async def access(req: AccessRequest, service: ServiceDep) -> AccessResponse:
    updated = await service.record_access(req.memory_ids, req.bump)
    return AccessResponse(updated=updated)


@router.post("/decay", tags=["lifecycle"], response_model=DecayResponse)
async def decay(req: DecayRequest, service: ServiceDep) -> DecayResponse:
    raw: list[dict[str, Any]] = await service.decay_report(req.memory_ids)
    return DecayResponse(reports=[DecayReport.model_validate(r) for r in raw])


# Literal-path GET handlers MUST appear before `/{memory_id}` so FastAPI's
# ordered matching doesn't route `/memory/stats` etc. to `get_memory("stats")`.


@router.get("/stats", tags=["store"], response_model=StatsResponse)
async def stats(service: ServiceDep) -> StatsResponse:
    payload: dict[str, Any] = await service.stats()
    return StatsResponse.model_validate(payload)


@router.get("/search", tags=["recall"], response_model=SearchResponse)
async def search(
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1)],
    memory_types: Annotated[list[MemoryType] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> SearchResponse:
    results = await service.search_by_text(q, memory_types=memory_types, limit=limit)
    return SearchResponse(query=q, results=[SearchResult.model_validate(r) for r in results])


@router.get("/timeline", tags=["store"], response_model=TimelineResponse)
async def timeline(
    service: ServiceDep,
    memory_types: Annotated[list[MemoryType] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> TimelineResponse:
    memories = await service.list_recent(memory_types=memory_types, limit=limit)
    return TimelineResponse(memories=[TimelineEntry.model_validate(m) for m in memories])


@router.get("/graph", tags=["relationships"], response_model=GraphResponse)
async def graph(
    service: ServiceDep,
    center_id: Annotated[str, Query(min_length=1)],
    depth: Annotated[int, Query(ge=1, le=3)] = 1,
) -> GraphResponse:
    result = await service.graph_around(center_id, depth=depth)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory {center_id} not found")
    return GraphResponse(
        center_id=result["center_id"],
        depth=depth,
        nodes=[GraphNode.model_validate(n) for n in result["nodes"]],
        edges=[GraphEdge.model_validate(e) for e in result["edges"]],
    )


@router.get("/{memory_id}", tags=["store"], response_model=Memory)
async def get_memory(memory_id: str, service: ServiceDep) -> Memory:
    got = await service.get_memory(memory_id)
    if got is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return got


@router.get(
    "/{memory_id}/history",
    tags=["store"],
    response_model=MemoryHistoryResponse,
)
async def memory_history(memory_id: str, service: ServiceDep) -> MemoryHistoryResponse:
    memory = await service.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    (report,) = await service.decay_report([memory_id])
    rels = await service.get_relationships(memory_id)
    return MemoryHistoryResponse(
        memory_id=memory_id,
        memory_type=memory.memory_type.value,
        created_at=memory.created_at.isoformat() if memory.created_at else None,
        updated_at=memory.updated_at.isoformat() if memory.updated_at else None,
        last_accessed_at=(memory.last_accessed_at.isoformat() if memory.last_accessed_at else None),
        access_count=memory.access_count,
        activation=memory.activation,
        effective_activation=report.get("effective_activation", memory.activation),
        relationships=rels,
    )


@router.post(
    "/relate",
    tags=["relationships"],
    status_code=status.HTTP_201_CREATED,
)
async def relate(relationship: Relationship, service: ServiceDep) -> Relationship:
    return await service.create_relationship(relationship)


@router.get(
    "/{memory_id}/relationships",
    tags=["relationships"],
)
async def memory_relationships(memory_id: str, service: ServiceDep) -> list[Relationship]:
    return await service.get_relationships(memory_id)


@router.post(
    "/conflicts/resolve",
    tags=["conflicts"],
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def resolve_conflict() -> dict[str, str]:
    _stub(4)
