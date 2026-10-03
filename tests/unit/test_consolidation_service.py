from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from apps.agent.models import MemoryCandidate
from memory.config import Settings
from memory.consolidation.models import ConsolidationJob
from memory.consolidation.service import ConsolidationService
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, SemanticMemory
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallResponse,
    RecallResult,
)
from models.llm import LLMResponse


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _persisted(
    memory: EpisodicMemory | SemanticMemory, mid: str
) -> EpisodicMemory | SemanticMemory:
    return memory.model_copy(update={"id": mid, "created_at": datetime.now(UTC)})


def _fake_service_create(memory_prefix: str) -> AsyncMock:
    counter = {"n": 0}

    async def _create(memory):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return memory.model_copy(
            update={
                "id": f"{memory_prefix}_{counter['n']}",
                "created_at": datetime.now(UTC),
            }
        )

    m = AsyncMock(side_effect=_create)
    return m


@pytest.mark.asyncio
async def test_process_persists_turn_and_ignores_ignore_candidates() -> None:
    service = AsyncMock()
    service.create_memory.side_effect = lambda m: m.model_copy(
        update={"id": "epi_turn_1", "created_at": datetime.now(UTC)}
    )
    llm = AsyncMock()

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="hello",
            assistant_reply="hi",
            candidates=[
                MemoryCandidate(
                    classification="IGNORE",
                    content="ignore me",
                    reason="pleasantry",
                )
            ],
            conversation_id="conv-1",
        )
    )

    assert result.source_memory_id == "epi_turn_1"
    assert len(result.outcomes) == 1
    assert result.outcomes[0].action == "ignored"
    assert result.outcomes[0].created_memory_id is None
    llm.complete.assert_not_awaited()
    service.create_relationship.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_candidate_creates_and_links_derived_from() -> None:
    service = AsyncMock()
    created_ids = iter(["epi_turn_1", "sem_1"])

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": next(created_ids), "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create
    service.recall.return_value = RecallResponse(query="q", weights=RankWeights(), memories=[])
    llm = AsyncMock()

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="I use OpenSearch.",
            assistant_reply="Noted.",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content="User uses OpenSearch as the vector store.",
                    importance=0.8,
                    confidence=0.9,
                    reason="stable preference",
                )
            ],
        )
    )

    assert result.outcomes[0].action == "created"
    assert result.outcomes[0].created_memory_id == "sem_1"
    service.create_relationship.assert_awaited_once()
    rel = service.create_relationship.await_args.args[0]
    assert rel.relationship_type == RelationshipType.DERIVED_FROM
    assert rel.from_id == "sem_1"
    assert rel.to_id == "epi_turn_1"


@pytest.mark.asyncio
async def test_duplicate_semantic_candidate_is_skipped() -> None:
    service = AsyncMock()

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": "epi_turn_1", "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create

    existing = SemanticMemory(id="sem_old", content="user uses OpenSearch", subject="opensearch")
    service.recall.return_value = RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_old",
                memory_type=MemoryType.SEMANTIC,
                content=existing.content,
                score=0.9,
                explanation=Explanation(semantic=0.9),
                memory=existing,
            )
        ],
    )

    llm = AsyncMock()
    llm.complete.return_value = LLMResponse(
        content='{"decisions":[{"existing_id":"sem_old","decision":"DUPLICATE","reason":"same fact"}]}',
        model="gpt-4o",
        latency_ms=1,
    )

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="I use OpenSearch",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content="User uses OpenSearch as the vector store.",
                    reason="restated preference",
                )
            ],
        )
    )

    # Only the source turn was created; the duplicate candidate was skipped
    assert service.create_memory.await_count == 1
    assert result.outcomes[0].action == "duplicate"
    assert result.outcomes[0].created_memory_id is None
    service.create_relationship.assert_not_awaited()


@pytest.mark.asyncio
async def test_supersedes_creates_edge_but_keeps_old_memory() -> None:
    service = AsyncMock()
    created_ids = iter(["epi_turn_1", "sem_new"])

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": next(created_ids), "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create

    existing = SemanticMemory(
        id="sem_old", content="user prefers OpenSearch", subject="vector_store"
    )
    service.recall.return_value = RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_old",
                memory_type=MemoryType.SEMANTIC,
                content=existing.content,
                score=0.85,
                explanation=Explanation(semantic=0.85),
                memory=existing,
            )
        ],
    )
    service.get_memory.return_value = existing

    llm = AsyncMock()
    llm.complete.return_value = LLMResponse(
        content='{"decisions":[{"existing_id":"sem_old","decision":"SUPERSEDES","reason":"switched to Qdrant"}]}',
        model="gpt-4o",
        latency_ms=1,
    )

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="I switched to Qdrant.",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content="User now prefers Qdrant over OpenSearch.",
                    reason="preference change",
                )
            ],
        )
    )

    assert result.outcomes[0].action == "superseded"
    assert result.outcomes[0].created_memory_id == "sem_new"
    # Two relationships written: DERIVED_FROM turn + SUPERSEDES sem_old
    rel_calls = service.create_relationship.await_args_list
    assert len(rel_calls) == 2
    rel_types = {call.args[0].relationship_type for call in rel_calls}
    assert RelationshipType.DERIVED_FROM in rel_types
    assert RelationshipType.SUPERSEDES in rel_types


@pytest.mark.asyncio
async def test_contradicts_creates_edge_and_keeps_old() -> None:
    service = AsyncMock()
    created_ids = iter(["epi_turn_1", "sem_new"])

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": next(created_ids), "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create

    existing = SemanticMemory(
        id="sem_old", content="deployment on Jetson succeeded", subject="jetson"
    )
    service.recall.return_value = RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_old",
                memory_type=MemoryType.SEMANTIC,
                content=existing.content,
                score=0.82,
                explanation=Explanation(semantic=0.82),
                memory=existing,
            )
        ],
    )
    service.get_memory.return_value = existing

    llm = AsyncMock()
    llm.complete.return_value = LLMResponse(
        content='{"decisions":[{"existing_id":"sem_old","decision":"CONTRADICTS","reason":"opposite outcome"}]}',
        model="gpt-4o",
        latency_ms=1,
    )

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="Actually the Jetson deployment failed.",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content="Deployment on Jetson failed due to memory pressure.",
                    reason="new observation",
                )
            ],
        )
    )

    assert result.outcomes[0].action == "contradicted"
    rel_types = {
        call.args[0].relationship_type for call in service.create_relationship.await_args_list
    }
    assert RelationshipType.CONTRADICTS in rel_types


@pytest.mark.asyncio
async def test_episodic_candidate_skips_conflict_detection() -> None:
    service = AsyncMock()
    created_ids = iter(["epi_turn_1", "epi_new"])

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": next(created_ids), "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create
    llm = AsyncMock()

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="It just rained.",
            candidates=[
                MemoryCandidate(
                    classification="EPISODIC",
                    content="Rain started at 3pm.",
                    reason="event",
                )
            ],
        )
    )

    assert result.outcomes[0].action == "created"
    service.recall.assert_not_awaited()
    llm.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_conflict_llm_failure_falls_back_to_create() -> None:
    service = AsyncMock()
    created_ids = iter(["epi_turn_1", "sem_new"])

    async def _create(memory):  # type: ignore[no-untyped-def]
        return memory.model_copy(update={"id": next(created_ids), "created_at": datetime.now(UTC)})

    service.create_memory.side_effect = _create

    existing = SemanticMemory(id="sem_old", content="uses OpenSearch")
    service.recall.return_value = RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_old",
                memory_type=MemoryType.SEMANTIC,
                content=existing.content,
                score=0.9,
                explanation=Explanation(semantic=0.9),
                memory=existing,
            )
        ],
    )

    llm = AsyncMock()
    llm.complete.side_effect = RuntimeError("boom")

    consolidator = ConsolidationService(service=service, llm=llm, settings=_settings())

    result = await consolidator.process(
        ConsolidationJob(
            user_message="I use OpenSearch",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content="user uses OpenSearch",
                    reason="fact",
                )
            ],
        )
    )

    assert result.outcomes[0].action == "created"
    assert result.outcomes[0].created_memory_id == "sem_new"
