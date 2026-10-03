from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from memory.config import Settings
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory
from memory.retrieval.models import RecallQuery
from memory.service import MemoryService
from memory.storage.interface import SearchHit


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _fake_embedder(dim: int = 8) -> AsyncMock:
    e = AsyncMock()
    e.model = "text-embedding-3-large"
    e.dim = dim
    e.embed.return_value = [[0.1] * dim]
    return e


@pytest.mark.asyncio
async def test_create_memory_embeds_when_vector_missing() -> None:
    store = AsyncMock()
    store.create_memory.side_effect = lambda m: m
    embedder = _fake_embedder(dim=4)

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    saved = await svc.create_memory(EpisodicMemory(content="something"))

    assert saved.content_vector == [0.1, 0.1, 0.1, 0.1]
    assert saved.metadata.embedding_model == "text-embedding-3-large"
    assert saved.metadata.embedding_dim == 4
    embedder.embed.assert_awaited_once_with(["something"])


@pytest.mark.asyncio
async def test_create_memory_skips_embedding_when_vector_provided() -> None:
    store = AsyncMock()
    store.create_memory.side_effect = lambda m: m
    embedder = _fake_embedder()

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    await svc.create_memory(EpisodicMemory(content="x", content_vector=[0.5, 0.6]))

    embedder.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_returns_empty_when_no_hits() -> None:
    store = AsyncMock()
    store.search_semantic.return_value = []
    store.search_keyword.return_value = []
    embedder = _fake_embedder()

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    resp = await svc.recall(RecallQuery(query="anything"))

    assert resp.query == "anything"
    assert resp.memories == []
    store.get_memories_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_orchestrates_and_ranks() -> None:
    store = AsyncMock()
    embedder = _fake_embedder()

    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=0.95),
        SearchHit(id="epi_b", memory_type=MemoryType.EPISODIC, score=0.4),
    ]
    store.search_keyword.return_value = [
        SearchHit(id="epi_b", memory_type=MemoryType.EPISODIC, score=12.0),
        SearchHit(id="sem_c", memory_type=MemoryType.SEMANTIC, score=8.0),
    ]

    now = datetime.now(UTC)
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="opensearch is used", importance=0.9, created_at=now),
        EpisodicMemory(id="epi_b", content="deploy failed", importance=0.5, created_at=now),
        SemanticMemory(id="sem_c", content="something else", importance=0.2, created_at=now),
    ]
    store.get_relationships_for_ids.return_value = [
        Relationship(
            id="rel_1",
            from_id="sem_a",
            to_id="epi_b",
            from_type=MemoryType.SEMANTIC,
            to_type=MemoryType.EPISODIC,
            relationship_type=RelationshipType.DERIVED_FROM,
            confidence=0.9,
            weight=1.0,
        ),
    ]

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    resp = await svc.recall(RecallQuery(query="opensearch usage", limit=5))

    assert resp.query == "opensearch usage"
    assert len(resp.memories) == 3
    ids = [r.id for r in resp.memories]
    # sem_a has the highest semantic (0.95) + highest importance (0.9) + relationship boost -> #1
    assert ids[0] == "sem_a"
    top = resp.memories[0]
    assert top.explanation.semantic == 0.95
    assert top.explanation.importance == 0.9
    assert top.explanation.relationship > 0.0
    # weights echoed
    assert resp.weights.semantic == 0.40


@pytest.mark.asyncio
async def test_recall_normalises_keyword_scores() -> None:
    store = AsyncMock()
    embedder = _fake_embedder()

    store.search_semantic.return_value = []
    store.search_keyword.return_value = [
        SearchHit(id="epi_a", memory_type=MemoryType.EPISODIC, score=10.0),
        SearchHit(id="epi_b", memory_type=MemoryType.EPISODIC, score=5.0),
    ]

    now = datetime.now(UTC)
    store.get_memories_by_ids.return_value = [
        EpisodicMemory(id="epi_a", content="a", created_at=now),
        EpisodicMemory(id="epi_b", content="b", created_at=now),
    ]
    store.get_relationships_for_ids.return_value = []

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    resp = await svc.recall(RecallQuery(query="q"))

    scores = {r.id: r.explanation.keyword for r in resp.memories}
    assert scores["epi_a"] == 1.0
    assert scores["epi_b"] == 0.5


@pytest.mark.asyncio
async def test_recall_skips_relationships_when_disabled() -> None:
    store = AsyncMock()
    embedder = _fake_embedder()

    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=1.0)
    ]
    store.search_keyword.return_value = []
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="x", created_at=datetime.now(UTC))
    ]

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    resp = await svc.recall(RecallQuery(query="q", include_relationships=False))

    store.get_relationships_for_ids.assert_not_awaited()
    assert resp.memories[0].explanation.relationship == 0.0


@pytest.mark.asyncio
async def test_recall_honours_custom_weights() -> None:
    store = AsyncMock()
    embedder = _fake_embedder()

    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=1.0)
    ]
    store.search_keyword.return_value = []
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="x", importance=0.5, created_at=datetime.now(UTC))
    ]
    store.get_relationships_for_ids.return_value = []

    svc = MemoryService(store=store, embedder=embedder, settings=_settings())
    from memory.retrieval.models import RankWeights

    resp = await svc.recall(
        RecallQuery(
            query="q",
            weights=RankWeights(
                semantic=1.0,
                keyword=0.0,
                importance=0.0,
                recency=0.0,
                activation=0.0,
                relationship=0.0,
            ),
        )
    )
    assert resp.weights.semantic == 1.0
    assert resp.memories[0].score == pytest.approx(1.0, abs=1e-4)
