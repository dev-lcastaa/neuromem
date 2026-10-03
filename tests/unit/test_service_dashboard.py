from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from memory.config import Settings
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory
from memory.service import MemoryService
from memory.storage.interface import SearchHit


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _fake_embedder() -> AsyncMock:
    e = AsyncMock()
    e.model = "text-embedding-3-large"
    e.dim = 4
    return e


@pytest.mark.asyncio
async def test_stats_aggregates_counts_and_averages() -> None:
    store = AsyncMock()
    store.count_memories_per_type.return_value = {
        "episodic": 5,
        "semantic": 7,
        "procedural": 2,
    }
    store.count_relationships.return_value = 10
    store.aggregate_averages.return_value = {
        "confidence": 0.83,
        "activation": 0.4,
        "importance": 0.65,
    }

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    stats = await svc.stats()

    assert stats["totals"]["memories"] == 14
    assert stats["totals"]["relationships"] == 10
    assert stats["counts"]["semantic"] == 7
    assert stats["averages"]["confidence"] == 0.83


@pytest.mark.asyncio
async def test_search_by_text_returns_hit_scored_memories() -> None:
    store = AsyncMock()
    store.search_keyword.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=12.5),
        SearchHit(id="epi_b", memory_type=MemoryType.EPISODIC, score=4.0),
    ]
    now = datetime.now(UTC)
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="A", created_at=now, importance=0.8),
        EpisodicMemory(id="epi_b", content="B", created_at=now, importance=0.4),
    ]

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    results = await svc.search_by_text("opensearch")

    assert [r["id"] for r in results] == ["sem_a", "epi_b"]
    assert results[0]["score"] == 12.5
    assert results[0]["memory_type"] == "semantic"


@pytest.mark.asyncio
async def test_search_by_text_empty_query_short_circuits() -> None:
    store = AsyncMock()
    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    assert await svc.search_by_text("   ") == []
    store.search_keyword.assert_not_called()


@pytest.mark.asyncio
async def test_list_recent_returns_dicts() -> None:
    store = AsyncMock()
    now = datetime.now(UTC)
    store.list_recent.return_value = [
        EpisodicMemory(id="epi_a", content="event", created_at=now, importance=0.5),
    ]

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    out = await svc.list_recent(limit=10)

    assert len(out) == 1
    assert out[0]["id"] == "epi_a"
    assert out[0]["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_graph_around_expands_to_neighbours() -> None:
    store = AsyncMock()
    now = datetime.now(UTC)
    center = SemanticMemory(id="sem_a", content="center", created_at=now)
    neighbour = EpisodicMemory(id="epi_b", content="neighbour", created_at=now)

    store.get_memory.return_value = center
    store.get_relationships_for_ids.side_effect = [
        [
            Relationship(
                id="rel_1",
                from_id="sem_a",
                to_id="epi_b",
                from_type=MemoryType.SEMANTIC,
                to_type=MemoryType.EPISODIC,
                relationship_type=RelationshipType.DERIVED_FROM,
                confidence=0.9,
            )
        ],
        [],
    ]
    store.get_memories_by_ids.return_value = [neighbour]

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    result = await svc.graph_around("sem_a", depth=2)

    assert result is not None
    assert result["center_id"] == "sem_a"
    ids = {n["id"] for n in result["nodes"]}
    assert ids == {"sem_a", "epi_b"}
    assert result["edges"][0]["relationship_type"] == "DERIVED_FROM"


@pytest.mark.asyncio
async def test_graph_around_returns_none_when_center_missing() -> None:
    store = AsyncMock()
    store.get_memory.return_value = None
    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())

    assert await svc.graph_around("missing") is None
