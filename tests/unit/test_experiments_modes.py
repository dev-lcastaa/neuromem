from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from experiments.modes import NaiveRAGMode, NeuroMemMode
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallResponse,
    RecallResult,
)
from memory.storage.interface import SearchHit


def _fake_embedder(dim: int = 4) -> AsyncMock:
    e = AsyncMock()
    e.model = "text-embedding-3-large"
    e.dim = dim
    e.embed.return_value = [[0.1] * dim]
    return e


@pytest.mark.asyncio
async def test_naive_rag_uses_semantic_search_only() -> None:
    store = AsyncMock()
    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=0.9),
        SearchHit(id="sem_b", memory_type=MemoryType.SEMANTIC, score=0.5),
    ]
    now = datetime.now(UTC)
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="fact A", created_at=now),
        SemanticMemory(id="sem_b", content="fact B", created_at=now),
    ]

    mode = NaiveRAGMode(embedder=_fake_embedder(), store=store)
    results = await mode.retrieve("question", k=5)

    assert [r.id for r in results] == ["sem_a", "sem_b"]
    assert results[0].explanation.semantic == 0.9
    # Only relies on semantic + mget; never touches keyword / relationships / recall
    store.search_semantic.assert_awaited_once()
    store.search_keyword.assert_not_called()
    store.get_relationships_for_ids.assert_not_called()


@pytest.mark.asyncio
async def test_naive_rag_returns_empty_on_no_hits() -> None:
    store = AsyncMock()
    store.search_semantic.return_value = []

    mode = NaiveRAGMode(embedder=_fake_embedder(), store=store)
    assert await mode.retrieve("q", 5) == []
    store.get_memories_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_neuromem_mode_delegates_to_service_recall() -> None:
    service = AsyncMock()
    now = datetime.now(UTC)
    service.recall.return_value = RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_x",
                memory_type=MemoryType.SEMANTIC,
                content="ok",
                score=0.7,
                explanation=Explanation(semantic=0.94),
                memory=SemanticMemory(id="sem_x", content="ok", created_at=now),
            )
        ],
    )

    mode = NeuroMemMode(service=service)
    results = await mode.retrieve("question", k=3)

    assert [r.id for r in results] == ["sem_x"]
    service.recall.assert_awaited_once()
    call = service.recall.await_args.args[0]
    assert call.query == "question"
    assert call.limit == 3
