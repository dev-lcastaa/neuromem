from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from memory.config import Settings
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory
from memory.retrieval.models import RecallQuery
from memory.service import MemoryService
from memory.storage.interface import SearchHit


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "strengthening_enabled": True,
        "activation_reinforcement": 0.05,
        "activation_decay_lambda": 0.05,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def _fake_embedder(dim: int = 8) -> AsyncMock:
    e = AsyncMock()
    e.model = "text-embedding-3-large"
    e.dim = dim
    e.embed.return_value = [[0.1] * dim]
    return e


@pytest.mark.asyncio
async def test_record_access_uses_default_bump_from_settings() -> None:
    store = AsyncMock()
    store.record_access_batch.return_value = 2

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    updated = await svc.record_access(["epi_a", "sem_b"])

    assert updated == 2
    store.record_access_batch.assert_awaited_once_with(["epi_a", "sem_b"], 0.05)


@pytest.mark.asyncio
async def test_record_access_accepts_bump_override() -> None:
    store = AsyncMock()
    store.record_access_batch.return_value = 1

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    await svc.record_access(["epi_a"], bump=0.2)

    store.record_access_batch.assert_awaited_once_with(["epi_a"], 0.2)


@pytest.mark.asyncio
async def test_decay_report_computes_effective_activation() -> None:
    store = AsyncMock()
    now = datetime.now(UTC)
    ten_days_ago = now - timedelta(days=10)
    memory = SemanticMemory(
        id="sem_a",
        content="x",
        activation=0.8,
        access_count=3,
        created_at=ten_days_ago,
        last_accessed_at=ten_days_ago,
    )
    store.get_memories_by_ids.return_value = [memory]

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    reports = await svc.decay_report(["sem_a"])

    assert len(reports) == 1
    rep = reports[0]
    assert rep["memory_id"] == "sem_a"
    assert rep["found"] is True
    assert rep["activation"] == 0.8
    assert rep["access_count"] == 3
    assert rep["age_days"] > 9.0
    assert rep["decay_factor"] < 1.0
    assert rep["effective_activation"] < 0.8


@pytest.mark.asyncio
async def test_decay_report_flags_missing_memory() -> None:
    store = AsyncMock()
    store.get_memories_by_ids.return_value = []

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    reports = await svc.decay_report(["epi_missing"])

    assert reports == [{"memory_id": "epi_missing", "found": False}]


@pytest.mark.asyncio
async def test_recall_schedules_strengthening_when_enabled() -> None:
    store = AsyncMock()
    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=0.9)
    ]
    store.search_keyword.return_value = []
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="x", created_at=datetime.now(UTC))
    ]
    store.get_relationships_for_ids.return_value = []
    store.record_access_batch.return_value = 1

    svc = MemoryService(store=store, embedder=_fake_embedder(), settings=_settings())
    await svc.recall(RecallQuery(query="q"))
    # Yield the loop so the fire-and-forget task can run
    await asyncio.sleep(0)

    store.record_access_batch.assert_awaited_once()
    call_ids, call_bump = store.record_access_batch.await_args.args
    assert call_ids == ["sem_a"]
    assert call_bump == 0.05


@pytest.mark.asyncio
async def test_recall_skips_strengthening_when_disabled() -> None:
    store = AsyncMock()
    store.search_semantic.return_value = [
        SearchHit(id="sem_a", memory_type=MemoryType.SEMANTIC, score=0.9)
    ]
    store.search_keyword.return_value = []
    store.get_memories_by_ids.return_value = [
        SemanticMemory(id="sem_a", content="x", created_at=datetime.now(UTC))
    ]
    store.get_relationships_for_ids.return_value = []

    svc = MemoryService(
        store=store,
        embedder=_fake_embedder(),
        settings=_settings(strengthening_enabled=False),
    )
    await svc.recall(RecallQuery(query="q"))
    await asyncio.sleep(0)

    store.record_access_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_recall_scores_use_activation_signal() -> None:
    store = AsyncMock()
    now = datetime.now(UTC)
    store.search_semantic.return_value = [
        SearchHit(id="sem_hot", memory_type=MemoryType.SEMANTIC, score=0.5),
        SearchHit(id="sem_cold", memory_type=MemoryType.SEMANTIC, score=0.5),
    ]
    store.search_keyword.return_value = []
    # Same semantic score, same importance, same recency; only activation differs
    store.get_memories_by_ids.return_value = [
        SemanticMemory(
            id="sem_hot",
            content="hot",
            activation=0.9,
            created_at=now,
            last_accessed_at=now,
        ),
        SemanticMemory(
            id="sem_cold",
            content="cold",
            activation=0.1,
            created_at=now,
            last_accessed_at=now,
        ),
    ]
    store.get_relationships_for_ids.return_value = []

    svc = MemoryService(
        store=store,
        embedder=_fake_embedder(),
        settings=_settings(strengthening_enabled=False),
    )
    resp = await svc.recall(RecallQuery(query="q"))

    ids = [r.id for r in resp.memories]
    assert ids[0] == "sem_hot"
    hot, cold = (
        {r.id: r for r in resp.memories}["sem_hot"],
        {r.id: r for r in resp.memories}["sem_cold"],
    )
    assert hot.explanation.activation > cold.explanation.activation
