from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.memory_api.main import create_app
from apps.memory_api.routers.memory import get_service
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallResponse,
    RecallResult,
)


@pytest.fixture
def service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(service: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_recall_returns_ranked_response(client: TestClient, service: AsyncMock) -> None:
    now = datetime.now(UTC)
    memory = SemanticMemory(
        id="sem_a", content="opensearch is used", importance=0.9, created_at=now
    )
    service.recall.return_value = RecallResponse(
        query="opensearch",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_a",
                memory_type=MemoryType.SEMANTIC,
                content="opensearch is used",
                score=0.91,
                explanation=Explanation(semantic=0.94, keyword=0.83, importance=0.9, recency=0.72),
                memory=memory,
            )
        ],
    )

    r = client.post("/memory/recall", json={"query": "opensearch"})

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "opensearch"
    assert body["memories"][0]["id"] == "sem_a"
    assert body["memories"][0]["score"] == 0.91
    assert body["memories"][0]["explanation"]["semantic"] == 0.94
    assert body["memories"][0]["memory"]["memory_type"] == "semantic"
    assert body["weights"]["semantic"] == 0.40
    service.recall.assert_awaited_once()


def test_recall_rejects_empty_query(client: TestClient) -> None:
    r = client.post("/memory/recall", json={"query": ""})
    assert r.status_code == 422


def test_recall_rejects_extra_fields(client: TestClient) -> None:
    r = client.post("/memory/recall", json={"query": "x", "bogus": 1})
    assert r.status_code == 422


def test_recall_accepts_full_options(client: TestClient, service: AsyncMock) -> None:
    service.recall.return_value = RecallResponse(query="q", weights=RankWeights(), memories=[])

    r = client.post(
        "/memory/recall",
        json={
            "query": "q",
            "memory_types": ["semantic"],
            "limit": 5,
            "candidate_pool": 25,
            "include_relationships": False,
            "weights": {
                "semantic": 1.0,
                "keyword": 0.0,
                "importance": 0.0,
                "recency": 0.0,
                "activation": 0.0,
                "relationship": 0.0,
            },
        },
    )

    assert r.status_code == 200
    call = service.recall.await_args.args[0]
    assert call.limit == 5
    assert call.memory_types == [MemoryType.SEMANTIC]
    assert call.include_relationships is False
    assert call.weights is not None and call.weights.semantic == 1.0
