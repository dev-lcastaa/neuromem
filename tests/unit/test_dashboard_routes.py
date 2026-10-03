from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.memory_api.main import create_app
from apps.memory_api.routers.memory import get_service


@pytest.fixture
def service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(service: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_stats_route(client: TestClient, service: AsyncMock) -> None:
    service.stats.return_value = {
        "counts": {"episodic": 3, "semantic": 4, "procedural": 1},
        "totals": {"memories": 8, "relationships": 5},
        "averages": {"confidence": 0.8, "activation": 0.4, "importance": 0.6},
    }

    r = client.get("/memory/stats")

    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["memories"] == 8
    assert body["averages"]["confidence"] == 0.8


def test_search_route(client: TestClient, service: AsyncMock) -> None:
    service.search_by_text.return_value = [
        {
            "id": "sem_a",
            "memory_type": "semantic",
            "content": "OpenSearch is the vector store",
            "score": 12.5,
            "importance": 0.8,
            "activation": 0.5,
            "created_at": "2026-08-15T00:00:00+00:00",
        }
    ]

    r = client.get(
        "/memory/search",
        params={"q": "opensearch", "limit": 10, "memory_types": ["semantic"]},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "opensearch"
    assert body["results"][0]["id"] == "sem_a"
    call = service.search_by_text.await_args
    assert call.args[0] == "opensearch"


def test_search_route_requires_query(client: TestClient) -> None:
    r = client.get("/memory/search", params={"q": ""})
    assert r.status_code == 422


def test_timeline_route(client: TestClient, service: AsyncMock) -> None:
    service.list_recent.return_value = [
        {
            "id": "epi_a",
            "memory_type": "episodic",
            "content": "event",
            "created_at": "2026-08-15T00:00:00+00:00",
            "importance": 0.5,
            "confidence": 0.9,
            "activation": 0.5,
            "access_count": 0,
        }
    ]
    r = client.get("/memory/timeline", params={"limit": 5})
    assert r.status_code == 200
    assert r.json()["memories"][0]["id"] == "epi_a"


def test_graph_route_404_when_missing(client: TestClient, service: AsyncMock) -> None:
    service.graph_around.return_value = None
    r = client.get("/memory/graph", params={"center_id": "sem_missing"})
    assert r.status_code == 404


def test_graph_route_returns_subgraph(client: TestClient, service: AsyncMock) -> None:
    service.graph_around.return_value = {
        "center_id": "sem_a",
        "nodes": [
            {
                "id": "sem_a",
                "memory_type": "semantic",
                "content": "center",
                "importance": 0.7,
                "activation": 0.6,
            },
            {
                "id": "epi_b",
                "memory_type": "episodic",
                "content": "neighbour",
                "importance": 0.5,
                "activation": 0.4,
            },
        ],
        "edges": [
            {
                "id": "rel_1",
                "from_id": "sem_a",
                "to_id": "epi_b",
                "relationship_type": "DERIVED_FROM",
                "confidence": 0.9,
            }
        ],
    }

    r = client.get("/memory/graph", params={"center_id": "sem_a", "depth": 2})

    assert r.status_code == 200
    body = r.json()
    assert body["center_id"] == "sem_a"
    assert body["depth"] == 2
    assert len(body["nodes"]) == 2
    assert body["edges"][0]["relationship_type"] == "DERIVED_FROM"
