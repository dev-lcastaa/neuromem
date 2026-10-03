from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.memory_api.main import create_app
from apps.memory_api.routers.memory import get_service
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory


@pytest.fixture
def service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(service: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_store_returns_created_memory(client: TestClient, service: AsyncMock) -> None:
    now = datetime.now(UTC)
    service.create_memory.return_value = EpisodicMemory(
        id="epi_abc",
        content="a thing",
        importance=0.7,
        created_at=now,
        updated_at=now,
    )

    r = client.post(
        "/memory/store",
        json={"memory_type": "episodic", "content": "a thing", "importance": 0.7},
    )

    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "epi_abc"
    assert body["memory_type"] == "episodic"
    assert body["importance"] == 0.7
    service.create_memory.assert_awaited_once()


def test_store_rejects_invalid_memory_type(client: TestClient) -> None:
    r = client.post(
        "/memory/store",
        json={"memory_type": "nonsense", "content": "x"},
    )
    assert r.status_code == 422


def test_get_memory_returns_document(client: TestClient, service: AsyncMock) -> None:
    service.get_memory.return_value = SemanticMemory(id="sem_x", content="fact", subject="sky")
    r = client.get("/memory/sem_x")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "sem_x"
    assert body["memory_type"] == "semantic"
    assert body["subject"] == "sky"
    service.get_memory.assert_awaited_once_with("sem_x")


def test_get_memory_404_when_missing(client: TestClient, service: AsyncMock) -> None:
    service.get_memory.return_value = None
    r = client.get("/memory/epi_missing")
    assert r.status_code == 404
    assert "epi_missing" in r.json()["detail"]


def test_relate_returns_created_relationship(client: TestClient, service: AsyncMock) -> None:
    now = datetime.now(UTC)
    service.create_relationship.return_value = Relationship(
        id="rel_1",
        from_id="epi_a",
        to_id="sem_b",
        from_type=MemoryType.EPISODIC,
        to_type=MemoryType.SEMANTIC,
        relationship_type=RelationshipType.DERIVED_FROM,
        created_at=now,
    )

    r = client.post(
        "/memory/relate",
        json={
            "from_id": "epi_a",
            "to_id": "sem_b",
            "from_type": "episodic",
            "to_type": "semantic",
            "relationship_type": "DERIVED_FROM",
        },
    )

    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "rel_1"
    assert body["relationship_type"] == "DERIVED_FROM"


def test_get_relationships_returns_list(client: TestClient, service: AsyncMock) -> None:
    service.get_relationships.return_value = [
        Relationship(
            id="rel_1",
            from_id="epi_a",
            to_id="sem_b",
            from_type=MemoryType.EPISODIC,
            to_type=MemoryType.SEMANTIC,
            relationship_type=RelationshipType.RELATED_TO,
        ),
    ]

    r = client.get("/memory/epi_a/relationships")

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["id"] == "rel_1"
    service.get_relationships.assert_awaited_once_with("epi_a")
