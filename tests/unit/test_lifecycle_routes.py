from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.memory_api.main import create_app
from apps.memory_api.routers.memory import get_service
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory


@pytest.fixture
def service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(service: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_access_route_uses_default_bump(client: TestClient, service: AsyncMock) -> None:
    service.record_access.return_value = 3

    r = client.post("/memory/access", json={"memory_ids": ["a", "b", "c"]})

    assert r.status_code == 200
    assert r.json() == {"updated": 3}
    service.record_access.assert_awaited_once_with(["a", "b", "c"], None)


def test_access_route_forwards_bump(client: TestClient, service: AsyncMock) -> None:
    service.record_access.return_value = 1

    r = client.post("/memory/access", json={"memory_ids": ["a"], "bump": 0.15})

    assert r.status_code == 200
    service.record_access.assert_awaited_once_with(["a"], 0.15)


def test_access_route_rejects_empty(client: TestClient) -> None:
    r = client.post("/memory/access", json={"memory_ids": []})
    assert r.status_code == 422


def test_decay_route_returns_reports(client: TestClient, service: AsyncMock) -> None:
    service.decay_report.return_value = [
        {
            "memory_id": "sem_a",
            "found": True,
            "activation": 0.8,
            "access_count": 3,
            "last_accessed_at": "2026-08-01T00:00:00+00:00",
            "age_days": 14.0,
            "decay_factor": 0.4966,
            "effective_activation": 0.3972,
        },
        {"memory_id": "epi_missing", "found": False},
    ]

    r = client.post("/memory/decay", json={"memory_ids": ["sem_a", "epi_missing"]})

    assert r.status_code == 200
    body = r.json()
    assert body["reports"][0]["effective_activation"] == 0.3972
    assert body["reports"][1]["found"] is False


def test_memory_history_route(client: TestClient, service: AsyncMock) -> None:
    now = datetime.now(UTC)
    accessed = now - timedelta(days=2)
    memory = SemanticMemory(
        id="sem_a",
        content="fact",
        activation=0.7,
        access_count=4,
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=1),
        last_accessed_at=accessed,
    )
    service.get_memory.return_value = memory
    service.decay_report.return_value = [
        {
            "memory_id": "sem_a",
            "found": True,
            "activation": 0.7,
            "access_count": 4,
            "last_accessed_at": accessed.isoformat(),
            "age_days": 2.0,
            "decay_factor": 0.9048,
            "effective_activation": 0.6334,
        }
    ]
    service.get_relationships.return_value = []

    r = client.get("/memory/sem_a/history")

    assert r.status_code == 200
    body = r.json()
    assert body["memory_id"] == "sem_a"
    assert body["memory_type"] == MemoryType.SEMANTIC.value
    assert body["access_count"] == 4
    assert body["activation"] == 0.7
    assert body["effective_activation"] == 0.6334


def test_memory_history_route_404(client: TestClient, service: AsyncMock) -> None:
    service.get_memory.return_value = None
    r = client.get("/memory/sem_missing/history")
    assert r.status_code == 404
