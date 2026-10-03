from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.memory_api.main import create_app
from apps.memory_api.routers.memory import get_consolidation_service
from memory.consolidation.models import (
    ConsolidationOutcome,
    ConsolidationResult,
)


@pytest.fixture
def consolidation() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(consolidation: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_consolidation_service] = lambda: consolidation
    return TestClient(app)


def test_consolidate_route_runs_synchronously(client: TestClient, consolidation: AsyncMock) -> None:
    consolidation.process.return_value = ConsolidationResult(
        conversation_id="conv-1",
        source_memory_id="epi_turn_1",
        outcomes=[
            ConsolidationOutcome(
                candidate_index=0,
                classification="SEMANTIC",
                action="created",
                created_memory_id="sem_1",
            )
        ],
    )

    r = client.post(
        "/memory/consolidate",
        json={
            "user_message": "I use OpenSearch.",
            "assistant_reply": "Noted.",
            "candidates": [
                {
                    "classification": "SEMANTIC",
                    "content": "user uses OpenSearch",
                    "importance": 0.7,
                    "confidence": 0.9,
                    "entities": ["opensearch"],
                    "source": "conversation",
                    "reason": "explicit preference",
                }
            ],
            "conversation_id": "conv-1",
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["source_memory_id"] == "epi_turn_1"
    assert body["outcomes"][0]["action"] == "created"
    assert body["outcomes"][0]["created_memory_id"] == "sem_1"
    consolidation.process.assert_awaited_once()


def test_consolidate_route_rejects_missing_message(client: TestClient) -> None:
    r = client.post("/memory/consolidate", json={"assistant_reply": "hi"})
    assert r.status_code == 422
