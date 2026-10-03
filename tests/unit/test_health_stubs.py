"""Health-endpoint tests without upstream dependencies.

`GET /health` and the memory stub routes work without network. OpenSearch/LLM/
embedding probes are covered by integration tests.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from apps.memory_api.main import create_app

    return TestClient(create_app())


def test_liveness() -> None:
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_health_llm_skipped_when_disabled() -> None:
    with _client() as client:
        r = client.get("/health/llm")
        assert r.status_code == 200
        assert r.json()["status"] == "skipped"


def test_health_embeddings_skipped_when_disabled() -> None:
    with _client() as client:
        r = client.get("/health/embeddings")
        assert r.status_code == 200
        assert r.json()["status"] == "skipped"


def test_memory_conflicts_resolve_returns_501() -> None:
    with _client() as client:
        r = client.post("/memory/conflicts/resolve")
        assert r.status_code == 501
