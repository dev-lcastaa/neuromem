"""Live health checks against the real OpenSearch cluster. Gated by `-m integration`."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_opensearch_reachable() -> None:
    from apps.memory_api.main import create_app

    with TestClient(create_app()) as client:
        r = client.get("/health/opensearch")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") in {"green", "yellow"}
        assert body.get("cluster_name")
        assert body.get("number_of_nodes", 0) >= 1
