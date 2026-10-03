from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memory.storage.opensearch import OpenSearchMemoryStore


def _mock_client() -> SimpleNamespace:
    return SimpleNamespace(
        indices=SimpleNamespace(exists=AsyncMock(), create=AsyncMock()),
        index=AsyncMock(),
        get=AsyncMock(),
        update=AsyncMock(),
        search=AsyncMock(),
        mget=AsyncMock(),
        bulk=AsyncMock(),
    )


@pytest.fixture
def mappings_dir(tmp_path: Path) -> Path:
    for name in ("episodic", "semantic", "procedural", "relationships"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"settings": {}, "mappings": {"properties": {}}}),
            encoding="utf-8",
        )
    return tmp_path


@pytest.mark.asyncio
async def test_record_access_empty_short_circuits(mappings_dir: Path) -> None:
    client = _mock_client()
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    updated = await store.record_access_batch([], bump=0.05)

    assert updated == 0
    client.bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_access_bulk_payload_shape(mappings_dir: Path) -> None:
    client = _mock_client()
    client.bulk.return_value = {
        "errors": False,
        "items": [
            {"update": {"status": 200}},
            {"update": {"status": 200}},
        ],
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    updated = await store.record_access_batch(["epi_a", "sem_b"], bump=0.1)

    assert updated == 2
    body = client.bulk.await_args.kwargs["body"]
    assert body[0] == {"update": {"_index": "neuromem-episodic", "_id": "epi_a"}}
    assert body[2] == {"update": {"_index": "neuromem-semantic", "_id": "sem_b"}}
    script = body[1]["script"]
    assert "activation" in script["source"]
    assert "access_count" in script["source"]
    assert "last_accessed_at" in script["source"]
    assert script["params"]["bump"] == 0.1


@pytest.mark.asyncio
async def test_record_access_counts_partial_success(mappings_dir: Path) -> None:
    client = _mock_client()
    client.bulk.return_value = {
        "errors": True,
        "items": [
            {"update": {"status": 200}},
            {"update": {"status": 404}},
            {"update": {"status": 200}},
        ],
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    updated = await store.record_access_batch(["epi_a", "sem_b", "pro_c"], bump=0.05)

    assert updated == 2
