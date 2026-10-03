from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from opensearchpy import NotFoundError

from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory
from memory.storage.opensearch import OpenSearchMemoryStore


def _mock_client() -> SimpleNamespace:
    return SimpleNamespace(
        indices=SimpleNamespace(
            exists=AsyncMock(),
            create=AsyncMock(),
        ),
        index=AsyncMock(),
        get=AsyncMock(),
        update=AsyncMock(),
        search=AsyncMock(),
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
async def test_bootstrap_creates_missing_indexes(mappings_dir: Path) -> None:
    client = _mock_client()
    client.indices.exists.side_effect = [False, True, False, True]
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.bootstrap()

    assert client.indices.exists.await_count == 4
    assert client.indices.create.await_count == 2


@pytest.mark.asyncio
async def test_bootstrap_skips_when_all_present(mappings_dir: Path) -> None:
    client = _mock_client()
    client.indices.exists.return_value = True
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.bootstrap()

    client.indices.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_memory_assigns_id_and_timestamps(mappings_dir: Path) -> None:
    client = _mock_client()
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    memory = EpisodicMemory(content="a thing happened", importance=0.7)
    saved = await store.create_memory(memory)

    assert saved.id is not None and saved.id.startswith("epi_")
    assert saved.created_at is not None
    assert saved.updated_at is not None

    call = client.index.await_args
    assert call.kwargs["index"] == "neuromem-episodic"
    assert call.kwargs["id"] == saved.id
    assert call.kwargs["refresh"] == "wait_for"
    body = call.kwargs["body"]
    assert body["content"] == "a thing happened"
    assert "id" not in body


@pytest.mark.asyncio
async def test_create_memory_preserves_provided_id(mappings_dir: Path) -> None:
    client = _mock_client()
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    fixed_at = datetime(2026, 1, 1, tzinfo=UTC)
    memory = SemanticMemory(
        id="sem_fixed", content="fact", created_at=fixed_at, updated_at=fixed_at
    )
    saved = await store.create_memory(memory)

    assert saved.id == "sem_fixed"
    assert saved.created_at == fixed_at
    assert client.index.await_args.kwargs["index"] == "neuromem-semantic"


@pytest.mark.asyncio
async def test_get_memory_returns_none_on_missing(mappings_dir: Path) -> None:
    client = _mock_client()

    async def _raise(**kwargs: Any) -> dict[str, Any]:
        raise NotFoundError(404, "not_found", {})

    client.get.side_effect = _raise
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    result = await store.get_memory("epi_missing")
    assert result is None


@pytest.mark.asyncio
async def test_get_memory_reconstructs_discriminated_type(mappings_dir: Path) -> None:
    client = _mock_client()
    client.get.return_value = {
        "_source": {
            "memory_type": "semantic",
            "content": "fact",
            "subject": "sky",
            "importance": 0.4,
            "confidence": 0.9,
            "activation": 0.5,
            "access_count": 0,
        }
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    got = await store.get_memory("sem_abc")

    assert isinstance(got, SemanticMemory)
    assert got.id == "sem_abc"
    assert got.subject == "sky"
    call = client.get.await_args
    assert call.kwargs["_source_excludes"] == ["content_vector"]


@pytest.mark.asyncio
async def test_get_memory_include_vector_omits_source_excludes(
    mappings_dir: Path,
) -> None:
    client = _mock_client()
    client.get.return_value = {
        "_source": {
            "memory_type": "episodic",
            "content": "x",
            "content_vector": [0.1, 0.2],
            "importance": 0.5,
            "confidence": 0.5,
            "activation": 0.5,
            "access_count": 0,
        }
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    got = await store.get_memory("epi_abc", include_vector=True)

    assert got is not None and got.content_vector == [0.1, 0.2]
    assert "_source_excludes" not in client.get.await_args.kwargs


@pytest.mark.asyncio
async def test_create_relationship_assigns_id_and_created_at(
    mappings_dir: Path,
) -> None:
    client = _mock_client()
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    rel = Relationship(
        from_id="epi_1",
        to_id="sem_2",
        from_type=MemoryType.EPISODIC,
        to_type=MemoryType.SEMANTIC,
        relationship_type=RelationshipType.DERIVED_FROM,
    )
    saved = await store.create_relationship(rel)

    assert saved.id is not None and saved.id.startswith("rel_")
    assert saved.created_at is not None
    assert client.index.await_args.kwargs["index"] == "neuromem-relationships"


@pytest.mark.asyncio
async def test_get_relationships_queries_both_directions(mappings_dir: Path) -> None:
    client = _mock_client()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "rel_1",
                    "_source": {
                        "from_id": "epi_a",
                        "to_id": "sem_b",
                        "from_type": "episodic",
                        "to_type": "semantic",
                        "relationship_type": "DERIVED_FROM",
                        "confidence": 0.9,
                        "weight": 1.0,
                        "created_by": "system",
                    },
                }
            ]
        }
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    rels = await store.get_relationships("epi_a")

    assert len(rels) == 1
    assert rels[0].id == "rel_1"
    body = client.search.await_args.kwargs["body"]
    should = body["query"]["bool"]["should"]
    assert {"term": {"from_id": "epi_a"}} in should
    assert {"term": {"to_id": "epi_a"}} in should
