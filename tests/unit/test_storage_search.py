from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memory.domain.enums import MemoryType
from memory.storage.opensearch import OpenSearchMemoryStore


def _mock_client() -> SimpleNamespace:
    return SimpleNamespace(
        indices=SimpleNamespace(exists=AsyncMock(), create=AsyncMock()),
        index=AsyncMock(),
        get=AsyncMock(),
        update=AsyncMock(),
        search=AsyncMock(),
        mget=AsyncMock(),
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
async def test_search_semantic_targets_all_indexes_by_default(
    mappings_dir: Path,
) -> None:
    client = _mock_client()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.search_semantic([0.1, 0.2, 0.3], size=25)

    call = client.search.await_args
    assert call.kwargs["index"] == ("neuromem-episodic,neuromem-semantic,neuromem-procedural")
    body = call.kwargs["body"]
    assert body["size"] == 25
    knn = body["query"]["knn"]["content_vector"]
    assert knn["vector"] == [0.1, 0.2, 0.3]
    assert knn["k"] == 25
    assert body["_source"] == ["memory_type"]


@pytest.mark.asyncio
async def test_search_semantic_filters_memory_types(mappings_dir: Path) -> None:
    client = _mock_client()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.search_semantic([0.0], memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL])

    assert client.search.await_args.kwargs["index"] == "neuromem-semantic,neuromem-procedural"


@pytest.mark.asyncio
async def test_search_semantic_parses_hits(mappings_dir: Path) -> None:
    client = _mock_client()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "sem_1",
                    "_score": 0.94,
                    "_source": {"memory_type": "semantic"},
                },
                {
                    "_id": "epi_2",
                    "_score": 0.5,
                    "_source": {"memory_type": "episodic"},
                },
            ]
        }
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    hits = await store.search_semantic([0.1])

    assert len(hits) == 2
    assert hits[0].id == "sem_1"
    assert hits[0].memory_type is MemoryType.SEMANTIC
    assert hits[0].score == 0.94


@pytest.mark.asyncio
async def test_search_keyword_uses_multi_match_lenient(mappings_dir: Path) -> None:
    client = _mock_client()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.search_keyword("what is opensearch", size=10)

    body = client.search.await_args.kwargs["body"]
    mm = body["query"]["multi_match"]
    assert mm["query"] == "what is opensearch"
    assert mm["lenient"] is True
    assert "content^2" in mm["fields"]


@pytest.mark.asyncio
async def test_get_memories_by_ids_uses_mget(mappings_dir: Path) -> None:
    client = _mock_client()
    client.mget.return_value = {
        "docs": [
            {
                "_id": "sem_a",
                "found": True,
                "_source": {
                    "memory_type": "semantic",
                    "content": "fact",
                    "importance": 0.5,
                    "confidence": 0.5,
                    "activation": 0.5,
                    "access_count": 0,
                },
            },
            {"_id": "epi_missing", "found": False},
        ]
    }
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    got = await store.get_memories_by_ids(["sem_a", "epi_missing"])

    assert len(got) == 1
    assert got[0].id == "sem_a"
    body = client.mget.await_args.kwargs["body"]
    assert body["docs"] == [
        {"_index": "neuromem-semantic", "_id": "sem_a"},
        {"_index": "neuromem-episodic", "_id": "epi_missing"},
    ]
    assert client.mget.await_args.kwargs["_source_excludes"] == ["content_vector"]


@pytest.mark.asyncio
async def test_get_memories_by_ids_empty_short_circuits(mappings_dir: Path) -> None:
    client = _mock_client()
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    assert await store.get_memories_by_ids([]) == []
    client.mget.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_relationships_for_ids_terms_query(mappings_dir: Path) -> None:
    client = _mock_client()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchMemoryStore(client=client, mappings_dir=mappings_dir)

    await store.get_relationships_for_ids(["epi_a", "sem_b"])

    body = client.search.await_args.kwargs["body"]
    should = body["query"]["bool"]["should"]
    assert {"terms": {"from_id": ["epi_a", "sem_b"]}} in should
    assert {"terms": {"to_id": ["epi_a", "sem_b"]}} in should
