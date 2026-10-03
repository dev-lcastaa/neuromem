"""End-to-end storage against the live OpenSearch cluster. Gated by `-m integration`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from opensearchpy import AsyncOpenSearch

from memory.config import get_settings
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory
from memory.storage.opensearch import OpenSearchMemoryStore

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def opensearch_client() -> AsyncIterator[AsyncOpenSearch]:
    s = get_settings()
    auth = None
    if s.opensearch_user and s.opensearch_password:
        auth = (s.opensearch_user, s.opensearch_password)
    client = AsyncOpenSearch(
        hosts=[s.opensearch_url],
        http_auth=auth,
        verify_certs=s.opensearch_verify_certs,
        ssl_show_warn=s.opensearch_verify_certs,
    )
    yield client
    await client.close()


@pytest_asyncio.fixture
async def store(
    opensearch_client: AsyncOpenSearch,
) -> AsyncIterator[OpenSearchMemoryStore]:
    s = OpenSearchMemoryStore(client=opensearch_client, mappings_dir=Path("opensearch/indexes"))
    await s.bootstrap()
    yield s


async def test_bootstrap_is_idempotent(store: OpenSearchMemoryStore) -> None:
    await store.bootstrap()
    await store.bootstrap()


async def test_create_get_relate_traverse(store: OpenSearchMemoryStore) -> None:
    ep = await store.create_memory(
        EpisodicMemory(content="deploy attempt 1 failed", importance=0.7)
    )
    assert ep.id is not None and ep.id.startswith("epi_")

    sem = await store.create_memory(
        SemanticMemory(
            content="opensearch requires vm.max_map_count>=262144",
            subject="opensearch",
            confidence=0.95,
        )
    )
    assert sem.id is not None and sem.id.startswith("sem_")

    got = await store.get_memory(ep.id)
    assert got is not None
    assert got.content == "deploy attempt 1 failed"
    assert got.importance == pytest.approx(0.7)

    rel = await store.create_relationship(
        Relationship(
            from_id=sem.id,
            to_id=ep.id,
            from_type=MemoryType.SEMANTIC,
            to_type=MemoryType.EPISODIC,
            relationship_type=RelationshipType.DERIVED_FROM,
            metadata={"reason": "test-integration"},  # type: ignore[arg-type]
        )
    )
    assert rel.id is not None and rel.id.startswith("rel_")

    rels_from_ep = await store.get_relationships(ep.id)
    rels_from_sem = await store.get_relationships(sem.id)

    assert any(r.id == rel.id for r in rels_from_ep)
    assert any(r.id == rel.id for r in rels_from_sem)


async def test_get_missing_returns_none(store: OpenSearchMemoryStore) -> None:
    assert await store.get_memory("epi_definitely_missing_xxxx") is None
