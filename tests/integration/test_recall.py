"""End-to-end recall against the live OpenSearch cluster + real OpenAI embeddings.

Runs `-m integration`. Expensive: makes real embedding calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
import ulid
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from memory.config import get_settings
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import EpisodicMemory, Relationship, SemanticMemory
from memory.retrieval.models import RecallQuery
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from models.embeddings import OpenAIEmbeddingProvider

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def service() -> AsyncIterator[MemoryService]:
    s = get_settings()
    if not s.openai_api_key or s.openai_api_key.startswith("sk-test"):
        pytest.skip("OPENAI_API_KEY not set for real embedding calls")

    auth = None
    if s.opensearch_user and s.opensearch_password:
        auth = (s.opensearch_user, s.opensearch_password)
    os_client = AsyncOpenSearch(
        hosts=[s.opensearch_url],
        http_auth=auth,
        verify_certs=s.opensearch_verify_certs,
        ssl_show_warn=s.opensearch_verify_certs,
    )
    ai_client = AsyncOpenAI(api_key=s.openai_api_key)
    store = OpenSearchMemoryStore(client=os_client, mappings_dir=Path("opensearch/indexes"))
    await store.bootstrap()
    embedder = OpenAIEmbeddingProvider(
        client=ai_client,
        model=s.openai_model_embedding,
        dim=s.embedding_dim,
        batch_max=s.embedding_batch_max,
    )
    yield MemoryService(store=store, embedder=embedder, settings=s)
    await os_client.close()
    await ai_client.close()


async def test_recall_returns_semantic_top_match(service: MemoryService) -> None:
    marker = f"rlx-{ulid.new()!s}"
    fact = await service.create_memory(
        SemanticMemory(
            content=(f"For experiment {marker}, the homelab uses OpenSearch for vector search."),
            subject="opensearch",
            importance=0.9,
        )
    )
    unrelated = await service.create_memory(
        EpisodicMemory(
            content=f"For experiment {marker}, made coffee this morning at 7am.",
            importance=0.1,
        )
    )
    assert fact.id and unrelated.id

    await asyncio.sleep(1)

    resp = await service.recall(
        RecallQuery(
            query=f"for experiment {marker}, which vector database am I using",
            limit=5,
        )
    )

    assert len(resp.memories) >= 1
    top = resp.memories[0]
    assert top.id == fact.id
    assert top.explanation.semantic > 0.0
    assert top.explanation.importance == pytest.approx(0.9, abs=1e-3)
    assert top.score > 0.0


async def test_recall_relationship_boost(service: MemoryService) -> None:
    marker = f"rlx-{ulid.new()!s}"
    fact = await service.create_memory(
        SemanticMemory(
            content=f"Jetson Orin Nano ({marker}) hosts the agent process.",
            subject="jetson",
            importance=0.7,
        )
    )
    event = await service.create_memory(
        EpisodicMemory(
            content=f"Deployment on Jetson ({marker}) failed due to memory pressure.",
            importance=0.6,
        )
    )
    assert fact.id and event.id

    await service.create_relationship(
        Relationship(
            from_id=event.id,
            to_id=fact.id,
            from_type=MemoryType.EPISODIC,
            to_type=MemoryType.SEMANTIC,
            relationship_type=RelationshipType.RELATED_TO,
            confidence=0.9,
        )
    )
    await asyncio.sleep(1)

    resp = await service.recall(RecallQuery(query=f"Jetson deployment issues {marker}", limit=10))

    by_id = {r.id: r for r in resp.memories}
    assert fact.id in by_id
    assert event.id in by_id
    # both endpoints should show a relationship boost
    assert by_id[fact.id].explanation.relationship > 0.0
    assert by_id[event.id].explanation.relationship > 0.0
