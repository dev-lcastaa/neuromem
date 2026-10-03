"""Live cognitive-lifecycle test: strengthening + decay reporting.

Phase 5 acceptance: repeatedly-useful memories become more retrievable.
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
from memory.domain.models import SemanticMemory
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


async def test_recall_strengthens_returned_memories(service: MemoryService) -> None:
    marker = f"lc-{ulid.new()!s}"
    memory = await service.create_memory(
        SemanticMemory(
            content=(f"The lifecycle experiment {marker} uses OpenSearch for vector storage."),
            subject="lifecycle",
            importance=0.6,
            activation=0.5,
        )
    )
    assert memory.id is not None
    await asyncio.sleep(1)

    # First recall bumps activation on the returned top result
    resp = await service.recall(
        RecallQuery(query=f"vector storage in experiment {marker}", limit=3)
    )
    assert any(r.id == memory.id for r in resp.memories)
    # Give the fire-and-forget strengthening task time to run + OpenSearch refresh
    await asyncio.sleep(2)

    before = await service.get_memory(memory.id)
    assert before is not None
    assert before.activation > 0.5, "activation should have been strengthened"
    assert before.access_count >= 1
    assert before.last_accessed_at is not None

    # Second explicit access adds another bump
    updated = await service.record_access([memory.id], bump=0.1)
    assert updated == 1
    await asyncio.sleep(2)

    after = await service.get_memory(memory.id)
    assert after is not None
    assert after.activation > before.activation
    assert after.access_count > before.access_count


async def test_decay_report_reflects_activation_and_age(
    service: MemoryService,
) -> None:
    marker = f"lc-decay-{ulid.new()!s}"
    memory = await service.create_memory(
        SemanticMemory(
            content=f"Decay report test {marker}",
            subject="decay",
            activation=0.8,
        )
    )
    assert memory.id is not None
    await asyncio.sleep(1)

    reports = await service.decay_report([memory.id, "sem_definitely_missing_xxx"])
    assert len(reports) == 2

    hit = next(r for r in reports if r["memory_id"] == memory.id)
    assert hit["found"] is True
    assert hit["activation"] == pytest.approx(0.8, abs=1e-3)
    assert hit["effective_activation"] <= hit["activation"]
    assert 0.0 <= hit["decay_factor"] <= 1.0

    missing = next(r for r in reports if r["memory_id"] == "sem_definitely_missing_xxx")
    assert missing["found"] is False
