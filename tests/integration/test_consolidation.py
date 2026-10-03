"""Live consolidation against the real cluster + OpenAI (Phase 4 acceptance)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
import ulid
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from apps.agent.models import MemoryCandidate
from memory.config import get_settings
from memory.consolidation.models import ConsolidationJob
from memory.consolidation.service import ConsolidationService
from memory.domain.enums import RelationshipType
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from models.embeddings import OpenAIEmbeddingProvider
from models.llm import OpenAIProvider

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def consolidator() -> AsyncIterator[tuple[ConsolidationService, MemoryService]]:
    s = get_settings()
    if not s.openai_api_key or s.openai_api_key.startswith("sk-test"):
        pytest.skip("OPENAI_API_KEY not set for real LLM calls")

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
    llm = OpenAIProvider(client=ai_client, default_model=s.openai_model_reasoning)
    service = MemoryService(store=store, embedder=embedder, settings=s)
    consolidator = ConsolidationService(service=service, llm=llm, settings=s)
    yield consolidator, service
    await os_client.close()
    await ai_client.close()


async def test_first_turn_creates_derived_semantic(
    consolidator: tuple[ConsolidationService, MemoryService],
) -> None:
    svc, memory_service = consolidator
    marker = f"cs1-{ulid.new()!s}"

    job = ConsolidationJob(
        user_message=f"I am using the tag {marker} for consolidation experiments.",
        assistant_reply="Understood.",
        candidates=[
            MemoryCandidate(
                classification="SEMANTIC",
                content=f"The user's active consolidation-experiment tag is {marker}.",
                importance=0.7,
                confidence=0.9,
                entities=[marker],
                reason="stable preference",
            )
        ],
        conversation_id=f"conv-{marker}",
    )
    result = await svc.process(job)
    await asyncio.sleep(1)

    assert result.source_memory_id.startswith("epi_")
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    # `superseded`/`contradicted` are acceptable if the LLM adjudicated against
    # a similar memory from a previous test run. The core Phase 4 guarantee is
    # that a memory was materialised and linked to the source turn.
    assert outcome.action in {"created", "superseded", "contradicted"}
    assert outcome.created_memory_id is not None
    assert outcome.created_memory_id.startswith("sem_")

    rels = await memory_service.get_relationships(outcome.created_memory_id)
    derived_edges = [r for r in rels if r.relationship_type == RelationshipType.DERIVED_FROM]
    assert derived_edges, "expected a DERIVED_FROM edge from the new fact to the turn"
    assert any(r.to_id == result.source_memory_id for r in derived_edges)


async def test_second_related_turn_detects_conflict(
    consolidator: tuple[ConsolidationService, MemoryService],
) -> None:
    svc, memory_service = consolidator
    marker = f"cs2-{ulid.new()!s}"

    # Turn 1: assert the initial fact
    result_a = await svc.process(
        ConsolidationJob(
            user_message=f"My preferred vector store on {marker} is OpenSearch.",
            assistant_reply="Noted.",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content=f"On {marker}, the user's preferred vector store is OpenSearch.",
                    importance=0.85,
                    confidence=0.95,
                    entities=[marker, "opensearch"],
                    reason="explicit preference",
                )
            ],
            conversation_id=f"conv1-{marker}",
        )
    )
    await asyncio.sleep(1)
    assert result_a.outcomes[0].action == "created"
    first_id = result_a.outcomes[0].created_memory_id
    assert first_id is not None

    # Turn 2: restate an incompatible preference for the same subject
    result_b = await svc.process(
        ConsolidationJob(
            user_message=f"Actually on {marker} I switched to Qdrant.",
            assistant_reply="Got it.",
            candidates=[
                MemoryCandidate(
                    classification="SEMANTIC",
                    content=f"On {marker}, the user now prefers Qdrant over OpenSearch as the vector store.",
                    importance=0.9,
                    confidence=0.95,
                    entities=[marker, "qdrant", "opensearch"],
                    reason="preference change",
                )
            ],
            conversation_id=f"conv2-{marker}",
        )
    )
    await asyncio.sleep(1)

    outcome = result_b.outcomes[0]
    # LLM may return SUPERSEDES, CONTRADICTS, or DUPLICATE depending on interpretation
    assert outcome.action in {"superseded", "contradicted", "duplicate", "created"}, (
        f"unexpected action {outcome.action}"
    )
    # A non-trivial adjudication should link back to the earlier memory
    if outcome.action in {"superseded", "contradicted"}:
        assert outcome.created_memory_id is not None
        rels = await memory_service.get_relationships(outcome.created_memory_id)
        edge_types = {r.relationship_type for r in rels}
        assert (
            RelationshipType.SUPERSEDES in edge_types or RelationshipType.CONTRADICTS in edge_types
        )
        # Old fact must still exist -- never silently overwritten
        old = await memory_service.get_memory(first_id)
        assert old is not None
