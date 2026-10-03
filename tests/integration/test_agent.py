"""Live agent turn against the real cluster + OpenAI. Phase 3 acceptance test."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from apps.agent.graph import build_agent_graph
from apps.agent.models import ChatMessage
from apps.agent.state import AgentState
from memory.config import get_settings
from memory.domain.models import SemanticMemory
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from models.embeddings import OpenAIEmbeddingProvider
from models.llm import OpenAIProvider

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def wired() -> AsyncIterator[tuple[MemoryService, object]]:
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
    graph = build_agent_graph(service=service, llm=llm, settings=s)
    yield service, graph
    await os_client.close()
    await ai_client.close()


async def test_agent_answers_from_prior_memory(
    wired: tuple[MemoryService, object],
) -> None:
    service, graph = wired

    # Seed a durable fact only the memory can supply
    await service.create_memory(
        SemanticMemory(
            content="Luis' preferred neuromem tag color is chartreuse.",
            subject="preferences.tag_color",
            importance=0.95,
        )
    )
    await asyncio.sleep(1)

    initial: AgentState = {
        "messages": [
            ChatMessage(
                role="user",
                content="What color did I say I prefer for neuromem tags?",
            )
        ]
    }
    final = await graph.ainvoke(initial)  # type: ignore[attr-defined]

    reply = str(final.get("final_response", ""))
    assert "chartreuse" in reply.lower()
    recalled = final.get("recalled_memories", [])
    assert len(recalled) >= 1
    assert any("chartreuse" in r.content.lower() for r in recalled)


async def test_agent_extracts_candidate_from_declarative_turn(
    wired: tuple[MemoryService, object],
) -> None:
    _service, graph = wired

    initial: AgentState = {
        "messages": [
            ChatMessage(
                role="user",
                content=("Remember that I prefer black coffee. Do not forget this."),
            )
        ]
    }
    final = await graph.ainvoke(initial)  # type: ignore[attr-defined]

    candidates = final.get("candidate_memories", [])
    assert candidates, "expected at least one memory candidate"
    assert any("coffee" in c.content.lower() and c.classification != "IGNORE" for c in candidates)
