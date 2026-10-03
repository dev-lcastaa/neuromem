from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from apps.agent.graph import build_agent_graph
from apps.agent.models import ChatMessage, MemoryCandidate
from apps.agent.state import AgentState
from memory.config import Settings
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallResponse,
    RecallResult,
)
from models.llm import LLMResponse


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _llm_response(content: str, model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(content=content, model=model, latency_ms=1, usage=None)


def _recall_response(memory_content: str = "The system uses OpenSearch.") -> RecallResponse:
    mem = SemanticMemory(
        id="sem_a",
        content=memory_content,
        importance=0.9,
        created_at=datetime.now(UTC),
    )
    return RecallResponse(
        query="q",
        weights=RankWeights(),
        memories=[
            RecallResult(
                id="sem_a",
                memory_type=MemoryType.SEMANTIC,
                content=memory_content,
                score=0.9,
                explanation=Explanation(semantic=0.94, importance=0.9),
                memory=mem,
            )
        ],
    )


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_and_produces_response() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        _llm_response('{"query":"opensearch","goal":"identify vector db"}'),
        _llm_response("You are using OpenSearch.", model="gpt-4o"),
        _llm_response(
            '{"candidates":[{"classification":"SEMANTIC",'
            '"content":"user uses opensearch",'
            '"importance":0.8,"confidence":0.9,'
            '"entities":["opensearch"],"source":"conversation",'
            '"reason":"stable preference"}]}'
        ),
    ]
    service = AsyncMock()
    service.recall.return_value = _recall_response()

    graph = build_agent_graph(service=service, llm=llm, settings=_settings())

    initial: AgentState = {
        "messages": [ChatMessage(role="user", content="what vector db am I using?")]
    }
    final: AgentState = await graph.ainvoke(initial)

    assert final["final_response"] == "You are using OpenSearch."
    assert final["current_goal"] == "identify vector db"
    assert len(final["recalled_memories"]) == 1
    assert final["recalled_memories"][0].id == "sem_a"

    candidates = final["candidate_memories"]
    assert len(candidates) == 1
    assert isinstance(candidates[0], MemoryCandidate)
    assert candidates[0].classification == "SEMANTIC"
    assert candidates[0].importance == 0.8

    assert llm.complete.await_count == 3
    service.recall.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_survives_malformed_extraction_json() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        _llm_response('{"query":"x","goal":"y"}'),
        _llm_response("Sure.", model="gpt-4o"),
        _llm_response("not json at all"),
    ]
    service = AsyncMock()
    service.recall.return_value = _recall_response()

    graph = build_agent_graph(service=service, llm=llm, settings=_settings())
    final = await graph.ainvoke({"messages": [ChatMessage(role="user", content="hi")]})

    assert final["final_response"] == "Sure."
    assert final["candidate_memories"] == []


@pytest.mark.asyncio
async def test_graph_appends_assistant_reply_to_messages() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        _llm_response('{"query":"x","goal":"y"}'),
        _llm_response("Answer.", model="gpt-4o"),
        _llm_response('{"candidates":[]}'),
    ]
    service = AsyncMock()
    service.recall.return_value = _recall_response()

    graph = build_agent_graph(service=service, llm=llm, settings=_settings())
    final = await graph.ainvoke({"messages": [ChatMessage(role="user", content="q")]})

    msgs = final["messages"]
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[1].content == "Answer."


@pytest.mark.asyncio
async def test_graph_skips_recall_when_no_user_message() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        _llm_response('{"query":"","goal":""}'),
        _llm_response("noop", model="gpt-4o"),
        _llm_response('{"candidates":[]}'),
    ]
    service = AsyncMock()
    service.recall.return_value = _recall_response()

    graph = build_agent_graph(service=service, llm=llm, settings=_settings())
    final = await graph.ainvoke({"messages": []})

    assert final["recalled_memories"] == []
    service.recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_strips_code_fence_from_json() -> None:
    llm = AsyncMock()
    llm.complete.side_effect = [
        _llm_response('```json\n{"query":"x","goal":"y"}\n```'),
        _llm_response("Ok.", model="gpt-4o"),
        _llm_response('```\n{"candidates":[]}\n```'),
    ]
    service = AsyncMock()
    service.recall.return_value = _recall_response()

    graph = build_agent_graph(service=service, llm=llm, settings=_settings())
    final = await graph.ainvoke({"messages": [ChatMessage(role="user", content="q")]})

    assert final["current_goal"] == "y"
    assert final["candidate_memories"] == []
