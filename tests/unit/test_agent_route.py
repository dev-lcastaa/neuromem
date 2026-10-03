from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from apps.agent.models import MemoryCandidate
from apps.memory_api.main import create_app
from apps.memory_api.routers.agent import get_agent_graph, get_consolidation_queue


@pytest.fixture
def agent_graph() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def consolidation_queue() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(agent_graph: AsyncMock, consolidation_queue: AsyncMock) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_agent_graph] = lambda: agent_graph
    app.dependency_overrides[get_consolidation_queue] = lambda: consolidation_queue
    return TestClient(app)


def test_chat_returns_expected_shape(
    client: TestClient,
    agent_graph: AsyncMock,
    consolidation_queue: AsyncMock,
) -> None:
    agent_graph.ainvoke.return_value = {
        "final_response": "hello back",
        "current_goal": "greet",
        "recalled_memories": [],
        "candidate_memories": [],
    }

    r = client.post("/agent/chat", json={"message": "hi"})

    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "hello back"
    assert body["goal"] == "greet"
    assert body["recalled_memories"] == []
    assert body["candidate_memories"] == []
    assert body["consolidation_queued"] is False
    agent_graph.ainvoke.assert_awaited_once()
    consolidation_queue.enqueue.assert_not_awaited()


def test_chat_enqueues_when_candidates_present(
    client: TestClient,
    agent_graph: AsyncMock,
    consolidation_queue: AsyncMock,
) -> None:
    agent_graph.ainvoke.return_value = {
        "final_response": "noted",
        "current_goal": "record preference",
        "recalled_memories": [],
        "candidate_memories": [
            MemoryCandidate(
                classification="SEMANTIC",
                content="user prefers X",
                reason="explicit",
            )
        ],
    }

    r = client.post(
        "/agent/chat",
        json={"message": "please remember I prefer X", "conversation_id": "conv-1"},
    )

    assert r.status_code == 200
    assert r.json()["consolidation_queued"] is True
    consolidation_queue.enqueue.assert_awaited_once()
    job = consolidation_queue.enqueue.await_args.args[0]
    assert job.user_message == "please remember I prefer X"
    assert job.assistant_reply == "noted"
    assert job.conversation_id == "conv-1"
    assert len(job.candidates) == 1


def test_chat_forwards_history(client: TestClient, agent_graph: AsyncMock) -> None:
    agent_graph.ainvoke.return_value = {"final_response": "ok"}

    client.post(
        "/agent/chat",
        json={
            "message": "second turn",
            "history": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ],
        },
    )

    state = agent_graph.ainvoke.await_args.args[0]
    msgs = state["messages"]
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert msgs[0].content == "first"
    assert msgs[-1].content == "second turn"


def test_chat_rejects_empty_message(client: TestClient) -> None:
    r = client.post("/agent/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_rejects_extra_fields(client: TestClient) -> None:
    r = client.post("/agent/chat", json={"message": "hi", "surprise": 1})
    assert r.status_code == 422
