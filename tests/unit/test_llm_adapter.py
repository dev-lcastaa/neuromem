from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from models.llm import OpenAIProvider


def _mock_chat_response(
    content: str = "pong",
    model: str = "gpt-4o-mini",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _fake_client(create_mock: AsyncMock) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))


@pytest.mark.asyncio
async def test_complete_returns_content_model_and_usage() -> None:
    create = AsyncMock(return_value=_mock_chat_response())
    provider = OpenAIProvider(client=_fake_client(create), default_model="gpt-4o-mini")

    resp = await provider.complete([{"role": "user", "content": "ping"}])

    assert resp.content == "pong"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    assert resp.latency_ms >= 0
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert kwargs["temperature"] == 0.2
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_json_mode_sets_response_format() -> None:
    create = AsyncMock(return_value=_mock_chat_response(content='{"ok":true}'))
    provider = OpenAIProvider(client=_fake_client(create), default_model="gpt-4o-mini")

    await provider.complete([{"role": "user", "content": "hi"}], response_format="json_object")

    kwargs = create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_override_model_wins_over_default() -> None:
    create = AsyncMock(return_value=_mock_chat_response(model="gpt-4o"))
    provider = OpenAIProvider(client=_fake_client(create), default_model="gpt-4o-mini")

    await provider.complete([{"role": "user", "content": "hi"}], model="gpt-4o")

    assert create.await_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_missing_usage_returns_none() -> None:
    resp_no_usage = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        model="gpt-4o-mini",
        usage=None,
    )
    create = AsyncMock(return_value=resp_no_usage)
    provider = OpenAIProvider(client=_fake_client(create), default_model="gpt-4o-mini")

    resp = await provider.complete([{"role": "user", "content": "hi"}])

    assert resp.content == "ok"
    assert resp.usage is None
