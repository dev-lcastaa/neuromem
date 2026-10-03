from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIError

from models.embeddings import NullEmbeddingProvider, OpenAIEmbeddingProvider


def _mock_embedding_response(
    vectors: list[list[float]],
    model: str = "text-embedding-3-large",
) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=len(vectors), total_tokens=len(vectors)),
    )


def _fake_client(create_mock: AsyncMock) -> Any:
    return SimpleNamespace(embeddings=SimpleNamespace(create=create_mock))


@pytest.mark.asyncio
async def test_null_provider_returns_zero_vectors() -> None:
    p = NullEmbeddingProvider(dim=4)
    vs = await p.embed(["a", "b"])
    assert vs == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert p.dim == 4


@pytest.mark.asyncio
async def test_empty_input_returns_empty_list_without_api_call() -> None:
    create = AsyncMock()
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create), model="text-embedding-3-large", dim=3072
    )
    assert await p.embed([]) == []
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_dim_assertion_passes_on_correct_size() -> None:
    dim = 3072
    create = AsyncMock(return_value=_mock_embedding_response([[0.1] * dim]))
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create), model="text-embedding-3-large", dim=dim
    )
    vs = await p.embed(["hi"])
    assert len(vs) == 1
    assert len(vs[0]) == dim


@pytest.mark.asyncio
async def test_dim_mismatch_raises() -> None:
    create = AsyncMock(return_value=_mock_embedding_response([[0.1] * 10]))
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create), model="text-embedding-3-large", dim=3072
    )
    with pytest.raises(ValueError, match="dim mismatch"):
        await p.embed(["hi"])


@pytest.mark.asyncio
async def test_batches_large_input() -> None:
    dim = 8

    async def _create(**kwargs: Any) -> SimpleNamespace:
        n = len(kwargs["input"])
        return _mock_embedding_response([[0.1] * dim] * n)

    create = AsyncMock(side_effect=_create)
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create),
        model="text-embedding-3-large",
        dim=dim,
        batch_max=2,
    )

    vs = await p.embed(["a", "b", "c", "d", "e"])

    assert len(vs) == 5
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_retries_on_api_error_then_succeeds() -> None:
    dim = 4
    attempts = {"n": 0}

    async def _create(**kwargs: Any) -> SimpleNamespace:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise APIError(
                message="boom",
                request=MagicMock(),
                body=None,
            )
        return _mock_embedding_response([[0.2] * dim])

    create = AsyncMock(side_effect=_create)
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create),
        model="text-embedding-3-large",
        dim=dim,
        initial_backoff_s=0.0,
    )

    vs = await p.embed(["hi"])

    assert len(vs) == 1
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_retries() -> None:
    async def _create(**kwargs: Any) -> SimpleNamespace:
        raise APIError(message="boom", request=MagicMock(), body=None)

    create = AsyncMock(side_effect=_create)
    p = OpenAIEmbeddingProvider(
        client=_fake_client(create),
        model="text-embedding-3-large",
        dim=4,
        max_retries=2,
        initial_backoff_s=0.0,
    )

    with pytest.raises(APIError):
        await p.embed(["hi"])
    assert create.await_count == 2
