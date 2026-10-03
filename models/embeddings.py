"""Embedding adapter interface + OpenAI implementation + a null provider for tests."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from openai import APIError, AsyncOpenAI, RateLimitError


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbeddingProvider:
    """Deterministic zero vectors; safe for unit tests and offline bootstraps."""

    model = "null"

    def __init__(self, dim: int = 3072) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dim: int,
        batch_max: int = 128,
        max_retries: int = 3,
        initial_backoff_s: float = 1.0,
    ) -> None:
        self._client = client
        self.model = model
        self.dim = dim
        self._batch_max = batch_max
        self._max_retries = max_retries
        self._initial_backoff_s = initial_backoff_s
        self._dim_asserted = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self._batch_max):
            batch = texts[start : start + self._batch_max]
            results.extend(await self._embed_batch(batch))
        return results

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        backoff = self._initial_backoff_s
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.embeddings.create(model=self.model, input=batch)
                vectors = [item.embedding for item in resp.data]
                if not self._dim_asserted and vectors:
                    actual = len(vectors[0])
                    if actual != self.dim:
                        raise ValueError(
                            f"Embedding dim mismatch: expected {self.dim}, "
                            f"got {actual} from model={self.model}"
                        )
                    self._dim_asserted = True
                return vectors
            except (RateLimitError, APIError) as err:
                last_err = err
                if attempt == self._max_retries - 1:
                    break
                await asyncio.sleep(backoff)
                backoff *= 2
        assert last_err is not None
        raise last_err
