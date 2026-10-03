"""LLM adapter interface + OpenAI implementation.

Providers are constructed with an already-built `AsyncOpenAI` client so tests can
inject a `respx`-mocked HTTP transport. No prompts live here; this is transport only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from openai import AsyncOpenAI

ResponseFormat = Literal["text", "json_object"]


@dataclass(slots=True, frozen=True)
class LLMResponse:
    content: str
    model: str
    latency_ms: int
    usage: dict[str, int] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: ResponseFormat | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


class OpenAIProvider:
    def __init__(self, client: AsyncOpenAI, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: ResponseFormat | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        resp = await self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        content = resp.choices[0].message.content or ""
        usage: dict[str, int] | None = None
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(
            content=content,
            model=resp.model,
            latency_ms=latency_ms,
            usage=usage,
        )
