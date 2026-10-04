"""Test-wide fixtures: neutralize env, isolate settings cache, guard against live network."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from memory.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("integration"):
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
        return

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HEALTH_CHECK_LLM", "false")
    monkeypatch.setenv("HEALTH_CHECK_EMBEDDINGS", "false")
    monkeypatch.setenv("OPENSEARCH_URL", "http://opensearch.test:9200")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("BOOTSTRAP_INDEXES", "false")
    monkeypatch.setenv("STRENGTHENING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeRabbitMQ:
    async def close(self) -> None:
        return None

    async def enqueue(self, job: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _fake_rabbitmq_for_unit_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if request.node.get_closest_marker("integration"):
        return

    async def connect(_cls, _url: str) -> _FakeRabbitMQ:
        return _FakeRabbitMQ()

    monkeypatch.setattr("apps.memory_api.main.RabbitMQBroker.connect", classmethod(connect))
