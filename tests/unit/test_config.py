from __future__ import annotations

import pytest

from memory.config import Settings, get_settings


def test_defaults_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in [
        "OPENSEARCH_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL_EMBEDDING",
        "EMBEDDING_DIM",
        "HEALTH_CHECK_LLM",
        "AGENT_ROLE_ID",
        "RABBITMQ_HOST",
        "RABBITMQ_PASSWORD",
    ]:
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.opensearch_url == "http://localhost:9200"
    assert s.openai_model_extraction == "gpt-4o-mini"
    assert s.openai_model_reasoning == "gpt-4o"
    assert s.openai_model_embedding == "text-embedding-3-large"
    assert s.embedding_dim == 3072
    assert s.embedding_batch_max == 128
    assert s.health_check_llm is False
    assert s.health_check_embeddings is False
    assert s.agent_role_id == "default"
    assert s.rabbitmq_url == "amqp://neuromem:@localhost:5672/"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("HEALTH_CHECK_LLM", "true")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.openai_api_key == "sk-abc"
    assert s.embedding_dim == 1536
    assert s.health_check_llm is True


def test_rabbitmq_url_encodes_credentials() -> None:
    settings = Settings(_env_file=None, rabbitmq_password="p@ss/word")  # type: ignore[call-arg]
    assert settings.rabbitmq_url == "amqp://neuromem:p%40ss%2Fword@localhost:5672/"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
