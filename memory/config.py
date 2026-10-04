"""NeuroMem runtime settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenSearch ---
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_verify_certs: bool = False

    # --- Async ingestion ---
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "neuromem"
    rabbitmq_password: str = ""
    consolidation_max_retries: int = 5
    consolidation_retry_delay_ms: int = 5000

    @property
    def rabbitmq_url(self) -> str:
        user = quote(self.rabbitmq_user, safe="")
        password = quote(self.rabbitmq_password, safe="")
        return f"amqp://{user}:{password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model_extraction: str = "gpt-4o-mini"
    openai_model_reasoning: str = "gpt-4o"
    openai_model_embedding: str = "text-embedding-3-large"

    # --- Embeddings ---
    embedding_dim: int = 3072
    embedding_batch_max: int = 128
    embedding_timeout_s: float = 30.0

    # --- Health probes (gated to avoid paid API calls on every deploy) ---
    health_check_llm: bool = False
    health_check_embeddings: bool = False

    # --- Logging ---
    log_level: str = "INFO"

    # --- Agent identity (provenance only; no scoping in POC) ---
    agent_role_id: str = "default"

    # --- Storage bootstrap ---
    bootstrap_indexes: bool = True
    mappings_dir: str = "opensearch/indexes"

    # --- Ranking (§10 baseline; weights fully configurable) ---
    rank_weight_semantic: float = 0.40
    rank_weight_keyword: float = 0.15
    rank_weight_importance: float = 0.20
    rank_weight_recency: float = 0.10
    rank_weight_activation: float = 0.10
    rank_weight_relationship: float = 0.05

    # decay applied to `created_at` age in days: recency = exp(-lambda * age_days)
    recency_decay_lambda: float = 0.05

    # --- Cognitive lifecycle (§18/§19) ---
    # bump applied to `activation` on every successful retrieval (capped at 1.0)
    activation_reinforcement: float = 0.05
    # per-day decay applied at query time: effective_activation = activation * exp(-lambda * age_days)
    activation_decay_lambda: float = 0.05
    # if False, recall does not schedule strengthening writes; useful for tests + read-only replay
    strengthening_enabled: bool = True

    # --- Consolidation ---
    # semantic score above which a nearby memory is considered for conflict adjudication
    consolidation_similarity_threshold: float = 0.75
    # how many near-neighbours to consider per candidate
    consolidation_conflict_pool: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
