"""FastAPI application factory + lifespan wiring for the memory API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from apps.agent.graph import build_agent_graph
from apps.memory_api.routers import agent, health, memory
from ingestion.jobs import OpenSearchIngestionJobs
from memory.config import get_settings
from memory.consolidation.service import ConsolidationService
from memory.logging import configure_logging, get_logger
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from messaging.rabbitmq import RabbitMQBroker
from models.embeddings import OpenAIEmbeddingProvider
from models.llm import OpenAIProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("memory_api")

    opensearch_auth = None
    if settings.opensearch_user and settings.opensearch_password:
        opensearch_auth = (settings.opensearch_user, settings.opensearch_password)

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key or "sk-noop")
    opensearch_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=opensearch_auth,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=settings.opensearch_verify_certs,
    )
    memory_store = OpenSearchMemoryStore(
        client=opensearch_client,
        mappings_dir=Path(settings.mappings_dir),
    )
    ingestion_jobs = OpenSearchIngestionJobs(
        client=opensearch_client,
        mappings_dir=Path(settings.mappings_dir),
    )

    embeddings_provider = OpenAIEmbeddingProvider(
        client=openai_client,
        model=settings.openai_model_embedding,
        dim=settings.embedding_dim,
        batch_max=settings.embedding_batch_max,
    )

    app.state.settings = settings
    app.state.opensearch = opensearch_client
    app.state.openai = openai_client
    app.state.memory_store = memory_store
    app.state.ingestion_jobs = ingestion_jobs
    llm_provider = OpenAIProvider(
        client=openai_client,
        default_model=settings.openai_model_reasoning,
    )
    app.state.llm = llm_provider
    app.state.embeddings = embeddings_provider
    memory_service = MemoryService(
        store=memory_store, embedder=embeddings_provider, settings=settings
    )
    app.state.memory_service = memory_service
    app.state.agent_graph = build_agent_graph(
        service=memory_service, llm=llm_provider, settings=settings
    )

    consolidation_service = ConsolidationService(
        service=memory_service, llm=llm_provider, settings=settings
    )
    ingestion_broker = await RabbitMQBroker.connect(settings.rabbitmq_url)
    app.state.consolidation_service = consolidation_service
    app.state.ingestion_broker = ingestion_broker

    if settings.bootstrap_indexes:
        try:
            await memory_store.bootstrap()
            await ingestion_jobs.bootstrap()
            logger.info("memory_api.bootstrap.ok")
        except Exception as err:
            logger.warning("memory_api.bootstrap.failed", error=str(err))

    logger.info(
        "memory_api.startup",
        opensearch=settings.opensearch_url,
        embedding_model=settings.openai_model_embedding,
        embedding_dim=settings.embedding_dim,
    )
    try:
        yield
    finally:
        await ingestion_broker.close()
        await openai_client.close()
        await opensearch_client.close()
        logger.info("memory_api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="NeuroMem Memory API",
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness + upstream readiness"},
            {"name": "store", "description": "Create / update memories"},
            {"name": "recall", "description": "Query and retrieve memories"},
            {
                "name": "consolidate",
                "description": "Extract durable memories from episodic events",
            },
            {"name": "relationships", "description": "Memory graph edges"},
            {"name": "lifecycle", "description": "Access, decay, strengthening"},
            {"name": "conflicts", "description": "Conflict detection & resolution"},
            {"name": "agent", "description": "LangGraph agent turn"},
        ],
    )
    app.include_router(health.router)
    app.include_router(memory.router)
    app.include_router(agent.router)
    return app


app = create_app()
