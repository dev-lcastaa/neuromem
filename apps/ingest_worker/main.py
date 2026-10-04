"""RabbitMQ consumer that runs consolidation outside the API process."""

from __future__ import annotations

import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from ingestion.jobs import OpenSearchIngestionJobs
from memory.config import get_settings
from memory.consolidation.models import ConsolidationJob
from memory.consolidation.service import ConsolidationService
from memory.logging import configure_logging, get_logger
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from messaging.rabbitmq import RabbitMQBroker
from models.embeddings import OpenAIEmbeddingProvider
from models.llm import OpenAIProvider


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("ingest_worker")
    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password)

    search_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=settings.opensearch_verify_certs,
    )
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key or "sk-noop")
    store = OpenSearchMemoryStore(search_client, Path(settings.mappings_dir))
    jobs = OpenSearchIngestionJobs(search_client, Path(settings.mappings_dir))
    embedder = OpenAIEmbeddingProvider(
        client=openai_client,
        model=settings.openai_model_embedding,
        dim=settings.embedding_dim,
        batch_max=settings.embedding_batch_max,
    )
    llm = OpenAIProvider(client=openai_client, default_model=settings.openai_model_reasoning)
    service = MemoryService(store=store, embedder=embedder, settings=settings)
    consolidator = ConsolidationService(service=service, llm=llm, settings=settings)
    broker = await RabbitMQBroker.connect(settings.rabbitmq_url)

    async def process_job(job: ConsolidationJob) -> None:
        await jobs.update(job.job_id, status="processing")
        result = await consolidator.process(job)
        await jobs.update(job.job_id, status="succeeded", result=result)
        logger.info(
            "ingestion.succeeded",
            job_id=job.job_id,
            source_memory_id=result.source_memory_id,
            outcomes=len(result.outcomes),
        )

    async def record_failure(job_id: str, attempts: int, error: str, dead: bool) -> None:
        await jobs.update(
            job_id,
            status="failed" if dead else "retrying",
            attempts=attempts,
            error=error,
        )

    try:
        logger.info("ingestion.worker_started", opensearch=settings.opensearch_url)
        await broker.consume(
            processor=process_job,
            on_failure=record_failure,
            max_retries=settings.consolidation_max_retries,
            retry_delay_ms=settings.consolidation_retry_delay_ms,
        )
    finally:
        await broker.close()
        await openai_client.close()
        await search_client.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
