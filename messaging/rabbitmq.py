"""Durable RabbitMQ transport for consolidation jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from memory.consolidation.models import ConsolidationJob
from memory.logging import get_logger

MAIN_EXCHANGE = "neuromem.ingest"
RETRY_EXCHANGE = "neuromem.retry"
DEAD_EXCHANGE = "neuromem.dead"
MAIN_QUEUE = "neuromem.consolidation"
RETRY_QUEUE = "neuromem.consolidation.retry"
DEAD_QUEUE = "neuromem.consolidation.dead"
MAIN_KEY = "consolidation"
RETRY_KEY = "retry"
DEAD_KEY = "dead"

FailureHandler = Callable[[str, int, str, bool], Awaitable[None]]
JobProcessor = Callable[[ConsolidationJob], Awaitable[None]]
_logger = get_logger("messaging.rabbitmq")


def is_final_attempt(attempt: int, max_retries: int) -> bool:
    return attempt >= max_retries


class RabbitMQBroker:
    def __init__(self, connection: Any, channel: Any, exchange: Any) -> None:
        self._connection = connection
        self._channel = channel
        self._exchange = exchange

    @classmethod
    async def connect(cls, url: str) -> RabbitMQBroker:
        connection = await aio_pika.connect_robust(url)
        channel = await connection.channel(publisher_confirms=True)
        await channel.set_qos(prefetch_count=1)
        exchange = await channel.declare_exchange(MAIN_EXCHANGE, ExchangeType.DIRECT, durable=True)
        retry_exchange = await channel.declare_exchange(
            RETRY_EXCHANGE, ExchangeType.DIRECT, durable=True
        )
        dead_exchange = await channel.declare_exchange(
            DEAD_EXCHANGE, ExchangeType.DIRECT, durable=True
        )
        queue = await channel.declare_queue(MAIN_QUEUE, durable=True)
        await queue.bind(exchange, routing_key=MAIN_KEY)
        retry_queue = await channel.declare_queue(
            RETRY_QUEUE,
            durable=True,
            arguments={
                "x-dead-letter-exchange": MAIN_EXCHANGE,
                "x-dead-letter-routing-key": MAIN_KEY,
            },
        )
        await retry_queue.bind(retry_exchange, routing_key=RETRY_KEY)
        dead_queue = await channel.declare_queue(DEAD_QUEUE, durable=True)
        await dead_queue.bind(dead_exchange, routing_key=DEAD_KEY)
        return cls(connection, channel, exchange)

    async def enqueue(self, job: ConsolidationJob) -> None:
        message = Message(
            body=job.model_dump_json().encode(),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=job.job_id,
            headers={"x-attempts": 0},
        )
        await self._exchange.publish(message, routing_key=MAIN_KEY)

    async def consume(
        self,
        *,
        processor: JobProcessor,
        on_failure: FailureHandler,
        max_retries: int,
        retry_delay_ms: int,
    ) -> None:
        queue = await self._channel.get_queue(MAIN_QUEUE)
        async with queue.iterator() as iterator:
            async for message in iterator:
                attempts = int((message.headers or {}).get("x-attempts", 0))
                job: ConsolidationJob | None = None
                try:
                    job = ConsolidationJob.model_validate_json(message.body)
                    await processor(job)
                except Exception as error:
                    dead = is_final_attempt(attempts, max_retries)
                    try:
                        await self._publish_failure(message, attempts, error, dead, retry_delay_ms)
                        if job is not None:
                            await on_failure(job.job_id, attempts + 1, str(error), dead)
                        _logger.warning(
                            "ingestion.dead_lettered" if dead else "ingestion.retry_scheduled",
                            job_id=job.job_id if job is not None else message.message_id,
                            attempt=attempts + 1,
                            error=str(error),
                        )
                    except Exception:
                        await message.nack(requeue=True)
                        continue
                await message.ack()

    async def _publish_failure(
        self,
        message: Any,
        attempts: int,
        error: Exception,
        dead: bool,
        retry_delay_ms: int,
    ) -> None:
        headers = {
            **(message.headers or {}),
            "x-attempts": attempts + 1,
            "x-last-error": str(error),
        }
        retry = Message(
            body=message.body,
            content_type=message.content_type or "application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message.message_id,
            headers=headers,
            expiration=None if dead else timedelta(milliseconds=retry_delay_ms),
        )
        exchange_name = DEAD_EXCHANGE if dead else RETRY_EXCHANGE
        routing_key = DEAD_KEY if dead else RETRY_KEY
        exchange = await self._channel.get_exchange(exchange_name)
        await exchange.publish(retry, routing_key=routing_key)

    async def close(self) -> None:
        await self._connection.close()
