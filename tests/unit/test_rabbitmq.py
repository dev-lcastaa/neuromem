from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from aio_pika import DeliveryMode

from memory.consolidation.models import ConsolidationJob
from messaging.rabbitmq import RabbitMQBroker, is_final_attempt


class FakeExchange:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, message, *, routing_key: str) -> None:  # type: ignore[no-untyped-def]
        self.published.append((message, routing_key))


class FakeChannel:
    def __init__(self, exchanges: dict[str, FakeExchange]) -> None:
        self.exchanges = exchanges

    async def get_exchange(self, name: str) -> FakeExchange:
        return self.exchanges[name]


@pytest.mark.asyncio
async def test_enqueue_publishes_persistent_job_with_idempotency_key() -> None:
    exchange = FakeExchange()
    broker = RabbitMQBroker(None, None, exchange)
    job = ConsolidationJob(user_message="Remember this")

    await broker.enqueue(job)

    message, routing_key = exchange.published[0]
    assert message.delivery_mode == DeliveryMode.PERSISTENT
    assert message.message_id == job.job_id
    assert routing_key == "consolidation"
    assert ConsolidationJob.model_validate_json(message.body).job_id == job.job_id


@pytest.mark.parametrize(
    ("attempt", "max_retries", "expected"),
    [(0, 5, False), (4, 5, False), (5, 5, True)],
)
def test_retry_cutoff(attempt: int, max_retries: int, expected: bool) -> None:
    assert is_final_attempt(attempt, max_retries) is expected


@pytest.mark.asyncio
async def test_failed_delivery_is_sent_to_delayed_retry_queue() -> None:
    retry_exchange = FakeExchange()
    channel = FakeChannel({"neuromem.retry": retry_exchange})
    broker = RabbitMQBroker(None, channel, FakeExchange())
    incoming = SimpleNamespace(
        headers={"x-attempts": 0},
        body=b'{"job_id":"job-1"}',
        content_type="application/json",
        message_id="job-1",
    )

    await broker._publish_failure(incoming, 0, RuntimeError("temporary"), False, 5000)

    retry, routing_key = retry_exchange.published[0]
    assert routing_key == "retry"
    assert retry.headers["x-attempts"] == 1
    assert retry.expiration == timedelta(milliseconds=5000)


@pytest.mark.asyncio
async def test_final_failure_is_sent_to_dead_letter_queue() -> None:
    dead_exchange = FakeExchange()
    channel = FakeChannel({"neuromem.dead": dead_exchange})
    broker = RabbitMQBroker(None, channel, FakeExchange())
    incoming = SimpleNamespace(
        headers={"x-attempts": 5},
        body=b'{"job_id":"job-1"}',
        content_type="application/json",
        message_id="job-1",
    )

    await broker._publish_failure(incoming, 5, RuntimeError("permanent"), True, 5000)

    dead, routing_key = dead_exchange.published[0]
    assert routing_key == "dead"
    assert dead.headers["x-attempts"] == 6
    assert dead.expiration is None
