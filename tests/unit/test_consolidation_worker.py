from __future__ import annotations

import asyncio

import pytest

from apps.agent.models import MemoryCandidate
from memory.consolidation.models import ConsolidationJob, ConsolidationResult
from memory.consolidation.queue import ConsolidationQueue, run_worker


def _job(text: str = "hi") -> ConsolidationJob:
    return ConsolidationJob(
        user_message=text,
        candidates=[MemoryCandidate(classification="IGNORE", content=text, reason="test")],
    )


@pytest.mark.asyncio
async def test_enqueue_and_qsize() -> None:
    q = ConsolidationQueue()
    assert q.qsize() == 0
    await q.enqueue(_job("a"))
    await q.enqueue(_job("b"))
    assert q.qsize() == 2


@pytest.mark.asyncio
async def test_worker_processes_jobs_in_order() -> None:
    q = ConsolidationQueue()
    processed: list[str] = []

    async def processor(job: ConsolidationJob) -> ConsolidationResult:
        processed.append(job.user_message)
        return ConsolidationResult(conversation_id=job.conversation_id, source_memory_id="epi_x")

    task = asyncio.create_task(run_worker(q, processor))
    try:
        await q.enqueue(_job("first"))
        await q.enqueue(_job("second"))
        await q.join()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert processed == ["first", "second"]


@pytest.mark.asyncio
async def test_worker_survives_processor_exception() -> None:
    q = ConsolidationQueue()
    processed: list[str] = []
    called = 0

    async def processor(job: ConsolidationJob) -> ConsolidationResult:
        nonlocal called
        called += 1
        if called == 1:
            raise RuntimeError("boom")
        processed.append(job.user_message)
        return ConsolidationResult(conversation_id=job.conversation_id, source_memory_id="epi_x")

    task = asyncio.create_task(run_worker(q, processor))
    try:
        await q.enqueue(_job("dies"))
        await q.enqueue(_job("survives"))
        await q.join()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert processed == ["survives"]
