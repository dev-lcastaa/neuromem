"""In-process asyncio queue + background worker for consolidation jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from memory.consolidation.models import ConsolidationJob, ConsolidationResult
from memory.logging import get_logger

_logger = get_logger("consolidation.worker")


class ConsolidationQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[ConsolidationJob] = asyncio.Queue()

    async def enqueue(self, job: ConsolidationJob) -> None:
        await self._queue.put(job)

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _get(self) -> ConsolidationJob:
        return await self._queue.get()

    def _task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


async def run_worker(
    queue: ConsolidationQueue,
    processor: Callable[[ConsolidationJob], Awaitable[ConsolidationResult]],
) -> None:
    while True:
        try:
            job = await queue._get()
        except asyncio.CancelledError:
            raise
        try:
            result = await processor(job)
            _logger.info(
                "consolidation.processed",
                conversation_id=result.conversation_id,
                source_memory_id=result.source_memory_id,
                outcomes=len(result.outcomes),
            )
        except asyncio.CancelledError:
            queue._task_done()
            raise
        except Exception as err:
            _logger.exception("consolidation.error", error=str(err))
        finally:
            queue._task_done()
