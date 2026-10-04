"""OpenSearch persistence for asynchronous ingestion job state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError
from pydantic import BaseModel, ConfigDict

from memory.consolidation.models import ConsolidationJob, ConsolidationResult

INDEX_INGEST_JOBS = "neuromem-ingest-jobs"


class IngestionJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    attempts: int = 0
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result: ConsolidationResult | None = None


class OpenSearchIngestionJobs:
    def __init__(self, client: AsyncOpenSearch, mappings_dir: Path) -> None:
        self._client = client
        self._mapping_path = mappings_dir / "ingest-jobs.json"

    async def bootstrap(self) -> None:
        if await self._client.indices.exists(index=INDEX_INGEST_JOBS):
            return
        mapping = json.loads(self._mapping_path.read_text(encoding="utf-8"))
        await self._client.indices.create(index=INDEX_INGEST_JOBS, body=mapping)

    async def create(self, job: ConsolidationJob) -> IngestionJobStatus:
        now = datetime.now(UTC)
        status = IngestionJobStatus(
            job_id=job.job_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        await self._client.index(
            index=INDEX_INGEST_JOBS,
            id=job.job_id,
            body=status.model_dump(mode="json"),
            refresh="wait_for",
        )
        return status

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        attempts: int | None = None,
        error: str | None = None,
        result: ConsolidationResult | None = None,
    ) -> None:
        patch: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            "error": error[:2000] if error else None,
        }
        if attempts is not None:
            patch["attempts"] = attempts
        if result is not None:
            patch["result"] = result.model_dump(mode="json")
        await self._client.update(
            index=INDEX_INGEST_JOBS,
            id=job_id,
            body={"doc": patch},
            refresh="wait_for",
        )

    async def get(self, job_id: str) -> IngestionJobStatus | None:
        try:
            response = await self._client.get(index=INDEX_INGEST_JOBS, id=job_id)
        except NotFoundError:
            return None
        return IngestionJobStatus.model_validate(response["_source"])
