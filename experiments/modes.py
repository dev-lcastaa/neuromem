"""Two retrieval modes that share the same LLM answer path.

Mode A (`rag`): naive semantic k-NN, no rank fusion, no relationships, no strengthening.
Mode B (`neuromem`): full `MemoryService.recall` hybrid pipeline.

Both return `list[RecallResult]` so the caller can score them identically.
"""

from __future__ import annotations

from typing import Protocol

from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallQuery,
    RecallResult,
)
from memory.service import MemoryService
from memory.storage.interface import MemoryStore
from models.embeddings import EmbeddingProvider


class RetrievalMode(Protocol):
    name: str

    async def retrieve(self, query: str, k: int) -> list[RecallResult]: ...


class NaiveRAGMode:
    name = "rag"

    def __init__(self, *, embedder: EmbeddingProvider, store: MemoryStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(self, query: str, k: int) -> list[RecallResult]:
        vec = (await self._embedder.embed([query]))[0]
        hits = await self._store.search_semantic(vec, size=k)
        if not hits:
            return []
        ids = [h.id for h in hits]
        memories = await self._store.get_memories_by_ids(ids)
        by_id = {m.id: m for m in memories if m.id is not None}
        out: list[RecallResult] = []
        for hit in hits:
            m = by_id.get(hit.id)
            if m is None or m.id is None:
                continue
            out.append(
                RecallResult(
                    id=m.id,
                    memory_type=m.memory_type,
                    content=m.content,
                    score=round(hit.score, 4),
                    explanation=Explanation(semantic=hit.score),
                    memory=m,
                )
            )
        return out


class NeuroMemMode:
    name = "neuromem"

    def __init__(self, *, service: MemoryService) -> None:
        self._service = service

    async def retrieve(self, query: str, k: int) -> list[RecallResult]:
        resp = await self._service.recall(RecallQuery(query=query, limit=k, weights=RankWeights()))
        return list(resp.memories)
