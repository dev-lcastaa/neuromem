"""Facade combining storage + embeddings + retrieval orchestration.

Route handlers depend only on `MemoryService`. Store and embedder are injected so
tests can swap in fakes.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any

from memory.config import Settings
from memory.domain.enums import MemoryType
from memory.domain.models import MemoryMetadata, Relationship
from memory.logging import get_logger
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallQuery,
    RecallResponse,
    RecallResult,
)
from memory.retrieval.ranking import (
    activation_score,
    normalise_by_max,
    recency_score,
    weighted_sum,
)
from memory.storage.interface import Memory, MemoryStore
from models.embeddings import EmbeddingProvider

_logger = get_logger("memory.service")


def _default_weights(s: Settings) -> RankWeights:
    return RankWeights(
        semantic=s.rank_weight_semantic,
        keyword=s.rank_weight_keyword,
        importance=s.rank_weight_importance,
        recency=s.rank_weight_recency,
        activation=s.rank_weight_activation,
        relationship=s.rank_weight_relationship,
    )


class DecayReport(dict[str, Any]):
    """Diagnostic dict for `/memory/decay`. Read-only; no writes happen."""


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    async def create_memory(self, memory: Memory) -> Memory:
        if memory.content_vector is None:
            vectors = await self._embedder.embed([memory.content])
            new_metadata = memory.metadata.model_copy(
                update={
                    "embedding_model": self._embedder.model,
                    "embedding_dim": self._embedder.dim,
                }
            )
            memory = memory.model_copy(
                update={"content_vector": vectors[0], "metadata": new_metadata}
            )
        return await self._store.create_memory(memory)

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._store.get_memory(memory_id)

    async def create_relationship(self, rel: Relationship) -> Relationship:
        return await self._store.create_relationship(rel)

    async def get_relationships(self, memory_id: str) -> list[Relationship]:
        return await self._store.get_relationships(memory_id)

    async def record_access(self, memory_ids: list[str], bump: float | None = None) -> int:
        effective_bump = self._settings.activation_reinforcement if bump is None else bump
        return await self._store.record_access_batch(memory_ids, effective_bump)

    async def decay_report(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        memories = await self._store.get_memories_by_ids(memory_ids)
        by_id = {m.id: m for m in memories if m.id is not None}
        reports: list[dict[str, Any]] = []
        for mid in memory_ids:
            memory = by_id.get(mid)
            if memory is None:
                reports.append({"memory_id": mid, "found": False})
                continue
            reference = memory.last_accessed_at or memory.created_at
            age_days = 0.0
            if reference is not None:
                age_days = max(0.0, (now - reference).total_seconds() / 86_400.0)
            decay_factor = math.exp(-self._settings.activation_decay_lambda * age_days)
            reports.append(
                {
                    "memory_id": mid,
                    "found": True,
                    "activation": memory.activation,
                    "access_count": memory.access_count,
                    "last_accessed_at": (
                        memory.last_accessed_at.isoformat() if memory.last_accessed_at else None
                    ),
                    "age_days": round(age_days, 3),
                    "decay_factor": round(decay_factor, 4),
                    "effective_activation": round(memory.activation * decay_factor, 4),
                }
            )
        return reports

    async def recall(self, query: RecallQuery) -> RecallResponse:
        weights = query.weights or _default_weights(self._settings)

        vec = (await self._embedder.embed([query.query]))[0]

        sem_hits = await self._store.search_semantic(
            vec, memory_types=query.memory_types, size=query.candidate_pool
        )
        kw_hits = await self._store.search_keyword(
            query.query,
            memory_types=query.memory_types,
            size=query.candidate_pool,
        )

        sem_by_id = {h.id: h.score for h in sem_hits}
        kw_raw = [h.score for h in kw_hits]
        kw_normed = normalise_by_max(kw_raw)
        kw_by_id = {h.id: n for h, n in zip(kw_hits, kw_normed, strict=True)}

        candidate_ids = list(set(sem_by_id) | set(kw_by_id))
        if not candidate_ids:
            return RecallResponse(query=query.query, weights=weights, memories=[])

        memories = await self._store.get_memories_by_ids(
            candidate_ids, include_vector=query.include_content_vector
        )
        by_id: dict[str, Memory] = {}
        for m in memories:
            if m.id is not None:
                by_id[m.id] = m

        rel_score_by_id: dict[str, float] = dict.fromkeys(by_id, 0.0)
        if query.include_relationships and by_id:
            rels = await self._store.get_relationships_for_ids(list(by_id))
            for rel in rels:
                if rel.from_id in by_id and rel.to_id in by_id:
                    boost = rel.confidence * rel.weight * 0.5
                    rel_score_by_id[rel.from_id] = min(1.0, rel_score_by_id[rel.from_id] + boost)
                    rel_score_by_id[rel.to_id] = min(1.0, rel_score_by_id[rel.to_id] + boost)

        now = datetime.now(UTC)
        results = [
            self._score_memory(
                memory=m,
                weights=weights,
                semantic=sem_by_id.get(mid, 0.0),
                keyword=kw_by_id.get(mid, 0.0),
                relationship=rel_score_by_id[mid],
                now=now,
            )
            for mid, m in by_id.items()
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[: query.limit]
        self._schedule_strengthening([r.id for r in top])
        return RecallResponse(query=query.query, weights=weights, memories=top)

    def _score_memory(
        self,
        *,
        memory: Memory,
        weights: RankWeights,
        semantic: float,
        keyword: float,
        relationship: float,
        now: datetime,
    ) -> RecallResult:
        recency = recency_score(memory.created_at, now, self._settings.recency_decay_lambda)
        activation = activation_score(
            activation=memory.activation,
            reference_at=memory.last_accessed_at or memory.created_at,
            now=now,
            decay_lambda=self._settings.activation_decay_lambda,
        )
        explanation = Explanation(
            semantic=semantic,
            keyword=keyword,
            importance=memory.importance,
            recency=recency,
            activation=activation,
            relationship=relationship,
        )
        score = weighted_sum(
            semantic=semantic,
            keyword=keyword,
            importance=memory.importance,
            recency=recency,
            activation=activation,
            relationship=relationship,
            w_semantic=weights.semantic,
            w_keyword=weights.keyword,
            w_importance=weights.importance,
            w_recency=weights.recency,
            w_activation=weights.activation,
            w_relationship=weights.relationship,
        )
        assert memory.id is not None
        return RecallResult(
            id=memory.id,
            memory_type=memory.memory_type,
            content=memory.content,
            score=round(score, 4),
            explanation=explanation,
            memory=memory,
        )

    def _schedule_strengthening(self, memory_ids: list[str]) -> None:
        if not memory_ids or not self._settings.strengthening_enabled:
            return
        if self._settings.activation_reinforcement <= 0:
            return

        async def _run() -> None:
            try:
                await self._store.record_access_batch(
                    memory_ids, self._settings.activation_reinforcement
                )
            except Exception as err:
                _logger.warning("strengthening.failed", error=str(err))

        task = asyncio.create_task(_run(), name="neuromem.strengthening")
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ------------------------------------------------------------------
    # Dashboard-facing read paths (Phase 7)
    # ------------------------------------------------------------------

    async def stats(self) -> dict[str, Any]:
        counts = await self._store.count_memories_per_type()
        rel_count = await self._store.count_relationships()
        averages = await self._store.aggregate_averages()
        return {
            "counts": counts,
            "totals": {
                "memories": sum(counts.values()),
                "relationships": rel_count,
            },
            "averages": {
                "confidence": round(averages.get("confidence", 0.0), 4),
                "activation": round(averages.get("activation", 0.0), 4),
                "importance": round(averages.get("importance", 0.0), 4),
            },
        }

    async def search_by_text(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        hits = await self._store.search_keyword(query, memory_types=memory_types, size=limit)
        if not hits:
            return []
        memories = await self._store.get_memories_by_ids([h.id for h in hits])
        by_id = {m.id: m for m in memories if m.id is not None}
        out: list[dict[str, Any]] = []
        for hit in hits:
            m = by_id.get(hit.id)
            if m is None:
                continue
            out.append(
                {
                    "id": hit.id,
                    "memory_type": m.memory_type.value,
                    "content": m.content,
                    "score": round(hit.score, 4),
                    "importance": m.importance,
                    "activation": m.activation,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
            )
        return out

    async def list_recent(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        memories = await self._store.list_recent(memory_types=memory_types, limit=limit)
        return [
            {
                "id": m.id,
                "memory_type": m.memory_type.value,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "importance": m.importance,
                "confidence": m.confidence,
                "activation": m.activation,
                "access_count": m.access_count,
            }
            for m in memories
            if m.id is not None
        ]

    async def graph_around(self, center_id: str, *, depth: int = 1) -> dict[str, Any] | None:
        depth = max(1, min(depth, 3))
        center = await self._store.get_memory(center_id)
        if center is None:
            return None
        assert center.id is not None

        nodes: dict[str, Memory] = {center.id: center}
        edges: list[dict[str, Any]] = []
        seen_edges: set[str] = set()
        frontier: list[str] = [center.id]

        for _ in range(depth):
            if not frontier:
                break
            rels = await self._store.get_relationships_for_ids(frontier)
            new_frontier: set[str] = set()
            for rel in rels:
                if rel.id is None or rel.id in seen_edges:
                    continue
                seen_edges.add(rel.id)
                edges.append(
                    {
                        "id": rel.id,
                        "from_id": rel.from_id,
                        "to_id": rel.to_id,
                        "relationship_type": rel.relationship_type.value,
                        "confidence": rel.confidence,
                    }
                )
                for endpoint in (rel.from_id, rel.to_id):
                    if endpoint not in nodes:
                        new_frontier.add(endpoint)
            if not new_frontier:
                break
            fetched = await self._store.get_memories_by_ids(list(new_frontier))
            for m in fetched:
                if m.id is not None:
                    nodes[m.id] = m
            frontier = list(new_frontier)

        return {
            "center_id": center_id,
            "nodes": [
                {
                    "id": mid,
                    "memory_type": m.memory_type.value,
                    "content": m.content,
                    "importance": m.importance,
                    "activation": m.activation,
                }
                for mid, m in nodes.items()
            ],
            "edges": edges,
        }


# Re-exports referenced by nothing else yet but kept for API stability with the plan.
__all__ = [
    "MemoryMetadata",
    "MemoryService",
    "MemoryType",
]
