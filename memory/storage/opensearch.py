"""OpenSearch-backed memory store: CRUD + relationships + idempotent index bootstrap."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError

from memory.domain.enums import MemoryType
from memory.domain.models import MemoryAdapter, Relationship
from memory.storage.ids import (
    INDEX_EPISODIC,
    INDEX_PROCEDURAL,
    INDEX_RELATIONSHIPS,
    INDEX_SEMANTIC,
    index_for_id,
    index_for_memory_type,
    new_memory_id,
    new_relationship_id,
)
from memory.storage.interface import Memory, SearchHit

_MAPPING_FILES = {
    INDEX_EPISODIC: "episodic.json",
    INDEX_SEMANTIC: "semantic.json",
    INDEX_PROCEDURAL: "procedural.json",
    INDEX_RELATIONSHIPS: "relationships.json",
}

_ALL_MEMORY_INDEXES = f"{INDEX_EPISODIC},{INDEX_SEMANTIC},{INDEX_PROCEDURAL}"

# Fields worth searching for text matches across memory types. multi_match with lenient=true
# ignores fields that don't exist in a given index's mapping.
_KEYWORD_FIELDS = [
    "content^2",
    "event",
    "context",
    "actions",
    "outcome",
    "subject",
    "object",
    "domain",
    "name",
    "goal",
    "preconditions",
    "expected_outcome",
]


def _now() -> datetime:
    return datetime.now(UTC)


def _memory_from_source(memory_id: str, source: dict[str, Any]) -> Memory:
    return MemoryAdapter.validate_python({"id": memory_id, **source})


def _relationship_from_source(rel_id: str, source: dict[str, Any]) -> Relationship:
    return Relationship.model_validate({"id": rel_id, **source})


class OpenSearchMemoryStore:
    def __init__(self, client: AsyncOpenSearch, mappings_dir: Path) -> None:
        self._client = client
        self._mappings_dir = mappings_dir

    async def bootstrap(self) -> None:
        for index_name, mapping_filename in _MAPPING_FILES.items():
            if await self._client.indices.exists(index=index_name):
                continue
            body = json.loads((self._mappings_dir / mapping_filename).read_text(encoding="utf-8"))
            await self._client.indices.create(index=index_name, body=body)

    async def create_memory(self, memory: Memory) -> Memory:
        updates: dict[str, Any] = {}
        if memory.id is None:
            updates["id"] = new_memory_id(memory.memory_type)
        now = _now()
        if memory.created_at is None:
            updates["created_at"] = now
        if memory.updated_at is None:
            updates["updated_at"] = now
        if updates:
            memory = memory.model_copy(update=updates)

        assert memory.id is not None
        index = index_for_memory_type(memory.memory_type)
        body = memory.model_dump(mode="json", exclude={"id"}, exclude_none=True)
        await self._client.index(index=index, id=memory.id, body=body, refresh="wait_for")
        return memory

    async def get_memory(self, memory_id: str, *, include_vector: bool = False) -> Memory | None:
        index = index_for_id(memory_id)
        kwargs: dict[str, Any] = {"index": index, "id": memory_id}
        if not include_vector:
            kwargs["_source_excludes"] = ["content_vector"]
        try:
            resp = await self._client.get(**kwargs)
        except NotFoundError:
            return None
        return _memory_from_source(memory_id, resp["_source"])

    async def update_memory(self, memory_id: str, patch: dict[str, Any]) -> Memory:
        index = index_for_id(memory_id)
        doc = {**patch, "updated_at": _now().isoformat()}
        await self._client.update(index=index, id=memory_id, body={"doc": doc}, refresh="wait_for")
        updated = await self.get_memory(memory_id)
        if updated is None:
            raise KeyError(f"Memory {memory_id} disappeared after update")
        return updated

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        updates: dict[str, Any] = {}
        if relationship.id is None:
            updates["id"] = new_relationship_id()
        if relationship.created_at is None:
            updates["created_at"] = _now()
        if updates:
            relationship = relationship.model_copy(update=updates)

        assert relationship.id is not None
        body = relationship.model_dump(mode="json", exclude={"id"}, exclude_none=True)
        await self._client.index(
            index=INDEX_RELATIONSHIPS,
            id=relationship.id,
            body=body,
            refresh="wait_for",
        )
        return relationship

    async def get_relationships(self, memory_id: str) -> list[Relationship]:
        query: dict[str, Any] = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"from_id": memory_id}},
                        {"term": {"to_id": memory_id}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": 1000,
        }
        resp = await self._client.search(index=INDEX_RELATIONSHIPS, body=query)
        hits = resp.get("hits", {}).get("hits", [])
        return [_relationship_from_source(h["_id"], h["_source"]) for h in hits]

    async def get_relationships_for_ids(self, ids: list[str]) -> list[Relationship]:
        if not ids:
            return []
        query: dict[str, Any] = {
            "query": {
                "bool": {
                    "should": [
                        {"terms": {"from_id": ids}},
                        {"terms": {"to_id": ids}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "size": 10_000,
        }
        resp = await self._client.search(index=INDEX_RELATIONSHIPS, body=query)
        hits = resp.get("hits", {}).get("hits", [])
        return [_relationship_from_source(h["_id"], h["_source"]) for h in hits]

    async def get_memories_by_ids(
        self, ids: list[str], *, include_vector: bool = False
    ) -> list[Memory]:
        if not ids:
            return []
        body = {"docs": [{"_index": index_for_id(i), "_id": i} for i in ids]}
        kwargs: dict[str, Any] = {"body": body}
        if not include_vector:
            kwargs["_source_excludes"] = ["content_vector"]
        resp = await self._client.mget(**kwargs)
        results: list[Memory] = []
        for doc in resp.get("docs", []):
            if not doc.get("found"):
                continue
            results.append(_memory_from_source(doc["_id"], doc["_source"]))
        return results

    async def search_semantic(
        self,
        query_vector: list[float],
        *,
        memory_types: list[MemoryType] | None = None,
        size: int = 50,
    ) -> list[SearchHit]:
        indexes = _resolve_indexes(memory_types)
        body: dict[str, Any] = {
            "size": size,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": query_vector,
                        "k": size,
                    }
                }
            },
            "_source": ["memory_type"],
        }
        resp = await self._client.search(index=indexes, body=body)
        return _hits_to_search_hits(resp.get("hits", {}).get("hits", []))

    async def search_keyword(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        size: int = 50,
    ) -> list[SearchHit]:
        indexes = _resolve_indexes(memory_types)
        body: dict[str, Any] = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": _KEYWORD_FIELDS,
                    "type": "best_fields",
                    "lenient": True,
                }
            },
            "_source": ["memory_type"],
        }
        resp = await self._client.search(index=indexes, body=body)
        return _hits_to_search_hits(resp.get("hits", {}).get("hits", []))

    async def record_access_batch(self, memory_ids: list[str], bump: float) -> int:
        if not memory_ids:
            return 0
        now_iso = _now().isoformat()
        body: list[dict[str, Any]] = []
        for mid in memory_ids:
            body.append({"update": {"_index": index_for_id(mid), "_id": mid}})
            body.append(
                {
                    "script": {
                        "source": (
                            "ctx._source.activation = "
                            "Math.min(1.0, ctx._source.activation + params.bump); "
                            "ctx._source.access_count = ctx._source.access_count + 1; "
                            "ctx._source.last_accessed_at = params.now;"
                        ),
                        "params": {"bump": bump, "now": now_iso},
                    }
                }
            )
        resp = await self._client.bulk(body=body, refresh="false")
        successes = 0
        for item in resp.get("items", []):
            status = item.get("update", {}).get("status")
            if status is not None and 200 <= int(status) < 300:
                successes += 1
        return successes

    async def count_memories_per_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for mtype, index in _INDEX_FOR_TYPE.items():
            resp = await self._client.count(index=index)
            counts[mtype.value] = int(resp.get("count", 0))
        return counts

    async def count_relationships(self) -> int:
        resp = await self._client.count(index=INDEX_RELATIONSHIPS)
        return int(resp.get("count", 0))

    async def aggregate_averages(self) -> dict[str, float]:
        body = {
            "size": 0,
            "aggs": {
                "avg_confidence": {"avg": {"field": "confidence"}},
                "avg_activation": {"avg": {"field": "activation"}},
                "avg_importance": {"avg": {"field": "importance"}},
            },
        }
        resp = await self._client.search(index=_ALL_MEMORY_INDEXES, body=body)
        aggs = resp.get("aggregations", {})

        def _val(name: str) -> float:
            raw = aggs.get(name, {}).get("value")
            return float(raw) if raw is not None else 0.0

        return {
            "confidence": _val("avg_confidence"),
            "activation": _val("avg_activation"),
            "importance": _val("avg_importance"),
        }

    async def list_recent(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        indexes = _resolve_indexes(memory_types)
        body: dict[str, Any] = {
            "size": limit,
            "sort": [{"created_at": {"order": "desc"}}],
            "_source": {"excludes": ["content_vector"]},
        }
        resp = await self._client.search(index=indexes, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [_memory_from_source(h["_id"], h["_source"]) for h in hits]


_INDEX_FOR_TYPE = {
    MemoryType.EPISODIC: INDEX_EPISODIC,
    MemoryType.SEMANTIC: INDEX_SEMANTIC,
    MemoryType.PROCEDURAL: INDEX_PROCEDURAL,
}


def _resolve_indexes(memory_types: list[MemoryType] | None) -> str:
    if not memory_types:
        return _ALL_MEMORY_INDEXES
    return ",".join(_INDEX_FOR_TYPE[t] for t in memory_types)


def _hits_to_search_hits(hits: list[dict[str, Any]]) -> list[SearchHit]:
    out: list[SearchHit] = []
    for h in hits:
        source = h.get("_source", {})
        raw_type = source.get("memory_type")
        if raw_type is None:
            continue
        out.append(
            SearchHit(
                id=h["_id"],
                memory_type=MemoryType(raw_type),
                score=float(h.get("_score", 0.0)),
            )
        )
    return out
