"""Abstract memory store contract. OpenSearch is the only implementation for the POC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from memory.domain.enums import MemoryType
from memory.domain.models import (
    EpisodicMemory,
    ProceduralMemory,
    Relationship,
    SemanticMemory,
)

Memory = EpisodicMemory | SemanticMemory | ProceduralMemory


@dataclass(slots=True, frozen=True)
class SearchHit:
    id: str
    memory_type: MemoryType
    score: float


@runtime_checkable
class MemoryStore(Protocol):
    async def bootstrap(self) -> None: ...

    async def create_memory(self, memory: Memory) -> Memory: ...

    async def get_memory(
        self, memory_id: str, *, include_vector: bool = False
    ) -> Memory | None: ...

    async def get_memories_by_ids(
        self, ids: list[str], *, include_vector: bool = False
    ) -> list[Memory]: ...

    async def update_memory(self, memory_id: str, patch: dict[str, Any]) -> Memory: ...

    async def create_relationship(self, relationship: Relationship) -> Relationship: ...

    async def get_relationships(self, memory_id: str) -> list[Relationship]: ...

    async def get_relationships_for_ids(self, ids: list[str]) -> list[Relationship]: ...

    async def search_semantic(
        self,
        query_vector: list[float],
        *,
        memory_types: list[MemoryType] | None = None,
        size: int = 50,
    ) -> list[SearchHit]: ...

    async def search_keyword(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        size: int = 50,
    ) -> list[SearchHit]: ...

    async def record_access_batch(self, memory_ids: list[str], bump: float) -> int: ...

    async def count_memories_per_type(self) -> dict[str, int]: ...

    async def count_relationships(self) -> int: ...

    async def aggregate_averages(self) -> dict[str, float]: ...

    async def list_recent(
        self,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 50,
    ) -> list[Memory]: ...
