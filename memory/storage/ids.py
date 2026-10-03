"""ID generation and index routing. ID prefix identifies the target OpenSearch index."""

from __future__ import annotations

import ulid

from memory.domain.enums import MemoryType

INDEX_EPISODIC = "neuromem-episodic"
INDEX_SEMANTIC = "neuromem-semantic"
INDEX_PROCEDURAL = "neuromem-procedural"
INDEX_RELATIONSHIPS = "neuromem-relationships"

_MEMORY_PREFIX = {
    MemoryType.EPISODIC: "epi",
    MemoryType.SEMANTIC: "sem",
    MemoryType.PROCEDURAL: "pro",
}

_PREFIX_INDEX = {
    "epi": INDEX_EPISODIC,
    "sem": INDEX_SEMANTIC,
    "pro": INDEX_PROCEDURAL,
    "rel": INDEX_RELATIONSHIPS,
}


def new_memory_id(memory_type: MemoryType) -> str:
    return f"{_MEMORY_PREFIX[memory_type]}_{ulid.new()!s}"


def new_relationship_id() -> str:
    return f"rel_{ulid.new()!s}"


def index_for_id(memory_or_rel_id: str) -> str:
    prefix, _, _ = memory_or_rel_id.partition("_")
    try:
        return _PREFIX_INDEX[prefix]
    except KeyError as err:
        raise ValueError(f"Unknown ID prefix: {memory_or_rel_id!r}") from err


def index_for_memory_type(memory_type: MemoryType) -> str:
    return _PREFIX_INDEX[_MEMORY_PREFIX[memory_type]]
