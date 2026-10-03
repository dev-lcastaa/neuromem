from __future__ import annotations

import pytest

from memory.domain.enums import MemoryType
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


def test_memory_id_prefix_matches_type() -> None:
    epi = new_memory_id(MemoryType.EPISODIC)
    sem = new_memory_id(MemoryType.SEMANTIC)
    pro = new_memory_id(MemoryType.PROCEDURAL)
    assert epi.startswith("epi_")
    assert sem.startswith("sem_")
    assert pro.startswith("pro_")
    assert len({epi, sem, pro}) == 3


def test_relationship_id_prefix() -> None:
    assert new_relationship_id().startswith("rel_")


def test_index_for_id_routes_correctly() -> None:
    assert index_for_id(new_memory_id(MemoryType.EPISODIC)) == INDEX_EPISODIC
    assert index_for_id(new_memory_id(MemoryType.SEMANTIC)) == INDEX_SEMANTIC
    assert index_for_id(new_memory_id(MemoryType.PROCEDURAL)) == INDEX_PROCEDURAL
    assert index_for_id(new_relationship_id()) == INDEX_RELATIONSHIPS


def test_index_for_memory_type() -> None:
    assert index_for_memory_type(MemoryType.EPISODIC) == INDEX_EPISODIC
    assert index_for_memory_type(MemoryType.SEMANTIC) == INDEX_SEMANTIC
    assert index_for_memory_type(MemoryType.PROCEDURAL) == INDEX_PROCEDURAL


def test_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ID prefix"):
        index_for_id("xxx_abc")
