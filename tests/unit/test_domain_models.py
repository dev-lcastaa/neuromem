from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import (
    EpisodicMemory,
    MemoryAdapter,
    ProceduralMemory,
    ProceduralStep,
    Relationship,
    SemanticMemory,
)


def test_episodic_defaults() -> None:
    m = EpisodicMemory(content="a happened")
    assert m.memory_type is MemoryType.EPISODIC
    assert m.content == "a happened"
    assert m.importance == 0.5
    assert m.confidence == 0.5
    assert m.activation == 0.5
    assert m.access_count == 0
    assert m.source_memory_ids == []
    assert m.related_memory_ids == []
    assert m.metadata.entities == []


def test_semantic_defaults() -> None:
    m = SemanticMemory(content="the sky is blue")
    assert m.memory_type is MemoryType.SEMANTIC
    assert m.subject is None


def test_procedural_with_steps() -> None:
    m = ProceduralMemory(
        content="deploy",
        name="deploy_service",
        steps=[
            ProceduralStep(order=1, action="build image"),
            ProceduralStep(order=2, action="push image"),
        ],
    )
    assert m.memory_type is MemoryType.PROCEDURAL
    assert len(m.steps) == 2
    assert m.steps[0].order == 1


def test_score_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        EpisodicMemory(content="x", importance=1.5)
    with pytest.raises(ValidationError):
        EpisodicMemory(content="x", confidence=-0.1)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        EpisodicMemory(content="x", not_a_field="oops")  # type: ignore[call-arg]


def test_discriminated_union_dispatches_by_memory_type() -> None:
    m = MemoryAdapter.validate_python({"memory_type": "semantic", "content": "x", "subject": "s"})
    assert isinstance(m, SemanticMemory)
    assert m.subject == "s"

    m2 = MemoryAdapter.validate_python(
        {
            "memory_type": "procedural",
            "content": "y",
            "steps": [{"order": 0, "action": "step-0"}],
        }
    )
    assert isinstance(m2, ProceduralMemory)


def test_relationship_defaults() -> None:
    r = Relationship(
        from_id="epi_1",
        to_id="sem_2",
        from_type=MemoryType.EPISODIC,
        to_type=MemoryType.SEMANTIC,
        relationship_type=RelationshipType.DERIVED_FROM,
    )
    assert r.confidence == 1.0
    assert r.weight == 1.0
    assert r.created_by == "system"
