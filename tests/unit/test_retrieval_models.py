from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory.domain.enums import MemoryType
from memory.retrieval.models import (
    Explanation,
    RankWeights,
    RecallQuery,
)


def test_rank_weights_defaults_match_plan() -> None:
    w = RankWeights()
    assert w.semantic == 0.40
    assert w.keyword == 0.15
    assert w.importance == 0.20
    assert w.recency == 0.10
    assert w.activation == 0.10
    assert w.relationship == 0.05


def test_rank_weights_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        RankWeights(mystery=1.0)  # type: ignore[call-arg]


def test_rank_weights_reject_negative() -> None:
    with pytest.raises(ValidationError):
        RankWeights(semantic=-0.1)


def test_recall_query_defaults() -> None:
    q = RecallQuery(query="hello")
    assert q.query == "hello"
    assert q.memory_types is None
    assert q.limit == 10
    assert q.candidate_pool == 50
    assert q.include_relationships is True
    assert q.include_content_vector is False
    assert q.weights is None


def test_recall_query_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        RecallQuery(query="")


def test_recall_query_limits() -> None:
    with pytest.raises(ValidationError):
        RecallQuery(query="x", limit=0)
    with pytest.raises(ValidationError):
        RecallQuery(query="x", limit=101)


def test_recall_query_accepts_memory_type_filter() -> None:
    q = RecallQuery(query="x", memory_types=[MemoryType.SEMANTIC])
    assert q.memory_types == [MemoryType.SEMANTIC]


def test_explanation_defaults_zero() -> None:
    e = Explanation()
    assert e.semantic == 0.0
    assert e.relationship == 0.0
