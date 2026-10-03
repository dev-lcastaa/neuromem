from __future__ import annotations

from datetime import UTC, datetime

from experiments.metrics import (
    answer_forbidden_present,
    answer_hit,
    irrelevant_count,
    precision_at_k,
    recall_hit,
    score_query,
    top1_match,
)
from experiments.models import QuerySpec
from memory.domain.enums import MemoryType
from memory.domain.models import SemanticMemory
from memory.retrieval.models import Explanation, RecallResult


def _mem(mid: str, content: str) -> RecallResult:
    now = datetime.now(UTC)
    return RecallResult(
        id=mid,
        memory_type=MemoryType.SEMANTIC,
        content=content,
        score=0.5,
        explanation=Explanation(),
        memory=SemanticMemory(id=mid, content=content, created_at=now),
    )


def test_precision_at_k_counts_substring_hits() -> None:
    memories = [
        _mem("a", "uses OpenSearch"),
        _mem("b", "unrelated coffee note"),
        _mem("c", "OpenSearch cluster health"),
    ]
    assert precision_at_k(memories, ["opensearch"]) == 2 / 3
    assert precision_at_k(memories, ["nothing"]) == 0.0
    assert precision_at_k([], ["opensearch"]) == 0.0
    assert precision_at_k(memories, []) == 0.0


def test_recall_hit_is_zero_when_none_match() -> None:
    memories = [_mem("a", "coffee")]
    assert recall_hit(memories, ["opensearch"]) == 0.0
    assert recall_hit(memories, ["coffee"]) == 1.0


def test_top1_match() -> None:
    memories = [_mem("a", "OpenSearch fact"), _mem("b", "unrelated")]
    assert top1_match(memories, ["opensearch"]) is True
    assert top1_match(memories, ["missing"]) is False
    assert top1_match([], ["x"]) is False


def test_irrelevant_count() -> None:
    memories = [
        _mem("a", "OpenSearch cluster"),
        _mem("b", "coffee note"),
        _mem("c", "another unrelated"),
    ]
    assert irrelevant_count(memories, ["opensearch"]) == 2


def test_answer_hit_requires_all_substrings() -> None:
    assert answer_hit("The user uses OpenSearch for vectors", ["opensearch", "vectors"]) is True
    assert answer_hit("Something else", ["opensearch"]) is False
    assert answer_hit("nothing", []) is True  # no expected -> vacuously true


def test_answer_forbidden_flags_leak() -> None:
    assert answer_forbidden_present("Using Pinecone", ["pinecone", "weaviate"]) is True
    assert answer_forbidden_present("Using OpenSearch", ["pinecone"]) is False


def test_score_query_returns_dict_of_fields() -> None:
    memories = [_mem("a", "OpenSearch is the vector database")]
    q = QuerySpec(
        id="q1",
        question="which vector db?",
        top_k=5,
        expected_top_content_contains=["opensearch"],
        expected_recall_contains_any=["opensearch"],
        expected_answer_contains=["opensearch"],
        expected_answer_missing=["pinecone"],
    )
    scored = score_query(memories, "You are using OpenSearch.", q)
    assert scored["precision_at_k"] == 1.0
    assert scored["recall_hit"] == 1.0
    assert scored["top1_match"] is True
    assert scored["irrelevant_retrievals"] == 0
    assert scored["answer_hit"] is True
    assert scored["answer_forbidden_present"] is False
