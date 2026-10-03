"""Per-query metric computation. Substring-based; no LLM-as-judge in the POC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.models import QuerySpec
from memory.retrieval.models import RecallResult


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    if not needles:
        return False
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _contains_all(text: str, needles: Sequence[str]) -> bool:
    if not needles:
        return True
    low = text.lower()
    return all(n.lower() in low for n in needles)


def precision_at_k(memories: list[RecallResult], expected: Sequence[str]) -> float:
    if not memories:
        return 0.0
    if not expected:
        return 0.0
    hits = sum(1 for m in memories if _contains_any(m.content, expected))
    return hits / len(memories)


def recall_hit(memories: list[RecallResult], expected: Sequence[str]) -> float:
    if not expected or not memories:
        return 0.0
    return 1.0 if any(_contains_any(m.content, expected) for m in memories) else 0.0


def top1_match(memories: list[RecallResult], expected: Sequence[str]) -> bool:
    if not memories or not expected:
        return False
    return _contains_any(memories[0].content, expected)


def irrelevant_count(memories: list[RecallResult], expected: Sequence[str]) -> int:
    if not memories or not expected:
        return 0
    return sum(1 for m in memories if not _contains_any(m.content, expected))


def answer_hit(answer: str, expected: Sequence[str]) -> bool:
    return _contains_all(answer, expected)


def answer_forbidden_present(answer: str, forbidden: Sequence[str]) -> bool:
    return _contains_any(answer, forbidden)


def score_query(
    memories: list[RecallResult],
    answer: str,
    query: QuerySpec,
) -> dict[str, Any]:
    top_k = memories[: query.top_k]
    return {
        "precision_at_k": precision_at_k(top_k, query.expected_recall_contains_any),
        "recall_hit": recall_hit(top_k, query.expected_recall_contains_any),
        "top1_match": top1_match(top_k, query.expected_top_content_contains),
        "irrelevant_retrievals": irrelevant_count(top_k, query.expected_recall_contains_any),
        "answer_hit": answer_hit(answer, query.expected_answer_contains),
        "answer_forbidden_present": answer_forbidden_present(answer, query.expected_answer_missing),
    }
