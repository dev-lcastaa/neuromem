"""Pure scoring helpers: recency decay, activation decay, keyword normalisation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime


def recency_score(created_at: datetime | None, now: datetime, decay_lambda: float) -> float:
    if created_at is None:
        return 0.0
    age_days = max(0.0, (now - created_at).total_seconds() / 86_400.0)
    return math.exp(-decay_lambda * age_days)


def activation_score(
    *,
    activation: float,
    reference_at: datetime | None,
    now: datetime,
    decay_lambda: float,
) -> float:
    """`effective_activation = activation * exp(-lambda * age_days)`.

    `reference_at` is normally `last_accessed_at or created_at` — the moment the
    memory was last "primed". If both are None the raw activation passes through.
    """
    if reference_at is None:
        return activation
    age_days = max(0.0, (now - reference_at).total_seconds() / 86_400.0)
    return activation * math.exp(-decay_lambda * age_days)


def normalise_by_max(scores: Iterable[float]) -> list[float]:
    values = list(scores)
    if not values:
        return []
    peak = max(values)
    if peak <= 0.0:
        return [0.0 for _ in values]
    return [v / peak for v in values]


def weighted_sum(
    *,
    semantic: float,
    keyword: float,
    importance: float,
    recency: float,
    activation: float,
    relationship: float,
    w_semantic: float,
    w_keyword: float,
    w_importance: float,
    w_recency: float,
    w_activation: float,
    w_relationship: float,
) -> float:
    return (
        semantic * w_semantic
        + keyword * w_keyword
        + importance * w_importance
        + recency * w_recency
        + activation * w_activation
        + relationship * w_relationship
    )
