from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from memory.retrieval.ranking import (
    activation_score,
    normalise_by_max,
    recency_score,
    weighted_sum,
)


def test_recency_none_created_at_returns_zero() -> None:
    assert recency_score(None, datetime.now(UTC), 0.05) == 0.0


def test_recency_zero_age_returns_one() -> None:
    now = datetime.now(UTC)
    assert recency_score(now, now, 0.05) == 1.0


def test_recency_decays_with_age() -> None:
    now = datetime.now(UTC)
    lam = 0.05
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)
    assert recency_score(seven_days_ago, now, lam) == math.exp(-lam * 7)
    assert recency_score(thirty_days_ago, now, lam) < recency_score(seven_days_ago, now, lam)


def test_recency_future_dates_clamped_to_zero_age() -> None:
    now = datetime.now(UTC)
    future = now + timedelta(days=5)
    assert recency_score(future, now, 0.05) == 1.0


def test_activation_passes_through_when_no_reference() -> None:
    assert (
        activation_score(
            activation=0.7, reference_at=None, now=datetime.now(UTC), decay_lambda=0.05
        )
        == 0.7
    )


def test_activation_decays_with_age() -> None:
    now = datetime.now(UTC)
    lam = 0.05
    ten_days_ago = now - timedelta(days=10)
    got = activation_score(activation=0.8, reference_at=ten_days_ago, now=now, decay_lambda=lam)
    assert math.isclose(got, 0.8 * math.exp(-lam * 10))


def test_activation_recent_reference_barely_decays() -> None:
    now = datetime.now(UTC)
    got = activation_score(activation=0.9, reference_at=now, now=now, decay_lambda=0.05)
    assert got == 0.9


def test_normalise_by_max_scales_to_one() -> None:
    assert normalise_by_max([1.0, 2.0, 5.0]) == [0.2, 0.4, 1.0]
    assert normalise_by_max([]) == []
    assert normalise_by_max([0.0, 0.0]) == [0.0, 0.0]


def test_weighted_sum_matches_manual_calc() -> None:
    total = weighted_sum(
        semantic=1.0,
        keyword=0.5,
        importance=0.8,
        recency=1.0,
        activation=0.2,
        relationship=0.0,
        w_semantic=0.4,
        w_keyword=0.15,
        w_importance=0.2,
        w_recency=0.1,
        w_activation=0.1,
        w_relationship=0.05,
    )
    expected = 0.4 + 0.075 + 0.16 + 0.1 + 0.02 + 0.0
    assert math.isclose(total, expected)
