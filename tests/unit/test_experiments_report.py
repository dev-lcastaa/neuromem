from __future__ import annotations

from datetime import UTC, datetime

from experiments.models import QueryMetrics
from experiments.report import build_report


def _row(mode: str, precision: float, top1: bool = True) -> QueryMetrics:
    return QueryMetrics(
        scenario_id="s1",
        query_id="q1",
        mode=mode,
        precision_at_k=precision,
        recall_hit=1.0,
        top1_match=top1,
        answer_hit=True,
        answer_forbidden_present=False,
        irrelevant_retrievals=0,
        retrieval_latency_ms=42.0,
        answer_latency_ms=123.0,
        retrieved_memory_ids=["sem_a"],
        answer="Answer text.",
    )


def test_build_report_includes_all_sections() -> None:
    rows = [_row("rag", 0.5), _row("neuromem", 0.9)]
    started = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
    finished = datetime(2026, 8, 15, 12, 5, tzinfo=UTC)

    md = build_report(
        rows=rows,
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072,
        started_at=started,
        finished_at=finished,
        run_ids={"s1": "eval-abc"},
    )

    assert "# NeuroMem — evaluation report" in md
    assert "gpt-4o" in md
    assert "text-embedding-3-large" in md
    assert "3072d" in md
    assert "## Aggregate (per mode)" in md
    assert "## Per-query detail" in md
    assert "## Run IDs" in md
    assert "eval-abc" in md
    assert "`rag`" in md
    assert "`neuromem`" in md


def test_aggregate_row_orders_alphabetically() -> None:
    rows = [_row("neuromem", 0.8), _row("rag", 0.4)]
    md = build_report(
        rows=rows,
        llm_model="m",
        embedding_model="e",
        embedding_dim=1,
        started_at=datetime(2026, 8, 15, tzinfo=UTC),
        finished_at=datetime(2026, 8, 15, tzinfo=UTC),
        run_ids={},
    )
    # Aggregate rows appear in sorted mode order
    neuromem_pos = md.index("`neuromem`")
    rag_pos = md.index("`rag`")
    assert neuromem_pos < rag_pos
