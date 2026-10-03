"""Markdown report generator for evaluation runs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean

from experiments.models import QueryMetrics


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _aggregate(rows: list[QueryMetrics]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[QueryMetrics]] = defaultdict(list)
    for r in rows:
        buckets[r.mode].append(r)
    out: dict[str, dict[str, float]] = {}
    for mode, xs in buckets.items():
        n = len(xs)
        out[mode] = {
            "queries": float(n),
            "avg_precision_at_k": mean(x.precision_at_k for x in xs) if n else 0.0,
            "avg_recall_hit": mean(x.recall_hit for x in xs) if n else 0.0,
            "top1_accuracy": mean(1.0 if x.top1_match else 0.0 for x in xs) if n else 0.0,
            "answer_accuracy": mean(1.0 if x.answer_hit else 0.0 for x in xs) if n else 0.0,
            "answer_forbidden_rate": (
                mean(1.0 if x.answer_forbidden_present else 0.0 for x in xs) if n else 0.0
            ),
            "avg_irrelevant": mean(x.irrelevant_retrievals for x in xs) if n else 0.0,
            "avg_retrieval_ms": mean(x.retrieval_latency_ms for x in xs) if n else 0.0,
            "avg_answer_ms": mean(x.answer_latency_ms for x in xs) if n else 0.0,
        }
    return out


def build_report(
    *,
    rows: list[QueryMetrics],
    llm_model: str,
    embedding_model: str,
    embedding_dim: int,
    started_at: datetime,
    finished_at: datetime,
    run_ids: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# NeuroMem — evaluation report")
    lines.append("")
    lines.append(f"- Started (UTC): `{started_at.isoformat()}`")
    lines.append(f"- Finished (UTC): `{finished_at.isoformat()}`")
    lines.append(f"- Duration: `{(finished_at - started_at).total_seconds():.1f}s`")
    lines.append(f"- LLM model: `{llm_model}`")
    lines.append(f"- Embedding model: `{embedding_model}` ({embedding_dim}d)")
    lines.append(f"- Scenarios run: `{len(run_ids)}`")
    lines.append("")

    lines.append("## Aggregate (per mode)")
    lines.append("")
    lines.append(
        "| Mode | Queries | P@K | Recall | Top-1 | Answer ✓ | Forbidden | Irrelevant | Retr. ms | Answ. ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    agg = _aggregate(rows)
    for mode in sorted(agg):
        a = agg[mode]
        lines.append(
            f"| `{mode}` | {int(a['queries'])} | {_fmt(a['avg_precision_at_k'])} "
            f"| {_fmt(a['avg_recall_hit'])} | {_fmt(a['top1_accuracy'])} "
            f"| {_fmt(a['answer_accuracy'])} | {_fmt(a['answer_forbidden_rate'])} "
            f"| {_fmt(a['avg_irrelevant'], 2)} "
            f"| {a['avg_retrieval_ms']:.1f} | {a['avg_answer_ms']:.1f} |"
        )
    lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append(
        "| Scenario | Query | Mode | P@K | Recall | Top-1 | Answer ✓ | Forbidden | Irrelevant | Retr. ms | Answ. ms |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r.scenario_id}` | `{r.query_id}` | `{r.mode}` "
            f"| {_fmt(r.precision_at_k)} | {_fmt(r.recall_hit)} "
            f"| {'✓' if r.top1_match else '✗'} | {'✓' if r.answer_hit else '✗'} "
            f"| {'⚠︎' if r.answer_forbidden_present else '·'} "
            f"| {r.irrelevant_retrievals} "
            f"| {r.retrieval_latency_ms:.0f} | {r.answer_latency_ms:.0f} |"
        )
    lines.append("")

    lines.append("## Run IDs")
    lines.append("")
    lines.append(
        "Seeded memory content is tagged with the run ID below. Kept in-place (no deletes) for later inspection."
    )
    lines.append("")
    lines.append("| Scenario | Run ID |")
    lines.append("|---|---|")
    for sid, rid in run_ids.items():
        lines.append(f"| `{sid}` | `{rid}` |")
    lines.append("")

    lines.append("## Sample answers")
    lines.append("")
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        key = (r.scenario_id, r.query_id, r.mode)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"### `{r.scenario_id}` / `{r.query_id}` / `{r.mode}`")
        lines.append("")
        lines.append("```")
        lines.append(r.answer.strip() or "(empty)")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
