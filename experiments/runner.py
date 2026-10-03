"""Scenario runner. Seeds memories, runs both retrieval modes, scores each query."""

from __future__ import annotations

import time
from collections.abc import Iterable

import ulid

from apps.agent.prompts import ANSWER_SYSTEM, ANSWER_USER_TEMPLATE
from experiments.metrics import score_query
from experiments.models import (
    QueryMetrics,
    QuerySpec,
    Scenario,
    SeededMemorySpec,
    memory_type_to_enum,
)
from experiments.modes import RetrievalMode
from memory.domain.enums import MemoryType
from memory.domain.models import (
    EpisodicMemory,
    MemoryMetadata,
    ProceduralMemory,
    Relationship,
    RelationshipMetadata,
    SemanticMemory,
)
from memory.retrieval.models import RecallResult
from memory.service import MemoryService
from memory.storage.interface import Memory
from models.llm import LLMProvider


def new_run_id() -> str:
    return f"eval-{str(ulid.new())[:12]}"


def _instantiate_memory(spec: SeededMemorySpec, run_id: str) -> Memory:
    content = spec.content.replace("{run_id}", run_id)
    metadata = MemoryMetadata(
        agent_id="eval",
        agent_role="eval",
        source=f"eval:{run_id}",
        entities=[run_id],
    )
    mtype = memory_type_to_enum(spec.memory_type)
    common = {
        "content": content,
        "importance": spec.importance,
        "confidence": spec.confidence,
        "activation": spec.activation,
        "metadata": metadata,
    }
    if mtype is MemoryType.EPISODIC:
        return EpisodicMemory(**common, event=spec.event)
    if mtype is MemoryType.SEMANTIC:
        return SemanticMemory(**common, subject=spec.subject)
    if mtype is MemoryType.PROCEDURAL:
        return ProceduralMemory(**common, name=spec.name)
    raise ValueError(f"Unknown memory type {spec.memory_type!r}")


def _format_memories(memories: Iterable[RecallResult]) -> str:
    lines: list[str] = []
    for i, m in enumerate(memories, start=1):
        lines.append(f"{i}. [{m.memory_type.value}] {m.content}")
    return "\n".join(lines) or "(none)"


async def _answer(llm: LLMProvider, question: str, memories: list[RecallResult], model: str) -> str:
    resp = await llm.complete(
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": ANSWER_USER_TEMPLATE.format(
                    memories_block=_format_memories(memories), message=question
                ),
            },
        ],
        model=model,
        temperature=0.0,
    )
    return resp.content.strip()


async def _run_query(
    scenario: Scenario,
    query: QuerySpec,
    modes: list[RetrievalMode],
    llm: LLMProvider,
    llm_model: str,
    run_id: str,
) -> list[QueryMetrics]:
    q_text = query.question.replace("{run_id}", run_id)
    rows: list[QueryMetrics] = []
    for mode in modes:
        t0 = time.perf_counter()
        memories = await mode.retrieve(q_text, query.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        answer = await _answer(llm, q_text, memories[: query.top_k], llm_model)
        answer_ms = (time.perf_counter() - t0) * 1000.0

        scored = score_query(memories, answer, query)
        rows.append(
            QueryMetrics(
                scenario_id=scenario.id,
                query_id=query.id,
                mode=mode.name,
                precision_at_k=float(scored["precision_at_k"]),
                recall_hit=float(scored["recall_hit"]),
                top1_match=bool(scored["top1_match"]),
                answer_hit=bool(scored["answer_hit"]),
                answer_forbidden_present=bool(scored["answer_forbidden_present"]),
                irrelevant_retrievals=int(scored["irrelevant_retrievals"]),
                retrieval_latency_ms=round(retrieval_ms, 1),
                answer_latency_ms=round(answer_ms, 1),
                retrieved_memory_ids=[m.id for m in memories[: query.top_k]],
                answer=answer,
            )
        )
    return rows


async def run_scenario(
    *,
    scenario: Scenario,
    service: MemoryService,
    modes: list[RetrievalMode],
    llm: LLMProvider,
    llm_model: str,
    refresh_delay_s: float = 1.5,
) -> tuple[str, list[QueryMetrics]]:
    """Seed the scenario's memories + relationships, then run each query across every mode.

    Returns the `run_id` used to tag inserted memories.
    """
    import asyncio

    run_id = new_run_id()
    key_to_id: dict[str, str] = {}
    for spec in scenario.memories:
        memory = _instantiate_memory(spec, run_id)
        saved = await service.create_memory(memory)
        if spec.key and saved.id:
            key_to_id[spec.key] = saved.id

    for rel_spec in scenario.relationships:
        if rel_spec.from_key not in key_to_id or rel_spec.to_key not in key_to_id:
            continue
        from_id = key_to_id[rel_spec.from_key]
        to_id = key_to_id[rel_spec.to_key]
        from_type = _lookup_memory_type(scenario, rel_spec.from_key)
        to_type = _lookup_memory_type(scenario, rel_spec.to_key)
        if from_type is None or to_type is None:
            continue
        await service.create_relationship(
            Relationship(
                from_id=from_id,
                to_id=to_id,
                from_type=from_type,
                to_type=to_type,
                relationship_type=rel_spec.relationship_type,
                confidence=rel_spec.confidence,
                weight=rel_spec.weight,
                metadata=RelationshipMetadata(reason=rel_spec.reason, source=f"eval:{run_id}"),
            )
        )

    await asyncio.sleep(refresh_delay_s)

    rows: list[QueryMetrics] = []
    for query in scenario.queries:
        rows.extend(await _run_query(scenario, query, modes, llm, llm_model, run_id))
    return run_id, rows


def _lookup_memory_type(scenario: Scenario, key: str) -> MemoryType | None:
    for spec in scenario.memories:
        if spec.key == key:
            return memory_type_to_enum(spec.memory_type)
    return None
