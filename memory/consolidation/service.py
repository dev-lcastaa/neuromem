"""Consolidation pipeline (§14).

Per-turn flow:

1. Persist a raw `EpisodicMemory` capturing the user message + assistant reply.
2. For each candidate produced by the agent's extraction node:
   - IGNORE  -> skip
   - EPISODIC -> create the memory, link DERIVED_FROM the source turn
   - SEMANTIC/PROCEDURAL -> recall near neighbours; if any are similar enough,
     ask the LLM to adjudicate DUPLICATE / SUPERSEDES / CONTRADICTS /
     INDEPENDENT. Persist based on that decision. Duplicates are skipped;
     superseded/contradicted memories keep their old rows and gain an edge.

Never silently overwrites — supersession and contradiction are additive edges.
"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from apps.agent.models import MemoryCandidate
from memory.config import Settings
from memory.consolidation.models import (
    ConflictDecision,
    ConsolidationDecision,
    ConsolidationJob,
    ConsolidationOutcome,
    ConsolidationResult,
)
from memory.consolidation.prompts import CONFLICT_SYSTEM, CONFLICT_USER_TEMPLATE
from memory.domain.enums import MemoryType, RelationshipType
from memory.domain.models import (
    EpisodicMemory,
    MemoryMetadata,
    ProceduralMemory,
    Relationship,
    RelationshipMetadata,
    SemanticMemory,
)
from memory.retrieval.models import RecallQuery
from memory.service import MemoryService
from memory.storage.interface import Memory
from models.llm import LLMProvider

_CONFLICT_CHECKED_TYPES = {MemoryType.SEMANTIC, MemoryType.PROCEDURAL}


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        parsed = json.loads(text)
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class ConsolidationService:
    def __init__(self, *, service: MemoryService, llm: LLMProvider, settings: Settings) -> None:
        self._service = service
        self._llm = llm
        self._settings = settings

    async def process(self, job: ConsolidationJob) -> ConsolidationResult:
        source = await self._persist_turn(job)
        assert source.id is not None

        outcomes: list[ConsolidationOutcome] = []
        for idx, cand in enumerate(job.candidates):
            outcome = await self._process_candidate(idx, cand, source.id, job.job_id)
            outcomes.append(outcome)

        return ConsolidationResult(
            conversation_id=job.conversation_id,
            source_memory_id=source.id,
            outcomes=outcomes,
        )

    async def _persist_turn(self, job: ConsolidationJob) -> EpisodicMemory:
        content = f"USER: {job.user_message}\nASSISTANT: {job.assistant_reply}"
        turn = EpisodicMemory(
            id=_stable_memory_id(job.job_id, "source", MemoryType.EPISODIC),
            content=content,
            event="agent_turn",
            source_conversation=job.conversation_id,
            importance=0.3,
            confidence=1.0,
            metadata=MemoryMetadata(
                agent_id=self._settings.agent_role_id,
                agent_role=self._settings.agent_role_id,
                source="agent",
            ),
        )
        persisted = await self._service.create_memory(turn)
        assert isinstance(persisted, EpisodicMemory)
        return persisted

    async def _process_candidate(
        self, index: int, cand: MemoryCandidate, source_id: str, job_id: str
    ) -> ConsolidationOutcome:
        if cand.classification == "IGNORE":
            return ConsolidationOutcome(
                candidate_index=index,
                classification=cand.classification,
                action="ignored",
            )

        memory_type = _MEMORY_TYPE_MAP[cand.classification]
        memory_id = _stable_memory_id(job_id, f"candidate:{index}:{memory_type.value}", memory_type)

        conflicts: list[ConsolidationDecision] = []
        if memory_type in _CONFLICT_CHECKED_TYPES:
            conflicts = await self._detect_conflicts(cand, memory_type, exclude_id=memory_id)

        duplicates = [d for d in conflicts if d.decision == "DUPLICATE"]
        if duplicates:
            return ConsolidationOutcome(
                candidate_index=index,
                classification=cand.classification,
                action="duplicate",
                conflicts=duplicates,
            )

        memory = _candidate_to_memory(
            cand,
            memory_type,
            self._settings,
            memory_id=memory_id,
        )
        created = await self._service.create_memory(memory)
        assert created.id is not None

        await self._service.create_relationship(
            Relationship(
                id=_stable_relationship_id(created.id, source_id, RelationshipType.DERIVED_FROM),
                from_id=created.id,
                to_id=source_id,
                from_type=memory_type,
                to_type=MemoryType.EPISODIC,
                relationship_type=RelationshipType.DERIVED_FROM,
                confidence=cand.confidence,
            )
        )

        superseded = [d for d in conflicts if d.decision == "SUPERSEDES"]
        contradicted = [d for d in conflicts if d.decision == "CONTRADICTS"]
        for dec in superseded:
            await self._link_conflict(created.id, memory_type, dec, RelationshipType.SUPERSEDES)
        for dec in contradicted:
            await self._link_conflict(created.id, memory_type, dec, RelationshipType.CONTRADICTS)

        action: str
        if superseded:
            action = "superseded"
        elif contradicted:
            action = "contradicted"
        else:
            action = "created"

        return ConsolidationOutcome(
            candidate_index=index,
            classification=cand.classification,
            action=action,
            created_memory_id=created.id,
            conflicts=[*superseded, *contradicted],
        )

    async def _detect_conflicts(
        self, cand: MemoryCandidate, memory_type: MemoryType, *, exclude_id: str
    ) -> list[ConsolidationDecision]:
        recall = await self._service.recall(
            RecallQuery(
                query=cand.content,
                memory_types=[memory_type],
                limit=self._settings.consolidation_conflict_pool,
                include_relationships=False,
            )
        )
        similar = [
            r
            for r in recall.memories
            if r.id != exclude_id
            if r.explanation.semantic >= self._settings.consolidation_similarity_threshold
        ]
        if not similar:
            return []

        existing_block = "\n".join(f"{i + 1}. [{r.id}] {r.content}" for i, r in enumerate(similar))
        prompt_user = CONFLICT_USER_TEMPLATE.format(
            new_content=cand.content, existing_block=existing_block
        )

        try:
            resp = await self._llm.complete(
                messages=[
                    {"role": "system", "content": CONFLICT_SYSTEM},
                    {"role": "user", "content": prompt_user},
                ],
                model=self._settings.openai_model_reasoning,
                response_format="json_object",
                temperature=0.0,
            )
        except Exception:
            return []

        parsed = _safe_json_loads(resp.content)
        raw = parsed.get("decisions", [])
        allowed_ids = {r.id for r in similar}
        decisions: list[ConsolidationDecision] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                dec = ConsolidationDecision.model_validate(item)
            except ValidationError:
                continue
            if dec.existing_id in allowed_ids:
                decisions.append(dec)
        return decisions

    async def _link_conflict(
        self,
        new_id: str,
        new_type: MemoryType,
        decision: ConsolidationDecision,
        rel_type: RelationshipType,
    ) -> None:
        target = await self._service.get_memory(decision.existing_id)
        if target is None:
            return
        await self._service.create_relationship(
            Relationship(
                id=_stable_relationship_id(new_id, decision.existing_id, rel_type),
                from_id=new_id,
                to_id=decision.existing_id,
                from_type=new_type,
                to_type=target.memory_type,
                relationship_type=rel_type,
                confidence=0.9,
                metadata=RelationshipMetadata(reason=decision.reason),
            )
        )


_MEMORY_TYPE_MAP: dict[str, MemoryType] = {
    "EPISODIC": MemoryType.EPISODIC,
    "SEMANTIC": MemoryType.SEMANTIC,
    "PROCEDURAL": MemoryType.PROCEDURAL,
}


def _candidate_to_memory(
    cand: MemoryCandidate,
    memory_type: MemoryType,
    settings: Settings,
    *,
    memory_id: str,
) -> Memory:
    metadata = MemoryMetadata(
        agent_id=settings.agent_role_id,
        agent_role=settings.agent_role_id,
        source="consolidation",
        entities=list(cand.entities),
    )
    base_kwargs: dict[str, Any] = {
        "id": memory_id,
        "content": cand.content,
        "importance": cand.importance,
        "confidence": cand.confidence,
        "metadata": metadata,
    }
    if memory_type is MemoryType.EPISODIC:
        return EpisodicMemory(**base_kwargs, event=cand.content[:200])
    if memory_type is MemoryType.SEMANTIC:
        return SemanticMemory(**base_kwargs)
    if memory_type is MemoryType.PROCEDURAL:
        return ProceduralMemory(**base_kwargs, name=cand.content[:60])
    raise ValueError(f"Unhandled memory type: {memory_type}")


def _stable_memory_id(job_id: str, key: str, memory_type: MemoryType) -> str:
    stable = uuid5(NAMESPACE_URL, f"neuromem:{job_id}:{key}")
    prefix = {
        MemoryType.EPISODIC: "epi",
        MemoryType.SEMANTIC: "sem",
        MemoryType.PROCEDURAL: "pro",
    }[memory_type]
    return f"{prefix}_{stable.hex}"


def _stable_relationship_id(from_id: str, to_id: str, rel_type: RelationshipType) -> str:
    stable = uuid5(NAMESPACE_URL, f"neuromem:{from_id}:{to_id}:{rel_type.value}")
    return f"rel_{stable.hex}"


_ConflictDecisionValue = ConflictDecision  # re-export for tests
