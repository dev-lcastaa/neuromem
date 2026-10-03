"""LangGraph agent per plan §12.

Graph:
    START -> classify_request -> recall_memory -> plan -> execute_tools
          -> evaluate_result -> answer -> memory_candidate_extraction -> END

`plan`, `execute_tools`, `evaluate_result` are passthroughs in Phase 3 to preserve
the plan's shape without forcing tool implementations. `memory_candidate_extraction`
emits candidates but does NOT persist them (Phase 4 consolidation owns writes).
"""

from __future__ import annotations

import json
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from apps.agent.models import ChatMessage, MemoryCandidate
from apps.agent.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER_TEMPLATE,
    CLASSIFY_REQUEST_SYSTEM,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)
from apps.agent.state import AgentState
from memory.config import Settings
from memory.retrieval.models import RecallQuery
from memory.service import MemoryService
from models.llm import LLMProvider

# LangGraph generic type params vary by version; we treat the compiled graph opaquely.
AgentGraph = Any


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.role == "user":
            return msg.content
    return ""


def _format_memories_block(state: AgentState) -> str:
    recalled = state.get("recalled_memories", [])
    if not recalled:
        return "(none)"
    lines = []
    for i, r in enumerate(recalled, start=1):
        lines.append(f"{i}. [{r.memory_type.value}] {r.content}")
    return "\n".join(lines)


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


def build_agent_graph(
    *,
    service: MemoryService,
    llm: LLMProvider,
    settings: Settings,
) -> AgentGraph:
    async def classify_request(state: AgentState) -> dict[str, Any]:
        user_msg = _last_user_message(state)
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": CLASSIFY_REQUEST_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=settings.openai_model_extraction,
            response_format="json_object",
            temperature=0.0,
        )
        parsed = _safe_json_loads(resp.content)
        query = str(parsed.get("query") or user_msg).strip() or user_msg
        goal = str(parsed.get("goal") or user_msg).strip() or user_msg
        return {"current_goal": goal, "observations": [f"classified_query={query!r}"]}

    async def recall_memory(state: AgentState) -> dict[str, Any]:
        goal = state.get("current_goal") or _last_user_message(state)
        if not goal:
            return {"recalled_memories": []}
        response = await service.recall(RecallQuery(query=goal, limit=5))
        return {"recalled_memories": list(response.memories)}

    async def _passthrough(_: AgentState) -> dict[str, Any]:
        return {}

    async def answer(state: AgentState) -> dict[str, Any]:
        user_msg = _last_user_message(state)
        memories_block = _format_memories_block(state)
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": ANSWER_USER_TEMPLATE.format(
                        memories_block=memories_block, message=user_msg
                    ),
                },
            ],
            model=settings.openai_model_reasoning,
            temperature=0.2,
        )
        reply = resp.content.strip()
        updated_messages = [
            *state.get("messages", []),
            ChatMessage(role="assistant", content=reply),
        ]
        return {"final_response": reply, "messages": updated_messages}

    async def memory_candidate_extraction(state: AgentState) -> dict[str, Any]:
        user_msg = _last_user_message(state)
        assistant_reply = state.get("final_response", "")
        resp = await llm.complete(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": EXTRACTION_USER_TEMPLATE.format(
                        user_message=user_msg, assistant_reply=assistant_reply
                    ),
                },
            ],
            model=settings.openai_model_extraction,
            response_format="json_object",
            temperature=0.0,
        )
        parsed = _safe_json_loads(resp.content)
        raw = parsed.get("candidates", [])
        candidates: list[MemoryCandidate] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                candidates.append(MemoryCandidate.model_validate(item))
            except ValidationError:
                continue
        return {"candidate_memories": candidates}

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("classify_request", classify_request)
    graph.add_node("recall_memory", recall_memory)
    graph.add_node("plan", _passthrough)  # type: ignore[arg-type]
    graph.add_node("execute_tools", _passthrough)  # type: ignore[arg-type]
    graph.add_node("evaluate_result", _passthrough)  # type: ignore[arg-type]
    graph.add_node("answer", answer)
    graph.add_node("memory_candidate_extraction", memory_candidate_extraction)

    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "recall_memory")
    graph.add_edge("recall_memory", "plan")
    graph.add_edge("plan", "execute_tools")
    graph.add_edge("execute_tools", "evaluate_result")
    graph.add_edge("evaluate_result", "answer")
    graph.add_edge("answer", "memory_candidate_extraction")
    graph.add_edge("memory_candidate_extraction", END)

    return graph.compile()
