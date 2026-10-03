"""Liveness + readiness endpoints. LLM/embedding probes are cost-gated."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/opensearch")
async def opensearch_health(request: Request) -> dict[str, Any]:
    client = request.app.state.opensearch
    try:
        info = await client.info()
        cluster = await client.cluster.health()
    except Exception as err:
        return {"status": "unreachable", "error": str(err)}
    return {
        "status": cluster.get("status"),
        "cluster_name": cluster.get("cluster_name"),
        "number_of_nodes": cluster.get("number_of_nodes"),
        "distribution": info.get("version", {}).get("distribution"),
        "version": info.get("version", {}).get("number"),
    }


@router.get("/llm")
async def llm_health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    if not settings.health_check_llm:
        return {"status": "skipped", "hint": "set HEALTH_CHECK_LLM=true to enable"}
    llm = request.app.state.llm
    started = time.perf_counter()
    resp = await llm.complete(
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        model=settings.openai_model_extraction,
        temperature=0.0,
    )
    warmup_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "model": resp.model,
        "content_len": len(resp.content),
        "warmup_ms": warmup_ms,
    }


@router.get("/embeddings")
async def embeddings_health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    if not settings.health_check_embeddings:
        return {
            "status": "skipped",
            "hint": "set HEALTH_CHECK_EMBEDDINGS=true to enable",
        }
    provider = request.app.state.embeddings
    started = time.perf_counter()
    vectors = await provider.embed(["neuromem health probe"])
    warmup_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "model": provider.model,
        "dim": len(vectors[0]) if vectors else 0,
        "warmup_ms": warmup_ms,
    }
