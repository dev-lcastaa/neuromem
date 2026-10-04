from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from apps.mcp_server.main import create_server


@asynccontextmanager
async def mcp_client() -> AsyncIterator[tuple[httpx.AsyncClient, list[httpx.Request]]]:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/memory/missing":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/memory/offline":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"ok": True})

    server = create_server("http://memory-api", transport=httpx.MockTransport(respond))
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8001",
            headers={"Accept": "application/json, text/event-stream"},
        ) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["result"]["serverInfo"]["name"] == "NeuroMem"
            yield client, requests


async def test_tool_discovery() -> None:
    async with mcp_client() as (client, requests):
        response = await client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
        tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
        assert set(tools) == {
            "store_memory",
            "recall_memories",
            "get_memory",
            "search_memories",
            "consolidate_memories",
            "relate_memories",
            "memory_graph",
            "memory_stats",
        }
        assert "ctx" not in tools["store_memory"]["inputSchema"]["properties"]
        assert tools["recall_memories"]["annotations"]["readOnlyHint"] is False
        assert tools["memory_stats"]["annotations"]["readOnlyHint"] is True
        assert not requests


@pytest.mark.parametrize(
    ("name", "arguments", "method", "path"),
    [
        (
            "store_memory",
            {"memory": {"memory_type": "semantic", "content": "Use OpenSearch"}},
            "POST",
            "/memory/store",
        ),
        ("recall_memories", {"query": {"query": "deployment"}}, "POST", "/memory/recall"),
        ("get_memory", {"memory_id": "abc"}, "GET", "/memory/abc"),
        (
            "search_memories",
            {"query": "deployment", "memory_types": ["semantic"], "limit": 5},
            "GET",
            "/memory/search",
        ),
        (
            "consolidate_memories",
            {"job": {"user_message": "Remember this"}},
            "POST",
            "/memory/consolidate",
        ),
        (
            "relate_memories",
            {
                "relationship": {
                    "from_id": "a",
                    "to_id": "b",
                    "from_type": "semantic",
                    "to_type": "episodic",
                    "relationship_type": "RELATED_TO",
                }
            },
            "POST",
            "/memory/relate",
        ),
        ("memory_graph", {"center_id": "abc", "depth": 2}, "GET", "/memory/graph"),
        ("memory_stats", {}, "GET", "/memory/stats"),
    ],
)
async def test_tool_routes(name: str, arguments: dict, method: str, path: str) -> None:
    async with mcp_client() as (client, requests):
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        result = response.json()["result"]
        assert result.get("isError", False) is False, result
        assert result["structuredContent"] == {"ok": True}
        assert requests[0].method == method
        assert requests[0].url.path == path
        if name == "search_memories":
            assert requests[0].url.params.get_list("memory_types") == ["semantic"]
            assert requests[0].url.params["limit"] == "5"
        if method == "POST":
            payload = json.loads(requests[0].content)
            for key, value in next(iter(arguments.values())).items():
                assert payload[key] == value


@pytest.mark.parametrize("memory_id", ["missing", "offline"])
async def test_upstream_failures_are_tool_errors(memory_id: str) -> None:
    async with mcp_client() as (client, requests):
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "get_memory", "arguments": {"memory_id": memory_id}},
            },
        )
        assert response.json()["result"]["isError"] is True
        assert len(requests) == 1


async def test_invalid_arguments_do_not_call_api() -> None:
    async with mcp_client() as (client, requests):
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_memories", "arguments": {"query": "test", "limit": 101}},
            },
        )
        assert response.json()["result"]["isError"] is True
        assert not requests


async def test_dns_rebinding_is_rejected() -> None:
    async with mcp_client() as (client, requests):
        response = await client.post(
            "/mcp",
            headers={"Host": "untrusted.example"},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "ping",
            },
        )
        assert response.status_code == 421
        assert not requests


async def test_configured_lan_host_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROMEM_HOST_IP", "192.168.1.208")
    server = create_server("http://memory-api")
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://192.168.1.208:8001",
            headers={
                "Accept": "application/json, text/event-stream",
                "Origin": "http://192.168.1.208:8501",
            },
        ) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                },
            )

    assert response.status_code == 200
