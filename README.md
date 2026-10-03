# NeuroMem POC

Experimental AI-agent memory service with managed storage and MCP tools. Chat inference and embeddings use the hosted OpenAI API; no local LLM, GPU, or Jetson is required. The POC tests one hypothesis: **does structured, consolidating, associative, decaying memory improve agent behavior over conventional vector RAG alone?**

See [NEUROMEM_POC_PLAN.md](NEUROMEM_POC_PLAN.md) for the full research plan and phase-by-phase deliverables.

---

## Topology

```text
Agent clients -> MCP :8001/mcp ----+
                  |
Dashboard :8501 ------------------+-> Memory API :8000
                      |          |
                      v          v
                  OpenSearch :9200   OpenAI API (HTTPS)
                  persistent volume  chat + embeddings
```

Docker Compose manages all four local services. The API bootstraps the four
`neuromem-*` indexes from the bundled mappings. OpenSearch starts as a single-node
POC cluster, not a highly available production cluster.

The storage engine remains **OpenSearch**, not Elasticsearch: the existing
adapter and vector mappings use OpenSearch k-NN APIs.

## Current decisions

| Area | Choice |
|------|--------|
| LLM | OpenAI `gpt-4o-mini` (extraction) + `gpt-4o` (reasoning) |
| Embeddings | OpenAI `text-embedding-3-large` at native 3072-dim |
| Vector store | OpenSearch k-NN (Lucene HNSW, cosine) |
| Multi-agent | Single global shared memory pool; `agent_id` recorded for provenance only |
| Consolidation | Async worker per turn (Phase 4) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## Prerequisites

- Docker Engine with Docker Compose, or Docker Desktop with Linux containers
- An OpenAI API key
- Internet access for container images and hosted model calls
- For Python development: Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)

On Linux, including a Jetson host, OpenSearch needs `vm.max_map_count` of at least
`262144`. Have a host administrator configure it before startup. Docker Desktop
users need the setting in its Linux VM/WSL environment if OpenSearch reports a
bootstrap failure. Allow several GB of container memory; the configured JVM heap
alone is 512 MB and does not include off-heap vector memory.

## First-time setup

```bash
cp .env.example .env             # fill in OPENAI_API_KEY at minimum
docker compose up --build -d
docker compose ps
```

In PowerShell, use `Copy-Item .env.example .env` for the first command. Do not
overwrite an existing environment file containing your secrets.

Endpoints:

- Dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs
- MCP Streamable HTTP: http://localhost:8001/mcp
- OpenSearch: http://localhost:9200

Compose waits for OpenSearch readiness before starting the API, then starts the
MCP server and dashboard after API liveness. Health probes do not make paid model
calls by default. `docker compose down` stops the stack and retains memory data;
`docker compose down -v` permanently deletes the storage volume.

This is a local trusted POC: OpenSearch security is disabled and the API/MCP
have no user authentication or tenant isolation. All published ports bind to
localhost. Do not expose them publicly; remote use requires authenticated TLS
access and appropriate authorization. No container mounts the Docker socket.

## Agent MCP connection

For a VS Code workspace MCP configuration, use:

```json
{
    "servers": {
        "neuromem": {
            "type": "http",
            "url": "http://localhost:8001/mcp"
        }
    }
}
```

Available tools: `store_memory`, `recall_memories`, `get_memory`,
`search_memories`, `consolidate_memories`, `relate_memories`, `memory_graph`,
and `memory_stats`. Tool inputs use the same typed models as the REST API.
Recall can reinforce retrieved memories; consolidation can supersede prior
memories. MCP annotations distinguish read-only tools from these writes.

The calling agent supplies its own LLM. NeuroMem uses its configured OpenAI
models for internal consolidation and embeddings, not to host that agent's model.
Store and recall use paid embeddings; consolidation can also incur chat charges.
Retrieved memories should be treated as untrusted data, not agent instructions.

To run MCP separately against an existing API:

```bash
uv sync --all-groups
uv run python -m apps.mcp_server.main
```

Set `MEMORY_API_URL` in the process environment to change its upstream API.
The default is `http://localhost:8000`. MCP listens on port 8001; use a firewall
to restrict direct Python deployments, which bind to all interfaces.

On Windows, targets in the Makefile assume a POSIX shell. Use WSL, Git Bash, or invoke the equivalent `uv run ...` commands directly.

## Verify OpenSearch reachability

```bash
curl http://localhost:9200/_cluster/health
curl http://localhost:9200/_plugins/_knn/stats
curl 'http://localhost:9200/_cat/indices/neuromem-*?v'
```

You should see the four `neuromem-*` indexes. Yellow health is normal on a
single-node cluster when mappings request replicas. Check
`http://localhost:8000/health/opensearch` for index readiness.

## Development commands

```bash
make install           # uv sync
make lint              # ruff check + format --check
make format            # ruff format + auto-fix
make typecheck         # mypy
make test              # pytest unit tests
make test-cov          # pytest with coverage
make test-integration  # pytest -m integration  (requires OpenSearch)
make run-api           # uvicorn memory_api on :8000
make run-dashboard     # streamlit on :8501
make run-mcp           # MCP tools on :8001/mcp
make health            # curl local health endpoints
```

## Repository layout

Phase 0 lays down the top-level scaffolding. Each subsequent phase populates its own module tree — see [NEUROMEM_POC_PLAN.md](NEUROMEM_POC_PLAN.md) §5 for the full target structure.

```text
neuromem-poc/
├── apps/
│   ├── memory_api/       FastAPI service (Phase 0.D)
│   ├── dashboard/        Streamlit UI (Phase 0.E)
│   └── mcp_server/       Agent-facing MCP tools over Streamable HTTP
├── memory/               Domain, storage, retrieval, consolidation, lifecycle
├── models/               LLM + embedding adapters
├── opensearch/indexes/   Index mapping JSON (source of truth for Dev Tools PUTs)
├── tests/                unit / integration / evaluation
└── docs/                 architecture, memory-model, retrieval, ...
```

## Cost guards

`text-embedding-3-large` and the chat models are paid APIs. Both health endpoints that exercise OpenAI are feature-flagged off by default:

```env
HEALTH_CHECK_LLM=false
HEALTH_CHECK_EMBEDDINGS=false
```

Unit tests mock the OpenAI client at the object level (no live network calls).

## Status

The POC includes storage, recall, consolidation, a LangGraph agent, dashboard,
experiments, and agent-facing MCP tools. The default deployment now manages its
own storage instead of requiring a pre-provisioned LAN cluster. The research
plan and older architecture documents retain historical deployment assumptions;
this README and the Compose file describe the current deployment.