# NeuroMem — Architecture (Phase 0)

## Two-host topology

```text
                     +--------------------------+
                     |     Dell OptiPlex        |
                     |                          |
                     |  OpenSearch 2.16+        |
                     |  3-node cluster          |
                     |  `homelab-opensearch`    |
                     |  http://192.168.1.208    |
                     |    :9200  API            |
                     |    :5601  Dashboards     |
                     +-------------+------------+
                                   |
                                   | LAN (9200 open only to LAN)
                                   |
                     +-------------+------------+
                     |    Jetson Orin Nano      |
                     |                          |
                     |  memory_api  (:8000)     |
                     |  dashboard   (:8501)     |
                     |  agent       (Phase 3+)  |
                     +-------------+------------+
                                   |
                                   | HTTPS
                                   v
                          +-----------------+
                          |  OpenAI API     |
                          |  gpt-4o-mini    |
                          |  gpt-4o         |
                          |  text-embedding |
                          |  -3-large       |
                          +-----------------+
```

## Component boundaries

| Layer | Concern | Module | Phase |
|-------|---------|--------|-------|
| Config | Environment settings, defaults | `memory/config.py` | 0.C |
| Logging | Structured JSON logs, request context | `memory/logging.py` | 0.C |
| LLM adapter | OpenAI chat completions (extraction + reasoning) | `models/llm.py` | 0.C |
| Embeddings adapter | OpenAI embeddings, batching, retry, dim assertion | `models/embeddings.py` | 0.C |
| Storage interface | CRUD, k-NN, relationships | `memory/storage/` | 1 |
| Retrieval | Semantic + keyword + hybrid + rank | `memory/retrieval/` | 2 |
| Domain models | `EpisodicMemory`, `SemanticMemory`, `ProceduralMemory` | `memory/domain/` | 1 |
| Consolidation | Extract facts/procedures from episodic events | `memory/consolidation/` | 4 |
| Lifecycle | Strengthening, decay | `memory/lifecycle/` | 5 |
| Memory API | REST surface, health, contract-visible stubs | `apps/memory_api/` | 0.D |
| Dashboard | Streamlit inspection UI | `apps/dashboard/` | 0.E, filled 7 |
| Agent | LangGraph orchestration | `apps/agent/` | 3 |

The **Memory Engine** (`memory/`) is independent of the LLM provider and the agent orchestration layer. Storage is behind an interface (Phase 1) so OpenSearch can be swapped without touching retrieval or consolidation logic.

## Data flow (Phase 0 → target)

Phase 0 wires only the top of this diagram. Solid boxes are implemented; dashed boxes are stubbed as 501 endpoints.

```text
    Client / Agent
         |
         v
    +-------------------+
    |  memory_api       |
    |  /health/*  OK    |
    |  /memory/*  501   |
    +---------+---------+
              |
     +--------+---------+---------+
     |                  |         |
     v                  v         v
  OpenSearch        OpenAI     OpenAI
   (3-node)     text-embedding chat/completions
                   -3-large   gpt-4o(-mini)
```

## Multi-agent memory model

The POC uses a **single global memory pool**. Every memory records `metadata.agent_id` (the writer) and `metadata.agent_role` (the role name), but no scoping is enforced at read time — any agent instance can read any memory. This is a deliberate simplification to test the shared-memory hypothesis of the POC.

## Cost, security, and reversibility

- **LAN-only**: OpenSearch security plugin is disabled; port 9200 must only be reachable from the LAN.
- **Cost guards**: `/health/llm` and `/health/embeddings` return `skipped` unless `HEALTH_CHECK_LLM=true` / `HEALTH_CHECK_EMBEDDINGS=true`. The unit test suite mocks all outbound HTTP with `respx`.
- **Storage**: no deletes in the POC (see §19 of the plan). Superseding and decay are additive.
- **Provenance**: every vector records `metadata.embedding_model` and `metadata.embedding_dim`, so a model swap can selectively re-embed stale documents rather than wipe the index.
