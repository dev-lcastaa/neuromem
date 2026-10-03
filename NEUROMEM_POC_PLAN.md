# NeuroMem POC — Agent Implementation Plan

## 1. Project Overview

**Project:** NeuroMem  
**Goal:** Build a local, experimental AI-agent memory system inspired by useful properties of human memory.

The system must demonstrate that an agent can:

1. Record experiences as episodic memories.
2. Extract durable facts into semantic memory.
3. Learn reusable procedures into procedural memory.
4. Recall memories using hybrid, temporal, importance, and relationship-aware retrieval.
5. Consolidate short-term experiences into long-term memories.
6. Strengthen frequently useful memories.
7. Decay memories that become stale or unused.
8. Detect and resolve conflicting memories.
9. Explain why a memory was retrieved.
10. Operate locally in the homelab using Docker and a local LLM.

This is a **POC/research project**, not an attempt to reproduce biological memory exactly.

---

# 2. Design Principles

The implementation agent MUST follow these principles:

- Prefer simple, observable mechanisms over opaque abstractions.
- Keep the Memory Engine independent from the LLM provider.
- Keep storage independent from agent orchestration.
- Make every memory operation inspectable.
- Do not automatically store every conversation as permanent memory.
- Preserve provenance for derived memories.
- Never silently overwrite conflicting facts.
- Prefer deterministic scoring where possible.
- Use LLMs for semantic extraction, classification, and reasoning—not basic database operations.
- Build the smallest useful version first.
- Every major component must have tests.
- Every important memory decision should be explainable.

---

# 3. Initial Homelab Deployment

## Target architecture

```text
                    User / API
                        |
                        v
                +---------------+
                |   LangGraph   |
                |     Agent     |
                +-------+-------+
                        |
          +-------------+-------------+
          |                           |
          v                           v
 +----------------+          +----------------+
 | Memory Engine  |          | Tool Executor  |
 +-------+--------+          +----------------+
         |
         +-------------------------------+
         |               |               |
         v               v               v
    Episodic         Semantic        Procedural
     Memory           Memory          Memory
         |               |               |
         +---------------+---------------+
                         |
                         v
                   +-----------+
                   | OpenSearch|
                   +-----------+

Jetson Orin Nano:
- Local LLM
- Embedding model
- LangGraph agent
- Memory Engine

Dell OptiPlex:
- OpenSearch
- OpenSearch Dashboards if useful
- Supporting Docker services

Raspberry Pis:
- Not required for Phase 1
- Reserved for later distributed experiments
```

Do not introduce Kubernetes, Kafka, Redis, Neo4j, or other infrastructure unless a concrete requirement emerges.

---

# 4. Recommended Technology Stack

## Core

- Python 3.12+
- FastAPI
- Pydantic
- LangGraph
- OpenSearch
- Docker / Docker Compose
- pytest

## LLM

The LLM must be configurable.

Initial target:

- llama.cpp server on Jetson
- Small local instruct model in the 1B–3B range

The application must communicate through an OpenAI-compatible API or an internal adapter so the model can later be replaced.

## Embeddings

Embedding model must also be configurable.

Do not hard-code a specific embedding model into the memory engine.

## UI

Initial UI:

- Streamlit

The UI is secondary to the memory engine.

---

# 5. Repository Structure

Create the repository with this structure:

```text
neuromem/
├── README.md
├── AGENTS.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── Makefile
│
├── apps/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes/
│   │   └── tools/
│   │
│   ├── memory_api/
│   │   └── main.py
│   │
│   └── dashboard/
│       └── app.py
│
├── memory/
│   ├── domain/
│   │   ├── models.py
│   │   ├── enums.py
│   │   └── relationships.py
│   │
│   ├── storage/
│   │   ├── interface.py
│   │   └── opensearch.py
│   │
│   ├── retrieval/
│   │   ├── semantic.py
│   │   ├── keyword.py
│   │   ├── hybrid.py
│   │   └── ranking.py
│   │
│   ├── consolidation/
│   │   ├── episodic.py
│   │   ├── semantic.py
│   │   ├── procedural.py
│   │   └── conflicts.py
│   │
│   ├── lifecycle/
│   │   ├── strengthening.py
│   │   ├── decay.py
│   │   └── pruning.py
│   │
│   └── service.py
│
├── models/
│   ├── llm.py
│   └── embeddings.py
│
├── opensearch/
│   ├── indexes/
│   └── scripts/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── experiments/
│   ├── datasets/
│   ├── scenarios/
│   └── results/
│
└── docs/
    ├── architecture.md
    ├── memory-model.md
    ├── retrieval.md
    ├── consolidation.md
    └── experiments.md
```

The exact structure may evolve, but domain logic must remain separated from infrastructure.

---

# 6. Memory Domain Model

All memory objects must contain a common base.

Minimum fields:

```python
id
memory_type
content
created_at
updated_at
last_accessed_at
importance
confidence
activation
access_count
embedding
metadata
source_memory_ids
related_memory_ids
```

## Memory types

### EpisodicMemory

Represents an event or experience.

Required concepts:

```text
event
context
timestamp
participants/entities
actions
outcome
source conversation
importance
```

Example:

```json
{
  "memory_type": "episodic",
  "content": "OpenSearch deployment failed because the host did not have enough available memory.",
  "importance": 0.78,
  "confidence": 0.96
}
```

### SemanticMemory

Represents a durable fact or generalized knowledge.

Example:

```json
{
  "memory_type": "semantic",
  "content": "The current homelab uses OpenSearch for vector search.",
  "confidence": 0.92,
  "source_memory_ids": ["evt_123", "evt_177"]
}
```

### ProceduralMemory

Represents knowledge of how to perform a task.

Minimum fields:

```text
name
goal
preconditions
steps
expected_outcome
success_count
failure_count
success_rate
source_memory_ids
```

---

# 7. Memory Relationships

Relationships are first-class data.

Initial relationship types:

```text
DERIVED_FROM
RELATED_TO
SUPPORTS
CONTRADICTS
CAUSED_BY
RESULTED_IN
USED_BY
PRECEDES
REINFORCES
SUPERSEDES
```

Example:

```text
evt_001
   |
   +--DERIVED_FROM--> fact_001

fact_001
   |
   +--CONTRADICTS--> fact_002

skill_001
   |
   +--DERIVED_FROM--> evt_009
```

Do not require a graph database for the POC.

Store relationship IDs and types in OpenSearch and implement graph traversal in application code.

---

# 8. OpenSearch Indexes

Create separate indexes initially:

```text
neuromem-episodic
neuromem-semantic
neuromem-procedural
neuromem-relationships
```

Each searchable memory should contain:

- Full text
- Vector embedding
- Metadata
- Timestamp
- Importance
- Confidence
- Activation
- Access count
- Relationship references

Use OpenSearch's vector capabilities for semantic retrieval.

The storage layer must expose an interface so OpenSearch can later be replaced.

---

# 9. Memory API

Implement these operations first:

```text
POST /memory/store
POST /memory/recall
POST /memory/consolidate
POST /memory/relate
POST /memory/access
POST /memory/decay

GET /memory/{id}
GET /memory/{id}/history
GET /memory/{id}/relationships

POST /memory/conflicts/resolve
```

The API should return structured JSON.

Example recall response:

```json
{
  "query": "What vector database am I using?",
  "memories": [
    {
      "id": "fact_001",
      "content": "The system uses OpenSearch for vector search.",
      "score": 0.91,
      "explanation": {
        "semantic": 0.94,
        "keyword": 0.83,
        "importance": 0.88,
        "recency": 0.72,
        "relationship": 0.91
      }
    }
  ]
}
```

The explanation is a core feature, not an optional debug field.

---

# 10. Retrieval Engine

Do not implement vector-only retrieval.

Initial retrieval pipeline:

```text
Query
  |
  +--> Semantic Search
  |
  +--> Keyword Search
  |
  +--> Temporal Filtering
  |
  +--> Relationship Expansion
  |
  v
Candidate Set
  |
  v
Re-ranking
  |
  v
Top Memories
```

Initial ranking formula:

```text
score =
    semantic_similarity * 0.40
  + keyword_score       * 0.15
  + importance          * 0.20
  + recency             * 0.10
  + access_frequency    * 0.10
  + relationship_score  * 0.05
```

Make all weights configurable.

Do not assume this formula is biologically correct.

It is the baseline to experiment against.

---

# 11. Working Memory

LangGraph state should represent temporary working memory.

Example:

```python
class AgentState:
    messages
    current_goal
    current_plan
    observations
    recalled_memories
    tool_results
    candidate_memories
    final_response
```

Working memory must not automatically become long-term memory.

---

# 12. Agent Graph

Initial LangGraph:

```text
START
  |
  v
classify_request
  |
  v
recall_memory
  |
  v
plan
  |
  v
execute_tools
  |
  v
evaluate_result
  |
  v
answer
  |
  v
memory_candidate_extraction
  |
  v
END
```

Consolidation should initially occur after the interaction rather than interrupting the primary agent flow.

---

# 13. Memory Candidate Extraction

After each interaction, ask the LLM to identify candidate memories.

The extraction prompt should classify information into:

```text
IGNORE
EPISODIC
SEMANTIC
PROCEDURAL
```

It must also produce:

```text
importance
confidence
entities
source
reason
```

Example:

```json
{
  "classification": "semantic",
  "content": "The user uses OpenSearch for vector storage.",
  "importance": 0.82,
  "confidence": 0.95,
  "reason": "Stable architectural preference."
}
```

The LLM must never directly write to storage.

It proposes memory candidates.

The Memory Engine validates and persists them.

---

# 14. Consolidation

Implement consolidation as a separate process.

Pipeline:

```text
Recent Episodic Memories
        |
        v
Importance Evaluation
        |
        v
Candidate Clustering
        |
        v
Fact / Procedure Extraction
        |
        v
Conflict Detection
        |
        v
Memory Update
        |
        v
Relationship Creation
```

Consolidation must preserve provenance.

A semantic fact should be traceable to the episodic memories that produced it.

---

# 15. Importance Scoring

Start with deterministic factors:

```text
importance =
    novelty
    + future_usefulness
    + repetition
    + task_relevance
    + explicit_user_signal
```

Normalize to:

```text
0.0 - 1.0
```

Explicit statements such as:

> "Remember that..."

should receive very high importance.

Do not treat all user statements equally.

---

# 16. Confidence

Confidence should represent how strongly the system believes the memory is true.

Sources should influence confidence.

Example:

```text
Explicit user statement     = high
Repeated consistent events  = high
Single inferred observation = medium
LLM inference only          = low
Contradicted fact           = reduced
```

Never confuse confidence with importance.

A fact can be:

```text
high confidence + low importance
```

or:

```text
high importance + medium confidence
```

---

# 17. Conflict Resolution

Never silently overwrite a memory.

Example:

```text
Old:
User prefers OpenSearch.

New:
User is switching to Qdrant.
```

Create:

```text
fact_old
    |
    +--SUPERSEDED_BY--> fact_new
```

or, if unresolved:

```text
fact_old
    |
    +--CONTRADICTS--> fact_new
```

The system should preserve history.

---

# 18. Memory Strengthening

Every successful retrieval should generate an access event.

Initial model:

```text
access_count += 1

activation += reinforcement_factor
```

Cap activation at 1.0.

Example:

```text
activation =
    min(1.0, activation + 0.05)
```

Repeated useful retrieval should increase ranking.

---

# 19. Memory Decay

Implement a configurable decay function.

Baseline:

```text
decay_factor = exp(-lambda * age)
```

Do not delete memories in the first version.

Instead:

```text
effective_activation =
    activation * decay_factor
```

This lets old memories become less likely to surface without destroying experimental data.

---

# 20. Evaluation Framework

The POC is not complete until it can be measured.

Create a test dataset containing known facts and experiences.

Example:

```text
Scenario 001:
User establishes OpenSearch preference.

Scenario 002:
User discusses Jetson deployment.

Scenario 003:
User changes embedding model.

Scenario 004:
User encounters OpenSearch failure.

Scenario 005:
User successfully resolves failure.
```

Evaluate:

### Recall accuracy

Does the agent retrieve the correct memory?

### Precision

Are irrelevant memories excluded?

### Consolidation accuracy

Does the system extract the correct durable facts?

### Conflict handling

Does it recognize changed information?

### Procedural learning

Can it reuse a successful procedure later?

### Memory efficiency

Does the system avoid storing useless information?

---

# 21. Baseline Comparison

Build two modes.

## Mode A — Traditional RAG

```text
Query
  |
Vector Search
  |
LLM
```

## Mode B — NeuroMem

```text
Query
  |
Memory Recall
  |
Hybrid Retrieval
  |
Relationships
  |
Long-Term Memory
  |
LLM
```

Use identical queries against both.

Measure:

```text
answer accuracy
retrieval precision
retrieval recall
latency
memory count
irrelevant retrievals
```

This comparison is critical.

The POC should answer:

> Does cognitive-style memory actually improve agent behavior?

---

# 22. Dashboard

Create a simple Streamlit dashboard.

Pages:

## Overview

Show:

```text
Total Memories
Episodic
Semantic
Procedural
Relationships
Average Confidence
Average Activation
```

## Memory Explorer

Search and inspect individual memories.

## Memory Graph

Display relationships.

## Timeline

Show memory creation and consolidation.

## Retrieval Trace

Show:

```text
Query
Candidates
Scores
Selected Memories
Why Selected
```

## Consolidation Trace

Show:

```text
Events
Candidates
Decisions
New Facts
Relationships
Conflicts
```

---

# 23. Logging

Use structured JSON logging.

Every memory lifecycle operation should produce an event.

Example:

```json
{
  "event": "memory_retrieved",
  "memory_id": "fact_001",
  "query": "What vector database am I using?",
  "score": 0.91
}
```

Other events:

```text
memory_created
memory_updated
memory_accessed
memory_consolidated
memory_strengthened
memory_decayed
memory_conflict_detected
memory_superseded
```

---

# 24. Testing Strategy

## Unit tests

Test:

- Memory models
- Ranking
- Decay
- Strengthening
- Importance scoring
- Conflict detection
- Relationship handling

## Integration tests

Test:

- OpenSearch storage
- Embeddings
- Retrieval
- Memory API

## Agent tests

Test:

- Recall node
- Consolidation node
- Tool interaction
- End-to-end memory lifecycle

## Evaluation tests

Run fixed scenarios and compare expected results.

---

# 25. Phase Plan

## Phase 0 — Bootstrap

Deliver:

```text
[ ] Repository
[ ] Python project
[ ] Docker Compose
[ ] OpenSearch
[ ] Environment configuration
[ ] Basic CI/test command
```

Success criteria:

OpenSearch starts and health checks pass.

---

## Phase 1 — Memory Storage

Deliver:

```text
[ ] Domain models
[ ] OpenSearch indexes
[ ] Storage interface
[ ] CRUD operations
[ ] Relationships
```

Success criteria:

A memory can be created, retrieved, updated, and related to another memory.

---

## Phase 2 — Retrieval

Deliver:

```text
[ ] Embeddings
[ ] Vector search
[ ] Keyword search
[ ] Hybrid retrieval
[ ] Ranking
[ ] Retrieval explanation
```

Success criteria:

A query retrieves the correct memory from a dataset of at least 50 memories.

---

## Phase 3 — Agent

Deliver:

```text
[ ] LangGraph
[ ] Local LLM adapter
[ ] Working memory
[ ] Recall node
[ ] Agent response
```

Success criteria:

Agent can answer a question using a memory created in an earlier interaction.

---

## Phase 4 — Consolidation

Deliver:

```text
[ ] Candidate extraction
[ ] Importance
[ ] Episodic memory
[ ] Semantic memory
[ ] Procedural memory
[ ] Provenance
```

Success criteria:

The agent can convert several related experiences into a durable fact.

---

## Phase 5 — Cognitive Lifecycle

Deliver:

```text
[ ] Strengthening
[ ] Decay
[ ] Conflict detection
[ ] Superseding
[ ] Relationship traversal
```

Success criteria:

Repeatedly useful memories become more retrievable, while stale memories become less prominent.

---

## Phase 6 — Evaluation

Deliver:

```text
[ ] Benchmark dataset
[ ] Traditional RAG baseline
[ ] NeuroMem evaluation
[ ] Accuracy metrics
[ ] Latency metrics
[ ] Retrieval metrics
```

Success criteria:

Produce a report showing whether NeuroMem improves retrieval and agent behavior.

---

## Phase 7 — Dashboard

Deliver:

```text
[ ] Memory explorer
[ ] Graph
[ ] Timeline
[ ] Retrieval trace
[ ] Consolidation trace
```

Success criteria:

A human can inspect how the agent's memory evolved.

---

# 26. Definition of Done for the POC

The POC is considered successful when the following scenario works end-to-end:

```text
1. User tells the agent a fact.
        |
        v
2. Agent records the experience.
        |
        v
3. Memory candidate is generated.
        |
        v
4. Candidate becomes semantic memory.
        |
        v
5. User asks about the fact later.
        |
        v
6. Memory engine retrieves it.
        |
        v
7. Agent uses it to answer.
        |
        v
8. Memory activation increases.
        |
        v
9. Related memories are discovered.
        |
        v
10. User changes the fact.
        |
        v
11. Conflict is detected.
        |
        v
12. New memory supersedes or conflicts with old memory.
        |
        v
13. Dashboard shows the complete history.
```

---

# 27. Agent Instructions

The implementation agent must work incrementally.

For each phase:

1. Inspect the existing repository.
2. Implement only the current phase.
3. Add tests.
4. Run tests.
5. Update documentation.
6. Verify Docker deployment where applicable.
7. Do not proceed to the next phase until the current phase passes its acceptance criteria.
8. Do not introduce unnecessary infrastructure.
9. Do not replace working components without justification.
10. Keep interfaces stable.

The implementation agent must maintain:

```text
docs/architecture.md
docs/memory-model.md
docs/retrieval.md
docs/consolidation.md
docs/experiments.md
```

as the implementation evolves.

---

# 28. First Implementation Task

Start with **Phase 0 only**.

The agent should:

1. Create the repository structure.
2. Create `pyproject.toml`.
3. Create Docker Compose.
4. Deploy OpenSearch.
5. Add health checks.
6. Add environment configuration.
7. Add a minimal FastAPI application.
8. Add pytest.
9. Add a basic README.
10. Add architecture documentation.
11. Verify everything starts successfully.

Do not implement the agent, memory engine, LangGraph, embeddings, or dashboard yet.

After Phase 0 passes, proceed to Phase 1.

---

# 29. Long-Term Research Direction

After the POC works, investigate:

```text
Memory graphs
Hierarchical memory
Associative activation
Context-dependent recall
Memory reconsolidation
Learned retrieval weights
Memory compression
Semantic clustering
Procedural skill optimization
Self-reflection
Multi-agent shared memory
Distributed memory across homelab nodes
```

Potential future architecture:

```text
                    Cognitive Agent
                          |
              +-----------+-----------+
              |                       |
        Working Memory          Memory Engine
                                      |
        +-------------+---------------+-------------+
        |             |               |             |
    Episodic      Semantic       Procedural      Graph
        |             |               |             |
        +-------------+---------------+-------------+
                                      |
                               OpenSearch
                                      |
                         +------------+------------+
                         |                         |
                    Local LLM                 Embeddings
                         |
                      Jetson
```

The central research question remains:

> **Can an agent with structured, consolidating, associative, decaying memory behave more effectively than an agent using conventional vector RAG alone?**

That is the hypothesis this POC should test.
