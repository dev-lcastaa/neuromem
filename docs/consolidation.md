# Consolidation (Phase 4)

Consolidation turns raw agent turns and extracted candidates into durable memories, wired to their sources with `DERIVED_FROM` edges and to conflicting facts with `SUPERSEDES` / `CONTRADICTS` edges. It **never overwrites** an existing memory.

## Trigger paths

| Path | Behaviour |
|---|---|
| Async (per turn) | `POST /agent/chat` persists a job status and publishes a persistent `ConsolidationJob` to RabbitMQ after the graph returns. A separate worker consumes it. The response includes `consolidation_queued` and `consolidation_job_id`. |
| Sync (manual) | `POST /memory/consolidate` runs the same pipeline in-line and returns the full `ConsolidationResult`. Useful for research, replay, and the future dashboard. |

RabbitMQ uses durable main, delayed-retry, and dead-letter queues. The worker processes one delivery at a time, retries failures up to the configured limit, and exposes status at `GET /agent/jobs/{job_id}`. Memory and relationship IDs are deterministic per job so redelivery is idempotent.

## Job shape

```json
{
  "job_id": "5f0c9df9-4750-4be7-8e72-91bdce9ee23c",
  "user_message": "I now prefer Qdrant over OpenSearch.",
  "assistant_reply": "Noted.",
  "conversation_id": "conv-123",
  "candidates": [
    {
      "classification": "SEMANTIC",
      "content": "User now prefers Qdrant as the vector store.",
      "importance": 0.85,
      "confidence": 0.9,
      "entities": ["qdrant", "opensearch"],
      "source": "conversation",
      "reason": "explicit preference change"
    }
  ]
}
```

`candidates` come directly from the agent's `memory_candidate_extraction` node (Phase 3). The consolidator does not call the extraction LLM.

## Pipeline

```text
                    Job (turn + candidates)
                            |
                            v
              +--------------------------+
              |  create source Episodic  |  raw turn, event="agent_turn"
              +-------------+------------+
                            |
             for each candidate ordered by input index
                            |
                            v
              +--------------------------+
              |  classification == IGNORE|  -> skip, record outcome
              +-------------+------------+
                            |
                            v (EPISODIC / SEMANTIC / PROCEDURAL)
              +--------------------------+
              |  candidate -> Memory obj |  auto-embed on write (Phase 2)
              +-------------+------------+
                            |
             SEMANTIC / PROCEDURAL only
                            |
                            v
              +--------------------------+
              |  recall(content, k=3)    |  filter to semantic >= threshold
              +-------------+------------+
                            |
                            v (if similar exist)
              +--------------------------+
              |  LLM adjudicates:        |
              |  DUPLICATE / SUPERSEDES  |
              |  CONTRADICTS / INDEPEND. |
              +-------------+------------+
                            |
                            v
              +--------------------------+
              |  DUPLICATE -> skip       |
              |  else -> create memory + |
              |  DERIVED_FROM turn +     |
              |  SUPERSEDES / CONTRADICTS|
              +--------------------------+
```

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| `RABBITMQ_HOST` | `localhost` | Broker hostname; Compose sets this to `rabbitmq`. |
| `RABBITMQ_PORT` | `5672` | Broker port on the private Compose network. |
| `RABBITMQ_USER` | `neuromem` | Broker account. |
| `RABBITMQ_PASSWORD` | required in deployment | Broker credential, supplied through Jenkins credentials. |
| `CONSOLIDATION_MAX_RETRIES` | `5` | Retries before the job is moved to the dead-letter queue. |
| `CONSOLIDATION_RETRY_DELAY_MS` | `5000` | Delay between attempts. |
| `CONSOLIDATION_SIMILARITY_THRESHOLD` | `0.75` | Semantic score above which a nearby memory is sent to the LLM for adjudication. |
| `CONSOLIDATION_CONFLICT_POOL` | `3` | How many near-neighbours to consider per candidate. |
| `OPENAI_MODEL_REASONING` | `gpt-4o` | Model used for conflict adjudication. |

## Non-overwrite guarantee

- `SUPERSEDES` and `CONTRADICTS` are additive edges — the old memory row is preserved and remains queryable.
- The old memory's `superseded_by` field is not mutated in Phase 4; Phase 5 (lifecycle) can decay superseded memories or the dashboard can filter them out.
- If the LLM fails or returns malformed JSON, the pipeline falls back to `INDEPENDENT` — the candidate is created as a fresh memory with only a `DERIVED_FROM` edge. Never a silent overwrite.

## Provenance

Every derived memory has a `DERIVED_FROM` edge to the raw turn `EpisodicMemory` (which itself carries `source_conversation`). One hop from any semantic fact leads to the raw event that produced it. A dashboard timeline is a graph traversal, not a text search.

## Non-goals (Phase 4)

- Clustering candidates within a job. Each candidate is processed independently.
- Multi-hop provenance (fact -> earlier fact -> older turn).
- Learned adjudication — the LLM does one shot with a small prompt; no calibration or self-check.
- Streaming responses.
- `superseded_by` field updates on the old memory (Phase 5).
- Rate limiting or queue admission control.

## Cost

Each turn with N semantic/procedural candidates issues **1 embedding call for the turn write + N embedding calls for recall + up to N chat calls for adjudication**. With `gpt-4o` at ~$5 / 1M input tokens and candidates well under 500 tokens each, expect fractions of a cent per turn.
