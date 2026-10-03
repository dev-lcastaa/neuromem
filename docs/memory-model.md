# NeuroMem — Memory Model

## Scoping decision

The POC treats memory as a **single global shared pool** across all agent instances that share a role. There is no per-agent or per-session partition at query time.

Rationale: the research question is whether shared, consolidated, associative memory helps collaborating agents. A private-memory design would answer a different question.

Every memory records provenance via `metadata`:

| Field | Meaning |
|-------|---------|
| `metadata.agent_id` | Concrete instance identifier of the writer (e.g. container hostname). |
| `metadata.agent_role` | Logical role name (e.g. `research-agent`). All instances of a role share memory. |
| `metadata.source` | Where the memory came from: `conversation`, `consolidation`, `tool`, ... |
| `metadata.entities` | Named entities referenced (agent-supplied). |
| `metadata.embedding_model` | Which embedding model produced `content_vector`. |
| `metadata.embedding_dim` | Its output dimensionality. |

Enforcement of scoping is deferred to a future phase; the fields exist now so we do not have to reindex when we add it.

## Base schema

Every memory type shares this set of fields, drawn from §6 of the outline:

| Field | Type | Purpose |
|-------|------|---------|
| `memory_type` | `keyword` | `episodic`, `semantic`, `procedural`. |
| `content` | `text` + `keyword` subfield | Human-readable body + exact-match/aggregation. |
| `content_vector` | `knn_vector` (3072) | Dense semantic vector (OpenAI `text-embedding-3-large`). |
| `created_at` / `updated_at` / `last_accessed_at` | `date` | Lifecycle timestamps. |
| `importance` / `confidence` / `activation` | `float` (0–1) | Scoring signals. |
| `access_count` | `integer` | Strengthening counter. |
| `source_memory_ids` | `keyword[]` | Provenance (for derived memories). |
| `related_memory_ids` | `keyword[]` | Denormalised edges for cheap read-side filters; canonical graph lives in `neuromem-relationships`. |

Document `_id` is set by the app (ULID); there is no separate `id` field in the mapping.

## Per-type additions

### Episodic (`neuromem-episodic`)

Captures an event.

- `event` — short summary of what happened.
- `context` — surrounding situation.
- `event_timestamp` — when the event occurred (distinct from `created_at`, which is when the memory was written).
- `participants` — actor identifiers.
- `actions`, `outcome` — free text.
- `source_conversation` — reference to the originating conversation.

### Semantic (`neuromem-semantic`)

Durable facts / generalised knowledge.

- `subject` / `predicate` / `object` — optional triple structure; free-form `content` is authoritative.
- `domain` — topic/category tag.
- `superseded_by` — pointer to the fact that replaced this one, once conflict resolution runs.

### Procedural (`neuromem-procedural`)

How-to knowledge.

- `name`, `goal`, `preconditions`, `expected_outcome`.
- `steps` — nested `{ order, action, expected_result }[]`.
- `success_count` / `failure_count` / `success_rate` — updated when a procedure is reused.

### Relationships (`neuromem-relationships`)

The graph layer. No vector.

- `from_id` / `to_id`, `from_type` / `to_type`.
- `relationship_type` ∈ { `DERIVED_FROM`, `RELATED_TO`, `SUPPORTS`, `CONTRADICTS`, `CAUSED_BY`, `RESULTED_IN`, `USED_BY`, `PRECEDES`, `REINFORCES`, `SUPERSEDES` }. Enforced in app code, not in OpenSearch.
- `confidence`, `weight`, `created_at`, `created_by`.
- `metadata.reason` — natural-language explanation of why the edge exists (feeds retrieval explanations).

Graph traversal is done in application code — no graph database is needed for the POC.

## Vector configuration

- Model: **OpenAI `text-embedding-3-large`**, native 3072 dimensions.
- Engine: **Lucene HNSW**, cosine similarity.
- Parameters: `m=16`, `ef_construction=128`, `ef_search=100`.
- OpenSearch ≥ 2.16 required (`remote_vector_index_build_stats` presence confirms this on the deployed cluster).

## Index settings

All four indexes: `number_of_shards: 1`, `number_of_replicas: 1`, `refresh_interval: 1s`, `dynamic: strict`. Strict mapping prevents typos from silently creating fields — safer for a research POC.

Deletes are **not performed** in the POC (see plan §19). Decay and superseding are additive; the raw history stays queryable.
