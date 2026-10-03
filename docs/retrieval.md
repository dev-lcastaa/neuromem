# Retrieval (Phase 2)

## Pipeline

```text
query text
    |
    v
+--------------------+       +----------------------+
| OpenAI embedding   |       | OpenAI               |
| text-embedding-3-  |       | (chat, not used yet) |
| large (3072-dim)   |       +----------------------+
+---------+----------+
          |
          v
+---------+---------------------------------+
|  MemoryStore.search_semantic (k-NN)       |
|  MemoryStore.search_keyword (multi_match) |
+---------+---------------------------------+
          |
          v
+-------------------+
| Union candidates  |
| dedup by memory_id|
+---------+---------+
          |
          v
+-------------------+     +-----------------------+
| mget documents    | --> | get_relationships_    |
| (no content_vec)  |     | for_ids (edge boost)  |
+---------+---------+     +-----------------------+
          |
          v
+-------------------+
| Rank each doc:    |
|   weighted sum of |
|   6 signals       |
+---------+---------+
          |
          v
top-K RecallResult[]  +  Explanation per result  +  weights echoed back
```

## Ranking

Baseline weights from §10 of the plan:

| Signal | Default weight | Source |
|---|---|---|
| `semantic` | 0.40 | OpenSearch k-NN cosine similarity, already 0–1 |
| `keyword` | 0.15 | OpenSearch `multi_match` BM25, normalised by max in the result set |
| `importance` | 0.20 | `memory.importance` (author-supplied) |
| `recency` | 0.10 | `exp(-λ_recency · age_days)` where `λ_recency=0.05` by default |
| `activation` | 0.10 | `memory.activation × exp(-λ_activation · age_days_since_last_access)` — see Cognitive Lifecycle below |
| `relationship` | 0.05 | Edge-weighted boost when candidates share an edge |

All weights and constants are settings (`RANK_WEIGHT_*`, `RECENCY_DECAY_LAMBDA`, `ACTIVATION_DECAY_LAMBDA`) and can be overridden per-query via the `weights` field of `RecallQuery`. **The default formula is a baseline for experimentation, not a claim of correctness.**

## Cognitive lifecycle (Phase 5)

### Strengthening (§18)

After every `POST /memory/recall`, the top-K results have their `activation` and `access_count` bumped in-place via an OpenSearch painless `_bulk` update. Strengthening is fire-and-forget — the recall response returns first, the bump runs on the next tick.

```
activation      = min(1.0, activation + ACTIVATION_REINFORCEMENT)  # default 0.05
access_count   += 1
last_accessed_at = now
```

Explicit strengthening is also exposed:

```
POST /memory/access
{ "memory_ids": ["sem_a", "epi_b"], "bump": 0.05 }
```

Set `STRENGTHENING_ENABLED=false` to disable both paths (tests default to this).

### Decay (§19)

Decay is **computed at query time, never persisted**. In the ranking formula:

```
effective_activation = activation × exp(-ACTIVATION_DECAY_LAMBDA × age_days)
```

where `age_days` is time since `last_accessed_at` (falling back to `created_at`). Memories that are not retrieved lose priming; memories that are retrieved regain it.

The dedicated diagnostic endpoint returns the same view without touching storage:

```
POST /memory/decay
{ "memory_ids": ["sem_a"] }
```

`GET /memory/{id}/history` returns the same lifecycle snapshot plus the memory's edges — the read-only view a dashboard timeline would render.

**No deletes.** Superseded, contradicted, and decayed memories all remain queryable; ranking simply pushes them down.

## Explanation

Every `RecallResult` includes an `explanation` object with the six component scores that produced the final `score`. The `RecallResponse` also echoes the `weights` used, so a downstream tool can replay the ranking or A/B two weight configs.

Example response (§9-compatible):

```json
{
  "query": "which vector database am I using",
  "weights": {
    "semantic": 0.40, "keyword": 0.15, "importance": 0.20,
    "recency": 0.10, "activation": 0.10, "relationship": 0.05
  },
  "memories": [
    {
      "id": "sem_01H...",
      "memory_type": "semantic",
      "content": "The homelab uses OpenSearch for vector search.",
      "score": 0.91,
      "explanation": {
        "semantic": 0.94,
        "keyword": 0.83,
        "importance": 0.9,
        "recency": 0.72,
        "activation": 0.0,
        "relationship": 0.0
      },
      "memory": { "...full Memory document..." }
    }
  ]
}
```

## Write-side embedding

`MemoryService.create_memory` embeds `content` automatically when `content_vector` is not supplied. The provider's `model` and `dim` are captured in `metadata.embedding_model` / `metadata.embedding_dim` so future migrations can detect and selectively re-embed stale vectors (see architecture.md).

## Relationship expansion

Cheap and app-side. After the candidate set is built, we fetch every edge whose endpoints are in the set (single `_search` on `neuromem-relationships` with a `terms` query). If both endpoints are candidates, both receive a boost of `confidence × weight × 0.5`, capped at 1.0 per memory. No graph traversal beyond one hop in the POC; expanding to N hops is a future refinement.

## Non-goals (Phase 2)

- Learned ranking. Weights are fixed per query; there is no online learning.
- Reranking with cross-encoders.
- Temporal filtering (`time_window_days` is not yet in `RecallQuery`).
- Diversity/MMR — pure score-descending order.
- Vector quantization or ANN parameter tuning.

## Cost note

Every `/memory/recall` call issues **one** OpenAI embedding request against `text-embedding-3-large`. Recall is otherwise free — all remaining work happens in OpenSearch and application code.
