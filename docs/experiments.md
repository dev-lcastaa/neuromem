# Experiments (Phase 6)

The benchmark harness answers the research question:

> **Can an agent with structured, consolidating, associative memory retrieve and answer better than one using conventional vector RAG alone?**

## Modes

| Mode | Retrieval | Ranking signals used | Notes |
|---|---|---|---|
| `rag` | Semantic k-NN only | Vector cosine only | Naive baseline |
| `neuromem` | `MemoryService.recall` (hybrid) | semantic + keyword + importance + recency + activation + relationship | Full Phase 2–5 pipeline |

Both modes use the **same answer prompt** (`apps/agent/prompts.ANSWER_SYSTEM`) and the same LLM. Retrieval is the only variable.

## Scenarios

YAML files under `experiments/scenarios/`. One file per scenario, five total (plan §20):

| ID | Focus |
|---|---|
| `001-opensearch-preference` | Basic fact recall |
| `002-jetson-deployment` | Multi-fact retrieval, correct subset selection |
| `003-embedding-model-change` | Newer overriding older; recency + importance matter |
| `004-opensearch-failure` | Episodic retrieval + technical detail |
| `005-failure-resolution` | Procedural retrieval + linking to earlier failure |

Each scenario declares a list of memories to seed and a list of queries with expected outcomes (substring-based). All content contains a per-run marker `{run_id}` so runs are isolated within the shared production index without deletes.

## Metrics (per query, per mode)

| Metric | Definition |
|---|---|
| `precision_at_k` | Fraction of top-K retrieved memories whose `content` matches any expected substring |
| `recall_hit` | 1.0 if any of the top-K matches an expected substring, else 0.0 |
| `top1_match` | Whether the top-1 memory's content matches an expected top substring |
| `answer_hit` | Whether the LLM answer contains all expected substrings |
| `answer_forbidden_present` | Whether the answer leaks any forbidden substring (e.g. wrong vendor name) |
| `irrelevant_retrievals` | Count of top-K memories with no expected substring |
| `retrieval_latency_ms` | Wall-clock time to fetch the retrieval set |
| `answer_latency_ms` | Wall-clock time for the LLM to produce the answer |

Aggregate table averages these across all queries per mode.

## Running

```powershell
$env:Path = "C:\Users\luis0\.local\bin;$env:Path"
$env:UV_LINK_MODE = "copy"
uv run --no-sync python -m experiments.main
```

Prints the path to the newly-written report under `experiments/results/report-<UTC-timestamp>.md`.

## Isolation & repeatability

- Every scenario run generates a fresh `run_id` (ULID prefix) and templates it into both the seeded memory content and the query text. Recall matches the marker first, so runs don't cross-contaminate.
- Seeded memories are **not deleted** after a run — they stay in the cluster for later inspection. The `## Run IDs` section of the report lets you find the specific tag if you want to query or delete manually via Dev Tools.
- Cost per full run: roughly one embedding call per query per mode + one chat completion per query per mode. With 5 scenarios × 2 queries × 2 modes that's ~20 calls each, well under $0.10 on `gpt-4o` + `text-embedding-3-large`.

## Interpreting the report

Two headline numbers to compare across modes in the Aggregate section:
1. **P@K** — do we surface relevant memories at all?
2. **Answer ✓** — does the resulting LLM answer contain the expected fact?

Beat the RAG baseline on **both** and the plan's central hypothesis has evidence in its favour. Beating it on one but not the other is worth writing up as a bounded finding.

## Non-goals

- LLM-as-judge scoring — the POC uses substring matching. It's blunter but reproducible and free.
- Statistical significance testing — sample size is too small for POC purposes.
- Cross-encoder reranking of retrieval outputs.
- Automatic seed cleanup between runs — deliberate, per plan §19 "no deletes".
