"""`python -m experiments.main` — end-to-end benchmark run against live cluster + OpenAI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from experiments.loader import load_scenarios
from experiments.modes import NaiveRAGMode, NeuroMemMode, RetrievalMode
from experiments.report import build_report
from experiments.runner import run_scenario
from memory.config import get_settings
from memory.logging import configure_logging, get_logger
from memory.service import MemoryService
from memory.storage.opensearch import OpenSearchMemoryStore
from models.embeddings import OpenAIEmbeddingProvider
from models.llm import OpenAIProvider

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"


async def _amain() -> Path:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("experiments.main")

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    scenarios = load_scenarios(SCENARIOS_DIR)
    if not scenarios:
        raise SystemExit(f"No scenarios found in {SCENARIOS_DIR}")

    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password)

    os_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=settings.opensearch_verify_certs,
    )
    ai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    store = OpenSearchMemoryStore(client=os_client, mappings_dir=Path(settings.mappings_dir))
    await store.bootstrap()
    embedder = OpenAIEmbeddingProvider(
        client=ai_client,
        model=settings.openai_model_embedding,
        dim=settings.embedding_dim,
        batch_max=settings.embedding_batch_max,
    )
    llm = OpenAIProvider(client=ai_client, default_model=settings.openai_model_reasoning)
    service = MemoryService(store=store, embedder=embedder, settings=settings)

    modes: list[RetrievalMode] = [
        NaiveRAGMode(embedder=embedder, store=store),
        NeuroMemMode(service=service),
    ]

    started_at = datetime.now(UTC)
    all_rows = []
    run_ids: dict[str, str] = {}
    try:
        for scenario in scenarios:
            logger.info("experiments.scenario.start", scenario=scenario.id)
            run_id, rows = await run_scenario(
                scenario=scenario,
                service=service,
                modes=modes,
                llm=llm,
                llm_model=settings.openai_model_reasoning,
            )
            run_ids[scenario.id] = run_id
            all_rows.extend(rows)
            logger.info(
                "experiments.scenario.done",
                scenario=scenario.id,
                run_id=run_id,
                rows=len(rows),
            )
    finally:
        await os_client.close()
        await ai_client.close()

    finished_at = datetime.now(UTC)
    report = build_report(
        rows=all_rows,
        llm_model=settings.openai_model_reasoning,
        embedding_model=settings.openai_model_embedding,
        embedding_dim=settings.embedding_dim,
        started_at=started_at,
        finished_at=finished_at,
        run_ids=run_ids,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = RESULTS_DIR / f"report-{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("experiments.report.written", path=str(out_path))
    return out_path


def main() -> None:
    path = asyncio.run(_amain())
    print(str(path))


if __name__ == "__main__":
    main()
