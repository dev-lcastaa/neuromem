from pathlib import Path

import yaml


def test_compose_manages_storage_and_bootstrap() -> None:
    config = yaml.safe_load((Path(__file__).parents[2] / "docker-compose.yml").read_text())
    services = config["services"]
    assert services["opensearch"]["environment"]["discovery.type"] == "single-node"
    assert services["opensearch"]["volumes"] == ["opensearch-data:/usr/share/opensearch/data"]
    api = services["memory_api"]
    assert api["depends_on"]["opensearch"]["condition"] == "service_healthy"
    assert api["depends_on"]["rabbitmq"]["condition"] == "service_healthy"
    assert api["environment"]["OPENSEARCH_URL"] == "http://opensearch:9200"
    assert api["environment"]["BOOTSTRAP_INDEXES"] == "true"
    assert api["ports"] == ["${NEUROMEM_HOST_IP:-192.168.1.208}:8100:8000"]
    mcp = services["mcp"]
    assert mcp["command"] == ["python", "-m", "apps.mcp_server.main"]
    assert mcp["environment"]["MEMORY_API_URL"] == "http://memory_api:8000"
    assert mcp["environment"]["NEUROMEM_HOST_IP"] == "${NEUROMEM_HOST_IP:-192.168.1.208}"
    assert mcp["depends_on"]["memory_api"]["condition"] == "service_healthy"
    assert services["opensearch"]["ports"] == ["${NEUROMEM_HOST_IP:-192.168.1.208}:9200:9200"]
    assert mcp["ports"] == ["${NEUROMEM_HOST_IP:-192.168.1.208}:8001:8001"]
    assert services["dashboard"]["ports"] == ["${NEUROMEM_HOST_IP:-192.168.1.208}:8501:8501"]
    assert config["networks"]["default"]["name"] == "neuromem-net"


def test_compose_runs_durable_internal_rabbitmq_and_worker() -> None:
    config = yaml.safe_load((Path(__file__).parents[2] / "docker-compose.yml").read_text())
    services = config["services"]
    broker = services["rabbitmq"]
    worker = services["ingest_worker"]
    assert broker["image"] == "rabbitmq:3.13-management-alpine"
    assert broker["volumes"] == ["rabbitmq-data:/var/lib/rabbitmq"]
    assert broker.get("ports", []) == []
    assert worker["command"] == ["python", "-m", "apps.ingest_worker.main"]
    assert worker["depends_on"]["rabbitmq"]["condition"] == "service_healthy"
    assert worker["environment"]["RABBITMQ_HOST"] == "rabbitmq"


def test_api_image_includes_bootstrap_mappings_and_package_readme() -> None:
    dockerfile = (Path(__file__).parents[2] / "apps/memory_api/Dockerfile").read_text()
    assert "COPY opensearch/indexes ./opensearch/indexes" in dockerfile
    assert "COPY README.md ./" in dockerfile
