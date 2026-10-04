from pathlib import Path

import yaml


def test_compose_manages_storage_and_bootstrap() -> None:
    config = yaml.safe_load((Path(__file__).parents[2] / "docker-compose.yml").read_text())
    services = config["services"]
    assert services["opensearch"]["environment"]["discovery.type"] == "single-node"
    assert services["opensearch"]["volumes"] == ["opensearch-data:/usr/share/opensearch/data"]
    api = services["memory_api"]
    assert api["depends_on"]["opensearch"]["condition"] == "service_healthy"
    assert api["environment"]["OPENSEARCH_URL"] == "http://opensearch:9200"
    assert api["environment"]["BOOTSTRAP_INDEXES"] == "true"
    assert api["ports"] == ["127.0.0.1:8100:8000"]
    mcp = services["mcp"]
    assert mcp["command"] == ["python", "-m", "apps.mcp_server.main"]
    assert mcp["environment"]["MEMORY_API_URL"] == "http://memory_api:8000"
    assert mcp["depends_on"]["memory_api"]["condition"] == "service_healthy"
    for service in services.values():
        assert all(port.startswith("127.0.0.1:") for port in service.get("ports", []))


def test_api_image_includes_bootstrap_mappings_and_package_readme() -> None:
    dockerfile = (Path(__file__).parents[2] / "apps/memory_api/Dockerfile").read_text()
    assert "COPY opensearch/indexes ./opensearch/indexes" in dockerfile
    assert "COPY README.md ./" in dockerfile
