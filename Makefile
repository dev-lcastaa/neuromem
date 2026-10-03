.PHONY: help install lint format typecheck test test-cov test-integration \
		run-api run-dashboard run-mcp health clean

PYTHON := uv run

help:
	@echo "install           - sync dependencies with uv"
	@echo "lint              - ruff check + format --check"
	@echo "format            - ruff format (writes)"
	@echo "typecheck         - mypy on memory, models, apps"
	@echo "test              - pytest unit tests"
	@echo "test-cov          - pytest unit tests with coverage"
	@echo "test-integration  - pytest integration tests (needs OpenSearch)"
	@echo "run-api           - uvicorn memory_api on :8000 with reload"
	@echo "run-dashboard     - streamlit dashboard on :8501"
	@echo "run-mcp           - MCP Streamable HTTP tools on :8001/mcp"
	@echo "health            - curl local health endpoints"
	@echo "clean             - remove caches"

install:
	uv sync --all-groups

lint:
	$(PYTHON) ruff check .
	$(PYTHON) ruff format --check .

format:
	$(PYTHON) ruff format .
	$(PYTHON) ruff check --fix .

typecheck:
	$(PYTHON) mypy memory models apps

test:
	$(PYTHON) pytest tests/unit -q

test-cov:
	$(PYTHON) pytest tests/unit --cov --cov-report=term-missing

test-integration:
	$(PYTHON) pytest -m integration -q

run-api:
	$(PYTHON) uvicorn apps.memory_api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	$(PYTHON) streamlit run apps/dashboard/app.py --server.port 8501

run-mcp:
	$(PYTHON) python -m apps.mcp_server.main

health:
	@curl -fsS http://localhost:8000/health && echo
	@curl -fsS http://localhost:8000/health/opensearch && echo

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
