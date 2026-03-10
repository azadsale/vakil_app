.PHONY: up down build logs migrate migrate-new shell-backend shell-db test lint clean

# =============================================================================
# Legal-CoPilot — Makefile
# =============================================================================

DOCKER_COMPOSE = docker compose
BACKEND_SERVICE = backend
DB_SERVICE     = postgres

# ---------------------------------------------------------------------------
# Stack Lifecycle
# ---------------------------------------------------------------------------

up:
	@echo "▶  Starting Legal-CoPilot stack..."
	$(DOCKER_COMPOSE) up -d --build
	@echo "✅ Stack running. Backend: http://localhost:8000 | Frontend: http://localhost:3000 | MinIO: http://localhost:9001"

down:
	$(DOCKER_COMPOSE) down

down-volumes:
	@echo "⚠  This will DELETE all data volumes."
	$(DOCKER_COMPOSE) down -v

build:
	$(DOCKER_COMPOSE) build --no-cache

logs:
	$(DOCKER_COMPOSE) logs -f

logs-backend:
	$(DOCKER_COMPOSE) logs -f $(BACKEND_SERVICE)

# ---------------------------------------------------------------------------
# Database / Alembic
# ---------------------------------------------------------------------------

migrate:
	@echo "▶  Running Alembic migrations..."
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) alembic upgrade head

migrate-new:
	@echo "Usage: make migrate-new MSG='your migration message'"
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) alembic revision --autogenerate -m "$(MSG)"

migrate-history:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) alembic history --verbose

migrate-downgrade:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) alembic downgrade -1

# ---------------------------------------------------------------------------
# Shell Access
# ---------------------------------------------------------------------------

shell-backend:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) /bin/bash

shell-db:
	$(DOCKER_COMPOSE) exec $(DB_SERVICE) psql -U vakil -d vakildb

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) pytest --cov=app --cov-report=term-missing --cov-fail-under=80 -v

test-fast:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) pytest -x -q

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) ruff check app/
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) mypy app/

format:
	$(DOCKER_COMPOSE) exec $(BACKEND_SERVICE) ruff format app/

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "Legal-CoPilot — Available Make Targets"
	@echo "======================================="
	@echo "  up                 Start all services (builds if needed)"
	@echo "  down               Stop all services"
	@echo "  down-volumes       Stop and DELETE all data volumes"
	@echo "  build              Force rebuild all images"
	@echo "  logs               Tail all service logs"
	@echo "  logs-backend       Tail backend logs only"
	@echo "  migrate            Run pending Alembic migrations"
	@echo "  migrate-new MSG='' Generate new Alembic revision"
	@echo "  migrate-history    Show migration history"
	@echo "  migrate-downgrade  Downgrade one revision"
	@echo "  shell-backend      bash into backend container"
	@echo "  shell-db           psql into postgres container"
	@echo "  test               Run pytest with coverage (>80% enforced)"
	@echo "  test-fast          Run pytest, stop on first failure"
	@echo "  lint               ruff + mypy"
	@echo "  format             ruff format"
	@echo "  clean              Remove __pycache__, .pyc, cache dirs"
	@echo ""