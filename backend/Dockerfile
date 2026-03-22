# =============================================================================
# Legal-CoPilot Backend — Multi-stage Dockerfile
# Base: Python 3.11-slim
# Stages: base → development | production
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: base — shared system deps + Python env
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for psycopg2 async driver, cryptography, pillow (OCR),
# and ffmpeg (audio format conversion for Sarvam AI — webm → wav)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-mar \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv for fast dependency resolution
RUN pip install --no-cache-dir --upgrade pip uv

# Copy dependency manifests first (layer cache)
COPY pyproject.toml ./

# ---------------------------------------------------------------------------
# Stage 2: development — includes dev/test tools, hot-reload via uvicorn
# ---------------------------------------------------------------------------
FROM base AS development

# Install all deps including dev group
RUN uv pip install --system --no-cache -e ".[dev]"

# Source code is bind-mounted in docker-compose for hot-reload
# COPY is intentionally omitted here

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]

# ---------------------------------------------------------------------------
# Stage 3: production — minimal image, no dev tools
# ---------------------------------------------------------------------------
FROM base AS production

# Install only production deps
RUN uv pip install --system --no-cache .

# Copy application source
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]