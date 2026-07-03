# =============================================================================
# Backend — Railway Dockerfile
#
# Build context is the REPO ROOT (Railway sets root dir to backend/ but
# builds from repo root when rootDirectory is set, so all COPY paths are
# relative to the repo root).
#
# Railway injects $PORT at runtime; CMD uses shell form so the variable
# is expanded correctly.
# Ollama is NOT available on Railway — use EMBEDDING_PROVIDER=sentence-transformers
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Python deps — copied from repo root
COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

# Application source
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY scripts ./scripts
COPY data    ./data

# Install package in editable mode so `import backend` resolves everywhere
RUN pip install --no-cache-dir -e .

RUN mkdir -p logs data/chroma backend/app/static/images

EXPOSE 8000

# Shell form — Railway's $PORT is expanded at container start
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
