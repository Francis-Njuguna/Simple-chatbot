# =============================================================================
# Amref Help Desk RAG — multi-stage Dockerfile
#
# Stages
#   base      : slim Python image + OS deps shared by both services
#   backend   : installs backend Python deps + copies backend/scripts/data
#   frontend  : installs ONLY streamlit + httpx, copies frontend/
# =============================================================================

# -----------------------------------------------------------------------------
# base — OS-level build tools only (no Python packages yet)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libpq-dev  : asyncpg / psycopg native build
# build-essential : compilers for hnswlib, tokenizers, grpcio, etc.
# curl       : healthcheck in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# -----------------------------------------------------------------------------
# backend — all server-side dependencies
# -----------------------------------------------------------------------------
FROM base AS backend

# Install backend Python deps first (layer-cached unless requirements change)
COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy source
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY scripts ./scripts
COPY data    ./data

# Install the backend package in editable mode so that
# `import backend` works from any working directory (fixes
# ModuleNotFoundError: No module named 'backend' in scripts/)
RUN pip install --no-cache-dir -e .

RUN mkdir -p logs data/chroma backend/app/static/images

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# frontend — only streamlit + httpx (tiny image, fast build)
# -----------------------------------------------------------------------------
FROM base AS frontend

# Install ONLY what the Streamlit app needs
COPY requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY frontend ./frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
