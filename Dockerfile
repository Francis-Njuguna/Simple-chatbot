FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p logs data/chroma backend/app/static/images

# ─────────────────────────────────────────────
# Backend stage
# ─────────────────────────────────────────────
FROM base AS backend

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────
# Frontend stage
# ─────────────────────────────────────────────
FROM base AS frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
