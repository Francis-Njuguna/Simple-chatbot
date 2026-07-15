# =============================================================================
# Frontend — Railway Dockerfile
#
# Minimal image: only Streamlit + httpx.
# Railway injects $PORT at runtime; CMD uses shell form so the variable
# is expanded correctly.
#
# The embeddable chat widget (frontend/widget/*) is copied into ./static and
# served by Streamlit's static file serving at:
#   https://<frontend-domain>/app/static/chat-widget.js
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip

# Only what Streamlit needs
COPY requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Copy only the frontend source
COPY frontend/streamlit_app.py ./streamlit_app.py

# Embeddable chat widget — served via Streamlit static serving
# (/app/static/chat-widget.js on the deployed domain)
COPY frontend/widget ./static

EXPOSE 8501

# Shell form — Railway's $PORT is expanded at container start
CMD streamlit run streamlit_app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableStaticServing true
