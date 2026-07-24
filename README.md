# Amref Help Desk RAG Chatbot

Production-ready Retrieval-Augmented Generation (RAG) chatbot for **Amref International University** that answers questions exclusively from the official Help Desk Knowledge Base at [helpdesk.amref.ac.ke](https://helpdesk.amref.ac.ke/knowledgebase.php).

## Features

- Full knowledge base crawling and ingestion (articles, categories, metadata)
- Hybrid retrieval: vector search, metadata filtering, MMR diversity, reranking
- Dual embedding stores in ChromaDB (text chunks + image metadata)
- Image retrieval with gallery display (max 3 images per response)
- Source citations with article title, URL, and confidence scores
- Conversation memory with PostgreSQL session history
- JWT authentication, rate limiting, feedback system, analytics logging
- Configurable LLM: OpenAI-compatible provider (Qwen / GPT-4o) or local Ollama models
- Streamlit chat UI with dark mode, categories filter, and search history
- Docker Compose deployment

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Streamlit UI   │────▶│  FastAPI Backend │────▶│ PostgreSQL  │
│  (frontend/)    │     │  (backend/app/)  │     │ (metadata)  │
└─────────────────┘     └────────┬─────────┘     └─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐ ┌─────────┐ ┌──────────┐
              │ ChromaDB│ │ OpenAI/ │ │ Help Desk│
              │ vectors │ │ Ollama  │ │   KB     │
              └─────────┘ └─────────┘ └──────────┘
```

### Module Structure

```
backend/app/
├── api/           # FastAPI routes and dependency injection
├── services/      # Business logic (RAG, history, auth, feedback)
├── rag/           # Embeddings, retrieval, LLM generation
├── ingest/        # Crawler, chunker, image processor, pipeline
├── database/      # SQLAlchemy models, PostgreSQL, ChromaDB
├── models/        # Pydantic request/response schemas
├── prompts/       # LLM prompt templates
├── utils/         # Security, logging, text processing
└── main.py        # Application entry point
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (async) |
| Frontend | Streamlit |
| Database | PostgreSQL |
| Vector DB | ChromaDB |
| LLM Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | OpenAI GPT-4o / Ollama |
| Package Manager | uv |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 16+ (or use Docker Compose)
- OpenAI-compatible API key (Qwen/OpenAI) or Ollama for local inference

## Setup

### 1. Clone and configure

```bash
cd amref-helpdesk-rag
cp .env.example .env
```

If you are using a local Ollama-backed Qwen model, set the following in `.env` instead of the OpenAI-compatible provider fields:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=<your-qwen-model-name>
```

For OpenAI-compatible Qwen endpoints, keep `LLM_PROVIDER=openai` and fill in `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL`.

### 2. Install dependencies

```bash
uv sync
```

### 3. Start PostgreSQL

Using Docker Compose (recommended):

```bash
docker compose up postgres -d
```

Or use a local PostgreSQL instance and update `.env` accordingly.

### 4. Run ingestion

Crawl the knowledge base, chunk articles, generate embeddings, and store in ChromaDB + PostgreSQL:

```bash
uv run python scripts/ingest.py
```

Or via API (requires authentication):

```bash
# Register first
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@amref.ac.ke","password":"securepass123"}'

# Trigger ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"force": false, "include_images": true}'
```

### 5. Add local images (optional)

Place screenshots in `data/images/` and register metadata in `data/images/metadata.json`:

```json
{
  "screenshot.png": {
    "caption": "LMS login screen",
    "alt_text": "Login page",
    "article_id": "2",
    "category": "LMS",
    "keywords": "login,lms"
  }
}
```

Re-run ingestion to index local images.

## Running

### Development (local)

```bash
# Backend
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# Frontend
uv run streamlit run frontend/streamlit_app.py --server.port 8501
```

- API docs: http://localhost:8000/docs
- Chat UI: http://localhost:8501

### Docker Compose (full stack)

```bash
docker compose up --build
```

Services:
- Backend: http://localhost:8000
- Streamlit: http://localhost:8501
- PostgreSQL: localhost:5432

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send a question, get answer + images + sources |
| `POST` | `/api/v1/feedback` | Submit rating for a response |
| `POST` | `/api/v1/ingest` | Trigger knowledge base ingestion (auth required) |
| `GET` | `/api/v1/history/{session_id}` | Get chat history for a session |
| `GET` | `/api/v1/history` | List recent sessions |
| `GET` | `/api/v1/articles` | List ingested articles |
| `GET` | `/api/v1/categories` | List article categories |
| `GET` | `/api/v1/images` | List indexed images |
| `POST` | `/api/v1/auth/register` | Register user |
| `POST` | `/api/v1/auth/login` | Login and get JWT token |
| `GET` | `/health` | Health check |

### Chat request example

```json
{
  "message": "How do I reset my student portal password?",
  "session_id": "optional-session-uuid",
  "category": "Student Portal"
}
```

### Chat response example

```json
{
  "answer": "To reset your student portal password...",
  "images": [{"image_id": "...", "filename": "...", "filepath": "/static/images/...", "score": 0.85}],
  "sources": [{"article_id": "9", "title": "Reset your student portal password", "url": "...", "score": 0.92}],
  "confidence": 0.89,
  "session_id": "...",
  "message_id": "..."
}
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `OPENAI_MODEL` | OpenAI or Qwen model name | `gpt-4o` |
| `OPENAI_API_BASE` | Base URL for OpenAI-compatible / Qwen endpoints | `https://<qwen-openai-compatible-endpoint>` |
| `OPENAI_API_KEY` | Optional API key for local OpenAI-compatible endpoints | `` |
| `OLLAMA_BASE_URL` | Local Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `qwen3:4b` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size (chars) | `500` |
| `TOP_K_RETRIEVAL` | Text chunks retrieved | `5` |
| `TOP_K_IMAGES` | Max images returned | `3` |

### Local Qwen / OpenAI-compatible Ollama

For local Qwen running on an Ollama server, keep `LLM_PROVIDER=openai`, set `OPENAI_API_BASE=http://localhost:11434`, and choose the local model name in `OPENAI_MODEL` (for example `qwen3:4b`). `OPENAI_API_KEY` can remain blank if the local server does not require authentication.

## Testing

```bash
uv run pytest tests/ -v
```

## Ingestion Pipeline

1. **Discover** — Crawl index and all category pages to find every `article=N` link
2. **Download** — Fetch each article HTML
3. **Extract** — Title, category, article ID, URL, clean text, inline images
4. **Chunk** — Split into ~500-token chunks with overlap
5. **Embed** — Generate vectors with `all-MiniLM-L6-v2`
6. **Store** — ChromaDB (embeddings) + PostgreSQL (metadata)
7. **Images** — Download article images + scan `data/images/` folder

## Deployment

1. Set production values in `.env` (strong `SECRET_KEY`, real DB credentials)
2. Use `docker compose up -d` for containerized deployment
3. Run ingestion after deploy: `docker compose exec backend python scripts/ingest.py`
4. Configure reverse proxy (nginx) for HTTPS
5. Set rate limits and monitor logs in `logs/app.log`

## Future Extensions

The modular architecture supports:

- Multilingual support (swap embedding model + prompts)
- OCR for PDF attachments in articles
- Voice input (Whisper integration)
- Additional document sources (SharePoint, Google Drive)
- Cross-encoder reranking for higher precision

## License

Internal use — Amref International University.

