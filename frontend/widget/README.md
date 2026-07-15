# Embeddable Chat Widget

A standalone, dependency-free floating chat widget for the Amref Help Desk
RAG backend. One JS file (`chat-widget.js`) + one small CSS file
(`chat-widget.css`, auto-loaded by the JS).

## Embed in your PHP site

Host `chat-widget.js` and `chat-widget.css` **side by side** anywhere
(same web server as your PHP site is fine), then add a single tag to your
template (e.g. `footer.php`):

```html
<script
  src="https://your-php-site.com/assets/chat-widget.js"
  data-api-base="https://your-fastapi-backend.com/api/v1"
  defer></script>
```

That's it — a black floating button appears bottom-right; clicking it slides
open a compact chat panel.

### Alternative: load the widget from the deployed frontend service

The Docker images (root `Dockerfile` frontend stage, `Dockerfile.frontend`,
and `frontend/Dockerfile`) copy this folder into Streamlit's `static/`
directory and run with `--server.enableStaticServing true`, so once the
frontend service is deployed (Railway/docker-compose) the widget is served
at:

```
https://<frontend-domain>/app/static/chat-widget.js
```

So on Railway you can embed without hosting the files yourself:

```html
<script
  src="https://your-frontend.up.railway.app/app/static/chat-widget.js"
  data-api-base="https://your-backend.up.railway.app/api/v1"
  defer></script>
```

### Script tag options

| Attribute          | Default                          | Purpose                                              |
| ------------------ | -------------------------------- | ---------------------------------------------------- |
| `data-api-base`    | `http://localhost:8000/api/v1`   | Base URL of the FastAPI API                          |
| `data-backend-url` | derived from `data-api-base`     | Backend root used to resolve `/static` image paths   |
| `data-title`       | `Amref Help Desk Assistant`      | Panel header title                                   |
| `data-css`         | `chat-widget.css` next to the JS | Custom stylesheet URL, or `false` to skip injection  |

## Backend endpoints used

- `POST /api/v1/chat` — `{ message, session_id?, category? }`; the returned
  `session_id` is stored in `sessionStorage` and echoed on subsequent
  requests (same conversation logic as the Streamlit frontend).
- `GET /api/v1/categories` — populates the category filter dropdown.
- `POST /api/v1/feedback` — `{ message_id, rating }` from the 1–5 star
  rating under each assistant answer.

## CORS

The FastAPI backend already ships with `CORSMiddleware` configured with
`allow_origins=["*"]` (see `backend/app/main.py`), so requests from your
PHP site's domain are permitted with no backend change. If you later want
to lock it down, replace `["*"]` with your exact origin, e.g.
`["https://your-php-site.com"]`.

## Local test

```bash
# terminal 1 — backend
uvicorn backend.app.main:app --reload

# terminal 2 — static server for the widget
cd frontend/widget && python -m http.server 8080
```

Open http://localhost:8080/example.html.
