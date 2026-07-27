# RAG Chatbot

Chatbot con **arquitectura hexagonal**, **LiteLLM**, ingestión RAG de **PDF / DOCX / XLSX**, UI estática, **SSE**, gestión con **UV**, **Docker** y **Kubernetes**.

## Arquitectura

```
                    ┌─────────────────────────┐
                    │  UI estática + FastAPI  │
                    │  /chat/stream (SSE)     │
                    │  /documents             │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                                           ▼
   ChatService + Guardrails                 IngestionService + MIME
          │                                           │
          │                              parse → chunk(ti) → embed
          │                                           │
   retrieve top-k  ◄────────────────────── PostgresVectorStore (pgvector)
          │
          ▼
       LLMPort.generate / generate_stream
```

| Capa | Responsabilidad |
|------|-----------------|
| `domain/` | Entidades, excepciones y puertos |
| `application/` | `ChatService`, `IngestionService`, guardarraíles, chunker |
| `infrastructure/` | Parsers, embeddings, Postgres/pgvector, FastAPI, UI estática |
| `core/` | Singletons `Env` (lazy) y `AsyncHttpClient` |

### Presentación

- UI en `/` (chat) y `/documents` (subir / listar / borrar).
- Chat por **SSE** (`POST /api/v1/chat/stream`): solo la conversación actual en `sessionStorage`.
- Guardarraíles: toxicidad (patrones) + umbral `RAG_MIN_SCORE` + prompts de alcance en BD.
- Ingestión: validación de **MIME**, extensión y magic bytes.
- Prompts markdown en tabla `prompts`: `system` (`{context}`) y `user_message` (`{question}`).

### Tablas protegidas en el chunking

1. Cada tabla se serializa a Markdown con cabeceras.
2. En el texto se sustituye por un token atómico `t{i}` (`t0`, `t1`, …).
3. Se hace el split del texto **sin partir** esos tokens.
4. Se restauran las tablas Markdown completas en cada chunk.

### Patrones

- **Singleton**: `Env` (lazy), `AsyncHttpClient`, `AppContainer`
- **Factory**: `LLMFactory`, `DocumentParserFactory`
- **Ports & Adapters**: hexagonal

## Requisitos

- Python ≥ 3.11
- [UV](https://docs.astral.sh/uv/)
- PostgreSQL con extensión **pgvector**
- Ollama (si usas modelos `ollama/...`), p.ej. chat + `nomic-embed-text`
- Docker / kubectl (opcional)

## Configuración

Copia `.env.example` a `.env` y ajusta:

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL async | `postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot` |
| `EMBEDDING_DIMENSION` | Dimensión de vectores | `768` |
| `LLM_PROVIDER` | Proveedor LLM | `litellm`, `ollama` o `mock` |
| `LITELLM_MODEL` | Modelo de chat | `ollama/qwen2.5:3b` |
| `LITELLM_EMBEDDING_MODEL` | Modelo de embeddings | `ollama/nomic-embed-text` |
| `LITELLM_API_BASE` | Base URL | `http://localhost:11434` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamaño de chunks | `800` / `100` |
| `RAG_TOP_K` | Fragmentos en el prompt | `4` |
| `RAG_MIN_SCORE` | Umbral de alcance documental | `0.25` |

## Desarrollo local

```bash
uv sync --extra dev
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
cp .env.example .env
uv run uvicorn chatbot.main:app --reload --host 0.0.0.0 --port 8000
```

Abre la UI en [http://localhost:8000/](http://localhost:8000/).

### API rápida

```bash
# Health
curl http://localhost:8000/api/v1/health

# Ingestar documento (MIME correcto obligatorio)
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@./mi-documento.pdf;type=application/pdf"

# Listar documentos
curl http://localhost:8000/api/v1/documents

# Chat clásico (JSON completo)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Resume el documento\"}"

# Chat streaming (SSE)
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "{\"message\": \"Resume el documento\"}"
```

Formatos soportados: `.pdf`, `.docx`, `.xlsx` / `.xlsm` (con MIME coincidente).

### Tests y calidad

```bash
uv run pytest -q
uv run pylint src/chatbot
```

## Docker Compose

```bash
docker compose up --build
```

Cambia modelos con:

```bash
LITELLM_MODEL=ollama/llama3.2:3b OLLAMA_MODEL=llama3.2:3b docker compose up --build
```

## Kubernetes

1. Construye e importa la imagen:

```bash
docker build -t rag-chatbot:0.1.0 .
```

2. Aplica manifiestos (`bash k8s/apply.sh` o `kubectl apply -f k8s/...`).

3. Port-forward:

```bash
kubectl -n rag-chatbot port-forward svc/chatbot 8000:8000
```

## Estructura del proyecto

```
src/chatbot/
  domain/                 # entidades, documents, ports
  application/services/   # ChatService, IngestionService, guardrails, chunker
  infrastructure/
    adapters/
      api/                # FastAPI + UI estática + MIME
      ingestion/          # PDF / DOCX / XLSX parsers
      llm/                # LiteLLM chat + embeddings (+ stream)
      persistence/        # conversaciones + vector store memoria
    config/
    container.py
```

## Licencia

Ver `LICENSE`.
