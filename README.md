# RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that lets users upload documents and ask questions grounded in their content. Built to showcase end-to-end AI engineering: clean architecture, dual vector backends, streaming UX, observability, and measurable retrieval quality.

## What this project demonstrates (portfolio)

This is a full-stack RAG application, not a toy demo. It shows how to design, ship, and evaluate a document Q&A system the way you would in a real product:

| Area | What you can point to |
|------|------------------------|
| **Architecture** | Hexagonal (ports & adapters) with clear separation between domain, application, and infrastructure |
| **RAG pipeline** | PDF / DOCX / XLSX ingestion → table-aware chunking → embeddings → top-k retrieval → LLM generation |
| **Vector search** | **PostgreSQL + pgvector** as the primary store, plus **Neo4j** as a switchable alternative — selectable from the UI without re-ingesting |
| **LLM integration** | Provider-agnostic layer via **LiteLLM** (Ollama locally; easy to swap models) |
| **API & UX** | FastAPI backend, static chat UI, **SSE streaming**, document management, citations |
| **Safety & quality** | Input guardrails, relevance threshold (`RAG_MIN_SCORE`), DB-backed prompts, MIME validation |
| **Observability** | Langfuse tracing hooks for chat and retrieval flows |
| **Evaluation** | BioASQ retrieval benchmarks (hit@k, recall@k, MRR) and optional RAGAS / DeepEval judges |
| **DevOps** | **UV** for Python, **Docker Compose** full stack, **Kubernetes** manifests, CI, Makefile shortcuts |

In short: upload corporate docs, ask questions in natural language, get answers backed by retrieved chunks — with the plumbing (architecture, infra, evals) visible and intentional.

## Architecture

```
                    ┌─────────────────────────┐
                    │  Static UI + FastAPI    │
                    │  /chat/stream (SSE)     │
                    │  /documents             │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                                           ▼
   ChatService + Guardrails                 IngestionService + MIME
          │                                           │
          │                              parse → chunk → embed
          │                                           │
   retrieve top-k  ◄────────────────────── RoutedVectorStore
          │
          ▼
       LLMPort.generate / generate_stream
```

| Layer | Responsibility |
|-------|----------------|
| `domain/` | Entities, exceptions, and ports |
| `application/` | `ChatService`, `IngestionService`, guardrails, chunker |
| `infrastructure/` | Parsers, embeddings, Postgres/pgvector, Neo4j, FastAPI, static UI |
| `core/` | Lazy singletons: `Env`, `AsyncHttpClient` |

### Features

- UI at `/` (chat) and `/documents` (upload / list / delete).
- Chat over **SSE** (`POST /api/v1/chat/stream`); current conversation kept in `sessionStorage`.
- Retrieval backend selector in the UI: **PostgreSQL** or **Neo4j**.
- Guardrails: toxicity patterns + `RAG_MIN_SCORE` threshold + scope prompts stored in the database.
- Ingestion: **MIME**, extension, and magic-byte validation.
- Markdown prompts in the `prompts` table: `system` (`{context}`) and `user_message` (`{question}`).

### Table-aware chunking

1. Each table is serialized to Markdown with headers.
2. In the text it is replaced by an atomic token `t{i}` (`t0`, `t1`, …).
3. Text is split **without breaking** those tokens.
4. Full Markdown tables are restored in each chunk.

### Design patterns

- **Singleton**: `Env` (lazy), `AsyncHttpClient`, `AppContainer`
- **Factory**: `LLMFactory`, `DocumentParserFactory`
- **Ports & Adapters**: hexagonal architecture

## Requirements

- Python ≥ 3.11
- [UV](https://docs.astral.sh/uv/)
- PostgreSQL with **pgvector** extension
- Neo4j 5.x or compatible with vector index
- Ollama (if using `ollama/...` models), e.g. chat + `nomic-embed-text`
- Docker / kubectl (optional)

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL | `postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot` |
| `EMBEDDING_DIMENSION` | Vector dimension | `768` |
| `VECTOR_BACKEND` | Default retrieval backend | `postgres` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j credentials | `neo4j` / `password` |
| `NEO4J_VECTOR_INDEX` | Vector index name | `chunk_embeddings` |
| `LLM_PROVIDER` | LLM provider | `litellm`, `ollama`, or `mock` |
| `LITELLM_MODEL` | Chat model | `ollama/qwen2.5:3b` |
| `LITELLM_EMBEDDING_MODEL` | Embedding model | `ollama/nomic-embed-text` |
| `LITELLM_API_BASE` | Base URL | `http://localhost:11434` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk size | `800` / `100` |
| `RAG_TOP_K` | Chunks in the prompt | `4` |
| `RAG_MIN_SCORE` | Document relevance threshold | `0.25` |

## Local development

Makefile shortcuts:

```bash
make setup      # .env + uv sync
make models     # ollama pull (host)
make up         # Postgres in Docker + local uvicorn
make up-docker  # full stack in Docker
make test
make help
```

Manual equivalent (Ollama/app on host; Postgres in Docker):

```bash
docker compose --profile db up -d postgres
ollama pull qwen3:4b
ollama pull nomic-embed-text
uv sync --extra dev
cp .env.example .env
uv run uvicorn chatbot.main:app --reload --host 0.0.0.0 --port 8000
```

Open the UI at [http://localhost:8000/](http://localhost:8000/).

Documents are always indexed in **PostgreSQL**; when `NEO4J_ENABLED=true`, they are also replicated to **Neo4j** so you can switch retrieval backends from the chat UI without re-ingesting.

### Quick API reference

```bash
# Health
curl http://localhost:8000/api/v1/health

# Ingest document (correct MIME required)
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@./my-document.pdf;type=application/pdf"

# List documents
curl http://localhost:8000/api/v1/documents

# Classic chat (full JSON response)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the document", "retrieval_backend": "postgres"}'

# Streaming chat (SSE)
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "Summarize the document", "retrieval_backend": "neo4j"}'
```

Supported formats: `.pdf`, `.docx`, `.xlsx` / `.xlsm` (with matching MIME type).

### Tests and quality

```bash
uv run pytest -q
uv run pylint src/chatbot
```

### BioASQ evaluation (retrieval)

Dataset: [rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq). Details in [`evals/README.md`](evals/README.md).

```bash
make eval-bioasq                              # retrieval only (hit@k / recall@k / mrr)
LIMIT=50 make eval-bioasq
EVAL_ARGS='--generate --ragas' make eval-bioasq   # later: answers + LLM judges
```

Output: `evals/results/bioasq/retrieval_metrics.json`.

## Docker Compose

Ollama uses **GPU by default** (`gpus: all`). If Docker fails with:

`failed to discover GPU vendor from CDI: no known GPU vendor found`

the host GPU works (`nvidia-smi`), but the **NVIDIA Container Toolkit** is missing from the Docker daemon. Install it once:

```bash
make setup-gpu
```

Then start the stack:

```bash
make up-docker
# or: docker compose --profile docker up --build
```

This starts `postgres`, `neo4j`, `ollama`, `ollama-init`, and `chatbot`.

Change models with:

```bash
LITELLM_MODEL=ollama/llama3.2:3b OLLAMA_MODEL=llama3.2:3b make up-docker
```

## Kubernetes

You need a cluster. Locally, `make up-k8s` creates one with **kind** if it does not exist:

```bash
make up-k8s      # kind + build image + apply
make k8s-status
make k8s-pf      # port-forward → http://localhost:8000
make down-k8s
make k8s-cluster-delete
```

If you see a `localhost:8080` error, there is no kubeconfig/cluster: run `make k8s-cluster` first (or `make up-k8s`).

Manual:

```bash
# kind create cluster --name rag-chatbot   # first time only
docker build -t rag-chatbot:0.1.0 .
kind load docker-image rag-chatbot:0.1.0 --name rag-chatbot
bash k8s/apply.sh
kubectl -n rag-chatbot port-forward svc/chatbot 8000:8000
```

## Project structure

```
src/chatbot/
  domain/                 # entities, documents, ports
  application/services/   # ChatService, IngestionService, guardrails, chunker
  infrastructure/
    adapters/
      api/                # FastAPI + static UI + MIME validation
      ingestion/          # PDF / DOCX / XLSX parsers
      llm/                # LiteLLM chat + embeddings (+ stream)
      persistence/        # Postgres, Neo4j, and backend router
    config/
    container.py
```

## Tech stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy (async), LiteLLM
- **Vector stores**: PostgreSQL + pgvector, Neo4j
- **LLM / embeddings**: Ollama (local), swappable via LiteLLM
- **Frontend**: Vanilla HTML/CSS/JS (static, served by FastAPI)
- **Tooling**: UV, pytest, pylint, Docker, Kubernetes (kind), GitHub Actions CI
- **Observability**: Langfuse
- **Evals**: BioASQ, RAGAS, DeepEval

## License

See `LICENSE`.
