# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
COPY uv.lock* ./

RUN if [ -f uv.lock ]; then uv sync --frozen --extra eval --no-dev; else uv sync --extra eval --no-dev; fi

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    LOG_JSON=true \
    HOST=0.0.0.0 \
    PORT=8000

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
