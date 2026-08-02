# Evaluación de retrieval (BioASQ) y respuestas del agente

Flujo para descargar [rag-datasets/rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq), indexar pasajes en memoria y medir **solo retrieval** frente a los pasajes gold.

Por defecto **no** genera respuestas ni lanza jueces LLM (RAGAS / DeepEval van con `--generate`).

Se ejecuta en **Docker** (`Dockerfile.eval` + perfil Compose `eval`).

## Métricas (default)

| Métrica | Qué mide |
|---------|----------|
| `hit_at_k` | ¿Hay al menos un pasaje gold en el top-k? |
| `recall_at_k` | Cobertura de pasajes gold recuperados |
| `mrr` | Reciprocal rank del primer pasaje gold |

## Jueces LLM (opcional)

| Flag | Qué hace |
|------|----------|
| `--generate` | Genera respuestas con el LLM de chat |
| `--ragas` | Juez RAGAS vía LangChain-Ollama (requiere `--generate`) |
| `--deepeval` | Juez **DeepEval** vía **LiteLLM/Ollama** (requiere `--generate`) |

> DeepEval y RAGAS pueden combinarse en la UI marcando ambos checkboxes.

## Smoke test DeepEval (solo plantilla)

Prueba rápida con **2 preguntas** del template, sin BioASQ ni Postgres:

```bash
# Host (Ollama en localhost:11434)
make eval-sync
make eval-deepeval-template

# Docker (Ollama del perfil eval)
make eval-deepeval-template-docker
```

Equivalente manual:

```bash
python -m evals --template --limit 2 --distractors 0 --generate --deepeval
```

Salida en `evals/results/template/` (`deepeval_metrics.json`, `rag_outputs.json`, etc.).

Otro JSON propio:

```bash
python -m evals --json-dataset mi-dataset.json --generate --deepeval --distractors 0
```

## Uso (Docker)

```bash
# Solo retrieval (rápido)
make eval-bioasq

LIMIT=50 DISTRACTORS=100 make eval-bioasq

# Respuestas + DeepEval
EVAL_ARGS='--generate --deepeval' make eval-bioasq
EVAL_ARGS='--generate --ragas --deepeval' make eval-bioasq
```

Imagen: `rag-chatbot-eval:0.1.0`. Resultados: `evals/results/bioasq/`.

## Dataset JSON personalizado

Plantilla incluida en `evals/templates/dataset.template.json` (también en `/static/dataset.template.json`).

```json
{
  "name": "Mi dataset",
  "passages": [{"id": "doc-1", "text": "..."}],
  "samples": [{
    "id": "q1",
    "question": "...",
    "answer": "...",
    "relevant_passage_ids": ["doc-1"]
  }]
}
```

Importación:

- **UI** (`/evals`): subir JSON o descargar plantilla.
- **API**: `POST /api/v1/evals/datasets/json/import` (multipart).
- **Plantilla**: `GET /api/v1/evals/datasets/template`.

Los IDs de pasajes/muestras en JSON son strings; al importar se mapean a enteros internos.

## UI y persistencia en PostgreSQL

La app expone **`/evals`** con:

1. **Datasets** — BioASQ (Hugging Face) o JSON custom en Postgres.
2. **Suites** — conjuntos de muestras con flags RAGAS / DeepEval y modelo LLM opcional.
3. **Ejecuciones** — evaluaciones en segundo plano con métricas de retrieval y jueces.
4. **A/B testing** — lanza dos variantes (p. ej. distinto `llm_model`) y compara runs lado a lado.

API REST bajo `/api/v1/evals/*`:

- `GET /evals/datasets`, `GET /evals/datasets/template`
- `POST /evals/datasets/json/import`, BioASQ import
- CRUD suites / runs
- `POST /evals/ab-test`, `GET /evals/compare?run_a=…&run_b=…`

## CLI con Postgres

```bash
python -m evals --import-db --limit 1
python -m evals --use-db --limit 20 --generate --deepeval
```

## Notas

- DeepEval usa `OllamaModel` nativo con `think=false` (mejor JSON que LiteLLM con qwen3 local).
- Override del juez: `DEEPEVAL_JUDGE_MODEL=qwen3:4b`.
- La caché va a `DEEPEVAL_CACHE_FOLDER` (por defecto `~/.cache/deepeval`), no a `.deepeval` en el cwd.
- En Docker eval, embeddings y juez apuntan a `http://ollama:11434`.
