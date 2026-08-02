# RAG Chatbot — atajos de desarrollo y despliegue
# Uso: make help

.DEFAULT_GOAL := help

COMPOSE   ?= docker compose
UV        ?= uv
KUBECTL   ?= kubectl
KIND      ?= $(shell command -v kind 2>/dev/null || echo $(HOME)/.local/bin/kind)
PORT      ?= 8000
HOST      ?= 0.0.0.0
K8S_NS    ?= rag-chatbot
IMAGE     ?= rag-chatbot:0.1.0
KIND_CLUSTER ?= rag-chatbot
PROFILE_DB     := --profile db
PROFILE_DOCKER := --profile docker

.PHONY: help
help: ## Muestra esta ayuda
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)
	@printf "\nEjemplos:\n  make setup && make up\n  make up-docker\n  make k8s-cluster && make up-k8s\n  make test\n\n"

# ── Setup ───────────────────────────────────────────────────────────────────

.PHONY: setup
setup: env sync ## Prepara .env y dependencias Python
	@echo "Listo. Siguiente: make up   (o make up-docker)"

.PHONY: env
env: ## Crea .env desde .env.example si no existe
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Creado .env desde .env.example"; \
	else \
		echo ".env ya existe"; \
	fi

.PHONY: sync
sync: ## Instala dependencias con uv (incl. dev)
	$(UV) sync --extra dev

.PHONY: models
models: ## Descarga modelos Ollama en el host
	ollama pull $${OLLAMA_MODEL:-qwen3:4b}
	ollama pull $${OLLAMA_EMBED_MODEL:-nomic-embed-text}

# ── Levantar ────────────────────────────────────────────────────────────────

.PHONY: up
up: env db ## Levanta Postgres + Neo4j (Docker) + app local
	@echo "App en http://localhost:$(PORT)/"
	$(UV) run uvicorn chatbot.main:app --reload --host $(HOST) --port $(PORT)

.PHONY: up-docker
up-docker: env ## Stack completo en Docker (postgres + ollama + app)
	$(COMPOSE) $(PROFILE_DOCKER) up --build -d
	@echo "App en http://localhost:$(PORT)/"
	@$(COMPOSE) $(PROFILE_DOCKER) ps

.PHONY: up-docker-fg
up-docker-fg: env ## Stack Docker en primer plano (logs)
	$(COMPOSE) $(PROFILE_DOCKER) up --build

.PHONY: db
db: ## Solo bases de datos en Docker (Postgres + Neo4j)
	$(COMPOSE) $(PROFILE_DB) up -d postgres neo4j
	@echo "Esperando a Postgres…"
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
		$(COMPOSE) $(PROFILE_DB) exec -T postgres pg_isready -U chatbot -d chatbot >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@$(COMPOSE) $(PROFILE_DB) exec -T postgres \
		psql -U chatbot -d chatbot -c 'CREATE EXTENSION IF NOT EXISTS vector;' >/dev/null
	@echo "Postgres listo en localhost:5432"
	@echo "Esperando a Neo4j…"
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		$(COMPOSE) $(PROFILE_DB) exec -T neo4j \
			cypher-shell -u neo4j -p password 'RETURN 1;' >/dev/null 2>&1 && break; \
		sleep 2; \
	done
	@echo "Neo4j listo en localhost:7687 (Browser: http://localhost:7474)"

.PHONY: neo4j
neo4j: ## Solo Neo4j en Docker
	$(COMPOSE) $(PROFILE_DB) up -d neo4j
	@echo "Esperando a Neo4j…"
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		$(COMPOSE) $(PROFILE_DB) exec -T neo4j \
			cypher-shell -u neo4j -p password 'RETURN 1;' >/dev/null 2>&1 && break; \
		sleep 2; \
	done
	@echo "Neo4j listo en localhost:7687 (Browser: http://localhost:7474)"

.PHONY: down
down: ## Para perfiles db y docker
	-$(COMPOSE) $(PROFILE_DOCKER) down
	-$(COMPOSE) $(PROFILE_DB) down

.PHONY: down-volumes
down-volumes: ## Para todo y borra volúmenes (datos)
	-$(COMPOSE) $(PROFILE_DOCKER) down -v
	-$(COMPOSE) $(PROFILE_DB) down -v

.PHONY: restart
restart: ## Reinicia el stack Docker completo
	$(COMPOSE) $(PROFILE_DOCKER) up -d --build --force-recreate
	@$(COMPOSE) $(PROFILE_DOCKER) ps

.PHONY: logs
logs: ## Logs del stack Docker (follow)
	$(COMPOSE) $(PROFILE_DOCKER) logs -f --tail=100

.PHONY: ps
ps: ## Estado de contenedores
	@$(COMPOSE) $(PROFILE_DOCKER) ps -a
	@$(COMPOSE) $(PROFILE_DB) ps -a

.PHONY: rebuild
rebuild: ## Rebuild forzado de la imagen chatbot
	$(COMPOSE) $(PROFILE_DOCKER) build --no-cache chatbot
	$(COMPOSE) $(PROFILE_DOCKER) up -d chatbot

# ── Kubernetes ──────────────────────────────────────────────────────────────

.PHONY: k8s-cluster
k8s-cluster: ## Crea cluster local kind (si no existe)
	@command -v $(KIND) >/dev/null 2>&1 || { \
		echo "Instalando kind en ~/.local/bin …"; \
		mkdir -p "$(HOME)/.local/bin"; \
		curl -fsSL "https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64" \
			-o "$(HOME)/.local/bin/kind"; \
		chmod +x "$(HOME)/.local/bin/kind"; \
	}
	@if $(KIND) get clusters 2>/dev/null | grep -qx "$(KIND_CLUSTER)"; then \
		echo "Cluster kind '$(KIND_CLUSTER)' ya existe"; \
	else \
		$(KIND) create cluster --name "$(KIND_CLUSTER)"; \
	fi
	@$(KUBECTL) cluster-info

.PHONY: k8s-image
k8s-image: ## Construye la imagen y la carga en kind
	docker build -t $(IMAGE) .
	@if $(KIND) get clusters 2>/dev/null | grep -qx "$(KIND_CLUSTER)"; then \
		$(KIND) load docker-image $(IMAGE) --name "$(KIND_CLUSTER)"; \
	else \
		echo "Aviso: no hay cluster kind '$(KIND_CLUSTER)'; imagen solo en Docker local"; \
	fi

.PHONY: k8s-apply
k8s-apply: ## Aplica manifiestos (namespace, postgres, neo4j, ollama, chatbot, ingress)
	bash k8s/apply.sh

.PHONY: up-k8s
up-k8s: k8s-cluster k8s-image k8s-apply ## kind + imagen + apply
	@echo "Namespace: $(K8S_NS)"
	@$(KUBECTL) -n $(K8S_NS) get pods,svc,ingress,jobs
	@echo "Siguiente: make k8s-pf"

.PHONY: k8s-status
k8s-status: ## Estado de pods/svc/ingress/jobs/pvc
	$(KUBECTL) -n $(K8S_NS) get pods,svc,ingress,jobs,pvc

.PHONY: k8s-logs
k8s-logs: ## Logs del deployment chatbot
	$(KUBECTL) -n $(K8S_NS) logs -f deploy/chatbot --tail=100

.PHONY: k8s-pf
k8s-pf: ## Espera chatbot Ready y hace port-forward → localhost:$(PORT)
	@echo "Esperando a que chatbot esté Ready…"
	$(KUBECTL) -n $(K8S_NS) wait --for=condition=available --timeout=300s deploy/chatbot
	@echo "App en http://localhost:$(PORT)/"
	$(KUBECTL) -n $(K8S_NS) port-forward svc/chatbot $(PORT):8000

.PHONY: k8s-wait
k8s-wait: ## Espera Postgres + Neo4j + chatbot Ready
	$(KUBECTL) -n $(K8S_NS) wait --for=condition=available --timeout=180s deploy/postgres
	$(KUBECTL) -n $(K8S_NS) wait --for=condition=available --timeout=300s deploy/neo4j
	$(KUBECTL) -n $(K8S_NS) wait --for=condition=available --timeout=300s deploy/chatbot
	@echo "Listo. Siguiente: make k8s-pf"

.PHONY: k8s-ollama-logs
k8s-ollama-logs: ## Logs de Ollama / job de pull
	-$(KUBECTL) -n $(K8S_NS) logs -f deploy/ollama --tail=80
	-$(KUBECTL) -n $(K8S_NS) logs -l app.kubernetes.io/component=ollama-init --tail=80

.PHONY: k8s-neo4j-logs
k8s-neo4j-logs: ## Logs del deployment Neo4j
	$(KUBECTL) -n $(K8S_NS) logs -f deploy/neo4j --tail=100

.PHONY: k8s-neo4j-pf
k8s-neo4j-pf: ## Port-forward Neo4j Browser/Bolt
	@echo "Neo4j Browser en http://localhost:7474/"
	$(KUBECTL) -n $(K8S_NS) port-forward svc/neo4j 7474:7474 7687:7687

.PHONY: down-k8s
down-k8s: ## Borra namespace y (opcional) el cluster kind
	-$(KUBECTL) delete namespace $(K8S_NS) --ignore-not-found
	@echo "Namespace eliminado. Para borrar el cluster: make k8s-cluster-delete"

.PHONY: k8s-cluster-delete
k8s-cluster-delete: ## Elimina el cluster kind local
	-$(KIND) delete cluster --name "$(KIND_CLUSTER)"

# ── Calidad ─────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Ejecuta pytest
	$(UV) run pytest -q

.PHONY: eval-sync
eval-sync: ## Instala extras de evaluación en el host (opcional; make eval-bioasq usa Docker)
	$(UV) sync --extra eval --extra dev

.PHONY: eval-bioasq
eval-bioasq: ## Evalúa retrieval BioASQ en Docker (LIMIT=10; EVAL_ARGS='--generate --ragas' opcional)
	@mkdir -p evals/results/bioasq
	$(COMPOSE) --profile eval build eval-bioasq
	$(COMPOSE) --profile eval run --rm eval-bioasq \
		--limit $${LIMIT:-10} \
		--distractors $${DISTRACTORS:-50} \
		--output-dir evals/results/bioasq \
		$(EVAL_ARGS)
	@echo "Resultados en evals/results/bioasq/"

.PHONY: eval-bioasq-local
eval-bioasq-local: ## Igual que eval-bioasq pero en el host (sin Docker)
	$(UV) run python -m evals \
		--limit $${LIMIT:-10} \
		--distractors $${DISTRACTORS:-50} \
		--output-dir evals/results/bioasq \
		$(EVAL_ARGS)

.PHONY: eval-deepeval-template
eval-deepeval-template: eval-sync ## Smoke test DeepEval con plantilla JSON (2 QA; Ollama en host)
	@mkdir -p evals/results/template
	$(UV) run python -m evals \
		--template \
		--limit 2 \
		--distractors 0 \
		--generate \
		--deepeval \
		--output-dir evals/results/template
	@echo "Resultados en evals/results/template/"

.PHONY: eval-deepeval-template-docker
eval-deepeval-template-docker: ## Igual en Docker (perfil eval + Ollama)
	@mkdir -p evals/results/template
	$(COMPOSE) --profile eval build eval-bioasq
	$(COMPOSE) --profile eval run --rm eval-bioasq \
		--template \
		--limit 2 \
		--distractors 0 \
		--generate \
		--deepeval \
		--output-dir evals/results/template
	@echo "Resultados en evals/results/template/"

.PHONY: lint
lint: ## Pylint sobre el paquete
	$(UV) run pylint src/chatbot

.PHONY: health
health: ## Ping a /api/v1/health
	@curl -sf http://localhost:$(PORT)/api/v1/health | python3 -m json.tool

.PHONY: shell
shell: ## Shell Python del proyecto
	$(UV) run python
