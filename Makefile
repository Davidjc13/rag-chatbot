# RAG Chatbot — atajos de desarrollo y despliegue
# Uso: make help

.DEFAULT_GOAL := help

COMPOSE   ?= docker compose
UV        ?= uv
KUBECTL   ?= kubectl
PORT      ?= 8000
HOST      ?= 0.0.0.0
K8S_NS    ?= rag-chatbot
IMAGE     ?= rag-chatbot:0.1.0
PROFILE_DB     := --profile db
PROFILE_DOCKER := --profile docker

.PHONY: help
help: ## Muestra esta ayuda
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)
	@printf "\nEjemplos:\n  make setup && make up\n  make up-docker\n  make up-k8s\n  make test\n\n"

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
up: env db ## Levanta Postgres (Docker) + app local (uvicorn --reload)
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
db: ## Solo Postgres (pgvector) en Docker
	$(COMPOSE) $(PROFILE_DB) up -d postgres
	@echo "Esperando a Postgres…"
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
		$(COMPOSE) $(PROFILE_DB) exec -T postgres pg_isready -U chatbot -d chatbot >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@$(COMPOSE) $(PROFILE_DB) exec -T postgres \
		psql -U chatbot -d chatbot -c 'CREATE EXTENSION IF NOT EXISTS vector;' >/dev/null
	@echo "Postgres listo en localhost:5432"

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

.PHONY: k8s-image
k8s-image: ## Construye la imagen para Kubernetes
	docker build -t $(IMAGE) .

.PHONY: k8s-apply
k8s-apply: ## Aplica manifiestos (namespace, ollama, chatbot, ingress)
	bash k8s/apply.sh

.PHONY: up-k8s
up-k8s: k8s-image k8s-apply ## Build imagen + aplica k8s
	@echo "Namespace: $(K8S_NS)"
	@$(KUBECTL) -n $(K8S_NS) get pods,svc,ingress,jobs
	@echo "Siguiente: make k8s-pf"

.PHONY: k8s-status
k8s-status: ## Estado de pods/svc/ingress/jobs
	$(KUBECTL) -n $(K8S_NS) get pods,svc,ingress,jobs

.PHONY: k8s-logs
k8s-logs: ## Logs del deployment chatbot
	$(KUBECTL) -n $(K8S_NS) logs -f deploy/chatbot --tail=100

.PHONY: k8s-pf
k8s-pf: ## Port-forward svc/chatbot → localhost:$(PORT)
	@echo "App en http://localhost:$(PORT)/"
	$(KUBECTL) -n $(K8S_NS) port-forward svc/chatbot $(PORT):8000

.PHONY: k8s-ollama-logs
k8s-ollama-logs: ## Logs de Ollama / job de pull
	-$(KUBECTL) -n $(K8S_NS) logs -f deploy/ollama --tail=80
	-$(KUBECTL) -n $(K8S_NS) logs -l app.kubernetes.io/component=ollama-init --tail=80

.PHONY: down-k8s
down-k8s: ## Elimina el namespace completo
	$(KUBECTL) delete namespace $(K8S_NS) --ignore-not-found

# ── Calidad ─────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Ejecuta pytest
	$(UV) run pytest -q

.PHONY: lint
lint: ## Pylint sobre el paquete
	$(UV) run pylint src/chatbot

.PHONY: health
health: ## Ping a /api/v1/health
	@curl -sf http://localhost:$(PORT)/api/v1/health | python3 -m json.tool

.PHONY: shell
shell: ## Shell Python del proyecto
	$(UV) run python
