#!/usr/bin/env bash
# Aplica todos los manifiestos de Kubernetes en orden.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! kubectl cluster-info >/dev/null 2>&1; then
  cat <<'EOF' >&2
No hay un cluster de Kubernetes alcanzable (kubectl apunta a nada / localhost:8080).

Opciones:
  1) Cluster local con kind:
       make k8s-cluster
       make up-k8s
  2) Sin Kubernetes (Docker Compose):
       make up-docker
EOF
  exit 1
fi

kubectl apply -f "$ROOT/namespace.yaml"
kubectl apply -f "$ROOT/configmap.yaml"
kubectl apply -f "$ROOT/postgres-deployment.yaml"
kubectl apply -f "$ROOT/neo4j-deployment.yaml"
kubectl apply -f "$ROOT/ollama-deployment.yaml"
kubectl apply -f "$ROOT/ollama-pull-job.yaml"
kubectl apply -f "$ROOT/chatbot-deployment.yaml"
kubectl apply -f "$ROOT/ingress.yaml"

echo "Despliegue aplicado. Espera a que los pods estén Ready:"
echo "  make k8s-status"
echo "  make k8s-pf"
