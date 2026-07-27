#!/usr/bin/env bash
# Aplica todos los manifiestos de Kubernetes en orden.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl apply -f "$ROOT/namespace.yaml"
kubectl apply -f "$ROOT/configmap.yaml"
kubectl apply -f "$ROOT/ollama-deployment.yaml"
kubectl apply -f "$ROOT/ollama-pull-job.yaml"
kubectl apply -f "$ROOT/chatbot-deployment.yaml"
kubectl apply -f "$ROOT/ingress.yaml"

echo "Despliegue aplicado. Espera a que el Job de pull termine y los pods estén Ready:"
echo "  kubectl -n rag-chatbot get pods,svc,ingress,jobs"
echo "  kubectl -n rag-chatbot port-forward svc/chatbot 8000:8000"
