#!/usr/bin/env bash
# Install (or upgrade) Prometheus + Grafana on the kind cluster and point them
# at our services. Idempotent.
#   kube-prometheus-stack Helm chart -> namespace monitoring (values: deploy/monitoring/values.yaml)
#   ServiceMonitors for scheduler, worker; Service + EndpointSlice + ServiceMonitor for the shards
#     on the laptop (address = host.docker.internal as the kind nodes resolve it)
#   Grafana dashboard ConfigMap
# Afterwards: scripts/k8s-dashboards.sh opens port-forwards for Grafana and Prometheus.
set -euo pipefail
cd "$(dirname "$0")/.."
kubectl config use-context kind-kgpu >/dev/null
echo "== helm: kube-prometheus-stack into namespace monitoring"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update prometheus-community >/dev/null
helm upgrade --install kps prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace -f deploy/monitoring/values.yaml \
  --wait --timeout 15m >/dev/null
kubectl -n monitoring get pods --no-headers | awk '{print "   "$1, $3}'
echo "== our scrape targets"
HOST_IP=$(docker exec kgpu-control-plane getent ahostsv4 host.docker.internal | awk 'NR==1{print $1}')
[ -n "$HOST_IP" ] || { echo "!! could not resolve host.docker.internal from a kind node"; exit 1; }
echo "   laptop as seen from the kind nodes: $HOST_IP (the shards' metrics endpoint, :9091)"
sed "s/HOST_IP/$HOST_IP/" deploy/monitoring/servicemonitors.yaml | kubectl apply -f - | sed 's/^/   /'
kubectl apply -f deploy/monitoring/dashboard-kgpu.yaml | sed 's/^/   /'
echo "== done. scripts/k8s-dashboards.sh for Grafana (admin / kgpu) and Prometheus."
