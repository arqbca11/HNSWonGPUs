#!/usr/bin/env bash
# Port-forward Grafana (http://127.0.0.1:3000, admin / kgpu, anonymous viewer
# allowed) and Prometheus (http://127.0.0.1:9090) from the kind cluster.
# Runs until Ctrl-C. Dashboards are for a human at this laptop, so a
# port-forward is enough (decision 33); the shards, which need a stable
# address, get the NodePort instead.
set -euo pipefail
kubectl config use-context kind-kgpu >/dev/null
kubectl -n monitoring port-forward svc/kps-grafana 3000:80 >/dev/null 2>&1 &
G=$!
kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 9090:9090 >/dev/null 2>&1 &
P=$!
trap 'kill $G $P 2>/dev/null' EXIT
echo "Grafana    http://127.0.0.1:3000/d/kgpu-round   (admin / kgpu)"
echo "Prometheus http://127.0.0.1:9090/targets"
echo "Ctrl-C to stop."
wait
