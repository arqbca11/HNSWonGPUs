#!/usr/bin/env bash
# Create the Phase 2 kind cluster (or reuse it), advertise CARDS_PER_NODE fake
# GPU cards on every GPU node as the extended resource kgpu.io/gpu-card, and
# print the result. Idempotent: rerunning re-applies the card count.
#
# An extended resource is a countable thing a node advertises that Kubernetes
# knows nothing about except the number: a pod requests N of it and the
# scheduler subtracts. That is exactly how the NVIDIA device plugin exposes
# real cards (nvidia.com/gpu). On kind there is no plugin, so this script
# writes the number into the node's status directly.
set -euo pipefail
cd "$(dirname "$0")/.."
CARDS_PER_NODE="${CARDS_PER_NODE:-2}"
if kind get clusters 2>/dev/null | grep -qx kgpu; then
  echo "kind cluster 'kgpu' already exists"
else
  kind create cluster --config deploy/kind/cluster.yaml --wait 120s
fi
kubectl config use-context kind-kgpu >/dev/null
kubectl wait --for=condition=Ready node --all --timeout=180s >/dev/null
for n in $(kubectl get nodes -l kgpu.io/gpu=true -o name); do
  kubectl patch "$n" --subresource=status --type=json \
    -p "[{\"op\":\"add\",\"path\":\"/status/capacity/kgpu.io~1gpu-card\",\"value\":\"$CARDS_PER_NODE\"}]" >/dev/null
done
# The kubelet copies capacity into allocatable on its next status sync (~10 s).
for _ in $(seq 1 60); do
  ok=1
  for n in $(kubectl get nodes -l kgpu.io/gpu=true -o name); do
    [ "$(kubectl get "$n" -o jsonpath='{.status.allocatable.kgpu\.io/gpu-card}')" = "$CARDS_PER_NODE" ] || ok=0
  done
  [ "$ok" = 1 ] && break; sleep 1
done
echo "== nodes"
kubectl get nodes -L kgpu.io/role,kgpu.io/gpu
echo "== taints"
kubectl get nodes -o custom-columns='NAME:.metadata.name,TAINTS:.spec.taints[*].key'
echo "== GPU cards (kgpu.io/gpu-card capacity / allocatable)"
kubectl get nodes -l kgpu.io/gpu=true -o custom-columns='NAME:.metadata.name,CAPACITY:.status.capacity.kgpu\.io/gpu-card,ALLOCATABLE:.status.allocatable.kgpu\.io/gpu-card'
