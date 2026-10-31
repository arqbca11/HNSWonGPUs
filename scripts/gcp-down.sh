#!/usr/bin/env bash
# Delete everything billable from scripts/gcp-up.sh: the shard VM, the GKE
# cluster (its node pools, load balancers and disks go with it). The Artifact
# Registry images stay (pennies). Asks once.
set -euo pipefail
export PATH="$PATH:/opt/homebrew/bin"
PROJECT="${PROJECT:-project-3456ed25-5d5d-4b0d-aa5}"; ZONE="${ZONE:-us-west1-b}"; CLUSTER="${CLUSTER:-kgpu}"; VM="${VM:-kgpu-shards}"
G="gcloud --project=$PROJECT --quiet"
echo "This deletes VM $VM and cluster $CLUSTER in $ZONE (project $PROJECT). Continue? [y/N]"; read -r a; [ "$a" = y ] || exit 1
$G compute instances delete "$VM" --zone "$ZONE" 2>&1 | tail -1 || true
$G container clusters delete "$CLUSTER" --zone "$ZONE" 2>&1 | tail -1 || true
$G compute routers nats delete kgpu-nat --router kgpu-router --region "${REGION:-us-west1}" 2>&1 | tail -1 || true
$G compute routers delete kgpu-router --region "${REGION:-us-west1}" 2>&1 | tail -1 || true
# deleting the cluster leaves its dynamically provisioned disks (Postgres's
# 2 GB, the store's 10 GB) behind; GKE labels them with the cluster's name
for d in $($G compute disks list --filter="labels.goog-k8s-cluster-name=$CLUSTER" --format='value(name)'); do
  $G compute disks delete "$d" --zone "$ZONE" 2>&1 | tail -1 || true
done
echo "== remaining compute instances / clusters / disks:"
$G compute instances list --format='value(name,zone,status)'; $G container clusters list --format='value(name,location,status)'; $G compute disks list --format='value(name,sizeGb)'
echo "== done; check the billing report tomorrow"
