#!/usr/bin/env bash
# Phase 4 step 4: the cloud topology without GPUs (roadmap, Phase 4; decisions
# 7, 8, 25, 26). A zonal GKE cluster in us-west1-b (us-west1-a stocked out, decision 29) with a small default pool
# for the control plane services and a `workers` pool of n2-standard-8 nodes
# (one CPU worker each, 8 threads), the Phase 2 manifests applied through the
# deploy/k8s-gcp overlay (images from Artifact Registry, internal load
# balancers), and a shard VM outside the cluster, as the database fleet would
# be, sized for 6-shard rounds (16 vCPUs: six 2-thread builds at once).
# Idempotent: every step checks before creating. scripts/gcp-down.sh deletes
# everything billable. Set a billing alert before the first run. No public
# IP anywhere (decision 28): private nodes, a NAT for outbound, IAP for ssh.
#
#   scripts/gcp-up.sh                 # cluster, workers, manifests, VM
#   WORKER_NODES=2 scripts/gcp-up.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/opt/homebrew/bin"
PROJECT="${PROJECT:-project-3456ed25-5d5d-4b0d-aa5}"; REGION="${REGION:-us-west1}"; ZONE="${ZONE:-us-west1-b}"
CLUSTER="${CLUSTER:-kgpu}"; WORKER_NODES="${WORKER_NODES:-2}"; WORKER_MACHINE="${WORKER_MACHINE:-n2-standard-8}"
VM="${VM:-kgpu-shards}"; VM_MACHINE="${VM_MACHINE:-n2d-standard-16}"   # n2d (AMD): the only 16-vCPU family us-west1-b had to give (decision 32)
REG="$REGION-docker.pkg.dev/$PROJECT/kgpu"
G="gcloud --project=$PROJECT --quiet"
echo "== Phase 4 cloud topology: project $PROJECT, zone $ZONE, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# the project-wide CPU cap must hold the whole topology (4 + 16 + 16 = 36; a new project gets 12: request "CPUs (all regions)" in the console)
cap=$($G compute project-info describe --format='value(quotas)' | tr ';' '\n' | grep CPUS_ALL_REGIONS | grep -o "'limit': [0-9.]*" | grep -o '[0-9.]*$')
echo "-- CPUS_ALL_REGIONS quota: ${cap:-?} (need 36)"
# 0. no public IPs anywhere (the project's org policy forbids them, and nothing
#    here needs one): a Cloud NAT lets private nodes and the VM reach the
#    internet (Docker Hub images, apt, the datasets); an IAP firewall rule lets
#    `gcloud compute ssh --tunnel-through-iap` reach the VM.
if ! $G compute routers describe kgpu-router --region "$REGION" >/dev/null 2>&1; then
  echo "-- creating Cloud Router + NAT for $REGION (private nodes and the VM reach the internet through it)"
  $G compute routers create kgpu-router --region "$REGION" --network default >/dev/null
  $G compute routers nats create kgpu-nat --router kgpu-router --region "$REGION" --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges >/dev/null
fi
if ! $G compute firewall-rules describe kgpu-allow-iap-ssh >/dev/null 2>&1; then
  $G compute firewall-rules create kgpu-allow-iap-ssh --network default --direction INGRESS --allow tcp:22 --source-ranges 35.235.240.0/20 >/dev/null
fi
# 0b. the nodes pull images with the project's default compute service account, which must be able to read the registry (a fresh project's cannot)
PROJECT_NUMBER=$($G projects describe "$PROJECT" --format='value(projectNumber)')
$G artifacts repositories add-iam-policy-binding kgpu --location "$REGION" --member "serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role roles/artifactregistry.reader >/dev/null
# 1. the cluster: zonal (the control plane fee is covered by the free credit for
#    one zonal cluster), private nodes (no external IPs), public control-plane endpoint
if ! $G container clusters describe "$CLUSTER" --zone "$ZONE" >/dev/null 2>&1; then
  echo "-- creating cluster $CLUSTER (private nodes; default pool: 1 x e2-standard-4 for scheduler, Postgres, the object store)"
  $G container clusters create "$CLUSTER" --zone "$ZONE" --num-nodes 1 --machine-type e2-standard-4 --disk-size 50 \
    --release-channel regular --no-enable-basic-auth --workload-pool="$PROJECT.svc.id.goog" \
    --enable-ip-alias --enable-private-nodes --master-ipv4-cidr 172.16.0.0/28 --enable-dns-access >/dev/null
fi
# private nodes switch on "authorized networks" with an empty list, so the
# public IP endpoint answers nobody; the DNS endpoint authenticates by the
# gcloud account instead of by the laptop's IP (decision 30)
$G container clusters update "$CLUSTER" --zone "$ZONE" --enable-dns-access >/dev/null 2>&1 || true
if ! $G container node-pools describe workers --cluster "$CLUSTER" --zone "$ZONE" >/dev/null 2>&1; then
  echo "-- creating node pool workers: $WORKER_NODES x $WORKER_MACHINE (spot: decision 27, the quota was granted the first day)"
  $G container node-pools create workers --cluster "$CLUSTER" --zone "$ZONE" --num-nodes "$WORKER_NODES" --machine-type "$WORKER_MACHINE" --disk-size 50 --spot >/dev/null
fi
export USE_GKE_GCLOUD_AUTH_PLUGIN=True PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"   # kubectl's GKE auth plugin (gcloud components install gke-gcloud-auth-plugin)
$G container clusters get-credentials "$CLUSTER" --zone "$ZONE" --dns-endpoint >/dev/null
echo "-- nodes:"; kubectl get nodes -o custom-columns='NAME:.metadata.name,POOL:.metadata.labels.cloud\.google\.com/gke-nodepool,CPU:.status.capacity.cpu,MEM:.status.capacity.memory' | sed 's/^/   /'
# 2. the manifests
kubectl apply -k deploy/k8s-gcp >/dev/null
echo "-- manifests applied (deploy/k8s-gcp); waiting for Postgres, the object store, the scheduler and the workers"
kubectl -n kgpu rollout status statefulset/postgres --timeout=300s >/dev/null
kubectl -n kgpu rollout status statefulset/objectstore --timeout=300s >/dev/null
kubectl -n kgpu rollout status deployment/scheduler --timeout=300s >/dev/null
# "wait available", not "rollout status": a Deployment whose pods sat Pending
# for longer than its progress deadline (here, until the pool existed) keeps a
# failed Progressing condition that rollout status reports as an error even
# once every pod is ready
kubectl -n kgpu wait --for=condition=available deployment/worker --timeout=600s >/dev/null
for svc in scheduler objectstore; do
  for _ in $(seq 1 120); do ip=$(kubectl -n kgpu get svc $svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null); [ -n "$ip" ] && break; sleep 5; done
  echo "   $svc internal IP: ${ip:-NOT YET}"
done
SCHED_IP=$(kubectl -n kgpu get svc scheduler -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
S3_IP=$(kubectl -n kgpu get svc objectstore -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
kubectl -n kgpu get pods -o wide | sed 's/^/   /'
# 3. the shard VM, outside the cluster, same VPC; its startup script installs the toolchain and clones the repo
if ! $G compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; then
  echo "-- creating the shard VM $VM ($VM_MACHINE, Debian 12)"
  $G compute instances create "$VM" --zone "$ZONE" --machine-type "$VM_MACHINE" --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 40GB --no-address --scopes cloud-platform --metadata-from-file startup-script=deploy/gcp/shard-vm-startup.sh >/dev/null
fi
echo "-- shard VM: $($G compute instances describe "$VM" --zone "$ZONE" --format='value(status,networkInterfaces[0].networkIP)') (no public IP; ssh through IAP)"
# the endpoints the VM's shards use, written to the VM's metadata so scripts there can read them
$G compute instances add-metadata "$VM" --zone "$ZONE" --metadata "kgpu-scheduler-url=http://$SCHED_IP:8080,kgpu-s3-endpoint=http://$S3_IP:9000" >/dev/null
echo "== up. scheduler http://$SCHED_IP:8080 and store http://$S3_IP:9000 (private IPs, reachable from the VM); ssh: gcloud compute ssh $VM --zone $ZONE --tunnel-through-iap"
