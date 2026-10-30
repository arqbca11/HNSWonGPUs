#!/usr/bin/env bash
# Run ON THE SHARD VM (gcloud compute ssh kgpu-shards -- sudo bash /opt/kgpu/HNSWonGPUs/deploy/gcp/vm-session.sh <step>):
# the Phase 4 closing-step work that belongs to the shard host, against the
# cluster's private endpoints (from the instance metadata). Steps:
#   data        download both datasets here (in-cloud, fast), split 6 and 50 ways, ground truth, upload to the cluster's store, verify
#   calibrate   the shard-side calibration: 2 threads on this VM's cores (and 8 for the ratio on the same machine), both datasets
#   rounds      real 6-shard rounds against the cluster: sift-128 under the policy the cluster runs
#   all         the three in order
# Output under /opt/kgpu/out/ (logs, CSVs); the laptop fetches them with gcloud compute scp.
set -euo pipefail
export PATH="/usr/local/go/bin:/usr/local/bin:$PATH"
cd /opt/kgpu/HNSWonGPUs
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
export SCHEDULER_URL="$(md kgpu-scheduler-url)" S3_ENDPOINT="$(md kgpu-s3-endpoint)" S3_ACCESS_KEY=kgpu S3_SECRET_KEY=kgpu-secret S3_BUCKET=kgpu
export KGPU_DATA_DIR=/opt/kgpu/data
OUT=/opt/kgpu/out; mkdir -p "$OUT" "$KGPU_DATA_DIR"
HOST=$(hostname)
step="${1:-all}"
echo "== vm-session $step on $HOST ($(nproc) vCPUs), scheduler $SCHEDULER_URL, store $S3_ENDPOINT, $(date -u +%Y-%m-%dT%H:%M:%SZ), commit $(git rev-parse --short HEAD)"
curl -sf "$SCHEDULER_URL/healthz" >/dev/null && echo "   scheduler reachable" || { echo "!! scheduler unreachable"; exit 1; }
curl -sf "$S3_ENDPOINT/health/live" >/dev/null && echo "   store reachable" || { echo "!! store unreachable"; exit 1; }
if [ "$step" = data ] || [ "$step" = all ]; then
  for ds in sift-128 gist-960-300k; do for n in 6 50; do
    echo "#### $ds $n"; (cd builder && uv run --quiet python -m kgpu_builder all "$ds" "$n" | grep -E 'already complete|complete,|split written|ground truth for|uploaded|verify:')
  done; done
fi
if [ "$step" = calibrate ] || [ "$step" = all ]; then
  (cd builder && uv run --quiet python -m kgpu_builder calibrate sift-128 6 --sizes 10000,20000,50000,100000,200000,400000 --threads 2,8 --repeats 2 --out "$OUT/$HOST-sift-128-6.csv" \
             && uv run --quiet python -m kgpu_builder calibrate gist-960-300k 6 --sizes 10000,30000,60000,120000 --threads 2,8 --repeats 2 --out "$OUT/$HOST-gist-960-300k-6.csv" \
             && uv run --quiet python -m kgpu_builder fit "$OUT/$HOST-sift-128-6.csv" "$OUT/$HOST-gist-960-300k-6.csv")
fi
if [ "$step" = rounds ] || [ "$step" = all ]; then
  for seed in 1 2; do
    echo "#### real round sift-128 6 shards seed $seed (policy: the cluster's)"
    TIMELINE_DIR="$OUT/timelines" DATASET=sift-128 TIME_SCALE=1 SCENARIO=skewed N_SHARDS=6 SEED=$seed ROUNDS=1 ROUND_TIMEOUT=30m MIN_WORKERS=2 LOG_FORMAT=text \
      BUILDER_CMD="uv run --project builder python -m kgpu_builder" SHARD_THREADS=2 go run ./shard/cmd/shardsim 2>&1 | grep -vE '^time=' | sed 's/^/   /'
  done
fi
echo "== vm-session $step done, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
