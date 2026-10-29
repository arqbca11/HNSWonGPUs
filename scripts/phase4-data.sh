#!/usr/bin/env bash
# Phase 4 step 1: put the datasets into the cluster's object store. Runs the
# builder's data pipeline for SIFT-128 and the GIST-960 subset at 6 and 50
# shards — download (resumable), Zipf split, exact per-shard ground truth,
# upload, verify — against the kind cluster's RustFS, and saves
# test-logs/phase4/step1-data.log. The split and ground truth are skipped
# when their local outputs exist (KGPU_DATA_DIR, default ./data, gitignored).
#
#   scripts/phase4-data.sh                      # both datasets, both splits
#   DATASETS=sift-128 SPLITS=6 scripts/phase4-data.sh
#
# Reaches the store through NodePort 30900 when kind maps it (clusters created
# after Phase 4 step 1), else through a temporary `kubectl port-forward`.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=test-logs/phase4/step1-data.log
DATASETS="${DATASETS:-sift-128 gist-960-300k}"; SPLITS="${SPLITS:-6 50}"
export KGPU_DATA_DIR="${KGPU_DATA_DIR:-$PWD/data}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "== Phase 4 step 1: datasets into the cluster's object store, $(date -u +%Y-%m-%dT%H:%M:%SZ), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+local changes')"
kubectl config use-context kind-kgpu >/dev/null
kubectl -n kgpu rollout status statefulset/objectstore --timeout=120s >/dev/null
PF=""
if curl -sf --max-time 2 http://127.0.0.1:30900/health/live >/dev/null; then
  export S3_ENDPOINT=http://127.0.0.1:30900; echo "   store reachable at the NodePort $S3_ENDPOINT"
else
  kubectl -n kgpu port-forward svc/objectstore 39000:9000 >/dev/null 2>&1 & PF=$!
  for _ in $(seq 1 40); do curl -sf --max-time 1 http://127.0.0.1:39000/health/live >/dev/null && break; sleep 0.25; done
  export S3_ENDPOINT=http://127.0.0.1:39000; echo "   NodePort 30900 not mapped on this cluster; port-forwarding svc/objectstore to $S3_ENDPOINT"
fi
trap '[ -n "$PF" ] && kill $PF 2>/dev/null || true' EXIT
export S3_ACCESS_KEY=$(kubectl -n kgpu get secret objectstore -o jsonpath='{.data.S3_ACCESS_KEY}' | base64 -d)
export S3_SECRET_KEY=$(kubectl -n kgpu get secret objectstore -o jsonpath='{.data.S3_SECRET_KEY}' | base64 -d)
export S3_BUCKET=$(kubectl -n kgpu get secret objectstore -o jsonpath='{.data.S3_BUCKET}' | base64 -d)
cd builder
for ds in $DATASETS; do
  echo; echo "#### $ds: download"
  uv run --quiet python -m kgpu_builder download "$ds"
  for n in $SPLITS; do
    dir="$KGPU_DATA_DIR/splits/$ds/$n"
    echo; echo "#### $ds, $n shards: split"
    if [ -f "$dir/manifest.json" ]; then echo "   (local split exists: $dir)"; python3 -c "
import json; m=json.load(open('$dir/manifest.json'))
print('   ' + ', '.join('shard %d: rank %d, %d vectors' % (s['shard_id'], s['rank'], s['size']) for s in m['shards'][:8]) + (' ...' if len(m['shards'])>8 else ''))"; else uv run --quiet python -m kgpu_builder split "$ds" "$n"; fi
    echo "#### $ds, $n shards: ground truth"
    if ls "$dir"/ground_truth/shard_*.npy >/dev/null 2>&1; then echo "   (local ground truth exists: $(ls "$dir"/ground_truth | wc -l | tr -d ' ') shards)"; else uv run --quiet python -m kgpu_builder groundtruth "$ds" "$n"; fi
    echo "#### $ds, $n shards: upload"
    uv run --quiet python -m kgpu_builder upload "$ds" "$n" | grep -v '^  put '
    echo "#### $ds, $n shards: verify"
    uv run --quiet python -m kgpu_builder verify "$ds" "$n" | tail -1
  done
done
cd ..
echo; echo "== objects in the store"
cd builder && uv run --quiet python - <<'PY'
from kgpu_builder.config import Config
from kgpu_builder.store import Store
st = Store(Config.from_env())
keys = st.list("")
total = 0
by = {}
for k in keys:
    n = st.size(k); total += n
    top = "/".join(k.split("/")[:3]); by[top] = by.get(top, 0) + n
for k, n in sorted(by.items()):
    print(f"   {k:40s} {n/1e6:9.1f} MB")
print(f"   {len(keys)} objects, {total/1e9:.2f} GB in bucket {st.bucket}")
PY
cd ..
echo; echo "== Result: PASS (log: $LOG)"
