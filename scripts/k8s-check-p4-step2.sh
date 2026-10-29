#!/usr/bin/env bash
# Phase 4 step 2: a real round on the kind cluster. The shards on the laptop
# run hnswlib builds as child processes (2 threads each) for local placements;
# the workers in the cluster run the same builder CLI (BUILDER=hnswlib) for
# GPU placements; vectors come from the cluster's object store and every
# finished build leaves index.bin + manifest.json under artifacts/. One round
# per policy on sift-128 at 6 shards; saves test-logs/phase4/step2-real-round.log.
#
#   scripts/k8s-check-p4-step2.sh                 # always_gpu then reactive
#   POLICIES=reactive scripts/k8s-check-p4-step2.sh
#   DATASET=gist-960-300k scripts/k8s-check-p4-step2.sh
#
# Sets the cluster to FAKE_TIME_SCALE=1 and BUILDER=hnswlib for the run and
# restores it after (the same patch-restart-verify as the step 7 runner).
# Laptop numbers are a functional check, not calibration (decision 7): the
# shards' builds, the workers' builds and the store all share this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${KGPU_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then KGPU_CAFFEINATED=1 exec caffeinate -i "$0" "$@"; fi
LOG="${LOG:-test-logs/phase4/step2-real-round.log}"
DATASET="${DATASET:-sift-128}"; N_SHARDS="${N_SHARDS:-6}"; WORKERS="${WORKERS:-2}"; POLICIES="${POLICIES:-always_gpu reactive}"
SEED="${SEED:-1}"; ROUND_TIMEOUT="${ROUND_TIMEOUT:-30m}"
K="kubectl -n kgpu"
mkdir -p "$(dirname "$LOG")" experiments/results/step2-timelines
exec > >(tee "$LOG") 2>&1
echo "== Phase 4 step 2: a real round, dataset $DATASET at $N_SHARDS shards, $(date -u +%Y-%m-%dT%H:%M:%SZ), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+local changes')"
kubectl config use-context kind-kgpu >/dev/null
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  scripts/k8s-build-load.sh scheduler >/dev/null && scripts/k8s-build-load.sh worker >/dev/null
  kubectl apply -k deploy/k8s >/dev/null
  echo "   images rebuilt and loaded; manifests applied"
fi
ORIG_SCALE=$($K get configmap worker -o jsonpath='{.data.FAKE_TIME_SCALE}'); ORIG_POLICY=$($K get configmap scheduler -o jsonpath='{.data.POLICY}'); ORIG_BUILDER=$($K get configmap worker -o jsonpath='{.data.BUILDER}')
echo "   cluster before: FAKE_TIME_SCALE=$ORIG_SCALE POLICY=$ORIG_POLICY BUILDER=$ORIG_BUILDER"
restart_and_verify() { # <deployment> <label> <expected log fragment>
  local deploy="$1" label="$2" expect="$3" old new
  old=$($K get pods -l "app=$label" -o name | tr '\n' ' ')
  $K rollout restart "deployment/$deploy" >/dev/null
  for _ in $(seq 1 360); do
    sleep 0.5
    local gone=1; for p in $old; do $K get "$p" >/dev/null 2>&1 && { gone=0; break; }; done
    [ "$gone" = 1 ] || continue
    $K rollout status "deployment/$deploy" --timeout=5s >/dev/null 2>&1 || continue
    new=$($K get pods -l "app=$label" --field-selector=status.phase=Running -o name); [ -n "$new" ] || continue
    local ok=1; for p in $new; do $K logs "$p" 2>/dev/null | grep -q -- "$expect" || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0
  done
  echo "!! $deploy: no new pod reporting $expect within 180 s"; return 1
}
CUR_POLICY=""
set_cluster() { # <policy>
  $K patch configmap worker --type merge -p '{"data":{"FAKE_TIME_SCALE":"1","BUILDER":"hnswlib"}}' >/dev/null
  $K patch configmap scheduler --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"1\",\"POLICY\":\"$1\"}}" >/dev/null
  [ -n "$CUR_POLICY" ] || restart_and_verify worker worker '"builder": "hnswlib"'
  restart_and_verify scheduler scheduler "\"policy\":\"$1\""
  CUR_POLICY="$1"
  echo "   cluster set and verified: FAKE_TIME_SCALE=1 BUILDER=hnswlib POLICY=$1"
}
PF=""
restore() {
  echo; echo "== restoring the cluster: FAKE_TIME_SCALE=$ORIG_SCALE POLICY=$ORIG_POLICY BUILDER=$ORIG_BUILDER"
  $K patch configmap worker --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$ORIG_SCALE\",\"BUILDER\":\"$ORIG_BUILDER\"}}" >/dev/null || true
  $K patch configmap scheduler --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$ORIG_SCALE\",\"POLICY\":\"$ORIG_POLICY\"}}" >/dev/null || true
  $K rollout restart deployment/scheduler deployment/worker >/dev/null || true
  $K rollout status deployment/scheduler --timeout=180s >/dev/null || true
  $K rollout status deployment/worker --timeout=180s >/dev/null || true
  [ -n "$PF" ] && kill "$PF" 2>/dev/null || true
  echo "   restored; log: $LOG"
}
trap restore EXIT
# the shards on the laptop reach the store through the NodePort or a port-forward
if curl -sf --max-time 2 http://127.0.0.1:30900/health/live >/dev/null; then export S3_ENDPOINT=http://127.0.0.1:30900; else
  kubectl -n kgpu port-forward svc/objectstore 39000:9000 >/dev/null 2>&1 & PF=$!
  for _ in $(seq 1 40); do curl -sf --max-time 1 http://127.0.0.1:39000/health/live >/dev/null && break; sleep 0.25; done
  export S3_ENDPOINT=http://127.0.0.1:39000
fi
export S3_ACCESS_KEY=$($K get secret objectstore -o jsonpath='{.data.S3_ACCESS_KEY}' | base64 -d) S3_SECRET_KEY=$($K get secret objectstore -o jsonpath='{.data.S3_SECRET_KEY}' | base64 -d) S3_BUCKET=$($K get secret objectstore -o jsonpath='{.data.S3_BUCKET}' | base64 -d)
echo "   store for the laptop's shards: $S3_ENDPOINT"
for policy in $POLICIES; do
  echo; echo "#### $DATASET, $N_SHARDS shards, $WORKERS workers, policy $policy, seed $SEED: real builds everywhere"
  set_cluster "$policy"
  dir="experiments/results/step2-timelines/$DATASET-$N_SHARDS-$policy"; rm -rf "$dir"; mkdir -p "$dir"
  t0=$(date +%s)
  TIMELINE_DIR="$dir" TIME_SCALE=1 DATASET="$DATASET" SCENARIO=skewed N_SHARDS="$N_SHARDS" WORKERS="$WORKERS" SEED="$SEED" ROUNDS=1 ROUND_TIMEOUT="$ROUND_TIMEOUT" \
    scripts/k8s-round.sh | sed -n '/^== timeline/,/^== scheduler log/p' | grep -v '^== scheduler log' | sed 's/^/   /'
  echo "   wall: $(( $(date +%s) - t0 )) s"
  echo "-- the builds' manifests in the store (recall, bytes, build time, threads):"
  absdir="$PWD/$dir"
  (cd builder && uv run --quiet python - "$absdir" <<'PY'
import json, sys, glob
from kgpu_builder.config import Config
from kgpu_builder.store import Store
st = Store(Config.from_env())
tl = json.load(open(glob.glob(sys.argv[1] + "/round-*.json")[0]))
rid = tl["round_id"]
for s in tl["shards"]:
    bid = f"{s['shard_id']}:{rid}"
    keys = [k for k in st.list(f"artifacts/{bid}/") if k.endswith("manifest.json")]
    if not keys:
        print(f"   shard {s['shard_id']:2d} {s['placement']:5s}: no manifest (state {s['build_state']})"); continue
    m = st.get_json(keys[-1])
    print(f"   shard {s['shard_id']:2d} {s['placement']:5s} attempt {m['attempt']}: {m['n_vectors']:7d} vectors, recall@10 {m['recall']:.3f}, {m['index_bytes']/1e6:6.1f} MB, build {m['build_s']:6.1f} s on {m['threads']} threads (+fetch {m['fetch_s']:.1f} gate {m['gate_s']:.1f} upload {m['upload_s']:.1f})")
PY
  )
done
echo; echo "== Result: PASS (log: $LOG)"
