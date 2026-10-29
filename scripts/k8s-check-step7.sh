#!/usr/bin/env bash
# Phase 3 step 7: validate the discrete-event simulator against the live kind
# cluster. For each configuration below, run seeded rounds live (the shard
# simulator on the laptop, the scheduler and fake workers in the cluster),
# keep the per-round timelines, replay the same seeds through the simulator
# with the live cadences, and print the two side by side
# (experiments/cmd/validate). Saves test-logs/phase3/step7-validation.log.
#
# The cluster normally runs at FAKE_TIME_SCALE=10; a validation run sets the
# scale to TIME_SCALE (default 1, so the live cadences — a 0.5 s report, a
# 2 s reconsider tick, a 10 s lease — mean the same modeled seconds the
# simulator defaults to) and the scheduler's POLICY per configuration, by
# patching the ConfigMaps and restarting the Deployments, and restores both
# on exit. A ConfigMap is a Kubernetes object holding plain key/value
# settings a pod reads as environment variables at start, so a change needs
# a restart to take effect.
#
#   scripts/k8s-check-step7.sh                 # every configuration, ~35 min at scale 1
#   ONLY=A scripts/k8s-check-step7.sh          # one configuration (A..F below)
#   TIME_SCALE=10 ONLY=A scripts/k8s-check-step7.sh
#   SKIP_HOUR=1 scripts/k8s-check-step7.sh     # without the two hour_build rounds
#   SKIP_BUILD=1 ...                           # don't rebuild/load the images first (they must already match the checkout)
#
# Runs under caffeinate on macOS: a sleeping laptop pauses Docker Desktop and
# the shard simulator, leases expire, and every GPU build in flight restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${KGPU_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  KGPU_CAFFEINATED=1 exec caffeinate -i "$0" "$@"
fi
LOG=test-logs/phase3/step7-validation.log
OUT=experiments/results/step7-timelines
TIME_SCALE="${TIME_SCALE:-1}"
ONLY="${ONLY:-}"; SKIP_HOUR="${SKIP_HOUR:-0}"
WORKER_MEM_BYTES=25769803776 # 24 GiB, decision 26; the simulator side is told the same
K="kubectl -n kgpu"
mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "== Phase 3 step 7: simulator vs live kind cluster, time scale $TIME_SCALE, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
kubectl config use-context kind-kgpu >/dev/null
# The cluster must run THIS checkout: the first full run compared Phase 2
# images (8 and 10 days stale) and the scheduler crashed on POLICY=reactive,
# which it did not know. Rebuild, load and apply unless told not to.
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  scripts/k8s-build-load.sh scheduler >/dev/null && scripts/k8s-build-load.sh worker >/dev/null
  kubectl apply -k deploy/k8s >/dev/null
  echo "   images rebuilt from $(git rev-parse --short HEAD)$(git diff --quiet || echo '+local changes') and loaded; manifests applied"
fi
ORIG_SCALE=$($K get configmap worker -o jsonpath='{.data.FAKE_TIME_SCALE}')
ORIG_POLICY=$($K get configmap scheduler -o jsonpath='{.data.POLICY}')
ORIG_MEM=$($K get configmap worker -o jsonpath='{.data.WORKER_MEM_BYTES}')
echo "   cluster before: FAKE_TIME_SCALE=$ORIG_SCALE POLICY=$ORIG_POLICY WORKER_MEM_BYTES=$ORIG_MEM"

CUR_POLICY=""
# restart_and_verify <deployment> <app label> <expected log fragment>: a
# `rollout restart` followed at once by `rollout status` can return before the
# controller has even begun the new rollout (bug 15: the first full run's F and
# G rounds were served by the PREVIOUS policy's pod), so wait until every pod
# that existed before the restart is gone, then read the setting back from the
# new pod's own startup line instead of trusting the ConfigMap.
restart_and_verify() {
  local deploy="$1" label="$2" expect="$3" old new
  old=$($K get pods -l "app=$label" -o name | tr '\n' ' ')
  $K rollout restart "deployment/$deploy" >/dev/null
  for _ in $(seq 1 360); do
    sleep 0.5
    local gone=1; for p in $old; do $K get "$p" >/dev/null 2>&1 && { gone=0; break; }; done
    [ "$gone" = 1 ] || continue
    $K rollout status "deployment/$deploy" --timeout=5s >/dev/null 2>&1 || continue
    new=$($K get pods -l "app=$label" --field-selector=status.phase=Running -o name)
    [ -n "$new" ] || continue
    local ok=1; for p in $new; do $K logs "$p" 2>/dev/null | grep -q -- "$expect" || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0
  done
  echo "!! $deploy: no new pod reporting $expect within 180 s"; return 1
}
set_cluster() { # set_cluster <policy>: patch both ConfigMaps, restart what changed, verify from the pods' own logs
  local policy="$1"
  $K patch configmap worker --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$TIME_SCALE\",\"WORKER_MEM_BYTES\":\"$WORKER_MEM_BYTES\"}}" >/dev/null
  $K patch configmap scheduler --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$TIME_SCALE\",\"POLICY\":\"$policy\"}}" >/dev/null
  [ -n "$CUR_POLICY" ] || restart_and_verify worker worker "\"mem_bytes\": $WORKER_MEM_BYTES"
  restart_and_verify scheduler scheduler "\"policy\":\"$policy\""
  CUR_POLICY="$policy"
  echo "   cluster set and verified from the new pods' startup lines: FAKE_TIME_SCALE=$TIME_SCALE POLICY=$policy WORKER_MEM_BYTES=$WORKER_MEM_BYTES ($($K get pods -l app=worker --field-selector=status.phase=Running -o name | wc -l | tr -d ' ') workers running)"
}
restore() {
  echo; echo "== restoring the cluster: FAKE_TIME_SCALE=$ORIG_SCALE POLICY=$ORIG_POLICY WORKER_MEM_BYTES=$ORIG_MEM"
  $K patch configmap worker --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$ORIG_SCALE\",\"WORKER_MEM_BYTES\":\"$ORIG_MEM\"}}" >/dev/null || true
  $K patch configmap scheduler --type merge -p "{\"data\":{\"FAKE_TIME_SCALE\":\"$ORIG_SCALE\",\"POLICY\":\"$ORIG_POLICY\"}}" >/dev/null || true
  $K scale deployment/worker --replicas=2 >/dev/null || true
  $K rollout restart deployment/scheduler deployment/worker >/dev/null || true
  $K rollout status deployment/scheduler --timeout=180s >/dev/null || true
  $K rollout status deployment/worker --timeout=180s >/dev/null || true
  echo "   restored; log: $LOG"
}
trap restore EXIT

FAILS=0
run_config() { # run_config <name> <scenario> <n_shards> <workers> <policy> <seed> <rounds> <timeout> <what it checks>
  local name="$1" scenario="$2" shards="$3" workers="$4" policy="$5" seed="$6" rounds="$7" timeout="$8" what="$9"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return; fi
  echo; echo "#### configuration $name: $scenario, $shards shards, $workers workers, policy $policy, seeds $seed..$((seed+rounds-1)) — $what"
  [ "$CUR_POLICY" = "$policy" ] || set_cluster "$policy"
  local dir="$OUT/$name"; rm -rf "$dir"; mkdir -p "$dir"
  local t0=$(date +%s)
  if TIMELINE_DIR="$dir" TIME_SCALE="$TIME_SCALE" SCENARIO="$scenario" N_SHARDS="$shards" WORKERS="$workers" SEED="$seed" ROUNDS="$rounds" ROUND_TIMEOUT="$timeout" \
       scripts/k8s-round.sh | sed -n '/^== timeline/,/^== round rows/p' | grep -v '^== round rows' | sed 's/^/   /'; then :; else
    echo "!! configuration $name: the live run failed"; FAILS=$((FAILS+1)); return
  fi
  echo "   live run took $(( $(date +%s) - t0 )) s wall for $rounds round(s)"
  echo "-- simulator replay with the live cadences (sharing off, decision 22):"
  go run ./experiments/cmd/validate -timelines "$dir" -policy "$policy" -workers "$workers" -worker_mem_gb 24 -lease_s 10 | sed 's/^/   /'
}

# name  scenario    shards workers policy      seed rounds timeout  what
run_config A  skewed      6  2 always_gpu 1 3 10m "the baseline: horizon-bound 30 s rounds, every build on a GPU"
run_config B  skewed      6  2 always_gpu 1 1 10m "the same seed again, for the live stack's own jitter (compare with A's first row)"
run_config C  skewed     50  3 always_gpu 1 3 15m "50 shards on 3 workers: the GPU queue binds, FCFS waits and a straggler lag"
run_config D  skewed     50  3 reactive   1 3 15m "the same with reactive: most builds stay local, a few offload; the policy path end to end"
run_config E  high_dim    6  2 reactive   1 3 15m "1536 dims: transfer rivals the GPU build, memory refuses the largest, preemption acts"
if [ "$SKIP_HOUR" != 1 ]; then
run_config F  hour_build  6  2 always_gpu 1 1 90m "one modeled hour on a CPU, ~7 min on a GPU: the cadence-to-build ratio of production (decision 27)"
run_config G  hour_build  6  2 reactive   1 1 90m "the same straggler under reactive: offloaded at submit, the small builds kept local"
fi

echo
[ "$FAILS" = 0 ] && echo "== Result: all live runs completed (agreement is per round above; a disagreement is a finding, not a failure)" || { echo "== Result: FAIL ($FAILS configuration(s) did not complete)"; exit 1; }
