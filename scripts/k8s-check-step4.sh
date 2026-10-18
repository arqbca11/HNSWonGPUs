#!/usr/bin/env bash
# Phase 2 step 4 check: the shards run OUTSIDE the cluster (decision 26) and
# the whole stack completes rounds on kind. Three runs via k8s-round.sh:
#   1. 6 shards, 2 workers (the default configuration)
#   2. 50 shards, 2 workers, the leaseholder of the largest build deleted mid-round
#   3. 50 shards, 4 workers (every card)
# Saves test-logs/phase2/step4-shards-outside.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log="test-logs/phase2/step4-shards-outside.log"; mkdir -p test-logs/phase2
exec > >(tee "$log") 2>&1
result=PASS
run() { echo; echo "#### $1"; shift; "$@" || result=FAIL; }
echo "== Phase 2 step 4: shards outside the cluster  ($(date '+%Y-%m-%d %H:%M:%S %Z'), commit $(git rev-parse --short HEAD))"
echo "   Only the scheduler's NodePort (127.0.0.1:30080) is reachable from here: $(kubectl -n kgpu get svc scheduler -o jsonpath='{.spec.type} nodePort={.spec.ports[0].nodePort}')"
echo "   No shard pods exist in the cluster: $(kubectl -n kgpu get pods -l app=shards --no-headers 2>/dev/null | wc -l | tr -d ' ') found"
run "1. 6 shards, 2 workers" env N_SHARDS=6 WORKERS=2 scripts/k8s-round.sh
run "2. 50 shards, 2 workers, a worker pod deleted mid-round (50 shards so the round is long enough to strike)" env N_SHARDS=50 WORKERS=2 SEED=2 KILL_WORKER=1 scripts/k8s-round.sh
run "3. 50 shards, 4 workers (every card)" env N_SHARDS=50 WORKERS=4 SEED=3 scripts/k8s-round.sh
kubectl -n kgpu scale deployment/worker --replicas=2 >/dev/null
echo; echo "== Result: $result"; echo "log: $log"; [ "$result" = PASS ]
