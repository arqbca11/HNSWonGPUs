#!/usr/bin/env bash
# Phase 2 step 5 check: metrics and dashboards. Proves: the scheduler, the
# workers and the simulator (on the laptop) expose the roadmap's metrics;
# Prometheus scrapes all three (the shards through the Service + EndpointSlice
# pointing at the laptop); after two rounds the numbers are there; Grafana has
# the dashboard. Saves test-logs/phase2/step5-monitoring.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log="test-logs/phase2/step5-monitoring.log"; mkdir -p test-logs/phase2
exec > >(tee "$log") 2>&1
kubectl config use-context kind-kgpu >/dev/null
K="kubectl -n kgpu"; M="kubectl -n monitoring"
say() { printf '\n== %s\n' "$*"; }; note() { printf '   %s\n' "$*"; }
result=PASS; reached_end=0; fail() { printf '\n!! FAIL: %s\n' "$*"; result=FAIL; }
PIDS=""
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null || true; done; [ "$reached_end" = 1 ] || result="ABORTED (script error before the end)"; say "Result: $result"; echo "log: $log"; [ "$result" = PASS ]; }
trap cleanup EXIT
pf() { kubectl -n "$1" port-forward "svc/$2" "$3" >/dev/null 2>&1 & PIDS="$PIDS $!"; disown; for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:${3%%:*}$4" >/dev/null 2>&1 && return 0; sleep 0.25; done; return 1; }
PROM=http://127.0.0.1:19090; GRAF=http://127.0.0.1:13000; API=http://127.0.0.1:30080
pq() { local v=""; for _ in 1 2 3; do v=$(curl -s "$PROM/api/v1/query" --data-urlencode "query=$1" | python3 -c 'import sys,json
try: r=json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else "")
except Exception: print("")'); [ -n "$v" ] && break; sleep 1; done; echo "$v"; }
pq_all() { curl -s "$PROM/api/v1/query" --data-urlencode "query=$1" | python3 -c 'import sys,json
for r in json.load(sys.stdin)["data"]["result"]: print("   ", {k:v for k,v in r["metric"].items() if k not in ("__name__","endpoint","service","namespace","container")}, r["value"][1])'; }
targets() { curl -s "$PROM/api/v1/targets" | python3 -c 'import sys,json
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    j=t["labels"].get("job",""); 
    if j in ("scheduler","worker","shards"): print("   %-10s %-22s %s" % (j, t["labels"].get("instance",""), t["health"]))' | sort; }

say "Phase 2 step 5: metrics and dashboards  ($(date '+%Y-%m-%d %H:%M:%S %Z'), commit $(git rev-parse --short HEAD))"
say "1. The scheduler exposes the roadmap's metrics at /metrics (through the NodePort)"
S=$(curl -s $API/metrics)
for m in round_duration_seconds shard_lag_seconds build_queue_wait_seconds placement_decisions_total lease_expirations_total gpu_queue_depth gpu_workers_live builds_leased rounds_open; do
  echo "$S" | grep -q "^$m" || fail "scheduler /metrics lacks $m"; done
echo "$S" | grep -E "^(gpu_queue_depth|gpu_workers_live|rounds_open|builds_leased) " | sed 's/^/   /'

say "2. Each worker exposes build_duration_seconds, worker_busy and builds_total on its probe port"
for p in $($K get pods -l app=worker --field-selector=status.phase=Running -o name); do
  W=$($K exec "$p" -- python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8081/metrics',timeout=3).read().decode())" 2>/dev/null)
  for m in build_duration_seconds worker_busy builds_total; do echo "$W" | grep -q "^$m" || fail "${p#pod/} /metrics lacks $m"; done
  note "${p#pod/}: $(echo "$W" | grep -E '^worker_busy|^builds_total\{outcome="done"\}' | tr '\n' ' ')"
done

say "3. Prometheus scrapes scheduler, workers and the shards on the laptop (Service shards -> Endpoints -> host.docker.internal:9091)"
pf monitoring kps-kube-prometheus-stack-prometheus 19090:9090 /-/ready || fail "prometheus port-forward"
note "Endpoints shards -> $($K get endpoints shards -o jsonpath='{.subsets[0].addresses[0].ip}:{.subsets[0].ports[0].port}') (the laptop, as the kind nodes see it)"
# a fresh install needs up to a minute before the operator has turned our ServiceMonitors into targets
for _ in $(seq 1 60); do [ "$(targets | grep -c " up$")" -ge 3 ] && break; sleep 2; done
note "targets before the rounds (shards is 'down' until a simulator runs: nothing listens on the laptop's 9091 yet):"; targets
C0=$(pq 'round_duration_seconds_count'); C0=${C0:-0}; P0=$(pq 'sum(placement_decisions_total)'); P0=${P0:-0}; B0=$(pq 'sum(build_duration_seconds_count{placement="gpu"})'); B0=${B0:-0}
note "before the rounds: round_duration_seconds_count=$C0 placement_decisions_total=$P0 gpu builds observed=$B0"
export SCHEDULER_URL=$API SCENARIO=skewed N_SHARDS=50 SEED=11 ROUNDS=2 ROUND_GAP=8s TIME_SCALE=10 LOG_FORMAT=text MIN_WORKERS=2 METRICS_ADDR=:9091
$K scale deployment/worker --replicas=2 >/dev/null; $K rollout status deployment/worker --timeout=120s >/dev/null
go run ./shard/cmd/shardsim > /tmp/kgpu-step5-sim.log 2>&1 &
SIM=$!
sleep 12   # into the first round: every target has been scraped at least once at 5 s interval
note "targets while a round runs:"; targets
targets | grep -q "^   scheduler .* up" || fail "scheduler target not up"
[ "$(targets | grep -c "^   worker .* up")" = 2 ] || fail "expected 2 worker targets up"
targets | grep -q "^   shards .* up" || fail "shards (laptop) target not up: Prometheus cannot reach the simulator's /metrics"
note "live: gpu_workers_live=$(pq gpu_workers_live) rounds_open=$(pq rounds_open) shards_active=$(pq shards_active) busy=$(pq 'sum(worker_busy)') queries/s=$(pq 'sum(rate(queries_total[30s]))')"
wait $SIM && note "simulator finished both rounds" || fail "simulator exit $?"
sleep 8  # one reaper tick and two scrapes after the last round

say "4. After two 50-shard rounds the numbers are in Prometheus"
C1=$(pq 'round_duration_seconds_count'); P1=$(pq 'sum(placement_decisions_total)')
note "round_duration_seconds_count: $C0 -> $C1 (expect +2); placement_decisions_total: $P0 -> $P1 (expect +100)"
[ "$(( ${C1%.*} - ${C0%.*} ))" = 2 ] || fail "round_duration_seconds_count did not grow by 2"
[ "$(( ${P1%.*} - ${P0%.*} ))" = 100 ] || fail "placement_decisions_total did not grow by 100"
note "last round duration (mean of all): $(pq 'round_duration_seconds_sum / round_duration_seconds_count') s; straggler lag mean: $(pq 'shard_lag_seconds_sum / shard_lag_seconds_count') s"
B1=$(pq 'sum(build_duration_seconds_count{placement="gpu"})')
note "GPU build durations observed by the workers: $B0 -> $B1 (expect +100)"
[ "$(( ${B1%.*} - ${B0%.*} ))" = 100 ] || fail "expected 100 more GPU build durations, got $(( ${B1%.*} - ${B0%.*} ))"
note "worker_busy reached 1 in the window: $(pq 'max(max_over_time(worker_busy[5m]))')"
[ "$(pq 'max(max_over_time(worker_busy[5m]))')" = 1 ] || fail "worker_busy never 1"
note "shard side (laptop) in the window: queries_total=$(pq 'sum(max_over_time(queries_total[5m]))') query_wait p99 needs_index=true: $(pq 'histogram_quantile(0.99, sum(rate(query_wait_seconds_bucket{needs_index="true"}[5m])) by (le))') s"
[ -n "$(pq 'sum(max_over_time(queries_total[5m]))')" ] || fail "no shard metrics reached Prometheus"
note "build_queue_wait p99 (observed at round end): $(pq 'histogram_quantile(0.99, sum(rate(build_queue_wait_seconds_bucket[5m])) by (le))') s"
note "lease_expirations_total by outcome:"; pq_all 'lease_expirations_total'

say "5. Grafana is up and has the KGPU dashboard (provisioned from the ConfigMap by the sidecar)"
pf monitoring kps-grafana 13000:80 /api/health || fail "grafana port-forward"
note "health: $(curl -s $GRAF/api/health | tr -d '\n ')"
D=$(curl -s "$GRAF/api/search?query=KGPU" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(" | ".join(d["uid"]+" "+d["title"] for d in r))')
note "dashboards: $D"; echo "$D" | grep -q "kgpu-round" || fail "dashboard kgpu-round not provisioned"
note "panels: $(curl -s "$GRAF/api/dashboards/uid/kgpu-round" | python3 -c 'import sys,json; d=json.load(sys.stdin)["dashboard"]; print(len(d["panels"]))')"
note "open with scripts/k8s-dashboards.sh -> http://127.0.0.1:3000/d/kgpu-round"
reached_end=1
