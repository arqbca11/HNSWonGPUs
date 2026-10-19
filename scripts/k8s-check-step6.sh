#!/usr/bin/env bash
# Phase 2 step 6 check: the failure injections, watched on the dashboard.
# Phase 2's done-when is "the dashboard shows a worker pod deleted mid-round
# and the round still completing"; this script runs the three injections and
# reads every number THROUGH GRAFANA'S DATASOURCE PROXY, so what this log
# records is what http://127.0.0.1:3000/d/kgpu-round showed. Three runs:
#   1. a worker pod force-deleted mid-round (grace 0): the lease expires on
#      the dashboard (lease_expirations panel +1), the killed pod's worker_busy
#      series ends, and the round duration still lands (the done-when)
#   2. the scheduler pod deleted mid-round: the shards stall ~1 s (bug 10),
#      the replacement pod closes the round and observes it
#   3. both GPU nodes drained mid-round: the pool empties while the queue
#      holds, uncordon refills it, the round completes
# Saves test-logs/phase2/step6-failures-dashboard.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log="test-logs/phase2/step6-failures-dashboard.log"; mkdir -p test-logs/phase2
exec > >(tee "$log") 2>&1
kubectl config use-context kind-kgpu >/dev/null
K="kubectl -n kgpu"
say() { printf '\n== %s\n' "$*"; }; note() { printf '   %s\n' "$*"; }
result=PASS; reached_end=0; fail() { printf '\n!! FAIL: %s\n' "$*"; result=FAIL; }
PIDS=""; RUNS="${TMPDIR:-/tmp}/kgpu-step6.$$"; mkdir -p "$RUNS"
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null || true; done; rm -rf "$RUNS"
  kubectl uncordon $(kubectl get nodes -l kgpu.io/gpu=true -o jsonpath='{.items[*].metadata.name}') >/dev/null 2>&1 || true
  if [ "$($K get configmap worker -o jsonpath='{.data.FAKE_TIME_SCALE}' 2>/dev/null)" != 10 ]; then
    $K patch configmap worker --type merge -p '{"data":{"FAKE_TIME_SCALE":"10"}}' >/dev/null 2>&1 || true
    $K rollout restart deployment/worker >/dev/null 2>&1 || true
  fi
  $K scale deployment/worker --replicas=2 >/dev/null 2>&1 || true
  [ "$reached_end" = 1 ] || result="ABORTED (script error before the end)"; say "Result: $result"; echo "log: $log"; [ "$result" = PASS ]; }
trap cleanup EXIT
pf() { kubectl -n "$1" port-forward "svc/$2" "$3" >/dev/null 2>&1 & PIDS="$PIDS $!"; disown; for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:${3%%:*}$4" >/dev/null 2>&1 && return 0; sleep 0.25; done; return 1; }
GRAF=http://127.0.0.1:13000; AUTH="admin:kgpu"

# every query below goes through Grafana's datasource proxy: Grafana forwards
# it to the same Prometheus the dashboard panels read, under Grafana's auth.
gq() { local v=""; for _ in 1 2 3; do v=$(curl -s -u "$AUTH" "$DSPROXY/api/v1/query" --data-urlencode "query=$1" | python3 -c 'import sys,json
try: r=json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else "")
except Exception: print("")'); [ -n "$v" ] && break; sleep 1; done; echo "$v"; }
show_range() { # expr start end: one line per series, 5 s steps, as the panel drew it
  curl -s -u "$AUTH" "$DSPROXY/api/v1/query_range" --data-urlencode "query=$1" --data-urlencode "start=$2" --data-urlencode "end=$3" --data-urlencode "step=5" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)["data"]["result"]
except Exception: d = []
if not d: print("   (no series)")
for s in d:
    lbl = {k: v for k, v in s["metric"].items() if k not in ("__name__","endpoint","service","namespace","container","instance","job")}
    print("   %s  %s" % (lbl or "{}", " ".join("%g" % float(v) for _, v in s["values"])))'
}

say "Phase 2 step 6: the failure injections, watched on the dashboard  ($(date '+%Y-%m-%d %H:%M:%S %Z'), commit $(git rev-parse --short HEAD))"

say "0. Grafana is up; its Prometheus datasource answers; nothing else holds the simulator's :9091"
pf monitoring kps-grafana 13000:80 /api/health || fail "grafana port-forward"
DSUID=$(curl -s -u "$AUTH" "$GRAF/api/datasources" | python3 -c 'import sys,json;print([d["uid"] for d in json.load(sys.stdin) if d["type"]=="prometheus"][0])')
DSPROXY="$GRAF/api/datasources/proxy/uid/$DSUID"
note "datasource uid: $DSUID (all numbers below come through this proxy)"
[ -n "$DSUID" ] || fail "no prometheus datasource in Grafana"
if lsof -nP -iTCP:9091 -sTCP:LISTEN >/dev/null 2>&1; then fail ":9091 is busy; stop the running simulator first"; reached_end=1; exit 1; fi
curl -s -u "$AUTH" "$GRAF/api/dashboards/uid/kgpu-round" >/dev/null || fail "dashboard kgpu-round not reachable"

say "1. A worker pod force-deleted mid-round: the dashboard shows the lease expiring and the round still completing (the phase's done-when)"
note "fake builds run 5x slower for this run (FAKE_TIME_SCALE 10 -> 2, restored after): at 10x the largest build sleeps ~0.6 s and the deletion usually lands between builds; at 2x it sleeps ~3 s, a window the kill cannot miss"
$K patch configmap worker --type merge -p '{"data":{"FAKE_TIME_SCALE":"2"}}' >/dev/null
$K rollout restart deployment/worker >/dev/null; $K rollout status deployment/worker --timeout=180s >/dev/null
LQ0=$(gq 'sum(lease_expirations_total{outcome="queued"})'); LQ0=${LQ0:-0}
RC0=$(gq 'max(round_duration_seconds_count)'); RC0=${RC0:-0}
note "before: lease_expirations{queued}=$LQ0, rounds observed=$RC0"
T1=$(date +%s)
env N_SHARDS=12 WORKERS=2 SEED=2 TIME_SCALE=2 KILL_WORKER=1 KILL_FORCE=1 scripts/k8s-round.sh > "$RUNS/run1.log" 2>&1 || fail "run 1: k8s-round.sh failed"
sed 's/^/   | /' "$RUNS/run1.log" | grep -vE '^\s+\| time=' | head -60
T1E=$(( $(date +%s) + 10 )); sleep 10   # one reaper tick and two scrapes past the round
KP=$(grep -o 'force-deleting worker pod [^ ]*' "$RUNS/run1.log" | awk '{print $4}')
note "the force-deleted pod: ${KP:-NONE FOUND}"
[ -n "$KP" ] || fail "run 1 never struck a worker"
LQ1=$(gq 'sum(lease_expirations_total{outcome="queued"})'); RC1=$(gq 'max(round_duration_seconds_count)')
note "after: lease_expirations{queued}=$LQ0 -> $LQ1 (the panel ticked +1), rounds observed=$RC0 -> $RC1 (the round still landed)"
[ "$(( ${LQ1%.*} - ${LQ0%.*} ))" = 1 ] || fail "lease_expirations{queued} did not grow by exactly 1"
[ "$(( ${RC1%.*} - ${RC0%.*} ))" = 1 ] || fail "round_duration_seconds_count did not grow by 1: the round did not land on the dashboard"
grep -qE 'done at attempt [2-9]' "$RUNS/run1.log" && note "the struck build: $(grep -oE '[0-9]+:[0-9]+: done at attempt [0-9]+ \(owner now [^)]*\)' "$RUNS/run1.log")" || fail "the struck build did not complete at attempt 2"
[ -z "$(gq "count(worker_busy{pod=\"$KP\"})")" ] && note "the killed pod's worker_busy series has ended (stale, off the panel)" || fail "the killed pod still has a live worker_busy series"
WB=$(gq 'count(count by (pod) (worker_busy))'); note "worker_busy series now: $WB pods (the survivor and the replacement)"
[ "${WB%.*}" = 2 ] || fail "expected worker_busy from exactly 2 pods, got $WB"
note "panel 'sum(worker_busy)' over the run window (5 s steps):"; show_range 'sum(worker_busy)' "$T1" "$T1E"
note "panel 'max(gpu_workers_live)' over the window (the killed worker counts until WORKER_STALE_AFTER=15s passes; the replacement registers early, so a brief 3 then back to 2):"
show_range 'max(gpu_workers_live)' "$T1" "$T1E"
note "panel 'sum(lease_expirations_total) by (outcome)' over the window:"; show_range 'sum(lease_expirations_total) by (outcome)' "$T1" "$T1E"
note "restoring FAKE_TIME_SCALE=10 on the workers"
$K patch configmap worker --type merge -p '{"data":{"FAKE_TIME_SCALE":"10"}}' >/dev/null
$K rollout restart deployment/worker >/dev/null; $K rollout status deployment/worker --timeout=180s >/dev/null

say "2. The scheduler pod deleted mid-round: the shards stall about a second (bug 10), the replacement closes the round and the dashboard keeps the story"
T2=$(date +%s)
env N_SHARDS=50 WORKERS=2 SEED=4 KILL_SCHEDULER=1 scripts/k8s-round.sh > "$RUNS/run2.log" 2>&1 || fail "run 2: k8s-round.sh failed"
sed 's/^/   | /' "$RUNS/run2.log" | grep -vE '^\s+\| time=' | head -60
T2E=$(( $(date +%s) + 10 )); sleep 10
NP=$(grep -o 'replacement [^ ]* ready' "$RUNS/run2.log" | awk '{print $2}')
note "the replacement scheduler pod: ${NP:-NONE FOUND}"
[ -n "$NP" ] || fail "run 2 never reported a replacement scheduler pod"
STALL=$(grep -o 'stalled for about [0-9.]*' "$RUNS/run2.log" | grep -o '[0-9.]*' || true)
RW=$(grep -oE '[0-9]+ retry warnings' "$RUNS/run2.log" | grep -oE '^[0-9]+' || echo "")
if [ -n "$STALL" ]; then
  note "the shards' stall: $STALL s (bug 10 regression: must stay ~1 s, not the old 10 s)"
  python3 -c "exit(0 if float('$STALL') < 3 else 1)" || fail "stall was $STALL s, expected < 3 s"
elif [ "$RW" = 0 ]; then
  note "no shard call failed at all: the shutdown delay (bug 10's fix) kept the dying pod serving while the endpoints switched"
else
  fail "retry warnings ($RW) but no stall measurement in the run log"
fi
RCN=$(gq "round_duration_seconds_count{pod=\"$NP\"}")
note "rounds observed by the replacement pod: ${RCN:-0} (its counters started at 0, decision 35, and the round it closed is there)"
[ "${RCN%.*}" = 1 ] || fail "the replacement scheduler pod observed ${RCN:-0} rounds, expected exactly 1"
note "the scheduler target through the window (up{job=\"scheduler\"}; the old pod's series ends, the new one's begins):"
show_range 'up{job="scheduler"}' "$T2" "$T2E"

say "3. Both GPU nodes drained mid-round: the pool empties on the dashboard while the queue holds, uncordon refills it, the round completes"
T3=$(date +%s)
env N_SHARDS=50 WORKERS=2 SEED=3 DRAIN_GPU=1 DRAIN_HOLD=20 scripts/k8s-round.sh > "$RUNS/run3.log" 2>&1 || fail "run 3: k8s-round.sh failed"
sed 's/^/   | /' "$RUNS/run3.log" | grep -vE '^\s+\| time=' | head -60
T3E=$(( $(date +%s) + 10 )); sleep 10
RC3=$(gq "round_duration_seconds_count{pod=\"$NP\"}")
note "rounds observed by the scheduler pod: 1 -> ${RC3:-?} (the drained round still landed)"
[ "${RC3%.*}" = 2 ] || fail "expected the scheduler pod to have observed 2 rounds after the drain run, got ${RC3:-0}"
note "panel 'max(gpu_workers_live)' over the window:"; show_range 'max(gpu_workers_live)' "$T3" "$T3E"
note "panel 'max(gpu_queue_depth)' over the window:"; show_range 'max(gpu_queue_depth)' "$T3" "$T3E"
VERDICT=$(python3 - "$DSPROXY" "$T3" "$T3E" <<'PY'
import sys, json, base64, urllib.request, urllib.parse
proxy, start, end = sys.argv[1:4]
def rng(q):
    data = urllib.parse.urlencode({"query": q, "start": start, "end": end, "step": "5"}).encode()
    req = urllib.request.Request(proxy + "/api/v1/query_range", data=data)
    req.add_header("Authorization", "Basic " + base64.b64encode(b"admin:kgpu").decode())
    r = json.load(urllib.request.urlopen(req))["data"]["result"]
    return dict(r[0]["values"]) if r else {}
live, queue = rng("max(gpu_workers_live)"), rng("max(gpu_queue_depth)")
zero = [t for t, v in live.items() if float(v) == 0]
held = [t for t in zero if float(queue.get(t, 0)) > 0]
final = float(sorted(live.items())[-1][1]) if live else -1
print("scrapes with 0 live workers: %d; of those with builds still queued: %d; live workers at the end: %g"
      % (len(zero), len(held), final))
sys.exit(0 if (zero and held and final == 2) else 1)
PY
) && note "$VERDICT" || { note "${VERDICT:-range query failed}"; fail "the dashboard never showed an empty pool holding a queue, or the pool did not recover to 2"; }

say "The roadmap's failure-injection list is now fully shown on the dashboard: worker kill (run 1, the done-when), scheduler kill (run 2), node drain (run 3); the Postgres restart was step 2's"
reached_end=1
