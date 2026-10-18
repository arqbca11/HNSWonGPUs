#!/usr/bin/env bash
# Phase 2 step 3b check: GPU cards as a countable resource. Proves: each GPU
# node advertises CARDS_PER_NODE of kgpu.io/gpu-card; every worker holds one;
# scaling the workers to the card total fills every card; one more stays
# Pending with "Insufficient kgpu.io/gpu-card"; nothing lands on a general
# node; the pool follows the worker count; scaling back frees the cards.
# Saves test-logs/phase2/step3b-gpu-cards.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log="test-logs/phase2/step3b-gpu-cards.log"; mkdir -p test-logs/phase2
exec > >(tee "$log") 2>&1
kubectl config use-context kind-kgpu >/dev/null
K="kubectl -n kgpu"
say() { printf '\n== %s\n' "$*"; }; note() { printf '   %s\n' "$*"; }
result=PASS; reached_end=0; fail() { printf '\n!! FAIL: %s\n' "$*"; result=FAIL; }
PF_PID=""; API=http://127.0.0.1:18091
pf() { $K port-forward svc/scheduler 18091:8080 >/dev/null 2>&1 & PF_PID=$!; disown
  for _ in $(seq 1 40); do curl -sf $API/livez >/dev/null 2>&1 && return 0; sleep 0.25; done; return 1; }
cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null || true
  $K scale deployment/worker --replicas=2 >/dev/null 2>&1 || true
  [ "$reached_end" = 1 ] || result="ABORTED (script error before the end)"; say "Result: $result"; echo "log: $log"; [ "$result" = PASS ]; }
trap cleanup EXIT
pods() { $K get pods -l app=worker -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,NODE:.spec.nodeName' --no-headers | sed 's/^/   /'; }
per_node() { $K get pods -l app=worker --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort | uniq -c | awk '{print "   "$2": "$1" worker(s)"}'; }
allocated() { kubectl describe node "$1" | awk '/Allocated resources/,0' | awk '$1=="kgpu.io/gpu-card"{print $2}'; }
live() { curl -s $API/workers | python3 scripts/jget.py "d['pool']['live_workers']"; }
wait_live() { for _ in $(seq 1 $(( $2 * 4 ))); do [ "$(live)" = "$1" ] && return 0; sleep 0.25; done; return 1; }

say "Phase 2 step 3b: GPU cards as a countable resource  ($(date '+%Y-%m-%d %H:%M:%S %Z'), commit $(git rev-parse --short HEAD))"
say "1. Every GPU node advertises its cards; general nodes advertise none"
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.metadata.labels.kgpu\.io/gpu,CARDS:.status.allocatable.kgpu\.io/gpu-card' --no-headers | sed 's/^/   /'
GPU_NODES=$(kubectl get nodes -l kgpu.io/gpu=true -o name | sed 's|node/||' | tr '\n' ' ')
TOTAL=0; for n in $GPU_NODES; do c=$(kubectl get node "$n" -o jsonpath='{.status.allocatable.kgpu\.io/gpu-card}'); [ -n "$c" ] || fail "$n advertises no cards"; TOTAL=$(( TOTAL + ${c:-0} )); done
for n in $(kubectl get nodes -l '!kgpu.io/gpu' -o name | sed 's|node/||'); do [ -z "$(kubectl get node "$n" -o jsonpath='{.status.allocatable.kgpu\.io/gpu-card}')" ] || fail "$n (not a GPU node) advertises cards"; done
note "GPU nodes: $GPU_NODES; total cards: $TOTAL"; [ "$TOTAL" -ge 2 ] || fail "need at least 2 cards for this check"

say "2. Each running worker holds exactly one card"
$K rollout status deployment/worker --timeout=120s >/dev/null; pods
for n in $GPU_NODES; do w=$($K get pods -l app=worker --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | grep -c "^$n$" || true); a=$(allocated "$n"); note "$n: $w worker(s), $a card(s) allocated"; [ "$w" = "${a:-0}" ] || fail "$n: workers and allocated cards differ"; done
for p in $($K get pods -l app=worker -o name); do r=$($K get "$p" -o jsonpath='{.spec.containers[0].resources.limits.kgpu\.io/gpu-card}'); [ "$r" = 1 ] || fail "$p requests $r cards, not 1"; done
pf || fail "port-forward failed"; note "pool: $(live) live workers"

say "3. Scale the workers to the card total ($TOTAL): every card is used, both nodes carry workers, none on a general node"
$K scale deployment/worker --replicas="$TOTAL" >/dev/null; $K rollout status deployment/worker --timeout=180s >/dev/null || fail "rollout to $TOTAL did not finish"
pods; per_node
for n in $GPU_NODES; do [ "$(allocated "$n")" = "$(kubectl get node "$n" -o jsonpath='{.status.allocatable.kgpu\.io/gpu-card}')" ] || fail "$n is not full"; done
for n in $($K get pods -l app=worker -o jsonpath='{.items[*].spec.nodeName}'); do case " $GPU_NODES " in *" $n "*) ;; *) fail "a worker landed on $n";; esac; done
wait_live "$TOTAL" 30 || fail "pool did not reach $TOTAL live workers (got $(live))"; note "pool: $(live) live workers"

say "4. One more worker than cards: it stays Pending, and the scheduler says why"
$K scale deployment/worker --replicas=$(( TOTAL + 1 )) >/dev/null; sleep 8
pods
PEND=$($K get pods -l app=worker --field-selector=status.phase=Pending -o name | head -1)
[ -n "$PEND" ] || fail "no Pending worker"
if [ -n "$PEND" ]; then
  msg=$($K get events --field-selector involvedObject.name="${PEND#pod/}",reason=FailedScheduling -o jsonpath='{.items[-1:].message}' | cut -c1-220)
  note "scheduler says: $msg"
  case "$msg" in *"Insufficient kgpu.io/gpu-card"*) ;; *) fail "the Pending reason does not cite the card resource";; esac
fi
[ "$(live)" = "$TOTAL" ] || fail "pool should still be $TOTAL, got $(live)"; note "pool: $(live) live workers (the Pending pod is not a worker yet)"

say "5. Scale back to 2: the extra pods go, the cards are freed, the pool follows"
$K scale deployment/worker --replicas=2 >/dev/null; $K rollout status deployment/worker --timeout=180s >/dev/null
for _ in $(seq 1 120); do [ "$($K get pods -l app=worker --no-headers | wc -l | tr -d ' ')" = 2 ] && break; sleep 0.5; done
pods; SUM=0; for n in $GPU_NODES; do a=$(allocated "$n"); SUM=$(( SUM + ${a:-0} )); done; note "cards allocated across GPU nodes: $SUM"; [ "$SUM" = 2 ] || fail "expected 2 cards allocated, got $SUM"
wait_live 2 30 || fail "pool did not return to 2 (got $(live))"; note "pool: $(live) live workers"
reached_end=1
