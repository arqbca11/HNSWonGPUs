#!/usr/bin/env bash
# Run shard-simulator rounds FROM THE LAPTOP against the scheduler on the kind
# cluster, the way a database fleet outside the cluster would (Phase 2
# decision 26). Scales the worker Deployment, runs the rounds, prints the
# timeline, the round rows from the API and the relevant pod logs.
#
#   scripts/k8s-round.sh                                 # 6 shards, 2 workers, skewed, 1 round
#   N_SHARDS=50 WORKERS=4 ROUNDS=2 scripts/k8s-round.sh
#   KILL_WORKER=1 scripts/k8s-round.sh                   # delete the leaseholder of the largest leased build mid-round
#   KILL_SCHEDULER=1 scripts/k8s-round.sh                # delete the scheduler pod 1 s into the round; report the shards' stall
#
# Not a saved step log by itself; scripts/k8s-check-step4.sh calls it and saves.
set -euo pipefail
cd "$(dirname "$0")/.."
export SCHEDULER_URL="${SCHEDULER_URL:-http://127.0.0.1:30080}" SCENARIO="${SCENARIO:-skewed}" N_SHARDS="${N_SHARDS:-6}" SEED="${SEED:-1}" \
       ROUNDS="${ROUNDS:-1}" TIME_SCALE="${TIME_SCALE:-10}" LOG_FORMAT=text MIN_WORKERS="${WORKERS:-2}" ROUND_TIMEOUT="${ROUND_TIMEOUT:-10m}"
WORKERS="${WORKERS:-2}"; KILL_WORKER="${KILL_WORKER:-0}"; KILL_SCHEDULER="${KILL_SCHEDULER:-0}"
K="kubectl -n kgpu"
note() { printf '   %s\n' "$*"; }
jget() { python3 scripts/jget.py "$1"; }
SCRATCH="${TMPDIR:-/tmp}/kgpu-k8s-round.$$"; mkdir -p "$SCRATCH"; export TIMELINE_DIR="$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "== k8s round from outside the cluster: scenario=$SCENARIO shards=$N_SHARDS workers=$WORKERS seed=$SEED rounds=$ROUNDS time_scale=$TIME_SCALE kill_worker=$KILL_WORKER"
note "scheduler: $SCHEDULER_URL (NodePort); the simulator runs on this machine, not in a pod"
kubectl config use-context kind-kgpu >/dev/null
$K scale deployment/worker --replicas="$WORKERS" >/dev/null
$K rollout status deployment/worker --timeout=180s >/dev/null
for _ in $(seq 1 120); do [ "$(curl -s "$SCHEDULER_URL/workers" | jget "d['pool']['live_workers']")" = "$WORKERS" ] && break; sleep 0.5; done
note "pool: $(curl -s "$SCHEDULER_URL/workers" | jget "str(d['pool']['live_workers'])+' live: '+', '.join(w['worker_id'] for w in d['workers'] if w['live'])")"
# the TIME_SCALE must match the workers' FAKE_TIME_SCALE
WTS=$($K get configmap worker -o jsonpath='{.data.FAKE_TIME_SCALE}'); [ "$WTS" = "$TIME_SCALE" ] || { echo "!! TIME_SCALE=$TIME_SCALE but the worker ConfigMap has FAKE_TIME_SCALE=$WTS"; exit 2; }

T0=$(date +%s)
go run ./shard/cmd/shardsim > "$SCRATCH/sim.log" 2>&1 &
SIM=$!
KILLED=""
if [ "$KILL_WORKER" = 1 ]; then
  # wait for the round id, then for a leased build; strike the largest one.
  RID=""; for _ in $(seq 1 240); do RID=$(grep -o 'round_id=[0-9]*' "$SCRATCH/sim.log" 2>/dev/null | head -1 | cut -d= -f2 || true); [ -n "$RID" ] && break; sleep 0.25; done
  # prefer a build big enough to still be running when the deletion lands (>= 100k vectors is ~0.4 s at 10x);
  # after 5 s of waiting take the largest leased build there is.
  OWNER=""; MIN=100000
  for i in $(seq 1 240); do
    [ "$i" -gt 20 ] && MIN=0
    OWNER=$(curl -s "$SCHEDULER_URL/rounds/$RID" | jget "(lambda bs: (sorted(bs, key=lambda b: -b['n_vectors'])[0]['lease_owner'] if bs else ''))([b for b in d['builds'] if b['state']=='leased' and b['n_vectors']>=$MIN])" || true)
    [ -n "$OWNER" ] && break; sleep 0.25
  done
  if [ -n "$OWNER" ]; then
    BID=$(curl -s "$SCHEDULER_URL/rounds/$RID" | jget "[b['build_id'] for b in d['builds'] if b.get('lease_owner')=='$OWNER'][0]")
    note "+$(( $(date +%s) - T0 )) s: deleting worker pod $OWNER while it builds $BID"
    $K delete pod "$OWNER" --wait=false >/dev/null; KILLED="$OWNER"; KILLED_BUILD="$BID"
  else
    note "no leased build appeared to strike"
  fi
fi
SCHED_KILLED_AT=""
if [ "$KILL_SCHEDULER" = 1 ]; then
  for _ in $(seq 1 240); do grep -q 'round started' "$SCRATCH/sim.log" 2>/dev/null && break; sleep 0.25; done
  sleep 1
  SP=$($K get pods -l app=scheduler -o jsonpath='{.items[0].metadata.name}')
  SCHED_KILLED_AT=$(date +%s.%N)
  $K delete pod "$SP" --wait=false >/dev/null
  note "+$(( $(date +%s) - T0 )) s: deleted scheduler pod $SP mid-round"
  for _ in $(seq 1 120); do NP=$($K get pods -l app=scheduler --field-selector=status.phase=Running -o jsonpath='{.items[?(@.status.containerStatuses[0].ready==true)].metadata.name}' 2>/dev/null | tr ' ' '\n' | grep -v "^$SP$" | head -1 || true); [ -n "$NP" ] && break; sleep 0.1; done
  note "replacement $NP ready $(python3 -c "import time;print(round(time.time()-$SCHED_KILLED_AT,1))") s after the deletion"
  sleep 1; note "the deleted pod's last lines while terminating: $($K logs "$SP" --tail=3 2>/dev/null | python3 -c 'import sys,json
out=[]
for l in sys.stdin:
    try: d=json.loads(l); out.append(d["msg"]+(" delay="+str(d["delay"]) if "delay" in d else ""))
    except Exception: pass
print(" | ".join(out) or "(gone)")')"
fi
wait "$SIM" && SIM_EXIT=0 || SIM_EXIT=$?
note "simulator exited $SIM_EXIT after $(( $(date +%s) - T0 )) s"

if [ -n "$SCHED_KILLED_AT" ]; then
  echo; echo "== the shards' view of the scheduler deletion (retry warnings in the simulator log)"
  python3 - "$SCRATCH/sim.log" "$SCHED_KILLED_AT" <<'PY'
import sys, re, datetime
lines = [l for l in open(sys.argv[1]) if "scheduler call failed, retrying" in l or "succeeded after retries" in l]
killed = float(sys.argv[2])
def ts(l):
    m = re.match(r'time=(\S+)', l); t = m.group(1)
    t = re.sub(r'(\.\d{6})\d*', r'\1', t)
    return datetime.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
fails = [l for l in lines if "failed" in l]
kinds = {}
for l in fails:
    k = ("EOF" if "EOF" in l else "timeout" if ("Timeout" in l or "deadline" in l or "timeout" in l) else
         "refused" if "refused" in l else "reset" if "reset" in l else "other")
    kinds[k] = kinds.get(k, 0) + 1
print("   %d retry warnings: %s" % (len(fails), ", ".join("%s x%d" % kv for kv in kinds.items())))
if fails:
    rel = [ts(l) - killed for l in fails]
    print("   first %.1f s, last %.1f s after the deletion; the shards were stalled for about %.1f s" % (min(rel), max(rel), max(rel)))
    try:
        from statistics import median
        print("   (the Service dropped the pod and the replacement was ready within ~1.5 s; anything beyond that is the stall this measures)")
    except Exception: pass
PY
fi
echo; echo "== timeline (simulator output)"
grep -vE '^time=' "$SCRATCH/sim.log" | sed 's/^/   /'
echo; echo "== round events (simulator log)"
grep -E 'round started|round finished|aborted|attempt=[2-9]|lost|ERROR|failed' "$SCRATCH/sim.log" | sed 's/^time=[^ ]* //' | sed 's/^/   /' | head -40
echo; echo "== round rows (from the scheduler API)"
for RID in $(grep -o 'round_id=[0-9]*' "$SCRATCH/sim.log" | cut -d= -f2 | sort -un); do
  curl -s "$SCHEDULER_URL/rounds/$RID" > "$SCRATCH/round.json"
  python3 - "$SCRATCH/round.json" <<'PY'
import sys, json
r = json.load(open(sys.argv[1])); rd = r["round"]; bs = r["builds"]; jobs = r["jobs"]
print("   round %s: n_shards=%s finished=%s" % (rd["round_id"], rd["n_shards"], rd["finished_at"]))
print("   builds: " + ", ".join("%s=%s/%s@%s" % (b["build_id"], b["placement"], b["state"], b["attempt"]) for b in bs[:12]) + (" ..." if len(bs) > 12 else ""))
print("   states: done=%d other=%d; attempts>1: %d" % (sum(1 for b in bs if b["state"] == "done"), sum(1 for b in bs if b["state"] != "done"), sum(1 for b in bs if b["attempt"] > 1)))
print("   queries recorded: %d, finished: %d" % (len(jobs), sum(1 for j in jobs if j["finished_at"])))
PY
done
if [ -n "$KILLED" ]; then
  echo; echo "== the deleted worker $KILLED (its last lines) and the build it held"
  $K logs "$KILLED" 2>/dev/null | grep -E 'shutdown|release|finish|deregister' | sed 's/^/   /' | tail -5 || note "(pod log gone)"
  curl -s "$SCHEDULER_URL/builds/$KILLED_BUILD" | jget "'   '+d['build_id']+': '+d['state']+' at attempt '+str(d['attempt'])+' (owner now '+str(d.get('lease_owner'))+')'"
  $K rollout status deployment/worker --timeout=180s >/dev/null
  note "workers now: $($K get pods -l app=worker --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.metadata.name}@{.spec.nodeName} {end}')"
  for p in $($K get pods -l app=worker --field-selector=status.phase=Running -o name); do
    note "${p#pod/}: $($K logs "$p" 2>/dev/null | grep -c '"msg": "claimed"' || true) claims, $($K logs "$p" 2>/dev/null | grep -c '"msg": "completed"' || true) completions in its lifetime (a replacement that claimed joined the round in flight)"
  done
fi
echo; echo "== scheduler log since the run started (counts, then everything that is not a placement)"
$K logs deploy/scheduler --since-time="$(date -u -r "$T0" +%Y-%m-%dT%H:%M:%SZ)" 2>/dev/null | python3 -c '
import sys, json, collections
c = collections.Counter(); other = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    c[d.get("msg")] += 1
    if d.get("msg") != "build placed":
        other.append("   " + str(d.get("level")) + " " + str(d.get("msg")) + " " + " ".join("%s=%s" % (k, v) for k, v in d.items() if k not in ("time", "level", "msg")))
print("   " + ", ".join("%s x%d" % (m, n) for m, n in c.most_common()))
print("\n".join(other[-20:]))'
echo
[ "$SIM_EXIT" = 0 ] && echo "== Result: PASS" || { echo "== Result: FAIL (simulator exit $SIM_EXIT)"; exit 1; }
