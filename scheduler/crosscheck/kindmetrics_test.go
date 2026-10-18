package crosscheck

// Cross-check of Phase 2 step 5: metrics and dashboards, on the implementer's
// stack in namespaces kgpu and monitoring of the kind cluster kgpu.
//
// Spec: docs/roadmap.md Phase 2 "Metrics" (names, types, emitters);
// docs/design/phase2-kubernetes.md decisions 28 (shard metrics from the shard,
// scraped from outside), 31 (round metrics observed exactly once by whichever
// closer stamps the round: the reaper's tick or GET /rounds/{id}), 32 (scrape
// wiring: ServiceMonitors, the shards Service + Endpoints), 33 (dashboards by
// port-forward), 34 (5 s scrape interval), 35 (every series from the first
// scrape) and the "Step 5" implementation notes (17-panel dashboard uid
// kgpu-round, from ConfigMap grafana-dashboard-kgpu).
//
// Written without reading the implementation. Run with KGPU_KIND=1.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// The roadmap's Metrics table plus the extras it lists in its step-5 bullet.
var (
	kmSchedulerFamilies = map[string]string{
		"round_duration_seconds":    "histogram",
		"shard_lag_seconds":         "histogram",
		"build_queue_wait_seconds":  "histogram",
		"placement_decisions_total": "counter",
		"lease_expirations_total":   "counter",
		"gpu_queue_depth":           "gauge",
		"gpu_workers_live":          "gauge",
		"builds_leased":             "gauge",
		"rounds_open":               "gauge",
	}
	kmWorkerFamilies = map[string]string{
		"build_duration_seconds": "histogram",
		"worker_busy":            "gauge",
		"builds_total":           "counter",
	}
	kmShardFamilies = map[string]string{
		"build_duration_seconds":     "histogram",
		"query_wait_seconds":         "histogram",
		"queries_total":              "counter",
		"shard_queue_depth":          "gauge",
		"shards_active":              "gauge",
		"local_builds_aborted_total": "counter",
	}
)

// checkFamilies asserts every family is declared with the roadmap's type and
// has at least one sample (decision 35: present from the first scrape).
func checkFamilies(t *testing.T, who string, e expo, want map[string]string) {
	t.Helper()
	names := make([]string, 0, len(want))
	for n := range want {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		typ, ok := e.Types[n]
		if !ok {
			t.Errorf("[%s] no '# TYPE %s' line (roadmap Phase 2 Metrics says %s emits a %s)", who, n, who, want[n])
			continue
		}
		if typ != want[n] {
			t.Errorf("[%s] '# TYPE %s %s', roadmap says %s", who, n, typ, want[n])
			continue
		}
		ss := e.find(sampleName(n, typ), nil)
		if len(ss) == 0 {
			t.Errorf("[%s] %s declared but has no %s sample (decision 35: every series exists from the first scrape)", who, n, sampleName(n, typ))
			continue
		}
		t.Logf("[%s] # TYPE %s %s; labels %v; %d %s series", who, n, typ, e.labelKeys(sampleName(n, typ)), len(ss), sampleName(n, typ))
	}
}

func histCount(e expo, name string, l map[string]string) float64 {
	var s float64
	for _, x := range e.find(name+"_count", l) {
		s += x.Value
	}
	return s
}

func histSum(e expo, name string, l map[string]string) float64 {
	var s float64
	for _, x := range e.find(name+"_sum", l) {
		s += x.Value
	}
	return s
}

func sumOf(e expo, name string, l map[string]string) float64 {
	var s float64
	for _, x := range e.find(name, l) {
		s += x.Value
	}
	return s
}

type schedSnap struct {
	RDCount, RDSum, SLCount, SLSum, BQCount, Placements, LeaseQueued, LeaseFailed float64
}

func snapScheduler(t *testing.T) schedSnap {
	e := schedMetrics(t)
	return schedSnap{
		RDCount:     histCount(e, "round_duration_seconds", nil),
		RDSum:       histSum(e, "round_duration_seconds", nil),
		SLCount:     histCount(e, "shard_lag_seconds", nil),
		SLSum:       histSum(e, "shard_lag_seconds", nil),
		BQCount:     histCount(e, "build_queue_wait_seconds", nil),
		Placements:  sumOf(e, "placement_decisions_total", nil),
		LeaseQueued: sumOf(e, "lease_expirations_total", map[string]string{"outcome": "queued"}),
		LeaseFailed: sumOf(e, "lease_expirations_total", map[string]string{"outcome": "failed"}),
	}
}

func (s schedSnap) String() string {
	return fmt.Sprintf("round_duration count=%.0f sum=%.3f; shard_lag count=%.0f sum=%.3f; build_queue_wait count=%.0f; placement_decisions=%.0f; lease_expirations queued=%.0f failed=%.0f",
		s.RDCount, s.RDSum, s.SLCount, s.SLSum, s.BQCount, s.Placements, s.LeaseQueued, s.LeaseFailed)
}

// waitShardsTarget waits for a scrape of the shards target newer than after
// whose health is want, returning the time it was seen.
func waitShardsTarget(t *testing.T, prom *kmForward, want string, after time.Time, d time.Duration) (promTarget, time.Time) {
	t.Helper()
	var got promTarget
	var seen time.Time
	poll(t, d, 250*time.Millisecond, fmt.Sprintf("shards target health=%s with a scrape after %s", want, after.Format("15:04:05.000")), func() bool {
		ts := targetsOfJob(promTargets(t, prom), "shards")
		if len(ts) != 1 {
			return false
		}
		got = ts[0]
		if got.Health == want && got.LastScrape.After(after) {
			seen = time.Now()
			return true
		}
		return false
	})
	return got, seen
}

var kmFinishedRE = regexp.MustCompile(`msg="round finished" round_id=(\d+) duration=(\S+) duration_modeled=\S+ straggler_lag=(\S+)`)

type simFinished struct {
	ID       int64
	Duration float64
	Lag      float64
}

func parseFinished(t *testing.T, out string) []simFinished {
	t.Helper()
	var r []simFinished
	for _, m := range kmFinishedRE.FindAllStringSubmatch(out, -1) {
		id, _ := strconv.ParseInt(m[1], 10, 64)
		d, err1 := time.ParseDuration(m[2])
		l, err2 := time.ParseDuration(m[3])
		if err1 != nil || err2 != nil {
			t.Fatalf("parse round finished line %q: %v %v", m[0], err1, err2)
		}
		r = append(r, simFinished{id, d.Seconds(), l.Seconds()})
	}
	return r
}

// killWorkerPython SIGKILLs the worker's python process inside the container
// (found by walking /proc: python runs under tini, so it is not pid 1).
func killWorkerPython(t *testing.T, pod string) string {
	t.Helper()
	out := kc(t, "exec", "-n", koNS, pod, "-c", "worker", "--", "sh", "-c",
		`for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in python*kgpu_worker*) kill -9 ${p#/proc/}; echo "killed ${p#/proc/}: $c";; esac; done`)
	return strings.TrimSpace(out)
}

func restartCount(t *testing.T, pod string) int {
	p, ok := getPodNS(t, pod)
	if !ok {
		return -1
	}
	return p.restarts()
}

func getPodNS(t *testing.T, name string) (kPod, bool) {
	so, _, err := kcRun("", "get", "pod", name, "-n", koNS, "-o", "json")
	if err != nil {
		return kPod{}, false
	}
	var p kPod
	if json.Unmarshal([]byte(so), &p) != nil {
		return kPod{}, false
	}
	return p, true
}

func TestKindMetrics(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set")
	}
	if _, _, err := kcRun("", "get", "deployment", "scheduler", "-n", koNS); err != nil {
		t.Fatalf("no scheduler Deployment in %s: %v", koNS, err)
	}
	t.Cleanup(func() { koRestoreWorkers(t) })
	scale := koTimeScale(t)
	t.Logf("[setup] worker FAKE_TIME_SCALE=%s; the simulator runs with the same TIME_SCALE (shardsim header)", scale)
	if scale != "10" {
		t.Logf("[setup] note: the brief says TIME_SCALE must be 10; the ConfigMap says %s", scale)
	}
	koWaitPool(t, koReplicas, 3*time.Minute)
	prom := kmPortForward(t, kmPromSvc, kmPromPort, 9090, "/-/ready")

	// 1. Scheduler exposition: roadmap names and types.
	t.Run("SchedulerExposition", func(t *testing.T) {
		code, body, err := httpText(koURL + "/metrics")
		if err != nil || code != http.StatusOK {
			t.Fatalf("GET %s/metrics: %d %v", koURL, code, err)
		}
		t.Logf("[scheduler] GET %s/metrics -> 200, %d bytes", koURL, len(body))
		e, err := parseExpo(body)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		checkFamilies(t, "scheduler", e, kmSchedulerFamilies)
		// Labelled families: every combination pre-created (decision 35).
		if got := e.labelKeys("placement_decisions_total"); strings.Join(got, ",") != "placement,policy" {
			t.Errorf("placement_decisions_total labels %v, roadmap says {placement, policy}", got)
		}
		if pl := e.labelValues("placement_decisions_total", "placement"); strings.Join(pl, ",") != "gpu,local" {
			t.Errorf("placement_decisions_total placement values %v, want both gpu and local present (decision 35)", pl)
		}
		lk := e.labelKeys("lease_expirations_total")
		t.Logf("[scheduler] lease_expirations_total labels %v, outcome values %v (the roadmap table lists it unlabelled; the brief names outcome=\"queued\")", lk, e.labelValues("lease_expirations_total", "outcome"))
		if _, ok := e.val("lease_expirations_total", map[string]string{"outcome": "queued"}); !ok {
			t.Errorf("no lease_expirations_total{outcome=\"queued\"} series")
		}
		// The roadmap does not label the round histograms; no label is expected.
		for _, n := range []string{"round_duration_seconds", "shard_lag_seconds"} {
			if lk := e.labelKeys(n + "_count"); len(lk) != 0 {
				t.Logf("[scheduler] note: %s carries labels %v not in the roadmap", n, lk)
			}
		}
		t.Logf("[scheduler] build_queue_wait_seconds labels %v (roadmap: unlabelled)", e.labelKeys("build_queue_wait_seconds_count"))
		// Values are sane for gauges.
		live, _ := e.val("gpu_workers_live", nil)
		if live != float64(koReplicas) {
			t.Errorf("gpu_workers_live=%v with %d Ready workers", live, koReplicas)
		}
		for _, g := range []string{"gpu_queue_depth", "builds_leased", "rounds_open"} {
			v, _ := e.val(g, nil)
			t.Logf("[scheduler] %s=%v (idle stack)", g, v)
			if v < 0 {
				t.Errorf("%s negative: %v", g, v)
			}
		}
	})

	// 2. Worker exposition on the probe port 8081.
	t.Run("WorkerExposition", func(t *testing.T) {
		ws, _ := kwPods(t, koNS, "app=worker")
		if len(ws) != koReplicas {
			t.Fatalf("%d worker pods, want %d", len(ws), koReplicas)
		}
		for _, w := range ws {
			name := w.Metadata.Name
			e := mustWorkerMetrics(t, name)
			t.Logf("[worker %s] GET :8081/metrics (via the API server's pod proxy)", name)
			checkFamilies(t, "worker "+name, e, kmWorkerFamilies)
			if pl := e.labelValues("build_duration_seconds_count", "placement"); strings.Join(pl, ",") != "gpu" {
				t.Errorf("[worker %s] build_duration_seconds placement values %v, want exactly [gpu] (roadmap: placement=local comes from the shard)", name, pl)
			}
			if lk := e.labelKeys("builds_total"); strings.Join(lk, ",") != "outcome" {
				t.Errorf("[worker %s] builds_total labels %v, want {outcome}", name, lk)
			}
			t.Logf("[worker %s] builds_total outcomes pre-created: %v", name, e.labelValues("builds_total", "outcome"))
			if v, ok := e.val("worker_busy", nil); !ok || (v != 0 && v != 1) {
				t.Errorf("[worker %s] worker_busy=%v ok=%v, want 0 or 1", name, v, ok)
			}
		}
	})

	// 3. The shards target (decisions 28, 32) and decision 35 on the shard
	// side. The simulator is held before its first round (MIN_WORKERS above
	// the live pool), so the first scrapes see a process with no round yet.
	t.Run("ShardsTargetAndFirstScrape", func(t *testing.T) {
		if err := koDial("127.0.0.1:9091", 300*time.Millisecond); err == nil {
			t.Fatalf("something already listens on 127.0.0.1:9091; cannot test the down state")
		}
		tg, _ := waitShardsTarget(t, prom, "down", time.Now(), 15*time.Second)
		t.Logf("[prometheus] no simulator running: shards target %s", tg)
		roundsBefore := koLatestRound(t)
		env := koSimEnv("skewed", 6, 5001, 1, scale, koReplicas+1) // waits for 3 live workers: never starts a round
		sim := startShardsim(t, env)

		var first expo
		firstAt := poll(t, 10*time.Second, 50*time.Millisecond, "simulator /metrics on 127.0.0.1:9091 (default METRICS_ADDR) before its first round", func() bool {
			e, err := shardMetrics("127.0.0.1:9091")
			if err != nil {
				return false
			}
			first = e
			return true
		})
		t.Logf("[shards] /metrics answered %.2fs after start, no round started (%d round-started lines)", firstAt.Seconds(), len(koStartedRounds(sim)))
		checkFamilies(t, "shards (before the first round)", first, kmShardFamilies)
		if pl := first.labelValues("build_duration_seconds_count", "placement"); strings.Join(pl, ",") != "local" {
			t.Errorf("[shards] build_duration_seconds placement values %v, want exactly [local] (decision 35)", pl)
		}
		if ni := first.labelValues("query_wait_seconds_count", "needs_index"); strings.Join(ni, ",") != "false,true" {
			t.Errorf("[shards] query_wait_seconds needs_index values %v, want both false and true (decision 35)", ni)
		}
		if k := first.labelValues("queries_total", "kind"); len(k) == 0 {
			t.Errorf("[shards] queries_total has no kind series before the first round (decision 35)")
		} else {
			t.Logf("[shards] queries_total kinds %v", k)
		}
		if sh := first.labelValues("shard_queue_depth", "shard"); len(sh) != 6 {
			t.Errorf("[shards] shard_queue_depth has series for shards %v, want 6 (N_SHARDS=6 is known at startup)", sh)
		}

		_, upAt := waitShardsTarget(t, prom, "up", sim.start, 15*time.Second)
		upAfter := upAt.Sub(sim.start)
		t.Logf("[prometheus] shards target up %.1fs after the simulator started (5 s scrape interval: expected within 10 s)", upAfter.Seconds())
		if upAfter > 10*time.Second {
			t.Errorf("shards target took %.1fs to go up, want <= 10 s", upAfter.Seconds())
		}
		time.Sleep(time.Second)
		upTS := promFirstTS(t, prom, `up{job="shards"} == 1`, sim.start.Add(-time.Second), time.Now())
		fams := make([]string, 0, len(kmShardFamilies))
		for n := range kmShardFamilies {
			fams = append(fams, n)
		}
		sort.Strings(fams)
		for _, n := range fams {
			ts := promFirstTS(t, prom, fmt.Sprintf(`count(%s{job="shards"})`, sampleName(n, kmShardFamilies[n])), sim.start.Add(-time.Second), time.Now())
			if ts.IsZero() || ts.After(upTS) {
				t.Errorf("[prometheus] the first successful scrape of the shards (%s) holds no %s (decision 35: every series from the first scrape)", upTS.Format("15:04:05"), sampleName(n, kmShardFamilies[n]))
			} else {
				t.Logf("[prometheus] %s present from the first successful scrape (%s)", sampleName(n, kmShardFamilies[n]), upTS.Format("15:04:05"))
			}
		}
		if len(koStartedRounds(sim)) != 0 || koLatestRound(t) != roundsBefore {
			t.Errorf("the held simulator started a round")
		}
		_ = sim.cmd.Process.Kill()
		<-sim.done
		killedAt := time.Now()
		tg, downAt := waitShardsTarget(t, prom, "down", killedAt, 15*time.Second)
		t.Logf("[prometheus] simulator killed: shards target down %.1fs later: %s", downAt.Sub(killedAt).Seconds(), tg)
	})

	// 4. Round metrics from a simulator run (decision 31) in real seconds, and
	// the shard-side counts of the same run.
	t.Run("RoundMetricsMatchSimulator", func(t *testing.T) {
		if err := koDial("127.0.0.1:9091", 300*time.Millisecond); err == nil {
			t.Fatalf("something already listens on 127.0.0.1:9091")
		}
		s0 := snapScheduler(t)
		t.Logf("[scheduler] before: %s", s0)
		sim := startShardsim(t, koSimEnv("skewed", 6, 5101, 3, scale, 1))

		// Keep the last exposition before the simulator exits.
		var last expo
		var maxActive float64
		for !sim.exited() {
			if e, err := shardMetrics("127.0.0.1:9091"); err == nil {
				last = e
				if v, ok := e.val("shards_active", nil); ok && v > maxActive {
					maxActive = v
				}
			}
			time.Sleep(200 * time.Millisecond)
			if time.Since(sim.start) > 4*time.Minute {
				t.Fatalf("simulator still running after 4 min")
			}
		}
		if code := sim.wait(t, time.Second); code != 0 {
			t.Fatalf("simulator exit %d:\n%s", code, tail(koSimOut(sim), 30))
		}
		exitAt := time.Now()
		fin := parseFinished(t, koSimOut(sim))
		started := koStartedRounds(sim)
		t.Logf("[shards] simulator exited; rounds started %v, finished lines %+v", started, fin)
		if len(started) != 3 || len(fin) != 3 {
			t.Fatalf("want 3 rounds started and finished, got %d / %d", len(started), len(fin))
		}
		if last.Types != nil {
			q := sumOf(last, "queries_total", nil)
			w := histCount(last, "query_wait_seconds", nil)
			lb := histCount(last, "build_duration_seconds", map[string]string{"placement": "local"})
			ab := sumOf(last, "local_builds_aborted_total", nil)
			t.Logf("[shards] last exposition before exit: queries_total=%.0f query_wait count=%.0f build_duration{local} count=%.0f aborted=%.0f; max shards_active seen=%.0f; queue depth series %d",
				q, w, lb, ab, maxActive, len(last.find("shard_queue_depth", nil)))
			if q == 0 || w == 0 {
				t.Errorf("[shards] no queries counted in the last exposition")
			}
			if len(last.find("build_duration_seconds_count", map[string]string{"placement": "local"})) == 0 {
				t.Errorf("[shards] after 3 rounds with no local build, build_duration_seconds{placement=\"local\"} has no series at all (decision 35: a zero-count series, not a gap)")
			}
			if maxActive > 6 {
				t.Errorf("shards_active reached %.0f with N_SHARDS=6", maxActive)
			}
		}

		// The API's view of the same rounds.
		gpuBuilds, builds := 0, 0
		var apiDur float64
		for _, id := range started {
			r, code := koRound(t, id)
			if code != http.StatusOK || r.Round.FinishedAt == nil {
				t.Fatalf("round %d: code %d finished_at %v", id, code, r.Round.FinishedAt)
			}
			d := r.Round.FinishedAt.Sub(r.Round.StartedAt).Seconds()
			apiDur += d
			for _, b := range r.Builds {
				builds++
				if b.Placement == "gpu" {
					gpuBuilds++
				}
			}
			t.Logf("[api] round %d: finished_at-started_at=%.3fs, %d builds", id, d, len(r.Builds))
		}
		time.Sleep(5 * time.Second) // >= 2 reaper ticks, so a second observation would have happened
		s1 := snapScheduler(t)
		t.Logf("[scheduler] after: %s", s1)
		n := float64(len(started))
		check := func(what string, got, want float64) {
			if got != want {
				t.Errorf("%s grew by %.0f, want %.0f (decision 31: exactly once per round)", what, got, want)
			} else {
				t.Logf("[decision 31] %s grew by %.0f = %.0f", what, got, want)
			}
		}
		check("round_duration_seconds_count", s1.RDCount-s0.RDCount, n)
		check("shard_lag_seconds_count", s1.SLCount-s0.SLCount, n)
		check("build_queue_wait_seconds_count (one per GPU build)", s1.BQCount-s0.BQCount, float64(gpuBuilds))
		check("placement_decisions_total (one per build)", s1.Placements-s0.Placements, float64(builds))

		// Real seconds: the histogram sum against the API timestamps and the simulator's printout.
		dSum := s1.RDSum - s0.RDSum
		var simDur, simLag float64
		for _, f := range fin {
			simDur += f.Duration
			simLag += f.Lag
		}
		t.Logf("[real seconds] round_duration_seconds_sum grew by %.3f; API finished_at-started_at total %.3f; simulator printed total %.3f (modeled would be ~%sx larger)", dSum, apiDur, simDur, scale)
		if math.Abs(dSum-apiDur) > 0.05*n {
			t.Errorf("round_duration_seconds_sum delta %.3f differs from the API's round durations %.3f", dSum, apiDur)
		}
		if math.Abs(dSum-simDur) > 0.5*n {
			t.Errorf("round_duration_seconds_sum delta %.3f differs from the simulator's printed durations %.3f by more than 0.5 s per round", dSum, simDur)
		}
		lSum := s1.SLSum - s0.SLSum
		t.Logf("[real seconds] shard_lag_seconds_sum grew by %.3f; simulator's straggler_lag total %.3f", lSum, simLag)
		if math.Abs(lSum-simLag) > 0.5*n {
			t.Errorf("shard_lag_seconds_sum delta %.3f differs from the simulator's straggler_lag total %.3f by more than 0.5 s per round", lSum, simLag)
		}

		// Prometheus holds the scheduler's value.
		poll(t, 20*time.Second, 500*time.Millisecond, "Prometheus round_duration_seconds_count == scheduler's", func() bool {
			v, ok := promScalar(t, prom, `sum(round_duration_seconds_count{job="scheduler"})`)
			return ok && v == s1.RDCount
		})
		t.Logf("[prometheus] round_duration_seconds_count = %.0f, same as the scheduler's /metrics", s1.RDCount)

		tg, downAt := waitShardsTarget(t, prom, "down", exitAt, 15*time.Second)
		t.Logf("[prometheus] simulator gone: shards target down %.1fs after exit: %s", downAt.Sub(exitAt).Seconds(), tg)
	})

	// 4. Decision 31: a round closed by GET /rounds/{id} and a round closed by
	// the reaper are each observed exactly once.
	t.Run("RoundObservedOnceByEitherCloser", func(t *testing.T) {
		// (a) Close by GET: report stream_done only once the build is done, then
		// GET straight away, before the reaper's next tick (REAP_INTERVAL 2s).
		ra := kmNewRound(t, "xcheck-metrics-get", 1)
		sa := kmSubmit(t, ra, 0, 100000)
		kmWaitBuild(t, sa.BuildID, 60*time.Second, "done", isDone)
		s0 := snapScheduler(t)
		t.Logf("[scheduler] before closing round %d: %s", ra, s0)
		kmReportDone(t, ra, 0)
		var r ldRound
		code, _ := koGetRetry(t, fmt.Sprintf("/rounds/%d", ra), &r, 10*time.Second)
		s1 := snapScheduler(t)
		t.Logf("[api] report stream_done, then GET /rounds/%d -> %d finished_at=%v; scheduler now: %s", ra, code, r.Round.FinishedAt, s1)
		if r.Round.FinishedAt == nil {
			t.Fatalf("round %d not finished on the GET right after the last report", ra)
		}
		if s1.RDCount-s0.RDCount != 1 {
			t.Errorf("right after the GET that returned the round finished, round_duration_seconds_count grew by %.0f, want 1 (decision 31: the GET stamps and observes)", s1.RDCount-s0.RDCount)
		}
		for i := 0; i < 3; i++ {
			koGetRetry(t, fmt.Sprintf("/rounds/%d", ra), nil, 10*time.Second)
		}
		time.Sleep(5 * time.Second)
		s2 := snapScheduler(t)
		t.Logf("[scheduler] after 3 more GETs and 5 s of reaper ticks: %s", s2)
		if d := s2.RDCount - s0.RDCount; d != 1 {
			t.Errorf("GET-closed round %d observed %.0f times in round_duration_seconds, want 1", ra, d)
		}
		if d := s2.SLCount - s0.SLCount; d != 1 {
			t.Errorf("GET-closed round %d observed %.0f times in shard_lag_seconds, want 1", ra, d)
		}
		if d := s2.BQCount - s0.BQCount; d != 1 {
			t.Errorf("GET-closed round %d (1 GPU build) added %.0f build_queue_wait observations, want 1", ra, d)
		}
		if dur := r.Round.FinishedAt.Sub(r.Round.StartedAt).Seconds(); math.Abs((s2.RDSum-s0.RDSum)-dur) > 0.05 {
			t.Errorf("round_duration_seconds_sum grew by %.3f, round %d lasted %.3f s by its timestamps", s2.RDSum-s0.RDSum, ra, dur)
		}

		// (b) Close by the reaper: the last report goes in first, nobody GETs the round.
		rb := kmNewRound(t, "xcheck-metrics-reaper", 1)
		sb := kmSubmit(t, rb, 0, 100000)
		kmReportDone(t, rb, 0)
		t.Logf("[api] round %d: report stream_done before the build finishes; no GET /rounds/%d until the metric moves", rb, rb)
		kmWaitBuild(t, sb.BuildID, 60*time.Second, "done", isDone)
		took := poll(t, 15*time.Second, 200*time.Millisecond, "round_duration_seconds_count +1 from the reaper", func() bool {
			return snapScheduler(t).RDCount-s2.RDCount >= 1
		})
		t.Logf("[scheduler] reaper observed round %d %.1fs after its build was done", rb, took.Seconds())
		for i := 0; i < 3; i++ {
			koGetRetry(t, fmt.Sprintf("/rounds/%d", rb), nil, 10*time.Second)
		}
		time.Sleep(5 * time.Second)
		s3 := snapScheduler(t)
		t.Logf("[scheduler] after 3 GETs and 5 s more: %s", s3)
		if d := s3.RDCount - s2.RDCount; d != 1 {
			t.Errorf("reaper-closed round %d observed %.0f times, want 1", rb, d)
		}
		if d := s3.SLCount - s2.SLCount; d != 1 {
			t.Errorf("reaper-closed round %d: shard_lag observed %.0f times, want 1", rb, d)
		}
		if d := s3.BQCount - s2.BQCount; d != 1 {
			t.Errorf("reaper-closed round %d: build_queue_wait observed %.0f times, want 1", rb, d)
		}
		var rr ldRound
		koGetRetry(t, fmt.Sprintf("/rounds/%d", rb), &rr, 10*time.Second)
		if rr.Round.FinishedAt == nil {
			t.Errorf("round %d not finished", rb)
		} else if dur := rr.Round.FinishedAt.Sub(rr.Round.StartedAt).Seconds(); math.Abs((s3.RDSum-s2.RDSum)-dur) > 0.05 {
			t.Errorf("round_duration_seconds_sum grew by %.3f, round %d lasted %.3f s by its timestamps", s3.RDSum-s2.RDSum, rb, dur)
		}
	})

	// 5. SIGKILL a worker mid-build: lease_expirations_total{outcome="queued"}
	// +1 exactly; worker_busy follows the leaseholder.
	t.Run("LeaseExpiryAndWorkerBusy", func(t *testing.T) {
		ws, _ := kwPods(t, koNS, "app=worker")
		if len(ws) != 2 {
			t.Fatalf("%d workers, want 2", len(ws))
		}
		doneBefore := map[string]float64{}
		restartsBefore := map[string]int{}
		for _, w := range ws {
			e := mustWorkerMetrics(t, w.Metadata.Name)
			doneBefore[w.Metadata.Name] = sumOf(e, "builds_total", map[string]string{"outcome": "done"})
			restartsBefore[w.Metadata.Name] = w.restarts()
		}
		t.Logf("[workers] builds_total{done} before: %v; restarts %v", doneBefore, restartsBefore)
		r := kmNewRound(t, "xcheck-metrics-kill", 1)
		s := kmSubmit(t, r, 0, kcLongN)
		kmReportDone(t, r, 0)
		b := kmWaitBuild(t, s.BuildID, 30*time.Second, "leased", func(b xBuild) bool { return b.State == "leased" && b.LeaseOwner != nil })
		owner := *b.LeaseOwner
		other := ""
		for _, w := range ws {
			if w.Metadata.Name != owner {
				other = w.Metadata.Name
			}
		}
		poll(t, 10*time.Second, 300*time.Millisecond, "worker_busy=1 on the leaseholder "+owner, func() bool {
			e, ok := workerMetrics(t, owner)
			v, _ := e.val("worker_busy", nil)
			return ok && v == 1
		})
		eo, _ := workerMetrics(t, other)
		vo, _ := eo.val("worker_busy", nil)
		t.Logf("[workers] attempt %d leased by %s: worker_busy=1 there; idle %s worker_busy=%v", b.Attempt, owner, other, vo)
		if vo != 0 {
			t.Errorf("idle worker %s reports worker_busy=%v", other, vo)
		}
		s0 := snapScheduler(t)
		t.Logf("[scheduler] before the kill: %s", s0)
		t.Logf("[exec %s] %s", owner, killWorkerPython(t, owner))
		killedAt := time.Now()

		b2 := kmWaitBuild(t, s.BuildID, 60*time.Second, "re-leased at attempt 2", func(b xBuild) bool { return b.Attempt >= 2 && b.State == "leased" && b.LeaseOwner != nil })
		s1 := snapScheduler(t)
		t.Logf("[scheduler] %.1fs after the kill, attempt %d leased by %s: %s", time.Since(killedAt).Seconds(), b2.Attempt, b2.owner(), s1)
		if d := s1.LeaseQueued - s0.LeaseQueued; d != 1 {
			t.Errorf("lease_expirations_total{outcome=\"queued\"} grew by %.0f across one expired lease, want 1", d)
		}
		if b2.Attempt != 2 {
			t.Errorf("build reclaimed at attempt %d, want 2", b2.Attempt)
		}
		newOwner := *b2.LeaseOwner
		poll(t, 15*time.Second, 300*time.Millisecond, "worker_busy=1 on the attempt-2 leaseholder "+newOwner, func() bool {
			e, ok := workerMetrics(t, newOwner)
			v, _ := e.val("worker_busy", nil)
			return ok && v == 1 && b2.State == "leased"
		})
		t.Logf("[workers] worker_busy=1 on %s while attempt 2 runs", newOwner)
		bd := kmWaitBuild(t, s.BuildID, 90*time.Second, "done", isDone)
		if bd.Attempt != 2 {
			t.Errorf("completed at attempt %d, want 2", bd.Attempt)
		}
		poll(t, 10*time.Second, 300*time.Millisecond, "worker_busy=0 on every worker after the build", func() bool {
			for _, n := range []string{owner, other} {
				e, ok := workerMetrics(t, n)
				if !ok {
					return false
				}
				if v, _ := e.val("worker_busy", nil); v != 0 {
					return false
				}
			}
			return true
		})
		t.Logf("[workers] worker_busy=0 on %s and %s after done", owner, other)
		en := mustWorkerMetrics(t, newOwner)
		doneNow := sumOf(en, "builds_total", map[string]string{"outcome": "done"})
		rcNow := restartCount(t, newOwner)
		want := doneBefore[newOwner] + 1
		if rcNow != restartsBefore[newOwner] {
			want = 1 // its process restarted: counters start again from zero
		}
		t.Logf("[workers] %s builds_total{done}=%.0f (before %.0f, restarts %d -> %d); want %.0f", newOwner, doneNow, doneBefore[newOwner], restartsBefore[newOwner], rcNow, want)
		if doneNow != want {
			t.Errorf("builds_total{outcome=\"done\"} on the attempt-2 worker %s = %.0f, want %.0f", newOwner, doneNow, want)
		}
		if rc := restartCount(t, owner); rc != restartsBefore[owner]+1 {
			t.Errorf("killed worker %s restart count %d, want %d", owner, rc, restartsBefore[owner]+1)
		}
		poll(t, 15*time.Second, 300*time.Millisecond, "the kill round observed by the reaper", func() bool {
			return snapScheduler(t).RDCount-s0.RDCount >= 1
		})
		time.Sleep(12 * time.Second) // past another lease length: nothing else may expire
		s2 := snapScheduler(t)
		t.Logf("[scheduler] 12 s after completion: %s", s2)
		if d := s2.LeaseQueued - s0.LeaseQueued; d != 1 {
			t.Errorf("lease_expirations_total{queued} total growth %.0f, want 1", d)
		}
		if d := s2.LeaseFailed - s0.LeaseFailed; d != 0 {
			t.Errorf("lease_expirations_total{failed} grew by %.0f, want 0", d)
		}
		if d := s2.RDCount - s0.RDCount; d != 1 {
			t.Errorf("kill round observed %.0f times, want 1", d)
		}
		// Prometheus saw the expiry too.
		poll(t, 20*time.Second, 500*time.Millisecond, "Prometheus lease_expirations_total{outcome=queued} == scheduler's", func() bool {
			v, ok := promScalar(t, prom, `sum(lease_expirations_total{job="scheduler",outcome="queued"})`)
			return ok && v == s2.LeaseQueued
		})
		t.Logf("[prometheus] lease_expirations_total{outcome=\"queued\"} = %.0f", s2.LeaseQueued)
		koWaitPool(t, koReplicas, 2*time.Minute)
	})

	// 6. Scrape wiring (decisions 32, 34): targets, 5 s interval, and what the
	// wiring does not cover (a simulator on another METRICS_ADDR).
	t.Run("ScrapeWiring", func(t *testing.T) {
		ts := promTargets(t, prom)
		ws, _ := kwPods(t, koNS, "app=worker")
		var wantPods []string
		for _, w := range ws {
			wantPods = append(wantPods, w.Metadata.Name)
		}
		sort.Strings(wantPods)
		for _, job := range []string{"scheduler", "worker", "shards"} {
			jt := targetsOfJob(ts, job)
			for _, x := range jt {
				t.Logf("[prometheus] target %s", x)
				if x.ScrapeInterval != "5s" {
					t.Errorf("target %s scrapeInterval %q, want 5s (decision 34, ServiceMonitors)", x.ScrapeURL, x.ScrapeInterval)
				}
				if !strings.HasSuffix(x.ScrapeURL, "/metrics") {
					t.Errorf("target %s does not scrape /metrics", x.ScrapeURL)
				}
			}
			want := map[string]int{"scheduler": 1, "worker": koReplicas, "shards": 1}[job]
			if len(jt) != want {
				t.Errorf("job %s has %d targets, want %d", job, len(jt), want)
			}
		}
		var gotPods []string
		for _, x := range targetsOfJob(ts, "worker") {
			gotPods = append(gotPods, x.Labels["pod"])
			if !strings.HasSuffix(x.ScrapeURL, ":8081/metrics") {
				t.Errorf("worker target %s not on the probe port 8081", x.ScrapeURL)
			}
		}
		sort.Strings(gotPods)
		if strings.Join(gotPods, ",") != strings.Join(wantPods, ",") {
			t.Errorf("worker targets for pods %v, live worker pods %v", gotPods, wantPods)
		}
		epIP := strings.TrimSpace(kc(t, "get", "endpoints", "shards", "-n", koNS, "-o", "jsonpath={.subsets[0].addresses[0].ip}"))
		if sh := targetsOfJob(ts, "shards"); len(sh) == 1 && sh[0].ScrapeURL != fmt.Sprintf("http://%s:9091/metrics", epIP) {
			t.Errorf("shards target %s, want http://%s:9091/metrics (the hand-written Endpoints)", sh[0].ScrapeURL, epIP)
		}
		var cfg struct {
			Data struct {
				YAML string `json:"yaml"`
			} `json:"data"`
		}
		promGetJSON(t, prom, "/api/v1/status/config", nil, &cfg)
		if !regexp.MustCompile(`(?m)^\s*scrape_interval:\s*5s\s*$`).MatchString(cfg.Data.YAML) {
			t.Errorf("Prometheus config has no scrape_interval: 5s")
		}
		t.Logf("[prometheus] global config: %s", strings.TrimSpace(regexp.MustCompile(`(?s)global:.*?\n[a-z]`).FindString(cfg.Data.YAML)))

		// A second simulator on :9092 is not scraped: only 9091 is wired.
		if err := koDial("127.0.0.1:9091", 300*time.Millisecond); err == nil {
			t.Fatalf("something listens on 9091")
		}
		env := koSimEnv("skewed", 6, 5201, 3, scale, 1)
		env["METRICS_ADDR"] = ":9092"
		sim := startShardsim(t, env)
		poll(t, 10*time.Second, 100*time.Millisecond, "simulator /metrics on 127.0.0.1:9092", func() bool {
			_, err := shardMetrics("127.0.0.1:9092")
			return err == nil
		})
		t.Logf("[shards] a simulator with METRICS_ADDR=:9092 serves on 127.0.0.1:9092")
		deadline := time.Now().Add(12 * time.Second)
		scrapes := 0
		var lastSeen time.Time
		for time.Now().Before(deadline) && !sim.exited() {
			for _, x := range promTargets(t, prom) {
				if strings.Contains(x.ScrapeURL, ":9092") {
					t.Errorf("Prometheus has a target on :9092: %s", x)
				}
				if x.Labels["job"] == "shards" && x.LastScrape.After(sim.start) && x.LastScrape != lastSeen {
					lastSeen = x.LastScrape
					scrapes++
					t.Logf("[prometheus] shards target scraped while the :9092 simulator runs: %s", x)
					if x.Health == "up" {
						t.Errorf("shards target up with nothing on 9091")
					}
				}
			}
			time.Sleep(500 * time.Millisecond)
		}
		if scrapes == 0 {
			t.Logf("[prometheus] note: no shards scrape observed during the :9092 run")
		}
		if code := sim.wait(t, 3*time.Minute); code != 0 {
			t.Errorf("simulator exit %d", code)
		}
		t.Logf("[wiring] covered: scheduler, the worker pods, and one laptop address on port 9091. A simulator on any other METRICS_ADDR is invisible to Prometheus.")
	})

	// 7. The dashboard: provisioned, 17 panels, every panel query answers.
	t.Run("DashboardPanelsReturnData", func(t *testing.T) {
		d, meta := dashboardFromConfigMap(t)
		ps := flatPanels(d.Panels)
		t.Logf("[configmap] %s/%s %s: uid=%q title=%q, %d panels", koNS, kmDashCM, meta, d.UID, d.Title, len(ps))
		if d.UID != kmDashUID {
			t.Errorf("dashboard uid %q, want %q", d.UID, kmDashUID)
		}
		if len(ps) != 17 {
			t.Errorf("%d panels, want 17 (step 5 implementation notes)", len(ps))
		}

		graf := kmPortForward(t, kmGrafSvc, kmGrafPort, 80, "/api/health")
		code, body, err := httpText(graf.base() + "/api/dashboards/uid/" + kmDashUID)
		if err != nil || code != http.StatusOK {
			t.Fatalf("anonymous GET /api/dashboards/uid/%s -> %d %v %.200s", kmDashUID, code, err, body)
		}
		var gd struct {
			Dashboard dashboard `json:"dashboard"`
		}
		if err := json.Unmarshal([]byte(body), &gd); err != nil {
			t.Fatalf("decode Grafana dashboard: %v", err)
		}
		gps := flatPanels(gd.Dashboard.Panels)
		t.Logf("[grafana] anonymous viewer GET /api/dashboards/uid/%s -> 200, %d panels, title %q", kmDashUID, len(gps), gd.Dashboard.Title)
		if len(gps) != len(ps) {
			t.Errorf("Grafana serves %d panels, the ConfigMap has %d", len(gps), len(ps))
		}
		// Admin login and the Prometheus datasource.
		req, _ := http.NewRequest("GET", graf.base()+"/api/datasources", nil)
		req.SetBasicAuth("admin", "kgpu")
		resp, err := kmHTTP.Do(req)
		if err != nil {
			t.Fatalf("GET /api/datasources: %v", err)
		}
		var dss []struct {
			UID, Name, Type, URL string
		}
		_ = json.NewDecoder(resp.Body).Decode(&dss)
		resp.Body.Close()
		t.Logf("[grafana] admin/kgpu GET /api/datasources -> %d %+v", resp.StatusCode, dss)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("admin/kgpu login failed: %d", resp.StatusCode)
		}
		dsUID := ""
		for _, ds := range dss {
			if ds.Type == "prometheus" {
				dsUID = ds.UID
			}
		}

		end := time.Now()
		start := end.Add(-15 * time.Minute) // the dashboard's default range
		var empty []string
		for _, p := range ps {
			if len(p.Targets) == 0 {
				t.Errorf("panel %d %q has no query", p.ID, p.Title)
			}
			for _, tg := range p.Targets {
				series, pts, errText := promRange(t, prom, tg.Expr, start, end, 15*time.Second)
				gf := "-"
				if dsUID != "" {
					gf = grafanaQuery(t, graf, dsUID, tg.Expr)
				}
				t.Logf("[panel %2d %-6s %q] %s -> prometheus: %d series, %d points %s; grafana /api/ds/query: %s", p.ID, p.Type, p.Title, tg.Expr, series, pts, errText, gf)
				if errText != "" {
					t.Errorf("panel %d %q query %q errors: %s", p.ID, p.Title, tg.Expr, errText)
				} else if pts == 0 {
					empty = append(empty, fmt.Sprintf("panel %d %q: %s", p.ID, p.Title, tg.Expr))
				}
			}
		}
		if len(empty) > 0 {
			t.Errorf("%d panel queries return no data over the last 15 min after rounds ran:\n  %s", len(empty), strings.Join(empty, "\n  "))
		}
	})
}

// promFirstTS returns the timestamp of the first point expr has in [from, to]
// at 1 s resolution (zero if none). A 1 s step with Prometheus's lookback
// means the first point lands at the first scrape that holds the series.
func promFirstTS(t *testing.T, f *kmForward, expr string, from, to time.Time) time.Time {
	t.Helper()
	q := url.Values{
		"query": {expr},
		"start": {strconv.FormatInt(from.Unix(), 10)},
		"end":   {strconv.FormatInt(to.Unix()+1, 10)},
		"step":  {"1"},
	}
	var r struct {
		Data struct {
			Result []promSeries `json:"result"`
		} `json:"data"`
	}
	promGetJSON(t, f, "/api/v1/query_range", q, &r)
	var first time.Time
	for _, s := range r.Data.Result {
		if len(s.Values) == 0 {
			continue
		}
		sec, _ := s.Values[0][0].(float64)
		ts := time.Unix(int64(sec), 0)
		if first.IsZero() || ts.Before(first) {
			first = ts
		}
	}
	return first
}

// grafanaQuery runs expr through Grafana's datasource proxy (/api/ds/query)
// over the last 15 minutes and summarises the answer.
func grafanaQuery(t *testing.T, g *kmForward, dsUID, expr string) string {
	body := map[string]any{
		"from": "now-15m", "to": "now",
		"queries": []map[string]any{{
			"refId": "A", "datasource": map[string]string{"uid": dsUID, "type": "prometheus"},
			"expr": expr, "range": true, "intervalMs": 15000, "maxDataPoints": 100,
		}},
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", g.base()+"/api/ds/query", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.SetBasicAuth("admin", "kgpu")
	resp, err := kmHTTP.Do(req)
	if err != nil {
		return "error " + err.Error()
	}
	defer resp.Body.Close()
	var r struct {
		Results map[string]struct {
			Error  string `json:"error"`
			Frames []struct {
				Data struct {
					Values [][]any `json:"values"`
				} `json:"data"`
			} `json:"frames"`
		} `json:"results"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&r)
	a := r.Results["A"]
	rows := 0
	for _, f := range a.Frames {
		if len(f.Data.Values) > 0 {
			rows += len(f.Data.Values[0])
		}
	}
	return fmt.Sprintf("%d, %d frames, %d rows %s", resp.StatusCode, len(a.Frames), rows, a.Error)
}
