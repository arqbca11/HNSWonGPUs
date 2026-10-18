package crosscheck

// Helpers for the cross-check of Phase 2 step 4: the shards run OUTSIDE the
// kind cluster (decisions 26, 27). The shard simulator is started on the
// laptop as a black-box process (startShardsim), pointed at the NodePort
// address http://127.0.0.1:30080, and everything else is observed through
// that same address (the scheduler API), kubectl on namespace kgpu (pods,
// logs, the worker Deployment's replica count), and the simulator's stdout.
//
// Unlike the earlier kind cross-checks there is no private namespace: the
// implementer's stack in kgpu is the system under test. The tests only run
// rounds against it, scale its worker Deployment (restored to 2 in Cleanup)
// and delete its scheduler pod (the Deployment replaces it).

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/KubernetesGPU/experiments/workload"
)

const (
	koNS       = "kgpu"
	koURL      = "http://127.0.0.1:30080"
	koReplicas = 2 // the worker Deployment's replica count in deploy/k8s/worker.yaml
)

var koHTTP = &http.Client{Timeout: 5 * time.Second}

// koGet does GET koURL+path, decoding JSON into out when out != nil.
func koGet(path string, out any) (int, string, error) {
	resp, err := koHTTP.Get(koURL + path)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if out != nil && resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(b, out); err != nil {
			return resp.StatusCode, string(b), err
		}
	}
	return resp.StatusCode, strings.TrimSpace(string(b)), nil
}

// koGetRetry retries transport errors for up to d (a scheduler pod may be
// restarting).
func koGetRetry(t *testing.T, path string, out any, d time.Duration) (int, string) {
	t.Helper()
	deadline := time.Now().Add(d)
	for {
		code, raw, err := koGet(path, out)
		if err == nil {
			return code, raw
		}
		if time.Now().After(deadline) {
			t.Fatalf("GET %s%s: %v", koURL, path, err)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

func koWorkers(t *testing.T) xWorkers {
	t.Helper()
	var w xWorkers
	code, raw := koGetRetry(t, "/workers", &w, 30*time.Second)
	if code != http.StatusOK {
		t.Fatalf("GET /workers -> %d %s", code, raw)
	}
	return w
}

func koRound(t *testing.T, id int64) (ldRound, int) {
	t.Helper()
	var r ldRound
	code, _ := koGetRetry(t, "/rounds/"+strconv.FormatInt(id, 10), &r, 60*time.Second)
	return r, code
}

// koLatestRound returns the highest round id the scheduler serves (0 if none).
func koLatestRound(t *testing.T) int64 {
	t.Helper()
	var id int64 = 0
	for {
		code, _ := koGetRetry(t, "/rounds/"+strconv.FormatInt(id+1, 10), nil, 30*time.Second)
		if code != http.StatusOK {
			return id
		}
		id++
	}
}

// koDial reports whether a TCP connection to addr opens within d.
func koDial(addr string, d time.Duration) error {
	c, err := net.DialTimeout("tcp", addr, d)
	if err != nil {
		return err
	}
	c.Close()
	return nil
}

// ---- the worker Deployment ----

func koScaleWorkers(t *testing.T, n int) {
	t.Helper()
	kc(t, "scale", "deployment/worker", "-n", koNS, fmt.Sprintf("--replicas=%d", n))
	t.Logf("[kubectl] scale deployment/worker -n %s --replicas=%d", koNS, n)
}

// koWaitPool waits until n worker pods are Ready (none terminating) and the
// scheduler's pool counts exactly n live workers.
func koWaitPool(t *testing.T, n int, d time.Duration) {
	t.Helper()
	start := time.Now()
	if n > 0 {
		waitWorkers(t, koNS, n, d)
	} else {
		poll(t, d, time.Second, "no worker pods in kgpu", func() bool {
			live, term := kwPods(t, koNS, "app=worker")
			return len(live) == 0 && len(term) == 0
		})
	}
	poll(t, d, 500*time.Millisecond, fmt.Sprintf("pool live_workers=%d", n), func() bool {
		return koWorkers(t).Pool.LiveWorkers == n
	})
	t.Logf("[api] GET /workers: pool live_workers=%d after %.1fs", n, time.Since(start).Seconds())
}

// koRestoreWorkers is the Cleanup that puts the worker Deployment back at
// its manifest's replica count and waits for it, whatever the test did.
func koRestoreWorkers(t *testing.T) {
	so, se, err := kcRun("", "scale", "deployment/worker", "-n", koNS, fmt.Sprintf("--replicas=%d", koReplicas))
	t.Logf("[cleanup] scale deployment/worker --replicas=%d: err=%v %s%s", koReplicas, err, strings.TrimSpace(so), strings.TrimSpace(se))
	so, se, err = kcRun("", "rollout", "status", "deployment/worker", "-n", koNS, "--timeout=180s")
	t.Logf("[cleanup] rollout status worker: err=%v %s%s", err, strings.TrimSpace(so), strings.TrimSpace(se))
	so, se, err = kcRun("", "rollout", "status", "deployment/scheduler", "-n", koNS, "--timeout=180s")
	t.Logf("[cleanup] rollout status scheduler: err=%v %s%s", err, strings.TrimSpace(so), strings.TrimSpace(se))
}

// koTimeScale reads FAKE_TIME_SCALE from the worker ConfigMap: the simulator's
// TIME_SCALE must equal it (shardsim header).
func koTimeScale(t *testing.T) string {
	t.Helper()
	v := strings.TrimSpace(kc(t, "get", "configmap", "worker", "-n", koNS, "-o", "jsonpath={.data.FAKE_TIME_SCALE}"))
	if v == "" {
		t.Fatalf("worker ConfigMap has no FAKE_TIME_SCALE")
	}
	return v
}

// ---- the simulator ----

// koSimEnv is the environment for one simulator run from the laptop.
func koSimEnv(scenario string, nShards int, seed int64, rounds int, scale string, minWorkers int) map[string]string {
	return map[string]string{
		"SCHEDULER_URL": koURL,
		"SCENARIO":      scenario,
		"N_SHARDS":      strconv.Itoa(nShards),
		"SEED":          strconv.FormatInt(seed, 10),
		"ROUNDS":        strconv.Itoa(rounds),
		"TIME_SCALE":    scale,
		"MIN_WORKERS":   strconv.Itoa(minWorkers),
		"ROUND_TIMEOUT": "5m",
		"LOG_FORMAT":    "text",
	}
}

var koStartedRE = regexp.MustCompile(`msg="round started" round_id=(\d+)`)

// koSimOut is everything the simulator printed (it writes its log and the
// timeline to stdout; stderr is appended in case anything lands there).
func koSimOut(p *simProc) string { return p.stdout.String() + p.stderr.String() }

// koStartedRounds returns the round ids from the simulator's "round started" lines.
func koStartedRounds(p *simProc) []int64 {
	var ids []int64
	for _, m := range koStartedRE.FindAllStringSubmatch(koSimOut(p), -1) {
		id, _ := strconv.ParseInt(m[1], 10, 64)
		ids = append(ids, id)
	}
	return ids
}

// koWaitStarted waits for the simulator's n-th "round started" line.
func koWaitStarted(t *testing.T, p *simProc, n int, d time.Duration) int64 {
	t.Helper()
	var id int64
	poll(t, d, 100*time.Millisecond, fmt.Sprintf("simulator's round-started line #%d", n), func() bool {
		ids := koStartedRounds(p)
		if len(ids) >= n {
			id = ids[n-1]
			return true
		}
		if p.exited() {
			t.Fatalf("simulator exited (code %d) before round #%d started:\n%s", p.cmd.ProcessState.ExitCode(), n, tail(koSimOut(p), 30))
		}
		return false
	})
	return id
}

// koLines returns the simulator's lines matching re.
func koLines(p *simProc, re *regexp.Regexp) []string {
	return grepLines(koSimOut(p), re)
}

// koTimeline parses the text timeline the simulator prints for round id.
func koTimeline(t *testing.T, p *simProc, id int64) ldTimeline {
	t.Helper()
	lines := strings.Split(koSimOut(p), "\n")
	hdr := regexp.MustCompile(`^round ` + strconv.FormatInt(id, 10) + `\s+scenario=`)
	fin := regexp.MustCompile(`msg="round finished" round_id=` + strconv.FormatInt(id, 10) + `\s`)
	tl := ldTimeline{Rows: map[int]ldTLRow{}}
	for i, l := range lines {
		if fin.MatchString(l) {
			tl.Finished = strings.TrimSpace(l)
		}
		if !hdr.MatchString(l) {
			continue
		}
		tl.Header = strings.TrimSpace(l)
		if i+1 < len(lines) {
			tl.Summary = strings.TrimSpace(lines[i+1])
		}
		for _, r := range lines[i+1:] {
			if strings.TrimSpace(r) == "" && len(tl.Rows) > 0 {
				break
			}
			m := ldRowRE.FindStringSubmatch(strings.TrimSpace(r))
			if m == nil {
				continue
			}
			sh, _ := strconv.Atoi(m[1])
			n, _ := strconv.ParseInt(m[2], 10, 64)
			att, _ := strconv.Atoi(m[4])
			q, _ := strconv.Atoi(m[9])
			tl.Rows[sh] = ldTLRow{Shard: sh, NVectors: n, Place: m[3], Att: att, BDone: secs(m[7]), Finish: secs(m[8]), Queries: q, MaxWait: secs(m[10])}
		}
	}
	return tl
}

// ---- following the pods' logs in kgpu ----

// koCollector follows every worker and scheduler container in kgpu with
// `kubectl logs -f` as soon as it appears, so the lines of a pod that is
// deleted (a scheduler pod, a scaled-down worker) are not lost.
type koCollector struct {
	mu        sync.Mutex
	lines     []kwLine
	following map[string]bool
	cmds      []*exec.Cmd
	stopc     chan struct{}
	wg        sync.WaitGroup
}

func startKoCollector(t *testing.T) *koCollector {
	c := &koCollector{following: map[string]bool{}, stopc: make(chan struct{})}
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		tk := time.NewTicker(500 * time.Millisecond)
		defer tk.Stop()
		for {
			c.scan()
			select {
			case <-c.stopc:
				return
			case <-tk.C:
			}
		}
	}()
	t.Cleanup(c.stop)
	return c
}

func (c *koCollector) scan() {
	so, _, err := kcRun("", "get", "pods", "-n", koNS, "-l", "app in (worker,scheduler)", "-o",
		`jsonpath={range .items[*]}{.metadata.name}{" "}{.status.containerStatuses[0].containerID}{"\n"}{end}`)
	if err != nil {
		return
	}
	for _, l := range strings.Split(strings.TrimSpace(so), "\n") {
		f := strings.Fields(l)
		if len(f) != 2 {
			continue
		}
		key := f[0] + " " + f[1]
		c.mu.Lock()
		done := c.following[key]
		c.following[key] = true
		c.mu.Unlock()
		if done {
			continue
		}
		pod := f[0]
		cmd := exec.Command("kubectl", "--context", kContext, "-n", koNS, "logs", "-f", pod)
		out, err := cmd.StdoutPipe()
		if err != nil || cmd.Start() != nil {
			continue
		}
		c.mu.Lock()
		c.cmds = append(c.cmds, cmd)
		c.mu.Unlock()
		go func() {
			sc := bufio.NewScanner(out)
			sc.Buffer(make([]byte, 1<<20), 1<<20)
			for sc.Scan() {
				raw := sc.Text()
				var m map[string]any
				_ = json.Unmarshal([]byte(raw), &m)
				c.mu.Lock()
				c.lines = append(c.lines, kwLine{pod, raw, m})
				c.mu.Unlock()
			}
			_ = cmd.Wait()
		}()
	}
}

func (c *koCollector) stop() {
	select {
	case <-c.stopc:
		return
	default:
	}
	close(c.stopc)
	c.wg.Wait()
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, cmd := range c.cmds {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}
}

// completions returns, per build id, the worker "completed" lines (pod@attempt).
func (c *koCollector) completions() map[string][]kwLine {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := map[string][]kwLine{}
	// A container's log may be followed twice if kubectl re-lists it under a
	// new container id after a restart; dedupe on pod+raw line.
	seen := map[string]bool{}
	for _, l := range c.lines {
		if !isWorkerPod(l.Pod) || l.str("msg") != "completed" {
			continue
		}
		k := l.Pod + "\x00" + l.Raw
		if seen[k] {
			continue
		}
		seen[k] = true
		m[l.str("build_id")] = append(m[l.str("build_id")], l)
	}
	return m
}

// schedulerLines returns the scheduler lines whose level is one of levels.
func (c *koCollector) schedulerLines(levels ...string) []kwLine {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []kwLine
	for _, l := range c.lines {
		if !isSchedulerPod(l.Pod) {
			continue
		}
		lv := strings.ToUpper(l.str("level"))
		for _, want := range levels {
			if strings.HasPrefix(lv, want) {
				out = append(out, l)
			}
		}
	}
	return out
}

// ---- watching the cluster for shard pods ----

// koPodWatch samples every pod in the cluster once a second and records any
// that is not part of the control plane or GPU pool: decision 26 says the
// cluster holds only the scheduler, Postgres and the workers.
type koPodWatch struct {
	mu         sync.Mutex
	samples    int
	unexpected map[string]string
	seenKgpu   map[string]bool
	stopc      chan struct{}
	done       chan struct{}
}

func startKoPodWatch(t *testing.T) *koPodWatch {
	w := &koPodWatch{unexpected: map[string]string{}, seenKgpu: map[string]bool{}, stopc: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(w.done)
		for {
			w.sample()
			select {
			case <-w.stopc:
				return
			case <-time.After(time.Second):
			}
		}
	}()
	t.Cleanup(w.stop)
	return w
}

func (w *koPodWatch) sample() {
	so, _, err := kcRun("", "get", "pods", "-A", "-o",
		`jsonpath={range .items[*]}{.metadata.namespace}{" "}{.metadata.name}{" "}{.metadata.labels.app}{"\n"}{end}`)
	if err != nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.samples++
	for _, l := range strings.Split(strings.TrimSpace(so), "\n") {
		f := strings.Fields(l)
		if len(f) < 2 {
			continue
		}
		ns, name, app := f[0], f[1], ""
		if len(f) > 2 {
			app = f[2]
		}
		bad := strings.Contains(strings.ToLower(name), "shard")
		switch ns {
		case "kube-system", "local-path-storage":
		case koNS:
			w.seenKgpu[app] = true
			if app != "postgres" && app != "scheduler" && app != "worker" {
				bad = true
			}
		default:
			bad = true
		}
		if bad {
			w.unexpected[ns+"/"+name] = "app=" + app
		}
	}
}

func (w *koPodWatch) stop() {
	select {
	case <-w.stopc:
	default:
		close(w.stopc)
	}
	<-w.done
}

// report stops the watch and fails the test if any unexpected pod was seen.
func (w *koPodWatch) report(t *testing.T) {
	t.Helper()
	w.stop()
	w.mu.Lock()
	defer w.mu.Unlock()
	var apps []string
	for a := range w.seenKgpu {
		apps = append(apps, a)
	}
	sort.Strings(apps)
	t.Logf("[k8s] pod watch: %d samples of `kubectl get pods -A` during the run; apps seen in %s: %v; unexpected pods: %v",
		w.samples, koNS, apps, w.unexpected)
	if w.samples == 0 {
		t.Errorf("pod watch took no samples")
	}
	if len(w.unexpected) > 0 {
		t.Errorf("decision 26: the cluster holds only the control plane and the GPU pool; saw %v", w.unexpected)
	}
}

// ---- the round-completes-correctly property, observed from outside ----

// koVerifyRound checks round id through the NodePort and against the pods'
// logs and the simulator's own output:
//
//   - the round is stamped finished, holds n_shards builds, every one done;
//   - every GPU build has exactly one worker "completed" line, at the attempt
//     the row records (invariants 3, 4);
//   - every recorded query started and finished in order, no needs_index
//     query started before its build was done;
//   - the recorded queries are exactly the seeded workload's (invariant 10):
//     per shard count, needs_index and duration, and arrivals on schedule;
//   - every shard reported stream_done;
//   - the simulator logged "round finished" with the API's query count and a
//     timeline whose attempts match Postgres.
func koVerifyRound(t *testing.T, c *koCollector, p *simProc, id int64, scale float64, wantShards int, d time.Duration) ldRound {
	t.Helper()
	var r ldRound
	poll(t, d, 300*time.Millisecond, fmt.Sprintf("round %d stamped finished via %s", id, koURL), func() bool {
		var code int
		r, code = koRound(t, id)
		return code == http.StatusOK && r.Round.FinishedAt != nil
	})
	fin := *r.Round.FinishedAt
	t.Logf("[api %s] GET /rounds/%d: scenario=%s seed=%d n_shards=%d finished_at-started_at=%.2fs, %d builds, %d queries",
		koURL, id, r.Round.Scenario, r.Round.Seed, r.Round.NShards, fin.Sub(r.Round.StartedAt).Seconds(), len(r.Builds), len(r.Jobs))
	if r.Round.NShards != wantShards || len(r.Builds) != wantShards {
		t.Errorf("round %d: n_shards=%d with %d builds, want %d", id, r.Round.NShards, len(r.Builds), wantShards)
	}

	// Builds, and exactly one completion each in the workers' logs.
	var comps map[string][]kwLine
	deadline := time.Now().Add(10 * time.Second)
	for {
		comps = c.completions()
		missing := 0
		for _, b := range r.Builds {
			if b.Placement == "gpu" && len(comps[b.BuildID]) == 0 {
				missing++
			}
		}
		if missing == 0 || time.Now().After(deadline) {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	time.Sleep(500 * time.Millisecond)
	comps = c.completions()
	attempts := map[int32]int{}
	pods := map[string]int{}
	for _, b := range r.Builds {
		attempts[b.Attempt]++
		if b.State != "done" || b.FinishedAt == nil {
			t.Errorf("build not done: %s", b)
			continue
		}
		if b.Placement != "gpu" {
			continue
		}
		cs := comps[b.BuildID]
		var desc []string
		for _, l := range cs {
			desc = append(desc, l.Pod+"@attempt"+l.str("attempt"))
			pods[l.Pod]++
		}
		if len(cs) != 1 {
			t.Errorf("build %s: %d worker 'completed' lines %v, want exactly 1", b.BuildID, len(cs), desc)
		} else if cs[0].str("attempt") != strconv.Itoa(int(b.Attempt)) {
			t.Errorf("build %s completed at attempt %s but the row says %d", b.BuildID, cs[0].str("attempt"), b.Attempt)
		}
	}
	t.Logf("[check] round %d builds by attempt: %v; completions by worker pod: %v", id, attempts, pods)

	// Queries.
	doneAt := map[int32]time.Time{}
	for _, b := range r.Builds {
		if b.FinishedAt != nil {
			doneAt[b.ShardID] = *b.FinishedAt
		}
	}
	unfinished, early, disorder := 0, 0, 0
	for _, j := range r.Jobs {
		if j.StartedAt == nil || j.FinishedAt == nil {
			unfinished++
			continue
		}
		if j.StartedAt.Before(j.ArrivedAt) || j.FinishedAt.Before(*j.StartedAt) {
			disorder++
		}
		if j.NeedsIndex && j.StartedAt.Before(doneAt[j.ShardID]) {
			early++
		}
	}
	if unfinished+early+disorder > 0 {
		t.Errorf("round %d: %d of %d queries unfinished, %d out of order, %d needs_index queries started before their build was done",
			id, unfinished, len(r.Jobs), disorder, early)
	}

	// The seeded workload.
	koCheckWorkload(t, r, scale)

	// Shard status.
	if len(r.ShardStatus) != r.Round.NShards {
		t.Errorf("round %d: %d shard_status rows, want %d", id, len(r.ShardStatus), r.Round.NShards)
	}
	for _, s := range r.ShardStatus {
		if !s.StreamDone {
			t.Errorf("shard %d: stream_done=false in a stamped round", s.ShardID)
		}
	}

	// The simulator's own record.
	var tl ldTimeline
	poll(t, 30*time.Second, 200*time.Millisecond, fmt.Sprintf("simulator's timeline for round %d", id), func() bool {
		tl = koTimeline(t, p, id)
		return tl.Finished != "" && len(tl.Rows) == wantShards
	})
	t.Logf("[shards] %s", tl.Finished)
	t.Logf("[shards] %s | %s", tl.Header, tl.Summary)
	if !strings.Contains(tl.Finished, fmt.Sprintf("queries=%d", len(r.Jobs))) {
		t.Errorf("simulator's round-finished line disagrees with the API's %d queries: %s", len(r.Jobs), tl.Finished)
	}
	tlQ := 0
	for _, b := range r.Builds {
		row, ok := tl.Rows[int(b.ShardID)]
		if !ok {
			t.Errorf("timeline has no row for shard %d", b.ShardID)
			continue
		}
		tlQ += row.Queries
		if row.Att != int(b.Attempt) || row.NVectors != b.NVectors {
			t.Errorf("timeline row for shard %d (att=%d n=%d) disagrees with the build row %s", b.ShardID, row.Att, row.NVectors, b)
		}
	}
	if tlQ != len(r.Jobs) {
		t.Errorf("timeline's per-shard query counts sum to %d, the API holds %d", tlQ, len(r.Jobs))
	}
	if !t.Failed() {
		t.Logf("[verify] round %d complete and correct from outside: %d builds done once each, %d queries finished, matching the seeded workload", id, len(r.Builds), len(r.Jobs))
	}
	return r
}

// koCheckWorkload compares every recorded query with the seeded workload the
// round row names (scenario, seed, n_shards), at the given time scale.
// Arrival is loose: at most 500 ms late, 100 ms early against round start +
// offset/scale (the round's started_at is the scheduler's clock, the arrivals
// the laptop's; both are this machine's clock here).
func koCheckWorkload(t *testing.T, r ldRound, scale float64) {
	t.Helper()
	sc, err := workload.PresetWithShards(r.Round.Scenario, r.Round.NShards)
	if err != nil {
		t.Fatalf("PresetWithShards(%q, %d): %v", r.Round.Scenario, r.Round.NShards, err)
	}
	w := workload.Generate(sc, r.Round.Seed)
	got := ldJobsByShard(r)
	total, mism, late, early := 0, 0, 0, 0
	var worstLate, worstEarly time.Duration
	for _, s := range w.Shards {
		total += len(s.Queries)
		js := got[s.ShardID]
		if len(js) != len(s.Queries) {
			t.Errorf("round %d shard %d: %d queries recorded, the workload (%s seed %d) has %d", r.Round.RoundID, s.ShardID, len(js), r.Round.Scenario, r.Round.Seed, len(s.Queries))
			continue
		}
		for i, j := range js {
			q := s.Queries[i]
			dm := q.Duration.Milliseconds()
			if j.NeedsIndex != q.NeedsIndex || j.DurationMs < dm-1 || j.DurationMs > dm+1 {
				mism++
			}
			want := r.Round.StartedAt.Add(time.Duration(float64(q.Arrival) / scale))
			dd := j.ArrivedAt.Sub(want)
			if dd > worstLate {
				worstLate = dd
			}
			if dd < worstEarly {
				worstEarly = dd
			}
			if dd > 500*time.Millisecond {
				late++
			}
			if dd < -100*time.Millisecond {
				early++
			}
		}
	}
	if len(r.Jobs) != total {
		t.Errorf("round %d: %d queries recorded, the workload has %d", r.Round.RoundID, len(r.Jobs), total)
	}
	if mism > 0 {
		t.Errorf("round %d: %d recorded queries disagree with the workload (needs_index or duration)", r.Round.RoundID, mism)
	}
	if late > 0 || early > 0 {
		t.Errorf("round %d: %d queries arrived >500 ms late and %d >100 ms early against schedule (worst %+.3fs / %+.3fs)",
			r.Round.RoundID, late, early, worstLate.Seconds(), worstEarly.Seconds())
	}
	t.Logf("[workload] round %d (%s seed %d, %d shards): %d recorded queries vs %d in the workload, %d field mismatches; arrivals worst late %+.3fs, worst early %+.3fs",
		r.Round.RoundID, r.Round.Scenario, r.Round.Seed, r.Round.NShards, len(r.Jobs), total, mism, worstLate.Seconds(), worstEarly.Seconds())
}
