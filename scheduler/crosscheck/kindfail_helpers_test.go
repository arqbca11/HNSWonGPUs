package crosscheck

// Helpers for the cross-check of Phase 2 step 6: the failure injections,
// watched on the dashboard (roadmap Phase 2 "Failure injection" and the
// phase's done-when; design decisions 29, 31, 32, 35, 36).
//
// Every dashboard number is read the way decision 36 says the dashboard is
// proven: through Grafana's datasource proxy,
// /api/datasources/proxy/uid/<prometheus uid>/api/v1/query, basic auth
// admin/kgpu, on a private kubectl port-forward to svc/kps-grafana. The
// datasource uid is discovered from Grafana's own /api/datasources, not
// assumed. Everything else is observed from outside: the scheduler API on the
// NodePort, kubectl on namespace kgpu, pods' logs, the simulator's stdout.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	kfGrafPort = 29310
	kfGrafUser = "admin"
	kfGrafPass = "kgpu"
)

// ---- Grafana's datasource proxy ----

type kfGraf struct {
	f   *kmForward
	uid string
}

// kfGrafana port-forwards Grafana and discovers the default Prometheus
// datasource's uid through Grafana's API.
func kfGrafana(t *testing.T) *kfGraf {
	t.Helper()
	f := kmPortForward(t, kmGrafSvc, kfGrafPort, 80, "/api/health")
	g := &kfGraf{f: f}
	code, body := g.get(t, "/api/datasources")
	if code != http.StatusOK {
		t.Fatalf("GET /api/datasources -> %d %.300s", code, body)
	}
	var ds []struct {
		UID       string `json:"uid"`
		Type      string `json:"type"`
		IsDefault bool   `json:"isDefault"`
		URL       string `json:"url"`
	}
	if err := json.Unmarshal([]byte(body), &ds); err != nil {
		t.Fatalf("decode datasources: %v", err)
	}
	for _, d := range ds {
		if d.Type == "prometheus" && (g.uid == "" || d.IsDefault) {
			g.uid = d.UID
			t.Logf("[grafana] Prometheus datasource uid=%q default=%v url=%s; every number below is read through /api/datasources/proxy/uid/%s (decision 36)", d.UID, d.IsDefault, d.URL, d.UID)
		}
	}
	if g.uid == "" {
		t.Fatalf("Grafana has no Prometheus datasource: %s", body)
	}
	return g
}

func (g *kfGraf) get(t *testing.T, path string) (int, string) {
	t.Helper()
	var last error
	for i := 0; i < 5; i++ {
		req, _ := http.NewRequest("GET", g.f.base()+path, nil)
		req.SetBasicAuth(kfGrafUser, kfGrafPass)
		resp, err := kmHTTP.Do(req)
		if err == nil {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return resp.StatusCode, string(b)
			}
			last = fmt.Errorf("status %d %.200s", resp.StatusCode, b)
		} else {
			last = err
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatalf("GET grafana %s: %v", path, last)
	return 0, ""
}

// query runs an instant PromQL query through the datasource proxy.
func (g *kfGraf) query(t *testing.T, expr string) []promSeries {
	t.Helper()
	path := "/api/datasources/proxy/uid/" + g.uid + "/api/v1/query?" + url.Values{"query": {expr}}.Encode()
	code, body := g.get(t, path)
	var r struct {
		Status string `json:"status"`
		Error  string `json:"error"`
		Data   struct {
			Result []promSeries `json:"result"`
		} `json:"data"`
	}
	if code != http.StatusOK || json.Unmarshal([]byte(body), &r) != nil || r.Status != "success" {
		t.Fatalf("grafana proxy query %q -> %d %.300s", expr, code, body)
	}
	return r.Data.Result
}

// scalar runs a query expected to return at most one series; ok=false if none.
func (g *kfGraf) scalar(t *testing.T, expr string) (float64, bool) {
	t.Helper()
	res := g.query(t, expr)
	if len(res) == 0 {
		return 0, false
	}
	if len(res) > 1 {
		t.Fatalf("query %q returned %d series, want 1: %v", expr, len(res), res)
	}
	return res[0].num(), true
}

// byLabel returns label value -> value for a query.
func (g *kfGraf) byLabel(t *testing.T, expr, label string) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	for _, s := range g.query(t, expr) {
		out[s.Metric[label]] = s.num()
	}
	return out
}

func fmtMap(m map[string]float64) string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	var parts []string
	for _, k := range ks {
		parts = append(parts, fmt.Sprintf("%s=%g", k, m[k]))
	}
	return "{" + strings.Join(parts, " ") + "}"
}

// scrapeWait sleeps long enough for two 5 s scrapes (decision 34) to land.
func scrapeWait(t *testing.T, why string) {
	t.Logf("[wait] 12 s for two 5 s scrapes: %s", why)
	time.Sleep(12 * time.Second)
}

// ---- the dashboard's gauges at rest ----

type kfRest struct {
	RoundsOpen, QueueDepth, Leased, Busy, BusySeries, Live, SchedUp float64
}

func (r kfRest) String() string {
	return fmt.Sprintf("max(rounds_open)=%g max(gpu_queue_depth)=%g max(builds_leased)=%g sum(worker_busy)=%g count(worker_busy)=%g max(gpu_workers_live)=%g count(up{job=scheduler}==1)=%g",
		r.RoundsOpen, r.QueueDepth, r.Leased, r.Busy, r.BusySeries, r.Live, r.SchedUp)
}

func (g *kfGraf) rest(t *testing.T) kfRest {
	t.Helper()
	v := func(e string) float64 { x, _ := g.scalar(t, e); return x }
	return kfRest{
		RoundsOpen: v("max(rounds_open)"),
		QueueDepth: v("max(gpu_queue_depth)"),
		Leased:     v("max(builds_leased)"),
		Busy:       v("sum(worker_busy)"),
		BusySeries: v("count(worker_busy)"),
		Live:       v("max(gpu_workers_live)"),
		SchedUp:    v(`count(up{job="scheduler"} == 1)`),
	}
}

// kfAtRest waits up to d for the dashboard's stat panels to come back to
// rest after an injection: no open round beyond baseRounds, an empty GPU
// queue, nothing leased, no worker busy, exactly wantWorkers worker_busy
// series (no ghost series from a deleted pod), wantWorkers live, one
// scheduler target up. Fails with the last values otherwise.
func kfAtRest(t *testing.T, g *kfGraf, wantWorkers int, baseRounds float64, d time.Duration) {
	t.Helper()
	want := kfRest{RoundsOpen: baseRounds, Busy: 0, BusySeries: float64(wantWorkers), Live: float64(wantWorkers), SchedUp: 1}
	start := time.Now()
	var r kfRest
	for {
		r = g.rest(t)
		if r == want {
			t.Logf("[dashboard at rest after %.1fs] %s", time.Since(start).Seconds(), r)
			return
		}
		if time.Since(start) > d {
			t.Errorf("dashboard not at rest after %s:\n  got  %s\n  want %s", d, r, want)
			return
		}
		time.Sleep(2 * time.Second)
	}
}

// ---- a background sampler of the dashboard's gauges ----

type kfSample struct {
	At time.Time
	R  kfRest
}

type kfSampler struct {
	mu      sync.Mutex
	samples []kfSample
	stopc   chan struct{}
	done    chan struct{}
}

// kfSample samples the gauges through Grafana every 2 s until stopped. It
// uses its own *testing.T-free path: errors are skipped, not fatal.
func kfStartSampler(t *testing.T, g *kfGraf) *kfSampler {
	s := &kfSampler{stopc: make(chan struct{}), done: make(chan struct{})}
	q := func(e string) (float64, bool) {
		path := "/api/datasources/proxy/uid/" + g.uid + "/api/v1/query?" + url.Values{"query": {e}}.Encode()
		req, _ := http.NewRequest("GET", g.f.base()+path, nil)
		req.SetBasicAuth(kfGrafUser, kfGrafPass)
		resp, err := kmHTTP.Do(req)
		if err != nil {
			return 0, false
		}
		defer resp.Body.Close()
		var r struct {
			Data struct {
				Result []promSeries `json:"result"`
			} `json:"data"`
		}
		if json.NewDecoder(resp.Body).Decode(&r) != nil || len(r.Data.Result) != 1 {
			return 0, len(r.Data.Result) == 0
		}
		return r.Data.Result[0].num(), true
	}
	go func() {
		defer close(s.done)
		for {
			var r kfRest
			ok := true
			for _, x := range []struct {
				e string
				p *float64
			}{
				{"max(rounds_open)", &r.RoundsOpen}, {"max(gpu_queue_depth)", &r.QueueDepth}, {"max(builds_leased)", &r.Leased},
				{"sum(worker_busy)", &r.Busy}, {"count(worker_busy)", &r.BusySeries}, {"max(gpu_workers_live)", &r.Live},
				{`count(up{job="scheduler"} == 1)`, &r.SchedUp},
			} {
				v, good := q(x.e)
				ok = ok && good
				*x.p = v
			}
			if ok {
				s.mu.Lock()
				s.samples = append(s.samples, kfSample{time.Now(), r})
				s.mu.Unlock()
			}
			select {
			case <-s.stopc:
				return
			case <-time.After(2 * time.Second):
			}
		}
	}()
	t.Cleanup(s.stop)
	return s
}

func (s *kfSampler) stop() {
	select {
	case <-s.stopc:
	default:
		close(s.stopc)
	}
	<-s.done
}

func (s *kfSampler) all() []kfSample {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]kfSample(nil), s.samples...)
}

// narrate logs the samples, collapsing runs of identical readings.
func (s *kfSampler) narrate(t *testing.T, t0 time.Time) {
	var prev string
	for _, x := range s.all() {
		cur := x.R.String()
		if cur == prev {
			continue
		}
		prev = cur
		t.Logf("[dashboard t+%5.1fs] %s", x.At.Sub(t0).Seconds(), cur)
	}
}

// ---- the scheduler and worker pods ----

func kfSchedulerPod(t *testing.T) string {
	t.Helper()
	live, _ := kwPods(t, koNS, "app=scheduler")
	if len(live) != 1 {
		t.Fatalf("want exactly one live scheduler pod, have %v", podNames(live))
	}
	return live[0].Metadata.Name
}

// kfWaitNewScheduler waits for a Ready scheduler pod other than old, with
// none terminating, and returns its name.
func kfWaitNewScheduler(t *testing.T, old string, d time.Duration) string {
	t.Helper()
	var name string
	took := poll(t, d, 500*time.Millisecond, "a new Ready scheduler pod, the old one gone", func() bool {
		live, term := kwPods(t, koNS, "app=scheduler")
		if len(live) != 1 || len(term) != 0 || live[0].Metadata.Name == old || !live[0].ready() {
			return false
		}
		name = live[0].Metadata.Name
		return true
	})
	t.Logf("[k8s] scheduler %s replaced by %s, Ready after %.1fs", old, name, took.Seconds())
	return name
}

// kfForceDelete deletes a pod with grace 0 and --force (the KILL_FORCE=1
// injection in the CLAUDE.md Commands table).
func kfForceDelete(t *testing.T, pod string) time.Time {
	t.Helper()
	at := time.Now()
	so, se, err := kcRun("", "delete", "pod", pod, "-n", koNS, "--grace-period=0", "--force", "--wait=false")
	if err != nil {
		t.Fatalf("force-delete %s: %v %s%s", pod, err, so, se)
	}
	t.Logf("[kubectl %s] delete pod %s --grace-period=0 --force: %s", at.Format("15:04:05.000"), pod, strings.TrimSpace(so))
	return at
}

// kfWorkerLinesFor returns the collector's lines from pod about build id.
func kfWorkerLinesFor(c *koCollector, pod, id string) []kwLine {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []kwLine
	seen := map[string]bool{}
	for _, l := range c.lines {
		if l.Pod != pod || l.str("build_id") != id || seen[l.Raw] {
			continue
		}
		seen[l.Raw] = true
		out = append(out, l)
	}
	return out
}

// kfMsgs summarises lines as msg@attempt.
func kfMsgs(ls []kwLine) []string {
	var out []string
	for _, l := range ls {
		out = append(out, l.str("msg")+"@"+l.str("attempt"))
	}
	return out
}

func kfHasRelease(ls []kwLine) bool {
	for _, l := range ls {
		if strings.Contains(strings.ToLower(l.str("msg")), "releas") {
			return true
		}
	}
	return false
}

// kfSchedulerLinesAbout returns scheduler lines carrying round_id or build_id.
func kfSchedulerLinesAbout(c *koCollector, roundID int64, buildID string) []kwLine {
	c.mu.Lock()
	defer c.mu.Unlock()
	rs := strconv.FormatInt(roundID, 10)
	var out []kwLine
	seen := map[string]bool{}
	for _, l := range c.lines {
		if !isSchedulerPod(l.Pod) || seen[l.Pod+l.Raw] {
			continue
		}
		if l.str("round_id") == rs || (buildID != "" && l.str("build_id") == buildID) {
			seen[l.Pod+l.Raw] = true
			out = append(out, l)
		}
	}
	return out
}

// ---- nodes ----

func kfUncordonGPU(t *testing.T) {
	for _, n := range []string{"kgpu-worker3", "kgpu-worker4"} {
		so, se, err := kcRun("", "uncordon", n)
		t.Logf("[cleanup] uncordon %s: err=%v %s%s", n, err, strings.TrimSpace(so), strings.TrimSpace(se))
	}
}

// kfGPUNodes returns the nodes labelled kgpu.io/gpu=true.
func kfGPUNodes(t *testing.T) []string {
	t.Helper()
	out := strings.Fields(kc(t, "get", "nodes", "-l", "kgpu.io/gpu=true", "-o", "jsonpath={.items[*].metadata.name}"))
	sort.Strings(out)
	return out
}

func kfWorkersByNode(t *testing.T) (map[string][]string, []string) {
	t.Helper()
	live, _ := kwPods(t, koNS, "app=worker")
	m := map[string][]string{}
	var pending []string
	for _, p := range live {
		if p.Spec.NodeName == "" {
			pending = append(pending, p.Metadata.Name)
			continue
		}
		m[p.Spec.NodeName] = append(m[p.Spec.NodeName], p.Metadata.Name+fmt.Sprintf("(ready=%v)", p.ready()))
	}
	return m, pending
}

// kfCrashWorker SIGKILLs the worker's python process inside its container in
// namespace kgpu: the in-container crash of decision 19, the one injection
// that makes a lease expire (the pod and its container restart in place).
func kfCrashWorker(t *testing.T, pod string) time.Time {
	t.Helper()
	out := kc(t, "exec", "-n", koNS, pod, "-c", "worker", "--", "sh", "-c",
		`for p in /proc/[0-9]*; do [ -r $p/cmdline ] || continue; c=$(tr '\0' ' ' < $p/cmdline); case "$c" in python*) echo "${p#/proc/}";; esac; done`)
	pids := strings.Fields(out)
	if len(pids) == 0 {
		t.Fatalf("no python process in %s", pod)
	}
	at := time.Now()
	kc(t, "exec", "-n", koNS, pod, "-c", "worker", "--", "sh", "-c", "kill -KILL "+pids[0])
	t.Logf("[exec %s %s] kill -KILL %s (the worker's python process)", pod, at.Format("15:04:05.000"), pids[0])
	return at
}

// kfLine formats a log line's message and its main fields.
func kfLine(l kwLine) string {
	var parts []string
	for _, k := range []string{"msg", "round_id", "build_id", "attempt", "duration_s", "lag_s", "level"} {
		if v := l.str(k); v != "" {
			parts = append(parts, k+"="+v)
		}
	}
	return strings.Join(parts, " ")
}
