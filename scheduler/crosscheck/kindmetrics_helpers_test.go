package crosscheck

// Helpers for the cross-check of Phase 2 step 5: metrics and dashboards
// (Phase 2 decisions 28, 31 to 35; roadmap Phase 2 "Metrics"). Everything is
// observed from outside: the scheduler's /metrics through the NodePort
// (127.0.0.1:30080), each worker's /metrics on its probe port 8081 through
// the API server's pod proxy, the simulator's /metrics on the laptop,
// Prometheus's HTTP API and Grafana's HTTP API through kubectl port-forward
// in namespace monitoring. Nothing here reads the implementation.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	kmMonNS    = "monitoring"
	kmPromSvc  = "svc/kps-kube-prometheus-stack-prometheus"
	kmGrafSvc  = "svc/kps-grafana"
	kmPromPort = 29090
	kmGrafPort = 29300
	kmDashCM   = "grafana-dashboard-kgpu"
	kmDashUID  = "kgpu-round"
)

// ---- the Prometheus text exposition format ----

type expoSample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

type expo struct {
	Types   map[string]string // family name -> counter|gauge|histogram|...
	Samples []expoSample
}

// parseExpo parses the text exposition format: "# TYPE name type" lines and
// "name{k="v",...} value [timestamp]" sample lines.
func parseExpo(text string) (expo, error) {
	e := expo{Types: map[string]string{}}
	for _, ln := range strings.Split(text, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		if strings.HasPrefix(ln, "#") {
			f := strings.Fields(ln)
			if len(f) >= 4 && f[1] == "TYPE" {
				e.Types[f[2]] = f[3]
			}
			continue
		}
		s := expoSample{Labels: map[string]string{}}
		i := strings.IndexAny(ln, "{ ")
		if i < 0 {
			return e, fmt.Errorf("bad sample line %q", ln)
		}
		s.Name = ln[:i]
		rest := ln[i:]
		if rest[0] == '{' {
			j := 1
			for {
				for j < len(rest) && (rest[j] == ',' || rest[j] == ' ') {
					j++
				}
				if j >= len(rest) {
					return e, fmt.Errorf("unterminated labels in %q", ln)
				}
				if rest[j] == '}' {
					j++
					break
				}
				eq := strings.IndexByte(rest[j:], '=')
				if eq < 0 {
					return e, fmt.Errorf("bad label in %q", ln)
				}
				key := rest[j : j+eq]
				j += eq + 1
				if j >= len(rest) || rest[j] != '"' {
					return e, fmt.Errorf("unquoted label value in %q", ln)
				}
				j++
				var val strings.Builder
				for j < len(rest) && rest[j] != '"' {
					if rest[j] == '\\' && j+1 < len(rest) {
						j++
						switch rest[j] {
						case 'n':
							val.WriteByte('\n')
						default:
							val.WriteByte(rest[j])
						}
					} else {
						val.WriteByte(rest[j])
					}
					j++
				}
				j++ // closing quote
				s.Labels[key] = val.String()
			}
			rest = rest[j:]
		}
		f := strings.Fields(rest)
		if len(f) < 1 {
			return e, fmt.Errorf("no value in %q", ln)
		}
		v, err := strconv.ParseFloat(f[0], 64)
		if err != nil {
			return e, fmt.Errorf("bad value in %q: %v", ln, err)
		}
		s.Value = v
		e.Samples = append(e.Samples, s)
	}
	return e, nil
}

// match reports whether s carries every label in want.
func (s expoSample) match(want map[string]string) bool {
	for k, v := range want {
		if s.Labels[k] != v {
			return false
		}
	}
	return true
}

// find returns the samples called name carrying every label in want.
func (e expo) find(name string, want map[string]string) []expoSample {
	var out []expoSample
	for _, s := range e.Samples {
		if s.Name == name && s.match(want) {
			out = append(out, s)
		}
	}
	return out
}

// val returns the single sample's value; ok=false if there is not exactly one.
func (e expo) val(name string, want map[string]string) (float64, bool) {
	ss := e.find(name, want)
	if len(ss) != 1 {
		return 0, false
	}
	return ss[0].Value, true
}

// labelValues returns the sorted distinct values of label key on samples of name.
func (e expo) labelValues(name, key string) []string {
	set := map[string]bool{}
	for _, s := range e.Samples {
		if s.Name == name {
			if v, ok := s.Labels[key]; ok {
				set[v] = true
			}
		}
	}
	var out []string
	for v := range set {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// labelKeys returns the sorted label names (le excluded) on samples of name.
func (e expo) labelKeys(name string) []string {
	set := map[string]bool{}
	for _, s := range e.Samples {
		if s.Name == name {
			for k := range s.Labels {
				if k != "le" {
					set[k] = true
				}
			}
		}
	}
	var out []string
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// sampleName returns the series a family is checked by: name_count for a
// histogram, the name itself otherwise.
func sampleName(name, typ string) string {
	if typ == "histogram" {
		return name + "_count"
	}
	return name
}

// ---- fetching /metrics ----

var kmHTTP = &http.Client{Timeout: 5 * time.Second}

func httpText(u string) (int, string, error) {
	resp, err := kmHTTP.Get(u)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), nil
}

// schedMetrics scrapes the scheduler through the NodePort.
func schedMetrics(t *testing.T) expo {
	t.Helper()
	var last error
	for i := 0; i < 20; i++ {
		code, body, err := httpText(koURL + "/metrics")
		if err == nil && code == http.StatusOK {
			e, perr := parseExpo(body)
			if perr != nil {
				t.Fatalf("parse scheduler /metrics: %v", perr)
			}
			return e
		}
		last = fmt.Errorf("code %d err %v", code, err)
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatalf("GET %s/metrics: %v", koURL, last)
	return expo{}
}

// workerMetrics scrapes one worker pod's probe port 8081 through the API
// server's pod proxy (no port-forward to leak). ok=false if unreachable (a
// container restarting).
func workerMetrics(t *testing.T, pod string) (expo, bool) {
	t.Helper()
	so, _, err := kcRun("", "get", "--raw", fmt.Sprintf("/api/v1/namespaces/%s/pods/%s:8081/proxy/metrics", koNS, pod))
	if err != nil {
		return expo{}, false
	}
	e, perr := parseExpo(so)
	if perr != nil {
		t.Fatalf("parse %s /metrics: %v", pod, perr)
	}
	return e, true
}

func mustWorkerMetrics(t *testing.T, pod string) expo {
	t.Helper()
	var e expo
	poll(t, 30*time.Second, 500*time.Millisecond, "worker "+pod+" /metrics on 8081", func() bool {
		var ok bool
		e, ok = workerMetrics(t, pod)
		return ok
	})
	return e
}

func shardMetrics(addr string) (expo, error) {
	code, body, err := httpText("http://" + addr + "/metrics")
	if err != nil {
		return expo{}, err
	}
	if code != http.StatusOK {
		return expo{}, fmt.Errorf("status %d", code)
	}
	return parseExpo(body)
}

// ---- the scheduler API (POST is not covered by the earlier helpers) ----

func kmDo(t *testing.T, method, path string, body any, out any) (int, string) {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, koURL+path, rd)
	if err != nil {
		t.Fatal(err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := kmHTTP.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if out != nil && resp.StatusCode < 300 {
		if err := json.Unmarshal(b, out); err != nil {
			t.Fatalf("%s %s: decode %v: %s", method, path, err, b)
		}
	}
	return resp.StatusCode, strings.TrimSpace(string(b))
}

func kmNewRound(t *testing.T, scenario string, nShards int) int64 {
	t.Helper()
	var r struct {
		RoundID int64 `json:"round_id"`
	}
	code, raw := kmDo(t, "POST", "/rounds", map[string]any{"scenario": scenario, "seed": 5, "n_shards": nShards}, &r)
	if code != http.StatusCreated || r.RoundID == 0 {
		t.Fatalf("POST /rounds -> %d %s", code, raw)
	}
	t.Logf("[api] POST /rounds {scenario:%s n_shards:%d} -> 201 round_id=%d", scenario, nShards, r.RoundID)
	return r.RoundID
}

func kmSubmit(t *testing.T, round int64, shard int, n int64) xSubmit {
	t.Helper()
	var s xSubmit
	code, raw := kmDo(t, "POST", "/builds", map[string]any{"round_id": round, "shard_id": shard, "n_vectors": n, "dim": 128}, &s)
	if code != http.StatusCreated || !s.Created {
		t.Fatalf("POST /builds -> %d %s", code, raw)
	}
	t.Logf("[api] POST /builds {round:%d shard:%d n:%d} -> 201 build_id=%s placement=%s gpu_total_ms=%d", round, shard, n, s.BuildID, s.Placement, s.GPUTotalMs)
	return s
}

// kmReportDone sends the shard's report saying its stream is exhausted and
// its backlog empty (Phase 1 API: POST /shards/{round}/{shard}/report).
func kmReportDone(t *testing.T, round int64, shard int) string {
	t.Helper()
	code, raw := kmDo(t, "POST", fmt.Sprintf("/shards/%d/%d/report", round, shard), map[string]any{
		"queue_depth": 0, "waiting_needs_index": 0, "oldest_wait_ms": 0, "build_progress": 0, "stream_done": true,
	}, nil)
	if code != http.StatusOK {
		t.Fatalf("POST /shards/%d/%d/report -> %d %s", round, shard, code, raw)
	}
	return raw
}

func kmBuild(t *testing.T, id string) xBuild {
	t.Helper()
	var b xBuild
	code, raw := kmDo(t, "GET", "/builds/"+id, nil, &b)
	if code != http.StatusOK {
		t.Fatalf("GET /builds/%s -> %d %s", id, code, raw)
	}
	return b
}

func kmWaitBuild(t *testing.T, id string, d time.Duration, desc string, pred func(xBuild) bool) xBuild {
	t.Helper()
	var b xBuild
	took := poll(t, d, 200*time.Millisecond, "build "+id+" "+desc, func() bool {
		b = kmBuild(t, id)
		return pred(b)
	})
	t.Logf("[api] build %s %s after %.1fs: %s", id, desc, took.Seconds(), b)
	return b
}

// ---- port-forwards into monitoring ----

type kmForward struct {
	cmd  *exec.Cmd
	port int
	out  *syncBuffer
}

func (f *kmForward) stop() {
	if f.cmd.Process != nil {
		_ = f.cmd.Process.Kill()
		_ = f.cmd.Wait()
		f.cmd.Process = nil
	}
}

func (f *kmForward) base() string { return fmt.Sprintf("http://127.0.0.1:%d", f.port) }

// kmPortForward starts kubectl port-forward to svc in monitoring and waits
// until readyPath answers 200.
func kmPortForward(t *testing.T, svc string, local, remote int, readyPath string) *kmForward {
	t.Helper()
	for attempt := 1; attempt <= 3; attempt++ {
		buf := &syncBuffer{}
		cmd := exec.Command("kubectl", "--context", kContext, "-n", kmMonNS, "port-forward", svc, fmt.Sprintf("%d:%d", local, remote))
		cmd.Stdout, cmd.Stderr = buf, buf
		if err := cmd.Start(); err != nil {
			t.Fatalf("start port-forward: %v", err)
		}
		f := &kmForward{cmd: cmd, port: local, out: buf}
		deadline := time.Now().Add(15 * time.Second)
		for time.Now().Before(deadline) {
			if code, _, err := httpText(f.base() + readyPath); err == nil && code == http.StatusOK {
				t.Logf("[port-forward] %s/%s:%d on 127.0.0.1:%d", kmMonNS, svc, remote, local)
				t.Cleanup(f.stop)
				return f
			}
			time.Sleep(300 * time.Millisecond)
		}
		f.stop()
		t.Logf("[port-forward] attempt %d to %s failed: %s", attempt, svc, buf.String())
		time.Sleep(time.Second)
	}
	t.Fatalf("port-forward to %s never answered %s", svc, readyPath)
	return nil
}

// ---- the Prometheus HTTP API ----

type promTarget struct {
	Labels           map[string]string `json:"labels"`
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
	ScrapePool       string            `json:"scrapePool"`
	ScrapeURL        string            `json:"scrapeUrl"`
	Health           string            `json:"health"`
	LastError        string            `json:"lastError"`
	LastScrape       time.Time         `json:"lastScrape"`
	ScrapeInterval   string            `json:"scrapeInterval"`
}

func (p promTarget) String() string {
	return fmt.Sprintf("{job=%s url=%s health=%s interval=%s last=%s err=%q}",
		p.Labels["job"], p.ScrapeURL, p.Health, p.ScrapeInterval, p.LastScrape.Format("15:04:05.000"), p.LastError)
}

func promGetJSON(t *testing.T, f *kmForward, path string, q url.Values, out any) {
	t.Helper()
	u := f.base() + path
	if q != nil {
		u += "?" + q.Encode()
	}
	var last error
	for i := 0; i < 5; i++ {
		code, body, err := httpText(u)
		if err == nil && code == http.StatusOK {
			if err := json.Unmarshal([]byte(body), out); err != nil {
				t.Fatalf("decode %s: %v", path, err)
			}
			return
		}
		last = fmt.Errorf("code %d err %v body %.300s", code, err, body)
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatalf("GET %s: %v", u, last)
}

func promTargets(t *testing.T, f *kmForward) []promTarget {
	t.Helper()
	var r struct {
		Data struct {
			ActiveTargets []promTarget `json:"activeTargets"`
		} `json:"data"`
	}
	promGetJSON(t, f, "/api/v1/targets", url.Values{"state": {"active"}}, &r)
	return r.Data.ActiveTargets
}

func targetsOfJob(ts []promTarget, job string) []promTarget {
	var out []promTarget
	for _, x := range ts {
		if x.Labels["job"] == job {
			out = append(out, x)
		}
	}
	return out
}

type promSeries struct {
	Metric map[string]string `json:"metric"`
	Value  [2]any            `json:"value"`
	Values [][2]any          `json:"values"`
}

func (s promSeries) num() float64 {
	if str, ok := s.Value[1].(string); ok {
		v, _ := strconv.ParseFloat(str, 64)
		return v
	}
	return 0
}

// promQuery runs an instant query and returns the vector.
func promQuery(t *testing.T, f *kmForward, expr string) []promSeries {
	t.Helper()
	var r struct {
		Status string `json:"status"`
		Error  string `json:"error"`
		Data   struct {
			ResultType string       `json:"resultType"`
			Result     []promSeries `json:"result"`
		} `json:"data"`
	}
	promGetJSON(t, f, "/api/v1/query", url.Values{"query": {expr}}, &r)
	if r.Status != "success" {
		t.Fatalf("query %q: %s %s", expr, r.Status, r.Error)
	}
	return r.Data.Result
}

// promScalar runs an instant query expected to return one series.
func promScalar(t *testing.T, f *kmForward, expr string) (float64, bool) {
	t.Helper()
	res := promQuery(t, f, expr)
	if len(res) != 1 {
		return 0, false
	}
	return res[0].num(), true
}

// promRange runs a range query and returns (series, total points, error text).
func promRange(t *testing.T, f *kmForward, expr string, start, end time.Time, step time.Duration) (int, int, string) {
	t.Helper()
	q := url.Values{
		"query": {expr},
		"start": {strconv.FormatInt(start.Unix(), 10)},
		"end":   {strconv.FormatInt(end.Unix(), 10)},
		"step":  {strconv.Itoa(int(step.Seconds()))},
	}
	u := f.base() + "/api/v1/query_range?" + q.Encode()
	code, body, err := httpText(u)
	if err != nil {
		return 0, 0, err.Error()
	}
	var r struct {
		Status string `json:"status"`
		Error  string `json:"error"`
		Data   struct {
			Result []promSeries `json:"result"`
		} `json:"data"`
	}
	if json.Unmarshal([]byte(body), &r) != nil || code != http.StatusOK || r.Status != "success" {
		return 0, 0, fmt.Sprintf("code %d %s %.200s", code, r.Error, body)
	}
	pts := 0
	for _, s := range r.Data.Result {
		for _, v := range s.Values {
			if str, ok := v[1].(string); ok && str != "NaN" {
				pts++
			}
		}
	}
	return len(r.Data.Result), pts, ""
}

// ---- the dashboard ConfigMap ----

type dashPanel struct {
	ID         int    `json:"id"`
	Type       string `json:"type"`
	Title      string `json:"title"`
	Datasource any    `json:"datasource"`
	Targets    []struct {
		Expr  string `json:"expr"`
		RefID string `json:"refId"`
	} `json:"targets"`
	Panels []dashPanel `json:"panels"`
}

type dashboard struct {
	UID    string      `json:"uid"`
	Title  string      `json:"title"`
	Panels []dashPanel `json:"panels"`
}

// flatPanels returns every non-row panel, rows' children included.
func flatPanels(ps []dashPanel) []dashPanel {
	var out []dashPanel
	for _, p := range ps {
		if p.Type == "row" {
			out = append(out, flatPanels(p.Panels)...)
			continue
		}
		out = append(out, p)
		out = append(out, flatPanels(p.Panels)...)
	}
	return out
}

func dashboardFromConfigMap(t *testing.T) (dashboard, string) {
	t.Helper()
	out := kc(t, "get", "configmap", kmDashCM, "-n", koNS, "-o", "json")
	var cm struct {
		Metadata struct {
			Labels map[string]string `json:"labels"`
		} `json:"metadata"`
		Data map[string]string `json:"data"`
	}
	if err := json.Unmarshal([]byte(out), &cm); err != nil {
		t.Fatalf("decode configmap: %v", err)
	}
	if len(cm.Data) != 1 {
		t.Fatalf("configmap %s has %d data keys, want 1", kmDashCM, len(cm.Data))
	}
	var key, raw string
	for k, v := range cm.Data {
		key, raw = k, v
	}
	var d dashboard
	if err := json.Unmarshal([]byte(raw), &d); err != nil {
		t.Fatalf("decode dashboard JSON %s: %v", key, err)
	}
	return d, fmt.Sprintf("key=%s labels=%v", key, cm.Metadata.Labels)
}
