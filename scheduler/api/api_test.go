package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/arqbca11/HNSWonGPUs/db"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	"github.com/arqbca11/HNSWonGPUs/scheduler/reaper"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

// Integration test of the HTTP surface against a real Postgres. Skipped
// unless TEST_DATABASE_URL is set; scripts/test-db.sh provides one.

type harness struct {
	t        *testing.T
	st       *store.Store
	srv      *httptest.Server
	log      *slog.Logger
	draining *atomic.Bool // false unless a test flips it; main sets it at SIGTERM
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("TEST_DATABASE_URL not set; run scripts/test-db.sh")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if err := db.Migrate(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `TRUNCATE shard_status, shard_jobs, builds, rounds, workers`); err != nil {
		t.Fatal(err)
	}
	st := store.New(pool)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	draining := new(atomic.Bool)
	srv := httptest.NewServer(New(st, policy.AlwaysGPU{}, costmodel.Default(), Config{WorkerStaleAfter: 30 * time.Second, Draining: draining}, log))
	t.Cleanup(srv.Close)
	return &harness{t: t, st: st, srv: srv, log: log, draining: draining}
}

// call does one request, logs it, decodes the JSON body into out (if any),
// and returns the status code.
func (h *harness) call(method, path string, body any, out any) int {
	h.t.Helper()
	var rd io.Reader
	shown := ""
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
		shown = " " + string(b)
	}
	req, _ := http.NewRequest(method, h.srv.URL+path, rd)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	line := string(bytes.TrimSpace(raw)) // a copy: truncating must not touch raw
	if len(line) > 160 {
		line = line[:160] + "..."
	}
	h.t.Logf("%s %s%s\n      -> %d %s", method, path, shown, resp.StatusCode, line)
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			h.t.Fatalf("decode %s %s: %v: %s", method, path, err, raw)
		}
	}
	return resp.StatusCode
}

func TestEndToEndRoundOverHTTP(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()

	t.Log("--- a worker with 8 GiB registers by heartbeat (workers use SQL, not HTTP)")
	if err := h.st.Heartbeat(ctx, "gpu-1", 8<<30, "node-1"); err != nil {
		t.Fatal(err)
	}
	var ws WorkersResponse
	h.call("GET", "/workers", nil, &ws)
	if ws.Pool.LiveWorkers != 1 {
		t.Fatalf("pool = %+v", ws.Pool)
	}

	t.Log("--- shard simulator starts a round of 2 shards")
	var cr CreateRoundResponse
	if code := h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 42, NShards: 2}, &cr); code != 201 {
		t.Fatalf("create round: %d", code)
	}
	r := cr.RoundID
	report := func(shard int32, req ReportRequest) ReportResponse {
		var rr ReportResponse
		if code := h.call("POST", fmt.Sprintf("/shards/%d/%d/report", r, shard), req, &rr); code != 200 {
			t.Fatalf("report: %d", code)
		}
		return rr
	}

	t.Log("--- shard 1 submits a 100k x 128 build; it knows nothing about future queries")
	var b1 SubmitBuildResponse
	code := h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 1, NVectors: 100_000, Dim: 128}, &b1)
	if code != 201 || b1.Placement != "gpu" || !b1.Created {
		t.Fatalf("submit 1: code=%d resp=%+v", code, b1)
	}
	t.Logf("    placed on gpu; estimates: cpu %d ms, gpu %d ms, mem %d MiB", b1.CPUBuildMs, b1.GPUTotalMs, b1.MemBytes>>20)

	t.Log("--- shard 1 resubmits the same build (retry after a timeout, say)")
	var dup SubmitBuildResponse
	code = h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 1, NVectors: 100_000, Dim: 128}, &dup)
	if code != 200 || dup.Created || dup.Placement != "gpu" || dup.BuildID != b1.BuildID {
		t.Fatalf("duplicate submit: code=%d resp=%+v", code, dup)
	}

	t.Log("--- shard 2 submits a 20M x 128 build: 19 GiB modeled memory, more than any live worker")
	var b2 SubmitBuildResponse
	code = h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 2, NVectors: 20_000_000, Dim: 128}, &b2)
	if code != 201 || b2.Placement != "local" {
		t.Fatalf("submit 2: code=%d resp=%+v", code, b2)
	}
	t.Logf("    placed local: %s", b2.Reason)

	t.Log("--- queries arrive at shard 1 while its build is queued on the gpu")
	if code := h.call("POST", fmt.Sprintf("/jobs/%d/1", r), ArrivalRequest{Seq: 0, DurationMs: 50, NeedsIndex: false}, nil); code != 201 {
		t.Fatalf("arrival: %d", code)
	}
	if code := h.call("POST", fmt.Sprintf("/jobs/%d/1", r), ArrivalRequest{Seq: 0, DurationMs: 50, NeedsIndex: false}, nil); code != 200 {
		t.Fatalf("repeat arrival should be 200, got %d", code)
	}
	h.call("POST", fmt.Sprintf("/jobs/%d/1", r), ArrivalRequest{Seq: 1, DurationMs: 50, NeedsIndex: true}, nil)

	t.Log("--- shard 1 reports: 2 waiting, 1 of them needs the index; the reply tells it the build is still queued")
	rr := report(1, ReportRequest{QueueDepth: 2, WaitingNeedsIndex: 1, OldestWaitMs: 120})
	if rr.Build.Placement != "gpu" || rr.Build.State != "queued" {
		t.Fatalf("report reply: %+v", rr.Build)
	}

	t.Log("--- shard 1's cpu is free, so query 0 (needs_index=false) runs now")
	if code := h.call("POST", fmt.Sprintf("/jobs/%d/1/0/start", r), nil, nil); code != 200 {
		t.Fatalf("job start: %d", code)
	}
	if code := h.call("POST", fmt.Sprintf("/jobs/%d/1/0/done", r), nil, nil); code != 200 {
		t.Fatalf("job done: %d", code)
	}
	if code := h.call("POST", fmt.Sprintf("/jobs/%d/1/0/done", r), nil, nil); code != 409 {
		t.Fatalf("finishing a finished job should be 409, got %d", code)
	}

	t.Log("--- the gpu worker claims and completes shard 1's build (SQL, as the Python worker will)")
	c, ok, err := h.st.Claim(ctx, "gpu-1", 8<<30, 30*time.Second)
	if err != nil || !ok || c.BuildID != b1.BuildID {
		t.Fatalf("claim: ok=%v id=%s err=%v", ok, c.BuildID, err)
	}
	if ok, _ := h.st.Complete(ctx, c.BuildID, c.Attempt); !ok {
		t.Fatal("complete rejected")
	}

	t.Log("--- shard 1's next report learns the build is done; query 1 (needs_index) runs; stream exhausted")
	rr = report(1, ReportRequest{QueueDepth: 1, WaitingNeedsIndex: 1, OldestWaitMs: 300})
	if rr.Build.State != "done" || rr.Build.Attempt != 1 {
		t.Fatalf("report reply after completion: %+v", rr.Build)
	}
	h.call("POST", fmt.Sprintf("/jobs/%d/1/1/start", r), nil, nil)
	h.call("POST", fmt.Sprintf("/jobs/%d/1/1/done", r), nil, nil)
	report(1, ReportRequest{QueueDepth: 0, StreamDone: true})

	t.Log("--- shard 2: one query arrives during its local build, then the build finishes, the query runs")
	h.call("POST", fmt.Sprintf("/jobs/%d/2", r), ArrivalRequest{Seq: 0, DurationMs: 50, NeedsIndex: true}, nil)
	report(2, ReportRequest{QueueDepth: 1, WaitingNeedsIndex: 1, OldestWaitMs: 900, BuildProgress: 0.6})
	if code := h.call("POST", "/builds/"+b2.BuildID+"/done", nil, nil); code != 200 {
		t.Fatalf("local done: %d", code)
	}
	if code := h.call("POST", "/builds/"+b2.BuildID+"/done", nil, nil); code != 409 {
		t.Fatalf("second local done should be 409, got %d", code)
	}
	if code := h.call("POST", "/builds/"+b1.BuildID+"/done", nil, nil); code != 409 {
		t.Fatalf("local-done on a gpu build should be 409, got %d", code)
	}
	h.call("POST", fmt.Sprintf("/jobs/%d/2/0/start", r), nil, nil)
	h.call("POST", fmt.Sprintf("/jobs/%d/2/0/done", r), nil, nil)

	t.Log("--- the round is still open: shard 2 has not said its stream is done")
	var rs RoundStatus
	h.call("GET", fmt.Sprintf("/rounds/%d", r), nil, &rs)
	if rs.Round.FinishedAt != nil {
		t.Fatal("round finished before shard 2 reported stream_done")
	}
	report(2, ReportRequest{QueueDepth: 0, StreamDone: true})

	t.Log("--- the reaper tick (or a GET) stamps the round finished")
	reaper.Tick(ctx, h.st, 5, 30*time.Second, h.log)
	h.call("GET", fmt.Sprintf("/rounds/%d", r), nil, &rs)
	if rs.Round.FinishedAt == nil {
		t.Fatal("round not finished")
	}
	if len(rs.Builds) != 2 || len(rs.Jobs) != 3 || len(rs.Statuses) != 2 {
		t.Fatalf("status: builds=%d jobs=%d statuses=%d", len(rs.Builds), len(rs.Jobs), len(rs.Statuses))
	}
	for _, b := range rs.Builds {
		t.Logf("    build %s: %s/%s attempt %d", b.BuildID, b.Placement, b.State, b.Attempt)
		if b.State != "done" {
			t.Fatalf("build %s state %s", b.BuildID, b.State)
		}
	}
	t.Logf("    round %d finished; duration %s", r, rs.Round.FinishedAt.Sub(rs.Round.StartedAt).Round(time.Millisecond))
}

func TestReportReflectsPreemption(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 1, NShards: 1}, &cr)
	r := cr.RoundID
	h.st.Heartbeat(ctx, "small", 1, "node-1") //nolint:errcheck
	var b SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 0, NVectors: 100_000, Dim: 128}, &b)
	if b.Placement != "local" {
		t.Fatalf("expected local (1-byte worker), got %s", b.Placement)
	}
	t.Log("--- shard 0 builds locally and reports a growing queue")
	var rr ReportResponse
	h.call("POST", fmt.Sprintf("/shards/%d/0/report", r), ReportRequest{QueueDepth: 5, WaitingNeedsIndex: 1, OldestWaitMs: 4000, BuildProgress: 0.2}, &rr)
	if rr.Build.Placement != "local" || rr.Build.State != "running" {
		t.Fatalf("reply: %+v", rr.Build)
	}
	t.Log("--- (Phase 3 reconsider) preempts the build to the gpu queue")
	if ok, _ := h.st.PreemptToGPU(ctx, b.BuildID, 5); !ok {
		t.Fatal("preempt failed")
	}
	t.Log("--- the shard's next report tells it: abort the local build")
	h.call("POST", fmt.Sprintf("/shards/%d/0/report", r), ReportRequest{QueueDepth: 6, WaitingNeedsIndex: 1, OldestWaitMs: 4500, BuildProgress: 0.3}, &rr)
	if rr.Build.Placement != "gpu" || rr.Build.State != "queued" {
		t.Fatalf("reply after preempt: %+v", rr.Build)
	}
	if code := h.call("POST", "/builds/"+b.BuildID+"/done", nil, nil); code != 409 {
		t.Fatalf("a preempted shard's done must be refused, got %d", code)
	}
	t.Log("--- a report for a shard with no build is 404; out-of-range fields are 400")
	if code := h.call("POST", fmt.Sprintf("/shards/%d/7/report", r), ReportRequest{}, nil); code != 404 {
		t.Fatalf("report without build: %d", code)
	}
	if code := h.call("POST", fmt.Sprintf("/shards/%d/0/report", r), ReportRequest{QueueDepth: 1, WaitingNeedsIndex: 2}, nil); code != 400 {
		t.Fatalf("bad report: %d", code)
	}
}

func TestValidationAndNotFound(t *testing.T) {
	h := newHarness(t)
	cases := []struct {
		method, path string
		body         any
		want         int
	}{
		{"POST", "/rounds", CreateRoundRequest{NShards: 0}, 400},
		{"POST", "/rounds", `{"n_shards": 1, "bogus": 1}`, 400},
		{"GET", "/rounds/999999", nil, 404},
		{"GET", "/rounds/abc", nil, 400},
		{"POST", "/builds", SubmitBuildRequest{RoundID: 999999, ShardID: 1, NVectors: 10, Dim: 4}, 404},
		{"POST", "/builds", SubmitBuildRequest{RoundID: 1, ShardID: 1, NVectors: 0, Dim: 4}, 400},
		{"POST", "/builds/nope/done", nil, 404},
		{"POST", "/jobs/999999/1/0/start", nil, 404},
		{"GET", "/builds/nope", nil, 404},
		{"POST", "/jobs/999999/1", ArrivalRequest{Seq: -1}, 400},
		{"GET", "/healthz", nil, 200},
	}
	for _, c := range cases {
		var body any = c.body
		if s, isStr := c.body.(string); isStr {
			body = json.RawMessage(s)
		}
		if got := h.call(c.method, c.path, body, nil); got != c.want {
			t.Fatalf("%s %s: got %d want %d", c.method, c.path, got, c.want)
		}
	}
}

func TestSubmitValidationAgainstRound(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 1, NShards: 2}, &cr)
	r := cr.RoundID

	t.Log("--- shard ids are the shard's own numbering; 7 and 42 are fine in a 2-shard round")
	var first, dup SubmitBuildResponse
	if code := h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 7, NVectors: 100_000, Dim: 128}, &first); code != 201 {
		t.Fatalf("shard 7: got %d", code)
	}
	if code := h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 42, NVectors: 10, Dim: 4}, nil); code != 201 {
		t.Fatalf("shard 42: got %d", code)
	}

	t.Log("--- a third distinct shard would let the round finish early: refused")
	if code := h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 99, NVectors: 10, Dim: 4}, nil); code != 409 {
		t.Fatalf("third shard in a 2-shard round: got %d want 409", code)
	}

	t.Log("--- a duplicate with a different body returns estimates for the STORED row, and is fine in a full round")
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 7, NVectors: 5, Dim: 128}, &dup)
	if dup.Created || dup.CPUBuildMs != first.CPUBuildMs || dup.GPUTotalMs != first.GPUTotalMs || dup.MemBytes != first.MemBytes {
		t.Fatalf("duplicate estimates differ: first %+v dup %+v", first, dup)
	}

	t.Log("--- finish the round; a duplicate is still 200, a new build would be 409")
	for range 2 {
		c, ok, _ := h.st.Claim(ctx, "w", 1<<40, 30*time.Second)
		if !ok {
			t.Fatal("expected a claimable build")
		}
		h.st.Complete(ctx, c.BuildID, c.Attempt) //nolint:errcheck
	}
	for _, shard := range []int32{7, 42} {
		h.call("POST", fmt.Sprintf("/shards/%d/%d/report", r, shard), ReportRequest{StreamDone: true}, nil)
	}
	var rs RoundStatus
	h.call("GET", fmt.Sprintf("/rounds/%d", r), nil, &rs)
	if rs.Round.FinishedAt == nil {
		t.Fatal("round should be finished")
	}
	if code := h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 42, NVectors: 10, Dim: 4}, nil); code != 200 {
		t.Fatalf("duplicate into a finished round: got %d want 200", code)
	}
}

// Decision 60 over HTTP: the report carries a batch of client-timestamped
// query records; bad batches are 400; a new arrival after stream_done is 409.
func TestReportWithJobBatch(t *testing.T) {
	h := newHarness(t)
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 1, NShards: 1}, &cr)
	r := cr.RoundID
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 0, NVectors: 1000, Dim: 4}, nil)
	t0 := time.Now().Add(-5 * time.Second)
	at := func(ms int) *time.Time { x := t0.Add(time.Duration(ms) * time.Millisecond); return &x }
	path := fmt.Sprintf("/shards/%d/0/report", r)

	t.Log("--- a report with two records, one finished")
	var rr ReportResponse
	code := h.call("POST", path, ReportRequest{QueueDepth: 1, Jobs: []store.JobRecord{
		{Seq: 0, DurationMs: 10, ArrivedAt: *at(0), StartedAt: at(1), FinishedAt: at(11)},
		{Seq: 1, DurationMs: 10, ArrivedAt: *at(2)},
	}}, &rr)
	if code != 200 || rr.Build.State != "queued" {
		t.Fatalf("batch report: %d %+v", code, rr)
	}
	var rs RoundStatus
	h.call("GET", fmt.Sprintf("/rounds/%d", r), nil, &rs)
	if len(rs.Jobs) != 2 || rs.Jobs[0].FinishedAt == nil || rs.Jobs[0].ArrivedAt.Sub(t0) != 0 {
		t.Fatalf("jobs: %+v", rs.Jobs)
	}
	t.Log("    both recorded with the shard's timestamps; the reply still carries the build")

	t.Log("--- bad batches are 400")
	for _, bad := range []store.JobRecord{
		{Seq: 5, DurationMs: 10}, // no arrived_at
		{Seq: 5, DurationMs: 10, ArrivedAt: *at(10), StartedAt: at(5)},   // started before arrived
		{Seq: 5, DurationMs: 10, ArrivedAt: *at(10), FinishedAt: at(20)}, // finished without started
	} {
		if code := h.call("POST", path, ReportRequest{Jobs: []store.JobRecord{bad}}, nil); code != 400 {
			t.Fatalf("bad record %+v: got %d", bad, code)
		}
	}
	t.Log("--- stream_done with the last record; then a new arrival is 409")
	if code := h.call("POST", path, ReportRequest{StreamDone: true, Jobs: []store.JobRecord{{Seq: 1, DurationMs: 10, ArrivedAt: *at(2), StartedAt: at(12), FinishedAt: at(22)}}}, nil); code != 200 {
		t.Fatalf("final batch: %d", code)
	}
	if code := h.call("POST", path, ReportRequest{StreamDone: true, Jobs: []store.JobRecord{{Seq: 2, DurationMs: 10, ArrivedAt: *at(30)}}}, nil); code != 409 {
		t.Fatalf("arrival after stream_done: got %d want 409", code)
	}
}

// Phase 2 step 5: the scheduler exposes the roadmap's metrics at /metrics, and
// finishing a round through the reaper observes its duration, straggler lag
// and the GPU builds' queue waits (decision 31: round metrics are observed
// when the round is stamped finished).
func TestMetricsExposedAndRoundObserved(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "metrics", Seed: 1, NShards: 1}, &cr)
	var sb SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: cr.RoundID, ShardID: 0, NVectors: 1000, Dim: 8}, &sb)
	c, ok, err := h.st.Claim(ctx, "w", 1<<30, time.Minute)
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	if _, err := h.st.Complete(ctx, c.BuildID, c.Attempt); err != nil {
		t.Fatal(err)
	}
	if err := h.st.ReportStatus(ctx, store.ShardStatus{RoundID: cr.RoundID, ShardID: 0, StreamDone: true}); err != nil {
		t.Fatal(err)
	}
	reaper.Tick(ctx, h.st, 5, 30*time.Second, h.log)

	resp, err := http.Get(h.srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	text := string(body)
	for _, name := range []string{"round_duration_seconds_count", "shard_lag_seconds_count", "build_queue_wait_seconds_count",
		"placement_decisions_total", "lease_expirations_total", "gpu_queue_depth", "gpu_workers_live", "builds_leased", "rounds_open"} {
		if !strings.Contains(text, name) {
			t.Errorf("/metrics lacks %s", name)
		}
	}
	line := func(prefix string) string {
		for _, l := range strings.Split(text, "\n") {
			if strings.HasPrefix(l, prefix) {
				return l
			}
		}
		return "<missing>"
	}
	t.Logf("after one round through the reaper:\n    %s\n    %s\n    %s\n    %s", line("round_duration_seconds_count"),
		line("shard_lag_seconds_count"), line(`build_queue_wait_seconds_count{placement="gpu"}`), line(`placement_decisions_total{placement="gpu"`))
	// Other tests in this package also finish rounds, so assert "at least".
	for _, prefix := range []string{"round_duration_seconds_count ", `build_queue_wait_seconds_count{placement="gpu"} `} {
		l := line(prefix)
		if l == "<missing>" || strings.HasSuffix(l, " 0") {
			t.Errorf("expected %s to have observations, got %q", prefix, l)
		}
	}
}

// Decision 40: past SIGTERM the scheduler is a lame duck — it answers but
// never stamps a round, so a round's metrics cannot land in a process
// Prometheus has stopped scraping. The replacement's reaper stamps instead.
func TestDrainingServesWithoutStamping(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()

	// A 1-byte worker: every build exceeds its memory, so placement is local
	// and the shard itself can finish the build (same trick as the preemption
	// test; a GPU build would sit queued with nobody to run it here).
	h.st.Heartbeat(ctx, "small", 1, "node-1") //nolint:errcheck

	// completeRound makes a one-shard round whose work is all done but whose
	// finished_at nobody has stamped yet (no reaper runs in this test).
	completeRound := func(seed int64) int64 {
		var cr CreateRoundResponse
		h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: seed, NShards: 1}, &cr)
		var b SubmitBuildResponse
		h.call("POST", "/builds", SubmitBuildRequest{RoundID: cr.RoundID, ShardID: 0, NVectors: 100_000, Dim: 128}, &b)
		if b.Placement != "local" {
			t.Fatalf("expected local (1-byte worker), got %s", b.Placement)
		}
		h.call("POST", "/builds/"+b.BuildID+"/done", nil, nil)
		h.call("POST", fmt.Sprintf("/shards/%d/0/report", cr.RoundID), ReportRequest{QueueDepth: 0, StreamDone: true}, nil)
		return cr.RoundID
	}

	t.Log("--- round A, healthy pod: a shard's poll stamps the round (decision 31)")
	a := completeRound(1)
	var rs RoundStatus
	h.call("GET", fmt.Sprintf("/rounds/%d", a), nil, &rs)
	if rs.Round.FinishedAt == nil {
		t.Fatal("healthy GET did not stamp a complete round")
	}
	t.Logf("    round %d stamped on read, as before", a)

	t.Log("--- round B, draining pod (flag set, as main does at SIGTERM): the poll answers but does not stamp")
	b := completeRound(2)
	h.draining.Store(true)
	t.Cleanup(func() { h.draining.Store(false) })
	for i := 1; i <= 3; i++ {
		h.call("GET", fmt.Sprintf("/rounds/%d", b), nil, &rs)
		if rs.Round.FinishedAt != nil {
			t.Fatalf("draining GET #%d stamped the round", i)
		}
	}
	if len(rs.Builds) != 1 || rs.Builds[0].State != "done" {
		t.Fatalf("draining answer should still carry the full status, got builds %+v", rs.Builds)
	}
	t.Logf("    3 polls answered truthfully (round open, build done), finished_at still NULL in Postgres")

	t.Log("--- writes that must survive the shutdown window (bug 10) still work while draining")
	if code := h.call("POST", fmt.Sprintf("/shards/%d/0/report", b), ReportRequest{QueueDepth: 0, StreamDone: true}, nil); code != 200 {
		t.Fatalf("shard report while draining: %d", code)
	}

	t.Log("--- the replacement's reaper stamps; the draining pod serves what is already stamped")
	reaper.Tick(ctx, h.st, 5, 30*time.Second, h.log)
	h.call("GET", fmt.Sprintf("/rounds/%d", b), nil, &rs)
	if rs.Round.FinishedAt == nil {
		t.Fatal("round not stamped by the reaper tick, or the draining pod hid it")
	}
	t.Logf("    round %d now shows finished_at %s: the cost of draining is at most one reaper interval",
		b, rs.Round.FinishedAt.Format(time.RFC3339Nano))
}

// waitMs asserts the estimate is present (a fresh submit always carries one,
// even the cold-start 0) and returns it.
func waitMs(t *testing.T, r SubmitBuildResponse) int64 {
	t.Helper()
	if r.GPUQueueWaitMs == nil {
		t.Fatalf("gpu_queue_wait_ms absent where an estimate was expected: %+v", r)
	}
	return *r.GPUQueueWaitMs
}

func TestSubmitQueueWaitEstimate(t *testing.T) {
	h := newHarness(t)
	ctx := context.Background()
	if err := h.st.Heartbeat(ctx, "gpu-1", 8<<30, "node-1"); err != nil {
		t.Fatal(err)
	}
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 1, NShards: 4}, &cr)
	r := cr.RoundID

	t.Log("--- idle worker, empty queue: the first build should see a 0 wait")
	var a SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 1, NVectors: 100_000, Dim: 128}, &a)
	if waitMs(t, a) != 0 {
		t.Fatalf("first build on an idle pool: want wait 0, got %d ms", waitMs(t, a))
	}
	t.Logf("    build 1 placed %s, gpu_queue_wait_ms %d", a.Placement, waitMs(t, a))

	t.Log("--- a second identical build joins behind the first: its wait is the first's whole gpu total")
	var b SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 2, NVectors: 100_000, Dim: 128}, &b)
	if diff := waitMs(t, b) - a.GPUTotalMs; diff < -5 || diff > 5 {
		t.Fatalf("second build: want wait ~= %d ms (one gpu total ahead), got %d ms", a.GPUTotalMs, waitMs(t, b))
	}
	t.Logf("    build 2 waits %d ms behind build 1 (gpu total %d ms): the replay queues them on the one worker", waitMs(t, b), a.GPUTotalMs)

	t.Log("--- a duplicate submit answers the original decision, with no fresh estimate (no gpu_queue_wait_ms at all)")
	var dup SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 2, NVectors: 100_000, Dim: 128}, &dup)
	if dup.Created || dup.GPUQueueWaitMs != nil {
		t.Fatalf("duplicate: want created=false and gpu_queue_wait_ms absent, got %+v", dup)
	}
	t.Log("    created=false, gpu_queue_wait_ms omitted (cross-check F1: a plain int64 served 0 for both)")

	t.Log("--- the worker claims build 1; a third build sees: leased remaining + build 2's total")
	if _, ok, err := h.st.Claim(ctx, "gpu-1", 8<<30, time.Minute); err != nil || !ok {
		t.Fatalf("claim: %v %v", err, ok)
	}
	var c SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 3, NVectors: 100_000, Dim: 128}, &c)
	if waitMs(t, c) <= waitMs(t, b) || waitMs(t, c) > 2*a.GPUTotalMs+5 {
		t.Fatalf("third build: want a wait in (one, two] gpu totals (%d..%d ms], got %d ms",
			waitMs(t, b), 2*a.GPUTotalMs, waitMs(t, c))
	}
	t.Logf("    build 3 waits %d ms: the lease's elapsed time already counts toward build 1", waitMs(t, c))

	t.Log("--- a build too big for every live worker reports -1: no wait would ever end (bug 11's truth)")
	var huge SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: r, ShardID: 4, NVectors: 20_000_000, Dim: 128}, &huge)
	if huge.Placement != "local" || waitMs(t, huge) != -1 {
		t.Fatalf("huge build: want local and wait -1, got %+v", huge)
	}
	t.Logf("    placed %s, gpu_queue_wait_ms %d", huge.Placement, waitMs(t, huge))
}

func TestSubmitQueueWaitColdStart(t *testing.T) {
	h := newHarness(t)
	var cr CreateRoundResponse
	h.call("POST", "/rounds", CreateRoundRequest{Scenario: "test", Seed: 1, NShards: 1}, &cr)

	t.Log("--- no workers at all: the cold-start rule keeps the wait at 0, and always_gpu still queues")
	var b SubmitBuildResponse
	h.call("POST", "/builds", SubmitBuildRequest{RoundID: cr.RoundID, ShardID: 0, NVectors: 100_000, Dim: 128}, &b)
	if b.Placement != "gpu" || waitMs(t, b) != 0 {
		t.Fatalf("cold start: want gpu with wait 0, got %+v", b)
	}
	t.Logf("    placed %s, gpu_queue_wait_ms %d: an empty pool may be about to fill (G1)", b.Placement, waitMs(t, b))
}
