package sim

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
)

// A stand-in for the builder CLI: the same streams and exit codes (Phase 4
// decision 17), no hnswlib. STUB_MODE: ok, gate (exit 3), hang (60 s per
// progress line, for the kill).
const stubBuilder = `import json, os, sys, time
mode = os.environ.get("STUB_MODE", "ok")
for i in range(1, 6):
    print(json.dumps({"progress": i / 5}), file=sys.stderr, flush=True)
    time.sleep(60 if mode == "hang" else 0.1)
if mode == "gate":
    print(json.dumps({"status": "failed", "reason": "recall@10 0.8400 < 0.90"}), flush=True); sys.exit(3)
print(json.dumps({"status": "done", "recall": 0.95, "index_bytes": 12345678, "build_s": 0.5, "gate_s": 0.1}), flush=True)
`

// Phase 4 decision 19: with a dataset the shard's local build is the builder
// CLI as a child process in its own process group — progress from its
// stderr, done or failed by its exit code, killed on preempt.
func TestRealBuildChildProcess(t *testing.T) {
	stub := filepath.Join(t.TempDir(), "stub.py")
	if err := os.WriteFile(stub, []byte(stubBuilder), 0o644); err != nil {
		t.Fatal(err)
	}
	cfgFor := func() RoundConfig {
		return RoundConfig{
			Dataset: "stub",
			Shard: Config{TimeScale: scale, PollInterval: 100 * time.Millisecond, Slice: 20 * time.Millisecond,
				Dataset: "stub", NShards: 1, BuilderCmd: []string{"python3", stub}, Threads: 2},
			MinWorkers: 1, WaitWorkers: 5 * time.Second, RoundTimeout: 60 * time.Second,
		}
	}
	oneShard := func(name string, idx bool) workload.Workload {
		return workload.Workload{
			Scenario: workload.Scenario{Name: name, NShards: 1, Dim: 128, Stream: workload.StreamProfile{Horizon: 20 * time.Second}},
			Seed:     1,
			Shards: []workload.ShardWorkload{{ShardID: 0, NVectors: 100_000, Queries: []workload.Query{
				q(200*time.Millisecond, 500*time.Millisecond, idx, workload.KindSingleShard, 1),
			}}},
		}
	}

	t.Run("ExitZeroIsDoneWithTheManifestSummarised", func(t *testing.T) {
		t.Setenv("STUB_MODE", "ok")
		h := newHarness(t)
		h.fakeWorker("tiny", 1, scale) // nothing fits: the build goes local
		time.Sleep(100 * time.Millisecond)
		tl, err := RunRound(context.Background(), h.api, oneShard("test-real-ok", true), cfgFor(), h.log)
		if err != nil {
			t.Fatal(err)
		}
		var done string
		for _, e := range tl.Shards[0].Events {
			t.Logf("  +%5.2fs %s", e.At.Seconds(), e.Msg)
			if strings.HasPrefix(e.Msg, "local build done") {
				done = e.Msg
			}
		}
		s := tl.Shards[0]
		if s.Placement != "local" || s.BuildState != "done" || !strings.Contains(done, "recall@10 0.950") || !strings.Contains(done, "12.3 MB") {
			t.Fatalf("want a done local build whose event quotes the manifest; got %s/%s %q", s.Placement, s.BuildState, done)
		}
		var ds *string
		h.pool.QueryRow(context.Background(), `SELECT dataset FROM rounds WHERE round_id=$1`, tl.RoundID).Scan(&ds)
		if ds == nil || *ds != "stub" {
			t.Fatalf("the round must carry the dataset (decision 15), got %v", ds)
		}
		if s.Queries[0].Started < s.BuildDone {
			t.Fatal("the index query ran before the child process finished")
		}
		t.Logf("[db] round %d dataset=%s; the child built (exit 0) -> POST /builds/{id}/done; the index query waited for it", tl.RoundID, *ds)
	})

	t.Run("ExitThreeIsFailedWithTheGatesReason", func(t *testing.T) {
		t.Setenv("STUB_MODE", "gate")
		h := newHarness(t)
		h.fakeWorker("tiny", 1, scale)
		time.Sleep(100 * time.Millisecond)
		tl, err := RunRound(context.Background(), h.api, oneShard("test-real-gate", false), cfgFor(), h.log)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range tl.Shards[0].Events {
			t.Logf("  +%5.2fs %s", e.At.Seconds(), e.Msg)
		}
		s := tl.Shards[0]
		var reason string
		h.pool.QueryRow(context.Background(), `SELECT fail_reason FROM builds WHERE round_id=$1 AND shard_id=0`, tl.RoundID).Scan(&reason)
		if s.BuildState != "failed" || reason != "GateRefused: recall@10 0.8400 < 0.90" {
			t.Fatalf("want failed with the gate's reason; got %s reason %q", s.BuildState, reason)
		}
		t.Logf("[db] build failed, fail_reason=%q (exit 3 -> POST /builds/{id}/fail); the round still completed because no query needed the index", reason)
	})

	t.Run("PreemptKillsTheChild", func(t *testing.T) {
		t.Setenv("STUB_MODE", "hang")
		h := newHarness(t)
		h.fakeWorker("tiny", 1, scale)
		time.Sleep(100 * time.Millisecond)
		w := oneShard("test-real-preempt", true)
		go func() {
			ctx := context.Background()
			deadline := time.Now().Add(10 * time.Second)
			for time.Now().Before(deadline) {
				var id string
				var progress float32
				err := h.pool.QueryRow(ctx, `SELECT b.build_id, COALESCE(s.build_progress, 0) FROM builds b
					LEFT JOIN shard_status s ON s.round_id = b.round_id AND s.shard_id = b.shard_id
					WHERE b.placement = 'local' AND b.state = 'running' LIMIT 1`).Scan(&id, &progress)
				if err == nil && progress > 0.1 { // the child reported its first progress line: it is running
					ok, _ := h.st.PreemptToGPU(ctx, id, 50)
					h.t.Logf("  [reconsider] preempt %s at progress %.0f%% -> %v", id, 100*progress, ok)
					h.fakeWorker("big", 8<<30, scale)
					return
				}
				time.Sleep(50 * time.Millisecond)
			}
		}()
		t0 := time.Now()
		tl, err := RunRound(context.Background(), h.api, w, cfgFor(), h.log)
		if err != nil {
			t.Fatal(err)
		}
		var aborted string
		for _, e := range tl.Shards[0].Events {
			t.Logf("  +%5.2fs %s", e.At.Seconds(), e.Msg)
			if strings.Contains(e.Msg, "aborted") {
				aborted = e.Msg
			}
		}
		s := tl.Shards[0]
		if !strings.Contains(aborted, "child process killed") || s.Placement != "gpu" || s.BuildState != "done" {
			t.Fatalf("want the child killed on preempt and a gpu completion; got %q %s/%s", aborted, s.Placement, s.BuildState)
		}
		if time.Since(t0) > 30*time.Second {
			t.Fatalf("the round took %s: the child's 60 s sleep was not cut short", time.Since(t0))
		}
		out, _ := exec.Command("sh", "-c", "ps -axo pid,command | grep '"+stub+"' | grep -v grep").Output()
		if strings.TrimSpace(string(out)) != "" {
			t.Fatalf("a stub process survived the kill:\n%s", out)
		}
		t.Logf("the child (which would have slept 60 s per progress line) was killed on the preempt within %s; no stub process survives; the gpu worker finished the build", time.Since(t0).Round(time.Millisecond))
	})
}

// Phase 4 decision 21: a real build that fails for good is rebuilt locally
// once, at the next attempt; if that fails too, the index queries run
// without the index and the round completes with the failure on record.
func TestFailedBuildRetriesOnceThenFallsBack(t *testing.T) {
	// a stub that refuses at the gate on every run, counting its runs
	dir := t.TempDir()
	stub := filepath.Join(dir, "stub.py")
	counter := filepath.Join(dir, "runs")
	script := `import json, os, sys
runs = open(sys.argv[-1]).read().count("x") if os.path.exists(sys.argv[-1]) else 0
open(sys.argv[-1], "a").write("x")
print(json.dumps({"progress": 1.0}), file=sys.stderr, flush=True)
print(json.dumps({"status": "failed", "reason": "recall@10 0.8000 < 0.90 (run %d, attempt %s)" % (runs + 1, sys.argv[sys.argv.index("--attempt") + 1])}), flush=True)
sys.exit(3)
`
	if err := os.WriteFile(stub, []byte(script), 0o644); err != nil {
		t.Fatal(err)
	}
	h := newHarness(t)
	h.fakeWorker("tiny", 1, scale)
	time.Sleep(100 * time.Millisecond)
	cfg := RoundConfig{
		Dataset: "stub",
		Shard: Config{TimeScale: scale, PollInterval: 100 * time.Millisecond, Slice: 20 * time.Millisecond,
			Dataset: "stub", NShards: 1, BuilderCmd: []string{"python3", stub, "--", counter}, Threads: 1},
		MinWorkers: 1, WaitWorkers: 5 * time.Second, RoundTimeout: 60 * time.Second,
	}
	// the stub ignores the real flags; the counter path is its last argument,
	// so give it as part of the command and let the build flags follow
	cfg.Shard.BuilderCmd = []string{"python3", stub}
	script2 := strings.ReplaceAll(script, `sys.argv[-1]`, `"`+counter+`"`)
	if err := os.WriteFile(stub, []byte(script2), 0o644); err != nil {
		t.Fatal(err)
	}
	w := workload.Workload{
		Scenario: workload.Scenario{Name: "test-retry-fallback", NShards: 1, Dim: 128, Stream: workload.StreamProfile{Horizon: 20 * time.Second}},
		Seed:     1,
		Shards: []workload.ShardWorkload{{ShardID: 0, NVectors: 100_000, Queries: []workload.Query{
			q(200*time.Millisecond, 300*time.Millisecond, true, workload.KindSingleShard, 1),
			q(300*time.Millisecond, 300*time.Millisecond, true, workload.KindSingleShard, 2),
		}}},
	}
	t0 := time.Now()
	tl, err := RunRound(context.Background(), h.api, w, cfg, h.log)
	if err != nil {
		t.Fatal(err)
	}
	var retried, fallback bool
	for _, e := range tl.Shards[0].Events {
		t.Logf("  +%5.2fs %s", e.At.Seconds(), e.Msg)
		retried = retried || strings.Contains(e.Msg, "local retry granted at attempt 1")
		fallback = fallback || strings.Contains(e.Msg, "will run without it")
	}
	s := tl.Shards[0]
	runs, _ := os.ReadFile(counter)
	var attempt, retries int32
	var reason string
	h.pool.QueryRow(context.Background(), `SELECT attempt, local_retries, fail_reason FROM builds WHERE round_id=$1 AND shard_id=0`, tl.RoundID).Scan(&attempt, &retries, &reason)
	t.Logf("[db] build: state=%s attempt=%d local_retries=%d fail_reason=%q; the stub ran %d times; round took %s", s.BuildState, attempt, retries, reason, len(runs), time.Since(t0).Round(time.Millisecond))
	if !retried || !fallback || s.BuildState != "failed" || len(runs) != 2 || attempt != 1 || retries != 1 || !strings.Contains(reason, "attempt 1") {
		t.Fatalf("want one local retry at attempt 1 (the stub run twice), then failed for good with the fallback; got retried=%v fallback=%v state=%s runs=%d attempt=%d retries=%d reason=%q",
			retried, fallback, s.BuildState, len(runs), attempt, retries, reason)
	}
	for _, ql := range s.Queries {
		if ql.Started < 0 || ql.Finished < 0 {
			t.Fatalf("index query %d never ran: the fallback must release it", ql.Seq)
		}
	}
	var finished *time.Time
	h.pool.QueryRow(context.Background(), `SELECT finished_at FROM rounds WHERE round_id=$1`, tl.RoundID).Scan(&finished)
	if finished == nil {
		t.Fatal("the round must complete with the failed build on record")
	}
	t.Logf("both index queries ran without the index (the exact-scan fallback) and the round completed; the retry built under attempt 1, its own artifact prefix")
}
