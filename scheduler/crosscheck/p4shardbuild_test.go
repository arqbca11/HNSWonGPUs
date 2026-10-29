package crosscheck

// Phase 4 step 2 cross-check: the shard's local build as a child process
// (decision 19) and burning queries (decision 20), driven only through
// shard/sim.RunRound (its exported Config/RoundConfig) against the in-process
// scheduler, with a stub builder in place of `python -m kgpu_builder`. The
// stub honours the CLI contract of decision 17: argv `build --build-id K:R
// --attempt A --dataset DS --n-shards N --shard K --threads T`, progress as
// JSON lines on stderr, one JSON document on stdout, exit 0 / 3 / 1 / 2.

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
	"github.com/arqbca11/HNSWonGPUs/shard/client"
	"github.com/arqbca11/HNSWonGPUs/shard/sim"
)

const p4StubHead = `#!/bin/bash
DIR=%q
echo "$$ $*" >> "$DIR/argv.log"
env | grep -E '^(S3_|RECALL_MIN|KGPU_)' > "$DIR/env.$$" || true
BID=; ATT=
while [ $# -gt 0 ]; do case "$1" in --build-id) BID=$2; shift 2;; --attempt) ATT=$2; shift 2;; *) shift;; esac; done
manifest() { printf '{"status":"done","kind":"hnswlib","build_id":"%%s","attempt":%%s,"n_vectors":1000,"dim":128,"index_bytes":4096,"index_sha256":"00","recall":0.99}\n' "$BID" "$ATT"; }
`

// p4Stub writes an executable stub builder whose body is appended to the
// common head; it returns the command and the directory it records into.
func p4Stub(t *testing.T, body string) ([]string, string) {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "stub-builder.sh")
	if err := os.WriteFile(p, []byte(fmt.Sprintf(p4StubHead, dir)+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return []string{p}, dir
}

func p4Workload(nv int64, qs ...workload.Query) workload.Workload {
	return workload.Workload{
		Scenario: workload.Scenario{Name: "xc-p4", NShards: 1, Dim: 128, Stream: workload.StreamProfile{Horizon: 10 * time.Second}},
		Seed:     1,
		Shards:   []workload.ShardWorkload{{ShardID: 0, NVectors: nv, Queries: qs}},
	}
}

func p4Q(at, dur time.Duration, idx bool, id int64) workload.Query {
	return workload.Query{Arrival: at, Duration: dur, NeedsIndex: idx, Kind: workload.KindSingleShard, ClusterQueryID: id}
}

func p4Cfg(dataset string, cmd []string, timeout time.Duration) sim.RoundConfig {
	return sim.RoundConfig{
		Dataset: dataset,
		Shard: sim.Config{TimeScale: 1, PollInterval: 100 * time.Millisecond, Slice: 20 * time.Millisecond,
			Dataset: dataset, NShards: 1, BuilderCmd: cmd, Threads: 2, BurnQueries: dataset != ""},
		MinWorkers: 1, WaitWorkers: 5 * time.Second, RoundTimeout: timeout,
	}
}

type p4Result struct {
	tl  *sim.Timeline
	err error
}

// p4Run runs one round with a 1-byte worker in the pool (every build goes
// local: it fits no worker) and returns the timeline and the build row.
func p4Run(t *testing.T, e *simEnv, w workload.Workload, cfg sim.RoundConfig) (p4Result, store.BuildRow, int64) {
	t.Helper()
	startFakeGPU(t, e, "xc-tiny", 1, 1, 1, false)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	tl, err := sim.RunRound(context.Background(), client.New(e.srv.URL), w, cfg, logger)
	round := e.waitFirstRoundQuick(t)
	var b store.BuildRow
	if round > 0 {
		b, _, _ = e.st.GetBuild(e.ctx, store.BuildID(0, round))
	}
	if tl != nil {
		for _, s := range tl.Shards {
			for _, ev := range s.Events {
				t.Logf("  timeline shard %d +%6.2fs %s", s.ShardID, ev.At.Seconds(), ev.Msg)
			}
		}
	}
	t.Logf("RunRound err=%v", err)
	if round > 0 {
		logBuild(t, "build after the round", b)
		t.Logf("[db] fail_reason=%q", derefS(b.FailReason))
	}
	return p4Result{tl, err}, b, round
}

func (e *simEnv) waitFirstRoundQuick(t *testing.T) int64 {
	var id int64
	if err := e.pool.QueryRow(e.ctx, `SELECT round_id FROM rounds ORDER BY round_id LIMIT 1`).Scan(&id); err != nil {
		return 0
	}
	return id
}

func readArgv(t *testing.T, dir string) []string {
	raw, _ := os.ReadFile(filepath.Join(dir, "argv.log"))
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

// Decision 19: stderr progress becomes build_progress in the shard's reports;
// exit 0 is POST /builds/{id}/done. Decision 17 gives the argv.
func TestXcP4S2ShardChildProgressReachesTheScheduler(t *testing.T) {
	e := setupSim(t)
	cmd, dir := p4Stub(t, `for i in 1 2 3 4 5 6 7 8 9 10; do sleep 0.3; echo "{\"progress\": 0.$i}" | sed 's/0.10/1.0/' >&2; done
echo '{"phase": "gate"}' >&2
manifest
exit 0
`)
	seen := map[float32]bool{}
	var mu sync.Mutex
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			if r := e.waitFirstRoundQuick(t); r > 0 {
				if ss, err := e.st.ListShardStatus(e.ctx, r); err == nil {
					mu.Lock()
					for _, s := range ss {
						seen[s.BuildProgress] = true
					}
					mu.Unlock()
				}
			}
			time.Sleep(30 * time.Millisecond)
		}
	}()
	res, b, round := p4Run(t, e, p4Workload(1000, p4Q(200*time.Millisecond, 50*time.Millisecond, false, 1)), p4Cfg("xtoy", cmd, 30*time.Second))
	close(stop)
	mu.Lock()
	var vals []float32
	for v := range seen {
		vals = append(vals, v)
	}
	mu.Unlock()
	argv := readArgv(t, dir)
	t.Logf("shard_status.build_progress values seen while polling: %v", vals)
	t.Logf("stub argv: %v", argv)
	if res.err != nil || b.State != "done" || b.Placement != "local" {
		t.Fatalf("decision 19: exit 0 is POST /builds/{id}/done; got err=%v %s/%s", res.err, b.Placement, b.State)
	}
	mid := 0
	for _, v := range vals {
		if v > 0 && v < 1 {
			mid++
		}
	}
	if mid < 2 {
		t.Errorf("decision 19: the child's progress should reach build_progress during the build; saw %v", vals)
	}
	want := fmt.Sprintf("build --build-id 0:%d --attempt %d --dataset xtoy --n-shards 1 --shard 0 --threads 2", round, b.Attempt)
	if len(argv) != 1 || !strings.HasSuffix(argv[0], want) {
		t.Errorf("decision 17 argv: want one invocation ending %q, got %v", want, argv)
	}
}

// Garbage on stderr (tracebacks, non-JSON, wrong types, out of range) must
// not break the build; exit 0 is still done. Report what progress was kept.
func TestXcP4S2ShardChildGarbageOnStderr(t *testing.T) {
	cases := []struct{ name, lines string }{
		{"non-JSON text", "echo 'Traceback (most recent call last):' >&2\necho '{not json' >&2\necho '' >&2\nprintf '\\x00\\xff\\xfe binary\\n' >&2"},
		{"progress as a string", `echo '{"progress": "abc"}' >&2`},
		{"progress above 1", `echo '{"progress": 7}' >&2`},
		{"progress below 0", `echo '{"progress": -1}' >&2`},
	}
	for _, c := range cases {
		t.Run(strings.ReplaceAll(c.name, " ", "_"), func(t *testing.T) {
			e := setupSim(t)
			cmd, _ := p4Stub(t, "echo '{\"progress\": 0.2}' >&2\nsleep 0.3\n"+c.lines+"\nsleep 0.4\necho '{\"progress\": 0.5}' >&2\nsleep 0.4\necho '{\"phase\": \"gate\"}' >&2\nmanifest\nexit 0\n")
			var maxP, minP float32
			stop := make(chan struct{})
			go func() {
				for {
					select {
					case <-stop:
						return
					default:
					}
					if r := e.waitFirstRoundQuick(t); r > 0 {
						if ss, _ := e.st.ListShardStatus(e.ctx, r); len(ss) > 0 {
							maxP = max(maxP, ss[0].BuildProgress)
							minP = min(minP, ss[0].BuildProgress)
						}
					}
					time.Sleep(20 * time.Millisecond)
				}
			}()
			res, b, _ := p4Run(t, e, p4Workload(1000), p4Cfg("xtoy", cmd, 30*time.Second))
			close(stop)
			t.Logf("%s on stderr: build_progress range observed [%v, %v]; RunRound err=%v; build %s", c.name, minP, maxP, res.err, b.State)
			if res.err != nil || b.State != "done" {
				t.Errorf("a stray stderr line (%s) should not break a build that exits 0 (decision 17: the exit code and stdout are the interface); got err=%v state=%s", c.name, res.err, b.State)
			}
			if maxP > 1 || minP < 0 {
				t.Errorf("build_progress left [0,1]: %v..%v", minP, maxP)
			}
		})
	}
}

// Exit 0 with an unparsable stdout: decision 17 says stdout is the manifest;
// decision 19 says exit 0 is done. Record which wins.
func TestXcP4S2ShardChildExit0UnparsableStdout(t *testing.T) {
	e := setupSim(t)
	cmd, _ := p4Stub(t, `echo '{"progress": 1.0}' >&2
echo 'this is not a manifest'
exit 0
`)
	res, b, _ := p4Run(t, e, p4Workload(1000), p4Cfg("xtoy", cmd, 15*time.Second))
	t.Logf("exit 0 with stdout 'this is not a manifest': RunRound err=%v, build %s/%s, reason %q  [spec silent: exit 0 is done (dec 19) vs stdout is the manifest (dec 17)]",
		res.err, b.Placement, b.State, derefS(b.FailReason))
	if b.State == "running" {
		t.Errorf("the build must not be left running forever (round cannot finish)")
	}
}

// Exit 3: POST /builds/{id}/fail with the gate's reason (decision 19); a
// reason of 2,000 characters is cut to 500 (brief); the round still finishes
// when no query needs the index.
func TestXcP4S2ShardChildExit3FailsWithTheReason(t *testing.T) {
	e := setupSim(t)
	long := "recall@10 0.42 < 0.90 " + strings.Repeat("x", 2000)
	cmd, _ := p4Stub(t, fmt.Sprintf(`echo '{"progress": 1.0}' >&2
echo '{"phase": "gate"}' >&2
printf '{"status":"failed","reason":"%%s"}\n' %q
exit 3
`, long))
	res, b, _ := p4Run(t, e, p4Workload(1000, p4Q(100*time.Millisecond, 50*time.Millisecond, false, 1)), p4Cfg("xtoy", cmd, 30*time.Second))
	r := derefS(b.FailReason)
	t.Logf("exit 3: build %s/%s, reason %d chars, prefix %q", b.Placement, b.State, len(r), r[:min(40, len(r))])
	if b.State != "failed" || !strings.Contains(r, "recall@10 0.42 < 0.90") {
		t.Errorf("decision 19: exit 3 is POST /builds/{id}/fail with the reason; got %s %q", b.State, r[:min(40, len(r))])
	}
	if len(r) > 500 {
		t.Errorf("the reason should be at most 500 characters; got %d", len(r))
	}
	if res.err != nil {
		t.Errorf("a failed build is terminal; the round (no needs_index queries) should finish: %v", res.err)
	}
}

// Exits other than 0 and 3: decision 17 defines 1 (anything else) and 2
// (usage); decision 19 says only what 0 and 3 do. Record what the shard does.
func TestXcP4S2ShardChildExit1And2(t *testing.T) {
	for _, code := range []int{1, 2} {
		t.Run(fmt.Sprintf("exit%d", code), func(t *testing.T) {
			e := setupSim(t)
			cmd, dir := p4Stub(t, fmt.Sprintf(`echo '{"progress": 0.3}' >&2
echo 'usage: kgpu_builder build ... error' >&2
printf '{"status":"failed","reason":"exit %d from the stub"}\n'
exit %d
`, code, code))
			res, b, _ := p4Run(t, e, p4Workload(1000), p4Cfg("xtoy", cmd, 12*time.Second))
			t.Logf("exit %d: RunRound err=%v; build %s/%s reason %q; the stub ran %d time(s)  [spec: decision 19 names only exit 0 and 3]",
				code, res.err, b.Placement, b.State, derefS(b.FailReason), len(readArgv(t, dir)))
			if b.State == "running" {
				t.Errorf("exit %d left the local build running: the round can never finish (spec gap: what a shard does with exit 1/2)", code)
			}
		})
	}
}

// A child that ignores SIGTERM and has forked a grandchild: a preempt must
// still kill both (decision 19: "a preempt ... kills the group"), and the
// build then completes on the GPU.
func TestXcP4S2ShardPreemptKillsTheWholeGroup(t *testing.T) {
	e := setupSim(t)
	cmd, dir := p4Stub(t, `trap '' TERM INT HUP
sleep 1000 &
echo $! > "$DIR/grandchild.pid"
echo $$ > "$DIR/child.pid"
i=0
while true; do i=$((i+1)); echo "{\"progress\": 0.$((i%9+1))}" >&2; sleep 0.2; done
`)
	var preemptAt time.Time
	var cpid, gpid int
	done := make(chan struct{})
	go func() {
		defer close(done)
		deadline := time.Now().Add(15 * time.Second)
		for time.Now().Before(deadline) {
			r := e.waitFirstRoundQuick(t)
			if r > 0 {
				ss, _ := e.st.ListShardStatus(e.ctx, r)
				if len(ss) > 0 && ss[0].BuildProgress > 0 {
					c, _ := os.ReadFile(filepath.Join(dir, "child.pid"))
					g, _ := os.ReadFile(filepath.Join(dir, "grandchild.pid"))
					fmt.Sscan(string(c), &cpid)
					fmt.Sscan(string(g), &gpid)
					ok, err := e.st.PreemptToGPU(e.ctx, store.BuildID(0, r), 50)
					preemptAt = time.Now()
					t.Logf("[reconsider] preempt at progress %.1f: ok=%v err=%v; child pid %d, grandchild pid %d", ss[0].BuildProgress, ok, err, cpid, gpid)
					startFakeGPU(t, e, "xc-big", 1<<40, 1<<40, 100, true)
					return
				}
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()
	// The shard must react before the round times out; check the processes
	// from a second goroutine so the result does not depend on RunRound.
	type death struct{ child, grand time.Duration }
	deathCh := make(chan death, 1)
	go func() {
		<-done
		var d death
		for i := 0; i < 200; i++ {
			if cpid > 0 && d.child == 0 && !alive(cpid) {
				d.child = time.Since(preemptAt)
			}
			if gpid > 0 && d.grand == 0 && !alive(gpid) {
				d.grand = time.Since(preemptAt)
			}
			if d.child > 0 && d.grand > 0 {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		deathCh <- d
	}()
	res, b, _ := p4Run(t, e, p4Workload(1000, p4Q(100*time.Millisecond, 50*time.Millisecond, false, 1)), p4Cfg("xtoy", cmd, 30*time.Second))
	d := <-deathCh
	t.Logf("after the preempt: child (ignores SIGTERM) gone after %v, grandchild gone after %v (0 = still alive after 10 s)", d.child, d.grand)
	t.Logf("round: err=%v; build %s/%s attempt %d", res.err, b.Placement, b.State, b.Attempt)
	if d.child == 0 || d.grand == 0 {
		t.Errorf("decision 19: a preempt kills the builder's process group; child dead=%v grandchild dead=%v", d.child > 0, d.grand > 0)
		if gpid > 0 {
			syscall.Kill(gpid, syscall.SIGKILL) //nolint:errcheck
		}
		if cpid > 0 {
			syscall.Kill(cpid, syscall.SIGKILL) //nolint:errcheck
		}
	}
	if b.Placement != "gpu" || b.State != "done" {
		t.Errorf("the preempted build should complete on the GPU; got %s/%s", b.Placement, b.State)
	}
}

func alive(pid int) bool {
	if err := syscall.Kill(pid, 0); err != nil {
		return false
	}
	// a zombie still answers kill(0); ps tells it apart
	out, _ := execOut("ps", "-o", "stat=", "-p", fmt.Sprint(pid))
	return !strings.HasPrefix(strings.TrimSpace(out), "Z") && strings.TrimSpace(out) != ""
}

// With no Dataset the modeled sleep is the local build (decision 19: "The
// modeled sleep remains the behaviour when the round has no dataset"): the
// builder command must never run, and the round records no dataset.
func TestXcP4S2ShardModeledPathUntouchedWithoutDataset(t *testing.T) {
	e := setupSim(t)
	cmd, dir := p4Stub(t, "manifest\nexit 0\n")
	cfg := p4Cfg("", cmd, 30*time.Second)
	cfg.Shard.Threads = 2
	res, b, round := p4Run(t, e, p4Workload(1000, p4Q(100*time.Millisecond, 50*time.Millisecond, false, 1)), cfg)
	_, statErr := os.Stat(filepath.Join(dir, "argv.log"))
	r := e.mustRound(t, round)
	t.Logf("modeled round: err=%v build %s/%s; stub ran: %v; rounds.dataset=%v", res.err, b.Placement, b.State, statErr == nil, derefS(r.Dataset))
	if statErr == nil {
		t.Errorf("decision 19: without a dataset the builder CLI must not run")
	}
	if res.err != nil || b.State != "done" || r.Dataset != nil {
		t.Errorf("the modeled round should complete as before with a NULL dataset")
	}
}

// RunRound with a Dataset records it on the round (decision 15: the shard
// simulator sets it when it creates the round).
func TestXcP4S2RunRoundRecordsTheDataset(t *testing.T) {
	e := setupSim(t)
	cmd, _ := p4Stub(t, "manifest\nexit 0\n")
	_, b, round := p4Run(t, e, p4Workload(1000), p4Cfg("xtoy", cmd, 30*time.Second))
	r := e.mustRound(t, round)
	t.Logf("round %d dataset=%v n_shards=%d; build %s", round, derefS(r.Dataset), r.NShards, b.State)
	if r.Dataset == nil || *r.Dataset != "xtoy" {
		t.Errorf("decision 15: RunRound with Dataset should create the round with it")
	}
}

// Decision 2 / the step 2 flow: "A stale attempt writes under its own attempt
// prefix and can never overwrite the winner." A GPU attempt that lost its
// lease (reaped back to queued) and is then reneged to local: which --attempt
// does the shard's local build use? If it reuses the stale GPU attempt's
// number, both write artifacts/<id>/<that attempt>/.
func TestXcP4S2RenegedBuildAttemptPrefixVersusStaleGPUAttempt(t *testing.T) {
	e := setupSim(t)
	cmd, dir := p4Stub(t, "echo '{\"progress\": 1.0}' >&2\nmanifest\nexit 0\n")
	// A big worker is live so the submit goes to the GPU; this test claims by hand.
	if err := e.st.Heartbeat(e.ctx, "xc-stale", 1<<40, ""); err != nil {
		t.Fatal(err)
	}
	hb := make(chan struct{})
	go func() {
		tk := time.NewTicker(300 * time.Millisecond)
		defer tk.Stop()
		for {
			select {
			case <-hb:
				return
			case <-tk.C:
				e.st.Heartbeat(e.ctx, "xc-stale", 1<<40, "") //nolint:errcheck
			}
		}
	}()
	defer close(hb)
	var claimedAttempt int32
	go func() {
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			c, ok, err := e.st.Claim(e.ctx, "xc-stale", 1<<40, 300*time.Millisecond)
			if err == nil && ok {
				claimedAttempt = c.Attempt
				t.Logf("[stale worker] claimed %s attempt %d with a 300 ms lease; never renews (paused)", c.BuildID, c.Attempt)
				for time.Now().Before(deadline) {
					b, _, _ := e.st.GetBuild(e.ctx, c.BuildID)
					if b.State == "queued" {
						ok, err := e.st.RenegeToLocal(e.ctx, c.BuildID)
						b2, _, _ := e.st.GetBuild(e.ctx, c.BuildID)
						t.Logf("[reaper] lease expired -> queued attempt %d; [reconsider] renege to local: ok=%v err=%v -> %s/%s attempt %d", b.Attempt, ok, err, b2.Placement, b2.State, b2.Attempt)
						return
					}
					time.Sleep(20 * time.Millisecond)
				}
			}
			time.Sleep(20 * time.Millisecond)
		}
	}()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	cfg := p4Cfg("xtoy", cmd, 30*time.Second)
	cfg.MinWorkers = 1
	tl, err := sim.RunRound(context.Background(), client.New(e.srv.URL), p4Workload(1000, p4Q(100*time.Millisecond, 50*time.Millisecond, false, 1)), cfg, logger)
	_ = tl
	argv := readArgv(t, dir)
	round := e.waitFirstRoundQuick(t)
	b := e.mustBuild(t, store.BuildID(0, round))
	logBuild(t, "after", b)
	t.Logf("RunRound err=%v; the shard ran the builder as: %v", err, argv)
	if claimedAttempt == 0 || len(argv) == 0 || argv[0] == "" {
		t.Skipf("the scenario did not happen (claimed attempt %d, stub runs %d)", claimedAttempt, len(argv))
	}
	stale := fmt.Sprintf("--attempt %d ", claimedAttempt)
	if strings.Contains(argv[0], stale) {
		t.Errorf("decision 2 / invariant 7: the local build writes under artifacts/%s/%d/, the same prefix the stale GPU attempt %d (lease lost, possibly still running) writes to; a stale attempt can then overwrite the winner's index.bin and manifest.json",
			b.BuildID, claimedAttempt, claimedAttempt)
	}
}

// Decision 20: with a dataset a query's modeled duration is spent busy on the
// CPU; without one it sleeps. Measured as this process's CPU time (the shard
// goroutines run in it; the stub builder is a child and not counted).
func TestXcP4S2BurnQueriesSpendCPUTime(t *testing.T) {
	var qs []workload.Query
	for i := 0; i < 8; i++ {
		qs = append(qs, p4Q(time.Duration(100+50*i)*time.Millisecond, 250*time.Millisecond, false, int64(i+1)))
	}
	modeled := 8 * 250 * time.Millisecond
	measure := func(label string, dataset string, burn bool) time.Duration {
		e := setupSim(t)
		cmd, _ := p4Stub(t, "manifest\nexit 0\n")
		cfg := p4Cfg(dataset, cmd, 30*time.Second)
		cfg.Shard.BurnQueries = burn
		var r0, r1 syscall.Rusage
		syscall.Getrusage(syscall.RUSAGE_SELF, &r0) //nolint:errcheck
		w0 := time.Now()
		res, b, _ := p4Run(t, e, p4Workload(1000, qs...), cfg)
		syscall.Getrusage(syscall.RUSAGE_SELF, &r1) //nolint:errcheck
		cpu := time.Duration(r1.Utime.Nano()+r1.Stime.Nano()-r0.Utime.Nano()-r0.Stime.Nano()) * time.Nanosecond
		t.Logf("%s: dataset=%q BurnQueries=%v: process CPU %v over wall %v for %v of modeled query time; round err=%v build %s",
			label, dataset, burn, cpu.Round(time.Millisecond), time.Since(w0).Round(time.Millisecond), modeled, res.err, b.State)
		return cpu
	}
	burn := measure("burn", "xtoy", true)
	sleep := measure("sleep", "", false)
	other := measure("dataset without BurnQueries", "xtoy", false)
	t.Logf("burn %v vs sleep %v vs dataset-without-burn %v", burn, sleep, other)
	if burn < modeled*7/10 {
		t.Errorf("decision 20: burning queries should cost about their modeled duration in CPU (>= 70%% of %v); got %v", modeled, burn)
	}
	if sleep > modeled*3/10 {
		t.Errorf("the sleeping path should cost little CPU; got %v", sleep)
	}
}

// Decision 20: shardsim refuses a time scale other than 1 with a dataset,
// with exit 2 and one line, before contacting anything.
func TestXcP4S2ShardsimRefusesTimeScaleWithDataset(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(503)
	}))
	defer srv.Close()
	cases := []map[string]string{
		{"DATASET": "sift-128", "TIME_SCALE": "10", "N_SHARDS": "6"},
		{"DATASET": "sift-128", "TIME_SCALE": "0.5", "N_SHARDS": "6"},
		{"DATASET": "sift-128", "TIME_SCALE": "1.0000001", "N_SHARDS": "6"},
		{"DATASET": "sift-128", "TIME_SCALE": "1", "N_SHARDS": "7"},
		{"DATASET": "no-such-dataset", "TIME_SCALE": "1", "N_SHARDS": "6"},
	}
	for _, vars := range cases {
		vars["SCHEDULER_URL"] = srv.URL
		vars["LOG_FORMAT"] = "text"
		before := hits.Load()
		p := startShardsim(t, vars)
		var code int
		select {
		case <-p.done:
			code = p.cmd.ProcessState.ExitCode()
		case <-time.After(8 * time.Second):
			code = -1
			_ = p.cmd.Process.Kill()
			<-p.done
		}
		errLines := strings.Split(strings.TrimSpace(p.stderr.String()+p.stdout.String()), "\n")
		t.Logf("DATASET=%s TIME_SCALE=%s N_SHARDS=%s: exit %d after %v; %d output line(s) (stderr+stdout), first %q; requests to the scheduler: %d",
			vars["DATASET"], vars["TIME_SCALE"], vars["N_SHARDS"], code, p.exitAt.Sub(p.start).Round(time.Millisecond), len(errLines), errLines[0], hits.Load()-before)
		if vars["TIME_SCALE"] != "1" {
			if code != 2 || hits.Load() != before {
				t.Errorf("decision 20: TIME_SCALE=%s with DATASET must exit 2 before touching anything; exit %d, %d requests", vars["TIME_SCALE"], code, hits.Load()-before)
			}
			if len(errLines) != 1 {
				t.Errorf("one-line message expected; got %d lines", len(errLines))
			}
		} else if code == -1 {
			t.Logf("  NOTE: still running after 8 s (waiting on the scheduler, not refusing the configuration up front)")
		}
	}
}

func execOut(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	return string(out), err
}
