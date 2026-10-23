package reconsider

import (
	"context"
	"io"
	"log/slog"
	"math"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/arqbca11/HNSWonGPUs/db"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

// Integration tests. Skipped unless TEST_DATABASE_URL points at a Postgres
// the test may wipe. scripts/test-db.sh starts one in a container.

const lease = 30 * time.Second

func newStore(t *testing.T) (*store.Store, *pgxpool.Pool) {
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
	return store.New(pool), pool
}

func quiet() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func submit(t *testing.T, s *store.Store, round int64, shard int32, placement string, n int64, mem int64) string {
	t.Helper()
	id := store.BuildID(shard, round)
	ins, err := s.SubmitBuild(context.Background(), store.Build{
		BuildID: id, RoundID: round, ShardID: shard, NVectors: n, Dim: 128,
		MemBytes: mem, Placement: placement,
	})
	if err != nil || !ins {
		t.Fatalf("submit %s: inserted=%v err=%v", id, ins, err)
	}
	return id
}

func getBuild(t *testing.T, s *store.Store, id string) store.BuildRow {
	t.Helper()
	b, found, err := s.GetBuild(context.Background(), id)
	if err != nil || !found {
		t.Fatalf("get %s: found=%v err=%v", id, found, err)
	}
	return b
}

// fakePolicy returns a fixed script of actions, whatever the input: the
// actuator's mechanics are under test, not the policy's judgement.
type fakePolicy struct {
	acts []policy.Action
	last policy.ReconsiderInput
}

func (f *fakePolicy) Name() string                        { return "fake" }
func (f *fakePolicy) Decide(policy.Input) policy.Decision { return policy.Decision{Placement: "gpu"} }
func (f *fakePolicy) Reconsider(in policy.ReconsiderInput) []policy.Action {
	f.last = in
	return f.acts
}

// The tick executes whatever the policy returns, as guarded row changes: one
// of each action kind lands, and an action whose row has moved on since the
// snapshot is skipped without damage.
func TestTickExecutesEveryActionKind(t *testing.T) {
	s, pool := newStore(t)
	ctx := context.Background()
	cm := costmodel.Default()
	cfg := Config{StaleAfter: time.Minute}
	if err := s.Heartbeat(ctx, "w1", 1<<40, "node-a"); err != nil {
		t.Fatal(err)
	}
	r, _ := s.CreateRound(ctx, "test", 1, 10)
	toPreempt := submit(t, s, r, 1, "local", 1000, 1<<20)
	toRenege := submit(t, s, r, 2, "gpu", 1000, 1<<20)
	toReprioritize := submit(t, s, r, 3, "gpu", 1000, 1<<20)
	toCopy := submit(t, s, r, 4, "local", 1000, 1<<20)
	toSkip := submit(t, s, r, 5, "local", 1000, 1<<20)
	if ok, _ := s.CompleteLocal(ctx, toSkip); !ok {
		t.Fatal("complete local failed")
	}
	t.Logf("five builds set up; %s is already done, so the fake policy's preempt of it must be skipped", toSkip)

	pol := &fakePolicy{acts: []policy.Action{
		{Kind: policy.ActPreempt, BuildID: toPreempt, Priority: 11, Reason: "scripted"},
		{Kind: policy.ActRenege, BuildID: toRenege, Reason: "scripted"},
		{Kind: policy.ActReprioritize, BuildID: toReprioritize, Priority: 77, Reason: "scripted"},
		{Kind: policy.ActSpeculate, BuildID: toCopy, Priority: 5, Reason: "scripted"},
		{Kind: policy.ActPreempt, BuildID: toSkip, Priority: 1, Reason: "scripted against a done build"},
	}}
	Tick(ctx, s, pol, cm, cfg, quiet())

	if b := getBuild(t, s, toPreempt); b.Placement != "gpu" || b.State != "queued" || b.Priority != 11 || !b.Moved {
		t.Fatalf("preempt: %s/%s priority=%v moved=%v", b.Placement, b.State, b.Priority, b.Moved)
	}
	t.Logf("%s preempted: local/running -> gpu/queued at priority 11, moved=true", toPreempt)
	if b := getBuild(t, s, toRenege); b.Placement != "local" || b.State != "running" || !b.Moved {
		t.Fatalf("renege: %s/%s moved=%v", b.Placement, b.State, b.Moved)
	}
	t.Logf("%s reneged: gpu/queued -> local/running, moved=true", toRenege)
	if b := getBuild(t, s, toReprioritize); b.Priority != 77 {
		t.Fatalf("reprioritize: priority=%v", b.Priority)
	}
	t.Logf("%s reprioritized to 77", toReprioritize)
	if b := getBuild(t, s, toCopy+":spec"); !b.Speculative || b.State != "queued" || b.Priority != 5 {
		t.Fatalf("speculate: speculative=%v %s priority=%v", b.Speculative, b.State, b.Priority)
	}
	t.Logf("%s:spec exists: a queued speculative gpu copy at priority 5", toCopy)
	if b := getBuild(t, s, toSkip); b.State != "done" || b.Placement != "local" {
		t.Fatalf("the skipped action touched a done build: %s/%s", b.Placement, b.State)
	}
	t.Logf("%s untouched: the guarded preempt saw done and updated zero rows", toSkip)
	_ = pool
}

// The real reactive policy, end to end through a tick: a local build with a
// painful backlog is preempted (and the sunk-cost control is not), a queued
// build no live worker can ever take reneges to local (bug 11's cure at the
// third placement-change site), and reactive_spec starts a copy that a later
// tick resolves when the GPU wins.
func TestTickWithReactivePolicy(t *testing.T) {
	s, pool := newStore(t)
	ctx := context.Background()
	cm := costmodel.Default()
	cfg := Config{StaleAfter: time.Minute}
	pol, err := policy.ByName("reactive")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Heartbeat(ctx, "w1", 8<<30, "node-a"); err != nil {
		t.Fatal(err)
	}

	t.Log("--- preempt: a 500k-vector local build at 10% with 20 queries queued behind it")
	r, _ := s.CreateRound(ctx, "test", 1, 10)
	hurting := submit(t, s, r, 1, "local", 500_000, cm.GPUMemBytes(500_000, 128))
	for seq := int32(0); seq < 2; seq++ {
		if _, err := s.RecordArrival(ctx, r, 1, seq, 1000, true); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.ReportStatus(ctx, store.ShardStatus{RoundID: r, ShardID: 1, QueueDepth: 20, WaitingNeedsIndex: 10, BuildProgress: 0.1}); err != nil {
		t.Fatal(err)
	}
	t.Logf("stay local ~ %.0fs remaining plus a 20-deep backlog at ~1s each; migrate ~ %.0fs gpu total on the idle worker",
		cm.CPUBuild(500_000, 128).Seconds()*0.9, cm.GPUTotal(500_000, 128).Seconds())

	t.Log("--- sunk-cost control: the same build shape at 95% with no backlog stays local")
	kept := submit(t, s, r, 2, "local", 500_000, cm.GPUMemBytes(500_000, 128))
	if err := s.ReportStatus(ctx, store.ShardStatus{RoundID: r, ShardID: 2, QueueDepth: 0, BuildProgress: 0.95}); err != nil {
		t.Fatal(err)
	}

	t.Log("--- renege: a queued 20M-vector build needs ~20 GB; the one live worker has 8 GB")
	stuck := submit(t, s, r, 3, "gpu", 20_000_000, cm.GPUMemBytes(20_000_000, 128))

	Tick(ctx, s, pol, cm, cfg, quiet())

	if b := getBuild(t, s, hurting); b.Placement != "gpu" || b.State != "queued" || !b.Moved || b.Priority <= 0 {
		t.Fatalf("hurting build after tick: %s/%s moved=%v priority=%v", b.Placement, b.State, b.Moved, b.Priority)
	} else {
		t.Logf("preempted %s at priority %.1f (the blocked work behind it: 20 queries x ~1s)", hurting, b.Priority)
	}
	if b := getBuild(t, s, kept); b.Placement != "local" || b.State != "running" {
		t.Fatalf("the 95%% build was moved: %s/%s — sunk cost ignored", b.Placement, b.State)
	}
	t.Logf("%s stayed local at 95%%: not worth killing", kept)
	if b := getBuild(t, s, stuck); b.Placement != "local" || b.State != "running" || !b.Moved {
		t.Fatalf("the fits-nowhere build after tick: %s/%s moved=%v", b.Placement, b.State, b.Moved)
	}
	t.Logf("%s reneged to local: its +Inf queue estimate told the truth no claim ever would", stuck)

	t.Log("--- speculation: reactive_spec copies the worst local build onto the idle pool")
	if _, err := pool.Exec(ctx, `TRUNCATE shard_status, shard_jobs, builds, rounds`); err != nil {
		t.Fatal(err)
	}
	spec, err := policy.ByName("reactive_spec")
	if err != nil {
		t.Fatal(err)
	}
	r2, _ := s.CreateRound(ctx, "test", 2, 10)
	raced := submit(t, s, r2, 1, "gpu", 500_000, cm.GPUMemBytes(500_000, 128))
	if ok, _ := s.RenegeToLocal(ctx, raced); !ok {
		t.Fatal("renege setup failed")
	}
	// moved=true now: reconsider cannot preempt it (decision 8), so the only
	// way the idle GPU can help this shard is a speculative copy.
	if _, err := s.RecordArrival(ctx, r2, 1, 0, 1000, true); err != nil {
		t.Fatal(err)
	}
	if err := s.ReportStatus(ctx, store.ShardStatus{RoundID: r2, ShardID: 1, QueueDepth: 5, WaitingNeedsIndex: 5, BuildProgress: 0.2}); err != nil {
		t.Fatal(err)
	}
	Tick(ctx, s, spec, cm, cfg, quiet())
	copyID := raced + ":spec"
	if b := getBuild(t, s, copyID); !b.Speculative || b.State != "queued" {
		t.Fatalf("after tick: copy %s speculative=%v state=%s", copyID, b.Speculative, b.State)
	}
	t.Logf("tick started copy %s; the local build keeps running (its progress is protected)", copyID)
	if b := getBuild(t, s, raced); b.Placement != "local" || b.State != "running" {
		t.Fatalf("the primary moved: %s/%s", b.Placement, b.State)
	}

	t.Log("--- the gpu wins the race: a worker completes the copy, the next tick settles it")
	c, ok, _ := s.Claim(ctx, "w1", 8<<30, lease)
	if !ok || c.BuildID != copyID {
		t.Fatalf("claim: ok=%v build=%s", ok, c.BuildID)
	}
	if ok, _ := s.Complete(ctx, copyID, c.Attempt); !ok {
		t.Fatal("complete copy failed")
	}
	Tick(ctx, s, spec, cm, cfg, quiet())
	if b := getBuild(t, s, raced); b.State != "done" || b.Placement != "gpu" {
		t.Fatalf("primary after resolve: %s/%s", b.Placement, b.State)
	}
	t.Logf("primary %s is done (gpu): the shard's next report reply says so and it aborts its local build", raced)
}

// buildInput is the host's mapping from rows to the policy's pure view; the
// time-scale conversions are the part worth checking by hand. Scale 10 means
// 1 wall second is 10 modeled seconds.
func TestBuildInputTimeScale(t *testing.T) {
	cm := costmodel.Default()
	gpuTotal := cm.GPUTotal(500_000, 128).Seconds()
	snap := store.ReconsiderSnapshot{
		Workers: []store.SnapshotWorker{{WorkerID: "w1", Node: "n", MemBytes: 1 << 40}, {WorkerID: "w2", Node: "n", MemBytes: 1 << 40}},
		Builds: []store.ReconsiderBuild{
			{BuildID: "1:1", RoundID: 1, ShardID: 1, NVectors: 500_000, Dim: 128, MemBytes: 1 << 20,
				Placement: "gpu", State: "leased", LeaseOwner: "w1", ElapsedS: 1}, // 1 wall s = 10 modeled s into the build
			{BuildID: "2:1", RoundID: 1, ShardID: 2, NVectors: 500_000, Dim: 128, MemBytes: 1 << 20,
				Placement: "gpu", State: "queued"},
			{BuildID: "3:1", RoundID: 1, ShardID: 3, NVectors: 500_000, Dim: 128, MemBytes: 1 << 20,
				Placement: "local", State: "running", ElapsedS: 2},
		},
		Status: []store.ReconsiderStatus{
			{RoundID: 1, ShardID: 3, QueueDepth: 4, Arrivals: 6, MeanQueryMs: 1500, OldestWaitMs: 500, RoundElapsedS: 3},
		},
	}
	in := buildInput(snap, cm, 10)

	var w1 policy.WorkerView
	for _, w := range in.Workers {
		if w.ID == "w1" {
			w1 = w
		}
	}
	want := gpuTotal - 10 // modeled total minus 1 wall s x scale 10
	if math.Abs(w1.BusyRemainingS-want) > 0.01 {
		t.Fatalf("w1 busy remaining %.2f, want %.2f (modeled total %.2f minus 10 modeled s elapsed)", w1.BusyRemainingS, want, gpuTotal)
	}
	t.Logf("leased build: 1 wall s at scale 10 = 10 modeled s done, %.2f of %.2f remaining", w1.BusyRemainingS, gpuTotal)

	views := map[string]policy.BuildView{}
	for _, v := range in.Builds {
		views[v.BuildID] = v
	}
	if v := views["2:1"]; v.Phase != "gpu_queued" || math.Abs(v.QueueWaitS) > 0.01 {
		// w2 is idle, so the replay starts the queued build immediately.
		t.Fatalf("queued view: phase=%s wait=%.2f, want gpu_queued at 0 (w2 is idle)", v.Phase, v.QueueWaitS)
	}
	v3 := views["3:1"]
	if v3.Phase != "local_running" || !v3.Status.Valid {
		t.Fatalf("local view: %+v", v3)
	}
	// 6 arrivals over 3 wall s = 30 modeled s -> 0.2/s; mean 1500 ms -> 1.5 s;
	// oldest wait 500 wall ms -> 5 modeled s.
	if math.Abs(v3.Status.ArrivalRateQPS-0.2) > 1e-9 || v3.Status.MeanQueryS != 1.5 || v3.Status.OldestWaitS != 5 {
		t.Fatalf("derived status: rate=%v mean=%v oldest=%v, want 0.2 / 1.5 / 5", v3.Status.ArrivalRateQPS, v3.Status.MeanQueryS, v3.Status.OldestWaitS)
	}
	t.Logf("derived: rate %.1f modeled qps, mean %.1fs, oldest wait %.1fs — wall facts x scale, modeled facts untouched", v3.Status.ArrivalRateQPS, v3.Status.MeanQueryS, v3.Status.OldestWaitS)
	if v3.QueueWaitS < 0 || math.IsInf(v3.QueueWaitS, 1) {
		t.Fatalf("local build's would-join wait: %v", v3.QueueWaitS)
	}

	t.Log("--- empty pool: the cold-start rule holds on every tick, nothing reads +Inf")
	snap.Workers = nil
	in = buildInput(snap, cm, 10)
	for _, v := range in.Builds {
		if v.QueueWaitS != 0 {
			t.Fatalf("empty pool: %s wait %v, want 0 (G1: an empty pool may be about to fill)", v.BuildID, v.QueueWaitS)
		}
	}
	t.Log("all waits read 0 with no workers: queued builds hold their place instead of stampeding local")
}
