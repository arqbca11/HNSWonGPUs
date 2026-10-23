package sim

import (
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

// Pure tests, no database: every number is computed by hand from the default
// cost model (A=1.4e-7, Speedup=10, GPUOverhead=2, Bandwidth=200e6, M=16):
//
//	100k x 128: local = 1.4e-7*1e5*ln(1e5)*128 = 20.6313 s
//	            gpu   = 2.0631 + 2 = 4.0631 s; transfer = (51.2+64)MB/200MBps = 0.576 s
//	            gpu total = 4.6391 s
//	150k x 128: local = 1.4e-7*1.5e5*ln(1.5e5)*128 = 32.0366 s
//	            gpu total = 3.2037 + 2 + 0.864 = 6.0677 s

const (
	local100k = 20.631288
	gpu100k   = local100k/10 + 2
	xfer100k  = (100_000*128*4 + 100_000*(128*4+128)) / 200e6
	total100k = gpu100k + xfer100k
	local150k = 32.036634
	total150k = local150k/10 + 2 + (150_000*128*4+150_000*(128*4+128))/200e6
)

func near(t *testing.T, what string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s: got %.6f, want %.6f (±%g)", what, got, want, tol)
	}
	t.Logf("    %s = %.4f (hand: %.4f)", what, got, want)
}

// oneShard builds a minimal workload by hand: one shard, n vectors, the given
// queries. The scenario carries a one-worker 16 GB pool unless the test
// overrides it in Config.
func oneShard(n int64, queries []workload.Query) workload.Workload {
	return workload.Workload{
		Scenario: workload.Scenario{Name: "hand", NShards: 1, Dim: 128,
			Stream: workload.StreamProfile{Horizon: time.Minute},
			Pool:   workload.PoolConfig{Workers: 1, WorkerMemBytes: 16 << 30}},
		Seed:   1,
		Shards: []workload.ShardWorkload{{ShardID: 0, NVectors: n, Queries: queries}},
	}
}

func q(at, dur time.Duration, idx bool) workload.Query {
	return workload.Query{Arrival: at, Duration: dur, NeedsIndex: idx, Kind: workload.KindSingleShard, ClusterQueryID: int64(at)}
}

// A local build occupies the CPU: both queries wait behind it, then run in
// arrival order. Round = build + q0 + q1, computed by hand.
func TestOneShardLocalByHand(t *testing.T) {
	w := oneShard(100_000, []workload.Query{
		q(1*time.Second, 2*time.Second, false),
		q(2*time.Second, 3*time.Second, true),
	})
	pol, _ := policy.ByName("static_threshold:200000") // 100k < 200k: local
	r := Run(w, pol, Config{})
	t.Log("one shard builds locally for 20.63 s; a 2 s and a 3 s query queue behind it")
	if !r.Completed || r.BuildsLocal != 1 || r.BuildsGPU != 0 {
		t.Fatalf("completed=%v local=%d gpu=%d", r.Completed, r.BuildsLocal, r.BuildsGPU)
	}
	near(t, "round", r.RoundS, local100k+2+3, 1e-3)
	if len(r.QueueWaits) != 0 {
		t.Fatalf("a local build was claimed? waits=%v", r.QueueWaits)
	}
	t.Log("the CPU ran nothing until its own build finished: the modeling assumption, by hand")
}

// A GPU build frees the CPU: the independent query runs on arrival; the
// index query waits for the build AND for the shard to learn of it at its
// next 0.5 s report — the report latency is part of the model.
func TestOneShardGPUByHand(t *testing.T) {
	w := oneShard(100_000, []workload.Query{
		q(1*time.Second, 1*time.Second, false),
		q(1*time.Second, 1*time.Second, true),
	})
	r := Run(w, policy.AlwaysGPU{}, Config{})
	t.Logf("gpu total by hand: %.4f s; the shard's reports run every 0.5 s", total100k)
	if !r.Completed || r.BuildsGPU != 1 {
		t.Fatalf("completed=%v gpu=%d", r.Completed, r.BuildsGPU)
	}
	// Build done at 4.639; the shard hears at the 5.0 s report; the index
	// query runs 5.0 -> 6.0. The free query ran at 1 -> 2.
	near(t, "round", r.RoundS, 6.0, 1e-3)
	near(t, "queue wait", r.QueueWaits[0], 0, 1e-9)
	t.Log("the index query started only after the report that said done: report latency is modeled")
}

// One worker, two equal builds: FCFS. The second build's queue wait is
// exactly the first build's GPU total.
func TestFCFSQueueWaitByHand(t *testing.T) {
	w := workload.Workload{
		Scenario: workload.Scenario{Name: "hand", NShards: 2, Dim: 128,
			Stream: workload.StreamProfile{Horizon: time.Minute},
			Pool:   workload.PoolConfig{Workers: 1, WorkerMemBytes: 16 << 30}},
		Seed: 1,
		Shards: []workload.ShardWorkload{
			{ShardID: 0, NVectors: 100_000}, {ShardID: 1, NVectors: 100_000},
		},
	}
	r := Run(w, policy.AlwaysGPU{}, Config{})
	if !r.Completed || r.BuildsGPU != 2 || len(r.QueueWaits) != 2 {
		t.Fatalf("completed=%v gpu=%d waits=%v", r.Completed, r.BuildsGPU, r.QueueWaits)
	}
	near(t, "first wait", r.QueueWaits[0], 0, 1e-9)
	near(t, "second wait", r.QueueWaits[1], total100k, 1e-3)
	near(t, "round (both done at build finish; no queries)", r.RoundS, 2*total100k, 1e-3)
	t.Log("the one worker served the queue FCFS; the straggler lag is half the round:")
	near(t, "straggler lag", r.StragglerLagS, total100k/2, 1e-3)
}

// Fluid link sharing (decision 22): two workers on ONE node claim equal
// builds together; their transfers each run at half rate and finish together,
// so both builds finish at 2*transfer + build. On separate nodes they finish
// at transfer + build.
func TestNodeLinkSharingByHand(t *testing.T) {
	w := workload.Workload{
		Scenario: workload.Scenario{Name: "hand", NShards: 2, Dim: 128,
			Stream: workload.StreamProfile{Horizon: time.Minute},
			Pool:   workload.PoolConfig{Workers: 2, WorkerMemBytes: 16 << 30}},
		Seed: 1,
		Shards: []workload.ShardWorkload{
			{ShardID: 0, NVectors: 100_000}, {ShardID: 1, NVectors: 100_000},
		},
	}
	shared := Run(w, policy.AlwaysGPU{}, Config{ShareNodeLinks: true, WorkersPerNode: 2})
	near(t, "same node, shared link: round", shared.RoundS, 2*xfer100k+gpu100k, 1e-3)
	// Bug 14: asserting only RoundS hid a build that "finished" at 2x transfer
	// without ever running its GPU phase — the surviving build set the round.
	// EVERY shard must finish at 2X+G, and no earlier.
	for i, fin := range shared.ShardFinishS {
		near(t, "  shared: shard finish", fin, 2*xfer100k+gpu100k, 1e-3)
		if fin < xfer100k+gpu100k {
			t.Fatalf("shard %d finished at %.3f, faster than an unshared build: sharing sped something up", i, fin)
		}
	}
	separate := Run(w, policy.AlwaysGPU{}, Config{ShareNodeLinks: true, WorkersPerNode: 1})
	near(t, "separate nodes: round", separate.RoundS, xfer100k+gpu100k, 1e-3)
	off := Run(w, policy.AlwaysGPU{}, Config{ShareNodeLinks: false, WorkersPerNode: 2})
	near(t, "sharing off (the fake worker's behaviour): round", off.RoundS, xfer100k+gpu100k, 1e-3)
	t.Log("two transfers on one link each ran at 1/2 and finished together; off mimics the live fake worker")

	t.Log("--- a staggered join: the second shard's DDL lands mid-transfer of the first (j = 0.2 s)")
	w.Shards[1].DDLOffset = 200 * time.Millisecond
	stag := Run(w, policy.AlwaysGPU{}, Config{ShareNodeLinks: true, WorkersPerNode: 2})
	// By hand with X = 0.576, join at j = 0.2: A runs alone to j (remaining
	// X-j), then both drain at 1/2, so A's transfer ends at j + 2(X-j) =
	// 2X - j; B has X - (X-j) = j left, alone, ending at exactly 2X — the
	// link moved 2X of work in 2X seconds, saturated throughout.
	near(t, "  A finish", stag.ShardFinishS[0], 2*xfer100k-0.2+gpu100k, 1e-3)
	near(t, "  B finish", stag.ShardFinishS[1], 2*xfer100k+gpu100k, 1e-3)
	t.Log("the fluid drains at the shared rate for exactly the shared interval: no stolen progress at the rate change (bug 14)")
}

// The reactive policy preempts a hurting local build mid-flight: the round
// recovers on the GPU, and the preempted build's queue wait measures from its
// SUBMIT (enqueued_at keeps the submit time).
func TestPreemptionInSim(t *testing.T) {
	// 150k vectors (< the 200k threshold: placed local), with a steady stream
	// of index-needing queries behind it — stay-local loses badly.
	var qs []workload.Query
	for i := 0; i < 20; i++ {
		qs = append(qs, q(time.Duration(i+1)*250*time.Millisecond, time.Second, true))
	}
	w := oneShard(150_000, qs)
	pol, _ := policy.ByName("reactive")
	r := Run(w, pol, Config{})
	if !r.Completed || r.Preempts != 1 || r.BuildsGPU != 1 {
		t.Fatalf("completed=%v preempts=%d gpu=%d", r.Completed, r.Preempts, r.BuildsGPU)
	}
	t.Logf("preempted once; round %.2f s", r.RoundS)
	// Stay local would be: 32.04 build + 20 s backlog = 52+ s. The preempt
	// (first possible tick is t=2, acted on by the shard at its next report)
	// puts the GPU total (6.07 s) plus drain (20 s) well under it.
	if r.RoundS >= local150k+20 {
		t.Fatalf("round %.2f s did not beat staying local (%.2f s): the preemption bought nothing", r.RoundS, local150k+20)
	}
	if r.QueueWaits[0] < 2 {
		t.Fatalf("the preempted build's wait %.2f s should measure from submit (it sat local past the first tick)", r.QueueWaits[0])
	}
	if r.WastedCPUS < 2 || r.WastedCPUS > 4 {
		t.Fatalf("wasted CPU %.2f s: the abort came at the shard's first report after the t=2 tick, so 2..2.5 s of build was thrown away", r.WastedCPUS)
	}
	t.Logf("the round beat the stay-local projection; the wait measures from submit; %.2f s of CPU recorded wasted", r.WastedCPUS)
}

// Speculation, GPU wins: with a huge margin the reactive_spec policy never
// preempts or reneges, so the idle worker's only move is a copy. The round
// ends at the copy's finish — by hand — and the primary never counts a wait.
func TestSpeculationGPUWinsByHand(t *testing.T) {
	w := oneShard(150_000, nil)
	pol := policy.NewReactiveSpec(policy.ReactiveConfig{MarginS: 1e9})
	r := Run(w, pol, Config{})
	if !r.Completed || r.Speculations != 1 || r.SpecGPUWins != 1 || r.BuildsGPU != 1 {
		t.Fatalf("completed=%v spec=%d gpuwins=%d gpu=%d", r.Completed, r.Speculations, r.SpecGPUWins, r.BuildsGPU)
	}
	// First tick t=2 starts the copy; it finishes at 2 + 6.0681 = 8.0681. The
	// resolve backdates the primary's finish to that moment (decision 16).
	near(t, "round = copy finish", r.RoundS, 2+total150k, 1e-3)
	near(t, "wasted gpu", r.WastedGPUS, 0, 1e-9)
	// The copy finished at 8.07, the tick at 10 settled it, the shard's 10.5
	// report aborted the local build: 10.5 s of CPU burned for nothing — the
	// cost speculation pays on the CPU side (design doc: report wasted
	// CPU-hours, not just wasted GPU time).
	near(t, "wasted cpu", r.WastedCPUS, 10.5, 1e-3)
	if len(r.QueueWaits) != 0 {
		t.Fatalf("the primary was never claimed; only the copy was: waits=%v", r.QueueWaits)
	}
	t.Log("the copy won at 8.07 s; the local build (32 s) was aborted; no GPU work wasted")
}

// Speculation, local wins: with Speedup 1 the GPU is slower than the CPU, the
// local build finishes first, and the copy is cancelled at the next tick —
// the wasted worker-seconds are the recorded cost of speculating.
func TestSpeculationLocalWinsByHand(t *testing.T) {
	w := oneShard(150_000, nil)
	pol := policy.NewReactiveSpec(policy.ReactiveConfig{MarginS: 1e9})
	cm := costmodel.Default()
	cm.Speedup = 1 // gpu build = local + 2 s overhead: the copy cannot win
	r := Run(w, pol, Config{CostModel: &cm})
	if !r.Completed || r.Speculations != 1 || r.SpecLocalWins != 1 || r.BuildsLocal != 1 {
		t.Fatalf("completed=%v spec=%d localwins=%d local=%d", r.Completed, r.Speculations, r.SpecLocalWins, r.BuildsLocal)
	}
	near(t, "round = local build", r.RoundS, local150k, 1e-3)
	// The copy was claimed at t=2 and cancelled at the first tick after the
	// local finish (32.04 -> tick 34): 32 worker-seconds burned.
	near(t, "wasted gpu seconds", r.WastedGPUS, 32, 1e-3)
	t.Log("the local build won; the copy was cancelled at the next tick and its worker-seconds recorded")
}

// The shrinking pool, both halves of decision 19. Every worker in a sim run
// shares one memory size, so the fits-nowhere renege is driven by the kill
// leaving the pool EMPTY (the mixed-memory renege is covered against real
// Postgres in the reconsider package): an empty pool HOLDS the queued build —
// the hang is visible in Completed=false, never hidden — while a surviving
// worker finishes the round normally after the lease requeues.
func TestKilledWorkerAndEmptyPoolHold(t *testing.T) {
	w := oneShard(100_000, nil)
	pol := policy.NewReactive(policy.ReactiveConfig{ThresholdN: 1}) // everything to the GPU at submit

	t.Log("--- the only worker dies at t=1; the lease requeues at t=6 into an empty pool")
	r := Run(w, pol, Config{Workers: 1, LeaseS: 5, KillWorkerAtS: 1, MaxModeledS: 120})
	if r.Completed {
		t.Fatalf("with the only worker dead, the round cannot complete (decision 19 holds the build); got round %.2f", r.RoundS)
	}
	t.Log("the requeued build held its place on the empty pool (decision 19); the safety cap ended the run with Completed=false")

	t.Log("--- the same kill with a second worker alive: the requeued build is reclaimed and finishes")
	r = Run(w, pol, Config{Workers: 2, LeaseS: 5, KillWorkerAtS: 1})
	if !r.Completed || r.BuildsGPU != 1 {
		t.Fatalf("completed=%v gpu=%d", r.Completed, r.BuildsGPU)
	}
	// Claimed at 0 by w00, killed at 1, requeued at 6, reclaimed by w01:
	// done at 6 + 4.6391.
	near(t, "round", r.RoundS, 6+total100k, 1e-3)
	t.Log("the lease expiry cost exactly LeaseS: the reaper's timetable, in modeled time")
}

// Determinism: the same workload, policy and config give byte-identical
// results, including under estimate noise; a different seed differs.
func TestDeterminism(t *testing.T) {
	sc, err := workload.Preset("skewed")
	if err != nil {
		t.Fatal(err)
	}
	w := workload.Generate(sc, 7)
	pol, _ := policy.ByName("reactive_spec")
	cfg := Config{EstimateSigma: 0.3, NoiseSeed: 42}
	a, b := Run(w, pol, cfg), Run(w, pol, cfg)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two identical runs differ:\n%+v\n%+v", a, b)
	}
	t.Logf("two runs of skewed/seed 7/sigma 0.3 are identical: round %.3f s, %d events", a.RoundS, a.Events)
	w2 := workload.Generate(sc, 8)
	c := Run(w2, pol, cfg)
	if c.RoundS == a.RoundS {
		t.Fatalf("different seeds gave the same round time %.6f: suspicious", c.RoundS)
	}
	t.Logf("seed 8 differs as it should: %.3f s", c.RoundS)
}

// Decision 23: noise perturbs only what policies SEE, never what executes.
// always_gpu ignores every estimate, so its rounds are identical at sigma 0
// and sigma 2 — if noise leaked into execution, they could not be.
func TestNoiseNeverTouchesExecution(t *testing.T) {
	sc, err := workload.Preset("uniform")
	if err != nil {
		t.Fatal(err)
	}
	w := workload.Generate(sc, 3)
	clean := Run(w, policy.AlwaysGPU{}, Config{})
	noisy := Run(w, policy.AlwaysGPU{}, Config{EstimateSigma: 2, NoiseSeed: 99})
	if clean.RoundS != noisy.RoundS || !reflect.DeepEqual(clean.QueueWaits, noisy.QueueWaits) {
		t.Fatalf("sigma=2 changed an always_gpu round (%.6f vs %.6f): noise leaked into execution",
			clean.RoundS, noisy.RoundS)
	}
	t.Logf("sigma 0 and sigma 2 identical under always_gpu (round %.3f s): the truth is never noised", clean.RoundS)
}

// All five policies play the default scenario; the oracle is constructed with
// the futures only the simulator may hand out (decision 1a). This is the
// pipeline's first policy-comparison table, as narration.
func TestAllFivePoliciesComplete(t *testing.T) {
	sc, err := workload.Preset("skewed")
	if err != nil {
		t.Fatal(err)
	}
	w := workload.Generate(sc, 1)
	pols := []policy.Policy{}
	for _, name := range []string{"always_gpu", "static_threshold", "reactive", "reactive_spec"} {
		p, err := policy.ByName(name)
		if err != nil {
			t.Fatal(err)
		}
		pols = append(pols, p)
	}
	pols = append(pols, policy.NewClairvoyant(Futures(w)))
	t.Log("skewed, seed 1, default pool (2 workers):")
	for _, p := range pols {
		r := Run(w, p, Config{})
		if !r.Completed {
			t.Fatalf("%s did not complete the round", p.Name())
		}
		t.Logf("  %-22s round %7.2f s  straggler %6.2f s  gpu %d local %d  preempts %d reneges %d spec %d",
			p.Name(), r.RoundS, r.StragglerLagS, r.BuildsGPU, r.BuildsLocal, r.Preempts, r.Reneges, r.Speculations)
	}
}
