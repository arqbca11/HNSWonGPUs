package crosscheck

// Phase 3 step 1 cross-check: the placement-policy interface and the five
// policies, written from docs/design/phase3-policies.md, the roadmap's Phase 3
// section ("Policy interface", "The policies"), CLAUDE.md invariant 8 and the
// modeling assumptions, and `go doc ./scheduler/policy`. No policy source was
// read. These tests are pure: no database, no cluster, no env gate.
//
// Hand arithmetic conventions used below (from the reactive flow diagram):
//   stay    = LocalBuildS*(1-progress) + QueueDepth*MeanQueryS
//   migrate = max((QueueDepth-WaitingNeedsIndex)*MeanQueryS,
//                 QueueWaitS + GPUBuildS + TransferS + WaitingNeedsIndex*MeanQueryS)
// ArrivalRateQPS is kept at 0 wherever exact arithmetic is asserted, because
// the docs do not say how the rate enters either side.

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

const (
	gib     int64 = 1 << 30
	wMem          = 16 * gib // every worker in these tests, unless stated
	smallMB       = 1 * gib  // a build that fits every worker
)

// ---- helpers ---------------------------------------------------------------

func pool(n int) policy.PoolState {
	if n == 0 {
		return policy.PoolState{}
	}
	return policy.PoolState{LiveWorkers: n, LargestMem: wMem}
}

func spec(shard int32, n int64, mem int64) policy.BuildSpec {
	return policy.BuildSpec{ShardID: shard, RoundID: 1, NVectors: n, Dim: 128, MemBytes: mem}
}

func status(depth, needsIdx int32, progress, mean float64) policy.ShardStatus {
	return policy.ShardStatus{QueueDepth: depth, WaitingNeedsIndex: needsIdx, BuildProgress: progress, MeanQueryS: mean, Valid: true}
}

func idleWorkers(n int) []policy.WorkerView {
	ws := make([]policy.WorkerView, n)
	for i := range ws {
		ws[i] = policy.WorkerView{ID: fmt.Sprintf("w%d", i+1), Node: "n1", MemBytes: wMem}
	}
	return ws
}

func busyWorkers(n int, rem float64) []policy.WorkerView {
	ws := idleWorkers(n)
	for i := range ws {
		ws[i].BusyRemainingS = rem
	}
	return ws
}

func bview(id, phase string, shard int32, est policy.Estimates, st policy.ShardStatus) policy.BuildView {
	return policy.BuildView{BuildID: id, Spec: spec(shard, 100_000, smallMB), Est: est, Phase: phase, Status: st}
}

func decide(t *testing.T, p policy.Policy, in policy.Input) (d policy.Decision) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s.Decide panicked on %+v: %v", p.Name(), in, r)
		}
	}()
	return p.Decide(in)
}

func reconsider(t *testing.T, p policy.Policy, in policy.ReconsiderInput) (acts []policy.Action) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s.Reconsider panicked: %v", p.Name(), r)
		}
	}()
	return p.Reconsider(in)
}

func actsOf(acts []policy.Action, kind policy.ActionKind, id string) []policy.Action {
	var out []policy.Action
	for _, a := range acts {
		if a.Kind == kind && (id == "" || a.BuildID == id) {
			out = append(out, a)
		}
	}
	return out
}

func fmtActs(acts []policy.Action) string {
	if len(acts) == 0 {
		return "[] (no actions)"
	}
	var b strings.Builder
	for _, a := range acts {
		fmt.Fprintf(&b, "{%s %s prio=%.3f %q} ", a.Kind, a.BuildID, a.Priority, a.Reason)
	}
	return b.String()
}

// canon sorts actions so two runs with permuted inputs can be compared as
// sets (the docs promise nothing about action order).
func canon(acts []policy.Action) []string {
	out := make([]string, 0, len(acts))
	for _, a := range acts {
		out = append(out, fmt.Sprintf("%s|%s|%.9f", a.Kind, a.BuildID, a.Priority))
	}
	sort.Strings(out)
	return out
}

func cloneBuilds(bs []policy.BuildView) []policy.BuildView {
	return append([]policy.BuildView(nil), bs...)
}

func approx(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func allPolicies(t *testing.T) []policy.Policy {
	t.Helper()
	futures := map[int32][]policy.QueryEvent{
		1: {{ArriveS: 1, DurS: 2}, {ArriveS: 2, DurS: 3, NeedsIndex: true}},
	}
	return []policy.Policy{
		policy.AlwaysGPU{},
		policy.StaticThreshold{N: 50_000},
		policy.NewReactive(policy.ReactiveConfig{}),
		policy.NewReactiveSpec(policy.ReactiveConfig{}),
		policy.NewClairvoyant(futures),
	}
}

// A world in which a reactive policy has something to do: a hurting local
// build (preempt), a stuck queued build (renege), a preempted queued build with
// a stale priority (reprioritize), a mild straggler and an idle worker.
func busyWorld() policy.ReconsiderInput {
	hurt := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
	stuck := bview("2:1", "gpu_queued", 2, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0, 1))
	stuck.QueueWaitS = 500
	stale := bview("3:1", "gpu_queued", 3, policy.Estimates{LocalBuildS: 30, GPUBuildS: 3}, status(40, 10, 0, 1))
	stale.Moved = true
	mild := bview("4:1", "local_running", 4, policy.Estimates{LocalBuildS: 20, GPUBuildS: 9}, status(0, 0, 0.5, 1))
	leased := bview("5:1", "gpu_leased", 5, policy.Estimates{LocalBuildS: 50, GPUBuildS: 5}, status(3, 3, 0, 1))
	return policy.ReconsiderInput{
		Workers: append(idleWorkers(1), policy.WorkerView{ID: "w2", Node: "n1", MemBytes: wMem, BusyRemainingS: 4}),
		Builds:  []policy.BuildView{hurt, stuck, stale, mild, leased},
	}
}

// ---- the test --------------------------------------------------------------

func TestPolicySpec(t *testing.T) {
	// ---- invariant 8: purity and determinism --------------------------------

	t.Run("Purity_RepeatedDecideIdentical", func(t *testing.T) {
		in := policy.Input{Build: spec(1, 120_000, smallMB), Est: policy.Estimates{LocalBuildS: 10, GPUBuildS: 3, TransferS: 1}, Pool: pool(2), Status: status(3, 1, 0, 1)}
		for _, p := range allPolicies(t) {
			first := decide(t, p, in)
			for i := 0; i < 50; i++ {
				if got := decide(t, p, in); got != first {
					t.Errorf("%s: call %d returned %+v, first call %+v; invariant 8 promises a pure function", p.Name(), i, got, first)
					break
				}
			}
			t.Logf("%s.Decide x51 on the same input -> %+v every time (invariant 8: pure, no clock, no randomness)", p.Name(), first)
		}
	})

	t.Run("Purity_RepeatedReconsiderIdentical", func(t *testing.T) {
		for _, p := range allPolicies(t) {
			first := reconsider(t, p, busyWorld())
			for i := 0; i < 50; i++ {
				got := reconsider(t, p, busyWorld())
				if !reflect.DeepEqual(got, first) {
					t.Errorf("%s: call %d returned %s, first call %s; invariant 8", p.Name(), i, fmtActs(got), fmtActs(first))
					break
				}
			}
			t.Logf("%s.Reconsider x51 on busyWorld -> %s", p.Name(), fmtActs(first))
		}
	})

	t.Run("Purity_ReconsiderDoesNotMutateInput", func(t *testing.T) {
		for _, p := range allPolicies(t) {
			w := busyWorld()
			wantB := cloneBuilds(w.Builds)
			wantW := append([]policy.WorkerView(nil), w.Workers...)
			reconsider(t, p, w)
			if !reflect.DeepEqual(w.Builds, wantB) || !reflect.DeepEqual(w.Workers, wantW) {
				t.Errorf("%s.Reconsider mutated its input slices (order or contents); a pure function must not", p.Name())
			} else {
				t.Logf("%s.Reconsider left the caller's Builds and Workers slices byte-identical", p.Name())
			}
		}
	})

	t.Run("Purity_ReconsiderPermutationInvariant", func(t *testing.T) {
		for _, p := range allPolicies(t) {
			base := busyWorld()
			want := canon(reconsider(t, p, base))
			n := len(base.Builds)
			// every rotation and the reversal of builds and workers
			for r := 0; r < n; r++ {
				for _, rev := range []bool{false, true} {
					w := busyWorld()
					bs := make([]policy.BuildView, n)
					for i := range bs {
						bs[i] = w.Builds[(i+r)%n]
					}
					if rev {
						for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
							bs[i], bs[j] = bs[j], bs[i]
						}
						w.Workers[0], w.Workers[1] = w.Workers[1], w.Workers[0]
					}
					w.Builds = bs
					got := canon(reconsider(t, p, w))
					if !reflect.DeepEqual(got, want) {
						t.Errorf("%s: rotation %d reversed=%v gives %v, canonical order gives %v; the decision depends on slice order", p.Name(), r, rev, got, want)
					}
				}
			}
			t.Logf("%s: %d permutations of builds/workers all give the same action set %v (docs promise no order; set equality is what matters)", p.Name(), 2*n, want)
		}
	})

	t.Run("EmptyWorld_NoActionsNoPanic", func(t *testing.T) {
		for _, p := range allPolicies(t) {
			a := reconsider(t, p, policy.ReconsiderInput{})
			b := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(3)})
			if len(a)+len(b) != 0 {
				t.Errorf("%s: no builds yet returned actions %s / %s", p.Name(), fmtActs(a), fmtActs(b))
			}
			t.Logf("%s.Reconsider on an empty world and on 3 idle workers with no builds -> no actions", p.Name())
		}
	})

	// ---- the memory filter (CLAUDE.md modeling assumptions; phase1 decision 19)

	t.Run("MemoryFilter_AllPolicies", func(t *testing.T) {
		est := policy.Estimates{LocalBuildS: 100, GPUBuildS: 1, TransferS: 0}
		big := spec(9, 10_000_000, wMem+1)
		exact := spec(9, 10_000_000, wMem)
		for _, p := range allPolicies(t) {
			d := decide(t, p, policy.Input{Build: big, Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
			t.Logf("%s: build needs %d B, largest live worker %d B, local 100 s vs GPU 1 s -> %s (%q)", p.Name(), big.MemBytes, wMem, d.Placement, d.Reason)
			if d.Placement != "local" {
				t.Errorf("%s placed a build that fits no live worker on %q; CLAUDE.md: 'must build locally'", p.Name(), d.Placement)
			}
			d = decide(t, p, policy.Input{Build: exact, Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
			t.Logf("%s: build needs exactly the largest worker's %d B -> %s", p.Name(), wMem, d.Placement)
			if d.Placement != "gpu" {
				t.Errorf("%s: a build needing exactly LargestMem should fit (it fits a worker of that size); got %q", p.Name(), d.Placement)
			}
		}
	})

	t.Run("MemoryFilter_EmptyPoolStillQueues", func(t *testing.T) {
		est := policy.Estimates{LocalBuildS: 100, GPUBuildS: 1}
		for _, p := range []policy.Policy{policy.AlwaysGPU{}, policy.StaticThreshold{N: 50_000}, policy.NewReactive(policy.ReactiveConfig{ThresholdN: 50_000}), policy.NewReactiveSpec(policy.ReactiveConfig{ThresholdN: 50_000})} {
			d := decide(t, p, policy.Input{Build: spec(9, 10_000_000, 64*gib), Est: est, Pool: pool(0), Status: status(0, 0, 0, 1)})
			t.Logf("%s: zero live workers, 64 GiB, 10M-vector build -> %s (phase 1 decision 19: an empty pool may be cold-starting, the memory rule applies only when a worker is live)", p.Name(), d.Placement)
			if d.Placement != "gpu" {
				t.Errorf("%s: empty pool turned a GPU-eligible build local; decision 19 says it still queues", p.Name())
			}
		}
		// The oracle with an empty pool: the docs say nothing. Record it.
		c := policy.NewClairvoyant(nil)
		d := decide(t, c, policy.Input{Build: spec(9, 10_000_000, 64*gib), Est: est, Pool: pool(0)})
		t.Logf("SPEC GAP probe: clairvoyant with zero live workers (local 100 s, GPU 1 s, no queries) -> %s prio=%.1f (%q). Neither decision 19 nor 1a says whether the oracle, which runs only in the simulator, honours the cold-start rule", d.Placement, d.Priority, d.Reason)
	})

	// ---- AlwaysGPU -------------------------------------------------------------

	t.Run("AlwaysGPU_QueuesEverythingFCFS", func(t *testing.T) {
		p := policy.AlwaysGPU{}
		for _, n := range []int64{1, 1000, 10_000_000} {
			d := decide(t, p, policy.Input{Build: spec(1, n, smallMB), Est: policy.Estimates{LocalBuildS: 0.1, GPUBuildS: 1000}, Pool: pool(1)})
			t.Logf("always_gpu: %d vectors, local 0.1 s vs GPU 1000 s -> %s prio=%v", n, d.Placement, d.Priority)
			if d.Placement != "gpu" || d.Priority != 0 {
				t.Errorf("roadmap policy 1: every build to the queue, FCFS (priority 0 per go doc); got %+v", d)
			}
		}
		if a := reconsider(t, p, busyWorld()); len(a) != 0 {
			t.Errorf("always_gpu returned reconsider actions %s; it is the baseline", fmtActs(a))
		}
		t.Logf("always_gpu.Reconsider on busyWorld -> no actions")
	})

	// ---- StaticThreshold -------------------------------------------------------

	t.Run("StaticThreshold_BoundaryPriorityNeverReconsiders", func(t *testing.T) {
		p := policy.StaticThreshold{N: 1000}
		cases := []struct {
			n    int64
			want string
		}{{999, "local"}, {1000, "gpu"}, {1001, "gpu"}, {0, "local"}}
		for _, c := range cases {
			d := decide(t, p, policy.Input{Build: spec(1, c.n, smallMB), Est: policy.Estimates{LocalBuildS: 5, GPUBuildS: 1}, Pool: pool(2)})
			t.Logf("static_threshold N=1000: %d vectors -> %s prio=%v (roadmap policy 2: n_vectors >= T to the GPU)", c.n, d.Placement, d.Priority)
			if d.Placement != c.want {
				t.Errorf("%d vectors: got %s want %s", c.n, d.Placement, c.want)
			}
			if c.want == "gpu" && d.Priority != float64(c.n) {
				t.Errorf("decision 10: static_threshold priority is the vector count; got %v for %d", d.Priority, c.n)
			}
		}
		small := decide(t, p, policy.Input{Build: spec(1, 2000, smallMB), Pool: pool(2)})
		large := decide(t, p, policy.Input{Build: spec(2, 9000, smallMB), Pool: pool(2)})
		if !(large.Priority > small.Priority) {
			t.Errorf("bigger builds claim first: 9000-vector prio %v must exceed 2000-vector prio %v", large.Priority, small.Priority)
		}
		t.Logf("2000 vs 9000 vectors -> prio %v vs %v: the bigger claims first", small.Priority, large.Priority)
		if a := reconsider(t, p, busyWorld()); len(a) != 0 {
			t.Errorf("static_threshold must never reconsider; got %s", fmtActs(a))
		}
		t.Logf("static_threshold.Reconsider on busyWorld (a hurting local build, a stuck queue) -> no actions")
	})

	// ---- ByName ----------------------------------------------------------------

	t.Run("ByName_AcceptsDocumentedNames", func(t *testing.T) {
		for _, name := range []string{"always_gpu", "static_threshold", "static_threshold:500", "reactive", "reactive_spec"} {
			p, err := policy.ByName(name)
			if err != nil || p == nil {
				t.Errorf("ByName(%q) = %v, %v; documented in the step 1 notes", name, p, err)
				continue
			}
			t.Logf("ByName(%q) -> %T, Name()=%q", name, p, p.Name())
		}
		p, _ := policy.ByName("static_threshold")
		if st, ok := p.(policy.StaticThreshold); !ok || st.N != policy.DefaultThresholdN {
			t.Errorf("bare static_threshold should use DefaultThresholdN=%d; got %#v", policy.DefaultThresholdN, p)
		}
		p, _ = policy.ByName("static_threshold:500")
		lo := decide(t, p, policy.Input{Build: spec(1, 499, smallMB), Pool: pool(1)})
		hi := decide(t, p, policy.Input{Build: spec(1, 500, smallMB), Pool: pool(1)})
		t.Logf("static_threshold:500: 499 -> %s, 500 -> %s", lo.Placement, hi.Placement)
		if lo.Placement != "local" || hi.Placement != "gpu" {
			t.Errorf("static_threshold:500 should split at 500")
		}
	})

	t.Run("ByName_RejectsBadNamesAndTheOracle", func(t *testing.T) {
		c := policy.NewClairvoyant(nil)
		bad := []string{"bogus", "clairvoyant", c.Name(), "static_threshold:", "static_threshold:abc", "static_threshold:1.5", "static_threshold:10:20", "always_gpu:3"}
		for _, name := range bad {
			p, err := policy.ByName(name)
			if err == nil {
				t.Errorf("ByName(%q) accepted (-> %T %q); want an error (decision 1a for the oracle; malformed otherwise)", name, p, p.Name())
				continue
			}
			t.Logf("ByName(%q) -> error %q", name, err)
		}
		// Not stated by the docs either way: record what happens.
		for _, name := range []string{"", "static_threshold:-5", "static_threshold:0", "REACTIVE", " reactive", "reactive ", "reactive:5", "reactive_spec:x", "always_gpu:"} {
			p, err := policy.ByName(name)
			if err != nil {
				t.Logf("SPEC GAP probe: ByName(%q) -> error %q", name, err)
			} else {
				t.Logf("SPEC GAP probe: ByName(%q) -> accepted, %T Name()=%q", name, p, p.Name())
			}
		}
	})

	t.Run("ByName_NameRoundTrips", func(t *testing.T) {
		for _, name := range []string{"always_gpu", "static_threshold", "static_threshold:500", "reactive", "reactive_spec"} {
			p, _ := policy.ByName(name)
			q, err := policy.ByName(p.Name())
			if err != nil {
				t.Errorf("ByName(%q).Name()=%q is not itself accepted by ByName: %v", name, p.Name(), err)
				continue
			}
			if !reflect.DeepEqual(p, q) {
				t.Errorf("ByName(%q) and ByName(Name()=%q) differ: %#v vs %#v (a logged policy name would not reproduce the run)", name, p.Name(), p, q)
			}
			t.Logf("ByName(%q).Name()=%q round-trips to an identical policy", name, p.Name())
		}
	})

	// ---- Reactive: Decide ------------------------------------------------------

	t.Run("Reactive_DecideLikeThresholdPriorityIsLocalSeconds", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{ThresholdN: 1000})
		est := policy.Estimates{LocalBuildS: 37.5, GPUBuildS: 2, TransferS: 0.5}
		lo := decide(t, p, policy.Input{Build: spec(1, 999, smallMB), Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
		hi := decide(t, p, policy.Input{Build: spec(1, 1000, smallMB), Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
		t.Logf("reactive ThresholdN=1000: 999 -> %s; 1000 -> %s prio=%v (LocalBuildS=%v)", lo.Placement, hi.Placement, hi.Priority, est.LocalBuildS)
		if lo.Placement != "local" || hi.Placement != "gpu" {
			t.Errorf("ReactiveConfig.ThresholdN: 'places like a static threshold'; want local/gpu, got %s/%s", lo.Placement, hi.Placement)
		}
		if !approx(hi.Priority, est.LocalBuildS) {
			t.Errorf("decision 10: reactive submit priority = the build's local duration (%v s); got %v", est.LocalBuildS, hi.Priority)
		}
		dz := decide(t, policy.NewReactive(policy.ReactiveConfig{}), policy.Input{Build: spec(1, policy.DefaultThresholdN, smallMB), Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
		dz1 := decide(t, policy.NewReactive(policy.ReactiveConfig{}), policy.Input{Build: spec(1, policy.DefaultThresholdN-1, smallMB), Est: est, Pool: pool(2), Status: status(0, 0, 0, 1)})
		t.Logf("SPEC GAP probe: zero ReactiveConfig at DefaultThresholdN (%d) -> %s, at N-1 -> %s. The docs say 'zero value means defaults' but name no ThresholdN default for reactive", policy.DefaultThresholdN, dz.Placement, dz1.Placement)
	})

	// ---- Reactive: stay vs migrate ---------------------------------------------

	t.Run("Reactive_PreemptArithmeticAndPriority", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		// LocalBuildS 100 at 50% -> remaining 50; backlog 10 x 1 s (4 need the index).
		// stay = 50 + 10 = 60; migrate = max(6, 0 + 5 + 1 + 4) = 10; wins by 50 > 2.
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{b}})
		t.Logf("local build 100 s at 50%%, backlog 10 x 1 s (4 need index), GPU 5+1 s, idle pool: stay=60, migrate=max(6,10)=10 -> %s", fmtActs(acts))
		pre := actsOf(acts, policy.ActPreempt, "1:1")
		if len(pre) != 1 {
			t.Fatalf("roadmap policy 3 / flow diagram: migrate recovers 50 s sooner (> 2 s margin), expected one preempt; got %s", fmtActs(acts))
		}
		// decision 10 (as corrected in the step 1 triage): priority after a
		// preemption = the backlog behind the build, 10 x 1 s = 10.
		if !approx(pre[0].Priority, 10) {
			t.Errorf("decision 10: preempted priority = the 10 s backlog behind the build; got %v", pre[0].Priority)
		}
		if len(acts) != 1 {
			t.Errorf("only one action expected for one build; got %s", fmtActs(acts))
		}
	})

	t.Run("Reactive_PreemptPriorityIsBacklogOnly", func(t *testing.T) {
		// Found in this cross-check (F3): the original decision 10 wording said
		// backlog + abandoned time, the code computed backlog + remaining, and
		// neither was stable across ticks. The triage corrected both to the
		// roadmap's literal rule: priority = the blocked work behind the build.
		// At 20%: LocalBuildS 100, backlog 10 x 1 s -> priority 10, whatever
		// the progress (progress only gates WHETHER to preempt, not the rank).
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.2, 1))
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
		pre := actsOf(acts, policy.ActPreempt, "1:1")
		t.Logf("local build 100 s at 20%%, backlog 10 x 1 s -> %s", fmtActs(acts))
		if len(pre) != 1 {
			t.Fatalf("stay 90 vs migrate 10: expected a preempt")
		}
		t.Logf("decision 10 (corrected): 'after a preemption it is the backlog behind the build' = 10; observed %v", pre[0].Priority)
		if !approx(pre[0].Priority, 10) {
			t.Errorf("preempt priority %v is not the backlog behind the build (10)", pre[0].Priority)
		}
	})

	t.Run("Reactive_RenegeComparisonProbe", func(t *testing.T) {
		// A queued build with a backlog of 10 x 1 s that does NOT need the index.
		// LocalBuildS 10, GPU 2, wait 15.
		//   recovery view (mirror of the flow): local = 10 + 10 = 20; gpu = max(10, 15+2) = 17 -> stay queued.
		//   build-only view: local 10 vs gpu 17 -> renege (blocking 10 s of runnable queries).
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "gpu_queued", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(10, 0, 0, 1))
		b.QueueWaitS = 15
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 15), Builds: []policy.BuildView{b}})
		t.Logf("SPEC GAP probe: queued build, 10 non-index queries waiting, local 10 s vs GPU 15+2 s -> %s. Roadmap: renege 'when its GPU estimate has grown past its local estimate'; the go doc says 'local from scratch now beats the queue'. Whether the backlog the local build would block counts is not stated. Recovery view: local 20 vs gpu 17 (stay); build-only view: 10 vs 17 (renege)", fmtActs(acts))
		if len(actsOf(acts, policy.ActRenege, "1:1")) == 1 {
			// Feed the reneged state back: the same shard, now building locally from 0%.
			r := b
			r.Phase, r.QueueWaitS = "local_running", 15
			back := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 15), Builds: []policy.BuildView{r}})
			t.Logf("FINDING: after that renege, the same policy's own stay-vs-migrate on the resulting state (Moved cleared to expose it) -> %s. It reneged into a placement its preempt rule would immediately undo; only Moved (decision 8) stops the ping-pong. Renege and preempt use different cost models", fmtActs(back))
		}
	})

	t.Run("Reactive_MarginBoundaryDefault2s", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		// LocalBuildS 10 at 0%, backlog 2 x 1 s both needing the index.
		// stay = 10 + 2 = 12; migrate = max(0, g + 0 + 2) = g + 2.
		for _, c := range []struct {
			g    float64
			want bool
		}{{8, false}, {7.5, true}, {9, false}, {11, false}} {
			b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: c.g}, status(2, 2, 0, 1))
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
			got := len(actsOf(acts, policy.ActPreempt, "1:1")) == 1
			t.Logf("GPU %v s: stay=12, migrate=%v, wins by %v (default margin 2 s, decision 8 'must win by MarginS') -> preempt=%v %s", c.g, c.g+2, 10-c.g, got, fmtActs(acts))
			if got != c.want {
				t.Errorf("GPU %v s: preempt=%v, want %v", c.g, got, c.want)
			}
		}
		// A custom margin: win by 5 with MarginS 6 -> no; with MarginS 4 -> yes.
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 5}, status(2, 2, 0, 1))
		for _, m := range []float64{6, 4} {
			acts := reconsider(t, policy.NewReactive(policy.ReactiveConfig{MarginS: m}), policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
			got := len(actsOf(acts, policy.ActPreempt, "")) == 1
			t.Logf("win by 5 s with MarginS=%v -> preempt=%v", m, got)
			if got != (m < 5) {
				t.Errorf("MarginS=%v: preempt=%v", m, got)
			}
		}
	})

	t.Run("Reactive_SunkCostViaBuildProgress", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		// LocalBuildS 100, GPU 30, backlog 4 x 1 s all needing index.
		// progress 0:    stay = 100+4 = 104, migrate = 34 -> preempt.
		// progress 0.75: stay = 25+4  = 29,  migrate = 34 -> leave it.
		for _, c := range []struct {
			prog float64
			want bool
		}{{0, true}, {0.5, true}, {0.75, false}, {0.9, false}} {
			b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 30}, status(4, 4, c.prog, 1))
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
			got := len(actsOf(acts, policy.ActPreempt, "")) == 1
			stay := 100*(1-c.prog) + 4
			t.Logf("progress %.2f: stay=%.0f, migrate=34 -> preempt=%v (decision 1b: sunk cost from build_progress)", c.prog, stay, got)
			if got != c.want {
				t.Errorf("progress %.2f: preempt=%v want %v", c.prog, got, c.want)
			}
		}
	})

	t.Run("Reactive_NonIndexBacklogFavoursMigration", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		// Same numbers, split of the backlog differs. LocalBuildS 20 at 0%, GPU 15, backlog 10 x 1 s.
		// all need index:  stay = 30, migrate = max(0, 15+10) = 25 -> wins by 5 -> preempt
		// none need index: stay = 30, migrate = max(10, 15)   = 15 -> wins by 15 -> preempt
		// Make it tighter: GPU 19.
		// all need index:  stay 30, migrate 29 -> no.   none: migrate max(10,19)=19 -> yes.
		for _, c := range []struct {
			idx  int32
			want bool
		}{{10, false}, {0, true}} {
			b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 20, GPUBuildS: 19}, status(10, c.idx, 0, 1))
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
			got := len(actsOf(acts, policy.ActPreempt, "")) == 1
			t.Logf("backlog 10 x 1 s, %d need the index: stay=30, migrate=%v -> preempt=%v (flow: non-index queries drain from the moment the CPU frees)", c.idx, map[int32]float64{10: 29, 0: 19}[c.idx], got)
			if got != c.want {
				t.Errorf("needs_index=%d: preempt=%v want %v", c.idx, got, c.want)
			}
		}
	})

	t.Run("Reactive_QueueWaitOfLocalBuildEntersMigrate", func(t *testing.T) {
		// The hurting build from PreemptArithmetic, but the host says a GPU
		// copy would wait 100 s. migrate = max(6, 100+5+1+4) = 110 > stay 60.
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
		b.QueueWaitS = 100
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(2, 100), Builds: []policy.BuildView{b}})
		t.Logf("same hurting build, BuildView.QueueWaitS=100 and both workers busy 100 s -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "")) != 0 {
			t.Errorf("flow diagram: migrate includes the GPU queue wait (110 s) which loses to stay (60 s); preempted anyway. SPEC GAP: go doc says BuildView.QueueWaitS is 'meaningful for gpu_queued' and ReconsiderInput carries no PoolState.QueueWaitS, so which wait a local build's migrate branch uses is not documented")
		}
	})

	t.Run("Reactive_NoActionWithoutValidReport", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		st := status(10, 4, 0.5, 1)
		st.Valid = false
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, st)
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{b}})
		t.Logf("the hurting build but Status.Valid=false (shard has not reported) -> %s", fmtActs(acts))
		if len(acts) != 0 {
			t.Errorf("the reactive policy decides 'from the shard's report alone'; with no report there is nothing to act on; got %s", fmtActs(acts))
		}
		// A preempted queued build with an invalid report must not be reprioritised to 0.
		q := bview("2:1", "gpu_queued", 2, policy.Estimates{LocalBuildS: 30, GPUBuildS: 3}, policy.ShardStatus{})
		q.Moved, q.Priority = true, 60
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 10), Builds: []policy.BuildView{q}})
		t.Logf("preempted queued build at prio 60, invalid (zero) report -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActReprioritize, "")) != 0 {
			t.Errorf("reprioritized from an invalid report: %s", fmtActs(acts))
		}
	})

	t.Run("Reactive_MemoryAwarePreempt", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
		b.Spec.MemBytes = wMem + 1
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{b}})
		t.Logf("the hurting build needs %d B, both live workers have %d B -> %s", b.Spec.MemBytes, wMem, fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "")) != 0 {
			t.Errorf("CLAUDE.md: a build that fits no live worker must build locally; preempting it would park it in a queue no worker can claim")
		}
		acts = reconsider(t, p, policy.ReconsiderInput{Builds: []policy.BuildView{b}})
		t.Logf("SPEC GAP probe: same build with zero live workers -> %s (decision 19's cold-start rule is stated for submit, not for reconsider)", fmtActs(acts))
	})

	// ---- Reactive: decision 8, moved at most once ------------------------------

	t.Run("Reactive_MovedAtMostOnce_RenegedNeverPreempted", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 1000, GPUBuildS: 1}, status(50, 10, 0, 1))
		b.Moved = true
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{b}})
		t.Logf("a reneged build (Moved, local_running): stay=1050, migrate=11, idle pool -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "")) != 0 {
			t.Errorf("decision 8: a reneged build cannot be preempted; got %s", fmtActs(acts))
		}
		b.Moved = false
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{b}})
		t.Logf("control: same build not Moved -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "")) != 1 {
			t.Errorf("control failed: the unmoved build should be preempted")
		}
	})

	t.Run("Reactive_MovedAtMostOnce_PreemptedNeverReneges", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		b := bview("1:1", "gpu_queued", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0, 1))
		b.QueueWaitS = 1000
		b.Moved = true
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 1000), Builds: []policy.BuildView{b}})
		t.Logf("a preempted build (Moved, gpu_queued), local 10 s vs queue 1000+2 s -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActRenege, "")) != 0 {
			t.Errorf("decision 8: a preempted build cannot renege back; got %s", fmtActs(acts))
		}
		b.Moved = false
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 1000), Builds: []policy.BuildView{b}})
		t.Logf("control: same build not Moved -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActRenege, "1:1")) != 1 {
			t.Errorf("control failed: the unmoved stuck build should renege")
		}
	})

	// ---- Reactive: renege ------------------------------------------------------

	t.Run("Reactive_RenegeOnlyQueuedPastMargin", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		// Empty backlog: local from scratch = 10; GPU = wait + 2.
		for _, c := range []struct {
			wait float64
			want bool
		}{{0, false}, {6, false}, {10, false}, {10.5, true}, {500, true}} {
			b := bview("1:1", "gpu_queued", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0, 1))
			b.QueueWaitS = c.wait
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, c.wait), Builds: []policy.BuildView{b}})
			got := len(actsOf(acts, policy.ActRenege, "1:1")) == 1
			t.Logf("queued build, local 10 s vs GPU wait %v + 2 s: local wins by %v (margin 2) -> renege=%v %s", c.wait, c.wait+2-10, got, fmtActs(acts))
			if got != c.want {
				t.Errorf("wait %v: renege=%v want %v (decision 8: reneging must win by MarginS)", c.wait, got, c.want)
			}
		}
		b := bview("1:1", "gpu_leased", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0, 1))
		b.QueueWaitS = 500
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 500), Builds: []policy.BuildView{b}})
		t.Logf("a LEASED build with the same hopeless numbers -> %s", fmtActs(acts))
		if len(acts) != 0 {
			t.Errorf("decision 4 / CLAUDE.md: a leased build is never reneged or preempted; got %s", fmtActs(acts))
		}
	})

	t.Run("Reactive_RenegeOfSpeculativeCopyProbe", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		orig := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0.5, 1))
		cp := bview("1:1:spec", "gpu_queued", 1, policy.Estimates{LocalBuildS: 10, GPUBuildS: 2}, status(0, 0, 0.5, 1))
		cp.Speculative, cp.QueueWaitS = true, 500
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 500), Builds: []policy.BuildView{orig, cp}})
		t.Logf("SPEC GAP probe: a speculative GPU copy stuck in the queue (wait 500 s) while its local original runs -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "1:1")) != 0 {
			t.Logf("SPEC GAP: the local ORIGINAL of an in-flight speculative copy was preempted (its own stay 5 s vs migrate 2 s). The shard then has two GPU rows for one build and loses its CPU progress, which roadmap policy 4 says speculation exists to avoid. No doc says whether an original with a copy in flight is preemptable")
		}
		cpL := cp
		cpL.Phase = "gpu_leased"
		acts2 := reconsider(t, policy.NewReactiveSpec(policy.ReactiveConfig{}), policy.ReconsiderInput{Workers: busyWorkers(1, 3), Builds: []policy.BuildView{orig, cpL}})
		t.Logf("SPEC GAP probe: reactive_spec, the copy LEASED (running on the only worker), original stay 5 s vs migrate 2 s -> %s", fmtActs(acts2))
		if len(actsOf(acts, policy.ActRenege, "1:1:spec")) != 0 {
			t.Errorf("reneging a speculative copy to local would run the same build twice on the shard's CPU; decisions 4/5 do not say, but this cannot be meant")
		}
	})

	// ---- Reactive: reprioritize ------------------------------------------------

	t.Run("Reactive_ReprioritizeOnlyPastDelta", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{})
		mk := func(prio float64) policy.BuildView {
			b := bview("3:1", "gpu_queued", 3, policy.Estimates{LocalBuildS: 30, GPUBuildS: 3}, status(40, 10, 0, 1))
			b.Moved, b.Priority = true, prio
			return b
		}
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 10), Builds: []policy.BuildView{mk(0)}})
		rp := actsOf(acts, policy.ActReprioritize, "3:1")
		t.Logf("preempted queued build at prio 0, report: 40 waiting x 1 s -> %s", fmtActs(acts))
		if len(rp) != 1 {
			t.Fatalf("decision 10: a preempted build's priority is kept current by reprioritize; expected one action at prio 0 vs blocked work >= 40 s")
		}
		target := rp[0].Priority
		t.Logf("target blocked work the policy computes: %v s (the report's backlog is 40 x 1 s = 40 s)", target)
		if target < 40-1e-9 {
			t.Errorf("blocked work behind the build is at least the 40 s backlog; got %v", target)
		}
		for _, c := range []struct {
			off  float64
			want bool
		}{{0, false}, {0.5, false}, {-0.5, false}, {1, false}, {-1, false}, {1.5, true}, {-1.5, true}} {
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 10), Builds: []policy.BuildView{mk(target + c.off)}})
			got := len(actsOf(acts, policy.ActReprioritize, "3:1")) == 1
			t.Logf("stored prio = target%+.1f -> reprioritize=%v (decision 10: only past a 1 s churn delta) %s", c.off, got, fmtActs(acts))
			if got != c.want {
				t.Errorf("offset %+.1f: reprioritize=%v want %v", c.off, got, c.want)
			}
		}
		// Unmoved queued builds keep their submit priority.
		u := mk(0)
		u.Moved = false
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 10), Builds: []policy.BuildView{u}})
		t.Logf("an UNMOVED queued build at prio 0 with the same backlog -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActReprioritize, "")) != 0 {
			t.Errorf("ReactiveConfig doc: reprioritize applies to 'a queued preempted build'; an unmoved build was reprioritized: %s", fmtActs(acts))
		}
	})

	t.Run("Reactive_PreemptThenNextTickDoesNotChurn", func(t *testing.T) {
		// Tick 1: preempt the hurting build (priority = backlog + abandoned local).
		p := policy.NewReactive(policy.ReactiveConfig{})
		st := status(10, 4, 0.5, 1)
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, st)
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 30), Builds: []policy.BuildView{b}})
		pre := actsOf(acts, policy.ActPreempt, "1:1")
		if len(pre) != 1 {
			t.Fatalf("tick 1 should preempt; got %s", fmtActs(acts))
		}
		t.Logf("tick 1: %s", fmtActs(acts))
		// Tick 2: the host applied it. The build is gpu_queued, Moved, at that
		// priority. The shard's backlog is unchanged; it reports progress 0
		// because it no longer builds locally (go doc: '0 if not building locally').
		st2 := st
		st2.BuildProgress = 0
		b2 := b
		b2.Phase, b2.Moved, b2.Priority, b2.Status = "gpu_queued", true, pre[0].Priority, st2
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: busyWorkers(1, 29), Builds: []policy.BuildView{b2}})
		t.Logf("tick 2, same backlog, build now queued at prio %v -> %s", pre[0].Priority, fmtActs(acts))
		if rp := actsOf(acts, policy.ActReprioritize, "1:1"); len(rp) != 0 {
			t.Errorf("decision 10: priority after preemption is 'the backlog plus the abandoned local time, kept current by reprioritize past a 1 s churn delta'. The blocked work did not change between ticks, yet tick 2 moved the priority %v -> %v. The abandoned-time term is not recoverable from a gpu_queued BuildView (progress now reads 0), so reprioritize silently drops it", pre[0].Priority, rp[0].Priority)
		}
	})

	t.Run("Decision10_PrioritiesComparableWithinReactive", func(t *testing.T) {
		p := policy.NewReactive(policy.ReactiveConfig{ThresholdN: 1})
		// A fresh submit whose local build takes 10 s, and a preempted build
		// with a 10 x 1 s backlog behind it: both are 10 s of blocked work.
		d := decide(t, p, policy.Input{Build: spec(7, 100_000, smallMB), Est: policy.Estimates{LocalBuildS: 10, GPUBuildS: 1}, Pool: pool(1), Status: status(0, 0, 0, 1)})
		b := bview("1:1", "local_running", 1, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{b}})
		pre := actsOf(acts, policy.ActPreempt, "1:1")
		t.Logf("submit prio for a 10 s local build = %v; preempt prio for a 10 s backlog = %s", d.Priority, fmtActs(pre))
		if d.Placement != "gpu" || len(pre) != 1 {
			t.Fatalf("setup: expected a gpu submit and a preempt")
		}
		if !approx(d.Priority, pre[0].Priority) {
			t.Errorf("decision 10: one currency, seconds of blocked work; 10 s at submit and 10 s after preemption should carry equal priorities, got %v vs %v", d.Priority, pre[0].Priority)
		}
		// Two preempted shards: the one hurting more claims first.
		lo := bview("2:1", "local_running", 2, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5}, status(10, 0, 0.5, 1))
		hi := bview("3:1", "local_running", 3, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5}, status(40, 0, 0.5, 1))
		acts = reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{lo, hi}})
		a, c := actsOf(acts, policy.ActPreempt, "2:1"), actsOf(acts, policy.ActPreempt, "3:1")
		t.Logf("two hurting shards, backlogs 10 s and 40 s -> %s", fmtActs(acts))
		if len(a) != 1 || len(c) != 1 || !(c[0].Priority > a[0].Priority) {
			t.Errorf("roadmap policy 3: 'the shard that is hurting most is served first'; want both preempted with 3:1 above 2:1")
		}
	})

	// ---- ReactiveSpec: speculation ---------------------------------------------

	// Two mild stragglers that do not win migration by the margin:
	//   A: local 20 at 50% -> remaining 10, no backlog: stay 10, migrate 9 (wins by 1)
	//   B: local 40 at 50% -> remaining 20, 2 index queries: stay 22, migrate 19+2=21 (wins by 1)
	// Worst projected recovery: B.
	mild := func() (policy.BuildView, policy.BuildView) {
		a := bview("A", "local_running", 1, policy.Estimates{LocalBuildS: 20, GPUBuildS: 9}, status(0, 0, 0.5, 1))
		b := bview("B", "local_running", 2, policy.Estimates{LocalBuildS: 40, GPUBuildS: 19}, status(2, 2, 0.5, 1))
		return a, b
	}

	t.Run("Spec_SpeculatesWorstOncePerTick", func(t *testing.T) {
		p := policy.NewReactiveSpec(policy.ReactiveConfig{})
		a, b := mild()
		for _, order := range [][]policy.BuildView{{a, b}, {b, a}} {
			acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(1), Builds: order})
			sp := actsOf(acts, policy.ActSpeculate, "")
			t.Logf("idle worker, empty queue, stragglers A (stay 10) and B (stay 22), order %s,%s -> %s", order[0].BuildID, order[1].BuildID, fmtActs(acts))
			if len(actsOf(acts, policy.ActPreempt, "")) != 0 {
				t.Fatalf("setup: neither build wins migration by the margin; got a preempt")
			}
			if len(sp) != 1 || sp[0].BuildID != "B" {
				t.Errorf("roadmap policy 4: one speculation on the worst projected recovery (B); got %s", fmtActs(sp))
			}
		}
		// Three idle workers and three stragglers: still one per tick.
		c := bview("C", "local_running", 3, policy.Estimates{LocalBuildS: 30, GPUBuildS: 14}, status(0, 0, 0.5, 1))
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(3), Builds: []policy.BuildView{a, b, c}})
		t.Logf("three idle workers, three stragglers -> %s", fmtActs(acts))
		if n := len(actsOf(acts, policy.ActSpeculate, "")); n != 1 {
			t.Errorf("decision 8: speculation is one per reconsider tick; got %d", n)
		}
		// Plain Reactive never speculates.
		acts = reconsider(t, policy.NewReactive(policy.ReactiveConfig{}), policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{a, b}})
		if len(actsOf(acts, policy.ActSpeculate, "")) != 0 {
			t.Errorf("reactive (without _spec) speculated: %s", fmtActs(acts))
		}
		t.Logf("plain reactive on the same world -> %s (no speculation)", fmtActs(acts))
	})

	t.Run("Spec_NoGoCases", func(t *testing.T) {
		p := policy.NewReactiveSpec(policy.ReactiveConfig{})
		a, b := mild()
		queued := bview("Q", "gpu_queued", 5, policy.Estimates{LocalBuildS: 30, GPUBuildS: 3}, status(0, 0, 0, 1))
		copyQ := bview("B:spec", "gpu_queued", 2, b.Est, b.Status)
		copyQ.Speculative = true
		copyL := copyQ
		copyL.Phase = "gpu_leased"
		cases := []struct {
			name string
			in   policy.ReconsiderInput
		}{
			{"GPU queue non-empty (one queued build, one idle worker)", policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{a, b, queued}}},
			{"a speculative copy queued", policy.ReconsiderInput{Workers: idleWorkers(1), Builds: []policy.BuildView{a, b, copyQ}}},
			{"a speculative copy leased (one worker busy with it, one idle)", policy.ReconsiderInput{Workers: append(idleWorkers(1), policy.WorkerView{ID: "w9", MemBytes: wMem, BusyRemainingS: 5}), Builds: []policy.BuildView{a, b, copyL}}},
			{"every worker busy", policy.ReconsiderInput{Workers: busyWorkers(2, 3), Builds: []policy.BuildView{a, b}}},
			{"no workers at all", policy.ReconsiderInput{Builds: []policy.BuildView{a, b}}},
		}
		for _, c := range cases {
			acts := reconsider(t, p, c.in)
			t.Logf("%s -> %s", c.name, fmtActs(acts))
			if n := len(actsOf(acts, policy.ActSpeculate, "")); n != 0 {
				t.Errorf("%s: speculated; roadmap policy 4 / decision 8 forbid it", c.name)
			}
		}
	})

	t.Run("Spec_MemoryAware", func(t *testing.T) {
		p := policy.NewReactiveSpec(policy.ReactiveConfig{})
		a, b := mild()
		b.Spec.MemBytes = 8 * gib
		w := []policy.WorkerView{{ID: "small", Node: "n1", MemBytes: 4 * gib}, {ID: "big", Node: "n1", MemBytes: 32 * gib, BusyRemainingS: 50}}
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: w, Builds: []policy.BuildView{a, b}})
		t.Logf("idle worker has 4 GiB; worst straggler B needs 8 GiB (fits only the busy 32 GiB worker); A needs 1 GiB -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActSpeculate, "B")) != 0 {
			t.Errorf("speculating B onto a pool whose only idle worker cannot hold it burns nothing useful and leaves a copy queued (which also blocks the next speculation); the copy is meant for 'an idle worker'")
		}
		if len(actsOf(acts, policy.ActSpeculate, "A")) != 1 {
			t.Logf("SPEC GAP probe: A (which fits the idle worker) was not speculated instead; the docs do not say whether memory-ineligible candidates are skipped or end the search")
		}
	})

	t.Run("Spec_NeverTargetsBuildPreemptedSameTick", func(t *testing.T) {
		p := policy.NewReactiveSpec(policy.ReactiveConfig{})
		a, _ := mild()
		hurt := bview("H", "local_running", 3, policy.Estimates{LocalBuildS: 100, GPUBuildS: 5, TransferS: 1}, status(10, 4, 0.5, 1))
		acts := reconsider(t, p, policy.ReconsiderInput{Workers: idleWorkers(2), Builds: []policy.BuildView{a, hurt}})
		t.Logf("H wins migration by 50 s (stay 60) and is also the worst projected recovery; A is mild -> %s", fmtActs(acts))
		if len(actsOf(acts, policy.ActPreempt, "H")) != 1 {
			t.Fatalf("setup: H should be preempted")
		}
		for _, s := range actsOf(acts, policy.ActSpeculate, "") {
			if s.BuildID == "H" {
				t.Errorf("a build preempted this tick was also speculated; the preempted build is already headed to the GPU queue")
			}
		}
		if len(actsOf(acts, policy.ActSpeculate, "")) > 0 {
			t.Logf("SPEC GAP probe: a speculation was issued in the same tick as a preemption that will fill the GPU queue; the docs test 'queue empty' on the input, not after this tick's own actions")
		}
	})

	t.Run("Spec_StillReactive", func(t *testing.T) {
		pr := policy.NewReactive(policy.ReactiveConfig{})
		ps := policy.NewReactiveSpec(policy.ReactiveConfig{})
		w := busyWorld()
		w.Workers = busyWorkers(2, 4) // nobody idle: speculation cannot fire
		r, s := canon(reconsider(t, pr, w)), canon(reconsider(t, ps, w))
		t.Logf("busyWorld with no idle worker: reactive -> %v; reactive_spec -> %v", r, s)
		if !reflect.DeepEqual(r, s) {
			t.Errorf("ReactiveSpec is 'Reactive plus speculation'; without an idle worker the two should agree")
		}
		d1 := decide(t, pr, policy.Input{Build: spec(1, 300_000, smallMB), Est: policy.Estimates{LocalBuildS: 9}, Pool: pool(1)})
		d2 := decide(t, ps, policy.Input{Build: spec(1, 300_000, smallMB), Est: policy.Estimates{LocalBuildS: 9}, Pool: pool(1)})
		if d1.Placement != d2.Placement || d1.Priority != d2.Priority {
			t.Errorf("Decide differs between reactive and reactive_spec: %+v vs %+v", d1, d2)
		}
	})

	// ---- Clairvoyant -----------------------------------------------------------

	cin := func(shard int32, local, gpu, transfer, wait float64) policy.Input {
		return policy.Input{Build: spec(shard, 100_000, smallMB), Est: policy.Estimates{LocalBuildS: local, GPUBuildS: gpu, TransferS: transfer}, Pool: policy.PoolState{LiveWorkers: 2, LargestMem: wMem, QueueWaitS: wait}}
	}

	t.Run("Clairvoyant_ExactReplayWithSkipRule", func(t *testing.T) {
		// Shard 1: local 10, GPU 3 + transfer 1, no wait -> GPU ready at 4.
		//   q1 t=1 2 s no-index; q2 t=2 3 s index; q3 t=3 1 s no-index.
		//   local: build 0-10, q1 10-12, q2 12-15, q3 15-16 -> 16.
		//   gpu:   q1 1-3, q2 not ready (skipped), q3 3-4, q2 4-7      -> 7.
		//   (strict FIFO without the skip rule: q2 4-7, q3 7-8 -> 8.)
		// Shard 2 (the skip rule decides the placement): local 5, GPU 5 + 1 -> ready at 6.
		//   q1 t=0 1 s index; q2 t=0.5 4 s no-index.
		//   local: build 0-5, q1 5-6, q2 6-10                           -> 10.
		//   gpu with skip: q2 0.5-4.5, q1 6-7                          -> 7  -> gpu.
		//   gpu without skip: q1 6-7, q2 7-11                          -> 11 -> local.
		// Shard 3: local 2, GPU 10, one index query t=1 1 s.
		//   local: build 0-2, q 2-3 -> 3; gpu: q 10-11 -> 11 -> local.
		fut := map[int32][]policy.QueryEvent{
			1: {{ArriveS: 1, DurS: 2}, {ArriveS: 2, DurS: 3, NeedsIndex: true}, {ArriveS: 3, DurS: 1}},
			2: {{ArriveS: 0, DurS: 1, NeedsIndex: true}, {ArriveS: 0.5, DurS: 4}},
			3: {{ArriveS: 1, DurS: 1, NeedsIndex: true}},
		}
		p := policy.NewClairvoyant(fut)
		d1 := decide(t, p, cin(1, 10, 3, 1, 0))
		t.Logf("shard 1: local recovery 16 by hand, GPU 7 -> %s prio=%v %q", d1.Placement, d1.Priority, d1.Reason)
		if d1.Placement != "gpu" || !approx(d1.Priority, 16) {
			t.Errorf("roadmap policy 5: choose the sooner (gpu, 7 < 16) and priority = would-be local finish 16; got %+v. (If prio is 10, it is the build's local duration: decision 10's wording, which conflicts with the flow's 'local finish time')", d1)
		}
		d2 := decide(t, p, cin(2, 5, 5, 1, 0))
		t.Logf("shard 2: local 10, GPU 7 with the skip rule (11 without) -> %s prio=%v %q", d2.Placement, d2.Priority, d2.Reason)
		if d2.Placement != "gpu" {
			t.Errorf("CLAUDE.md skip rule: a not-ready index query waits without blocking others; the GPU replay is 7 s, local 10 s; got %s", d2.Placement)
		}
		if !approx(d2.Priority, 10) {
			t.Errorf("shard 2 priority should be its would-be local finish, 10; got %v", d2.Priority)
		}
		d3 := decide(t, p, cin(3, 2, 10, 0, 0))
		t.Logf("shard 3: local 3, GPU 11 -> %s %q", d3.Placement, d3.Reason)
		if d3.Placement != "local" {
			t.Errorf("shard 3 should build locally (3 < 11); got %s", d3.Placement)
		}
	})

	t.Run("Clairvoyant_QueueWaitAndLPT", func(t *testing.T) {
		p := policy.NewClairvoyant(map[int32][]policy.QueryEvent{})
		near := decide(t, p, cin(1, 10, 1, 0, 0))
		far := decide(t, p, cin(1, 10, 1, 0, 20))
		t.Logf("no queries, local 10 s vs GPU 1 s: queue wait 0 -> %s prio=%v; queue wait 20 -> %s", near.Placement, near.Priority, far.Placement)
		if near.Placement != "gpu" || far.Placement != "local" {
			t.Errorf("the GPU finish includes PoolState.QueueWaitS (decision 9); want gpu then local")
		}
		big := decide(t, p, cin(2, 40, 1, 0, 0))
		if !(big.Priority > near.Priority) {
			t.Errorf("longest-processing-time-first: a 40 s would-be local finish must outrank a 10 s one; got %v vs %v", big.Priority, near.Priority)
		}
		t.Logf("would-be local finish 40 s -> prio %v, 10 s -> prio %v", big.Priority, near.Priority)
	})

	t.Run("Clairvoyant_LateArrivalsAfterLocalBuild", func(t *testing.T) {
		// Queries after the build finishes run on arrival: local 4, query t=20 1 s.
		// local recovery 21; GPU (6) also 21 (query no-index runs on arrival, 20-21).
		// Add an index query at t=30 3 s: local 33; gpu 33. A tie; make GPU later:
		// GPU ready at 40 -> index query 40-43 -> gpu 43 -> local.
		p := policy.NewClairvoyant(map[int32][]policy.QueryEvent{1: {{ArriveS: 20, DurS: 1}, {ArriveS: 30, DurS: 3, NeedsIndex: true}}})
		d := decide(t, p, cin(1, 4, 40, 0, 0))
		t.Logf("local build 4 s, queries at 20 and 30 (index); GPU ready at 40: local 33 vs gpu 43 -> %s prio=%v", d.Placement, d.Priority)
		if d.Placement != "local" {
			t.Errorf("want local; got %+v", d)
		}
		// Same stream, GPU ready at 2: gpu 33 vs local 33 -> a tie. Record it.
		tie := decide(t, p, cin(1, 4, 2, 0, 0))
		t.Logf("SPEC GAP probe: equal recovery (33 vs 33) -> %s (the flow says 'earlier?' and names no tie-break)", tie.Placement)
	})

	t.Run("Clairvoyant_ShardWithNoFuture", func(t *testing.T) {
		withEmpty := policy.NewClairvoyant(map[int32][]policy.QueryEvent{7: {}})
		missing := policy.NewClairvoyant(map[int32][]policy.QueryEvent{1: {{ArriveS: 0, DurS: 100, NeedsIndex: true}}})
		nilMap := policy.NewClairvoyant(nil)
		in := cin(7, 10, 3, 1, 0)
		a, b, c := decide(t, withEmpty, in), decide(t, missing, in), decide(t, nilMap, in)
		t.Logf("shard 7 with an empty stream -> %+v; absent from the map -> %+v; nil futures -> %+v", a, b, c)
		if a.Placement != b.Placement || a.Priority != b.Priority || a.Placement != c.Placement || a.Priority != c.Priority {
			t.Errorf("a shard with no future entry should be treated as an empty stream; decisions differ")
		}
		if a.Placement != "gpu" || !approx(a.Priority, 10) {
			t.Errorf("empty stream: local 10 vs GPU 4 -> gpu with prio 10; got %+v", a)
		}
		t.Logf("SPEC GAP: the docs do not say what the oracle does for a shard missing from its futures (empty stream, error, or panic); observed: treated like an empty stream")
	})

	t.Run("Clairvoyant_UnsortedStreamSameAsSorted", func(t *testing.T) {
		sorted := []policy.QueryEvent{{ArriveS: 1, DurS: 2}, {ArriveS: 2, DurS: 3, NeedsIndex: true}, {ArriveS: 3, DurS: 1}}
		shuffled := []policy.QueryEvent{sorted[2], sorted[0], sorted[1]}
		a := decide(t, policy.NewClairvoyant(map[int32][]policy.QueryEvent{1: sorted}), cin(1, 10, 3, 1, 0))
		b := decide(t, policy.NewClairvoyant(map[int32][]policy.QueryEvent{1: shuffled}), cin(1, 10, 3, 1, 0))
		t.Logf("shard 1's stream in arrival order -> %+v; the same events permuted -> %+v", a, b)
		if a.Placement != b.Placement || !approx(a.Priority, b.Priority) {
			t.Errorf("the CPU runs queries in arrival order (CLAUDE.md), so the replay should not depend on slice order; or the docs should say the stream must be sorted (SPEC GAP either way)")
		}
		if !reflect.DeepEqual(shuffled, []policy.QueryEvent{sorted[2], sorted[0], sorted[1]}) {
			t.Errorf("NewClairvoyant/Decide reordered the caller's slice in place (a side effect on its input)")
		}
	})

	t.Run("Clairvoyant_NotDeployable", func(t *testing.T) {
		c := policy.NewClairvoyant(nil)
		_, err := policy.ByName(c.Name())
		t.Logf("Clairvoyant.Name()=%q; ByName of it -> err=%v (decision 1a: not reachable from config)", c.Name(), err)
		if err == nil {
			t.Errorf("decision 1a: ByName must never return the oracle")
		}
		acts := reconsider(t, c, busyWorld())
		t.Logf("clairvoyant.Reconsider on busyWorld -> %s (roadmap: it decides at submit with exact futures)", fmtActs(acts))
	})
}

