package crosscheck

// Phase 3 step 7 cross-check: the simulator validated against the live
// cluster, black-box. Spec: docs/design/phase3-policies.md decisions 26–30
// and the step 7 implementation notes (files, findings and results tables);
// experiments/workload/DESIGN.md (the straggler override and hour_build rows);
// docs/bugs.md entries 14 and 15; `go doc` of experiments/sim (Config.TickPhaseS,
// Result.ShardPlacement), experiments/workload (SizeDist.Straggler,
// PresetNames, PresetWithShards, Generate), shard/sim.Timeline, costmodel, and
// the comparer's -h text. Pure: no database, no cluster, no env gate.
//
// Hand numbers (default cost model, formulas from the costmodel package doc,
// computed in sim_helpers_test.go's sx* helpers, not by calling costmodel):
//   n=12.3M dim=128: CPU 3598.3 s  GPU build 361.8 s  transfer 70.85 s
//                    GPU total 432.7 s  memory 12.595e9 B = 11.73 GiB
//   n=100k  dim=128: CPU 20.63 s  GPU total 4.639 s
//   n=50k   dim=128: GPU total 3.258 s

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

const (
	vdStraggler = int64(12_300_000)
	vdGiB       = int64(1) << 30
	// vdLiveMem is the live cluster's WORKER_MEM_BYTES (decision 26). The
	// step 7 results table's high_dim row E reproduces only with it, not
	// with the high_dim preset's own 1.13 GB (see WorkerMemFlag below).
	vdLiveMem = 24 * vdGiB
)

// ---------------------------------------------------------------------------
// Decision 27: hour_build and SizeDist.Straggler
// ---------------------------------------------------------------------------

func TestStep7Straggler(t *testing.T) {
	t.Run("PresetNamesEndWithHourBuild", func(t *testing.T) {
		names := workload.PresetNames()
		t.Logf("PresetNames() = %v", names)
		want := []string{"uniform", "skewed", "bimodal", "downstream_heavy", "staggered", "shrinking_pool", "high_dim", "production", "hour_build"}
		if !reflect.DeepEqual(names, want) {
			t.Errorf("PresetNames = %v, want %v (go doc PresetNames: the roadmap's eight in roadmap order, then the step 7 addition)", names, want)
		}
	})

	t.Run("HourBuildIsSkewedWithRank1Grown", func(t *testing.T) {
		// Decision 27 / DESIGN.md hour_build row: exactly one shard (skewed's
		// rank-1, i.e. its largest; ties to the lowest id) becomes 12.3M;
		// every other size, every DDL offset and every query stream equal
		// skewed's for the same seed. Checked at shard counts and seeds the
		// implementer's TestStragglerOverride does not use (it uses 6 and 50,
		// three seeds).
		for _, n := range []int{2, 6, 13, 50} {
			for _, seed := range []int64{1, 2, 3, 7, 42, 1000003} {
				sk := vdWorkload(t, "skewed", n, seed)
				hb := vdWorkload(t, "hour_build", n, seed)
				if err := hb.Validate(); err != nil {
					t.Errorf("n=%d seed=%d: hour_build workload fails Validate: %v (a straggler workload must stay structurally valid)", n, seed, err)
				}
				big, bigN := -1, int64(-1)
				for i, s := range sk.Shards {
					if s.NVectors > bigN {
						big, bigN = i, s.NVectors
					}
				}
				var changed []int
				for i := range sk.Shards {
					a, b := sk.Shards[i], hb.Shards[i]
					if a.NVectors != b.NVectors {
						changed = append(changed, i)
					}
					if a.DDLOffset != b.DDLOffset || a.ShardID != b.ShardID {
						t.Errorf("n=%d seed=%d shard %d: DDL offset/id differ from skewed (%v/%d vs %v/%d)", n, seed, i, b.DDLOffset, b.ShardID, a.DDLOffset, a.ShardID)
					}
					if !reflect.DeepEqual(a.Queries, b.Queries) {
						t.Errorf("n=%d seed=%d shard %d: query stream differs from skewed's (%d vs %d queries): decision 27 'no extra random draw'", n, seed, i, len(b.Queries), len(a.Queries))
					}
				}
				if len(changed) != 1 || changed[0] != big || hb.Shards[big].NVectors != vdStraggler {
					t.Errorf("n=%d seed=%d: shards changed %v (want exactly [%d], skewed's largest at %d vectors) and hour_build size there = %d (want 12.3M)",
						n, seed, changed, big, bigN, hb.Shards[max(big, 0)].NVectors)
					continue
				}
				if n == 6 || seed == 1 {
					t.Logf("n=%d seed=%d: skewed's largest is shard %d (%d vectors); hour_build has %d there and is identical elsewhere (sizes, DDL offsets, %d query streams)",
						n, seed, big, bigN, hb.Shards[big].NVectors, n)
				}
			}
		}
	})

	t.Run("HourBuildScenarioIsSkewedPlusStraggler", func(t *testing.T) {
		for _, n := range []int{6, 50} {
			sk, _ := workload.PresetWithShards("skewed", n)
			hb, err := workload.PresetWithShards("hour_build", n)
			if err != nil {
				t.Fatalf("PresetWithShards(hour_build, %d): %v", n, err)
			}
			t.Logf("n=%d hour_build scenario: dim=%d sizes=%+v pool=%+v", n, hb.Dim, hb.Sizes, hb.Pool)
			if hb.Sizes.Straggler != vdStraggler {
				t.Errorf("hour_build Sizes.Straggler = %d, want 12.3M (decision 27)", hb.Sizes.Straggler)
			}
			hb2 := hb
			hb2.Name, hb2.Sizes.Straggler = sk.Name, 0
			if !reflect.DeepEqual(hb2, sk) {
				t.Errorf("n=%d: hour_build minus its name and straggler differs from skewed:\n hb=%+v\n sk=%+v", n, hb2, sk)
			}
			wantW := 2
			if n >= 50 {
				wantW = 3
			}
			if hb.Pool.Workers != wantW || hb.Pool.WorkerMemBytes != 24*vdGiB || hb.Dim != 128 {
				t.Errorf("n=%d: pool %+v dim %d, want %d workers of 24 GiB at dim 128 (DESIGN.md presets: 2/3 workers; decision 26)", n, hb.Pool, hb.Dim, wantW)
			}
		}
	})

	t.Run("StragglerOnUniformTiesGoToLowestID", func(t *testing.T) {
		// Uniform [100k, 100k]: every draw ties, so the straggler must land on
		// shard 0 (go doc SizeDist.Straggler: "ties to the lowest id").
		sc, _ := workload.PresetWithShards("uniform", 6)
		base := workload.Generate(sc, 5)
		sc.Sizes.Straggler = 777_777
		w := workload.Generate(sc, 5)
		var sizes []int64
		for _, s := range w.Shards {
			sizes = append(sizes, s.NVectors)
		}
		t.Logf("uniform [%d,%d] with Straggler=777777, seed 5: sizes %v", sc.Sizes.Min, sc.Sizes.Max, sizes)
		vdCheckOneReplaced(t, base, w, 0, 777_777)
	})

	t.Run("StragglerOnRandomUniformAndBimodal", func(t *testing.T) {
		cases := []struct {
			name string
			sc   func() workload.Scenario
		}{
			{"uniform[10k,200k]", func() workload.Scenario {
				sc, _ := workload.PresetWithShards("uniform", 8)
				sc.Sizes.Min, sc.Sizes.Max = 10_000, 200_000
				return sc
			}},
			{"uniform[10k,12k] (ties likely)", func() workload.Scenario {
				sc, _ := workload.PresetWithShards("uniform", 50)
				sc.Sizes.Min, sc.Sizes.Max = 10_000, 10_003
				return sc
			}},
			{"bimodal@6", func() workload.Scenario { sc, _ := workload.PresetWithShards("bimodal", 6); return sc }},
			{"bimodal@50", func() workload.Scenario { sc, _ := workload.PresetWithShards("bimodal", 50); return sc }},
			{"staggered (DDL offsets drawn after sizes)", func() workload.Scenario { sc, _ := workload.PresetWithShards("staggered", 6); return sc }},
			{"high_dim", func() workload.Scenario { sc, _ := workload.PresetWithShards("high_dim", 6); return sc }},
		}
		for _, c := range cases {
			for _, seed := range []int64{1, 2, 3, 99} {
				sc := c.sc()
				base := workload.Generate(sc, seed)
				big, bigN := -1, int64(-1)
				for i, s := range base.Shards {
					if s.NVectors > bigN {
						big, bigN = i, s.NVectors
					}
				}
				sc.Sizes.Straggler = 5_000_000
				w := workload.Generate(sc, seed)
				if seed == 1 {
					t.Logf("%s seed %d: largest draw is shard %d (%d vectors, lowest id among ties); expect it alone replaced by 5M", c.name, seed, big, bigN)
				}
				vdCheckOneReplaced(t, base, w, big, 5_000_000)
			}
		}
	})

	t.Run("StragglerZeroOrNegativeIsOff", func(t *testing.T) {
		// go doc: "Straggler, if > 0, overrides". Zero and negative are off.
		sc, _ := workload.PresetWithShards("skewed", 6)
		base := workload.Generate(sc, 4)
		sc.Sizes.Straggler = -5
		w := workload.Generate(sc, 4)
		if !reflect.DeepEqual(base.Shards, w.Shards) {
			t.Errorf("Straggler=-5 changed the workload; go doc says only > 0 overrides")
		} else {
			t.Logf("Straggler=-5: workload identical to Straggler=0 (go doc: 'if > 0')")
		}
	})

	t.Run("StragglerCostModelNumbers", func(t *testing.T) {
		m := costmodel.Default()
		cpu := m.CPUBuild(vdStraggler, 128).Seconds()
		gpu := m.GPUBuild(vdStraggler, 128).Seconds()
		xf := m.Transfer(vdStraggler, 128).Seconds()
		tot := m.GPUTotal(vdStraggler, 128).Seconds()
		mem := m.GPUMemBytes(vdStraggler, 128)
		t.Logf("costmodel.Default() on 12.3M x 128: CPU %.1f s, GPU build %.1f s, transfer %.2f s, GPU total %.1f s, memory %d B = %.2f GiB",
			cpu, gpu, xf, tot, mem, float64(mem)/float64(vdGiB))
		t.Logf("hand (costmodel doc formulas): CPU %.1f s, GPU build %.1f s, transfer %.2f s, total %.1f s, memory %d B",
			sxCPU(vdStraggler, 128), sxGPUBuild(vdStraggler, 128, 10), sxXfer(vdStraggler, 128), sxGPUTotal(vdStraggler, 128), sxMem(vdStraggler, 128))
		if math.Abs(cpu-sxCPU(vdStraggler, 128)) > 1e-3 || math.Abs(tot-sxGPUTotal(vdStraggler, 128)) > 1e-3 || mem != sxMem(vdStraggler, 128) {
			t.Errorf("costmodel disagrees with its own documented formulas at 12.3M")
		}
		if math.Abs(cpu-3600)/3600 > 0.05 {
			t.Errorf("CPU build %.1f s is not 'one modeled hour' within 5%% (decision 27)", cpu)
		}
		if mem > 24*vdGiB {
			t.Errorf("memory %d B exceeds decision 26's 24 GiB: the straggler could never be offloaded", mem)
		}
		if mem <= 8*vdGiB {
			t.Errorf("memory %d B fits 8 GiB, contradicting decision 26 ('did not fit 8')", mem)
		}
		// Doc numbers. Decision 27 / DESIGN.md / results row F, as corrected
		// after this test's first run (F1: the first draft said 69 s and
		// ~11.5 GiB): "362 s plus a 71 s transfer on a GPU (433 s), 11.7 GiB".
		if math.Abs(gpu-362) > 1.0 {
			t.Errorf("GPU build %.1f s vs the docs' 362 s", gpu)
		}
		if math.Abs(tot-433) > 1.0 {
			t.Errorf("GPU total %.1f s vs the docs' 433 s", tot)
		}
		if math.Abs(xf-71) > 1.0 {
			t.Errorf("DOC NUMBER: transfer = %.2f s by the cost model, the docs say 71 s (decision 27, DESIGN.md hour_build row, results row F)", xf)
		}
		gibs := float64(mem) / float64(vdGiB)
		if math.Abs(gibs-11.7) > 0.15 {
			t.Logf("note: modeled memory is %.2f GiB (%.2f GB); the docs say '~11.5 GiB' — approximate, conclusion (fits 24, not 8) unaffected", gibs, float64(mem)/1e9)
		}
	})
}

// vdCheckOneReplaced asserts w equals base except shard idx's size == size.
func vdCheckOneReplaced(t *testing.T, base, w workload.Workload, idx int, size int64) {
	t.Helper()
	if err := w.Validate(); err != nil {
		t.Errorf("seed %d: workload with straggler fails Validate: %v", w.Seed, err)
	}
	for i := range base.Shards {
		a, b := base.Shards[i], w.Shards[i]
		if i == idx {
			if b.NVectors != size {
				t.Errorf("seed %d: shard %d (largest draw, lowest id) has %d, want the straggler %d", w.Seed, i, b.NVectors, size)
			}
		} else if a.NVectors != b.NVectors {
			t.Errorf("seed %d: shard %d changed size %d -> %d; only shard %d should", w.Seed, i, a.NVectors, b.NVectors, idx)
		}
		if a.DDLOffset != b.DDLOffset || !reflect.DeepEqual(a.Queries, b.Queries) {
			t.Errorf("seed %d: shard %d DDL offset or stream changed (no extra random draw promised)", w.Seed, i)
		}
	}
}

// ---------------------------------------------------------------------------
// Decision 29: Config.TickPhaseS
// ---------------------------------------------------------------------------

// vdPreemptCase: shard 0 (100k) submits local at 0 on a one-or-two-worker
// pool; the script preempts it at the first tick it sees (prio 1). Shard 1
// (50k), if present, goes to the GPU at submit and never moves. Returns the
// result and the times of every tick the script was called at (counted).
func vdPreemptCase(t *testing.T, label string, cfg sim.Config, withQueries, withOther bool, preemptOnTick int) (sim.Result, int) {
	t.Helper()
	calls := 0
	sc := &sxScript{
		decide: sxGPUAt(map[int32]float64{1: 10}),
		recon: func(in policy.ReconsiderInput) []policy.Action {
			calls++
			if calls != preemptOnTick {
				return nil
			}
			if b, ok := sxFindView(in.Builds, 0, false); ok && b.Phase == "local_running" {
				return []policy.Action{{Kind: policy.ActPreempt, BuildID: b.BuildID, Priority: 1, Reason: "xcheck"}}
			}
			return nil
		},
	}
	var qs []workload.Query
	if withQueries {
		// All arrive before the earliest abort the test uses (1.0), so every
		// one of them is downstream of the preemption.
		qs = []workload.Query{sxQ(0.1, 0.2, false), sxQ(0.2, 0.2, false), sxQ(0.3, 0.1, true), sxQ(0.4, 0.05, true)}
	}
	shards := []workload.ShardWorkload{sxShard(0, 100_000, qs...)}
	workers := 1
	if withOther {
		shards = append(shards, sxShard(0, 50_000))
		workers = 2
	}
	w := sxWorkload(t, 128, workers, 16*vdGiB, 0.05, shards...)
	r := sxRun(t, label, w, sc, cfg)
	t.Logf("%s: wastedCPU=%.4f placement=%v (reconsider called %d times)", label, r.WastedCPUS, r.ShardPlacement, calls)
	return r, calls
}

// vdNextReportAfter is the first report instant strictly after x on the
// k*poll grid: a report AT the tick's instant is published before the tick
// (go doc sim: same-instant ordering), so the shard learns at the next one.
func vdNextReportAfter(x, poll float64) float64 {
	return math.Floor(x/poll+1e-9)*poll + poll
}

func TestStep7TickPhase(t *testing.T) {
	T := sxGPUTotal(100_000, 128)
	T50 := sxGPUTotal(50_000, 128)
	t.Logf("hand: GPU total 100k = %.4f s, 50k = %.4f s", T, T50)

	t.Run("FirstTickAtPhaseShiftsPreemptionByHand", func(t *testing.T) {
		// Decision 29 / go doc TickPhaseS: the first tick fires at p (0 < p <=
		// ReconsiderS). The preempt is executed at the tick and the free
		// worker claims in the same instant, so the build finishes at p + T;
		// default p = ReconsiderS = 2. The shard aborts its local build at
		// its next report after p, so wasted CPU = that report instant.
		for _, p := range []float64{0, 2, 1, 0.5, 0.7, 1.3, 1.999, 0.001} {
			eff := p
			if p == 0 {
				eff = 2
			}
			r, _ := vdPreemptCase(t, fmt.Sprintf("phase %.3f", p), sim.Config{TickPhaseS: p}, false, false, 1)
			if r.Preempts != 1 || !r.Completed {
				t.Errorf("phase %.3f: preempts=%d completed=%v, want 1 / true", p, r.Preempts, r.Completed)
				continue
			}
			sxExpect(t, fmt.Sprintf("phase %.3f: round", p), r.RoundS, eff+T, "decision 29: preempt at the first tick p, claimed at once; round = p + GPU total")
			sxExpect(t, fmt.Sprintf("phase %.3f: shift vs default", p), (2+T)-r.RoundS, 2-eff, "decision 29: shift = ReconsiderS - p")
			sxExpect(t, fmt.Sprintf("phase %.3f: wasted CPU", p), r.WastedCPUS, vdNextReportAfter(eff, 0.5), "shard aborts at its next report after the tick (decision 21)")
			if !reflect.DeepEqual(r.ShardPlacement, []string{"gpu"}) {
				t.Errorf("phase %.3f: ShardPlacement = %v, want [gpu] (a preempted build finishes on the GPU)", p, r.ShardPlacement)
			}
		}
	})

	t.Run("EverythingDownstreamShiftsOnTheReportGrid", func(t *testing.T) {
		// With queries behind the build (all arriving before the earliest
		// abort) and an unrelated GPU shard, phase 1.0 vs 2.0 vs 1.5 (all on
		// the 0.5 s report grid): the preempted shard's finish and the round
		// shift by exactly 2 - p; the unrelated shard does not move.
		base, _ := vdPreemptCase(t, "phase 2 (default)", sim.Config{}, true, true, 1)
		for _, p := range []float64{1.0, 1.5, 0.5} {
			r, _ := vdPreemptCase(t, fmt.Sprintf("phase %.1f", p), sim.Config{TickPhaseS: p}, true, true, 1)
			if len(r.ShardFinishS) != 2 || len(base.ShardFinishS) != 2 {
				t.Fatalf("want 2 shard finishes")
			}
			sxExpect(t, fmt.Sprintf("p=%.1f: shard 0 finish shift", p), base.ShardFinishS[0]-r.ShardFinishS[0], 2-p, "decision 29: everything downstream shifts by ReconsiderS - p")
			sxExpect(t, fmt.Sprintf("p=%.1f: round shift", p), base.RoundS-r.RoundS, 2-p, "decision 29")
			sxExpect(t, fmt.Sprintf("p=%.1f: shard 1 finish (unrelated, GPU from submit)", p), r.ShardFinishS[1], T50, "not downstream of the tick")
			sxExpect(t, fmt.Sprintf("p=%.1f: wasted CPU shift", p), base.WastedCPUS-r.WastedCPUS, 2-p, "abort at next report, on-grid phase")
		}
	})

	t.Run("OffGridPhaseShiftIsQuantizedByReports", func(t *testing.T) {
		// Not a promise in the docs; a probe of decision 29's "shifts the whole
		// timeline by exactly the phase". With queries waiting on the shard,
		// the abort and the index's arrival are both seen at the shard's next
		// 0.5 s report, so an off-grid phase shifts the shard's finish by a
		// report-grid amount, not by ReconsiderS - p. Hand, p = 0.7: abort at
		// 1.0; build done 0.7 + T = 5.339, seen at report 5.5; the two index
		// queries (0.1 + 0.05) run 5.5 to 5.65. Default: 7.15. Shift 1.5, not 1.3.
		base, _ := vdPreemptCase(t, "default", sim.Config{}, true, true, 1)
		r, _ := vdPreemptCase(t, "phase 0.7", sim.Config{TickPhaseS: 0.7}, true, true, 1)
		want := vdNextReportAfter(0.7+T, 0.5) + 0.15
		sxExpect(t, "phase 0.7: shard 0 finish", r.ShardFinishS[0], want, "report-quantized: index seen at next report after p+T, then 0.10 + 0.05 of index queries")
		t.Logf("shift of shard 0's finish = %.4f s, while ReconsiderS - p = 1.3 s: 'exactly the phase' holds for the build's own finish and for on-grid phases; off-grid, everything a shard learns through a report moves on the 0.5 s grid (G-item)", base.ShardFinishS[0]-r.ShardFinishS[0])
	})

	t.Run("OutOfRangePhaseBehavesAsDefault", func(t *testing.T) {
		// go doc: "if > 0 ... at most ReconsiderS ... Zero means ReconsiderS".
		// Negative and > ReconsiderS are not stated; the brief expects
		// "behaves as the default" (clamping > R to R gives the same).
		def, _ := vdPreemptCase(t, "default", sim.Config{}, true, true, 1)
		for _, p := range []float64{-1, -0.5, 2.0000001, 2.5, 3, 100} {
			r, _ := vdPreemptCase(t, fmt.Sprintf("phase %g", p), sim.Config{TickPhaseS: p}, true, true, 1)
			if !sxNear(r.RoundS, def.RoundS) || !reflect.DeepEqual(r.ShardFinishS, def.ShardFinishS) || !sxNear(r.WastedCPUS, def.WastedCPUS) {
				t.Errorf("TickPhaseS=%g: round %.4f, finishes %s, wasted %.4f; default gives %.4f, %s, %.4f (out-of-range phase should behave as the default)",
					p, r.RoundS, sxFloats(r.ShardFinishS), r.WastedCPUS, def.RoundS, sxFloats(def.ShardFinishS), def.WastedCPUS)
			} else {
				t.Logf("ok: TickPhaseS=%g behaves exactly as the default (round %.4f)", p, r.RoundS)
			}
		}
	})

	t.Run("PhaseIsRelativeToConfiguredInterval", func(t *testing.T) {
		// ReconsiderS = 4: phase 3 is in range (preempt at 3), phase 4 is
		// the default (preempt at 4), phase 0 is the default.
		for _, c := range []struct{ p, want float64 }{{0, 4}, {4, 4}, {3, 3}, {0.5, 0.5}} {
			r, _ := vdPreemptCase(t, fmt.Sprintf("R=4 phase %.1f", c.p), sim.Config{ReconsiderS: 4, TickPhaseS: c.p}, false, false, 1)
			sxExpect(t, fmt.Sprintf("R=4 phase %.1f: round", c.p), r.RoundS, c.want+T, "go doc TickPhaseS: first tick at p <= ReconsiderS; zero means ReconsiderS")
		}
	})

	t.Run("LaterTicksFollowEveryInterval", func(t *testing.T) {
		// go doc: "later ticks follow every ReconsiderS". The script preempts
		// on the SECOND tick: at p + 2 (p = 0.5 -> 2.5), default 4.
		for _, c := range []struct{ p, want float64 }{{0, 4}, {0.5, 2.5}, {1.25, 3.25}} {
			r, _ := vdPreemptCase(t, fmt.Sprintf("2nd-tick preempt, phase %.2f", c.p), sim.Config{TickPhaseS: c.p}, false, false, 2)
			sxExpect(t, fmt.Sprintf("phase %.2f: round", c.p), r.RoundS, c.want+T, "second tick at p + ReconsiderS")
		}
	})

	t.Run("DeterministicUnderAnyPhase", func(t *testing.T) {
		// Invariant 10 / go doc Run: same workload, policy, config -> same Result.
		w := vdWorkload(t, "high_dim", 6, 2)
		for _, p := range []float64{0.25, 0.75, 1.9} {
			a := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, TickPhaseS: p, LeaseS: 10})
			b := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, TickPhaseS: p, LeaseS: 10})
			if !reflect.DeepEqual(a, b) {
				t.Errorf("phase %.2f: two runs differ:\n%+v\n%+v", p, a, b)
			} else {
				t.Logf("ok: high_dim seed 2 reactive at phase %.2f twice: identical (round %.4f, preempts %d)", p, a.RoundS, a.Preempts)
			}
		}
	})

	t.Run("AlwaysGPUIsPhaseInvariant", func(t *testing.T) {
		// always_gpu returns no reconsider actions, so the tick's phase can
		// change nothing about the round (decision 29's sensitivity comes only
		// from what the first tick does).
		for _, c := range []struct {
			sc   string
			n    int
			seed int64
		}{{"skewed", 6, 1}, {"skewed", 50, 2}, {"hour_build", 6, 1}, {"high_dim", 6, 3}, {"staggered", 6, 1}} {
			w := vdWorkload(t, c.sc, c.n, c.seed)
			def := vdReplay(t, w, "always_gpu", sim.Config{LeaseS: 10})
			ok := true
			for _, p := range []float64{0.1, 0.5, 0.7, 1.3, 2} {
				r := vdReplay(t, w, "always_gpu", sim.Config{TickPhaseS: p, LeaseS: 10})
				r.Events, def.Events = 0, 0 // the tick count itself may differ
				if !reflect.DeepEqual(r, def) {
					ok = false
					t.Errorf("%s/%d seed %d always_gpu: phase %.1f gives round %.4f finishes %s, default %.4f %s", c.sc, c.n, c.seed, p, r.RoundS, sxFloats(r.ShardFinishS), def.RoundS, sxFloats(def.ShardFinishS))
				}
			}
			if ok {
				t.Logf("ok: %s/%d seed %d always_gpu: identical Result (Events aside) at phases 0.1..2; round %.4f", c.sc, c.n, c.seed, def.RoundS)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// Result.ShardPlacement
// ---------------------------------------------------------------------------

func TestStep7ShardPlacement(t *testing.T) {
	vdCheck := func(t *testing.T, label string, n int, r sim.Result) {
		t.Helper()
		g, l, e := 0, 0, 0
		for _, p := range r.ShardPlacement {
			switch p {
			case "gpu":
				g++
			case "local":
				l++
			case "":
				e++
			default:
				t.Errorf("%s: unexpected placement %q (go doc: gpu, local or \"\")", label, p)
			}
		}
		if len(r.ShardPlacement) != n {
			t.Errorf("%s: %d placements for %d shards (go doc: per shard)", label, len(r.ShardPlacement), n)
		}
		if g != r.BuildsGPU || l != r.BuildsLocal {
			t.Errorf("%s: placements gpu=%d local=%d but BuildsGPU=%d BuildsLocal=%d", label, g, l, r.BuildsGPU, r.BuildsLocal)
		}
		if r.Completed && e != 0 {
			t.Errorf("%s: completed round has %d empty placements (go doc: \"\" only if unfinished)", label, e)
		}
	}

	t.Run("ConsistentWithBuildCountsAcrossPresetsAndPolicies", func(t *testing.T) {
		for _, name := range workload.PresetNames() {
			if name == "production" {
				continue // hours of modeled time per round; covered by the counts invariant elsewhere
			}
			for _, n := range []int{6, 50} {
				w := vdWorkload(t, name, n, 3)
				for _, pol := range []string{"always_gpu", "static_threshold", "reactive", "reactive_spec"} {
					r := vdReplay(t, w, pol, sim.Config{})
					vdCheck(t, fmt.Sprintf("%s/%d %s", name, n, pol), n, r)
					if n == 6 {
						t.Logf("%s/%d %s: completed=%v gpu=%d local=%d placement=%v", name, n, pol, r.Completed, r.BuildsGPU, r.BuildsLocal, r.ShardPlacement)
					}
				}
			}
		}
	})

	t.Run("ShardOrderMatchesSizes", func(t *testing.T) {
		// static_threshold (DefaultThresholdN = 200k) places by size, so on
		// hour_build the 12.3M straggler must read "gpu" and every build under
		// 100k "local", at its own shard's index (go doc: per shard, in order).
		w := vdWorkload(t, "hour_build", 6, 1)
		r := vdReplay(t, w, "static_threshold", sim.Config{})
		for i, s := range w.Shards {
			t.Logf("shard %d: %d vectors -> %q", i, s.NVectors, r.ShardPlacement[i])
			if s.NVectors == vdStraggler && r.ShardPlacement[i] != "gpu" {
				t.Errorf("shard %d (the 12.3M straggler) placed %q under static_threshold, want gpu", i, r.ShardPlacement[i])
			}
			if s.NVectors < 100_000 && r.ShardPlacement[i] != "local" {
				t.Errorf("shard %d (%d vectors, below the 200k threshold) placed %q, want local", i, s.NVectors, r.ShardPlacement[i])
			}
		}
	})

	t.Run("IncompleteRoundLeavesUnfinishedEmpty", func(t *testing.T) {
		// hour_build with every build forced local and a 100 s modeled cap:
		// the five small builds finish locally, the 12.3M one (an hour) does
		// not. go doc: Completed=false when the cap fires; "" if unfinished.
		w := vdWorkload(t, "hour_build", 6, 1)
		r := sim.Run(w, &sxScript{decide: sxLocal}, sim.Config{MaxModeledS: 100})
		t.Logf("all-local, cap 100 s: completed=%v gpu=%d local=%d placement=%v", r.Completed, r.BuildsGPU, r.BuildsLocal, r.ShardPlacement)
		if r.Completed {
			t.Errorf("round completed under a 100 s cap with a one-hour local build")
		}
		vdCheck(t, "capped", 6, r)
		for i, s := range w.Shards {
			want := "local"
			if s.NVectors == vdStraggler {
				want = ""
			}
			if i < len(r.ShardPlacement) && r.ShardPlacement[i] != want {
				t.Errorf("shard %d (%d vectors): placement %q, want %q", i, s.NVectors, r.ShardPlacement[i], want)
			}
		}
	})

	t.Run("KilledPoolLeavesQueuedEmpty", func(t *testing.T) {
		// One worker, always_gpu, the busiest worker killed at 0.1 s: the pool
		// is empty forever, nothing GPU-placed finishes. go doc Result: an
		// unfinished build counts nowhere; placement "".
		w := vdWorkload(t, "skewed", 6, 1)
		r := vdReplay(t, w, "always_gpu", sim.Config{Workers: 1, KillWorkerAtS: 0.1, MaxModeledS: 300})
		t.Logf("1 worker killed at 0.1 s: completed=%v gpu=%d local=%d placement=%v", r.Completed, r.BuildsGPU, r.BuildsLocal, r.ShardPlacement)
		vdCheck(t, "killed", 6, r)
		if r.Completed || r.BuildsGPU != 0 {
			t.Errorf("want an incomplete round with no GPU finishes")
		}
	})
}

// ---------------------------------------------------------------------------
// Decision 28/29: the comparer as a black-box process
// ---------------------------------------------------------------------------

// vdRow is the comparer's main row for one round, parsed by position from
// its header "round seed scale live_s sim_s delta_s | live_lag sim_lag | ...
// phase envelope verdict".
type vdRow struct {
	liveS, simS, deltaS float64
	envLo, envHi        float64
	verdict             string
}

func vdParseRow(t *testing.T, out string, roundID int64) (vdRow, bool) {
	t.Helper()
	id := strconv.FormatInt(roundID, 10)
	for _, line := range strings.Split(out, "\n") {
		f := strings.Fields(line)
		if len(f) < 8 || f[0] != id || !strings.Contains(line, "|") {
			continue
		}
		var r vdRow
		r.liveS, _ = strconv.ParseFloat(f[3], 64)
		r.simS, _ = strconv.ParseFloat(f[4], 64)
		r.deltaS, _ = strconv.ParseFloat(f[5], 64)
		m := vdEnvRe.FindStringSubmatchIndex(line)
		if m != nil {
			r.envLo, _ = strconv.ParseFloat(line[m[2]:m[3]], 64)
			r.envHi, _ = strconv.ParseFloat(line[m[4]:m[5]], 64)
			r.verdict = strings.TrimSpace(line[m[1]:])
		}
		return r, true
	}
	return vdRow{}, false
}

func TestStep7Comparer(t *testing.T) {
	sk1 := vdWorkload(t, "skewed", 6, 1)
	skR := vdReplay(t, sk1, "always_gpu", sim.Config{LeaseS: 10})
	t.Logf("simulator's own replay, skewed 6 seed 1 always_gpu: round %.4f lag %.4f placement %v", skR.RoundS, skR.StragglerLagS, skR.ShardPlacement)

	t.Run("AgreesWithTheSimulatorsOwnReplay", func(t *testing.T) {
		dir := vdWrite(t, vdTimeline(sk1, skR, 11, 1, 0))
		out, code := vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-strict")
		row, ok := vdParseRow(t, out, 11)
		if !ok {
			t.Fatalf("no row for round 11")
		}
		if code != 0 || row.verdict != "agree" {
			t.Errorf("verdict %q exit %d, want agree / 0 with -strict (decision 28)", row.verdict, code)
		}
		if math.Abs(row.simS-skR.RoundS) > 0.006 {
			t.Errorf("comparer's sim_s %.2f, our sim.Run %.4f: it does not replay with the documented defaults", row.simS, skR.RoundS)
		} else {
			t.Logf("ok: comparer sim_s %.2f = sim.Run %.4f; verdict %q", row.simS, skR.RoundS, row.verdict)
		}
	})

	t.Run("ReplaysReactiveHighDimLikeSimRun", func(t *testing.T) {
		// A policy- and phase-sensitive case: the comparer's default-phase
		// replay must be exactly sim.Run with the documented cadences, pool
		// from the preset and sharing off; and the step 7 results row E says
		// 93.27, 105.13, 95.01.
		want := map[int64]float64{1: 93.27, 2: 105.13, 3: 95.01}
		for seed := int64(1); seed <= 3; seed++ {
			w := vdWorkload(t, "high_dim", 6, seed)
			r := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10})
			dir := vdWrite(t, vdTimeline(w, r, 20+seed, 1, 0))
			out, _ := vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false")
			row, ok := vdParseRow(t, out, 20+seed)
			if !ok {
				t.Errorf("seed %d: no row", seed)
				continue
			}
			t.Logf("seed %d: sim.Run %.4f (preempts %d), comparer sim_s %.2f, doc row E %.2f, envelope %.2f..%.2f", seed, r.RoundS, r.Preempts, row.simS, want[seed], row.envLo, row.envHi)
			if math.Abs(row.simS-r.RoundS) > 0.006 {
				t.Errorf("seed %d: comparer sim_s %.2f != sim.Run %.4f", seed, row.simS, r.RoundS)
			}
			if math.Abs(r.RoundS-want[seed]) > 0.006 {
				t.Errorf("seed %d: sim.Run %.4f != the results table's %.2f (decision 29 says no constant changed)", seed, r.RoundS, want[seed])
			}
		}
	})

	t.Run("WorkerMemFlagIsHonoured", func(t *testing.T) {
		// -h: -worker_mem_gb is the live WORKER_MEM_BYTES in GiB, 0 = the
		// preset's. high_dim's preset memory refuses some builds (decision 25),
		// 24 GiB refuses none, so the replay must differ accordingly.
		w := vdWorkload(t, "high_dim", 6, 1)
		pre := vdReplay(t, w, "reactive", sim.Config{LeaseS: 10})
		live := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10})
		t.Logf("sim.Run high_dim seed 1 reactive: preset memory %d B -> %.2f (g/l %d/%d); 24 GiB -> %.2f (g/l %d/%d)",
			w.Scenario.Pool.WorkerMemBytes, pre.RoundS, pre.BuildsGPU, pre.BuildsLocal, live.RoundS, live.BuildsGPU, live.BuildsLocal)
		dir := vdWrite(t, vdTimeline(w, live, 25, 1, 0))
		out, _ := vdRun(t, "-timelines", dir, "-policy", "reactive", "-shards=false")
		row, _ := vdParseRow(t, out, 25)
		if math.Abs(row.simS-pre.RoundS) > 0.006 {
			t.Errorf("no -worker_mem_gb: sim_s %.2f, want the preset-memory replay %.2f", row.simS, pre.RoundS)
		}
		out, _ = vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false")
		row, _ = vdParseRow(t, out, 25)
		if math.Abs(row.simS-live.RoundS) > 0.006 || vdVerdict(row.verdict) != "agree" {
			t.Errorf("-worker_mem_gb 24: sim_s %.2f verdict %q, want %.2f agree", row.simS, row.verdict, live.RoundS)
		} else {
			t.Logf("ok: the flag selects the memory; default = preset's")
		}
	})

	t.Run("RowEAndHMemory", func(t *testing.T) {
		// First run (F2): decision 28 listed E as "transfer and memory bind",
		// but E's numbers reproduce only at the live 24 GiB, where no high_dim
		// build exceeds a worker. The doc was reworded and configuration H
		// added: high_dim at the preset's own 1.13 GB workers, where the
		// two largest builds fit no worker. Pin both halves of the corrected
		// claim with the cost model's memory need per build.
		m := costmodel.Default()
		for seed := int64(1); seed <= 3; seed++ {
			w := vdWorkload(t, "high_dim", 6, seed)
			refused, presetRefused := 0, 0
			var need []string
			for _, s := range w.Shards {
				b := m.GPUMemBytes(s.NVectors, w.Scenario.Dim)
				need = append(need, fmt.Sprintf("%d:%.2fGB", s.NVectors, float64(b)/1e9))
				if b > vdLiveMem {
					refused++
				}
				if b > w.Scenario.Pool.WorkerMemBytes {
					presetRefused++
				}
			}
			t.Logf("seed %d: memory needs %v; refused at 24 GiB (E): %d; refused at the preset's %.3f GB (H): %d", seed, need, refused, float64(w.Scenario.Pool.WorkerMemBytes)/1e9, presetRefused)
			if refused != 0 {
				t.Errorf("seed %d: decision 28 now says E refuses nothing at 24 GiB, but %d builds exceed it", seed, refused)
			}
			if presetRefused != 2 {
				t.Errorf("seed %d: decision 28 says H's two largest builds fit no worker at the preset's memory, got %d refused", seed, presetRefused)
			}
		}
	})

	t.Run("ResultsTableReproduces", func(t *testing.T) {
		// The step 7 results table's "Simulated" column, rerun from the
		// seeds: A, C, D, F, G.
		cases := []struct {
			cfg, sc, pol string
			n            int
			want         []float64
		}{
			{"A", "skewed", "always_gpu", 6, []float64{30.00, 29.99, 30.08}},
			{"C", "skewed", "always_gpu", 50, []float64{48.56, 49.84, 48.22}},
			{"D", "skewed", "reactive", 50, []float64{30.17, 30.09, 30.13}},
			{"F", "hour_build", "always_gpu", 6, []float64{438.20}},
			{"G", "hour_build", "reactive", 6, []float64{438.20}},
		}
		for _, c := range cases {
			for i, want := range c.want {
				seed := int64(i + 1)
				r := vdReplay(t, vdWorkload(t, c.sc, c.n, seed), c.pol, sim.Config{LeaseS: 10})
				t.Logf("%s %s %d %s seed %d: sim %.4f (table %.2f) gpu/local %d/%d preempts %d lag %.2f", c.cfg, c.sc, c.n, c.pol, seed, r.RoundS, want, r.BuildsGPU, r.BuildsLocal, r.Preempts, r.StragglerLagS)
				if math.Abs(r.RoundS-want) > 0.006 {
					t.Errorf("%s seed %d: simulated %.4f, the results table says %.2f", c.cfg, seed, r.RoundS, want)
				}
			}
		}
	})

	t.Run("RefusesWorkloadMismatch", func(t *testing.T) {
		// Decision 28: "refuses to continue if any shard's size differs".
		cases := []struct {
			name string
			edit func(tl *timelineT)
		}{
			{"one shard's n_vectors + 1", func(tl *timelineT) { tl.Shards[3].NVectors++ }},
			{"a shard missing", func(tl *timelineT) { tl.Shards = tl.Shards[:5] }},
			{"seed says 2, sizes are seed 1's", func(tl *timelineT) { tl.Seed = 2 }},
			{"scenario says hour_build", func(tl *timelineT) { tl.Scenario = "hour_build" }},
			{"two shards' sizes swapped", func(tl *timelineT) {
				tl.Shards[0].NVectors, tl.Shards[1].NVectors = tl.Shards[1].NVectors, tl.Shards[0].NVectors
			}},
			{"unknown scenario", func(tl *timelineT) { tl.Scenario = "no_such_preset" }},
		}
		for _, c := range cases {
			tl := vdTimeline(sk1, skR, 31, 1, 0)
			c.edit(&tl)
			dir := vdWrite(t, tl)
			out, code := vdRun(t, "-timelines", dir, "-policy", "always_gpu")
			row, hasRow := vdParseRow(t, out, 31)
			if code == 0 || (hasRow && strings.HasPrefix(row.verdict, "agree")) {
				t.Errorf("%s: exit %d, row %v %+v: the comparer did not refuse (decision 28)", c.name, code, hasRow, row)
			} else {
				t.Logf("ok: %s -> refused, exit %d", c.name, code)
			}
		}
	})

	t.Run("ScalesCadencesByTimeScale", func(t *testing.T) {
		// -h: poll_s/reconsider_s/lease_s are WALL seconds; decision 28: the
		// replay's cadences are those x the file's time_scale. A live round
		// recorded at scale 10 with wall cadences 0.05/0.2/1 is modeled at
		// 0.5/2/10, the defaults: the replay must be the default one. With
		// wall cadences 0.5/2/10 at scale 10 the replay must be sim.Run at
		// 5/20/100. high_dim seed 2 reactive is cadence-sensitive.
		w := vdWorkload(t, "high_dim", 6, 2)
		def := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10})
		slow := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, PollS: 5, ReconsiderS: 20, LeaseS: 100})
		t.Logf("sim.Run: default cadences %.4f; cadences x10 (poll 5, reconsider 20, lease 100) %.4f", def.RoundS, slow.RoundS)
		if math.Abs(def.RoundS-slow.RoundS) < 0.05 {
			t.Fatalf("test needs a cadence-sensitive case; both %.4f", def.RoundS)
		}
		dir := vdWrite(t, vdTimeline(w, def, 41, 10, 0))
		out, _ := vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-poll_s", "0.05", "-reconsider_s", "0.2", "-lease_s", "1", "-shards=false")
		row, ok := vdParseRow(t, out, 41)
		if !ok {
			t.Fatalf("no row")
		}
		if math.Abs(row.simS-def.RoundS) > 0.006 || math.Abs(row.liveS-def.RoundS) > 0.006 || row.verdict != "agree" {
			t.Errorf("scale 10, wall 0.05/0.2/1: live_s %.2f sim_s %.2f verdict %q; want both %.2f (modeled) and agree", row.liveS, row.simS, row.verdict, def.RoundS)
		} else {
			t.Logf("ok: scale 10 with wall cadences 0.05/0.2/1 replays the default %.2f; live_s printed in modeled seconds", row.simS)
		}
		out, _ = vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false")
		row, _ = vdParseRow(t, out, 41)
		if math.Abs(row.simS-slow.RoundS) > 0.006 {
			t.Errorf("scale 10, wall 0.5/2/10: sim_s %.2f; want sim.Run at modeled 5/20/100 = %.4f", row.simS, slow.RoundS)
		} else {
			t.Logf("ok: scale 10 with wall 0.5/2/10 replays modeled 5/20/100: %.2f", row.simS)
		}
	})

	t.Run("ToleranceIsMaxOf2sAnd5Percent", func(t *testing.T) {
		// Decision 28: agree within max(2 s, 5 %) of the simulated round.
		hb := vdWorkload(t, "hour_build", 6, 1)
		hbR := vdReplay(t, hb, "always_gpu", sim.Config{LeaseS: 10})
		cases := []struct {
			w     workload.Workload
			r     sim.Result
			delta float64
			want  string
			why   string
		}{
			{sk1, skR, 1.9, "agree", "30 s round: 5% = 1.5 < 2 s, so 2 s applies"},
			{sk1, skR, -1.9, "agree", "symmetric"},
			{sk1, skR, 2.2, "disagree", "beyond 2 s"},
			{sk1, skR, -2.2, "disagree", "beyond 2 s, live faster"},
			{hb, hbR, 21.0, "agree", "438 s round: 5% = 21.9 s > 2 s"},
			{hb, hbR, -21.0, "agree", "symmetric"},
			{hb, hbR, 23.0, "disagree", "beyond 5%"},
		}
		for i, c := range cases {
			id := int64(50 + i)
			dir := vdWrite(t, vdTimeline(c.w, c.r, id, 1, c.delta))
			out, code := vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-shards=false")
			row, ok := vdParseRow(t, out, id)
			got := vdVerdict(row.verdict)
			if !ok || got != c.want || code != 0 {
				t.Errorf("%s round %.2f%+.1f: verdict %q (exit %d), want %s (%s)", c.w.Scenario.Name, c.r.RoundS, c.delta, row.verdict, code, c.want, c.why)
			} else {
				t.Logf("ok: %s round %.2f%+.1f -> %q (%s)", c.w.Scenario.Name, c.r.RoundS, c.delta, row.verdict, c.why)
			}
		}
		// The flags move the band.
		dir := vdWrite(t, vdTimeline(sk1, skR, 60, 1, 1.0))
		out, _ := vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-tol_s", "0.5", "-tol_frac", "0.01", "-shards=false")
		if row, _ := vdParseRow(t, out, 60); vdVerdict(row.verdict) != "disagree" {
			t.Errorf("-tol_s 0.5 -tol_frac 0.01 with delta 1.0 s on 30 s: verdict %q, want DISAGREE", row.verdict)
		}
		out, _ = vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-tol_s", "0.5", "-tol_frac", "0.05", "-shards=false")
		if row, _ := vdParseRow(t, out, 60); vdVerdict(row.verdict) != "agree" {
			t.Errorf("-tol_s 0.5 -tol_frac 0.05 (1.5 s) with delta 1.0 s: verdict %q, want agree", row.verdict)
		}
	})

	t.Run("ExitCodeIsZeroOnDisagreementUnlessStrict", func(t *testing.T) {
		// Decision 28: "a disagreement is a finding to triage, never a failed
		// run"; -h: -strict exits 1 if any round disagrees.
		dir := vdWrite(t, vdTimeline(sk1, skR, 70, 1, 0), vdTimeline(sk1, skR, 71, 1, 10))
		out, code := vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-shards=false")
		if code != 0 {
			t.Errorf("one of two rounds disagrees, no -strict: exit %d, want 0", code)
		}
		if !strings.Contains(out, "1/2") {
			t.Errorf("summary does not say 1/2 rounds agree")
		}
		_, code = vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-shards=false", "-strict")
		if code != 1 {
			t.Errorf("-strict with a disagreement: exit %d, want 1", code)
		} else {
			t.Logf("ok: exit 0 without -strict, 1 with it")
		}
	})

	t.Run("PlacementDifferenceAloneDoesNotDisagree", func(t *testing.T) {
		// Decision 28 defines agreement on round time only; the g/l split is
		// printed side by side. A live round with one shard placed locally
		// but the same round time must still agree (D's rows: 8/42 live vs
		// 7/43 sim, "agree").
		tl := vdTimeline(sk1, skR, 80, 1, 0)
		tl.Shards[4].Placement = "local"
		tl.BuildsGPU, tl.BuildsLocal = 5, 1
		dir := vdWrite(t, tl)
		out, _ := vdRun(t, "-timelines", dir, "-policy", "always_gpu", "-shards=false")
		row, _ := vdParseRow(t, out, 80)
		if vdVerdict(row.verdict) != "agree" {
			t.Errorf("verdict %q; decision 28 defines agreement on round time only", row.verdict)
		}
		if !strings.Contains(out, "5/1") {
			t.Errorf("the live g/l split 5/1 is not printed")
		}
	})

	t.Run("EnvelopeContainsDefaultReplay", func(t *testing.T) {
		// Decision 29: the comparer replays eight phases and prints min..max;
		// the default phase is one of the phases a live ticker could have, so
		// the envelope must contain the default replay. Also compare with our
		// own eight-phase grid (2k/8, k=1..8) and the doc's numbers for row E.
		docEnv := map[int64][2]float64{1: {89.8, 93.8}, 2: {90.6, 105.6}, 3: {92.0, 104.8}}
		for seed := int64(1); seed <= 3; seed++ {
			w := vdWorkload(t, "high_dim", 6, seed)
			def := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10})
			lo, hi := math.Inf(1), math.Inf(-1)
			var grid []string
			for k := 1; k <= 8; k++ {
				r := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10, TickPhaseS: 2 * float64(k) / 8})
				lo, hi = math.Min(lo, r.RoundS), math.Max(hi, r.RoundS)
				grid = append(grid, fmt.Sprintf("%.2f", r.RoundS))
			}
			dir := vdWrite(t, vdTimeline(w, def, 90+seed, 1, 0))
			out, _ := vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false")
			row, ok := vdParseRow(t, out, 90+seed)
			t.Logf("seed %d: default %.2f; our 8-phase grid %v -> %.2f..%.2f; comparer envelope %.2f..%.2f; doc %.1f..%.1f",
				seed, def.RoundS, grid, lo, hi, row.envLo, row.envHi, docEnv[seed][0], docEnv[seed][1])
			if !ok || row.envLo > def.RoundS+0.005 || row.envHi < def.RoundS-0.005 {
				t.Errorf("seed %d: envelope %.2f..%.2f does not contain the default replay %.2f", seed, row.envLo, row.envHi, def.RoundS)
			}
			if math.Abs(row.envLo-docEnv[seed][0]) > 0.06 || math.Abs(row.envHi-docEnv[seed][1]) > 0.06 {
				t.Errorf("seed %d: comparer envelope %.2f..%.2f differs from the results table's %.1f..%.1f", seed, row.envLo, row.envHi, docEnv[seed][0], docEnv[seed][1])
			}
		}
	})

	t.Run("ProbesWhereTheSpecIsSilent", func(t *testing.T) {
		// Not promises: behaviour recorded for the G-items of the report.
		// (1) Shards listed out of id order, each with its own id's size.
		tl := vdTimeline(sk1, skR, 100, 1, 0)
		for i, j := 0, len(tl.Shards)-1; i < j; i, j = i+1, j-1 {
			tl.Shards[i], tl.Shards[j] = tl.Shards[j], tl.Shards[i]
		}
		out, code := vdRun(t, "-timelines", vdWrite(t, tl), "-policy", "always_gpu", "-shards=false")
		row, _ := vdParseRow(t, out, 100)
		t.Logf("probe 1, shards in reverse order (sizes keyed by shard_id correct): exit %d verdict %q — the comparer matches by POSITION, and its message names the position as the shard (\"shard 0 has 100000\" is shard_id 5)", code, row.verdict)
		// (2) No -policy at all.
		out, code = vdRun(t, "-timelines", vdWrite(t, vdTimeline(sk1, skR, 101, 1, 0)), "-shards=false")
		t.Logf("probe 2, -policy omitted: exit %d, header %q", code, strings.SplitN(out, "\n", 2)[0])
		// (3) An empty directory.
		out, code = vdRun(t, "-timelines", t.TempDir(), "-policy", "always_gpu")
		t.Logf("probe 3, empty timeline directory: exit %d, output %q", code, strings.TrimSpace(out))
		// (4) Bug 15 at the comparer: a round that ran under reactive (hour_build,
		// whose straggler sets the round under any policy) compared with
		// -policy always_gpu. The Timeline has no policy field, so the comparer
		// can only trust the flag.
		w := vdWorkload(t, "hour_build", 6, 1)
		re := vdReplay(t, w, "reactive", sim.Config{LeaseS: 10})
		out, code = vdRun(t, "-timelines", vdWrite(t, vdTimeline(w, re, 102, 1, 0)), "-policy", "always_gpu", "-shards=false", "-strict")
		row, _ = vdParseRow(t, out, 102)
		t.Logf("probe 4, reactive round (preempts %d) compared as always_gpu: exit %d verdict %q — the comparer cannot tell the policy that ran; only the runner's pod-log check (decision 30) can", re.Preempts, code, row.verdict)
	})

	t.Run("InsideEnvelopeOutsideToleranceVerdict", func(t *testing.T) {
		// Decision 29: outside the default replay's tolerance but inside the
		// envelope is "agree within tick-phase envelope". high_dim seed 2:
		// default 105.13, doc envelope 90.6..105.6; live 91.58 (row E).
		w := vdWorkload(t, "high_dim", 6, 2)
		def := vdReplay(t, w, "reactive", sim.Config{WorkerMemBytes: vdLiveMem, LeaseS: 10})
		dir := vdWrite(t, vdTimeline(w, def, 95, 1, 91.58-def.RoundS))
		out, code := vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false", "-strict")
		row, _ := vdParseRow(t, out, 95)
		if !strings.Contains(row.verdict, "envelope") || strings.Contains(row.verdict, "DISAGREE") {
			t.Errorf("live 91.58 vs default %.2f inside envelope %.2f..%.2f: verdict %q, want 'agree within tick-phase envelope'", def.RoundS, row.envLo, row.envHi, row.verdict)
		}
		t.Logf("verdict %q; -strict exit %d (whether an envelope-only agreement passes -strict is not stated: G-item)", row.verdict, code)
		// Far outside the envelope too: DISAGREE.
		dir = vdWrite(t, vdTimeline(w, def, 96, 1, 80-def.RoundS))
		out, _ = vdRun(t, "-timelines", dir, "-policy", "reactive", "-worker_mem_gb", "24", "-shards=false")
		if row, _ := vdParseRow(t, out, 96); vdVerdict(row.verdict) != "disagree" {
			t.Errorf("live 80 s, far below the envelope: verdict %q, want DISAGREE", row.verdict)
		}
	})
}
