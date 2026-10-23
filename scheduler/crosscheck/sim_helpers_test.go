package crosscheck

// Helpers for the Phase 3 steps 4–6 cross-check of the discrete-event
// simulator (sim_test.go) and the sweep runner (sweep_test.go). Written from
// docs/design/phase3-policies.md decisions 12–25, the roadmap's Phase 3
// section, and `go doc` of experiments/sim, experiments/workload,
// scheduler/policy and scheduler/costmodel. No implementation source was read.
//
// Every expected number is computed here from the documented cost model v0
// (costmodel package doc), NOT by calling costmodel, so a drift in either the
// model or the simulator's use of it shows up:
//
//   CPU build  = A * n * ln(n) * dim                 A = 1.4e-7
//   GPU build  = CPU / Speedup + GPUOverhead          Speedup 10, overhead 2 s
//   transfer   = (n*dim*4 + n*(dim*4 + 2*M*4)) / BW   M 16, BW 200e6 B/s
//   GPU memory = MemFactor * n*dim*4                  MemFactor 2

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

const sxEps = 1e-6 // seconds; the model goes through time.Duration (ns)

func sxCPU(n int64, dim int32) float64 {
	return 1.4e-7 * float64(n) * math.Log(float64(n)) * float64(dim)
}
func sxGPUBuild(n int64, dim int32, speedup float64) float64 {
	return sxCPU(n, dim)/speedup + 2
}
func sxXfer(n int64, dim int32) float64 {
	vec := float64(n) * float64(dim) * 4
	idx := float64(n) * (float64(dim)*4 + 2*16*4)
	return (vec + idx) / 200e6
}
func sxGPUTotal(n int64, dim int32) float64 { return sxGPUBuild(n, dim, 10) + sxXfer(n, dim) }
func sxMem(n int64, dim int32) int64        { return 2 * n * int64(dim) * 4 }

func sxDur(s float64) time.Duration { return time.Duration(math.Round(s * 1e9)) }

// sxQ is one hand-made query: arrival (s from round start), duration, needs_index.
func sxQ(arr, dur float64, idx bool) workload.Query {
	return workload.Query{Arrival: sxDur(arr), Duration: sxDur(dur), NeedsIndex: idx, Kind: workload.KindSingleShard}
}

// sxShard is one hand-made shard: DDL offset (s), size, queries.
func sxShard(ddl float64, n int64, qs ...workload.Query) workload.ShardWorkload {
	return workload.ShardWorkload{DDLOffset: sxDur(ddl), NVectors: n, Queries: qs}
}

// sxWorkload builds a hand-made round. Shard IDs and cluster query IDs are
// assigned here; the scenario's pool is what Config's zero values pick up.
func sxWorkload(t *testing.T, dim int32, workers int, mem int64, meanQ float64, shards ...workload.ShardWorkload) workload.Workload {
	t.Helper()
	var horizon time.Duration
	var lo, hi int64 = math.MaxInt64, 0
	qid := int64(1)
	for i := range shards {
		shards[i].ShardID = int32(i)
		if shards[i].NVectors < lo {
			lo = shards[i].NVectors
		}
		if shards[i].NVectors > hi {
			hi = shards[i].NVectors
		}
		for j := range shards[i].Queries {
			shards[i].Queries[j].ClusterQueryID = qid
			qid++
			if a := shards[i].Queries[j].Arrival + time.Second; a > horizon {
				horizon = a
			}
		}
	}
	if horizon == 0 {
		horizon = time.Second
	}
	sc := workload.Scenario{
		Name:    "xcheck_hand",
		NShards: len(shards),
		Dim:     dim,
		Sizes:   workload.SizeDist{Kind: "uniform", Min: lo, Max: hi},
		Stream: workload.StreamProfile{
			Horizon:           horizon,
			MeanQueryDuration: sxDur(meanQ),
			HotShard:          -1,
		},
		DDL:  workload.ArrivalPattern{Kind: "all_at_once"},
		Pool: workload.PoolConfig{Workers: workers, WorkerMemBytes: mem},
	}
	w := workload.Workload{Scenario: sc, Seed: 1, Shards: shards}
	if err := w.Validate(); err != nil {
		t.Logf("note: hand-made workload does not pass Validate (%v); running it anyway", err)
	}
	return w
}

func sxModel(speedup float64) *costmodel.Model {
	m := costmodel.Default()
	m.Speedup = speedup
	return &m
}

func sxRun(t *testing.T, label string, w workload.Workload, p policy.Policy, cfg sim.Config) sim.Result {
	t.Helper()
	r := sim.Run(w, p, cfg)
	t.Logf("%s: policy=%s completed=%v round=%.4fs lag=%.4fs gpu=%d local=%d preempts=%d reneges=%d reprio=%d spec=%d (gpu-wins %d, local-wins %d) wasted=%.4fs queueWaits=%s shardFinish=%s events=%d",
		label, p.Name(), r.Completed, r.RoundS, r.StragglerLagS, r.BuildsGPU, r.BuildsLocal,
		r.Preempts, r.Reneges, r.Reprioritizes, r.Speculations, r.SpecGPUWins, r.SpecLocalWins,
		r.WastedGPUS, sxFloats(r.QueueWaits), sxFloats(r.ShardFinishS), r.Events)
	return r
}

func sxFloats(xs []float64) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%.4f", x)
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func sxSorted(xs []float64) []float64 {
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	return c
}

func sxNear(a, b float64) bool {
	if math.IsInf(a, 1) && math.IsInf(b, 1) {
		return true
	}
	return math.Abs(a-b) <= sxEps
}

func sxExpect(t *testing.T, what string, got, want float64, spec string) {
	t.Helper()
	if !sxNear(got, want) {
		t.Errorf("%s = %.6f, want %.6f (%s)", what, got, want, spec)
		return
	}
	t.Logf("ok: %s = %.6f as computed by hand (%s)", what, got, spec)
}

func sxExpectSet(t *testing.T, what string, got, want []float64, spec string) {
	t.Helper()
	g, w := sxSorted(got), sxSorted(want)
	if len(g) != len(w) {
		t.Errorf("%s = %s, want %s (%s)", what, sxFloats(g), sxFloats(w), spec)
		return
	}
	for i := range g {
		if !sxNear(g[i], w[i]) {
			t.Errorf("%s = %s, want %s (%s)", what, sxFloats(g), sxFloats(w), spec)
			return
		}
	}
	t.Logf("ok: %s = %s as computed by hand (%s)", what, sxFloats(g), spec)
}

// nextReport is the first report instant at or after x on a k*poll grid
// (reports are unstaggered, decision 21). Equality is avoided in the tests.
func sxNextReport(x, poll float64) float64 { return math.Ceil(x/poll-1e-9) * poll }

// sxScript is a test policy: scripted decisions and actions, recording every
// input the simulator hands it. It is deterministic (a pure function of its
// inputs as far as its decisions go); recording is only for observation.
type sxScript struct {
	decide func(policy.Input) policy.Decision
	recon  func(policy.ReconsiderInput) []policy.Action

	inputs []policy.Input
	ticks  []policy.ReconsiderInput
}

func (s *sxScript) Name() string { return "xcheck_script" }
func (s *sxScript) Decide(in policy.Input) policy.Decision {
	s.inputs = append(s.inputs, in)
	return s.decide(in)
}
func (s *sxScript) Reconsider(in policy.ReconsiderInput) []policy.Action {
	cp := policy.ReconsiderInput{
		Workers: append([]policy.WorkerView(nil), in.Workers...),
		Builds:  append([]policy.BuildView(nil), in.Builds...),
	}
	s.ticks = append(s.ticks, cp)
	if s.recon == nil {
		return nil
	}
	return s.recon(in)
}

func sxLocal(policy.Input) policy.Decision {
	return policy.Decision{Placement: "local", Reason: "script"}
}

func sxGPUAt(pr map[int32]float64) func(policy.Input) policy.Decision {
	return func(in policy.Input) policy.Decision {
		if p, ok := pr[in.Build.ShardID]; ok {
			return policy.Decision{Placement: "gpu", Priority: p, Reason: "script"}
		}
		return policy.Decision{Placement: "local", Reason: "script"}
	}
}

func sxFindView(bs []policy.BuildView, shard int32, spec bool) (policy.BuildView, bool) {
	for _, b := range bs {
		if b.Spec.ShardID == shard && b.Speculative == spec {
			return b, true
		}
	}
	return policy.BuildView{}, false
}

func fmtF(x float64) string { return fmt.Sprintf("%g", x) }
