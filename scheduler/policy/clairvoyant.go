package policy

import (
	"fmt"
	"sort"
)

// QueryEvent is one future query of a shard's stream, known only to the
// simulator's workload generator.
type QueryEvent struct {
	ArriveS    float64 // seconds after the DDL
	DurS       float64
	NeedsIndex bool
}

// Clairvoyant is the oracle (decision 1a): it is constructed with every
// shard's full future query stream, computes the exact shard recovery time
// for both placements, and picks the sooner. Priority is the shard's
// would-be local finish, so the worst would-be straggler gets a GPU first
// (longest-processing-time-first). It is not deployable — a real shard
// cannot declare its future — so ByName never returns it; the simulator
// constructs it directly. The gap between it and the reactive policies is
// the measured cost of not knowing the future.
//
// Edges, pinned by the cross-check: a shard absent from the futures (or a
// nil map) is an empty stream; a recovery tie goes local (no transfer, no
// queue slot spent); an empty pool still queues, the same cold-start rule
// every deployable policy applies.
type Clairvoyant struct {
	futures map[int32][]QueryEvent // by shard, sorted by arrival at construction
}

func NewClairvoyant(futures map[int32][]QueryEvent) Clairvoyant {
	sorted := make(map[int32][]QueryEvent, len(futures))
	for shard, qs := range futures {
		cp := append([]QueryEvent(nil), qs...)
		sort.SliceStable(cp, func(i, j int) bool { return cp[i].ArriveS < cp[j].ArriveS })
		sorted[shard] = cp
	}
	return Clairvoyant{futures: sorted}
}

func (Clairvoyant) Name() string { return "clairvoyant" }

func (p Clairvoyant) Decide(in Input) Decision {
	if reason, no := fitsNowhere(in.Build, in.Pool); no {
		return Decision{Placement: "local", Reason: reason}
	}
	qs := p.futures[in.Build.ShardID]
	local := replayShard(qs, in.Est.LocalBuildS, localReady)
	gpuDone := in.Pool.QueueWaitS + in.Est.GPUBuildS + in.Est.TransferS
	gpu := replayShard(qs, gpuDone, gpuReady)
	if local <= gpu {
		return Decision{Placement: "local",
			Reason: fmt.Sprintf("oracle: local recovers at %.1fs, gpu at %.1fs", local, gpu)}
	}
	return Decision{Placement: "gpu", Priority: local,
		Reason: fmt.Sprintf("oracle: gpu recovers at %.1fs, local at %.1fs; priority = would-be local finish", gpu, local)}
}

func (Clairvoyant) Reconsider(ReconsiderInput) []Action { return nil }

// localReady: a local build holds the CPU, so every query is runnable only
// once the build is done (and it has arrived).
func localReady(q QueryEvent, buildDoneS float64) float64 {
	return maxf(q.ArriveS, buildDoneS)
}

// gpuReady: the CPU is free from the start; only index-needing queries wait
// for the GPU result. They wait without blocking the others (the model's
// skip rule).
func gpuReady(q QueryEvent, buildDoneS float64) float64 {
	if q.NeedsIndex {
		return maxf(q.ArriveS, buildDoneS)
	}
	return q.ArriveS
}

// replayShard runs the stream exactly: one CPU, queries in arrival order
// among those ready, a not-yet-ready query waiting without blocking the
// rest. Returns when the shard is done: build finished, stream exhausted,
// backlog drained.
func replayShard(qs []QueryEvent, buildDoneS float64, ready func(QueryEvent, float64) float64) float64 {
	done := buildDoneS
	ran := make([]bool, len(qs))
	cpuFree := 0.0
	for n := 0; n < len(qs); n++ {
		// The next query to run: ready earliest; among those, earliest arrival
		// (qs is arrival-sorted, so the first eligible wins ties).
		best, bestReady := -1, 0.0
		for i, q := range qs {
			if ran[i] {
				continue
			}
			r := maxf(ready(q, buildDoneS), cpuFree)
			if best == -1 || r < bestReady {
				best, bestReady = i, r
			}
		}
		cpuFree = bestReady + qs[best].DurS
		ran[best] = true
		if cpuFree > done {
			done = cpuFree
		}
	}
	return done
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
