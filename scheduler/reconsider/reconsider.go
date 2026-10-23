// Package reconsider is the scheduler's second periodic loop (Phase 3 step
// 3), beside the reaper: on every tick it settles finished speculation races,
// reads the whole visible world from Postgres, asks the policy's Reconsider
// what to change, and executes the returned actions as guarded row updates.
//
// The division of labor is decision 3: the policy is a pure function that
// returns intentions (preempt this build, renege that one, start a copy,
// change a priority); this package is the actuator. Every write is a guarded
// UPDATE, so an action computed from a snapshot that a concurrent claim,
// completion or second scheduler has overtaken updates zero rows and is
// skipped — the next tick recomputes from the new truth. Running two
// schedulers is therefore safe, like the reaper.
//
// Time scale: shard reports, build timestamps and worker heartbeats are wall
// clock, while the cost model's durations are modeled time. Under
// FAKE_TIME_SCALE=s a worker finishes a modeled 60 s build in 6 wall seconds,
// so every wall-derived number here (a lease's elapsed time, an arrival rate,
// an oldest wait) is converted to modeled seconds with the same s before it
// meets a cost-model estimate. In production s is 1 and the conversion
// vanishes.
package reconsider

import (
	"context"
	"log/slog"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/metrics"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

type Config struct {
	Interval   time.Duration
	StaleAfter time.Duration // workers unseen for longer are not in the pool
	TimeScale  float64       // wall seconds x this = modeled seconds; <= 0 means 1
}

// Run ticks until ctx is cancelled. A failed tick is logged and the next one
// tries again; like the reaper it never returns an error.
func Run(ctx context.Context, st *store.Store, pol policy.Policy, cm costmodel.Model, cfg Config, log *slog.Logger) {
	t := time.NewTicker(cfg.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			Tick(ctx, st, pol, cm, cfg, log)
		}
	}
}

// Tick runs one reconsider pass. Exposed so tests can drive it on demand.
func Tick(ctx context.Context, st *store.Store, pol policy.Policy, cm costmodel.Model, cfg Config, log *slog.Logger) {
	// Settle races first: a copy that already finished (or lost) changes what
	// the policy should see this tick.
	resolved, err := st.ResolveSpeculations(ctx)
	if err != nil {
		log.Error("resolve speculations failed", "err", err)
	}
	for _, r := range resolved {
		metrics.SpeculationsResolved.WithLabelValues(r.Winner).Inc()
		if r.Winner == "gpu" {
			log.Info("speculation resolved: the gpu copy won; the local build will be aborted",
				"build_id", r.BuildID, "round_id", r.RoundID, "attempt", 0, "winner", r.Winner)
		} else {
			log.Info("speculation resolved: the local build won; the copy was cancelled",
				"build_id", r.BuildID, "round_id", r.RoundID, "attempt", 0, "winner", r.Winner,
				"copy_state_at_cancel", r.CopyState, "gpu_work_wasted", r.CopyState == "leased")
		}
	}

	snap, err := st.GetReconsiderSnapshot(ctx, cfg.StaleAfter)
	if err != nil {
		log.Error("reconsider snapshot failed", "err", err)
		return
	}
	if len(snap.Builds) == 0 {
		return
	}
	in := buildInput(snap, cm, cfg.TimeScale)
	acts := pol.Reconsider(in)
	if len(acts) == 0 {
		return
	}

	byID := map[string]store.ReconsiderBuild{}
	for _, b := range snap.Builds {
		byID[b.BuildID] = b
	}
	// Preempts first (step 1 triage, G7): a same-tick speculation judged the
	// queue empty on the tick's input, and applying the preempts first keeps
	// the actual claim order consistent with what the policy assumed.
	for _, pass := range [][]policy.ActionKind{
		{policy.ActPreempt},
		{policy.ActRenege, policy.ActReprioritize, policy.ActSpeculate},
	} {
		for _, a := range acts {
			for _, kind := range pass {
				if a.Kind == kind {
					execute(ctx, st, a, byID[a.BuildID], log)
				}
			}
		}
	}
}

// execute applies one action as its guarded store write. Zero rows updated
// means the row moved on (claimed, completed, already moved) between the
// snapshot and now: the action is skipped and the next tick recomputes.
func execute(ctx context.Context, st *store.Store, a policy.Action, b store.ReconsiderBuild, log *slog.Logger) {
	var applied bool
	var err error
	switch a.Kind {
	case policy.ActPreempt:
		applied, err = st.PreemptToGPU(ctx, a.BuildID, a.Priority)
	case policy.ActRenege:
		applied, err = st.RenegeToLocal(ctx, a.BuildID)
	case policy.ActReprioritize:
		applied, err = st.Reprioritize(ctx, a.BuildID, a.Priority)
	case policy.ActSpeculate:
		applied, err = st.SpeculateCopy(ctx, a.BuildID, a.Priority)
	default:
		log.Error("unknown reconsider action", "action", string(a.Kind), "build_id", a.BuildID)
		return
	}
	if err != nil {
		log.Error("reconsider action failed", "action", string(a.Kind),
			"build_id", a.BuildID, "round_id", b.RoundID, "attempt", b.Attempt, "err", err)
		return
	}
	outcome := "applied"
	if !applied {
		outcome = "skipped"
	}
	metrics.ReconsiderActions.WithLabelValues(string(a.Kind), outcome).Inc()
	log.Info("reconsider "+string(a.Kind)+" "+outcome,
		"build_id", a.BuildID, "round_id", b.RoundID, "attempt", b.Attempt,
		"priority", a.Priority, "reason", a.Reason)
}

// buildInput turns the store snapshot into the policy's pure view. The host
// computes every number a policy sees (decision 9): cost-model estimates per
// build, the queue replay's waits, and the per-shard aggregates.
func buildInput(snap store.ReconsiderSnapshot, cm costmodel.Model, timeScale float64) policy.ReconsiderInput {
	scale := timeScale
	if scale <= 0 {
		scale = 1
	}

	// A leased build occupies its worker for its modeled remaining time:
	// modeled total minus modeled elapsed (wall x scale), floored at 0 (an
	// overdue lease means "about to free or be reaped" — soon either way).
	busy := map[string]float64{}
	for _, b := range snap.Builds {
		if b.State != "leased" || b.LeaseOwner == "" {
			continue
		}
		rem := cm.GPUTotal(b.NVectors, b.Dim).Seconds() - b.ElapsedS*scale
		if rem < 0 {
			rem = 0
		}
		if rem > busy[b.LeaseOwner] {
			busy[b.LeaseOwner] = rem
		}
	}
	workers := make([]policy.WorkerView, 0, len(snap.Workers))
	for _, w := range snap.Workers {
		workers = append(workers, policy.WorkerView{ID: w.WorkerID, Node: w.Node, MemBytes: w.MemBytes, BusyRemainingS: busy[w.WorkerID]})
	}

	// The queued jobs in claim order (the snapshot's order), for the replay.
	var queue []policy.QueuedJob
	for _, b := range snap.Builds {
		if b.State != "queued" {
			continue
		}
		queue = append(queue, policy.QueuedJob{
			BuildID: b.BuildID, Priority: b.Priority, Seq: len(queue), MemBytes: b.MemBytes,
			GPUBuildS: cm.GPUBuild(b.NVectors, b.Dim).Seconds(),
			TransferS: cm.Transfer(b.NVectors, b.Dim).Seconds(),
		})
	}

	// The queue replay (step 2). The cold-start rule (G1) applies on every
	// tick exactly as at submit: an empty pool may be about to fill, so every
	// wait reads 0 and nothing stampedes back to local — a SHRUNKEN pool (one
	// worker left of four) produces real, longer waits, and reneging reacts
	// to those. A build whose memory fits no live worker still reads +Inf,
	// which is what lets reneging rescue it (bug 11's cure, this time at the
	// third place placement changes).
	emptyPool := len(workers) == 0
	var est map[string]policy.JobEstimate
	if !emptyPool {
		est = policy.ReplayQueue(workers, queue)
	}

	status := map[[2]int64]policy.ShardStatus{}
	for _, st := range snap.Status {
		rate := 0.0
		if st.RoundElapsedS > 0 {
			// arrivals per modeled second: wall elapsed x scale is the round's
			// modeled age. Recorded durations are already modeled (the shard
			// reports duration_ms unscaled).
			rate = float64(st.Arrivals) / (st.RoundElapsedS * scale)
		}
		status[[2]int64{st.RoundID, int64(st.ShardID)}] = policy.ShardStatus{
			QueueDepth:        st.QueueDepth,
			WaitingNeedsIndex: st.WaitingNeedsIndex,
			OldestWaitS:       float64(st.OldestWaitMs) / 1000 * scale,
			BuildProgress:     float64(st.BuildProgress),
			ArrivalRateQPS:    rate,
			MeanQueryS:        st.MeanQueryMs / 1000,
			Valid:             true,
		}
	}

	in := policy.ReconsiderInput{Workers: workers}
	for _, b := range snap.Builds {
		v := policy.BuildView{
			BuildID: b.BuildID,
			Spec: policy.BuildSpec{ShardID: b.ShardID, RoundID: b.RoundID,
				NVectors: b.NVectors, Dim: b.Dim, MemBytes: b.MemBytes},
			Est: policy.Estimates{
				LocalBuildS: cm.CPUBuild(b.NVectors, b.Dim).Seconds(),
				GPUBuildS:   cm.GPUBuild(b.NVectors, b.Dim).Seconds(),
				TransferS:   cm.Transfer(b.NVectors, b.Dim).Seconds(),
			},
			Priority:    b.Priority,
			Moved:       b.Moved,
			Speculative: b.Speculative,
			Status:      status[[2]int64{b.RoundID, int64(b.ShardID)}],
		}
		switch {
		case b.Placement == "local" && b.State == "running":
			v.Phase = "local_running"
			if !emptyPool {
				// The wait this build would join at if preempted now (G2). It
				// joins at the back like any new job (decision 13): an upper
				// bound, since a preempt would grant it the backlog's priority.
				v.QueueWaitS = policy.NewJobWait(workers, queue, policy.QueuedJob{
					BuildID: b.BuildID, MemBytes: b.MemBytes,
					GPUBuildS: v.Est.GPUBuildS, TransferS: v.Est.TransferS,
				})
			}
		case b.State == "queued":
			v.Phase = "gpu_queued"
			if !emptyPool {
				v.QueueWaitS = est[b.BuildID].StartS
			}
		case b.State == "leased":
			v.Phase = "gpu_leased"
		}
		in.Builds = append(in.Builds, v)
	}
	return in
}
