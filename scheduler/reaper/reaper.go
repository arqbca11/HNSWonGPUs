// Package reaper is the scheduler's periodic sweep: expired leases go back
// to the queue, and rounds whose work is all done get their finished_at.
// Both statements are idempotent, so two schedulers running this is safe.
package reaper

import (
	"context"
	"log/slog"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/metrics"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

// Config for the sweep.
type Config struct {
	Interval    time.Duration
	MaxAttempts int32         // a build whose lease expires on this attempt is failed, not requeued; <= 0 disables
	StaleAfter  time.Duration // for the live-workers gauge; the API's WorkerStaleAfter
}

// Run ticks until ctx is cancelled. It never returns an error: a failed tick
// is logged and the next tick tries again.
func Run(ctx context.Context, st *store.Store, cfg Config, log *slog.Logger) {
	t := time.NewTicker(cfg.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			Tick(ctx, st, cfg.MaxAttempts, cfg.StaleAfter, log)
		}
	}
}

// Tick runs one sweep. Exposed so tests and the API can run it on demand.
func Tick(ctx context.Context, st *store.Store, maxAttempts int32, staleAfter time.Duration, log *slog.Logger) {
	if staleAfter <= 0 {
		staleAfter = 30 * time.Second
	}
	reaped, err := st.Reap(ctx, maxAttempts)
	if err != nil {
		log.Error("reap failed", "err", err)
	}
	for _, r := range reaped {
		metrics.LeaseExpirations.WithLabelValues(r.Outcome).Inc()
		if r.Outcome == "failed" {
			log.Error("lease expired, retry budget exhausted, build failed",
				"build_id", r.BuildID, "round_id", r.RoundID, "attempt", r.Attempt, "lease_owner", r.LeaseOwner,
				"max_attempts", maxAttempts)
			continue
		}
		log.Warn("lease expired, build back to queued",
			"build_id", r.BuildID, "round_id", r.RoundID, "attempt", r.Attempt, "lease_owner", r.LeaseOwner)
	}
	finished, err := st.FinishCompleteRounds(ctx)
	if err != nil {
		log.Error("finish rounds failed", "err", err)
	}
	metrics.ObserveFinishedRounds(finished)
	for _, fr := range finished {
		log.Info("round finished", "round_id", fr.RoundID, "duration", fr.Duration.Round(time.Millisecond),
			"straggler_lag", fr.StragglerLag.Round(time.Millisecond), "gpu_builds_claimed", len(fr.QueueWaits))
	}
	if q, err := st.Stats(ctx); err == nil {
		metrics.GPUQueueDepth.Set(float64(q.GPUQueued))
		metrics.BuildsLeased.Set(float64(q.Leased))
		metrics.RoundsOpen.Set(float64(q.RoundsOpen))
	} else {
		log.Error("stats failed", "err", err)
	}
	if p, err := st.Pool(ctx, staleAfter); err == nil {
		metrics.GPUWorkersLive.Set(float64(p.LiveWorkers))
	}
}
