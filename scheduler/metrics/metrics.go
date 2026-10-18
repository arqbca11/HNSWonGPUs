// Package metrics holds the scheduler's Prometheus metrics. Names follow
// docs/roadmap.md (Phase 2, Metrics). Everything is registered on the default
// registry and served by promhttp at /metrics.
//
// Where each one is measured:
//
//	round_duration_seconds, shard_lag_seconds, build_queue_wait_seconds
//	    when the reaper's tick finishes a round (store.FinishCompleteRounds):
//	    the scheduler learns that a round is over only then, and the per-shard
//	    finish times it needs are all in Postgres by then.
//	placement_decisions_total      at POST /builds, when the policy decides
//	lease_expirations_total        in the reaper, per outcome (requeued, failed)
//	gpu_queue_depth, gpu_workers_live, builds_leased
//	    gauges refreshed every reaper tick from one query each
package metrics

import (
	"github.com/arqbca11/KubernetesGPU/scheduler/store"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RoundDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "round_duration_seconds",
		Help:    "Wall time from a round's creation to its last shard finishing (real seconds, not modeled).",
		Buckets: prometheus.ExponentialBuckets(1, 2, 12), // 1 s .. ~34 min
	})
	ShardLag = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "shard_lag_seconds",
		Help:    "Per round: the last shard's finish minus the median shard's finish. The straggler effect.",
		Buckets: prometheus.ExponentialBuckets(0.25, 2, 12),
	})
	BuildQueueWait = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "build_queue_wait_seconds",
		Help:    "Per GPU build: time from enqueue to first claim. Observed when its round finishes.",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 12),
	}, []string{"placement"})
	PlacementDecisions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "placement_decisions_total",
		Help: "Placement decisions made at submit, by placement and policy.",
	}, []string{"placement", "policy"})
	LeaseExpirations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "lease_expirations_total",
		Help: "Leases the reaper found expired, by outcome (queued = back in the queue, failed = retry budget exhausted).",
	}, []string{"outcome"})
	GPUQueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_queue_depth",
		Help: "GPU builds in state queued (nobody is building them).",
	})
	BuildsLeased = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "builds_leased",
		Help: "GPU builds currently leased to a worker.",
	})
	GPUWorkersLive = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_workers_live",
		Help: "Workers with a heartbeat inside WORKER_STALE_AFTER.",
	})
	RoundsOpen = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "rounds_open",
		Help: "Rounds created but not finished.",
	})
)

// ObserveFinishedRounds records the round histograms. Both places that stamp
// rounds finished (the reaper's tick and GET /rounds/{id}, which stamps on
// read so a polling shard never sees a stale "open") call it, so a round is
// observed exactly once wherever it was closed.
func ObserveFinishedRounds(finished []store.FinishedRound) {
	for _, fr := range finished {
		RoundDuration.Observe(fr.Duration.Seconds())
		ShardLag.Observe(fr.StragglerLag.Seconds())
		for _, w := range fr.QueueWaits {
			BuildQueueWait.WithLabelValues("gpu").Observe(w.Seconds())
		}
	}
}

// Init creates every label combination up front, so each series exists from
// the first scrape (a dashboard reading a missing counter sees a gap, not 0).
func Init(policyName string) {
	BuildQueueWait.WithLabelValues("gpu")
	for _, o := range []string{"queued", "failed"} {
		LeaseExpirations.WithLabelValues(o)
	}
	for _, p := range []string{"gpu", "local"} {
		PlacementDecisions.WithLabelValues(p, policyName)
	}
}
