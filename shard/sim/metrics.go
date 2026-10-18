package sim

// The shard side's Prometheus metrics (docs/roadmap.md, Phase 2, Metrics).
// The shards run outside the cluster (Phase 2 decision 26), so the simulator
// serves these itself (shardsim's METRICS_ADDR) and Prometheus is pointed at
// the laptop (decision 28). Real seconds throughout: at TIME_SCALE=10 a 20 s
// modeled local build is observed as 2 s.

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	buildDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "build_duration_seconds",
		Help:    "Wall time of one index build, by placement (local builds are measured here).",
		Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800},
	}, []string{"placement"})
	localBuildsAborted = promauto.NewCounter(prometheus.CounterOpts{
		Name: "local_builds_aborted_total",
		Help: "Local builds killed by a preemption to the GPU queue; their CPU progress is lost.",
	})
	queryWait = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "query_wait_seconds",
		Help:    "Per query: time from arrival at the shard to the start of its run, by whether it needed the index.",
		Buckets: prometheus.ExponentialBuckets(0.01, 2, 14), // 10 ms .. ~80 s
	}, []string{"needs_index"})
	queriesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "queries_total",
		Help: "Queries run on the shards' CPUs, by kind.",
	}, []string{"kind"})
	shardQueueDepth = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "shard_queue_depth",
		Help: "Queries waiting on a shard's CPU (set at every report beat).",
	}, []string{"shard"})
	shardsActive = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "shards_active",
		Help: "Shards whose round is in progress.",
	})
)
