"""The worker's Prometheus metrics (docs/roadmap.md, Phase 2, Metrics).
Served by the health server at /metrics on the same port as the probes.

    build_duration_seconds{placement="gpu"}  wall time of one build on this worker
                                             (real seconds; at FAKE_TIME_SCALE=10 a 65 s
                                             modeled build is observed as 6.5 s)
    worker_busy                              1 while a build is in progress
    builds_total{outcome}                    done | failed | released | lost
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

BUILD_DURATION = Histogram(
    "build_duration_seconds", "Wall time of one index build, by placement.",
    ["placement"], buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
)
WORKER_BUSY = Gauge("worker_busy", "1 while this worker is building, else 0.")
BUILDS_TOTAL = Counter("builds_total", "Builds this worker finished with, by outcome.", ["outcome"])

for _o in ("done", "failed", "released", "lost"):
    BUILDS_TOTAL.labels(outcome=_o)   # expose every series from the start
BUILD_DURATION.labels(placement="gpu")
