package policy

import "fmt"

// DefaultThresholdN is the bare `static_threshold` cutoff: at the fake
// workloads' scale it sends only the largest builds to the GPU. The Phase 3
// sweep varies it; nothing is tuned to this default.
const DefaultThresholdN int64 = 200_000

// StaticThreshold is the OpenSearch-style rule: builds with N or more
// vectors go to the GPU queue, the rest build locally. Within the queue,
// bigger builds claim first (longest-first), so one skewed giant does not
// wait behind a line of mid-sized builds.
type StaticThreshold struct {
	N int64
}

// Name carries the threshold, so a logged or metric-labelled name tells the
// sweep's arms apart and ByName(p.Name()) reproduces the policy exactly.
func (p StaticThreshold) Name() string { return fmt.Sprintf("static_threshold:%d", p.N) }

func (p StaticThreshold) Decide(in Input) Decision {
	if reason, no := fitsNowhere(in.Build, in.Pool); no {
		return Decision{Placement: "local", Reason: reason}
	}
	if in.Build.NVectors >= p.N {
		return Decision{
			Placement: "gpu",
			Priority:  float64(in.Build.NVectors),
			Reason:    fmt.Sprintf("%d vectors >= threshold %d", in.Build.NVectors, p.N),
		}
	}
	return Decision{Placement: "local", Reason: fmt.Sprintf("%d vectors < threshold %d", in.Build.NVectors, p.N)}
}

func (StaticThreshold) Reconsider(ReconsiderInput) []Action { return nil }
