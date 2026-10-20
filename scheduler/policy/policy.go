// Package policy decides where a build runs. A Policy is a pure function of
// the input it is handed: no I/O, no clock, no randomness (invariant 8). The
// scheduler and the Phase 3 simulator call the same code.
//
// Every estimate a policy sees — build times, transfer, the GPU queue wait —
// is computed by the host and passed in (Phase 3 decisions 9 and 7): the
// host may add noise to all of them at once for the estimate-error sweep,
// and the step 2 queue-replay estimator plugs in without touching a policy.
// All times are float64 seconds relative to the moment of the call, so a
// policy never needs a clock.
package policy

import (
	"fmt"
	"strconv"
	"strings"
)

// BuildSpec is what the shard tells us about the build. Nothing about the
// queries to come: a real shard does not know them (Phase 1 decision 28).
type BuildSpec struct {
	ShardID  int32
	RoundID  int64
	NVectors int64
	Dim      int32
	MemBytes int64 // modeled GPU memory need, from the cost model
}

// Estimates are the host's cost-model numbers for one build, possibly noised
// (Phase 3 decision 7). LocalBuildS is the full build from zero progress.
type Estimates struct {
	LocalBuildS float64
	GPUBuildS   float64
	TransferS   float64
}

// ShardStatus is the shard's latest load report (Phase 1 decision 28: queue
// depth, waiting-for-index count, oldest wait, build progress), plus two
// numbers the host derives for the policy: the arrival rate estimated from
// successive reports, and the scenario's mean query duration. Valid is false
// when the shard has not reported yet.
type ShardStatus struct {
	QueueDepth        int32
	WaitingNeedsIndex int32
	OldestWaitS       float64
	BuildProgress     float64 // 0..1 of the local build; 0 if not building locally
	ArrivalRateQPS    float64
	MeanQueryS        float64
	Valid             bool
}

// PoolState is what the scheduler knows about live GPU workers at submit.
// QueueWaitS is the host's estimate of how long a newly queued build waits
// before a worker starts it (0 until the step 2 queue replay provides it).
type PoolState struct {
	LiveWorkers int
	LargestMem  int64 // biggest memory capacity among live workers; 0 if none
	QueueWaitS  float64
}

// Input is everything a deployable policy may know at submit time.
type Input struct {
	Build  BuildSpec
	Est    Estimates
	Pool   PoolState
	Status ShardStatus // the submitting shard's latest report, if any
}

type Decision struct {
	Placement string  // "gpu" or "local"
	Priority  float64 // higher claims first; only meaningful for gpu
	Reason    string  // one line for the log and the API response
}

// WorkerView is one live worker as reconsider sees it.
type WorkerView struct {
	ID             string
	Node           string // which machine; the per-node transfer term (step 2) groups by it
	MemBytes       int64
	BusyRemainingS float64 // of its in-flight build; 0 means idle
}

// BuildView is one non-terminal build as reconsider sees it. Phase is the
// policy-level state the host maps from its own rows: "local_running",
// "gpu_queued" or "gpu_leased". QueueWaitS is the host's current estimate of
// this queued build's wait before a worker starts it (meaningful for
// gpu_queued). Moved says placement already changed once after submit
// (Phase 3 decision 8); Speculative marks a decision-5 copy.
type BuildView struct {
	BuildID     string
	Spec        BuildSpec
	Est         Estimates
	Phase       string
	Priority    float64
	QueueWaitS  float64
	Status      ShardStatus // the owning shard's latest report
	Moved       bool
	Speculative bool
}

// ReconsiderInput is the whole visible world on one reconsider tick.
type ReconsiderInput struct {
	Workers []WorkerView
	Builds  []BuildView
}

type ActionKind string

const (
	// ActPreempt kills a local build and queues it on the GPU at Priority.
	ActPreempt ActionKind = "preempt"
	// ActRenege moves a still-queued GPU build back to local.
	ActRenege ActionKind = "renege"
	// ActSpeculate starts a GPU copy of a local build (decision 5) at
	// Priority, without killing the local one.
	ActSpeculate ActionKind = "speculate"
	// ActReprioritize sets a queued GPU build's priority to Priority.
	ActReprioritize ActionKind = "reprioritize"
)

// Action is one row change reconsider asks the host to make (decision 3).
type Action struct {
	Kind     ActionKind
	BuildID  string
	Priority float64
	Reason   string
}

// Policy is the one interface every placement policy implements. Decide runs
// when a shard submits; Reconsider runs on the host's timer and may return
// no actions. Both are pure.
type Policy interface {
	Name() string
	Decide(Input) Decision
	Reconsider(ReconsiderInput) []Action
}

// fitsNowhere is the memory filter every policy applies first: a build whose
// modeled memory exceeds every live worker must build locally — but with
// zero live workers it still queues, because an empty pool may be
// cold-starting (Phase 5) and "nothing fits" is a statement about workers we
// can see, not about workers we cannot.
func fitsNowhere(b BuildSpec, p PoolState) (string, bool) {
	if p.LiveWorkers > 0 && b.MemBytes > p.LargestMem {
		return fmt.Sprintf("needs %d bytes, largest live worker has %d", b.MemBytes, p.LargestMem), true
	}
	return "", false
}

// AlwaysGPU is the baseline: every build goes to the GPU queue, first come
// first served (priority 0, the claim's enqueued_at tiebreak is the order).
type AlwaysGPU struct{}

func (AlwaysGPU) Name() string { return "always_gpu" }

func (AlwaysGPU) Decide(in Input) Decision {
	if reason, no := fitsNowhere(in.Build, in.Pool); no {
		return Decision{Placement: "local", Reason: reason}
	}
	return Decision{Placement: "gpu", Priority: 0, Reason: "always_gpu"}
}

func (AlwaysGPU) Reconsider(ReconsiderInput) []Action { return nil }

// ByName returns the deployable policy for a config value. The clairvoyant
// oracle is deliberately absent: it exists only in the simulator (decision
// 1a), constructed with the workload's future streams.
//
//	always_gpu | static_threshold[:N] | reactive | reactive_spec
func ByName(name string) (Policy, error) {
	base, arg, _ := strings.Cut(name, ":")
	switch base {
	case "", "always_gpu":
		return AlwaysGPU{}, nil
	case "static_threshold":
		t := DefaultThresholdN
		if arg != "" {
			v, err := strconv.ParseInt(arg, 10, 64)
			if err != nil || v <= 0 {
				return nil, fmt.Errorf("static_threshold: bad threshold %q", arg)
			}
			t = v
		}
		return StaticThreshold{N: t}, nil
	case "reactive":
		return NewReactive(ReactiveConfig{}), nil
	case "reactive_spec":
		return NewReactiveSpec(ReactiveConfig{}), nil
	}
	return nil, fmt.Errorf("unknown policy %q", name)
}
