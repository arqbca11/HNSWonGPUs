package policy

import "fmt"

// ReactiveConfig tunes the reactive policies. The zero value means defaults.
type ReactiveConfig struct {
	// ThresholdN is the submit-time rule: at submit the policy knows only the
	// build and the pool (the backlog that would justify more does not exist
	// yet), so it places like a static threshold and lets reconsider correct.
	ThresholdN int64
	// MarginS guards against flapping (decision 8): migrating must win by
	// this many seconds, and reneging must win by it too.
	MarginS float64
	// ReprioritizeDeltaS: a queued preempted build's priority is moved to the
	// shard's current blocked work only when it changed by more than this,
	// so the queue is not churned by every report.
	ReprioritizeDeltaS float64
}

func (c ReactiveConfig) withDefaults() ReactiveConfig {
	if c.ThresholdN == 0 {
		c.ThresholdN = DefaultThresholdN
	}
	if c.MarginS == 0 {
		c.MarginS = 2
	}
	if c.ReprioritizeDeltaS == 0 {
		c.ReprioritizeDeltaS = 1
	}
	return c
}

// Reactive is the core deployable policy. At submit it places like a static
// threshold. On every reconsider it compares, for each shard building
// locally, two futures computed from the shard's report alone (nothing about
// future queries), and preempts when migrating recovers the shard sooner by
// more than the margin. It also reneges a queued GPU build back to local
// when local from scratch now beats the queue, and keeps a preempted build's
// priority equal to the blocked work behind it.
//
// Priority currency (decision 10): seconds of blocked shard work. At submit
// that is the build's own local duration (longest-first, the same currency
// the clairvoyant uses); after a preemption it is the backlog plus the local
// time thrown away, so the shard hurting most claims first.
type Reactive struct {
	cfg ReactiveConfig
}

func NewReactive(cfg ReactiveConfig) Reactive { return Reactive{cfg: cfg.withDefaults()} }

func (p Reactive) Name() string { return "reactive" }

func (p Reactive) Decide(in Input) Decision {
	if reason, no := fitsNowhere(in.Build, in.Pool); no {
		return Decision{Placement: "local", Reason: reason}
	}
	if in.Build.NVectors >= p.cfg.ThresholdN {
		return Decision{
			Placement: "gpu",
			Priority:  in.Est.LocalBuildS,
			Reason:    fmt.Sprintf("%d vectors >= threshold %d; reconsider corrects", in.Build.NVectors, p.cfg.ThresholdN),
		}
	}
	return Decision{Placement: "local", Reason: fmt.Sprintf("%d vectors < threshold %d; reconsider watches the backlog", in.Build.NVectors, p.cfg.ThresholdN)}
}

func (p Reactive) Reconsider(in ReconsiderInput) []Action {
	// The memory filter applies to preemption too (bug 11): a build preempted
	// into a queue no live worker can claim would sit there forever, hanging
	// the round. With an empty pool a local build keeps its progress — unlike
	// submit, where queueing costs nothing.
	var largestMem int64
	for _, w := range in.Workers {
		if w.MemBytes > largestMem {
			largestMem = w.MemBytes
		}
	}
	var acts []Action
	for _, b := range in.Builds {
		switch b.Phase {
		case "local_running":
			if b.Spec.MemBytes > largestMem {
				continue
			}
			if a, ok := p.preemptIfWorth(b); ok {
				acts = append(acts, a)
			}
		case "gpu_queued":
			if b.Speculative {
				continue
			}
			if a, ok := p.renegeIfWorth(b); ok {
				acts = append(acts, a)
			} else if a, ok := p.reprioritizeIfDrifted(b); ok {
				acts = append(acts, a)
			}
		}
	}
	return acts
}

// stayLocal is the shard's recovery time if the local build runs to the end:
// the remaining build (sunk cost enters here: the further along, the less
// there is to save), then draining the backlog it will have by then.
func stayLocalS(b BuildView) float64 {
	remaining := b.Est.LocalBuildS * (1 - clamp01(b.Status.BuildProgress))
	backlogAtFinish := float64(b.Status.QueueDepth) + b.Status.ArrivalRateQPS*remaining
	return remaining + backlogAtFinish*b.Status.MeanQueryS
}

// migrate is the shard's recovery time if the build is killed and queued on
// the GPU: the freed CPU drains the non-index backlog at once, while the
// index-needing queries wait for queue + build + transfer. The slower of the
// two drains is the recovery.
func migrateS(b BuildView) float64 {
	nonIndex := float64(b.Status.QueueDepth-b.Status.WaitingNeedsIndex) * b.Status.MeanQueryS
	gpuDone := b.QueueWaitS + b.Est.GPUBuildS + b.Est.TransferS
	index := gpuDone + float64(b.Status.WaitingNeedsIndex)*b.Status.MeanQueryS
	if nonIndex > index {
		return nonIndex
	}
	return index
}

// blockedWorkS is the priority of a preempted build: the backlog behind it
// plus the local build time being abandoned.
func blockedWorkS(b BuildView) float64 {
	return float64(b.Status.QueueDepth)*b.Status.MeanQueryS +
		b.Est.LocalBuildS*(1-clamp01(b.Status.BuildProgress))
}

func (p Reactive) preemptIfWorth(b BuildView) (Action, bool) {
	if !b.Status.Valid || b.Moved {
		return Action{}, false // no report to judge by, or already moved once (decision 8)
	}
	stay, mig := stayLocalS(b), migrateS(b)
	if mig+p.cfg.MarginS >= stay {
		return Action{}, false
	}
	return Action{
		Kind: ActPreempt, BuildID: b.BuildID, Priority: blockedWorkS(b),
		Reason: fmt.Sprintf("stay local %.1fs, migrate %.1fs, margin %.1fs", stay, mig, p.cfg.MarginS),
	}, true
}

func (p Reactive) renegeIfWorth(b BuildView) (Action, bool) {
	if b.Moved {
		return Action{}, false // moved at most once (decision 8)
	}
	gpuDone := b.QueueWaitS + b.Est.GPUBuildS + b.Est.TransferS
	if b.Est.LocalBuildS+p.cfg.MarginS >= gpuDone {
		return Action{}, false
	}
	return Action{
		Kind: ActRenege, BuildID: b.BuildID,
		Reason: fmt.Sprintf("local from scratch %.1fs beats gpu queue %.1fs by the margin", b.Est.LocalBuildS, gpuDone),
	}, true
}

func (p Reactive) reprioritizeIfDrifted(b BuildView) (Action, bool) {
	if !b.Moved || !b.Status.Valid {
		return Action{}, false // only preempted builds carry blocked-work priority
	}
	want := blockedWorkS(b)
	if diff := want - b.Priority; diff < p.cfg.ReprioritizeDeltaS && diff > -p.cfg.ReprioritizeDeltaS {
		return Action{}, false
	}
	return Action{
		Kind: ActReprioritize, BuildID: b.BuildID, Priority: want,
		Reason: fmt.Sprintf("blocked work now %.1fs, priority was %.1f", want, b.Priority),
	}, true
}

// ReactiveSpec is Reactive plus speculation: when a worker is idle and the
// GPU queue is empty, start a GPU copy of the local build with the worst
// projected recovery, without killing it (decision 5). The first copy to
// finish wins; the host cancels the loser and records the wasted work. One
// speculation per tick, and never while one is already in flight.
type ReactiveSpec struct {
	Reactive
}

func NewReactiveSpec(cfg ReactiveConfig) ReactiveSpec {
	return ReactiveSpec{Reactive: NewReactive(cfg)}
}

func (p ReactiveSpec) Name() string { return "reactive_spec" }

func (p ReactiveSpec) Reconsider(in ReconsiderInput) []Action {
	acts := p.Reactive.Reconsider(in)
	preempted := map[string]bool{}
	for _, a := range acts {
		if a.Kind == ActPreempt {
			preempted[a.BuildID] = true
		}
	}
	if a, ok := p.speculate(in, preempted); ok {
		acts = append(acts, a)
	}
	return acts
}

func (p ReactiveSpec) speculate(in ReconsiderInput, preempted map[string]bool) (Action, bool) {
	var idleMem int64
	for _, w := range in.Workers {
		if w.BusyRemainingS == 0 && w.MemBytes > idleMem {
			idleMem = w.MemBytes
		}
	}
	if idleMem == 0 {
		return Action{}, false // no idle worker
	}
	for _, b := range in.Builds {
		if b.Phase == "gpu_queued" || b.Speculative {
			return Action{}, false // queue not empty, or a speculation already in flight
		}
	}
	best, bestStay := BuildView{}, 0.0
	for _, b := range in.Builds {
		if b.Phase != "local_running" || !b.Status.Valid || preempted[b.BuildID] || b.Spec.MemBytes > idleMem {
			continue
		}
		if stay := stayLocalS(b); stay > bestStay {
			best, bestStay = b, stay
		}
	}
	if bestStay == 0 {
		return Action{}, false
	}
	return Action{
		Kind: ActSpeculate, BuildID: best.BuildID, Priority: blockedWorkS(best),
		Reason: fmt.Sprintf("idle worker, empty queue; worst projected recovery %.1fs", bestStay),
	}, true
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
