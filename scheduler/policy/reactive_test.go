package policy

import "testing"

// A shard 20% into a 100 s local build with 5 one-second queries waiting (all
// needing the index), no new arrivals: staying costs 80+5 s, migrating costs
// a 15 s GPU trip plus the 5 s drain. The same shard at 95% has only 5 s
// left: the sunk cost makes staying the right call.
func hurtingBuild(progress float64) BuildView {
	return BuildView{
		BuildID: "3:7",
		Spec:    BuildSpec{ShardID: 3, RoundID: 7, NVectors: 150_000, MemBytes: 1 << 30},
		Est:     Estimates{LocalBuildS: 100, GPUBuildS: 12, TransferS: 3},
		Phase:   "local_running",
		Status: ShardStatus{
			QueueDepth: 5, WaitingNeedsIndex: 5, MeanQueryS: 1, ArrivalRateQPS: 0, BuildProgress: progress, Valid: true,
		},
	}
}

func TestReactivePreemption(t *testing.T) {
	p := NewReactive(ReactiveConfig{})

	t.Log("--- a shard at 20% with a blocked backlog: migrating recovers at ~20s vs ~85s local, so the build is preempted")
	acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{hurtingBuild(0.2)}})
	if len(acts) != 1 || acts[0].Kind != ActPreempt || acts[0].BuildID != "3:7" {
		t.Fatalf("expected one preempt of 3:7, got %+v", acts)
	}
	t.Logf("    preempt %s, priority %.1f (%s)", acts[0].BuildID, acts[0].Priority, acts[0].Reason)
	if want := 5*1 + 100*0.8; acts[0].Priority != want {
		t.Fatalf("priority should be the blocked work %.1f, got %.1f", want, acts[0].Priority)
	}

	t.Log("--- the same shard at 95%: only 5s of build left, the GPU trip cannot beat it — sunk cost holds the preemption back")
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{hurtingBuild(0.95)}}); len(acts) != 0 {
		t.Fatalf("expected no action at 95%% progress, got %+v", acts)
	}

	t.Log("--- a build that already moved once is never moved again (decision 8)")
	moved := hurtingBuild(0.2)
	moved.Moved = true
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{moved}}); len(acts) != 0 {
		t.Fatalf("expected no action for a moved build, got %+v", acts)
	}

	t.Log("--- no report yet: nothing to judge by, nothing happens")
	silent := hurtingBuild(0.2)
	silent.Status = ShardStatus{}
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{silent}}); len(acts) != 0 {
		t.Fatalf("expected no action without a report, got %+v", acts)
	}

	t.Log("--- the margin stops a photo-finish migration (stay 21s vs migrate 20s, margin 2s)")
	close := hurtingBuild(0)
	close.Est = Estimates{LocalBuildS: 16, GPUBuildS: 12, TransferS: 3} // stay 16+5=21, migrate 15+5=20
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{close}}); len(acts) != 0 {
		t.Fatalf("expected the margin to hold, got %+v", acts)
	}

	t.Log("--- growth counts: arrivals during the remaining build enlarge the backlog the local branch must drain")
	growing := hurtingBuild(0.2)
	growing.Status.ArrivalRateQPS = 2 // 80 s remaining -> 160 more queries behind the build
	acts = p.Reconsider(ReconsiderInput{Builds: []BuildView{growing}})
	if len(acts) != 1 || acts[0].Kind != ActPreempt {
		t.Fatalf("expected a preempt with a growing backlog, got %+v", acts)
	}
}

func TestReactiveRenegeAndReprioritize(t *testing.T) {
	p := NewReactive(ReactiveConfig{})

	t.Log("--- a queued GPU build whose queue has grown past its local time reneges back to local")
	stuck := BuildView{
		BuildID: "1:7", Spec: BuildSpec{ShardID: 1, RoundID: 7},
		Est: Estimates{LocalBuildS: 20, GPUBuildS: 5, TransferS: 1}, Phase: "gpu_queued", QueueWaitS: 50,
	}
	acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{stuck}})
	if len(acts) != 1 || acts[0].Kind != ActRenege || acts[0].BuildID != "1:7" {
		t.Fatalf("expected one renege of 1:7, got %+v", acts)
	}
	t.Logf("    renege %s (%s)", acts[0].BuildID, acts[0].Reason)

	t.Log("--- the same build with a short queue stays put; a preempted build never reneges back (decision 8)")
	short := stuck
	short.QueueWaitS = 10 // gpu 16 vs local 20: queue still wins
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{short}}); len(acts) != 0 {
		t.Fatalf("expected no renege with a short queue, got %+v", acts)
	}
	once := stuck
	once.Moved = true
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{once}}); len(acts) != 0 {
		t.Fatalf("expected no second move, got %+v", acts)
	}

	t.Log("--- a preempted build's priority follows the blocked work behind it, but only past the churn delta")
	preempted := BuildView{
		BuildID: "2:7", Spec: BuildSpec{ShardID: 2, RoundID: 7},
		Est: Estimates{LocalBuildS: 100, GPUBuildS: 12, TransferS: 3}, Phase: "gpu_queued", Moved: true, Priority: 10,
		Status: ShardStatus{QueueDepth: 30, MeanQueryS: 1, Valid: true}, // blocked work = 30 + 100 = 130
	}
	acts = p.Reconsider(ReconsiderInput{Builds: []BuildView{preempted}})
	if len(acts) != 1 || acts[0].Kind != ActReprioritize || acts[0].Priority != 130 {
		t.Fatalf("expected reprioritize to 130, got %+v", acts)
	}
	t.Logf("    reprioritize %s -> %.0f (%s)", acts[0].BuildID, acts[0].Priority, acts[0].Reason)
	steady := preempted
	steady.Priority = 129.5 // within the 1 s delta
	if acts := p.Reconsider(ReconsiderInput{Builds: []BuildView{steady}}); len(acts) != 0 {
		t.Fatalf("expected no churn inside the delta, got %+v", acts)
	}
}

func TestReactiveSpeculation(t *testing.T) {
	p := NewReactiveSpec(ReactiveConfig{})
	idle := []WorkerView{{ID: "w1", Node: "n1", MemBytes: 8 << 30, BusyRemainingS: 0}}

	// Two local builds, neither preemption-worthy (stay 15s and 19s against a
	// 20s GPU trip plus margin), one clearly the worse straggler.
	mild := hurtingBuild(0.9)
	mild.BuildID, mild.Spec.ShardID = "1:7", 1
	worse := hurtingBuild(0.9)
	worse.BuildID, worse.Spec.ShardID = "2:7", 2
	worse.Est.LocalBuildS = 140 // 14 s remaining at 90%: the worst projected recovery

	t.Log("--- idle worker, empty queue: speculate on the worst projected straggler, without killing it")
	acts := p.Reconsider(ReconsiderInput{Workers: idle, Builds: []BuildView{mild, worse}})
	if len(acts) != 1 || acts[0].Kind != ActSpeculate || acts[0].BuildID != "2:7" {
		t.Fatalf("expected one speculation on 2:7, got %+v", acts)
	}
	t.Logf("    speculate %s, priority %.1f (%s)", acts[0].BuildID, acts[0].Priority, acts[0].Reason)

	t.Log("--- no speculation while the queue has work, while a copy is in flight, while every worker is busy, or when the build fits no idle worker")
	queued := BuildView{BuildID: "9:7", Phase: "gpu_queued", Est: Estimates{LocalBuildS: 1, GPUBuildS: 1}}
	if acts := p.Reconsider(ReconsiderInput{Workers: idle, Builds: []BuildView{mild, worse, queued}}); len(acts) != 0 {
		t.Fatalf("queue not empty: got %+v", acts)
	}
	spec := BuildView{BuildID: "2:7:spec", Phase: "gpu_leased", Speculative: true}
	if acts := p.Reconsider(ReconsiderInput{Workers: idle, Builds: []BuildView{mild, worse, spec}}); len(acts) != 0 {
		t.Fatalf("speculation already in flight: got %+v", acts)
	}
	busy := []WorkerView{{ID: "w1", MemBytes: 8 << 30, BusyRemainingS: 4}}
	if acts := p.Reconsider(ReconsiderInput{Workers: busy, Builds: []BuildView{mild, worse}}); len(acts) != 0 {
		t.Fatalf("no idle worker: got %+v", acts)
	}
	huge := worse
	huge.Spec.MemBytes = 16 << 30
	if acts := p.Reconsider(ReconsiderInput{Workers: idle, Builds: []BuildView{huge}}); len(acts) != 0 {
		t.Fatalf("fits no idle worker: got %+v", acts)
	}

	t.Log("--- a build being preempted this very tick is not also speculated on")
	preemptWorthy := hurtingBuild(0.2) // from TestReactivePreemption: this one preempts
	acts = p.Reconsider(ReconsiderInput{Workers: idle, Builds: []BuildView{preemptWorthy, mild}})
	var kinds []ActionKind
	for _, a := range acts {
		kinds = append(kinds, a.Kind)
		if a.Kind == ActSpeculate && a.BuildID == preemptWorthy.BuildID {
			t.Fatalf("speculated on the build being preempted: %+v", acts)
		}
	}
	t.Logf("    actions this tick: %v", kinds)
}
