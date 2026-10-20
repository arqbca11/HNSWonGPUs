package policy

import "testing"

func TestClairvoyant(t *testing.T) {
	pool := PoolState{LiveWorkers: 2, LargestMem: 8 << 30}
	est := Estimates{LocalBuildS: 10, GPUBuildS: 3, TransferS: 2} // gpu done at 5 with an empty queue
	build := func(shard int32) BuildSpec {
		return BuildSpec{ShardID: shard, RoundID: 1, NVectors: 100_000, MemBytes: 1 << 30}
	}

	t.Log("--- no queries: the oracle compares pure build times (local 10s vs gpu 5s) and queues with priority = would-be local finish")
	p := NewClairvoyant(map[int32][]QueryEvent{})
	d := p.Decide(Input{Build: build(0), Est: est, Pool: pool})
	t.Logf("    %s prio %.1f (%s)", d.Placement, d.Priority, d.Reason)
	if d.Placement != "gpu" || d.Priority != 10 {
		t.Fatalf("got %s/%.1f want gpu/10", d.Placement, d.Priority)
	}

	t.Log("--- a 20s GPU queue flips the same build to local: the oracle sees the queue through the host's estimate")
	d = p.Decide(Input{Build: build(0), Est: est, Pool: PoolState{LiveWorkers: 2, LargestMem: 8 << 30, QueueWaitS: 20}})
	t.Logf("    %s (%s)", d.Placement, d.Reason)
	if d.Placement != "local" {
		t.Fatalf("got %s want local", d.Placement)
	}

	t.Log("--- with a stream, both futures are replayed exactly: local blocks every query behind the build, gpu frees the CPU")
	p = NewClairvoyant(map[int32][]QueryEvent{
		1: {
			{ArriveS: 1, DurS: 2, NeedsIndex: false},
			{ArriveS: 2, DurS: 3, NeedsIndex: true},
		},
	})
	// local: both wait for the build at 10: q1 10..12, q2 12..15 -> done 15.
	// gpu:   q1 runs 1..3; q2 waits for the index at 5, runs 5..8 -> done 8.
	d = p.Decide(Input{Build: build(1), Est: est, Pool: pool})
	t.Logf("    %s prio %.1f (%s)", d.Placement, d.Priority, d.Reason)
	if d.Placement != "gpu" || d.Priority != 15 {
		t.Fatalf("got %s/%.1f want gpu/15 (the replayed local finish)", d.Placement, d.Priority)
	}

	t.Log("--- the skip rule: an index query that is not ready waits without blocking the queries behind it")
	p = NewClairvoyant(map[int32][]QueryEvent{
		2: {
			{ArriveS: 0, DurS: 2, NeedsIndex: true},  // ready at 5 on gpu
			{ArriveS: 1, DurS: 1, NeedsIndex: false}, // must not wait for it
		},
	})
	// gpu: q2 runs 1..2 (skips past q1), q1 runs 5..7 -> done 7, not 8.
	// local: q1 10..12, q2 12..13 -> done 13.
	d = p.Decide(Input{Build: build(2), Est: est, Pool: pool})
	t.Logf("    %s prio %.1f (%s)", d.Placement, d.Priority, d.Reason)
	if d.Placement != "gpu" || d.Priority != 13 {
		t.Fatalf("got %s/%.1f want gpu/13", d.Placement, d.Priority)
	}

	t.Log("--- the memory filter comes first, future or no future")
	d = p.Decide(Input{Build: BuildSpec{ShardID: 2, MemBytes: 16 << 30}, Est: est, Pool: pool})
	if d.Placement != "local" {
		t.Fatalf("got %s want local (fits nowhere)", d.Placement)
	}

	t.Log("--- pure: the same input twice gives the identical decision, and the oracle never reconsiders")
	a, b := p.Decide(Input{Build: build(2), Est: est, Pool: pool}), p.Decide(Input{Build: build(2), Est: est, Pool: pool})
	if a != b {
		t.Fatal("decision is not deterministic")
	}
	if acts := p.Reconsider(ReconsiderInput{}); len(acts) != 0 {
		t.Fatalf("the oracle decides at submit; got %+v", acts)
	}
}

func TestByName(t *testing.T) {
	t.Log("--- every deployable policy is reachable by config name; the oracle deliberately is not (decision 1a)")
	for name, want := range map[string]string{
		"":                       "always_gpu",
		"always_gpu":             "always_gpu",
		"static_threshold":       "static_threshold",
		"static_threshold:50000": "static_threshold",
		"reactive":               "reactive",
		"reactive_spec":          "reactive_spec",
	} {
		p, err := ByName(name)
		if err != nil || p.Name() != want {
			t.Fatalf("ByName(%q): got %v, %v; want %s", name, p, err, want)
		}
		t.Logf("    %-22q -> %s", name, p.Name())
	}
	if p, err := ByName("static_threshold:50000"); err != nil {
		t.Fatal(err)
	} else if th := p.(StaticThreshold); th.N != 50000 {
		t.Fatalf("threshold arg not parsed: %+v", th)
	}
	for _, bad := range []string{"clairvoyant", "static_threshold:none", "static_threshold:-1", "nonsense"} {
		if _, err := ByName(bad); err == nil {
			t.Fatalf("ByName(%q) should fail", bad)
		}
		t.Logf("    %-22q -> rejected, as it must be", bad)
	}
}
