package policy

import "testing"

func TestStaticThreshold(t *testing.T) {
	p := StaticThreshold{N: 200_000}
	pool := PoolState{LiveWorkers: 2, LargestMem: 8 << 30}
	cases := []struct {
		name     string
		in       Input
		want     string
		wantPrio float64
	}{
		{"above the threshold goes to gpu", Input{Build: BuildSpec{NVectors: 500_000, MemBytes: 1 << 30}, Pool: pool}, "gpu", 500_000},
		{"exactly the threshold goes to gpu", Input{Build: BuildSpec{NVectors: 200_000, MemBytes: 1 << 30}, Pool: pool}, "gpu", 200_000},
		{"below the threshold builds locally", Input{Build: BuildSpec{NVectors: 199_999, MemBytes: 1 << 30}, Pool: pool}, "local", 0},
		{"memory filter overrides the threshold", Input{Build: BuildSpec{NVectors: 500_000, MemBytes: 16 << 30}, Pool: pool}, "local", 0},
	}
	for _, c := range cases {
		d := p.Decide(c.in)
		t.Logf("%-40s -> %-5s prio %.0f (%s)", c.name, d.Placement, d.Priority, d.Reason)
		if d.Placement != c.want || d.Priority != c.wantPrio {
			t.Fatalf("%s: got %s/%.0f want %s/%.0f", c.name, d.Placement, d.Priority, c.want, c.wantPrio)
		}
	}

	t.Log("--- bigger builds claim first: priority is the vector count, so the skewed giant never waits behind mid-sized builds")
	big := p.Decide(Input{Build: BuildSpec{NVectors: 8_000_000, MemBytes: 1 << 30}, Pool: pool})
	mid := p.Decide(Input{Build: BuildSpec{NVectors: 400_000, MemBytes: 1 << 30}, Pool: pool})
	if big.Priority <= mid.Priority {
		t.Fatalf("priority ordering: big %.0f should beat mid %.0f", big.Priority, mid.Priority)
	}

	t.Log("--- a static policy never reconsiders")
	if acts := p.Reconsider(ReconsiderInput{}); len(acts) != 0 {
		t.Fatalf("expected no actions, got %v", acts)
	}
}
