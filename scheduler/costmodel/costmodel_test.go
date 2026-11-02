package costmodel

import (
	"math"
	"testing"
	"time"
)

func TestDefaultsAreInTheIntendedRange(t *testing.T) {
	m := Default()
	cpu := m.CPUBuild(100_000, 128)
	gpu := m.GPUTotal(100_000, 128)
	t.Logf("100k x 128: cpu=%s gpu(total incl. transfer)=%s mem=%d MiB", cpu, gpu, m.GPUMemBytes(100_000, 128)>>20)
	if cpu < 15*time.Second || cpu > 30*time.Second {
		t.Fatalf("cpu build for 100k x 128 = %s, want roughly 20 s", cpu)
	}
	if gpu >= cpu {
		t.Fatalf("gpu total %s should beat cpu %s at this size", gpu, cpu)
	}
	// Tiny builds: the fixed GPU overhead and transfer make the CPU win.
	cpu, gpu = m.CPUBuild(1_000, 128), m.GPUTotal(1_000, 128)
	t.Logf("1k x 128: cpu=%s gpu=%s", cpu, gpu)
	if gpu <= cpu {
		t.Fatalf("for a tiny build gpu %s should lose to cpu %s", gpu, cpu)
	}
}

func TestMonotonicInSize(t *testing.T) {
	m := Default()
	prev := time.Duration(0)
	for _, n := range []int64{1_000, 10_000, 100_000, 1_000_000} {
		d := m.CPUBuild(n, 128)
		if d <= prev {
			t.Fatalf("cpu build not increasing at n=%d: %s <= %s", n, d, prev)
		}
		prev = d
	}
	if m.CPUBuild(0, 128) != 0 || m.CPUBuild(1, 128) != 0 {
		t.Fatal("degenerate sizes should cost nothing")
	}
}

// v1 (Phase 5 decision 6): with P = 1, D0 = 0 and no gate, save or second
// bandwidth, the v1 formulas are v0's to the last digit, so every Phase 1–3
// number stands. These are the numbers the Python test pins.
func TestDefaultIsV0ToTheDigit(t *testing.T) {
	m := Default()
	cpu, gpu := m.CPUBuild(100_000, 128).Seconds(), m.GPUTotal(100_000, 128).Seconds()
	t.Logf("default 100k x 128: cpu=%.9f gpu_total=%.9f local_total=%.9f", cpu, gpu, m.LocalTotal(100_000, 128).Seconds())
	if d := cpu - 20.631162433; d > 1e-6 || d < -1e-6 {
		t.Fatalf("v0's cpu build moved: %.9f", cpu)
	}
	if d := gpu - 4.639116243; d > 1e-6 || d < -1e-6 {
		t.Fatalf("v0's gpu total moved: %.9f", gpu)
	}
	if m.LocalTotal(100_000, 128) != m.CPUBuild(100_000, 128) {
		t.Fatal("without gate, save or store round trip the local path is the build alone")
	}
	if m.Gate(100_000, 128) != 0 || m.Save(100_000, 128) != 0 || m.Upload(100_000, 128) != m.IndexBytesDuration(100_000, 128) {
		t.Fatal("v0 terms must be zero / the single bandwidth")
	}
}

// IndexBytesDuration is the v0 upload: index bytes over the one bandwidth.
func (m Model) IndexBytesDuration(n int64, dim int32) time.Duration {
	return secs(float64(m.IndexBytes(n, dim)) / m.Bandwidth)
}

// The calibrated block, term by term, by hand (Phase 4 evidence 7 and the
// Phase 5 step 1 refit): the shard VM's 408164 x 128 build — the straggler
// of the live rounds — and a 10k x 960 one.
func TestCalibratedTermsByHand(t *testing.T) {
	m := Calibrated()
	n, dim := int64(408164), int32(128)
	ln := math.Log(float64(n))
	shape := math.Pow(float64(n), 1.220) * ln * (128 + 187)
	wantBuild := 2.30e-9 * shape
	wantGate := 1.79e-8 * 10000 * ln * (128 + 187)
	idx := float64(n * (128*4 + 2*16*4))
	vec := float64(n * 128 * 4)
	wantSave, wantFetch, wantUpload := idx/6.6e8, vec/1.57e8, idx/5.0e7
	got := map[string][2]float64{
		"build":                              {m.CPUBuild(n, dim).Seconds(), wantBuild},
		"gate":                               {m.Gate(n, dim).Seconds(), wantGate},
		"save":                               {m.Save(n, dim).Seconds(), wantSave},
		"fetch":                              {m.Fetch(n, dim).Seconds(), wantFetch},
		"upload":                             {m.Upload(n, dim).Seconds(), wantUpload},
		"gpu build":                          {m.GPUBuild(n, dim).Seconds(), (wantBuild+wantGate)/2.12 + 1.5 + wantSave},
		"transfer":                           {m.Transfer(n, dim).Seconds(), wantFetch + wantUpload},
		"gpu total":                          {m.GPUTotal(n, dim).Seconds(), (wantBuild+wantGate)/2.12 + 1.5 + wantSave + wantFetch + wantUpload},
		"local total (shard uses the store)": {m.LocalTotal(n, dim).Seconds(), wantBuild + wantGate + wantSave + wantFetch + wantUpload},
	}
	for _, k := range []string{"build", "gate", "save", "fetch", "upload", "gpu build", "transfer", "gpu total", "local total (shard uses the store)"} {
		g, w := got[k][0], got[k][1]
		t.Logf("calibrated 408164 x 128: %-36s = %8.3f s (by hand %8.3f)", k, g, w)
		if math.Abs(g-w) > 1e-6*math.Max(1, w) {
			t.Fatalf("%s: got %.6f want %.6f", k, g, w)
		}
	}
	// the live rounds' straggler took 35.2 s on the worker; v0 estimated 24.0; v1 must land nearer
	if g := m.GPUTotal(n, dim).Seconds(); g < 30 || g > 42 {
		t.Fatalf("v1's worker path for the 408k build = %.1f s; the live rounds measured 35.2 (v0 said 24.0)", g)
	}
	// the dimension term: a 10k x 960 build costs under 3x a 10k x 128 one on these cores (decision 35)
	r := m.CPUBuild(10_000, 960).Seconds() / m.CPUBuild(10_000, 128).Seconds()
	// the fit's D0 (187) puts the ratio at 3.6 over the whole grid; the single
	// 10k pair measured 2.8; v0 says 7.5 — what matters is well under half of v0
	t.Logf("calibrated 10k: build(960) / build(128) = %.2f (measured 2.8 at 10k on the VM; v0 would say 7.5)", r)
	if r < 2.5 || r > 4.5 {
		t.Fatalf("dimension ratio %.2f out of the measured range", r)
	}
	// a database-fleet shard pays no store round trip
	m2 := m
	m2.ShardUsesStore = false
	if m2.LocalTotal(n, dim) != m.CPUBuild(n, dim)+m.Gate(n, dim)+m.Save(n, dim) {
		t.Fatal("ShardUsesStore=false must drop fetch and upload from the local path")
	}
}

// Cost model v2 (Phase 5 decision 14): the L4 block, term by term by hand, on
// the live rounds' 408164 x 128 build. The worker's path should land near
// what the card did in the session (15.0 s claim-to-done on the second
// build; 20.4 with the process's warm-up) where v1's CPU-worker block said
// 39.7; and the local side is Calibrated()'s, unchanged.
func TestCalibratedL4TermsByHand(t *testing.T) {
	m := CalibratedL4()
	n, dim := int64(408164), int32(128)
	ln := math.Log(float64(n))
	w := 128.0 + 154
	wantBuild := 1.154 + 1.295e-8*math.Pow(float64(n), 0.891)*ln*w
	wantGate := 2.07e-8 * 10000 * ln * w
	idx := float64(n * (128*4 + 2*16*4))
	vec := float64(n * 128 * 4)
	wantSave, wantFetch, wantUpload := idx/2.9e8, vec/1.39e8, idx/1.01e8
	gpuBuild := wantBuild + wantGate + 1.5 + wantSave
	checks := []struct {
		name      string
		got, want float64
	}{
		{"worker build + gate (its own shape)", m.WorkerBuild(n, dim).Seconds(), wantBuild + wantGate},
		{"gpu build (floor, build, gate, overhead, save)", m.GPUBuild(n, dim).Seconds(), gpuBuild},
		{"transfer", m.Transfer(n, dim).Seconds(), wantFetch + wantUpload},
		{"gpu total", m.GPUTotal(n, dim).Seconds(), gpuBuild + wantFetch + wantUpload},
		{"local total (the shard side, as Calibrated)", m.LocalTotal(n, dim).Seconds(), Calibrated().CPUBuild(n, dim).Seconds() + Calibrated().Gate(n, dim).Seconds() + m.Save(n, dim).Seconds() + m.Transfer(n, dim).Seconds()},
	}
	for _, c := range checks {
		t.Logf("L4 block, 408164 x 128: %-46s = %8.3f s (by hand %8.3f)", c.name, c.got, c.want)
		if math.Abs(c.got-c.want) > 1e-6*math.Max(1, c.want) {
			t.Fatalf("%s: got %.6f want %.6f", c.name, c.got, c.want)
		}
	}
	if g := m.GPUTotal(n, dim).Seconds(); g < 12 || g > 22 {
		t.Fatalf("the L4 block's worker path for the 408k build = %.1f s; the session measured 15.0 s (20.4 with the warm-up); v1 said 39.7", g)
	}
	// the small build: the card's floor makes it slower than the shard (evidence 10: 0.5x at 10k)
	small := m.WorkerBuild(10_000, 128).Seconds()
	t.Logf("L4 block, 10k x 128: worker build %.2f s against the shard's %.2f s (the floor dominates)", small, m.CPUBuild(10_000, 128).Seconds())
	if small < m.CPUBuild(10_000, 128).Seconds() {
		t.Fatal("at 10k the card should not beat the shard (evidence 10)")
	}
	// WorkerA 0 falls back to v1's speedup form (with this block's own save bandwidth)
	v1 := m
	v1.WorkerA = 0
	wantV1 := (v1.CPUBuild(n, dim).Seconds()+v1.Gate(n, dim).Seconds())/v1.Speedup + v1.GPUOverhead + v1.Save(n, dim).Seconds()
	if math.Abs(v1.GPUBuild(n, dim).Seconds()-wantV1) > 1e-6 || m.WorkerBuild(1, dim).Seconds() != 1.154 {
		t.Fatalf("WorkerA 0 must be v1's form (%.3f vs %.3f); a 1-vector build pays only the floor (%.3f)", v1.GPUBuild(n, dim).Seconds(), wantV1, m.WorkerBuild(1, dim).Seconds())
	}
}
