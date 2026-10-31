package crosscheck

// Phase 5 step 1 cross-check: cost model v1. Black box. Spec: Phase 5
// decisions 6 (the terms), 7 (Default is v0 to the digit, Calibrated is the
// block, COST_MODEL / -cost select it), 8 (the calibration point; tested from
// Python), the "Step 1 (GPU-free): cost model v1" section, speedup.md
// evidence 9, `go doc ./scheduler/costmodel`, `go doc ./scheduler/cmd/scheduler`,
// the sweep's and the comparer's -h. The implementation was not read.
//
// Pure except TestXcP5S1PythonMirrorAgrees (runs the worker's Python mirror
// through uv) and the process tests (build and run the sweep, the comparer,
// the scheduler binary). No database is needed.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
)

// ---------------------------------------------------------------- hand model

// p5Terms is every documented term in seconds, computed by hand from the
// decision 6 formulas.
type p5Terms struct {
	CPU, Gate, Save, Fetch, Upload, GPUBuild, Transfer, GPUTotal, Local float64
	VecBytes, IdxBytes, Mem                                             float64
}

func p5Hand(m costmodel.Model, n int64, dim int32) p5Terms {
	nf, d := float64(n), float64(dim)
	bpd := float64(m.BytesPerDim)
	var t p5Terms
	t.VecBytes = nf * d * bpd
	t.IdxBytes = nf * (d*bpd + 2*float64(m.M)*4)
	t.Mem = m.MemFactor * t.VecBytes
	if n >= 2 {
		t.CPU = m.A * math.Pow(nf, m.P) * math.Log(nf) * (d + m.D0)
		t.Gate = m.GateCost * m.GateQueries * math.Log(nf) * (d + m.D0)
	}
	if m.SaveBandwidth > 0 {
		t.Save = t.IdxBytes / m.SaveBandwidth
	}
	t.Fetch = t.VecBytes / m.Bandwidth
	bo := m.BandwidthOut
	if bo == 0 {
		bo = m.Bandwidth
	}
	t.Upload = t.IdxBytes / bo
	t.GPUBuild = (t.CPU+t.Gate)/m.Speedup + m.GPUOverhead + t.Save
	t.Transfer = t.Fetch + t.Upload
	t.GPUTotal = t.GPUBuild + t.Transfer
	t.Local = t.CPU + t.Gate + t.Save
	if m.ShardUsesStore {
		t.Local += t.Transfer
	}
	return t
}

func p5Got(m costmodel.Model, n int64, dim int32) p5Terms {
	s := func(d time.Duration) float64 { return d.Seconds() }
	return p5Terms{
		CPU: s(m.CPUBuild(n, dim)), Gate: s(m.Gate(n, dim)), Save: s(m.Save(n, dim)),
		Fetch: s(m.Fetch(n, dim)), Upload: s(m.Upload(n, dim)), GPUBuild: s(m.GPUBuild(n, dim)),
		Transfer: s(m.Transfer(n, dim)), GPUTotal: s(m.GPUTotal(n, dim)), Local: s(m.LocalTotal(n, dim)),
		VecBytes: float64(m.VectorBytes(n, dim)), IdxBytes: float64(m.IndexBytes(n, dim)),
		Mem: float64(m.GPUMemBytes(n, dim)),
	}
}

// p5Close: durations are whole nanoseconds and the spec does not say whether a
// composite (GPUBuild, Transfer, GPUTotal, LocalTotal) is rounded once or is a
// sum of separately truncated terms, so allow 10 ns (up to five terms, a few ns
// each), plus 1e-12 relative for float association order in very long builds.
func p5Close(got, want float64) bool {
	return math.Abs(got-want) <= 10e-9+1e-12*math.Abs(want)
}

func p5Compare(t *testing.T, label string, got, want p5Terms) int {
	t.Helper()
	gv, wv := []float64{got.CPU, got.Gate, got.Save, got.Fetch, got.Upload, got.GPUBuild, got.Transfer, got.GPUTotal, got.Local, got.VecBytes, got.IdxBytes, got.Mem},
		[]float64{want.CPU, want.Gate, want.Save, want.Fetch, want.Upload, want.GPUBuild, want.Transfer, want.GPUTotal, want.Local, want.VecBytes, want.IdxBytes, want.Mem}
	names := []string{"cpu", "gate", "save", "fetch", "upload", "gpu_build", "transfer", "gpu_total", "local_total", "vector_bytes", "index_bytes", "gpu_mem"}
	bad := 0
	for i := range gv {
		tol := p5Close(gv[i], wv[i])
		if i >= 9 { // byte counts are integers; allow truncation of a fractional MemFactor product
			tol = math.Abs(gv[i]-wv[i]) <= 1
		}
		if !tol {
			bad++
			t.Errorf("%s: %s = %.9f, the decision 6 formula gives %.9f", label, names[i], gv[i], wv[i])
		}
	}
	return bad
}

// p5Custom sets every v1 term to a non-default value, so no term can hide
// behind a zero.
func p5Custom() costmodel.Model {
	m := costmodel.Default()
	m.A, m.P, m.D0 = 3.1e-9, 1.17, 55
	m.Speedup, m.GPUOverhead = 3.3, 0.7
	m.GateCost, m.GateQueries = 2.2e-8, 1234
	m.SaveBandwidth, m.Bandwidth, m.BandwidthOut = 4.4e8, 1.1e8, 3.3e7
	m.ShardUsesStore = true
	m.MemFactor = 2.5
	return m
}

var p5Sizes = []int64{2, 3, 10, 1000, 68027, 100000, 408164, 1_000_000, 10_000_000}
var p5Dims = []int32{1, 96, 128, 960, 1536}

// ---------------------------------------------------------------- the model

func TestXcP5S1DefaultIsV0ToTheDigit(t *testing.T) {
	m := costmodel.Default()
	t.Logf("Default() = %+v", m)
	t.Logf("spec (decision 7): Default() stays v0: A 1.4e-7, P 1, D0 0, Speedup 10, GPUOverhead 2, Bandwidth 2e8, no gate/save/second bandwidth, ShardUsesStore false")
	type kv struct {
		name      string
		got, want float64
	}
	for _, c := range []kv{{"A", m.A, 1.4e-7}, {"P", m.P, 1}, {"D0", m.D0, 0}, {"Speedup", m.Speedup, 10},
		{"GPUOverhead", m.GPUOverhead, 2}, {"Bandwidth", m.Bandwidth, 2e8}, {"GateCost", m.GateCost, 0},
		{"GateQueries", m.GateQueries, 0}, {"SaveBandwidth", m.SaveBandwidth, 0}, {"BandwidthOut", m.BandwidthOut, 0},
		{"M", float64(m.M), 16}, {"BytesPerDim", float64(m.BytesPerDim), 4}} {
		if c.got != c.want {
			t.Errorf("Default().%s = %g, v0 is %g", c.name, c.got, c.want)
		}
	}
	if m.ShardUsesStore {
		t.Errorf("Default().ShardUsesStore = true; v0 has no store round trip on the shard")
	}
	cpu, tot := m.CPUBuild(100000, 128), m.GPUTotal(100000, 128)
	t.Logf("100k x 128: CPUBuild %.9f s, GPUTotal %.9f s, LocalTotal %.9f s, Gate %v, Save %v, Fetch %v, Upload %v",
		cpu.Seconds(), tot.Seconds(), m.LocalTotal(100000, 128).Seconds(), m.Gate(100000, 128), m.Save(100000, 128),
		m.Fetch(100000, 128), m.Upload(100000, 128))
	t.Logf("spec (brief, decision 7): cpu 20.631162433 s, gpu total 4.639116243 s; by hand: 1.4e-7·1e5·ln(1e5)·128 = 20.6311624 and 20.6311624/10 + 2 + 51.2e6/2e8 + 64e6/2e8 = 4.6391162")
	if cpu != time.Duration(20631162433) {
		t.Errorf("CPUBuild(100k,128) = %d ns, want 20631162433", cpu)
	}
	if tot != time.Duration(4639116243) {
		t.Errorf("GPUTotal(100k,128) = %d ns, want 4639116243", tot)
	}
	if m.LocalTotal(100000, 128) != cpu {
		t.Errorf("v0's local path is the build alone: LocalTotal %v != CPUBuild %v", m.LocalTotal(100000, 128), cpu)
	}
	if m.Gate(100000, 128) != 0 || m.Save(100000, 128) != 0 {
		t.Errorf("v0 has no gate and no save term")
	}
	// The committed demo CSV (Phase 3) carries v0's 100k numbers too: its
	// queue_wait_p50_s for uniform/always_gpu is the 4.639116243 GPU total.
	bad := 0
	for _, n := range p5Sizes {
		for _, d := range p5Dims {
			bad += p5Compare(t, fmt.Sprintf("Default n=%d dim=%d", n, d), p5Got(m, n, d), p5Hand(m, n, d))
		}
	}
	t.Logf("=> every term of Default() over %d sizes x %d dims matches v1's formula with v0's constants: %d mismatches", len(p5Sizes), len(p5Dims), bad)
}

func TestXcP5S1CalibratedIsTheBlock(t *testing.T) {
	m := costmodel.Calibrated()
	t.Logf("Calibrated() = %+v", m)
	t.Logf("spec (decision 7, step 1 results): A 2.30e-9, P 1.220, D0 187, Speedup 2.12, GPUOverhead 1.5, Gate 1.79e-8, GateQueries 10000, SaveBandwidth 6.6e8, Bandwidth 1.57e8, BandwidthOut 5.0e7, ShardUsesStore true")
	type kv struct {
		name      string
		got, want float64
	}
	for _, c := range []kv{{"A", m.A, 2.30e-9}, {"P", m.P, 1.220}, {"D0", m.D0, 187}, {"Speedup", m.Speedup, 2.12},
		{"GPUOverhead", m.GPUOverhead, 1.5}, {"GateCost", m.GateCost, 1.79e-8}, {"GateQueries", m.GateQueries, 10000},
		{"SaveBandwidth", m.SaveBandwidth, 6.6e8}, {"Bandwidth", m.Bandwidth, 1.57e8}, {"BandwidthOut", m.BandwidthOut, 5.0e7},
		{"M", float64(m.M), 16}, {"BytesPerDim", float64(m.BytesPerDim), 4}} {
		if math.Abs(c.got-c.want) > 1e-12*math.Abs(c.want) {
			t.Errorf("Calibrated().%s = %g, the block says %g", c.name, c.got, c.want)
		}
	}
	if !m.ShardUsesStore {
		t.Errorf("Calibrated().ShardUsesStore = false; the block (this rig) has the shard use the store")
	}
	t.Logf("MemFactor %g (the spec does not state the block's MemFactor; Default's is %g)", m.MemFactor, costmodel.Default().MemFactor)
	// The live straggler: 408164 x 128. Evidence 9 / decision 37 numbers for orientation.
	g := p5Got(m, 408164, 128)
	t.Logf("408164 x 128 on the block: cpu %.3f gate %.3f save %.3f fetch %.3f upload %.3f | gpu_build %.3f transfer %.3f gpu_total %.3f | local_total %.3f",
		g.CPU, g.Gate, g.Save, g.Fetch, g.Upload, g.GPUBuild, g.Transfer, g.GPUTotal, g.Local)
	bad := 0
	for _, n := range p5Sizes {
		for _, d := range p5Dims {
			bad += p5Compare(t, fmt.Sprintf("Calibrated n=%d dim=%d", n, d), p5Got(m, n, d), p5Hand(m, n, d))
		}
	}
	t.Logf("=> every term of Calibrated() matches the decision 6 formulas by hand: %d mismatches", bad)
}

func TestXcP5S1EveryTermByHand(t *testing.T) {
	m := p5Custom()
	t.Logf("a model with every v1 term non-zero and non-default: %+v", m)
	bad := 0
	for _, n := range p5Sizes {
		for _, d := range p5Dims {
			bad += p5Compare(t, fmt.Sprintf("custom n=%d dim=%d", n, d), p5Got(m, n, d), p5Hand(m, n, d))
		}
	}
	g := p5Got(m, 100000, 128)
	t.Logf("100k x 128: cpu %.6f gate %.6f save %.6f fetch %.6f upload %.6f gpu_build %.6f transfer %.6f gpu_total %.6f local %.6f",
		g.CPU, g.Gate, g.Save, g.Fetch, g.Upload, g.GPUBuild, g.Transfer, g.GPUTotal, g.Local)
	t.Logf("spec (decision 6, go doc): GPUBuild = (CPU+gate)/Speedup + GPUOverhead + save; Transfer = fetch + upload; GPUTotal = GPUBuild + Transfer; LocalTotal = CPU + gate + save [+ transfer if ShardUsesStore]")
	t.Logf("=> %d mismatches over %d points", bad, len(p5Sizes)*len(p5Dims))
}

func TestXcP5S1BandwidthOutZeroFallsBackToFetch(t *testing.T) {
	m := p5Custom()
	m.BandwidthOut = 0
	n, d := int64(250000), int32(128)
	up, want := m.Upload(n, d).Seconds(), float64(m.IndexBytes(n, d))/m.Bandwidth
	t.Logf("BandwidthOut 0, Bandwidth %g: Upload(%d,%d) = %.9f s; index bytes %d / Bandwidth = %.9f", m.Bandwidth, n, d, up, m.IndexBytes(n, d), want)
	t.Logf("spec (decision 6, go doc field comment): BandwidthOut 0 = same as Bandwidth (v0)")
	if !p5Close(up, want) {
		t.Errorf("Upload with BandwidthOut 0 = %.9f, want index/Bandwidth = %.9f", up, want)
	}
	m.BandwidthOut = 1e7
	t.Logf("BandwidthOut 1e7: Upload = %.6f s (index / 1e7 = %.6f)", m.Upload(n, d).Seconds(), float64(m.IndexBytes(n, d))/1e7)
	if !p5Close(m.Upload(n, d).Seconds(), float64(m.IndexBytes(n, d))/1e7) {
		t.Errorf("Upload ignores BandwidthOut")
	}
	if !p5Close(m.Fetch(n, d).Seconds(), float64(m.VectorBytes(n, d))/m.Bandwidth) {
		t.Errorf("Fetch must stay on Bandwidth when BandwidthOut is set")
	}
}

func TestXcP5S1SaveBandwidthZeroMeansNoSave(t *testing.T) {
	m := p5Custom()
	m.SaveBandwidth = 0
	n, d := int64(400000), int32(960)
	t.Logf("SaveBandwidth 0: Save(%d,%d) = %v; GPUBuild %.6f; (CPU+gate)/Speedup + overhead = %.6f",
		n, d, m.Save(n, d), m.GPUBuild(n, d).Seconds(), (m.CPUBuild(n, d).Seconds()+m.Gate(n, d).Seconds())/m.Speedup+m.GPUOverhead)
	t.Logf("spec (decision 6): save = index bytes / SaveBandwidth, 0 when SaveBandwidth is 0 (no division by zero, no +Inf)")
	if m.Save(n, d) != 0 {
		t.Errorf("Save with SaveBandwidth 0 = %v, want 0", m.Save(n, d))
	}
	if !p5Close(m.GPUBuild(n, d).Seconds(), (m.CPUBuild(n, d).Seconds()+m.Gate(n, d).Seconds())/m.Speedup+m.GPUOverhead) {
		t.Errorf("GPUBuild carries a save term although SaveBandwidth is 0")
	}
	m.GateCost = 0
	if m.Gate(n, d) != 0 {
		t.Errorf("GateCost 0: Gate = %v, want 0 (go doc: zero when the model has no gate term)", m.Gate(n, d))
	}
}

func TestXcP5S1ShardUsesStoreAddsTheRoundTrip(t *testing.T) {
	m := p5Custom()
	n, d := int64(204082), int32(128)
	m.ShardUsesStore = false
	off := m.LocalTotal(n, d)
	m.ShardUsesStore = true
	on := m.LocalTotal(n, d)
	base := m.CPUBuild(n, d) + m.Gate(n, d) + m.Save(n, d)
	t.Logf("204082 x 128: LocalTotal without the store %v, with %v; CPU+gate+save %v; Transfer %v", off, on, base, m.Transfer(n, d))
	t.Logf("spec (decision 6): local total = build + gate + save, plus fetch + upload when ShardUsesStore")
	if !p5Close(off.Seconds(), base.Seconds()) {
		t.Errorf("ShardUsesStore false: LocalTotal %v != CPU+gate+save %v", off, base)
	}
	if !p5Close(on.Seconds(), (base + m.Transfer(n, d)).Seconds()) {
		t.Errorf("ShardUsesStore true: LocalTotal %v != CPU+gate+save+transfer %v", on, base+m.Transfer(n, d))
	}
	if m.GPUTotal(n, d) != m.GPUTotal(n, d) || !p5Close(m.GPUTotal(n, d).Seconds(), (m.GPUBuild(n, d) + m.Transfer(n, d)).Seconds()) {
		t.Errorf("GPUTotal != GPUBuild + Transfer")
	}
}

func TestXcP5S1TinyBuildsCostNothing(t *testing.T) {
	t.Logf("spec (brief): a build of fewer than 2 vectors costs 0 (ln(1) = 0, ln(0) = -Inf). The design doc and go doc do not say which terms, so CPU build and gate are asserted; the rest is logged")
	for _, m := range []struct {
		name string
		m    costmodel.Model
	}{{"Default", costmodel.Default()}, {"Calibrated", costmodel.Calibrated()}, {"custom", p5Custom()}} {
		for _, n := range []int64{0, 1} {
			g := p5Got(m.m, n, 128)
			t.Logf("%s n=%d: cpu %v gate %v save %v fetch %v upload %v gpu_build %v transfer %v gpu_total %v local %v",
				m.name, n, m.m.CPUBuild(n, 128), m.m.Gate(n, 128), m.m.Save(n, 128), m.m.Fetch(n, 128), m.m.Upload(n, 128),
				m.m.GPUBuild(n, 128), m.m.Transfer(n, 128), m.m.GPUTotal(n, 128), m.m.LocalTotal(n, 128))
			if g.CPU != 0 || g.Gate != 0 {
				t.Errorf("%s n=%d: CPUBuild %v, Gate %v; want 0", m.name, n, g.CPU, g.Gate)
			}
			for _, v := range []float64{g.Save, g.Fetch, g.Upload, g.GPUBuild, g.Transfer, g.GPUTotal, g.Local} {
				if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
					t.Errorf("%s n=%d: a term is %v (NaN, Inf or negative)", m.name, n, v)
				}
			}
		}
	}
}

// ---------------------------------------------------------------- Python mirror

// p5Python evaluates every term in the worker's mirror for the given models
// (by constructor keyword) and points, returning seconds keyed like p5Terms.
const p5PyScript = `
import json, sys
from kgpu_worker.costmodel import Model
spec = json.load(sys.stdin)
out = []
for case in spec:
    if case["kind"] == "default":
        m = Model()
    elif case["kind"] == "calibrated":
        m = Model.calibrated()
    elif case["kind"] == "env":
        m = Model.from_env(case["env"])
    else:
        m = Model(**case["kw"])
    rows = []
    for n, d in case["points"]:
        rows.append(dict(cpu=m.cpu_build_s(n, d), gate=m.gate_s(n, d), save=m.save_s(n, d), fetch=m.fetch_s(n, d),
                         upload=m.upload_s(n, d), gpu_build=m.gpu_build_s(n, d), transfer=m.transfer_s(n, d),
                         gpu_total=m.gpu_total_s(n, d), local=m.local_total_s(n, d), vec=m.vector_bytes(n, d),
                         idx=m.index_bytes(n, d), mem=m.gpu_mem_bytes(n, d)))
    out.append(rows)
print(json.dumps(out))
`

type p5PyCase struct {
	Kind   string            `json:"kind"`
	KW     map[string]any    `json:"kw,omitempty"`
	Env    map[string]string `json:"env,omitempty"`
	Points [][2]int64        `json:"points"`
}

type p5PyRow struct {
	CPU, Gate, Save, Fetch, Upload float64
	GPUBuild                       float64 `json:"gpu_build"`
	Transfer                       float64
	GPUTotal                       float64 `json:"gpu_total"`
	Local                          float64
	Vec, Idx, Mem                  float64
}

func p5RunPython(t *testing.T, cases []p5PyCase) [][]p5PyRow {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv not on PATH: the Python mirror cannot be run")
	}
	root, _ := filepath.Abs("../..")
	script := filepath.Join(t.TempDir(), "mirror.py")
	if err := os.WriteFile(script, []byte(p5PyScript), 0o644); err != nil {
		t.Fatal(err)
	}
	in, _ := json.Marshal(cases)
	cmd := exec.Command(uv, "run", "--project", filepath.Join(root, "worker"), "python", script)
	cmd.Dir = filepath.Join(root, "worker")
	cmd.Stdin = bytes.NewReader(in)
	var so, se bytes.Buffer
	cmd.Stdout, cmd.Stderr = &so, &se
	if err := cmd.Run(); err != nil {
		t.Fatalf("python mirror: %v\n%s", err, se.String())
	}
	var out [][]p5PyRow
	if err := json.Unmarshal(so.Bytes(), &out); err != nil {
		t.Fatalf("python output: %v: %s", err, so.String())
	}
	return out
}

func p5PyKW(m costmodel.Model) map[string]any {
	return map[string]any{"a": m.A, "p": m.P, "d0": m.D0, "speedup": m.Speedup, "gpu_overhead": m.GPUOverhead,
		"gate": m.GateCost, "gate_queries": m.GateQueries, "save_bandwidth": m.SaveBandwidth, "bandwidth": m.Bandwidth,
		"bandwidth_out": m.BandwidthOut, "shard_uses_store": m.ShardUsesStore, "m": m.M, "mem_factor": m.MemFactor,
		"bytes_per_dim": m.BytesPerDim}
}

func TestXcP5S1PythonMirrorAgrees(t *testing.T) {
	var pts [][2]int64
	for _, n := range append([]int64{0, 1}, p5Sizes...) {
		for _, d := range p5Dims {
			pts = append(pts, [2]int64{n, int64(d)})
		}
	}
	custom := p5Custom()
	noOut := p5Custom()
	noOut.BandwidthOut, noOut.SaveBandwidth, noOut.ShardUsesStore = 0, 0, false
	// The env block: COST_MODEL=calibrated plus every override name.
	envM := costmodel.Calibrated()
	envM.A, envM.P, envM.D0, envM.Speedup, envM.GPUOverhead = 4e-9, 1.1, 99, 3, 2.5
	envM.GateCost, envM.GateQueries, envM.SaveBandwidth, envM.Bandwidth, envM.BandwidthOut = 1e-8, 500, 9e8, 1.2e8, 4e7
	envM.ShardUsesStore, envM.MemFactor = false, 3
	env := map[string]string{"COST_MODEL": "calibrated", "COST_A": "4e-9", "COST_P": "1.1", "COST_D0": "99", "COST_SPEEDUP": "3",
		"COST_GPU_OVERHEAD": "2.5", "COST_GATE": "1e-8", "COST_GATE_QUERIES": "500", "COST_SAVE_BANDWIDTH": "9e8",
		"COST_BANDWIDTH": "1.2e8", "COST_BANDWIDTH_OUT": "4e7", "COST_SHARD_USES_STORE": "false", "COST_MEM_FACTOR": "3"}
	cases := []struct {
		label string
		goM   costmodel.Model
		py    p5PyCase
	}{
		{"Default() vs Model()", costmodel.Default(), p5PyCase{Kind: "default", Points: pts}},
		{"Calibrated() vs Model.calibrated()", costmodel.Calibrated(), p5PyCase{Kind: "calibrated", Points: pts}},
		{"custom (all terms)", custom, p5PyCase{Kind: "kw", KW: p5PyKW(custom), Points: pts}},
		{"custom (BandwidthOut 0, no save, no store)", noOut, p5PyCase{Kind: "kw", KW: p5PyKW(noOut), Points: pts}},
		{"from_env(COST_MODEL=calibrated + every COST_* override) vs the same Go model", envM, p5PyCase{Kind: "env", Env: env, Points: pts}},
	}
	var pyc []p5PyCase
	for _, c := range cases {
		pyc = append(pyc, c.py)
	}
	res := p5RunPython(t, pyc)
	t.Logf("spec (decision 6, the mirror's docstring): the Python mirror's formulas and defaults stay identical to Go; from_env reads the same COST_* names as the scheduler")
	for i, c := range cases {
		bad := 0
		for j, p := range pts {
			g := p5Got(c.goM, p[0], int32(p[1]))
			r := res[i][j]
			gv := []float64{g.CPU, g.Gate, g.Save, g.Fetch, g.Upload, g.GPUBuild, g.Transfer, g.GPUTotal, g.Local, g.VecBytes, g.IdxBytes, g.Mem}
			pv := []float64{r.CPU, r.Gate, r.Save, r.Fetch, r.Upload, r.GPUBuild, r.Transfer, r.GPUTotal, r.Local, r.Vec, r.Idx, r.Mem}
			names := []string{"cpu", "gate", "save", "fetch", "upload", "gpu_build", "transfer", "gpu_total", "local", "vec", "idx", "mem"}
			for k := range gv {
				// Go rounds to whole ns; Python keeps the float.
				if !p5Close(gv[k], pv[k]) {
					bad++
					if bad <= 10 {
						t.Errorf("%s n=%d dim=%d %s: Go %.9f, Python %.9f", c.label, p[0], p[1], names[k], gv[k], pv[k])
					}
				}
			}
		}
		r := res[i][len(pts)/2]
		t.Logf("%s: %d points x 12 terms, %d disagreements (sample n=%d dim=%d: python cpu %.6f gpu_total %.6f local %.6f)",
			c.label, len(pts), bad, pts[len(pts)/2][0], pts[len(pts)/2][1], r.CPU, r.GPUTotal, r.Local)
	}
}

// ---------------------------------------------------------------- the sweep

func TestXcP5S1SweepDefaultMatchesDemo(t *testing.T) {
	root, _ := filepath.Abs("../..")
	demo, err := os.ReadFile(filepath.Join(root, "experiments/results/demo/rounds.csv"))
	if err != nil {
		t.Fatal(err)
	}
	dh, drows := swParse(t, demo)
	// The demo script's first sweep, verbatim (scripts/sim-sweep-demo.sh).
	b, err := swRun(t, "-scenarios", "uniform,skewed,bimodal,downstream_heavy,staggered,high_dim,production", "-policies", "all", "-rounds", "20", "-cost", "default")
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	nh, nrows := swParse(t, b)
	t.Logf("demo CSV: %d rows, %d columns; new sweep (-cost default, no overrides): %d rows, %d columns", len(drows), len(dh), len(nrows), len(nh))
	added := []string{}
	have := map[string]bool{}
	for _, c := range dh {
		have[c] = true
	}
	for _, c := range nh {
		if !have[c] {
			added = append(added, c)
		}
	}
	t.Logf("columns the new sweep adds: %v", added)
	key := func(r swRow) string { return r["scenario"] + "|" + r["policy"] + "|" + r["seed"] }
	byKey := map[string]swRow{}
	for _, r := range nrows {
		byKey[key(r)] = r
	}
	bad, missing := 0, 0
	for _, d := range drows {
		r, ok := byKey[key(d)]
		if !ok {
			missing++
			continue
		}
		for _, c := range dh {
			if r[c] != d[c] {
				bad++
				if bad <= 15 {
					t.Errorf("%s column %s: demo %q, now %q", key(d), c, d[c], r[c])
				}
			}
		}
	}
	t.Logf("spec (decision 7, brief): with -cost default and no overrides every round equals the committed demo row: %d demo rows missing from the new sweep, %d differing cells", missing, bad)
	if missing > 0 {
		t.Errorf("%d demo rows have no counterpart", missing)
	}
	want := map[string]string{"cost_model": "default", "cost_p": "1", "cost_d0": "0", "gate": "0", "gate_queries": "0",
		"save_bandwidth": "0", "bandwidth_out": "0", "shard_uses_store": "false"}
	if len(nrows) > 0 {
		r := nrows[0]
		for k, v := range want {
			if r[k] != v {
				t.Errorf("-cost default echoes %s = %q, v0 is %q", k, r[k], v)
			}
		}
		t.Logf("first row's v1 echo: cost_model=%s cost_a=%s cost_p=%s cost_d0=%s gate=%s gate_queries=%s save_bandwidth=%s bandwidth=%s bandwidth_out=%s shard_uses_store=%s gpu_overhead=%s",
			r["cost_model"], r["cost_a"], r["cost_p"], r["cost_d0"], r["gate"], r["gate_queries"], r["save_bandwidth"], r["bandwidth"], r["bandwidth_out"], r["shard_uses_store"], r["gpu_overhead"])
	}
}

func TestXcP5S1SweepEchoesCalibratedAndOverrides(t *testing.T) {
	b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "2", "-cost", "calibrated")
	if err != nil {
		t.Fatalf("sweep -cost calibrated: %v", err)
	}
	_, rows := swParse(t, b)
	want := map[string]float64{"cost_a": 2.3e-9, "cost_p": 1.22, "cost_d0": 187, "gate": 1.79e-8, "gate_queries": 10000,
		"save_bandwidth": 6.6e8, "bandwidth": 1.57e8, "bandwidth_out": 5e7, "gpu_overhead": 1.5, "speedup": 2.12}
	r := rows[0]
	t.Logf("-cost calibrated row: cost_model=%s %v", r["cost_model"], p5Pick(r))
	t.Logf("spec (decision 7, brief): the CSV echoes the model it ran: cost_model and every v1 term")
	if r["cost_model"] != "calibrated" {
		t.Errorf("cost_model = %q, want calibrated", r["cost_model"])
	}
	for k, v := range want {
		if g := r.f(t, k); math.Abs(g-v) > 1e-9*math.Abs(v) {
			t.Errorf("-cost calibrated: %s = %g, the block's is %g", k, g, v)
		}
	}
	if r["shard_uses_store"] != "true" {
		t.Errorf("-cost calibrated: shard_uses_store = %q, want true", r["shard_uses_store"])
	}
	calRound := r.f(t, "round_s")

	ov := []string{"-scenarios", "skewed", "-policies", "reactive", "-rounds", "2", "-cost", "calibrated",
		"-cost_a", "3e-9", "-cost_p", "1.1", "-cost_d0", "50", "-gate", "2e-8", "-gate_queries", "500",
		"-save_bandwidth", "1e9", "-bandwidth", "1e8", "-bandwidth_out", "2e7", "-shard_uses_store", "false",
		"-gpu_overhead", "3", "-speedup", "4"}
	b, err = swRun(t, ov...)
	if err != nil {
		t.Fatalf("sweep with every override: %v", err)
	}
	_, rows = swParse(t, b)
	r = rows[0]
	t.Logf("every override on top of calibrated: %v", p5Pick(r))
	want = map[string]float64{"cost_a": 3e-9, "cost_p": 1.1, "cost_d0": 50, "gate": 2e-8, "gate_queries": 500,
		"save_bandwidth": 1e9, "bandwidth": 1e8, "bandwidth_out": 2e7, "gpu_overhead": 3, "speedup": 4}
	for k, v := range want {
		if g := r.f(t, k); math.Abs(g-v) > 1e-9*math.Abs(v) {
			t.Errorf("override: %s = %g, asked %g", k, g, v)
		}
	}
	if r["shard_uses_store"] != "false" || r["cost_model"] != "calibrated" {
		t.Errorf("override: shard_uses_store %q (asked false), cost_model %q (start point calibrated)", r["shard_uses_store"], r["cost_model"])
	}

	// An override reaches the model, not just the CSV: a huge A makes rounds longer.
	b, err = swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "2", "-cost", "calibrated", "-cost_a", "2.3e-7")
	if err != nil {
		t.Fatal(err)
	}
	_, rows = swParse(t, b)
	t.Logf("-cost calibrated round_s %.3f; with -cost_a 2.3e-7 (100x) round_s %s", calRound, rows[0]["round_s"])
	if rows[0].f(t, "round_s") <= calRound {
		t.Errorf("-cost_a 100x did not lengthen the round: the override is echoed but not applied?")
	}

	// -bandwidth_out 0 = same as -bandwidth: what is echoed is not specified; log it.
	b, err = swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "1", "-cost", "calibrated", "-bandwidth_out", "0")
	if err == nil {
		_, rows = swParse(t, b)
		t.Logf("-bandwidth_out 0 echoes bandwidth_out=%s, bandwidth=%s (spec silent on whether the echo is 0 or the effective rate)", rows[0]["bandwidth_out"], rows[0]["bandwidth"])
	}
}

func p5Pick(r swRow) string {
	var s []string
	for _, k := range []string{"cost_a", "cost_p", "cost_d0", "gate", "gate_queries", "save_bandwidth", "bandwidth", "bandwidth_out", "shard_uses_store", "gpu_overhead", "speedup", "round_s"} {
		s = append(s, k+"="+r[k])
	}
	return strings.Join(s, " ")
}

func TestXcP5S1SweepRefusesOutOfRange(t *testing.T) {
	base := []string{"-scenarios", "skewed", "-policies", "reactive", "-rounds", "1"}
	t.Logf("spec (brief): the sweep refuses out-of-range cost values with a non-zero exit. The docs do not list the ranges; these are the unambiguous ones (negative rates, constants, counts; a zero speedup; an unknown model)")
	must := [][]string{
		{"-cost", "bogus"}, {"-cost", "Calibrated"}, {"-cost_a", "-1e-9"}, {"-cost_p", "-0.5"}, {"-gate", "-2"}, {"-gate_queries", "-2"},
		{"-save_bandwidth", "-5"}, {"-bandwidth", "-1"}, {"-bandwidth_out", "-5"}, {"-gpu_overhead", "-2"},
		{"-speedup", "0"}, {"-speedup", "-1"}, {"-shard_uses_store", "maybe"}, {"-cost_p", "NaN"}, {"-bandwidth", "Inf"},
	}
	for _, a := range must {
		_, err := swRun(t, append(append([]string{}, base...), a...)...)
		if err == nil {
			t.Errorf("sweep %s %s exited 0; an out-of-range value must be refused", a[0], a[1])
		}
	}
	t.Logf("ambiguous values (logged, not asserted): ")
	for _, a := range [][]string{{"-cost_p", "0"}, {"-cost_d0", "-5"}, {"-cost_d0", "-200"}, {"-cost", "calibrated", "-cost_d0", "-200"}} {
		_, err := swRun(t, append(append([]string{}, base...), a...)...)
		t.Logf("  %v -> refused=%v", a, err != nil)
	}
}

// ---------------------------------------------------------------- the comparer

func TestXcP5S1ComparerCalibratedOnStep4(t *testing.T) {
	root, _ := filepath.Abs("../..")
	tl := filepath.Join(root, "experiments/results/step4-timelines")
	out, code := vdRun(t, "-timelines", tl, "-policy", "reactive", "-workers", "2", "-worker_mem_gb", "24",
		"-dataset", "sift-128", "-lease_s", "30", "-cost", "calibrated", "-shards=false")
	if code != 0 {
		t.Fatalf("comparer exit %d", code)
	}
	re := regexp.MustCompile(`(?m)^(\d+)\s+\d+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)`)
	ms := re.FindAllStringSubmatch(out, -1)
	t.Logf("spec (step 1 results, evidence 9): the live rounds replayed on the block: 45.9 and 46.4 s against 41.5 live")
	if len(ms) != 2 {
		t.Fatalf("expected 2 round rows, parsed %d", len(ms))
	}
	wantSim := []float64{45.9, 46.4}
	for i, m := range ms {
		live, _ := strconv.ParseFloat(m[2], 64)
		sim, _ := strconv.ParseFloat(m[3], 64)
		t.Logf("round %s: live %.2f, sim %.2f", m[1], live, sim)
		if math.Abs(sim-wantSim[i]) > 0.1 {
			t.Errorf("round %s: sim %.2f, the doc says ~%.1f", m[1], sim, wantSim[i])
		}
		if math.Abs(live-41.5) > 0.1 {
			t.Errorf("round %s: live %.2f, the doc says 41.5", m[1], live)
		}
	}
	_, code = vdRun(t, "-timelines", tl, "-policy", "reactive", "-workers", "2", "-dataset", "sift-128", "-cost", "bogus", "-shards=false")
	t.Logf("-cost bogus -> exit %d", code)
	if code == 0 {
		t.Errorf("the comparer accepted -cost bogus")
	}
}

func TestXcP5S1ComparerTakesTheV1TermFlags(t *testing.T) {
	root, _ := filepath.Abs("../..")
	tl := filepath.Join(root, "experiments/results/step4-timelines")
	base := []string{"-timelines", tl, "-policy", "reactive", "-workers", "2", "-worker_mem_gb", "24",
		"-dataset", "sift-128", "-lease_s", "30", "-cost", "calibrated", "-shards=false", "-phases", "1"}
	t.Logf("spec: decision 7 ('-cost calibrated on the sweep and the comparer, with the COST_* / flag overrides applied on top') and the step 1 table ('experiments/cmd/sweep/main.go, experiments/cmd/validate/main.go: -cost default|calibrated and the v1 term flags'); the brief: 'the comparer takes -cost and the same term flags' as the sweep")
	flags := [][]string{{"-cost_a", "2.3e-9"}, {"-cost_p", "1.22"}, {"-cost_d0", "187"}, {"-gate", "1.79e-8"}, {"-gate_queries", "10000"},
		{"-save_bandwidth", "6.6e8"}, {"-bandwidth", "1.57e8"}, {"-bandwidth_out", "5e7"}, {"-shard_uses_store", "true"},
		{"-gpu_overhead", "1.5"}, {"-speedup", "2.12"}}
	var missing []string
	for _, f := range flags {
		out, code := vdRun(t, append(append([]string{}, base...), f...)...)
		if code != 0 && strings.Contains(out, "flag provided but not defined") {
			missing = append(missing, f[0])
		} else if code != 0 {
			t.Errorf("%s %s: exit %d", f[0], f[1], code)
		}
	}
	t.Logf("=> v1 term flags the comparer does not define: %v", missing)
	if len(missing) > 0 {
		t.Errorf("the comparer lacks %d of the sweep's v1 term flags: %v", len(missing), missing)
	}
}

// ---------------------------------------------------------------- the scheduler's COST_MODEL

func TestXcP5S1SchedulerCostModelEnv(t *testing.T) {
	root, _ := filepath.Abs("../..")
	bin := filepath.Join(t.TempDir(), "scheduler")
	bc := exec.Command("go", "build", "-o", bin, "./scheduler/cmd/scheduler")
	bc.Dir = root
	if out, err := bc.CombinedOutput(); err != nil {
		t.Fatalf("build scheduler: %v %s", err, out)
	}
	// With TEST_DATABASE_URL the scheduler gets all the way up and its startup
	// lines show which model it chose; without it, a run that passes the
	// configuration check is visible as "still retrying the database".
	dbURL := os.Getenv("TEST_DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://x:y@127.0.0.1:1/none?sslmode=disable"
	}
	run := func(env ...string) (int, string) {
		cmd := exec.Command(bin)
		cmd.Env = append([]string{"PATH=" + os.Getenv("PATH"), "DATABASE_URL=" + dbURL, "LOG_FORMAT=text",
			"LISTEN_ADDR=127.0.0.1:0", "RECONSIDER_INTERVAL=0", "SHUTDOWN_DELAY=0s"}, env...)
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		select {
		case err := <-done:
			var ee *exec.ExitError
			if errors.As(err, &ee) {
				return ee.ExitCode(), buf.String()
			}
			return 0, buf.String()
		case <-time.After(4 * time.Second):
			_ = cmd.Process.Kill()
			<-done
			return -1, buf.String() // still running: it accepted its configuration
		}
	}
	modelLines := func(out string) string {
		var keep []string
		for _, l := range strings.Split(out, "\n") {
			ll := strings.ToLower(l)
			if strings.Contains(ll, "cost") || strings.Contains(ll, "model") || strings.Contains(ll, "start") || strings.Contains(ll, "listen") {
				keep = append(keep, strings.TrimSpace(l))
			}
		}
		if len(keep) > 4 {
			keep = keep[:4]
		}
		return strings.Join(keep, " | ")
	}
	t.Logf("database for the valid runs: %v (set: %v)", strings.SplitN(dbURL, "@", 2)[len(strings.SplitN(dbURL, "@", 2))-1], os.Getenv("TEST_DATABASE_URL") != "")
	t.Logf("spec (brief; go doc ./scheduler/cmd/scheduler: 'COST_MODEL default (v0, the placeholder) or calibrated'): anything else exits 2")
	for _, v := range []string{"bogus", "Calibrated", "CALIBRATED", "v1", "v0", " calibrated"} {
		code, out := run("COST_MODEL=" + v)
		state := fmt.Sprintf("exit %d", code)
		if code == -1 {
			state = "accepted (still running after 4 s)"
		}
		t.Logf("COST_MODEL=%q -> %s; startup: %s", v, state, p5Tail(modelLines(out)+" "+strings.TrimSpace(p5Tail(out, 200)), 600))
		if code != 2 {
			t.Errorf("COST_MODEL=%q: %s, want exit 2 (only default|calibrated are documented)", v, state)
		}
	}
	for _, v := range []string{"default", "calibrated", ""} {
		code, out := run("COST_MODEL=" + v)
		t.Logf("COST_MODEL=%q (valid) -> exit %d (-1 = still running); startup: %s", v, code, p5Tail(modelLines(out), 600))
		if code == 2 {
			t.Errorf("COST_MODEL=%q refused: %s", v, p5Tail(out, 300))
		}
	}
	for _, e := range []string{"COST_P=abc", "COST_SHARD_USES_STORE=maybe", "COST_BANDWIDTH=-1", "COST_SPEEDUP=0", "COST_P=NaN"} {
		code, out := run(e)
		t.Logf("%s -> exit %d (-1 = accepted; spec silent on malformed COST_* values); %s", e, code, p5Tail(modelLines(out), 300))
	}
}

func p5Tail(s string, n int) string {
	if len(s) > n {
		return "…" + s[len(s)-n:]
	}
	return s
}

// TestXcP5S1DefaultIsV0AtEveryPoint pins Default()'s Transfer and GPUTotal to
// what cost model v0 (the committed HEAD before step 1, 1aa4bde) returned, in
// nanoseconds. The values were recorded by building HEAD's
// scheduler/costmodel from `git archive HEAD` and calling its exported
// Default().Transfer / GPUTotal (no source read); they are the points where
// the working tree's v1 Default() was seen to differ. HEAD's sweep with these
// numbers reproduces experiments/results/demo/rounds.csv byte for byte.
func TestXcP5S1DefaultIsV0AtEveryPoint(t *testing.T) {
	v0 := []struct {
		n             int64
		dim           int32
		transfer, tot int64
	}{
		{2, 96, 8960, 2000010823},
		{2, 384, 32000, 2000039452},
		{2, 1536, 124160, 2000153970},
		{3, 1, 2040, 2000002086},
		{3, 96, 13440, 2000017869},
		{3, 128, 17280, 2000023186},
		{7, 96, 31359, 2000049666},
		{999, 768, 31328639, 2105515804},
		{999, 1536, 62017920, 2210392251},
		{1000, 384, 16000000, 2053136092},
		{12345, 1, 8394600, 2010022832},
		{12345, 96, 55305600, 2211615923},
		{12345, 128, 71107200, 2279520964},
		{102040, 960, 3983641600, 21800365587},
		{136055, 1536, 8446294399, 45030767218},
		{999999, 384, 15999984000, 92272089111},
	}
	m := costmodel.Default()
	t.Logf("spec (decision 7): Default() stays v0 to the digit, so every Phase 1-3 test, sweep and saved log keeps its numbers")
	bad := 0
	for _, c := range v0 {
		tr, tot := int64(m.Transfer(c.n, c.dim)), int64(m.GPUTotal(c.n, c.dim))
		ok := tr == c.transfer && tot == c.tot
		t.Logf("n=%d dim=%d: Transfer v0 %d ns, now %d (%+d); GPUTotal v0 %d ns, now %d (%+d) %s",
			c.n, c.dim, c.transfer, tr, tr-c.transfer, c.tot, tot, tot-c.tot, map[bool]string{true: "", false: "<- differs"}[ok])
		if !ok {
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("Default() differs from v0 at %d of %d points by 1-2 ns (Transfer = Fetch + Upload truncated separately, where v0 truncated one (vector+index bytes)/Bandwidth, is the likely cause; black box, not confirmed in source). Effect: 12 queue_wait cells of the committed demo CSV change in the last digit (TestXcP5S1SweepDefaultMatchesDemo)", bad, len(v0))
	}
}
