package crosscheck

// Phase 5 cross-check: cost model v2 (the worker's own shape). Black box.
// Spec: Phase 5 decisions 6-8, 12, 13, 14 and "The first GPU session" in
// docs/design/phase5-gpu.md, `go doc ./scheduler/costmodel`,
// `go doc ./scheduler/cmd/scheduler`, the sweep's and the comparer's -h, and the
// brief. The implementation (costmodel.go, the sweep's and the comparer's
// main.go, the Python mirror) was not read.
//
// v2 by hand (decision 14, go doc):
//
//	when WorkerA > 0:
//	  WorkerBuild = WorkerFixed + WorkerA·n^WorkerP·ln(n)·(dim+WorkerD0)
//	                + WorkerGate·GateQueries·ln(n)·(dim+WorkerD0)   (n < 2: WorkerFixed only)
//	  GPUBuild    = WorkerBuild + GPUOverhead + save
//	when WorkerA = 0: WorkerBuild = 0 and GPUBuild is v1's (CPU+gate)/Speedup + GPUOverhead + save
//	everything else (CPU build, gate, save, fetch, upload, transfer, local total) as v1.
//
// Reuses the step 1 cross-check's helpers (p5Hand, p5Got, p5Compare, p5Close,
// p5Sizes, p5Dims, p5Tail) and the sweep's / comparer's runners (swRun,
// swParse, vdRun). Pure except the process tests; no database needed, though
// the scheduler-binary test shows the startup line when TEST_DATABASE_URL is set.

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

// p5v2Hand is p5Hand with decision 14's worker shape, plus WorkerBuild.
func p5v2Hand(m costmodel.Model, n int64, dim int32) (p5Terms, float64) {
	t := p5Hand(m, n, dim) // v1 terms; GPUBuild in the v1 form
	if m.WorkerA <= 0 {
		return t, 0
	}
	wb := m.WorkerFixed
	if n >= 2 {
		nf, d := float64(n), float64(dim)
		wb += m.WorkerA*math.Pow(nf, m.WorkerP)*math.Log(nf)*(d+m.WorkerD0) +
			m.WorkerGate*m.GateQueries*math.Log(nf)*(d+m.WorkerD0)
	}
	t.GPUBuild = wb + m.GPUOverhead + t.Save
	t.GPUTotal = t.GPUBuild + t.Transfer
	return t, wb
}

// p5v2Custom: a v2 model where every term is non-zero and none equals the L4 block.
func p5v2Custom() costmodel.Model {
	m := p5Custom()
	m.WorkerFixed, m.WorkerA, m.WorkerP, m.WorkerD0, m.WorkerGate = 0.83, 2.2e-8, 0.95, 77, 3.3e-8
	return m
}

func p5v2CheckModel(t *testing.T, label string, m costmodel.Model, sizes []int64) int {
	t.Helper()
	bad := 0
	for _, n := range sizes {
		for _, d := range p5Dims {
			want, wb := p5v2Hand(m, n, d)
			bad += p5Compare(t, fmt.Sprintf("%s n=%d dim=%d", label, n, d), p5Got(m, n, d), want)
			if g := m.WorkerBuild(n, d).Seconds(); !p5Close(g, wb) {
				bad++
				t.Errorf("%s n=%d dim=%d: WorkerBuild = %.9f, decision 14 gives %.9f", label, n, d, g, wb)
			}
		}
	}
	return bad
}

// ---------------------------------------------------------------- the model

func TestXcP5V2CalibratedL4IsTheBlock(t *testing.T) {
	l4, cal := costmodel.CalibratedL4(), costmodel.Calibrated()
	t.Logf("CalibratedL4() = %+v", l4)
	t.Logf("Calibrated()   = %+v", cal)
	t.Logf("spec (decision 14, brief): CalibratedL4 = Calibrated with WorkerFixed 1.154, WorkerA 1.295e-8, WorkerP 0.891, WorkerD0 154, WorkerGate 2.07e-8, Bandwidth 1.39e8, BandwidthOut 1.01e8, SaveBandwidth 2.9e8; everything else as Calibrated (GateQueries 10000, ShardUsesStore true)")
	want := cal
	want.WorkerFixed, want.WorkerA, want.WorkerP, want.WorkerD0, want.WorkerGate = 1.154, 1.295e-8, 0.891, 154, 2.07e-8
	want.Bandwidth, want.BandwidthOut, want.SaveBandwidth = 1.39e8, 1.01e8, 2.9e8
	if l4 != want {
		t.Errorf("CalibratedL4() differs from the block:\n got  %+v\n want %+v", l4, want)
	}
	if l4.GateQueries != 10000 || !l4.ShardUsesStore {
		t.Errorf("GateQueries %g ShardUsesStore %v; the brief says 10000 and true", l4.GateQueries, l4.ShardUsesStore)
	}
	if err := l4.Validate(); err != nil {
		t.Errorf("CalibratedL4().Validate() = %v; the shipped block must validate", err)
	}
	t.Logf("=> the block equals Calibrated() with the eight named fields replaced: %v", l4 == want)
}

func TestXcP5V2L4TotalsOnTheLiveStraggler(t *testing.T) {
	m := costmodel.CalibratedL4()
	n, d := int64(408164), int32(128)
	g, l := m.GPUTotal(n, d).Seconds(), m.LocalTotal(n, d).Seconds()
	h, wb := p5v2Hand(m, n, d)
	t.Logf("408164 x 128 on CalibratedL4: WorkerBuild %.3f, GPUBuild %.3f, Transfer %.3f, GPUTotal %.3f | CPUBuild %.3f Gate %.3f Save %.3f LocalTotal %.3f",
		m.WorkerBuild(n, d).Seconds(), m.GPUBuild(n, d).Seconds(), m.Transfer(n, d).Seconds(), g,
		m.CPUBuild(n, d).Seconds(), m.Gate(n, d).Seconds(), m.Save(n, d).Seconds(), l)
	t.Logf("by hand: worker build %.6f, GPU total %.6f, local total %.6f", wb, h.GPUTotal, h.Local)
	t.Logf("spec (brief; decision 14 'the block's worker path is 13.2 s where the card took 15.0'): GPU total 13.109 s, local total 71.258 s")
	if math.Abs(g-13.109) > 0.0005 {
		t.Errorf("GPUTotal = %.4f, the brief says 13.109", g)
	}
	if math.Abs(l-71.258) > 0.0005 {
		t.Errorf("LocalTotal = %.4f, the brief says 71.258", l)
	}
	t.Logf("note: the local total uses the card's fetch (1.39e8), upload (1.01e8) and save (2.9e8) rates, because the model has one set of bandwidths for both sides; with Calibrated()'s rates the shard's path would be %.3f s",
		costmodel.Calibrated().LocalTotal(n, d).Seconds())
}

func TestXcP5V2EveryTermByHand(t *testing.T) {
	sizes := append([]int64{0, 1}, p5Sizes...)
	b1 := p5v2CheckModel(t, "CalibratedL4", costmodel.CalibratedL4(), sizes)
	c := p5v2Custom()
	t.Logf("custom v2 model (every term non-zero): %+v", c)
	b2 := p5v2CheckModel(t, "custom", c, sizes)
	t.Logf("spec (decision 14, go doc): GPU build = WorkerFixed + WorkerA·n^WorkerP·ln(n)·(dim+WorkerD0) + WorkerGate·GateQueries·ln(n)·(dim+WorkerD0) + GPUOverhead + save; every other term as v1")
	t.Logf("=> %d sizes x %d dims x 13 terms: CalibratedL4 %d mismatches, custom %d", len(sizes), len(p5Dims), b1, b2)
}

func TestXcP5V2WorkerAZeroKeepsV1(t *testing.T) {
	m := p5Custom()
	m.WorkerFixed, m.WorkerA, m.WorkerP, m.WorkerD0, m.WorkerGate = 7, 0, 0.9, 150, 5e-8
	t.Logf("a v1 model with WorkerA 0 but the other four worker terms set (fixed 7 s, P 0.9, D0 150, gate 5e-8): %+v", m)
	t.Logf("spec (decision 14): when WorkerA is 0 the v1 speedup form stands; WorkerBuild is 0 (brief)")
	bad := 0
	for _, n := range append([]int64{0, 1}, p5Sizes...) {
		for _, d := range p5Dims {
			bad += p5Compare(t, fmt.Sprintf("WorkerA=0 n=%d dim=%d", n, d), p5Got(m, n, d), p5Hand(m, n, d))
			if w := m.WorkerBuild(n, d); w != 0 {
				bad++
				t.Errorf("WorkerA=0 n=%d dim=%d: WorkerBuild = %v, want 0", n, d, w)
			}
		}
	}
	g := m.GPUBuild(408164, 128).Seconds()
	t.Logf("GPUBuild(408164,128) = %.6f s (v1 form %.6f; a leaked 7 s floor would make it %.6f)", g, p5Hand(m, 408164, 128).GPUBuild, p5Hand(m, 408164, 128).GPUBuild+7)
	t.Logf("=> %d mismatches against v1 with the worker terms ignored", bad)
}

func TestXcP5V2DefaultAndCalibratedUnchanged(t *testing.T) {
	t.Logf("spec (decision 14, brief): Default() and Calibrated() are unchanged to the digit; both have no worker shape")
	for _, c := range []struct {
		name string
		m    costmodel.Model
	}{{"Default", costmodel.Default()}, {"Calibrated", costmodel.Calibrated()}} {
		m := c.m
		t.Logf("%s(): WorkerFixed %g WorkerA %g WorkerP %g WorkerD0 %g WorkerGate %g", c.name, m.WorkerFixed, m.WorkerA, m.WorkerP, m.WorkerD0, m.WorkerGate)
		if m.WorkerFixed != 0 || m.WorkerA != 0 || m.WorkerP != 0 || m.WorkerD0 != 0 || m.WorkerGate != 0 {
			t.Errorf("%s() carries worker-shape terms", c.name)
		}
		bad := 0
		for _, n := range append([]int64{0, 1}, p5Sizes...) {
			for _, d := range p5Dims {
				bad += p5Compare(t, fmt.Sprintf("%s n=%d dim=%d", c.name, n, d), p5Got(m, n, d), p5Hand(m, n, d))
				if m.WorkerBuild(n, d) != 0 {
					bad++
					t.Errorf("%s n=%d: WorkerBuild %v != 0", c.name, n, m.WorkerBuild(n, d))
				}
			}
		}
		t.Logf("%s(): %d mismatches against the v1 formulas", c.name, bad)
	}
	m := costmodel.Default()
	if cpu, tot := m.CPUBuild(100000, 128), m.GPUTotal(100000, 128); cpu != 20631162433 || tot != 4639116243 {
		t.Errorf("Default() 100k x 128: CPUBuild %d ns GPUTotal %d ns, v0 pins 20631162433 / 4639116243", cpu, tot)
	} else {
		t.Logf("Default() 100k x 128 still CPUBuild 20631162433 ns, GPUTotal 4639116243 ns (v0's pinned numbers)")
	}
}

func TestXcP5V2TinyBuildsPayTheFloor(t *testing.T) {
	t.Logf("spec (brief, go doc WorkerBuild): a build of fewer than 2 vectors pays WorkerFixed only (plus overhead and save in GPU build)")
	for _, m := range []costmodel.Model{costmodel.CalibratedL4(), p5v2Custom()} {
		for _, n := range []int64{0, 1} {
			for _, d := range []int32{1, 128, 960} {
				wb, gb := m.WorkerBuild(n, d).Seconds(), m.GPUBuild(n, d).Seconds()
				wantGB := m.WorkerFixed + m.GPUOverhead + m.Save(n, d).Seconds()
				t.Logf("WorkerFixed %.3f n=%d dim=%d: WorkerBuild %.9f, GPUBuild %.9f (want %.9f), CPUBuild %v",
					m.WorkerFixed, n, d, wb, gb, wantGB, m.CPUBuild(n, d))
				if !p5Close(wb, m.WorkerFixed) {
					t.Errorf("n=%d: WorkerBuild %.9f, want the floor %.9f", n, wb, m.WorkerFixed)
				}
				if !p5Close(gb, wantGB) {
					t.Errorf("n=%d: GPUBuild %.9f, want floor + overhead + save %.9f", n, gb, wantGB)
				}
			}
		}
	}
}

func TestXcP5V2ValidateRefusesBadWorkerTerms(t *testing.T) {
	t.Logf("spec (brief): Validate() refuses negative or non-finite COST_WORKER_* values")
	set := map[string]func(*costmodel.Model, float64){
		"WorkerFixed": func(m *costmodel.Model, v float64) { m.WorkerFixed = v },
		"WorkerA":     func(m *costmodel.Model, v float64) { m.WorkerA = v },
		"WorkerP":     func(m *costmodel.Model, v float64) { m.WorkerP = v },
		"WorkerD0":    func(m *costmodel.Model, v float64) { m.WorkerD0 = v },
		"WorkerGate":  func(m *costmodel.Model, v float64) { m.WorkerGate = v },
	}
	for _, base := range []struct {
		name string
		m    costmodel.Model
	}{{"CalibratedL4", costmodel.CalibratedL4()}, {"Calibrated", costmodel.Calibrated()}} {
		for name, f := range set {
			for _, v := range []float64{-1, -1e-12, math.NaN(), math.Inf(1), math.Inf(-1)} {
				m := base.m
				f(&m, v)
				if err := m.Validate(); err == nil {
					t.Errorf("%s with %s = %g: Validate() = nil, want a refusal", base.name, name, v)
				}
			}
			m := base.m
			f(&m, 0)
			t.Logf("%s with %s = 0: Validate() = %v (zero is not refused by the brief's rule; WorkerA 0 is the v1 fallback)", base.name, name, m.Validate())
		}
	}
	m := costmodel.CalibratedL4()
	m.WorkerA = -1
	t.Logf("example refusal: WorkerA -1 -> %v", m.Validate())
	m = costmodel.CalibratedL4()
	m.WorkerP = 0
	t.Logf("open: WorkerA > 0 with WorkerP 0 (a constant-in-n build) -> Validate() = %v (spec silent)", m.Validate())
}

// ---------------------------------------------------------------- Python mirror

const p5v2PyScript = `
import json, sys
from kgpu_worker.costmodel import Model
spec = json.load(sys.stdin)
out = []
for case in spec:
    k = case["kind"]
    if k == "l4":
        m = Model.calibrated_l4()
    elif k == "calibrated":
        m = Model.calibrated()
    elif k == "default":
        m = Model()
    elif k == "env":
        m = Model.from_env(case["env"])
    else:
        m = Model(**case["kw"])
    rows = []
    for n, d in case["points"]:
        rows.append(dict(cpu=m.cpu_build_s(n, d), gate=m.gate_s(n, d), save=m.save_s(n, d), fetch=m.fetch_s(n, d),
                         upload=m.upload_s(n, d), gpu_build=m.gpu_build_s(n, d), transfer=m.transfer_s(n, d),
                         gpu_total=m.gpu_total_s(n, d), local=m.local_total_s(n, d), vec=m.vector_bytes(n, d),
                         idx=m.index_bytes(n, d), mem=m.gpu_mem_bytes(n, d), wb=m.worker_build_s(n, d)))
    out.append(rows)
print(json.dumps(out))
`

type p5v2PyRow struct {
	p5PyRow
	WB float64 `json:"wb"`
}

func p5v2RunPython(t *testing.T, cases []p5PyCase) [][]p5v2PyRow {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv not on PATH: the Python mirror cannot be run")
	}
	root, _ := filepath.Abs("../..")
	script := filepath.Join(t.TempDir(), "mirror_v2.py")
	if err := os.WriteFile(script, []byte(p5v2PyScript), 0o644); err != nil {
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
	var out [][]p5v2PyRow
	if err := json.Unmarshal(so.Bytes(), &out); err != nil {
		t.Fatalf("python output: %v: %s", err, so.String())
	}
	return out
}

func TestXcP5V2PythonMirrorAgrees(t *testing.T) {
	var pts [][2]int64
	for _, n := range append([]int64{0, 1}, p5Sizes...) {
		for _, d := range p5Dims {
			pts = append(pts, [2]int64{n, int64(d)})
		}
	}
	custom := p5v2Custom()
	kw := p5PyKW(custom)
	kw["worker_fixed"], kw["worker_a"], kw["worker_p"], kw["worker_d0"], kw["worker_gate"] =
		custom.WorkerFixed, custom.WorkerA, custom.WorkerP, custom.WorkerD0, custom.WorkerGate
	// COST_MODEL=calibrated-l4 plus the five worker overrides.
	envM := costmodel.CalibratedL4()
	envM.WorkerFixed, envM.WorkerA, envM.WorkerP, envM.WorkerD0, envM.WorkerGate = 2.5, 3e-8, 0.85, 99, 1e-8
	env := map[string]string{"COST_MODEL": "calibrated-l4", "COST_WORKER_FIXED": "2.5", "COST_WORKER_A": "3e-8",
		"COST_WORKER_P": "0.85", "COST_WORKER_D0": "99", "COST_WORKER_GATE": "1e-8"}
	// COST_MODEL=calibrated + COST_WORKER_A switches a v1 block to v2.
	envV := costmodel.Calibrated()
	envV.WorkerFixed, envV.WorkerA, envV.WorkerP, envV.WorkerD0 = 1.2, 1e-8, 0.9, 150
	envVm := map[string]string{"COST_MODEL": "calibrated", "COST_WORKER_FIXED": "1.2", "COST_WORKER_A": "1e-8",
		"COST_WORKER_P": "0.9", "COST_WORKER_D0": "150"}
	// COST_MODEL=calibrated-l4 with COST_WORKER_A=0 falls back to v1.
	envZ := costmodel.CalibratedL4()
	envZ.WorkerA = 0
	cases := []struct {
		label string
		goM   costmodel.Model
		py    p5PyCase
	}{
		{"CalibratedL4() vs Model.calibrated_l4()", costmodel.CalibratedL4(), p5PyCase{Kind: "l4", Points: pts}},
		{"Calibrated() vs Model.calibrated()", costmodel.Calibrated(), p5PyCase{Kind: "calibrated", Points: pts}},
		{"Default() vs Model()", costmodel.Default(), p5PyCase{Kind: "default", Points: pts}},
		{"custom v2 (kw)", custom, p5PyCase{Kind: "kw", KW: kw, Points: pts}},
		{"from_env(calibrated-l4 + COST_WORKER_*)", envM, p5PyCase{Kind: "env", Env: env, Points: pts}},
		{"from_env(calibrated + COST_WORKER_FIXED/A/P/D0)", envV, p5PyCase{Kind: "env", Env: envVm, Points: pts}},
		{"from_env(calibrated-l4 + COST_WORKER_A=0)", envZ, p5PyCase{Kind: "env", Env: map[string]string{"COST_MODEL": "calibrated-l4", "COST_WORKER_A": "0"}, Points: pts}},
	}
	var pyc []p5PyCase
	for _, c := range cases {
		pyc = append(pyc, c.py)
	}
	res := p5v2RunPython(t, pyc)
	t.Logf("spec (brief, decision 14): the Python Model mirrors all of v2: calibrated_l4(), worker_build_s, from_env with the same names and spellings")
	names := []string{"cpu", "gate", "save", "fetch", "upload", "gpu_build", "transfer", "gpu_total", "local", "vec", "idx", "mem", "worker_build"}
	for i, c := range cases {
		bad := 0
		for j, p := range pts {
			g := p5Got(c.goM, p[0], int32(p[1]))
			r := res[i][j]
			gv := []float64{g.CPU, g.Gate, g.Save, g.Fetch, g.Upload, g.GPUBuild, g.Transfer, g.GPUTotal, g.Local, g.VecBytes, g.IdxBytes, g.Mem, c.goM.WorkerBuild(p[0], int32(p[1])).Seconds()}
			pv := []float64{r.CPU, r.Gate, r.Save, r.Fetch, r.Upload, r.GPUBuild, r.Transfer, r.GPUTotal, r.Local, r.Vec, r.Idx, r.Mem, r.WB}
			for k := range gv {
				if !p5Close(gv[k], pv[k]) {
					bad++
					if bad <= 10 {
						t.Errorf("%s n=%d dim=%d %s: Go %.9f, Python %.9f", c.label, p[0], p[1], names[k], gv[k], pv[k])
					}
				}
			}
		}
		k := len(pts) - 9 // a large build
		t.Logf("%s: %d points x 13 terms, %d disagreements (n=%d dim=%d: python worker_build %.4f gpu_total %.4f local %.4f)",
			c.label, len(pts), bad, pts[k][0], pts[k][1], res[i][k].WB, res[i][k].GPUTotal, res[i][k].Local)
	}
}

// ---------------------------------------------------------------- the sweep

func TestXcP5V2SweepDefaultStillMatchesDemo(t *testing.T) {
	root, _ := filepath.Abs("../..")
	demo, err := os.ReadFile(filepath.Join(root, "experiments/results/demo/rounds.csv"))
	if err != nil {
		t.Fatal(err)
	}
	dh, drows := swParse(t, demo)
	b, err := swRun(t, "-scenarios", "uniform,skewed,bimodal,downstream_heavy,staggered,high_dim,production", "-policies", "all", "-rounds", "20", "-cost", "default")
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	nh, nrows := swParse(t, b)
	have := map[string]bool{}
	for _, c := range dh {
		have[c] = true
	}
	var added []string
	for _, c := range nh {
		if !have[c] {
			added = append(added, c)
		}
	}
	t.Logf("demo %d rows x %d cols; new sweep %d rows x %d cols; columns added since the demo: %v", len(drows), len(dh), len(nrows), len(nh), added)
	key := func(r swRow) string { return r["scenario"] + "|" + r["policy"] + "|" + r["seed"] }
	by := map[string]swRow{}
	for _, r := range nrows {
		by[key(r)] = r
	}
	bad, missing := 0, 0
	for _, d := range drows {
		r, ok := by[key(d)]
		if !ok {
			missing++
			continue
		}
		for _, c := range dh {
			if r[c] != d[c] {
				bad++
				if bad <= 10 {
					t.Errorf("%s %s: demo %q, now %q", key(d), c, d[c], r[c])
				}
			}
		}
	}
	t.Logf("spec (brief, decision 14): -cost default with no flags still reproduces experiments/results/demo/rounds.csv cell for cell: %d missing rows, %d differing cells", missing, bad)
	if missing > 0 {
		t.Errorf("%d demo rows missing", missing)
	}
	for _, c := range []string{"worker_fixed", "worker_a", "worker_p", "worker_d0", "worker_gate"} {
		if !strings.Contains(","+strings.Join(nh, ",")+",", ","+c+",") {
			t.Errorf("the sweep's CSV lacks column %s (brief: echoes worker_fixed worker_a worker_p worker_d0 worker_gate)", c)
		} else if len(nrows) > 0 {
			if v, _ := strconv.ParseFloat(nrows[0][c], 64); v != 0 {
				t.Errorf("-cost default echoes %s = %q, want 0", c, nrows[0][c])
			}
		}
	}
}

func TestXcP5V2SweepEchoesL4AndOverrides(t *testing.T) {
	b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "2", "-cost", "calibrated-l4")
	if err != nil {
		t.Fatalf("sweep -cost calibrated-l4: %v", err)
	}
	_, rows := swParse(t, b)
	r := rows[0]
	pick := func(r swRow) string {
		var s []string
		for _, k := range []string{"cost_model", "worker_fixed", "worker_a", "worker_p", "worker_d0", "worker_gate", "bandwidth", "bandwidth_out", "save_bandwidth", "gate_queries", "shard_uses_store", "speedup", "builds_gpu", "builds_local", "round_s"} {
			s = append(s, k+"="+r[k])
		}
		return strings.Join(s, " ")
	}
	t.Logf("-cost calibrated-l4 row 1: %s", pick(r))
	t.Logf("spec (decision 14, brief): the sweep takes -cost calibrated-l4 and echoes worker_fixed worker_a worker_p worker_d0 worker_gate")
	if r["cost_model"] != "calibrated-l4" {
		t.Errorf("cost_model = %q, want calibrated-l4", r["cost_model"])
	}
	want := map[string]float64{"worker_fixed": 1.154, "worker_a": 1.295e-8, "worker_p": 0.891, "worker_d0": 154, "worker_gate": 2.07e-8,
		"bandwidth": 1.39e8, "bandwidth_out": 1.01e8, "save_bandwidth": 2.9e8, "gate_queries": 10000, "cost_a": 2.3e-9, "cost_p": 1.22, "cost_d0": 187}
	for k, v := range want {
		if g := r.f(t, k); math.Abs(g-v) > 1e-9*math.Abs(v) {
			t.Errorf("-cost calibrated-l4: %s = %g, the block's is %g", k, g, v)
		}
	}
	l4Round := r.f(t, "round_s")

	b, err = swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "2", "-cost", "calibrated-l4",
		"-worker_fixed", "2.5", "-worker_a", "3e-8", "-worker_p", "0.85", "-worker_d0", "99", "-worker_gate", "1e-8")
	if err != nil {
		t.Fatalf("sweep with the five worker overrides: %v", err)
	}
	_, rows = swParse(t, b)
	r = rows[0]
	t.Logf("with -worker_fixed 2.5 -worker_a 3e-8 -worker_p 0.85 -worker_d0 99 -worker_gate 1e-8: %s", pick(r))
	for k, v := range map[string]float64{"worker_fixed": 2.5, "worker_a": 3e-8, "worker_p": 0.85, "worker_d0": 99, "worker_gate": 1e-8} {
		if g := r.f(t, k); math.Abs(g-v) > 1e-9*math.Abs(v) {
			t.Errorf("override: %s = %g, asked %g", k, g, v)
		}
	}

	// The override reaches the model: a 200 s floor makes every GPU path long.
	b, err = swRun(t, "-scenarios", "skewed", "-policies", "always_gpu", "-rounds", "1", "-cost", "calibrated-l4")
	if err != nil {
		t.Fatal(err)
	}
	_, base := swParse(t, b)
	b, err = swRun(t, "-scenarios", "skewed", "-policies", "always_gpu", "-rounds", "1", "-cost", "calibrated-l4", "-worker_fixed", "200")
	if err != nil {
		t.Fatal(err)
	}
	_, slow := swParse(t, b)
	t.Logf("always_gpu skewed seed 1: round_s %s on the block, %s with -worker_fixed 200", base[0]["round_s"], slow[0]["round_s"])
	if slow[0].f(t, "round_s") < base[0].f(t, "round_s")+150 {
		t.Errorf("-worker_fixed 200 did not lengthen an always-GPU round by ~200 s per queued build: the override is echoed but not applied?")
	}

	// -worker_a 0 = v1's speedup form on the same block.
	b, err = swRun(t, "-scenarios", "skewed", "-policies", "always_gpu", "-rounds", "1", "-cost", "calibrated-l4", "-worker_a", "0")
	if err != nil {
		t.Fatalf("-worker_a 0: %v", err)
	}
	_, v1 := swParse(t, b)
	t.Logf("-worker_a 0 (help: '0 = v1's speedup form'): round_s %s worker_a %s (block: %s, the slower v1 form expected for always_gpu)", v1[0]["round_s"], v1[0]["worker_a"], base[0]["round_s"])
	if v1[0].f(t, "worker_a") != 0 {
		t.Errorf("-worker_a 0 echoed %s", v1[0]["worker_a"])
	}
	if v1[0]["round_s"] == base[0]["round_s"] {
		t.Errorf("-worker_a 0 left the round unchanged; the v1 form (speedup 2.12) is much slower than the L4 shape")
	}
	_ = l4Round
}

func TestXcP5V2SweepRefusesBadWorkerValues(t *testing.T) {
	base := []string{"-scenarios", "skewed", "-policies", "reactive", "-rounds", "1"}
	t.Logf("spec (brief): -1 = keep; negative other than -1, NaN, Inf are refused; -cost takes calibrated-l4 exactly")
	for _, f := range []string{"-worker_fixed", "-worker_a", "-worker_p", "-worker_d0", "-worker_gate"} {
		for _, v := range []string{"-2", "-0.5", "-1e-9", "NaN", "Inf", "-Inf"} {
			if _, err := swRun(t, append(append([]string{}, base...), "-cost", "calibrated-l4", f, v)...); err == nil {
				t.Errorf("sweep %s %s exited 0; must be refused", f, v)
			}
		}
		if _, err := swRun(t, append(append([]string{}, base...), "-cost", "calibrated-l4", f, "-1")...); err != nil {
			t.Errorf("sweep %s -1 (keep) refused: %v", f, err)
		}
	}
	for _, v := range []string{"Calibrated-L4", "calibrated_l4", "l4", "calibrated-L4", "calibratedl4"} {
		if _, err := swRun(t, append(append([]string{}, base...), "-cost", v)...); err == nil {
			t.Errorf("sweep -cost %q exited 0; only calibrated-l4 is the spelling", v)
		}
	}
}

// ---------------------------------------------------------------- the comparer

func TestXcP5V2ComparerOnGPUTimelines(t *testing.T) {
	root, _ := filepath.Abs("../..")
	tl := filepath.Join(root, "experiments/results/gpu-timelines")
	out, code := vdRun(t, "-timelines", tl, "-policy", "reactive", "-workers", "1", "-worker_mem_gb", "22",
		"-dataset", "sift-128", "-lease_s", "30", "-trust_policy", "-cost", "calibrated-l4")
	if code != 0 && code != 1 {
		t.Fatalf("comparer exit %d", code)
	}
	t.Logf("comparer exit %d (DISAGREE is expected: the live rounds ran another model)", code)
	re := regexp.MustCompile(`(?m)^(\d+)\s+\d+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([-+]?[\d.]+)`)
	ms := re.FindAllStringSubmatch(out, -1)
	t.Logf("spec (decision 14, brief): sim rounds about 31.6 and 32.8 s, live 50.65 / 52.89; the 408k and 204k builds on the GPU")
	if len(ms) != 2 {
		t.Fatalf("expected 2 round rows, parsed %d", len(ms))
	}
	wantSim, wantLive := []float64{31.6, 32.8}, []float64{50.65, 52.89}
	for i, m := range ms {
		live, _ := strconv.ParseFloat(m[2], 64)
		sim, _ := strconv.ParseFloat(m[3], 64)
		t.Logf("round %s: live %.2f sim %.2f", m[1], live, sim)
		if math.Abs(sim-wantSim[i]) > 0.1 {
			t.Errorf("round %s: sim %.2f, the doc says ~%.1f", m[1], sim, wantSim[i])
		}
		if math.Abs(live-wantLive[i]) > 0.01 {
			t.Errorf("round %s: live %.2f, the doc says %.2f", m[1], live, wantLive[i])
		}
	}
	// Per-shard rows: "<shard> <n_vectors> <live place>/<att> <sim place> ..."
	sh := regexp.MustCompile(`(?m)^\s+(\d+)\s+(\d+)\s+(gpu|local)\s*/\d+\s+(gpu|local)\s`)
	rows := sh.FindAllStringSubmatch(out, -1)
	found := map[string]int{}
	for _, r := range rows {
		if r[2] == "408164" || r[2] == "204082" {
			t.Logf("shard %s n=%s: live %s, sim %s", r[1], r[2], r[3], r[4])
			if r[4] != "gpu" {
				t.Errorf("n=%s: sim placed it %s; decision 14 says both large builds go to the card", r[2], r[4])
			}
			found[r[2]]++
		}
	}
	if found["408164"] != 2 || found["204082"] != 2 {
		t.Errorf("per-shard rows for the 408k/204k builds: %v, want 2 rounds each", found)
	}
}

func TestXcP5V2ComparerTakesTheWorkerFlags(t *testing.T) {
	root, _ := filepath.Abs("../..")
	tl := filepath.Join(root, "experiments/results/gpu-timelines")
	base := []string{"-timelines", tl, "-policy", "reactive", "-workers", "1", "-worker_mem_gb", "22",
		"-dataset", "sift-128", "-lease_s", "30", "-trust_policy", "-cost", "calibrated-l4", "-shards=false"}
	t.Logf("spec (decision 14): 'The scheduler, the worker, the sweep (-worker_fixed -worker_a -worker_p -worker_d0 -worker_gate, echoed as columns) and the comparer read the five new terms'; step 1's finding B: a live round run with an override must be replayable")
	var missing []string
	for _, f := range [][]string{{"-worker_fixed", "1.154"}, {"-worker_a", "1.295e-8"}, {"-worker_p", "0.891"}, {"-worker_d0", "154"}, {"-worker_gate", "2.07e-8"}} {
		out, code := vdRun(t, append(append([]string{}, base...), f...)...)
		if strings.Contains(out, "flag provided but not defined") {
			missing = append(missing, f[0])
		} else if code != 0 && code != 1 {
			t.Errorf("%s %s: exit %d", f[0], f[1], code)
		}
	}
	t.Logf("=> worker-shape flags the comparer does not define: %v", missing)
	if len(missing) > 0 {
		t.Errorf("the comparer lacks %v: a round whose scheduler ran with COST_WORKER_* overrides cannot be replayed (only the block as shipped)", missing)
	}
	_, code := vdRun(t, "-timelines", tl, "-policy", "reactive", "-workers", "1", "-dataset", "sift-128", "-trust_policy", "-cost", "Calibrated-L4", "-shards=false")
	if code == 0 { // a refusal is any non-zero exit: the comparer's log.Fatalf exits 1, the flag package 2
		t.Errorf("the comparer accepted -cost Calibrated-L4 (exit %d)", code)
	}
}

// ---------------------------------------------------------------- the scheduler binary

func TestXcP5V2SchedulerCostModelEnv(t *testing.T) {
	root, _ := filepath.Abs("../..")
	bin := filepath.Join(t.TempDir(), "scheduler")
	bc := exec.Command("go", "build", "-o", bin, "./scheduler/cmd/scheduler")
	bc.Dir = root
	if out, err := bc.CombinedOutput(); err != nil {
		t.Fatalf("build scheduler: %v %s", err, out)
	}
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
			return -1, buf.String()
		}
	}
	costLine := func(out string) string {
		for _, l := range strings.Split(out, "\n") {
			if strings.Contains(l, "cost_model") {
				return strings.TrimSpace(l)
			}
		}
		return p5Tail(strings.TrimSpace(out), 300)
	}
	t.Logf("database: real=%v", os.Getenv("TEST_DATABASE_URL") != "")
	t.Logf("spec (brief, go doc ./scheduler/cmd/scheduler): COST_MODEL default|calibrated|calibrated-l4, anything else exits 2; COST_WORKER_FIXED/A/P/D0/GATE read; negative or non-finite refused (Validate)")
	code, out := run("COST_MODEL=calibrated-l4")
	t.Logf("COST_MODEL=calibrated-l4 -> %d (-1 = still running, accepted); %s", code, costLine(out))
	if code == 2 {
		t.Errorf("COST_MODEL=calibrated-l4 refused: %s", p5Tail(out, 400))
	}
	if os.Getenv("TEST_DATABASE_URL") != "" && !strings.Contains(out, "calibrated-l4") {
		t.Errorf("the startup line does not name cost_model=calibrated-l4: %s", p5Tail(out, 400))
	}
	for _, v := range []string{"Calibrated-L4", "calibrated_l4", "CALIBRATED-L4", "l4", "calibrated-l4 ", "calibratedl4"} {
		code, out := run("COST_MODEL=" + v)
		t.Logf("COST_MODEL=%q -> exit %d; %s", v, code, p5Tail(strings.TrimSpace(out), 200))
		if code != 2 {
			t.Errorf("COST_MODEL=%q: exit %d, want 2", v, code)
		}
	}
	for _, name := range []string{"COST_WORKER_FIXED", "COST_WORKER_A", "COST_WORKER_P", "COST_WORKER_D0", "COST_WORKER_GATE"} {
		for _, v := range []string{"-1", "NaN", "Inf", "abc"} {
			code, out := run("COST_MODEL=calibrated-l4", name+"="+v)
			t.Logf("%s=%s -> exit %d; %s", name, v, code, p5Tail(strings.TrimSpace(out), 160))
			if code != 2 {
				t.Errorf("%s=%s: exit %d, want 2 (negative or non-finite refused at startup)", name, v, code)
			}
		}
	}
	code, out = run("COST_MODEL=calibrated", "COST_WORKER_FIXED=1.2", "COST_WORKER_A=1e-8", "COST_WORKER_P=0.9", "COST_WORKER_D0=150", "COST_WORKER_GATE=2e-8")
	t.Logf("COST_MODEL=calibrated + all five COST_WORKER_* (valid) -> %d; %s", code, costLine(out))
	if code == 2 {
		t.Errorf("valid worker overrides refused: %s", p5Tail(out, 300))
	}
	if os.Getenv("TEST_DATABASE_URL") != "" {
		t.Logf("observability (logged, not asserted): does the startup line carry the worker terms? worker_a present: %v", strings.Contains(out, "worker_a"))
	}
}
