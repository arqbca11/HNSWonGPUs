package crosscheck

// Phase 3 step 5 cross-check: the sweep runner (experiments/cmd/sweep) as a
// black-box process. Spec: decision 24 (one self-describing CSV row per round,
// byte-identical reruns, non-overridden axes echo the scenario, percentile =
// sorted[int(p*(n-1))]), the roadmap's Sweeps table, `go doc
// ./experiments/cmd/sweep` and its -h flags. Pure: no database.

import (
	"bytes"
	"encoding/csv"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

var (
	swOnce sync.Once
	swBin  string
	swErr  error
)

func swBuild(t *testing.T) string {
	t.Helper()
	swOnce.Do(func() {
		dir, err := os.MkdirTemp("", "xcheck-sweep")
		if err != nil {
			swErr = err
			return
		}
		swBin = filepath.Join(dir, "sweep")
		root, _ := filepath.Abs("../..")
		cmd := exec.Command("go", "build", "-o", swBin, "./experiments/cmd/sweep")
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			swErr = err
			swBin = string(out)
		}
	})
	if swErr != nil {
		t.Fatalf("building the runner: %v %s", swErr, swBin)
	}
	return swBin
}

// swRun runs the runner with args (plus -out FILE) and returns the CSV bytes.
func swRun(t *testing.T, args ...string) ([]byte, error) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "rows.csv")
	cmd := exec.Command(swBuild(t), append(args, "-out", out)...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err := cmd.Run()
	t.Logf("sweep %s → exit err=%v; stderr: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	if err != nil {
		return nil, err
	}
	return os.ReadFile(out)
}

type swRow map[string]string

func swParse(t *testing.T, b []byte) ([]string, []swRow) {
	t.Helper()
	recs, err := csv.NewReader(bytes.NewReader(b)).ReadAll()
	if err != nil || len(recs) == 0 {
		t.Fatalf("CSV parse: %v (%d records)", err, len(recs))
	}
	hdr := recs[0]
	var rows []swRow
	for _, r := range recs[1:] {
		m := swRow{}
		for i, h := range hdr {
			m[h] = r[i]
		}
		rows = append(rows, m)
	}
	return hdr, rows
}

func (r swRow) f(t *testing.T, k string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(r[k], 64)
	if err != nil {
		t.Fatalf("column %q = %q: %v", k, r[k], err)
	}
	return v
}

func swPct(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[int(p*float64(len(s)-1))]
}

func TestSweepCSV(t *testing.T) {
	t.Run("ByteIdenticalReruns", func(t *testing.T) {
		args := []string{"-scenarios", "skewed,staggered", "-policies", "all", "-rounds", "3", "-sigma", "0,0.6", "-kill_at", "4"}
		a, err := swRun(t, args...)
		if err != nil {
			t.Fatal(err)
		}
		b, err := swRun(t, args...)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("two identical invocations wrote different CSVs (decision 24, invariant 10)")
		}
		_, rows := swParse(t, a)
		t.Logf("ok: %d bytes, %d rows, byte-identical across reruns (with noise and a kill)", len(a), len(rows))
		if want := 2 * 5 * 2 * 3; len(rows) != want {
			t.Errorf("rows = %d, want scenarios×policies×sigma×rounds = %d (one row per round)", len(rows), want)
		}
		pols := map[string]bool{}
		for _, r := range rows {
			pols[r["policy"]] = true
		}
		t.Logf("policies under 'all': %v", pols)
		if !pols["clairvoyant"] || len(pols) != 5 {
			t.Errorf("'all' should be the five policies including the oracle")
		}
	})

	t.Run("HeaderCarriesEveryAxis", func(t *testing.T) {
		b, err := swRun(t, "-rounds", "1", "-policies", "always_gpu")
		if err != nil {
			t.Fatal(err)
		}
		hdr, _ := swParse(t, b)
		t.Logf("header: %s", strings.Join(hdr, ","))
		has := map[string]bool{}
		for _, h := range hdr {
			has[h] = true
		}
		// Decision 24 / roadmap Sweeps axes, plus identity and the Result.
		for _, c := range []string{"scenario", "policy", "seed", "sigma", "zipf", "load", "needs_index", "dim", "speedup",
			"workers", "workers_per_node", "n_shards", "completed", "round_s", "straggler_s", "wasted_gpu_s"} {
			if !has[c] {
				t.Errorf("header lacks %q", c)
			}
		}
		// Flags that change results (see -h) but are not axes in the row.
		for _, c := range []string{"poll_s", "reconsider_s", "noise_seed"} {
			if !has[c] {
				t.Logf("note: header has no %q column although -%s changes results", c, c)
			}
		}
	})

	t.Run("NonOverriddenAxesEchoThePreset", func(t *testing.T) {
		sc, _ := workload.Preset("skewed")
		b, err := swRun(t, "-scenarios", "skewed", "-policies", "always_gpu", "-rounds", "1")
		if err != nil {
			t.Fatal(err)
		}
		_, rows := swParse(t, b)
		r := rows[0]
		t.Logf("row: %v", r)
		chk := func(col string, want float64, what string) {
			if got := r.f(t, col); math.Abs(got-want) > 1e-9 {
				t.Errorf("%s = %v, want the preset's %v (%s)", col, got, want, what)
			} else {
				t.Logf("ok: %s = %v echoes the preset (%s)", col, got, what)
			}
		}
		chk("n_shards", float64(sc.NShards), "NShards")
		chk("dim", float64(sc.Dim), "Dim")
		chk("workers", float64(sc.Pool.Workers), "Pool.Workers")
		chk("worker_mem_bytes", float64(sc.Pool.WorkerMemBytes), "Pool.WorkerMemBytes")
		chk("zipf", sc.Sizes.ZipfExponent, "Sizes.ZipfExponent")
		chk("needs_index", sc.Stream.NeedsIndexFraction, "Stream.NeedsIndexFraction")
		chk("speedup", 10, "cost model default")
		chk("load", 1, "multiplier default")
		chk("sigma", 0, "default")

		b2, err := swRun(t, "-scenarios", "skewed", "-policies", "always_gpu", "-rounds", "1",
			"-zipf", "1.5", "-dim", "768", "-workers", "4", "-needs_index", "1", "-speedup", "40", "-shards", "8", "-load", "2", "-workers_per_node", "2")
		if err != nil {
			t.Fatal(err)
		}
		_, rows2 := swParse(t, b2)
		r = rows2[0]
		t.Logf("overridden row: %v", r)
		for col, want := range map[string]float64{"zipf": 1.5, "dim": 768, "workers": 4, "needs_index": 1, "speedup": 40, "n_shards": 8, "load": 2, "workers_per_node": 2} {
			if got := r.f(t, col); got != want {
				t.Errorf("override %s = %v, want %v", col, got, want)
			}
		}
		// The uniform preset has no zipf exponent: what does its row say?
		b3, err := swRun(t, "-scenarios", "uniform", "-policies", "always_gpu", "-rounds", "1")
		if err == nil {
			_, rows3 := swParse(t, b3)
			u, _ := workload.Preset("uniform")
			t.Logf("uniform row zipf=%q (preset Sizes=%+v) — echoes the preset's own field", rows3[0]["zipf"], u.Sizes)
		}
	})

	t.Run("RowsAreSelfDescribing", func(t *testing.T) {
		// Decision 24: "Rows are self-describing … so CSVs concatenate
		// safely." Two runs whose rows agree on every configuration column
		// must agree on the results; otherwise a concatenated CSV mixes runs
		// that cannot be told apart.
		// F2's fix (steps 4-6 triage): reconsider_s, poll_s, lease_s and
		// noise_seed are configuration columns since this round found them
		// missing — every knob that can change a result is recorded.
		cfgCols := []string{"scenario", "policy", "seed", "n_shards", "dim", "workers", "workers_per_node", "worker_mem_bytes",
			"share_links", "sigma", "zipf", "load", "needs_index", "speedup", "kill_at",
			"cost_a", "bandwidth", "gpu_overhead", // Phase 4 step 4: the calibrated constants are knobs too
			"reconsider_s", "poll_s", "lease_s", "noise_seed"}
		resCols := []string{"round_s", "straggler_s", "builds_gpu", "builds_local", "preempts", "reneges", "speculations", "queue_wait_p50_s", "queue_wait_p99_s"}
		base := []string{"-scenarios", "downstream_heavy", "-policies", "reactive_spec", "-rounds", "3", "-sigma", "0.6"}
		variants := [][]string{{"-poll_s", "2"}, {"-reconsider_s", "6"}, {"-noise_seed", "9"}}
		a, err := swRun(t, base...)
		if err != nil {
			t.Fatal(err)
		}
		_, ra := swParse(t, a)
		for _, v := range variants {
			b, err := swRun(t, append(append([]string{}, base...), v...)...)
			if err != nil {
				t.Fatal(err)
			}
			_, rb := swParse(t, b)
			indist := 0
			for i := range ra {
				same := true
				for _, c := range cfgCols {
					if ra[i][c] != rb[i][c] {
						same = false
					}
				}
				diff := false
				for _, c := range resCols {
					if ra[i][c] != rb[i][c] {
						diff = true
					}
				}
				if same && diff {
					indist++
					if indist == 1 {
						t.Logf("%v: seed %s — config columns identical, results differ: round_s %s vs %s, preempts %s vs %s, speculations %s vs %s",
							v, ra[i]["seed"], ra[i]["round_s"], rb[i]["round_s"], ra[i]["preempts"], rb[i]["preempts"], ra[i]["speculations"], rb[i]["speculations"])
					}
				}
			}
			if indist > 0 {
				t.Errorf("flag %s changes results in %d/%d rows but is not recorded in the row: concatenated CSVs cannot tell the runs apart (decision 24)", v[0], indist, len(ra))
			} else {
				t.Logf("flag %v: no row both identically configured and differently resulting", v)
			}
		}
	})

	t.Run("RowMatchesSimRunAndPercentileRule", func(t *testing.T) {
		// The row is the simulator's Result for that cell: re-run sim.Run
		// with the runner's documented defaults (share_links on, workers per
		// node 1, noise seed 1, poll 0.5, reconsider 2) and compare.
		b, err := swRun(t, "-scenarios", "skewed", "-policies", "always_gpu,reactive_spec,clairvoyant", "-rounds", "2", "-sigma", "0.3")
		if err != nil {
			t.Fatal(err)
		}
		_, rows := swParse(t, b)
		sc, _ := workload.Preset("skewed")
		for _, r := range rows {
			seed, _ := strconv.ParseInt(r["seed"], 10, 64)
			w := workload.Generate(sc, seed)
			var p policy.Policy
			if r["policy"] == "clairvoyant" {
				p = policy.NewClairvoyant(sim.Futures(w))
			} else {
				p, _ = policy.ByName(r["policy"])
			}
			res := sim.Run(w, p, sim.Config{ShareNodeLinks: true, WorkersPerNode: 1, NoiseSeed: 1, EstimateSigma: 0.3, PollS: 0.5, ReconsiderS: 2})
			cmp := func(col string, want float64) {
				got := r.f(t, col)
				if math.Abs(got-want) > 1e-6*math.Max(1, math.Abs(want)) {
					t.Errorf("%s seed %d: %s = %v, sim.Run says %v", r["policy"], seed, col, got, want)
				}
			}
			cmp("round_s", res.RoundS)
			cmp("straggler_s", res.StragglerLagS)
			cmp("builds_gpu", float64(res.BuildsGPU))
			cmp("builds_local", float64(res.BuildsLocal))
			cmp("speculations", float64(res.Speculations))
			cmp("wasted_gpu_s", res.WastedGPUS)
			cmp("queue_wait_p50_s", swPct(res.QueueWaits, 0.5))
			cmp("queue_wait_p99_s", swPct(res.QueueWaits, 0.99))
			t.Logf("%s seed %d: CSV round_s=%s p50=%s p99=%s; sim.Run round=%.8f p50=%.8f p99=%.8f (sorted[int(p*(n-1))] over %d waits)",
				r["policy"], seed, r["round_s"], r["queue_wait_p50_s"], r["queue_wait_p99_s"], res.RoundS, swPct(res.QueueWaits, 0.5), swPct(res.QueueWaits, 0.99), len(res.QueueWaits))
		}
	})

	t.Run("StrictNamesFailLoudly", func(t *testing.T) {
		// ByName is strict (step 1 F5) so a typo'd sweep arm cannot run as
		// the wrong policy; the runner must surface that as a failure.
		for _, args := range [][]string{
			{"-policies", "reactiv"}, {"-policies", "static_threshold:"}, {"-policies", "always_gpu:"},
			{"-scenarios", "skewd"}, {"-sigma", "abc"},
		} {
			_, err := swRun(t, append(args, "-rounds", "1")...)
			if err == nil {
				t.Errorf("%v: exit 0, want a failure", args)
			}
		}
	})
}
