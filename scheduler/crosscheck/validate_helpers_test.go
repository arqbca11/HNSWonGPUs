package crosscheck

// Helpers for the Phase 3 step 7 cross-check (validate_test.go): the
// hour_build preset and SizeDist.Straggler (decision 27), Config.TickPhaseS
// and Result.ShardPlacement (decision 29), and the comparer
// experiments/cmd/validate as a black-box process (decision 28). Written from
// docs/design/phase3-policies.md decisions 26–30 and the step 7 notes,
// experiments/workload/DESIGN.md, docs/bugs.md entries 14 and 15, and `go doc`
// of experiments/sim, experiments/workload, shard/sim.Timeline and the
// comparer's -h text. No implementation source was read.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	shardsim "github.com/arqbca11/HNSWonGPUs/shard/sim"
)

// timelineT lets the mismatch table edit a Timeline in place.
type timelineT = shardsim.Timeline

var (
	vdOnce sync.Once
	vdBin  string
	vdErr  error
)

// vdBuild compiles the comparer once per test binary.
func vdBuild(t *testing.T) string {
	t.Helper()
	vdOnce.Do(func() {
		dir, err := os.MkdirTemp("", "xcheck-validate")
		if err != nil {
			vdErr = err
			return
		}
		vdBin = filepath.Join(dir, "validate")
		root, _ := filepath.Abs("../..")
		cmd := exec.Command("go", "build", "-o", vdBin, "./experiments/cmd/validate")
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			vdErr = err
			vdBin = string(out)
		}
	})
	if vdErr != nil {
		t.Fatalf("building the comparer: %v %s", vdErr, vdBin)
	}
	return vdBin
}

// vdRun runs the comparer and returns its combined output and exit code.
func vdRun(t *testing.T, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command(vdBuild(t), args...)
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	err := cmd.Run()
	code := 0
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		code = ee.ExitCode()
	} else if err != nil {
		t.Fatalf("running the comparer: %v", err)
	}
	t.Logf("$ validate %s\n(exit %d)\n%s", strings.Join(args, " "), code, buf.String())
	return buf.String(), code
}

// vdWorkload regenerates a preset round exactly as the comparer is documented
// to (scenario + seed + shard count from the file).
func vdWorkload(t *testing.T, scenario string, n int, seed int64) workload.Workload {
	t.Helper()
	sc, err := workload.PresetWithShards(scenario, n)
	if err != nil {
		t.Fatalf("PresetWithShards(%s, %d): %v", scenario, n, err)
	}
	return workload.Generate(sc, seed)
}

// vdReplay is the simulator's replay the comparer is documented to run:
// defaults (cadences 0.5 / 2 / LeaseS as given), sharing off (decision 22),
// pool from the preset.
func vdReplay(t *testing.T, w workload.Workload, pol string, cfg sim.Config) sim.Result {
	t.Helper()
	p, err := policy.ByName(pol)
	if err != nil {
		t.Fatalf("policy.ByName(%s): %v", pol, err)
	}
	cfg.ShareNodeLinks = false
	return sim.Run(w, p, cfg)
}

func vdSec(s float64) time.Duration { return time.Duration(s * 1e9) }

// vdTimeline manufactures a live timeline that says the live round did
// exactly what r says, shifted by delta modeled seconds on the round time and
// every shard's finish, recorded at timeScale (real = modeled / scale).
func vdTimeline(w workload.Workload, r sim.Result, roundID int64, timeScale, delta float64) shardsim.Timeline {
	tl := shardsim.Timeline{
		RoundID:         roundID,
		Scenario:        w.Scenario.Name,
		Seed:            w.Seed,
		TimeScale:       timeScale,
		StartedAt:       time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC),
		Duration:        vdSec((r.RoundS + delta) / timeScale),
		DurationModeled: vdSec(r.RoundS + delta),
		StragglerLag:    vdSec(r.StragglerLagS / timeScale),
		BuildsGPU:       r.BuildsGPU,
		BuildsLocal:     r.BuildsLocal,
	}
	for i, s := range w.Shards {
		pl := ""
		if i < len(r.ShardPlacement) {
			pl = r.ShardPlacement[i]
		}
		fin := 0.0
		if i < len(r.ShardFinishS) {
			fin = r.ShardFinishS[i]
		}
		tl.Shards = append(tl.Shards, shardsim.ShardLine{
			ShardID:      s.ShardID,
			NVectors:     s.NVectors,
			Placement:    pl,
			Attempt:      1,
			BuildState:   "succeeded",
			BuildStarted: -1,
			BuildDone:    vdSec((fin + delta) / timeScale),
			Finished:     vdSec((fin + delta) / timeScale),
		})
	}
	return tl
}

// vdWrite writes timelines as round-<id>.json into a fresh dir.
func vdWrite(t *testing.T, tls ...shardsim.Timeline) string {
	t.Helper()
	dir := t.TempDir()
	for _, tl := range tls {
		b, err := json.MarshalIndent(tl, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("round-%d.json", tl.RoundID)), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

var (
	vdAgreeRe    = regexp.MustCompile(`(?i)\bagree`)
	vdDisagreeRe = regexp.MustCompile(`DISAGREE`)
	vdFloatRe    = regexp.MustCompile(`-?\d+\.\d+`)
	vdEnvRe      = regexp.MustCompile(`(-?\d+\.\d+)\s*\.\.\s*(-?\d+\.\d+)`)
)

// vdHasNum reports whether the output prints x to two decimals.
func vdHasNum(out string, x float64) bool {
	want := fmt.Sprintf("%.2f", x)
	for _, f := range vdFloatRe.FindAllString(out, -1) {
		if f == want {
			return true
		}
		if v, err := strconv.ParseFloat(f, 64); err == nil && fmt.Sprintf("%.2f", v) == want {
			return true
		}
	}
	return false
}

// vdEnvelope returns the first "a..b" range in the output.
func vdEnvelope(out string) (lo, hi float64, ok bool) {
	m := vdEnvRe.FindStringSubmatch(out)
	if m == nil {
		return 0, 0, false
	}
	lo, _ = strconv.ParseFloat(m[1], 64)
	hi, _ = strconv.ParseFloat(m[2], 64)
	return lo, hi, true
}

// vdVerdict classifies the comparer's output: "disagree", "agree" or "".
func vdVerdict(out string) string {
	if vdDisagreeRe.MatchString(out) {
		return "disagree"
	}
	if vdAgreeRe.MatchString(out) {
		return "agree"
	}
	return ""
}
