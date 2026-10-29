// validate: Phase 3 step 7's comparer. Reads the per-round timeline files a
// live shard-simulator run wrote (shard/sim.Timeline, one JSON per round),
// regenerates each round's workload from the scenario and seed the file
// names, replays it through the discrete-event simulator with the live
// stack's cadences, and prints the two side by side: round time, straggler
// lag, the GPU/local split and every shard's finish. A round "agrees" when
// its round time is within max(-tol_s, -tol_frac x simulated) of the
// simulator's; a disagreement is a finding to triage, not a failure, so the
// exit code is 0 unless -strict.
//
//	go run ./experiments/cmd/validate -timelines <dir> -policy reactive -workers 2
//
// Cadences: the live stack's report, reconsider and lease intervals are
// wall-clock settings; under FAKE_TIME_SCALE=s they are s times longer in
// modeled seconds, so the simulator is told poll_s*s, reconsider_s*s and
// lease_s*s, read from each timeline's time_scale. Link sharing is off: the
// live fake worker sleeps and cannot contend (decision 22).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	esim "github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	ssim "github.com/arqbca11/HNSWonGPUs/shard/sim"
)

func main() {
	var (
		dir         = flag.String("timelines", "", "directory of round-<id>.json files written by shardsim (TIMELINE_DIR)")
		polName     = flag.String("policy", "", "the policy the live scheduler ran (its POLICY setting)")
		workers     = flag.Int("workers", 0, "live worker count (0 = the preset's)")
		workerMemGB = flag.Float64("worker_mem_gb", 0, "live WORKER_MEM_BYTES in GiB (0 = the preset's)")
		pollS       = flag.Float64("poll_s", 0.5, "shardsim POLL_INTERVAL, wall seconds")
		reconsiderS = flag.Float64("reconsider_s", 2, "scheduler RECONSIDER_INTERVAL, wall seconds")
		leaseS      = flag.Float64("lease_s", 10, "worker LEASE_SECONDS, wall seconds")
		tolS        = flag.Float64("tol_s", 2, "agreement tolerance, modeled seconds (the larger of this and tol_frac applies)")
		tolFrac     = flag.Float64("tol_frac", 0.05, "agreement tolerance as a fraction of the simulated round time")
		strict      = flag.Bool("strict", false, "exit 1 if any round disagrees")
		perShard    = flag.Bool("shards", true, "print the per-shard finish table")
		phases      = flag.Int("phases", 8, "replay this many reconsider-tick phases (decision 29: the k-th at k/phases of the interval) and report the round-time envelope; the main row is the default phase")
		trustPolicy = flag.Bool("trust_policy", false, "skip the check that the timelines' submit reasons carry -policy's own wording (bug 15: a round compared under the wrong policy can still agree)")
	)
	flag.Parse()
	if *dir == "" || *polName == "" {
		flag.Usage()
		os.Exit(2)
	}
	pol, err := policy.ByName(*polName)
	if err != nil {
		log.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(*dir, "round-*.json"))
	if err != nil || len(files) == 0 {
		log.Fatalf("no round-*.json in %s", *dir)
	}
	sort.Slice(files, func(i, j int) bool { return roundID(files[i]) < roundID(files[j]) })

	type row struct {
		tl           ssim.Timeline
		res          esim.Result
		liveS, simS  float64
		agree        bool
		worstShard   int
		worstDeltaS  float64
		meanAbsDelta float64
		sortedMaxAbs float64 // max |delta| after sorting both sides' finishes: order-free
		envLo, envHi float64 // round time over the replayed tick phases
		inEnvelope   bool    // live within [envLo - tol, envHi + tol]
	}
	var rows []row
	for _, f := range files {
		var tl ssim.Timeline
		b, err := os.ReadFile(f)
		if err != nil {
			log.Fatal(err)
		}
		if err := json.Unmarshal(b, &tl); err != nil {
			log.Fatalf("%s: %v", f, err)
		}
		ts := tl.TimeScale
		if ts <= 0 {
			ts = 1
		}
		sc, err := workload.PresetWithShards(tl.Scenario, len(tl.Shards))
		if err != nil {
			log.Fatal(err)
		}
		w := workload.Generate(sc, tl.Seed)
		for i, s := range w.Shards {
			if int(tl.Shards[i].ShardID) != i {
				log.Fatalf("%s: position %d holds shard_id %d — shards must be listed in id order", f, i, tl.Shards[i].ShardID)
			}
			if s.NVectors != tl.Shards[i].NVectors {
				log.Fatalf("%s: shard %d has %d vectors live but %d regenerated — not the same workload (scenario %q seed %d)",
					f, i, tl.Shards[i].NVectors, s.NVectors, tl.Scenario, tl.Seed)
			}
		}
		// Which policy decided the live round? The timeline carries no policy
		// field, but every submit event quotes the policy's own reason, and
		// each policy words it differently (bug 15 was seen through exactly
		// this). Refuse a timeline whose reasons belong to another policy.
		if !*trustPolicy {
			for _, s := range tl.Shards {
				for _, ev := range s.Events {
					m := submitReason.FindStringSubmatch(ev.Msg)
					if m == nil {
						continue
					}
					if by := policyOfReason(m[1]); by != "" && by != policyFamily(*polName) {
						log.Fatalf("%s: shard %d's submit reason %q is %s's wording, not %s's — the live round ran under another policy (bug 15); pass -trust_policy to override",
							f, s.ShardID, m[1], by, *polName)
					}
				}
			}
		}
		cfg := esim.Config{
			Workers: *workers, WorkerMemBytes: int64(*workerMemGB * float64(1<<30)),
			ShareNodeLinks: false,
			PollS:          *pollS * ts, ReconsiderS: *reconsiderS * ts, LeaseS: *leaseS * ts,
		}
		res := esim.Run(w, pol, cfg)
		r := row{tl: tl, res: res, liveS: tl.DurationModeled.Seconds(), simS: res.RoundS}
		tol := math.Max(*tolS, *tolFrac*r.simS)
		r.agree = res.Completed && math.Abs(r.liveS-r.simS) <= tol
		// The live ticker's phase relative to the round start is arbitrary
		// (decision 29): replay a spread of phases and record the envelope.
		r.envLo, r.envHi = r.simS, r.simS
		for k := 1; k <= *phases; k++ {
			pc := cfg
			pc.TickPhaseS = cfg.ReconsiderS * float64(k) / float64(*phases)
			pr := esim.Run(w, pol, pc)
			if pr.Completed {
				r.envLo, r.envHi = math.Min(r.envLo, pr.RoundS), math.Max(r.envHi, pr.RoundS)
			}
		}
		r.inEnvelope = r.liveS >= r.envLo-tol && r.liveS <= r.envHi+tol
		sum := 0.0
		for i, s := range tl.Shards {
			d := s.Finished.Seconds()*ts - res.ShardFinishS[i]
			sum += math.Abs(d)
			if math.Abs(d) > math.Abs(r.worstDeltaS) {
				r.worstDeltaS, r.worstShard = d, i
			}
		}
		r.meanAbsDelta = sum / float64(len(tl.Shards))
		// Order-free view: live builds of equal priority claim in the order
		// their submits landed (50 goroutines within ~50 ms), the simulator
		// in shard order; which shard finishes k-th is therefore a lottery,
		// but the k-th finish time itself should agree.
		live := make([]float64, len(tl.Shards))
		for i, s := range tl.Shards {
			live[i] = s.Finished.Seconds() * ts
		}
		simF := append([]float64(nil), res.ShardFinishS...)
		sort.Float64s(live)
		sort.Float64s(simF)
		for i := range live {
			r.sortedMaxAbs = math.Max(r.sortedMaxAbs, math.Abs(live[i]-simF[i]))
		}
		rows = append(rows, r)
	}

	fmt.Printf("validate: %s policy=%s workers=%d worker_mem=%.0fGiB (0 = the preset's) cadences(wall s): poll=%.2g reconsider=%.2g lease=%.2g; sharing off, 1 worker per node (irrelevant with sharing off); tolerance max(%.1f s, %.0f%%); envelope over %d tick phases\n",
		rows[0].tl.Scenario, *polName, *workers, *workerMemGB, *pollS, *reconsiderS, *leaseS, *tolS, *tolFrac*100, *phases)
	fmt.Printf("%-6s %-5s %-5s %9s %9s %8s | %9s %9s | %-11s %-11s | %-22s %-10s %-17s %s\n",
		"round", "seed", "scale", "live_s", "sim_s", "delta_s", "live_lag", "sim_lag", "live g/l", "sim g/l", "worst shard (live-sim)", "sorted max", "phase envelope", "verdict")
	agreeN := 0
	var sumDelta, maxAbs float64
	for _, r := range rows {
		verdict := "DISAGREE"
		if r.agree {
			verdict = "agree"
			agreeN++
		} else if r.inEnvelope {
			verdict = "agree within tick-phase envelope"
			agreeN++
		}
		if !r.res.Completed {
			verdict = "SIM INCOMPLETE"
		}
		d := r.liveS - r.simS
		sumDelta += d
		maxAbs = math.Max(maxAbs, math.Abs(d))
		fmt.Printf("%-6d %-5d %-5g %9.2f %9.2f %+8.2f | %9.2f %9.2f | %-11s %-11s | shard %-2d %+8.2f s      %7.2f s  %7.2f..%-7.2f %s\n",
			r.tl.RoundID, r.tl.Seed, r.tl.TimeScale, r.liveS, r.simS, d,
			r.tl.StragglerLag.Seconds()*r.tl.TimeScale, r.res.StragglerLagS,
			fmt.Sprintf("%d/%d", r.tl.BuildsGPU, r.tl.BuildsLocal), fmt.Sprintf("%d/%d", r.res.BuildsGPU, r.res.BuildsLocal),
			r.worstShard, r.worstDeltaS, r.sortedMaxAbs, r.envLo, r.envHi, verdict)
	}
	fmt.Printf("summary: %d/%d rounds agree; live - sim: mean %+.2f s, max |delta| %.2f s\n", agreeN, len(rows), sumDelta/float64(len(rows)), maxAbs)

	if *perShard {
		for _, r := range rows {
			ts := r.tl.TimeScale
			fmt.Printf("\nround %d seed %d, per shard (modeled s): shard  n_vectors  live place/att  sim place  live finish   sim finish   delta\n", r.tl.RoundID, r.tl.Seed)
			for i, s := range r.tl.Shards {
				live := s.Finished.Seconds() * ts
				simPlace := "?"
				if i < len(r.res.ShardPlacement) && r.res.ShardPlacement[i] != "" {
					simPlace = r.res.ShardPlacement[i]
				}
				fmt.Printf("                                           %-6d %-10d %-5s/%-9d %-9s %11.2f %12.2f %+8.2f\n",
					s.ShardID, s.NVectors, s.Placement, s.Attempt, simPlace, live, r.res.ShardFinishS[i], live-r.res.ShardFinishS[i])
			}
			fmt.Printf("                                           mean |delta| %.2f s by shard, max |delta| %.2f s with both sides' finishes sorted (order-free); sim: preempts=%d reneges=%d speculations=%d\n",
				r.meanAbsDelta, r.sortedMaxAbs, r.res.Preempts, r.res.Reneges, r.res.Speculations)
		}
	}
	if *strict && agreeN != len(rows) {
		os.Exit(1)
	}
}

func roundID(path string) int {
	base := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(path), "round-"), ".json")
	var id int
	fmt.Sscanf(base, "%d", &id)
	return id
}

// submitReason captures the policy's reason from the shard's submit event
// ("submitted: placement=gpu (always_gpu); cpu_build=...").
var submitReason = regexp.MustCompile(`^submitted: placement=\w+ \((.*)\); cpu_build=`)

// policyOfReason names the policy family whose Decide writes this reason, or
// "" when the wording is shared (the memory filter's "needs N bytes, largest
// live worker has M" is written by every policy).
func policyOfReason(r string) string {
	switch {
	case r == "always_gpu":
		return "always_gpu"
	case strings.HasPrefix(r, "oracle:"):
		return "clairvoyant"
	case strings.Contains(r, "; reconsider "):
		return "reactive"
	case strings.Contains(r, " threshold "):
		return "static_threshold"
	}
	return ""
}

// policyFamily folds the configured name to the family that writes its
// submit reasons: reactive_spec decides submits exactly like reactive.
func policyFamily(name string) string {
	base, _, _ := strings.Cut(name, ":")
	if base == "reactive_spec" {
		return "reactive"
	}
	if base == "" {
		return "always_gpu"
	}
	return base
}
