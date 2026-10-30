// sweep is the seeded experiment runner (Phase 3 step 5): it crosses
// scenarios x policies x sweep axes x seeds, runs each cell through the
// discrete-event simulator, and writes ONE CSV ROW PER ROUND. Runs are
// sequential and seeded, so the same invocation writes a byte-identical CSV.
//
//	go run ./experiments/cmd/sweep -scenarios skewed,uniform -policies all -rounds 20 -out rounds.csv
//	go run ./experiments/cmd/sweep -scenarios skewed -sigma 0,0.3,0.6 -rounds 20 -out sigma.csv
//
// Every axis flag takes a comma list and multiplies the grid, which is how
// the roadmap's sweeps run one axis at a time (workers, workers per node,
// zipf exponent, estimate-error sigma, query load, needs_index fraction,
// shard count, dimension, GPU speedup). "clairvoyant" as a policy constructs
// the oracle with each workload's future streams — the one consumer allowed
// to (decision 1a). -kill_at injects the shrinking-pool worker death.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

// Columns is the CSV schema, one row per simulated round. Overridden axes
// echo their value; non-overridden ones echo the scenario's own, and every
// knob that can change a result is a column — reconsider_s, poll_s, lease_s
// and noise_seed included (cross-check F2) — so a row is self-describing and
// two CSVs concatenate safely. The zipf column echoes the preset's exponent
// field and reads 0 when the size distribution is not zipf.
var columns = []string{
	"scenario", "policy", "seed",
	"n_shards", "dim", "workers", "workers_per_node", "worker_mem_bytes", "share_links",
	"sigma", "zipf", "load", "needs_index", "speedup", "kill_at",
	"cost_a", "bandwidth", "gpu_overhead",
	"reconsider_s", "poll_s", "lease_s", "noise_seed",
	"completed", "round_s", "straggler_s",
	"builds_gpu", "builds_local",
	"preempts", "reneges", "reprioritizes",
	"speculations", "spec_gpu_wins", "spec_local_wins", "wasted_gpu_s", "wasted_cpu_s",
	"queue_wait_p50_s", "queue_wait_p99_s", "events",
}

func main() {
	var (
		scenarios   = flag.String("scenarios", "skewed", "comma list of workload presets, or 'all'")
		policies    = flag.String("policies", "all", "comma list: always_gpu, static_threshold[:N], reactive, reactive_spec, clairvoyant, or 'all'")
		rounds      = flag.Int("rounds", 10, "seeded rounds per grid cell")
		seed0       = flag.Int64("seed0", 1, "first seed; cell c uses seed0..seed0+rounds-1")
		out         = flag.String("out", "-", "output CSV path, '-' for stdout")
		shards      = flag.Int("shards", 0, "override the scenario's shard count (0 = preset)")
		workers     = flag.String("workers", "", "comma list: override the pool's worker count")
		workerMemGB = flag.Float64("worker_mem_gb", 0, "override the pool's per-worker memory, GiB (0 = preset)")
		perNode     = flag.String("workers_per_node", "1", "comma list: workers sharing one node's link")
		shareLinks  = flag.Bool("share_links", true, "transfers share a node's link (off mimics the live fake worker)")
		sigmas      = flag.String("sigma", "0", "comma list: log-normal estimate-error sigma")
		zipfs       = flag.String("zipf", "", "comma list: override the zipf size exponent")
		loads       = flag.String("load", "1", "comma list: multiply the scenario's query rate")
		needsIdx    = flag.String("needs_index", "", "comma list: override the needs_index fraction")
		dims        = flag.String("dim", "", "comma list: override the vector dimension")
		speedups    = flag.String("speedup", "", "comma list: override the cost model's GPU speedup")
		costA       = flag.Float64("cost_a", 0, "override the cost model's A (s per n·ln(n)·dim): a calibrated constant (Phase 4 step 3's fit)")
		bandwidth   = flag.Float64("bandwidth", 0, "override the cost model's transfer bandwidth, bytes/s (calibrated)")
		gpuOverhead = flag.Float64("gpu_overhead", -1, "override the cost model's fixed GPU overhead, s (calibrated; -1 = the default)")
		killAt      = flag.Float64("kill_at", 0, "kill the busiest worker at this modeled second (shrinking pool)")
		reconsider  = flag.Float64("reconsider_s", 2, "reconsider tick interval, modeled seconds")
		leaseS      = flag.Float64("lease_s", 30, "a dead worker's lease requeues after this many modeled seconds")
		poll        = flag.Float64("poll_s", 0.5, "shard report interval, modeled seconds")
		noiseSeed   = flag.Int64("noise_seed", 1, "sub-seed for the estimate noise draws")
	)
	flag.Parse()

	scList := splitOr(*scenarios, workload.PresetNames())
	polList := splitOr(*policies, []string{"always_gpu", "static_threshold", "reactive", "reactive_spec", "clairvoyant"})

	w := csv.NewWriter(os.Stdout)
	if *out != "-" {
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			log.Fatal(err)
		}
		f, err := os.Create(*out)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		w = csv.NewWriter(f)
	}
	defer w.Flush()
	if err := w.Write(columns); err != nil {
		log.Fatal(err)
	}

	rowsWritten := 0
	for _, scName := range scList {
		for _, sigma := range floats(*sigmas) {
			for _, zipf := range floatsOr(*zipfs, -1) {
				for _, load := range floats(*loads) {
					for _, ni := range floatsOr(*needsIdx, -1) {
						for _, dim := range intsOr(*dims, 0) {
							for _, spd := range floatsOr(*speedups, -1) {
								for _, nw := range intsOr(*workers, 0) {
									for _, wpn := range intsOr(*perNode, 1) {
										for _, polName := range polList {
											for r := 0; r < *rounds; r++ {
												seed := *seed0 + int64(r)
												row, err := runCell(cell{
													scenario: scName, policyName: polName, seed: seed,
													shards: *shards, workers: nw, workerMemGB: *workerMemGB,
													perNode: wpn, shareLinks: *shareLinks,
													sigma: sigma, zipf: zipf, load: load, needsIdx: ni,
													dim: int32(dim), speedup: spd, killAt: *killAt,
													costA: *costA, bandwidth: *bandwidth, gpuOverhead: *gpuOverhead,
													reconsiderS: *reconsider, pollS: *poll, leaseS: *leaseS, noiseSeed: *noiseSeed,
												})
												if err != nil {
													log.Fatal(err)
												}
												if err := w.Write(row); err != nil {
													log.Fatal(err)
												}
												rowsWritten++
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	w.Flush()
	fmt.Fprintf(os.Stderr, "sweep: %d rows\n", rowsWritten)
}

type cell struct {
	scenario, policyName string
	seed                 int64
	shards, workers      int
	workerMemGB          float64
	perNode              int
	shareLinks           bool
	sigma, zipf, load    float64
	needsIdx             float64
	dim                  int32
	speedup              float64
	costA, bandwidth     float64 // 0 = the cost model's defaults
	gpuOverhead          float64 // -1 = the default
	killAt               float64
	reconsiderS, pollS   float64
	leaseS               float64
	noiseSeed            int64
}

func runCell(c cell) ([]string, error) {
	n := workload.DefaultShards
	if c.shards > 0 {
		n = c.shards
	}
	sc, err := workload.PresetWithShards(c.scenario, n)
	if err != nil {
		return nil, err
	}
	if c.zipf >= 0 && sc.Sizes.Kind == "zipf" {
		sc.Sizes.ZipfExponent = c.zipf
	}
	sc.Stream.ClusterQueryRate *= c.load
	if c.needsIdx >= 0 {
		sc.Stream.NeedsIndexFraction = c.needsIdx
	}
	if c.dim > 0 {
		sc.Dim = c.dim
	}
	w := workload.Generate(sc, c.seed)

	cm := costmodel.Default()
	if c.speedup > 0 {
		cm.Speedup = c.speedup
	}
	if c.costA > 0 {
		cm.A = c.costA
	}
	if c.bandwidth > 0 {
		cm.Bandwidth = c.bandwidth
	}
	if c.gpuOverhead >= 0 {
		cm.GPUOverhead = c.gpuOverhead
	}
	cfg := sim.Config{
		CostModel: &cm, WorkersPerNode: c.perNode, ShareNodeLinks: c.shareLinks,
		ReconsiderS: c.reconsiderS, PollS: c.pollS, LeaseS: c.leaseS,
		EstimateSigma: c.sigma, NoiseSeed: c.noiseSeed, KillWorkerAtS: c.killAt,
	}
	if c.workers > 0 {
		cfg.Workers = c.workers
	}
	if c.workerMemGB > 0 {
		cfg.WorkerMemBytes = int64(c.workerMemGB * (1 << 30))
	}

	var pol policy.Policy
	if c.policyName == "clairvoyant" {
		pol = policy.NewClairvoyant(sim.Futures(w))
	} else {
		pol, err = policy.ByName(c.policyName)
		if err != nil {
			return nil, err
		}
	}
	res := sim.Run(w, pol, cfg)

	mem := cfg.WorkerMemBytes
	if mem == 0 {
		mem = sc.Pool.WorkerMemBytes
	}
	nw := cfg.Workers
	if nw == 0 {
		nw = sc.Pool.Workers
	}
	f := func(x float64) string { return strconv.FormatFloat(x, 'g', 10, 64) }
	return []string{
		sc.Name, pol.Name(), strconv.FormatInt(c.seed, 10),
		strconv.Itoa(sc.NShards), strconv.Itoa(int(sc.Dim)), strconv.Itoa(nw), strconv.Itoa(c.perNode),
		strconv.FormatInt(mem, 10), strconv.FormatBool(c.shareLinks),
		f(c.sigma), f(sc.Sizes.ZipfExponent), f(c.load), f(sc.Stream.NeedsIndexFraction), f(cm.Speedup), f(c.killAt),
		f(cm.A), f(cm.Bandwidth), f(cm.GPUOverhead),
		f(c.reconsiderS), f(c.pollS), f(c.leaseS), strconv.FormatInt(c.noiseSeed, 10),
		strconv.FormatBool(res.Completed), f(res.RoundS), f(res.StragglerLagS),
		strconv.Itoa(res.BuildsGPU), strconv.Itoa(res.BuildsLocal),
		strconv.Itoa(res.Preempts), strconv.Itoa(res.Reneges), strconv.Itoa(res.Reprioritizes),
		strconv.Itoa(res.Speculations), strconv.Itoa(res.SpecGPUWins), strconv.Itoa(res.SpecLocalWins), f(res.WastedGPUS), f(res.WastedCPUS),
		f(pct(res.QueueWaits, 0.5)), f(pct(res.QueueWaits, 0.99)), strconv.Itoa(res.Events),
	}, nil
}

// pct is the timeline's percentile: sorted[int(p*(n-1))], 0 for no samples.
func pct(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[int(p*float64(len(s)-1))]
}

func splitOr(s string, all []string) []string {
	if s == "all" {
		return all
	}
	return strings.Split(s, ",")
}

func floats(s string) []float64 {
	var out []float64
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			log.Fatalf("bad float %q", p)
		}
		out = append(out, v)
	}
	return out
}

// floatsOr: an empty flag means "don't override": one cell with the sentinel.
func floatsOr(s string, sentinel float64) []float64 {
	if s == "" {
		return []float64{sentinel}
	}
	return floats(s)
}

func intsOr(s string, sentinel int) []int {
	if s == "" {
		return []int{sentinel}
	}
	var out []int
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			log.Fatalf("bad int %q", p)
		}
		out = append(out, v)
	}
	return out
}
