// Package costmodel is the cost model: how long a build takes, on a shard's
// CPU or on a worker, and how much GPU memory it needs, as pure functions of
// n_vectors and dim.
//
// Version 1 (Phase 5 decision 6, GPU-free step 1). Phase 4's calibration on
// the cloud topology found v0's shape wrong in three places — time grows
// like n^1.2·ln(n), not n·ln(n); the per-dimension cost is sub-linear on
// x86 (a 960-dim build costs 2.8× a 128-dim one, not 7.5×); and a worker's
// path has a gate, a save and an upload three times slower than the fetch,
// which v0 did not have (Phase 4 decisions 35, 37) — so every term is now
// its own, with a constant the calibration measures:
//
//	CPU build     = A · n^P · ln(n) · (dim + D0)
//	gate          = Gate · GateQueries · ln(n) · (dim + D0)   (recall check of the finished index)
//	save          = index bytes / SaveBandwidth                (write + hash before publishing)
//	fetch         = vector bytes / Bandwidth                    (the store → the builder)
//	upload        = index bytes / BandwidthOut                  (the builder → the store)
//	GPU build     = (CPU build + gate) / Speedup + GPUOverhead + save          (v1: a CPU worker)
//	              = WorkerFixed + WorkerA · n^WorkerP · ln(n) · (dim + WorkerD0)  (v2, when WorkerA > 0: the
//	                + WorkerGate · GateQueries · ln(n) · (dim + WorkerD0)        worker's own shape — a GPU's
//	                + GPUOverhead + save                                          floor and a different curve)
//	transfer      = fetch + upload
//	GPU total     = GPU build + transfer                        (a worker's whole path)
//	local total   = CPU build + gate + save [+ fetch + upload if ShardUsesStore]
//	vector bytes  = n · dim · 4                                 (float32)
//	index bytes   = n · (dim·4 + 2·M·4)                         (vectors + M links each way)
//	GPU memory    = MemFactor · vector bytes
//
// Default() is still v0 (P = 1, D0 = 0, no gate, no save, one bandwidth):
// every Phase 1–3 test and sweep keeps its numbers. Calibrated() is the
// CPU-worker topology's measured block (Phase 4 evidence 7, Phase 5 evidence
// 9); CalibratedL4() the GPU topology's (Phase 5 decision 14, evidence 10,
// 11), where the worker has its own shape instead of a speedup — the first
// GPU session showed the card's time is a floor plus a nearly linear term,
// against the shard's n^1.22·ln(n), so no single ratio fits. The Python
// worker reimplements the same formulas; keep the two in step.
//
// The two buckets the estimator and the simulator see: GPUBuild is what the
// worker's CPU/GPU does (build, gate, overhead, save — the save is disk, not
// link, and is counted here so that Transfer stays "the link"), Transfer is
// what crosses the network and may share a node's link (Phase 3 decision 24).
package costmodel

import (
	"fmt"
	"math"
	"time"
)

type Model struct {
	A              float64 // seconds per (n^P · ln n · (dim + D0)) on a shard CPU
	P              float64 // the exponent on n; 1 is v0's n·ln(n)
	D0             float64 // the dimension offset: the per-vector cost that does not scale with dim; 0 is v0
	Speedup        float64 // worker build = shard build / Speedup ...
	GPUOverhead    float64 // ... + this many seconds (process start, manifest, the rest)
	GateCost       float64 // seconds per (query · ln n · (dim + D0)) on a shard CPU; 0 = no gate term (v0)
	GateQueries    float64 // held-out queries the gate runs (a rig setting; the split's query file)
	SaveBandwidth  float64 // bytes/s writing and hashing the index; 0 = no save term (v0)
	Bandwidth      float64 // bytes/s fetching vectors from the store
	BandwidthOut   float64 // bytes/s uploading the index to the store; 0 = same as Bandwidth (v0)
	ShardUsesStore bool    // the shard fetches its vectors from the store and uploads its index (this rig; a database would not)
	WorkerFixed    float64 // v2: the worker's per-build floor, s (a GPU's context, launch and conversion overhead)
	WorkerA        float64 // v2: the worker's own A; 0 = use the Speedup form (v1)
	WorkerP        float64 // v2: the worker's own exponent on n
	WorkerD0       float64 // v2: the worker's own dimension offset
	WorkerGate     float64 // v2: the worker's gate cost per (query · ln n · (dim + WorkerD0))
	M              int     // HNSW links per node, for the index size
	MemFactor      float64 // GPU memory need as a multiple of the vector bytes
	BytesPerDim    int     // 4 for float32
}

// Default is cost model v0, tuned so that a 100k x 128 build takes about
// 20 s on a CPU and about 4 s on a GPU, which keeps a 6-shard round under a
// minute. It is the placeholder the Phase 1–3 tests and sweeps were written
// against; the measured model is Calibrated().
func Default() Model {
	return Model{
		A:           1.4e-7,
		P:           1,
		D0:          0,
		Speedup:     10,
		GPUOverhead: 2,
		Bandwidth:   200e6,
		M:           16,
		MemFactor:   2,
		BytesPerDim: 4,
	}
}

// Calibrated is the cloud topology's measured block (Phase 4 step 4, both
// datasets, the shard VM's 2-thread fit for the shard side and the worker
// node's 8-thread fit for the worker side; Phase 5 step 1 refit): the
// numbers the cluster runs with and the comparison is rerun on. Bandwidths:
// the fetch is the worker's median; the upload is the store's measured
// upload rate (46–54 MB/s on the data step); the save bandwidth is the
// laptop rehearsal's until the extended calibration point runs in the cloud.
// A GPU replaces Speedup and GPUOverhead in Phase 5 proper.
func Calibrated() Model {
	return Model{
		A:              2.30e-9,
		P:              1.220,
		D0:             187,
		Speedup:        2.12,
		GPUOverhead:    1.5,
		GateCost:       1.79e-8,
		GateQueries:    10000,
		SaveBandwidth:  6.6e8,
		Bandwidth:      1.57e8,
		BandwidthOut:   5.0e7,
		ShardUsesStore: true,
		M:              16,
		MemFactor:      2,
		BytesPerDim:    4,
	}
}

// CalibratedL4 is the GPU topology's measured block (Phase 5 decision 14;
// the first GPU session, evidence 10): the shard side as Calibrated(), the
// worker an NVIDIA L4 building CAGRA and converting to HNSW on 6 CPU threads,
// fitted with a fixed term (19 points, the process's warm-up point dropped,
// sigma 0.097); the card's fetch, upload and save bandwidths. Speedup is
// kept only as a fallback value and is not used while WorkerA > 0.
func CalibratedL4() Model {
	m := Calibrated()
	m.WorkerFixed = 1.154
	m.WorkerA = 1.295e-8
	m.WorkerP = 0.891
	m.WorkerD0 = 154
	m.WorkerGate = 2.07e-8
	m.Bandwidth = 1.39e8
	m.BandwidthOut = 1.01e8
	m.SaveBandwidth = 2.9e8
	return m
}

func (m Model) VectorBytes(n int64, dim int32) int64 {
	return n * int64(dim) * int64(m.BytesPerDim)
}

func (m Model) IndexBytes(n int64, dim int32) int64 {
	return n * (int64(dim)*int64(m.BytesPerDim) + int64(2*m.M*4))
}

func (m Model) GPUMemBytes(n int64, dim int32) int64 {
	return int64(math.Ceil(m.MemFactor * float64(m.VectorBytes(n, dim))))
}

// shape is n^P · ln(n) · (dim + D0), the size term shared by the build and the gate.
func (m Model) shape(n int64, dim int32) float64 {
	p := m.P
	if p == 0 {
		p = 1
	}
	return math.Pow(float64(n), p) * math.Log(float64(n)) * (float64(dim) + m.D0)
}

// CPUBuild is the build alone on a shard's CPU.
func (m Model) CPUBuild(n int64, dim int32) time.Duration {
	if n < 2 {
		return 0
	}
	return secs(m.A * m.shape(n, dim))
}

// Gate is the recall check of the finished index on a shard's CPU: GateQueries
// searches, each a graph walk whose cost grows with ln(n) and the distance's
// width. Zero when the model has no gate term (v0).
func (m Model) Gate(n int64, dim int32) time.Duration {
	if n < 2 || m.GateCost == 0 || m.GateQueries == 0 {
		return 0
	}
	return secs(m.GateCost * m.GateQueries * math.Log(float64(n)) * (float64(dim) + m.D0))
}

// Save is writing the index to disk and hashing it before it is published.
func (m Model) Save(n int64, dim int32) time.Duration {
	if m.SaveBandwidth <= 0 {
		return 0
	}
	return secs(float64(m.IndexBytes(n, dim)) / m.SaveBandwidth)
}

// Fetch is the vectors from the store to the builder.
func (m Model) Fetch(n int64, dim int32) time.Duration {
	return secs(float64(m.VectorBytes(n, dim)) / m.Bandwidth)
}

// Upload is the index from the builder to the store.
func (m Model) Upload(n int64, dim int32) time.Duration {
	bw := m.BandwidthOut
	if bw <= 0 {
		bw = m.Bandwidth
	}
	return secs(float64(m.IndexBytes(n, dim)) / bw)
}

// WorkerBuild is the worker's own build + gate when it has its own shape
// (v2, WorkerA > 0); 0 otherwise. A build of fewer than 2 vectors pays the
// floor and nothing else.
func (m Model) WorkerBuild(n int64, dim int32) time.Duration {
	if m.WorkerA <= 0 {
		return 0
	}
	if n < 2 {
		return secs(m.WorkerFixed)
	}
	p := m.WorkerP
	if p == 0 {
		p = 1
	}
	ln := math.Log(float64(n))
	w := float64(dim) + m.WorkerD0
	build := m.WorkerA * math.Pow(float64(n), p) * ln * w
	gate := 0.0
	if m.WorkerGate > 0 && m.GateQueries > 0 {
		gate = m.WorkerGate * m.GateQueries * ln * w
	}
	return secs(m.WorkerFixed + build + gate)
}

// GPUBuild is the worker's compute: with its own shape (v2) the floor, its
// build and its gate; otherwise (v1) the shard's build and gate at the
// worker's speed; plus the fixed overhead and the save either way.
func (m Model) GPUBuild(n int64, dim int32) time.Duration {
	if m.WorkerA > 0 {
		return secs(m.WorkerBuild(n, dim).Seconds() + m.GPUOverhead + m.Save(n, dim).Seconds())
	}
	return secs((m.CPUBuild(n, dim).Seconds()+m.Gate(n, dim).Seconds())/m.Speedup + m.GPUOverhead + m.Save(n, dim).Seconds())
}

// Transfer is what crosses the link: the vectors in and the index out. With
// one bandwidth it is computed as v0 did, in one expression, so that the
// nanosecond truncation happens once and v0's numbers hold to the digit
// (decision 7; the step 1 cross-check's A found a 1–2 ns drift otherwise).
func (m Model) Transfer(n int64, dim int32) time.Duration {
	if m.BandwidthOut <= 0 {
		return secs(float64(m.VectorBytes(n, dim)+m.IndexBytes(n, dim)) / m.Bandwidth)
	}
	return m.Fetch(n, dim) + m.Upload(n, dim)
}

// Validate refuses a model no estimate can be made with: a non-positive A, P,
// Speedup, Bandwidth or MemFactor, a negative offset or term, or any NaN or
// infinity (the step 1 cross-check's C and D: a speedup of 0 makes the GPU
// time infinite, and NaN passes every "< 0" check).
func (m Model) Validate() error {
	checks := []struct {
		name   string
		v      float64
		min    float64 // the smallest legal value, exclusive when strict
		strict bool
	}{
		{"COST_A", m.A, 0, true}, {"COST_P", m.P, 0, true}, {"COST_D0", m.D0, 0, false},
		{"COST_SPEEDUP", m.Speedup, 0, true}, {"COST_GPU_OVERHEAD", m.GPUOverhead, 0, false},
		{"COST_GATE", m.GateCost, 0, false}, {"COST_GATE_QUERIES", m.GateQueries, 0, false},
		{"COST_SAVE_BANDWIDTH", m.SaveBandwidth, 0, false}, {"COST_BANDWIDTH", m.Bandwidth, 0, true},
		{"COST_BANDWIDTH_OUT", m.BandwidthOut, 0, false}, {"COST_MEM_FACTOR", m.MemFactor, 0, true},
		{"COST_WORKER_FIXED", m.WorkerFixed, 0, false}, {"COST_WORKER_A", m.WorkerA, 0, false}, {"COST_WORKER_P", m.WorkerP, 0, false},
		{"COST_WORKER_D0", m.WorkerD0, 0, false}, {"COST_WORKER_GATE", m.WorkerGate, 0, false},
	}
	for _, c := range checks {
		if math.IsNaN(c.v) || math.IsInf(c.v, 0) {
			return fmt.Errorf("%s must be a finite number, got %g", c.name, c.v)
		}
		if (c.strict && c.v <= c.min) || (!c.strict && c.v < c.min) {
			if c.strict {
				return fmt.Errorf("%s must be positive, got %g", c.name, c.v)
			}
			return fmt.Errorf("%s must be non-negative, got %g", c.name, c.v)
		}
	}
	return nil
}

// GPUTotal is a worker's whole path, what a fake worker sleeps for.
func (m Model) GPUTotal(n int64, dim int32) time.Duration {
	return m.GPUBuild(n, dim) + m.Transfer(n, dim)
}

// LocalTotal is a shard's whole path for a local build: the build, the gate
// and the save on its own CPU, plus the store round trip when the rig makes
// the shard fetch its vectors and upload its index (ShardUsesStore). What a
// fake shard sleeps for, and the policies' local estimate.
func (m Model) LocalTotal(n int64, dim int32) time.Duration {
	d := m.CPUBuild(n, dim) + m.Gate(n, dim) + m.Save(n, dim)
	if m.ShardUsesStore {
		d += m.Transfer(n, dim)
	}
	return d
}

func secs(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }
