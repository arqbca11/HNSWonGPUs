// Package sim is the discrete-event simulator (Phase 3 steps 4–6): it plays
// one seeded workload round against one placement policy in modeled time —
// the clock jumps from event to event instead of sleeping — so a 200-round
// sweep takes seconds instead of days.
//
// It calls the SAME policy code the live scheduler calls (invariant 8:
// policies are pure): the same Decide at submit, the same Reconsider on a
// modeled tick, the same ReplayQueue/NewJobWait estimates, with the host-side
// mappings mirrored from scheduler/api (submit) and scheduler/reconsider (the
// tick). What the simulator reproduces of the live system, and where it
// deliberately differs, is decision 21 in docs/design/phase3-policies.md:
//
//   - Policies see only REPORTED state: each shard publishes its load every
//     PollS (like the live poll interval) and reacts to placement changes at
//     its own next report, so the report-latency the live system has is in
//     the model. Between reports a policy's view is stale, exactly as live.
//   - Claims mirror the claim statement: highest priority first, enqueued
//     order within (seq assigned at row insert, kept across a preemption,
//     like enqueued_at), memory-aware; simultaneous frees resolve in
//     ascending worker ID (the estimator's pinned order).
//   - The reconsider tick runs every ReconsiderS of modeled time: settle
//     speculation races first, then snapshot, then the policy's actions as
//     guarded transitions (a stale action is skipped, as a zero-row UPDATE).
//   - Transfers optionally share a node's link as a fluid (decision 22):
//     ShareNodeLinks on models what real GPUs on one machine will see
//     (decision 12's model); off models the Phase 1–3 fake worker, which
//     sleeps its whole GPUTotal without contending — use off when comparing
//     against the live cluster (step 7).
//   - Estimate error (decision 23): every estimate a policy sees is the true
//     duration times a seeded log-normal factor, drawn once per build and
//     per estimate kind, so a build's misestimate persists across ticks.
//     Execution always uses the truth. The sweep's σ is Config.EstimateSigma.
//
// Known simplifications (also decision 21): submit is instantaneous (no HTTP),
// reports are not staggered across shards, a killed worker leaves the pool
// immediately (live: after WORKER_STALE_AFTER) and a cancelled copy frees its
// worker at the tick (live: at the worker's next failed renew).
package sim

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

// Config is one run's knobs. Zero values mean: pool from the workload's
// scenario, cost model Default(), ReconsiderS 2, PollS 0.5, LeaseS 30,
// WorkersPerNode 1, no noise, no kill.
type Config struct {
	CostModel      *costmodel.Model
	Workers        int
	WorkerMemBytes int64
	WorkersPerNode int  // workers i share node i/WorkersPerNode
	ShareNodeLinks bool // fluid per-node transfer sharing (decision 22)
	ReconsiderS    float64
	PollS          float64
	LeaseS         float64 // how long a dead worker's lease takes to requeue
	EstimateSigma  float64 // log-normal σ on every estimate policies see (decision 23)
	NoiseSeed      int64   // sub-seed for the noise draws; the workload seed is mixed in
	KillWorkerAtS  float64 // >0: at this time the busiest worker dies (shrinking_pool)
	MaxModeledS    float64 // safety cap; default 6 h modeled
}

func (c Config) withDefaults(w workload.Workload) Config {
	if c.CostModel == nil {
		m := costmodel.Default()
		c.CostModel = &m
	}
	if c.Workers == 0 {
		c.Workers = w.Scenario.Pool.Workers
	}
	if c.WorkerMemBytes == 0 {
		c.WorkerMemBytes = w.Scenario.Pool.WorkerMemBytes
	}
	if c.WorkersPerNode <= 0 {
		c.WorkersPerNode = 1
	}
	if c.ReconsiderS <= 0 {
		c.ReconsiderS = 2
	}
	if c.PollS <= 0 {
		c.PollS = 0.5
	}
	if c.LeaseS <= 0 {
		c.LeaseS = 30
	}
	if c.MaxModeledS <= 0 {
		// Big enough for the production preset's worst case (a 10M x 1536
		// local build is ~9.6 modeled hours); small enough to end a genuine
		// hang (an empty pool holding a queued build) quickly in wall time.
		c.MaxModeledS = 24 * 3600
	}
	return c
}

// Result is one simulated round, the numbers one CSV row carries.
type Result struct {
	Completed     bool    // false: the safety cap fired (e.g. zero workers forever)
	RoundS        float64 // decision 30 semantics: latest real-build or query finish
	StragglerLagS float64 // last shard's finish minus the median shard's (interpolated)
	BuildsGPU     int     // real builds that finished on the GPU path
	BuildsLocal   int     // real builds that finished locally
	Preempts      int
	Reneges       int
	Reprioritizes int
	Speculations  int // copies created
	SpecGPUWins   int
	SpecLocalWins int
	WastedGPUS    float64   // worker-seconds spent on losing copies
	QueueWaits    []float64 // per claimed real build: first claim minus submit
	ShardFinishS  []float64 // per shard: max(build finish, last query finish)
	Events        int
}

const eps = 1e-9

type build struct {
	id          string
	shard       int32
	n           int64
	dim         int32
	mem         int64
	est         policy.Estimates // noised: what every policy and estimator sees
	trueLocalS  float64          // what execution actually takes
	trueGPUS    float64
	trueXferS   float64
	placement   string // local | gpu
	state       string // running | queued | leased | done | failed
	priority    float64
	moved       bool
	speculative bool
	seq         int // insert order: the claim's enqueued_at tiebreak, kept across moves
	submitAt    float64
	firstClaim  float64 // -1 until a worker claims it
	finishAt    float64
	copyB       *build // primary -> its copy; non-nil also after the copy failed (one ever)
	primary     *build // copy -> primary
}

type worker struct {
	id          string
	node        string
	mem         int64
	alive       bool
	cur         *build
	claimAt     float64
	xferRem     float64 // >0: transfer phase, full-rate seconds remaining
	buildDoneAt float64 // valid once the transfer is done
}

const (
	taskIdle  = -1
	taskBuild = -2
)

type shard struct {
	id         int32
	ddlAt      float64
	queries    []workload.Query
	durPrefix  []float64 // prefix sums of durations, for the reported mean
	next       int       // first not-yet-arrived query
	queue      []int     // arrived, waiting query indices (arrival order)
	task       int       // taskIdle, taskBuild, or a running query's index
	taskDone   float64
	buildStart float64
	localWant  bool
	indexReady bool // what the SHARD knows (own finish, or a report reply)
	b          *build
	nextReport float64
	rep        struct {
		valid      bool
		queueDepth int32
		waiting    int32
		oldestS    float64
		progress   float64
		arrivals   int
		meanDurS   float64
	}
	lastQueryFin float64
}

type engine struct {
	cfg     Config
	cm      costmodel.Model
	pol     policy.Policy
	t       float64
	shards  []*shard
	workers []*worker
	builds  []*build // insert order; copies included
	nextSeq int
	requeue []struct {
		at float64
		b  *build
	}
	killPending bool
	nextTick    float64
	res         Result
}

// Run plays one round. Deterministic: the same workload, policy and config
// give the same Result, whatever the host.
func Run(w workload.Workload, pol policy.Policy, cfg Config) Result {
	cfg = cfg.withDefaults(w)
	e := &engine{cfg: cfg, cm: *cfg.CostModel, pol: pol, killPending: cfg.KillWorkerAtS > 0}
	e.nextTick = cfg.ReconsiderS // like the live ticker: the first tick fires after one interval
	e.res.QueueWaits = []float64{}

	// Noise factors (decision 23): one per build and estimate kind, drawn in
	// shard order from a seeded PCG, so a run is reproducible and a build's
	// misestimate is consistent across every tick that looks at it.
	noise := make([][3]float64, len(w.Shards))
	for i := range noise {
		noise[i] = [3]float64{1, 1, 1}
	}
	if cfg.EstimateSigma > 0 {
		rng := rand.New(rand.NewPCG(uint64(cfg.NoiseSeed)^uint64(w.Seed), 0x6b67_7075_7369_6d21))
		for i := range noise {
			for k := 0; k < 3; k++ {
				noise[i][k] = math.Exp(cfg.EstimateSigma * rng.NormFloat64())
			}
		}
	}

	for i := 0; i < cfg.Workers; i++ {
		e.workers = append(e.workers, &worker{
			id:   fmt.Sprintf("w%02d", i),
			node: fmt.Sprintf("node%02d", i/cfg.WorkersPerNode),
			mem:  cfg.WorkerMemBytes, alive: true,
		})
	}
	for i, sw := range w.Shards {
		s := &shard{id: sw.ShardID, ddlAt: sw.DDLOffset.Seconds(), queries: sw.Queries, task: taskIdle}
		s.durPrefix = make([]float64, len(sw.Queries)+1)
		for j, q := range sw.Queries {
			s.durPrefix[j+1] = s.durPrefix[j] + q.Duration.Seconds()
		}
		dim := w.Scenario.Dim
		n := sw.NVectors
		b := &build{
			id: fmt.Sprintf("%d:sim", sw.ShardID), shard: sw.ShardID,
			n: n, dim: dim,
			mem:        e.cm.GPUMemBytes(n, dim),
			trueLocalS: e.cm.CPUBuild(n, dim).Seconds(),
			trueGPUS:   e.cm.GPUBuild(n, dim).Seconds(),
			trueXferS:  e.cm.Transfer(n, dim).Seconds(),
			firstClaim: -1,
		}
		b.est = policy.Estimates{
			LocalBuildS: b.trueLocalS * noise[i][0],
			GPUBuildS:   b.trueGPUS * noise[i][1],
			TransferS:   b.trueXferS * noise[i][2],
		}
		s.b = b // bound at the DDL; placement decided then
		e.shards = append(e.shards, s)
	}

	for !e.done() {
		if !e.step(w) {
			break
		}
	}
	e.finish()
	return e.res
}

// done: every shard finished its real work AND every copy is terminal (the
// ticks keep running after the round's real end only to settle stragglers'
// copies, whose cancels the round metrics ignore, decision 17).
func (e *engine) done() bool {
	for _, s := range e.shards {
		if s.b.state != "done" && s.b.state != "failed" {
			return false
		}
		if s.next < len(s.queries) || len(s.queue) > 0 || s.task != taskIdle {
			return false
		}
	}
	for _, b := range e.builds {
		if b.speculative && b.state != "done" && b.state != "failed" {
			return false
		}
	}
	return true
}

// step advances to the next event time and dispatches everything due there.
// Dispatch order at one instant is fixed (workers, requeues, kill, DDLs,
// arrivals, CPU completions, reports, tick, then claims and CPU dispatch), so
// a run is deterministic.
func (e *engine) step(w workload.Workload) bool {
	next := math.Inf(1)
	add := func(t float64) {
		if t >= e.t-eps && t < next {
			next = t
		}
	}
	for _, s := range e.shards {
		if !e.submitted(s) {
			add(s.ddlAt)
		}
		if s.next < len(s.queries) {
			add(s.queries[s.next].Arrival.Seconds())
		}
		if s.task != taskIdle {
			add(s.taskDone)
		}
		if e.submitted(s) {
			add(s.nextReport)
		}
	}
	for _, wk := range e.workers {
		if !wk.alive || wk.cur == nil {
			continue
		}
		if wk.xferRem > eps {
			add(e.t + wk.xferRem*e.linkShare(wk))
		} else {
			add(wk.buildDoneAt)
		}
	}
	for _, r := range e.requeue {
		add(r.at)
	}
	if e.killPending {
		add(e.cfg.KillWorkerAtS)
	}
	add(e.nextTick)
	if math.IsInf(next, 1) || next > e.cfg.MaxModeledS {
		return false // nothing scheduled, or the safety cap: report incomplete
	}

	// Advance the clock; in-flight transfers drain at their node's shared
	// rate, which is constant BETWEEN events — so every worker's share is
	// snapshotted before any remaining is mutated. Bug 14: computing the
	// share inside the mutation loop let a transfer that reached zero early
	// in the loop vanish from its node-mates' k, advancing them at the full
	// rate for the same interval — half an interval of stolen progress, and
	// an overshoot below zero that then skipped a whole GPU build.
	dt := next - e.t
	shares := make([]float64, len(e.workers))
	for i, wk := range e.workers {
		if wk.alive && wk.cur != nil && wk.xferRem > eps {
			shares[i] = e.linkShare(wk)
		}
	}
	for i, wk := range e.workers {
		if shares[i] > 0 {
			wk.xferRem -= dt / shares[i]
		}
	}
	e.t = next
	e.res.Events++

	// 1. Workers: transfers that finished start their build; builds that
	// finished complete their row (a copy's win is settled at the tick).
	// The phase test is by MEANING, not by value (bug 12's lesson): any
	// remaining at or below zero with no build deadline set means the
	// transfer just ended — it must enter its build, never fall through.
	for _, wk := range e.workers {
		if !wk.alive || wk.cur == nil {
			continue
		}
		if wk.xferRem > eps {
			continue
		}
		if wk.buildDoneAt == 0 {
			wk.xferRem = 0
			wk.buildDoneAt = e.t + wk.cur.trueGPUS
			continue
		}
		if wk.buildDoneAt <= e.t+eps {
			b := wk.cur
			b.state, b.finishAt = "done", e.t
			wk.cur = nil
		}
	}
	// 2. Expired leases of dead workers go back to the queue (the reaper).
	keep := e.requeue[:0]
	for _, r := range e.requeue {
		if r.at <= e.t+eps {
			if r.b.state == "leased" {
				r.b.state = "queued" // seq (enqueued_at) kept; the next claim re-increments attempt
			}
		} else {
			keep = append(keep, r)
		}
	}
	e.requeue = keep
	// 3. The shrinking pool's kill: the busiest worker dies; its lease
	// requeues after LeaseS (the reaper's timetable).
	if e.killPending && e.cfg.KillWorkerAtS <= e.t+eps {
		e.killPending = false
		var victim *worker
		for _, wk := range e.workers {
			if !wk.alive {
				continue
			}
			if victim == nil || (wk.cur != nil && victim.cur == nil) {
				victim = wk
			}
		}
		if victim != nil {
			victim.alive = false
			if victim.cur != nil {
				e.requeue = append(e.requeue, struct {
					at float64
					b  *build
				}{e.t + e.cfg.LeaseS, victim.cur})
				victim.cur = nil
			}
		}
	}
	// 4. DDLs: submit and decide placement.
	for _, s := range e.shards {
		if !e.submitted(s) && s.ddlAt <= e.t+eps {
			e.submit(s)
		}
	}
	// 5. Query arrivals.
	for _, s := range e.shards {
		for s.next < len(s.queries) && s.queries[s.next].Arrival.Seconds() <= e.t+eps {
			s.queue = append(s.queue, s.next)
			s.next++
		}
	}
	// 6. CPU completions.
	for _, s := range e.shards {
		if s.task == taskIdle || s.taskDone > e.t+eps {
			continue
		}
		if s.task == taskBuild {
			// Like the live LocalDone: accepted only if the row is still a
			// running local build; a row moved under us means the work is
			// discarded and the CPU is simply free (the 409 case).
			if s.b.placement == "local" && s.b.state == "running" {
				s.b.state, s.b.finishAt = "done", e.t
				s.indexReady = true
			}
		} else {
			s.lastQueryFin = e.t
		}
		s.task = taskIdle
	}
	// 7. Shard reports: publish the load report, then act on the reply.
	for _, s := range e.shards {
		if e.submitted(s) && s.nextReport <= e.t+eps {
			e.report(s)
			s.nextReport += e.cfg.PollS
		}
	}
	// 8. The reconsider tick.
	if e.nextTick <= e.t+eps {
		e.tick()
		e.nextTick += e.cfg.ReconsiderS
	}
	// 9. Claims: idle workers in ascending ID take the highest-priority
	// queued build that fits (the claim statement's ORDER BY and filter).
	e.claims()
	// 10. CPUs pick their next work.
	for _, s := range e.shards {
		e.dispatchCPU(s)
	}
	return true
}

func (e *engine) submitted(s *shard) bool { return s.b.placement != "" }

// linkShare is how many alive workers on wk's node are mid-transfer (>= 1);
// each proceeds at 1/k of the link. Without ShareNodeLinks every transfer is
// alone (the fake worker's behaviour: it sleeps, it does not contend).
func (e *engine) linkShare(wk *worker) float64 {
	if !e.cfg.ShareNodeLinks {
		return 1
	}
	k := 0.0
	for _, o := range e.workers {
		if o.alive && o.cur != nil && o.xferRem > eps && o.node == wk.node {
			k++
		}
	}
	if k < 1 {
		k = 1
	}
	return k
}

// submit mirrors scheduler/api submitBuild: pool state with the NewJobWait
// scalar (cold-start 0 on an empty pool), the policy's Decide, then the row.
func (e *engine) submit(s *shard) {
	b := s.b
	b.submitAt, b.seq = e.t, e.nextSeq
	e.nextSeq++
	e.builds = append(e.builds, b)

	live, largest := 0, int64(0)
	for _, wk := range e.workers {
		if wk.alive {
			live++
			if wk.mem > largest {
				largest = wk.mem
			}
		}
	}
	wait := 0.0
	if live > 0 {
		wait = policy.NewJobWait(e.workerViews(), e.queuedJobs(), policy.QueuedJob{
			BuildID: b.id, MemBytes: b.mem, GPUBuildS: b.est.GPUBuildS, TransferS: b.est.TransferS,
		})
	}
	dec := e.pol.Decide(policy.Input{
		Build: policy.BuildSpec{ShardID: s.id, RoundID: 1, NVectors: b.n, Dim: b.dim, MemBytes: b.mem},
		Est:   b.est,
		Pool:  policy.PoolState{LiveWorkers: live, LargestMem: largest, QueueWaitS: wait},
	})
	if dec.Placement == "local" {
		b.placement, b.state = "local", "running"
		s.localWant = true
	} else {
		b.placement, b.state, b.priority = "gpu", "queued", dec.Priority
	}
	s.nextReport = e.t + e.cfg.PollS
}

// report publishes the shard's load (what the policies will see until the
// next report) and acts on the current row state, exactly the live shard's
// apply(): done -> index ready (abort a still-running local build: the copy
// won); gpu while building -> preempted, abort; local/running while idle ->
// start (a renege, or the initial placement).
func (e *engine) report(s *shard) {
	s.rep.valid = true
	s.rep.queueDepth = int32(len(s.queue))
	waiting, oldest := int32(0), 0.0
	for _, qi := range s.queue {
		if s.queries[qi].NeedsIndex {
			waiting++
		}
		if w := e.t - s.queries[qi].Arrival.Seconds(); w > oldest {
			oldest = w
		}
	}
	s.rep.waiting, s.rep.oldestS = waiting, oldest
	s.rep.progress = 0
	if s.task == taskBuild && s.b.trueLocalS > 0 {
		s.rep.progress = (e.t - s.buildStart) / s.b.trueLocalS
	}
	s.rep.arrivals = s.next
	if s.next > 0 {
		s.rep.meanDurS = s.durPrefix[s.next] / float64(s.next)
	}

	b := s.b
	switch {
	case (b.state == "done" || b.state == "failed") && !s.indexReady:
		s.indexReady = true
		if s.task == taskBuild {
			s.task = taskIdle // the speculative copy won; stop burning the CPU
		}
	case b.placement == "gpu" && (b.state == "queued" || b.state == "leased") && s.task == taskBuild:
		s.task = taskIdle // preempted: the local progress is lost
	case b.placement == "local" && b.state == "running" && s.task != taskBuild && !s.indexReady && !s.localWant:
		s.localWant = true // reneged back to us
	}
}

// tick is the reconsider loop's Tick in modeled time: settle races, build the
// policy's view from REPORTED state, execute the actions as guarded
// transitions.
func (e *engine) tick() {
	// Resolve speculation races (decision 16). The copy's win backdates the
	// primary's finish to the copy's (the index existed from that moment).
	for _, b := range e.builds {
		if !b.speculative {
			continue
		}
		p := b.primary
		if b.state == "done" && p.placement == "local" && p.state == "running" {
			p.placement, p.state, p.finishAt = "gpu", "done", b.finishAt
			e.res.SpecGPUWins++
		}
		if (b.state == "queued" || b.state == "leased") && (p.state == "done" || p.state == "failed") {
			b.state, b.finishAt = "failed", e.t
			e.res.SpecLocalWins++
			for _, wk := range e.workers {
				if wk.cur == b {
					e.res.WastedGPUS += e.t - wk.claimAt
					wk.cur = nil // freed at the tick (slightly kinder than the live renew lag)
				}
			}
		}
	}

	in := policy.ReconsiderInput{Workers: e.workerViews()}
	queue := e.queuedJobs()
	emptyPool := len(in.Workers) == 0
	var est map[string]policy.JobEstimate
	if !emptyPool {
		est = policy.ReplayQueue(in.Workers, queue)
	}
	for _, b := range e.builds {
		if b.state == "done" || b.state == "failed" {
			continue
		}
		s := e.shards[b.shard]
		v := policy.BuildView{
			BuildID:     b.id,
			Spec:        policy.BuildSpec{ShardID: b.shard, RoundID: 1, NVectors: b.n, Dim: b.dim, MemBytes: b.mem},
			Est:         b.est,
			Priority:    b.priority,
			Moved:       b.moved,
			Speculative: b.speculative,
		}
		if s.rep.valid {
			rate := 0.0
			if e.t > 0 {
				rate = float64(s.rep.arrivals) / e.t
			}
			v.Status = policy.ShardStatus{
				QueueDepth: s.rep.queueDepth, WaitingNeedsIndex: s.rep.waiting,
				OldestWaitS: s.rep.oldestS, BuildProgress: s.rep.progress,
				ArrivalRateQPS: rate, MeanQueryS: s.rep.meanDurS, Valid: true,
			}
		}
		switch {
		case b.placement == "local" && b.state == "running":
			v.Phase = "local_running"
			if !emptyPool {
				v.QueueWaitS = policy.NewJobWait(in.Workers, queue, policy.QueuedJob{
					BuildID: b.id, MemBytes: b.mem, GPUBuildS: b.est.GPUBuildS, TransferS: b.est.TransferS,
				})
			}
		case b.state == "queued":
			v.Phase = "gpu_queued"
			if !emptyPool {
				v.QueueWaitS = est[b.id].StartS
			}
		case b.state == "leased":
			v.Phase = "gpu_leased"
		}
		in.Builds = append(in.Builds, v)
	}

	acts := e.pol.Reconsider(in)
	byID := map[string]*build{}
	for _, b := range e.builds {
		byID[b.id] = b
	}
	for _, pass := range [][]policy.ActionKind{
		{policy.ActPreempt},
		{policy.ActRenege, policy.ActReprioritize, policy.ActSpeculate},
	} {
		for _, a := range acts {
			for _, kind := range pass {
				if a.Kind == kind {
					e.execute(a, byID[a.BuildID])
				}
			}
		}
	}
}

// execute applies one action with the store's guards; a guard that fails is a
// skipped action, like a zero-row UPDATE.
func (e *engine) execute(a policy.Action, b *build) {
	if b == nil {
		return
	}
	switch a.Kind {
	case policy.ActPreempt:
		// Never a moved build; never while a copy could still win (bug 12).
		if b.placement == "local" && b.state == "running" && !b.moved &&
			(b.copyB == nil || b.copyB.state == "failed") {
			b.placement, b.state, b.priority, b.moved = "gpu", "queued", a.Priority, true
			e.res.Preempts++
		}
	case policy.ActRenege:
		// Not guarded on moved (decision 20's escape); the policy owns the
		// judgment rule.
		if b.placement == "gpu" && b.state == "queued" && !b.speculative {
			b.placement, b.state, b.moved = "local", "running", true
			e.res.Reneges++ // the shard starts building at its next report
		}
	case policy.ActReprioritize:
		if b.placement == "gpu" && b.state == "queued" {
			b.priority = a.Priority
			e.res.Reprioritizes++
		}
	case policy.ActSpeculate:
		if b.placement == "local" && b.state == "running" && !b.speculative && b.copyB == nil {
			c := &build{
				id: b.id + ":spec", shard: b.shard, n: b.n, dim: b.dim, mem: b.mem, est: b.est,
				trueLocalS: b.trueLocalS, trueGPUS: b.trueGPUS, trueXferS: b.trueXferS,
				placement: "gpu", state: "queued", priority: a.Priority,
				speculative: true, seq: e.nextSeq, submitAt: e.t, firstClaim: -1,
				primary: b,
			}
			e.nextSeq++
			b.copyB = c
			e.builds = append(e.builds, c)
			e.res.Speculations++
		}
	}
}

// claims mirrors the claim statement until nothing more is claimable now.
func (e *engine) claims() {
	for {
		claimed := false
		for _, wk := range e.workers {
			if !wk.alive || wk.cur != nil {
				continue
			}
			var best *build
			for _, b := range e.builds {
				if b.state != "queued" || b.mem > wk.mem {
					continue
				}
				if best == nil || b.priority > best.priority ||
					(b.priority == best.priority && b.seq < best.seq) {
					best = b
				}
			}
			if best == nil {
				continue
			}
			best.state = "leased"
			if best.firstClaim < 0 {
				best.firstClaim = e.t
			}
			wk.cur, wk.claimAt = best, e.t
			if e.cfg.ShareNodeLinks && best.trueXferS > eps {
				wk.xferRem, wk.buildDoneAt = best.trueXferS, 0
			} else {
				wk.xferRem, wk.buildDoneAt = 0, e.t+best.trueXferS+best.trueGPUS
			}
			claimed = true
		}
		if !claimed {
			return
		}
	}
}

// dispatchCPU starts the shard's next piece of work: the local build if the
// scheduler wants one, else the oldest READY query (one that does not need
// the index, or the index the shard knows about).
func (e *engine) dispatchCPU(s *shard) {
	if s.task != taskIdle {
		return
	}
	if s.localWant {
		s.localWant = false
		s.task, s.buildStart = taskBuild, e.t
		s.taskDone = e.t + s.b.trueLocalS
		return
	}
	for i, qi := range s.queue {
		q := s.queries[qi]
		if q.NeedsIndex && !s.indexReady {
			continue
		}
		s.queue = append(s.queue[:i], s.queue[i+1:]...)
		s.task = qi
		s.taskDone = e.t + q.Duration.Seconds()
		return
	}
}

func (e *engine) workerViews() []policy.WorkerView {
	var out []policy.WorkerView
	for _, wk := range e.workers {
		if !wk.alive {
			continue
		}
		busy := 0.0
		if wk.cur != nil {
			// The host's number, so the NOISED total (what its cost model says)
			// minus the real elapsed, floored at 0 — same as the live mapping.
			busy = wk.cur.est.GPUBuildS + wk.cur.est.TransferS - (e.t - wk.claimAt)
			if busy < 0 {
				busy = 0
			}
		}
		out = append(out, policy.WorkerView{ID: wk.id, Node: wk.node, MemBytes: wk.mem, BusyRemainingS: busy})
	}
	return out
}

func (e *engine) queuedJobs() []policy.QueuedJob {
	var qs []*build
	for _, b := range e.builds {
		if b.state == "queued" {
			qs = append(qs, b)
		}
	}
	sort.SliceStable(qs, func(i, j int) bool {
		if qs[i].priority != qs[j].priority {
			return qs[i].priority > qs[j].priority
		}
		return qs[i].seq < qs[j].seq
	})
	out := make([]policy.QueuedJob, len(qs))
	for i, b := range qs {
		out[i] = policy.QueuedJob{
			BuildID: b.id, Priority: b.priority, Seq: i, MemBytes: b.mem,
			GPUBuildS: b.est.GPUBuildS, TransferS: b.est.TransferS,
		}
	}
	return out
}

// Futures turns a workload into what the clairvoyant oracle is constructed
// with: each shard's full future query stream, arrival offsets relative to
// that shard's DDL (a pre-DDL arrival clamps to 0 — it is backlog the moment
// the build decision is made). Only the simulator may call this (decision 1a).
func Futures(w workload.Workload) map[int32][]policy.QueryEvent {
	out := make(map[int32][]policy.QueryEvent, len(w.Shards))
	for _, sw := range w.Shards {
		qs := make([]policy.QueryEvent, 0, len(sw.Queries))
		for _, q := range sw.Queries {
			at := (q.Arrival - sw.DDLOffset).Seconds()
			if at < 0 {
				at = 0
			}
			qs = append(qs, policy.QueryEvent{ArriveS: at, DurS: q.Duration.Seconds(), NeedsIndex: q.NeedsIndex})
		}
		out[sw.ShardID] = qs
	}
	return out
}

// finish computes the round's numbers with the live semantics: finished_at is
// the latest real-build or query finish (decision 30), the straggler lag is
// last minus interpolated-median shard finish (fillRoundStats), queue waits
// are per claimed real build (attempt > 0, cross-check F2), and copies are
// invisible to all of it (decision 17).
func (e *engine) finish() {
	e.res.Completed = true
	for _, s := range e.shards {
		if s.b.state != "done" && s.b.state != "failed" {
			e.res.Completed = false
		}
		fin := s.b.finishAt
		if s.lastQueryFin > fin {
			fin = s.lastQueryFin
		}
		e.res.ShardFinishS = append(e.res.ShardFinishS, fin)
		if fin > e.res.RoundS {
			e.res.RoundS = fin
		}
		if s.b.placement == "gpu" {
			e.res.BuildsGPU++
		} else {
			e.res.BuildsLocal++
		}
	}
	for _, b := range e.builds {
		if !b.speculative && b.firstClaim >= 0 {
			e.res.QueueWaits = append(e.res.QueueWaits, b.firstClaim-b.submitAt)
		}
	}
	fins := append([]float64(nil), e.res.ShardFinishS...)
	sort.Float64s(fins)
	if n := len(fins); n > 0 {
		// percentile_cont(0.5): interpolated median, as the live SQL computes.
		mid := 0.5 * float64(n-1)
		lo, hi := int(math.Floor(mid)), int(math.Ceil(mid))
		median := fins[lo] + (mid-float64(lo))*(fins[hi]-fins[lo])
		e.res.StragglerLagS = fins[n-1] - median
	}
}
