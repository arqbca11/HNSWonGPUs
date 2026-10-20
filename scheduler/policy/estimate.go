package policy

import (
	"math"
	"sort"
)

// The GPU finish estimator (Phase 3 step 2): list scheduling by replaying the
// claim. It is pure like everything in this package — the host hands it the
// pool and the queue, every duration a float64 of seconds relative to the
// call, and it returns when each queued build would start and finish.
//
// The replay mirrors what the real system does, not abstract list scheduling
// (decision 12): whenever a worker is free it claims the highest-priority
// queued job THAT FITS ITS MEMORY (the claim's own filter), ties broken by
// enqueued order, exactly the claim statement's ORDER BY. A job that fits no
// worker in the pool is never claimed and estimates to +Inf — the estimator
// states the truth bug 11 hid.
//
// Transfers share the node's link (decision 12): a claimed job first moves
// its bytes for TransferS seconds at the full link (the cost model's whole
// transfer term, vectors in plus index out, taken upfront), and while k
// transfers are in flight on one node each proceeds at 1/k — two 10 s
// transfers started together finish together at 20 s. Workers whose Node is
// "" are each alone on their own link ("unknown" must not invent contention
// between unrelated machines). The GPU build itself is not shared: one build
// per worker, full speed.
//
// Approximations, pinned: a busy worker is opaque — BusyRemainingS is taken
// as a whole (whether it is mid-transfer or mid-build is not visible to the
// host), so an in-flight build does not contend for its node's link in the
// replay. And the replay freezes the world: no future submits, preempts or
// worker deaths.

// QueuedJob is one gpu-queued build as the estimator sees it. Seq is the
// enqueued order (0 first), the claim's tiebreak within equal priority; hosts
// that pass the queue already in claim order just number it 0..n-1.
type QueuedJob struct {
	BuildID   string
	Priority  float64
	Seq       int
	GPUBuildS float64
	TransferS float64 // at the node's full link; sharing is the replay's job
	MemBytes  int64
}

// JobEstimate is the replay's answer for one job: StartS is when a worker
// claims it (its QueueWaitS), DoneS when transfer and build are finished.
// Both are +Inf for a job that fits no worker in the pool.
type JobEstimate struct {
	StartS float64
	DoneS  float64
}

// ReplayQueue replays the whole queue over the pool and returns an estimate
// per BuildID. It is deterministic: input order never matters (jobs are
// sorted by priority DESC then Seq; simultaneous worker frees are resolved in
// ascending worker ID — the real claim is a race, so any one coherent order
// is an approximation, and this one is pinned). BuildIDs are expected unique,
// as they are the system's idempotency key; duplicates collapse into one map
// entry with an unspecified value. A job whose TransferS is zero (or
// negative) moves no bytes and takes no share of its node's link.
func ReplayQueue(workers []WorkerView, queue []QueuedJob) map[string]JobEstimate {
	out := make(map[string]JobEstimate, len(queue))
	if len(workers) == 0 {
		// No pool, no claims: every estimate is +Inf. The cold-start rule
		// (an empty pool may be about to fill) is the host's to apply.
		for _, j := range queue {
			out[j.BuildID] = JobEstimate{StartS: math.Inf(1), DoneS: math.Inf(1)}
		}
		return out
	}

	var largestMem int64
	for _, w := range workers {
		if w.MemBytes > largestMem {
			largestMem = w.MemBytes
		}
	}
	pending := make([]QueuedJob, 0, len(queue))
	for _, j := range queue {
		if j.MemBytes > largestMem { // fits no worker: never claimed (bug 11's truth)
			out[j.BuildID] = JobEstimate{StartS: math.Inf(1), DoneS: math.Inf(1)}
			continue
		}
		pending = append(pending, j)
	}
	sort.SliceStable(pending, func(i, k int) bool {
		if pending[i].Priority != pending[k].Priority {
			return pending[i].Priority > pending[k].Priority
		}
		return pending[i].Seq < pending[k].Seq
	})

	// One replay worker per live worker. An empty Node is the worker's own
	// link, keyed uniquely so unknowns never contend with each other.
	type rw struct {
		view   WorkerView
		node   string
		freeAt float64 // when its current occupation ends; claims happen then
		idle   bool
	}
	rws := make([]*rw, len(workers))
	for i, w := range workers {
		node := w.Node
		if node == "" {
			node = "\x00worker:" + w.ID
		}
		rws[i] = &rw{view: w, node: node, freeAt: w.BusyRemainingS, idle: w.BusyRemainingS <= 0}
	}
	sort.Slice(rws, func(i, k int) bool { return rws[i].view.ID < rws[k].view.ID })

	// Active phases: a transfer holds remaining full-rate seconds and drains
	// at 1/k of its node's concurrency; a build ends at a fixed time.
	type xfer struct {
		w         *rw
		job       QueuedJob
		remaining float64
	}
	type bld struct {
		w     *rw
		job   QueuedJob
		endAt float64
	}
	var xfers []*xfer
	var blds []*bld
	nodeCount := func(node string) float64 {
		n := 0
		for _, x := range xfers {
			if x.w.node == node {
				n++
			}
		}
		return float64(n)
	}

	const eps = 1e-9
	t := 0.0
	for {
		// Every idle worker claims, in worker-ID order, the highest-priority
		// pending job that fits it. Claiming changes node concurrency, so the
		// loop restarts until nothing more is claimable now.
		claimed := true
		for claimed {
			claimed = false
			for _, w := range rws {
				if !w.idle {
					continue
				}
				for pi, j := range pending {
					if j.MemBytes > w.view.MemBytes {
						continue
					}
					pending = append(pending[:pi], pending[pi+1:]...)
					w.idle = false
					est := out[j.BuildID]
					est.StartS = t
					out[j.BuildID] = est
					if j.TransferS > eps {
						xfers = append(xfers, &xfer{w: w, job: j, remaining: j.TransferS})
					} else {
						blds = append(blds, &bld{w: w, job: j, endAt: t + j.GPUBuildS})
					}
					claimed = true
					break
				}
			}
		}

		if len(xfers) == 0 && len(blds) == 0 {
			// Nothing in flight. Pending jobs, if any, wait for an initially
			// busy worker; otherwise the replay is finished.
			next := math.Inf(1)
			for _, w := range rws {
				if !w.idle && w.freeAt > t+eps {
					next = math.Min(next, w.freeAt)
				}
			}
			if math.IsInf(next, 1) {
				break
			}
			t = next
			for _, w := range rws {
				if !w.idle && w.freeAt <= t+eps {
					w.idle = true
				}
			}
			continue
		}

		// Next event: an initially busy worker freeing, a transfer finishing
		// at its node's current rate, or a build finishing.
		next := math.Inf(1)
		for _, w := range rws {
			if !w.idle && w.freeAt > t+eps && w.freeAt < next {
				// only the initial busy phase uses freeAt; claimed jobs live in xfers/blds
				inFlight := false
				for _, x := range xfers {
					if x.w == w {
						inFlight = true
					}
				}
				for _, b := range blds {
					if b.w == w {
						inFlight = true
					}
				}
				if !inFlight {
					next = w.freeAt
				}
			}
		}
		for _, x := range xfers {
			next = math.Min(next, t+x.remaining*nodeCount(x.w.node))
		}
		for _, b := range blds {
			next = math.Min(next, b.endAt)
		}

		dt := next - t
		for _, x := range xfers {
			x.remaining -= dt / nodeCount(x.w.node)
		}
		t = next

		// Transfers that finished start their build; builds that finished free
		// their worker; initially busy workers whose time came go idle.
		var stillX []*xfer
		for _, x := range xfers {
			if x.remaining <= eps {
				blds = append(blds, &bld{w: x.w, job: x.job, endAt: t + x.job.GPUBuildS})
			} else {
				stillX = append(stillX, x)
			}
		}
		xfers = stillX
		var stillB []*bld
		for _, b := range blds {
			if b.endAt <= t+eps {
				est := out[b.job.BuildID]
				est.DoneS = b.endAt
				out[b.job.BuildID] = est
				b.w.idle = true
			} else {
				stillB = append(stillB, b)
			}
		}
		blds = stillB
		for _, w := range rws {
			if !w.idle && w.freeAt <= t+eps {
				inFlight := false
				for _, x := range xfers {
					if x.w == w {
						inFlight = true
					}
				}
				for _, b := range blds {
					if b.w == w {
						inFlight = true
					}
				}
				if !inFlight {
					w.idle = true
				}
			}
		}
	}
	return out
}

// NewJobWait is the submit-time question (decision 13): how long would this
// not-yet-queued build wait before a worker starts it? The build joins at
// literally priority 0, with a Seq after every queued job — so it sits behind
// the whole queue when priorities are non-negative, which every shipped
// policy's are; a hypothetical negative-priority job would rank below it.
// Exact for the FCFS baseline, an upper bound for a policy that would grant
// the build higher priority. With an empty pool it returns +Inf; the host
// applies the cold-start rule (keep 0) itself, because "no workers yet" is
// the host's knowledge, not the replay's.
func NewJobWait(workers []WorkerView, queue []QueuedJob, job QueuedJob) float64 {
	job.Priority = 0
	job.Seq = len(queue)
	maxSeq := 0
	for _, q := range queue {
		if q.Seq >= maxSeq {
			maxSeq = q.Seq + 1
		}
	}
	if maxSeq > job.Seq {
		job.Seq = maxSeq
	}
	all := append(append([]QueuedJob(nil), queue...), job)
	return ReplayQueue(workers, all)[job.BuildID].StartS
}
