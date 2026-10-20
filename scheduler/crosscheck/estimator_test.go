package crosscheck

// Phase 3 step 2 cross-check: the GPU finish estimator (queue replay).
// Written from the roadmap's "GPU finish estimate: replay the queue" paragraph,
// docs/design/phase3-policies.md (decisions 11, 12, 13, the queue-replay flow
// diagram, the step 2 implementation notes), CLAUDE.md invariant 8, and
// `go doc ./scheduler/policy` (ReplayQueue, NewJobWait, QueuedJob,
// JobEstimate, WorkerView). No estimator source was read.
//
// The model these tests replay by hand (decision 12):
//   - whenever a worker is free it claims the highest-priority queued job that
//     fits its memory (MemBytes <= worker MemBytes), ties by Seq;
//   - StartS = the claim moment; the job's whole TransferS is then streamed
//     over the worker's node link, shared as a fluid: k concurrent transfers on
//     one node each run at 1/k of the link; a worker with Node "" has a private
//     link; a busy worker's in-flight build never contends (opaque lump);
//   - after its transfer the job builds for GPUBuildS; DoneS = transfer end +
//     GPUBuildS, and the worker is free again at DoneS;
//   - a job fitting no worker gets StartS = DoneS = +Inf and blocks nobody.

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

const estEps = 1e-6

func ew(id, node string, mem int64, busy float64) policy.WorkerView {
	return policy.WorkerView{ID: id, Node: node, MemBytes: mem, BusyRemainingS: busy}
}

func ej(id string, prio float64, seq int, build, transfer float64, mem int64) policy.QueuedJob {
	return policy.QueuedJob{BuildID: id, Priority: prio, Seq: seq, GPUBuildS: build, TransferS: transfer, MemBytes: mem}
}

func estSame(a, b float64) bool {
	if math.IsInf(a, 1) || math.IsInf(b, 1) {
		return math.IsInf(a, 1) && math.IsInf(b, 1)
	}
	return math.Abs(a-b) < estEps
}

func replay(t *testing.T, ws []policy.WorkerView, q []policy.QueuedJob) map[string]policy.JobEstimate {
	t.Helper()
	t.Logf("input workers: %s", fmtWorkers(ws))
	t.Logf("input queue:   %s", fmtJobs(q))
	got := policy.ReplayQueue(ws, q)
	t.Logf("ReplayQueue -> %s", fmtEsts(got, q))
	return got
}

func fmtWorkers(ws []policy.WorkerView) string {
	s := ""
	for _, w := range ws {
		s += fmt.Sprintf("{%s node=%q mem=%d busy=%gs} ", w.ID, w.Node, w.MemBytes, w.BusyRemainingS)
	}
	return s
}

func fmtJobs(q []policy.QueuedJob) string {
	s := ""
	for _, j := range q {
		s += fmt.Sprintf("{%s prio=%g seq=%d build=%gs transfer=%gs mem=%d} ", j.BuildID, j.Priority, j.Seq, j.GPUBuildS, j.TransferS, j.MemBytes)
	}
	return s
}

func fmtEsts(m map[string]policy.JobEstimate, q []policy.QueuedJob) string {
	s := ""
	for _, j := range q {
		e, ok := m[j.BuildID]
		if !ok {
			s += fmt.Sprintf("%s:<missing> ", j.BuildID)
			continue
		}
		s += fmt.Sprintf("%s:start=%g done=%g ", j.BuildID, e.StartS, e.DoneS)
	}
	return s
}

func want(t *testing.T, got map[string]policy.JobEstimate, id string, start, done float64, why string) {
	t.Helper()
	e, ok := got[id]
	if !ok {
		t.Errorf("%s: no estimate returned (spec: ReplayQueue returns an estimate per BuildID)", id)
		return
	}
	if !estSame(e.StartS, start) || !estSame(e.DoneS, done) {
		t.Errorf("%s: got start=%g done=%g, want start=%g done=%g  [%s]", id, e.StartS, e.DoneS, start, done, why)
		return
	}
	t.Logf("ok %s: start=%g done=%g  [%s]", id, start, done, why)
}

func cloneW(ws []policy.WorkerView) []policy.WorkerView { return append([]policy.WorkerView(nil), ws...) }
func cloneQ(q []policy.QueuedJob) []policy.QueuedJob   { return append([]policy.QueuedJob(nil), q...) }

func TestEstimatorSpec(t *testing.T) {
	inf := math.Inf(1)
	const big = int64(100)

	// ---- the roadmap / flow diagram example ------------------------------

	t.Run("RoadmapExample", func(t *testing.T) {
		t.Logf("spec: design doc flow 'Queue replay': w1 busy 4 s, w2 free; j1 6 s, j2 3 s by priority; j1->w2 done 6, j2->w1 done 7; the next build's earliest free is w2 at 6")
		ws := []policy.WorkerView{ew("w1", "n1", big, 4), ew("w2", "n2", big, 0)}
		q := []policy.QueuedJob{ej("j1", 2, 0, 6, 0, 1), ej("j2", 1, 1, 3, 0, 1)}
		got := replay(t, ws, q)
		want(t, got, "j1", 0, 6, "t=0 only w2 is free, it claims j1 (highest prio): 0+0 transfer+6 build = 6")
		want(t, got, "j2", 4, 7, "t=4 w1 frees, claims j2: 4+3 = 7")
		nw := policy.NewJobWait(ws, q, ej("new", 0, 2, 1, 0, 1))
		t.Logf("NewJobWait(new) = %g; hand: w1 busy until 7, w2 free at 6 -> 6 (decision 13: joins behind every queued job)", nw)
		if !estSame(nw, 6) {
			t.Errorf("NewJobWait: got %g, want 6", nw)
		}
	})

	// ---- fluid link sharing (decision 12, decision 11) --------------------

	t.Run("TwoTransfersOneNodeShareTheLink", func(t *testing.T) {
		t.Logf("spec: decision 12 'two 10 s transfers started together both finish at 20 s'")
		ws := []policy.WorkerView{ew("w1", "nodeA", big, 0), ew("w2", "nodeA", big, 0)}
		q := []policy.QueuedJob{ej("j1", 0, 0, 1, 10, 1), ej("j2", 0, 1, 1, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "j1", 0, 21, "both at 1/2: 10 s of transfer takes 20 s, +1 s build = 21")
		want(t, got, "j2", 0, 21, "same as j1")
	})

	t.Run("ThreeTransfersOneNodeEachAtOneThird", func(t *testing.T) {
		t.Logf("spec: decision 12 'k concurrent transfers each proceed at 1/k'")
		ws := []policy.WorkerView{ew("w1", "n", big, 0), ew("w2", "n", big, 0), ew("w3", "n", big, 0)}
		q := []policy.QueuedJob{ej("a", 0, 0, 0, 6, 1), ej("b", 0, 1, 0, 6, 1), ej("c", 0, 2, 0, 6, 1)}
		got := replay(t, ws, q)
		for _, id := range []string{"a", "b", "c"} {
			want(t, got, id, 0, 18, "three at 1/3: 6 s of transfer takes 18 s, build 0")
		}
	})

	t.Run("UnequalConcurrentTransfersRateRisesWhenOneEnds", func(t *testing.T) {
		t.Logf("spec: decision 12 fluid sharing; the GPU build phase does not use the link")
		ws := []policy.WorkerView{ew("w1", "n", big, 0), ew("w2", "n", big, 0)}
		q := []policy.QueuedJob{ej("short", 0, 0, 10, 4, 1), ej("long", 0, 1, 0, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "short", 0, 18, "0..8 both at 1/2: short's 4 s done at 8; then it builds 10 s -> 18")
		want(t, got, "long", 0, 14, "0..8 long moves 4 of 10 at 1/2; from 8 it is alone at full rate, 6 more -> 14 (short's build does not hold the link)")
	})

	t.Run("StaggeredJoinSlowsTheEarlierTransfer", func(t *testing.T) {
		t.Logf("spec: decision 12 fluid sharing with a join mid-transfer")
		ws := []policy.WorkerView{ew("w1", "n", big, 0), ew("w2", "n", big, 4)}
		q := []policy.QueuedJob{ej("j1", 2, 0, 0, 10, 1), ej("j2", 1, 1, 0, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "j1", 0, 16, "0..4 alone at full rate: 4 done, 6 left; from 4 at 1/2: 12 s more -> 16")
		want(t, got, "j2", 4, 20, "w2 frees at 4 and claims j2; 4..16 at 1/2 moves 6; 16..20 alone moves 4 -> 20")
	})

	t.Run("SeparateNodesRunAtFullRate", func(t *testing.T) {
		t.Logf("spec: decision 11/12: sharing is per node")
		ws := []policy.WorkerView{ew("w1", "nodeA", big, 0), ew("w2", "nodeB", big, 0)}
		q := []policy.QueuedJob{ej("j1", 0, 0, 1, 10, 1), ej("j2", 0, 1, 1, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "j1", 0, 11, "own link: 10 + 1")
		want(t, got, "j2", 0, 11, "own link: 10 + 1")
	})

	t.Run("EmptyNodeIsAPrivateLink", func(t *testing.T) {
		t.Logf("spec: decision 11 'The estimator gives an unknown-node worker its own private link, because unknown must not invent contention'")
		ws := []policy.WorkerView{ew("w1", "", big, 0), ew("w2", "", big, 0)}
		q := []policy.QueuedJob{ej("j1", 0, 0, 0, 10, 1), ej("j2", 0, 1, 0, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "j1", 0, 10, "both Node \"\": no shared link, 10 s each")
		want(t, got, "j2", 0, 10, "both Node \"\": no shared link, 10 s each")

		ws2 := []policy.WorkerView{ew("w1", "", big, 0), ew("w2", "nodeA", big, 0), ew("w3", "nodeA", big, 0)}
		q2 := []policy.QueuedJob{ej("k1", 0, 0, 0, 10, 1), ej("k2", 0, 1, 0, 10, 1), ej("k3", 0, 2, 0, 10, 1)}
		got2 := replay(t, ws2, q2)
		// All three start at 0 (three idle workers). The unknown-node one is
		// alone; the two on nodeA share. Which job lands where depends on the
		// worker tiebreak, so check the multiset of finishes.
		var tens, twenties int
		for _, id := range []string{"k1", "k2", "k3"} {
			e := got2[id]
			if !estSame(e.StartS, 0) {
				t.Errorf("%s: start %g, want 0 (three idle workers)", id, e.StartS)
			}
			switch {
			case estSame(e.DoneS, 10):
				tens++
			case estSame(e.DoneS, 20):
				twenties++
			}
		}
		t.Logf("finishes: %d at 10 (the private-link worker), %d at 20 (two sharing nodeA); hand: exactly 1 and 2", tens, twenties)
		if tens != 1 || twenties != 2 {
			t.Errorf("mixed \"\" and nodeA: want one finish at 10 and two at 20, got %d and %d", tens, twenties)
		}
	})

	t.Run("BusyWorkerIsOpaqueNoContention", func(t *testing.T) {
		t.Logf("spec: decision 12 pinned approximation 'a busy worker is opaque (... an in-flight build's own transfer does not contend)'")
		ws := []policy.WorkerView{ew("w1", "n", big, 5), ew("w2", "n", big, 0)}
		q := []policy.QueuedJob{ej("j", 0, 0, 0, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "j", 0, 10, "w2 idle claims at 0; w1's in-flight lump does not share the link: 10")
	})

	t.Run("TransferOfLaterJobOnSameWorkerSharesWithNeighbour", func(t *testing.T) {
		t.Logf("spec: decision 12; a worker's second job contends with its node neighbour's transfer if they overlap")
		// w1 and w2 on node n. a(transfer 2, build 0) on w1; b(transfer 10, build 0) on w2.
		ws := []policy.WorkerView{ew("w1", "n", big, 0), ew("w2", "n", big, 0)}
		q := []policy.QueuedJob{ej("a", 3, 0, 0, 2, 1), ej("b", 2, 1, 0, 10, 1), ej("c", 1, 2, 0, 2, 1)}
		got := replay(t, ws, q)
		// t=0: w1 claims a, w2 claims b; both at 1/2. a's 2 s done at 4 (b has 2 done, 8 left).
		// t=4: a's worker frees and claims c; b and c share at 1/2. c's 2 s done at 8 (b: 4 done, 6 left).
		// 8..14: b alone moves its last 6.
		want(t, got, "a", 0, 4, "0..4 at 1/2 moves 2")
		want(t, got, "c", 4, 8, "a's worker frees at 4; 4..8 at 1/2 moves 2")
		want(t, got, "b", 0, 14, "0..8 at 1/2 moves 4; 8..14 alone moves 6")
	})

	// ---- the claim mirror (decision 12) ----------------------------------

	t.Run("SmallIdleWorkerDoesNotWaitBehindBigJob", func(t *testing.T) {
		t.Logf("spec: decision 12 'a small idle worker never waits behind a big job it could never take'; textbook list scheduling would put jBig on the earliest-free (small) worker")
		ws := []policy.WorkerView{ew("big", "n1", 100, 2), ew("small", "n2", 10, 0)}
		q := []policy.QueuedJob{ej("jBig", 10, 0, 5, 0, 50), ej("jSmall", 1, 1, 3, 0, 5)}
		got := replay(t, ws, q)
		want(t, got, "jSmall", 0, 3, "t=0 small is free, jBig (50) doesn't fit its 10, so it takes jSmall: 0+3")
		want(t, got, "jBig", 2, 7, "t=2 big frees and takes jBig: 2+5")
	})

	t.Run("PriorityOrderUnderPerWorkerMemoryFilter", func(t *testing.T) {
		t.Logf("spec: decision 12: a free worker takes the highest-priority job that fits ITS memory")
		ws := []policy.WorkerView{ew("big", "n1", 100, 4), ew("small", "n2", 10, 0)}
		q := []policy.QueuedJob{ej("A", 9, 0, 2, 0, 50), ej("B", 5, 1, 3, 0, 5), ej("C", 1, 2, 1, 0, 5)}
		got := replay(t, ws, q)
		want(t, got, "B", 0, 3, "t=0 small: A doesn't fit, highest fitting is B: 0..3")
		want(t, got, "C", 3, 4, "t=3 small free again: A still doesn't fit, takes C: 3..4")
		want(t, got, "A", 4, 6, "t=4 big frees: takes A: 4..6")
	})

	t.Run("PriorityDescThenSeqTiebreak", func(t *testing.T) {
		t.Logf("spec: go doc ReplayQueue 'jobs are sorted by priority DESC then Seq'; QueuedJob.Seq 'the claim's tiebreak within equal priority'")
		ws := []policy.WorkerView{ew("w", "n", big, 0)}
		// Given deliberately out of order in the slice.
		q := []policy.QueuedJob{ej("A", 1, 0, 1, 0, 1), ej("B", 5, 2, 1, 0, 1), ej("C", 5, 1, 1, 0, 1)}
		got := replay(t, ws, q)
		want(t, got, "C", 0, 1, "priority 5 seq 1 is first")
		want(t, got, "B", 1, 2, "priority 5 seq 2 second")
		want(t, got, "A", 2, 3, "priority 1 last even though seq 0")
	})

	t.Run("SimultaneousFreeResolvedByWorkerID", func(t *testing.T) {
		t.Logf("spec: go doc ReplayQueue 'simultaneous worker frees are resolved by worker ID' (direction unstated; the result must be one coherent outcome and independent of input order)")
		ws := []policy.WorkerView{ew("a", "n1", 100, 0), ew("b", "n2", 10, 0)}
		q := []policy.QueuedJob{ej("j1", 10, 0, 10, 0, 5), ej("j2", 5, 1, 1, 0, 50)}
		got := replay(t, ws, q)
		e2 := got["j2"]
		// If "a" picks first: a takes j1, b cannot take j2 (50 > 10), j2 waits until 10.
		// If "b" picks first: b takes j1, a takes j2 at 0.
		switch {
		case estSame(e2.StartS, 10) && estSame(e2.DoneS, 11):
			t.Logf("outcome: ascending ID order ('a' claims first): j1 on a 0..10, j2 waits for a: 10..11")
		case estSame(e2.StartS, 0) && estSame(e2.DoneS, 1):
			t.Logf("outcome: descending ID order ('b' claims first): j1 on b, j2 on a 0..1")
		default:
			t.Errorf("j2: start=%g done=%g is neither coherent outcome (10/11 or 0/1)", e2.StartS, e2.DoneS)
		}
		want(t, got, "j1", 0, 10, "j1 starts at 0 either way")
		rev := policy.ReplayQueue([]policy.WorkerView{ws[1], ws[0]}, q)
		if !reflect.DeepEqual(rev, got) {
			t.Errorf("worker input order changed the tiebreak: %v vs %v", rev, got)
		}
	})

	t.Run("MemoryExactlyEqualFits", func(t *testing.T) {
		t.Logf("spec: decision 12 'fits its memory'; a job needing exactly the worker's capacity fits (claim filter mem_bytes <= cap)")
		ws := []policy.WorkerView{ew("w", "n", 64, 0)}
		q := []policy.QueuedJob{ej("eq", 0, 0, 2, 1, 64), ej("over", 0, 1, 2, 1, 65)}
		got := replay(t, ws, q)
		want(t, got, "eq", 0, 3, "64 <= 64 fits: 0 + 1 transfer + 2 build")
		want(t, got, "over", inf, inf, "65 > 64 fits nowhere: +Inf (decision 12)")
	})

	t.Run("FitsNowhereIsInfAndDoesNotBlock", func(t *testing.T) {
		t.Logf("spec: decision 12 'A job that fits no live worker estimates +Inf'; JobEstimate 'Both are +Inf for a job that fits no worker'")
		ws := []policy.WorkerView{ew("w1", "n", 10, 0), ew("w2", "n", 20, 1)}
		q := []policy.QueuedJob{ej("huge", 100, 0, 1, 1, 1000), ej("ok1", 1, 1, 2, 0, 5), ej("ok2", 0, 2, 2, 0, 15)}
		got := replay(t, ws, q)
		want(t, got, "huge", inf, inf, "1000 exceeds both 10 and 20")
		want(t, got, "ok1", 0, 2, "w1 free at 0; huge is skipped, ok1 (5<=10) next by priority: 0..2")
		want(t, got, "ok2", 1, 3, "w2 frees at 1; ok2 (15<=20): 1..3 (does not wait behind huge)")
	})

	t.Run("EmptyPoolEveryJobInf", func(t *testing.T) {
		t.Logf("spec: decision 12/13: with no workers no job fits any live worker -> +Inf; go doc NewJobWait 'With an empty pool it returns +Inf'")
		q := []policy.QueuedJob{ej("a", 1, 0, 1, 1, 1), ej("b", 0, 1, 0, 0, 0)}
		got := replay(t, nil, q)
		want(t, got, "a", inf, inf, "no worker")
		want(t, got, "b", inf, inf, "no worker, even a zero-size job")
		if len(got) != 2 {
			t.Errorf("want 2 entries, got %d", len(got))
		}
	})

	t.Run("EmptyQueueEmptyMap", func(t *testing.T) {
		got := policy.ReplayQueue([]policy.WorkerView{ew("w", "n", 1, 0)}, nil)
		t.Logf("ReplayQueue(1 worker, nil queue) -> %v (len %d); expected empty", got, len(got))
		if len(got) != 0 {
			t.Errorf("want empty map, got %v", got)
		}
	})

	t.Run("EveryBuildIDGetsAnEstimate", func(t *testing.T) {
		t.Logf("spec: go doc ReplayQueue 'returns an estimate per BuildID'")
		ws := []policy.WorkerView{ew("w", "n", 10, 0)}
		q := []policy.QueuedJob{ej("a", 0, 0, 1, 0, 1), ej("b", 0, 1, 1, 0, 100), ej("c", 0, 2, 1, 0, 1)}
		got := replay(t, ws, q)
		if len(got) != len(q) {
			t.Errorf("want %d entries, got %d", len(q), len(got))
		}
	})

	// ---- boundary probes ---------------------------------------------------

	t.Run("ZeroTransferAndZeroBuild", func(t *testing.T) {
		ws := []policy.WorkerView{ew("w", "n", big, 0)}
		q := []policy.QueuedJob{ej("zero", 2, 0, 0, 0, 1), ej("next", 1, 1, 3, 2, 1)}
		got := replay(t, ws, q)
		want(t, got, "zero", 0, 0, "0 transfer + 0 build: done at its start")
		want(t, got, "next", 0, 5, "the worker is free again at 0: 0 + 2 transfer + 3 build")
	})

	t.Run("ZeroTransferDoesNotSlowANeighbour", func(t *testing.T) {
		ws := []policy.WorkerView{ew("w1", "n", big, 0), ew("w2", "n", big, 0)}
		q := []policy.QueuedJob{ej("nolink", 1, 0, 5, 0, 1), ej("link", 0, 1, 0, 10, 1)}
		got := replay(t, ws, q)
		want(t, got, "nolink", 0, 5, "no transfer: just the 5 s build")
		want(t, got, "link", 0, 10, "the only transfer on the node: full rate, 10")
	})

	t.Run("ManyJobsOneWorkerSerialSum", func(t *testing.T) {
		ws := []policy.WorkerView{ew("w", "n", big, 2)}
		q := []policy.QueuedJob{}
		for i := 0; i < 4; i++ {
			q = append(q, ej(fmt.Sprintf("j%d", i), 0, i, 2, 1, 1))
		}
		got := replay(t, ws, q)
		for i := 0; i < 4; i++ {
			s := 2 + 3*float64(i)
			want(t, got, fmt.Sprintf("j%d", i), s, s+3, fmt.Sprintf("busy 2, then %d earlier jobs of 1+2 = 3 s each", i))
		}
	})

	t.Run("BusyRemainingTie", func(t *testing.T) {
		ws := []policy.WorkerView{ew("w1", "a", big, 3), ew("w2", "b", big, 3)}
		q := []policy.QueuedJob{ej("x", 1, 0, 1, 0, 1), ej("y", 0, 1, 1, 0, 1)}
		got := replay(t, ws, q)
		want(t, got, "x", 3, 4, "both free at 3: each takes one")
		want(t, got, "y", 3, 4, "both free at 3: each takes one")

		ws2 := []policy.WorkerView{ew("w1", "n", big, 3), ew("w2", "n", big, 3)}
		q2 := []policy.QueuedJob{ej("x", 1, 0, 1, 2, 1), ej("y", 0, 1, 1, 2, 1)}
		got2 := replay(t, ws2, q2)
		want(t, got2, "x", 3, 8, "same node, both claim at 3: 2 s transfer at 1/2 = 4 -> 7, +1 build = 8")
		want(t, got2, "y", 3, 8, "same as x")
	})

	// ---- purity (invariant 8) ---------------------------------------------

	richW := []policy.WorkerView{ew("w1", "n1", 100, 1.5), ew("w2", "n1", 30, 0), ew("w3", "", 100, 0.5), ew("w4", "n2", 10, 0)}
	richQ := []policy.QueuedJob{
		ej("a", 3, 0, 4, 2, 80), ej("b", 3, 1, 1, 5, 20), ej("c", 1, 2, 2, 1, 5),
		ej("d", 7, 3, 3, 3, 30), ej("e", 0, 4, 1, 1, 500), ej("f", 1, 5, 0.5, 4, 10),
	}

	t.Run("DeterministicAcrossRepeatedCalls", func(t *testing.T) {
		t.Logf("spec: invariant 8 (pure), go doc ReplayQueue 'It is deterministic'")
		first := replay(t, richW, richQ)
		for i := 0; i < 20; i++ {
			if again := policy.ReplayQueue(richW, richQ); !reflect.DeepEqual(again, first) {
				t.Fatalf("call %d differs: %v vs %v", i, again, first)
			}
		}
		t.Logf("20 repeat calls identical")
	})

	t.Run("PermutedInputsSameAnswer", func(t *testing.T) {
		t.Logf("spec: go doc ReplayQueue 'input order never matters'")
		base := policy.ReplayQueue(richW, richQ)
		t.Logf("base: %s", fmtEsts(base, richQ))
		n := 0
		for r := 0; r < len(richW); r++ {
			for s := 0; s < len(richQ); s++ {
				ws := append(cloneW(richW[r:]), richW[:r]...)
				q := append(cloneQ(richQ[s:]), richQ[:s]...)
				if (r+s)%2 == 1 {
					for i, j := 0, len(q)-1; i < j; i, j = i+1, j-1 {
						q[i], q[j] = q[j], q[i]
					}
				}
				if got := policy.ReplayQueue(ws, q); !reflect.DeepEqual(got, base) {
					t.Errorf("rotation w=%d q=%d differs: %s", r, s, fmtEsts(got, richQ))
				}
				n++
			}
		}
		t.Logf("%d permutations, all identical to base", n)
		nb := policy.NewJobWait(richW, richQ, ej("new", 0, 99, 1, 1, 5))
		rw := cloneW(richW)
		rw[0], rw[3] = rw[3], rw[0]
		rq := cloneQ(richQ)
		rq[0], rq[5] = rq[5], rq[0]
		if nb2 := policy.NewJobWait(rw, rq, ej("new", 0, 99, 1, 1, 5)); !estSame(nb, nb2) {
			t.Errorf("NewJobWait depends on input order: %g vs %g", nb, nb2)
		}
	})

	t.Run("InputsNotMutated", func(t *testing.T) {
		t.Logf("spec: invariant 8 (pure functions of the state passed in)")
		ws, q := cloneW(richW), cloneQ(richQ)
		_ = policy.ReplayQueue(ws, q)
		_ = policy.NewJobWait(ws, q, ej("new", 0, 99, 1, 1, 5))
		if !reflect.DeepEqual(ws, richW) || !reflect.DeepEqual(q, richQ) {
			t.Errorf("caller slices mutated:\nworkers %s\nqueue %s", fmtWorkers(ws), fmtJobs(q))
		} else {
			t.Logf("worker and queue slices unchanged after both calls")
		}
	})

	t.Run("RichScenarioByHand", func(t *testing.T) {
		t.Logf("a mixed case, every number by hand (decisions 11, 12)")
		got := replay(t, richW, richQ)
		// Claim order: d(7) a(3,seq0) b(3,seq1) c(1,seq2) f(1,seq5) e(0).
		// t=0 free: w2(n1,30), w4(n2,10) by ID.
		//   w2: d(30<=30) fits -> d starts 0, transfer 3 on n1 (alone so far).
		//   w4: a(80) no, b(20) no, c(5) yes -> c starts 0, transfer 1 on n2 alone -> 1, build 2 -> done 3.
		// t=0.5 w3 ("" private, 100) frees: a(80) fits -> a starts 0.5, transfer 2 private -> 2.5, build 4 -> 6.5.
		// t=1.5 w1 (n1,100) frees: b(20) -> b starts 1.5, transfer 5 on n1, shared with d.
		//   d at 1.5 has moved 1.5 of 3, 1.5 left. From 1.5 both at 1/2.
		//   d's 1.5 left takes 3 s -> transfer ends 4.5; b moved 1.5 by 4.5, 3.5 left alone -> 8.0.
		//   d done 4.5+3 = 7.5. b done 8.0+1 = 9.0.
		// t=3 w4 frees: remaining f(10<=10) fits -> f starts 3, transfer 4 on n2 alone -> 7, build 0.5 -> 7.5.
		// e(500) fits nobody -> +Inf.
		want(t, got, "d", 0, 7.5, "w2: 0..1.5 full (1.5), 1.5..4.5 half (1.5), build 3")
		want(t, got, "c", 0, 3, "w4: 1 transfer + 2 build")
		want(t, got, "a", 0.5, 6.5, "w3 private link: 0.5+2+4")
		want(t, got, "b", 1.5, 9, "w1: 1.5..4.5 half (1.5), 4.5..8 full (3.5), build 1")
		want(t, got, "f", 3, 7.5, "w4 again at 3: 4 transfer on n2 alone + 0.5")
		want(t, got, "e", inf, inf, "500 fits nobody")
	})

	// ---- NewJobWait (decision 13) -----------------------------------------

	t.Run("NewJobWaitJoinsAtPriorityZeroBehindEveryone", func(t *testing.T) {
		t.Logf("spec: decision 13 'the new build is appended at priority 0 behind every queued job'; the job's own Priority and Seq must not let it jump the queue")
		ws := []policy.WorkerView{ew("w", "n", big, 0)}
		q := []policy.QueuedJob{ej("a", 0, 0, 3, 0, 1), ej("b", 0, 1, 4, 0, 1)}
		w1 := policy.NewJobWait(ws, q, ej("new", 0, 2, 1, 0, 1))
		w2 := policy.NewJobWait(ws, q, ej("new", 1e9, -1, 1, 0, 1))
		t.Logf("queue a(3s), b(4s) on one idle worker; NewJobWait = %g (prio 0) and %g (job passed with prio 1e9, seq -1); hand: 3+4 = 7 both", w1, w2)
		if !estSame(w1, 7) {
			t.Errorf("NewJobWait: got %g, want 7", w1)
		}
		if !estSame(w2, 7) {
			t.Errorf("NewJobWait with a high Priority on the job: got %g, want 7 (decision 13 says it joins at priority 0 behind every queued job)", w2)
		}
	})

	t.Run("NewJobWaitExactForFCFS", func(t *testing.T) {
		t.Logf("spec: decision 13 'exact for the FCFS baseline': under FCFS (all priority 0) NewJobWait equals the replay of the queue with the new job appended last")
		ws := []policy.WorkerView{ew("w1", "n", big, 1), ew("w2", "n", big, 0)}
		q := []policy.QueuedJob{ej("a", 0, 0, 2, 4, 1), ej("b", 0, 1, 1, 6, 1), ej("c", 0, 2, 3, 2, 1)}
		nj := ej("new", 0, 3, 1, 1, 1)
		w := policy.NewJobWait(ws, q, nj)
		full := policy.ReplayQueue(ws, append(cloneQ(q), nj))
		// Hand: t=0 w2 claims a (transfer 4 alone on n): 0..1 moves 1, 3 left.
		//  t=1 w1 frees, claims b; a and b at 1/2. a's 3 left ends at 7 (b moved 3, 3 left); a builds 2 -> 9.
		//  7..9 b alone moves 2 (1 left). t=9 w2 frees, claims c; b and c at 1/2.
		//  b's last 1 ends at 11, builds 1 -> 12. c moved 1 by 11, alone 11..12 moves 1, builds 3 -> 15.
		//  w1 frees at 12 (w2 at 15): new is claimed at 12.
		t.Logf("NewJobWait = %g; ReplayQueue(queue+new).StartS = %g; hand: 12 (a 0..9 on w2, b 1..12 on w1, c 9..15 on w2, new claimed by w1 at 12)", w, full["new"].StartS)
		want(t, full, "a", 0, 9, "w2: 0..1 full (1), 1..7 half (3), build 2")
		want(t, full, "b", 1, 12, "w1: 1..7 half (3), 7..9 full (2), 9..11 half (1), build 1")
		want(t, full, "c", 9, 15, "w2: 9..11 half (1), 11..12 full (1), build 3")
		if !estSame(w, 12) || !estSame(w, full["new"].StartS) {
			t.Errorf("NewJobWait: got %g, want 12 (= replay start %g)", w, full["new"].StartS)
		}
	})

	t.Run("NewJobWaitEmptyPoolIsInf", func(t *testing.T) {
		t.Logf("spec: go doc NewJobWait 'With an empty pool it returns +Inf; the host applies the cold-start rule (keep 0) itself'")
		w := policy.NewJobWait(nil, nil, ej("new", 0, 0, 1, 1, 1))
		t.Logf("NewJobWait(no workers) = %g", w)
		if !math.IsInf(w, 1) {
			t.Errorf("got %g, want +Inf", w)
		}
	})

	t.Run("NewJobWaitFitsNowhereIsInf", func(t *testing.T) {
		t.Logf("spec: decision 13 'a build fitting no live worker gets +Inf'")
		ws := []policy.WorkerView{ew("w", "n", 10, 0)}
		w := policy.NewJobWait(ws, nil, ej("new", 0, 0, 1, 1, 11))
		t.Logf("one idle worker mem 10, new job mem 11 -> %g", w)
		if !math.IsInf(w, 1) {
			t.Errorf("got %g, want +Inf", w)
		}
	})

	t.Run("NewJobWaitIdlePoolIsZeroBusyPoolIsRemaining", func(t *testing.T) {
		ws := []policy.WorkerView{ew("w", "n", 10, 0)}
		if w := policy.NewJobWait(ws, nil, ej("new", 0, 0, 1, 1, 5)); !estSame(w, 0) {
			t.Errorf("idle worker, empty queue: got %g, want 0", w)
		} else {
			t.Logf("idle worker, empty queue -> 0 (claimed immediately)")
		}
		ws2 := []policy.WorkerView{ew("w1", "n", 10, 2.5), ew("w2", "n", 10, 4)}
		if w := policy.NewJobWait(ws2, nil, ej("new", 0, 0, 1, 1, 5)); !estSame(w, 2.5) {
			t.Errorf("two busy workers (2.5, 4), empty queue: got %g, want 2.5", w)
		} else {
			t.Logf("busy 2.5 and 4, empty queue -> 2.5 (earliest free)")
		}
	})

	t.Run("NewJobWaitRespectsMemoryFilter", func(t *testing.T) {
		t.Logf("spec: decisions 12/13: a new job that fits only the big busy worker waits for it, not for the idle small one")
		ws := []policy.WorkerView{ew("big", "n1", 100, 5), ew("small", "n2", 10, 0)}
		w := policy.NewJobWait(ws, nil, ej("new", 0, 0, 1, 0, 50))
		t.Logf("NewJobWait = %g; hand: 5", w)
		if !estSame(w, 5) {
			t.Errorf("got %g, want 5", w)
		}
		// And a small queued job ahead does not delay it if the small worker takes it.
		q := []policy.QueuedJob{ej("s", 0, 0, 100, 0, 5)}
		w2 := policy.NewJobWait(ws, q, ej("new", 0, 1, 1, 0, 50))
		t.Logf("with a 100 s small job queued ahead (taken by the idle small worker at 0): NewJobWait = %g; hand: 5", w2)
		if !estSame(w2, 5) {
			t.Errorf("got %g, want 5", w2)
		}
	})

	t.Run("NewJobWaitBehindNegativePriorityProbe", func(t *testing.T) {
		t.Logf("PROBE (spec gap): decision 13 says both 'at priority 0' and 'behind every queued job'; these differ only when a queued job has negative priority. No assertion; recorded for the report.")
		ws := []policy.WorkerView{ew("w", "n", big, 0)}
		q := []policy.QueuedJob{ej("neg", -5, 0, 10, 0, 1)}
		w := policy.NewJobWait(ws, q, ej("new", 0, 1, 1, 0, 1))
		t.Logf("queued 'neg' prio -5 (10 s) on one idle worker; NewJobWait = %g (0 = joins at priority 0 ahead of it; 10 = behind every queued job)", w)
	})
}
