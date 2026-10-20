package policy

import (
	"math"
	"testing"
)

// Every expected number in this file is computed by hand in the test's
// narration, so the log reads as the worked example.

func almost(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestReplayRoadmapExample(t *testing.T) {
	// The roadmap's own example: w1 busy 4 s, w2 free; j1 (6 s) outranks j2 (3 s).
	workers := []WorkerView{
		{ID: "w1", Node: "n1", MemBytes: 8 << 30, BusyRemainingS: 4},
		{ID: "w2", Node: "n2", MemBytes: 8 << 30, BusyRemainingS: 0},
	}
	queue := []QueuedJob{
		{BuildID: "j1", Priority: 2, Seq: 0, GPUBuildS: 6},
		{BuildID: "j2", Priority: 1, Seq: 1, GPUBuildS: 3},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("w1 busy 4s, w2 idle; j1 (pri 2, 6s) claimed by w2 at 0, done 6; j2 (pri 1, 3s) claimed by w1 at 4, done 7")
	t.Logf("got j1=%+v j2=%+v", got["j1"], got["j2"])
	if !almost(got["j1"].StartS, 0) || !almost(got["j1"].DoneS, 6) {
		t.Fatalf("j1: want start 0 done 6, got %+v", got["j1"])
	}
	if !almost(got["j2"].StartS, 4) || !almost(got["j2"].DoneS, 7) {
		t.Fatalf("j2: want start 4 done 7, got %+v", got["j2"])
	}
}

func TestReplayTransferSharingOneNode(t *testing.T) {
	// Two idle workers. On one node two 10 s transfers each run at half the
	// link: both finish at 20, builds at 25. On two nodes: 10 and 15.
	queue := []QueuedJob{
		{BuildID: "a", Priority: 1, Seq: 0, GPUBuildS: 5, TransferS: 10},
		{BuildID: "b", Priority: 1, Seq: 1, GPUBuildS: 5, TransferS: 10},
	}
	same := ReplayQueue([]WorkerView{
		{ID: "w1", Node: "n1", MemBytes: 1 << 30},
		{ID: "w2", Node: "n1", MemBytes: 1 << 30},
	}, queue)
	t.Logf("same node: both claimed at 0, link shared, transfers 0..20, builds 20..25: a=%+v b=%+v", same["a"], same["b"])
	for _, id := range []string{"a", "b"} {
		if !almost(same[id].StartS, 0) || !almost(same[id].DoneS, 25) {
			t.Fatalf("%s on one node: want start 0 done 25, got %+v", id, same[id])
		}
	}
	diff := ReplayQueue([]WorkerView{
		{ID: "w1", Node: "n1", MemBytes: 1 << 30},
		{ID: "w2", Node: "n2", MemBytes: 1 << 30},
	}, queue)
	t.Logf("two nodes: full link each, transfers 0..10, builds 10..15: a=%+v b=%+v", diff["a"], diff["b"])
	for _, id := range []string{"a", "b"} {
		if !almost(diff[id].DoneS, 15) {
			t.Fatalf("%s on two nodes: want done 15, got %+v", id, diff[id])
		}
	}
}

func TestReplayTransferSharingStaggered(t *testing.T) {
	// w1 idle, w2 busy 4 s, one node. j1 transfers alone 0..4 (6 s of work
	// left), shares 4..16 (spends 12 wall seconds on them), builds 16..21.
	// j2 starts at 4, has 4 s of transfer left when j1's ends at 16, finishes
	// transferring at 20, builds 20..25.
	workers := []WorkerView{
		{ID: "w1", Node: "n1", MemBytes: 1 << 30},
		{ID: "w2", Node: "n1", MemBytes: 1 << 30, BusyRemainingS: 4},
	}
	queue := []QueuedJob{
		{BuildID: "j1", Priority: 2, Seq: 0, GPUBuildS: 5, TransferS: 10},
		{BuildID: "j2", Priority: 1, Seq: 1, GPUBuildS: 5, TransferS: 10},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("hand replay: j1 start 0 done 21, j2 start 4 done 25; got j1=%+v j2=%+v", got["j1"], got["j2"])
	if !almost(got["j1"].StartS, 0) || !almost(got["j1"].DoneS, 21) {
		t.Fatalf("j1: want start 0 done 21, got %+v", got["j1"])
	}
	if !almost(got["j2"].StartS, 4) || !almost(got["j2"].DoneS, 25) {
		t.Fatalf("j2: want start 4 done 25, got %+v", got["j2"])
	}
}

func TestReplayClaimIsMemoryAware(t *testing.T) {
	// The small idle worker must not sit behind the big job it cannot take:
	// it claims the lower-priority small job at once, like the real claim's
	// per-worker memory filter.
	workers := []WorkerView{
		{ID: "big", Node: "n1", MemBytes: 100, BusyRemainingS: 5},
		{ID: "small", Node: "n2", MemBytes: 10},
	}
	queue := []QueuedJob{
		{BuildID: "jbig", Priority: 10, Seq: 0, GPUBuildS: 2, MemBytes: 50},
		{BuildID: "jsmall", Priority: 1, Seq: 1, GPUBuildS: 3, MemBytes: 5},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("jsmall claimed by the idle small worker at 0 despite lower priority; jbig waits for big at 5: %+v %+v", got["jsmall"], got["jbig"])
	if !almost(got["jsmall"].StartS, 0) || !almost(got["jsmall"].DoneS, 3) {
		t.Fatalf("jsmall: want start 0 done 3, got %+v", got["jsmall"])
	}
	if !almost(got["jbig"].StartS, 5) || !almost(got["jbig"].DoneS, 7) {
		t.Fatalf("jbig: want start 5 done 7, got %+v", got["jbig"])
	}
}

func TestReplayFitsNowhereIsInfinite(t *testing.T) {
	workers := []WorkerView{{ID: "w1", Node: "n1", MemBytes: 10}}
	queue := []QueuedJob{
		{BuildID: "huge", Priority: 99, Seq: 0, GPUBuildS: 1, MemBytes: 20},
		{BuildID: "ok", Priority: 1, Seq: 1, GPUBuildS: 2, MemBytes: 5},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("huge fits no worker: +Inf, and it does not block ok: %+v %+v", got["huge"], got["ok"])
	if !math.IsInf(got["huge"].StartS, 1) || !math.IsInf(got["huge"].DoneS, 1) {
		t.Fatalf("huge: want +Inf, got %+v", got["huge"])
	}
	if !almost(got["ok"].StartS, 0) || !almost(got["ok"].DoneS, 2) {
		t.Fatalf("ok: want start 0 done 2, got %+v", got["ok"])
	}
	empty := ReplayQueue(nil, queue)
	if !math.IsInf(empty["ok"].StartS, 1) {
		t.Fatalf("zero workers: want +Inf for every job, got %+v", empty["ok"])
	}
}

func TestReplayFIFOWithinEqualPriority(t *testing.T) {
	workers := []WorkerView{{ID: "w1", Node: "n1", MemBytes: 1 << 30}}
	queue := []QueuedJob{
		{BuildID: "second", Priority: 0, Seq: 1, GPUBuildS: 2},
		{BuildID: "first", Priority: 0, Seq: 0, GPUBuildS: 3},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("equal priority: enqueued order wins; first 0..3, second 3..5: %+v %+v", got["first"], got["second"])
	if !almost(got["first"].StartS, 0) || !almost(got["second"].StartS, 3) {
		t.Fatalf("want first at 0 then second at 3, got %+v %+v", got["first"], got["second"])
	}
}

func TestReplayBusyWorkerIsOpaque(t *testing.T) {
	// A busy worker's in-flight build might be mid-transfer, but the host
	// cannot see that, so the replay charges no link contention for it: the
	// idle worker's 10 s transfer on the same node runs at the full link.
	workers := []WorkerView{
		{ID: "w1", Node: "n1", MemBytes: 1 << 30, BusyRemainingS: 8},
		{ID: "w2", Node: "n1", MemBytes: 1 << 30},
	}
	queue := []QueuedJob{{BuildID: "j", Priority: 1, Seq: 0, GPUBuildS: 5, TransferS: 10}}
	got := ReplayQueue(workers, queue)
	t.Logf("documented approximation: the opaque busy build does not share the link; j transfers 0..10, builds 10..15: %+v", got["j"])
	if !almost(got["j"].DoneS, 15) {
		t.Fatalf("want done 15, got %+v", got["j"])
	}
}

func TestReplayUnknownNodesDoNotContend(t *testing.T) {
	// Two workers with Node "" are unknowns: each gets its own link, so two
	// 10 s transfers run in parallel at full rate, not shared.
	workers := []WorkerView{
		{ID: "w1", MemBytes: 1 << 30},
		{ID: "w2", MemBytes: 1 << 30},
	}
	queue := []QueuedJob{
		{BuildID: "a", Priority: 1, Seq: 0, GPUBuildS: 5, TransferS: 10},
		{BuildID: "b", Priority: 1, Seq: 1, GPUBuildS: 5, TransferS: 10},
	}
	got := ReplayQueue(workers, queue)
	t.Logf("empty Node = own link: both done at 15, no phantom contention: %+v %+v", got["a"], got["b"])
	for _, id := range []string{"a", "b"} {
		if !almost(got[id].DoneS, 15) {
			t.Fatalf("%s: want done 15, got %+v", id, got[id])
		}
	}
}

func TestReplayDeterministicUnderPermutation(t *testing.T) {
	workers := []WorkerView{
		{ID: "w2", Node: "n1", MemBytes: 1 << 30, BusyRemainingS: 2},
		{ID: "w1", Node: "n1", MemBytes: 1 << 30},
	}
	queue := []QueuedJob{
		{BuildID: "a", Priority: 3, Seq: 0, GPUBuildS: 4, TransferS: 2},
		{BuildID: "b", Priority: 1, Seq: 1, GPUBuildS: 1, TransferS: 3},
		{BuildID: "c", Priority: 2, Seq: 2, GPUBuildS: 2, TransferS: 1},
	}
	base := ReplayQueue(workers, queue)
	permW := []WorkerView{workers[1], workers[0]}
	permQ := []QueuedJob{queue[2], queue[0], queue[1]}
	again := ReplayQueue(permW, permQ)
	t.Logf("permuted workers and queue give the identical answer: %+v", base)
	for id, want := range base {
		if got := again[id]; !almost(got.StartS, want.StartS) || !almost(got.DoneS, want.DoneS) {
			t.Fatalf("%s: permuted input changed the estimate: %+v vs %+v", id, want, got)
		}
	}
}

func TestNewJobWaitJoinsAtTheBack(t *testing.T) {
	// One worker, one queued job of 10 s: a new build joins at priority 0
	// behind it and waits 10 s. An empty pool says +Inf (the host applies the
	// cold-start rule itself).
	workers := []WorkerView{{ID: "w1", Node: "n1", MemBytes: 1 << 30}}
	queue := []QueuedJob{{BuildID: "q1", Priority: 5, Seq: 0, GPUBuildS: 10}}
	wait := NewJobWait(workers, queue, QueuedJob{BuildID: "new", GPUBuildS: 3, MemBytes: 1})
	t.Logf("new job behind a 10s queue on one worker: wait %.1fs", wait)
	if !almost(wait, 10) {
		t.Fatalf("want 10, got %v", wait)
	}
	if w := NewJobWait(nil, nil, QueuedJob{BuildID: "new", GPUBuildS: 3}); !math.IsInf(w, 1) {
		t.Fatalf("empty pool: want +Inf, got %v", w)
	}
}
