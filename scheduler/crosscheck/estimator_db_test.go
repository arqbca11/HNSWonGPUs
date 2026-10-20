package crosscheck

// Phase 3 step 2 cross-check, database side: workers.node through Heartbeat
// (decision 11), GetQueueSnapshot (go doc: workers live within maxAge by
// worker_id; queued in claim order priority DESC, enqueued_at; leased by
// build_id with seconds since the claim), and the submit response's
// gpu_queue_wait_ms (decision 13, go doc SubmitBuildResponse: 0 with an empty
// pool, -1 when no live worker's memory could take the build, absent on a
// duplicate). Written without reading store.go or api.go.

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/api"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

// A build size whose transfer term is large enough to see link sharing in ms:
// with costmodel.Default(), 50 000 x 1536 has transfer 3.104 s and GPU build
// about 13.633 s (logged by each test from the model itself).
const (
	estN   int64 = 50_000
	estDim int32 = 1536
)

type submitRaw struct {
	resp api.SubmitBuildResponse
	raw  map[string]json.RawMessage
	code int
	body string
}

func (e *env) submitSized(t *testing.T, round int64, shard int32, n int64, dim int32) submitRaw {
	t.Helper()
	var r api.SubmitBuildResponse
	code, body := e.do(t, "POST", "/builds", api.SubmitBuildRequest{RoundID: round, ShardID: shard, NVectors: n, Dim: dim}, &r)
	if code >= 300 {
		t.Fatalf("POST /builds shard %d: %d %s", shard, code, body)
	}
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatalf("decode raw: %v", err)
	}
	_, has := raw["gpu_queue_wait_ms"]
	t.Logf("POST /builds shard %d (n=%d dim=%d) -> %d placement=%s created=%v gpu_total_ms=%d gpu_queue_wait_ms=%d (key present: %v)",
		shard, n, dim, code, r.Placement, r.Created, r.GPUTotalMs, r.GPUQueueWaitMs, has)
	return submitRaw{resp: r, raw: raw, code: code, body: body}
}

func msOf(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func TestEstimatorDB(t *testing.T) {
	cm := costmodel.Default()
	T := msOf(cm.Transfer(estN, estDim))
	B := msOf(cm.GPUBuild(estN, estDim))
	mem := cm.GPUMemBytes(estN, estDim)
	bigMem := 16 * mem

	t.Run("HeartbeatRoundTripsNode", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 11: workers.node reported in the heartbeat, '' = unknown; go doc Store.Heartbeat")
		must(t, e.st.Heartbeat(e.ctx, "w-b", 100, ""))
		must(t, e.st.Heartbeat(e.ctx, "w-a", 200, "kind-gpu-1"))
		ws, err := e.st.ListWorkers(e.ctx)
		must(t, err)
		got := map[string]string{}
		for _, w := range ws {
			got[w.WorkerID] = w.Node
			t.Logf("[db] workers: %s node=%q mem=%d", w.WorkerID, w.Node, w.MemBytes)
		}
		if got["w-a"] != "kind-gpu-1" || got["w-b"] != "" {
			t.Errorf("ListWorkers nodes = %v, want w-a=kind-gpu-1 w-b=\"\"", got)
		}
		snap, err := e.st.GetQueueSnapshot(e.ctx, time.Minute)
		must(t, err)
		t.Logf("[db] snapshot workers: %+v", snap.Workers)
		if len(snap.Workers) != 2 || snap.Workers[0].WorkerID != "w-a" || snap.Workers[1].WorkerID != "w-b" {
			t.Fatalf("snapshot workers = %+v, want [w-a, w-b] by worker_id", snap.Workers)
		}
		if snap.Workers[0].Node != "kind-gpu-1" || snap.Workers[0].MemBytes != 200 || snap.Workers[1].Node != "" || snap.Workers[1].MemBytes != 100 {
			t.Errorf("snapshot node/mem wrong: %+v", snap.Workers)
		}
		// A later heartbeat reports the current node (spec says the heartbeat
		// reports it; pinned reading: the latest report wins).
		must(t, e.st.Heartbeat(e.ctx, "w-a", 200, "kind-gpu-2"))
		snap, err = e.st.GetQueueSnapshot(e.ctx, time.Minute)
		must(t, err)
		t.Logf("[db] after w-a re-heartbeats with node kind-gpu-2: %+v", snap.Workers)
		if snap.Workers[0].Node != "kind-gpu-2" {
			t.Errorf("w-a node after second heartbeat = %q, want kind-gpu-2", snap.Workers[0].Node)
		}
		var wr api.WorkersResponse
		code, body := e.do(t, "GET", "/workers", nil, &wr)
		t.Logf("GET /workers -> %d %s", code, body)
		if !strings.Contains(body, `"node":"kind-gpu-2"`) {
			t.Errorf("GET /workers does not carry node: %s", body)
		}
	})

	t.Run("SnapshotExcludesStaleWorkers", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc QueueSnapshot.Workers 'live within maxAge'")
		must(t, e.st.Heartbeat(e.ctx, "w-old", 100, "n"))
		time.Sleep(400 * time.Millisecond)
		must(t, e.st.Heartbeat(e.ctx, "w-new", 100, "n"))
		snap, err := e.st.GetQueueSnapshot(e.ctx, 200*time.Millisecond)
		must(t, err)
		t.Logf("w-old heartbeat 400 ms before w-new; snapshot(maxAge 200 ms).Workers = %+v; expected only w-new", snap.Workers)
		if len(snap.Workers) != 1 || snap.Workers[0].WorkerID != "w-new" {
			t.Errorf("want only w-new, got %+v", snap.Workers)
		}
	})

	t.Run("SnapshotQueuedInClaimOrderAndLeased", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc QueueSnapshot: Queued in claim order priority DESC, enqueued_at; Leased by build_id; local builds are not GPU-queued")
		round := e.createRound(t, 6)
		type sub struct {
			shard int32
			prio  float64
			place string
			n     int64
		}
		subs := []sub{{0, 1, "gpu", 1000}, {1, 5, "gpu", 2000}, {2, 5, "gpu", 3000}, {3, 0, "gpu", 4000}, {4, 9, "local", 5000}}
		for _, s := range subs {
			id := store.BuildID(s.shard, round)
			ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: id, RoundID: round, ShardID: s.shard, NVectors: s.n, Dim: 64, MemBytes: s.n * 10, Placement: s.place, Priority: s.prio})
			if err != nil || !ok {
				t.Fatalf("SubmitBuild %s: %v %v", id, ok, err)
			}
			t.Logf("submitted %s placement=%s priority=%g mem=%d", id, s.place, s.prio, s.n*10)
			time.Sleep(5 * time.Millisecond)
		}
		snap, err := e.st.GetQueueSnapshot(e.ctx, time.Minute)
		must(t, err)
		order := ids(snap.Queued)
		t.Logf("[db] snapshot queued: %v", order)
		wantOrder := []string{store.BuildID(1, round), store.BuildID(2, round), store.BuildID(0, round), store.BuildID(3, round)}
		if fmt.Sprint(order) != fmt.Sprint(wantOrder) {
			t.Errorf("queued order %v, want %v (prio 5 by enqueue, then 1, then 0; the local build absent)", order, wantOrder)
		}
		for _, q := range snap.Queued {
			if q.MemBytes != q.NVectors*10 || q.Dim != 64 {
				t.Errorf("queued row fields wrong: %+v", q)
			}
		}
		if len(snap.Leased) != 0 {
			t.Errorf("nothing claimed yet, leased = %+v", snap.Leased)
		}

		must(t, e.st.Heartbeat(e.ctx, "w1", 1<<40, "n"))
		c, ok, err := e.st.Claim(e.ctx, "w1", 1<<40, time.Minute)
		if err != nil || !ok {
			t.Fatalf("Claim: %v %v", ok, err)
		}
		t.Logf("w1 claimed %s attempt %d; the snapshot's first queued row was %s (the claim and the snapshot must agree)", c.BuildID, c.Attempt, wantOrder[0])
		if c.BuildID != order[0] {
			t.Errorf("claim took %s, snapshot's claim order said %s", c.BuildID, order[0])
		}
		time.Sleep(300 * time.Millisecond)
		snap, err = e.st.GetQueueSnapshot(e.ctx, time.Minute)
		must(t, err)
		t.Logf("[db] after claim + 300 ms: queued %v leased %+v", ids(snap.Queued), snap.Leased)
		if len(snap.Leased) != 1 || snap.Leased[0].BuildID != c.BuildID || snap.Leased[0].LeaseOwner != "w1" || snap.Leased[0].NVectors != 2000 {
			t.Fatalf("leased = %+v, want one row %s owned by w1", snap.Leased, c.BuildID)
		}
		if el := snap.Leased[0].ElapsedS; el < 0.25 || el > 5 {
			t.Errorf("ElapsedS = %g, want about 0.3 (seconds since the claim)", el)
		}
		if fmt.Sprint(ids(snap.Queued)) != fmt.Sprint(wantOrder[1:]) {
			t.Errorf("queued after claim %v, want %v", ids(snap.Queued), wantOrder[1:])
		}

		// The claim's memory filter vs the snapshot order: a small worker skips
		// the head if it does not fit; the snapshot keeps the global order.
		must(t, e.st.Heartbeat(e.ctx, "w-small", 15000, "n"))
		c2, ok, err := e.st.Claim(e.ctx, "w-small", 15000, time.Minute)
		if err != nil || !ok {
			t.Fatalf("Claim small: %v %v", ok, err)
		}
		t.Logf("w-small (cap 15000) claimed %s: head %s needs 30000, next %s needs 10000 (decision 12 mirrors this filter)", c2.BuildID, wantOrder[1], wantOrder[2])
		if c2.BuildID != wantOrder[2] {
			t.Errorf("small claim took %s, want %s", c2.BuildID, wantOrder[2])
		}
		snap, err = e.st.GetQueueSnapshot(e.ctx, time.Minute)
		must(t, err)
		t.Logf("[db] leased now %+v (by build_id)", snap.Leased)
		if len(snap.Leased) != 2 || snap.Leased[0].BuildID > snap.Leased[1].BuildID {
			t.Errorf("leased should be two rows sorted by build_id: %+v", snap.Leased)
		}
	})

	t.Run("APIQueueWaitEmptyPoolIsZero", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 13 / go doc SubmitBuildResponse: '0 with an empty pool (the cold-start rule)'")
		round := e.createRound(t, 6)
		r := e.submitSized(t, round, 0, estN, estDim)
		if _, has := r.raw["gpu_queue_wait_ms"]; !has {
			t.Errorf("gpu_queue_wait_ms absent on a fresh submit: %s", r.body)
		}
		if estWaitMs(t, r) != 0 {
			t.Errorf("empty pool: gpu_queue_wait_ms = %d, want 0", estWaitMs(t, r))
		}
	})

	t.Run("APIQueueWaitFitsNowhereIsMinusOne", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 13 '-1 for +Inf'; go doc: '-1 when no live worker's memory could ever take this build'")
		must(t, e.st.Heartbeat(e.ctx, "w-tiny", 1, "n"))
		round := e.createRound(t, 6)
		r := e.submitSized(t, round, 0, estN, estDim)
		t.Logf("worker mem 1 byte, build needs %d -> expected -1", mem)
		if estWaitMs(t, r) != -1 {
			t.Errorf("gpu_queue_wait_ms = %d, want -1", estWaitMs(t, r))
		}
	})

	nodeCase := func(t *testing.T, nodeA, nodeB string, wantMs float64, why string) {
		e := setup(t)
		must(t, e.st.Heartbeat(e.ctx, "w1", bigMem, nodeA))
		must(t, e.st.Heartbeat(e.ctx, "w2", bigMem, nodeB))
		t.Logf("workers w1 node=%q w2 node=%q, both idle, mem %d; each build: transfer T=%.3f ms, GPU build B=%.3f ms, mem %d", nodeA, nodeB, bigMem, T, B, mem)
		round := e.createRound(t, 6)
		r0 := e.submitSized(t, round, 0, estN, estDim)
		r1 := e.submitSized(t, round, 1, estN, estDim)
		r2 := e.submitSized(t, round, 2, estN, estDim)
		if r0.resp.Placement != "gpu" || r1.resp.Placement != "gpu" || r2.resp.Placement != "gpu" {
			t.Fatalf("always_gpu should queue all three: %s %s %s", r0.resp.Placement, r1.resp.Placement, r2.resp.Placement)
		}
		if estWaitMs(t, r0) != 0 || estWaitMs(t, r1) != 0 {
			t.Errorf("first two builds each find an idle worker: want 0 and 0, got %d and %d", estWaitMs(t, r0), estWaitMs(t, r1))
		}
		got := float64(estWaitMs(t, r2))
		t.Logf("third build: gpu_queue_wait_ms = %d; hand: %s = %.3f ms", estWaitMs(t, r2), why, wantMs)
		if math.Abs(got-wantMs) > 2 {
			t.Errorf("third build's wait %v ms, want %.3f ms (±2) [%s]", got, wantMs, why)
		}
	}

	t.Run("APIQueueWaitSameNodeSharesLink", func(t *testing.T) {
		t.Logf("spec: decisions 11, 12, 13: two builds claimed together on one node share its link, so both finish at 2T+B and the third waits that long")
		nodeCase(t, "nodeA", "nodeA", 2*T+B, "2T + B")
	})

	t.Run("APIQueueWaitSeparateNodesFullRate", func(t *testing.T) {
		t.Logf("spec: decisions 11, 12, 13: separate nodes, each build finishes at T+B")
		nodeCase(t, "nodeA", "nodeB", T+B, "T + B")
	})

	t.Run("APIQueueWaitUnknownNodesPrivateLinks", func(t *testing.T) {
		t.Logf("spec: decision 11: node '' = a private link per worker")
		nodeCase(t, "", "", T+B, "T + B (unknown nodes do not contend)")
	})

	t.Run("APIQueueWaitCountsLeasedRemaining", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: step 2 notes 'a leased build's remaining time is its GPU total minus the time since its claim, floored at 0'; decision 13 joins at the back")
		must(t, e.st.Heartbeat(e.ctx, "w1", bigMem, "n"))
		round := e.createRound(t, 6)
		r0 := e.submitSized(t, round, 0, estN, estDim)
		c, ok, err := e.st.Claim(e.ctx, "w1", bigMem, time.Minute)
		if err != nil || !ok {
			t.Fatalf("Claim: %v %v", ok, err)
		}
		t.Logf("w1 claimed %s (attempt %d)", c.BuildID, c.Attempt)
		claimed := time.Now()
		r1 := e.submitSized(t, round, 1, estN, estDim)
		elapsed := msOf(time.Since(claimed))
		total := T + B
		t.Logf("hand: w1 busy for GPU total %.3f ms minus time since claim (< %.1f ms) -> wait in [%.1f, %.1f]; response's own gpu_total_ms %d", total, elapsed, total-elapsed-1, total+1, r0.resp.GPUTotalMs)
		w1 := float64(estWaitMs(t, r1))
		if w1 < total-elapsed-1000 || w1 > total+2 {
			t.Errorf("wait behind the leased build = %v ms, want about %.0f ms", w1, total)
		}
		r2 := e.submitSized(t, round, 2, estN, estDim)
		w2 := float64(estWaitMs(t, r2))
		t.Logf("hand: the third waits for the leased build's remainder plus the queued second build's T+B: about %.0f ms", 2*total)
		if w2 < 2*total-1000 || w2 > 2*total+2 {
			t.Errorf("wait behind leased + one queued = %v ms, want about %.0f ms", w2, 2*total)
		}
	})

	t.Run("APIDuplicateOmitsQueueWait", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 13 'absent on a duplicate, whose answer is the original decision'; go doc SubmitBuildResponse 'Absent on a duplicate'")
		must(t, e.st.Heartbeat(e.ctx, "w1", bigMem, "n"))
		round := e.createRound(t, 6)
		first := e.submitSized(t, round, 0, estN, estDim)
		_ = e.submitSized(t, round, 1, estN, estDim) // so a fresh estimate for shard 0's slot would be non-zero
		dup := e.submitSized(t, round, 0, estN, estDim)
		t.Logf("first: created=%v body=%s", first.resp.Created, first.body)
		t.Logf("duplicate: code=%d created=%v body=%s", dup.code, dup.resp.Created, dup.body)
		if dup.resp.Created {
			t.Errorf("duplicate reported created=true")
		}
		if _, has := dup.raw["gpu_queue_wait_ms"]; has {
			t.Errorf("duplicate response carries gpu_queue_wait_ms (=%s); spec says absent", string(dup.raw["gpu_queue_wait_ms"]))
		}
		b := e.mustBuild(t, store.BuildID(0, round))
		logBuild(t, "after duplicate", b)
		if b.Attempt != 0 || b.State != "queued" {
			t.Errorf("duplicate changed the row: %+v", b)
		}
	})
}

// estWaitMs dereferences the response's gpu_queue_wait_ms, failing if it is
// absent: every subtest here but the duplicate one expects a fresh estimate.
// (The field became *int64 in the triage of this cross-check's own finding.)
func estWaitMs(t *testing.T, r submitRaw) int64 {
	t.Helper()
	if r.resp.GPUQueueWaitMs == nil {
		t.Fatalf("gpu_queue_wait_ms absent where an estimate was expected: %s", r.body)
	}
	return *r.resp.GPUQueueWaitMs
}

func ids(q []store.SnapshotQueued) []string {
	out := make([]string, len(q))
	for i, r := range q {
		out[i] = r.BuildID
	}
	return out
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

var _ = http.StatusOK
