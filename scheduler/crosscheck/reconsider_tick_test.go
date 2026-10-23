package crosscheck

// Phase 3 step 3 cross-check, the actuator: reconsider.Tick driven directly
// with the real deployable policies (policy.ByName "reactive" and
// "reactive_spec") against real Postgres rows. Expected outcomes are hand
// computed from the roadmap's stay-vs-migrate arithmetic ("The policies",
// the Reactive migration flow in docs/design/phase3-policies.md), the
// default margin of 2 s (decision 8), the 1 s reprioritize delta (decision
// 10), cost model v0 (go doc costmodel) and decisions 14-19. Written
// without reading reconsider.go, store.go or the policy source.

import (
	"fmt"
	"math"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/api"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
	"io"
	"log/slog"
)

func TestReconsiderTick(t *testing.T) {
	big := int64(1) << 40
	L := rxSecs(rxCM.CPUBuild(rxN, rxDim))       // 20.631 s
	G := rxSecs(rxCM.GPUTotal(rxN, rxDim))       // 4.639 s (build 4.063 + transfer 0.576)
	GBig := rxSecs(rxCM.GPUTotal(rxBigN, rxDim)) // 65.519 s
	LBig := rxSecs(rxCM.CPUBuild(rxBigN, rxDim)) // 519.99 s
	mem := rxCM.GPUMemBytes(rxN, rxDim)
	const margin = 2.0

	t.Logf("cost model v0 at dim 128: 100k local L=%.3f s, GPU total G=%.3f s, mem %d; 2M local %.3f s, GPU total %.3f s; margin %.0f s (decision 8)", L, G, mem, LBig, GBig, margin)

	// hurting sets up a round with shard 0 building 100k locally at progress
	// prog with q non-index queries of 1 s waiting (meanMs from one record).
	hurting := func(t *testing.T, e *env, nShards int32, prog float32, q int32) (int64, string) {
		round := e.createRound(t, nShards)
		id := e.rxSubmit(t, round, 0, rxN, "local", 0)
		e.rxReport(t, round, 0, q, 0, prog, 1000, false)
		e.rxAgeRound(t, round)
		return round, id
	}

	t.Run("PreemptsHurtingLocalBuild", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: roadmap policy 3 / Reactive migration flow; decision 14 (Tick executes preempts as guarded updates); decision 10 (priority after a preempt = the backlog behind the build)")
		round, id := hurting(t, e, 2, 0, 50)
		e.rxHeartbeat(t, "w1", big)
		stay := L*(1-0) + 50*1.0
		migrate := math.Max(50*1.0, 0+G+0)
		t.Logf("hand: stay = remaining %.3f + backlog 50x1 s = %.3f; migrate = max(non-index drain 50, wait 0 + G %.3f + index drain 0) = %.3f; stay - migrate = %.3f > margin 2 -> PREEMPT at priority 50 (the backlog)", L, stay, G, migrate, stay-migrate)
		lines := e.rxTick(t, "reactive", 1)
		b := e.rxRow(t, "after tick", id)
		if b.Placement != "gpu" || b.State != "queued" || !b.Moved {
			t.Fatalf("want gpu/queued moved=true after the tick, got %s/%s moved=%v", b.Placement, b.State, b.Moved)
		}
		if math.Abs(b.Priority-50) > 0.01 {
			t.Errorf("priority %g, want 50 (decision 10: the backlog behind the build, queue_depth 50 x mean 1 s)", b.Priority)
		}
		var rep api.ReportResponse
		code, body := e.do(t, "POST", fmt.Sprintf("/shards/%d/0/report", round), api.ReportRequest{QueueDepth: 50}, &rep)
		t.Logf("shard 0's next report -> %d %s (the shard learns through its report reply)", code, body)
		if rep.Build.Placement != "gpu" || rep.Build.State != "queued" {
			t.Errorf("report reply %+v, want gpu/queued", rep.Build)
		}
		// CLAUDE.md conventions: every log line about a build carries build_id, attempt and round_id.
		for _, l := range lines {
			if strings.Contains(l, "build_id=") && (!strings.Contains(l, "attempt=") || !strings.Contains(l, "round_id=")) {
				t.Errorf("CLAUDE.md convention: log line about a build lacks attempt or round_id: %s", l)
			}
		}
	})

	t.Run("ReprioritizeTracksBacklogStrictlyPastOneSecond", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 10: a preempted build's priority is recomputed as the backlog behind it, 'strictly past a 1 s churn delta'; go doc Reprioritize")
		round, id := hurting(t, e, 2, 0, 50)
		e.rxHeartbeat(t, "w1", big)
		e.rxTick(t, "reactive", 1)
		b := e.rxRow(t, "after preempt", id)
		if b.State != "queued" || !b.Moved {
			t.Fatalf("fixture: preempt did not happen")
		}
		steps := []struct {
			q    int32
			want float64
			why  string
		}{
			{50, 50, "unchanged backlog -> unchanged priority (decision 10: stable across ticks)"},
			{51, 50, "backlog 51 vs priority 50: delta exactly 1 s, not strictly more -> unchanged"},
			{52, 52, "delta 2 s > 1 -> reprioritize to 52"},
			{51, 52, "delta back down exactly 1 -> unchanged"},
			{40, 40, "delta -12 -> reprioritize to 40"},
		}
		for _, s := range steps {
			e.rxReport(t, round, 0, s.q, 0, 0, 1000, false)
			e.rxTick(t, "reactive", 1)
			got := e.rxRow(t, fmt.Sprintf("after report q=%d", s.q), id)
			t.Logf("hand: %s", s.why)
			if math.Abs(got.Priority-s.want) > 0.01 {
				t.Errorf("q=%d: priority %g, want %g (%s)", s.q, got.Priority, s.want, s.why)
			}
			if got.State != "queued" || got.Placement != "gpu" {
				t.Errorf("q=%d: the preempted build moved again: %s/%s", s.q, got.Placement, got.State)
			}
		}
	})

	t.Run("SunkCostRestrainsPreemption", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: roadmap 'Sunk cost is in the comparison through build_progress'; decision 8 margin 2 s")
		round := e.createRound(t, 2)
		a := e.rxSubmit(t, round, 0, rxN, "local", 0)
		b := e.rxSubmit(t, round, 1, rxN, "local", 0)
		e.rxReport(t, round, 0, 50, 0, 0.95, 1000, false)
		e.rxReport(t, round, 1, 50, 0, 0.85, 1000, false)
		e.rxAgeRound(t, round)
		e.rxHeartbeat(t, "w1", big)
		ra, rb := L*(1-0.95), L*(1-float64(float32(0.85)))
		t.Logf("hand: shard 0 at 95%%: remaining %.3f; stay = %.3f + 50 vs migrate = max(50, %.3f) = 50 -> gain %.3f < 2 -> LEAVE", ra, ra, G, ra)
		t.Logf("hand: shard 1 at 85%%: remaining %.3f; gain %.3f > 2 -> PREEMPT", rb, rb)
		e.rxTick(t, "reactive", 1)
		ga, gb := e.rxRow(t, "shard 0 (95%)", a), e.rxRow(t, "shard 1 (85%)", b)
		if ga.Placement != "local" || ga.State != "running" || ga.Moved {
			t.Errorf("shard 0 at 95%% was moved: %s/%s", ga.Placement, ga.State)
		}
		if gb.Placement != "gpu" || gb.State != "queued" || !gb.Moved {
			t.Errorf("shard 1 at 85%% was not preempted: %s/%s", gb.Placement, gb.State)
		}
	})

	t.Run("NoPreemptOnEmptyPoolOrWhenFitsNowhere", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: bug 11 fix 'never preempts a build that fits no live worker — including when the pool is empty'; decision 19")
		_, id := hurting(t, e, 2, 0, 50)
		e.rxTick(t, "reactive", 1)
		b := e.rxRow(t, "no workers at all", id)
		if b.Placement != "local" || b.Moved {
			t.Errorf("preempted onto an empty pool")
		}
		e.rxHeartbeat(t, "w-small", mem-1)
		e.rxTick(t, "reactive", 1)
		b = e.rxRow(t, fmt.Sprintf("one worker of %d bytes, build needs %d", mem-1, mem), id)
		if b.Placement != "local" || b.Moved {
			t.Errorf("preempted a build that fits no live worker")
		}
		e.rxHeartbeat(t, "w-ok", mem)
		e.rxTick(t, "reactive", 1)
		b = e.rxRow(t, fmt.Sprintf("control: a worker of exactly %d bytes joins", mem), id)
		if b.Placement != "gpu" || !b.Moved {
			t.Errorf("control: with a fitting idle worker the same hurting build should be preempted")
		}
	})

	t.Run("RenegesQueuedBuildWhenShrunkenPoolMakesQueueSlow", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: roadmap 'move a queued GPU build back to local when its GPU estimate has grown past its local estimate'; decision 19 'a shrunken pool produces genuinely longer replayed waits and reneging reacts to those'; per-tick ReplayQueue (step 3 notes)")
		round := e.createRound(t, 6)
		var bigs []string
		for s := int32(1); s <= 5; s++ {
			bigs = append(bigs, e.rxSubmit(t, round, s, rxBigN, "gpu", 1000))
		}
		y := e.rxSubmit(t, round, 0, rxN, "gpu", 0)
		e.rxReport(t, round, 0, 0, 0, 0, 0, false)
		e.rxAgeRound(t, round)
		e.rxHeartbeat(t, "w1", big)
		wait := 5 * GBig
		t.Logf("hand: one worker, five 2M builds ahead at priority 1000: shard 0 starts at 5 x %.3f = %.3f s; GPU recovery %.3f vs local from scratch %.3f + margin -> RENEGE. The k-th 2M build finishes at k x %.3f <= %.3f < its local %.3f -> stays", GBig, wait, wait+G, L, GBig, wait, LBig)
		e.rxTick(t, "reactive", 1)
		yb := e.rxRow(t, "shard 0", y)
		if yb.Placement != "local" || yb.State != "running" || !yb.Moved {
			t.Errorf("shard 0 was not reneged: %s/%s moved=%v", yb.Placement, yb.State, yb.Moved)
		}
		for _, id := range bigs {
			bb := e.rxRow(t, "2M", id)
			if bb.Placement != "gpu" || bb.State != "queued" || bb.Moved {
				t.Errorf("%s moved", id)
			}
		}
		// Decision 8: a reneged build is never preempted, however much it hurts.
		for i := 2; i <= 6; i++ {
			e.rxHeartbeat(t, fmt.Sprintf("w%d", i), big)
		}
		e.rxReport(t, round, 0, 100, 0, 0, 1000, false)
		t.Logf("now six idle workers (wait 0 for a would-be preempt) and shard 0 reports 100 x 1 s non-index backlog at progress 0: stay %.1f vs migrate 100 would preempt, but the build has moved once already", L+100)
		e.rxTick(t, "reactive", 1)
		yb = e.rxRow(t, "shard 0 after the hurting report", y)
		if yb.Placement != "local" || yb.State != "running" {
			t.Errorf("decision 8: a reneged build was preempted back")
		}
	})

	t.Run("EmptyPoolHoldsQueuedBuilds", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 19 'with zero live workers every QueueWaitS reads 0, so queued builds hold their place'")
		round := e.createRound(t, 6)
		for s := int32(1); s <= 5; s++ {
			e.rxSubmit(t, round, s, rxBigN, "gpu", 1000)
		}
		y := e.rxSubmit(t, round, 0, rxN, "gpu", 0)
		e.rxReport(t, round, 0, 0, 0, 0, 0, false)
		e.rxAgeRound(t, round)
		e.rxHeartbeat(t, "w-stale", big)
		if _, err := e.pool.Exec(e.ctx, `UPDATE workers SET last_seen = now() - interval '1 hour'`); err != nil {
			t.Fatal(err)
		}
		t.Logf("same queue as the shrunken-pool case, but the only worker last heartbeated 1 h ago (StaleAfter 1 min): an empty pool. hand: wait reads 0 -> GPU %.3f < local %.3f -> HOLD", G, L)
		e.rxTick(t, "reactive", 1)
		yb := e.rxRow(t, "shard 0", y)
		if yb.Placement != "gpu" || yb.State != "queued" || yb.Moved {
			t.Errorf("decision 19: queued build reneged on an empty pool")
		}
	})

	t.Run("FitsNowhereQueuedBuildEscapesByRenege", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 19 'a queued build that fits no live worker still reads +Inf, and reneging it to local is exactly how a stuck too-big build escapes the queue — bug 11's cure'")
		round := e.createRound(t, 2)
		y := e.rxSubmit(t, round, 0, rxN, "gpu", 0)
		z := e.rxSubmit(t, round, 1, 10_000, "gpu", 0)
		e.rxAgeRound(t, round)
		e.rxHeartbeat(t, "w-small", 50_000_000)
		zl, zg := rxSecs(rxCM.CPUBuild(10_000, rxDim)), rxSecs(rxCM.GPUTotal(10_000, rxDim))
		t.Logf("hand: worker 50 MB; shard 0 (100k) needs %d -> wait +Inf -> RENEGE; shard 1 (10k, %d bytes) fits: local %.3f vs GPU %.3f, local wins by %.3f < margin -> stays", mem, rxCM.GPUMemBytes(10_000, rxDim), zl, zg, zg-zl)
		e.rxTick(t, "reactive", 1)
		yb := e.rxRow(t, "shard 0 (fits nowhere)", y)
		if yb.Placement != "local" || yb.State != "running" || !yb.Moved {
			t.Errorf("fits-nowhere queued build was not reneged: %s/%s", yb.Placement, yb.State)
		}
		zb := e.rxRow(t, "shard 1 (fits)", z)
		if zb.Placement != "gpu" || zb.State != "queued" {
			t.Errorf("fitting build was moved")
		}
	})

	t.Run("FitsNowhereOnEmptyPoolHolds", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 19's first clause: zero live workers -> every QueueWaitS reads 0 (the +Inf clause is about live workers that cannot take it)")
		round := e.createRound(t, 1)
		y := e.rxSubmit(t, round, 0, rxBigN, "gpu", 0)
		e.rxAgeRound(t, round)
		e.rxTick(t, "reactive", 1)
		yb := e.rxRow(t, "2M build, no workers", y)
		if yb.Placement != "gpu" || yb.State != "queued" {
			t.Errorf("reneged on an empty pool (spec reading: hold); got %s/%s", yb.Placement, yb.State)
		}
	})

	t.Run("PreemptedFitsNowhereBuildEscapesByRenege", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 20 (the step 3 triage's F3 fix, owner approved): a moved build whose memory fits no live worker reneges anyway — impossibility is not a cost judgment, and preemption still requires never-moved, so no loop can form. Probe: preempt onto a big worker, then the big worker shrinks away")
		_, id := hurting(t, e, 2, 0, 50)
		e.rxHeartbeat(t, "w1", big)
		e.rxTick(t, "reactive", 1)
		if b := e.rxRow(t, "after preempt", id); !b.Moved || b.State != "queued" {
			t.Fatalf("fixture: no preempt")
		}
		if _, err := e.pool.Exec(e.ctx, `UPDATE workers SET last_seen = now() - interval '1 hour' WHERE worker_id = 'w1'`); err != nil {
			t.Fatal(err)
		}
		e.rxHeartbeat(t, "w-small", 1000)
		t.Logf("w1 is now stale; the only live worker has 1000 bytes. The preempted build reads +Inf: without the escape this is bug 11's hang through preempt-then-shrink (bug 13).")
		e.rxTick(t, "reactive", 1)
		b := e.rxRow(t, "after tick", id)
		if b.Placement != "local" || b.State != "running" || !b.Moved {
			t.Errorf("decision 20: the stranded build must escape to local; got %s/%s moved=%v", b.Placement, b.State, b.Moved)
		}
		if _, ok, _ := e.st.Claim(e.ctx, "w-small", 1000, time.Minute); ok {
			t.Errorf("something is still claimable after the escape")
		}
		t.Logf("the build escaped to local/running (moved stays true); the queue is empty and the round can finish on the shard's CPU")
		// And the loop really is closed: another tick must not preempt it back.
		e.rxHeartbeat(t, "w1", big) // the big worker returns
		e.rxTick(t, "reactive", 1)
		if b := e.rxRow(t, "after the big worker returns", id); b.Placement != "local" {
			t.Errorf("the escaped build was preempted again: flapping")
		}
		t.Logf("the big worker came back and the next tick left the escaped build alone: no flapping loop")
	})

	t.Run("Decision18LeaseElapsedIsScaled", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 18: a lease's elapsed wall time is multiplied by FAKE_TIME_SCALE before it meets a cost-model duration")
		round, id := hurting(t, e, 2, 0, 10)
		x := e.rxSubmit(t, round, 1, rxBigN, "gpu", 0)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if c.BuildID != x {
			t.Fatalf("claim %s", c.BuildID)
		}
		time.Sleep(100 * time.Millisecond)
		stay := L + 10
		t.Logf("hand: stay = %.3f + 10 = %.3f; migrate = max(10, wait + %.3f). Preempt iff wait < %.3f.", L, stay, G, stay-margin-G)
		t.Logf("hand, scale 1: w1's 2M lease has %.3f - ~0.1 = ~%.1f s left -> migrate %.1f -> LEAVE", GBig, GBig-0.1, GBig-0.1+G)
		e.rxTick(t, "reactive", 1)
		b := e.rxRow(t, "after scale-1 tick", id)
		if b.Placement != "local" || b.Moved {
			t.Errorf("scale 1: preempted although the only worker is busy for ~65 s")
		}
		t.Logf("hand, scale 100000: ~0.1 s wall = ~10000 modeled s >= %.3f -> w1 counts as free now -> migrate max(10, %.3f) = 10 -> PREEMPT", GBig, G)
		e.rxTick(t, "reactive", 100000)
		b = e.rxRow(t, "after scale-1e5 tick", id)
		if b.Placement != "gpu" || !b.Moved {
			t.Errorf("scale 1e5: not preempted; the lease's elapsed time does not seem to be scaled")
		}
	})

	t.Run("Decision18SubmitEstimateScalesLeaseElapsed", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 18 'a lease's elapsed time (here and in step 2's submit-time estimate)'; go doc api.Config.TimeScale")
		logger := slog.New(slog.NewTextHandler(io.Discard, nil))
		fast := httptest.NewServer(api.New(e.st, policy.AlwaysGPU{}, rxCM, api.Config{WorkerStaleAfter: time.Minute, TimeScale: 1000}, logger))
		defer fast.Close()
		round := e.createRound(t, 3)
		e.rxSubmit(t, round, 0, rxBigN, "gpu", 0)
		e.rxHeartbeat(t, "w1", big)
		e.rxClaim(t, "w1", big, time.Minute)
		claimed := time.Now()
		time.Sleep(200 * time.Millisecond)
		// Submit through the scale-1000 server first: 0.2 s wall = 200 modeled s > 65.5.
		slow := e.srv
		e.srv = fast
		r1 := e.submitSized(t, round, 1, rxN, rxDim)
		e.srv = slow
		t.Logf("hand, scale 1000: w1's remaining = max(0, %.3f - 200) = 0 -> gpu_queue_wait_ms 0", GBig)
		if estWaitMs(t, r1) != 0 {
			t.Errorf("scale 1000: wait %d ms, want 0", estWaitMs(t, r1))
		}
		r2 := e.submitSized(t, round, 2, rxN, rxDim)
		el := time.Since(claimed).Seconds()
		lo, hi := (GBig-el+G)*1000-5, (GBig-0.2+G)*1000+5
		t.Logf("hand, scale 1: w1's remaining %.3f - %.3f..0.2, then shard 1 (queued ahead, %.3f) -> wait in [%.0f, %.0f] ms", GBig, el, G, lo, hi)
		if w := float64(estWaitMs(t, r2)); w < lo || w > hi {
			t.Errorf("scale 1: wait %v ms, want in [%.0f, %.0f]", w, lo, hi)
		}
	})

	// ------------------------------------------------------------ speculation

	// specWorld: shards 0 and 1 build 100k locally at 75%, backlogs of
	// needs-index queries (2 and 40 x 0.5 s): neither preempt pays, shard 1's
	// projected recovery is the worst.
	specWorld := func(t *testing.T, e *env) (int64, string, string) {
		round := e.createRound(t, 3)
		a := e.rxSubmit(t, round, 0, rxN, "local", 0)
		b := e.rxSubmit(t, round, 1, rxN, "local", 0)
		e.rxReport(t, round, 0, 2, 2, 0.75, 500, false)
		e.rxReport(t, round, 1, 40, 40, 0.75, 500, false)
		e.rxAgeRound(t, round)
		rem := L * 0.25
		t.Logf("hand: both at 75%%: remaining %.3f; all-needs-index backlog cancels out: gain from preempting = %.3f - %.3f = %.3f < margin -> no preempt. Projected stay-local recovery: shard 0 %.3f + 1 = %.3f, shard 1 %.3f + 20 = %.3f -> shard 1 is the worst", rem, rem, G, rem-G, rem, rem+1, rem, rem+20)
		return round, a, b
	}

	t.Run("SpeculationLifecycleGPUWins", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: roadmap policy 4; ReactiveSpec doc 'when a worker is idle and the GPU queue is empty, start a GPU copy of the local build with the worst projected recovery... One speculation per tick, and never while one is already in flight'; Speculation sequence diagram; decisions 15, 16")
		round, a, b := specWorld(t, e)
		e.rxHeartbeat(t, "w1", big)
		e.rxTick(t, "reactive", 1)
		if _, ok, _ := e.st.GetBuild(e.ctx, b+":spec"); ok {
			t.Errorf("plain reactive created a copy")
		}
		e.rxTick(t, "reactive_spec", 1)
		cp := b + ":spec"
		c, ok, _ := e.st.GetBuild(e.ctx, cp)
		if !ok {
			t.Fatalf("reactive_spec with an idle worker and an empty queue created no copy of %s", b)
		}
		rxLogRow(t, "copy", c)
		if !c.Speculative || c.Placement != "gpu" || c.State != "queued" {
			t.Errorf("copy shape wrong")
		}
		if _, ok, _ := e.st.GetBuild(e.ctx, a+":spec"); ok {
			t.Errorf("shard 0 also got a copy in the same tick (one per tick)")
		}
		for _, id := range []string{a, b} {
			if r := e.rxRow(t, "primary", id); r.Placement != "local" || r.State != "running" || r.Moved {
				t.Errorf("primary %s changed by speculation", id)
			}
		}
		e.rxTick(t, "reactive_spec", 1)
		if _, ok, _ := e.st.GetBuild(e.ctx, a+":spec"); ok {
			t.Errorf("second copy while the first is queued")
		}
		cl := e.rxClaim(t, "w1", big, time.Minute)
		if cl.BuildID != cp {
			t.Fatalf("w1 claimed %s, want the copy", cl.BuildID)
		}
		e.rxHeartbeat(t, "w2", big)
		t.Logf("copy leased on w1; w2 idle; queue empty -> still no copy for shard 0 (one in flight)")
		e.rxTick(t, "reactive_spec", 1)
		if _, ok, _ := e.st.GetBuild(e.ctx, a+":spec"); ok {
			t.Errorf("second copy while the first is leased")
		}
		e.rxReport(t, round, 1, 100, 0, 0, 500, false)
		t.Logf("shard 1 now reports progress 0 and 100 non-index queries: stay %.1f vs migrate max(50, %.3f) -> a preempt would pay, but its copy is in flight (decision 15)", L+50, G)
		e.rxTick(t, "reactive_spec", 1)
		if r := e.rxRow(t, "shard 1 primary", b); r.Placement != "local" || r.State != "running" || r.Moved {
			t.Errorf("primary preempted while its copy races")
		}
		if ok, _ := e.st.Complete(e.ctx, cp, cl.Attempt); !ok {
			t.Fatal("complete copy")
		}
		cdone := e.rxRow(t, "copy completed by w1", cp)
		e.rxTick(t, "reactive_spec", 1)
		p := e.rxRow(t, "shard 1 primary after tick", b)
		if p.Placement != "gpu" || p.State != "done" || p.FinishedAt == nil || !p.FinishedAt.Equal(*cdone.FinishedAt) {
			t.Errorf("decision 16: tick should settle the GPU win: primary gpu/done at the copy's finish; got %s/%s %s", p.Placement, p.State, tptr(p.FinishedAt))
		}
		var rep api.ReportResponse
		code, body := e.do(t, "POST", fmt.Sprintf("/shards/%d/1/report", round), api.ReportRequest{BuildProgress: 0.9}, &rep)
		t.Logf("shard 1's next report -> %d %s (the shard aborts its local build on 'done')", code, body)
		if rep.Build.State != "done" || rep.Build.BuildID != b {
			t.Errorf("reply %+v, want %s done", rep.Build, b)
		}
		e.rxTick(t, "reactive_spec", 1)
		ca, ok, _ := e.st.GetBuild(e.ctx, a+":spec")
		t.Logf("after the race settled: shard 0 is the only local build, w1 and w2 idle, queue empty, nothing in flight -> copy of shard 0 expected; present=%v", ok)
		if !ok {
			t.Errorf("no copy for shard 0 once the first speculation was settled")
		} else {
			rxLogRow(t, "shard 0 copy", ca)
		}
	})

	t.Run("SpeculationLocalWinsCancelsCopy", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: Speculation diagram 'local build completes first: mark GPU copy cancelled, worker stops at its next failed renew'; decision 16")
		_, _, b := specWorld(t, e)
		e.rxHeartbeat(t, "w1", big)
		e.rxTick(t, "reactive_spec", 1)
		cp := b + ":spec"
		cl := e.rxClaim(t, "w1", big, time.Minute)
		if cl.BuildID != cp {
			t.Fatalf("claimed %s", cl.BuildID)
		}
		code, body := e.do(t, "POST", "/builds/"+b+"/done", nil, nil)
		t.Logf("shard 1 POST /builds/%s/done -> %d %s", b, code, body)
		if code < 200 || code >= 300 {
			t.Fatalf("local done refused while the copy races: %d", code)
		}
		e.rxTick(t, "reactive_spec", 1)
		c := e.rxRow(t, "copy after tick", cp)
		if c.State != "failed" || c.FailReason == nil {
			t.Errorf("copy not cancelled by the tick")
		}
		if ok, _ := e.st.Renew(e.ctx, cp, cl.Attempt, time.Minute); ok {
			t.Errorf("worker renewed a cancelled copy")
		}
		if ok, _ := e.st.Complete(e.ctx, cp, cl.Attempt); ok {
			t.Errorf("worker completed a cancelled copy")
		}
		p := e.rxRow(t, "primary", b)
		if p.Placement != "local" || p.State != "done" {
			t.Errorf("primary %s/%s, want local/done", p.Placement, p.State)
		}
	})

	t.Run("SpeculationOnePerTickWithTwoIdleWorkers", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 8 'Speculation is additionally one per reconsider tick'")
		_, a, b := specWorld(t, e)
		e.rxHeartbeat(t, "w1", big)
		e.rxHeartbeat(t, "w2", big)
		e.rxTick(t, "reactive_spec", 1)
		var n int
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM builds WHERE speculative`).Scan(&n))
		t.Logf("[db] speculative rows after one tick with two idle workers: %d (want 1)", n)
		if n != 1 {
			t.Errorf("copies %d, want 1", n)
		}
		if _, ok, _ := e.st.GetBuild(e.ctx, b+":spec"); !ok {
			t.Errorf("the copy is not of the worst shard %s", b)
		}
		_ = a
	})

	t.Run("SpeculationRespectsMemoryAndNonEmptyQueue", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: CLAUDE.md memory filter (speculation is one of its sites, bug 11); ReactiveSpec 'when ... the GPU queue is empty'")
		round, _, _ := specWorld(t, e)
		e.rxHeartbeat(t, "w-small", mem-1)
		e.rxTick(t, "reactive_spec", 1)
		var n int
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM builds WHERE speculative`).Scan(&n))
		t.Logf("idle worker of %d bytes, builds need %d -> copies %d (want 0)", mem-1, mem, n)
		if n != 0 {
			t.Errorf("speculated onto a worker the copy cannot fit")
		}
		e.rxHeartbeat(t, "w1", big)
		z := e.rxSubmit(t, round, 2, 10_000, "gpu", 0)
		t.Logf("a big idle worker joins, but %s is queued (and fits) -> queue not empty -> no copy", z)
		e.rxTick(t, "reactive_spec", 1)
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM builds WHERE speculative`).Scan(&n))
		if n != 0 {
			t.Errorf("speculated with a non-empty queue")
		}
		if zb := e.rxRow(t, "queued z", z); zb.State != "queued" {
			t.Errorf("z moved")
		}
	})

	// ------------------------------------------------------------ concurrency

	t.Run("TwoSchedulersTickingConcurrently", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 14 'Two schedulers ticking concurrently are therefore safe, like two reapers'")
		_, id := hurting(t, e, 4, 0, 50)
		e.rxHeartbeat(t, "w1", big)
		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() { defer wg.Done(); e.rxTick(t, "reactive", 1) }()
		}
		wg.Wait()
		b := e.rxRow(t, "after two concurrent ticks", id)
		if b.Placement != "gpu" || b.State != "queued" || !b.Moved || math.Abs(b.Priority-50) > 0.01 {
			t.Errorf("want one preempt at priority 50, got %+v", b)
		}
		// Speculation from two schedulers: still one copy.
		e2 := setup(t)
		_, _, sb := specWorld(t, e2)
		e2.rxHeartbeat(t, "w1", big)
		e2.rxHeartbeat(t, "w2", big)
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() { defer wg.Done(); e2.rxTick(t, "reactive_spec", 1) }()
		}
		wg.Wait()
		var n int
		must(t, e2.pool.QueryRow(e2.ctx, `SELECT count(*) FROM builds WHERE speculative`).Scan(&n))
		t.Logf("[db] copies after two concurrent reactive_spec ticks: %d (want 1, of %s)", n, sb)
		if n != 1 {
			t.Errorf("copies %d, want 1", n)
		}
	})

	t.Run("TickRenegeRacingAClaim", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 4 (renege guarded against a concurrent claim), decision 14 (a stale action updates zero rows and is skipped)")
		e.rxHeartbeat(t, "w-small", 1000)
		claimWins, renegeWins := 0, 0
		for i := 0; i < 20; i++ {
			round := e.createRound(t, 1)
			id := e.rxSubmit(t, round, 0, rxN, "gpu", 0)
			var wg sync.WaitGroup
			var cl store.Claimed
			var cOK bool
			start := make(chan struct{})
			wg.Add(2)
			go func() { defer wg.Done(); <-start; e.rxTick(t, "reactive", 1) }()
			go func() {
				defer wg.Done()
				<-start
				time.Sleep(time.Duration(i%5) * time.Millisecond)
				cl, cOK, _ = e.st.Claim(e.ctx, "w-ghost", big, time.Minute)
			}()
			close(start)
			wg.Wait()
			b := e.mustBuild(t, id)
			switch {
			case cOK && cl.BuildID == id && b.State == "leased" && b.Placement == "gpu" && !b.Moved && b.Attempt == 1:
				claimWins++
			case !cOK && b.State == "running" && b.Placement == "local" && b.Moved && b.Attempt == 0:
				renegeWins++
			default:
				t.Errorf("iteration %d incoherent: claim ok=%v %s; row %s/%s moved=%v attempt=%d", i, cOK, cl.BuildID, b.Placement, b.State, b.Moved, b.Attempt)
			}
			e.rxCloseRound(t, round)
		}
		t.Logf("20 races of a renege-worthy (fits-nowhere for the live pool) build vs an unregistered big worker's claim: claim won %d, renege won %d; every outcome coherent", claimWins, renegeWins)
	})
}
