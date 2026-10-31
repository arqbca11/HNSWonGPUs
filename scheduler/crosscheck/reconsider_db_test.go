package crosscheck

// Phase 3 step 3 cross-check, store and schema side: the guarded operations
// the reconsider timer executes (PreemptToGPU, RenegeToLocal, Reprioritize,
// SpeculateCopy, ResolveSpeculations; decisions 8, 14, 15, 16), migration
// 0006 at the SQL level, and rounds ignoring the copies (decision 17).
// Written from the spec and `go doc` alone, never reading store.go.

import (
	"errors"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/api"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

func TestReconsiderDB(t *testing.T) {
	big := int64(1) << 40

	// ---------------------------------------------------------------- moves

	t.Run("PreemptSetsMovedAndOnlyOnce", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 8 (moved-at-most-once), decision 15 ('moved ... guarded with NOT moved'), go doc PreemptToGPU")
		round := e.createRound(t, 2)
		id := e.rxSubmit(t, round, 0, rxN, "local", 0)
		before := e.rxRow(t, "before", id)
		if before.Moved || before.Speculative {
			t.Errorf("fresh build: moved=%v speculative=%v, want false/false (migration 0006 defaults)", before.Moved, before.Speculative)
		}
		ok, err := e.st.PreemptToGPU(e.ctx, id, 12.5)
		t.Logf("PreemptToGPU(%s, 12.5) -> %v %v", id, ok, err)
		if err != nil || !ok {
			t.Fatalf("first preempt of a running local build must succeed")
		}
		a := e.rxRow(t, "after preempt", id)
		if a.Placement != "gpu" || a.State != "queued" || !a.Moved || a.Priority != 12.5 || a.Attempt != before.Attempt || !a.EnqueuedAt.Equal(before.EnqueuedAt) {
			t.Errorf("want gpu/queued moved=true priority=12.5, attempt and enqueued_at untouched; got %+v", a)
		}
		// Claim it and release it back to queued: moved survives the cycle.
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		ok, _ = e.st.Release(e.ctx, c.BuildID, c.Attempt)
		t.Logf("Release(%s, %d) -> %v; the build is queued again with moved still true", c.BuildID, c.Attempt, ok)
		if b := e.rxRow(t, "after release", id); !b.Moved {
			t.Errorf("moved did not survive a claim/release cycle")
		}
		// Decision 20 (updated in the step 3 triage, as decision 8's F3 fix):
		// the store leaves the renege door OPEN for a moved build — it is the
		// impossibility escape's path — and the policy is what refuses a
		// judgment renege of a moved build. The loop is closed on the other
		// side: a moved build can never be preempted again.
		ok, err = e.st.RenegeToLocal(e.ctx, id)
		t.Logf("RenegeToLocal on the preempted (moved) build -> %v %v; decision 20: the escape door is open at the store", ok, err)
		if !ok {
			t.Errorf("the store refused the escape renege of a moved build (decision 20)")
		}
		if b := e.rxRow(t, "after escape renege", id); b.Placement != "local" || b.State != "running" || !b.Moved {
			t.Errorf("escape renege: want local/running with moved still true, got %s/%s moved=%v", b.Placement, b.State, b.Moved)
		}
		ok, err = e.st.PreemptToGPU(e.ctx, id, 1)
		t.Logf("PreemptToGPU after the escape -> %v %v (want false: no flapping loop can form)", ok, err)
		if ok {
			t.Errorf("a moved build was preempted again: decision 8's loop guard is gone")
		}
	})

	t.Run("RenegeSetsMovedAndBlocksPreempt", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 8 'a reneged build cannot be preempted'; go doc RenegeToLocal 'started_at becomes now ... moved becomes true'")
		round := e.createRound(t, 2)
		id := e.rxSubmit(t, round, 0, rxN, "gpu", 3)
		time.Sleep(50 * time.Millisecond)
		t0 := time.Now()
		ok, err := e.st.RenegeToLocal(e.ctx, id)
		t.Logf("RenegeToLocal(%s) -> %v %v", id, ok, err)
		if err != nil || !ok {
			t.Fatalf("renege of a queued GPU build must succeed")
		}
		a := e.rxRow(t, "after renege", id)
		// Phase 4 decision 38 (bug 19): a renege takes the next attempt (0 -> 1 here)
		if a.Placement != "local" || a.State != "running" || !a.Moved || a.Attempt != 1 {
			t.Errorf("want local/running moved=true attempt 1 (Phase 4 decision 38); got %+v", a)
		}
		if a.StartedAt == nil || a.StartedAt.Before(t0.Add(-time.Second)) || a.StartedAt.Sub(a.EnqueuedAt) < 40*time.Millisecond {
			t.Errorf("started_at should be the renege time (now), after enqueued_at by >= 50 ms; started=%s enqueued=%s", tptr(a.StartedAt), a.EnqueuedAt.Format(time.RFC3339Nano))
		}
		ok, err = e.st.PreemptToGPU(e.ctx, id, 99)
		t.Logf("PreemptToGPU on the reneged (moved) build -> %v %v (want false)", ok, err)
		if ok {
			t.Errorf("preempt of a reneged build succeeded")
		}
		ok, _ = e.st.RenegeToLocal(e.ctx, id)
		if ok {
			t.Errorf("second renege succeeded")
		}
		b := e.rxRow(t, "after refused preempt", id)
		if b.Placement != "local" || b.State != "running" || b.Priority != 3 {
			t.Errorf("refused moves changed the row")
		}
	})

	t.Run("ReprioritizeOnlyWhileQueued", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc Reprioritize 'false if the build is no longer queued: a claimed or moved build keeps whatever priority it was claimed at'")
		round := e.createRound(t, 5)
		queued := e.rxSubmit(t, round, 0, rxN, "gpu", 1)
		leased := e.rxSubmit(t, round, 1, rxN, "gpu", 50)
		local := e.rxSubmit(t, round, 2, rxN, "local", 0)
		preempted := e.rxSubmit(t, round, 3, rxN, "local", 0)
		reneged := e.rxSubmit(t, round, 4, rxN, "gpu", 0)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if c.BuildID != leased {
			t.Fatalf("claim took %s, want %s (priority 50)", c.BuildID, leased)
		}
		if ok, _ := e.st.PreemptToGPU(e.ctx, preempted, 5); !ok {
			t.Fatalf("preempt fixture failed")
		}
		if ok, _ := e.st.RenegeToLocal(e.ctx, reneged); !ok {
			t.Fatalf("renege fixture failed")
		}
		cases := []struct {
			id   string
			want bool
			why  string
		}{
			{queued, true, "queued GPU build"},
			{preempted, true, "preempted build, now queued (decision 10: this is exactly who gets reprioritized)"},
			{leased, false, "leased (claimed)"},
			{local, false, "running local"},
			{reneged, false, "reneged to local"},
		}
		for _, cs := range cases {
			before := e.mustBuild(t, cs.id)
			ok, err := e.st.Reprioritize(e.ctx, cs.id, 77)
			after := e.mustBuild(t, cs.id)
			t.Logf("Reprioritize(%s [%s], 77) -> %v %v; priority %g -> %g", cs.id, cs.why, ok, err, before.Priority, after.Priority)
			if ok != cs.want {
				t.Errorf("%s: Reprioritize returned %v, want %v", cs.why, ok, cs.want)
			}
			if cs.want && after.Priority != 77 {
				t.Errorf("%s: priority %g, want 77", cs.why, after.Priority)
			}
			if !cs.want && after.Priority != before.Priority {
				t.Errorf("%s: refused reprioritize still changed priority", cs.why)
			}
			if after.State != before.State || after.Placement != before.Placement || after.Attempt != before.Attempt {
				t.Errorf("%s: reprioritize touched more than priority", cs.why)
			}
		}
		if ok, _ := e.st.Complete(e.ctx, leased, c.Attempt); !ok {
			t.Fatalf("complete")
		}
		if ok, _ := e.st.Reprioritize(e.ctx, leased, 1); ok {
			t.Errorf("Reprioritize on a done build returned true")
		}
		if ok, _ := e.st.Reprioritize(e.ctx, "no:such", 1); ok {
			t.Errorf("Reprioritize on a missing build returned true")
		}
	})

	// ---------------------------------------------------------------- copies

	t.Run("SpeculateCopyRowShapeAndOneEver", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 15 (build_id + ':spec', speculative, GPU-side, one copy ever by primary key); go doc SpeculateCopy")
		round := e.createRound(t, 3)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		queued := e.rxSubmit(t, round, 1, rxN, "gpu", 0)
		done := e.rxSubmit(t, round, 2, rxN, "local", 0)
		if ok, _ := e.st.CompleteLocal(e.ctx, done); !ok {
			t.Fatal("CompleteLocal fixture")
		}
		p := e.mustBuild(t, primary)
		cp := e.rxCopy(t, primary, 6.25)
		c := e.rxRow(t, "copy", cp)
		if c.BuildID != primary+":spec" || !c.Speculative || c.Placement != "gpu" || c.State != "queued" || c.Attempt != 0 || c.Moved || c.Priority != 6.25 {
			t.Errorf("copy row: want %s:spec speculative gpu/queued attempt 0 moved=false priority 6.25; got %+v", primary, c)
		}
		if c.RoundID != p.RoundID || c.ShardID != p.ShardID || c.NVectors != p.NVectors || c.Dim != p.Dim || c.MemBytes != p.MemBytes {
			t.Errorf("copy should carry the primary's round/shard/size/mem: primary %+v copy %+v", p, c)
		}
		pa := e.rxRow(t, "primary after copy", primary)
		if pa.Placement != "local" || pa.State != "running" || pa.Moved {
			t.Errorf("decision 5: the primary keeps running locally; got %+v", pa)
		}
		for _, cs := range []struct {
			id, why string
		}{
			{primary, "second copy of the same primary"},
			{queued, "a queued GPU build (not running local)"},
			{done, "a done local build"},
			{cp, "the copy itself"},
			{"9:9", "a missing build"},
		} {
			ok, err := e.st.SpeculateCopy(e.ctx, cs.id, 1)
			t.Logf("SpeculateCopy(%s) [%s] -> %v %v (want false)", cs.id, cs.why, ok, err)
			if ok {
				t.Errorf("%s: SpeculateCopy returned true", cs.why)
			}
		}
		var n int
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM builds WHERE round_id = $1`, round).Scan(&n))
		t.Logf("[db] builds rows in round: %d (want 4: three primaries and one copy)", n)
		if n != 4 {
			t.Errorf("rows = %d, want 4", n)
		}
		// One copy ever: fail the copy, then try again.
		e.rxHeartbeat(t, "w1", big)
		cl, _, _ := e.st.Claim(e.ctx, "w1", big, time.Minute)
		t.Logf("w1 claimed %s (copy at priority 6.25 should beat the priority-0 queued build)", cl.BuildID)
		if cl.BuildID != cp {
			t.Fatalf("claim took %s, want the copy %s", cl.BuildID, cp)
		}
		if ok, _ := e.st.Fail(e.ctx, cp, cl.Attempt, "crosscheck: GPU gave up"); !ok {
			t.Fatal("Fail copy")
		}
		e.rxRow(t, "copy after Fail", cp)
		ok, err := e.st.SpeculateCopy(e.ctx, primary, 1)
		t.Logf("SpeculateCopy(%s) after its copy failed -> %v %v (decision 15: 'one copy ever per build ... a second buys nothing')", primary, ok, err)
		if ok {
			t.Errorf("a second copy was created after the first failed")
		}
	})

	t.Run("CopyClaimableByOrdinaryClaimSQL", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 15 'the claim statement, lease fencing, reaper and the Python worker handle it with zero changes'; the claim SQL from roadmap 'The four operations'")
		round := e.createRound(t, 3)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		other := e.rxSubmit(t, round, 1, rxN, "gpu", 1)
		cp := e.rxCopy(t, primary, 9)
		var id string
		var att int32
		err := e.pool.QueryRow(e.ctx, roadmapClaimSQL, "py-worker", big).Scan(&id, &att)
		t.Logf("roadmap claim SQL as py-worker -> %s attempt %d err %v (want the copy: priority 9 > 1)", id, att, err)
		if err != nil || id != cp || att != 1 {
			t.Errorf("claim SQL took %s attempt %d, want %s attempt 1", id, att, cp)
		}
		c := e.rxRow(t, "copy after claim", cp)
		if c.State != "leased" || deref(c.LeaseOwner) != "py-worker" {
			t.Errorf("copy not leased to py-worker")
		}
		// The worker's renew and complete (store's Go reference) work on the copy as on any build.
		if ok, _ := e.st.Renew(e.ctx, cp, att, time.Minute); !ok {
			t.Errorf("Renew on the copy failed")
		}
		if ok, _ := e.st.Complete(e.ctx, cp, att); !ok {
			t.Errorf("Complete on the copy failed")
		}
		e.rxRow(t, "copy after complete", cp)
		err = e.pool.QueryRow(e.ctx, roadmapClaimSQL, "py-worker", big).Scan(&id, &att)
		t.Logf("second claim -> %s attempt %d err %v (want the ordinary build %s)", id, att, err, other)
		if id != other {
			t.Errorf("second claim took %s, want %s", id, other)
		}
		// The memory filter applies to copies like to any build.
		p2 := e.rxSubmit(t, round, 2, rxN, "local", 0)
		cp2 := e.rxCopy(t, p2, 100)
		_, ok, _ := e.st.Claim(e.ctx, "small", rxCM.GPUMemBytes(rxN, rxDim)-1, time.Minute)
		t.Logf("a worker one byte too small claims -> ok=%v (want false: copy %s needs %d)", ok, cp2, rxCM.GPUMemBytes(rxN, rxDim))
		if ok {
			t.Errorf("a too-small worker claimed the copy")
		}
	})

	t.Run("PreemptRefusedWhileCopyInFlight", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 15 'the store also refuses to preempt a primary whose copy is in flight'; go doc PreemptToGPU 'false if ... has a live copy'")
		round := e.createRound(t, 2)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		ok, err := e.st.PreemptToGPU(e.ctx, primary, 5)
		t.Logf("PreemptToGPU(primary) with the copy queued -> %v %v (want false)", ok, err)
		if ok {
			t.Errorf("preempted while copy queued")
		}
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		ok, err = e.st.PreemptToGPU(e.ctx, primary, 5)
		t.Logf("PreemptToGPU(primary) with the copy leased -> %v %v (want false)", ok, err)
		if ok {
			t.Errorf("preempted while copy leased")
		}
		p := e.rxRow(t, "primary", primary)
		if p.Placement != "local" || p.State != "running" || p.Moved {
			t.Errorf("refused preempt changed the primary")
		}
		// After the copy fails (its retry budget, say), it is no longer racing.
		if ok, _ := e.st.Fail(e.ctx, cp, c.Attempt, "crosscheck: budget"); !ok {
			t.Fatal("Fail copy")
		}
		ok, err = e.st.PreemptToGPU(e.ctx, primary, 5)
		t.Logf("PreemptToGPU(primary) after the copy failed -> %v %v (doc: refused only while a copy is 'still racing'/'live'; a failed copy is neither)", ok, err)
		if !ok {
			t.Errorf("preempt refused although the copy is failed (not live)")
		}
		e.rxRow(t, "primary after", primary)
	})

	t.Run("PreemptWhileCopyDoneButUnresolvedKeepsTheWin", func(t *testing.T) {
		e := setup(t)
		t.Logf("probe: a copy that has completed but not yet been settled by ResolveSpeculations. Decision 16: the index exists from the copy's finished_at. Is the race 'live'? If a preempt slips into this window (a tick's resolve runs first, but a worker can complete between resolve and the tick's actions), the won index must not be thrown away.")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if ok, _ := e.st.Complete(e.ctx, cp, c.Attempt); !ok {
			t.Fatal("complete copy")
		}
		copyRow := e.rxRow(t, "copy done", cp)
		ok, err := e.st.PreemptToGPU(e.ctx, primary, 5)
		t.Logf("PreemptToGPU(primary) with the copy done-but-unresolved -> %v %v", ok, err)
		res, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("ResolveSpeculations -> %+v", res)
		p := e.rxRow(t, "primary after resolve", primary)
		if ok {
			t.Errorf("PreemptToGPU succeeded on a primary whose copy had already WON: the primary is now %s/%s and will be built a third time on a GPU; the copy's index (finished_at %s) is thrown away and the shard waits for a whole new GPU build", p.Placement, p.State, tptr(copyRow.FinishedAt))
		}
		if p.State != "done" {
			t.Errorf("primary not done after resolve: %s/%s", p.Placement, p.State)
		}
	})

	t.Run("RenegeRefusedOnCopy", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc RenegeToLocal 'false if ... a speculative copy (a copy is cancelled, never turned into a second local build)'")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		ok, err := e.st.RenegeToLocal(e.ctx, cp)
		t.Logf("RenegeToLocal(copy) -> %v %v", ok, err)
		if ok {
			t.Errorf("a copy was reneged to local")
		}
		c := e.rxRow(t, "copy", cp)
		if c.Placement != "gpu" || c.State != "queued" || c.Moved {
			t.Errorf("refused renege changed the copy")
		}
	})

	// ------------------------------------------------------------- resolution

	t.Run("ResolveGPUWinPrimaryTakesCopyFinish", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16 and go doc ResolveSpeculations: copy done while primary runs -> primary gpu/done with the copy's finished_at; the shard's next report reply says done; idempotent")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if ok, _ := e.st.Complete(e.ctx, cp, c.Attempt); !ok {
			t.Fatal("complete copy")
		}
		copyRow := e.rxRow(t, "copy done", cp)
		time.Sleep(30 * time.Millisecond) // so now() at resolve differs from the copy's finish
		res, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("ResolveSpeculations -> %+v", res)
		if len(res) != 1 || res[0].BuildID != primary || res[0].Winner != "gpu" || res[0].RoundID != round {
			t.Errorf("want one {BuildID:%s RoundID:%d Winner:gpu}, got %+v", primary, round, res)
		}
		p := e.rxRow(t, "primary after resolve", primary)
		if p.Placement != "gpu" || p.State != "done" || p.FinishedAt == nil || !p.FinishedAt.Equal(*copyRow.FinishedAt) {
			t.Errorf("primary want gpu/done finished_at=%s (the copy's); got %s/%s %s", tptr(copyRow.FinishedAt), p.Placement, p.State, tptr(p.FinishedAt))
		}
		t.Logf("unspecified, observed: primary attempt=%d moved=%v after a GPU win", p.Attempt, p.Moved)
		res2, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("second ResolveSpeculations -> %+v (want none)", res2)
		if len(res2) != 0 {
			t.Errorf("not idempotent: %+v", res2)
		}
		p2 := e.mustBuild(t, primary)
		if !p2.FinishedAt.Equal(*p.FinishedAt) || p2.State != "done" {
			t.Errorf("second resolve changed the primary")
		}
		ok, err := e.st.CompleteLocal(e.ctx, primary)
		t.Logf("the shard's late CompleteLocal(primary) -> %v %v (want false: the build is done on the GPU's account)", ok, err)
		if ok {
			t.Errorf("late CompleteLocal succeeded after the GPU win")
		}
		var rep api.ReportResponse
		code, body := e.do(t, "POST", fmt.Sprintf("/shards/%d/0/report", round), api.ReportRequest{BuildProgress: 0.4}, &rep)
		t.Logf("shard 0 report -> %d %s", code, body)
		if rep.Build.BuildID != primary || rep.Build.State != "done" {
			t.Errorf("decision 16: the shard's next report reply says done for its real build %s; got %+v", primary, rep.Build)
		}
		code, body = e.do(t, "POST", "/builds/"+primary+"/done", nil, nil)
		t.Logf("shard's late POST /builds/%s/done -> %d %s", primary, code, body)
		if code >= 200 && code < 300 {
			t.Errorf("late /done accepted after the GPU won")
		}
		if p3 := e.mustBuild(t, primary); p3.Placement != "gpu" || !p3.FinishedAt.Equal(*copyRow.FinishedAt) {
			t.Errorf("late /done changed the primary: %+v", p3)
		}
	})

	t.Run("ResolveLocalWinCancelsLeasedCopyAndFencesWorker", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16: primary terminal, copy leased -> copy failed with a cancel reason; the worker's renew/complete then update zero rows; CopyState 'leased' is the wasted GPU work")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if ok, _ := e.st.CompleteLocal(e.ctx, primary); !ok {
			t.Fatal("CompleteLocal")
		}
		pBefore := e.rxRow(t, "primary done locally", primary)
		res, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("ResolveSpeculations -> %+v", res)
		if len(res) != 1 || res[0].BuildID != primary || res[0].Winner != "local" || res[0].CopyState != "leased" {
			t.Errorf("want one {BuildID:%s Winner:local CopyState:leased}, got %+v", primary, res)
		}
		cr := e.rxRow(t, "copy after cancel", cp)
		if cr.State != "failed" || cr.FailReason == nil || *cr.FailReason == "" {
			t.Errorf("copy want failed with a reason, got %s reason=%s", cr.State, deref(cr.FailReason))
		}
		for _, op := range []struct {
			name string
			f    func() (bool, error)
		}{
			{"Renew", func() (bool, error) { return e.st.Renew(e.ctx, cp, c.Attempt, time.Minute) }},
			{"Complete", func() (bool, error) { return e.st.Complete(e.ctx, cp, c.Attempt) }},
			{"Fail", func() (bool, error) { return e.st.Fail(e.ctx, cp, c.Attempt, "worker") }},
			{"Release", func() (bool, error) { return e.st.Release(e.ctx, cp, c.Attempt) }},
		} {
			ok, err := op.f()
			t.Logf("worker w1's %s(copy, attempt %d) -> %v %v (want false: lease ended from outside)", op.name, c.Attempt, ok, err)
			if ok {
				t.Errorf("%s succeeded on a cancelled copy", op.name)
			}
		}
		cr2 := e.rxRow(t, "copy after the worker's attempts", cp)
		if cr2.State != "failed" {
			t.Errorf("worker revived the cancelled copy")
		}
		reaped, err := e.st.Reap(e.ctx, 0)
		must(t, err)
		t.Logf("Reap -> %+v (the cancelled copy holds no lease to reap)", reaped)
		for _, r := range reaped {
			if r.BuildID == cp {
				t.Errorf("reaper acted on a cancelled copy")
			}
		}
		res2, _ := e.st.ResolveSpeculations(e.ctx)
		t.Logf("second ResolveSpeculations -> %+v (want none)", res2)
		if len(res2) != 0 {
			t.Errorf("not idempotent: %+v", res2)
		}
		pAfter := e.rxRow(t, "primary", primary)
		if pAfter.Placement != "local" || pAfter.State != "done" || !pAfter.FinishedAt.Equal(*pBefore.FinishedAt) {
			t.Errorf("resolve touched the winning primary")
		}
	})

	t.Run("ResolveLocalWinCancelsQueuedCopy", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16: primary terminal, copy queued -> copy failed; CopyState 'queued'; nothing left to claim")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		if ok, _ := e.st.CompleteLocal(e.ctx, primary); !ok {
			t.Fatal("CompleteLocal")
		}
		res, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("ResolveSpeculations -> %+v", res)
		if len(res) != 1 || res[0].Winner != "local" || res[0].CopyState != "queued" {
			t.Errorf("want one local win with CopyState queued, got %+v", res)
		}
		cr := e.rxRow(t, "copy", cp)
		if cr.State != "failed" {
			t.Errorf("copy state %s, want failed", cr.State)
		}
		_, ok, _ := e.st.Claim(e.ctx, "w1", big, time.Minute)
		t.Logf("a worker claims -> ok=%v (want false)", ok)
		if ok {
			t.Errorf("cancelled copy was claimable")
		}
	})

	t.Run("ResolveBothDoneNeedsNothing", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16 'both sides finishing before a tick needs nothing: done is done, built twice'")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		if ok, _ := e.st.Complete(e.ctx, cp, c.Attempt); !ok {
			t.Fatal("complete copy")
		}
		if ok, _ := e.st.CompleteLocal(e.ctx, primary); !ok {
			t.Fatal("CompleteLocal")
		}
		p := e.rxRow(t, "primary", primary)
		cr := e.rxRow(t, "copy", cp)
		res, err := e.st.ResolveSpeculations(e.ctx)
		must(t, err)
		t.Logf("ResolveSpeculations -> %+v (want none)", res)
		if len(res) != 0 {
			t.Errorf("resolve acted on a both-done pair: %+v", res)
		}
		p2, c2 := e.rxRow(t, "primary after", primary), e.rxRow(t, "copy after", cp)
		if p2.Placement != p.Placement || p2.State != "done" || !p2.FinishedAt.Equal(*p.FinishedAt) || c2.State != "done" || !c2.FinishedAt.Equal(*cr.FinishedAt) {
			t.Errorf("both-done rows changed")
		}
	})

	t.Run("ReaperTreatsCopyAsOrdinaryBuild", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 15 'reaper ... handle it with zero changes'; decision 42 retry budget; decision 15 one copy ever")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, 10*time.Millisecond)
		time.Sleep(50 * time.Millisecond)
		reaped, err := e.st.Reap(e.ctx, 1)
		must(t, err)
		t.Logf("lease of %d ms expired; Reap(maxAttempts=1) -> %+v (want the copy failed: attempt %d reached the budget)", 10, reaped, c.Attempt)
		if len(reaped) != 1 || reaped[0].BuildID != cp || reaped[0].Outcome != "failed" {
			t.Errorf("want the copy reaped to failed, got %+v", reaped)
		}
		e.rxRow(t, "copy", cp)
		res, _ := e.st.ResolveSpeculations(e.ctx)
		t.Logf("ResolveSpeculations with primary running and copy failed -> %+v (want none: no winner yet)", res)
		if len(res) != 0 {
			t.Errorf("resolve acted with no winner: %+v", res)
		}
		p := e.rxRow(t, "primary", primary)
		if p.Placement != "local" || p.State != "running" {
			t.Errorf("primary changed")
		}
		if ok, _ := e.st.SpeculateCopy(e.ctx, primary, 1); ok {
			t.Errorf("a second copy was created after the first was reaped to failed")
		}
	})

	// ----------------------------------------------------------------- races

	t.Run("RaceClaimVersusRenegeExactlyOneWins", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 4 'the guard makes it race-free against a concurrent claim'; decision 14 zero rows = skipped")
		e.rxHeartbeat(t, "w1", big)
		claimWins, renegeWins := 0, 0
		for i := 0; i < 30; i++ {
			round := e.createRound(t, 1)
			id := store.BuildID(0, round)
			if ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: id, RoundID: round, ShardID: 0, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "gpu"}); !ok || err != nil {
				t.Fatalf("submit: %v %v", ok, err)
			}
			var wg sync.WaitGroup
			var cOK, rOK bool
			var cl store.Claimed
			start := make(chan struct{})
			wg.Add(2)
			go func() { defer wg.Done(); <-start; cl, cOK, _ = e.st.Claim(e.ctx, "w1", big, time.Minute) }()
			go func() { defer wg.Done(); <-start; rOK, _ = e.st.RenegeToLocal(e.ctx, id) }()
			close(start)
			wg.Wait()
			b := e.mustBuild(t, id)
			switch {
			case cOK && !rOK && cl.BuildID == id && b.State == "leased" && b.Placement == "gpu" && b.Attempt == 1 && !b.Moved:
				claimWins++
			case rOK && !cOK && b.State == "running" && b.Placement == "local" && b.Attempt == 1 && b.Moved: // the renege took attempt 1 (Phase 4 decision 38)
				renegeWins++
			default:
				t.Errorf("iteration %d incoherent: claim ok=%v (%s) renege ok=%v row=%s/%s attempt=%d moved=%v", i, cOK, cl.BuildID, rOK, b.Placement, b.State, b.Attempt, b.Moved)
			}
			e.rxCloseRound(t, round)
		}
		t.Logf("30 races: claim won %d, renege won %d, incoherent 0 expected", claimWins, renegeWins)
	})

	t.Run("RaceResolveVersusCopyComplete", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc ResolveSpeculations 'If the copy finishes in the same instant, exactly one side's guard holds'")
		e.rxHeartbeat(t, "w1", big)
		cancelled, completed := 0, 0
		for i := 0; i < 30; i++ {
			round := e.createRound(t, 1)
			primary := store.BuildID(0, round)
			if ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: primary, RoundID: round, ShardID: 0, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "local"}); !ok || err != nil {
				t.Fatalf("submit: %v %v", ok, err)
			}
			if ok, _ := e.st.SpeculateCopy(e.ctx, primary, 1); !ok {
				t.Fatal("copy")
			}
			cp := primary + ":spec"
			c, ok, _ := e.st.Claim(e.ctx, "w1", big, time.Minute)
			if !ok || c.BuildID != cp {
				t.Fatalf("claim %+v %v", c, ok)
			}
			if ok, _ := e.st.CompleteLocal(e.ctx, primary); !ok {
				t.Fatal("CompleteLocal")
			}
			var wg sync.WaitGroup
			var compOK bool
			var res []store.SpeculationResolved
			start := make(chan struct{})
			wg.Add(2)
			go func() { defer wg.Done(); <-start; compOK, _ = e.st.Complete(e.ctx, cp, c.Attempt) }()
			go func() { defer wg.Done(); <-start; res, _ = e.st.ResolveSpeculations(e.ctx) }()
			close(start)
			wg.Wait()
			cr := e.mustBuild(t, cp)
			p := e.mustBuild(t, primary)
			resolvedThis := false
			for _, r := range res {
				if r.BuildID == primary {
					resolvedThis = true
				}
			}
			switch {
			case compOK && cr.State == "done" && !resolvedThis:
				completed++
			case !compOK && cr.State == "failed" && resolvedThis:
				cancelled++
			default:
				t.Errorf("iteration %d incoherent: Complete=%v copy=%s resolved=%v res=%+v", i, compOK, cr.State, resolvedThis, res)
			}
			if p.Placement != "local" || p.State != "done" {
				t.Errorf("iteration %d: primary %s/%s, want local/done", i, p.Placement, p.State)
			}
			e.rxCloseRound(t, round)
		}
		t.Logf("30 races (primary already done locally): copy completed first %d, copy cancelled first %d", completed, cancelled)
	})

	t.Run("RaceResolveVersusLocalDone", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16 'one guarded statement per direction keeps every interleaving coherent': copy done, then resolve (GPU win) races the shard's CompleteLocal")
		e.rxHeartbeat(t, "w1", big)
		gpuWins, localWins := 0, 0
		for i := 0; i < 30; i++ {
			round := e.createRound(t, 1)
			primary := store.BuildID(0, round)
			if ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: primary, RoundID: round, ShardID: 0, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "local"}); !ok || err != nil {
				t.Fatalf("submit: %v %v", ok, err)
			}
			if ok, _ := e.st.SpeculateCopy(e.ctx, primary, 1); !ok {
				t.Fatal("copy")
			}
			cp := primary + ":spec"
			c, ok, _ := e.st.Claim(e.ctx, "w1", big, time.Minute)
			if !ok || c.BuildID != cp {
				t.Fatalf("claim %+v %v", c, ok)
			}
			if ok, _ := e.st.Complete(e.ctx, cp, c.Attempt); !ok {
				t.Fatal("complete copy")
			}
			var wg sync.WaitGroup
			var locOK bool
			var res []store.SpeculationResolved
			start := make(chan struct{})
			wg.Add(2)
			go func() { defer wg.Done(); <-start; locOK, _ = e.st.CompleteLocal(e.ctx, primary) }()
			go func() { defer wg.Done(); <-start; res, _ = e.st.ResolveSpeculations(e.ctx) }()
			close(start)
			wg.Wait()
			p := e.mustBuild(t, primary)
			cr := e.mustBuild(t, cp)
			resolved := len(res) == 1 && res[0].BuildID == primary && res[0].Winner == "gpu"
			switch {
			case resolved && !locOK && p.Placement == "gpu" && p.State == "done" && p.FinishedAt.Equal(*cr.FinishedAt):
				gpuWins++
			case !resolved && locOK && p.Placement == "local" && p.State == "done":
				localWins++
			default:
				t.Errorf("iteration %d incoherent: CompleteLocal=%v resolve=%+v primary=%s/%s", i, locOK, res, p.Placement, p.State)
			}
			if cr.State != "done" {
				t.Errorf("copy %s, want done", cr.State)
			}
			e.rxCloseRound(t, round)
		}
		t.Logf("30 races (copy already done): resolve marked the GPU win %d, shard's CompleteLocal first %d", gpuWins, localWins)
	})

	// ------------------------------------------------------------- migration

	t.Run("Migration0006GeneratedFKColumns", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: migration 0006: shard_status.build_id and shard_jobs.build_id GENERATED ALWAYS AS (shard_id || ':' || round_id) STORED REFERENCES builds; bug 3's rule survives; never the copy")
		round := e.createRound(t, 2)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		e.rxCopy(t, primary, 1)
		e.rxReport(t, round, 0, 3, 1, 0.5, 250, false)
		var sb, jb string
		must(t, e.pool.QueryRow(e.ctx, `SELECT build_id FROM shard_status WHERE round_id=$1 AND shard_id=0`, round).Scan(&sb))
		must(t, e.pool.QueryRow(e.ctx, `SELECT build_id FROM shard_jobs WHERE round_id=$1 AND shard_id=0 AND seq=0`, round).Scan(&jb))
		t.Logf("[db] shard_status.build_id=%q shard_jobs.build_id=%q (want %q, the real build, never the copy)", sb, jb, primary)
		if sb != primary || jb != primary {
			t.Errorf("generated build_id wrong")
		}
		_, err := e.pool.Exec(e.ctx, `INSERT INTO shard_jobs (round_id, shard_id, seq, duration_ms, needs_index, build_id) VALUES ($1, 0, 5, 1, false, 'x')`, round)
		t.Logf("explicit build_id insert into shard_jobs -> %v (code %s; want 428C9, generated always)", err, pgCode(err))
		if err == nil {
			t.Errorf("explicit value accepted into a generated column")
		}
		// A shard whose only row is a copy (inserted by SQL; the store never
		// makes one without a primary) has no build in the FK's eyes.
		if err := e.rxRawBuild(store.BuildID(1, round)+":spec", round, 1, "gpu", "queued", true); err != nil {
			t.Fatalf("raw copy insert: %v", err)
		}
		t.Logf("raw-inserted an orphan copy %s:spec (no primary)", store.BuildID(1, round))
		err = e.st.ReportStatus(e.ctx, store.ShardStatus{RoundID: round, ShardID: 1})
		t.Logf("ReportStatus for shard 1 (only a copy) -> %v (want ErrNoBuild: decision 33 / bug 3)", err)
		if !errors.Is(err, store.ErrNoBuild) {
			t.Errorf("report from a copy-only shard: %v, want ErrNoBuild", err)
		}
		_, err = e.st.RecordArrival(e.ctx, round, 1, 0, 10, false)
		t.Logf("RecordArrival for shard 1 -> %v (want ErrNoBuild)", err)
		if !errors.Is(err, store.ErrNoBuild) {
			t.Errorf("arrival at a copy-only shard: %v, want ErrNoBuild", err)
		}
		code, body := e.do(t, "POST", fmt.Sprintf("/shards/%d/1/report", round), report(false), nil)
		t.Logf("POST /shards/%d/1/report -> %d %s (want 404, as for any shard without a build)", round, code, body)
		if code != http.StatusNotFound {
			t.Errorf("API report from a copy-only shard: %d", code)
		}
	})

	t.Run("Migration0006PartialUniqueAndCheck", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: migration 0006: builds_one_real_per_shard, builds_one_copy_per_shard (partial unique), builds_speculative_gpu CHECK; builds_round_shard_unique dropped")
		round := e.createRound(t, 3)
		e.rxSubmit(t, round, 0, rxN, "local", 0)
		err := e.rxRawBuild("other-real", round, 0, "gpu", "queued", false)
		t.Logf("second real build for (round, shard 0) under another id -> %v (code %s, want 23505)", err, pgCode(err))
		if pgCode(err) != "23505" {
			t.Errorf("second real build: want unique violation")
		}
		err = e.rxRawBuild("copy-a", round, 0, "gpu", "queued", true)
		t.Logf("one copy beside the real build -> %v (want ok)", err)
		if err != nil {
			t.Errorf("a copy beside the real build was refused")
		}
		err = e.rxRawBuild("copy-b", round, 0, "gpu", "queued", true)
		t.Logf("a second copy for (round, shard 0) -> %v (code %s, want 23505)", err, pgCode(err))
		if pgCode(err) != "23505" {
			t.Errorf("second copy: want unique violation")
		}
		err = e.rxRawBuild("copy-local", round, 1, "local", "running", true)
		t.Logf("a local speculative row -> %v (code %s, want 23514 check violation)", err, pgCode(err))
		if pgCode(err) != "23514" {
			t.Errorf("local copy: want check violation")
		}
		_, err = e.pool.Exec(e.ctx, `UPDATE builds SET placement='local', state='running' WHERE build_id='copy-a'`)
		t.Logf("UPDATE the copy to local/running -> %v (code %s, want 23514)", err, pgCode(err))
		if pgCode(err) != "23514" {
			t.Errorf("copy turned local: want check violation")
		}
		var nOld, nIdx int
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM pg_constraint WHERE conname IN ('builds_round_shard_unique','shard_status_build_fk','shard_jobs_build_fk')`).Scan(&nOld))
		must(t, e.pool.QueryRow(e.ctx, `SELECT count(*) FROM pg_indexes WHERE indexname IN ('builds_one_real_per_shard','builds_one_copy_per_shard')`).Scan(&nIdx))
		t.Logf("[db] old 0004 constraints remaining: %d (want 0); new partial unique indexes: %d (want 2)", nOld, nIdx)
		if nOld != 0 || nIdx != 2 {
			t.Errorf("constraint/index set wrong")
		}
	})

	t.Run("SubmitBuildWithNonConventionalIDProbe", func(t *testing.T) {
		e := setup(t)
		t.Logf("probe: migration 0006 re-anchors the report FK on build_id = shard_id:round_id. Store.SubmitBuild takes the build_id from the caller; does it insist on the convention, or can a build exist that its own shard can never report against?")
		round := e.createRound(t, 1)
		ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: "custom-id", RoundID: round, ShardID: 0, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "local"})
		t.Logf("SubmitBuild(build_id=custom-id, shard 0) -> %v %v", ok, err)
		// F4's fix (step 3 triage): the store now insists on the convention,
		// because a build under any other id could never be reported against.
		if ok || err == nil {
			t.Errorf("a non-conventional build_id was accepted; its shard's reports would forever fail the generated FK")
		}
		conventional := store.BuildID(0, round)
		ok, err = e.st.SubmitBuild(e.ctx, store.Build{BuildID: conventional, RoundID: round, ShardID: 0, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "local"})
		t.Logf("SubmitBuild(build_id=%s, shard 0) -> %v %v; ReportStatus must now work", conventional, ok, err)
		if err != nil || !ok {
			t.Fatalf("the conventional id must be accepted")
		}
		if err := e.st.ReportStatus(e.ctx, store.ShardStatus{RoundID: round, ShardID: 0}); err != nil {
			t.Errorf("ReportStatus for the conventional build -> %v, want ok", err)
		}
	})

	// ---------------------------------------------------------------- rounds

	t.Run("RoundClosesWithUnresolvedCopy", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 17 'a round may close while a losing copy is still being cancelled'; go doc FinishCompleteRounds")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		e.rxClaim(t, "w1", big, time.Minute)
		e.rxReport(t, round, 0, 0, 0, 0, 0, true)
		if ok, _ := e.st.CompleteLocal(e.ctx, primary); !ok {
			t.Fatal("CompleteLocal")
		}
		p := e.rxRow(t, "primary", primary)
		e.rxRow(t, "copy (still leased)", cp)
		fin, err := e.st.FinishCompleteRounds(e.ctx)
		must(t, err)
		t.Logf("FinishCompleteRounds -> %+v", fin)
		r := e.mustRound(t, round)
		t.Logf("[db] round finished_at=%s (want the primary's finish %s)", tptr(r.FinishedAt), tptr(p.FinishedAt))
		if r.FinishedAt == nil || !r.FinishedAt.Equal(*p.FinishedAt) {
			t.Errorf("round did not close at the primary's finish while the copy was leased")
		}
		if len(fin) == 1 && len(fin[0].QueueWaits) != 0 {
			t.Errorf("QueueWaits %v include the copy's claim; decision 17 excludes copies", fin[0].QueueWaits)
		}
		res, _ := e.st.ResolveSpeculations(e.ctx)
		t.Logf("ResolveSpeculations after the round closed -> %+v (decision 17: 'the sweep cancels it afterwards regardless of round state')", res)
		if len(res) != 1 || res[0].Winner != "local" || res[0].CopyState != "leased" {
			t.Errorf("copy not cancelled after the round closed: %+v", res)
		}
		e.rxRow(t, "copy", cp)
	})

	t.Run("RoundWaitsForGPUWinResolution", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 17 (completion ignores copies), decision 16 (primary takes the copy's finished_at): a done copy does not close the round; the resolved primary does, at the copy's finish")
		round := e.createRound(t, 1)
		r0 := e.mustRound(t, round)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 1)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute)
		e.rxReport(t, round, 0, 0, 0, 0.3, 0, true)
		if ok, _ := e.st.Complete(e.ctx, cp, c.Attempt); !ok {
			t.Fatal("complete copy")
		}
		cr := e.rxRow(t, "copy done", cp)
		fin, err := e.st.FinishCompleteRounds(e.ctx)
		must(t, err)
		t.Logf("FinishCompleteRounds before resolve -> %+v (want none: the real build still runs locally)", fin)
		if len(fin) != 0 || e.mustRound(t, round).FinishedAt != nil {
			t.Errorf("round closed on the copy's completion alone")
		}
		time.Sleep(30 * time.Millisecond)
		if res, _ := e.st.ResolveSpeculations(e.ctx); len(res) != 1 {
			t.Fatalf("resolve: %+v", res)
		}
		fin, err = e.st.FinishCompleteRounds(e.ctx)
		must(t, err)
		t.Logf("FinishCompleteRounds after resolve -> %+v", fin)
		r := e.mustRound(t, round)
		t.Logf("[db] round finished_at=%s, copy finished_at=%s", tptr(r.FinishedAt), tptr(cr.FinishedAt))
		if r.FinishedAt == nil || !r.FinishedAt.Equal(*cr.FinishedAt) {
			t.Errorf("round should finish at the copy's finished_at")
		}
		if len(fin) == 1 {
			if want := cr.FinishedAt.Sub(r0.StartedAt); fin[0].Duration != want {
				t.Errorf("Duration %v, want %v", fin[0].Duration, want)
			}
			pr := e.rxRow(t, "primary after resolve", primary)
			cr2 := e.rxRow(t, "copy", cp)
			t.Logf("evidence: primary started_at - enqueued_at = %v (attempt %d, never claimed: its started_at is the LOCAL start); copy started_at - enqueued_at = %v (attempt %d)",
				pr.StartedAt.Sub(pr.EnqueuedAt), pr.Attempt, cr2.StartedAt.Sub(cr2.EnqueuedAt), cr2.Attempt)
			t.Logf("QueueWaits=%v: the primary was never claimed (attempt 0) and the copy is excluded, so want none (go doc: 'builds never claimed are skipped')", fin[0].QueueWaits)
			if len(fin[0].QueueWaits) != 0 {
				t.Errorf("QueueWaits %v, want empty", fin[0].QueueWaits)
			}
		}
	})

	t.Run("RoundStatsExcludeCopies", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 17: straggler lag, queue-wait stats and finished_at exclude speculative rows, even a copy cancelled later")
		round := e.createRound(t, 3)
		g := e.rxSubmit(t, round, 0, rxN, "gpu", 0)
		l1 := e.rxSubmit(t, round, 1, rxN, "local", 0)
		l2 := e.rxSubmit(t, round, 2, rxN, "local", 0)
		cp := e.rxCopy(t, l2, 1)
		e.rxHeartbeat(t, "w1", big)
		time.Sleep(40 * time.Millisecond)
		c := e.rxClaim(t, "w1", big, time.Minute) // the copy (priority 1) beats g (priority 0)
		if c.BuildID != cp {
			t.Fatalf("first claim %s, want copy", c.BuildID)
		}
		time.Sleep(40 * time.Millisecond)
		e.rxHeartbeat(t, "w2", big)
		c2 := e.rxClaim(t, "w2", big, time.Minute)
		if c2.BuildID != g {
			t.Fatalf("second claim %s, want %s", c2.BuildID, g)
		}
		for s := int32(0); s < 3; s++ {
			e.rxReport(t, round, s, 0, 0, 0, 0, true)
		}
		if ok, _ := e.st.Complete(e.ctx, g, c2.Attempt); !ok {
			t.Fatal("complete g")
		}
		time.Sleep(60 * time.Millisecond)
		if ok, _ := e.st.CompleteLocal(e.ctx, l1); !ok {
			t.Fatal("l1")
		}
		time.Sleep(120 * time.Millisecond)
		if ok, _ := e.st.CompleteLocal(e.ctx, l2); !ok {
			t.Fatal("l2")
		}
		time.Sleep(300 * time.Millisecond)
		res, _ := e.st.ResolveSpeculations(e.ctx)
		t.Logf("copy cancelled 300 ms after the last real finish: %+v", res)
		cr := e.rxRow(t, "copy", cp)
		var fins []time.Time
		var gRow store.BuildRow
		for _, id := range []string{g, l1, l2} {
			b := e.rxRow(t, "real", id)
			fins = append(fins, *b.FinishedAt)
			if id == g {
				gRow = b
			}
		}
		sort.Slice(fins, func(i, j int) bool { return fins[i].Before(fins[j]) })
		fin, err := e.st.FinishCompleteRounds(e.ctx)
		must(t, err)
		t.Logf("FinishCompleteRounds -> %+v", fin)
		if len(fin) != 1 {
			t.Fatalf("round did not close")
		}
		r := e.mustRound(t, round)
		wantLag := fins[2].Sub(fins[1])
		// g's first claim minus enqueue: started_at was the claim (no release).
		wantWait := gRow.StartedAt.Sub(gRow.EnqueuedAt)
		t.Logf("hand: finished_at = last real finish %s; lag = last - median = %v; queue waits = [%v] (g only; the copy %s waited %v and must not appear); copy finished_at %s",
			fins[2].Format(time.RFC3339Nano), wantLag, wantWait, c.BuildID, cr.StartedAt.Sub(cr.EnqueuedAt), tptr(cr.FinishedAt))
		if r.FinishedAt == nil || !r.FinishedAt.Equal(fins[2]) {
			t.Errorf("finished_at %s, want %s", tptr(r.FinishedAt), fins[2].Format(time.RFC3339Nano))
		}
		if d := fin[0].StragglerLag - wantLag; d > time.Microsecond || d < -time.Microsecond { // the lag comes back through float seconds; ns rounding is not a finding
			t.Errorf("StragglerLag %v, want %v", fin[0].StragglerLag, wantLag)
		}
		if len(fin[0].QueueWaits) != 1 || fin[0].QueueWaits[0] != wantWait {
			t.Errorf("QueueWaits %v, want [%v]", fin[0].QueueWaits, wantWait)
		}
	})

	t.Run("NShardsCapCountsRealBuildsOnly", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 17 'SubmitBuild's n_shards cap counts real builds only'")
		round := e.createRound(t, 2)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		e.rxCopy(t, primary, 1)
		ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: store.BuildID(1, round), RoundID: round, ShardID: 1, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "gpu"})
		t.Logf("SubmitBuild shard 1 into n_shards=2 with one real build + one copy -> %v %v (want inserted)", ok, err)
		if err != nil || !ok {
			t.Errorf("the copy counted against the cap")
		}
		ok, err = e.st.SubmitBuild(e.ctx, store.Build{BuildID: store.BuildID(2, round), RoundID: round, ShardID: 2, NVectors: rxN, Dim: rxDim, MemBytes: 1, Placement: "gpu"})
		t.Logf("SubmitBuild shard 2 (third real) -> %v %v (want ErrRoundFull)", ok, err)
		if !errors.Is(err, store.ErrRoundFull) {
			t.Errorf("third real build: %v, want ErrRoundFull", err)
		}
	})

	t.Run("StatsCountCopies", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 17 'gpu_queue_depth and builds_leased DO count copies'; go doc Stats")
		round := e.createRound(t, 2)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		e.rxSubmit(t, round, 1, rxN, "gpu", 0)
		e.rxCopy(t, primary, 5)
		s, err := e.st.Stats(e.ctx)
		must(t, err)
		t.Logf("Stats with one real queued + one queued copy: %+v (want GPUQueued 2)", s)
		if s.GPUQueued != 2 {
			t.Errorf("GPUQueued %d, want 2", s.GPUQueued)
		}
		e.rxHeartbeat(t, "w1", big)
		e.rxClaim(t, "w1", big, time.Minute)
		s, _ = e.st.Stats(e.ctx)
		t.Logf("Stats after w1 claims the copy: %+v (want GPUQueued 1, Leased 1)", s)
		if s.GPUQueued != 1 || s.Leased != 1 {
			t.Errorf("Stats %+v, want queued 1 leased 1", s)
		}
	})

	// --------------------------------------------------------------- snapshot

	t.Run("ReconsiderSnapshotContents", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: go doc GetReconsiderSnapshot / ReconsiderBuild / ReconsiderStatus; decision 18 (aggregates derived in SQL)")
		round := e.createRound(t, 4)
		done := e.createRound(t, 1)
		e.rxSubmit(t, done, 0, rxN, "local", 0) // running, but in a finished round: excluded (a queued one would still be claimable: the claim ignores rounds)
		e.rxCloseRound(t, done)
		local := e.rxSubmit(t, round, 0, rxN, "local", 0)
		q1 := e.rxSubmit(t, round, 1, rxN, "gpu", 1)
		q2 := e.rxSubmit(t, round, 2, rxN, "gpu", 3)
		fin := e.rxSubmit(t, round, 3, rxN, "local", 0)
		if ok, _ := e.st.CompleteLocal(e.ctx, fin); !ok {
			t.Fatal("CompleteLocal")
		}
		cp := e.rxCopy(t, local, 2)
		e.rxHeartbeat(t, "w1", big)
		c := e.rxClaim(t, "w1", big, time.Minute) // q2 (priority 3)
		if c.BuildID != q2 {
			t.Fatalf("claim %s, want %s", c.BuildID, q2)
		}
		now := time.Now()
		jobs := []store.JobRecord{{Seq: 0, DurationMs: 100, ArrivedAt: now}, {Seq: 1, DurationMs: 200, ArrivedAt: now}, {Seq: 2, DurationMs: 600, NeedsIndex: true, ArrivedAt: now}}
		must(t, e.st.ReportBatch(e.ctx, store.ShardStatus{RoundID: round, ShardID: 0, QueueDepth: 2, WaitingNeedsIndex: 1, OldestWaitMs: 1500, BuildProgress: 0.25}, jobs))
		t.Logf("shard 0 reported depth 2, needs 1, oldest 1500 ms, progress 0.25, with 3 records of 100/200/600 ms (mean 300)")
		if _, err := e.pool.Exec(e.ctx, `UPDATE rounds SET started_at = now() - interval '10 seconds' WHERE round_id=$1`, round); err != nil {
			t.Fatal(err)
		}
		time.Sleep(200 * time.Millisecond)
		snap, err := e.st.GetReconsiderSnapshot(e.ctx, time.Minute)
		must(t, err)
		var got []string
		byID := map[string]store.ReconsiderBuild{}
		for _, b := range snap.Builds {
			got = append(got, b.BuildID)
			byID[b.BuildID] = b
			t.Logf("[snapshot] build %s placement=%s state=%s prio=%g attempt=%d moved=%v spec=%v owner=%q elapsed=%.3f", b.BuildID, b.Placement, b.State, b.Priority, b.Attempt, b.Moved, b.Speculative, b.LeaseOwner, b.ElapsedS)
		}
		for _, s := range snap.Status {
			t.Logf("[snapshot] status %+v", s)
		}
		t.Logf("[snapshot] workers %+v", snap.Workers)
		want := map[string]bool{local: true, q1: true, q2: true, cp: true}
		if len(got) != 4 {
			t.Errorf("builds %v, want exactly %v (non-terminal, open rounds, copies included)", got, want)
		}
		for _, id := range got {
			if !want[id] {
				t.Errorf("unexpected build %s in snapshot", id)
			}
		}
		if b, ok := byID[cp]; !ok || !b.Speculative {
			t.Errorf("copy missing or not marked speculative")
		}
		if b := byID[q2]; b.LeaseOwner != "w1" || b.ElapsedS < 0.15 || b.ElapsedS > 5 {
			t.Errorf("leased %s: owner %q elapsed %g, want w1 and ~0.2 s", q2, b.LeaseOwner, b.ElapsedS)
		}
		if b := byID[q1]; b.ElapsedS != 0 || b.LeaseOwner != "" {
			t.Errorf("queued %s: elapsed %g owner %q, want 0 and ''", q1, b.ElapsedS, b.LeaseOwner)
		}
		if b := byID[local]; b.ElapsedS < 0.15 {
			t.Errorf("local %s elapsed %g, want >= ~0.2 s (since started_at)", local, b.ElapsedS)
		}
		// claim order among the queued ones: copy (2) before q1 (1)
		ic, i1 := -1, -1
		for i, id := range got {
			if id == cp {
				ic = i
			}
			if id == q1 {
				i1 = i
			}
		}
		if ic > i1 {
			t.Errorf("claim order: copy (priority 2) should precede %s (priority 1): %v", q1, got)
		}
		if len(snap.Status) != 1 {
			t.Fatalf("status rows %d, want 1", len(snap.Status))
		}
		s := snap.Status[0]
		if s.ShardID != 0 || s.QueueDepth != 2 || s.WaitingNeedsIndex != 1 || s.OldestWaitMs != 1500 || s.BuildProgress != 0.25 || s.Arrivals != 3 || s.MeanQueryMs != 300 {
			t.Errorf("status %+v, want depth 2 needs 1 oldest 1500 progress 0.25 arrivals 3 mean 300", s)
		}
		if s.RoundElapsedS < 10 || s.RoundElapsedS > 12 {
			t.Errorf("RoundElapsedS %g, want ~10.2 (wall seconds since the backdated start)", s.RoundElapsedS)
		}
	})
}
