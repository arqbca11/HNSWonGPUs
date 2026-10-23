package crosscheck

// Phase 3 step 3 cross-check with the real Python worker as a black-box
// subprocess (worker_helpers_test.go): decision 15 says the worker handles a
// speculative copy "with zero changes", and decision 16 says a cancelled
// copy looks to its worker exactly like a lost lease. The tick is the real
// reconsider.Tick with reactive_spec.

import (
	"strings"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
)

func TestReconsiderWorker(t *testing.T) {
	t.Run("PythonWorkerBuildsCopyAndTickSettlesGPUWin", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 15 (copy claimable with zero worker changes), decision 16 (GPU win: primary gpu/done at the copy's finished_at), Speculation diagram")
		round := e.createRound(t, 1)
		primary := e.rxSubmit(t, round, 0, rxN, "local", 0)
		cp := e.rxCopy(t, primary, 3)
		w := startWorker(t, "py-gpu", map[string]string{"FAKE_TIME_SCALE": "20"})
		t.Logf("copy modeled GPU total %v -> ~%v wall at scale 20", rxCM.GPUTotal(rxN, rxDim), rxCM.GPUTotal(rxN, rxDim)/20)
		done := e.waitBuild(t, cp, 15*time.Second, "copy done by the Python worker", func(b store.BuildRow) bool { return b.State == "done" })
		rxLogRow(t, "copy", done)
		// Complete clears lease_owner, so the worker's own log is the evidence of who built it.
		for _, l := range w.find(func(l logLine) bool { return l.str("build_id") == cp }) {
			t.Logf("[worker] %s", l.str("raw"))
		}
		if len(w.find(forBuild(cp, 1, msgHas("claimed")))) == 0 || len(w.find(forBuild(cp, 1, msgHas("completed")))) == 0 {
			t.Errorf("py-gpu did not log claimed and completed for %s attempt 1", cp)
		}
		p := e.rxRow(t, "primary before tick (still running locally)", primary)
		if p.State != "running" {
			t.Fatalf("primary changed before any tick")
		}
		e.rxTick(t, "reactive_spec", 20)
		p = e.rxRow(t, "primary after tick", primary)
		if p.Placement != "gpu" || p.State != "done" || p.FinishedAt == nil || !p.FinishedAt.Equal(*done.FinishedAt) {
			t.Errorf("want primary gpu/done at the copy's finished_at %s; got %s/%s %s", tptr(done.FinishedAt), p.Placement, p.State, tptr(p.FinishedAt))
		}
	})

	t.Run("PythonWorkerFencedOutByCancelledCopy", func(t *testing.T) {
		e := setup(t)
		t.Logf("spec: decision 16 'the worker's every write stays guarded by attempt and state=leased, so its next renew or complete updates zero rows and it stops — exactly a lost lease'; Speculation diagram 'worker stops at its next failed renew'")
		round := e.createRound(t, 2)
		primary := e.rxSubmit(t, round, 0, 1_000_000, "local", 0)
		cp := e.rxCopy(t, primary, 3)
		w := startWorker(t, "py-gpu", map[string]string{"FAKE_TIME_SCALE": "10", "LEASE_SECONDS": "30", "RENEW_INTERVAL_SECONDS": "0.3"})
		t.Logf("copy modeled GPU total %v -> ~%v wall at scale 10; renew every 0.3 s", rxCM.GPUTotal(1_000_000, rxDim), rxCM.GPUTotal(1_000_000, rxDim)/10)
		e.waitBuild(t, cp, 10*time.Second, "copy leased to py-gpu", leasedTo("py-gpu"))
		time.Sleep(500 * time.Millisecond)
		code, body := e.do(t, "POST", "/builds/"+primary+"/done", nil, nil)
		t.Logf("shard 0's local build finishes first: POST /builds/%s/done -> %d %s", primary, code, body)
		tCancel := time.Now()
		e.rxTick(t, "reactive_spec", 10)
		c := e.rxRow(t, "copy after tick", cp)
		if c.State != "failed" {
			t.Fatalf("tick did not cancel the leased copy")
		}
		lost := w.waitLog(t, 3*time.Second, "lost ownership of the copy", forBuild(cp, 1, msgHas("lost")))
		t.Logf("worker noticed after %v: %s", time.Since(tCancel), lost.str("raw"))
		time.Sleep(3500 * time.Millisecond) // past the copy's whole modeled build time
		after := e.rxRow(t, "copy one full build time later", cp)
		if after.State != "failed" || !strings.Contains(deref(after.FailReason), deref(c.FailReason)) {
			t.Errorf("the fenced worker changed the cancelled copy: %s %s", after.State, deref(after.FailReason))
		}
		for _, l := range w.find(forBuild(cp, 1, func(l logLine) bool {
			m := strings.ToLower(l.str("msg"))
			return strings.Contains(m, "complet") || strings.Contains(m, "releas")
		})) {
			t.Errorf("worker logged a write after the cancel: %s", l.str("raw"))
		}
		// And the worker goes back to work.
		next := queueGPU(t, e, round, 1, 10_000, rxDim, -1, 0)
		e.waitBuild(t, next, 10*time.Second, "next build done by the same worker", func(b store.BuildRow) bool { return b.State == "done" })
		t.Logf("the worker built %s after the cancelled copy: a cancel is just a lost lease", next)
	})
}
