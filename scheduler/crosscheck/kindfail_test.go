package crosscheck

// Cross-check of Phase 2 step 6: the failure injections, watched on the
// dashboard. Written from the spec only: CLAUDE.md (invariants; the Commands
// rows for scripts/k8s-round.sh: KILL_FORCE=1 "makes it grace 0 so the lease
// must expire"), docs/roadmap.md Phase 2 "Failure injection" and the phase's
// done-when ("the dashboard shows a worker pod deleted mid-round and the round
// still completing"), and docs/design/phase2-kubernetes.md decisions 19, 24,
// 29, 31, 32, 35, 36.
//
// The system under test is the implementer's live stack in namespace kgpu on
// the kind-kgpu cluster. Every dashboard number is read through Grafana's
// datasource proxy (decision 36). The tests scale the worker Deployment,
// delete pods and drain GPU nodes; Cleanup uncordons both GPU nodes and puts
// the worker Deployment back at 2 replicas, whatever happened.
//
// Run: KGPU_KIND=1 go test -count=1 -v -timeout 60m -run TestKindFail ./scheduler/crosscheck/

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

const (
	kfScale     = 10.0       // FAKE_TIME_SCALE in the worker ConfigMap; the simulator must match
	kfLongN     = 8_000_000  // ~276 s of modeled GPU time, ~27.6 s real at scale 10, 7.6 GiB < 8 GiB worker
	kfMidN      = 2_000_000  // ~65 s modeled, ~6.5 s real
	kfShortN    = 100_000    // ~4.6 s modeled, ~0.5 s real
	kfPanelExpr = `sum(lease_expirations_total) by (outcome)` // current-series total, for before/after deltas; the panel itself reads windowed increase (decision 37)
)

func TestKindFail(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set")
	}
	if _, _, err := kcRun("", "get", "deployment", "scheduler", "-n", koNS); err != nil {
		t.Fatalf("no scheduler Deployment in %s: %v", koNS, err)
	}
	if err := koDial("127.0.0.1:9091", 300*time.Millisecond); err == nil {
		t.Fatalf("port 9091 is already taken on the laptop; another simulator is running")
	}
	t.Cleanup(func() {
		kfUncordonGPU(t)
		koRestoreWorkers(t)
		if err := koDial("127.0.0.1:9091", 300*time.Millisecond); err == nil {
			t.Errorf("[cleanup] port 9091 still taken after the tests")
		} else {
			t.Logf("[cleanup] port 9091 free")
		}
	})
	scale := koTimeScale(t)
	t.Logf("[setup] worker FAKE_TIME_SCALE=%s; the simulator runs with TIME_SCALE=%s (shardsim header)", scale, scale)
	if scale != "10" {
		t.Fatalf("FAKE_TIME_SCALE=%s, these tests assume 10", scale)
	}
	koWaitPool(t, koReplicas, 3*time.Minute)
	g := kfGrafana(t)
	base, _ := g.scalar(t, "max(rounds_open)")
	t.Logf("[setup] dashboard before any injection: %s", g.rest(t))
	c := startKoCollector(t)

	// 1. The done-when, from a different angle than "the panel went up":
	// the round's largest GPU build's leaseholder is force-deleted with grace
	// 0. CLAUDE.md says KILL_FORCE=1 "makes it grace 0 so the lease must
	// expire" and the roadmap says the panel shows lease_expirations{queued}
	// +1; decision 19 says a forced deletion still delivers SIGTERM and the
	// worker releases, so NO lease expires. The two cannot both hold. The
	// test infers which path ran from the build row alone (a release puts the
	// build back before lease_until; the reaper only moves EXPIRED leases,
	// invariant 5) and requires the dashboard's counter to agree exactly.
	t.Run("ForceDeletedLeaseholderCountedOnce", func(t *testing.T) {
		sched := kfSchedulerPod(t)
		exp0 := g.byLabel(t, kfPanelExpr, "outcome")
		rd0, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		t.Logf("[dashboard before] scheduler pod %s; panel %q = %s; round_duration_seconds_count = %g", sched, kfPanelExpr, fmtMap(exp0), rd0)

		p := startShardsim(t, koSimEnv("skewed", 6, 61, 1, scale, koReplicas))
		id := koWaitStarted(t, p, 1, 2*time.Minute)
		t.Logf("[shards] round %d started (skewed, 6 shards, seed 61)", id)

		var target xBuild
		poll(t, 90*time.Second, 100*time.Millisecond, "the round's largest build leased", func() bool {
			r, code := koRound(t, id)
			if code != http.StatusOK || len(r.Builds) < 6 {
				return false
			}
			b, _ := r.largest()
			if b.Placement != "gpu" {
				t.Fatalf("largest build placed %s under always_gpu: %s", b.Placement, b)
			}
			if b.State == "done" {
				t.Fatalf("the largest build finished before it was seen leased: %s", b)
			}
			if b.State == "leased" && b.LeaseOwner != nil {
				target = b
				return true
			}
			return false
		})
		owner := target.owner()
		t.Logf("[api] largest build %s n=%d leased: attempt=%d owner=%s lease_until=%s", target.BuildID, target.NVectors, target.Attempt, owner, target.LeaseUntil.Format("15:04:05.000"))
		killAt := kfForceDelete(t, owner)

		// Follow the row at 100 ms until attempt 1 leaves "leased".
		var left xBuild
		poll(t, 60*time.Second, 100*time.Millisecond, "attempt 1 leaves leased", func() bool {
			left = kmBuild(t, target.BuildID)
			return left.Attempt > target.Attempt || left.State != "leased"
		})
		leftAt := time.Now()
		completedInGrace := left.State == "done" && left.Attempt == target.Attempt
		released := !completedInGrace && leftAt.Before(*target.LeaseUntil)
		path := "lease expired and was reaped"
		switch {
		case completedInGrace:
			path = "completed by the dying worker at attempt 1 (SIGTERM path: finish within the budget)"
		case released:
			path = "released by the worker before its lease_until (no expiry possible: invariant 5)"
		}
		t.Logf("[api] %.1fs after the kill: %s; lease_until was kill%+.1fs -> %s", leftAt.Sub(killAt).Seconds(), left, target.LeaseUntil.Sub(killAt).Seconds(), path)

		r := koVerifyRound(t, c, p, id, kfScale, 6, 5*time.Minute)
		if code := p.wait(t, 2*time.Minute); code != 0 {
			t.Errorf("simulator exited %d", code)
		}
		fin, _ := r.build(target.BuildID)
		t.Logf("[api] build %s finally: %s", target.BuildID, fin)
		wantAtt := target.Attempt + 1
		if completedInGrace {
			wantAtt = target.Attempt
		}
		if fin.Attempt != wantAtt {
			t.Errorf("build %s ended at attempt %d, want %d after %q", target.BuildID, fin.Attempt, wantAtt, path)
		}
		t.Logf("[logs] %s's lines about %s (the stream is cut by the kill): %v", owner, target.BuildID, kfMsgs(kfWorkerLinesFor(c, owner, target.BuildID)))

		scrapeWait(t, "the round's last numbers reach Prometheus")
		exp1 := g.byLabel(t, kfPanelExpr, "outcome")
		rd1, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		dq, df := exp1["queued"]-exp0["queued"], exp1["failed"]-exp0["failed"]
		t.Logf("[dashboard after] panel %s -> %s: queued %+g failed %+g; round_duration_seconds_count %g -> %g (%+g)", kfPanelExpr, fmtMap(exp1), dq, df, rd0, rd1, rd1-rd0)

		t.Run("CounterMatchesWhatHappened", func(t *testing.T) {
			// Decision 19: a grace-0 forced deletion still delivers SIGTERM, so the
			// dying worker finishes or releases if it wins the race to the kill;
			// the lease expires only if it loses (decision 38). Whichever of the
			// three happened, the panel must move by exactly that.
			want := 1.0
			if released || completedInGrace {
				want = 0
			}
			if dq != want || df != 0 {
				t.Errorf("the build went back to the queue by %q, so lease_expirations{queued} should move by %g and {failed} by 0; the dashboard moved them %+g and %+g", path, want, dq, df)
			}
		})
		t.Run("RoundObservedOnce", func(t *testing.T) {
			if rd1-rd0 != 1 {
				t.Errorf("round_duration_seconds_count moved %+g on the dashboard, want +1 (decision 31)", rd1-rd0)
			}
		})
		t.Run("KilledPodSeriesEnded", func(t *testing.T) {
			res := g.query(t, fmt.Sprintf(`worker_busy{pod=%q}`, owner))
			if len(res) != 0 {
				t.Errorf("worker_busy for the deleted pod %s still on the dashboard: %v", owner, res)
			} else {
				t.Logf("[dashboard] worker_busy{pod=%s}: no current series (target gone)", owner)
			}
		})
		t.Run("ReplacementSeriesPreCreated", func(t *testing.T) {
			live, _ := kwPods(t, koNS, "app=worker")
			for _, w := range live {
				if w.Metadata.Name == owner {
					continue
				}
				up, _ := g.scalar(t, fmt.Sprintf(`count_over_time(up{job="worker",pod=%q}[10m])`, w.Metadata.Name))
				busy, _ := g.scalar(t, fmt.Sprintf(`count_over_time(worker_busy{pod=%q}[10m])`, w.Metadata.Name))
				outs := g.byLabel(t, fmt.Sprintf(`builds_total{pod=%q}`, w.Metadata.Name), "outcome")
				t.Logf("[dashboard] %s: %g scrapes of up, %g samples of worker_busy; builds_total by outcome %s", w.Metadata.Name, up, busy, fmtMap(outs))
				if busy < up {
					t.Errorf("%s: worker_busy present in %g of %g scrapes; decision 35 says from the first", w.Metadata.Name, busy, up)
				}
				if len(outs) == 0 {
					t.Errorf("%s: no builds_total series (decision 35)", w.Metadata.Name)
				}
			}
		})
		t.Run("GaugesAtRest", func(t *testing.T) { kfAtRest(t, g, koReplicas, base, 90*time.Second) })
	})

	// 2. Two injections on the same build in one round, then the scheduler
	// pod replaced: does each injection count once, does the "Lease
	// expirations (total)" panel keep its total, and does the new scheduler
	// pod's every series exist from its first scrape (decision 35)?
	t.Run("TwoKillsOneBuildThenSchedulerRestart", func(t *testing.T) {
		koWaitPool(t, koReplicas, 3*time.Minute)
		sched := kfSchedulerPod(t)
		exp0 := g.byLabel(t, kfPanelExpr, "outcome")
		rd0, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		t.Logf("[dashboard before] panel %s; round_duration_seconds_count{pod=%s}=%g", fmtMap(exp0), sched, rd0)

		rid := kmNewRound(t, "xcheck-step6-twokills", 1)
		s := kmSubmit(t, rid, 0, kfLongN)
		kmReportDone(t, rid, 0)
		expiries := 0
		for k := int32(1); k <= 2; k++ {
			b := kmWaitBuild(t, s.BuildID, 2*time.Minute, fmt.Sprintf("leased at attempt %d", k), func(b xBuild) bool {
				return b.State == "leased" && b.Attempt == k && b.LeaseOwner != nil
			})
			time.Sleep(2 * time.Second) // mid-build
			b = kmBuild(t, s.BuildID)
			if b.State != "leased" || b.Attempt != k {
				t.Fatalf("build moved on before the kill: %s", b)
			}
			killAt := kfForceDelete(t, b.owner())
			var left xBuild
			poll(t, 60*time.Second, 100*time.Millisecond, fmt.Sprintf("attempt %d leaves leased", k), func() bool {
				left = kmBuild(t, s.BuildID)
				return left.Attempt > k || left.State != "leased"
			})
			rel := time.Now().Before(*b.LeaseUntil)
			if !rel {
				expiries++
			}
			t.Logf("[api] kill #%d of %s: attempt %d left leased %.1fs after the kill (lease_until kill%+.1fs) -> released=%v: %s",
				k, b.owner(), k, time.Since(killAt).Seconds(), b.LeaseUntil.Sub(killAt).Seconds(), rel, left)
		}
		fin := kmWaitBuild(t, s.BuildID, 3*time.Minute, "done", isDone)
		if fin.Attempt != 3 {
			t.Errorf("build %s done at attempt %d, want 3 (two kills, invariant 4)", s.BuildID, fin.Attempt)
		}
		poll(t, 30*time.Second, 300*time.Millisecond, "round stamped finished", func() bool {
			r, code := koRound(t, rid)
			return code == http.StatusOK && r.Round.FinishedAt != nil
		})
		time.Sleep(time.Second)
		comps := c.completions()[s.BuildID]
		t.Logf("[logs] worker 'completed' lines for %s: %d %v", s.BuildID, len(comps), kfMsgs(comps))
		if len(comps) != 1 {
			t.Errorf("want exactly one completion of %s, saw %d", s.BuildID, len(comps))
		}
		scrapeWait(t, "the expiries and the round reach Prometheus")
		exp1 := g.byLabel(t, kfPanelExpr, "outcome")
		rd1, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		dq := exp1["queued"] - exp0["queued"]
		t.Logf("[dashboard after] panel %s (queued %+g, %d expiries inferred from the row); round_duration_seconds_count %+g", fmtMap(exp1), dq, expiries, rd1-rd0)
		t.Run("EachInjectionCountedOnce", func(t *testing.T) {
			if dq != float64(expiries) || exp1["failed"] != exp0["failed"] {
				t.Errorf("lease_expirations moved queued %+g failed %+g; %d attempts ended by expiry", dq, exp1["failed"]-exp0["failed"], expiries)
			}
			if rd1-rd0 != 1 {
				t.Errorf("round_duration_seconds_count moved %+g, want +1", rd1-rd0)
			}
		})

		// A real expiry for the panel to hold: the in-container crash of
		// decision 19 (kill -9 of the worker process, the pod stays).
		var crashExp map[string]float64
		t.Run("InContainerCrashExpiresOnce", func(t *testing.T) {
			koWaitPool(t, koReplicas, 3*time.Minute)
			r2 := kmNewRound(t, "xcheck-step6-crash", 1)
			s2 := kmSubmit(t, r2, 0, kfMidN)
			kmReportDone(t, r2, 0)
			b := kmWaitBuild(t, s2.BuildID, time.Minute, "leased", func(b xBuild) bool { return b.State == "leased" && b.LeaseOwner != nil })
			crashAt := kfCrashWorker(t, b.owner())
			var left xBuild
			poll(t, 60*time.Second, 100*time.Millisecond, "attempt 1 leaves leased", func() bool {
				left = kmBuild(t, s2.BuildID)
				return left.Attempt > 1 || left.State != "leased"
			})
			t.Logf("[api] attempt 1 left leased %.1fs after the crash (lease_until crash%+.1fs): %s", time.Since(crashAt).Seconds(), b.LeaseUntil.Sub(crashAt).Seconds(), left)
			fin := kmWaitBuild(t, s2.BuildID, 2*time.Minute, "done", isDone)
			poll(t, 30*time.Second, 300*time.Millisecond, "round stamped finished", func() bool {
				r, code := koRound(t, r2)
				return code == http.StatusOK && r.Round.FinishedAt != nil
			})
			scrapeWait(t, "the expiry reaches Prometheus")
			crashExp = g.byLabel(t, kfPanelExpr, "outcome")
			t.Logf("[dashboard] panel %s -> %s; build ended %s", fmtMap(exp1), fmtMap(crashExp), fin)
			if fin.Attempt != 2 || crashExp["queued"]-exp1["queued"] != 1 {
				t.Errorf("in-container crash: want attempt 2 and lease_expirations{queued} +1, got attempt %d and %+g", fin.Attempt, crashExp["queued"]-exp1["queued"])
			}
		})
		if crashExp == nil {
			crashExp = g.byLabel(t, kfPanelExpr, "outcome")
		}

		// Now replace the scheduler pod (a plain delete, as KILL_SCHEDULER=1 does).
		koWaitPool(t, koReplicas, 3*time.Minute)
		panelBefore := crashExp["queued"]
		so, se, err := kcRun("", "delete", "pod", sched, "-n", koNS, "--wait=false")
		t.Logf("[kubectl] delete pod %s: err=%v %s%s", sched, err, strings.TrimSpace(so), strings.TrimSpace(se))
		nsched := kfWaitNewScheduler(t, sched, 2*time.Minute)
		scrapeWait(t, "the new scheduler pod is scraped")
		scrapeWait(t, "and once more")

		t.Run("NewSchedulerSeriesFromFirstScrape", func(t *testing.T) {
			up, _ := g.scalar(t, fmt.Sprintf(`count_over_time(up{job="scheduler",pod=%q}[10m])`, nsched))
			for _, sel := range []string{
				`lease_expirations_total{outcome="queued",`, `lease_expirations_total{outcome="failed",`,
				`round_duration_seconds_count{`, `shard_lag_seconds_count{`, `gpu_queue_depth{`, `rounds_open{`,
			} {
				expr := fmt.Sprintf(`%spod=%q}`, sel, nsched)
				n, _ := g.scalar(t, "count_over_time("+expr+"[10m])")
				v, ok := g.scalar(t, expr)
				t.Logf("[dashboard] %s: now %g (present=%v), in %g of %g scrapes", expr, v, ok, n, up)
				if !ok || n < up {
					t.Errorf("%s not present from the new pod's first scrape (%g of %g), decision 35", expr, n, up)
				}
			}
			for _, fam := range []string{"placement_decisions_total", "build_queue_wait_seconds_count"} {
				res := g.query(t, fmt.Sprintf(`%s{pod=%q}`, fam, nsched))
				t.Logf("[dashboard] %s{pod=%s}: %d series", fam, nsched, len(res))
				if len(res) == 0 {
					t.Errorf("%s has no series on the new pod (decision 35)", fam)
				}
			}
		})
		t.Run("LeaseExpiryPanelSurvivesSchedulerRestart", func(t *testing.T) {
			// Decision 37: the panel reads windowed increase(), not the raw sum,
			// precisely so an expiry recorded before a scheduler replacement stays
			// on the dashboard (a counter lives in process memory; the raw sum of
			// current series fell to 0 here when this test first ran).
			raw := g.byLabel(t, kfPanelExpr, "outcome")
			panel := g.byLabel(t, `sum(increase(lease_expirations_total[1h])) by (outcome)`, "outcome")
			t.Logf("[dashboard] raw sum %s (allowed to reset: pod %s replaced by %s held %g); panel-style increase over 1h: %s",
				fmtMap(raw), sched, nsched, panelBefore, fmtMap(panel))
			if panel["queued"] < panelBefore {
				t.Errorf("the windowed-increase panel shows %g queued expiries after the scheduler pod was replaced, want at least the %g recorded before it (decision 37)", panel["queued"], panelBefore)
			}
		})
		t.Run("GaugesAtRest", func(t *testing.T) { kfAtRest(t, g, koReplicas, base, 90*time.Second) })
	})

	// 3. Decision 31 says a round is observed exactly once by whichever closer
	// stamps it; decision 29 keeps a deleted scheduler pod serving 3 s after
	// SIGTERM. Adversarial moment: the scheduler pod is deleted exactly as
	// the round becomes complete. If the OLD pod (still running its reaper in
	// its last seconds, already removed from the Service and hence from
	// Prometheus's targets) stamps the round, its observation is made in a
	// process nobody will scrape again. The dashboard must still show the
	// round once: (old pod's max count - its count before) + new pod's count.
	t.Run("SchedulerDeletedAtRoundClose", func(t *testing.T) {
		koWaitPool(t, koReplicas, 3*time.Minute)
		lost := 0
		for trial := 1; trial <= 4; trial++ {
			variant := "A: last report then delete at once"
			if trial%2 == 0 {
				variant = "B: delete ~1 s before the worker completes the build"
			}
			t.Logf("---- trial %d, variant %s ----", trial, variant)
			old := kfSchedulerPod(t)
			scrapeWait(t, "the current scheduler pod's count is fresh in Prometheus")
			before, ok := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, old))
			if !ok {
				t.Fatalf("no round_duration_seconds_count for %s", old)
			}
			rid := kmNewRound(t, fmt.Sprintf("xcheck-step6-close-%d", trial), 1)
			var delAt time.Time
			del := func() {
				delAt = time.Now()
				so, se, err := kcRun("", "delete", "pod", old, "-n", koNS, "--wait=false")
				t.Logf("[kubectl %s] delete pod %s: err=%v %s%s", delAt.Format("15:04:05.000"), old, err, strings.TrimSpace(so), strings.TrimSpace(se))
			}
			// Watch the old pod's own /metrics through the API server's pod
			// proxy while it lives: the in-process count.
			inproc := before
			stopWatch := make(chan struct{})
			watchDone := make(chan struct{})
			go func() {
				defer close(watchDone)
				for {
					so, _, err := kcRun("", "get", "--raw", fmt.Sprintf("/api/v1/namespaces/%s/pods/%s:8080/proxy/metrics", koNS, old))
					if err == nil {
						if e, perr := parseExpo(so); perr == nil {
							if v, ok := e.val("round_duration_seconds_count", nil); ok {
								inproc = v
							}
						}
					}
					select {
					case <-stopWatch:
						return
					case <-time.After(200 * time.Millisecond):
					}
				}
			}()
			if trial%2 == 1 {
				s := kmSubmit(t, rid, 0, kfShortN)
				kmWaitBuild(t, s.BuildID, time.Minute, "done", isDone)
				kmReportDone(t, rid, 0)
				del()
			} else {
				s := kmSubmit(t, rid, 0, kfMidN)
				kmReportDone(t, rid, 0)
				b := kmWaitBuild(t, s.BuildID, time.Minute, "leased", func(b xBuild) bool { return b.State == "leased" && b.StartedAt != nil })
				end := b.StartedAt.Add(time.Duration(float64(s.GPUTotalMs)/kfScale) * time.Millisecond)
				t.Logf("[plan] build started %s, gpu_total_ms=%d -> expected done ~%s; deleting the scheduler at done-1s", b.StartedAt.Format("15:04:05.000"), s.GPUTotalMs, end.Format("15:04:05.000"))
				time.Sleep(time.Until(end.Add(-time.Second)))
				del()
			}
			nsched := kfWaitNewScheduler(t, old, 2*time.Minute)
			close(stopWatch)
			<-watchDone
			var r ldRound
			poll(t, 60*time.Second, 300*time.Millisecond, fmt.Sprintf("round %d stamped finished", rid), func() bool {
				var code int
				r, code = koRound(t, rid)
				return code == http.StatusOK && r.Round.FinishedAt != nil
			})
			t.Logf("[api] round %d finished_at=%s (delete at %s, %+.2fs)", rid, r.Round.FinishedAt.Format("15:04:05.000"), delAt.Format("15:04:05.000"), r.Round.FinishedAt.Sub(delAt).Seconds())
			scrapeWait(t, "the new pod scraped twice")
			oldMax, _ := g.scalar(t, fmt.Sprintf(`max_over_time(round_duration_seconds_count{pod=%q}[10m])`, old))
			newVal, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, nsched))
			dash := (oldMax - before) + newVal
			who := "?"
			switch {
			case inproc-before == 1:
				who = old + " (the deleted pod, in its shutdown window)"
			case newVal >= 1:
				who = nsched + " (the replacement)"
			}
			t.Logf("[result trial %d] old pod %s: count before %g, last in-process %g (pod proxy), max ever scraped %g; new pod %s: %g -> the dashboard holds %g observation(s) of round %d; closer: %s",
				trial, old, before, inproc, oldMax, nsched, newVal, dash, rid, who)
			for _, l := range kfSchedulerLinesAbout(c, rid, "") {
				t.Logf("[logs %s] %s", l.Pod, kfLine(l))
			}
			if dash != 1 {
				lost++
				t.Errorf("trial %d (%s): round %d shows %g times on the dashboard, want exactly 1 (decisions 31, 36; done-when 'the round still completing' on the dashboard)", trial, variant, rid, dash)
			}
		}
		t.Logf("[summary] %d of 4 trials lost or doubled the round on the dashboard", lost)
		kfAtRest(t, g, koReplicas, base, 90*time.Second)
	})

	// 4. Drain ONE GPU node mid-round. Decision 24: "two nodes also make a
	// drain halve the pool instead of emptying it". The round is hand-driven
	// with 2 x replicas builds of ~6.5 s real each (longer than the worker's
	// 5 s finish budget, so an evicted worker must RELEASE its build, not
	// finish it), so the round is still open while the node stays drained.
	// Checked: the drained node's workers leave; the pool on the API and the
	// dashboard holds what the other node's cards allow and is never 0 on the
	// dashboard; every build completes exactly once; graceful eviction costs
	// no lease expiry (roadmap Lifecycle: on SIGTERM a worker finishes or
	// releases); the round is observed once; the gauges come back to rest.
	drain := func(t *testing.T, replicas int) {
		koScaleWorkers(t, replicas)
		koWaitPool(t, replicas, 3*time.Minute)
		gpus := kfGPUNodes(t)
		byNode, _ := kfWorkersByNode(t)
		t.Logf("[k8s] GPU nodes %v; workers by node before the drain %v", gpus, byNode)
		sched := kfSchedulerPod(t)
		scrapeWait(t, "the pool size is fresh on the dashboard")
		exp0 := g.byLabel(t, kfPanelExpr, "outcome")
		rd0, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		t0 := time.Now()
		smp := kfStartSampler(t, g)
		nb := 2 * replicas
		rid := kmNewRound(t, fmt.Sprintf("xcheck-step6-drain-%d", replicas), nb)
		var ids []string
		for sh := 0; sh < nb; sh++ {
			ids = append(ids, kmSubmit(t, rid, sh, kfMidN).BuildID)
			kmReportDone(t, rid, sh)
		}
		// Wait for the first wave: `replicas` builds leased.
		var r ldRound
		poll(t, time.Minute, 100*time.Millisecond, fmt.Sprintf("%d builds leased", replicas), func() bool {
			r, _ = koRound(t, rid)
			n := 0
			for _, b := range r.Builds {
				if b.State == "leased" {
					n++
				}
			}
			return n == replicas
		})
		// Drain the GPU node running the most workers.
		byNode, _ = kfWorkersByNode(t)
		placed := fmt.Sprint(byNode)
		node := ""
		for _, n := range gpus {
			if node == "" || len(byNode[n]) > len(byNode[node]) {
				node = n
			}
		}
		other := gpus[0]
		if other == node {
			other = gpus[1]
		}
		survivors := len(byNode[other]) // workers NOT on the drained node, before the drain
		leasedOn := map[string]int32{}
		for _, b := range r.Builds {
			if b.State == "leased" {
				leasedOn[b.BuildID] = b.Attempt
			}
		}
		t.Logf("[drain t+%.1fs] draining %s (workers there: %v; on %s: %v); leased builds %v", time.Since(t0).Seconds(), node, byNode[node], other, byNode[other], leasedOn)
		// Poll the API's pool every 200 ms from the drain to the held check:
		// the dashboard's 5 s scrapes can miss a short empty window.
		apiMin, apiSamples := 1<<30, 0
		var apiZeroAt []string
		stopAPI := make(chan struct{})
		apiDone := make(chan struct{})
		go func() {
			defer close(apiDone)
			for {
				var ww xWorkers
				if code, _, err := koGet("/workers", &ww); err == nil && code == http.StatusOK {
					apiSamples++
					if ww.Pool.LiveWorkers < apiMin {
						apiMin = ww.Pool.LiveWorkers
					}
					if ww.Pool.LiveWorkers == 0 {
						apiZeroAt = append(apiZeroAt, fmt.Sprintf("t+%.1fs", time.Since(t0).Seconds()))
					}
				}
				select {
				case <-stopAPI:
					return
				case <-time.After(200 * time.Millisecond):
				}
			}
		}()
		so, se, err := kcRun("", "drain", node, "--ignore-daemonsets", "--delete-emptydir-data", "--timeout=120s")
		t.Logf("[kubectl t+%.1fs] drain %s: err=%v\n%s%s", time.Since(t0).Seconds(), node, err, tail(strings.TrimSpace(so), 8), tail(strings.TrimSpace(se), 4))
		if err != nil {
			t.Fatalf("drain failed")
		}
		time.Sleep(12 * time.Second) // the evicted workers' heartbeats go stale or deregister, plus a scrape
		close(stopAPI)
		<-apiDone
		t.Logf("[api] GET /workers polled %d times every 200 ms from the drain on: min pool live_workers=%d; zero at %v", apiSamples, apiMin, apiZeroAt)
		byNode, pending := kfWorkersByNode(t)
		w := koWorkers(t)
		dashLive, _ := g.scalar(t, "max(gpu_workers_live)")
		dashQ, _ := g.scalar(t, "max(gpu_queue_depth)")
		r, _ = koRound(t, rid)
		open := r.Round.FinishedAt == nil
		t.Logf("[held t+%.1fs] workers by node %v, unscheduled %v; GET /workers live=%d; dashboard max(gpu_workers_live)=%g max(gpu_queue_depth)=%g; round still open=%v", time.Since(t0).Seconds(), byNode, pending, w.Pool.LiveWorkers, dashLive, dashQ, open)
		if len(byNode[node]) != 0 {
			t.Errorf("drained node %s still runs workers %v", node, byNode[node])
		}
		if !open {
			t.Logf("[note] the round closed before the held check; the pool check below is still meaningful")
		}
		wantHeld := replicas
		if wantHeld > 2 {
			wantHeld = 2 // the other node's two cards (decisions 22, 24)
		}
		if w.Pool.LiveWorkers != wantHeld || dashLive != float64(wantHeld) {
			t.Errorf("pool while %s is drained: API %d, dashboard %g; want %d (what %s's cards allow)", node, w.Pool.LiveWorkers, dashLive, wantHeld, other)
		}
		so, se, err = kcRun("", "uncordon", node)
		t.Logf("[kubectl t+%.1fs] uncordon %s: err=%v %s%s", time.Since(t0).Seconds(), node, err, strings.TrimSpace(so), strings.TrimSpace(se))
		poll(t, 3*time.Minute, 300*time.Millisecond, fmt.Sprintf("round %d stamped finished", rid), func() bool {
			var code int
			r, code = koRound(t, rid)
			return code == http.StatusOK && r.Round.FinishedAt != nil
		})
		t.Logf("[api t+%.1fs] round %d finished; duration %.1fs", time.Since(t0).Seconds(), rid, r.Round.FinishedAt.Sub(r.Round.StartedAt).Seconds())
		time.Sleep(time.Second)
		comps := c.completions()
		for _, b := range r.Builds {
			cs := comps[b.BuildID]
			pre, wasLeased := leasedOn[b.BuildID]
			t.Logf("[build] %s attempt=%d state=%s (leased at drain: %v@%d) completions %v", b.BuildID, b.Attempt, b.State, wasLeased, pre, kfMsgs(cs))
			if b.State != "done" || len(cs) != 1 || cs[0].str("attempt") != fmt.Sprint(b.Attempt) {
				t.Errorf("build %s: state %s, %d completions %v, row attempt %d; want done once at the row's attempt", b.BuildID, b.State, len(cs), kfMsgs(cs), b.Attempt)
			}
		}
		koWaitPool(t, replicas, 3*time.Minute)
		scrapeWait(t, "the round's last numbers reach Prometheus")
		smp.stop()
		smp.narrate(t, t0)
		minLive := 1e9
		for _, x := range smp.all() {
			if x.R.Live < minLive {
				minLive = x.R.Live
			}
		}
		exp1 := g.byLabel(t, kfPanelExpr, "outcome")
		rd1, _ := g.scalar(t, fmt.Sprintf(`round_duration_seconds_count{pod=%q}`, sched))
		t.Logf("[dashboard] min of max(gpu_workers_live) over %d samples: %g; lease expirations %s -> %s; round_duration_seconds_count %+g", len(smp.all()), minLive, fmtMap(exp0), fmtMap(exp1), rd1-rd0)
		t.Run("PoolThroughTheDrain", func(t *testing.T) {
			// Decision 38 (refining 24): "a drain halves the pool" holds when the
			// workers are spread across the nodes, which every-card-in-use forces.
			// With fewer replicas than nodes Kubernetes may co-schedule them all on
			// one node (decision 22 rejected anti-affinity), and a one-node drain
			// then empties the pool for the moment rescheduling takes.
			if survivors == 0 {
				t.Logf("[decision 38] every worker sat on the drained node (placement %s): the pool may empty; it dipped to API min %d (zero at %v), dashboard min %g, and recovered to %d", placed, apiMin, apiZeroAt, minLive, replicas)
				return
			}
			if minLive < 1 || apiMin < 1 {
				t.Errorf("pool emptied during a ONE-node drain with %d workers on the surviving node: dashboard min %g, API min %d at %v (decisions 24, 38); workers at the drain were placed %s", survivors, minLive, apiMin, apiZeroAt, placed)
			}
		})
		t.Run("NoLeaseExpiryFromEviction", func(t *testing.T) {
			if exp1["queued"] != exp0["queued"] || exp1["failed"] != exp0["failed"] {
				t.Errorf("a graceful eviction cost lease expiries: %s -> %s", fmtMap(exp0), fmtMap(exp1))
			}
		})
		t.Run("RoundObservedOnce", func(t *testing.T) {
			if rd1-rd0 != 1 {
				t.Errorf("round_duration_seconds_count moved %+g, want +1", rd1-rd0)
			}
		})
		t.Run("GaugesAtRest", func(t *testing.T) { kfAtRest(t, g, replicas, base, 90*time.Second) })
	}
	t.Run("DrainOneGPUNodeAllFourCards", func(t *testing.T) {
		t.Cleanup(func() { kfUncordonGPU(t) })
		drain(t, 4)
		koScaleWorkers(t, koReplicas)
		koWaitPool(t, koReplicas, 3*time.Minute)
	})
	t.Run("DrainOneGPUNodeTwoReplicas", func(t *testing.T) {
		t.Cleanup(func() { kfUncordonGPU(t) })
		drain(t, 2)
	})
}
