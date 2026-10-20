package crosscheck

// Cross-check of the Phase 2 amendment "the draining scheduler" (design
// decision 40, read with decisions 18, 29, 31 and 37 and the implementation
// notes section "Amendment after close: the draining scheduler").
//
// The promise under test, in the doc's words: "Past SIGTERM the scheduler is a
// lame duck: it answers but never stamps a round." GET /rounds/{id} on a dying
// pod answers truthfully from Postgres and does not stamp (so it never
// observes round metrics); every other endpoint keeps writing (bug 10 must not
// return); the replacement stamps, so each round is observed exactly once
// (decision 31) on a pod Prometheus scrapes.
//
// Black box only, against the implementer's live namespace kgpu on kind
// context kind-kgpu: the API on the NodePort, one specific pod (dying or new)
// through the API server's pod proxy behind a private `kubectl proxy`, rows
// through psql in postgres-0, metrics from each pod's /metrics and from
// Prometheus through Grafana's datasource proxy (decision 36). The dying pod's
// "draining" log line is used only to know SIGTERM has been delivered, so a
// stamp after that moment cannot be blamed on the reaper's last pre-signal
// tick.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const lkProxyPort = 28197

// ---- kubectl proxy: reach any pod, even a terminating one, by name ----

type lkProxy struct {
	cmd *exec.Cmd
	out *syncBuffer
}

func lkStartProxy(t *testing.T) *lkProxy {
	t.Helper()
	buf := &syncBuffer{}
	cmd := exec.Command("kubectl", "--context", kContext, "proxy", fmt.Sprintf("--port=%d", lkProxyPort))
	cmd.Stdout, cmd.Stderr = buf, buf
	if err := cmd.Start(); err != nil {
		t.Fatalf("start kubectl proxy: %v", err)
	}
	p := &lkProxy{cmd: cmd, out: buf}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	poll(t, 15*time.Second, 200*time.Millisecond, "kubectl proxy answering", func() bool {
		code, _, err := httpText(fmt.Sprintf("http://127.0.0.1:%d/version", lkProxyPort))
		return err == nil && code == 200
	})
	t.Logf("[kubectl proxy] on 127.0.0.1:%d; a pod is reached at /api/v1/namespaces/kgpu/pods/<pod>:8080/proxy/<path>", lkProxyPort)
	return p
}

var lkHTTP = &http.Client{Timeout: 2 * time.Second}

// do sends one request to one named scheduler pod.
func (p *lkProxy) do(pod, method, path string, body any) (int, string, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	u := fmt.Sprintf("http://127.0.0.1:%d/api/v1/namespaces/%s/pods/%s:8080/proxy%s", lkProxyPort, koNS, pod, path)
	req, err := http.NewRequest(method, u, rd)
	if err != nil {
		return 0, "", err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := lkHTTP.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, strings.TrimSpace(string(b)), nil
}

// roundCount reads round_duration_seconds_count from one pod's /metrics.
func (p *lkProxy) roundCount(pod string) (float64, error) {
	code, body, err := p.do(pod, "GET", "/metrics", nil)
	if err != nil {
		return 0, err
	}
	if code != 200 {
		return 0, fmt.Errorf("status %d %.120s", code, body)
	}
	e, err := parseExpo(body)
	if err != nil {
		return 0, err
	}
	v, ok := e.val("round_duration_seconds_count", nil)
	if !ok {
		return 0, fmt.Errorf("round_duration_seconds_count absent")
	}
	return v, nil
}

func (p *lkProxy) mustRoundCount(t *testing.T, pod string) float64 {
	t.Helper()
	var v float64
	var err error
	for i := 0; i < 20; i++ {
		if v, err = p.roundCount(pod); err == nil {
			return v
		}
		time.Sleep(300 * time.Millisecond)
	}
	t.Fatalf("%s /metrics: %v", pod, err)
	return 0
}

// getRound GETs /rounds/{id} on one pod and returns finished_at ("" = null).
func (p *lkProxy) getRound(pod string, id int64) (int, string, error) {
	code, body, err := p.do(pod, "GET", fmt.Sprintf("/rounds/%d", id), nil)
	if err != nil || code != 200 {
		return code, body, err
	}
	var r ldRound
	if err := json.Unmarshal([]byte(body), &r); err != nil {
		return code, "", fmt.Errorf("decode: %v: %.200s", err, body)
	}
	if r.Round.FinishedAt == nil {
		return code, "", nil
	}
	return code, r.Round.FinishedAt.UTC().Format("15:04:05.000000"), nil
}

// ---- Postgres in namespace kgpu (read only) ----

func lkPsql(t *testing.T, sql string) string {
	t.Helper()
	return strings.TrimSpace(kc(t, "exec", "-n", koNS, "postgres-0", "-c", "postgres", "--",
		"psql", "-U", "postgres", "-d", "kgpu", "-At", "-v", "ON_ERROR_STOP=1", "-c", sql))
}

// lkFinished returns rounds.finished_at for id as HH:MM:SS.micro UTC, "" if NULL.
func lkFinished(t *testing.T, id int64) string {
	t.Helper()
	return lkPsql(t, fmt.Sprintf(
		"SELECT coalesce(to_char(finished_at AT TIME ZONE 'UTC','HH24:MI:SS.US'),'') FROM rounds WHERE round_id=%d", id))
}

// ---- following the dying pod's log ----

type lkFollow struct {
	cmd *exec.Cmd
	buf *syncBuffer
}

func lkFollowLogs(t *testing.T, pod string) *lkFollow {
	t.Helper()
	buf := &syncBuffer{}
	cmd := exec.Command("kubectl", "--context", kContext, "-n", koNS, "logs", "-f", "--tail=0", pod)
	cmd.Stdout, cmd.Stderr = buf, buf
	if err := cmd.Start(); err != nil {
		t.Fatalf("follow logs of %s: %v", pod, err)
	}
	f := &lkFollow{cmd: cmd, buf: buf}
	t.Cleanup(f.stop)
	time.Sleep(500 * time.Millisecond) // let the stream attach before anything happens
	return f
}

func (f *lkFollow) stop() {
	if f.cmd.Process != nil {
		_ = f.cmd.Process.Kill()
		_ = f.cmd.Wait()
		f.cmd.Process = nil
	}
}

func (f *lkFollow) lines(sub string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(f.buf.String()))
	for sc.Scan() {
		if strings.Contains(sc.Text(), sub) {
			out = append(out, sc.Text())
		}
	}
	return out
}

// ---- building blocks ----

// lkCompletedBuild creates a one-shard round and drives its only build to
// done, without the shard reporting stream_done: the round is NOT complete
// yet (decision 30: completion needs stream_done).
func lkCompletedBuild(t *testing.T, scenario string) (int64, xBuild) {
	t.Helper()
	id := kmNewRound(t, scenario, 1)
	s := kmSubmit(t, id, 0, 20_000)
	if s.Placement == "local" {
		code, raw := kmDo(t, "POST", "/builds/"+s.BuildID+"/done", nil, nil)
		if code != 200 {
			t.Fatalf("POST /builds/%s/done -> %d %s", s.BuildID, code, raw)
		}
		t.Logf("[api] POST /builds/%s/done -> 200 (local build reported finished by the shard)", s.BuildID)
	}
	b := kmWaitBuild(t, s.BuildID, 120*time.Second, "state=done", func(b xBuild) bool { return b.State == "done" })
	return id, b
}

// lkDelete deletes a pod without waiting and returns when it was asked.
func lkDelete(t *testing.T, pod string) time.Time {
	t.Helper()
	at := time.Now()
	so, se, err := kcRun("", "delete", "pod", pod, "-n", koNS, "--wait=false")
	if err != nil {
		t.Fatalf("delete %s: %v %s%s", pod, err, so, se)
	}
	t.Logf("[kubectl %s] delete pod %s (graceful, SIGTERM; decision 29 says it keeps serving ~3 s)", at.Format("15:04:05.000"), pod)
	return at
}

// lkWaitDraining waits for the dying pod's "draining" log line, the evidence
// SIGTERM has been delivered. Returns false if it never appears in d.
func lkWaitDraining(t *testing.T, f *lkFollow, del time.Time, d time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if ls := f.lines("draining"); len(ls) > 0 {
			t.Logf("[log +%.0fms] dying pod: %s", time.Since(del).Seconds()*1000, strings.TrimSpace(ls[0]))
			return true
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Logf("[log] no 'draining' line from the dying pod within %s of the delete; log so far: %q", d, f.buf.String())
	return false
}

type lkObs struct {
	at       time.Duration
	code     int
	finished string
	err      error
	pg       string
	count    float64
	countErr error
}

// lkWatchDying polls the dying pod (GET /rounds/{id}, its /metrics) and
// Postgres until the pod stops answering or d passes.
func lkWatchDying(t *testing.T, p *lkProxy, pod string, id int64, del time.Time, d time.Duration) []lkObs {
	t.Helper()
	var obs []lkObs
	deadline := time.Now().Add(d)
	fails := 0
	for time.Now().Before(deadline) && fails < 3 {
		var o lkObs
		o.code, o.finished, o.err = p.getRound(pod, id)
		o.at = time.Since(del)
		o.count, o.countErr = p.roundCount(pod)
		o.pg = lkFinished(t, id)
		if o.err != nil || o.code != 200 {
			fails++
		}
		obs = append(obs, o)
		fin := o.finished
		if fin == "" {
			fin = "null"
		}
		pg := o.pg
		if pg == "" {
			pg = "NULL"
		}
		cnt := fmt.Sprintf("%.0f", o.count)
		if o.countErr != nil {
			cnt = "unreachable"
		}
		t.Logf("[dying %s +%5.0fms] GET /rounds/%d -> %d finished_at=%s (err=%v); its round_duration_seconds_count=%s; Postgres finished_at=%s",
			pod, o.at.Seconds()*1000, id, o.code, fin, o.err, cnt, pg)
		time.Sleep(150 * time.Millisecond)
	}
	return obs
}

func lkReportDone(p *lkProxy, pod string, round int64, shard int) (int, string, error) {
	return p.do(pod, "POST", fmt.Sprintf("/shards/%d/%d/report", round, shard), map[string]any{
		"queue_depth": 0, "waiting_needs_index": 0, "oldest_wait_ms": 0, "build_progress": 0, "stream_done": true,
	})
}

// lkQuiet checks no other round is being stamped right now: the scheduler's
// count is stable over 2.5 s (more than one reaper interval), so the exact
// counts below are about our rounds only.
func lkQuiet(t *testing.T, p *lkProxy, pod string) float64 {
	t.Helper()
	a := p.mustRoundCount(t, pod)
	time.Sleep(2500 * time.Millisecond)
	b := p.mustRoundCount(t, pod)
	if a != b {
		t.Fatalf("scheduler %s stamped rounds on its own (%v -> %v in 2.5 s): someone else is driving rounds, the exact counts would be meaningless", pod, a, b)
	}
	open := lkPsql(t, "SELECT count(*) FROM rounds WHERE finished_at IS NULL")
	t.Logf("[pre] %s round_duration_seconds_count=%.0f, stable over 2.5 s; rounds with finished_at NULL in Postgres: %s (abandoned or ours, never complete)", pod, a, open)
	return a
}

type lkCycleResult struct {
	oldDelta, newCount float64
	oldPod, newPod     string
	round              int64
}

// lkCycle runs one replacement with one round completing around it.
//
// mode "in-window": the build is done before the delete; the shard's
// stream_done report goes to the dying pod after SIGTERM, so the round
// becomes complete only inside the lame-duck window: the dying pod must not
// stamp it, the replacement must.
//
// mode "before-delete": the stream_done report goes through the NodePort a
// moment before the delete, racing the old pod's last reaper tick against
// SIGTERM: either pod may stamp, but exactly one must (decision 31).
func lkCycle(t *testing.T, p *lkProxy, g *kfGraf, mode string) lkCycleResult {
	t.Helper()
	old := kfSchedulerPod(t)
	base := lkQuiet(t, p, old)
	id, b := lkCompletedBuild(t, "xcheck-lameduck-"+mode)
	if pg := lkFinished(t, id); pg != "" {
		t.Fatalf("round %d already stamped (%s) before the shard reported stream_done; decision 30 says completion needs it", id, pg)
	}
	t.Logf("[pg] round %d: its only build is done (finished_at %s) but finished_at is NULL: the shard has not reported stream_done (decision 30)", id, b.FinishedAt.UTC().Format("15:04:05.000000"))

	f := lkFollowLogs(t, old)
	if mode == "before-delete" {
		raw := kmReportDone(t, id, 0)
		t.Logf("[api] POST /shards/%d/0/report {stream_done:true} via NodePort -> 200 %s; the round is complete now; deleting at once", id, raw)
	}
	if mode == "stamped-before-delete" {
		raw := kmReportDone(t, id, 0)
		t.Logf("[api] POST /shards/%d/0/report {stream_done:true} via NodePort -> 200 %s; waiting for the healthy pod's reaper to stamp it, then deleting at once", id, raw)
		took := poll(t, 10*time.Second, 50*time.Millisecond, "the healthy pod's reaper stamping round", func() bool { return lkFinished(t, id) != "" })
		t.Logf("[pg] round %d stamped %.2fs after the report, by the still-healthy %s; deleting it now, inside the 5 s scrape interval: decision 40's accepted residual race", id, took.Seconds(), old)
	}
	del := lkDelete(t, old)
	sig := lkWaitDraining(t, f, del, 2500*time.Millisecond)
	if !sig {
		t.Errorf("the dying pod never logged 'draining' (implementation notes: 'shutdown requested; draining: serving without stamping'); cannot place SIGTERM")
	}

	if mode == "in-window" {
		code, body, err := lkReportDone(p, old, id, 0)
		t.Logf("[dying %s +%.0fms] POST /shards/%d/0/report {stream_done:true} -> %d %s err=%v", old, time.Since(del).Seconds()*1000, id, code, body, err)
		if err != nil || code != 200 {
			t.Fatalf("a shard report to the draining pod was refused (%d %v): decision 40 says every endpoint except stamp-on-read keeps writing, or bug 10's stall returns", code, err)
		}
		sd := lkPsql(t, fmt.Sprintf("SELECT stream_done FROM shard_status WHERE round_id=%d AND shard_id=0", id))
		t.Logf("[pg] shard_status(%d,0).stream_done=%s: the dying pod recorded the report; the round is now complete in Postgres", id, sd)
		if sd != "t" {
			t.Errorf("stream_done not recorded by the draining pod: %q", sd)
		}
	}

	obs := lkWatchDying(t, p, old, id, del, 8*time.Second)
	served, sawOpen := 0, false
	var lastCount float64 = -1
	for _, o := range obs {
		if o.err == nil && o.code == 200 {
			served++
			// Truthful: what the dying pod says must be what Postgres held
			// around that moment. A null answer while Postgres says stamped
			// is fine only if the stamp came between the two reads; a
			// non-null answer must equal Postgres (stamps never change).
			if o.finished != "" && o.finished != o.pg {
				t.Errorf("dying pod said finished_at=%s, Postgres %s", o.finished, o.pg)
			}
			if o.finished == "" {
				sawOpen = true
			}
		} else if o.err == nil && o.code != 200 {
			t.Logf("[dying] non-200 %d (the pod is shutting down)", o.code)
		}
		if o.countErr == nil {
			lastCount = o.count
		}
	}
	if served == 0 {
		t.Errorf("the dying pod answered no GET /rounds/%d after SIGTERM; decision 29 says it serves ~3 s", id)
	}
	if mode == "in-window" {
		if !sawOpen {
			t.Errorf("the dying pod never answered finished_at=null for round %d although the round was complete-but-unstamped; did it stamp on read? (decision 40)", id)
		}
		for _, o := range obs {
			if o.countErr == nil && o.count != base {
				t.Errorf("the dying pod's round_duration_seconds_count moved %v -> %v at +%.0fms: it observed a round past SIGTERM (decision 40)", base, o.count, o.at.Seconds()*1000)
				break
			}
		}
	}
	if lastCount < 0 {
		t.Fatalf("never read the dying pod's /metrics after SIGTERM")
	}
	lastServed := time.Duration(0)
	for _, o := range obs {
		if o.err == nil && o.code == 200 {
			lastServed = o.at
		}
	}
	t.Logf("[dying] served %d of %d GETs; last 200 at +%.0fms after the delete (decision 29: ~3 s after SIGTERM); its final round_duration_seconds_count=%.0f (was %.0f)",
		served, len(obs), lastServed.Seconds()*1000, lastCount, base)
	if ls := f.lines(fmt.Sprintf("%d", id)); len(ls) > 0 {
		t.Logf("[log] dying pod lines mentioning %d: %v", id, ls)
	}

	nw := kfWaitNewScheduler(t, old, 90*time.Second)
	took := poll(t, 20*time.Second, 200*time.Millisecond, fmt.Sprintf("round %d stamped in Postgres", id), func() bool { return lkFinished(t, id) != "" })
	pg := lkFinished(t, id)
	t.Logf("[pg] round %d finished_at=%s, stamped %.1fs after %s became Ready (reaper interval 2 s)", id, pg, took.Seconds(), nw)
	if want := b.FinishedAt.UTC().Format("15:04:05.000000"); pg != want {
		t.Errorf("round finished_at %s != its last child's finish %s (phase 1 decision 18: finished_at is the latest build or job finish, not the stamp time)", pg, want)
	}
	time.Sleep(time.Second)
	nc := p.mustRoundCount(t, nw)
	oldDelta := lastCount - base
	t.Logf("[metrics] old %s observed %.0f round(s) in this cycle; new %s round_duration_seconds_count=%.0f; sum=%.0f, decision 31 wants exactly 1",
		old, oldDelta, nw, nc, oldDelta+nc)
	if oldDelta+nc != 1 {
		t.Errorf("round %d observed %.0f times across the replacement (old %.0f + new %.0f); decision 31 says exactly once", id, oldDelta+nc, oldDelta, nc)
	}
	if mode == "in-window" && nc != 1 {
		t.Errorf("the replacement observed %.0f rounds, want 1 (decision 40: the replacement stamps)", nc)
	}

	// The scraped view: the observation must land somewhere Prometheus sees it.
	if g != nil {
		scrapeWait(t, "the new pod's count to reach Prometheus")
		byPod := g.byLabel(t, `max_over_time(round_duration_seconds_count{namespace="kgpu"}[10m])`, "pod")
		t.Logf("[grafana proxy] max_over_time(round_duration_seconds_count[10m]) by pod: %s", fmtMap(byPod))
		if v, ok := byPod[nw]; !ok || v != nc {
			t.Errorf("Prometheus has %s at %v (present=%v), the pod itself says %v", nw, v, ok, nc)
		}
		if v, ok := byPod[old]; ok {
			if v != lastCount {
				t.Logf("[note] Prometheus's last scrape of the dying %s saw %v, the pod's final value was %v: %v observation(s) never scraped", old, v, lastCount, lastCount-v)
				if mode == "in-window" {
					t.Errorf("dying pod's scraped count %v != its final %v in in-window mode", v, lastCount)
				}
			} else {
				t.Logf("[grafana proxy] the dying %s's last scrape (%v) equals its final value: nothing it observed was lost", old, v)
			}
		} else {
			t.Logf("[note] Prometheus has no series for the dying %s at all", old)
		}
	}
	return lkCycleResult{oldDelta: oldDelta, newCount: nc, oldPod: old, newPod: nw, round: id}
}

func TestKindLameDuck(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set")
	}
	t.Cleanup(func() {
		so, _, _ := kcRun("", "-n", koNS, "get", "pods", "-o", "wide")
		t.Logf("[cleanup] namespace kgpu after the test:\n%s", so)
	})
	p := lkStartProxy(t)
	g := kfGrafana(t)
	waitPodsReady(t, koNS, "app=scheduler", 1, 90*time.Second)

	t.Run("DyingPodAnswersOpenAndNeverStamps", func(t *testing.T) {
		t.Logf("Decision 40: a round that becomes complete inside the dying pod's window is answered as open (finished_at null, truthful) and not stamped by that pod; the replacement stamps it and its round_duration_seconds_count is exactly 1")
		r := lkCycle(t, p, g, "in-window")
		t.Logf("[result] round %d: dying %s observed %.0f, replacement %s observed %.0f", r.round, r.oldPod, r.oldDelta, r.newPod, r.newCount)
	})

	t.Run("DyingPodStillRecordsWrites", func(t *testing.T) {
		t.Logf("Decision 40: every endpoint other than stamp-on-read keeps writing while draining (POST /builds, POST /jobs, POST /shards/.../report), or bug 10's stall would return")
		old := kfSchedulerPod(t)
		id := kmNewRound(t, "xcheck-lameduck-writes", 2)
		s0 := kmSubmit(t, id, 0, 20_000)
		f := lkFollowLogs(t, old)
		del := lkDelete(t, old)
		if !lkWaitDraining(t, f, del, 2500*time.Millisecond) {
			t.Errorf("no 'draining' log line; cannot be sure the writes below landed after SIGTERM")
		}
		type call struct {
			method, path string
			body         any
			want         int
		}
		calls := []call{
			{"POST", "/builds", map[string]any{"round_id": id, "shard_id": 1, "n_vectors": 20_000, "dim": 128}, 201},
			{"POST", fmt.Sprintf("/jobs/%d/0", id), map[string]any{"seq": 1, "duration_ms": 10, "needs_index": false}, 201},
			{"POST", fmt.Sprintf("/shards/%d/0/report", id), map[string]any{"queue_depth": 1, "waiting_needs_index": 0, "oldest_wait_ms": 5, "build_progress": 0, "stream_done": false}, 200},
		}
		for _, c := range calls {
			code, body, err := p.do(old, c.method, c.path, c.body)
			t.Logf("[dying %s +%.0fms] %s %s -> %d %.200s err=%v (API table wants %d)", old, time.Since(del).Seconds()*1000, c.method, c.path, code, body, err, c.want)
			if err != nil || code != c.want {
				t.Errorf("%s %s on the draining pod -> %d err=%v, want %d (decision 40: writes continue)", c.method, c.path, code, err, c.want)
			}
		}
		b1 := lkPsql(t, fmt.Sprintf("SELECT build_id||' '||placement||' '||state FROM builds WHERE round_id=%d AND shard_id=1", id))
		j := lkPsql(t, fmt.Sprintf("SELECT count(*) FROM shard_jobs WHERE round_id=%d AND shard_id=0 AND seq=1", id))
		st := lkPsql(t, fmt.Sprintf("SELECT queue_depth||' '||oldest_wait_ms||' '||stream_done FROM shard_status WHERE round_id=%d AND shard_id=0", id))
		t.Logf("[pg] builds(shard 1)=%q; shard_jobs(shard 0, seq 1) rows=%s; shard_status(shard 0)=%q (queue_depth oldest_wait_ms stream_done)", b1, j, st)
		if b1 == "" {
			t.Errorf("the build submitted to the draining pod is not in Postgres")
		}
		if j != "1" {
			t.Errorf("the arrival posted to the draining pod is not in Postgres (%s rows)", j)
		}
		if st != "1 5 false" {
			t.Errorf("the report posted to the draining pod is not recorded: %q, want \"1 5 false\"", st)
		}
		nw := kfWaitNewScheduler(t, old, 90*time.Second)
		code, body, err := p.do(nw, "GET", fmt.Sprintf("/rounds/%d", id), nil)
		var r ldRound
		_ = json.Unmarshal([]byte(body), &r)
		t.Logf("[new %s] GET /rounds/%d -> %d err=%v: %d builds, %d jobs, %d shard statuses, finished_at=%v (shard 0 still has a queued query and no stream_done, so open is right)",
			nw, id, code, err, len(r.Builds), len(r.Jobs), len(r.ShardStatus), r.Round.FinishedAt)
		if code != 200 || len(r.Builds) != 2 || len(r.Jobs) != 1 || len(r.ShardStatus) != 1 {
			t.Errorf("the replacement does not see what the draining pod wrote (Postgres is the only state, invariant 1)")
		}
		if r.Round.FinishedAt != nil {
			t.Errorf("round %d stamped although shard 0 never reported stream_done", id)
		}
		_ = s0
	})

	t.Run("ExactlyOnceAcrossReplacements", func(t *testing.T) {
		t.Logf("Decision 31 through decision 40: over several replacements, each with a round completing in or right before the window, every round is observed exactly once (never zero, never twice) summed over the dying and the new pod")
		modes := []string{"in-window", "before-delete", "stamped-before-delete", "in-window", "before-delete", "stamped-before-delete"}
		var rows []string
		for i, m := range modes {
			t.Logf("---- repetition %d/%d, mode %s ----", i+1, len(modes), m)
			r := lkCycle(t, p, g, m)
			rows = append(rows, fmt.Sprintf("rep %d %-21s round %d: dying %s +%.0f, new %s %.0f", i+1, m, r.round, r.oldPod, r.oldDelta, r.newPod, r.newCount))
		}
		t.Logf("[summary]\n%s", strings.Join(rows, "\n"))
	})

	// Leave the stack as found: one Ready scheduler, two workers.
	waitPodsReady(t, koNS, "app=scheduler", 1, 90*time.Second)
	ws, _ := kwPods(t, koNS, "app=worker")
	t.Logf("[cleanup] workers untouched: %v", podNames(ws))
}
