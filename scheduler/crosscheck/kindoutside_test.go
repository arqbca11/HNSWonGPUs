package crosscheck

// Cross-check of Phase 2 step 4: the shards run outside the cluster.
//
// Spec (docs/design/phase2-kubernetes.md): decision 26, "the shard simulator
// is a client on the laptop ... the cluster holds only the control plane and
// the GPU pool"; decision 27, "the scheduler is reachable from outside
// through a NodePort (port 30080), which kind maps to the laptop with
// extraPortMappings on the control-plane node; the in-cluster scheduler:8080
// address is unchanged"; the system diagram, "the shard simulator runs on the
// laptop and reaches the scheduler through one address, a NodePort mapped to
// 127.0.0.1:30080"; deploy/k8s/scheduler.yaml (type NodePort, port 8080,
// nodePort 30080); deploy/kind/cluster.yaml (containerPort 30080 -> hostPort
// 30080, listenAddress 127.0.0.1). The simulator's knobs are the env vars in
// the header of shard/cmd/shardsim/main.go.
//
// Run with KGPU_KIND=1 against the kind-kgpu cluster with the stack applied
// in namespace kgpu. The tests run rounds against that stack, scale its
// worker Deployment (restored to 2 in Cleanup) and delete its scheduler pod.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestKindOutside(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set: needs the kind-kgpu cluster with the stack in namespace kgpu")
	}
	t.Cleanup(func() { koRestoreWorkers(t) })
	scaleStr := koTimeScale(t)
	scale, err := strconv.ParseFloat(scaleStr, 64)
	if err != nil {
		t.Fatalf("FAKE_TIME_SCALE %q: %v", scaleStr, err)
	}
	t.Logf("[k8s] worker ConfigMap FAKE_TIME_SCALE=%s; every simulator here runs with TIME_SCALE=%s (shardsim header: must be the same number)", scaleStr, scaleStr)
	c := startKoCollector(t)
	koWaitPool(t, koReplicas, 3*time.Minute)

	t.Run("NodePortIsTheOnlyDoor", func(t *testing.T) { koTestDoor(t) })
	t.Run("BimodalRoundFromLaptop", func(t *testing.T) { koTestBimodal(t, c, scaleStr, scale) })
	t.Run("SchedulerPodDeletedMidRound", func(t *testing.T) { koTestSchedulerDeleted(t, c, scaleStr, scale) })
	t.Run("TwoSimulatorsAtOnce", func(t *testing.T) { koTestTwoSims(t, c, scaleStr, scale) })
	t.Run("NoWorkersThenScaleUp", func(t *testing.T) { koTestNoWorkers(t, c, scaleStr, scale) })
	t.Run("FiftyShardsFourWorkers", func(t *testing.T) { koTestFifty(t, c, scaleStr, scale) })
	t.Run("SimulatorOnlyReachesNodePort", func(t *testing.T) { koTestOtherAddrs(t, scaleStr) })
}

// ---- 1. the NodePort is the only door ----

// Decision 27 and the manifests: from the laptop, 127.0.0.1:30080 answers;
// the Service's ClusterIP and the scheduler pod's IP do not (they are inside
// the cluster's network), and the mapping is bound to 127.0.0.1 only, so the
// laptop's LAN address does not answer either. From inside the cluster both
// scheduler:8080 (unchanged) and <any node>:30080 answer.
func koTestDoor(t *testing.T) {
	var svc struct {
		Spec struct {
			Type      string `json:"type"`
			ClusterIP string `json:"clusterIP"`
			Ports     []struct {
				Port     int `json:"port"`
				NodePort int `json:"nodePort"`
			} `json:"ports"`
		} `json:"spec"`
	}
	if err := json.Unmarshal([]byte(kc(t, "get", "svc", "scheduler", "-n", koNS, "-o", "json")), &svc); err != nil {
		t.Fatalf("decode svc: %v", err)
	}
	t.Logf("[k8s] svc/scheduler: type=%s clusterIP=%s ports=%+v", svc.Spec.Type, svc.Spec.ClusterIP, svc.Spec.Ports)
	if svc.Spec.Type != "NodePort" || len(svc.Spec.Ports) != 1 || svc.Spec.Ports[0].Port != 8080 || svc.Spec.Ports[0].NodePort != 30080 {
		t.Errorf("decision 27: svc/scheduler should be type NodePort with port 8080 and nodePort 30080; got %+v", svc.Spec)
	}
	out, err := exec.Command("docker", "port", "kgpu-control-plane", "30080/tcp").CombinedOutput()
	t.Logf("[docker] port kgpu-control-plane 30080/tcp -> %q (err=%v)", strings.TrimSpace(string(out)), err)
	if strings.TrimSpace(string(out)) != "127.0.0.1:30080" {
		t.Errorf("deploy/kind/cluster.yaml: the control-plane node's 30080 should be published on 127.0.0.1:30080 only; docker says %q", strings.TrimSpace(string(out)))
	}

	for _, p := range []string{"/livez", "/healthz", "/workers"} {
		code, body, err := koGet(p, nil)
		t.Logf("[laptop] GET %s%s -> %d %.120s (err=%v)", koURL, p, code, body, err)
		if err != nil || code != http.StatusOK {
			t.Errorf("decision 27: %s%s should answer 200 from the laptop; got %d %v", koURL, p, code, err)
		}
	}

	sched, _ := kwPods(t, koNS, "app=scheduler")
	if len(sched) != 1 {
		t.Fatalf("want one scheduler pod, got %v", sched)
	}
	podIP := sched[0].Status.PodIP
	nodes := strings.Fields(kc(t, "get", "nodes", "-o", `jsonpath={range .items[*]}{.status.addresses[?(@.type=="InternalIP")].address}{" "}{end}`))
	for _, addr := range []string{svc.Spec.ClusterIP + ":8080", podIP + ":8080"} {
		err := koDial(addr, 3*time.Second)
		t.Logf("[laptop] TCP connect %s -> err=%v", addr, err)
		if err == nil {
			t.Errorf("system diagram: the shards reach the scheduler through one address; %s (ClusterIP or pod IP) answered from the laptop", addr)
		}
	}
	for _, n := range nodes {
		err := koDial(n+":30080", 2*time.Second)
		t.Logf("[laptop] TCP connect node %s:30080 -> err=%v (observation: Docker Desktop does not route the kind network to the host)", n, err)
	}
	if lan, err := exec.Command("ipconfig", "getifaddr", "en0").Output(); err == nil && strings.TrimSpace(string(lan)) != "" {
		addr := strings.TrimSpace(string(lan)) + ":30080"
		err := koDial(addr, 2*time.Second)
		t.Logf("[laptop] TCP connect LAN address %s -> err=%v", addr, err)
		if err == nil {
			t.Errorf("cluster.yaml listenAddress 127.0.0.1: the NodePort should not be reachable on the LAN address %s", addr)
		}
	} else {
		t.Logf("[laptop] no en0 address; LAN check skipped")
	}

	// Inside the cluster, from a worker pod (the image has python).
	workers, _ := kwPods(t, koNS, "app=worker")
	if len(workers) == 0 {
		t.Fatalf("no worker pod to probe from")
	}
	from := workers[0].Metadata.Name
	urls := []string{"http://scheduler:8080/livez", "http://scheduler.kgpu.svc.cluster.local:8080/healthz"}
	for _, n := range nodes {
		urls = append(urls, "http://"+n+":30080/livez")
	}
	for _, u := range urls {
		so, se, err := kcRun("", "exec", "-n", koNS, from, "-c", "worker", "--", "python", "-c",
			fmt.Sprintf(`import urllib.request as u; r=u.urlopen(%q, timeout=5); print(r.status, r.read()[:60])`, u))
		t.Logf("[in-cluster %s] GET %s -> %s %s (err=%v)", from, u, strings.TrimSpace(so), strings.TrimSpace(se), err)
		if err != nil || !strings.HasPrefix(strings.TrimSpace(so), "200") {
			t.Errorf("decision 27: from inside the cluster %s should answer 200 (in-cluster address unchanged; a NodePort listens on every node)", u)
		}
	}
}

// ---- 2. a round from the laptop ----

// Decision 26 and the step 4 notes: the simulator on the laptop, through the
// NodePort, completes a 6-shard round on 2 workers. Uses bimodal (the
// implementer's check used skewed). No shard pod appears while it runs.
func koTestBimodal(t *testing.T, c *koCollector, scaleStr string, scale float64) {
	koWaitPool(t, koReplicas, 2*time.Minute)
	watch := startKoPodWatch(t)
	before := koLatestRound(t)
	p := startShardsim(t, koSimEnv("bimodal", 6, 4201, 1, scaleStr, 1))
	code := p.wait(t, 3*time.Minute)
	watch.report(t)
	t.Logf("[shards] simulator exited %d after %.1fs", code, p.exitAt.Sub(p.start).Seconds())
	if code != 0 {
		t.Fatalf("simulator exit %d, want 0 (header: exits 0 when its rounds complete):\n%s", code, tail(koSimOut(p), 40))
	}
	ids := koStartedRounds(p)
	if len(ids) != 1 || ids[0] <= before {
		t.Fatalf("want one new round after %d, simulator started %v", before, ids)
	}
	r := koVerifyRound(t, c, p, ids[0], scale, 6, time.Minute)
	for _, b := range r.Builds {
		if b.Attempt != 1 {
			t.Errorf("no failure injected, every build should finish at attempt 1: %s", b)
		}
	}
	if errs := koLines(p, regexp.MustCompile(`level=(ERROR|WARN)`)); len(errs) > 0 {
		t.Errorf("simulator logged %d ERROR/WARN lines in an undisturbed round: %v", len(errs), errs)
	}
}

// ---- 3. the scheduler pod is deleted mid-round ----

// Decision 2 (the scheduler is stateless; Postgres is the only state,
// invariant 1) and the simulator's 60 s client retry: deleting the scheduler
// pod while a round runs leaves the simulator running, the Deployment
// replaces the pod, the NodePort routes to the new one, and the round
// completes, served from Postgres by the new pod. Two rounds, so the second
// starts across or after the gap.
func koTestSchedulerDeleted(t *testing.T, c *koCollector, scaleStr string, scale float64) {
	koWaitPool(t, koReplicas, 2*time.Minute)
	old, _ := kwPods(t, koNS, "app=scheduler")
	if len(old) != 1 {
		t.Fatalf("want one scheduler pod, got %v", old)
	}
	oldName := old[0].Metadata.Name
	t.Logf("[k8s] scheduler pod before: %v", old[0])

	p := startShardsim(t, koSimEnv("uniform", 30, 4301, 2, scaleStr, 1))
	r1 := koWaitStarted(t, p, 1, time.Minute)
	time.Sleep(400 * time.Millisecond)

	// Probe the NodePort every 100 ms to see the gap.
	var mu sync.Mutex
	var fails []time.Time
	var oks int
	stop := make(chan struct{})
	probeDone := make(chan struct{})
	go func() {
		defer close(probeDone)
		cl := &http.Client{Timeout: time.Second}
		for {
			select {
			case <-stop:
				return
			default:
			}
			resp, err := cl.Get(koURL + "/healthz")
			mu.Lock()
			if err != nil || resp.StatusCode != 200 {
				fails = append(fails, time.Now())
			} else {
				oks++
			}
			mu.Unlock()
			if resp != nil {
				resp.Body.Close()
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	deleteAt := time.Now()
	kc(t, "delete", "pod", oldName, "-n", koNS, "--wait=false")
	t.Logf("[kubectl] delete pod %s (graceful) while round %d runs", oldName, r1)

	var newPod kPod
	poll(t, 2*time.Minute, 500*time.Millisecond, "a replacement scheduler pod Ready and the old one gone", func() bool {
		live, term := kwPods(t, koNS, "app=scheduler")
		_, oldExists := getPod(t, koNS, oldName)
		if len(live) == 1 && len(term) == 0 && !oldExists && live[0].Metadata.Name != oldName && live[0].ready() {
			newPod = live[0]
			return true
		}
		return false
	})
	readyAt := time.Now()
	t.Logf("[k8s] replacement scheduler pod Ready %.1fs after the delete: %v", readyAt.Sub(deleteAt).Seconds(), newPod)

	code := p.wait(t, 5*time.Minute)
	close(stop)
	<-probeDone
	mu.Lock()
	if len(fails) > 0 {
		t.Logf("[laptop] NodePort /healthz probe: %d ok, %d failed, failures from %+.2fs to %+.2fs after the delete",
			oks, len(fails), fails[0].Sub(deleteAt).Seconds(), fails[len(fails)-1].Sub(deleteAt).Seconds())
	} else {
		t.Logf("[laptop] NodePort /healthz probe: %d ok, no failed probe: the Service moved from the old pod to the new one without a visible gap", oks)
	}
	mu.Unlock()
	t.Logf("[shards] simulator exited %d after %.1fs", code, p.exitAt.Sub(p.start).Seconds())
	warnRE := regexp.MustCompile(`^time=(\S+) level=(ERROR|WARN)`)
	kinds := map[string]int{}
	var lastWarn time.Time
	for _, l := range koLines(p, regexp.MustCompile(`level=(ERROR|WARN)`)) {
		t.Logf("[shards observation] %s", l)
		kind := "other"
		switch {
		case strings.Contains(l, "level=ERROR"):
			kind = "ERROR"
		case strings.Contains(l, "Client.Timeout"):
			kind = "client timeout"
		case strings.Contains(l, "EOF"):
			kind = "EOF"
		case strings.Contains(l, "connection refused"):
			kind = "refused"
		}
		kinds[kind]++
		if m := warnRE.FindStringSubmatch(l); m != nil {
			if ts, err := time.Parse("2006-01-02T15:04:05.000-07:00", m[1]); err == nil && ts.After(lastWarn) {
				lastWarn = ts
			}
		}
	}
	if !lastWarn.IsZero() {
		t.Logf("[shards] retry lines by kind %v; the last one %+.2fs after the delete (the replacement pod was Ready at %+.2fs)",
			kinds, lastWarn.Sub(deleteAt).Seconds(), readyAt.Sub(deleteAt).Seconds())
	}
	for _, l := range c.schedulerLines("ERROR") {
		if l.Pod == oldName || l.Pod == newPod.Metadata.Name {
			t.Logf("[scheduler observation %s] %s", l.Pod, l.Raw)
		}
	}
	if code != 0 {
		t.Fatalf("simulator exit %d after the scheduler pod was replaced, want 0:\n%s", code, tail(koSimOut(p), 40))
	}
	ids := koStartedRounds(p)
	if len(ids) != 2 {
		t.Fatalf("simulator started rounds %v, want 2", ids)
	}
	for _, id := range ids {
		r := koVerifyRound(t, c, p, id, scale, 30, time.Minute)
		if id == r1 {
			t.Logf("[check] round %d: started %s, scheduler deleted %s, finished %s",
				id, r.Round.StartedAt.Format("15:04:05.000"), deleteAt.UTC().Format("15:04:05.000"), r.Round.FinishedAt.Format("15:04:05.000"))
			if !r.Round.FinishedAt.After(deleteAt) {
				t.Errorf("test validity: round %d finished before the delete; the deletion did not land mid-round", id)
			}
		}
	}
	live, _ := kwPods(t, koNS, "app=scheduler")
	if len(live) != 1 || live[0].Metadata.Name != newPod.Metadata.Name {
		t.Errorf("only the replacement scheduler pod should be serving; pods now %v", live)
	}
	t.Logf("[check] GET /rounds/%d after the run was answered by %s, the only scheduler pod (the round's data lives in Postgres, invariant 1)", r1, newPod.Metadata.Name)
}

// ---- 4. two simulators at once ----

// Decision 26: the shards are clients; nothing in the spec limits how many
// fleets talk to one scheduler. Two simulators on the laptop with different
// seeds run two concurrent rounds; neither's builds or queries leak into the
// other's, and both complete correctly.
func koTestTwoSims(t *testing.T, c *koCollector, scaleStr string, scale float64) {
	koWaitPool(t, koReplicas, 2*time.Minute)
	watch := startKoPodWatch(t)
	a := startShardsim(t, koSimEnv("staggered", 6, 4401, 1, scaleStr, 1))
	b := startShardsim(t, koSimEnv("staggered", 6, 4402, 1, scaleStr, 1))
	ra := koWaitStarted(t, a, 1, time.Minute)
	rb := koWaitStarted(t, b, 1, time.Minute)
	t.Logf("[shards] simulator A started round %d (seed 4401), B round %d (seed 4402)", ra, rb)
	ca, cb := a.wait(t, 3*time.Minute), b.wait(t, 3*time.Minute)
	watch.report(t)
	t.Logf("[shards] A exited %d after %.1fs, B exited %d after %.1fs", ca, a.exitAt.Sub(a.start).Seconds(), cb, b.exitAt.Sub(b.start).Seconds())
	if ca != 0 || cb != 0 {
		t.Fatalf("exit codes A=%d B=%d, want 0 and 0\nA:\n%s\nB:\n%s", ca, cb, tail(koSimOut(a), 20), tail(koSimOut(b), 20))
	}
	if ra == rb {
		t.Fatalf("both simulators report round %d; two fleets must get two rounds", ra)
	}
	// Overlap: the two rounds ran at the same time.
	rA := koVerifyRound(t, c, a, ra, scale, 6, time.Minute)
	rB := koVerifyRound(t, c, b, rb, scale, 6, time.Minute)
	if rA.Round.Seed != 4401 || rB.Round.Seed != 4402 {
		t.Errorf("round rows carry seeds %d and %d, want 4401 and 4402", rA.Round.Seed, rB.Round.Seed)
	}
	overlap := rA.Round.StartedAt.Before(*rB.Round.FinishedAt) && rB.Round.StartedAt.Before(*rA.Round.FinishedAt)
	t.Logf("[check] round %d %s..%s, round %d %s..%s, overlap=%v", ra, rA.Round.StartedAt.Format("15:04:05.000"), rA.Round.FinishedAt.Format("15:04:05.000"),
		rb, rB.Round.StartedAt.Format("15:04:05.000"), rB.Round.FinishedAt.Format("15:04:05.000"), overlap)
	if !overlap {
		t.Errorf("test validity: the two rounds did not overlap")
	}
	for _, x := range []ldRound{rA, rB} {
		for _, bl := range x.Builds {
			if bl.RoundID != x.Round.RoundID || bl.BuildID != fmt.Sprintf("%d:%d", bl.ShardID, x.Round.RoundID) {
				t.Errorf("round %d holds a foreign build %s", x.Round.RoundID, bl)
			}
		}
	}
	// Each simulator logged only its own round.
	for name, pr := range map[string]*simProc{"A": a, "B": b} {
		other := rb
		if name == "B" {
			other = ra
		}
		if ls := koLines(pr, regexp.MustCompile(`round_id=`+strconv.FormatInt(other, 10)+`(\s|$)`)); len(ls) > 0 {
			t.Errorf("simulator %s logged lines about the other fleet's round %d: %v", name, other, ls[:min(3, len(ls))])
		}
	}
}

// ---- 5. no workers, then workers ----

// MIN_WORKERS (shardsim header): "wait for this many live workers before the
// first round; default 1 (0 = don't wait)". Phase 1 decision 19: with zero
// live workers the builds still go on the GPU queue. With the worker
// Deployment at 0, a simulator with MIN_WORKERS=1 starts no round, and one
// with MIN_WORKERS=0 starts its round whose builds sit queued; once the
// workers are scaled back both complete.
func koTestNoWorkers(t *testing.T, c *koCollector, scaleStr string, scale float64) {
	koScaleWorkers(t, 0)
	koWaitPool(t, 0, 2*time.Minute)
	before := koLatestRound(t)

	wait := startShardsim(t, koSimEnv("bimodal", 6, 4501, 1, scaleStr, 1))
	nowait := startShardsim(t, koSimEnv("bimodal", 6, 4502, 1, scaleStr, 0))
	rn := koWaitStarted(t, nowait, 1, time.Minute)
	t.Logf("[shards] MIN_WORKERS=0 simulator started round %d with an empty pool", rn)

	time.Sleep(20 * time.Second)
	if ids := koStartedRounds(wait); len(ids) > 0 {
		t.Errorf("MIN_WORKERS=1 with zero live workers: the simulator should not start a round; it started %v", ids)
	}
	if wait.exited() {
		t.Errorf("MIN_WORKERS=1 simulator exited (%d) while waiting for workers:\n%s", wait.cmd.ProcessState.ExitCode(), tail(koSimOut(wait), 20))
	}
	if latest := koLatestRound(t); latest != rn || rn != before+1 {
		t.Errorf("rounds after 20 s with no workers: latest=%d (before %d, MIN_WORKERS=0 round %d); want only the MIN_WORKERS=0 round", latest, before, rn)
	}
	r, _ := koRound(t, rn)
	states := map[string]int{}
	for _, b := range r.Builds {
		states[b.Placement+"/"+b.State+"/attempt"+strconv.Itoa(int(b.Attempt))]++
	}
	t.Logf("[api] round %d after 20 s with zero workers: %d builds %v, finished_at=%v", rn, len(r.Builds), states, r.Round.FinishedAt)
	if len(r.Builds) != 6 || states["gpu/queued/attempt0"] != 6 {
		t.Errorf("Phase 1 decision 19: with zero live workers every build goes on the GPU queue and stays queued; got %v", states)
	}
	if r.Round.FinishedAt != nil {
		t.Errorf("round %d finished with no workers", rn)
	}
	if nowait.exited() {
		t.Errorf("MIN_WORKERS=0 simulator exited (%d) while its builds were queued", nowait.cmd.ProcessState.ExitCode())
	}

	upAt := time.Now()
	koScaleWorkers(t, koReplicas)
	koWaitPool(t, koReplicas, 3*time.Minute)
	cw, cn := wait.wait(t, 4*time.Minute), nowait.wait(t, 4*time.Minute)
	t.Logf("[shards] after scaling up: MIN_WORKERS=1 exited %d, MIN_WORKERS=0 exited %d", cw, cn)
	if cw != 0 || cn != 0 {
		t.Fatalf("exit codes %d/%d, want 0/0\nwait:\n%s\nnowait:\n%s", cw, cn, tail(koSimOut(wait), 20), tail(koSimOut(nowait), 20))
	}
	rw := koStartedRounds(wait)
	if len(rw) != 1 {
		t.Fatalf("MIN_WORKERS=1 simulator started rounds %v, want one", rw)
	}
	gotW := koVerifyRound(t, c, wait, rw[0], scale, 6, time.Minute)
	if gotW.Round.StartedAt.Before(upAt) {
		t.Errorf("MIN_WORKERS=1 round %d started at %s, before the workers were scaled up at %s", rw[0], gotW.Round.StartedAt, upAt.UTC())
	}
	gotN := koVerifyRound(t, c, nowait, rn, scale, 6, time.Minute)
	for _, b := range gotN.Builds {
		if b.StartedAt != nil && b.StartedAt.Before(upAt) {
			t.Errorf("build %s started at %s before any worker existed", b.BuildID, b.StartedAt)
		}
	}
}

// ---- 6. fifty shards, four workers ----

// Step 4 notes: "a 50-shard, 4-worker round using every card". Here with a
// different scenario from the laptop; every card's worker takes part.
func koTestFifty(t *testing.T, c *koCollector, scaleStr string, scale float64) {
	koScaleWorkers(t, 4)
	t.Cleanup(func() { koScaleWorkers(t, koReplicas); koWaitPool(t, koReplicas, 3*time.Minute) })
	koWaitPool(t, 4, 3*time.Minute)
	wp, _ := kwPods(t, koNS, "app=worker")
	perNode := map[string]int{}
	for _, p := range wp {
		perNode[p.Spec.NodeName]++
	}
	t.Logf("[k8s] 4 workers by node: %v", perNode)
	watch := startKoPodWatch(t)
	p := startShardsim(t, koSimEnv("downstream_heavy", 50, 4601, 1, scaleStr, 4))
	code := p.wait(t, 5*time.Minute)
	watch.report(t)
	t.Logf("[shards] simulator exited %d after %.1fs", code, p.exitAt.Sub(p.start).Seconds())
	if code != 0 {
		t.Fatalf("simulator exit %d:\n%s", code, tail(koSimOut(p), 40))
	}
	ids := koStartedRounds(p)
	if len(ids) != 1 {
		t.Fatalf("rounds %v, want one", ids)
	}
	r := koVerifyRound(t, c, p, ids[0], scale, 50, 2*time.Minute)
	comps := c.completions()
	by := map[string]int{}
	for _, b := range r.Builds {
		for _, l := range comps[b.BuildID] {
			by[l.Pod]++
		}
	}
	t.Logf("[check] completions by worker pod: %v", by)
	for _, w := range wp {
		if by[w.Metadata.Name] == 0 {
			t.Errorf("worker %s (node %s) completed no build of the 50; every card should be in use", w.Metadata.Name, w.Spec.NodeName)
		}
	}
}

// ---- 7. the simulator reaches only the NodePort ----

// System diagram: one address. A simulator pointed at the Service's ClusterIP
// or the scheduler pod's IP cannot reach the scheduler from the laptop: it
// starts no round and nothing new appears in the scheduler.
func koTestOtherAddrs(t *testing.T, scaleStr string) {
	koWaitPool(t, koReplicas, 2*time.Minute)
	clusterIP := strings.TrimSpace(kc(t, "get", "svc", "scheduler", "-n", koNS, "-o", "jsonpath={.spec.clusterIP}"))
	sched, _ := kwPods(t, koNS, "app=scheduler")
	if len(sched) != 1 {
		t.Fatalf("want one scheduler pod, got %v", sched)
	}
	before := koLatestRound(t)
	urls := []string{"http://" + clusterIP + ":8080", "http://" + sched[0].Status.PodIP + ":8080"}
	var ps []*simProc
	for i, u := range urls {
		env := koSimEnv("bimodal", 6, int64(4701+i), 1, scaleStr, 0) // MIN_WORKERS=0: go straight to creating a round
		env["SCHEDULER_URL"] = u
		ps = append(ps, startShardsim(t, env))
	}
	deadline := time.Now().Add(150 * time.Second)
	for time.Now().Before(deadline) && !(ps[0].exited() && ps[1].exited()) {
		time.Sleep(time.Second)
	}
	for i, p := range ps {
		lines := strings.Split(strings.TrimSpace(koSimOut(p)), "\n")
		if p.exited() {
			t.Logf("[shards %s] exited %d after %.1fs; last lines:\n%s", urls[i], p.cmd.ProcessState.ExitCode(), p.exitAt.Sub(p.start).Seconds(), tail(koSimOut(p), 6))
			if p.cmd.ProcessState.ExitCode() == 0 {
				t.Errorf("simulator against %s exited 0; it cannot have completed a round there", urls[i])
			}
		} else {
			t.Logf("[shards %s] still running after 150 s (%d lines); last lines:\n%s", urls[i], len(lines), tail(koSimOut(p), 6))
		}
		if ids := koStartedRounds(p); len(ids) > 0 {
			t.Errorf("simulator against %s started rounds %v; that address should be unreachable from the laptop", urls[i], ids)
		}
	}
	if latest := koLatestRound(t); latest != before {
		t.Errorf("rounds through %s: before %d, after %d; nothing should have reached the scheduler", koURL, before, latest)
	}
}
