package crosscheck

// Cross-check of Phase 2 step 3b on the kind cluster `kgpu`: GPU cards as a
// countable node resource. Spec: docs/design/phase2-kubernetes.md decisions
// 22 to 25 and the "Step 3b" implementation notes; deploy/kind/cluster.yaml;
// deploy/k8s/worker.yaml; docs/bugs.md entry 9.
//
// The promises under test: two GPU nodes (label kgpu.io/gpu=true, taint
// kgpu.io/gpu=true:NoSchedule) each advertise kgpu.io/gpu-card: 2; each worker
// pod requests one card (request = limit); Kubernetes caps workers per node at
// the card count and reports "Insufficient kgpu.io/gpu-card" beyond it;
// `replicas` picks how many cards are in use; worker registration waits for
// the schema instead of crashing (decision 25).
//
// Cards are cluster-wide, so the implementer's workers in `kgpu` are scaled to
// 0 for the run (replica count recorded first) and restored in a cleanup. The
// node-level actions (capacity patch, drain) run last and are undone in
// cleanups no matter what. The cross-check's namespace is deleted at the end.
//
// Run: KGPU_KIND=1 scripts/test-db.sh -count=1 -v -run KindCards -timeout 40m ./scheduler/crosscheck/

import (
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestKindCards(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set: kind tests need the kgpu cluster and take minutes")
	}
	if _, err := exec.LookPath("kubectl"); err != nil {
		t.Fatalf("kubectl not on PATH: %v", err)
	}
	gpus := gpuNodes(t)
	if len(gpus) != 2 {
		t.Fatalf("want two nodes labelled %s=true (decision 24), got %v", kGPUKey, gpus)
	}
	t.Logf("[setup] GPU nodes %v", gpus)

	// 1. The implementer's workers: record, scale to 0, restore last.
	origReplicas, err := strconv.Atoi(strings.TrimSpace(kc(t, "get", "deployment/worker", "-n", kOrigNS, "-o", "jsonpath={.spec.replicas}")))
	if err != nil {
		t.Fatalf("read %s worker replicas: %v", kOrigNS, err)
	}
	t.Logf("[setup] %s worker Deployment has replicas=%d; pods %v", kOrigNS, origReplicas, ccPods(t, kOrigNS, "app=worker"))
	t.Cleanup(func() {
		so, se, err := kcRun("", "scale", "deployment/worker", "-n", kOrigNS, fmt.Sprintf("--replicas=%d", origReplicas))
		t.Logf("[cleanup] restore %s workers to %d: err=%v %s", kOrigNS, origReplicas, err, strings.TrimSpace(so+se))
		ps := waitWorkers(t, kOrigNS, origReplicas, 4*time.Minute)
		for _, p := range ps {
			if p.Status.Phase != "Running" {
				t.Errorf("%s worker %v not Running after restore", kOrigNS, p)
			}
		}
		t.Logf("[cleanup] %s workers back: %v", kOrigNS, ps)
	})
	// 2. The cross-check namespace: deleted before the workers are restored
	// (cleanups run last-registered first), so its cards are free again.
	deleteXcheckNS()
	t.Cleanup(deleteXcheckNS)
	// 3. Node state: uncordoned with capacity 2, whatever happened.
	t.Cleanup(func() {
		for _, n := range gpus {
			so, se, err := kcRun("", "uncordon", n)
			t.Logf("[cleanup] uncordon %s: err=%v %s", n, err, strings.TrimSpace(so+se))
			out, err := patchNodeCapacity(n, "2")
			t.Logf("[cleanup] capacity %s=2 on %s: err=%v %s", kcCard, n, err, out)
		}
		deadline := time.Now().Add(60 * time.Second)
		for _, n := range gpus {
			for {
				c, a := nodeCards(t, n)
				if (c == "2" && a == "2") || time.Now().After(deadline) {
					t.Logf("[cleanup] %s: capacity=%s allocatable=%s", n, c, a)
					break
				}
				time.Sleep(time.Second)
			}
		}
	})

	scaleWorkers(t, kOrigNS, 0)
	poll(t, 2*time.Minute, time.Second, kOrigNS+" worker pods gone", func() bool {
		return len(ccPods(t, kOrigNS, "app=worker")) == 0
	})
	t.Logf("[setup] %s workers scaled to 0; cards allocated per node now %v", kOrigNS, cardsAllocated(t))

	var api *kwAPI
	t.Run("ReverseOrderStartWaitsForSchema", func(t *testing.T) { api = testReverseStart(t) })
	if api == nil {
		t.Fatal("stack did not come up; remaining subtests need it")
	}
	// The parent owns the port-forward, so it outlives each subtest.
	api.pf = forward(t, "svc/scheduler", kPortA)
	t.Run("NodeAndPodCardFacts", func(t *testing.T) { testCardFacts(t, gpus) })
	t.Run("FourWorkersFillCardsFifthPending", func(t *testing.T) { testFillCards(t, api, gpus) })
	t.Run("NoCardPodIgnoresCap", func(t *testing.T) { testNoCardPod(t, gpus) })
	t.Run("KilledWorkerKeepsItsCard", func(t *testing.T) { testKillKeepsCard(t, api) })
	t.Run("TwoCardPodFragmentation", func(t *testing.T) { testFragmentation(t, api, gpus) })
	t.Run("CapacityPatchedDownUnderLoad", func(t *testing.T) { testCapacityDown(t, api, gpus) })
	t.Run("DrainOneGPUNodeHalvesPool", func(t *testing.T) { testDrainHalves(t, api, gpus) })
}

// ---- 1. decision 25: workers applied first, before Postgres and scheduler ----

// The worker Deployment goes in first (with only the Secret and ConfigMaps it
// references), then Postgres, then the scheduler. Kubernetes has no ordering
// between them (bug 9). Promise: the workers never crash-loop (restart count
// 0 throughout), log that they wait for the schema once Postgres is up but
// empty, are not Ready until the scheduler has migrated, then register.
func testReverseStart(t *testing.T) *kwAPI {
	rendered := renderXcheckPerl(t)
	stageA := strings.Join([]string{
		docsSelect(rendered, "Namespace", ""),
		docsSelect(rendered, "Secret", ""),
		docsSelect(rendered, "ConfigMap", "worker"),
		docsSelect(rendered, "Deployment", "worker"),
	}, "\n---\n")
	start := time.Now()
	kcApply(t, stageA)
	t.Logf("[stage A] only the worker Deployment (+ its Secret and ConfigMap) exists in %s: no Postgres, no scheduler", kNS)

	checkNoRestarts := func(stage string) []ccPod {
		bound, unbound := myWorkers(t)
		for _, p := range bound {
			if p.restarts() != 0 {
				t.Errorf("[%s] %s restarted %d time(s): a crash loop (decision 25 / bug 9); last state %+v",
					stage, p.Metadata.Name, p.restarts(), p.Status.ContainerStatuses)
			}
			if p.ready() {
				t.Errorf("[%s] %s is Ready before the schema exists; readiness must stay false (decision 25)", stage, p.Metadata.Name)
			}
		}
		if len(unbound) > 0 {
			t.Errorf("[%s] unbound workers %v (4 cards should be free)", stage, unbound)
		}
		return bound
	}

	// Wait for both containers to start, then hold past the liveness budget
	// (initialDelay 5 s + 3 x 10 s period).
	poll(t, 2*time.Minute, time.Second, "2 worker containers running", func() bool {
		b, _ := myWorkers(t)
		if len(b) != 2 {
			return false
		}
		for _, p := range b {
			if p.Status.Phase != "Running" {
				return false
			}
		}
		return true
	})
	for i := 0; i < 9; i++ {
		ps := checkNoRestarts("stage A")
		if i%3 == 0 {
			t.Logf("[stage A t=%.0fs] %v", time.Since(start).Seconds(), ps)
		}
		time.Sleep(5 * time.Second)
	}
	bound, _ := myWorkers(t)
	for _, p := range bound {
		all := strings.TrimSpace(podLogs(t, p.Metadata.Name, false))
		n := 0
		if all != "" {
			n = len(strings.Split(all, "\n"))
		}
		t.Logf("[stage A] %s wrote %d log line(s) in ~45 s with no Postgres Service at all (observation: the spec promises no log here)", p.Metadata.Name, n)
		lines := strings.Split(strings.TrimSpace(podLogs(t, p.Metadata.Name, false)), "\n")
		if len(lines) > 3 {
			lines = lines[len(lines)-3:]
		}
		t.Logf("[stage A] %s last log lines (no database at all):\n  %s", p.Metadata.Name, strings.Join(lines, "\n  "))
	}

	// Stage B: Postgres, empty database, still no scheduler to migrate it.
	kcApply(t, strings.Join([]string{docsSelect(rendered, "Service", "postgres"), docsSelect(rendered, "StatefulSet", "")}, "\n---\n"))
	waitPodsReady(t, kNS, "app=postgres", 1, 4*time.Minute)
	t.Logf("[stage B] Postgres Ready at t=%.0fs; tables in public: %q", time.Since(start).Seconds(),
		psql(t, "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'"))
	pb := forwardProbe(t, bound[0].Metadata.Name)
	var h503, l200, samples int
	for i := 0; i < 10; i++ {
		checkNoRestarts("stage B")
		hc, _ := pb.probe("/healthz")
		lc, _ := pb.probe("/livez")
		samples++
		if hc == 503 {
			h503++
		}
		if lc == 200 {
			l200++
		}
		time.Sleep(5 * time.Second)
	}
	pb.pf.stop()
	t.Logf("[stage B] %s probes over %d samples (50 s, past the liveness budget): /healthz 503 on %d, /livez 200 on %d",
		bound[0].Metadata.Name, samples, h503, l200)
	if h503 != samples || l200 != samples {
		t.Errorf("stage B: want /healthz 503 and /livez 200 on every sample while the schema is missing (decision 25)")
	}
	waitRe := regexp.MustCompile(`schema not ready`)
	for _, p := range bound {
		ls := grepLines(podLogs(t, p.Metadata.Name, false), waitRe)
		t.Logf("[stage B] %s: %d log line(s) matching %q; first: %.200s", p.Metadata.Name, len(ls), waitRe, firstOr(ls))
		if len(ls) == 0 {
			t.Errorf("%s never logged that it is waiting for the schema (bug 9 fix)", p.Metadata.Name)
		}
	}

	// Stage C: the scheduler, which migrates; workers must register.
	kcApply(t, strings.Join([]string{docsSelect(rendered, "Service", "scheduler"), docsSelect(rendered, "ConfigMap", "scheduler"), docsSelect(rendered, "Deployment", "scheduler")}, "\n---\n"))
	waitPodsReady(t, kNS, "app=scheduler", 1, 4*time.Minute)
	tSched := time.Now()
	ps, _ := waitBound(t, 2, 0, 90*time.Second)
	t.Logf("[stage C] both workers Ready %.1fs after the scheduler became Ready", time.Since(tSched).Seconds())
	for _, p := range ps {
		if p.restarts() != 0 {
			t.Errorf("%s restart count %d at the end, want 0 (decision 25)", p.Metadata.Name, p.restarts())
		}
	}
	local := &kwAPI{pf: forward(t, "svc/scheduler", kPortA)}
	waitPool(t, local, names(ps), 30*time.Second)
	local.pf.stop()
	t.Logf("[check] reverse-order start: restarts 0 on %v throughout; not Ready until migrated; registered under their pod names", names(ps))
	return &kwAPI{}
}

func firstOr(ls []string) string {
	if len(ls) == 0 {
		return "<none>"
	}
	return ls[0]
}

// ---- 2. node-level facts ----

// Decisions 22 and 24: both GPU nodes advertise capacity = allocatable = 2,
// no other node has the resource; every worker pod's card request equals its
// limit (1); nothing but workers consumes cards anywhere in the cluster.
func testCardFacts(t *testing.T, gpus []string) {
	for _, n := range nodes(t) {
		c, a := nodeCards(t, n.Metadata.Name)
		isGPU := n.Metadata.Labels[kGPUKey] == "true"
		var taints []string
		for _, tn := range n.Spec.Taints {
			taints = append(taints, fmt.Sprintf("%s=%s:%s", tn.Key, tn.Value, tn.Effect))
		}
		t.Logf("[node] %s gpu-label=%v taints=%v capacity %s=%q allocatable=%q", n.Metadata.Name, isGPU, taints, kcCard, c, a)
		if isGPU {
			if c != "2" || a != "2" {
				t.Errorf("GPU node %s: capacity %q allocatable %q, want 2 and 2", n.Metadata.Name, c, a)
			}
			if !contains(taints, kGPUKey+"=true:NoSchedule") {
				t.Errorf("GPU node %s lacks the taint %s=true:NoSchedule", n.Metadata.Name, kGPUKey)
			}
		} else if c != "" || a != "" {
			t.Errorf("non-GPU node %s advertises %s (capacity %q allocatable %q)", n.Metadata.Name, kcCard, c, a)
		}
	}
	bound, _ := myWorkers(t)
	checkWorkerCardSpec(t, bound)
	for _, p := range ccPods(t, "", "") {
		if p.cards() > 0 && !(p.Metadata.Labels["app"] == "worker" && (p.Metadata.Namespace == kNS || p.Metadata.Namespace == kOrigNS)) {
			t.Errorf("non-worker pod %s/%s requests %d card(s)", p.Metadata.Namespace, p.Metadata.Name, p.cards())
		}
	}
	alloc := cardsAllocated(t)
	per := byNode(bound)
	for _, g := range gpus {
		t.Logf("[cards] %s: allocated %d, my workers there %v", g, alloc[g], per[g])
		if alloc[g] != len(per[g]) {
			t.Errorf("%s: %d cards allocated but %d workers on it", g, alloc[g], len(per[g]))
		}
	}
	t.Logf("[observe] default 2 replicas placed %v (spec: one per node is a spreading preference, not a guarantee)", per)
}

// ---- 3. replicas fill the cards; the fifth is refused ----

// Decision 22: scaled to 4, every card is used, two per node, the pool shows
// 4; scaled to 5, the fifth stays Pending with "Insufficient kgpu.io/gpu-card"
// and the pool stays 4. Left at 5 for the next two tests.
func testFillCards(t *testing.T, api *kwAPI, gpus []string) {
	scaleWorkers(t, kNS, 4)
	bound, _ := waitBound(t, 4, 0, 3*time.Minute)
	checkWorkerCardSpec(t, bound)
	alloc := cardsAllocated(t)
	per := byNode(bound)
	for _, g := range gpus {
		t.Logf("[cards] %s: allocated %d/2, workers %v", g, alloc[g], per[g])
		if alloc[g] != 2 || len(per[g]) != 2 {
			t.Errorf("%s: %d cards allocated, %d workers; want 2 and 2 with all 4 cards in use", g, alloc[g], len(per[g]))
		}
	}
	waitPool(t, api, names(bound), 45*time.Second)

	scaleWorkers(t, kNS, 5)
	var pend ccPod
	poll(t, 60*time.Second, time.Second, "a fifth, unbound worker pod", func() bool {
		_, ub := myWorkers(t)
		if len(ub) == 1 {
			pend = ub[0]
			return true
		}
		return false
	})
	msg := waitUnschedulable(t, pend.Metadata.Name, kcInsuffRe, 60*time.Second)
	t.Logf("[k8s] fifth worker %s PodScheduled=False: %q", pend.Metadata.Name, msg)
	for i := 0; i < 10; i++ {
		b, ub := myWorkers(t)
		if len(b) != 4 || len(ub) != 1 {
			t.Errorf("with 5 replicas: %d bound, %d unbound; want 4 and 1", len(b), len(ub))
		}
		time.Sleep(time.Second)
	}
	w := ccPool(t, api)
	t.Logf("[api] pool with 5 replicas: live_workers=%d ids=%v", w.Pool.LiveWorkers, w.liveIDs())
	if w.Pool.LiveWorkers != 4 {
		t.Errorf("pool %d with one replica Pending, want 4", w.Pool.LiveWorkers)
	}
}

// ---- 4. a pod with no card request is not capped ----

// With every card in use and a worker Pending on cards, a pause pod with the
// worker's toleration and selector but NO card request still lands on a GPU
// node: the resource caps only pods that request it. The spec states the
// toleration is the access control and "only workers there" is a convention
// (system diagram text); decision 22 promises a cap on card requesters, not on
// pods. A one-card non-worker pod, by contrast, is refused like a worker.
func testNoCardPod(t *testing.T, gpus []string) {
	kcApply(t, cardPausePod("xc-nocard", 0))
	defer deletePod(t, "xc-nocard")
	var p ccPod
	poll(t, 90*time.Second, time.Second, "xc-nocard Running", func() bool {
		var ok bool
		p, ok = ccPodByName(t, kNS, "xc-nocard")
		return ok && p.Status.Phase == "Running"
	})
	t.Logf("[k8s] xc-nocard (toleration + selector, no card) %v while all 4 cards are allocated %v", p, cardsAllocated(t))
	if !contains(gpus, p.Spec.NodeName) {
		t.Errorf("xc-nocard landed on %s, not a GPU node", p.Spec.NodeName)
	}
	_, ub := myWorkers(t)
	if len(ub) != 1 {
		t.Errorf("fifth worker should still be Pending, unbound=%v", ub)
	}
	alloc := cardsAllocated(t)
	for _, g := range gpus {
		if alloc[g] != 2 {
			t.Errorf("%s allocated %d after the no-card pod, want 2", g, alloc[g])
		}
	}

	kcApply(t, cardPausePod("xc-onecard", 1))
	defer deletePod(t, "xc-onecard")
	msg := waitUnschedulable(t, "xc-onecard", kcInsuffRe, 60*time.Second)
	t.Logf("[k8s] xc-onecard (a non-worker pod requesting 1 card) Pending: %q. Nothing but the card count stops a non-worker pod from taking a card when one is free.", msg)
}

// ---- 5. a crashed worker keeps its card ----

// With 4 bound workers and a Pending fifth: kill one worker's python in the
// container (decision 19: the crash test). The kubelet restarts the container
// in place: same pod, same node, restart count +1, exit 137. The node's
// allocated card count never drops below 2, and the Pending fifth never gets
// the card. Then scale back to 4: the Pending pod is the one removed.
func testKillKeepsCard(t *testing.T, api *kwAPI) {
	bound, ub := myWorkers(t)
	if len(bound) != 4 || len(ub) != 1 {
		t.Fatalf("precondition: want 4 bound + 1 Pending, got %v / %v", bound, ub)
	}
	victim := bound[0]
	node := victim.Spec.NodeName
	pendName := ub[0].Metadata.Name
	r0 := victim.restarts()
	t.Logf("[setup] victim %s on %s (restarts %d); Pending %s; allocated %v", victim.Metadata.Name, node, r0, pendName, cardsAllocated(t))
	signalPython(t, victim.Metadata.Name, "KILL")
	start := time.Now()
	var minAlloc, samples = 99, 0
	restartedAt := time.Duration(-1)
	deadline := start.Add(90 * time.Second)
	for time.Now().Before(deadline) {
		alloc := cardsAllocated(t)[node]
		samples++
		if alloc < minAlloc {
			minAlloc = alloc
		}
		if p, ok := ccPodByName(t, kNS, pendName); !ok || p.Spec.NodeName != "" {
			t.Errorf("t=%.1fs: the Pending fifth %s got bound (%v): it took the crashed worker's card", time.Since(start).Seconds(), pendName, p)
			break
		}
		v, ok := ccPodByName(t, kNS, victim.Metadata.Name)
		if !ok || v.Metadata.UID != victim.Metadata.UID {
			t.Errorf("victim pod %s disappeared or was replaced; want an in-place container restart", victim.Metadata.Name)
			break
		}
		if restartedAt < 0 && v.restarts() == r0+1 && v.ready() {
			restartedAt = time.Since(start)
			t.Logf("[k8s] t=%.1fs victim restarted in place and Ready: %v lastState=%+v", restartedAt.Seconds(), v, v.Status.ContainerStatuses[0].LastState.Terminated)
			if term := v.Status.ContainerStatuses[0].LastState.Terminated; term == nil || term.ExitCode != 137 {
				t.Errorf("victim's last termination %+v, want exit 137", term)
			}
			deadline = time.Now().Add(10 * time.Second) // keep watching a little longer
		}
		time.Sleep(300 * time.Millisecond)
	}
	t.Logf("[check] %d samples on %s over %.1fs: minimum allocated cards %d (want 2 throughout)", samples, node, time.Since(start).Seconds(), minAlloc)
	if minAlloc != 2 {
		t.Errorf("allocated cards on %s dropped to %d during the restart", node, minAlloc)
	}
	if restartedAt < 0 {
		t.Errorf("victim never came back Ready with restart count %d", r0+1)
	}
	waitPool(t, api, names(bound), 60*time.Second)

	scaleWorkers(t, kNS, 4)
	after, _ := waitBound(t, 4, 0, 60*time.Second)
	if strings.Join(names(after), ",") != strings.Join(names(bound), ",") {
		t.Errorf("scale 5->4 removed a running worker: before %v after %v", names(bound), names(after))
	} else {
		t.Logf("[check] scale 5->4 removed the Pending %s; the 4 card holders are unchanged", pendName)
	}
}

// ---- 6. fragmentation: a 2-card pod ----

// 3 workers leave 1 free card on one node: a pod requesting 2 cards must stay
// Pending with Insufficient. At 2 workers (1+1) there are 2 free cards in
// total but 1 per node: still Pending. At 1 worker one node has 2 free: the
// pod lands there. Ends with the pod deleted and 4 workers.
func testFragmentation(t *testing.T, api *kwAPI, gpus []string) {
	scaleWorkers(t, kNS, 3)
	b3, _ := waitBound(t, 3, 0, 2*time.Minute)
	t.Logf("[cards] 3 workers: allocated %v", cardsAllocated(t))
	kcApply(t, cardPausePod("xc-twocards", 2))
	defer deletePod(t, "xc-twocards")
	msg := waitUnschedulable(t, "xc-twocards", kcInsuffRe, 60*time.Second)
	t.Logf("[k8s] 3 workers (1 card free on one node): xc-twocards Pending: %q", msg)
	_ = b3

	scaleWorkers(t, kNS, 2)
	b2, _ := waitBound(t, 2, 0, 2*time.Minute)
	per := byNode(b2)
	time.Sleep(5 * time.Second)
	p, _ := ccPodByName(t, kNS, "xc-twocards")
	t.Logf("[cards] 2 workers %v: allocated %v; xc-twocards %v", per, cardsAllocated(t), p)
	if len(per) == 2 {
		_, _, m := p.cond("PodScheduled")
		if p.Spec.NodeName != "" {
			t.Errorf("1 free card per node, yet the 2-card pod was bound to %s", p.Spec.NodeName)
		} else {
			t.Logf("[check] 2 cards free in total, 1 per node: the 2-card pod stays Pending (%q); cards do not pool across nodes", m)
		}
	} else {
		t.Logf("[observe] scale-down left both workers on one node %v; skipping the 1+1 case", per)
	}

	scaleWorkers(t, kNS, 1)
	b1, _ := waitBound(t, 1, 0, 2*time.Minute)
	empty := ""
	for _, g := range gpus {
		if g != b1[0].Spec.NodeName {
			empty = g
		}
	}
	poll(t, 60*time.Second, time.Second, "xc-twocards Running", func() bool {
		p, _ = ccPodByName(t, kNS, "xc-twocards")
		return p.Status.Phase == "Running"
	})
	alloc := cardsAllocated(t)
	t.Logf("[k8s] 1 worker on %s: xc-twocards %v; allocated %v", b1[0].Spec.NodeName, p, alloc)
	if p.Spec.NodeName != empty || alloc[empty] != 2 {
		t.Errorf("xc-twocards on %s with %d allocated there, want the empty node %s with 2", p.Spec.NodeName, alloc[empty], empty)
	}
	deletePod(t, "xc-twocards")
	scaleWorkers(t, kNS, 4)
	b4, _ := waitBound(t, 4, 0, 3*time.Minute)
	waitPool(t, api, names(b4), 60*time.Second)
}

// ---- 7. capacity patched down under load ----

// Observation, not a stated promise: patch one GPU node's card capacity from
// 2 to 1 while 2 workers run on it. Expected Kubernetes behaviour: nothing is
// evicted (extended resources are checked only at scheduling time), the node
// is over-allocated (2 allocated, 1 allocatable), and a replacement for a
// worker deleted there cannot land anywhere until the capacity is restored.
func testCapacityDown(t *testing.T, api *kwAPI, gpus []string) {
	bound, _ := waitBound(t, 4, 0, time.Minute)
	node := gpus[0]
	onNode := byNode(bound)[node]
	if len(onNode) != 2 {
		t.Fatalf("precondition: 2 workers on %s, got %v", node, onNode)
	}
	out, err := patchNodeCapacity(node, "1")
	if err != nil {
		t.Fatalf("patch capacity: %v %s", err, out)
	}
	t.Logf("[kubectl] patch node %s status capacity %s=1: %s", node, kcCard, out)
	poll(t, 60*time.Second, time.Second, node+" allocatable 1", func() bool {
		c, a := nodeCards(t, node)
		return c == "1" && a == "1"
	})
	uids := map[string]string{}
	for _, p := range bound {
		uids[p.Metadata.Name] = p.Metadata.UID
	}
	for i := 0; i < 20; i++ {
		b, _ := myWorkers(t)
		for _, p := range b {
			if uids[p.Metadata.Name] != p.Metadata.UID || p.Status.Phase != "Running" {
				t.Errorf("worker %v replaced or not Running after the capacity patch (expected no eviction)", p)
			}
		}
		if len(b) != 4 {
			t.Errorf("t=%ds after the capacity patch: %d bound workers, want 4 (no eviction)", i, len(b))
			break
		}
		time.Sleep(time.Second)
	}
	c, a := nodeCards(t, node)
	t.Logf("[observe] 20 s after the patch: %s capacity=%s allocatable=%s allocated=%d; workers unchanged %v; pool %d",
		node, c, a, cardsAllocated(t)[node], byNode(bound)[node], ccPool(t, api).Pool.LiveWorkers)

	victim := onNode[0]
	kc(t, "delete", "pod", victim, "-n", kNS, "--wait")
	t.Logf("[kubectl] delete pod %s (graceful) on the over-allocated node", victim)
	var repl ccPod
	poll(t, 60*time.Second, time.Second, "a replacement worker unbound", func() bool {
		_, ub := myWorkers(t)
		if len(ub) == 1 {
			repl = ub[0]
			return true
		}
		return false
	})
	msg := waitUnschedulable(t, repl.Metadata.Name, kcInsuffRe, 60*time.Second)
	t.Logf("[k8s] replacement %s Pending: %q; allocated %v", repl.Metadata.Name, msg, cardsAllocated(t))

	out, err = patchNodeCapacity(node, "2")
	if err != nil {
		t.Fatalf("restore capacity: %v %s", err, out)
	}
	t.Logf("[kubectl] restore %s capacity %s=2", node, kcCard)
	b4, _ := waitBound(t, 4, 0, 2*time.Minute)
	if p, ok := ccPodByName(t, kNS, repl.Metadata.Name); ok {
		t.Logf("[k8s] replacement after restore: %v", p)
		if p.Spec.NodeName != node {
			t.Errorf("replacement bound to %s, want %s", p.Spec.NodeName, node)
		}
	}
	c, a = nodeCards(t, node)
	if c != "2" || a != "2" {
		t.Errorf("%s capacity %s allocatable %s after restore, want 2/2", node, c, a)
	}
	waitPool(t, api, names(b4), 60*time.Second)
}

// ---- 8. drain one GPU node: the shrinking pool ----

// Decision 24: "a drain halves the pool instead of emptying it". With 4
// workers each holding a long build, drain one GPU node: exactly its two
// workers are evicted (their builds released to queued), the other two keep
// their pods and keep building to done at attempt 1, the pool drops to 2, the
// replacements stay Pending until uncordon, and every build ends done.
func testDrainHalves(t *testing.T, api *kwAPI, gpus []string) {
	bound, _ := waitBound(t, 4, 0, time.Minute)
	per := byNode(bound)
	drain, keep := gpus[1], gpus[0]
	if len(per[drain]) != 2 || len(per[keep]) != 2 {
		t.Fatalf("precondition: 2 workers per node, got %v", per)
	}
	keepUID := map[string]string{}
	for _, p := range bound {
		if p.Spec.NodeName == keep {
			keepUID[p.Metadata.Name] = p.Metadata.UID
		}
	}
	round := api.round(t, "xcheck-cards-drain", 4)
	var ids []string
	for s := 0; s < 4; s++ {
		ids = append(ids, api.submit(t, round, s, kcLongN).BuildID)
	}
	owner := map[string]string{}
	poll(t, 60*time.Second, 300*time.Millisecond, "4 builds leased by 4 distinct workers", func() bool {
		seen := map[string]bool{}
		for _, id := range ids {
			b := api.build(t, id)
			if b.State != "leased" {
				return false
			}
			owner[id] = b.owner()
			seen[b.owner()] = true
		}
		return len(seen) == 4
	})
	t.Logf("[api] leases: %v", owner)
	onDrain := map[string]bool{}
	for _, n := range per[drain] {
		onDrain[n] = true
	}

	t0 := time.Now()
	so, se, err := kcRun("", "drain", drain, "--ignore-daemonsets", "--delete-emptydir-data", "--timeout=120s")
	t.Logf("[kubectl] drain %s took %.1fs: err=%v\n%s%s", drain, time.Since(t0).Seconds(), err, so, se)
	if err != nil {
		t.Fatalf("drain failed")
	}
	evicted := regexp.MustCompile(`pod/(worker-\S+) evicted`).FindAllStringSubmatch(so+se, -1)
	var ev []string
	for _, m := range evicted {
		ev = append(ev, m[1])
	}
	t.Logf("[k8s] evicted workers: %v (on %s: %v)", ev, drain, per[drain])
	for _, e := range ev {
		if !onDrain[e] {
			t.Errorf("drain evicted %s, which was not on %s", e, drain)
		}
	}
	if len(ev) != 2 {
		t.Errorf("drain evicted %d workers, want exactly the 2 on %s", len(ev), drain)
	}

	// The pool halves; the replacements stay Pending.
	var survivors []string
	for n := range keepUID {
		survivors = append(survivors, n)
	}
	waitPool(t, api, survivors, 30*time.Second)
	var ub []ccPod
	poll(t, 60*time.Second, time.Second, "2 replacement workers unbound", func() bool {
		_, ub = myWorkers(t)
		return len(ub) == 2
	})
	for _, p := range ub {
		_, _, m := p.cond("PodScheduled")
		t.Logf("[k8s] replacement %s Pending: %q", p.Metadata.Name, m)
		if !regexp.MustCompile(kcInsuffRe).MatchString(m) {
			t.Errorf("replacement %s Pending without citing the card resource: %q", p.Metadata.Name, m)
		}
	}
	// The released builds and the surviving ones.
	for _, id := range ids {
		b := api.build(t, id)
		t.Logf("[api] t=%.1fs after drain: %s (was leased by %s)", time.Since(t0).Seconds(), b, owner[id])
		if onDrain[owner[id]] {
			if b.State == "leased" && b.owner() == owner[id] {
				t.Errorf("build %s still leased by the evicted %s", id, owner[id])
			}
		} else if b.State != "leased" || b.owner() != owner[id] || b.Attempt != 1 {
			t.Errorf("build %s on a surviving worker was disturbed: %s", id, b)
		}
	}
	// Hold Pending for a while; survivors unchanged.
	for i := 0; i < 10; i++ {
		b, u := myWorkers(t)
		if len(u) != 2 || len(b) != 2 {
			t.Errorf("while cordoned: bound %v unbound %v; want 2 and 2", names(b), names(u))
			break
		}
		for _, p := range b {
			if keepUID[p.Metadata.Name] != p.Metadata.UID || p.restarts() != 0 {
				t.Errorf("survivor %v changed", p)
			}
		}
		time.Sleep(time.Second)
	}

	so, se, err = kcRun("", "uncordon", drain)
	t.Logf("[kubectl] uncordon %s: err=%v %s", drain, err, strings.TrimSpace(so+se))
	b4, _ := waitBound(t, 4, 0, 2*time.Minute)
	t.Logf("[k8s] after uncordon: %v", byNode(b4))
	waitPool(t, api, names(b4), 60*time.Second)

	for _, id := range ids {
		b := api.waitBuild(t, id, 4*time.Minute, "build done", isDone)
		want := int32(1)
		if onDrain[owner[id]] {
			want = 2
		}
		t.Logf("[check] %s done at attempt %d (first leased by %s, evicted=%v)", id, b.Attempt, owner[id], onDrain[owner[id]])
		if b.Attempt != want {
			t.Errorf("build %s done at attempt %d, want %d", id, b.Attempt, want)
		}
	}
}
