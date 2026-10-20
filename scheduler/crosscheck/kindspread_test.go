package crosscheck

// Cross-check of design decision 39 (Phase 2, amendment after close): the GPU
// workers have a soft spread across the GPU nodes. The promises tested, all
// from the decision row and the "Amendment after close: the worker soft
// spread" implementation notes:
//
//   - fresh scheduling prefers balance: with both GPU nodes schedulable and
//     free cards on each, a new or replacement worker pod lands on the node
//     with fewer workers (counts differ by at most 1);
//   - the spread is soft (ScheduleAnyway): with one GPU node cordoned or full,
//     pods still schedule by packing onto the other node; nothing goes Pending
//     for spread reasons;
//   - at 4 replicas (every card in use) it is 2 per node, forced by the card
//     resource (decision 22), and a 5th replica is Pending for cards, not for
//     spread;
//   - the spread is evaluated only at scheduling time (no assertion is made
//     about placement after a rolling update, per the decision's caveat).
//
// Black box: kubectl against the live kind cluster (namespace kgpu, worker
// Deployment `worker`). No manifest under deploy/ is read; the worker pod
// template's constraints are never inspected, only where pods land.
//
// Run: KGPU_KIND=1 go test -count=1 -v -timeout 20m -run TestKindSpread ./scheduler/crosscheck/

import (
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"
)

const (
	spNS     = "kgpu"
	spHogNS  = "kgpu-spread-xcheck"
	spNodeA  = "kgpu-worker3"
	spNodeB  = "kgpu-worker4"
	spSettle = 3 * time.Minute
)

// spView is one observation of the worker pods in namespace kgpu.
type spView struct {
	byNode      map[string][]string // bound, non-terminating pods per node
	pending     []string            // non-terminating pods with no node
	unsched     []string            // pods with PodScheduled=False, with message
	terminating int
	notReady    int
}

func (v spView) count(node string) int { return len(v.byNode[node]) }

func (v spView) total() int {
	n := len(v.pending)
	for _, ps := range v.byNode {
		n += len(ps)
	}
	return n
}

func (v spView) String() string {
	var nodes []string
	for n := range v.byNode {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)
	var parts []string
	for _, n := range nodes {
		parts = append(parts, fmt.Sprintf("%s=%v", n, v.byNode[n]))
	}
	return fmt.Sprintf("{%s pending=%v terminating=%d notReady=%d}", strings.Join(parts, " "), v.pending, v.terminating, v.notReady)
}

func spObserve(t *testing.T) spView {
	t.Helper()
	v := spView{byNode: map[string][]string{}}
	for _, p := range ccPods(t, spNS, "app=worker") {
		if p.Metadata.DeletionTimestamp != nil {
			v.terminating++
			continue
		}
		if s, reason, msg := p.cond("PodScheduled"); s == "False" {
			v.unsched = append(v.unsched, fmt.Sprintf("%s: %s %s", p.Metadata.Name, reason, msg))
		}
		if p.Spec.NodeName == "" {
			v.pending = append(v.pending, p.Metadata.Name)
			continue
		}
		v.byNode[p.Spec.NodeName] = append(v.byNode[p.Spec.NodeName], p.Metadata.Name)
		if !p.ready() {
			v.notReady++
		}
	}
	return v
}

// spSettleTo waits until exactly n worker pods are bound and Ready with none
// terminating and wantPending unbound. Every sample along the way is checked:
// if forbidPending, any pod observed with PodScheduled=False fails the test
// (the spread is soft; nothing may go Pending for spread reasons).
func spSettleTo(t *testing.T, n, wantPending int, forbidPending bool) spView {
	t.Helper()
	var v spView
	var seenUnsched []string
	took := poll(t, spSettle, time.Second, fmt.Sprintf("%d Ready bound workers, %d pending, none terminating", n, wantPending), func() bool {
		v = spObserve(t)
		seenUnsched = append(seenUnsched, v.unsched...)
		bound := v.total() - len(v.pending)
		return bound == n && len(v.pending) == wantPending && v.terminating == 0 && v.notReady == 0
	})
	t.Logf("[k8s] settled after %.1fs: %v", took.Seconds(), v)
	if forbidPending && len(seenUnsched) > 0 {
		t.Errorf("a worker pod was observed with PodScheduled=False while settling (decision 39: the spread is ScheduleAnyway, nothing goes Pending): %v", spDedup(seenUnsched))
	} else if forbidPending {
		t.Logf("[k8s] no worker pod was ever observed with PodScheduled=False while settling (sampled every 1 s)")
	}
	return v
}

func spDedup(xs []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, x := range xs {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}

func spScale(t *testing.T, n int) {
	t.Helper()
	kc(t, "scale", "deployment/worker", "-n", spNS, fmt.Sprintf("--replicas=%d", n))
	t.Logf("[kubectl] scale deployment/worker -n %s --replicas=%d", spNS, n)
}

// spDelete deletes a worker pod gracefully without waiting; the ReplicaSet
// creates the replacement at once, which is a fresh scheduling decision.
func spDelete(t *testing.T, pod string) {
	t.Helper()
	kc(t, "delete", "pod", pod, "-n", spNS, "--wait=false")
	t.Logf("[kubectl] delete pod %s -n %s (graceful, --wait=false)", pod, spNS)
}

// spNewPod returns the one pod in after that was not in before, and its node.
func spNewPod(t *testing.T, before, after spView) (string, string) {
	t.Helper()
	old := map[string]bool{}
	for _, ps := range before.byNode {
		for _, p := range ps {
			old[p] = true
		}
	}
	var name, node string
	n := 0
	for nd, ps := range after.byNode {
		for _, p := range ps {
			if !old[p] {
				name, node = p, nd
				n++
			}
		}
	}
	if n != 1 {
		t.Fatalf("expected exactly one new worker pod, found %d (before %v, after %v)", n, before, after)
	}
	return name, node
}

func spCordon(t *testing.T, node string) {
	t.Helper()
	kc(t, "cordon", node)
	t.Logf("[kubectl] cordon %s", node)
}

func spUncordon(t *testing.T, node string) {
	t.Helper()
	kc(t, "uncordon", node)
	t.Logf("[kubectl] uncordon %s", node)
}

func spOther(node string) string {
	if node == spNodeA {
		return spNodeB
	}
	return spNodeA
}

// spHog: a pause pod outside namespace kgpu (so it is not a worker and does
// not count in the workers' spread) that takes `cards` cards on `node`.
func spHog(name, node string, cards int) string {
	return fmt.Sprintf(`apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
  labels:
    xcheck: spread
spec:
  terminationGracePeriodSeconds: 0
  nodeSelector:
    kubernetes.io/hostname: %s
  tolerations:
    - key: kgpu.io/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule
  containers:
    - name: pause
      image: %s
      imagePullPolicy: IfNotPresent
      resources:
        limits:
          %s: "%d"
`, name, spHogNS, node, kPauseImg, kcCard, cards)
}

// spRestore puts the cluster back: hog namespace gone, both GPU nodes
// uncordoned, 2 replicas Ready, and (best effort) one per node.
func spRestore(t *testing.T) {
	so, se, err := kcRun("", "delete", "namespace", spHogNS, "--ignore-not-found", "--wait=true", "--timeout=120s")
	t.Logf("[cleanup] delete namespace %s: err=%v %s", spHogNS, err, strings.TrimSpace(so+se))
	for _, n := range []string{spNodeA, spNodeB} {
		so, se, err := kcRun("", "uncordon", n)
		t.Logf("[cleanup] uncordon %s: err=%v %s", n, err, strings.TrimSpace(so+se))
	}
	so, se, err = kcRun("", "scale", "deployment/worker", "-n", spNS, "--replicas=2")
	t.Logf("[cleanup] scale worker to 2: err=%v %s", err, strings.TrimSpace(so+se))
	var v spView
	deadline := time.Now().Add(spSettle)
	for time.Now().Before(deadline) {
		v = spObserve(t)
		if v.total() == 2 && len(v.pending) == 0 && v.terminating == 0 && v.notReady == 0 {
			break
		}
		time.Sleep(time.Second)
	}
	if v.count(spNodeA) == 2 || v.count(spNodeB) == 2 {
		node := spNodeA
		if v.count(spNodeB) == 2 {
			node = spNodeB
		}
		t.Logf("[cleanup] both workers on %s; deleting one so the fresh replacement rebalances", node)
		_, _, _ = kcRun("", "delete", "pod", v.byNode[node][0], "-n", spNS, "--wait=false")
		deadline = time.Now().Add(spSettle)
		for time.Now().Before(deadline) {
			v = spObserve(t)
			if v.total() == 2 && len(v.pending) == 0 && v.terminating == 0 && v.notReady == 0 {
				break
			}
			time.Sleep(time.Second)
		}
	}
	t.Logf("[cleanup] final worker placement: %v", v)
}

func TestKindSpread(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set")
	}
	if _, _, err := kcRun("", "get", "deployment", "worker", "-n", spNS); err != nil {
		t.Fatalf("no worker Deployment in %s: %v", spNS, err)
	}
	t.Cleanup(func() { spRestore(t) })

	nodes := kfGPUNodes(t)
	t.Logf("[setup] GPU nodes (label kgpu.io/gpu=true): %v", nodes)
	if len(nodes) != 2 || nodes[0] != spNodeA || nodes[1] != spNodeB {
		t.Fatalf("expected GPU nodes [%s %s] (decision 24), got %v", spNodeA, spNodeB, nodes)
	}
	for _, n := range nodes {
		c, a := nodeCards(t, n)
		t.Logf("[setup] %s: %s capacity=%s allocatable=%s", n, kcCard, c, a)
		if a != "2" {
			t.Fatalf("%s allocatable cards = %q, want 2 (decision 24)", n, a)
		}
		if u := strings.TrimSpace(kc(t, "get", "node", n, "-o", "jsonpath={.spec.unschedulable}")); u == "true" {
			t.Logf("[setup] %s was cordoned at start; uncordoning", n)
			spUncordon(t, n)
		}
	}
	spScale(t, 2)
	start := spSettleTo(t, 2, 0, false)
	t.Logf("[setup] starting placement: %v (not asserted: it may come from a rollout, which decision 39 says can pack)", start)
	if start.count(spNodeA) != 1 {
		// Get to one per node with a fresh replacement, so later subtests start from balance.
		var node string
		if start.count(spNodeA) == 2 {
			node = spNodeA
		} else {
			node = spNodeB
		}
		spDelete(t, start.byNode[node][0])
		start = spSettleTo(t, 2, 0, false)
		t.Logf("[setup] after one fresh replacement: %v", start)
	}

	t.Run("ReplacementLandsOnEmptierNode", func(t *testing.T) {
		// Decision 39: "a fresh replacement pod lands on the emptier node".
		// From one per node, delete the worker on one node; its card frees
		// (once it terminates) and that node now has 0 workers vs 1. Three
		// trials, alternating nodes, so a coin flip would pass with p=1/8.
		for i, node := range []string{spNodeB, spNodeA, spNodeB} {
			before := spObserve(t)
			if before.count(spNodeA) != 1 || before.count(spNodeB) != 1 {
				t.Fatalf("trial %d precondition: want one worker per node, have %v", i+1, before)
			}
			victim := before.byNode[node][0]
			t.Logf("[trial %d] placement %v; deleting %s on %s", i+1, before, victim, node)
			spDelete(t, victim)
			after := spSettleTo(t, 2, 0, true)
			name, landed := spNewPod(t, before, after)
			t.Logf("[trial %d] replacement %s landed on %s; placement now %v", i+1, name, landed, after)
			if landed != node {
				t.Errorf("[trial %d] replacement landed on %s, which already had a worker; decision 39 says fresh scheduling prefers the emptier node %s (maxSkew 1)", i+1, landed, node)
				return
			}
			t.Logf("[trial %d] as decision 39 promises: the replacement went to the emptier node; counts %s=%d %s=%d", i+1, spNodeA, after.count(spNodeA), spNodeB, after.count(spNodeB))
		}
	})

	t.Run("ScaleFromZeroOnePerNode", func(t *testing.T) {
		// Decision 39 / the user's summary: "a 2-replica pool ends up one per
		// node after fresh scheduling". Scaling 0 -> 2 makes both pods fresh
		// scheduling decisions with no old pod holding any card. Two trials.
		for i := 1; i <= 2; i++ {
			spScale(t, 0)
			spSettleTo(t, 0, 0, false)
			spScale(t, 2)
			v := spSettleTo(t, 2, 0, true)
			t.Logf("[trial %d] after 0 -> 2: %v", i, v)
			if v.count(spNodeA) != 1 || v.count(spNodeB) != 1 {
				t.Errorf("[trial %d] 0 -> 2 placed %s=%d %s=%d; decision 39: fresh scheduling with free cards on both nodes ends one per node", i, spNodeA, v.count(spNodeA), spNodeB, v.count(spNodeB))
				return
			}
			t.Logf("[trial %d] one per node, as decision 39 promises", i)
		}
	})

	t.Run("ScaleUpFromOneGoesToEmptyNode", func(t *testing.T) {
		// 1 -> 2: the second pod is a fresh decision with the first already
		// bound; it should land on the other node. Two trials.
		for i := 1; i <= 2; i++ {
			spScale(t, 1)
			one := spSettleTo(t, 1, 0, false)
			var survivorNode string
			for n, ps := range one.byNode {
				if len(ps) == 1 {
					survivorNode = n
				}
			}
			t.Logf("[trial %d] at 1 replica the survivor is on %s: %v", i, survivorNode, one)
			spScale(t, 2)
			two := spSettleTo(t, 2, 0, true)
			name, landed := spNewPod(t, one, two)
			t.Logf("[trial %d] new pod %s landed on %s; placement %v", i, name, landed, two)
			if landed == survivorNode {
				t.Errorf("[trial %d] 1 -> 2 packed the new worker onto %s next to the survivor, though %s had 2 free cards; decision 39 promises fresh scheduling prefers balance", i, landed, spOther(survivorNode))
				return
			}
			t.Logf("[trial %d] the new pod went to the empty node, as decision 39 promises", i)
		}
	})

	t.Run("CordonedNodePacksThenRebalances", func(t *testing.T) {
		// Decision 39: "with one node cordoned the replacement still schedules
		// by packing (nothing goes Pending)"; implementation notes: "after
		// uncordon, the next fresh scheduling rebalanced".
		before := spObserve(t)
		if before.count(spNodeA) != 1 || before.count(spNodeB) != 1 {
			t.Fatalf("precondition: want one worker per node, have %v", before)
		}
		spCordon(t, spNodeB)
		uncordoned := false
		defer func() {
			if !uncordoned {
				spUncordon(t, spNodeB)
			}
		}()
		victim := before.byNode[spNodeB][0]
		t.Logf("[cordon] %s cordoned; deleting its worker %s", spNodeB, victim)
		spDelete(t, victim)
		after := spSettleTo(t, 2, 0, true)
		name, landed := spNewPod(t, before, after)
		t.Logf("[cordon] replacement %s landed on %s; placement %v", name, landed, after)
		if landed != spNodeA || after.count(spNodeA) != 2 {
			t.Fatalf("with %s cordoned the replacement should pack onto %s (the only schedulable GPU node, 1 free card); got %s, placement %v", spNodeB, spNodeA, landed, after)
		}
		t.Logf("[cordon] as decision 39 promises: soft spread, the replacement packed (skew 2 > maxSkew 1) instead of going Pending")

		spUncordon(t, spNodeB)
		uncordoned = true
		victim = after.byNode[spNodeA][0]
		t.Logf("[uncordon] %s uncordoned; nothing moves by itself (decision 39: evaluated only at scheduling). Placement still %v", spNodeB, spObserve(t))
		time.Sleep(5 * time.Second)
		still := spObserve(t)
		t.Logf("[uncordon] 5 s later: %v", still)
		if still.count(spNodeA) != 2 {
			t.Errorf("a running worker moved after uncordon (%v); decision 39 says Kubernetes never relocates a running pod", still)
		}
		t.Logf("[uncordon] deleting %s on %s to force one fresh scheduling decision", victim, spNodeA)
		spDelete(t, victim)
		re := spSettleTo(t, 2, 0, true)
		name, landed = spNewPod(t, still, re)
		t.Logf("[uncordon] replacement %s landed on %s; placement %v", name, landed, re)
		if landed != spNodeB {
			t.Errorf("after uncordon the fresh replacement landed on %s (now %v); implementation notes: the next fresh scheduling rebalances onto %s", landed, re, spNodeB)
			return
		}
		t.Logf("[uncordon] rebalanced to one per node, as the implementation notes record")
	})

	t.Run("FullNodePacks", func(t *testing.T) {
		// Decision 39 / the user's summary: "with one GPU node ... full, pods
		// still schedule by packing onto the other node". Scale to 1, fill the
		// other node's two cards with a non-worker pause pod, scale to 2.
		spScale(t, 1)
		one := spSettleTo(t, 1, 0, false)
		var x string
		for n, ps := range one.byNode {
			if len(ps) == 1 {
				x = n
			}
		}
		y := spOther(x)
		t.Logf("[full] survivor on %s; filling %s's 2 cards with a pause pod in namespace %s", x, y, spHogNS)
		so, se, err := kcRun("", "create", "namespace", spHogNS)
		t.Logf("[kubectl] create namespace %s: err=%v %s", spHogNS, err, strings.TrimSpace(so+se))
		kcApply(t, spHog("hog", y, 2))
		poll(t, 2*time.Minute, time.Second, "hog Running on "+y, func() bool {
			p, ok := ccPodByName(t, spHogNS, "hog")
			return ok && p.Spec.NodeName == y && p.Status.Phase == "Running"
		})
		alloc := cardsAllocated(t)
		t.Logf("[full] hog Running on %s; cards in use per node (pods in all namespaces): %v", y, alloc)

		spScale(t, 2)
		two := spSettleTo(t, 2, 0, true)
		name, landed := spNewPod(t, one, two)
		t.Logf("[full] new worker %s landed on %s; placement %v", name, landed, two)
		if landed != x {
			t.Fatalf("with %s full the new worker should pack onto %s, got %s", y, x, landed)
		}
		t.Logf("[full] as decision 39 promises: with %s full the second worker packed onto %s and was never Pending", y, x)

		so, se, err = kcRun("", "delete", "pod", "hog", "-n", spHogNS, "--grace-period=0", "--wait=true")
		t.Logf("[full] delete hog: err=%v %s", err, strings.TrimSpace(so+se))
		victim := two.byNode[x][0]
		t.Logf("[full] %s has free cards again; deleting %s on %s to force a fresh decision", y, victim, x)
		spDelete(t, victim)
		re := spSettleTo(t, 2, 0, true)
		name, landed = spNewPod(t, two, re)
		t.Logf("[full] replacement %s landed on %s; placement %v", name, landed, re)
		if landed != y {
			t.Errorf("after the hog left, the fresh replacement landed on %s (placement %v); decision 39: fresh scheduling prefers the emptier node %s", landed, re, y)
		}
	})

	t.Run("FourReplicasTwoPerNodeFifthPendingForCards", func(t *testing.T) {
		// Decision 39: "the soft rule changes nothing at 4 replicas (every card
		// in use forces 2 per node anyway) and never blocks scheduling".
		spScale(t, 4)
		v := spSettleTo(t, 4, 0, true)
		if v.count(spNodeA) != 2 || v.count(spNodeB) != 2 {
			t.Fatalf("4 replicas placed %s=%d %s=%d, want 2 and 2 (decision 22 cards, decision 39)", spNodeA, v.count(spNodeA), spNodeB, v.count(spNodeB))
		}
		t.Logf("[4] 2 per node, every card in use: %v", v)

		spScale(t, 5)
		v5 := spSettleTo(t, 4, 1, false)
		var msg string
		poll(t, time.Minute, time.Second, "the 5th worker has PodScheduled=False", func() bool {
			for _, p := range ccPods(t, spNS, "app=worker") {
				if p.Metadata.DeletionTimestamp == nil && p.Spec.NodeName == "" {
					s, _, m := p.cond("PodScheduled")
					msg = m
					return s == "False"
				}
			}
			return false
		})
		t.Logf("[5] placement %v; the unbound pod's PodScheduled message: %q", v5, msg)
		if !strings.Contains(msg, "Insufficient "+kcCard) {
			t.Errorf("the 5th worker is Pending but not for cards: %q (decision 22: the card resource is the cap)", msg)
		}
		if strings.Contains(strings.ToLower(msg), "topology spread") {
			t.Errorf("the 5th worker's scheduling failure cites topology spread: %q; decision 39 says the spread never blocks scheduling", msg)
		}
		t.Logf("[5] Pending only for %s, not for spread: the soft rule adds no filter (decision 39)", kcCard)
		spScale(t, 2)
		back := spSettleTo(t, 2, 0, false)
		t.Logf("[4->2] after scale-down: %v (which pods the ReplicaSet removes is not in spec; not asserted)", back)
	})
}
