package crosscheck

// Phase 3 step 7 cross-check on the live kind cluster: decision 30 and bug 15.
// The promise: after a ConfigMap change and `kubectl rollout restart`, the
// policy actually serving is proven only by the NEW pod's own startup line
// ("policy":"…"), after every pre-restart pod is gone — `rollout status`
// issued right after the restart may return before the restart has begun.
//
// This test switches the implementer's namespace kgpu (kind context
// kind-kgpu) from always_gpu to reactive and back, touching ONLY the scheduler
// ConfigMap's POLICY key (FAKE_TIME_SCALE, WORKER_MEM_BYTES and the worker
// replica count are read, never written) and restoring it in a cleanup. For
// each switch it records, side by side: when `rollout status` returned and
// what the cluster looked like at that instant (which pods existed, which were
// Ready, which policy each had logged), and when the decision-30 check (old
// pods gone, new pod's startup line names the policy) succeeded.
//
// Needs KGPU_KIND=1. Takes about a minute.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"
)

var vkPolicyRe = regexp.MustCompile(`"msg":"scheduler listening".*?"policy":"([^"]*)"`)

type vkPod struct {
	name        string
	ready       bool
	terminating bool
	policy      string // from the pod's own startup line; "" if not logged yet
}

func (p vkPod) String() string {
	return fmt.Sprintf("%s(ready=%v terminating=%v policy=%q)", p.name, p.ready, p.terminating, p.policy)
}

func vkSchedulerPods(t *testing.T) []vkPod {
	t.Helper()
	var list struct {
		Items []struct {
			Metadata struct {
				Name              string  `json:"name"`
				DeletionTimestamp *string `json:"deletionTimestamp"`
			} `json:"metadata"`
			Status struct {
				Conditions []struct{ Type, Status string } `json:"conditions"`
			} `json:"status"`
		} `json:"items"`
	}
	so, se, err := kcRun("", "-n", kOrigNS, "get", "pods", "-l", "app=scheduler", "-o", "json")
	if err != nil {
		t.Fatalf("get pods: %v %s", err, se)
	}
	if err := json.Unmarshal([]byte(so), &list); err != nil {
		t.Fatalf("decode pods: %v", err)
	}
	var out []vkPod
	for _, it := range list.Items {
		p := vkPod{name: it.Metadata.Name, terminating: it.Metadata.DeletionTimestamp != nil}
		for _, c := range it.Status.Conditions {
			if c.Type == "Ready" && c.Status == "True" {
				p.ready = true
			}
		}
		if logs, _, err := kcRun("", "-n", kOrigNS, "logs", p.name); err == nil {
			if m := vkPolicyRe.FindStringSubmatch(logs); m != nil {
				p.policy = m[1]
			}
		}
		out = append(out, p)
	}
	return out
}

func vkCM(t *testing.T, cm, key string) string {
	t.Helper()
	return strings.TrimSpace(kc(t, "-n", kOrigNS, "get", "cm", cm, "-o", "jsonpath={.data."+key+"}"))
}

// vkSwitch sets POLICY, restarts, times `rollout status`, then waits for the
// decision-30 proof. Returns seconds to status-return and to proof.
func vkSwitch(t *testing.T, want string) (statusS, provenS float64) {
	t.Helper()
	before := vkSchedulerPods(t)
	old := map[string]bool{}
	for _, p := range before {
		old[p.name] = true
	}
	t.Logf("[switch to %s] before: %v", want, before)
	kc(t, "-n", kOrigNS, "patch", "cm", "scheduler", "--type", "merge", "-p", fmt.Sprintf(`{"data":{"POLICY":%q}}`, want))
	t0 := time.Now()
	kc(t, "-n", kOrigNS, "rollout", "restart", "deploy/scheduler")
	so, se, err := kcRun("", "-n", kOrigNS, "rollout", "status", "deploy/scheduler", "--timeout=180s")
	statusS = time.Since(t0).Seconds()
	if err != nil {
		t.Fatalf("rollout status: %v %s", err, se)
	}
	at := vkSchedulerPods(t)
	t.Logf("[switch to %s] `rollout status` returned after %.2f s: %q", want, statusS, strings.TrimSpace(so))
	t.Logf("[switch to %s] pods at that instant: %v", want, at)
	stale, draining := 0, 0
	for _, p := range at {
		if old[p.name] && p.ready && !p.terminating {
			stale++
		}
		if old[p.name] && p.terminating {
			draining++
		}
	}
	if draining > 0 {
		t.Logf("[switch to %s] at status-return %d pre-restart pod(s) were still alive, terminating (draining, decision 40) and logged the PREVIOUS policy: a client holding a connection to it (keep-alive through the NodePort) could still be answered under the old policy; `rollout status` does not wait for them", want, draining)
	}
	if stale > 0 {
		t.Logf("[switch to %s] BUG-15 WINDOW OBSERVED: rollout status had returned while %d pre-restart pod(s) were still Ready and not terminating, i.e. still serving the previous policy", want, stale)
	} else {
		t.Logf("[switch to %s] at status-return no pre-restart pod was Ready-and-not-terminating (the race did not show this time; decision 30's check is still the only proof)", want)
	}
	var proven []vkPod
	d := poll(t, 180*time.Second, 500*time.Millisecond, "old scheduler pods gone and the new pod's startup line naming "+want, func() bool {
		ps := vkSchedulerPods(t)
		for _, p := range ps {
			if old[p.name] {
				return false
			}
		}
		if len(ps) != 1 || !ps[0].ready || ps[0].policy == "" {
			return false
		}
		proven = ps
		return true
	})
	provenS = time.Since(t0).Seconds()
	_ = d
	t.Logf("[switch to %s] decision-30 proof after %.2f s: %v", want, provenS, proven)
	if proven[0].policy != want {
		t.Errorf("[switch to %s] the new pod's startup line says policy %q: the ConfigMap change did not reach the serving process", want, proven[0].policy)
	}
	return statusS, provenS
}

func TestStep7KindPolicySwitch(t *testing.T) {
	if !kindEnabled() {
		t.Skip("KGPU_KIND=1 not set: needs the kind cluster kgpu")
	}
	orig := vkCM(t, "scheduler", "POLICY")
	scale := vkCM(t, "scheduler", "FAKE_TIME_SCALE")
	wscale := vkCM(t, "worker", "FAKE_TIME_SCALE")
	wmem := vkCM(t, "worker", "WORKER_MEM_BYTES")
	reps := strings.TrimSpace(kc(t, "-n", kOrigNS, "get", "deploy", "worker", "-o", "jsonpath={.spec.replicas}"))
	t.Logf("cluster before: scheduler POLICY=%s FAKE_TIME_SCALE=%s; worker FAKE_TIME_SCALE=%s WORKER_MEM_BYTES=%s replicas=%s", orig, scale, wscale, wmem, reps)
	if orig != "always_gpu" || scale != "10" || wmem != "25769803776" || reps != "2" {
		t.Fatalf("cluster is not in the expected idle state (always_gpu, scale 10, 24 GiB, 2 workers); refusing to touch it")
	}
	t.Cleanup(func() {
		if cur := vkCM(t, "scheduler", "POLICY"); cur != orig {
			t.Logf("cleanup: POLICY is %s, restoring %s", cur, orig)
			vkSwitch(t, orig)
		}
		after := fmt.Sprintf("POLICY=%s FAKE_TIME_SCALE=%s worker FAKE_TIME_SCALE=%s WORKER_MEM_BYTES=%s replicas=%s",
			vkCM(t, "scheduler", "POLICY"), vkCM(t, "scheduler", "FAKE_TIME_SCALE"), vkCM(t, "worker", "FAKE_TIME_SCALE"),
			vkCM(t, "worker", "WORKER_MEM_BYTES"), strings.TrimSpace(kc(t, "-n", kOrigNS, "get", "deploy", "worker", "-o", "jsonpath={.spec.replicas}")))
		t.Logf("cluster after: %s", after)
		if !strings.Contains(after, "POLICY=always_gpu FAKE_TIME_SCALE=10 worker FAKE_TIME_SCALE=10 WORKER_MEM_BYTES=25769803776 replicas=2") {
			t.Errorf("cluster NOT restored: %s", after)
		}
	})

	s1, p1 := vkSwitch(t, "reactive")
	s2, p2 := vkSwitch(t, orig)
	t.Logf("summary: always_gpu->reactive: rollout status %.2f s, proven %.2f s; reactive->always_gpu: status %.2f s, proven %.2f s", s1, p1, s2, p2)
	t.Logf("reading: the gap between the two numbers is how long a runner trusting `rollout status` could have started a round before (or while) the intended policy was the one serving (bug 15)")
}
