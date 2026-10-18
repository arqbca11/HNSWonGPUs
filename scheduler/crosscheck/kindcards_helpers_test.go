package crosscheck

// Helpers for the cross-check of Phase 2 step 3b: GPU cards as the extended
// resource kgpu.io/gpu-card (decisions 22 to 25). Black box: kubectl (node
// status, pod specs and scheduling conditions), the scheduler API through a
// port-forward, and a shell inside the worker container for the kill test.
//
// Card accounting is done the way `kubectl describe node` does it: the sum of
// the card requests of every pod bound to the node that has not terminated.

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	kcCard     = "kgpu.io/gpu-card"
	kcCardJP   = `kgpu\.io/gpu-card`
	kcPortAPI  = 28093
	kcLongN    = 8000000
	kcInsuffRe = `Insufficient kgpu\.io/gpu-card`
)

// ---- rendering with perl ----

// renderXcheckPerl renders deploy/k8s with kustomize and rewrites the namespace
// with perl: only lines that are exactly `namespace: kgpu` or `name: kgpu`
// change (the database name and password, also "kgpu", must not).
func renderXcheckPerl(t *testing.T) string {
	t.Helper()
	orig := kc(t, "kustomize", xcRepoRoot()+"/deploy/k8s")
	cmd := exec.Command("perl", "-pe", `s/^(\s*)namespace: kgpu$/${1}namespace: kgpu-xcheck/; s/^(\s*)name: kgpu$/${1}name: kgpu-xcheck/`)
	cmd.Stdin = strings.NewReader(orig)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("perl rewrite: %v", err)
	}
	res := string(out)
	left := regexp.MustCompile(`(?m)^\s*(namespace|name): kgpu\s*$`).FindAllString(res, -1)
	if len(left) > 0 {
		t.Fatalf("render still references namespace kgpu: %q", left)
	}
	nOrig := len(regexp.MustCompile(`(?m)^\s*namespace: kgpu$`).FindAllString(orig, -1))
	nNew := len(regexp.MustCompile(`(?m)^\s*namespace: kgpu-xcheck$`).FindAllString(res, -1))
	if nOrig == 0 || nOrig != nNew {
		t.Fatalf("namespace rewrite count: %d in the original, %d rewritten", nOrig, nNew)
	}
	if strings.Contains(res, ".kgpu.svc") || strings.Contains(res, ".kgpu:") {
		t.Fatalf("render contains a cross-namespace DNS name into kgpu")
	}
	t.Logf("[render] kustomize deploy/k8s, perl rewrote %d namespace lines and the Namespace name to %s; no kgpu namespace reference remains", nNew, kNS)
	return res
}

// docsSelect returns the documents of the given kind, and of the given
// metadata name if name != "".
func docsSelect(rendered, kind, name string) string {
	var keep []string
	kre := regexp.MustCompile(`(?m)^kind: ` + kind + `$`)
	nre := regexp.MustCompile(`(?m)^  name: ` + regexp.QuoteMeta(name) + `$`)
	for _, d := range strings.Split(rendered, "\n---\n") {
		if kre.MatchString(d) && (name == "" || nre.MatchString(d)) {
			keep = append(keep, strings.TrimPrefix(d, "---\n"))
		}
	}
	return strings.Join(keep, "\n---\n")
}

// ---- pods, with the fields the card tests need ----

type ccPod struct {
	Metadata struct {
		Name              string            `json:"name"`
		Namespace         string            `json:"namespace"`
		UID               string            `json:"uid"`
		Labels            map[string]string `json:"labels"`
		DeletionTimestamp *string           `json:"deletionTimestamp"`
	} `json:"metadata"`
	Spec struct {
		NodeName   string `json:"nodeName"`
		Containers []struct {
			Name      string `json:"name"`
			Resources struct {
				Requests map[string]string `json:"requests"`
				Limits   map[string]string `json:"limits"`
			} `json:"resources"`
		} `json:"containers"`
	} `json:"spec"`
	Status struct {
		Phase      string `json:"phase"`
		Conditions []struct {
			Type    string `json:"type"`
			Status  string `json:"status"`
			Reason  string `json:"reason"`
			Message string `json:"message"`
		} `json:"conditions"`
		ContainerStatuses []struct {
			Ready        bool `json:"ready"`
			RestartCount int  `json:"restartCount"`
			LastState    struct {
				Terminated *struct {
					ExitCode int    `json:"exitCode"`
					Reason   string `json:"reason"`
				} `json:"terminated"`
			} `json:"lastState"`
		} `json:"containerStatuses"`
	} `json:"status"`
}

func (p ccPod) cond(typ string) (status, reason, msg string) {
	for _, c := range p.Status.Conditions {
		if c.Type == typ {
			return c.Status, c.Reason, c.Message
		}
	}
	return "", "", ""
}

func (p ccPod) ready() bool { s, _, _ := p.cond("Ready"); return s == "True" }

func (p ccPod) restarts() int {
	n := 0
	for _, c := range p.Status.ContainerStatuses {
		n += c.RestartCount
	}
	return n
}

func (p ccPod) cards() int {
	n := 0
	for _, c := range p.Spec.Containers {
		if v, ok := c.Resources.Requests[kcCard]; ok {
			x, _ := strconv.Atoi(v)
			n += x
		}
	}
	return n
}

func (p ccPod) terminated() bool {
	return p.Status.Phase == "Succeeded" || p.Status.Phase == "Failed"
}

func (p ccPod) String() string {
	node := p.Spec.NodeName
	if node == "" {
		node = "<unbound>"
	}
	return fmt.Sprintf("%s(uid=%s node=%s phase=%s ready=%v restarts=%d cards=%d)",
		p.Metadata.Name, short(p.Metadata.UID), node, p.Status.Phase, p.ready(), p.restarts(), p.cards())
}

func ccPods(t *testing.T, ns, selector string) []ccPod {
	t.Helper()
	args := []string{"get", "pods", "-o", "json"}
	if ns == "" {
		args = append(args, "-A")
	} else {
		args = append(args, "-n", ns)
	}
	if selector != "" {
		args = append(args, "-l", selector)
	}
	var list struct {
		Items []ccPod `json:"items"`
	}
	if err := json.Unmarshal([]byte(kc(t, args...)), &list); err != nil {
		t.Fatalf("decode pods: %v", err)
	}
	sort.Slice(list.Items, func(i, j int) bool { return list.Items[i].Metadata.Name < list.Items[j].Metadata.Name })
	return list.Items
}

func ccPodByName(t *testing.T, ns, name string) (ccPod, bool) {
	t.Helper()
	for _, p := range ccPods(t, ns, "") {
		if p.Metadata.Name == name {
			return p, true
		}
	}
	return ccPod{}, false
}

// myWorkers returns the cross-check's worker pods that are not being deleted.
func myWorkers(t *testing.T) (bound, unbound []ccPod) {
	t.Helper()
	for _, p := range ccPods(t, kNS, "app=worker") {
		if p.Metadata.DeletionTimestamp != nil {
			continue
		}
		if p.Spec.NodeName == "" {
			unbound = append(unbound, p)
		} else {
			bound = append(bound, p)
		}
	}
	return
}

// cardsAllocated sums the card requests of every non-terminated pod bound to
// each node, across all namespaces.
func cardsAllocated(t *testing.T) map[string]int {
	t.Helper()
	m := map[string]int{}
	for _, p := range ccPods(t, "", "") {
		if p.Spec.NodeName != "" && !p.terminated() {
			m[p.Spec.NodeName] += p.cards()
		}
	}
	return m
}

func byNode(ps []ccPod) map[string][]string {
	m := map[string][]string{}
	for _, p := range ps {
		m[p.Spec.NodeName] = append(m[p.Spec.NodeName], p.Metadata.Name)
	}
	return m
}

// ---- nodes ----

func gpuNodes(t *testing.T) []string {
	t.Helper()
	var names []string
	for _, n := range nodes(t) {
		if n.Metadata.Labels[kGPUKey] == "true" {
			names = append(names, n.Metadata.Name)
		}
	}
	sort.Strings(names)
	return names
}

// nodeCards returns the node's capacity and allocatable for the card resource
// ("" if the node does not advertise it).
func nodeCards(t *testing.T, node string) (capacity, allocatable string) {
	t.Helper()
	out := kc(t, "get", "node", node, "-o",
		`jsonpath={.status.capacity.`+kcCardJP+`}{" "}{.status.allocatable.`+kcCardJP+`}`)
	f := strings.Split(out, " ")
	if len(f) != 2 {
		return "", ""
	}
	return strings.TrimSpace(f[0]), strings.TrimSpace(f[1])
}

func patchNodeCapacity(node, value string) (string, error) {
	so, se, err := kcRun("", "patch", "node", node, "--subresource=status", "--type=merge",
		"-p", fmt.Sprintf(`{"status":{"capacity":{"%s":"%s"}}}`, kcCard, value))
	return strings.TrimSpace(so + se), err
}

// ---- workers ----

func scaleWorkers(t *testing.T, ns string, n int) {
	t.Helper()
	kc(t, "scale", "deployment/worker", "-n", ns, fmt.Sprintf("--replicas=%d", n))
	t.Logf("[kubectl] scale deployment/worker -n %s --replicas=%d", ns, n)
}

// waitBound waits for exactly n bound, Ready, non-terminating worker pods and
// `pending` unbound ones in the cross-check namespace.
func waitBound(t *testing.T, n, pending int, d time.Duration) (bound, unbound []ccPod) {
	t.Helper()
	took := poll(t, d, time.Second, fmt.Sprintf("%d Ready bound workers and %d unbound", n, pending), func() bool {
		all := ccPods(t, kNS, "app=worker")
		bound, unbound = nil, nil
		for _, p := range all {
			if p.Metadata.DeletionTimestamp != nil {
				return false
			}
			if p.Spec.NodeName == "" {
				unbound = append(unbound, p)
			} else {
				bound = append(bound, p)
			}
		}
		if len(bound) != n || len(unbound) != pending {
			return false
		}
		for _, p := range bound {
			if !p.ready() {
				return false
			}
		}
		return true
	})
	t.Logf("[k8s] after %.1fs: bound %v; unbound %v; by node %v", took.Seconds(), bound, unbound, byNode(bound))
	return
}

// checkWorkerCardSpec: every worker pod requests exactly one card, request
// equal to limit (decision 22, worker.yaml).
func checkWorkerCardSpec(t *testing.T, ps []ccPod) {
	t.Helper()
	for _, p := range ps {
		for _, c := range p.Spec.Containers {
			req, lim := c.Resources.Requests[kcCard], c.Resources.Limits[kcCard]
			t.Logf("[spec] %s container %s: requests %s=%q limits %s=%q", p.Metadata.Name, c.Name, kcCard, req, kcCard, lim)
			if req != "1" || lim != "1" {
				t.Errorf("%s: card request %q limit %q, want 1 and 1 (decision 22)", p.Metadata.Name, req, lim)
			}
		}
	}
}

// ---- the pool through the API ----

type ccWorkers struct {
	Pool struct {
		LiveWorkers int `json:"live_workers"`
	} `json:"pool"`
	Workers []struct {
		WorkerID string `json:"worker_id"`
		Live     bool   `json:"live"`
	} `json:"workers"`
}

func (w ccWorkers) liveIDs() []string {
	var ids []string
	for _, x := range w.Workers {
		if x.Live {
			ids = append(ids, x.WorkerID)
		}
	}
	sort.Strings(ids)
	return ids
}

func ccPool(t *testing.T, api *kwAPI) ccWorkers {
	t.Helper()
	code, raw := api.do(t, "GET", "/workers", nil)
	var w ccWorkers
	if code != 200 || json.Unmarshal([]byte(raw), &w) != nil {
		t.Fatalf("GET /workers -> %d %s", code, raw)
	}
	return w
}

// waitPool waits until live_workers == n and the live ids equal the names.
func waitPool(t *testing.T, api *kwAPI, names []string, d time.Duration) ccWorkers {
	t.Helper()
	sort.Strings(names)
	want := strings.Join(names, ",")
	var w ccWorkers
	took := poll(t, d, 500*time.Millisecond, fmt.Sprintf("pool of %d live workers = %v", len(names), names), func() bool {
		w = ccPool(t, api)
		return w.Pool.LiveWorkers == len(names) && strings.Join(w.liveIDs(), ",") == want
	})
	t.Logf("[api] GET /workers after %.1fs: live_workers=%d live ids=%v", took.Seconds(), w.Pool.LiveWorkers, w.liveIDs())
	return w
}

func names(ps []ccPod) []string {
	var n []string
	for _, p := range ps {
		n = append(n, p.Metadata.Name)
	}
	sort.Strings(n)
	return n
}

// ---- test pods ----

// cardPausePod: a pause pod with the worker's selector and toleration and the
// given card limit (0 = no card resource at all).
func cardPausePod(name string, cards int) string {
	extra := ""
	if cards > 0 {
		extra = fmt.Sprintf("      resources:\n        limits:\n          %s: \"%d\"\n", kcCard, cards)
	}
	return pausePod(name, map[string]string{"xcheck": "cards"}, extra+workerShape)
}

func deletePod(t *testing.T, name string) {
	t.Helper()
	so, se, err := kcRun("", "delete", "pod", name, "-n", kNS, "--grace-period=0", "--wait", "--ignore-not-found")
	t.Logf("[kubectl] delete pod %s: err=%v %s", name, err, strings.TrimSpace(so+se))
}

// waitUnschedulable waits until the pod is unbound with PodScheduled=False and
// a message matching re, and returns the message.
func waitUnschedulable(t *testing.T, name, re string, d time.Duration) string {
	t.Helper()
	rx := regexp.MustCompile(re)
	var msg string
	poll(t, d, time.Second, name+" unschedulable with "+re, func() bool {
		p, ok := ccPodByName(t, kNS, name)
		if !ok || p.Spec.NodeName != "" {
			return false
		}
		s, _, m := p.cond("PodScheduled")
		msg = m
		return s == "False" && rx.MatchString(m)
	})
	return msg
}
