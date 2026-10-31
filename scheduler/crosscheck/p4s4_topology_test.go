package crosscheck

// Phase 4 step 4 cross-check: the cloud topology without GPUs, black-box.
//
// Spec: docs/design/phase4-hnswlib.md decisions 25-32 and the "Step 4: the
// cloud topology without GPUs" implementation notes; docs/roadmap.md Phase 4
// "The cloud topology without GPUs"; the caller's brief (the overlay's
// promised shape, the live cluster's promised shape, the sweep's calibrated-
// constant flags); `go doc ./scheduler/costmodel` (Default) and
// `go doc ./scheduler/api.WorkersResponse` for GET /workers.
//
// Three groups, all prefixed TestXcP4S4:
//   - Render*: `kubectl kustomize deploy/k8s-gcp` (and deploy/k8s for the
//     base) parsed and checked. Needs only the kubectl binary.
//   - Live*:   the GKE cluster, read-only (get, logs, exec of a GET, port-
//     forward). Skips unless the current kubectl context is the GKE one and it
//     answers within 30 s.
//   - Sweep*:  the sweep runner as a process (pure, no cluster, no database).
//
// Nothing here mutates the cluster: no apply, delete, scale or rollout.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
)

const (
	p4s4Context  = "gke_project-3456ed25-5d5d-4b0d-aa5_us-west1-b_kgpu"
	p4s4NS       = "kgpu"
	p4s4Registry = "us-west1-docker.pkg.dev/project-3456ed25-5d5d-4b0d-aa5/kgpu/"
	p4s4PoolKey  = "cloud.google.com/gke-nodepool"
	p4s4ILBKey   = "networking.gke.io/load-balancer-type"
	p4s4GPUKey   = "kgpu.io/gpu"
	p4s4CardRes  = "kgpu.io/gpu-card"
	p4s4VPC      = "10.138.0.0/20"
)

// ---------------------------------------------------------------------------
// A minimal reader for kustomize's canonical block YAML (two-space indent,
// sequences at the parent key's indent, no anchors). Every scalar stays a
// string, quotes removed, so `cpu: "7"` and `replicas: 2` read as "7" and "2".
// The module has no YAML dependency and adding one needs the owner's say.

type p4s4Line struct {
	indent int
	text   string
}

func p4s4ParseYAML(t *testing.T, src string) []map[string]any {
	t.Helper()
	var docs []map[string]any
	for _, chunk := range regexp.MustCompile(`(?m)^---\s*$`).Split(src, -1) {
		var lines []p4s4Line
		for _, raw := range strings.Split(chunk, "\n") {
			trimmed := strings.TrimLeft(raw, " ")
			if strings.TrimSpace(trimmed) == "" || strings.HasPrefix(trimmed, "#") {
				continue
			}
			lines = append(lines, p4s4Line{len(raw) - len(trimmed), strings.TrimRight(trimmed, " ")})
		}
		if len(lines) == 0 {
			continue
		}
		v, next := p4s4Node(t, lines, 0, lines[0].indent)
		if next != len(lines) {
			t.Fatalf("yaml: stopped at line %d of %d: %q", next, len(lines), lines[next].text)
		}
		m, ok := v.(map[string]any)
		if !ok {
			t.Fatalf("yaml: document is not a map: %#v", v)
		}
		docs = append(docs, m)
	}
	return docs
}

func p4s4IsSeq(s string) bool { return s == "-" || strings.HasPrefix(s, "- ") }

func p4s4Node(t *testing.T, ls []p4s4Line, i, indent int) (any, int) {
	if i >= len(ls) || ls[i].indent < indent {
		return nil, i
	}
	if p4s4IsSeq(ls[i].text) {
		var seq []any
		for i < len(ls) && ls[i].indent == indent && p4s4IsSeq(ls[i].text) {
			rest := strings.TrimPrefix(strings.TrimPrefix(ls[i].text, "-"), " ")
			if rest == "" {
				var v any
				v, i = p4s4Node(t, ls, i+1, indent+1)
				seq = append(seq, v)
				continue
			}
			if _, _, isKV := p4s4SplitKV(rest); isKV {
				ls[i] = p4s4Line{indent + 2, rest} // the item is a map starting on this line
				var v any
				v, i = p4s4Node(t, ls, i, indent+2)
				seq = append(seq, v)
				continue
			}
			seq = append(seq, p4s4Scalar(rest))
			i++
		}
		return seq, i
	}
	m := map[string]any{}
	for i < len(ls) && ls[i].indent == indent && !p4s4IsSeq(ls[i].text) {
		k, rest, ok := p4s4SplitKV(ls[i].text)
		if !ok {
			t.Fatalf("yaml: expected key at indent %d: %q", indent, ls[i].text)
		}
		i++
		switch {
		case rest == "":
			if i < len(ls) && ls[i].indent > indent {
				m[k], i = p4s4Node(t, ls, i, ls[i].indent)
			} else if i < len(ls) && ls[i].indent == indent && p4s4IsSeq(ls[i].text) {
				m[k], i = p4s4Node(t, ls, i, indent)
			} else {
				m[k] = nil
			}
		case rest == "|" || rest == "|-" || rest == ">" || rest == ">-":
			var b []string
			for i < len(ls) && ls[i].indent > indent {
				b = append(b, ls[i].text)
				i++
			}
			m[k] = strings.Join(b, "\n")
		default:
			m[k] = p4s4Scalar(rest)
		}
	}
	return m, i
}

// p4s4SplitKV splits "key: value" or "key:"; a quoted key is unquoted.
func p4s4SplitKV(s string) (string, string, bool) {
	if strings.HasPrefix(s, `"`) || strings.HasPrefix(s, `'`) {
		q := s[:1]
		end := strings.Index(s[1:], q)
		if end < 0 || !strings.HasPrefix(s[end+2:], ":") {
			return "", "", false
		}
		return s[1 : end+1], strings.TrimSpace(s[end+3:]), true
	}
	if j := strings.Index(s, ": "); j >= 0 {
		return s[:j], strings.TrimSpace(s[j+2:]), true
	}
	if strings.HasSuffix(s, ":") {
		return s[:len(s)-1], "", true
	}
	return "", "", false
}

func p4s4Scalar(s string) any {
	switch s {
	case "[]":
		return []any{}
	case "{}":
		return map[string]any{}
	}
	if strings.HasPrefix(s, `"`) {
		if u, err := strconv.Unquote(s); err == nil {
			return u
		}
	}
	if len(s) >= 2 && strings.HasPrefix(s, "'") && strings.HasSuffix(s, "'") {
		return strings.ReplaceAll(s[1:len(s)-1], "''", "'")
	}
	return s
}

// p4s4Get walks a path of map keys and list indices ("0", "1"...).
func p4s4Get(v any, path ...string) any {
	for _, p := range path {
		switch x := v.(type) {
		case map[string]any:
			v = x[p]
		case []any:
			n, err := strconv.Atoi(p)
			if err != nil || n >= len(x) {
				return nil
			}
			v = x[n]
		default:
			return nil
		}
	}
	return v
}

func p4s4Str(v any, path ...string) string {
	x := p4s4Get(v, path...)
	if x == nil {
		return ""
	}
	return fmt.Sprint(x)
}

// p4s4Flatten turns an object into path -> scalar, for diffs.
func p4s4Flatten(prefix string, v any, out map[string]string) {
	switch x := v.(type) {
	case map[string]any:
		if len(x) == 0 {
			out[prefix] = "{}"
		}
		for k, c := range x {
			p4s4Flatten(prefix+"."+k, c, out)
		}
	case []any:
		if len(x) == 0 {
			out[prefix] = "[]"
		}
		for i, c := range x {
			p4s4Flatten(fmt.Sprintf("%s[%d]", prefix, i), c, out)
		}
	case nil:
		out[prefix] = "<null>"
	default:
		out[prefix] = fmt.Sprint(x)
	}
}

// ---------------------------------------------------------------------------
// Rendering.

var (
	p4s4RenderOnce sync.Once
	p4s4Overlay    map[string]map[string]any // "Kind/name" -> object
	p4s4Base       map[string]map[string]any
	p4s4RenderErr  error
	p4s4RenderRaw  [2]string
)

func p4s4Render(t *testing.T) (overlay, base map[string]map[string]any) {
	t.Helper()
	if _, err := exec.LookPath("kubectl"); err != nil {
		t.Skip("kubectl not on PATH; the render tests need its built-in kustomize")
	}
	p4s4RenderOnce.Do(func() {
		root := repoRoot(t)
		for i, dir := range []string{"deploy/k8s-gcp", "deploy/k8s"} {
			out, err := exec.Command("kubectl", "kustomize", filepath.Join(root, dir)).Output()
			if err != nil {
				p4s4RenderErr = fmt.Errorf("kubectl kustomize %s: %v", dir, err)
				return
			}
			p4s4RenderRaw[i] = string(out)
		}
	})
	if p4s4RenderErr != nil {
		t.Fatal(p4s4RenderErr)
	}
	index := func(src string) map[string]map[string]any {
		m := map[string]map[string]any{}
		for _, d := range p4s4ParseYAML(t, src) {
			key := p4s4Str(d, "kind") + "/" + p4s4Str(d, "metadata", "name")
			if _, dup := m[key]; dup {
				t.Fatalf("rendered twice: %s", key)
			}
			m[key] = d
		}
		return m
	}
	overlay, base = index(p4s4RenderRaw[0]), index(p4s4RenderRaw[1])
	return overlay, base
}

func p4s4Obj(t *testing.T, objs map[string]map[string]any, key string) map[string]any {
	t.Helper()
	o, ok := objs[key]
	if !ok {
		var ks []string
		for k := range objs {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		t.Fatalf("rendered overlay has no %s; it has %v", key, ks)
	}
	return o
}

func p4s4Container(t *testing.T, obj map[string]any, name string) map[string]any {
	t.Helper()
	cs, _ := p4s4Get(obj, "spec", "template", "spec", "containers").([]any)
	for _, c := range cs {
		if p4s4Str(c, "name") == name {
			return c.(map[string]any)
		}
	}
	t.Fatalf("no container %q in %s", name, p4s4Str(obj, "metadata", "name"))
	return nil
}

// Decision 25 / brief: images come from Artifact Registry, built for amd64 and
// tagged p4. The base's images (kgpu-*:dev, kind-loaded) must not leak.
func TestXcP4S4RenderImagesFromArtifactRegistry(t *testing.T) {
	ov, _ := p4s4Render(t)
	want := map[string]string{
		"Deployment/scheduler": p4s4Registry + "kgpu-scheduler:p4",
		"Deployment/worker":    p4s4Registry + "kgpu-worker:p4",
	}
	for key, img := range want {
		obj := p4s4Obj(t, ov, key)
		name := strings.TrimPrefix(key, "Deployment/")
		got := p4s4Str(p4s4Container(t, obj, name), "image")
		t.Logf("[render] %s container %q image = %s (spec, decision 25: %s)", key, name, got, img)
		if got != img {
			t.Errorf("%s image = %q, want %q", key, got, img)
		}
	}
	// No kgpu image anywhere is a kind-local :dev tag.
	for key, obj := range ov {
		cs, _ := p4s4Get(obj, "spec", "template", "spec", "containers").([]any)
		for _, c := range cs {
			img := p4s4Str(c, "image")
			if strings.Contains(img, "kgpu-") && !strings.HasPrefix(img, p4s4Registry) {
				t.Errorf("%s still uses a non-registry kgpu image %q", key, img)
			}
		}
	}
}

// Decision 25 / brief: workers select the `workers` pool by GKE's pool label,
// with no stand-in GPU node selector, no toleration for its taint and no fake
// card resource.
func TestXcP4S4RenderWorkersOnPoolNoFakeCard(t *testing.T) {
	ov, _ := p4s4Render(t)
	w := p4s4Obj(t, ov, "Deployment/worker")
	pod := p4s4Get(w, "spec", "template", "spec")
	ns, _ := p4s4Get(pod, "nodeSelector").(map[string]any)
	t.Logf("[render] worker nodeSelector = %v (spec: exactly %s=workers, no %s)", ns, p4s4PoolKey, p4s4GPUKey)
	if p4s4Str(ns, p4s4PoolKey) != "workers" {
		t.Errorf("nodeSelector[%s] = %q, want workers", p4s4PoolKey, p4s4Str(ns, p4s4PoolKey))
	}
	if _, has := ns[p4s4GPUKey]; has {
		t.Errorf("nodeSelector still has %s", p4s4GPUKey)
	}
	if len(ns) != 1 {
		t.Errorf("nodeSelector has %d keys, want only the pool label: %v", len(ns), ns)
	}
	tols, _ := p4s4Get(pod, "tolerations").([]any)
	t.Logf("[render] worker tolerations = %v (spec: none for %s)", tols, p4s4GPUKey)
	for _, tol := range tols {
		if p4s4Str(tol, "key") == p4s4GPUKey {
			t.Errorf("worker still tolerates %s: %v", p4s4GPUKey, tol)
		}
	}
	// No pod template anywhere asks for the fake card.
	for key, obj := range ov {
		cs, _ := p4s4Get(obj, "spec", "template", "spec", "containers").([]any)
		for _, c := range cs {
			for _, kind := range []string{"requests", "limits"} {
				if p4s4Get(c, "resources", kind, p4s4CardRes) != nil {
					t.Errorf("%s container %s still has %s in %s", key, p4s4Str(c, "name"), p4s4CardRes, kind)
				}
			}
		}
	}
	affinity := p4s4Get(pod, "affinity")
	t.Logf("[render] worker affinity = %v; topologySpreadConstraints = %v", affinity, p4s4Get(pod, "topologySpreadConstraints"))
	if strings.Contains(fmt.Sprint(affinity), p4s4GPUKey) {
		t.Errorf("worker affinity still refers to %s: %v", p4s4GPUKey, affinity)
	}
}

// Decision 25 / brief: one worker per n2-standard-8 node: CPU request 7 (so
// two cannot fit on 8 vCPUs), memory limit 6Gi.
func TestXcP4S4RenderWorkerResources(t *testing.T) {
	ov, _ := p4s4Render(t)
	c := p4s4Container(t, p4s4Obj(t, ov, "Deployment/worker"), "worker")
	res := p4s4Get(c, "resources")
	t.Logf("[render] worker resources = %v (spec: requests.cpu 7, limits.memory 6Gi; decision 25)", res)
	if got := p4s4Str(res, "requests", "cpu"); got != "7" {
		t.Errorf("requests.cpu = %q, want 7", got)
	}
	if got := p4s4Str(res, "limits", "memory"); got != "6Gi" {
		t.Errorf("limits.memory = %q, want 6Gi", got)
	}
	if lc := p4s4Str(res, "limits", "cpu"); lc != "" {
		t.Logf("[render] note: limits.cpu = %s (the spec names none)", lc)
	}
	t.Logf("[render] requests.memory = %q (the spec does not name it; decision 22 had 512Mi with a 2Gi limit)", p4s4Str(res, "requests", "memory"))
}

// Decision 25 / brief: the scheduler and the store are internal load
// balancers, with no fixed nodePort left over from the kind base.
func TestXcP4S4RenderInternalLoadBalancers(t *testing.T) {
	ov, _ := p4s4Render(t)
	for _, name := range []string{"scheduler", "objectstore"} {
		svc := p4s4Obj(t, ov, "Service/"+name)
		typ := p4s4Str(svc, "spec", "type")
		ann := p4s4Str(svc, "metadata", "annotations", p4s4ILBKey)
		ports, _ := p4s4Get(svc, "spec", "ports").([]any)
		t.Logf("[render] Service/%s type=%s %s=%q ports=%v (spec: LoadBalancer, Internal, no nodePort)", name, typ, p4s4ILBKey, ann, ports)
		if typ != "LoadBalancer" {
			t.Errorf("Service/%s type = %q, want LoadBalancer", name, typ)
		}
		if ann != "Internal" {
			t.Errorf("Service/%s annotation %s = %q, want Internal", name, p4s4ILBKey, ann)
		}
		if len(ports) == 0 {
			t.Errorf("Service/%s has no ports", name)
		}
		for _, p := range ports {
			if np := p4s4Get(p, "nodePort"); np != nil {
				t.Errorf("Service/%s port %s still fixes nodePort %v", name, p4s4Str(p, "name"), np)
			}
		}
	}
	// No other Service is exposed outside the cluster.
	for key, obj := range ov {
		if strings.HasPrefix(key, "Service/") && key != "Service/scheduler" && key != "Service/objectstore" {
			typ := p4s4Str(obj, "spec", "type")
			t.Logf("[render] %s type=%q", key, typ)
			if typ == "LoadBalancer" || typ == "NodePort" {
				t.Errorf("%s is exposed (%s); only the scheduler and the store should be", key, typ)
			}
		}
	}
}

// Decision 25 / brief: real builds at time scale 1, hnswlib with 8 threads.
func TestXcP4S4RenderConfigMapsRealBuilds(t *testing.T) {
	ov, _ := p4s4Render(t)
	want := map[string]map[string]string{
		"ConfigMap/worker":    {"FAKE_TIME_SCALE": "1", "BUILDER": "hnswlib", "BUILD_THREADS": "8"},
		"ConfigMap/scheduler": {"FAKE_TIME_SCALE": "1"},
	}
	for key, kv := range want {
		data := p4s4Get(p4s4Obj(t, ov, key), "data")
		t.Logf("[render] %s data = %v", key, data)
		for k, v := range kv {
			if got := p4s4Str(data, k); got != v {
				t.Errorf("%s %s = %q, want %q", key, k, got, v)
			}
		}
	}
}

// Decision 31(b): the store's pod gets fsGroup 10001 so RustFS can write the
// GKE persistent disk.
func TestXcP4S4RenderObjectstoreFsGroup(t *testing.T) {
	ov, _ := p4s4Render(t)
	sts := p4s4Obj(t, ov, "StatefulSet/objectstore")
	sc := p4s4Get(sts, "spec", "template", "spec", "securityContext")
	t.Logf("[render] StatefulSet/objectstore pod securityContext = %v (decision 31b: fsGroup 10001)", sc)
	if got := p4s4Str(sc, "fsGroup"); got != "10001" {
		t.Errorf("fsGroup = %q, want 10001", got)
	}
}

// Brief: everything namespaced lands in kgpu, and the Namespace is kgpu.
func TestXcP4S4RenderEverythingInKgpu(t *testing.T) {
	ov, _ := p4s4Render(t)
	var keys []string
	for k := range ov {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	t.Logf("[render] overlay objects: %v", keys)
	for _, k := range keys {
		o := ov[k]
		if p4s4Str(o, "kind") == "Namespace" {
			if n := p4s4Str(o, "metadata", "name"); n != p4s4NS {
				t.Errorf("Namespace object is %q, want %s", n, p4s4NS)
			}
			continue
		}
		if ns := p4s4Str(o, "metadata", "namespace"); ns != p4s4NS {
			t.Errorf("%s namespace = %q, want %s", k, ns, p4s4NS)
		}
	}
}

// Brief: the base's other objects are still there and unchanged: the Postgres
// StatefulSet, the Secrets, the worker Service. And the overlay drops nothing.
func TestXcP4S4RenderBaseObjectsUntouched(t *testing.T) {
	ov, base := p4s4Render(t)
	for k := range base {
		if _, ok := ov[k]; !ok {
			t.Errorf("base object %s is missing from the overlay", k)
		}
	}
	for k := range ov {
		if _, ok := base[k]; !ok {
			t.Logf("[render] overlay adds %s (not in the base)", k)
		}
	}
	for _, k := range []string{"StatefulSet/postgres", "Secret/postgres", "Secret/objectstore", "Service/worker", "Service/postgres", "Namespace/kgpu"} {
		b, o := base[k], ov[k]
		same := reflect.DeepEqual(b, o)
		t.Logf("[render] %s identical to the base: %v", k, same)
		if b == nil {
			t.Errorf("base has no %s", k)
			continue
		}
		if !same {
			fb, fo := map[string]string{}, map[string]string{}
			p4s4Flatten("", b, fb)
			p4s4Flatten("", o, fo)
			t.Errorf("%s differs from the base: %v", k, p4s4DiffPaths(fb, fo))
		}
	}
}

func p4s4DiffPaths(a, b map[string]string) []string {
	var out []string
	seen := map[string]bool{}
	for k, v := range a {
		seen[k] = true
		if w, ok := b[k]; !ok {
			out = append(out, fmt.Sprintf("%s: %q -> (absent)", k, v))
		} else if w != v {
			out = append(out, fmt.Sprintf("%s: %q -> %q", k, v, w))
		}
	}
	for k, w := range b {
		if !seen[k] {
			out = append(out, fmt.Sprintf("%s: (absent) -> %q", k, w))
		}
	}
	sort.Strings(out)
	return out
}

// Decision 25 and the step 4 notes list what the overlay changes: images, the
// pool label for the fake card and its taint, CPU requests, internal load
// balancers, real builds at time scale 1 (and decision 31b's fsGroup; the
// brief adds the 6Gi memory limit). Every other difference between the
// rendered base and the rendered overlay is a change the design doc does not
// record. This test lists them all and fails if there are any.
func TestXcP4S4RenderOnlyDocumentedChanges(t *testing.T) {
	ov, base := p4s4Render(t)
	documented := []*regexp.Regexp{
		regexp.MustCompile(`^Deployment/(scheduler|worker)\.spec\.template\.spec\.containers\[\d+\]\.image$`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.nodeSelector\.`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.tolerations`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.containers\[\d+\]\.resources\.(requests|limits)\.kgpu\.io/gpu-card$`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.containers\[\d+\]\.resources\.requests\.cpu$`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.containers\[\d+\]\.resources\.limits\.memory$`),
		regexp.MustCompile(`^Service/(scheduler|objectstore)\.spec\.type$`),
		regexp.MustCompile(`^Service/(scheduler|objectstore)\.metadata\.annotations`),
		regexp.MustCompile(`^Service/(scheduler|objectstore)\.spec\.ports\[\d+\]\.nodePort$`),
		regexp.MustCompile(`^ConfigMap/(scheduler|worker)\.data\.FAKE_TIME_SCALE$`),
		regexp.MustCompile(`^ConfigMap/worker\.data\.(BUILDER|BUILD_THREADS)$`),
		regexp.MustCompile(`^StatefulSet/objectstore\.spec\.template\.spec\.securityContext`),
		// decision 33 (this test's F1): the policy, the lease and renew
		// intervals, the worker's memory request, the scheduler's target port
		regexp.MustCompile(`^ConfigMap/scheduler\.data\.POLICY$`),
		regexp.MustCompile(`^ConfigMap/worker\.data\.(LEASE_SECONDS|RENEW_INTERVAL_SECONDS)$`),
		regexp.MustCompile(`^Deployment/worker\.spec\.template\.spec\.containers\[\d+\]\.resources\.requests\.memory$`),
		regexp.MustCompile(`^Service/scheduler\.spec\.ports\[\d+\]\.targetPort$`),
		// decision 36: the in-place rolling update for a one-per-node pool
		regexp.MustCompile(`^Deployment/worker\.spec\.strategy\.`),
		// decision 35 / evidence 7: the calibrated cost constants on both ConfigMaps
		regexp.MustCompile(`^ConfigMap/(scheduler|worker)\.data\.(COST_A|COST_SPEEDUP|COST_GPU_OVERHEAD|COST_BANDWIDTH|KGPU_ESTIMATE_SIGMA)$`),
	}
	var undocumented, doc []string
	for k := range ov {
		fb, fo := map[string]string{}, map[string]string{}
		p4s4Flatten(k, base[k], fb)
		p4s4Flatten(k, ov[k], fo)
		for _, d := range p4s4DiffPaths(fb, fo) {
			path := d[:strings.Index(d, ": ")]
			ok := false
			for _, re := range documented {
				if re.MatchString(path) {
					ok = true
					break
				}
			}
			if ok {
				doc = append(doc, d)
			} else {
				undocumented = append(undocumented, d)
			}
		}
	}
	sort.Strings(doc)
	sort.Strings(undocumented)
	t.Logf("[render] base -> overlay, documented changes (%d):\n  %s", len(doc), strings.Join(doc, "\n  "))
	t.Logf("[render] base -> overlay, changes the design doc does not record (%d):\n  %s", len(undocumented), strings.Join(undocumented, "\n  "))
	if len(undocumented) > 0 {
		t.Errorf("the overlay makes %d change(s) that decisions 25, 31, 33 / the step 4 notes do not list (spec gap unless recorded): %v", len(undocumented), undocumented)
	}
}

// ---------------------------------------------------------------------------
// Live cluster (read-only).

var (
	p4s4LiveOnce   sync.Once
	p4s4LiveReason string
)

func p4s4Env() []string {
	env := os.Environ()
	env = append(env, "USE_GKE_GCLOUD_AUTH_PLUGIN=True")
	for i, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			env[i] = "PATH=/opt/homebrew/share/google-cloud-sdk/bin:" + strings.TrimPrefix(e, "PATH=")
		}
	}
	return env
}

func p4s4Kubectl(ctx context.Context, args ...string) (string, string, error) {
	cmd := exec.CommandContext(ctx, "kubectl", append([]string{"--context", p4s4Context}, args...)...)
	cmd.Env = p4s4Env()
	var so, se bytes.Buffer
	cmd.Stdout, cmd.Stderr = &so, &se
	err := cmd.Run()
	return so.String(), se.String(), err
}

// p4s4Live skips unless the current context is the GKE one and answers in 30 s.
func p4s4Live(t *testing.T) {
	t.Helper()
	p4s4LiveOnce.Do(func() {
		if _, err := exec.LookPath("kubectl"); err != nil {
			p4s4LiveReason = "kubectl not on PATH"
			return
		}
		cur := exec.Command("kubectl", "config", "current-context")
		cur.Env = p4s4Env()
		out, err := cur.Output()
		if err != nil || strings.TrimSpace(string(out)) != p4s4Context {
			p4s4LiveReason = fmt.Sprintf("current kubectl context is %q, not %s", strings.TrimSpace(string(out)), p4s4Context)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, se, err := p4s4Kubectl(ctx, "--request-timeout=25s", "get", "namespace", p4s4NS); err != nil {
			p4s4LiveReason = fmt.Sprintf("GKE cluster not reachable within 30 s: %v %s", err, strings.TrimSpace(se))
		}
	})
	if p4s4LiveReason != "" {
		t.Skip(p4s4LiveReason)
	}
}

func p4s4KJSON(t *testing.T, args ...string) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	so, se, err := p4s4Kubectl(ctx, append(args, "-o", "json")...)
	if err != nil {
		t.Fatalf("kubectl %s: %v %s", strings.Join(args, " "), err, se)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(so), &m); err != nil {
		t.Fatalf("decode kubectl %s: %v", strings.Join(args, " "), err)
	}
	return m
}

func p4s4Items(m map[string]any) []any { x, _ := m["items"].([]any); return x }

type p4s4NodeInfo struct {
	name, pool, itype, cpu string
	internal, external     []string
}

func p4s4Nodes(t *testing.T) []p4s4NodeInfo {
	t.Helper()
	var out []p4s4NodeInfo
	for _, n := range p4s4Items(p4s4KJSON(t, "get", "nodes")) {
		nd := p4s4NodeInfo{
			name:  p4s4Str(n, "metadata", "name"),
			pool:  p4s4Str(n, "metadata", "labels", p4s4PoolKey),
			itype: p4s4Str(n, "metadata", "labels", "node.kubernetes.io/instance-type"),
			cpu:   p4s4Str(n, "status", "capacity", "cpu"),
		}
		addrs, _ := p4s4Get(n, "status", "addresses").([]any)
		for _, a := range addrs {
			switch p4s4Str(a, "type") {
			case "InternalIP":
				nd.internal = append(nd.internal, p4s4Str(a, "address"))
			case "ExternalIP":
				nd.external = append(nd.external, p4s4Str(a, "address"))
			}
		}
		out = append(out, nd)
	}
	return out
}

// Decision 25: one default-pool node (e2-standard-4, 4 vCPU) and two in the
// `workers` pool (n2-standard-8, 8 vCPU each); nothing else.
func TestXcP4S4LiveNodePools(t *testing.T) {
	p4s4Live(t)
	nodes := p4s4Nodes(t)
	count := map[string]int{}
	for _, n := range nodes {
		t.Logf("[live] node %s pool=%s type=%s cpu=%s internal=%v", n.name, n.pool, n.itype, n.cpu, n.internal)
		count[n.pool]++
		switch n.pool {
		case "default-pool":
			if n.cpu != "4" {
				t.Errorf("default-pool node %s has %s CPU, want 4", n.name, n.cpu)
			}
		case "workers":
			if n.cpu != "8" {
				t.Errorf("workers node %s has %s CPU, want 8", n.name, n.cpu)
			}
		default:
			t.Errorf("node %s is in an unexpected pool %q", n.name, n.pool)
		}
	}
	t.Logf("[live] pools: %v (decision 25: default-pool 1, workers 2)", count)
	if count["default-pool"] != 1 || count["workers"] != 2 || len(nodes) != 3 {
		t.Errorf("pool sizes = %v over %d nodes, want default-pool 1 and workers 2", count, len(nodes))
	}
}

// Decision 28: no public IP anywhere; every node's address is private.
func TestXcP4S4LiveNoExternalIPs(t *testing.T) {
	p4s4Live(t)
	_, vpc, _ := net.ParseCIDR(p4s4VPC)
	for _, n := range p4s4Nodes(t) {
		t.Logf("[live] node %s ExternalIP=%v InternalIP=%v (decision 28: no ExternalIP)", n.name, n.external, n.internal)
		if len(n.external) > 0 {
			t.Errorf("node %s has an external IP %v", n.name, n.external)
		}
		for _, ip := range n.internal {
			if !vpc.Contains(net.ParseIP(ip)) {
				t.Errorf("node %s internal IP %s is outside %s", n.name, ip, p4s4VPC)
			}
		}
	}
}

// Decision 25: the two worker pods run one per `workers` node; postgres-0,
// objectstore-0 and the scheduler run on the default pool.
func TestXcP4S4LivePlacement(t *testing.T) {
	p4s4Live(t)
	pool := map[string]string{}
	for _, n := range p4s4Nodes(t) {
		pool[n.name] = n.pool
	}
	workerNodes := map[string]int{}
	var sched, pg, os3 int
	for _, p := range p4s4Items(p4s4KJSON(t, "-n", p4s4NS, "get", "pods")) {
		name := p4s4Str(p, "metadata", "name")
		app := p4s4Str(p, "metadata", "labels", "app")
		node := p4s4Str(p, "spec", "nodeName")
		phase := p4s4Str(p, "status", "phase")
		t.Logf("[live] pod %s app=%s phase=%s node=%s pool=%s", name, app, phase, node, pool[node])
		if phase != "Running" {
			continue
		}
		switch app {
		case "worker":
			workerNodes[node]++
			if pool[node] != "workers" {
				t.Errorf("worker pod %s runs on %s in pool %q, want workers", name, node, pool[node])
			}
		case "scheduler", "postgres", "objectstore":
			if pool[node] != "default-pool" {
				t.Errorf("%s pod %s runs in pool %q, want default-pool", app, name, pool[node])
			}
			switch app {
			case "scheduler":
				sched++
			case "postgres":
				pg++
			case "objectstore":
				os3++
			}
		}
	}
	t.Logf("[live] running worker pods per node: %v; scheduler=%d postgres=%d objectstore=%d", workerNodes, sched, pg, os3)
	if len(workerNodes) != 2 {
		t.Errorf("running workers sit on %d nodes, want 2 (one per workers node): %v", len(workerNodes), workerNodes)
	}
	for n, c := range workerNodes {
		if c != 1 {
			t.Errorf("node %s runs %d workers, want 1", n, c)
		}
	}
	if sched != 1 || pg != 1 || os3 != 1 {
		t.Errorf("running scheduler/postgres/objectstore = %d/%d/%d, want 1/1/1", sched, pg, os3)
	}
}

func p4s4ILBs(t *testing.T) map[string]string {
	t.Helper()
	_, vpc, _ := net.ParseCIDR(p4s4VPC)
	ips := map[string]string{}
	for _, name := range []string{"scheduler", "objectstore"} {
		svc := p4s4KJSON(t, "-n", p4s4NS, "get", "service", name)
		ing, _ := p4s4Get(svc, "status", "loadBalancer", "ingress").([]any)
		typ := p4s4Str(svc, "spec", "type")
		ann := p4s4Str(svc, "metadata", "annotations", p4s4ILBKey)
		t.Logf("[live] Service/%s type=%s %s=%q ingress=%v", name, typ, p4s4ILBKey, ann, ing)
		if typ != "LoadBalancer" || ann != "Internal" {
			t.Errorf("Service/%s type=%q annotation=%q, want LoadBalancer/Internal", name, typ, ann)
		}
		if len(ing) != 1 {
			t.Errorf("Service/%s has %d ingress entries, want 1", name, len(ing))
			continue
		}
		ip := p4s4Str(ing[0], "ip")
		if parsed := net.ParseIP(ip); parsed == nil || !vpc.Contains(parsed) {
			t.Errorf("Service/%s ingress IP %q is not in %s", name, ip, p4s4VPC)
		}
		ips[name] = ip
	}
	return ips
}

// Decision 25: the scheduler and the store each have a private ingress IP in
// the VPC's subnet.
func TestXcP4S4LiveInternalLBIPs(t *testing.T) {
	p4s4Live(t)
	ips := p4s4ILBs(t)
	t.Logf("[live] internal LB IPs: %v (decision 25: private, in %s)", ips, p4s4VPC)
}

// Decisions 25 and 28: the scheduler's private IP answers inside the VPC and
// not from the laptop. Inside: a worker pod (it has Python; the scheduler
// image has no shell) GETs /healthz on the scheduler's ILB IP and /health on
// the store's. Outside: the laptop's own GET must fail.
func TestXcP4S4LiveSchedulerPrivateFromLaptop(t *testing.T) {
	p4s4Live(t)
	ips := p4s4ILBs(t)
	if ips["scheduler"] == "" {
		t.Fatal("no scheduler IP")
	}
	url := "http://" + ips["scheduler"] + ":8080/healthz"
	cl := &http.Client{Timeout: 5 * time.Second}
	resp, err := cl.Get(url)
	if err == nil {
		resp.Body.Close()
		t.Errorf("laptop GET %s answered %d; decision 28 says the scheduler is reachable only inside the VPC", url, resp.StatusCode)
	} else {
		t.Logf("[live] laptop GET %s failed as expected: %v", url, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	// The scheduler pod itself: the brief allows exec "or" port-forward. Record
	// whether exec can run anything there at all.
	_, se, err := p4s4Kubectl(ctx, "-n", p4s4NS, "exec", "deploy/scheduler", "--", "sh", "-c", "true")
	t.Logf("[live] exec into the scheduler pod: err=%v %s (an image without a shell cannot curl from inside)", err, strings.TrimSpace(se))

	py := fmt.Sprintf(`import urllib.request as u
print(u.urlopen("http://%s:8080/healthz", timeout=5).read().decode().strip())
print(u.urlopen("http://%s:9000/health", timeout=5).status)`, ips["scheduler"], ips["objectstore"])
	so, se, err := p4s4Kubectl(ctx, "-n", p4s4NS, "exec", "deploy/worker", "--", "python", "-c", py)
	t.Logf("[live] from a worker pod inside the VPC: GET scheduler /healthz and store /health -> %q err=%v %s", strings.TrimSpace(so), err, strings.TrimSpace(se))
	if err != nil {
		t.Errorf("in-VPC GET of the private IPs failed: %v %s", err, se)
	} else if !strings.Contains(so, `"ok"`) || !strings.Contains(so, "200") {
		t.Errorf("in-VPC answers = %q, want the scheduler's ok and the store's 200", so)
	}
}

// p4s4PortForward forwards a local port to the scheduler Service and waits for
// /healthz.
func p4s4PortForward(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	cmd := exec.Command("kubectl", "--context", p4s4Context, "-n", p4s4NS, "port-forward", "svc/scheduler", fmt.Sprintf("%d:8080", port))
	cmd.Env = p4s4Env()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	cl := &http.Client{Timeout: 3 * time.Second}
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if r, err := cl.Get(base + "/healthz"); err == nil {
			r.Body.Close()
			t.Logf("[live] port-forward %s -> svc/scheduler:8080, /healthz %d", base, r.StatusCode)
			if r.StatusCode == 200 {
				return base
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatalf("port-forward to the scheduler never answered /healthz 200")
	return ""
}

// API table GET /workers (phase 1), decision 21 of phase 2 (live flag), brief:
// two live workers with mem_bytes 24 GiB, on the workers pool, building with
// hnswlib. WorkersResponse carries no builder field (go doc), so the builder is
// read from each worker's own startup line instead.
func TestXcP4S4LiveWorkersViaPortForward(t *testing.T) {
	p4s4Live(t)
	base := p4s4PortForward(t)
	resp, err := http.Get(base + "/workers")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var raw map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		t.Fatal(err)
	}
	t.Logf("[live] GET /workers -> %d %v", resp.StatusCode, raw)
	pool := map[string]string{}
	for _, n := range p4s4Nodes(t) {
		pool[n.name] = n.pool
	}
	const gib24 = int64(24) << 30
	if got := p4s4Str(raw, "pool", "live_workers"); got != "2" {
		t.Errorf("pool.live_workers = %s, want 2", got)
	}
	ws, _ := raw["workers"].([]any)
	live := 0
	for _, w := range ws {
		if b, _ := p4s4Get(w, "live").(bool); !b {
			t.Logf("[live] stale registration %s (kept by decision 21)", p4s4Str(w, "worker_id"))
			continue
		}
		live++
		mem, _ := p4s4Get(w, "mem_bytes").(float64)
		node := p4s4Str(w, "node")
		t.Logf("[live] live worker %s node=%s (pool %s) mem_bytes=%.0f", p4s4Str(w, "worker_id"), node, pool[node], mem)
		if int64(mem) != gib24 {
			t.Errorf("worker %s mem_bytes = %.0f, want %d (24 GiB)", p4s4Str(w, "worker_id"), mem, gib24)
		}
		if pool[node] != "workers" {
			t.Errorf("worker %s reports node %q in pool %q, want the workers pool", p4s4Str(w, "worker_id"), node, pool[node])
		}
		if b := p4s4Get(w, "builder"); b != nil && fmt.Sprint(b) != "hnswlib" {
			t.Errorf("worker %s builder = %v, want hnswlib", p4s4Str(w, "worker_id"), b)
		}
	}
	if live != 2 {
		t.Errorf("%d live workers, want 2", live)
	}
	if len(ws) > 0 && p4s4Get(ws[0], "builder") == nil {
		t.Logf("[live] GET /workers has no builder field (WorkersResponse/store.Worker define none): spec gap for the brief's 'builder hnswlib'; reading the workers' startup lines")
	}
	// The builder each running worker announced at startup.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	for _, p := range p4s4Items(p4s4KJSON(t, "-n", p4s4NS, "get", "pods", "-l", "app=worker")) {
		name := p4s4Str(p, "metadata", "name")
		so, se, err := p4s4Kubectl(ctx, "-n", p4s4NS, "logs", name)
		if err != nil {
			t.Errorf("logs %s: %v %s", name, err, se)
			continue
		}
		var start map[string]any
		for _, line := range strings.Split(so, "\n") {
			var m map[string]any
			if json.Unmarshal([]byte(line), &m) == nil && m["msg"] == "worker starting" {
				start = m
				break
			}
		}
		t.Logf("[live] %s startup line: %v", name, start)
		if start == nil {
			t.Errorf("%s: no 'worker starting' line in its log", name)
			continue
		}
		if fmt.Sprint(start["builder"]) != "hnswlib" {
			t.Errorf("%s started with builder %v, want hnswlib", name, start["builder"])
		}
		if mem, _ := start["mem_bytes"].(float64); int64(mem) != gib24 {
			t.Errorf("%s started with mem_bytes %v, want 24 GiB", name, start["mem_bytes"])
		}
	}
}

// The live cluster runs what the checked-in overlay renders: images, the two
// ConfigMaps, the worker's resources and node selector, the Services' type and
// annotation. A drift means the committed overlay is not what was measured.
func TestXcP4S4LiveMatchesRender(t *testing.T) {
	p4s4Live(t)
	ov, _ := p4s4Render(t)
	for _, name := range []string{"scheduler", "worker"} {
		want := p4s4Get(p4s4Obj(t, ov, "ConfigMap/"+name), "data")
		got := p4s4Get(p4s4KJSON(t, "-n", p4s4NS, "get", "configmap", name), "data")
		fw, fg := map[string]string{}, map[string]string{}
		p4s4Flatten("", want, fw)
		p4s4Flatten("", got, fg)
		d := p4s4DiffPaths(fw, fg)
		t.Logf("[live] ConfigMap/%s live vs rendered: %d differences %v", name, len(d), d)
		if len(d) > 0 {
			t.Errorf("ConfigMap/%s drifted from the overlay: %v", name, d)
		}
	}
	for _, name := range []string{"scheduler", "worker"} {
		r := p4s4Obj(t, ov, "Deployment/"+name)
		l := p4s4KJSON(t, "-n", p4s4NS, "get", "deployment", name)
		rc := p4s4Container(t, r, name)
		lc := p4s4Container(t, l, name)
		for _, field := range []string{"image"} {
			t.Logf("[live] Deployment/%s %s live=%s rendered=%s", name, field, p4s4Str(lc, field), p4s4Str(rc, field))
			if p4s4Str(lc, field) != p4s4Str(rc, field) {
				t.Errorf("Deployment/%s %s drifted", name, field)
			}
		}
		fr, fl := map[string]string{}, map[string]string{}
		p4s4Flatten("", p4s4Get(rc, "resources"), fr)
		p4s4Flatten("", p4s4Get(lc, "resources"), fl)
		if d := p4s4DiffPaths(fr, fl); len(d) > 0 {
			t.Errorf("Deployment/%s resources drifted: %v", name, d)
		} else {
			t.Logf("[live] Deployment/%s resources match the render: %v", name, p4s4Get(lc, "resources"))
		}
		rs := fmt.Sprint(p4s4Get(r, "spec", "template", "spec", "nodeSelector"))
		ls := fmt.Sprint(p4s4Get(l, "spec", "template", "spec", "nodeSelector"))
		if rs != ls {
			t.Errorf("Deployment/%s nodeSelector live=%s rendered=%s", name, ls, rs)
		}
	}
	sts := p4s4KJSON(t, "-n", p4s4NS, "get", "statefulset", "objectstore")
	fg := p4s4Str(sts, "spec", "template", "spec", "securityContext", "fsGroup")
	t.Logf("[live] StatefulSet/objectstore fsGroup=%s (decision 31b: 10001)", fg)
	if fg != "10001" {
		t.Errorf("live objectstore fsGroup = %q, want 10001", fg)
	}
	for _, name := range []string{"scheduler", "objectstore"} {
		svc := p4s4KJSON(t, "-n", p4s4NS, "get", "service", name)
		ports, _ := p4s4Get(svc, "spec", "ports").([]any)
		for _, p := range ports {
			t.Logf("[live] Service/%s port %s: port=%s nodePort=%s (auto-allocated by Kubernetes for a LoadBalancer; not fixed by the overlay)", name, p4s4Str(p, "name"), p4s4Str(p, "port"), p4s4Str(p, "nodePort"))
		}
	}
}

// ---------------------------------------------------------------------------
// The sweep runner's calibrated-constant flags. Reuses swRun/swParse from
// sweep_test.go (the Phase 3 step 5 cross-check).

func p4s4Float(t *testing.T, r swRow, k string) float64 {
	t.Helper()
	if _, ok := r[k]; !ok {
		t.Fatalf("CSV has no column %q", k)
	}
	return r.f(t, k)
}

// Brief: without the flags, cost_a, bandwidth and gpu_overhead echo
// costmodel.Default() (and speedup its Speedup).
func TestXcP4S4SweepDefaultsEcho(t *testing.T) {
	d := costmodel.Default()
	b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "2")
	if err != nil {
		t.Fatal(err)
	}
	hdr, rows := swParse(t, b)
	t.Logf("[sweep] header: %v", hdr)
	t.Logf("[sweep] costmodel.Default() = A %g, Bandwidth %g, GPUOverhead %g, Speedup %g", d.A, d.Bandwidth, d.GPUOverhead, d.Speedup)
	for _, r := range rows {
		got := [4]float64{p4s4Float(t, r, "cost_a"), p4s4Float(t, r, "bandwidth"), p4s4Float(t, r, "gpu_overhead"), p4s4Float(t, r, "speedup")}
		want := [4]float64{d.A, d.Bandwidth, d.GPUOverhead, d.Speedup}
		t.Logf("[sweep] seed %s echoes cost_a=%g bandwidth=%g gpu_overhead=%g speedup=%g", r["seed"], got[0], got[1], got[2], got[3])
		if got != want {
			t.Errorf("seed %s: echoed %v, want Default() %v", r["seed"], got, want)
		}
	}
}

// Brief: the step 3 rehearsal's constants are echoed exactly.
func TestXcP4S4SweepCalibratedEcho(t *testing.T) {
	b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive,static_threshold", "-rounds", "2",
		"-cost_a", "6.08e-8", "-bandwidth", "1.81e8", "-gpu_overhead", "1.18", "-speedup", "2.05")
	if err != nil {
		t.Fatal(err)
	}
	_, rows := swParse(t, b)
	if len(rows) != 4 {
		t.Errorf("%d rows, want 4 (2 policies x 2 seeds)", len(rows))
	}
	for _, r := range rows {
		got := [4]float64{p4s4Float(t, r, "cost_a"), p4s4Float(t, r, "bandwidth"), p4s4Float(t, r, "gpu_overhead"), p4s4Float(t, r, "speedup")}
		t.Logf("[sweep] %s seed %s echoes cost_a=%s bandwidth=%s gpu_overhead=%s speedup=%s; round_s=%s local=%s gpu=%s",
			r["policy"], r["seed"], r["cost_a"], r["bandwidth"], r["gpu_overhead"], r["speedup"], r["round_s"], r["builds_local"], r["builds_gpu"])
		if got != [4]float64{6.08e-8, 1.81e8, 1.18, 2.05} {
			t.Errorf("echoed %v, want [6.08e-8 1.81e8 1.18 2.05]", got)
		}
	}
}

// Brief: two runs differing only in -cost_a differ in round_s when there are
// local builds. static_threshold on skewed places five of six builds local;
// a larger A (slower shard CPU) cannot make the round shorter.
func TestXcP4S4SweepCostAMovesRoundS(t *testing.T) {
	common := []string{"-scenarios", "skewed", "-policies", "static_threshold", "-rounds", "3"}
	run := func(a string) []swRow {
		b, err := swRun(t, append(append([]string(nil), common...), "-cost_a", a)...)
		if err != nil {
			t.Fatal(err)
		}
		_, rows := swParse(t, b)
		return rows
	}
	lo, hi := run("6.08e-8"), run("1.4e-7")
	if len(lo) != len(hi) || len(lo) == 0 {
		t.Fatalf("row counts %d vs %d", len(lo), len(hi))
	}
	differ := 0
	for i := range lo {
		l, h := lo[i], hi[i]
		t.Logf("[sweep] seed %s: cost_a %s -> round_s %s (local %s) | cost_a %s -> round_s %s (local %s)",
			l["seed"], l["cost_a"], l["round_s"], l["builds_local"], h["cost_a"], h["round_s"], h["builds_local"])
		if l["seed"] != h["seed"] {
			t.Fatalf("rows out of step: seed %s vs %s", l["seed"], h["seed"])
		}
		if p4s4Float(t, l, "builds_local") == 0 || p4s4Float(t, h, "builds_local") == 0 {
			t.Errorf("seed %s has no local build; the comparison needs one", l["seed"])
		}
		if l["round_s"] != h["round_s"] {
			differ++
		}
		if p4s4Float(t, h, "round_s") < p4s4Float(t, l, "round_s") {
			t.Errorf("seed %s: a slower shard CPU (larger A) made the round shorter: %s < %s", l["seed"], h["round_s"], l["round_s"])
		}
		// Nothing but cost_a and the outcome columns may change.
		for _, k := range []string{"scenario", "policy", "n_shards", "dim", "workers", "speedup", "bandwidth", "gpu_overhead", "sigma", "zipf"} {
			if l[k] != h[k] {
				t.Errorf("seed %s: column %s changed %s -> %s with only -cost_a changed", l["seed"], k, l[k], h[k])
			}
		}
	}
	t.Logf("[sweep] %d of %d seeds differ in round_s", differ, len(lo))
	if differ == 0 {
		t.Errorf("no seed's round_s moved when only -cost_a changed")
	}
}

// Brief: the CSV echoes "the cost model's effective values". An out-of-range
// override (negative A, negative bandwidth, a negative overhead other than
// the documented -1 sentinel) is a typo or a sign error from a fit; the spec
// does not say what happens. Acceptable: refused (non-zero exit), or honoured
// and echoed. Not acceptable: silently replaced by the default, which runs
// and labels a sweep the caller did not ask for.
func TestXcP4S4SweepOutOfRangeOverrideNotSilentlyReplaced(t *testing.T) {
	d := costmodel.Default()
	cases := []struct {
		flag, val, col string
		def            float64
	}{
		{"-cost_a", "-1e-7", "cost_a", d.A},
		{"-bandwidth", "-5", "bandwidth", d.Bandwidth},
		{"-gpu_overhead", "-2", "gpu_overhead", d.GPUOverhead},
	}
	for _, c := range cases {
		b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive", "-rounds", "1", c.flag, c.val)
		if err != nil {
			t.Logf("[sweep] %s %s refused: %v (acceptable)", c.flag, c.val, err)
			continue
		}
		_, rows := swParse(t, b)
		for _, r := range rows {
			got := p4s4Float(t, r, c.col)
			asked, _ := strconv.ParseFloat(c.val, 64)
			t.Logf("[sweep] %s %s accepted; CSV %s = %g (default %g), round_s=%s", c.flag, c.val, c.col, got, c.def, r["round_s"])
			if got != asked {
				t.Errorf("%s %s ran with %s=%g (the default) and exit 0: the override was silently dropped", c.flag, c.val, c.col, got)
			}
		}
	}
}
