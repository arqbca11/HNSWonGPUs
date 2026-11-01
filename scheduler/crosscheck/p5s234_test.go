package crosscheck

// Phase 5 steps 2 and 3 cross-check (GPU-free), black-box.
//
// Spec: docs/design/phase5-gpu.md decisions 9 (the GPU overlay), 11 (the cold
// start) and the "Steps 2–4 (GPU-free)" notes; docs/roadmap.md Phase 5
// "Before the GPUs"; the caller's brief; `go doc` of experiments/sim.Config,
// scheduler/api.Config, scheduler/reconsider.Config, scheduler/policy and
// scheduler/cmd/scheduler (COLD_START); `sweep -h` (-cold_start_s).
// No implementation source was read. Step 4 (the cuVS scaffold) is checked by
// builder/tests/test_xcheck_cagra.py and worker/tests/test_xcheck_cagra.py.
//
// Groups:
//   - TestXcP5S2Render*: `kubectl kustomize deploy/k8s-gcp-gpu` and
//     `deploy/k8s-gcp`, parsed with p4s4ParseYAML. Needs only kubectl.
//   - TestXcP5S3Sim*, TestXcP5S3Sweep*: the simulator and the sweep (pure).
//   - TestXcP5S3API*, TestXcP5S3Reconsider*, TestXcP5S3Scheduler*: Postgres
//     (TEST_DATABASE_URL), skipped without it.
//
// Hand numbers (cost model v0, Default(), n = 100k, dim = 128):
//   CPU build 20.6314 s; GPU path = 20.6314/10 + 2 + 0.576 = 4.6391 s.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/scheduler/api"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	rc "github.com/arqbca11/HNSWonGPUs/scheduler/reconsider"
)

// =========================================================================
// Step 2: the GPU overlay (decision 9)

const (
	p5GPUImage  = "us-west1-docker.pkg.dev/project-3456ed25-5d5d-4b0d-aa5/kgpu/kgpu-worker-gpu:p5"
	p5AccelKey  = "cloud.google.com/gke-accelerator"
	p5AccelVal  = "nvidia-l4"
	p5GPURes    = "nvidia.com/gpu"
	p5WorkerMem = "24159191040" // 22.5 GiB, an L4's usable memory (decision 9 as amended after this test's G2)
)

var (
	p5RenderOnce sync.Once
	p5RenderRaw  [2]string // 0: k8s-gcp-gpu, 1: k8s-gcp
	p5RenderErr  error
)

func p5Render(t *testing.T) (gpu, gcp map[string]map[string]any, rawGPU, rawGCP string) {
	t.Helper()
	if _, err := exec.LookPath("kubectl"); err != nil {
		t.Skip("kubectl not on PATH; the render tests need its built-in kustomize")
	}
	p5RenderOnce.Do(func() {
		root := repoRoot(t)
		for i, dir := range []string{"deploy/k8s-gcp-gpu", "deploy/k8s-gcp"} {
			var stderr bytes.Buffer
			cmd := exec.Command("kubectl", "kustomize", filepath.Join(root, dir))
			cmd.Stderr = &stderr
			out, err := cmd.Output()
			if err != nil {
				p5RenderErr = fmt.Errorf("kubectl kustomize %s: %v %s", dir, err, stderr.String())
				return
			}
			p5RenderRaw[i] = string(out)
		}
	})
	if p5RenderErr != nil {
		t.Fatal(p5RenderErr)
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
	return index(p5RenderRaw[0]), index(p5RenderRaw[1]), p5RenderRaw[0], p5RenderRaw[1]
}

func p5Keys(m map[string]map[string]any) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

func TestXcP5S2RenderWorkerDeployment(t *testing.T) {
	gpu, _, _, _ := p5Render(t)
	t.Logf("[render] deploy/k8s-gcp-gpu objects: %v", p5Keys(gpu))
	w := p4s4Obj(t, gpu, "Deployment/worker")
	c := p4s4Container(t, w, "worker")
	img := p4s4Str(c, "image")
	lim := p4s4Str(c, "resources", "limits", p5GPURes)
	req := p4s4Str(c, "resources", "requests", p5GPURes)
	ns, _ := p4s4Get(w, "spec", "template", "spec", "nodeSelector").(map[string]any)
	tols, _ := p4s4Get(w, "spec", "template", "spec", "tolerations").([]any)
	reps := p4s4Str(w, "spec", "replicas")
	t.Logf("[render] worker: image=%s limits[%s]=%q requests[%s]=%q nodeSelector=%v tolerations=%v replicas=%s",
		img, p5GPURes, lim, p5GPURes, req, ns, tols, reps)
	t.Logf("[render] worker resources (all): %v", p4s4Get(c, "resources"))
	t.Logf("spec: decision 9 ('the cuVS image (kgpu-worker-gpu:p5), nvidia.com/gpu: 1 in place of the CPU pool's selector ... GKE's accelerator label selects the pool family and its nvidia.com/gpu=present taint is tolerated'); the brief: replicas 2")
	if img != p5GPUImage {
		t.Errorf("worker image %q, want %q", img, p5GPUImage)
	}
	if lim != "1" {
		t.Errorf("resources.limits[%s] = %q, want 1", p5GPURes, lim)
	}
	if req != "" && req != "1" {
		t.Errorf("resources.requests[%s] = %q: an extended resource's request must equal its limit", p5GPURes, req)
	}
	if !reflect.DeepEqual(ns, map[string]any{p5AccelKey: p5AccelVal}) {
		t.Errorf("nodeSelector = %v, want exactly {%s: %s} (no gke-nodepool, no kgpu.io/gpu)", ns, p5AccelKey, p5AccelVal)
	}
	found := false
	for _, x := range tols {
		if p4s4Str(x, "key") == p5GPURes && p4s4Str(x, "value") == "present" && p4s4Str(x, "effect") == "NoSchedule" {
			op := p4s4Str(x, "operator")
			found = op == "" || op == "Equal"
		}
		if p4s4Str(x, "key") == p4s4GPUKey {
			t.Errorf("the GPU worker still tolerates the kind cluster's %s taint: %v", p4s4GPUKey, x)
		}
	}
	if !found {
		t.Errorf("no toleration {key %s, value present, effect NoSchedule} on the worker: %v", p5GPURes, tols)
	}
	if reps != "0" {
		t.Errorf("replicas = %q, want 0 (decision 11 as amended after this test's G1: the pool is cold between sessions, a session scales up)", reps)
	}
	t.Logf("note: decision 11 says 'the GPU overlay is applied with the workers scaled to zero and scaled up at a session's start', but the rendered Deployment says replicas %s: `kubectl apply` of this overlay sets 2 replicas, i.e. it asks the autoscaler for GPU nodes immediately. Whatever scales it to zero is outside the manifest (spec tension, reported)", reps)
}

func TestXcP5S2RenderConfigMaps(t *testing.T) {
	gpu, gcp, _, _ := p5Render(t)
	wg := p4s4Obj(t, gpu, "ConfigMap/worker")
	sg := p4s4Obj(t, gpu, "ConfigMap/scheduler")
	t.Logf("[render] k8s-gcp-gpu ConfigMap/worker data: %v", p4s4Get(wg, "data"))
	t.Logf("[render] k8s-gcp-gpu ConfigMap/scheduler data: %v", p4s4Get(sg, "data"))
	t.Logf("spec: decision 9 (BUILDER=cagra, a real card's memory, 'the cluster's cost configuration becomes COST_MODEL=calibrated (v1's block) in both overlays'); brief: BUILD_THREADS \"6\", WORKER_MEM_BYTES \"25769803776\"")
	want := map[string]string{"BUILDER": "cagra", "BUILD_THREADS": "6", "WORKER_MEM_BYTES": p5WorkerMem, "COST_MODEL": "calibrated"}
	for k, v := range want {
		if g := p4s4Str(wg, "data", k); g != v {
			t.Errorf("GPU worker ConfigMap %s = %q, want %q", k, g, v)
		}
	}
	if g := p4s4Str(sg, "data", "COST_MODEL"); g != "calibrated" {
		t.Errorf("GPU scheduler ConfigMap COST_MODEL = %q, want calibrated", g)
	}
	for name, objs := range map[string]map[string]map[string]any{"k8s-gcp": gcp, "k8s-gcp-gpu": gpu} {
		for _, cm := range []string{"ConfigMap/scheduler", "ConfigMap/worker"} {
			o := p4s4Obj(t, objs, cm)
			data, _ := p4s4Get(o, "data").(map[string]any)
			var costKeys []string
			for k := range data {
				if strings.HasPrefix(k, "COST_") {
					costKeys = append(costKeys, k)
				}
			}
			sort.Strings(costKeys)
			t.Logf("[render] %s %s: COST_* keys %v, COST_MODEL=%q, COLD_START=%q", name, cm, costKeys, p4s4Str(o, "data", "COST_MODEL"), p4s4Str(o, "data", "COLD_START"))
			if p4s4Str(o, "data", "COST_MODEL") != "calibrated" {
				t.Errorf("%s %s: COST_MODEL = %q, want calibrated (decision 9)", name, cm, p4s4Str(o, "data", "COST_MODEL"))
			}
			if len(costKeys) != 1 {
				t.Errorf("%s %s carries per-constant overrides %v besides COST_MODEL: the block is now Calibrated(), the old COST_A/COST_SPEEDUP/... must be gone", name, cm, costKeys)
			}
		}
	}
	t.Logf("note: WORKER_MEM_BYTES %s is exactly 24 GiB and is the same value the CPU overlay already carries; decision 9 says the GPU overlay sets 'a real card's memory', but an L4's usable memory is 24 GB as marketed (about 22.5 GiB as the driver reports it). The modeled capacity is therefore about 1.5 GiB above the card: a build modeled between ~22.5 and 24 GiB would be claimed by a worker whose card cannot hold it (spec gap until the session measures it, reported)", p5WorkerMem)
}

func TestXcP5S2RenderDCGMExporter(t *testing.T) {
	gpu, gcp, _, _ := p5Render(t)
	ds := p4s4Obj(t, gpu, "DaemonSet/dcgm-exporter")
	svc := p4s4Obj(t, gpu, "Service/dcgm-exporter")
	t.Logf("spec: decision 9 ('adds NVIDIA's DCGM exporter as a DaemonSet on the GPU nodes for the Phase 2 dashboard'); brief: namespace kgpu, the accelerator selector and toleration, container port 9400, Service port 9400")
	ns := p4s4Str(ds, "metadata", "namespace")
	sel, _ := p4s4Get(ds, "spec", "template", "spec", "nodeSelector").(map[string]any)
	tols, _ := p4s4Get(ds, "spec", "template", "spec", "tolerations").([]any)
	t.Logf("[render] DaemonSet/dcgm-exporter: namespace=%s nodeSelector=%v tolerations=%v", ns, sel, tols)
	if ns != "kgpu" {
		t.Errorf("DaemonSet namespace %q, want kgpu", ns)
	}
	if !reflect.DeepEqual(sel, map[string]any{p5AccelKey: p5AccelVal}) {
		t.Errorf("DaemonSet nodeSelector %v, want {%s: %s}", sel, p5AccelKey, p5AccelVal)
	}
	tolOK := false
	for _, x := range tols {
		if p4s4Str(x, "key") == p5GPURes && p4s4Str(x, "value") == "present" && p4s4Str(x, "effect") == "NoSchedule" {
			tolOK = true
		}
	}
	if !tolOK {
		t.Errorf("DaemonSet lacks the nvidia.com/gpu=present:NoSchedule toleration: %v", tols)
	}
	cs, _ := p4s4Get(ds, "spec", "template", "spec", "containers").([]any)
	portOK := false
	var podGPU []string
	for _, c := range cs {
		ports, _ := p4s4Get(c, "ports").([]any)
		for _, p := range ports {
			if p4s4Str(p, "containerPort") == "9400" {
				portOK = true
			}
		}
		if p4s4Str(c, "resources", "limits", p5GPURes) != "" {
			podGPU = append(podGPU, p4s4Str(c, "name"))
		}
		t.Logf("[render] dcgm container %s: image=%s ports=%v securityContext=%v volumeMounts=%v env=%v",
			p4s4Str(c, "name"), p4s4Str(c, "image"), ports, p4s4Get(c, "securityContext"), p4s4Get(c, "volumeMounts"), p4s4Get(c, "env"))
	}
	if !portOK {
		t.Errorf("no dcgm-exporter container exposes containerPort 9400")
	}
	if len(podGPU) > 0 {
		t.Errorf("the exporter requests a card (%v): it would take a worker's GPU", podGPU)
	}
	sns := p4s4Str(svc, "metadata", "namespace")
	sports, _ := p4s4Get(svc, "spec", "ports").([]any)
	ssel, _ := p4s4Get(svc, "spec", "selector").(map[string]any)
	plabels, _ := p4s4Get(ds, "spec", "template", "metadata", "labels").(map[string]any)
	t.Logf("[render] Service/dcgm-exporter: namespace=%s ports=%v selector=%v; DaemonSet pod labels=%v", sns, sports, ssel, plabels)
	if sns != "kgpu" {
		t.Errorf("Service namespace %q, want kgpu", sns)
	}
	svcPort := false
	for _, p := range sports {
		if p4s4Str(p, "port") == "9400" {
			svcPort = true
		}
	}
	if !svcPort {
		t.Errorf("Service dcgm-exporter has no port 9400: %v", sports)
	}
	if len(ssel) == 0 {
		t.Errorf("Service dcgm-exporter has no selector")
	}
	for k, v := range ssel {
		if fmt.Sprint(plabels[k]) != fmt.Sprint(v) {
			t.Errorf("Service selector %s=%v does not match the DaemonSet's pod labels %v: the Service would have no endpoints", k, v, plabels)
		}
	}
	for _, k := range []string{"DaemonSet/dcgm-exporter", "Service/dcgm-exporter"} {
		if _, ok := gcp[k]; ok {
			t.Errorf("%s also appears in deploy/k8s-gcp: it belongs to the GPU overlay only", k)
		}
	}
	var sms []string
	for k := range gpu {
		if strings.HasPrefix(k, "ServiceMonitor/") || strings.HasPrefix(k, "PodMonitor/") {
			sms = append(sms, k)
		}
	}
	t.Logf("note: the overlay renders monitors %v; decision 9 says the exporter feeds 'the Phase 2 dashboard', but no ServiceMonitor/PodMonitor for dcgm-exporter is in the overlay, so Prometheus scraping it depends on something outside deploy/k8s-gcp-gpu (spec gap: where is it wired?)", sms)
	t.Logf("note (not asserted, outside the written spec): the exporter does not request nvidia.com/gpu, so the device plugin will not inject the driver's libraries or /dev/nvidia* into it; it is not privileged and mounts only pod-resources. On GKE the driver lives on the host under /home/kubernetes/bin/nvidia; an exporter without access to it usually fails to load NVML. Worth checking in the first GPU session")
}

func TestXcP5S2RenderOnlyDocumentedObjectsDiffer(t *testing.T) {
	gpu, gcp, _, _ := p5Render(t)
	allowedChanged := map[string]bool{"Deployment/worker": true, "ConfigMap/worker": true, "ConfigMap/scheduler": true}
	allowedAdded := map[string]bool{"DaemonSet/dcgm-exporter": true, "Service/dcgm-exporter": true}
	t.Logf("spec: decision 9 ('The overlay keeps everything of deploy/k8s-gcp and changes the worker only ... and it adds NVIDIA's DCGM exporter'); brief: only the worker Deployment, the two ConfigMaps and the two dcgm objects may differ")
	for _, k := range p5Keys(gcp) {
		g, ok := gpu[k]
		if !ok {
			t.Errorf("deploy/k8s-gcp's %s is missing from the GPU overlay", k)
			continue
		}
		same := reflect.DeepEqual(gcp[k], g)
		fa, fb := map[string]string{}, map[string]string{}
		p4s4Flatten(k, gcp[k], fa)
		p4s4Flatten(k, g, fb)
		diff := p4s4DiffPaths(fa, fb)
		if same {
			t.Logf("[render] %s identical", k)
		} else {
			t.Logf("[render] %s differs (%d paths):\n    %s", k, len(diff), strings.Join(diff, "\n    "))
		}
		if !same && !allowedChanged[k] {
			t.Errorf("%s differs between deploy/k8s-gcp and the GPU overlay; decision 9 allows only the worker (and the ConfigMaps): %v", k, diff)
		}
	}
	for _, k := range p5Keys(gpu) {
		if _, ok := gcp[k]; ok {
			continue
		}
		t.Logf("[render] the GPU overlay adds %s", k)
		if !allowedAdded[k] {
			t.Errorf("the GPU overlay adds %s, which decision 9 does not mention", k)
		}
	}
	for k := range allowedAdded {
		if _, ok := gpu[k]; !ok {
			t.Errorf("the GPU overlay lacks %s", k)
		}
	}
}

func TestXcP5S2RenderNoFakeCardAnywhere(t *testing.T) {
	gpu, gcp, rawGPU, rawGCP := p5Render(t)
	t.Logf("spec: the roadmap ('the only change from the kind manifests is the resource name (kgpu.io/gpu-card → nvidia.com/gpu)'); brief: no kgpu.io/gpu-card in either render")
	for name, raw := range map[string]string{"k8s-gcp-gpu": rawGPU, "k8s-gcp": rawGCP} {
		n := strings.Count(raw, p4s4CardRes)
		t.Logf("[render] %s: %d occurrence(s) of %s, %d of nvidia.com/gpu", name, n, p4s4CardRes, strings.Count(raw, p5GPURes))
		if n > 0 {
			t.Errorf("%s still mentions the fake card %s", name, p4s4CardRes)
		}
	}
	// The CPU overlay must not ask for a real card either.
	c := p4s4Container(t, p4s4Obj(t, gcp, "Deployment/worker"), "worker")
	if p4s4Str(c, "resources", "limits", p5GPURes) != "" {
		t.Errorf("deploy/k8s-gcp's CPU worker requests %s", p5GPURes)
	}
	// Only the GPU worker and nothing else requests a card.
	for _, k := range p5Keys(gpu) {
		cs, _ := p4s4Get(gpu[k], "spec", "template", "spec", "containers").([]any)
		for _, cc := range cs {
			if l := p4s4Str(cc, "resources", "limits", p5GPURes); l != "" {
				t.Logf("[render] %s container %s requests %s: %s", k, p4s4Str(cc, "name"), p5GPURes, l)
				if k != "Deployment/worker" {
					t.Errorf("%s requests a card", k)
				}
			}
		}
	}
}

// =========================================================================
// Step 3: the cold start in the simulator (decision 11)

const (
	p5N   int64 = 100_000
	p5Dim int32 = 128
	p5Mem int64 = 16 << 30
)

func TestXcP5S3SimAlwaysGPUColdStartByHand(t *testing.T) {
	gpuTot := sxGPUTotal(p5N, p5Dim)
	t.Logf("hand: CPU build %.4f s, GPU path %.4f s (v0: CPU/10 + 2 + transfer %.4f)", sxCPU(p5N, p5Dim), gpuTot, sxXfer(p5N, p5Dim))
	w := sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true)))
	ag, err := policy.ByName("always_gpu")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("spec: decision 11 / sim.Config.ColdStartS: 'the workers are not alive until ColdStartS after the first GPU build is queued'; brief: always_gpu with ColdStartS 10 -> round 16.0 s, queue wait 10")
	r := sxRun(t, "always_gpu cold 10", w, ag, sim.Config{ColdStartS: 10})
	done := 10 + gpuTot
	learn := sxNextReport(done, 0.5)
	t.Logf("hand: claimed at 10.0 (pool alive), finishes %.4f, the shard learns at the next report %.1f, the index query runs %.1f -> %.1f", done, learn, learn, learn+1)
	sxExpect(t, "round_s", r.RoundS, learn+1, "decision 11, brief: 16.0")
	sxExpect(t, "round_s (brief's number)", r.RoundS, 16.0, "brief")
	sxExpectSet(t, "queue waits", r.QueueWaits, []float64{10}, "first claim minus submit = the provisioning time")
	if r.BuildsGPU != 1 || !r.Completed {
		t.Errorf("builds_gpu %d completed %v, want 1 true", r.BuildsGPU, r.Completed)
	}

	// The submit-time estimate a policy sees: a scripted always-GPU policy records it.
	s := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0})}
	r2 := sxRun(t, "scripted gpu cold 10", sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true))), s, sim.Config{ColdStartS: 10})
	if len(s.inputs) != 1 {
		t.Fatalf("policy saw %d submits, want 1", len(s.inputs))
	}
	in := s.inputs[0]
	t.Logf("submit input: Pool{LiveWorkers %d, LargestMem %d, QueueWaitS %.4f}, Est{Local %.4f GPU %.4f Transfer %.4f}", in.Pool.LiveWorkers, in.Pool.LargestMem, in.Pool.QueueWaitS, in.Est.LocalBuildS, in.Est.GPUBuildS, in.Est.TransferS)
	sxExpect(t, "submit-time Pool.QueueWaitS", in.Pool.QueueWaitS, 10, "decision 11: an empty pool reports the provisioning time still to come")
	sxExpect(t, "scripted round_s", r2.RoundS, 16.0, "same round as always_gpu")
	t.Logf("note: at submit the cold pool reports LiveWorkers=%d LargestMem=%d; a policy that filters on LargestMem (the memory rule 'a build whose need exceeds every live worker's capacity must build locally') would see 0 here and keep every build local while the pool is cold. The docs do not say what LargestMem is while the pool is cold (spec gap if 0)", in.Pool.LiveWorkers, in.Pool.LargestMem)
}

func TestXcP5S3SimZeroChangesNothing(t *testing.T) {
	gpuTot := sxGPUTotal(p5N, p5Dim)
	mk := func() sim.Config { return sim.Config{} }
	s := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0})}
	r := sxRun(t, "scripted gpu cold 0", sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true))), s, mk())
	learn := sxNextReport(gpuTot, 0.5)
	t.Logf("spec: decision 11 'until then it is 0 everywhere, so nothing changes'; Phase 3's rule: an empty pool reports wait 0 at submit. hand: GPU done %.4f, learnt %.1f, index query done %.1f", gpuTot, learn, learn+1)
	sxExpect(t, "round_s", r.RoundS, learn+1, "ColdStartS 0: the worker is alive from t=0")
	sxExpectSet(t, "queue waits", r.QueueWaits, []float64{0}, "no provisioning")
	if len(s.inputs) == 1 {
		sxExpect(t, "submit-time Pool.QueueWaitS", s.inputs[0].Pool.QueueWaitS, 0, "Phase 3's rule kept at 0")
	}
	ag, _ := policy.ByName("always_gpu")
	a := sxRun(t, "always_gpu cold 0 (zero value)", sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true))), ag, sim.Config{})
	b := sxRun(t, "always_gpu cold 0 (explicit)", sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true))), ag, sim.Config{ColdStartS: 0})
	if !reflect.DeepEqual(a, b) {
		t.Errorf("ColdStartS 0 explicit differs from the zero value: %+v vs %+v", a, b)
	}
}

func TestXcP5S3SimReactiveColdStart10vs30(t *testing.T) {
	re, err := policy.ByName("reactive")
	if err != nil {
		t.Fatal(err)
	}
	loc, gpu := sxCPU(p5N, p5Dim), sxGPUTotal(p5N, p5Dim)
	t.Logf("spec: decision 11 'The reactive policy then keeps a build local while the pool is cold whenever local time beats cold start + GPU path ... (10 s: offloaded; 30 s: kept local)'. hand: local %.3f s; cold 10 + GPU %.3f = %.3f (GPU wins); cold 30 + GPU = %.3f (local wins)", loc, gpu, 10+gpu, 30+gpu)
	for _, c := range []struct {
		cold float64
		want string
	}{{0, "gpu"}, {10, "gpu"}, {60, "local"}} { // 60, not the doc's 30: staying local is judged WITH the backlog (two queued queries and ~1 query/s arriving over the 18.6 s left adds ~20 s), so 30 + 4.64 still beats it; the design doc's example was corrected to 60
		w := sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true)))
		r := sxRun(t, fmt.Sprintf("reactive cold %g", c.cold), w, re, sim.Config{ColdStartS: c.cold})
		got := ""
		if len(r.ShardPlacement) == 1 {
			got = r.ShardPlacement[0]
		}
		t.Logf("cold %g: final placement %q (gpu=%d local=%d preempts=%d reneges=%d), round %.4f", c.cold, got, r.BuildsGPU, r.BuildsLocal, r.Preempts, r.Reneges, r.RoundS)
		if got != c.want {
			t.Errorf("reactive at ColdStartS %g placed %q, want %q (decision 11)", c.cold, got, c.want)
		}
		if c.want == "local" {
			// the local build: CPU busy 0..20.63, both queries wait behind it
			sxExpect(t, "round_s (kept local)", r.RoundS, loc+2, "local build then the two queued 1 s queries")
		}
		if c.cold == 10 {
			// submit local (100k < the 200k threshold); preempted at the first tick, t=2, nothing
			// was queued before, so the pool comes alive at 12; the build runs 12 -> 16.64.
			want := sxNextReport(2+10+sxGPUTotal(p5N, p5Dim), 0.5) + 1
			sxExpect(t, "round_s (offloaded at the first tick, cold 10)", r.RoundS, want, "hand: learnt at the next report after 16.64, then the index query")
		}
	}
	// Why: the pure policy, called directly with the view the simulator hands it.
	st := policy.ShardStatus{QueueDepth: 2, WaitingNeedsIndex: 1, BuildProgress: 2 / loc, MeanQueryS: 1, Valid: true}
	est := policy.Estimates{LocalBuildS: loc, GPUBuildS: sxGPUBuild(p5N, p5Dim, 10), TransferS: sxXfer(p5N, p5Dim)}
	view := policy.BuildView{BuildID: "0:1", Spec: policy.BuildSpec{NVectors: p5N, Dim: p5Dim, MemBytes: sxMem(p5N, p5Dim)}, Est: est, Phase: "local_running", QueueWaitS: 8, Status: st}
	none := re.Reconsider(policy.ReconsiderInput{Builds: []policy.BuildView{view}})
	one := re.Reconsider(policy.ReconsiderInput{Workers: []policy.WorkerView{{ID: "w", MemBytes: p5Mem}}, Builds: []policy.BuildView{view}})
	t.Logf("diagnosis: Reactive.Reconsider on the same local_running view (queue wait 8 s, 2 s into a 20.6 s build, one query needing the index): with no live worker -> %+v; with one idle live worker -> %+v", none, one)
	if len(none) == 0 && len(one) > 0 {
		t.Logf("=> the reactive policy refuses to preempt into a pool with no live worker, whatever the wait says; and the simulated pool is only provisioned after a GPU build is queued. A sub-threshold build therefore never leaves the CPU while the pool is cold, at any ColdStartS > 0 (see TestXcP5S3SweepReactiveColdDiscontinuity)")
	}
}

func TestXcP5S3SimLocalViewBeforeAnyQueue(t *testing.T) {
	s := &sxScript{decide: sxLocal}
	w := sxWorkload(t, p5Dim, 1, p5Mem, 1, sxShard(0, p5N, sxQ(1, 1, false), sxQ(1, 1, true)))
	sxRun(t, "local only, cold 10", w, s, sim.Config{ColdStartS: 10})
	t.Logf("spec: sim.Config.ColdStartS (provisioning starts when the first GPU build is queued) + BuildView.QueueWaitS (for local_running: the wait it would join at if preempted now). Nothing is ever queued here, so at every tick a preemption would wait the whole 10 s")
	for k := 0; k < 3 && k < len(s.ticks); k++ {
		for _, b := range s.ticks[k].Builds {
			at := 2 * float64(k+1)
			sxExpect(t, fmt.Sprintf("t=%.0f local_running QueueWaitS", at), b.QueueWaitS, 10, "nothing queued yet: the full provisioning time is still to come")
		}
	}
}

func TestXcP5S3SweepReactiveColdDiscontinuity(t *testing.T) {
	b, err := swRun(t, "-scenarios", "uniform", "-policies", "reactive", "-rounds", "1", "-cold_start_s", "0,0.5,10")
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	_, rows := swParse(t, b)
	t.Logf("spec: decision 11 'The reactive policy then keeps a build local while the pool is cold whenever local time beats cold start + GPU path'. In `uniform` every build is below the reactive's 200k submit threshold, so every offload is a reconsider preemption; at cold start 0 the policy preempts all six. A 0.5 s cold start barely changes cold start + GPU path, so the offloads should survive it")
	got := map[string]swRow{}
	for _, r := range rows {
		got[r["cold_start_s"]] = r
		t.Logf("uniform reactive cold_start_s=%s: builds_gpu=%s builds_local=%s preempts=%s round_s=%s", r["cold_start_s"], r["builds_gpu"], r["builds_local"], r["preempts"], r["round_s"])
	}
	if got["0"]["builds_gpu"] != got["0.5"]["builds_gpu"] {
		t.Errorf("a 0.5 s cold start moves builds_gpu from %s to %s: the reactive never preempts into an empty pool and the simulated pool is never provisioned without a queued GPU build, so the cold start acts as an off switch, not a cost", got["0"]["builds_gpu"], got["0.5"]["builds_gpu"])
	}
}

func TestXcP5S3SimColdClockStartsAtFirstGPUQueue(t *testing.T) {
	// shard 0 builds locally from t=0; shard 1's DDL arrives at t=5 and goes to the GPU.
	s := &sxScript{decide: sxGPUAt(map[int32]float64{1: 0})}
	w := sxWorkload(t, p5Dim, 1, p5Mem, 1,
		sxShard(0, p5N, sxQ(1, 1, false)),
		sxShard(5, p5N, sxQ(6, 1, false), sxQ(6, 1, true)))
	r := sxRun(t, "local@0 + gpu@5, cold 10", w, s, sim.Config{ColdStartS: 10})
	t.Logf("spec: sim.Config.ColdStartS 'the workers are not alive until ColdStartS after the first GPU build is queued' (not after the round starts)")
	for _, in := range s.inputs {
		t.Logf("submit shard %d: Pool{LiveWorkers %d QueueWaitS %.4f}", in.Build.ShardID, in.Pool.LiveWorkers, in.Pool.QueueWaitS)
		sxExpect(t, fmt.Sprintf("shard %d submit-time QueueWaitS", in.Build.ShardID), in.Pool.QueueWaitS, 10, "no GPU build queued yet: the whole provisioning time is still to come")
	}
	sxExpectSet(t, "queue waits", r.QueueWaits, []float64{10}, "queued at 5, workers alive at 15")
	done := 15 + sxGPUTotal(p5N, p5Dim)
	t.Logf("hand: shard 1's GPU build runs 15.0 -> %.4f", done)
	if len(r.ShardFinishS) == 2 {
		sxExpect(t, "shard 1 finish", r.ShardFinishS[1], sxNextReport(done, 0.5)+1, "learnt at the next report, then the 1 s index query")
	}
}

func TestXcP5S3SimColdWaitCountsDown(t *testing.T) {
	// two workers; shard 0 to the GPU at t=0, shard 2 local at t=0, shard 1 to the GPU at t=3.
	s := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0, 1: 0})}
	w := sxWorkload(t, p5Dim, 2, p5Mem, 1,
		sxShard(0, p5N, sxQ(1, 1, true)),
		sxShard(3, p5N, sxQ(4, 1, true)),
		sxShard(0, 4*p5N, sxQ(1, 1, false)))
	r := sxRun(t, "gpu@0, gpu@3, local@0, 2 workers, cold 10", w, s, sim.Config{ColdStartS: 10})
	t.Logf("spec: decision 11: 'the queue-wait estimate every policy sees at submit and at every reconsider tick is the provisioning time still to come'; policy.BuildView.QueueWaitS: for gpu_queued its current wait, for local_running the wait it would join at if preempted now")
	for _, in := range s.inputs {
		want := 10.0
		if in.Build.ShardID == 1 {
			want = 7
		}
		t.Logf("submit shard %d: QueueWaitS %.4f", in.Build.ShardID, in.Pool.QueueWaitS)
		sxExpect(t, fmt.Sprintf("shard %d submit-time QueueWaitS", in.Build.ShardID), in.Pool.QueueWaitS, want, "provisioning still to come at its submit")
	}
	sxExpectSet(t, "queue waits", r.QueueWaits, []float64{10, 7}, "both claimed when the pool comes alive at 10")
	// Ticks fire at t = 2, 4, 6, 8 (ReconsiderS 2, first at ReconsiderS); the pool is alive from 10.
	for k := 0; k < 4 && k < len(s.ticks); k++ {
		at := 2 * float64(k+1)
		tk := s.ticks[k]
		var parts []string
		for _, b := range tk.Builds {
			parts = append(parts, fmt.Sprintf("shard %d %s wait %.4f", b.Spec.ShardID, b.Phase, b.QueueWaitS))
		}
		t.Logf("tick at t=%.0f: %d live workers; %s", at, len(tk.Workers), strings.Join(parts, "; "))
		if len(tk.Workers) != 0 {
			t.Errorf("tick at t=%.0f sees %d live workers before the pool is provisioned", at, len(tk.Workers))
		}
		for _, b := range tk.Builds {
			switch {
			case b.Phase == "gpu_queued":
				sxExpect(t, fmt.Sprintf("t=%.0f shard %d gpu_queued QueueWaitS", at, b.Spec.ShardID), b.QueueWaitS, 10-at, "decision 11: provisioning still to come")
			case b.Phase == "local_running" && k == 0:
				// one queued build ahead, two workers: the honest wait is also 10 - t
				sxExpect(t, fmt.Sprintf("t=%.0f local_running QueueWaitS", at), b.QueueWaitS, 10-at, "decision 11: provisioning still to come")
			case b.Phase == "local_running":
				t.Logf("  (t=%.0f local_running: two builds already wait for two workers; decision 11's rule says %.0f, a replay after the pool comes alive would add one GPU path; logged, not asserted)", at, 10-at)
			}
		}
	}
	if len(s.ticks) < 4 {
		t.Errorf("only %d reconsider ticks recorded", len(s.ticks))
	}
}

// =========================================================================
// Step 3: the sweep's -cold_start_s axis

func TestXcP5S3SweepColdStartAxis(t *testing.T) {
	b, err := swRun(t, "-scenarios", "skewed", "-policies", "reactive,always_gpu", "-rounds", "2", "-cold_start_s", "0,120")
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	hdr, rows := swParse(t, b)
	has := false
	for _, h := range hdr {
		has = has || h == "cold_start_s"
	}
	t.Logf("spec: decision 11 ('-cold_start_s, a sweep axis, echoed as a column'); the step 2–4 notes: 'at 0 both offload all six builds (30.0 s); at 120 reactive keeps all six local (56.7 s) while always-GPU waits for the pool (140.6 s)'")
	if !has {
		t.Fatalf("no cold_start_s column in %v", hdr)
	}
	if len(rows) != 8 {
		t.Errorf("%d rows, want 2 policies x 2 cold starts x 2 rounds = 8", len(rows))
	}
	round := map[string]float64{}
	for _, r := range rows {
		t.Logf("row: policy=%s seed=%s cold_start_s=%s builds_gpu=%s builds_local=%s round_s=%s queue_wait_p50_s=%s completed=%s",
			r["policy"], r["seed"], r["cold_start_s"], r["builds_gpu"], r["builds_local"], r["round_s"], r["queue_wait_p50_s"], r["completed"])
		c := r.f(t, "cold_start_s")
		key := fmt.Sprintf("%s@%g/%s", r["policy"], c, r["seed"])
		round[key] = r.f(t, "round_s")
		g, l := r["builds_gpu"], r["builds_local"]
		switch {
		case c == 120 && r["policy"] == "reactive":
			if g != "0" || l != "6" {
				t.Errorf("%s: gpu %s local %s, want 0 / 6 (doc)", key, g, l)
			}
		case c == 120 && r["policy"] == "always_gpu":
			if g != "6" || l != "0" {
				t.Errorf("%s: gpu %s local %s, want 6 / 0", key, g, l)
			}
			if r.f(t, "queue_wait_p50_s") < 120-1e-6 {
				t.Errorf("%s: median queue wait %s < the 120 s provisioning time", key, r["queue_wait_p50_s"])
			}
		case c == 0:
			if g != "6" {
				t.Errorf("%s: gpu %s, the doc says both offload all six at 0", key, g)
			}
		}
	}
	for _, seed := range []string{"1", "2"} {
		ag0, ag120, re120 := round["always_gpu@0/"+seed], round["always_gpu@120/"+seed], round["reactive@120/"+seed]
		if ag120 == 0 {
			continue
		}
		t.Logf("seed %s: always_gpu %.2f -> %.2f at 120; reactive at 120 %.2f", seed, ag0, ag120, re120)
		if !(ag120 > ag0+100) || !(ag120 > re120) {
			t.Errorf("seed %s: always_gpu at 120 (%.2f) should be 'much longer' than at 0 (%.2f) and than reactive at 120 (%.2f)", seed, ag120, ag0, re120)
		}
	}
	t.Logf("doc numbers vs seed 1 (the doc names no seed or round count; logged): reactive@0 %.1f always_gpu@0 %.1f (doc 30.0), reactive@120 %.1f (doc 56.7), always_gpu@120 %.1f (doc 140.6)",
		round["reactive@0/1"], round["always_gpu@0/1"], round["reactive@120/1"], round["always_gpu@120/1"])
}

func TestXcP5S3SweepRefusesBadColdStart(t *testing.T) {
	base := []string{"-scenarios", "skewed", "-policies", "reactive", "-rounds", "1"}
	t.Logf("spec (brief; same rule as the step 1 triage C): -cold_start_s refuses negative, NaN and Inf with a non-zero exit")
	for _, v := range []string{"-1", "-0.5", "NaN", "nan", "Inf", "+Inf", "-Inf", "abc", "0,-5", "0,,120"} {
		_, err := swRun(t, append(append([]string{}, base...), "-cold_start_s", v)...)
		t.Logf("-cold_start_s %q -> refused=%v", v, err != nil)
		if err == nil {
			t.Errorf("-cold_start_s %q accepted", v)
		}
	}
	b, err := swRun(t, append(append([]string{}, base...), "-cold_start_s", "0")...)
	if err != nil {
		t.Fatalf("-cold_start_s 0 refused: %v", err)
	}
	_, rows := swParse(t, b)
	t.Logf("-cold_start_s 0 echoes %q", rows[0]["cold_start_s"])
	b2, _ := swRun(t, base...)
	_, rows2 := swParse(t, b2)
	if rows[0]["cold_start_s"] != rows2[0]["cold_start_s"] || rows[0]["round_s"] != rows2[0]["round_s"] {
		t.Errorf("explicit 0 (%v, %v) differs from the default (%v, %v)", rows[0]["cold_start_s"], rows[0]["round_s"], rows2[0]["cold_start_s"], rows2[0]["round_s"])
	}
}

func TestXcP5S3DemoCSVCarriesColdStartZero(t *testing.T) {
	root := repoRoot(t)
	t.Logf("spec: decision 11 (0 everywhere until measured, so nothing changes) and the brief: the committed demo CSVs carry cold_start_s = 0 and `-cost default` without -cold_start_s reproduces every cell")
	for _, f := range []string{"rounds.csv", "sigma.csv", "shrinking.csv"} {
		raw, err := os.ReadFile(filepath.Join(root, "experiments/results/demo", f))
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		hdr, rows := swParse(t, raw)
		idx := -1
		for i, h := range hdr {
			if h == "cold_start_s" {
				idx = i
			}
		}
		nonzero := 0
		for _, r := range rows {
			if r["cold_start_s"] != "0" {
				nonzero++
			}
		}
		t.Logf("demo/%s: %d rows, cold_start_s at column %d, rows with cold_start_s != \"0\": %d", f, len(rows), idx, nonzero)
		if idx < 0 || nonzero > 0 {
			t.Errorf("demo/%s: cold_start_s column index %d, %d non-zero rows", f, idx, nonzero)
		}
	}
	demo, _ := os.ReadFile(filepath.Join(root, "experiments/results/demo/rounds.csv"))
	dh, drows := swParse(t, demo)
	b, err := swRun(t, "-scenarios", "uniform,skewed,bimodal,downstream_heavy,staggered,high_dim,production", "-policies", "all", "-rounds", "20", "-cost", "default")
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	_, nrows := swParse(t, b)
	key := func(r swRow) string { return r["scenario"] + "|" + r["policy"] + "|" + r["seed"] }
	by := map[string]swRow{}
	for _, r := range nrows {
		by[key(r)] = r
	}
	bad, missing := 0, 0
	for _, d := range drows {
		r, ok := by[key(d)]
		if !ok {
			missing++
			continue
		}
		for _, c := range dh {
			if r[c] != d[c] {
				bad++
				if bad <= 10 {
					t.Errorf("%s %s: demo %q, now %q", key(d), c, d[c], r[c])
				}
			}
		}
	}
	t.Logf("demo rows %d, rerun rows %d, missing %d, differing cells %d (every column of the demo, cold_start_s included)", len(drows), len(nrows), missing, bad)
	if missing > 0 || len(nrows) != len(drows) {
		t.Errorf("rerun has %d rows for the demo's %d (%d missing)", len(nrows), len(drows), missing)
	}
}

// =========================================================================
// Step 3: the scheduler (API, reconsider timer, binary)

func p5ColdEnv(t *testing.T, coldS float64, pol policy.Policy) *env {
	t.Helper()
	e := setup(t)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := httptest.NewServer(api.New(e.st, pol, costmodel.Default(), api.Config{WorkerStaleAfter: time.Minute, ColdStartS: coldS}, logger))
	t.Cleanup(srv.Close)
	e.srv = srv
	return e
}

func p5Submit(t *testing.T, e *env, round int64, shard int32, n int64, dim int32) (api.SubmitBuildResponse, map[string]any) {
	t.Helper()
	var r api.SubmitBuildResponse
	code, body := e.do(t, "POST", "/builds", api.SubmitBuildRequest{RoundID: round, ShardID: shard, NVectors: n, Dim: dim}, &r)
	if code != http.StatusCreated {
		t.Fatalf("POST /builds shard %d: %d %s", shard, code, body)
	}
	var raw map[string]any
	_ = json.Unmarshal([]byte(body), &raw)
	t.Logf("POST /builds shard %d (n=%d dim=%d) -> 201 %s", shard, n, dim, strings.TrimSpace(body))
	return r, raw
}

func p5Wait(r api.SubmitBuildResponse) string {
	if r.GPUQueueWaitMs == nil {
		return "<absent>"
	}
	return fmt.Sprint(*r.GPUQueueWaitMs)
}

func TestXcP5S3APIColdStartEmptyPool(t *testing.T) {
	e := p5ColdEnv(t, 150, policy.AlwaysGPU{})
	round := e.createRound(t, 6)
	t.Logf("spec: decision 11 + api.Config.ColdStartS: 'with no live worker the queue-wait estimate a policy sees is this instead of 0'; brief: POST /builds replies gpu_queue_wait_ms = 150000")
	r, _ := p5Submit(t, e, round, 0, 10000, 64)
	if r.GPUQueueWaitMs == nil || *r.GPUQueueWaitMs != 150000 {
		t.Errorf("empty pool, ColdStartS 150: gpu_queue_wait_ms = %s, want 150000", p5Wait(r))
	}
	logBuild(t, "after submit (no workers)", e.mustBuild(t, r.BuildID))

	// A live worker: the step 2 replay applies instead.
	e.rxHeartbeat(t, "w-1", p5Mem)
	r2, _ := p5Submit(t, e, round, 1, 10000, 64)
	cm := costmodel.Default()
	want := cm.GPUTotal(10000, 64).Milliseconds()
	t.Logf("hand: one idle worker, shard 0's GPU build queued ahead (GPUTotal %v): the newcomer waits ~%d ms behind it (PoolState doc: 'joined at priority 0 behind every queued job')", cm.GPUTotal(10000, 64), want)
	if r2.GPUQueueWaitMs == nil {
		t.Fatalf("live worker: gpu_queue_wait_ms absent")
	}
	if *r2.GPUQueueWaitMs == 150000 {
		t.Errorf("live worker: still the cold-start 150000; the replay should apply")
	}
	if d := *r2.GPUQueueWaitMs - want; d < -2 || d > 2 {
		t.Errorf("live worker: gpu_queue_wait_ms = %d, the replay's wait is ~%d", *r2.GPUQueueWaitMs, want)
	}

	// A stale worker: no live worker again.
	if _, err := e.pool.Exec(e.ctx, `UPDATE workers SET last_seen = now() - interval '1 hour'`); err != nil {
		t.Logf("could not age the worker (%v); skipping the stale case", err)
		return
	}
	r3, _ := p5Submit(t, e, round, 2, 10000, 64)
	t.Logf("worker aged past WorkerStaleAfter: gpu_queue_wait_ms %s", p5Wait(r3))
	if r3.GPUQueueWaitMs == nil || *r3.GPUQueueWaitMs != 150000 {
		t.Errorf("stale-only pool: gpu_queue_wait_ms = %s, want 150000 (no live worker)", p5Wait(r3))
	}
}

func TestXcP5S3APIColdStartUnsetKeepsZero(t *testing.T) {
	e := p5ColdEnv(t, 0, policy.AlwaysGPU{})
	round := e.createRound(t, 6)
	t.Logf("spec: decision 11 'COLD_START ... default 0 ... with the Phase 3 rule kept as the default': an empty pool reports 0")
	r, raw := p5Submit(t, e, round, 0, 10000, 64)
	if _, present := raw["gpu_queue_wait_ms"]; !present || r.GPUQueueWaitMs == nil || *r.GPUQueueWaitMs != 0 {
		t.Errorf("ColdStartS 0, empty pool: gpu_queue_wait_ms = %s (present %v), want an explicit 0", p5Wait(r), present)
	}
	// Live idle worker, empty queue: also 0 — and with ColdStartS set, too.
	e2 := p5ColdEnv(t, 150, policy.AlwaysGPU{})
	round2 := e2.createRound(t, 6)
	e2.rxHeartbeat(t, "w-1", p5Mem)
	r2, _ := p5Submit(t, e2, round2, 0, 10000, 64)
	if r2.GPUQueueWaitMs == nil || *r2.GPUQueueWaitMs != 0 {
		t.Errorf("ColdStartS 150, one idle live worker, empty queue: gpu_queue_wait_ms = %s, want 0 (the pool is warm)", p5Wait(r2))
	}
	// A duplicate omits the field (existing contract), cold or not.
	code, body := e2.do(t, "POST", "/builds", api.SubmitBuildRequest{RoundID: round2, ShardID: 0, NVectors: 10000, Dim: 64}, nil)
	t.Logf("duplicate POST /builds -> %d %s (the existing contract omits gpu_queue_wait_ms on a duplicate)", code, strings.TrimSpace(body))
	if strings.Contains(body, "gpu_queue_wait_ms") {
		t.Errorf("a duplicate carries gpu_queue_wait_ms: %s", body)
	}
}

func TestXcP5S3APIColdStartReachesThePolicy(t *testing.T) {
	// reactive at submit "places like a static threshold"; whether the cold wait
	// changes its submit decision is not specified. Log what it does at 150 s vs 0.
	re, err := policy.ByName("reactive")
	if err != nil {
		t.Fatal(err)
	}
	for _, cold := range []float64{0, 150} {
		e := p5ColdEnv(t, cold, re)
		round := e.createRound(t, 6)
		r, _ := p5Submit(t, e, round, 0, p5N, p5Dim)
		t.Logf("reactive, ColdStartS %g, empty pool: placement %s, reason %q, gpu_queue_wait_ms %s, cpu_build_ms %d, gpu_total_ms %d",
			cold, r.Placement, r.Reason, p5Wait(r), r.CPUBuildMs, r.GPUTotalMs)
		if cold == 150 && r.GPUQueueWaitMs != nil && *r.GPUQueueWaitMs != 150000 {
			t.Errorf("ColdStartS 150: gpu_queue_wait_ms %d", *r.GPUQueueWaitMs)
		}
	}
}

// p5Rec records what the reconsider timer hands the policy.
type p5Rec struct{ ticks []policy.ReconsiderInput }

func (p *p5Rec) Name() string { return "xcheck_p5_recorder" }
func (p *p5Rec) Decide(policy.Input) policy.Decision {
	return policy.Decision{Placement: "gpu", Reason: "xcheck"}
}
func (p *p5Rec) Reconsider(in policy.ReconsiderInput) []policy.Action {
	p.ticks = append(p.ticks, policy.ReconsiderInput{
		Workers: append([]policy.WorkerView(nil), in.Workers...),
		Builds:  append([]policy.BuildView(nil), in.Builds...),
	})
	return nil
}

func p5Tick(t *testing.T, e *env, cold float64) policy.ReconsiderInput {
	t.Helper()
	rec := &p5Rec{}
	var buf lockedBuf
	log := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	rc.Tick(e.ctx, e.st, rec, rxCM, rc.Config{Interval: time.Second, StaleAfter: time.Minute, TimeScale: 1, ColdStartS: cold}, log)
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if l != "" {
			t.Logf("  [tick cold %g] %s", cold, l)
		}
	}
	if len(rec.ticks) != 1 {
		t.Fatalf("the policy's Reconsider ran %d times in one Tick", len(rec.ticks))
	}
	in := rec.ticks[0]
	t.Logf("tick (ColdStartS %g): %d workers", cold, len(in.Workers))
	for _, b := range in.Builds {
		t.Logf("  view %s phase=%s QueueWaitS=%.4f", b.BuildID, b.Phase, b.QueueWaitS)
	}
	return in
}

func p5View(in policy.ReconsiderInput, phase string) (policy.BuildView, bool) {
	for _, b := range in.Builds {
		if b.Phase == phase && !b.Speculative {
			return b, true
		}
	}
	return policy.BuildView{}, false
}

func TestXcP5S3ReconsiderColdStartEmptyPool(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 6)
	e.rxAgeRound(t, round)
	e.rxSubmit(t, round, 0, rxN, "local", 0)
	e.rxSubmit(t, round, 1, rxN, "gpu", 0)
	t.Logf("spec: decision 11 'at submit and at every reconsider tick'; reconsider.Config.ColdStartS 'a cold pool's provisioning time: the wait an empty pool reports'; brief: the timer reports 150 for an empty pool's local-running and queued views")
	in := e.p5TickCheck(t, 150, 150)
	if len(in.Workers) != 0 {
		t.Errorf("%d workers in an empty pool", len(in.Workers))
	}
	t.Logf("--- ColdStartS 0 (Phase 3's rule): the empty pool reports 0")
	e.p5TickCheck(t, 0, 0)

	t.Logf("--- a live worker joins: the replay's wait applies instead of the cold start")
	e.rxHeartbeat(t, "w-1", p5Mem)
	in = p5Tick(t, e, 150)
	q, okq := p5View(in, "gpu_queued")
	l, okl := p5View(in, "local_running")
	if !okq || !okl {
		t.Fatalf("views missing (queued %v local %v)", okq, okl)
	}
	gpu := rxSecs(rxCM.GPUTotal(rxN, rxDim))
	t.Logf("hand: one idle worker; the queued build is at the head (wait 0); a local build preempted now would join behind it (~%.4f s)", gpu)
	if q.QueueWaitS >= 150 || l.QueueWaitS >= 150 {
		t.Errorf("live worker: queued %.4f, local %.4f — still the cold start", q.QueueWaitS, l.QueueWaitS)
	}
	if math.Abs(q.QueueWaitS) > 1e-6 {
		t.Errorf("live idle worker: the head of the queue waits %.4f, want 0", q.QueueWaitS)
	}
	if math.Abs(l.QueueWaitS-gpu) > 0.01 {
		t.Errorf("live idle worker: local_running would join at %.4f, the replay says ~%.4f", l.QueueWaitS, gpu)
	}
}

func (e *env) p5TickCheck(t *testing.T, cold, want float64) policy.ReconsiderInput {
	t.Helper()
	in := p5Tick(t, e, cold)
	for _, ph := range []string{"local_running", "gpu_queued"} {
		v, ok := p5View(in, ph)
		if !ok {
			t.Errorf("ColdStartS %g: no %s view", cold, ph)
			continue
		}
		if math.Abs(v.QueueWaitS-want) > 1e-6 {
			t.Errorf("ColdStartS %g, empty pool: %s QueueWaitS = %.4f, want %g", cold, ph, v.QueueWaitS, want)
		} else {
			t.Logf("ok: ColdStartS %g, empty pool: %s QueueWaitS = %.4f", cold, ph, v.QueueWaitS)
		}
	}
	return in
}

// The binary: COLD_START parsed from the environment.

var (
	p5BinOnce sync.Once
	p5Bin     string
	p5BinErr  error
)

func p5SchedulerBin(t *testing.T) string {
	t.Helper()
	p5BinOnce.Do(func() {
		dir, err := os.MkdirTemp("", "xcheck-p5-sched")
		if err != nil {
			p5BinErr = err
			return
		}
		p5Bin = filepath.Join(dir, "scheduler")
		cmd := exec.Command("go", "build", "-o", p5Bin, "./scheduler/cmd/scheduler")
		cmd.Dir = repoRoot(t)
		if out, err := cmd.CombinedOutput(); err != nil {
			p5BinErr = fmt.Errorf("%v %s", err, out)
		}
	})
	if p5BinErr != nil {
		t.Fatal(p5BinErr)
	}
	return p5Bin
}

func p5FreePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// p5Start runs the scheduler with env; it returns the base URL once POST
// /rounds answers, or ("", exit code, output) if it exited.
func p5Start(t *testing.T, dbURL string, extra ...string) (string, int, *lockedBuf, func()) {
	t.Helper()
	addr := p5FreePort(t)
	cmd := exec.Command(p5SchedulerBin(t))
	cmd.Env = append([]string{"PATH=" + os.Getenv("PATH"), "DATABASE_URL=" + dbURL, "LOG_FORMAT=text",
		"LISTEN_ADDR=" + addr, "RECONSIDER_INTERVAL=0", "SHUTDOWN_DELAY=0s"}, extra...)
	buf := &lockedBuf{}
	cmd.Stdout, cmd.Stderr = buf, buf
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	stop := func() {
		_ = cmd.Process.Kill()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
		}
	}
	base := "http://" + addr
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-done:
			var ee *exec.ExitError
			code := 0
			if errors.As(err, &ee) {
				code = ee.ExitCode()
			}
			done <- err
			return "", code, buf, func() {}
		default:
		}
		resp, err := http.Get(base + "/workers")
		if err == nil {
			resp.Body.Close()
			return base, -1, buf, stop
		}
		time.Sleep(100 * time.Millisecond)
	}
	stop()
	t.Fatalf("scheduler did not come up: %s", buf.String())
	return "", 0, nil, nil
}

func p5Post(t *testing.T, url string, body any, out any) (int, string) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if out != nil && resp.StatusCode < 300 {
		_ = json.Unmarshal(raw, out)
	}
	return resp.StatusCode, string(raw)
}

func TestXcP5S3SchedulerColdStartEnv(t *testing.T) {
	e := setup(t) // skips without TEST_DATABASE_URL; migrates and truncates
	dbURL := os.Getenv("TEST_DATABASE_URL")
	t.Logf("spec: `go doc ./scheduler/cmd/scheduler`: 'COLD_START default 0; a duration (e.g. 150s)'; brief: with no live worker POST /builds replies gpu_queue_wait_ms = 150000; unset keeps 0")
	cases := []struct {
		env  []string
		want int64
	}{{[]string{"COLD_START=150s"}, 150000}, {nil, 0}, {[]string{"COLD_START=2m30s"}, 150000}, {[]string{"COLD_START=0"}, 0}}
	for i, c := range cases {
		if _, err := e.pool.Exec(e.ctx, `TRUNCATE shard_status, shard_jobs, builds, rounds, workers`); err != nil {
			t.Fatal(err)
		}
		base, code, out, stop := p5Start(t, dbURL, c.env...)
		if base == "" {
			t.Errorf("%v: scheduler exited %d: %s", c.env, code, p5Tail(out.String(), 400))
			continue
		}
		var rr api.CreateRoundResponse
		if sc, body := p5Post(t, base+"/rounds", api.CreateRoundRequest{Scenario: "xcheck", Seed: int64(i), NShards: 6}, &rr); sc != http.StatusCreated {
			stop()
			t.Fatalf("POST /rounds: %d %s", sc, body)
		}
		var r api.SubmitBuildResponse
		sc, body := p5Post(t, base+"/builds", api.SubmitBuildRequest{RoundID: rr.RoundID, ShardID: 0, NVectors: 10000, Dim: 64}, &r)
		lines := strings.Split(strings.TrimSpace(out.String()), "\n")
		if len(lines) > 3 {
			lines = lines[:3]
		}
		stop()
		t.Logf("env %v: POST /builds -> %d %s; first log lines: %v; any line mentions cold_start: %v", c.env, sc, strings.TrimSpace(body), lines, strings.Contains(strings.ToLower(out.String()), "cold"))
		if r.GPUQueueWaitMs == nil || *r.GPUQueueWaitMs != c.want {
			t.Errorf("env %v: gpu_queue_wait_ms = %s, want %d", c.env, p5Wait(r), c.want)
		}
	}
	t.Logf("malformed COLD_START (spec silent on what is refused; the scheduler's other bad values exit 2, step 1 triage D):")
	for _, v := range []string{"abc", "150", "-5s", "NaN"} {
		base, code, out, stop := p5Start(t, dbURL, "COLD_START="+v)
		state := fmt.Sprintf("exit %d: %s", code, p5Tail(strings.TrimSpace(out.String()), 200))
		if base != "" {
			state = "accepted (serving)"
			if _, err := e.pool.Exec(e.ctx, `TRUNCATE shard_status, shard_jobs, builds, rounds, workers`); err == nil {
				var rr api.CreateRoundResponse
				p5Post(t, base+"/rounds", api.CreateRoundRequest{Scenario: "xcheck", Seed: 99, NShards: 6}, &rr)
				var r api.SubmitBuildResponse
				p5Post(t, base+"/builds", api.SubmitBuildRequest{RoundID: rr.RoundID, ShardID: 0, NVectors: 10000, Dim: 64}, &r)
				state += ", empty pool reports gpu_queue_wait_ms " + p5Wait(r)
			}
			stop()
		}
		t.Logf("  COLD_START=%q -> %s", v, state)
		if v == "abc" || v == "-5s" {
			if base != "" {
				t.Errorf("COLD_START=%q accepted: a non-duration or a negative provisioning time should be refused at startup like every other bad value", v)
			}
		}
	}
}
