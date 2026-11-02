# Phase 5: GPUs and cuVS

## Status and scope

**In progress, GPU-free part first** (the GPU quota is pending; Phase 4 decision 27). Step 1, cost model v1, is done (decisions 6–8 below). Add a real GPU build implementation to the cloud topology Phase 4 leaves behind (a managed cluster with CPU workers, MinIO and the scheduler, plus the shard VM outside it — Phase 4 decisions 7 and 8), run it for a few hours on a GPU node pool added to that cluster, and rerun the comparison with a measured GPU cost model. Scheduler, Postgres, the shard VM and the shard simulator do not change; the worker gets a second builder image behind the same `build(job)` interface.

**Done when** the policy comparison includes real GPU numbers and a spot preemption has been observed and recovered from.

## System diagram

The same manifests as Phase 2, already applied to the managed cluster in Phase 4's closing step; this phase adds the GPU node pool, autoscaling from zero. The worker image changes; nothing else does.

```mermaid
flowchart TB
    subgraph cloud["managed cluster (e.g. GKE)"]
        subgraph std["standard node pool"]
            SD["scheduler<br/>(LoadBalancer or Ingress for the shards)"]
            PG["postgres"]
            MO["minio"]
            MON["prometheus, grafana,<br/>DCGM exporter"]
        end
        subgraph gpu["GPU node pool: spot, 0…N machines × M cards<br/>taint gpu-only, device plugin advertises nvidia.com/gpu: M per node"]
            W["worker (cuVS image) ×(cards in use)<br/>requests nvidia.com/gpu: 1"]
        end
    end
    SH["shards: the database fleet on a CPU VM<br/>in the same region, outside the cluster<br/>(Phase 2 decision 26, Phase 4 decision 8)"] -- HTTP --> SD
    AS["cluster autoscaler"] -- "pending worker pod → add node<br/>idle node → remove" --> gpu
    SP["spot reclaim"] -. "node disappears mid-build" .-> gpu
```

## Flows

### Cold start

With the pool at zero, the first GPU job waits for a node. That delay is part of the GPU finish estimate, so the cost-based policy sends more work to CPUs while the pool is cold.

```mermaid
sequenceDiagram
    participant A as scheduler
    participant P as Postgres
    participant K as autoscaler
    participant N as new GPU node
    participant W as worker pod

    A->>P: first GPU build queued, workers table empty
    Note over A: estimate = node provision time + GPU time
    W->>K: pod pending (no node satisfies nvidia.com/gpu)
    K->>N: provision node (minutes)
    N->>W: pod starts, image pulled
    W->>P: register, heartbeat, claim
```

### Spot preemption

Recovery goes through the lease path; Kubernetes replaces the pod when a node is available again. What is measured: work lost per preemption, and whether a shorter renew interval would reduce it.

```mermaid
sequenceDiagram
    participant W1 as worker on node A
    participant P as Postgres
    participant R as reaper
    participant W2 as worker on node B

    W1->>P: claim, attempt 1
    Note over W1: node A reclaimed (30 s notice on most clouds)
    W1->>P: release if the notice is caught: state=queued (guarded)
    R->>P: otherwise lease expires → queued
    W2->>P: claim, attempt 2, rebuild from scratch
```

## Design decisions

| # | Decision | Alternatives | Why |
| --- | --- | --- | --- |
| 1 | Separate GPU worker image on a RAPIDS/cuVS base; CPU image unchanged | One image with both | The GPU image is large and needs CUDA. Keeping the CPU image lets the kind cluster stay the everyday environment. |
| 2 | Build CAGRA with cuVS, convert to HNSW, save; verify it loads in hnswlib | Serve CAGRA directly | The query path stays CPU hnswlib, matching the industry pattern (build on GPU, search on CPU). |
| 3 | Memory constraint in placement uses measured peak GPU memory per size | The modeled `mem_bytes` from Phase 1 | Phase 1's model is a placeholder; this phase measures it. |
| 4 | Spot nodes, autoscaling from zero, billing alert first, tear down after each session | On-demand nodes kept up | Cost. Preemption is also a feature here: it is the failure this phase wants to observe. |
| 5 | Cold-start delay added to the GPU finish estimate | Ignore it | It is real, minutes long, and the cost-based policy should visibly prefer CPUs while the pool is cold. |
| 6 | **Cost model v1: every term the Phase 4 calibration found is its own, with a constant the calibration measures** (owner's choice, "full refit"): build = A·n^P·ln(n)·(dim + D0); gate = Gate·GateQueries·ln(n)·(dim + D0); save = index bytes / SaveBandwidth; fetch = vector bytes / Bandwidth; upload = index bytes / BandwidthOut; the worker's path = (build + gate)/Speedup + GPUOverhead + save + fetch + upload; the shard's local path = build + gate + save, plus the fetch and the upload when the rig makes the shard use the store (`ShardUsesStore`: this topology does, a database fleet would not). P and D0 come from Phase 4 decision 35 (n^1.2, and a per-vector cost that does not scale with dim — 187 on the VM's x86 cores, 0 on the laptop's), the gate, save and separate upload bandwidth from decision 37. Speedup is no longer the ratio of two A's but the geometric mean, over the worker's calibration points, of the shard model's build time over the worker's measured one (2.12 on the cloud topology), since P and D0 differ between the two fits. The simulator's two buckets stay: `GPUBuild` is what the worker computes (build, gate, overhead, save — the save is disk, counted here so the other bucket stays "the link"), `Transfer` is what crosses the link (fetch + upload) and may share a node's link (Phase 3 decision 24). | Keep n·ln(n) and patch only D0 and the upload rate; defer until GPU timings exist | On the cloud data v1 fits both datasets together with sigma 0.08 at 2 threads against v0's 0.49: every estimate within ±20 % instead of a factor of 1.6. The live rounds replay within 4.4 s (over) instead of 10 s (under). |
| 7 | **`Default()` stays v0 to the digit; `Calibrated()` is the measured block.** v1's formulas with P = 1, D0 = 0, no gate, no save and one bandwidth are v0's, so every Phase 1–3 test, sweep and saved log keeps its numbers; the measured block is a second constructor (`costmodel.Calibrated()`, `Model.calibrated()`), chosen by `COST_MODEL=calibrated` on the scheduler and the worker, `-cost calibrated` on the sweep and the comparer, with the `COST_*` / flag overrides applied on top. The block's bandwidths: the fetch is the worker node's median (1.57e8), the upload the cloud store's measured upload rate (5.0e7, the data step), the save the laptop's extended calibration point (6.6e8) until that point runs in the cloud. | Replace the defaults with the measured numbers (rewrites a hundred pinned tests and every saved log for no new information) | The placeholder and the measurement are both legitimate models; naming which one a run used (the sweep's `cost_model` column) is worth more than pretending there was only ever one. |
| 8 | **The calibration point times the worker's whole path** (`calibrate-point --upload-prefix`, on by default in the grid): after the build and the gate it saves the index to disk and hashes it (the save term) and uploads it to the store under `calibration/scratch/<host>/`, deleting it afterwards (the upload term); it records the gate's query count. The CSV gains `gate_queries, save_s, upload_s, upload_bytes`; a v0 CSV still reads, its v1 columns absent. `fit --v1` fits A, P and D0 in log space with a one-dimensional search on D0, the gate per query, the three bandwidths, and emits the v1 `COST_*` block; `--worker-csv` takes the worker side's CSVs when it is another machine, as on the cloud topology. | Fold save and upload into the overhead (Phase 4 decision 37 showed it cannot scale) | The term the model lacked was exactly the part of the path the point did not time. |
| 9 | **The GPU pool is a second overlay on the Phase 4 one, and two spot L4 pools autoscaling from zero** (`deploy/k8s-gcp-gpu/`, `GPU_POOLS=1 scripts/gcp-up.sh`): `gpu-l4x1` (g2-standard-8, one card) and `gpu-l4x2` (g2-standard-24, two cards) for the workers-per-node measurement, both spot, 0..N nodes so nothing bills between sessions, GKE installing the driver and advertising `nvidia.com/gpu`. The overlay keeps everything of `deploy/k8s-gcp` and changes the worker only: the cuVS image (`kgpu-worker-gpu:p5`), `nvidia.com/gpu: 1` in place of the CPU pool's selector (the card is what places the pod; GKE's accelerator label selects the pool family and its `nvidia.com/gpu=present` taint is tolerated), `BUILDER=cagra`, the card's usable memory (22.5 GiB: what an L4's 24 GB reports; the first session measures it), more memory for CUDA, `replicas: 0` between sessions (decision 11); and it adds NVIDIA's DCGM exporter as a DaemonSet on the GPU nodes (privileged, GKE's driver path mounted, as NVIDIA's manifest runs it; unexercised until a card exists) with its ServiceMonitor in `deploy/monitoring/` for the Phase 2 dashboard. The cluster's cost configuration becomes `COST_MODEL=calibrated` (v1's block) in both overlays. Written, rendered (`kubectl kustomize`) and cross-checked before the quota exists; applied once it does. | A separate GPU cluster; on-demand GPU nodes kept up; GKE's managed DCGM metrics (billed per sample) | Phase 4 decision 25's promise: Phase 5 adds the GPU pool to the same cluster and nothing else changes. Autoscaling from zero is also the cold start the phase wants to measure (decision 5). |
| 10 | **The cuVS builder converts with `hierarchy="cpu"` and proves the file loads in stock hnswlib before publishing** (amends decision 2 with the detail the docs demand): `cagra.build` on the card, then `hnsw.from_cagra(hnsw.IndexParams(hierarchy="cpu", M=16, ef_construction=200), index)` and `hnsw.save`. The docs say a conversion with the default hierarchy writes a base-layer-only file "not compatible with the original hnswlib", searchable only through cuVS's own wrapper; the `cpu` hierarchy builds the upper layers on the CPU and its file "is compatible with the original hnswlib library". Since the query path is stock hnswlib (decision 2), the builder loads the saved file with hnswlib before the gate — the gate itself then runs through hnswlib, so a file the query path could not read fails the build with a reason, never reaches the store as a manifest. The GPU image is `rapidsai/base:26.10-cuda12-py3.12` (RAPIDS 26.10, CUDA 12, the cuVS wheel included) with the builder and the worker installed on top (`worker/Dockerfile.gpu`); the CPU image is unchanged (decision 1). The builder is selected per job by `--builder cagra` on the CLI and `BUILDER=cagra` on the worker, the same child-process path as hnswlib (Phase 4 decision 18); without cuVS importable the CLI refuses in one line (exit 1), which is what every laptop and kind run sees, and the tests skip. | Serve the base-layer file through cuVS's wrapper (a second query path); Faiss's cuVS-backed CAGRA (the roadmap's fallback, kept) | The file format is the contract between the GPU side and the CPU side; checking it on the worker, with the library that will read it, is cheaper than any later surprise. |
| 11 | **The cold start is a configured provisioning time that an empty pool reports as its queue wait** (decision 5 made concrete; GPU-free step 3): `COLD_START` on the scheduler (a duration, default 0) and `ColdStartS` in the simulator and the sweep (`-cold_start_s`, a sweep axis, echoed as a column). With no live worker, the queue-wait estimate every policy sees at submit and at every reconsider tick is the provisioning time still to come instead of Phase 3's 0 (that rule — "no workers yet must not scare a policy off the queue" — assumed a pool that is only transiently empty; a pool that autoscales from zero is empty by design until asked). The simulator models the pool itself: its workers are not alive until `ColdStartS` after the first GPU build is queued, so a cold round pays the provisioning time once. The reactive policy then keeps a build local while the pool is cold whenever its local recovery beats cold start + GPU path — recovery, not bare build time: the policy's arithmetic counts the backlog a report shows (Phase 3), so on the by-hand round a 10 s cold start is preempted into the cold pool and a 60 s one is not (the steps 2–4 cross-check corrected a first draft that said 30: with two queued queries and an estimated query per second arriving over the 18.6 s left, staying local costs ~39 s, which 30 + 4.6 still beats). For that to happen at all, a preemption must be allowed to join a pool with no live worker: Phase 3's reactive refused one (the memory filter against no workers, bug 11's guard), which with a cold pool meant nothing below the submit threshold could ever be offloaded — the cross-check's finding A. The rule now: a pool with no live worker that reports a finite, positive wait is COLD and may be joined (its workers are coming, the wait is the price); one that reports 0 is EMPTY and is still never joined. Decision 9's overlay keeps the worker Deployment at 0 replicas between sessions (a pending GPU pod is what provisions a node) and a session scales it up; what the live scheduler reports while cold is the whole provisioning time at every ask, where the simulator counts it down from the first queued build — the live side cannot tell when provisioning began (a stale worker's replacement is a cold start too), the difference is at most one provisioning period, and the GPU session's measurement decides whether the live side should carry a clock (G7 below). What the constant is — node provisioning plus the image pull, minutes — is the first measurement of the GPU session; until then it is 0 everywhere, so nothing changes. The worker Deployment's replicas, not only the autoscaler, make the pool cold between sessions: the GPU overlay carries `replicas: 0` and a session scales it up (`kubectl -n kgpu scale deploy/worker --replicas 2`). | A separate "pool cold" flag on the policies' input (the policies are pure functions of the wait they see; a longer wait is the honest signal); a scheduler-driven scaler (Phase 6's Kueue/KEDA territory) | A term the estimate can carry today, measured when the GPUs come, with the Phase 3 rule kept as the default. |
| 12 | **The GPU containers must find the node's driver where GKE mounts it** (the first GPU session's first failure): GKE installs the NVIDIA driver on the host and mounts it into a card-requesting container at `/usr/local/nvidia`, but the RAPIDS image's library path names only its own CUDA runtime, so the runtime could not load `libcuda` and reported "CUDA driver version is insufficient" (the node's driver, 580, was newer than the runtime, 12.9: the message is misleading). `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64` in the worker's ConfigMap, the calibration Job and the image. Also from the session: GKE's spot L4 pool stocked out in `us-west1-b` ("GCE out of resources"), so an on-demand pool spanning all three zones was added (`gpu-l4x1-od`, landed in `us-west1-c`); the measured cold start is 353 s (about 2.5 min for the node with a failed spot attempt first, 2.3 min to pull the 6 GB image, under a second to register), now `COLD_START=350s`; the L4 reports 22563 MiB free (`WORKER_MEM_BYTES` set to it); the DCGM exporter started as configured and serves the card's utilisation, memory, temperature and power; cuVS 26.10 imports and builds inside the worker. | A custom image with the driver baked in (forbidden by the driver's licence and version-locked to the node) | The driver is the node's; the container only needs to know where. |
| 13 | **The GPU's build time has its own shape, and the speedup-constant form of the model cannot express it** (the session's calibration, speedup.md evidence 10): on the L4, build + conversion + load is 1.15 s + 5.5e-8 · n^0.98 · (dim + 150) with sigma 0.095 — a fixed cost and then LINEAR in n, where the shard's CPU grows as n^1.22·ln(n). The ratio to the shard VM therefore runs from 0.5× at 10k vectors (the GPU is slower: the fixed cost dominates) to 9.5× at 400k and keeps growing; a single `Speedup` fitted as the geometric mean comes out 2.6 with a spread of 1.2, which is no model at all. Cost model v2 gives the worker its own shape: worker build = WorkerFixed + WorkerA · n^WorkerP · ln(n) · (dim + WorkerD0), with the speedup form kept as the fallback when WorkerA is 0 (the CPU-worker topology, v1's block); the calibration's `fit` gets a `--fixed` option that fits the worker's fixed term. The first build of a process pays a one-off warm-up (6.8 s for 10k against 1.2 s on the repeat: the CUDA context and kernels), excluded from the fit and charged nowhere yet. | Keep a size-dependent speedup table; fit the ratio as a power law | The two sides are different machines with different physics; the model should say so, as decision 6 said for the dim term. |
| 14 | **Cost model v2: the worker has its own shape** (decision 13 carried out): worker build = WorkerFixed + WorkerA · n^WorkerP · ln(n) · (dim + WorkerD0) + WorkerGate · GateQueries · ln(n) · (dim + WorkerD0), then the overhead and the save as before; when WorkerA is 0 the v1 speedup form stands, so `Calibrated()` (the CPU-worker block) and `Default()` are unchanged to the digit. `CalibratedL4()` (`COST_MODEL=calibrated-l4`, `-cost calibrated-l4`) is the first GPU session's block: floor 1.154 s, A 1.295e-8, P 0.891, D0 148, gate 2.07e-8 per query, the card's fetch 1.39e8, upload 1.01e8, save 2.9e8 B/s; sigma 0.097 — fitted by `fit --v1 --v2 --fixed --drop-warmup` with the shard VM's CSVs against the L4's (`--worker-csv`): `--fixed` searches a per-build floor on a grid like D0 (the residuals are of the whole time, floor included), `--drop-warmup` leaves out the process's first point (the one-off CUDA warm-up). The sigma the simulator takes is the larger of the two sides' fit errors; no ratio spread exists any more. On the live rounds' 408k build the block's worker path is 13.2 s where the card took 15.0 (v1 said 39.7); the session's two rounds replayed on it come out 31.6 and 32.8 s with both large builds on the card, against the 50.6 and 52.9 the wrong model produced — the second session's prediction. The scheduler, the worker, the sweep (`-worker_fixed -worker_a -worker_p -worker_d0 -worker_gate`, echoed as columns) and the comparer read the five new terms. | A size-dependent speedup table; a second ratio for small builds | Decision 13's reason; and the fallback keeps every earlier number. |

## Step 1 (GPU-free): cost model v1

| Path | What |
| --- | --- |
| `scheduler/costmodel/costmodel.go` | The model: the terms above, `Default()` (v0) and `Calibrated()`; `LocalTotal` for the shard's whole local path; the two buckets `GPUBuild` and `Transfer`. |
| `worker/kgpu_worker/costmodel.py` | The mirror, `Model.calibrated()`, `COST_MODEL` and the v1 `COST_*` names in `from_env`. |
| `scheduler/cmd/scheduler/main.go` | `COST_MODEL` and the v1 variables into the scheduler's model. |
| `experiments/sim/sim.go`, `scheduler/api/api.go`, `scheduler/reconsider/reconsider.go` | The local path is `LocalTotal` now (the fake shard's sleep, the policies' local estimate, the reconsider timer's). |
| `experiments/cmd/sweep/main.go`, `experiments/cmd/validate/main.go` | `-cost default|calibrated` and the v1 term flags; the sweep's CSV echoes them (`cost_model`, `cost_p`, `cost_d0`, `gate`, `gate_queries`, `save_bandwidth`, `bandwidth_out`, `shard_uses_store`). |
| `builder/kgpu_builder/calibrate.py`, `cli.py` | The extended point (decision 8), `fit_points_v1`, `speedup_v1`, `env_overrides_v1`, `fit --v1 --worker-csv --gate-queries`. |
| `experiments/results/calibration/laptop-v1-sift-128-6.csv` | The laptop's extended points (save 660–740 MB/s, upload 150–215 MB/s to the local store). |
| `experiments/results/calibrated/`, `calibrated-v0/` | The Phase 3 comparison on v1's block (speedup.md evidence 9) and, kept beside it, Phase 4's rerun on v0 with calibrated constants. |

**Results.** The cloud CSVs refitted (`fit --v1`, the shard VM's two datasets against the worker node's): build = 2.30e-9 · n^1.220 · ln(n) · (dim + 187) at 2 threads with sigma 0.079 over both datasets (v0: 0.49), the worker node at 8 threads 1.53e-9 · n^1.198 · ln(n) · (dim + 142) with sigma 0.091; speedup 2.126 with a spread of 0.113 across the grid; gate 1.79e-8 s per query·ln(n)·(dim + D0). The laptop's v1 fit of its own data gives D0 = 0 (Phase 4 decision 35's point: the dimension term is the machine's). The live rounds replayed on the block: 45.9 and 46.4 s against 41.5 live (v0 with calibrated constants: 31.5), every local shard within 0.1–2 s, the straggler 4.4 s over — the over-prediction is the worker's upload at the data step's 50 MB/s, to be measured by the extended point on the next cloud session. The comparison rerun on the block (evidence 9) says what Phase 4's did: with CPU workers no policy moves a round at the median, now with the shard's store round trip making every path longer (bimodal 78 s, hour_build 2708, production 24139 at v1's steeper n^1.22; the CSV was regenerated after the save bandwidth moved from the 400 MB/s placeholder to the laptop's measured 660 MB/s, a change of a few seconds on the longest rounds).

### Steps 2–4 (GPU-free): the GPU pool overlay, the cold-start term, the cuVS scaffold

| Path | What |
| --- | --- |
| `deploy/k8s-gcp-gpu/` | Step 2 (decision 9): the overlay on `deploy/k8s-gcp` — `worker-gpu.yaml` (`nvidia.com/gpu: 1`, GKE's accelerator label and taint), `config-gpu.yaml` (`BUILDER=cagra`, the card's memory, `COST_MODEL=calibrated`), `dcgm-exporter.yaml`; `GPU_POOLS=1 scripts/gcp-up.sh` creates the two spot L4 pools autoscaling from zero. Rendered with `kubectl kustomize`; applied once the quota exists. |
| `scheduler/api/api.go`, `scheduler/reconsider/reconsider.go`, `scheduler/cmd/scheduler/main.go`, `experiments/sim/sim.go`, the sweep and the comparer | Step 3 (decision 11): `COLD_START`, `ColdStartS`, `-cold_start_s`. |
| `builder/kgpu_builder/cagra.py` | Step 4 (decision 10): `build_index_cagra` — CAGRA on the card, `from_cagra(hierarchy="cpu")`, `hnsw.save`, and stock hnswlib's `load_index` as the proof; `cuvs_available()`, `CuvsUnavailable`. |
| `builder/kgpu_builder/build.py`, `cli.py` | `BuildSpec.builder` ("hnswlib" or "cagra"), the manifest's `kind` ("cagra-hnsw") and `cuvs_version`; `build --builder cagra`, which fails fast with a reason where cuVS is not importable, before any fetch. |
| `worker/kgpu_worker/builder.py`, `config.py` | `BUILDER=cagra`: the same child process with `--builder cagra` (invariant 9); the Artifact's kind from the manifest. |
| `worker/Dockerfile.gpu` | The GPU image on `rapidsai/base:26.10-cuda12-py3.12` (decision 10); built for amd64 and pushed as `kgpu-worker-gpu:p5` before the first GPU session. |
| `builder/tests/test_cagra.py`, `worker/tests/test_hnswlib_builder.py` | The refusal path and the CLI's fast failure where cuVS is absent; the worker's flag and kind; the GPU-only build-and-gate test, skipped without cuVS. |

The cold-start sweep check (seed 1): `skewed` × reactive and always-GPU at `-cold_start_s 0,120` — at 0 both offload all six builds (30.0 s); at 120 reactive keeps all six local (56.7 s) while always-GPU waits for the pool (140.6 s). What remains for the GPU session: build and push the GPU image, measure the cold start and the card's build times (the calibration with `--builder cagra`, a `builder` column added to the point schema then), refit Speedup and GPUOverhead on the GPU side, and the measurements the roadmap lists.

### The first GPU session (2026-11-02)

The quota (one L4 across all regions) landed on 2026-11-01. The session, in order, with what each step found: the GPU image built on `rapidsai/base:26.10-cuda12-py3.12` (a plain `pip install` refuses the image's managed Python; the sanctioned flag) and pushed, 6 GB; the topology up with `GPU_POOLS=1 GPU_L4X2_MAX=0 WORKER_NODES=0` (no two-card pool, no CPU worker pool, the worker Deployment at 0); the scheduler image rebuilt as `p5` (the running `p4` knew neither `COST_MODEL` nor `COLD_START`: the startup line showed it); the data step on the VM (5 min); the worker scaled to one — the spot pool stocked out, the on-demand pool across three zones took it, **353 s to a registered worker** (decision 12); the card checked from inside the worker (cuVS 26.10, the L4, 22563 MiB free, the DCGM exporter serving); the GPU calibration Job failed on the driver path (decision 12), then ran: **20 points in 4 minutes**, the card's shape (decision 13); the live rounds against the one GPU worker (below).

| Build (vectors × dims) | Shard VM, 2 threads | CPU worker node, 8 threads | **L4 (cagra → hnsw, 6 CPU threads for the conversion)** | L4 over shard |
| --- | --- | --- | --- | --- |
| 10k × 128 | 0.61 s | 0.28 s | 1.22 s | 0.5× |
| 100k × 128 | 10.2 s | 4.5 s | 2.26 s | 4.5× |
| 400k × 128 | 63.3 s | 27.5 s | 6.66 s | 9.5× |
| 10k × 960 | 1.59 s | 0.86 s | 2.09 s | 0.8× |
| 120k × 960 | 51.7 s | 27.7 s | 6.61 s | 7.8× |

Recall at the gate on the GPU-built graphs: 0.999 → 0.988 (sift), 0.998 → 0.961 (gist), above hnswlib's own at the same sizes. The card's fetch 139 MB/s, upload 99 MB/s (median), save 290 MB/s. Logs: `test-logs/phase5/gpu-calibration.log`; CSVs `experiments/results/calibration/gpu-l4-*.csv`.

**The live rounds against the one GPU worker** (`test-logs/phase5/gpu-rounds.log`, timelines `experiments/results/gpu-timelines/`): 50.6 s and 52.9 s — *longer* than Phase 4's 41.5 s with two CPU workers, and the log says why. The scheduler ran cost model v1's CPU-worker block (speedup 2.12), which overestimated the card by 2–2.6× (the worker's own estimate for the 408k build was 39.7 s; the card took 20.4 s the first time, with the process's warm-up, and 15.0 s the second), so the reactive policy sent only the 408k build (and, in round 2, the 136k) to the GPU, kept the 204k build local, then preempted it when the backlog grew and reneged it when the queue behind the 408k's inflated estimate looked long — a build restarted from scratch at 38–40 s, the straggler of both rounds (`att 1` on a local placement is that history: decision 38 gives the renege a new attempt). Under a model that knew the card, the 408k and the 204k would both have been done on the GPU by about 30 s. This is the measurement the session was for: the model's shape, not the hardware, decided the round, and cost model v2 (decision 13) is what fixes it; the second GPU session validates v2 live.

### Cross-check of cost model v2

A fresh agent wrote 14 Go, 10 builder and 44 worker checks (`scheduler/crosscheck/p5v2_costmodel_test.go`, `builder/tests/test_xcheck_costmodel_v2.py`, `worker/tests/test_xcheck_costmodel_v2.py`; log `test-logs/phase5/costmodel-v2-crosscheck.log`) from decisions 13–14: the L4 block field by field, every term by hand on 55 points, Go against Python on seven models, the fallback, tiny builds, validation, the sweep against the demo and with the overrides, the comparer on the session's timelines, the scheduler binary, the fit on the committed CSVs and on exact synthetic data, the worker process with every bad value. 55 passed; the 13 failures came down to three bugs. Triage:

| Finding | Kind | Resolution |
| --- | --- | --- |
| A: `--drop-warmup` dropped the first row in argument order, so the documented fit command, whose glob lists the gist file first, kept the real warm-up point and printed a different block (sigma 0.37). | Bug | The earliest measured point per thread count is dropped (by the point's timestamp). The floor grid was also refined (120 steps), which moved the block's constants within their flat trade-off (floor 1.154, A 1.295e-8, P 0.891, D0 154, gate 2.07e-8; the same sigma 0.097 and the same 13.1 s on the straggler); every pinned number follows. |
| B: the comparer had none of the five worker flags. | Bug (as step 1's B) | Added. |
| C: the worker accepted NaN and infinity in the `COST_WORKER_*` values — NaN fails every comparison, so `COST_WORKER_A=nan` silently fell back to the v1 form — where the scheduler exits 2. | Bug (as step 1's E) | The worker's config refuses non-finite values in every `COST_*` number. |
| G1: the fit's v2 block says `COST_MODEL=calibrated`; the v0 block's "no environment block" note did not say which block. | Pinned / fixed | The block sets every field where the two blocks differ, so `calibrated` plus the overrides is the L4 block; the v0 note names itself. |
| G2: the startup line did not show whether the worker's own shape was in force. | Observability | `worker_shape` echoed. |
| G3: one set of bandwidths serves both sides (the L4 block's local path fetches and uploads at the card's node's rates). | Pinned | Per-side bandwidths would be a v3 term; the difference is 2 s on the 408k straggler's local path. |
| G4: the floor grid's resolution, and A recovered only within ~25 % on exact data because a small floor error trades against P. | Pinned | The grid is 120 steps to 98 % of the smallest time; the trade-off is real (the same sigma at several floor/A/P triples), which is why the block's constants are quoted as a set. |
| G5, G6, G7: a single-size worker CSV with `--fixed` prints v0's shape without a note; `WorkerP` 0 is read as 1; the small presets are insensitive to the worker terms. | Pinned | As decision 8's G6; as `P`; the query horizon. |

### Cross-check of steps 2–4

A fresh agent wrote 43 checks (`scheduler/crosscheck/p5s234_test.go`, `builder/tests/test_xcheck_cagra.py`, `worker/tests/test_xcheck_cagra.py`; log `test-logs/phase5/step234-crosscheck.log`) from decisions 9–11 and the step notes: both overlays rendered and compared object by object, the cold start by hand in the simulator (the clock's start, the countdown, the sweep axis, the demo CSV), on the API and the reconsider timer against a database and on the scheduler binary, the cuVS refusal path against a store that counts connections, the worker's flag, kind, gate and cancel paths with the cagra builder, the GPU image's text. 37 passed, 3 failed, 1 needs a store. Triage:

| Finding | Kind | Resolution |
| --- | --- | --- |
| A: the shipped reactive policy never preempted into a cold pool — its memory filter against "no live worker" (bug 11's guard) skipped every local build, so below the submit threshold nothing could ever be offloaded while cold, and a 0.5 s cold start switched every offload off in `uniform`. The step's own by-hand test had used the zero-value policy, which offloads at submit, and missed it. | Bug | A pool with no worker that reports a finite positive wait is cold and may be joined; one reporting 0 is empty and is not (decision 11 amended). The by-hand tests use the shipped policy; the kept-local example is 60 s (the backlog arithmetic, above). |
| B: `COLD_START=-5s` started the scheduler and every policy saw a −5000 ms wait. | Bug (validation) | Refused at startup; the startup line echoes `cold_start_s`. |
| G1: the overlay rendered `replicas: 2` while decision 11 said the pool is scaled to zero between sessions. | Spec gap | `replicas: 0`; the session scales up. |
| G2: `WORKER_MEM_BYTES` was the modeled 24 GiB; an L4 reports ~22.5 GiB usable. | Spec gap | 22.5 GiB in the overlay; measured in the first session. |
| G3, G4: nothing scraped the DCGM exporter, and it did not mount the driver. | Spec gap | A ServiceMonitor in `deploy/monitoring/`; privileged with GKE's driver path, as NVIDIA's manifest; flagged unexercised. |
| G5, G6: stale comments on `PoolState.QueueWaitS` and the reply field; no startup echo. | Doc / observability | Fixed. |
| G7: the simulator counts the cold wait down; the live scheduler reports the constant. | Pinned | Decision 11 says why, and that the GPU measurement decides. |
| G8: while cold the policies see `LiveWorkers 0, LargestMem 0`. | Pinned | `fitsNowhere` applies only with live workers, so a cold pool refuses nothing at submit; the reconsider filter is the rule above. |
| G9, G10, G11: the cold wait ignores builds already queued ahead; hnswlib's argv carries no `--builder`; a manifest without `kind` takes the builder's; the doc's numbers are seed 1. | Pinned | Documented here. |

### Cross-check of step 1, cost model v1

A fresh agent wrote 15 Go checks (`scheduler/crosscheck/p5s1_costmodel_test.go`), 12 builder checks (`builder/tests/test_xcheck_costmodel_v1.py`) and 8 worker checks (`worker/tests/test_xcheck_costmodel_v1.py`) from decisions 6–8 and the formulas alone: every term by hand, Go against Python on 55 points × 12 terms, `Default()` against a build of the previous commit at every point, the sweep against the committed demo cell by cell, the comparer on the step 4 timelines, the scheduler binary with every `COST_MODEL` spelling, `fit --v1` on the cloud CSVs and on exact synthetic data, the extended point against a throwaway store. Logs `test-logs/phase5/step1-crosscheck.log` (Go). 28 passed first time; 7 failed, two of them real bugs of the step. Triage:

| Finding | Kind | Resolution |
| --- | --- | --- |
| A: `Default()` was not v0 to the digit — `Transfer` truncated fetch and upload to nanoseconds separately where v0 truncated once, a 1–2 ns drift that moved 12 queue-wait cells of the committed demo. Found by building the previous commit beside the working tree. | Bug (against decision 7's promise) | With one bandwidth `Transfer` is computed in v0's single expression; the demo regenerates byte for byte (its new columns aside). |
| B: the comparer lacked seven of the eleven v1 term flags, so a live round run with any v1 override could not be replayed. | Bug | Added, with the same keep-sentinels as the sweep. |
| C: the sweep accepted `-speedup 0`, negative speedups, `NaN` and `Inf` (NaN passes every `< 0` check). | Bug (validation) | Refused; the message names the keep value per flag (0 for P). |
| D: the scheduler accepted `Calibrated`, `CALIBRATED`, `v1`, `v0` for `COST_MODEL`, and `COST_SPEEDUP=0`, `COST_BANDWIDTH=-1`, `COST_P=NaN`; which model ran was unobservable. | Bug | Exactly `default` and `calibrated`; `Model.Validate()` refuses non-positive or non-finite constants at startup (exit 2); the startup line carries `cost_model`. |
| E: the worker silently took v0 for any unknown `COST_MODEL`, so one env block could mean two models. | Spec gap with a hazard | The worker parses the two spellings and the boolean exactly as the scheduler does (Go's `ParseBool` set), validates the v1 fields, and logs `cost_model` at startup. |
| G1: which terms are zero below two vectors. | Pinned | The build and the gate (they are per-graph); fetch, save, upload and the overhead are charged — a one-vector build still moves bytes and starts a process. |
| G2: rounding — per term or per composite. | Pinned | Each Go term is a whole-nanosecond `Duration`; composites may differ from a float sum by a few ns; Python keeps floats; the Go–Python agreement is "within nanosecond rounding". |
| G3, G10: `fit --v1` assumes 500 gate queries on a v0 CSV, and `--gate-queries` defaults to 10000. | Pinned | Documented in the fit's docstring and `--help`: v0 points ran `run_point`'s default of 500; the block's query count is the split's query file (10000 sift-128, 1000 gist). |
| G4: `COST_GPU_OVERHEAD=1.5` is not fitted. | Pinned | Decision 8: the constant the terms do not cover (process start, the manifest), set from the live rounds (decision 37); the GPU calibration refits it. |
| G5: what `KGPU_ESTIMATE_SIGMA` means in v1. | Pinned | The largest of the shard fit's sigma, the worker fit's and the speedup's spread across the grid (0.113 on the cloud): the worst multiplicative error any estimate the scheduler makes carries, which is what the simulator's log-normal factor models. |
| G6, G7: a single-size fit printed P 1.000, D0 0 as if fitted; two unlabelled "threads 8" lines with `--worker-csv`. | Spec gap | A note names the unidentifiable shape; the side is labelled at the end of the line (the line's shape is parsed and kept). |
| G8, G9, G11, G13: the worker's upload and save rates win when both sides carry them; the block omits terms no point measured; an untimed upload records 0; `MemFactor` stays 2. | Pinned | Documented in decisions 7 and 8. |
| G12: the boolean's accepted spellings differed between the sides. | Fixed with E | Both accept Go's `ParseBool` set. |

## What the published figures change here

- **GPU memory sizing is a hard constraint.** An L4 has 24 GB; the OpenSearch RFC built 5M x 1536 on a 24 GB A10G, and 10M at that width does not fit. The placement memory constraint added in this phase must use measured peak memory per size and dimension, and the node pool choice (L4 24 GB vs A100/H100 40 to 80 GB) decides which shards can be offloaded at all.
- **Cold start is comparable to the build.** A GPU build of 1 to 10M vectors is 17 s to 4 minutes (RFC); node provisioning is minutes. The cold-start term in the estimate is therefore of the same order as the work, which is why the cost-based policy should visibly prefer CPUs while the pool is cold.
- **A spot preemption costs up to a whole build**, minutes of GPU work, not the seconds the fake builds suggested. Record lost GPU-minutes per preemption; whether checkpointing partial graphs is worth it becomes a real question (noted below).
- **Transfer inside the cluster matters.** 6 to 61 GB per shard means the worker's network path to the object store is part of the GPU cost; place workers and MinIO in the same zone and measure the bandwidth the worker actually gets.
- **Expected GPU numbers to compare against:** 1M x 128 in 17 s and 10M x 128 in 227 s on an A10G; 50M in under 30 minutes on an H100.

## What the worker-count question changes here

Phase 2 decisions 22 to 24 put a fake card resource (`kgpu.io/gpu-card`) on two stand-in nodes, one card per worker. This phase is where it becomes real:

- **The only manifest change is the resource name.** The device plugin advertises `nvidia.com/gpu: M` on each GPU node; the worker's limit becomes `nvidia.com/gpu: 1`; the taint, selector, probes and grace period are the Phase 2 ones. A worker beyond the card total stays Pending with the same "Insufficient" message the kind check already asserts.
- **Workers per node is a measurement of this phase.** On a two-card machine, run one worker and then two, at 128 and at 1536 dimensions, and record per-build transfer time and build time. The ratio of two-worker to one-worker throughput is the per-node bandwidth term the Phase 3 cost model needs; the expectation is "almost 2x at 128 dims, well under 2x at 1536". The node pool choice is part of the experiment: one-card and two-card machines of the same card type (e.g. L4) cost about the same per card and differ only in this sharing.
- **Builds per card is a measurement too, and probably a win for small builds.** A build has phases: vectors stream in (the card's compute idle), the graph is built (compute saturated), the result copies out. Two builds sharing a card overlap one's transfer with the other's compute, and a small build (most of a Zipf distribution) cannot fill a modern card by itself, so two at once can approach twice the throughput. The losses: two large builds just interleave and the bigger one, the likely straggler, finishes later; and both graphs must fit in card memory at once, where overflow is a crash. Two ways to get there: the device plugin's time-slicing (each card advertised as N slots; blunt, because Kubernetes cannot see memory or build size) or in-process concurrency in the worker, which admits a second build only when the modeled memory of both fits and ideally only while the first is in a transfer phase. The worker knows every build's memory estimate, which is exactly what the device plugin lacks, so the second is the design to try; each build keeps its own lease and attempt, so the invariants hold. Measure one build per card against two with memory-aware admission, at 128 and 1536 dimensions, reporting throughput and the largest build's finish time, since round time is set by the straggler. The modeling assumption "a worker runs one build at a time" stays the default behind a config flag (owner's question).

## Open questions

- Is checkpointing a partially built CAGRA graph feasible, so a spot preemption loses minutes rather than a whole build? Only worth asking if measured preemption loss is large.

- Faiss with cuVS-backed CAGRA as the fallback build path if the direct cuVS API causes trouble.
- Which cloud. GKE is the roadmap's example because its GPU node pools and autoscaler are well documented.
