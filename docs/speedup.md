# Observed speedup from offloading builds to GPUs

The project's headline question — **how much does offloading an index build to a GPU actually speed things up?** — answered honestly at every point in time. The honest answer has three levels, because "speedup" means three different things:

1. **Per-build**: how much faster one build finishes on a GPU than on its shard's CPU.
2. **Per-round**: how much sooner the whole cluster recovers from the DDL (round completion time — one slow shard holds everyone up).
3. **Per-shard recovery**: how much sooner the *other* shards drain their blocked queries, even when the round clock doesn't move.

Each number is also labeled by how we know it: **assumed** (a cost-model parameter), **simulated** (the discrete-event simulator running on that assumption), or **measured** (a real build on real hardware). Nothing is measured yet; that is Phase 5's job.

This doc is updated whenever a sweep, a live validation or a real measurement produces a new number: the current-answer table below is rewritten in place, and a numbered entry is appended to the evidence log (append at the end, never insert).

## Current answer

| Question | Current answer | How we know |
| --- | --- | --- |
| Per-build speedup | **~5–9× effective**, from an *assumed* 10× compute speedup minus a 2 s GPU overhead and a 200 MB/s transfer; small builds lose most of it to overhead | Assumed (cost model); simulator and fake worker both run on it |
| Per-round speedup, current presets | **~0 at the median in the query-bound presets, on any model** (evidence 8, 9, 11: the round is the drain of the queries); **9.5× on `hour_build` with the L4 block** (2708 → 284 s, evidence 11), none on `production` (its largest build fits no worker). The first live GPU rounds were slower than the CPU-worker ones because the scheduler's model did not know the card's shape (evidence 10); the second session validates v2's 32 s prediction. Every policy — including all-local — produces the same round time, because the presets are bound by physics placement can't touch (see evidence 1); the full comparison (evidence 5) confirms it at 50 seeds across all nine scenarios | Simulated |
| Per-round speedup once the largest build fits a worker | **~9.5× on the measured model** (`hour_build` on the L4 block, evidence 11); earlier **~7.0×** on the placeholder (`production` preset with 160 GB workers: ~9.7 h all-local → ~1.4 h reactive), and the dynamic policy beats naive always-offload by a further **~20%** (see evidence 2) | Simulated |
| Per-shard recovery, `production` preset | The five offloadable shards recover **~2.1 h sooner** per round; the round clock doesn't move because the sixth shard's build fits no GPU | Simulated |
| Per-build speedup, measured | **On an L4, a curve, not a number: 0.5× at 10k vectors, 4.5× at 100k, 9.5× at 400k over the shard VM's 2 threads** (the card has a 1.15 s floor and then grows linearly with n; evidence 10). For a CPU worker with 8 threads over the shard's 2: 2.1–2.2× (evidence 7, 9) | Measured (cloud; decision 7 says the number that counts comes from the cloud topology) |
| The cost model's CPU constant and its error, measured | **Cost model v1 (Phase 5 decision 6): build = 2.30e-9 · n^1.22 · ln(n) · (dim + 187) on the shard VM at 2 threads, sigma 0.08 over both datasets** (evidence 9); v0's one constant per dataset had sigma 0.49 together because its "time ∝ dim" is wrong on x86 (evidence 7, decision 35: a 960-dim build costs 2.8–3.6× a 128-dim one, not 7.5×, and 8.0× on the laptop — the dimension term is the machine's) | Measured (cloud), refitted |
| Does the simulator's number survive contact with the live stack? | **Yes. With a GPU, on cost model v2: placement identical in 8 of 8 live rounds, the offloaded and query-bound shards within 0.01 s, one local build 4–6 s slow on the shared VM** (evidence 12, decision 15). With real CPU builds on v1: within 4.4 s of a 41.5 s round (evidence 9). On v0 with calibrated constants it was 10 s under, because v0 charged the index's upload at the fetch's rate and had no gate or save (evidence 8, decision 37). With modeled builds: within ~2 s at 50 shards and within 1 s over 7- and 9-minute rounds, 18 of 18 live rounds on the kind cluster agree with the simulator's replay at the same seed and cadences (evidence 4). The simulator cannot know the live reconsider ticker's phase; at 1536 dims that moves a round by up to ~15 s, and the comparer reports the envelope | Measured (fake builds: the live stack sleeps for the same modeled times the simulator uses, so this validates the *mechanics*, not the cost model) |

## The assumption underneath everything

All modeled times come from one place: `scheduler/costmodel/costmodel.go`, `Default()`. A local build takes `A · n · ln(n) · dim` seconds (`A = 1.4e-7`); a GPU build takes the local time **÷ 10** (`Speedup`), **+ 2 s** fixed overhead (`GPUOverhead`), **+ the vectors over a 200 MB/s node link** (`Bandwidth`, shared by workers on the same node). So the effective per-build speedup grows with build size:

| Build | Local (modeled) | GPU (modeled) | Effective speedup |
| --- | --- | --- | --- |
| 100k vectors × dim 128 | 20.6 s | 4.3 s | **~4.8×** — the 2 s overhead eats small builds |
| 1M × 128 | 248 s | 29 s | **~8.4×** |
| 10M × 1536 (the `production` preset's largest) | ~9.6 h | ~63 min | **~9.2×** — but it never happens: the build needs ~125 GB of GPU memory (index bytes × `MemFactor` 2) against the preset's 80 GB workers, so it builds locally under *every* policy |

The 10× is a placeholder chosen in Phase 1, in the plausible range for graph-index builds but not from our hardware or our libraries. Both the live fake worker and the simulator obey it, which is what makes step 7's simulator-vs-live comparison meaningful — and what makes the whole column worthless until Phase 5 replaces the placeholder with a measurement.

**External reference points for the placeholder**: [NVIDIA's FAISS + cuVS benchmarks](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs) report CAGRA building **up to 12.3× faster than CPU HNSW** on Deep100M (same build-on-GPU, serve-as-HNSW path we assume; [Meta's writeup](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/) matches); [Zilliz's CAGRA post](https://zilliz.com/blog/Milvus-introduces-GPU-index-CAGRA) quotes a 50M-vector rebuild at 8–12 h on a 32-core CPU vs <30 min on one H100 (~16–24×); the CAGRA paper (arXiv 2308.15136) reports 2.2–27× depending on dataset and recall target. So 10× is a defensible prior, with three caveats for Phase 5: published numbers are at chosen recall targets (build speed trades against graph quality), on H100-class hardware, and sometimes exclude the CAGRA→HNSW conversion and transfer that our pipeline must count (the cost model already charges the transfer). NVIDIA's own harness is [cuvs_bench](https://docs.rapids.ai/api/cuvs/stable/cuvs_bench/), a candidate to reuse or cross-check in Phase 5.

## Evidence log

### Evidence 1 — the all-local baseline: round time is physics-bound at the preset loads

The demo sweep compares policies against each other but had no "never offload" baseline, so it couldn't answer *this doc's* question. Added one: `static_threshold` with an impossibly large threshold offloads nothing (the sweep CSV confirms `builds_gpu=0, builds_local=6` on every row). Two runs, reproducible by seed:

```
go run ./experiments/cmd/sweep -scenarios skewed,uniform,high_dim \
  -policies static_threshold:9999999999999,always_gpu,reactive,reactive_spec \
  -rounds 20 -out baseline.csv
go run ./experiments/cmd/sweep -scenarios production \
  -policies static_threshold:999999999999999,always_gpu,reactive_spec \
  -rounds 5 -out prod.csv
```

(Sweep CSVs outside `experiments/results/demo/` are gitignored; the commands above regenerate these byte-identically.)

**Result: the all-local baseline ties every GPU policy on round time, on every preset tried.**

- `skewed` and `uniform`: median round ≈ 30 s under all five policies. The query streams' horizon outlives the builds, so the round ends when the streams drain, wherever the builds ran.
- `high_dim`: ≈ 536 s under all policies. The binding build exceeds every worker's memory and builds locally for everyone.
- `production`: round time is **byte-identical per seed across policies** (~34,900 s ≈ 9.7 h). The 10M × 1536 build fits no 80 GB worker, builds locally under every policy, and alone sets the round. A ~9× per-build speedup on the other five builds moves the round time by exactly zero.

Where offloading *does* show: in `production`, the straggler's lag over the rest of the cluster grows from ~25,300 s (all-local) to ~32,900 s (GPU policies) — read correctly, that means the five offloadable shards finish their builds and drain their blocked queries **~2.1 hours earlier** per round. Real blocked work unblocked; invisible to the round metric while one shard is un-offloadable.

**Consequence for the experiment design** (feeds the FULL policy comparison after step 7): if round time itself is to discriminate between policies, the presets must make the largest build offloadable — bigger worker memory, a size cap, or a dedicated scenario — otherwise every policy ties by construction. Raising query load (`-load`) or shrinking the horizon are the other levers that move the binding constraint from stream physics to placement.

### Evidence 2 — ~7× round-level speedup once the largest build fits a worker

Evidence 1 showed `production`'s round pinned by the 10M × 1536 build that no 80 GB worker can hold. Lifting just that constraint (160 GB workers; the build needs ~125 GB) makes the round-level value of offloading — and of *dynamic* placement specifically — visible:

```
go run ./experiments/cmd/sweep -scenarios production \
  -policies static_threshold:999999999999999,always_gpu,reactive,reactive_spec \
  -rounds 5 -worker_mem_gb 160 -out prod160.csv
```

Median round time over 5 seeds:

| Policy | Median round | vs all-local |
| --- | --- | --- |
| all-local baseline | 34,905 s (~9.7 h) | — |
| `always_gpu` | 6,183 s (~1.7 h) | 5.6× |
| `reactive` / `reactive_spec` | 4,959 s (~1.4 h) | **7.0×** |

Two findings. First, the headline: with every build offloadable, **offloading cuts simulated cluster recovery ~7×** on the production-scale workload (6 shards of 1M–10M × 1536 on 2 GPU workers). Second, the scheduling result: `reactive` beats `always_gpu` by **~20%** and cuts its straggler lag from 2,200–4,500 s to a tight ~690 s, because it keeps small builds local instead of queueing everything behind two workers — the first preset configuration where round time discriminates between policies at all, confirming evidence 1's consequence. Both numbers inherit the assumed 10×/2 s/200 MB/s cost model and await step 7 (live validation) and Phase 5 (measured build times).

### Evidence 3 — pre-registered prediction for Phase 5's measurement

Written down *before* anything is measured, so the measurement can grade it. The simulator maps any measured per-build speedup N to a round-recovery factor M (production preset, 160 GB workers, all-local baseline fixed at 9.7 h):

```
go run ./experiments/cmd/sweep -scenarios production \
  -policies static_threshold:999999999999999,reactive \
  -rounds 5 -worker_mem_gb 160 -speedup 4,6,8,10,12,15 -out prodN.csv
```

| If N = | 4× | 6× | 8× | 10× | 12× | 15× |
| --- | --- | --- | --- | --- | --- | --- |
| M = | 3.6× | 4.5× | 5.8× | 7.0× | 8.2× | 9.7× |

M is sub-linear in N because the transfer and the query-drain tail don't shrink with the GPU. **Prediction**: on an H100-class GPU, N ≈ 8–12× (the published 12.3× minus the CAGRA→HNSW conversion we must count, plus our GPU-friendly 1536 dims), hence M ≈ 5.8–8.2×; on a modest cloud GPU (L4/A10G, the likelier Phase 5 rig), N ≈ 4–8× with a point estimate of ~6×, hence M ≈ 3.6–5.8×, point ~4.5×. The policy-vs-policy ratios (reactive ~20% over always-offload, 3–6× tighter straggler lag) are predicted to survive any N in that range, widening slightly at low N where queueing penalizes always-offload more.

### Evidence 4 — the live kind cluster reproduces the simulator's rounds (Phase 3 step 7)

**Setup.** `scripts/k8s-check-step7.sh` (eight configurations, eighteen rounds; H added after the step's cross-check): the kind cluster at `FAKE_TIME_SCALE=1` (so the live 0.5 s report, 2 s reconsider tick and 10 s lease are the modeled seconds the simulator defaults to), images rebuilt from the checkout, 24 GiB modeled worker memory on both sides (decision 26), every policy switch verified from the scheduler pod's startup line (bug 15). Each live round's timeline replayed by `experiments/cmd/validate` from the same scenario and seed with link sharing off (decision 22). Log: `test-logs/phase3/step7-validation.log`; the table is in the Phase 3 design doc, step 7.

**Result.** 18/18 agree within max(2 s, 5 %), one of them only inside the tick-phase envelope (decision 29):

| Regime | Live − simulated | Example |
| --- | --- | --- |
| Horizon-bound (skewed 6/2, both policies at 50/3 under reactive) | +0.01 s | 30.01 vs 30.00 |
| Queue-bound (skewed 50/3 always_gpu) | +0.2 .. +2.1 s, live slower | 51.96 vs 49.84 |
| Transfer-bound at 1536 dims (high_dim 6/2 reactive) | −1.9 .. −13.6 s; envelope over tick phases 90.6..105.6 s contains every live round | 91.58 vs 105.13 (default phase), 90.63..105.63 (envelope) |
| One modeled hour on the CPU, 433 s on the GPU (71 s transfer + 362 s build; hour_build, both policies) | +0.2 s over 438 s | 438.42 vs 438.20 |
| Memory-bound (high_dim at the preset's 1.13 GB workers: the two largest builds fit no worker and build locally, both sides) | +0.7 .. +0.8 s over 537 s | 536.82 vs 536.06 |

**What it does and does not validate.** The live workers are fake: they sleep for the cost model's times, so this is a validation of the *mechanics* the simulator replays — report latency, claim order, preemption and reneging at the tick, lease renewal over minutes, the transfer term's arithmetic — not of the 10×/2 s/200 MB/s cost model, which both sides share and Phases 4–5 replace. The systematic +1.4 s at 50 shards is control-plane latency (50 shards' reports and claims through one HTTP server and Postgres) the simulator does not model; it is 3 % of a 50 s round and would be 0.04 % of an hour-long one. The submit order of 50 concurrent shards and the ticker's phase are the two live inputs the simulator cannot know; the first only permutes which shard finishes k-th, the second can reorder a preempted queue and is reported as an envelope. No simulator or cost-model constant was changed to obtain these numbers.

**Consequence for the ledger.** Evidence 2's ~7× and ~20 % are now numbers from a validated instrument, still built on an assumed cost model. The next change to this column comes from Phase 4's measured CPU constants and Phase 5's measured GPU speedup, not from more simulation.

### Evidence 5 — the full policy comparison on the validated simulator (Phase 3 closed)

**Setup.** `scripts/sim-sweep-full.sh`: nine scenarios × five policies × 50 seeds at the baseline, every roadmap sweep axis one at a time around `skewed` and `high_dim` at 30 seeds; 8,711 rounds, all completed; `experiments/results/full/`. The table of results is in the Phase 3 design doc ("The full policy comparison").

**Result, as speedups.** Where the round is horizon-bound (the 6-shard presets at their loads) every policy ties at the 30 s horizon: the *round-level* speedup from offloading is 1.0× there, by construction. Where something else binds:

| Arm | always_gpu → reactive | reactive → clairvoyant (what knowing the future is worth) | The threshold policy |
| --- | --- | --- | --- |
| 50 shards on 3 workers | 49.0 → 30.2 s p50: **1.6×** | 30.2 → 30.2: nothing left | 32.7 |
| dim 1536 | 95.1 → 93.9 p50, 120 → 105.5 p99: 1.01× / **1.14×** | 93.9 → 89.5, 105.5 → 90.4: 5 % / 17 % | 259.6: **2.8× worse** than reactive |
| dim 768 | 53.2 → 54.3 p50, 67.0 → 58.7 p99: 0.98× / **1.14×** | 54.3 → 52.0, 58.7 → 52.9: 4 % / 11 % | 135.8: 2.5× worse |
| 1 worker | 32.2 → 30.0: **1.07×** | — | 32.7 |
| σ = 0.6 | reactive p99 32.8 vs reactive_spec 30.3: **speculation recovers 8 %** | the oracle degrades to 30.8 / 33.4 | 32.7 |

**What it changes in the current-answer table.** Nothing at the median of the current presets (still ~0, physics-bound). The round-level speedup *of the dynamic policy over naive offloading* is now a measured 1.6× at the queue-bound 50-shard arm (evidence 2's 20 % was the production preset at 160 GB workers) and ~1.14× at p99 in the transfer-bound high-dimension arms, with the oracle a further 5–17 % ahead. The fixed threshold is not a contender. All of it inherits the assumed 10× / 2 s / 200 MB/s cost model; Phase 4's calibration and Phase 5's GPU measurement rerun this sweep on measured constants.

### Evidence 6 — the calibration method, rehearsed on the laptop (Phase 4 step 3)

**Setup.** `python -m kgpu_builder calibrate` (Phase 4 decision 23): real hnswlib builds of prefixes of SIFT-128's 408k shard (10k–400k vectors) and the GIST subset's 122k shard (10k–120k × 960 dims) at 2 and 8 threads, two repeats, each point in its own process, the vectors fetched from the kind cluster's store through a port-forward; then `fit` (decision 24). Ten-core laptop: a rehearsal of the method, not a ledger number (decision 7). Log `test-logs/phase4/step3-calibration.log`; points and figure under `experiments/results/calibration/`.

**Result.**

| Quantity | Measured | Phase 1 placeholder |
| --- | --- | --- |
| A (s per n·ln(n)·dim), 2 threads, log-space fit | 6.08e-8 (a linear-space fit says 7.46e-8 and hides a −0.21 mean log residual — the step 3 cross-check's finding) | 1.4e-7 (2.3× pessimistic for these cores) |
| A, 8 threads | 2.97e-8 | — |
| 8 threads over 2 ("speedup" with CPU workers) | **2.05×** | 10× was the GPU placeholder |
| Shape | t ∝ n^1.27–1.30 (5 % residuals); the n·ln(n) fit is off 0.69..1.52 across the grid | n·ln(n) |
| Estimate error of the n·ln(n) fit (RMS of the centred log residuals) | **0.22–0.25** | the sweeps assumed 0, 0.3, 0.6 |
| Per-build fixed cost (fetch + gate), 8 threads | 1.18 s | 2 s |
| Fetch from the object store | 153–213 MB/s (median 181) | 200 MB/s |
| Recall@10 at the gate's ef_search 100 | 0.998 → 0.984 (SIFT 10k → 400k), 0.989 → 0.939 (GIST 10k → 120k) | gate at 0.9 |

**What it changes.** Nothing in the current answers' *round-level* numbers yet — those wait for the cloud calibration — but two things about how they will be produced: the sweeps' sigma axis has a measured point (0.25) instead of three guesses, and the cost model's shape is now an open question with data behind it (the power law would predict the hour-long build 1.9× faster than n·ln(n) scaled from the small shards does). Both go to the Phase 3 decisions when the cloud data arrives.

### Evidence 7 — the calibration on the cloud topology, both sides (Phase 4 step 4)

**Setup.** The same grid as evidence 6, run where it counts (Phase 4 decision 7): on the shard VM (`kgpu-shards`, n2d-standard-16, AMD EPYC 7B13, 16 vCPUs, `deploy/gcp/vm-session.sh calibrate`) and on a worker node (n2-standard-8, Intel Xeon 2.8 GHz, as a Job with the worker image, `deploy/k8s-gcp/calibrate-job.yaml`), each reading the splits from the cluster's object store over the VPC. 2 and 8 threads, two repeats, sift-128 10k–400k and gist-960 10k–120k; 40 points a side. CSVs: `experiments/results/calibration/kgpu-shards-*.csv` and `worker-*.csv`; logs `test-logs/phase4/step4-calibration-shard.log` and `-worker.log`; figure `kgpu-shards-calibration.png`.

**Result.** Per dataset, because the two do not share a constant (below):

| Quantity | Shard VM (AMD) | Worker node (Intel) | Laptop rehearsal (evidence 6) |
| --- | --- | --- | --- |
| A, sift-128, 2 threads | 6.53e-8 (sigma 0.26) | 7.00e-8 (sigma 0.26) | 6.08e-8 (both datasets) |
| A, sift-128, 8 threads | 2.14e-8 (sigma 0.23) | 2.91e-8 (sigma 0.22) | 2.97e-8 |
| A, gist-960, 2 threads | 2.81e-8 (sigma 0.27) | 3.85e-8 (sigma 0.31) | — |
| A, gist-960, 8 threads | 1.07e-8 (sigma 0.42) | 1.43e-8 (sigma 0.27) | — |
| 8 threads over 2, same machine, sift | 3.06× | 2.41× | 2.05× |
| Shape, sift (power law) | n^1.29 / n^1.24 | n^1.29 / n^1.26 | n^1.27 / n^1.30 |
| Shape, gist | n^1.39 / n^1.56 | n^1.42 / n^1.39 | — |
| Fetch from the store | 165 MB/s | 151–158 MB/s | 153–213 |
| Fetch + gate per point, sift / gist | 1.3 / 3.1 s | 1.35 / 3.2 s | 1.18 |
| Both datasets in one fit, 2 threads | A 4.66e-8, sigma **0.49**, ratios 0.39..2.08 | A 5.51e-8, sigma **0.41** | sigma 0.22 |

**The block applied to the cluster for the sift-128 rounds** (`deploy/k8s-gcp/config-calibrated.yaml`): `COST_A=6.525e-08` (the shard's), `COST_SPEEDUP=2.24` (6.525 / 2.911), `COST_GPU_OVERHEAD=1.35`, `COST_BANDWIDTH=1.57e8`, `KGPU_ESTIMATE_SIGMA=0.26`.

**Three things it says.** (1) **The sift constant is stable across three machines** (6.1, 6.5, 7.0e-8 at 2 threads) and the laptop rehearsal was within 10 % of the cloud: the method transfers. (2) **The dimension term is wrong, and wrong differently per machine.** On the cloud's x86 cores a 960-dim build at equal n costs 2.8× (2 threads) to 2.2× (8 threads) a 128-dim one, against the model's 7.5×; the laptop's Apple cores gave 8.0×. hnswlib's distance kernels fill wide vector units (AVX-512: 16 floats per instruction) with 960 floats and waste them on 128, so time per dimension falls as dim grows; narrow NEON units have less to waste. One A for both datasets has sigma 0.49 (every estimate off by ×1.6 either way); per dataset, 0.26. So A is per dataset (decision 35) and the dim term joins the shape question for the Phase 5 refit. (3) **"Speedup" across a family mix is partly the family.** At 2 threads the AMD VM is 7 % faster than the Intel node, at 8 threads 36 % faster; the 2.24× the cluster runs with is what the scheduler actually sees (a shard build here against a worker build there), and the same-machine ratios bracket the thread effect at 2.4–3.1×.

### Evidence 8 — live rounds on the calibrated cloud topology, and the comparison rerun on its constants (Phase 4 step 4)

**Setup.** Two sift-128 rounds (`skewed`, 6 shards, seeds 1 and 2) with the shards on the VM building locally as real child processes at 2 threads and two CPU workers in the cluster at 8, policy reactive, the cluster running evidence 7's block. Log `test-logs/phase4/step4-rounds.log`; timelines `experiments/results/step4-timelines/`. Then the step 7 comparer replayed each round in the simulator on the same constants (it now takes `-dataset` and the cost flags). Then the roadmap's eight scenarios × every policy × 20 seeds in the simulator on those constants with sigma 0.26: `experiments/results/calibrated/rounds.csv` and `rounds.png`.

**Result, the live rounds.**

| | Round 1 | Round 2 |
| --- | --- | --- |
| Round time, live | 41.5 s | 41.5 s |
| Placement | 408k, 204k, 136k offloaded; 82k, 102k, 68k local | same |
| Straggler | the 408k build on a worker, 35.3 s (its own estimate 24.0 s) | 34.8 s |
| Simulator on the same constants | 31.5 s | 31.8 s |
| Local shards, live − sim | 0.0–0.1 s | 0.0–2.3 s |
| Offloaded straggler, live − sim | +10.0 s | +9.7 s |

**What it says.** (1) **The local side of the model is right**: every locally built shard finishes within a tenth of a second of the simulator's prediction (one 2.3 s case where the drain ran long). (2) **The offloaded side is short by a size-dependent 6–46 %**, because the worker's path is fetch + build + gate + save + upload + manifest, and the model has the fetch, the build, the upload at the fetch's rate (the store's upload is three times slower, 46–54 MB/s against 157) and a 1.35 s constant standing in for the gate and the save, which both grow with the build (Phase 4 decision 37). (3) So for a real round **the simulator's number carries a −25 % bias on an offloaded straggler** until the Phase 5 refit adds the result-transfer term; the comparison below is therefore honest about local builds and optimistic about offloaded ones, in a direction that flatters offloading.

**Result, the comparison on the calibrated constants** (900 rounds; median round seconds):

| Scenario | always_gpu | static_threshold | reactive | reactive_spec | clairvoyant |
| --- | --- | --- | --- | --- | --- |
| uniform, skewed, downstream_heavy, shrinking_pool | 30 | 30 | 30 | 30 | 30 |
| staggered | 40 | 40 | 40 | 40 | 40 |
| bimodal | 52 | 52 | 52 | 52 | 52 |
| high_dim | 257 | 257 | 257 | 257 | 257 |
| hour_build | 850 | 847 | 847 | 847 | 847 |
| production | 16400 | 16400 | 16400 | 16400 | 16400 |

**What it says.** With CPU workers the per-round speedup is **zero at the median in every preset**: a 2.24× build speedup minus a queue and a transfer cannot move a round that is bound by its query horizon (evidence 1) or by a build that fits no worker (`production`). This is evidence 2's condition restated with measured numbers — the round moves only when the largest build fits a worker *and* the worker is much faster than the shard — and it is why Phase 5's GPU measurement is the one that decides anything. The policies agree with each other to the second, which is also what evidence 5 found at the placeholder constants: the dynamic policies lose nothing to the clairvoyant one.

### Evidence 9 — cost model v1 and the comparison rerun on it (Phase 5 step 1, GPU-free)

**Setup.** The refit the Phase 4 findings asked for (Phase 5 decisions 6–8): build = A·n^P·ln(n)·(dim + D0), a gate term, a save term, separate fetch and upload bandwidths, the shard's store round trip as a flag. Fitted on the Phase 4 cloud CSVs (`fit --v1`, the shard VM's two datasets, `--worker-csv` the worker node's), the laptop's extended point for the save bandwidth; the block is `costmodel.Calibrated()`. Then the two live rounds replayed on it, and the roadmap's eight scenarios × every policy × 20 seeds with sigma 0.11 (`experiments/results/calibrated/`; Phase 4's rerun on v0 moved to `calibrated-v0/`).

**Result.**

| Quantity | v0 with calibrated constants (evidence 7–8) | v1 (this entry) |
| --- | --- | --- |
| Estimate error, both datasets in one fit, shard VM at 2 threads | sigma 0.49 (ratios 0.39..2.08) | **sigma 0.08** (0.86..1.20) |
| The same on the worker node at 8 threads | 0.41 | 0.09 |
| Live rounds replayed (41.5 s live) | 31.5 s (−10.0, under) | 45.9 s (+4.4, over) |
| Local shards, live − sim | 0.0–2.3 s | 0.0–2.0 s |
| Dimension ratio, 960 over 128 dims at equal n | 7.5× (measured 2.8) | 3.6× |
| Speedup, CPU worker over shard | 2.24 (ratio of A's) | 2.12 (geometric mean over the grid, spread 0.11) |
| Comparison, policies within a second of each other | all 9 presets | all 9 presets |

**What it says.** (1) **The model is now right to ±20 % on both sides and both datasets**, from a factor of 1.6; the sweeps' sigma axis takes 0.11. (2) The remaining 4 s on the straggler is the upload term's placeholder (the store's 50 MB/s from the data step, not a worker's measurement) — the extended calibration point will replace it on the next cloud session; the model's error is now in a constant, not its shape. (3) **The comparison's verdict does not change**: with CPU workers, no policy moves a round at the median in any preset, and the rig's store round trip on the local side makes every path longer; what decides the project is still Phase 5's GPU measurement, which now lands in a model that can hold it.

### Evidence 10 — the first GPU session: the card measured, and a round run under the wrong model (Phase 5)

**Setup.** One NVIDIA L4 (g2-standard-8, on-demand after the spot pool stocked out; us-west1-c) in the Phase 4 cluster, the GPU worker image on RAPIDS 26.10 with cuVS 26.10, builds as CAGRA on the card converted to an hnswlib file on 6 CPU threads (Phase 5 decision 10). The calibration grid as a Job on the node (`test-logs/phase5/gpu-calibration.log`, `experiments/results/calibration/gpu-l4-*.csv`); then two live sift-128 rounds with the six shards on the VM and that one worker, the scheduler still on v1's CPU-worker block (`test-logs/phase5/gpu-rounds.log`).

**Result, the card.**

| Build | Shard VM, 2 threads | L4, build + conversion + load | L4 over shard |
| --- | --- | --- | --- |
| 10k × 128 | 0.61 s | 1.22 s | **0.5×** |
| 100k × 128 | 10.2 s | 2.26 s | 4.5× |
| 400k × 128 | 63.3 s | 6.66 s | **9.5×** |
| 120k × 960 | 51.7 s | 6.61 s | 7.8× |
| The card's shape (19 points, the warm-up excluded) | — | 1.15 s + 5.5e-8 · n^0.98 · (dim + 150), sigma 0.095 | — |
| Cold start (pool at zero → registered worker) | — | 353 s (node ~2.5 min, image pull 2.3 min) | — |
| Recall@10 at the gate (sift 10k → 400k) | 0.998 → 0.984 (hnswlib) | 0.999 → 0.988 | — |
| The card's fetch / upload / save | — | 139 / 99 / 290 MB/s | — |

**Result, the rounds.** 50.6 and 52.9 s against Phase 4's 41.5 s with two CPU workers: with the scheduler believing the GPU to be a 2.1× CPU, the reactive policy kept the 204k build local, preempted it, reneged it, and the restarted build was the straggler. The worker's own estimates were 2–2.6× its elapsed.

**What it says.** (1) **The per-build speedup is not a number but a curve**: the card has a 1.15 s floor and then grows linearly with the vectors, so it loses to the shard below ~20k vectors and wins 9.5× at 400k, with no sign of flattening — the roadmap's "10× per build" is the large-build end of a curve that starts below 1×. (2) **A model with the wrong shape loses the round even with the right hardware**: the first-ever GPU rounds were slower than the CPU-worker ones because of the estimate, not the card. Decision 13's cost model v2 gives the worker its own shape; the second session validates it. (3) The gate passes on GPU-built graphs with recall above hnswlib's own at every size, so the conversion is sound.

### Evidence 11 — cost model v2 on the L4, and the comparison rerun on it (Phase 5)

**Setup.** Cost model v2 (Phase 5 decision 14): the worker's own shape, fitted on the session's L4 points against the shard VM's (`fit --v1 --v2 --fixed --drop-warmup`): floor 1.15 s + 1.05e-8 · n^0.91 · ln(n) · (dim + 154), sigma 0.097, the card's bandwidths. The session's two rounds replayed on it; then the roadmap's eight scenarios × every policy × 20 seeds with sigma 0.10, with the presets' pool (two workers at 6 shards, three at 50) and with one worker as in the session (`experiments/results/calibrated-l4/`).

**Result, the replay.** The two live rounds, which ran under v1's CPU-worker block and took 50.6 and 52.9 s, come out at **31.6 and 32.8 s** under the L4 block, with the 408k and 204k builds both on the card and done by 30 s: the second session's prediction.

**Result, the comparison** (median round seconds; the CPU-worker block's numbers from evidence 9 for comparison):

| Scenario | CPU-worker block (v1), any policy | L4 block, presets' pool, reactive | L4 block, one worker: always_gpu / static / reactive / clairvoyant |
| --- | --- | --- | --- |
| uniform, skewed, shrinking_pool | 30 | 30 | 37–38 / 30 / 30–39 / 30 |
| downstream_heavy | 30–31 | 30 | 38 / 31 / 40 / 30 |
| staggered | 40 | 40 | 43 / 40 / 40 / 40 |
| bimodal | 78 | 76 | 76 / 76 / 76 / 76 |
| high_dim | 193 | 184 | 184 |
| **hour_build** | **2708** | **284** | 308 / 284 / 284 / 284 |
| production | 24139 | 23678 | 23678 |

**What it says.** (1) **The first round-level GPU number: 9.5× on the hour-long build** (2708 → 284 s), because the straggler that holds the round is a build the card finishes in minutes; `production` does not move because its largest build fits no 24 GiB worker and stays local, as evidence 2 predicted. (2) **The small presets stay at their query horizon** (30 s): the round is the drain of the queries, not the builds, and no card changes that — evidence 1's finding survives the hardware. (3) **With one card the policies finally differ**: always-offload queues everything behind one worker (37–43 s), the size threshold and the clairvoyant stay at 30, and reactive loses on `uniform` and `downstream_heavy` (39–40 s) by preempting into a single-card queue — the first preset where a dynamic policy is beaten by a static one, and the sweep that the second session should reproduce live. (4) The model's error is now 0.10 on both sides; the two numbers above that used to be guesses, the per-build speedup and the round-level gain, are measured and derived respectively.

### Evidence 12 — the second GPU session: v2 validated live (Phase 5)

**Setup.** The first session's topology again, the scheduler on the L4 block; eight live rounds (skewed and uniform, two seeds, reactive and the static threshold) against one L4; `test-logs/phase5/gpu-rounds-v2.log`.

**Result.**

| Rounds | Live | Simulator | Agreement |
| --- | --- | --- | --- |
| skewed, reactive | 37.5 / 37.0 s | 31.6 / 32.8 s | placement identical; the offloaded shards within 0.01–0.6 s; one local build 4–6 s slow (decision 15) |
| uniform, reactive | 30.0 / 30.0 s | 30.0 / 30.0 s | within 0.01 s |
| skewed, static | 38.2 / 36.2 s | 31.6 / 32.8 s | as reactive |
| uniform, static | 30.0 / 30.0 s | 30.0 / 30.0 s | within 0.01 s |

**What it says.** (1) **The model's shape was the whole difference**: the same card, the same shards, the same policy, 51 s under the CPU-worker block and 37 s under the L4 block, with the placement the simulator predicted. (2) **The offloaded side is now exact to the second**; what remains is the rig's shared VM on the local side (decision 15), not the model. (3) The query-bound presets are at their horizon on hardware and simulator alike, as every entry since evidence 1 has said.

*(Next expected entries: the two-card measurements, workers per node and builds per card, once the quota allows a second card; a spot preemption observed once a spot L4 is available.)*
