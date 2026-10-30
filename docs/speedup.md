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
| Per-round speedup, current presets | **~0 at the median.** Every policy — including all-local — produces the same round time, because the presets are bound by physics placement can't touch (see evidence 1); the full comparison (evidence 5) confirms it at 50 seeds across all nine scenarios | Simulated |
| Per-round speedup once the largest build fits a worker | **~7.0×** (`production` preset with 160 GB workers: ~9.7 h all-local → ~1.4 h reactive), and the dynamic policy beats naive always-offload by a further **~20%** (see evidence 2) | Simulated |
| Per-shard recovery, `production` preset | The five offloadable shards recover **~2.1 h sooner** per round; the round clock doesn't move because the sixth shard's build fits no GPU | Simulated |
| Per-build speedup, measured | **Unknown for a GPU** (Phase 5). Measured for a CPU worker with 8 threads over a shard with 2, on the laptop: **2.0×** (evidence 6, a rehearsal) | Measured (laptop; decision 7 says the number that counts comes from the cloud topology) |
| The cost model's CPU constant and its error, measured | **A ≈ 6.1e-8 s per n·ln(n)·dim at 2 threads on this laptop (log-space fit), 2.3× below Phase 1's 1.4e-7; the n·ln(n) shape is off by up to ±50 % across 10k–400k (time grows like n^1.3), so the estimate error for the sweeps' sigma axis is ~0.25 RMS** (evidence 6) | Measured (laptop rehearsal) |
| Does the simulator's number survive contact with the live stack? | **Yes, within ~2 s at 50 shards and within 1 s over 7- and 9-minute rounds**: 18 of 18 live rounds on the kind cluster agree with the simulator's replay at the same seed and cadences (evidence 4). The simulator cannot know the live reconsider ticker's phase; at 1536 dims that moves a round by up to ~15 s, and the comparer reports the envelope | Measured (fake builds: the live stack sleeps for the same modeled times the simulator uses, so this validates the *mechanics*, not the cost model) |

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

*(Next expected entries: Phase 4's closing step — the same calibration on the cloud topology, whose A, speedup, bandwidth and sigma replace the placeholders and rerun the full comparison; Phase 5 — the measured hnswlib-vs-cuVS number replacing the assumed 10×, graded against evidence 3's prediction.)*
