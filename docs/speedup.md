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
| Per-round speedup, current presets | **~0 at the median.** Every policy — including all-local — produces the same round time, because the presets are bound by physics placement can't touch (see evidence 1) | Simulated |
| Per-round speedup once the largest build fits a worker | **~7.0×** (`production` preset with 160 GB workers: ~9.7 h all-local → ~1.4 h reactive), and the dynamic policy beats naive always-offload by a further **~20%** (see evidence 2) | Simulated |
| Per-shard recovery, `production` preset | The five offloadable shards recover **~2.1 h sooner** per round; the round clock doesn't move because the sixth shard's build fits no GPU | Simulated |
| Per-build speedup, measured | **Unknown.** hnswlib (CPU) vs cuVS (GPU) on real hardware is Phase 5 | — |
| Does the simulator's number survive contact with the live stack? | **Yes, within ~2 s at 50 shards and within 0.3 s over a 7-minute round**: 15 of 15 live rounds on the kind cluster agree with the simulator's replay at the same seed and cadences (evidence 4). The simulator cannot know the live reconsider ticker's phase; at 1536 dims that moves a round by up to ~15 s, and the comparer reports the envelope | Measured (fake builds: the live stack sleeps for the same modeled times the simulator uses, so this validates the *mechanics*, not the cost model) |

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

**Setup.** `scripts/k8s-check-step7.sh`: the kind cluster at `FAKE_TIME_SCALE=1` (so the live 0.5 s report, 2 s reconsider tick and 10 s lease are the modeled seconds the simulator defaults to), images rebuilt from the checkout, 24 GiB modeled worker memory on both sides (decision 26), every policy switch verified from the scheduler pod's startup line (bug 15). Seven configurations, fifteen rounds; each live round's timeline replayed by `experiments/cmd/validate` from the same scenario and seed with link sharing off (decision 22). Log: `test-logs/phase3/step7-validation.log`; the table is in the Phase 3 design doc, step 7.

**Result.** 15/15 agree within max(2 s, 5 %), one of them only inside the tick-phase envelope (decision 29):

| Regime | Live − simulated | Example |
| --- | --- | --- |
| Horizon-bound (skewed 6/2, both policies at 50/3 under reactive) | +0.01 s | 30.01 vs 30.00 |
| Queue-bound (skewed 50/3 always_gpu) | +0.2 .. +2.1 s, live slower | 51.96 vs 49.84 |
| Transfer-bound at 1536 dims (high_dim 6/2 reactive) | −1.9 .. −13.6 s; envelope over tick phases 90.6..105.6 s contains every live round | 91.58 vs 105.13 (default phase), 90.63..105.63 (envelope) |
| One modeled hour on the CPU, 7 min on the GPU (hour_build, both policies) | +0.2 s over 438 s | 438.42 vs 438.20 |

**What it does and does not validate.** The live workers are fake: they sleep for the cost model's times, so this is a validation of the *mechanics* the simulator replays — report latency, claim order, preemption and reneging at the tick, lease renewal over minutes, the transfer term's arithmetic — not of the 10×/2 s/200 MB/s cost model, which both sides share and Phases 4–5 replace. The systematic +1.4 s at 50 shards is control-plane latency (50 shards' reports and claims through one HTTP server and Postgres) the simulator does not model; it is 3 % of a 50 s round and would be 0.04 % of an hour-long one. The submit order of 50 concurrent shards and the ticker's phase are the two live inputs the simulator cannot know; the first only permutes which shard finishes k-th, the second can reorder a preempted queue and is reported as an envelope. No simulator or cost-model constant was changed to obtain these numbers.

**Consequence for the ledger.** Evidence 2's ~7× and ~20 % are now numbers from a validated instrument, still built on an assumed cost model. The next change to this column comes from Phase 4's measured CPU constants and Phase 5's measured GPU speedup, not from more simulation.

*(Next expected entries: the FULL policy comparison on the validated simulator; Phase 4 — the measured CPU build curve and MinIO throughput replacing A and the bandwidth; Phase 5 — the measured hnswlib-vs-cuVS number replacing the assumed 10×, graded against evidence 3's prediction.)*
