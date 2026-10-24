# Observed speedup from offloading builds to GPUs

The project's headline question — **how much does offloading an index build to a GPU actually speed things up?** — answered honestly at every point in time. The honest answer has three levels, because "speedup" means three different things:

1. **Per-build**: how much faster one build finishes on a GPU than on its shard's CPU.
2. **Per-round**: how much sooner the whole cluster recovers from the DDL (round completion time — one slow shard holds everyone up).
3. **Per-shard recovery**: how much sooner the *other* shards drain their blocked queries, even when the round clock doesn't move.

Each number is also labeled by how we know it: **assumed** (a cost-model parameter), **simulated** (the discrete-event simulator running on that assumption), or **measured** (a real build on real hardware). Nothing is measured yet; that is Phase 5's job.

This doc is updated whenever a sweep, a live validation or a real measurement produces a new number: the current-answer table below is rewritten in place, and a dated entry is appended to the evidence log (ascending by date — append at the end, never insert).

## Current answer (as of 2026-10-23)

| Question | Current answer | How we know |
| --- | --- | --- |
| Per-build speedup | **~5–9× effective**, from an *assumed* 10× compute speedup minus a 2 s GPU overhead and a 200 MB/s transfer; small builds lose most of it to overhead | Assumed (cost model); simulator and fake worker both run on it |
| Per-round speedup, current presets | **~0 at the median.** Every policy — including all-local — produces the same round time, because the presets are bound by physics placement can't touch (see first entry 2026-10-23) | Simulated |
| Per-round speedup once the largest build fits a worker | **~7.0×** (`production` preset with 160 GB workers: ~9.7 h all-local → ~1.4 h reactive), and the dynamic policy beats naive always-offload by a further **~20%** (see second entry 2026-10-23) | Simulated |
| Per-shard recovery, `production` preset | The five offloadable shards recover **~2.1 h sooner** per round; the round clock doesn't move because the sixth shard's build fits no GPU | Simulated |
| Per-build speedup, measured | **Unknown.** hnswlib (CPU) vs cuVS (GPU) on real hardware is Phase 5 | — |
| Does the simulator's number survive contact with the live stack? | **Untested.** Phase 3 step 7 compares simulator predictions to the live kind cluster | — |

## The assumption underneath everything

All modeled times come from one place: `scheduler/costmodel/costmodel.go`, `Default()`. A local build takes `A · n · ln(n) · dim` seconds (`A = 1.4e-7`); a GPU build takes the local time **÷ 10** (`Speedup`), **+ 2 s** fixed overhead (`GPUOverhead`), **+ the vectors over a 200 MB/s node link** (`Bandwidth`, shared by workers on the same node). So the effective per-build speedup grows with build size:

| Build | Local (modeled) | GPU (modeled) | Effective speedup |
| --- | --- | --- | --- |
| 100k vectors × dim 128 | 20.6 s | 4.3 s | **~4.8×** — the 2 s overhead eats small builds |
| 1M × 128 | 248 s | 29 s | **~8.4×** |
| 10M × 1536 (the `production` preset's largest) | ~9.6 h | ~63 min | **~9.2×** — but it never happens: the build needs ~125 GB of GPU memory (index bytes × `MemFactor` 2) against the preset's 80 GB workers, so it builds locally under *every* policy |

The 10× is a placeholder chosen in Phase 1, in the plausible range for graph-index builds but not from our hardware or our libraries. Both the live fake worker and the simulator obey it, which is what makes step 7's simulator-vs-live comparison meaningful — and what makes the whole column worthless until Phase 5 replaces the placeholder with a measurement.

## Evidence log

### 2026-10-23 — first all-local baseline: round time is physics-bound at the preset loads

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

### 2026-10-23 — second entry: ~7× round-level speedup once the largest build fits a worker

The first entry showed `production`'s round pinned by the 10M × 1536 build that no 80 GB worker can hold. Lifting just that constraint (160 GB workers; the build needs ~125 GB) makes the round-level value of offloading — and of *dynamic* placement specifically — visible:

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

Two findings. First, the headline: with every build offloadable, **offloading cuts simulated cluster recovery ~7×** on the production-scale workload (6 shards of 1M–10M × 1536 on 2 GPU workers). Second, the scheduling result: `reactive` beats `always_gpu` by **~20%** and cuts its straggler lag from 2,200–4,500 s to a tight ~690 s, because it keeps small builds local instead of queueing everything behind two workers — the first preset configuration where round time discriminates between policies at all, confirming the first entry's consequence. Both numbers inherit the assumed 10×/2 s/200 MB/s cost model and await step 7 (live validation) and Phase 5 (measured build times).

*(Next expected entries: step 7 — does the live kind cluster reproduce the simulator's times?; Phase 5 — the measured hnswlib-vs-cuVS number replacing the assumed 10×.)*
