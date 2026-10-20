# Phase 3: placement policies and experiments

## Status and scope

**In progress.** Started 2026-10-19. This is the core of the project: five policies behind one interface, compared on identical seeded workloads across the scenarios. Builds are still fake.

The centerpiece is **reactive migration**: a deployable policy that knows only what a real shard reports, and moves a local build to the GPU when the shard's growing query backlog says it should. The **clairvoyant** cost-based policy, which knows every shard's future queries, is kept as an oracle to measure what the unknowable future costs. (Model revision of 2026-10-02; see Phase 1 decisions 28 to 32.)

**Done when** the policy comparison exists for every scenario plus the sweeps in the roadmap, with the simulator validated against a few live runs.

## System diagram

The policy package is pure and is linked into two hosts: the live scheduler and a discrete-event simulator. Both feed it the same inputs; only the clock differs.

```mermaid
flowchart LR
    subgraph policy["policy package (pure functions)"]
        D["decide(build, shard_status, pool_state)<br/>→ placement, priority"]
        RC["reconsider(pool_state, all_builds, all_shard_status)<br/>→ preempt / renege / speculate / reprioritise"]
        EST["gpu_finish_estimate<br/>(queue replay)"]
        D --> EST
        RC --> EST
    end

    subgraph live["live scheduler"]
        API["HTTP API"] --> D
        T["timer"] --> RC
        D & RC --> PG[("Postgres")]
    end

    subgraph sim["discrete-event simulator"]
        EV["event loop<br/>(modeled time)"] --> D
        EV --> RC
        D & RC --> ST["in-memory state"]
    end

    RUN["experiment runner<br/>seeded scenarios → CSV"] --> sim
    RUN -. "a few configs" .-> live
```

## Flows

### Queue replay: the GPU finish estimate

List scheduling over the current queue. Possible only because the scheduler sees every queued job and every live worker.

```mermaid
flowchart TB
    A["workers with remaining time<br/>w1: 4 s, w2: 0 s"] --> B["queued jobs ahead, by priority<br/>j1: 6 s, j2: 3 s"]
    B --> C["assign each to earliest-free worker<br/>j1 → w2 (free at 0, done 6)<br/>j2 → w1 (free at 4, done 7)"]
    C --> D["this build → earliest free = w2 at 6<br/>+ GPU time + transfer = estimate"]
```

### Reactive migration: the reconsider step for one locally building shard

Inputs are the shard's latest report and the pool. Nothing about the future.

```mermaid
flowchart LR
    IN["report: queue_depth, waiting_needs_index,<br/>oldest_wait_ms, build_progress<br/>pool: workers, in-flight, queue"] --> L["stay local =<br/>remaining build (from progress)<br/>+ drain whole backlog"]
    IN --> G["migrate =<br/>max( drain non-index backlog from now,<br/>GPU queue wait + build + transfer<br/>+ drain index backlog )"]
    L & G --> CMP{"migrate recovers sooner<br/>by more than a margin?"}
    CMP -- no --> OUT1["leave it"]
    CMP -- yes --> OUT2["preempt: running→queued, placement gpu<br/>priority = blocked work behind it"]
```

The backlog is estimated from the report alone: `queue_depth` queries with the scenario's mean duration, split by `waiting_needs_index`. The arrival rate is estimated from successive reports. The margin guards against flapping, and `build_progress` carries the sunk cost: a build at ninety percent has little remaining time to save.

### Clairvoyant cost-based decision (oracle, simulator only)

```mermaid
flowchart LR
    IN["shard: n_vectors, dim, mem_bytes,<br/>FULL future query stream"] --> L["local finish =<br/>CPU build, then queries as they arrive"]
    IN --> G["GPU finish =<br/>replay the stream: non-index queries run on arrival,<br/>index queries wait for the GPU estimate"]
    L & G --> CMP{"earlier?"}
    CMP -- local --> OUT1["placement local"]
    CMP -- gpu --> OUT2["placement gpu<br/>priority = local finish time<br/>(largest first)"]
```

Priority equals the shard's would-be local finish time, so the worst would-be straggler gets a GPU first (longest-processing-time-first). This is the design the project started with; it is kept because it bounds what any deployable policy can achieve.

### Speculation

```mermaid
sequenceDiagram
    participant RC as reconsider (timer)
    participant P as Postgres
    participant W as idle worker
    participant S as shard k (building locally)

    RC->>P: queue empty and a worker idle?
    RC->>P: pick the local build with the latest projected finish
    RC->>P: INSERT speculative GPU copy (build_id k:r:spec, state queued)
    W->>P: claim the copy
    alt GPU copy completes first
        W->>P: complete (guarded)
        RC->>P: mark local copy cancelled, shard stops its build
    else local build completes first
        S->>P: done
        RC->>P: mark GPU copy cancelled, worker stops at its next failed renew
    end
    RC->>P: record wasted work = time spent on the loser
```

## Design decisions

| # | Decision | Alternatives | Why |
| --- | --- | --- | --- |
| 1 | Policies are pure functions of passed-in state; no I/O, no clock, seeded randomness only | Policies query Postgres directly | The simulator can then reuse them byte for byte, and the sweep comparison measures the policy, not two implementations of it. |
| 1a | Deployable policies see only `shard_status` reports; the future query stream is passed only to the clairvoyant oracle, and only in the simulator | One cost-based policy that knows the jobs (the original design) | A real shard cannot declare its future. Separating "what a scheduler can know" from "what the simulator knows" makes the gap measurable. |
| 1b | Preemption of a local build is a first-class `reconsider` action, with the sunk cost read from the shard's reported `build_progress` | Speculation only (never kill a local build) | Freeing the CPU is the only way to help queries that don't need the index while the build is still running. Speculation stays as the alternative for backlogs dominated by `needs_index` queries. |
| 2 | Priority is a number computed by the policy and stored on the row; the claim query orders by it | Separate queues per priority class | One queue, one index, one claim statement. The policy owns the meaning of the number. |
| 3 | `reconsider` runs on a timer in the scheduler and expresses every action as a row change | A long-lived planner that holds the queue in memory | Keeps invariant 1 (all state in Postgres) and makes reconsider restart-safe. |
| 4 | Reneging moves a queued GPU job back to local by a guarded `UPDATE … WHERE state = 'queued'` | Cancel and resubmit | A leased job is never reneged; only queued ones. The guard makes it race-free against a concurrent claim. |
| 5 | A speculative copy is a second `builds` row with its own `build_id` suffix | A flag on the original row | Two rows, two leases, two attempt counters. The original invariants hold unchanged. |
| 6 | Simulator for sweeps, live cluster for validation only | Sweeps on the live cluster | Fake jobs sleep in real time; a 200-round sweep would take days. Disagreement between the two is itself a result to write up. |
| 7 | Estimate error injected as log-normal noise on every estimate the policy sees | Only on build times | The policy never sees truth; the sweep over σ measures its fragility end to end. |
| 8 | **The flapping guard is a margin plus moved-at-most-once** (answers the flapping open question): migrating or reneging must win by `MarginS` (default 2 s, the order of the GPU overhead), and a build whose placement already changed once after submit (`Moved`) is never moved again — a preempted build cannot renege back, a reneged build cannot be preempted. Speculation is additionally one per reconsider tick and never while a copy is in flight | Margin only (a build could still ping-pong when estimates drift past the margin twice); a cooldown timer (needs a clock, breaking purity) | The pair is clock-free and bounds the worst case absolutely: a build loses progress at most once. |
| 9 | **The host computes every number a policy sees and passes it in**: cost-model estimates per build (`Estimates`), the GPU queue wait (`PoolState.QueueWaitS`, `BuildView.QueueWaitS`), the arrival rate and mean query duration on the report (`ShardStatus`). All times are float64 seconds relative to the call, so policies never read a clock | Policies call the cost model and estimator themselves (pure, so allowed) | One choke point implements decision 7: the host noises everything at once. The step 2 queue-replay estimator plugs in by filling `QueueWaitS` without touching any policy, and the simulator feeds modeled time through the same fields. |
| 10 | **One priority currency: seconds of blocked shard work.** At submit the reactive policy sets priority to the build's local duration and the clairvoyant to the would-be local *recovery* (the replayed finish) — longest-processing-time-first either way — and after a preemption it is **the backlog behind the build** (the roadmap's words), recomputed identically by `reprioritize` strictly past a 1 s churn delta, so an unchanged backlog means an unchanged priority. `static_threshold` uses the vector count, which orders identically within that policy. *(Corrected in the step 1 triage: first written as "backlog plus the abandoned local time", while the code computed backlog plus* remaining *time — unstable one tick after the preemption, when progress reads 0. Cross-check F3/F4.)* | Priority classes; FIFO within placement; backlog plus abandoned or remaining local time (not recomputable from a queued build's view, and the remaining variant jumps after the preemption) | The claim orders by one number (decision 2), so every priority written by one policy must be comparable, and a priority must be recomputable from the state a queued build actually carries. Seconds of blocked work is the thing the round loses by serving this build later. |
| 11 | **`workers` gets a `node` column** (owner approved, 2026-10-20), reported in the worker's heartbeat: in Kubernetes `spec.nodeName` via the downward API (the same mechanism as `WORKER_ID`), on Compose the one shared `laptop` node (which is the truth: every Compose worker shares the laptop's link), and `""` = unknown. The estimator gives an unknown-node worker its own private link, because "unknown" must not invent contention between unrelated machines. | Pass the grouping in scheduler config; infer it from worker-ID prefixes | Only the worker knows which machine it landed on, and the per-node transfer term (the worker-count consequences, 2026-10-17) needs the grouping at estimate time. |
| 12 | **The estimator replays the claim, not textbook list scheduling** (`ReplayQueue`): whenever a worker is free it takes the highest-priority queued job **that fits its memory**, ties by enqueued order — the claim statement's own ORDER BY and filter — so a small idle worker never waits behind a big job it could never take. Transfers share the node's link as a fluid: k concurrent transfers each proceed at 1/k (two 10 s transfers started together both finish at 20 s), the cost model's whole transfer term taken upfront at the claim. A job that fits no live worker estimates **+Inf** — the truth bug 11 hid. Pinned approximations: a busy worker is opaque (`BusyRemainingS` is one lump, so an in-flight build's own transfer does not contend), and the replay freezes the world (no future submits, preemptions or worker deaths). | The roadmap's literal "earliest-free worker" (wrong under heterogeneous memory); full-speed transfers (ignores the sharing decision 11 exists for); serializing transfers per node (too pessimistic) | The estimate should predict what the real claim will do, and the per-node sharing is what step 2 was gated on. |
| 13 | **Submit-time `PoolState.QueueWaitS` joins at the back** (`NewJobWait`): the new build is appended at priority 0 behind every queued job, so the scalar is exact for the FCFS baseline and an upper bound under a policy that would rank it higher. An empty pool keeps 0 (the cold-start rule, G1 — applied by the host, since "no workers yet" is the host's knowledge, not the replay's); a build fitting no live worker gets +Inf. The submit response exposes it as `gpu_queue_wait_ms` (-1 for +Inf; absent on a duplicate, whose answer is the original decision, not a fresh estimate). | The policy's would-be priority (circular: that priority is what Decide is about to compute); the front of the queue (underestimates whenever preempted builds carry real priorities) | One scalar must exist before the decision it feeds, and the bound errs toward "the queue looks slow", which leans a borderline decision local — the direction that never wastes a kill. The simulator calls `ReplayQueue` directly and is not bound by this. |

## Implementation notes

### Step 1: the policy interface and the five policies (done 2026-10-19)

| Path | What |
| --- | --- |
| `scheduler/policy/policy.go` | The interface (`Decide`, `Reconsider`), the input types (`Input`, `ReconsiderInput`, `BuildView`, `WorkerView`, `ShardStatus`, `Estimates`), the four action kinds, the shared memory filter, `AlwaysGPU`, and `ByName` (`always_gpu`, `static_threshold[:N]`, `reactive`, `reactive_spec`; the clairvoyant is deliberately absent, decision 1a). |
| `scheduler/policy/threshold.go` | `StaticThreshold`: N or more vectors to the GPU, bigger first; never reconsiders. Default N 200 000. |
| `scheduler/policy/reactive.go` | `Reactive` (stay-vs-migrate from the report alone, preempt past the margin, renege a stuck queued build, reprioritize to the blocked work) and `ReactiveSpec` (plus one speculation per tick onto an idle worker when the queue is empty). |
| `scheduler/policy/clairvoyant.go` | The oracle: constructed with the futures, replays both placements exactly (including the skip rule for not-ready index queries), priority = would-be local finish. |
| `scheduler/api/api.go` | The live host now fills `Input.Est` from the cost model; `QueueWaitS` stays 0 until step 2. `Reconsider` is not yet wired to a timer (step 3). |

`WorkerView` already carries a `Node` field for step 2's per-node transfer term, but the pure types need no schema; whether `workers` gets a `node` column is the owner question flagged below, due before step 2.

### Cross-check of step 1, the policies (2026-10-19)

A fresh agent wrote 40 pure subtests (`TestPolicySpec` in `scheduler/crosscheck/policy_test.go`, saved as `test-logs/phase3/step1-crosscheck-policies.log`) from the docs and `go doc` alone, never reading the policy source. 34 passed first time — purity under repetition and permutation, the memory filter at submit for all five, the margin and sunk-cost boundaries by hand, both directions of moved-at-most-once, speculation's no-go cases, three hand-replayed oracle streams including one that flips without the skip rule. 6 failed, and two passing probes carried findings. Triage:

| Finding | Kind | Resolution |
| --- | --- | --- |
| **F1: preemption ignored the memory filter** — a local build fitting no live worker (or facing an empty pool) was preempted into a queue nobody can claim, hanging the round. | Implementation bug | **Bug 11** (`docs/bugs.md`), fixed in its own commit: reconsider never preempts a build that fits no live worker; an empty pool means no preemption, stricter than submit's cold-start rule because a preemption destroys progress. |
| **F3/F4: three versions of the preempt priority** — the roadmap said "the blocked work behind the build", decision 10 said backlog plus abandoned time, the code computed backlog plus *remaining* time; the code's version also jumped one tick after every preemption (a queued shard reports progress 0), so each preempt bred a spurious reprioritize. | Implementation and doc both wrong | Decision 10 corrected in place to the roadmap's rule: priority = the backlog behind the build, recomputed identically everywhere; stable across ticks when the backlog is unchanged. The agent's two tests pinning the old wording updated with it. |
| **F2: the reprioritize delta fired at exactly ±1 s** where the doc says "more than". | Implementation bug | Comparison made strict. |
| **Renege and preempt used different cost models** (probe): renege compared bare build times, so a queued build with 10 runnable non-index queries waiting was reneged into a *worse* recovery (20 s vs 17 s) — which the policy's own preempt rule would immediately want to undo, decision 8 alone stopping the ping-pong. | Spec gap leaning bug | Renege now uses the same recovery arithmetic as preemption with the roles reversed (going local = stay-local with progress 0, which is what a queued shard reports); without a report it degrades to the old build-time comparison. |
| **The original of an in-flight speculative copy was preempted** (probe): the shard would get two GPU rows and lose the CPU progress the copy exists to protect. | Spec gap leaning bug | A build whose speculative copy is in flight is never preempted; pinned in the implementer's tests. |
| **F5: ByName parsed leniently** — `reactive:5`, `always_gpu:` and `static_threshold:` ran silently with defaults, so a typo'd sweep arm would run as the wrong configuration. | Implementation bug | Strict: any argument on a parameterless policy, or an empty threshold, is an error. `""` = always_gpu is now in the documented grammar (the deployed default since Phase 1). |
| **F6: `Name()` lost the threshold** — logs, metrics and CSV rows could not tell the sweep's arms apart, and a logged name did not reproduce the run. | Spec gap | `StaticThreshold.Name()` is now `static_threshold:N`; `ByName(p.Name())` round-trips for every deployable policy. |
| G1–G7: the oracle on an empty pool queues (the same cold-start rule); a local build's migrate branch reads `BuildView.QueueWaitS` ("the wait it would join at"); a shard missing from the futures is an empty stream and a recovery tie goes local; reactive's default threshold is `DefaultThresholdN`; arrivals enter stay-local as growth during the remaining build; speculation judges "queue empty" on the tick's input, so it may coexist with a same-tick preemption of another shard (the host applies preempts first); action order is unspecified, only the set is deterministic. | Spec gaps (documentation) | All pinned in the package's doc comments and here. |

### Step 2: the GPU finish estimator, queue replay (done 2026-10-20)

| Path | What |
| --- | --- |
| `db/migrations/0005_workers_node.sql` | `workers.node` (decision 11), default `''` = unknown. |
| `scheduler/policy/estimate.go` | Pure, like the whole package: `ReplayQueue` (the claim mirror with fluid per-node link sharing, decision 12; returns each queued build's start and finish, +Inf for one that fits no worker) and `NewJobWait` (the submit-time scalar, decision 13). |
| `scheduler/store/store.go` | `Heartbeat` carries the node; `GetQueueSnapshot` returns the estimator's raw inputs: live workers, queued GPU builds in claim order, leased builds with seconds since their claim. |
| `scheduler/api/api.go` | `submitBuild` fills `PoolState.QueueWaitS` (snapshot → cost model → `NewJobWait`; a leased build's remaining time is its GPU total minus the time since its claim, floored at 0), logs it, and answers `gpu_queue_wait_ms` (-1 when the build fits no live worker). |
| `worker/kgpu_worker/config.py`, `store.py`, `worker.py` | `WORKER_NODE` (default empty), written by every heartbeat. |
| `deploy/k8s/worker.yaml`, `deploy/compose.yaml` | The downward API's `spec.nodeName`; Compose's one shared `laptop` node. |

`BuildView.QueueWaitS` on reconsider is step 3's wiring: the host will call `ReplayQueue` once per tick and read each queued build's start (and each local build's would-be wait) from it.

## What the published build-time figures change here (2026-10-06)

The Phase 1 doc's "Cost model v0 against published numbers" section collects the data: CPU HNSW builds of 8 minutes for 1M x 128 on 8 vCPUs and 2.6 hours for 10M; GPU builds 10 to 50x faster for the build step; 61 GB of vectors to move for 10M x 1536; 24 GB GPUs that hold 5M x 1536 but not 10M.

- **Transfer is a first-class term in the GPU finish estimate.** At 128 dimensions it was noise; at 1536 it rivals the GPU build itself. The queue-replay estimate must add transfer per build with the measured bandwidth, and the reactive policy's "migrate" branch pays it on top of the queue wait.
- **Two scenarios are added:** `high_dim` (1536 dimensions, so transfer and memory bind) and `production_size` (1 to 10M vectors, 768 to 1536 dimensions; simulator only, since it does not sleep). The sweep table gets a dimension axis (128, 768, 1536).
- **The memory filter is central, not a corner case.** With production sizes some builds fit no worker and the bimodal scenario is the normal one; policies must handle "fits nowhere" as a routine outcome, and the GPU memory estimate's error is a sweep variable alongside time-estimate error.
- **Sunk cost dominates preemption.** A local build is hours; abandoning it at 60% throws away an hour of CPU. The reactive policy's `build_progress` term and the flapping margin matter more than the fake-scale runs suggested; the simulator should report wasted CPU-hours, not just wasted GPU time.
- **The speedup sweep** should cover 10x (end-to-end service figures) to 40x (build-step benchmarks), since which regime the GPU is in changes the threshold policy's optimum.

## What the worker-count question changes here (2026-10-17)

Phase 2 decision 23: how many workers run per card and per machine is a parameter to measure, not an assumption. Two consequences for the policies and the simulator:

- **The transfer term is per node, not per build.** Workers on one machine share its network link and host, so when two builds stream their vectors in at once each gets a share of the bandwidth. The cost model's transfer estimate becomes `bytes / (bandwidth_node / concurrent transfers on that node)`, and the queue-replay estimator (decision 2 of the flows above) needs to know which worker sits on which machine. `workers` therefore needs a `node` column or the pool state a node grouping; the schema question goes to the owner before Phase 3 starts.
- **Builds per card is a third axis.** If Phase 5 confirms that a card can overlap one build's transfer with another's compute (see the Phase 5 doc), the simulator's worker model needs the phases (transfer in, build, transfer out) and a per-card concurrency limit with memory-aware admission, so the GPU finish estimate can account for a second build sharing the card. Until then the simulator keeps one build per worker.
- **The pool is machines × cards, and the sweep varies both.** The simulator's pool model gains a topology, and the roadmap's sweep table gains "workers per node at a fixed card count". With the per-node bandwidth term, the simulator can predict the knee where a second card on one box stops adding throughput (expected to depend on dimension: invisible at 128, binding at 1536), and Phase 5 confirms it on hardware.

## Open questions

- Should `reconsider` be able to pre-empt a leased job (cancel it mid-build to hand the GPU to a worse straggler)? Not in Phase 3. It adds wasted work and the speculation policy already covers the idle-GPU case.
- Where does the shard's local build estimate come from? In Phase 3 it is the same cost model the scheduler uses, corrected by reported `build_progress`. In Phase 4 the shard measures.
- How is the arrival rate estimated from reports? First version: difference in cumulative arrivals between reports over the report interval, smoothed. The report may need a cumulative `arrived` count for this.
- Flapping: decided as decision 8 (margin plus moved-at-most-once).
- The `workers` table and the per-node transfer term: decided as decision 11 (owner approved, 2026-10-20) — `workers.node`, reported by the worker's heartbeat.
