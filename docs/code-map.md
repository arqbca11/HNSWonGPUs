# Code map: where each important action lives

One entry per action the system takes, pointing at the code that decides it, the code that writes it, and the code that reacts to it. Pointers are file + function (not line numbers, which drift); `grep -n "func Name"` or `go doc` finds the spot. The Python worker mirrors the Go store's SQL — when an entry names both, the Go side is the reference the integration tests check and the Python side is what actually runs in the pool.

**Order:** entries follow a build's lifecycle, then the loops around it, then the pure machinery (policies, estimator, cost model), then observability and configuration. **Update this file whenever a step adds, moves or renames one of these pieces** (working agreement in CLAUDE.md).

---

## The build lifecycle

### A round begins
- `POST /rounds` → `scheduler/api/api.go` `createRound` → `scheduler/store/store.go` `CreateRound` (the `n_shards` it expects is how "every shard is done" becomes decidable, migration 0002).

### Submit: a build enters and placement is decided
- `POST /builds` → `scheduler/api/api.go` `submitBuild`: duplicate check first (invariant 6 — a resubmit returns the stored decision via `writeDuplicate`, computed from the stored row, bug 1), then cost-model estimates, the submit-time queue-wait estimate, and the policy call.
- The decision itself: `scheduler/policy/policy.go` `Policy.Decide`, selected by `ByName` (strict grammar: `"" | always_gpu | static_threshold[:N] | reactive | reactive_spec`); the memory filter every policy applies first is `fitsNowhere` in the same file.
- The write: `scheduler/store/store.go` `SubmitBuild` — locks the round row first (bugs 4 and 6: snapshot-before-lock), enforces the `n_shards` cap and the finished-round refusal, insists the id is `BuildID(shard, round)` (cross-check F4: the generated FKs make the convention load-bearing), and inserts gpu→`queued` or local→`running`.
- The reply's waiting estimate: `gpu_queue_wait_ms` (`*int64`: absent on a duplicate, step 2 cross-check F1) from `submitBuild` → `queueWait` → the estimator (below).

### Claim: a worker takes a queued build
- The statement (invariant 2: workers pull, `FOR UPDATE SKIP LOCKED`; invariant 4: `attempt + 1` is the fencing token): Go reference `scheduler/store/store.go` `Claim`; what the pool really runs is `worker/kgpu_worker/store.py` `claim`.
- The worker's loop around it: `worker/kgpu_worker/worker.py` (claim → build behind the one `build(job) -> artifact` interface → complete, heartbeating and renewing as it goes).
- A speculative copy is claimed by this same statement with zero changes (decision 15).

### Renew, complete, fail, release: the leaseholder's writes
- All guarded by `build_id AND attempt AND state='leased'` (invariant 3; zero rows = lost ownership, stop): `scheduler/store/store.go` `Renew`, `Complete`, `Fail`, `Release`; Python `worker/kgpu_worker/store.py` `renew`, `complete`, `fail`, `release`.
- Renew deliberately has no `lease_until > now()` check: a late worker that still owns the row may continue (doc on `Renew`).

### Reap: expired leases return (or die)
- `scheduler/reaper/reaper.go` `Run`/`Tick` (also stamps finished rounds and refreshes the gauges) → `scheduler/store/store.go` `Reap`: requeue, or fail once the retry budget `MAX_ATTEMPTS` is spent (decision 42, bug 5's poison pill). Idempotent (invariant 5); the reaper is the first sanctioned writer without the attempt guard.

### A local build runs and finishes
- The shard's CPU runs it: `shard/sim/shard.go` `cpu` → `runLocalBuild` (progress slices, cancellation checks; the CPU runs no queries while a local build or an undecided DDL holds it).
- Completion: `POST /builds/{id}/done` → `scheduler/api/api.go` `localBuildDone` → `scheduler/store/store.go` `CompleteLocal` (guarded by `placement='local' AND state='running'` — a preempted shard's late done is refused, and that refusal is how the shard learns).

### Queries: the cost that makes placement measurable
- Recording: `scheduler/store/store.go` `RecordArrival`, `StartJob`, `FinishJob` (per event) and `ReportBatch` (the batched path, decision 60 — one statement per batch, timestamps only ever fill in).
- The shard side: `shard/sim/shard.go` `arrivals` (the seeded stream), `runQuery` (the virtual CPU clock, bug 8), `drainDirtyLocked` (what goes into the next report).

### The report loop: how a shard and the scheduler stay in sync
- `POST /shards/{round}/{shard}/report` → `scheduler/api/api.go` `shardReport` → `ReportStatus`/`ReportBatch`; the reply carries the build's placement/state/attempt (`BuildSummary`).
- The shard acts on the reply in `shard/sim/shard.go` `apply`: done → index ready (and, if it was still building locally, the speculation's copy won → abort); failed → stuck queries; gpu while building → preempted, abort; local while idle → renege, start building. The attempt counter is the only sure sign of a lost-and-reclaimed lease.
- Bug 3's rule (no report without a build) is Postgres's: migration 0004's FKs, re-anchored by migration 0006 on `builds.build_id` through stored generated columns.

### Round completion
- `scheduler/store/store.go` `FinishCompleteRounds` (+ `fillRoundStats`): all real builds terminal, all streams done, all queries drained; `finished_at` is the latest finish, not `now()`; speculative copies invisible (decision 17); queue waits need `attempt > 0` (cross-check F2).
- Stamped from two places, observed exactly once: the reaper's tick, and `GET /rounds/{id}` stamp-on-read — unless draining (decision 40, the lame-duck scheduler; the `Draining` flag is set in `scheduler/cmd/scheduler/main.go`).

## Placement changes in flight (Phase 3 step 3)

### The reconsider timer: the actuator
- `scheduler/reconsider/reconsider.go` `Run`/`Tick` (decision 14): settle speculation races first, snapshot, map, decide, execute — preempts before everything (G7). `buildInput` is the snapshot→policy mapping (time-scale conversions, decision 18; per-tick queue replay; cold-start rule every tick, decision 19). `execute` applies one action as its guarded write; zero rows = skipped, next tick recomputes.
- The snapshot: `scheduler/store/store.go` `GetReconsiderSnapshot` (live workers, non-terminal builds in claim order, reports with SQL-derived arrival/mean aggregates).
- Started in `scheduler/cmd/scheduler/main.go` (`RECONSIDER_INTERVAL`, `FAKE_TIME_SCALE`).

### Preempt: kill a local build, queue it on the GPU
- Decided: `scheduler/policy/reactive.go` `preemptIfWorth` (stay-vs-migrate from the report alone; sunk cost via `build_progress`; never on a moved build, never without a report) with the memory filter and the copy-in-flight check in `Reactive.Reconsider` (bug 11).
- Written: `scheduler/store/store.go` `PreemptToGPU` — sets `moved`, refuses a second move and any primary whose copy is not `failed` (bug 12: a done-but-unsettled copy is a race already won).
- The shard learns from the report reply and aborts (`apply` above); its priority currency is `blockedWorkS` (decision 10: seconds of blocked shard work).

### Renege: a queued GPU build goes back local
- Decided: `scheduler/policy/reactive.go` `renegeIfWorth` — the same recovery arithmetic reversed, and first the **impossibility escape** (decision 20, bug 13): a build reading +Inf (fits no live worker) reneges even if already moved.
- Written: `scheduler/store/store.go` `RenegeToLocal` — sets `moved`, deliberately not guarded on it (the escape needs the door open; no loop, because preempt requires never-moved).

### Reprioritize: a preempted build tracks its backlog
- Decided: `scheduler/policy/reactive.go` `reprioritizeIfDrifted` (strictly past the 1 s churn delta); written: `scheduler/store/store.go` `Reprioritize` (queued builds only).

### Speculate: a GPU copy races the local build
- Decided: `scheduler/policy/reactive.go` `ReactiveSpec.speculate` (idle worker + empty queue → the worst projected recovery; one per tick; never while one is in flight).
- Written: `scheduler/store/store.go` `SpeculateCopy` — the second `builds` row, `build_id + ":spec"` (decisions 5 and 15; one copy ever, by primary key). Schema room: `db/migrations/0006_speculation.sql`.

### Cancelling the losing copy
- `scheduler/store/store.go` `ResolveSpeculations` (decision 16, the third lease-ending path after complete and reap): GPU won → the primary becomes `gpu/done` at the copy's `finished_at`; local won → the copy is `failed` and its worker's next renew/complete hit zero rows and it stops, exactly like a lost lease — the Python worker needed no code for this.
- The shard's abort on a GPU win: `shard/sim/shard.go` `apply` (done while `localBusy`).

## The pure machinery

### The policy interface and the five policies
- `scheduler/policy/policy.go`: `Policy` (`Decide`/`Reconsider`), the input/action types, `fitsNowhere`, `AlwaysGPU`, `ByName`.
- `scheduler/policy/threshold.go` `StaticThreshold`; `scheduler/policy/reactive.go` `Reactive` and `ReactiveSpec`; `scheduler/policy/clairvoyant.go` the oracle (simulator only, decision 1a — deliberately absent from `ByName`).
- Invariant 8: all pure — no I/O, no clock; the host computes every number they see (decision 9).

### The GPU finish estimator (queue replay)
- `scheduler/policy/estimate.go`: `ReplayQueue` (the claim mirror — memory-aware, priority DESC then Seq, fluid per-node link sharing, +Inf for fits-nowhere; decision 12) and `NewJobWait` (the submit-time "joins at the back" scalar, decision 13).
- Its two callers: `scheduler/api/api.go` `queueWait` (at submit, via `GetQueueSnapshot`) and `scheduler/reconsider/reconsider.go` `buildInput` (every tick).

### The cost model
- `scheduler/costmodel/costmodel.go` (v0: `CPUBuild`, `GPUBuild`, `Transfer`, `GPUTotal`, `GPUMemBytes`; constants overridable by `COST_*` env). The Python worker re-implements the same formulas in `worker/kgpu_worker/costmodel.py` to know how long to sleep — keep the two in step.

### Workers and the pool
- Heartbeat/registration: `scheduler/store/store.go` `Heartbeat` (carries the node, decision 11; latest wins), `Deregister` (clean shutdown, decision 49), `Pool` (live = seen within `WORKER_STALE_AFTER`); Python `worker/kgpu_worker/store.py` `heartbeat`.
- Where a worker learns its node: Kubernetes downward API `spec.nodeName` in `deploy/k8s/worker.yaml`; Compose's one shared `laptop` in `deploy/compose.yaml`.

### The discrete-event simulator and the experiment pipeline
- One modeled round against one policy: `experiments/sim/sim.go` `Run` — the event loop is `engine.step` (fixed dispatch order for determinism); `submit` mirrors the live API's mapping, `tick` mirrors the reconsider loop's (resolve-then-decide, preempts first), `execute` applies actions with the store's guards, `claims` mirrors the claim statement, `report` is the live shard's publish-and-apply. `Futures` is the one place the oracle gets its streams (decision 1a). Fidelity choices and simplifications: decisions 21–23.
- The sweep runner (one CSV row per round, byte-identical reruns): `experiments/cmd/sweep/main.go` — the schema is `columns`, a cell's scenario/cost-model overrides are `runCell` (decision 24).
- Step 7's comparer (simulator vs live, decision 28): `experiments/cmd/validate/main.go` — reads `shard/sim.Timeline` JSONs, regenerates the workload from scenario and seed (refusing on any size mismatch, shards out of id order, or submit reasons worded by a policy other than `-policy`, decision 30), replays with `sim.Run` at the live cadences × `time_scale`, sharing off; prints per-round and per-shard deltas (by shard and order-free), each side's placement, and the tick-phase envelope (replays at eight `TickPhaseS` values, decision 29). Driven by `scripts/k8s-check-step7.sh` (patches the ConfigMaps' `FAKE_TIME_SCALE`/`POLICY`, restarts, keeps `TIMELINE_DIR`, restores on exit); `scripts/k8s-round.sh` honours an external `TIMELINE_DIR`.
- The straggler override (decision 27): `experiments/workload/generate.go` `genSizes` — after the distribution's draws, `SizeDist.Straggler` replaces the largest; the `hour_build` preset in `presets.go`.
- The plots: `experiments/analysis/` (`plot_rounds.py`, `plot_sigma.py`, shared `plotlib.py` with the Go side's percentile). The committed demo: `scripts/sim-sweep-demo.sh` → `experiments/results/demo/`.

### The shard simulator and workload
- One shard = three goroutines: `shard/sim/shard.go` `Run` (`arrivals`, `cpu`, `reports`); the round driver and timeline are `shard/sim/round.go` and `shard/sim/timeline.go` (`BuildTimeline` skips copies from the per-shard lines, counts them in `Speculations`).
- Seeded workloads: `experiments/workload/` (scenario presets; `PRINT_WORKLOAD=1` to inspect).
- The HTTP client both the simulator and tests use: `shard/client/client.go`.

## The data and the real builds (Phase 4)

### Step 1: datasets, splits, ground truth, the object store
- The Python package: `builder/kgpu_builder/` (its own uv project, decision 12 of the Phase 4 doc). The catalogue is `datasets.py`; the Zipf split and its manifest `split.py` (`zipf_sizes`, `plan`, `write_split`; decision 9); exact per-shard top-k `groundtruth.py` (`exact_topk`, `recall_at`; decision 14); the S3 client `store.py` (`Store`: bytes/JSON/npy, list, exists, delete_prefix; decision 11); the command line `cli.py` (`python -m kgpu_builder download|split|groundtruth|upload|verify|all`).
- Where the data lives: locally under `KGPU_DATA_DIR` (default `data/`, gitignored); in the store under `datasets/<ds>/` and `splits/<ds>/<n>/{manifest.json,vectors/,ground_truth/}`.
- The object store itself: RustFS (decision 13) — `deploy/compose.yaml` service `objectstore`; `deploy/k8s/objectstore.yaml` (Secret `objectstore` with the endpoint, keys and bucket; NodePort 30900; a StatefulSet); the kind port mapping in `deploy/kind/cluster.yaml`; a throwaway one in `scripts/test-db.sh`.
- Putting the data on the cluster: `scripts/phase4-data.sh`.

### Step 2: the real build, on both sides of the placement
- The build itself: `builder/kgpu_builder/build.py` (`run_build`: fetch, `build_index` in ten batches, `gate`, publish index then manifest; `GateFailed`) behind `cli.py`'s `build` command (exit 0/3/1, stdout manifest, stderr progress; decision 17).
- A round's data: `rounds.dataset` (migration 0007, decision 15) — set in `shard/sim/round.go` from `RoundConfig.Dataset`, stored by `store.CreateRoundWithDataset`, read by the worker's `store.claim` into `Job.dataset`/`n_shards`.
- The split sizes on the Go side: `experiments/workload/splits.go` (`ApplySplit`, kind `split`; the JSON copies under `experiments/workload/splits/`; decision 16).
- The worker's real build: `worker/kgpu_worker/builder.py` `HnswlibBuilder.build` — the CLI in a process group, progress from stderr, `cancel` → `killpg`, exit 3 → `GateRefused` → the worker's existing fail path (decision 18). Config: `BUILDER`, `BUILDER_CMD`, `BUILD_THREADS`, `RECALL_MIN`, `S3_*`.
- The shard's real build: `shard/sim/shard.go` `runRealBuild` (exec in a process group; progress into `b.progress`; cancel → `SIGKILL` the group; exit 0 → `LocalDone`, exit 3 → `LocalFail` with the gate's reason; decision 19) and `burn` for queries (decision 20); the scheduler side is `store.FailLocal` behind `POST /builds/{id}/fail`.
- A failed build's second chance (decision 21): `store.RetryLocal` behind `POST /builds/{id}/retry-local` (guarded on `local_retries = 0`, the row back to local/running at the next attempt), asked by the shard's `retryLocal` from `apply` when the summary says failed; after that `nextReady` releases the needs-index queries (the fallback) and the round completes with the failure on record.
- Running it: `scripts/k8s-check-p4-step2.sh` (the cluster at time scale 1 with `BUILDER=hnswlib`, the laptop's shards with `DATASET` and `BUILDER_CMD`).

### Step 3: calibration — measured build times into cost-model constants
- The grid and the point: `builder/kgpu_builder/calibrate.py` (`run_grid` spawns `calibrate-point` per (size, threads, repeat) and appends CSV rows; `run_point` builds a prefix of the largest shard, gates it on an exact top-10, reports peak RSS; decision 23).
- The fit: `calibrate.fit_points` (A per thread count through the origin, sigma of the log residuals, a power-law fit beside it), `residual_table`, `env_overrides` (the `COST_*` block; decision 24); the CLI's `calibrate`, `calibrate-point`, `fit` in `cli.py`.
- The figure: `experiments/analysis/plot_calibration.py` (measured vs fitted on log-log, the residual ratio below).
- Where the numbers go: `experiments/results/calibration/<host>-<dataset>-<n>.csv` (the laptop rehearsal is committed as a rehearsal); the cloud topology's calibration will be the one the scheduler's `COST_*` are set from.

## Observability and configuration

### Metrics
- Scheduler: `scheduler/metrics/metrics.go` — round histograms observed exactly once wherever the round was stamped (`ObserveFinishedRounds`), placement/lease counters, the gauges refreshed each reaper tick, and step 3's `reconsider_actions_total{action,outcome}` and `speculations_resolved_total{winner}` (a local win with a leased copy is the recorded wasted work).
- Simulator: `shard/sim/metrics.go`. ServiceMonitors, the Grafana dashboard and the Helm values: `deploy/monitoring/` (applied by `scripts/k8s-monitoring-up.sh`).

### Configuration entry points
- Scheduler env vars: the header comment of `scheduler/cmd/scheduler/main.go` (policy, intervals, `FAKE_TIME_SCALE`, cost model overrides). Worker env vars: `worker/kgpu_worker/config.py`. Simulator env vars: the top of `shard/cmd/shardsim/main.go`.
- Stacks: `deploy/compose.yaml` (knobs at the top), `deploy/k8s/` (kind cluster: `scripts/kind-up.sh`).

### Migrations (the schema is the spec)
- `db/migrations/`, applied by the scheduler at startup in filename order (`db/migrate.go`, advisory-locked): 0001 the core tables and the claim/reaper indexes; 0002 `rounds.n_shards`; 0003 the query stream and `shard_status`; 0004 the shard FKs (bug 3); 0005 `workers.node` (decision 11); 0006 `moved`/`speculative`, the partial unique indexes and the generated-column FKs (decision 15).
