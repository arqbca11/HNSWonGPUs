# Test logs

Narrated test output, kept so what was checked and what happened can be read without rerunning anything. **For a plain-language description of every test and the scenario it checks, read [`docs/tests.md`](../docs/tests.md) first**; the logs are the evidence, the catalog is the explanation. Each log has a header (date, commit, versions), one line per test step as the test narrates it, `[db]` lines showing rows as Postgres held them, the Docker events for the throwaway Postgres container, and the result.

| File | What |
| --- | --- |
| `latest.log` | The most recent run of `scripts/test-db.sh`, whatever it covered. Overwritten every run. |
| `phase1/step1-store.log` | Step 1: the store's lease operations and fencing (Go, `scheduler/store`). |
| `phase1/step2-scheduler.log` | Step 2: the HTTP API, policy v0 and cost model (Go, `scheduler/api`, `scheduler/policy`, `scheduler/costmodel`). |
| `phase1/step2-crosscheck.log` | The independent cross-check agent's tests for steps 1 and 2, written from the spec without reading the implementation (Go, `scheduler/crosscheck`, non-worker tests). |
| `phase1/step3-worker.log` | Step 3: the Python worker (`worker/tests`). |
| `phase1/step3-crosscheck-worker.log` | The cross-check agent's tests for the worker, run as a black-box process (Go, `scheduler/crosscheck`, `Worker*` tests). |
| `phase1/step4-compose-failures.log` | Step 4: the operator's failure script on the Compose stack (crash, pause, scheduler restart, duplicate submit). |
| `phase1/step4-crosscheck-compose.log` | The cross-check agent's nine Compose scenarios (`Compose*` tests, need `KGPU_COMPOSE=1`). |
| `phase1/step5-workload.log` | Step 5: the workload generator's 13 tests (written by an independent agent). |
| `phase1/step5-shardsim.log` | Step 5: the shard simulator against the real API and Postgres (end-to-end round, local build blocking, preemption). |
| `phase1/step5-compose-round.log` | Step 5: the first real six-shard round on the Compose stack, with the timeline. |
| `phase1/step5-crosscheck-shardsim.log` | The cross-check agent's nine black-box tests of the `shardsim` binary. |
| `phase1/step6-compose-failures-load.log` | Step 6: the four failure tests injected into live rounds on the Compose stack. |
| `phase1/step6-crosscheck-load.log` | The cross-check agent's eight under-load scenarios (`Load*` tests, need `KGPU_COMPOSE=1`). |
| `phase1/step7-compose-round-50.log` | Step 7: one 50-shard, 3-worker round with the timeline. |
| `phase1/step7-failures-load-50.log` | Step 7: the four failure tests injected into live 50-shard rounds. |
| `phase1/step7-crosscheck-load-50.log` | The cross-check agent's suite at 50 shards and 3 workers, with the batching, fidelity and GPU-queue scale tests. |
| `phase2/step1-kind-cluster.log` | Phase 2 step 1: the kind cluster's GPU node admits only worker-shaped pods. |
| `phase2/step2-postgres-scheduler.log` | Phase 2 step 2: Postgres and the scheduler on kind; pod deletions survived. |
| `phase2/step12-crosscheck-kind.log` | The cross-check agent's nine tests of steps 1 and 2 on kind, in its own namespace. |
| `phase2/step3-workers.log` | Phase 2 step 3: the workers on the GPU node; graceful deletion, in-container crash, forced deletion. |
| `phase2/step3-crosscheck-kind-workers.log` | The cross-check agent's nine worker scenarios on kind: two reapers, Postgres outage, liveness freeze, SIGTERM both branches, node drain. |
| `phase2/step3b-gpu-cards.log` | Phase 2 step 3b: GPU cards as a countable resource on two GPU nodes; filling every card, one too many Pending. |
| `phase2/step3b-crosscheck-kind-cards.log` | The cross-check agent's eight card scenarios: reverse-order startup, fragmentation, kill keeps the card, capacity patched down, one-node drain halving the pool. |
| `phase2/step4-shards-outside.log` | Phase 2 step 4: simulator rounds from the laptop against the cluster; 6/2, 50/2 with a worker deleted mid-round, 50/4, 50/2 with the scheduler pod deleted mid-round (bug 10 closed). |
| `phase2/step4-crosscheck-kind-outside.log` | The cross-check agent's seven from-the-laptop scenarios: the NodePort as the only door, scheduler deletion mid-round (found bug 10), two simulators at once, zero workers then scale-up. |
| `phase2/step5-monitoring.log` | Phase 2 step 5: metrics on all three sides, Prometheus targets including the laptop, two rounds' numbers, the Grafana dashboard. |
| `phase2/step5-crosscheck-kind-metrics.log` | The cross-check agent's eight metrics scenarios: names and types, exactly-once round observation by either closer, lease expiry under a kill, scrape wiring, every dashboard panel. |
| `phase2/step6-failures-dashboard.log` | Phase 2 step 6: the failure injections read through Grafana's datasource proxy; a leaseholder crashed and its pod deleted mid-build (the done-when), the scheduler pod deleted mid-round, both GPU nodes drained and uncordoned. |
| `phase2/step6-crosscheck-kind-failures.log` | The cross-check agent's 21 checks in 5 scenarios: release vs expiry told apart by the build row, two kills on one build, the scheduler deleted at the moment a round closes (4 trials), one-node drains at 4 and 2 replicas; found F1-F3 (decisions 37, 38). |
| `phase2/amendment-crosscheck-kind-spread.log` | The cross-check agent's 6 placement scenarios for the worker soft spread (decision 39): replacements to the emptier node, scale-up from 0 and 1, cordoned and card-full nodes packing without Pending, 4 replicas unchanged, the 5th Pending for cards only. |
| `phase2/amendment-draining-scheduler.log` | The draining scheduler (decision 40): a healthy poll stamps a round, a draining poll answers without stamping, shard reports still write, the reaper's tick stamps and the draining pod serves it. |
| `phase2/amendment-crosscheck-kind-lameduck.log` | The cross-check agent's 3 lame-duck scenarios over 8 scheduler replacements: the dying pod polled through the pod proxy never stamps and keeps recording writes; exactly-once observation in three timing modes; the residual race's real width measured. |
| `phase3/step1-policies.log` | Phase 3 step 1: the five placement policies as pure functions — threshold placement, preemption with sunk cost, the memory guard (bug 11) and the flapping guard, recovery-based reneging, reprioritizing, speculation's no-go cases, the oracle's exact replays, the strict config names. |
| `phase3/step1-crosscheck-policies.log` | The cross-check agent's 40 pure subtests of the policy package, written from the docs and `go doc` alone: purity under permutation, hand-computed boundaries, both flapping directions, three replayed oracle streams; found bug 11 and the F2–F6 findings. |
| `phase3/step2-estimator.log` | Phase 3 step 2: the queue-replay estimator — hand-replayed fluid transfer sharing on one node and two, the memory-aware claim mirror, fits-nowhere as +Inf, the node column round-tripping through heartbeats, and the submit response's `gpu_queue_wait_ms` against a live queue. |
| `phase3/step2-crosscheck-estimator.log` | The cross-check agent's 44 subtests of the estimator, written from the docs and `go doc` alone: every fluid-sharing shape by hand, a four-worker six-job scenario replayed entirely by hand, the snapshot's claim order confirmed by a real claim, the per-node term end to end through the API; found F1 (the duplicate's `gpu_queue_wait_ms`) and the G1–G7 gaps. |
| `phase3/step3-reconsider.log` | Phase 3 step 3: the reconsider timer — every action kind executed as a guarded row change, the real reactive policy preempting a hurting build and sparing a 95%-done one, a fits-nowhere build reneged (bug 11's cure), the speculation lifecycle in the store and end to end through the API with the local build aborted only when the copy's `done` arrives. |
| `phase3/step3-crosscheck-reconsider.log` | The cross-check agent's 44 subtests of the reconsider timer, written from the docs, migrations and `go doc` alone: the copy claimed by the roadmap's claim SQL typed in by hand, ninety repetitions of three races each with exactly one coherent winner, decision 18's scaling proven at scale 1 vs 10⁵, the real Python worker building a copy and fenced out by a cancel in 115 ms; found bugs 12 and 13 (F1, F3), F2, F4 and the G1–G7 gaps. |

Per-step logs are the state of the tests at the end of that step, including any fixes from the step's cross-check round, regenerated whenever the step's code changes. `history/` (local only, not committed) keeps every timestamped run.

Produce a saved log with `scripts/test-db.sh --save phase1/<name> <args>`.
