# Test catalog

Every test in the repo, in plain words: the scenario it sets up and what it proves. Tests marked **cross-check** were written by the independent agent from the spec alone, without reading the implementation. The narrated output of each group is in `test-logs/phase1/` (see the log column).

Legend for "proves": the invariants are numbered in `CLAUDE.md`; decisions are numbered in `docs/design/phase1-scheduler.md`.

## Contents

Each step has its own tests first, then any scripts, then the independent cross-check's. Phase 1 sections come before Phase 2.

- [Step 1: the store (Go, `scheduler/store`). Log: `step1-store.log`](#step-1-the-store-go-schedulerstore-log-step1-storelog)
- [Step 2: scheduler API, policy, cost model (Go). Log: `step2-scheduler.log`](#step-2-scheduler-api-policy-cost-model-go-log-step2-schedulerlog)
- [Steps 1 and 2, cross-check (Go, `scheduler/crosscheck`, non-worker tests). Log: `step2-crosscheck.log`](#steps-1-and-2-cross-check-go-schedulercrosscheck-non-worker-tests-log-step2-crosschecklog)
- [Step 3: the Python worker (`worker/tests`). Log: `step3-worker.log`](#step-3-the-python-worker-workertests-log-step3-workerlog)
- [Step 3, cross-check of the worker as a black-box process (Go, `scheduler/crosscheck`, `Worker*` tests). Log: `step3-crosscheck-worker.log`](#step-3-cross-check-of-the-worker-as-a-black-box-process-go-schedulercrosscheck-worker-tests-log-step3-crosscheck-workerlog)
- [Step 4: Compose failure tests (`scripts/compose-failures.sh`). Log: `step4-compose-failures.log`](#step-4-compose-failure-tests-scriptscompose-failuressh-log-step4-compose-failureslog)
- [Step 4, cross-check of the Compose stack (Go, `scheduler/crosscheck`, `Compose*` tests; need `KGPU_COMPOSE=1`). Log: `step4-crosscheck-compose.log`](#step-4-cross-check-of-the-compose-stack-go-schedulercrosscheck-compose-tests-need-kgpucompose1-log-step4-crosscheck-composelog)
- [Step 5: shard simulator (Go, `shard/sim`). Log: `step5-shardsim.log`](#step-5-shard-simulator-go-shardsim-log-step5-shardsimlog)
- [Step 5: workload generator (Go, `experiments/workload`, written by an independent agent). Log: `step5-workload.log`](#step-5-workload-generator-go-experimentsworkload-written-by-an-independent-agent-log-step5-workloadlog)
- [Step 5, cross-check of the shard simulator as a black-box process (Go, `scheduler/crosscheck`, `Shardsim*` tests). Log: `step5-crosscheck-shardsim.log`](#step-5-cross-check-of-the-shard-simulator-as-a-black-box-process-go-schedulercrosscheck-shardsim-tests-log-step5-crosscheck-shardsimlog)
- [Step 6: shard client unit tests (Go, `shard/client`). Log: `step5-shardsim.log`](#step-6-shard-client-unit-tests-go-shardclient-log-step5-shardsimlog)
- [Step 6: failure tests under simulator load (`scripts/compose-failures-load.sh`). Log: `step6-compose-failures-load.log`](#step-6-failure-tests-under-simulator-load-scriptscompose-failures-loadsh-log-step6-compose-failures-loadlog)
- [Step 6, cross-check under load (Go, `scheduler/crosscheck`, `Load*` tests; need `KGPU_COMPOSE=1`). Log: `step6-crosscheck-load.log`](#step-6-cross-check-under-load-go-schedulercrosscheck-load-tests-need-kgpucompose1-log-step6-crosscheck-loadlog)
- [Step 7: 50 shards and 3 workers (`scripts/compose-round.sh`, `scripts/compose-failures-load.sh`). Logs: `step7-compose-round-50.log`, `step7-failures-load-50.log`](#step-7-50-shards-and-3-workers-scriptscompose-roundsh-scriptscompose-failures-loadsh-logs-step7-compose-round-50log-step7-failures-load-50log)
- [Step 7, cross-check at 50 shards (Go, `scheduler/crosscheck`, `Load*` tests with `KGPU_XCHECK_SHARDS=50 KGPU_XCHECK_WORKERS=3`; need `KGPU_COMPOSE=1`). Log: `step7-crosscheck-load-50.log`](#step-7-cross-check-at-50-shards-go-schedulercrosscheck-load-tests-with-kgpuxcheckshards50-kgpuxcheckworkers3-need-kgpucompose1-log-step7-crosscheck-load-50log)
- [Phase 2, step 1: the kind cluster (`scripts/kind-check-gpu-node.sh`). Log: `phase2/step1-kind-cluster.log`](#phase-2-step-1-the-kind-cluster-scriptskind-check-gpu-nodesh-log-phase2step1-kind-clusterlog)
- [Phase 2, step 2: Postgres and the scheduler on kind (`scripts/k8s-check-step2.sh`). Log: `phase2/step2-postgres-scheduler.log`](#phase-2-step-2-postgres-and-the-scheduler-on-kind-scriptsk8s-check-step2sh-log-phase2step2-postgres-schedulerlog)
- [Phase 2, steps 1 and 2, cross-check on kind (Go, `scheduler/crosscheck`, `TestKind` subtests; need `KGPU_KIND=1`). Log: `phase2/step12-crosscheck-kind.log`](#phase-2-steps-1-and-2-cross-check-on-kind-go-schedulercrosscheck-testkind-subtests-need-kgpukind1-log-phase2step12-crosscheck-kindlog)
- [Phase 2, step 3: the workers on kind (`scripts/k8s-check-step3.sh`). Log: `phase2/step3-workers.log`](#phase-2-step-3-the-workers-on-kind-scriptsk8s-check-step3sh-log-phase2step3-workerslog)
- [Phase 2, step 3, cross-check on kind (Go, `scheduler/crosscheck`, `TestKindWorkers` subtests; need `KGPU_KIND=1`). Log: `phase2/step3-crosscheck-kind-workers.log`](#phase-2-step-3-cross-check-on-kind-go-schedulercrosscheck-testkindworkers-subtests-need-kgpukind1-log-phase2step3-crosscheck-kind-workerslog)
- [Phase 2, step 3b: GPU cards as a countable resource (`scripts/k8s-check-gpu-cards.sh`). Log: `phase2/step3b-gpu-cards.log`](#phase-2-step-3b-gpu-cards-as-a-countable-resource-scriptsk8s-check-gpu-cardssh-log-phase2step3b-gpu-cardslog)
- [Phase 2, step 3b, cross-check on kind (Go, `scheduler/crosscheck`, `TestKindCards` subtests; need `KGPU_KIND=1`). Log: `phase2/step3b-crosscheck-kind-cards.log`](#phase-2-step-3b-cross-check-on-kind-go-schedulercrosscheck-testkindcards-subtests-need-kgpukind1-log-phase2step3b-crosscheck-kind-cardslog)
- [Phase 2, step 4: shards outside the cluster (`scripts/k8s-check-step4.sh`, via `scripts/k8s-round.sh`). Log: `phase2/step4-shards-outside.log`](#phase-2-step-4-shards-outside-the-cluster-scriptsk8s-check-step4sh-via-scriptsk8s-roundsh-log-phase2step4-shards-outsidelog)
- [Phase 2, step 4, cross-check from the laptop (Go, `scheduler/crosscheck`, `TestKindOutside` subtests; need `KGPU_KIND=1`). Log: `phase2/step4-crosscheck-kind-outside.log`](#phase-2-step-4-cross-check-from-the-laptop-go-schedulercrosscheck-testkindoutside-subtests-need-kgpukind1-log-phase2step4-crosscheck-kind-outsidelog)
- [Phase 2, step 5: metrics and dashboards (`scripts/k8s-check-step5.sh`). Log: `phase2/step5-monitoring.log`](#phase-2-step-5-metrics-and-dashboards-scriptsk8s-check-step5sh-log-phase2step5-monitoringlog)
- [Phase 2, step 5, cross-check of metrics and dashboards (Go, `scheduler/crosscheck`, `TestKindMetrics` subtests; need `KGPU_KIND=1`). Log: `phase2/step5-crosscheck-kind-metrics.log`](#phase-2-step-5-cross-check-of-metrics-and-dashboards-go-schedulercrosscheck-testkindmetrics-subtests-need-kgpukind1-log-phase2step5-crosscheck-kind-metricslog)

## Step 1: the store (Go, `scheduler/store`). Log: `step1-store.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| ClaimOrdersByPriorityThenAge | Three builds queued with priorities 1, 5, 3. One worker claims three times, then once more on the empty queue. | Claims come out highest priority first (5, 3, 1), each with attempt 1; an empty queue returns nothing. |
| ConcurrentClaimsNeverShareABuild | Twenty builds; eight goroutines claim in a loop at the same time until the queue is empty. | Every build is claimed exactly once. `FOR UPDATE SKIP LOCKED` keeps concurrent workers off the same row (invariant 2). |
| FencingAfterReap | Worker 1 claims. Its lease is forced into the past (as if it stopped renewing). The reaper runs. Worker 2 claims. Worker 1 "wakes up" and tries renew, complete and fail with its old attempt number. | The reaper requeues only expired leases and is idempotent (invariant 5). The re-claim bumps attempt (invariant 4). Every write from the old owner matches zero rows and the row is untouched (invariant 3). Worker 2 completes once; a done row is never reaped. This is the SIGSTOP scenario at the SQL level. |
| ReleaseReturnsToQueueAndFences | Worker 1 claims then releases (graceful shutdown). Worker 2 claims. Worker 1 tries to complete. | Release puts the build back; the next claim gets attempt 2; the releasing owner's later write is refused (decision 15). |
| ClaimRespectsWorkerMemory | A 16 GiB build at high priority and a 1 GiB build. An 8 GiB worker claims, then a 24 GiB worker. | The small worker skips the build that does not fit even though it is first in line; the big worker takes it. "Fits" is `mem_bytes <= capacity` (decisions 7, 26). |
| DuplicateSubmitIsNoop | Submit a build, let a worker claim it, then submit the same build id again with a different priority and placement. | The second submit inserts nothing and changes nothing, even mid-lease (invariant 6). |
| LocalBuildsAreNeverReaped | Submit a local build. A worker tries to claim. The reaper runs. The shard reports it done, twice. | Local builds sit in state `running`, invisible to claims and the reaper; the first done is accepted, the second refused (decision 5). |
| HeartbeatAndPoolState | No workers, then two heartbeats, then one worker's last_seen pushed ten minutes into the past. | Pool state counts only recently seen workers and reports the largest memory among them; heartbeat is an upsert. |
| RoundFinishesOnlyWhenEveryShardIsDone | A two-shard round. Shard 1 submits, two queries arrive, its build completes, both queries finish, it reports stream done. Then shard 2 goes through the same, one step at a time. | The round is stamped finished only when both shards have terminal builds, finished queries and `stream_done`; the stamp is the last query's finish time, not the time of the check; running the check again changes nothing (decisions 18, 30). |
| PreemptAndRenegeTransitions | A running local build is preempted to the GPU queue with a priority, then reneged back to local, then preempted again and claimed by a worker, then renege is attempted on the leased build. | Preempt and renege flip placement and state under their guards; attempt is untouched by both; a shard that did not notice its build was taken away is refused at done; a leased build is never moved (decision 31). |
| SchemaRefusesOrphanShardRows | Insert a status row and a query row for a shard with no build, and a second build for the same shard in the same round, straight at Postgres. | The foreign keys and the unique constraint from migration 0004 refuse all three; through the store the same facts come back as `ErrNoBuild` (decision 33). |
| ReportBatchIsIdempotentAndFillsIn | Three batches for one shard: two arrivals with one started; a resend plus a finish and a start; a stale resend of the first batch; then a final `stream_done` batch, a resend after it, a new arrival after it, and a batch for a shard with no build. | Records upsert and timestamps only fill in, so stale resends change nothing; a resend after `stream_done` is fine but a new arrival is refused with nothing written; no build means `ErrNoBuild`; the round completes from batched records (decisions 60, 63). |
| ConcurrentDuplicateIntoLastSlotIsNoop | Eight goroutines submit the same build id at once into a round with one slot, five times over. | Exactly one insert and seven no-ops each time, never "round full": the duplicate check runs after the round lock (bug log 6, invariant 6). |
| ReapFailsBuildAfterRetryBudget | A "poison" build. Three workers in turn claim it and die (lease forced to expire). The reaper runs each time with a budget of 3. A fourth worker claims. | The first two expiries requeue; the third marks the build failed with a reason; the queue is clear and the next worker gets the healthy build; with the budget disabled an expiry always requeues (decision 42, bug log 5). |

## Step 2: scheduler API, policy, cost model (Go). Log: `step2-scheduler.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| api: EndToEndRoundOverHTTP | A full two-shard round through HTTP: a worker heartbeats; a round is created; shard 1's 100k build goes to the GPU and a retry of the same submit returns the original decision; shard 2's 20M build is placed local because it fits no worker; queries arrive at shard 1; shard 1 reports and sees its build still queued; a query that does not need the index runs; the worker claims and completes via SQL; shard 1's next report sees `done` and runs its index-dependent query; shard 2 finishes locally and its query runs; the round stays open until shard 2 reports stream done; a reaper tick stamps it. | The whole protocol a shard will follow, end to end, with the status codes the API table promises. |
| api: ReportReflectsPreemption | Shard 0 builds locally and reports a growing queue. The build is preempted to the GPU queue (as Phase 3's reconsider would). The shard reports again, then tries to report its local build done. | The report reply is how a shard learns of a preemption; its late done is refused with 409; a report for a shard with no build is 404; out-of-range report fields are 400 (decision 29). |
| api: ReportWithJobBatch | A report carrying two records, one finished; three malformed records; a final batch with `stream_done`; then a new arrival. | The records land with the shard's timestamps and the reply still carries the build; malformed records are 400; a new arrival after `stream_done` is 409 (decision 63). |
| api: ValidationAndNotFound | A table of bad requests: zero shards, an unknown JSON field, a bogus round id, a non-integer path, an unknown round, zero vectors, unknown build and job, and the health endpoint. | Each gets the documented status code; unknown fields are rejected, not ignored (decision 21). |
| api: SubmitValidationAgainstRound | Shards numbered 7 and 42 submit into a two-shard round; a third shard tries; shard 7 resubmits with a different body; the round is finished; shard 42 resubmits. | Shard ids need not be dense; the third distinct shard is refused with 409 (the cap); a duplicate's estimates come from the stored row; a duplicate into a finished round is still the idempotent 200 (decision 27). |
| policy: AlwaysGPU | Four inputs: fits, exactly fits, too big for every live worker, no live workers at all. | GPU, GPU, local, GPU. The memory rule applies only when a worker is live, since an empty pool may be cold-starting (decision 19). The same input twice gives the same answer (invariant 8). |
| costmodel: DefaultsAreInTheIntendedRange | A 100k x 128 build and a 1k x 128 build. | The big build takes about 20 s on CPU and less on the GPU; the tiny one is faster on CPU because of the fixed GPU overhead and transfer. The defaults produce the tradeoff the project studies. |
| costmodel: MonotonicInSize | Builds of 1k, 10k, 100k, 1M vectors. | CPU time strictly increases with size; degenerate sizes cost nothing. |
| `TestMetricsExposedAndRoundObserved` | One round is created, claimed, completed and reported through the API, then the reaper ticks; `/metrics` is read. | Every roadmap metric is exposed, and finishing a round observes its duration and the GPU builds' queue waits (Phase 2 decision 31). |

## Steps 1 and 2, cross-check (Go, `scheduler/crosscheck`, non-worker tests). Log: `step2-crosscheck.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| ReportUpsertsStatusAndRepliesWithBuild | A shard reports twice. | One status row per shard, updated in place; the reply carries the build's placement, state and attempt (decision 29). |
| ReportReplyTracksWorkerProgress | A shard reports before a claim, after a claim, after completion. | The reply says queued, then leased with attempt 1, then done. |
| ReportFromShardWithoutBuildIs404 | A report for a shard that never submitted. | 404, and nothing is written. |
| InvalidReportRejectedAndPreviousKept | A valid report, then one with more index-waiting queries than queue depth. | 400, and the earlier report is still what is stored. |
| ArrivalRecordedOnceAndRepeatUnchanged | A query arrival, then the same seq again with different fields. | 201 then 200; the row is unchanged (decision 28). |
| JobStartDoneStatusCodes | Start and done on a nonexistent query, done before start, start twice, done twice. | 404 for no such query, 409 for the wrong state. |
| PreemptMovesRunningLocalToGPUQueue | A running local build is preempted. The shard reports and tries done. A worker claims. | Exactly the columns decision 31 names change, including that `enqueued_at` keeps the submit time (this test caught the implementation resetting it); the first claim gives attempt 1. |
| RenegeMovesQueuedGPUToLocal | A queued GPU build is reneged to local. A worker tries to claim. The shard reports and completes it. | Placement and state flip, attempt untouched, no claim possible, done succeeds. |
| TransitionsRefusedOutsideTheirGuards | Preempt and renege attempted on a leased build and on builds in the wrong state; the lease holder renews. | Neither transition touches a leased build or a build in the wrong state; the holder's renew still works. |
| RoundWaitsForStreamDoneThenStampsLatestFinish | All builds and queries done but one shard has not said stream done; then it does. | The round waits for `stream_done`; the stamp equals the latest child finish. |
| RoundStaysOpenWhileAQueryIsUnfinished | Everything done except one query that has started but not finished. | The round stays open. |
| StreamDoneFromBuildlessShardDoesNotCount | Two real shards, one still streaming; a `stream_done` report written for a shard 99 with no build. | The report is refused at the store (decision 33); the round does not finish. This test caught the original completion bug (bug log 3). |
| SubmitValidationAndCap | Zero and negative shard counts; an unknown round; a third shard into a two-shard round; a duplicate in a full round; sparse shard ids. | 400, 404, 409, 200, accepted, in that order (decision 27). |
| ConcurrentSubmitsRespectCap | Twelve shards submit at the same moment over HTTP into a round capped at 3. | Exactly 3 get 201 and 9 get 409. This test caught the lock-then-count race (bug log 4). |
| ConcurrentStoreSubmitsRespectCap | The same, through the store directly. | Exactly 3 inserted, 9 `ErrRoundFull`. |
| SubmitWithJobsFieldRejected | A submit carrying the old spec's `jobs` field. | 400, nothing written: a shard cannot declare its future (decisions 21, 28). |

## Step 3: the Python worker (`worker/tests`). Log: `step3-worker.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| test_matches_go_defaults | The Python cost model computes the 100k x 128 and 1k x 128 numbers. | They equal the Go test's output to the microsecond. If either side's formula drifts, this fails. |
| test_env_overrides | `COST_SPEEDUP` and `COST_GPU_OVERHEAD` set in the environment. | The overrides take effect, same variable names as the scheduler. |
| test_claim_order_and_attempt | Two builds at priorities 1 and 5; the Python claim runs three times. | Priority order, attempt 1, round id returned, nothing on the empty queue: the Python SQL behaves like the Go SQL. |
| test_fencing_after_reap | The full SIGSTOP walk from the Python side: claim, expire, reap, re-claim, stale renew/complete/fail/release, live renew/complete, second complete. | Each stale write is refused; each live write is accepted once (decision 13: the two SQL mirrors agree). |
| test_memory_filter_and_release | A 16 GiB and a 1 GiB build; an 8 GiB worker claims, releases, re-claims; a 24 GiB worker claims. | Memory filter and release behave as in Go. |
| test_heartbeat_upsert | Two heartbeats with different capacities. | One row, capacity updated. |
| test_worker_claims_builds_completes | Two queued builds; a real `Worker` object runs in a thread with fake builds sped up 50x. | Both builds reach done with attempt 1; the worker is registered; it exits cleanly with `builds_done=2`. |
| test_lease_lost_mid_build_abandons_without_completing | A worker claims a long build. The test plays reaper and a second worker: requeues the row and leases it to `gpu-other` with attempt 2. | The worker's renew is rejected, it cancels its build, logs `lost ownership`, and never writes to the row again; the row stays with `gpu-other` (decision 36). |
| test_sigterm_releases_long_build_and_finishes_short_one | SIGTERM with 0.8 s of build left and a 0.2 s budget; then SIGTERM on a short build with a generous budget. | The long build is released at once (queued, attempt unchanged, no owner); the short one is finished before exit (decision 37). |
| test_two_workers_never_share_a_build | Twelve builds, two workers at once. | All twelve done, every attempt 1, no lost ownership on either side. |
| test_registers_before_first_claim | A worker starts with a build waiting. | Its `workers` row exists before its first claim's `started_at` (cross-check finding). |
| test_build_error_marks_failed_with_reason_and_worker_continues | One build is listed in `FAKE_FAIL_BUILD_IDS`, another is healthy. | The failing build ends `failed` with `RuntimeError: injected failure…` as the reason and no lease; the worker completes the healthy one and does not crash (decisions 40, 41). |
| test_bad_config_is_refused_before_connecting | Zero bandwidth, renew interval equal to the lease, negative time scale, a non-numeric cost constant, an unknown log format. | Each is refused with a message naming the variable, before any connection (decision 39). |
| test_survives_lost_database_connections_mid_build | Mid-build, every one of the worker's backend connections is terminated server-side with `pg_terminate_backend` (what a Postgres restart does to clients). | All three threads reconnect; the lease is renewed in time (no reap, attempt stays 1); the build completes; the worker goes on to claim and complete the next one (bug log 7, decision 48). |
| test_renew_outage_past_lease_deadline_assumes_lost | Every renew raises a connection error for longer than the lease. | The worker assumes the lease is lost, cancels the build, counts it lost and writes nothing (decision 48). |
| `test_waits_for_schema_at_startup` | The `workers` table is hidden; a worker starts; after 1.5 s the table is restored. | The worker waits for a missing schema instead of crashing (bug 9, Phase 2 decision 25) and registers on its next retry. |
| `test_metrics_served_on_the_health_port` | GET `/metrics` on the probe server. | The worker's metrics are served on the probe port with every `builds_total` outcome present from the start (Phase 2 decision 35). |

## Step 3, cross-check of the worker as a black-box process (Go, `scheduler/crosscheck`, `Worker*` tests). Log: `step3-crosscheck-worker.log`

These start the real `python -m kgpu_worker` as a subprocess, configure it only through its documented environment variables, and watch Postgres and its log.

| Test | Scenario | Proves |
| --- | --- | --- |
| WorkerHeartbeatsBeforeFirstClaimAndEveryInterval | A worker starts with a build waiting; the test compares its registration time with the claim time and watches `last_seen` move. | Registered before the first claim; heartbeats keep arriving at the configured interval. This test caught the heartbeat race. |
| WorkerCompletesBuildForScaledModelTime | One build; measure how long the worker takes. | Claim then complete with attempt 1; the build lasts the cost model's time divided by `FAKE_TIME_SCALE` (1.63 s measured vs 1.63 s modeled); every build log line carries build_id, attempt and round_id. |
| WorkerTextLogFormatCarriesBuildFields | Same with `LOG_FORMAT=text`. | The three fields are present in text logs too. |
| WorkerClaimsOnlyWhatFitsItsMemory | A worker with 5000 bytes of capacity; builds of 5001 (higher priority) and 5000 bytes. | The 5001-byte build is never claimed; the 5000-byte one is (fits means `<=`). |
| WorkerRenewKeepsLeaseAliveUnderReaper | A 3.3 s build with a 1 s lease while the test runs the reaper every 100 ms. | The lease is never reaped; `lease_until` advanced 15 times. The renewer does its job. |
| WorkerKill9MidBuildAnotherWorkerCompletesOnce | Worker A claims; the test sends SIGKILL mid-build; the reaper runs; worker B is started. | Reap reports attempt 1 owned by A; B completes attempt 2; exactly one completion. Roadmap failure test 1. |
| WorkerSIGSTOPPastLeaseLosesOwnershipAndWritesNothing | Worker A is paused with SIGSTOP past its lease; the reaper and a second claim move on; A gets SIGCONT. | A's renew is rejected, it logs `lost ownership` for attempt 1, and the row is byte-for-byte unchanged a full build time later; A exits 0 on SIGTERM. Roadmap failure test 2. |
| WorkerReclaimedMidBuildCancelsWithinARenewInterval | Mid-build, the test requeues and re-leases the row to another owner. | `lost ownership` appears within one renew interval, about 5 s before the build would have ended; no complete, fail or release follows (decision 36). |
| WorkerSIGTERMReleasesBuildOverBudget | SIGTERM with more remaining time than the budget. | Row back to queued with attempt 1 and lease columns NULL; the next claim gets attempt 2; exit 0 (decisions 24, 37). |
| WorkerSIGTERMFinishesBuildWithinBudgetAndStopsClaiming | SIGTERM with 2.75 s remaining and the default 5 s budget; a second build is queued. | The current build reaches done; the second stays untouched at attempt 0; exit 0. |
| WorkerBadConfigIsRefusedBeforeAnyClaim | `COST_BANDWIDTH=0` with a build waiting. | Exit 2 with the variable named on stderr; the build is still queued at attempt 0; no worker row (decision 39). This scenario was the original poison-pill crash (bug log 5). |
| WorkerBuildErrorRecordsFailWithReason | One build in `FAKE_FAIL_BUILD_IDS`, one healthy. | The failing build ends `failed` with the exception text and no lease; the worker completes the other and is still running (decision 41). |

## Step 4: Compose failure tests (`scripts/compose-failures.sh`). Log: `step4-compose-failures.log`

Not a test suite but an operator's script: it brings the real stack up and does to it what the roadmap's failure tests describe, narrating what Postgres and the container logs show.

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | `docker compose up --build`, wait for health | Start order by health works: Postgres, then scheduler (which applies the schema), then two workers register. |
| 2 | One round, one small build | The stack does the normal thing end to end. |
| 3 | SIGKILL the worker's python process from inside its container (tini is PID 1), mid-build | A crash leaves no chance to release. The lease expires, the reaper requeues, the other worker completes attempt 2, exactly once. The restart policy brings the crashed worker back. (An external `docker kill` is a manual stop and is not restarted; `kill -9 1` from inside is ignored by the kernel.) Roadmap failure test 1. |
| 4 | `docker pause` the leaseholder past its lease, then `docker unpause` | The paused process wakes holding attempt 1, its renew is rejected, it logs `lost ownership` and writes nothing more for that attempt; attempt 2 is completed exactly once, by whichever worker claimed it (possibly the woken one, re-claiming). Roadmap failure test 2. |
| 5 | `docker compose restart scheduler` while a build is leased | The worker is unaffected (talks to Postgres, not the scheduler); the build completes as attempt 1; the restarted scheduler stamps the round. Roadmap failure test 3. |
| 6 | Resubmit a finished build with a different body | 200 `created:false`, the stored decision. Roadmap failure test 4. |

## Step 4, cross-check of the Compose stack (Go, `scheduler/crosscheck`, `Compose*` tests; need `KGPU_COMPOSE=1`). Log: `step4-crosscheck-compose.log`

These bring up their own copy of the stack (project `kgpu-xcheck`) and drive it only through the API and the `docker` CLI.

| Test | Scenario | Proves |
| --- | --- | --- |
| ComposeKill9LeaseholderMidBuild | Crash the worker process holding a 6.5 s build, 2 s in (SIGKILL to its pid from inside the container); watch the row, the scheduler log, every worker log and the crashed container. | The reaper requeues after the 5 s lease, the other worker completes attempt 2, exactly one `completed` line exists, and the restart policy brings the crashed container back (roadmap failure test 1, decision 47). The agent's first version used `docker kill`, which Docker treats as a manual stop; see decision 47. |
| ComposePauseLeaseholderPastLease | `docker pause` the leaseholder 1 s in, for 8 s against a 5 s lease; unpause while attempt 2 is still building elsewhere. | Reaped and re-claimed while paused; on resume the worker logs `renew rejected` and `lost ownership` for attempt 1 and writes nothing; one completion at attempt 2; the done row is unchanged afterwards (roadmap failure test 2, decisions 25 and 36, invariant 3). |
| ComposeSchedulerRestartMidRound | `docker compose restart scheduler` while a build is leased, then resubmit, add the second shard, try a third, report stream done, read the round. | The worker never notices (done at attempt 1, no reap); state survives the restart (duplicate is 200 `created:false`, cap still 409); the round is stamped with the latest build finish (roadmap failure test 3, decisions 3, 18, 27). |
| ComposeReaperResumesAfterSchedulerOutage | Stop the scheduler, SIGKILL the leaseholder, wait past the lease, start the scheduler. | The new scheduler's reaper requeues the expired lease exactly once and attempt 2 completes once; the reaper keeps no state (decision 3, invariant 5). |
| ComposeDuplicateSubmit | Resubmit a finished build with a different body; then, repeatedly, 8 simultaneous submits of one new build, into a round's last slot and into a round with spare capacity. | A sequential duplicate is a no-op returning stored estimates (roadmap failure test 4, invariant 6); simultaneous duplicates give one 201 and seven 200 `created:false` even into the last slot (decision 26). Caught bug log 6. |
| ComposeDockerStopLeaseholder/LongRemainderIsReleased | `docker stop` the leaseholder 0.5 s into a 6.5 s build (budget 5 s). | The worker cancels, releases attempt 1, and exits 0 well inside `stop_grace_period`; another worker claims attempt 2 within 0.4 s, without a reap; one completion; the stopped container stays stopped (decisions 15, 37, 47). |
| ComposeDockerStopLeaseholder/ShortRemainderIsFinished | `docker stop` the leaseholder 3 s into a 6.5 s build. | The worker logs `finishing current build before exit`, completes attempt 1, then exits 0 inside the grace period (decision 37). |
| ComposeScaleToThreeWorkers | `--scale worker=3`, submit three 6.5 s builds, then scale back to 2. | The pool reports 3 live workers; three builds are leased at once by three distinct owners, each done once with attempt 1; the pool drops back to 2 after scale-down (decisions 6, 46, invariant 2). |
| ComposePostgresRestartMidBuild | `docker compose restart postgres` 1 s into a leased build, then check the build, the pool, the round and a fresh build. | The stack recovers, data survives on the named volume, the in-flight build completes exactly once, and no worker crashes (decision 48). Caught bug log 7. |

## Step 5: shard simulator (Go, `shard/sim`). Log: `step5-shardsim.log`

Integration tests: the real scheduler API over HTTP, real Postgres, and an in-test fake worker that claims and completes GPU builds. Workloads are hand-built, so these do not depend on the generator.

| Test | Scenario | Proves |
| --- | --- | --- |
| TwoShardRoundEndToEnd | Two shards, an 8 GiB worker, a fan-out query that spans both shards and needs the index, independent queries, a single-shard query; the round runs at 20x. | Both builds go to the GPU and finish at attempt 1; every query finishes; a query that needs the index never starts before its shard's build is done; an independent query runs at once on a free CPU; the fan-out query's latency is its slowest piece's; the round is stamped and the timeline has every shard, query and cluster query (decisions 32, 52, 56). |
| LocalBuildBlocksThenDrains | A worker with 1 byte of capacity, so the build is placed local; two queries arrive during it, one needing the index. | Both queries start only after the local build finishes: the CPU was occupied (modeling assumption; decision 32). |
| PreDDLQueriesRunAndAreBackfilled | A staggered DDL 1 s in; one query arrives and finishes before it, one arrives before it but needs the index, one arrives after. | The first runs before the submit and is backfilled with its true times; the second waits for the GPU build; every recorded run time is within a few ms of the scaled duration, since no HTTP sits inside the measured interval (decisions 59, 60). |
| PreemptionAbortsLocalBuild | A local build is preempted to the GPU queue at about 20% progress (as Phase 3's reconsider would), and a capable worker appears. | The shard aborts the local build at its next slice, the independent query runs on the freed CPU while the GPU builds, the index query waits for the GPU completion, and the build ends done at attempt 1 (decisions 31, 54). |

## Step 5: workload generator (Go, `experiments/workload`, written by an independent agent). Log: `step5-workload.log`

Pure functions; no database. The generator's modeling choices and their references are in `experiments/workload/DESIGN.md`.

| Test | Scenario | Proves |
| --- | --- | --- |
| Determinism | Every preset generated twice with seed 42, once with 43. | Same seed gives byte-identical JSON; a different seed differs (invariant 10). |
| PresetsValidate | All 6 presets at 6 and 50 shards, seeds 1 to 5; a bad name; 0 shards. | Every preset passes `Validate()`; bad inputs are errors. |
| ZipfSkew | The skewed preset at exponents 0.5, 1.0, 1.5. | Largest/median ratio is at least 3 at 1.0, rises with the exponent, stays mild at 0.5; sizes stay within [Min, Max]. |
| BimodalTwoClusters | Bimodal at 6 and 50 shards, 5 seeds. | Two size clusters with nothing between, the right heavy count, and some but not all heavy builds exceed the advised worker memory. |
| ClusterQueryRate | uniform, skewed, downstream_heavy at 6 and 50 shards, 3 seeds. | The number of cluster-level queries is within 4 standard deviations of rate times horizon. |
| PerShardLoadMatchesTarget | Skewed at 6 and 50 shards, 4 seeds. | Mean piece duration matches config; per-shard CPU utilisation is 0.4 within 15% at both scales. |
| NeedsIndexFraction | Skewed at 50 shards, fractions 0, 0.5, 0.8, 1. | The fraction by cluster query is within 0.03; all pieces of one query agree; local DDL never needs the index. |
| FanOutPieces | Skewed at 6 and 50 shards. | Fan-out pieces share id and arrival and land on distinct shards; width stays in range and covers it; single-shard queries have one piece; the fan-out fraction matches. |
| HotShard | downstream_heavy at 6 and 50 shards over a 600 s horizon. | The hot shard gets about M times a typical shard's single-shard queries and about 4 times its CPU load. |
| StreamFiniteAndSorted | All presets at 50 shards. | Shard ids in order; streams sorted; arrivals within [0, Horizon]; durations positive; non-fan-out ids unique. |
| LocalDDL | Skewed at 50 shards over a 400 s horizon. | Local DDL count is about rate times shards times horizon; mean duration about the configured 2 s. |
| DDLOffsets | Staggered at 6 and 50 shards; every other preset. | Staggered offsets lie in [0, Spread] and span it, and the horizon outlasts the spread; all-at-once offsets are zero. |
| PresetTable | All presets at 6 and 50 shards. | Prints each preset's parameters (documentation only). |

## Step 5, cross-check of the shard simulator as a black-box process (Go, `scheduler/crosscheck`, `Shardsim*` tests). Log: `step5-crosscheck-shardsim.log`

These run the real `shardsim` binary against an in-process scheduler, a reaper loop and a fake GPU worker, and observe only through Postgres, the binary's stdout and its timeline JSON.

| Test | Scenario | Proves |
| --- | --- | --- |
| ShardsimPrintWorkloadMatchesGenerate | `PRINT_WORKLOAD=1` run twice for four preset/seed/shard combinations, up to 50 shards, compared with `Generate` in the test process. | The binary replays exactly the seeded workload, deterministically across processes (invariant 10, decision 51). |
| ShardsimGPURoundRecordsEveryQueryOnce | A default six-shard round at 20x against two fake GPU workers. | Every workload query is recorded once, in order, and finished; builds end gpu/done at attempt 1, each claimed once; one CPU per shard (no overlapping runs); the round is stamped (decisions 28, 30, 32, 53). |
| ShardsimNeedsIndexWaitsForGPUBuild | The GPU is held back one second while queries pile up. | No `needs_index` query starts before its shard's build finished; independent queries are not blocked by them (decision 32). |
| ShardsimLocalBuildOccupiesCPU | The only worker has one byte of memory, so every build goes local. | A local build holds the CPU for cpu_build divided by the time scale (2.194 s against 2.187 s expected) and no query starts before it is done; local builds stay at attempt 0 (decisions 7, 32, 55). |
| ShardsimReportsCadenceAndStickyStreamDone | `shard_status` sampled every 20 ms during a round with a delayed GPU. | Reports come every `POLL_INTERVAL` and carry the backlog; `stream_done` comes after the last arrival and never flips back (decisions 29, 30, 33). |
| ShardsimHonoursPreemption | A running 200k local build is preempted at 0.4 s of 4.4 s, then a fake worker claims it. | The shard aborts within 100 ms, independent queries start on the freed CPU 189 ms later, `needs_index` queries wait for the GPU, the build ends gpu/done at attempt 1 by the worker's guarded complete (decisions 31, 54). |
| ShardsimSIGTERMStopsPromptly | SIGTERM mid-round at time scale 1. | The process exits within about 10 ms and records no further arrivals (decision 57). |
| ShardsimBadScenarioExitsBeforeScheduler | An unknown `SCENARIO`. | Exit 2 within milliseconds, zero HTTP requests, no round created (decision 57). |
| ShardsimTimelineJSONAgreesWithPostgres | Two bimodal rounds with mixed GPU and local placement and `TIMELINE_DIR` set. | The per-round JSON matches Postgres field for field (counts, placements, attempts, timestamps within 5 ms, exact duration); each query's kind and cluster id match the workload; rounds use seed, seed+1 (decision 56). |

## Step 6: shard client unit tests (Go, `shard/client`). Log: `step5-shardsim.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| RetriesTransientFailuresThenSucceeds | A fake scheduler answers 503 twice, then 200. | The call succeeds after three requests with backoff; the caller never sees the failures (decision 61). |
| GivesUpAfterRetryFor | A fake scheduler always answers 502. | The call fails with a clear error once the retry budget is spent. |
| NonTransientErrorsAreNotRetried | A fake scheduler answers 404. | One request, immediate error: 4xx is not retried. |
| JobStartTreats409AsRecorded | Start and done answer 409. | Both count as recorded, the lost-reply case. |
| `TestBlackHoleDialFailsFastAndIsRetried` | Dial a reserved, unroutable address (a pod IP nobody owns) with a short dial timeout and a small retry budget. | One connection attempt is bounded by the dial timeout, not the 10 s request timeout, so the retry can re-dial (bug 10, decision 29). |

## Step 6: failure tests under simulator load (`scripts/compose-failures-load.sh`). Log: `step6-compose-failures-load.log`

The stack with the simulator running continuous six-shard rounds at 2x; each failure is injected into a live round.

| Step | Scenario | Proves |
| --- | --- | --- |
| 0 | A baseline round | About 3,400 queries recorded and finished, six builds done at attempt 1, round stamped. |
| 1 | Crash the worker holding a lease mid-build | The build finishes at attempt 2 exactly once; the shard sees leased → queued → leased → done through its reports; every query still finishes; the pool recovers. |
| 2 | Pause the leaseholder past its lease, then unpause | Attempt 2 once; the paused worker logs lost ownership for attempt 1; the round completes. |
| 3 | Restart the scheduler mid-round | Shards retry failed calls and succeed (decision 61); no shard fails; builds stay at attempt 1; the round completes. |
| 4 | Resubmit a build with a different body during the round | 200 `created:false`, the row unchanged. |
| 5 | `docker stop` the simulator | It logs the stop and exits cleanly. |

## Step 6, cross-check under load (Go, `scheduler/crosscheck`, `Load*` tests; need `KGPU_COMPOSE=1`). Log: `step6-crosscheck-load.log`

Their own stack with the simulator running continuous rounds; after each injected failure the round is verified: stamped within 1 ms of its last child finish, every build done with exactly one `completed` line at the recorded attempt, every query finished in order with no `needs_index` query before its build, every shard `stream_done`, the simulator's timeline agreeing.

| Test | Scenario | Proves |
| --- | --- | --- |
| LoadCrashLeaseholderMidBuild | SIGKILL the python process inside the leaseholder of a build of at least 100k vectors, mid-round. | One reap, the next attempt completes once, the round is stamped with every query finished, the pool returns to 2 (roadmap failure test 1). |
| LoadPauseLeaseholderPastLease | `docker pause` 8 s against a 5 s lease, mid-round. | The worker logs `renew rejected` and `lost ownership` for attempt 1 and never completes it; attempt 2 completes once (roadmap failure test 2). |
| LoadSchedulerRestartMidRound | `docker compose restart scheduler` mid-round. | Every build stays at attempt 1; the shards log 12 retried and 12 recovered calls; the round completes (roadmap failure test 3, decision 61). |
| LoadDuplicateSubmitDuringRound | Eight concurrent duplicates into a full live round; a new shard 99; a duplicate after the round finished. | All duplicates 200 `created:false` with the stored decision and the row unchanged; the new shard 409; the late duplicate still 200 (roadmap failure test 4, bug log 6). |
| LoadLargestBuildKilledTwiceReachesAttempt3 | Crash the largest build's leaseholder at attempt 1 and again at attempt 2. | Exactly one `lease expired` per attempt, done at attempt 3, timeline `att=3`; that shard finishes last (20.1 s against 15 s), straggler lag 5.1 s, and the round's duration equals its finish: one slow shard holds up the round. |
| LoadPostgresRestartMidRound | `docker compose restart postgres` mid-round under load. | The round completes correctly and neither worker restarts (decision 48 under real traffic). |
| LoadDockerStopLeaseholderMidRound | `docker stop` the largest build's leaseholder with about 3.8 s left against a 5 s budget. | It logs `finishing current build before exit`, completes attempt 1, claims nothing more, deregisters at once, stays stopped (decisions 37, 49). |
| LoadFinalReportShowsEmptyBacklog | Audit the settled `shard_status` of every round the run produced. | Every round's reports settle to an empty backlog; at stamp time a report can still lag by one interval (decision 62). |

## Step 7: 50 shards and 3 workers (`scripts/compose-round.sh`, `scripts/compose-failures-load.sh`). Logs: `step7-compose-round-50.log`, `step7-failures-load-50.log`

The same scripts as steps 5 and 6 with `N_SHARDS=50 WORKERS=3`.

| Run | Scenario | Proves |
| --- | --- | --- |
| compose-round at 50 | One skewed round, 50 shards, 3 workers, 10x: about 30,000 queries. | All 50 builds done at attempt 1; every query recorded and finished; the round stamped; the GPU queue visible as late build starts and higher fan-out latency; one batched call per shard per poll interval carries it (decision 60). |
| compose-failures-load at 50 | The four failure tests injected into live 50-shard rounds. | The same outcomes as at 6 shards: one completion per build at the recovered attempt, every query finished, shard retries across the scheduler restart, duplicate a no-op. The Phase 1 done-when condition at 50 shards. |

## Step 7, cross-check at 50 shards (Go, `scheduler/crosscheck`, `Load*` tests with `KGPU_XCHECK_SHARDS=50 KGPU_XCHECK_WORKERS=3`; need `KGPU_COMPOSE=1`). Log: `step7-crosscheck-load-50.log`

The step 6 suite parametrised by shard and worker count, plus three scale tests and a stronger round check.

| Test | Scenario | Proves |
| --- | --- | --- |
| verifyRound workload check (every Load test) | The round's workload is regenerated from the round row's scenario and seed and compared with `shard_jobs`. | Every query, including pre-DDL backfill, is recorded in order with matching `needs_index` and duration, arrivals within 8 ms of the schedule (decisions 59, 60). |
| LoadScaleReportsAreBatched | `shard_jobs` sampled per shard every 50 ms through a whole live round. | Each shard's records change at most once per report interval, about 35 records per change; the scheduler logs nothing per query (decisions 60, 63). |
| LoadScaleQueryRunTimeFidelity | Every query's recorded run time compared with its exact scaled duration, across a whole round. | Aggregate within 3 percent, each within the larger of 5 percent and 1 ms, none short (decision 65). Caught bug log 8. |
| LoadScaleRoundSetByGPUQueue | A 50-shard round's build start and finish times against total GPU work. | Builds wait seconds in the queue; the last build finishes within 15 percent of total GPU work divided by workers; shard finish order follows service order; the straggler is among the last three served (decision 67). |

## Phase 2, step 1: the kind cluster (`scripts/kind-check-gpu-node.sh`). Log: `phase2/step1-kind-cluster.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | List nodes, labels and taints after `kind-up`. | One node carries `kgpu.io/gpu=true` and the `NoSchedule` taint; the rest carry neither. |
| 2 | Six plain pause pods with no toleration. | None lands on the GPU node: the taint repels everything that does not tolerate it. |
| 3 | A worker-shaped pod with the toleration and the node selector. | It lands on the GPU node and nowhere else. |
| 4 | A pod with the selector but no toleration. | It stays Pending with no node; the scheduler's event cites the untolerated taint. |

## Phase 2, step 2: Postgres and the scheduler on kind (`scripts/k8s-check-step2.sh`). Log: `phase2/step2-postgres-scheduler.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | Read the scheduler pod's startup log and query `schema_migrations` inside `postgres-0`. | The scheduler connected, applied all four migrations and listened; the claim is bound. |
| 2 | A round and a build through the Service, via port-forward. | The API works inside the cluster; readiness and liveness answer. |
| 3 | Delete the scheduler pod. | The Deployment replaces it under a new name; the round is still there (state lives in Postgres, decision 3). |
| 4 | Delete `postgres-0`. | The StatefulSet recreates it on the same claim with the data intact; the scheduler logs errors during the outage, keeps restart count 0, becomes ready again and serves a new round (decisions 11, 13, 14). |

## Phase 2, steps 1 and 2, cross-check on kind (Go, `scheduler/crosscheck`, `TestKind` subtests; need `KGPU_KIND=1`). Log: `phase2/step12-crosscheck-kind.log`

The agent deploys its own copy of the stack into a `kgpu-xcheck` namespace with two scheduler replicas, and deletes the namespace afterwards.

| Test | Scenario | Proves |
| --- | --- | --- |
| Kind/GPUNodeShape | Read every node's labels, taints and allocatable resources. | Exactly one node carries the GPU label and the NoSchedule taint; the general nodes are untainted (decision 9). |
| Kind/GPUScheduling | Eight plain pods, a worker-shaped pod, a selector-only pod and six toleration-only pods. | Plain pods never land on the GPU node; the worker shape does; selector-only stays Pending citing the taint; a toleration alone is enough to land there (decisions 5, 10). |
| Kind/GPUResourceRequestPending | A worker-shaped pod that also requests `nvidia.com/gpu: 1`. | On kind it stays Pending with "Insufficient nvidia.com/gpu": the one change Phase 5 makes. |
| Kind/TwoReplicasMigrateAndServe | Two scheduler replicas start together on an empty database, four times; once more while the test holds the migration's advisory lock. | Migrations are applied once, never duplicated, never crash a replica; both wait on the lock and neither becomes ready without a schema; both serve the API and see each other's writes (Phase 1 decision 11, Phase 2 decision 2). |
| Kind/DNSAndServices | A throwaway pod resolves names and calls the Services. | `postgres` is headless and resolves to the pod; the scheduler Service resolves by short name and FQDN and answers `/healthz` and `/livez` in-cluster (decision 13). |
| Kind/ProcessKillAndPodReplace | SIGKILL the scheduler process from the node, then force-delete the pod. | The kubelet restarts the container in place (exit 137, restart count 1); the controller replaces a deleted pod under a new name; no data is lost either way. |
| Kind/PostgresOutageReadinessNotLiveness | Postgres scaled to zero for 45 s, then back. | `/healthz` returns 503 and the pod leaves the ready endpoints while `/livez` stays 200 and the restart count stays 0; recovery is automatic (decisions 6, 11, 14). |
| Kind/StatefulSetDeletedClaimSurvives | Delete the Postgres StatefulSet, keep the claim, re-apply. | A new pod reattaches the same claim and volume; rounds and migrations survive; the scheduler recovers without a restart (decisions 3, 13). |
| Kind/OriginalNamespaceUntouched | Snapshot the `kgpu` namespace before and after the run. | The cross-check never touched the implementer's namespace. |

## Phase 2, step 3: the workers on kind (`scripts/k8s-check-step3.sh`). Log: `phase2/step3-workers.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | List the worker pods and the scheduler's pool. | Both run on the GPU node and are registered under their pod names (decisions 1, 5, 17). |
| 2 | A 2M-vector build; probe the leaseholder mid-build from inside. | Done at attempt 1, completed once; `/livez` and `/healthz` answer 200 while the build runs (decision 16). |
| 3 | `kubectl delete pod` the leaseholder mid-build. | SIGTERM: the build is released within a second, the Deployment's replacement lands on the GPU node, and the build finishes at attempt 2 exactly once (decision 37 of Phase 1). |
| 4 | SIGKILL the worker's python process inside its container (tini is PID 1). | The lease expires, the reaper requeues, attempt 2 completes once, the container is restarted in place with restart count 1 (decisions 18, 19). |
| 5 | `kubectl delete pod --grace-period=0 --force` the leaseholder (observation). | The release path still runs and no lease expires: a forced deletion is a graceful stop in disguise, not a crash (decision 19). |

## Phase 2, step 3, cross-check on kind (Go, `scheduler/crosscheck`, `TestKindWorkers` subtests; need `KGPU_KIND=1`). Log: `phase2/step3-crosscheck-kind-workers.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindWorkers/WorkersPinnedNamedAndTini | Inspect the worker pods, their env, their process tree and their probes. | Both on the GPU node; `WORKER_ID` equals the pod name in env and pool; PID 1 is tini with one python child; grace 30 s; both probes 200 (decisions 5, 16 to 18). |
| KindWorkers/ScaleToThreeThenBackToTwo | Scale to 3, submit three builds, scale back to 2. | All three on the GPU node, pool 3, three simultaneous leases; scale-down releases the long build at once and deregisters within a second; each build completed once (Phase 1 decisions 37, 49). |
| KindWorkers/TwoSchedulersReapOnceCompleteOnce | Two scheduler replicas; SIGKILL both leaseholders at the same moment. | One reap line per build across both schedulers, each done at attempt 2 once, containers exit 137 and restart in place (Phase 2 decision 2). |
| KindWorkers/PostgresOutageMidBuild | Postgres scaled to zero for 40 s during a build. | `/healthz` 503 from about 8 s, `/livez` 200 throughout, Ready flips, 0 restarts, the lease is lost at the deadline and the build completes once at attempt 2 after recovery (Phase 1 decision 48). |
| KindWorkers/LivenessFreezeRestartsContainer | SIGSTOP the worker's python. | `/livez` times out; the lease is reaped at about 10 s and the other worker finishes attempt 2; the kubelet restarts the container (decision 20 now shortens the wait). |
| KindWorkers/SIGTERMFinishInsideBudgetReleaseOutside | Delete the leaseholder with about 3 s left, then another with about 7 s left. | Finish path: completed at attempt 1 by the terminating pod; release path: off the pod in 0.2 s, done at attempt 2 elsewhere; no reaps (Phase 1 decision 37). |
| KindWorkers/DrainGPUNode | `kubectl drain` the GPU node mid-build, submit more, uncordon. | Both builds released, pool 0, replacements Pending on the taint, five builds queued; after uncordon all done once (released at attempt 2, new at attempt 1). |
| KindWorkers/BuildLogLinesCarryIDs | Collect every worker and scheduler log line naming a build. | Each carries `attempt` and `round_id` (CLAUDE.md convention). |
| KindWorkers/ImplementerWorkersBack | Snapshot the `kgpu` workers around the drain. | They return to 2 Running and Ready on the GPU node; the node is schedulable. |

## Phase 2, step 3b: GPU cards as a countable resource (`scripts/k8s-check-gpu-cards.sh`). Log: `phase2/step3b-gpu-cards.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | Read every node's allocatable `kgpu.io/gpu-card`. | Each GPU node advertises its cards; general nodes advertise none (decisions 22, 24). |
| 2 | Compare running workers per GPU node with the node's allocated cards; read each pod's resource limit. | Every worker holds exactly one card. |
| 3 | Scale the workers to the card total (4). | Every card is used, both GPU nodes carry workers, none on a general node, the pool shows 4. |
| 4 | Scale to one more than the cards. | The extra pod stays Pending with "Insufficient kgpu.io/gpu-card" and the pool stays 4: Kubernetes, not convention, caps the workers (decision 22). |
| 5 | Scale back to 2. | The cards are freed and the pool follows. |

## Phase 2, step 3b, cross-check on kind (Go, `scheduler/crosscheck`, `TestKindCards` subtests; need `KGPU_KIND=1`). Log: `phase2/step3b-crosscheck-kind-cards.log`

The agent scales the implementer's workers to zero for the run, since cards are cluster-wide, and restores them.

| Test | Scenario | Proves |
| --- | --- | --- |
| KindCards/ReverseOrderStartWaitsForSchema | Apply the worker Deployment first, then Postgres, then the scheduler, holding each stage past the liveness budget. | Workers never crash-loop (restarts 0), stay not ready with liveness 200, log that they wait, and register within a second of the migration (decision 25, bug 9). |
| KindCards/NodeAndPodCardFacts | Read every node's card capacity and allocatable, and every pod's card request and limit. | Both GPU nodes advertise 2 = 2, no other node advertises any, each worker requests exactly one card, nothing else consumes cards (decisions 22, 24). |
| KindCards/FourWorkersFillCardsFifthPending | Scale workers to 4, then 5. | Four fill every card, two per node, pool 4; the fifth stays Pending with "Insufficient kgpu.io/gpu-card" and the pool stays 4. |
| KindCards/NoCardPodIgnoresCap | All cards in use; a GPU-shaped pod with no card request, then one requesting a card. | The cap applies only to pods that request a card: the no-card pod lands on a full node; the requesting non-worker is refused like a worker. |
| KindCards/KilledWorkerKeepsItsCard | Four workers plus a Pending fifth; SIGKILL the python inside one. | The container restarts in place, the node's allocated count never drops, the Pending pod never gets the card. |
| KindCards/TwoCardPodFragmentation | A pod requesting two cards with 3, 2 (one per node) and 1 workers running. | Cards are per node: Pending while no single node has two free, even with two free in total; lands once a node has two. |
| KindCards/CapacityPatchedDownUnderLoad | Patch one node's capacity from 2 to 1 with two workers on it, delete one, restore. | Kubernetes evicts nothing (the node runs over-allocated); the replacement stays Pending until capacity is restored. |
| KindCards/DrainOneGPUNodeHalvesPool | Four workers with four long builds; drain one GPU node, then uncordon. | Exactly that node's two workers are evicted and release; the other two keep building; the pool halves; replacements wait; every build ends done once (decision 24). |

## Phase 2, step 4: shards outside the cluster (`scripts/k8s-check-step4.sh`, via `scripts/k8s-round.sh`). Log: `phase2/step4-shards-outside.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 0 | Read the scheduler Service and list shard pods. | The only door is the NodePort on 30080; no shard pods exist in the cluster (decisions 26, 27). |
| 1 | The simulator on the laptop runs a 6-shard round against 2 workers. | The default configuration completes end to end on Kubernetes from outside: all builds done at attempt 1, every query recorded and finished. |
| 2 | A 50-shard round against 2 workers; the leaseholder of the largest leased build is deleted mid-round. | The round completes, nothing completes twice, the replacement pod joins the round in flight and claims builds; the deleted pod finishes its short build inside its budget (the release path under a real round is in the step 3 cross-check). |
| 3 | A 50-shard round against 4 workers, every card in use. | The pool scales to the card total and the round completes. |
| 4 | A 50-shard round against 2 workers; the scheduler pod is deleted a second into the round. | The replacement serves the round from Postgres and the shards see no retries at all: the shutdown delay and the dial timeout close the 10 s stall (bug 10, decision 29). The deleted pod's last log line shows it serving the delay. |

## Phase 2, step 4, cross-check from the laptop (Go, `scheduler/crosscheck`, `TestKindOutside` subtests; need `KGPU_KIND=1`). Log: `phase2/step4-crosscheck-kind-outside.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindOutside/NodePortIsTheOnlyDoor | Probe the scheduler from the laptop at every address it has, then from inside a pod. | From the laptop only `127.0.0.1:30080` answers; inside the cluster `scheduler:8080` and every node's 30080 do (decision 27). |
| KindOutside/BimodalRoundFromLaptop | A bimodal 6-shard round on 2 workers from the laptop. | Every build done once; every query matches the seeded workload; no shard pods in the cluster (decision 26). |
| KindOutside/SchedulerPodDeletedMidRound | Delete the scheduler pod half a second into a 30-shard round. | The replacement serves the round from Postgres and both rounds complete correctly; the shards' retry stall is measured (found bug 10). |
| KindOutside/TwoSimulatorsAtOnce | Two simulators with different seeds run overlapping rounds. | Neither round's builds or queries leak into the other's. |
| KindOutside/NoWorkersThenScaleUp | Workers at zero: a `MIN_WORKERS=1` and a `MIN_WORKERS=0` simulator; then scale to 2. | The first starts no round, the second leaves its builds queued; both complete after the scale-up. |
| KindOutside/FiftyShardsFourWorkers | A 50-shard `downstream_heavy` round on 4 workers from the laptop. | Every worker completes builds; every query matches the workload. |
| KindOutside/SimulatorOnlyReachesNodePort | Point a simulator at the ClusterIP and at a pod IP. | It cannot connect, creates no round, and retries without limit (decision 30). |

## Phase 2, step 5: metrics and dashboards (`scripts/k8s-check-step5.sh`). Log: `phase2/step5-monitoring.log`

| Step | Scenario | Proves |
| --- | --- | --- |
| 1 | Read the scheduler's `/metrics` through the NodePort. | All nine scheduler metrics are present. |
| 2 | Read each worker's `/metrics` on its probe port from inside the pod. | `build_duration_seconds`, `worker_busy` and `builds_total` are served per worker. |
| 3 | Start a two-round, 50-shard simulator on the laptop and read Prometheus's targets mid-round. | Prometheus scrapes the scheduler, both workers and the shards on the laptop through the Service + Endpoints (decision 32); the laptop target is down when no simulator runs. |
| 4 | After the rounds, query Prometheus. | `round_duration_seconds_count` +2, `placement_decisions_total` +100, the workers' build durations +100, `worker_busy` reached 1, the shards' `queries_total` and query-wait p99 and the build queue wait p99 are there (decision 31). |
| 5 | Grafana's health and search APIs through a port-forward. | Grafana is up and the 17-panel KGPU dashboard is provisioned from the ConfigMap (decision 33). |

## Phase 2, step 5, cross-check of metrics and dashboards (Go, `scheduler/crosscheck`, `TestKindMetrics` subtests; need `KGPU_KIND=1`). Log: `phase2/step5-crosscheck-kind-metrics.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindMetrics/SchedulerExposition | Parse the scheduler's `/metrics` through the NodePort. | All nine families have the roadmap's types; every placement and outcome series exists at zero (decision 35). |
| KindMetrics/WorkerExposition | Read each worker's probe port through the API-server proxy. | The gpu build-duration histogram, the busy gauge and `builds_total` with every outcome are served per worker. |
| KindMetrics/ShardsTargetAndFirstScrape | Start a simulator held before its first round; watch its `/metrics` and the Prometheus target; kill it. | The target goes down to up within 10 s and back; every shard family, including the per-shard queue gauges, exists from the first scrape (decision 35; found the simulator bug). |
| KindMetrics/RoundMetricsMatchSimulator | Three 6-shard rounds; compare scheduler, API, simulator and Prometheus. | Once per round for duration and lag, once per GPU build for queue wait; the sums in real seconds match the API and the simulator; the local build-duration series exists at zero. |
| KindMetrics/RoundObservedOnceByEitherCloser | One round closed by a GET right after the last report, one by the reaper with no GET. | Each is observed exactly once however many GETs and ticks follow (decision 31). |
| KindMetrics/LeaseExpiryAndWorkerBusy | SIGKILL the leaseholder's python during an 8M-vector build. | `lease_expirations_total{queued}` +1 exactly, attempt 2 completes elsewhere, `worker_busy` follows the leaseholder. |
| KindMetrics/ScrapeWiring | Inspect Prometheus's targets and config; run a simulator on another port. | Scheduler, both workers and the laptop's 9091 are scraped every 5 s; another port is not (decision 32). |
| KindMetrics/DashboardPanelsReturnData | Read the dashboard ConfigMap, fetch the dashboard anonymously, run every panel query. | 17 panels in both places; all 25 expressions return data through Prometheus and through Grafana (decision 33). |

## Phase 2, step 6: the failure injections on the dashboard (`scripts/k8s-check-step6.sh`). Log: `phase2/step6-failures-dashboard.log`

Every number is read through Grafana's datasource proxy, so the log records what the dashboard's own datasource served.

| Step | Scenario | Proves |
| --- | --- | --- |
| 0 | Grafana's health and datasource APIs; the simulator's port is free. | The dashboard's Prometheus datasource answers through the proxy, so the runs below are read the way a panel reads them. |
| 1 | A 12-shard round at `FAKE_TIME_SCALE=2` (restored after); the owner of the round's largest build crashed in-container (SIGKILL, decision 19) and its pod deleted while it builds it. | The phase's done-when: `lease_expirations{queued}` +1 on the panel, the killed pod's `worker_busy` series ends and the replacement's begins, the struck build completes at attempt 2, and the round duration still lands (+1). |
| 2 | A 50-shard round; the scheduler pod deleted mid-round. | The replacement closes the round and observes it exactly once from fresh counters (decisions 31, 35); the shards' stall stays ~1 s or no call fails at all (bug 10); `up{job="scheduler"}` shows the old target end and the new one begin. |
| 3 | A 50-shard round; both GPU nodes drained (`kubectl drain`) when it starts, held 20 s after the pool empties, then uncordoned. | The dashboard shows `gpu_workers_live` at 0 while `gpu_queue_depth` holds queued builds, the pool recovers to 2, and the round duration still lands (+1). This is the picture Phase 3's CPU-fallback policies answer. |

## Phase 2, step 6, cross-check of the failure injections on the dashboard (Go, `scheduler/crosscheck`, `TestKindFail` subtests; need `KGPU_KIND=1`). Log: `phase2/step6-crosscheck-kind-failures.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindFail/ForceDeletedLeaseholderCountedOnce | A 6-shard skewed simulator round. The largest build's leaseholder is force-deleted with grace 0; the build row is polled every 100 ms to tell whether the dying worker completed it, released it before `lease_until`, or the lease was reaped; the dashboard is read through Grafana's proxy before and after. | A grace-0 deletion is a graceful stop in disguise (decisions 19, 38): the dying worker finishes or releases if it wins the race to SIGKILL, and the lease-expiry panel moves by exactly what happened (found F1: the spec had claimed it must expire). The round is observed once, the killed pod's `worker_busy` series ends, the replacement's series exist from its first scrape, the gauges return to rest. |
| KindFail/TwoKillsOneBuildThenSchedulerRestart | One 8M-vector build (~28 s). Its owner is force-deleted mid-build, then the next owner too; a second build's worker is crashed in-container with `kill -9`; finally the scheduler pod is deleted. | Two forced kills give two releases (0 expiries) and one completion at attempt 3 (invariants 3, 4); the in-container crash gives exactly +1 expiry and attempt 2; the new scheduler pod's series exist at zero from its first scrape (decision 35); the windowed-increase panel keeps the expiry across the pod replacement (decision 37; found F2: the raw-sum panel fell to 0). |
| KindFail/SchedulerDeletedAtRoundClose | Four one-shard rounds; the scheduler pod is deleted either right after the last report on a finished build or 1 s before the worker completes; the old pod's own `/metrics` is watched through the pod proxy. | Each round shows exactly once on the dashboard — (old pod's max minus before) plus the new pod's count = 1 in 4 of 4 trials — and the replacement closed every one (decisions 29, 31). |
| KindFail/DrainOneGPUNodeAllFourCards | Workers at 4 (every card); a hand-driven round of eight ~6.5 s builds; the node with the most workers drained once 4 builds are leased, held 12 s, uncordoned. | The pool drops to 2 and holds at 2 on the API and the dashboard, never 0 (decision 24). Evicted builds are released, not expired; every build completes once; the round is observed once; the gauges return to rest. |
| KindFail/DrainOneGPUNodeTwoReplicas | Workers at 2, which Kubernetes had co-scheduled on one GPU node; a round of four ~6.5 s builds; that node drained mid-round; the pool polled every 200 ms. | A one-node drain can empty a co-scheduled pool for ~0.5 s and the 5 s scrape can miss it (found F3; decision 38 refines 24: never-empty is asserted only when the workers were spread, recovery otherwise). Releases, single completions and the round observation stay correct. |

## Phase 2, amendment, cross-check of the worker soft spread (Go, `scheduler/crosscheck`, `TestKindSpread` subtests; need `KGPU_KIND=1`). Log: `phase2/amendment-crosscheck-kind-spread.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindSpread/ReplacementLandsOnEmptierNode | With one worker per GPU node, delete one worker, three times on alternating nodes, and watch where the replacement lands. | A fresh replacement goes to the emptier node, restoring one per node (decision 39). |
| KindSpread/ScaleFromZeroOnePerNode | Scale the worker pool from 0 to 2, twice. | Two freshly scheduled workers land one per GPU node (decision 39). |
| KindSpread/ScaleUpFromOneGoesToEmptyNode | Scale from 2 to 1 and back to 2, twice. | The added worker lands on the node with no worker, not next to the survivor (decision 39). |
| KindSpread/CordonedNodePacksThenRebalances | Cordon one GPU node and delete its worker; then uncordon and delete one of the packed pair. | The spread is soft: the replacement packs instead of going Pending; nothing moves on uncordon (spread is scheduling-time only); the next fresh scheduling rebalances (decision 39). |
| KindSpread/FullNodePacks | At 1 replica, fill the other node's two cards with a non-worker pod in another namespace and scale to 2; then free the cards and replace one worker. | A card-full node makes the new worker pack, never Pending; a foreign namespace's pods don't count toward the balance; once cards are free, fresh scheduling rebalances (decision 39). |
| KindSpread/FourReplicasTwoPerNodeFifthPendingForCards | Scale to 4, then 5, then back to 2. | Four replicas sit 2 per node; the 5th stays Pending citing only `Insufficient kgpu.io/gpu-card`, never the spread (decisions 22, 39). |

## Phase 2, amendment: the draining scheduler (Go, `scheduler/api`, needs `TEST_DATABASE_URL`). Log: `phase2/amendment-draining-scheduler.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| TestDrainingServesWithoutStamping | Two complete-but-unstamped one-shard rounds. The first is polled by a healthy pod; the draining flag (what main sets at SIGTERM) is flipped and the second is polled three times, a shard report is posted, and the reaper ticks. | A healthy poll stamps on read (decision 31); a draining poll answers the full, truthful status without stamping, three times; a shard report still writes while draining (bug 10 must not come back); the reaper's next tick stamps and the draining pod serves it (decision 40). |

## Phase 2, amendment, cross-check of the draining scheduler (Go, `scheduler/crosscheck`, `TestKindLameDuck` subtests; need `KGPU_KIND=1`). Log: `phase2/amendment-crosscheck-kind-lameduck.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| KindLameDuck/DyingPodAnswersOpenAndNeverStamps | A round's build is done; the scheduler pod is deleted; after SIGTERM the shard's `stream_done` report goes to the dying pod, which is polled every ~230 ms through the pod proxy until it stops serving. | A draining scheduler records the report, answers the complete-but-unstamped round as open (truthfully, matching Postgres at every poll), never stamps and never observes; the replacement stamps with `finished_at` equal to the build's finish and its round count exactly 1; Prometheus's last scrape of the dying pod equals its final value (decisions 18, 31, 40). |
| KindLameDuck/DyingPodStillRecordsWrites | After SIGTERM, a build submit, a query arrival and a shard report are sent to the dying pod. | Every endpoint except stamp-on-read keeps writing in the 3 s window (201/201/200 with the rows in Postgres), and the replacement serves them — bug 10's stall cannot return through the drain (decision 40, invariant 1). |
| KindLameDuck/ExactlyOnceAcrossReplacements | Six scheduler replacements: rounds completing inside the window, just before the delete, or stamped by the healthy pod and the pod deleted at once. | Each round is observed exactly once summed over dying and new pod, never zero, never twice (decisions 31, 40) — and a round stamped within one scrape interval before a deletion is observed but never scraped, the residual race's measured width. |

## Phase 3, step 1: the policy interface and the five policies (Go, `scheduler/policy`, pure, no database). Log: `phase3/step1-policies.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| TestAlwaysGPU | The baseline's four placements: fits, exactly fits, too big for every worker, empty pool. | Everything queues except what fits no live worker; an empty pool still queues (it may be cold-starting); pure. |
| TestStaticThreshold | Builds above, at and below the threshold; a too-big build above it; two GPU builds of different sizes. | N or more vectors goes to the GPU with priority = vector count (the skewed giant claims first), the memory filter overrides, and the policy never reconsiders. |
| TestReactivePreemption | One shard 20% into a 100 s local build with a blocked backlog, then the same shard at 95%, moved, silent, a photo-finish case, a too-big build, an empty pool, a build with a live speculative copy, and a growing backlog. | Migrating past the margin preempts with priority = the backlog behind the build; sunk cost (progress) holds a nearly-done build; a moved build never moves again (decision 8); no report means no action; the margin stops a 1 s win; a build fitting no live worker — or facing an empty pool — keeps its progress (bug 11); an original with a copy in flight is never preempted; arrivals during the remaining build count. |
| TestReactiveRenegeAndReprioritize | A queued GPU build behind a 50 s queue vs a 10 s queue; one whose non-index backlog makes going local worse; a preempted build whose backlog grew, drifted exactly the delta, and sat unchanged. | Renege compares recoveries, not bare build times, so it never reneges a shard into a worse recovery; a preempted build never reneges back; its priority follows the backlog strictly past the churn delta and is stable across ticks when the backlog is unchanged. |
| TestReactiveSpeculation | An idle worker with an empty queue and two mild local stragglers; then a non-empty queue, a copy in flight, all workers busy, a build that fits no idle worker, and a build being preempted the same tick. | Exactly one speculation, on the worst projected recovery, and never in any of the five no-go cases. |
| TestClairvoyant | No queries; a 20 s GPU queue; a two-query stream replayed on both placements; the skip rule; the memory filter; purity. | The oracle picks the sooner replayed recovery, sets priority to the would-be local finish (15.0 and 13.0 computed by hand), lets a not-ready index query wait without blocking, and never reconsiders. |
| TestByName | Every config name, the threshold argument, nine bad names, and `ByName(p.Name())` for each. | All four deployable policies are reachable by name and every name round-trips (`static_threshold` carries its N); a typo'd sweep arm errors instead of running as the wrong policy; the clairvoyant is not deployable (decision 1a). |

## Phase 3, step 1, cross-check of the policies (Go, `scheduler/crosscheck`, `TestPolicySpec` subtests, pure, no database). Log: `phase3/step1-crosscheck-policies.log`

Written from the docs and `go doc` alone; found bug 11 and the F2–F6 / probe findings triaged in the design doc.

| Test | Scenario | Proves |
| --- | --- | --- |
| PolicySpec/Purity_RepeatedDecideIdentical | Each of the five policies decides the same input 51 times. | Decide is deterministic (invariant 8). |
| PolicySpec/Purity_RepeatedReconsiderIdentical | Each policy reconsiders a busy world 51 times. | Reconsider is deterministic (invariant 8). |
| PolicySpec/Purity_ReconsiderDoesNotMutateInput | Input slices compared before and after Reconsider. | No side effects on the caller's state. |
| PolicySpec/Purity_ReconsiderPermutationInvariant | Builds and workers in 10 rotations and reversals. | The action set does not depend on input order. |
| PolicySpec/EmptyWorld_NoActionsNoPanic | No builds, then idle workers only. | Every policy returns no actions and does not panic. |
| PolicySpec/MemoryFilter_AllPolicies | A build one byte too big, and one exactly the largest worker's size, for all five policies. | Too big goes local; exactly fitting fits. |
| PolicySpec/MemoryFilter_EmptyPoolStillQueues | Zero live workers and a 64 GiB build. | Deployable policies still queue at submit (the cold-start rule); the oracle does the same. |
| PolicySpec/AlwaysGPU_QueuesEverythingFCFS | Tiny to huge builds, including one where the GPU is much slower. | Always gpu at priority 0; never reconsiders. |
| PolicySpec/StaticThreshold_BoundaryPriorityNeverReconsiders | N=1000 with 999, 1000, 1001 and 0 vectors; two sizes; a busy world. | The `>=` boundary, priority = vector count, no reconsider. |
| PolicySpec/ByName_AcceptsDocumentedNames | The five documented spellings. | They resolve; the bare threshold uses the default; `:500` splits at 500. |
| PolicySpec/ByName_RejectsBadNamesAndTheOracle | Malformed names, suffixes on parameterless policies, `clairvoyant`. | Config errors are rejected (found F5: parsing was lenient) and the oracle is unreachable (decision 1a). |
| PolicySpec/ByName_NameRoundTrips | `ByName(p.Name())` for each name. | A logged name reproduces the policy (found F6: the threshold was lost). |
| PolicySpec/Reactive_DecideLikeThresholdPriorityIsLocalSeconds | The ThresholdN boundary; a 37.5 s build. | Submit behaves like a threshold with priority = local seconds (decision 10). |
| PolicySpec/Reactive_PreemptArithmeticAndPriority | A 100 s build at 50% with a 10-query backlog, by hand: stay 60, migrate 10. | Preempts, with priority = the 10 s backlog behind it (decision 10 as corrected). |
| PolicySpec/Reactive_PreemptPriorityIsBacklogOnly | The same build at 20% progress, where the old formulas disagree. | Preempt priority is the backlog alone, whatever the progress (found F3: code and doc each said a third thing). |
| PolicySpec/Reactive_RenegeComparisonProbe | A queued build with a non-index backlog, local 10 s vs GPU 17 s. | Renege counts the backlog the local build would block (found the renege/preempt cost-model asymmetry; recovery 20 vs 17 now stays queued). |
| PolicySpec/Reactive_MarginBoundaryDefault2s | Wins by exactly 2, by 2.5, by 1 and by -1; custom margins. | Preempt must win by more than MarginS (default 2 s). |
| PolicySpec/Reactive_SunkCostViaBuildProgress | The same build at progress 0, 0.5, 0.75 and 0.9. | build_progress carries the sunk cost (decision 1b). |
| PolicySpec/Reactive_NonIndexBacklogFavoursMigration | The same backlog, all needing the index vs none. | Non-index queries drain as soon as the CPU frees. |
| PolicySpec/Reactive_QueueWaitOfLocalBuildEntersMigrate | A hurting build with a 100 s GPU wait. | The queue wait enters the migrate branch (G2 pinned). |
| PolicySpec/Reactive_NoActionWithoutValidReport | A hurting build, and a preempted queued build, with Valid=false. | No preempt or reprioritize without a report. |
| PolicySpec/Reactive_MemoryAwarePreempt | A hurting local build that fits no live worker. | Never preempted into a queue no worker can claim (found F1 = bug 11). |
| PolicySpec/Reactive_MovedAtMostOnce_RenegedNeverPreempted | A Moved local build with a huge migrate win, plus a control. | A reneged build is never preempted (decision 8). |
| PolicySpec/Reactive_MovedAtMostOnce_PreemptedNeverReneges | A Moved queued build with a 1000 s wait, plus a control. | A preempted build never reneges (decision 8). |
| PolicySpec/Reactive_RenegeOnlyQueuedPastMargin | Waits around the 2 s margin; a leased build with hopeless numbers. | Renege only past the margin, and only for queued builds (decision 4). |
| PolicySpec/Reactive_RenegeOfSpeculativeCopyProbe | A speculative copy stuck in the queue while its original runs. | The copy is not reneged, and the original is no longer preempted (the probe that found it). |
| PolicySpec/Reactive_ReprioritizeOnlyPastDelta | Stored priority at target +0, ±0.5, ±1 and ±1.5; an unmoved build. | Reprioritize strictly past 1 s, and only for preempted builds (found F2: fired at exactly ±1). |
| PolicySpec/Reactive_PreemptThenNextTickDoesNotChurn | Preempt, then the next tick with an unchanged backlog. | The priority stays put (found F4: it used to jump by the whole build time). |
| PolicySpec/Decision10_PrioritiesComparableWithinReactive | A 10 s submit vs a 10 s backlog preemption; two backlogs. | One currency of seconds; the shard hurting most claims first. |
| PolicySpec/Spec_SpeculatesWorstOncePerTick | Two or three mild stragglers in both orders. | Exactly one speculation, on the worst projected recovery; plain reactive never speculates. |
| PolicySpec/Spec_NoGoCases | Queue non-empty, a copy queued or leased, all workers busy, no workers. | No speculation in any of them. |
| PolicySpec/Spec_MemoryAware | The worst straggler doesn't fit the only idle worker. | It is not speculated; a fitting one is chosen instead. |
| PolicySpec/Spec_NeverTargetsBuildPreemptedSameTick | A hurting build that is also the worst recovery. | Preempted, never speculated in the same tick. |
| PolicySpec/Spec_StillReactive | A busy world with no idle worker. | reactive_spec acts and decides exactly like reactive. |
| PolicySpec/Clairvoyant_ExactReplayWithSkipRule | Three hand-replayed streams; one flips on the skip rule (7 vs 11). | Exact replay, the sooner placement wins, priority = would-be local recovery. |
| PolicySpec/Clairvoyant_QueueWaitAndLPT | Queue wait 0 vs 20; local finish 10 vs 40. | The queue wait enters the GPU finish; longest-first priority. |
| PolicySpec/Clairvoyant_LateArrivalsAfterLocalBuild | Queries arriving after the build; an equal-recovery tie. | Queries after the build run on arrival; a tie goes local (G4 pinned). |
| PolicySpec/Clairvoyant_ShardWithNoFuture | An empty stream, a missing map entry, nil futures. | All three treated alike, as no queries. |
| PolicySpec/Clairvoyant_UnsortedStreamSameAsSorted | A permuted query stream. | Same decision; the caller's slice is untouched. |
| PolicySpec/Clairvoyant_NotDeployable | ByName of the oracle's name; its Reconsider. | Unreachable from config (decision 1a); no reconsider actions. |

## Phase 3, step 2: the GPU finish estimator, queue replay (Go, `scheduler/policy` pure plus `scheduler/store` and `scheduler/api` with the database). Log: `phase3/step2-estimator.log`

| Test | Scenario | Proves |
| --- | --- | --- |
| TestReplayRoadmapExample | The roadmap's worked example: w1 busy 4 s, w2 free; a 6 s job outranking a 3 s one. | j1 starts at 0 and finishes at 6, j2 at 4 and 7 — list scheduling over remaining times, by hand. |
| TestReplayTransferSharingOneNode | Two idle workers run two 10 s transfers, first on one node, then on two. | On one node the link is shared (both transfers end at 20, builds at 25); on two nodes they run at full rate (done at 15) — decision 12's fluid sharing. |
| TestReplayTransferSharingStaggered | One worker claims at 0, its node-mate at 4; both transfer 10 s. | The rates change as transfers join and leave: j1 done at 21, j2 at 25, replayed by hand in the narration. |
| TestReplayClaimIsMemoryAware | A small idle worker, a big busy one, a high-priority big job and a low-priority small one. | The small worker claims the small job at once instead of waiting behind a job it could never take — the claim's own per-worker memory filter (decision 12). |
| TestReplayFitsNowhereIsInfinite | A job bigger than every worker, beside a normal one; then zero workers. | Fits-nowhere estimates +Inf without blocking the rest (the truth bug 11 hid); an empty pool is +Inf for everything. |
| TestReplayFIFOWithinEqualPriority | Two priority-0 jobs enqueued in order. | Equal priority falls back to enqueued order, the claim's tiebreak. |
| TestReplayBusyWorkerIsOpaque | A busy node-mate while a 10 s transfer runs. | BusyRemainingS is one lump: an in-flight build's own transfer does not contend (decision 12's pinned approximation). |
| TestReplayUnknownNodesDoNotContend | Two workers with empty Node, two 10 s transfers. | Unknown nodes get private links: "unknown" never invents contention (decision 11). |
| TestReplayDeterministicUnderPermutation | Workers and queue permuted. | The replay is a pure function of the set, not the order (invariant 8). |
| TestNewJobWaitJoinsAtTheBack | One worker, a 10 s queue, then an empty pool. | The submit-time scalar joins at priority 0 behind the queue (decision 13); an empty pool says +Inf and the host applies the cold-start rule. |
| TestGetQueueSnapshot (store) | Two workers heartbeat with nodes; three queued builds; one claimed. | The snapshot returns workers with nodes, the queue in claim order (priority DESC), and the leased build with its elapsed-since-claim. |
| TestHeartbeatAndPoolState (store, extended) | Heartbeats now carry a node. | `workers.node` round-trips through Heartbeat and ListWorkers (decision 11). |
| TestSubmitQueueWaitEstimate (api) | Four submits against one worker: empty queue, one ahead, one leased plus one queued, and a build too big for the pool. | `gpu_queue_wait_ms` reads 0, one GPU total, between one and two totals (the lease's elapsed time already counts), and -1 for fits-nowhere. |
| TestSubmitQueueWaitColdStart (api) | A submit with no workers at all. | The wait stays 0 and always_gpu still queues: the cold-start rule is the host's (decision 13, G1). |

## Phase 3, step 2, cross-check of the estimator (Go, `scheduler/crosscheck`, `TestEstimatorSpec` pure and `TestEstimatorDB` with the database). Log: `phase3/step2-crosscheck-estimator.log`

Written from the docs and `go doc` alone; found F1 (the duplicate response's `gpu_queue_wait_ms`) and the G1–G7 gaps triaged in the design doc.

| Test | Scenario | Proves |
| --- | --- | --- |
| EstimatorSpec/RoadmapExample | w1 busy 4 s, w2 free, jobs of 6 s and 3 s, then a new build. | The roadmap's worked example: 0–6 on w2, 4–7 on w1, the next build waits 6. |
| EstimatorSpec/TwoTransfersOneNodeShareTheLink | Two idle workers on one node take two 10 s transfers. | Each runs at half the link; both finish at 20 (decision 12's own example). |
| EstimatorSpec/ThreeTransfersOneNodeEachAtOneThird | Three workers on one node, three 6 s transfers. | k transfers each run at 1/k: all finish at 18. |
| EstimatorSpec/UnequalConcurrentTransfersRateRisesWhenOneEnds | Transfers of 4 s and 10 s start together on one node. | The short one ends at 8; the long one speeds up and ends at 14; a build does not hold the link. |
| EstimatorSpec/StaggeredJoinSlowsTheEarlierTransfer | A second worker on the node frees at 4 s and joins mid-transfer. | The earlier transfer slows from the join (16), the later ends at 20. |
| EstimatorSpec/SeparateNodesRunAtFullRate | Two workers on different nodes. | Sharing is per node; each transfer runs at full rate. |
| EstimatorSpec/EmptyNodeIsAPrivateLink | Workers with unknown node, alone and mixed with a shared node. | An unknown node never invents contention (decision 11). |
| EstimatorSpec/BusyWorkerIsOpaqueNoContention | A busy worker and an idle one on the same node. | An in-flight build's transfer does not slow a new claim. |
| EstimatorSpec/TransferOfLaterJobOnSameWorkerSharesWithNeighbour | A worker finishes and claims a second job while its neighbour is still transferring. | The second claim shares the link from its claim moment. |
| EstimatorSpec/SmallIdleWorkerDoesNotWaitBehindBigJob | A big job queued first, only a small worker idle. | The small worker takes the job it fits; the big job waits for the big worker. |
| EstimatorSpec/PriorityOrderUnderPerWorkerMemoryFilter | Three jobs, a small idle worker and a big busy one. | Each free worker takes the highest-priority job that fits its own memory. |
| EstimatorSpec/PriorityDescThenSeqTiebreak | Jobs given out of order with a priority tie. | Order is priority descending, then enqueue order. |
| EstimatorSpec/SimultaneousFreeResolvedByWorkerID | Two idle workers of different memory and a job only one can take. | Ties between workers are resolved one coherent way (ascending ID, G2), the same for any input order. |
| EstimatorSpec/MemoryExactlyEqualFits | Jobs needing exactly the worker's memory and one byte more. | Equal fits; one byte over is +Inf. |
| EstimatorSpec/FitsNowhereIsInfAndDoesNotBlock | A huge job at top priority ahead of small ones. | The huge job is +Inf and the rest run as if it were absent. |
| EstimatorSpec/EmptyPoolEveryJobInf | A queue and no workers. | Every job is +Inf and still has an entry. |
| EstimatorSpec/EmptyQueueEmptyMap | A worker and no queue. | The result is empty. |
| EstimatorSpec/EveryBuildIDGetsAnEstimate | A queue that includes a job fitting nowhere. | Every queued build gets an estimate. |
| EstimatorSpec/ZeroTransferAndZeroBuild | A zero-cost job ahead of a normal one on one worker. | It finishes at its start and frees the worker at once. |
| EstimatorSpec/ZeroTransferDoesNotSlowANeighbour | A no-transfer job and a 10 s transfer on one node. | A job with no transfer takes no share of the link (G7 pinned). |
| EstimatorSpec/ManyJobsOneWorkerSerialSum | Four equal jobs on one worker busy 2 s. | Starts and finishes are the running sum. |
| EstimatorSpec/BusyRemainingTie | Two workers freeing at the same moment, on separate and on shared nodes. | Both claim at once; on a shared node they split the link. |
| EstimatorSpec/DeterministicAcrossRepeatedCalls | The same rich input 20 times. | Identical answers (invariant 8). |
| EstimatorSpec/PermutedInputsSameAnswer | 24 reorderings of the workers and the queue. | Input order never changes the answer, for ReplayQueue and NewJobWait. |
| EstimatorSpec/InputsNotMutated | Both functions called on copies. | The caller's slices are unchanged. |
| EstimatorSpec/RichScenarioByHand | Four workers of mixed memory and node, six jobs. | Every start and finish matches a hand replay of the claim and link sharing. |
| EstimatorSpec/NewJobWaitJoinsAtPriorityZeroBehindEveryone | A new job passed in with a huge priority. | It still joins at the back at priority 0 (decision 13). |
| EstimatorSpec/NewJobWaitExactForFCFS | An all-priority-0 queue on two workers sharing a node. | NewJobWait equals the full replay's start for the appended job. |
| EstimatorSpec/NewJobWaitEmptyPoolIsInf | No workers. | +Inf; the host owns the cold-start 0. |
| EstimatorSpec/NewJobWaitFitsNowhereIsInf | A new job larger than the only worker. | +Inf. |
| EstimatorSpec/NewJobWaitIdlePoolIsZeroBusyPoolIsRemaining | Empty queue with an idle pool, then with a busy one. | 0 when idle, otherwise the earliest free time. |
| EstimatorSpec/NewJobWaitRespectsMemoryFilter | A big new job, a small idle worker and a big busy one. | It waits for the worker it fits, not the idle one. |
| EstimatorSpec/NewJobWaitBehindNegativePriorityProbe | A queued job at priority -5. | Records that the new job goes ahead of negative priorities (G1 probe). |
| EstimatorDB/HeartbeatRoundTripsNode | Workers heartbeat with a node and with "", then one changes node. | Node is stored, read back by ListWorkers, the snapshot and GET /workers, and updated by the latest heartbeat (G4). |
| EstimatorDB/SnapshotExcludesStaleWorkers | One worker's heartbeat is older than maxAge. | The snapshot lists only live workers. |
| EstimatorDB/SnapshotQueuedInClaimOrderAndLeased | Five builds of mixed priority and placement, then two claims. | Queued comes in claim order with local builds left out, the real claim agrees, leased rows carry owner and time since claim. |
| EstimatorDB/APIQueueWaitEmptyPoolIsZero | Submit with no workers. | `gpu_queue_wait_ms` is present and 0. |
| EstimatorDB/APIQueueWaitFitsNowhereIsMinusOne | Submit larger than the only worker's memory. | `gpu_queue_wait_ms` is -1. |
| EstimatorDB/APIQueueWaitSameNodeSharesLink | Two idle workers on one node, three equal submits. | The third waits 2T+B: link sharing reaches the API. |
| EstimatorDB/APIQueueWaitSeparateNodesFullRate | The same on separate nodes. | The third waits T+B. |
| EstimatorDB/APIQueueWaitUnknownNodesPrivateLinks | The same with unknown nodes. | The third waits T+B: unknown nodes do not contend. |
| EstimatorDB/APIQueueWaitCountsLeasedRemaining | One worker holds a lease, then two more submits. | A leased build counts as its GPU total minus time since claim. |
| EstimatorDB/APIDuplicateOmitsQueueWait | Resubmit an existing build. | The duplicate response leaves out `gpu_queue_wait_ms` (found F1: a plain int64 served 0 for both meanings). |
