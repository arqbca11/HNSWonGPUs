# Bug log

Significant bugs found in this project, written up so the cause is understood, not just the fix. One entry per bug: what was observed, what was actually wrong, how it was fixed, and what it teaches. Newest first.

Entries are added when a bug changes how we think about the system, is subtle enough that it would be made again, or was found by a method worth repeating (an independent cross-check, a failure test). Typos and one-line slips are not logged.

---

## 14. The simulator's shared link gave finishing transfers' bandwidth away mid-interval — and could skip a whole GPU build

**Found** by the Phase 3 steps 4–6 cross-check agent, from hand numbers alone (`NodeLinkSharingThreeModes` and three sibling probes): with two builds sharing one node's link, the second finished 0.208 s early — exactly half the event grid's last interval — and when both started together, one "finished" at 2× the transfer time **without ever running its GPU build**. The agent inferred the mechanism from the numbers before anyone read the code. Fixed in this commit, `experiments/sim/sim.go`. **Related:** entry 12 (the same lesson — a phase test written against a state *value* instead of the state's *meaning*), entry 8 (also a per-interval accounting error that only showed at a rate boundary).

**Symptom.** Under `ShareNodeLinks` with two or more workers per node, transfers finished early by half an interval per rate change; in the worst alignment a transfer's remaining time went negative, slipped past the transfer-finish test, and fell through to "build finished" — a build marked done at the moment its bytes arrived, its GPU phase never run, its worker freed to take the next job. Every `workers_per_node ≥ 2` sweep arm was wrong; the committed demo (workers per node = 1) was untouched.

**Cause, two layers.** (1) The clock advance iterated workers in order, computing each one's share-of-link *during* the mutation: when an earlier worker's remaining hit zero in the loop, later node-mates saw k−1 sharers for the same elapsed interval and drained at the full rate — stealing exactly `dt·(1/1 − 1/2)` of progress. The share is constant *between* events by construction; the code let one event's own processing change it mid-interval. (2) The transfer-finish test was `remaining > −ε && no deadline set`: a value test. The overshoot from (1) made remaining more negative than ε, the value test failed, and control fell through to the build-completion branch whose deadline (the zero value, never set) read as "due at t=0".

**Fix.** Snapshot every worker's share before mutating any remaining; and make the phase test mean what it says — any remaining at or below zero with no build deadline set is a transfer that just ended and must enter its build, unconditionally. The strengthened implementer's test now asserts every *shard's* finish, not just the round's: the original asserted only `RoundS`, which the one correctly-simulated build set, hiding its skipped twin — the agent guessed that blind spot exactly.

**In production.** Not production code — but it is the measurement instrument. The workers-per-node sweep exists to decide whether a second card on one machine adds a second card's throughput (Phase 2 decision 23); this bug would have answered that question with manufactured evidence: sharing *speeding builds up*, the opposite of the physics being modeled.

**Lesson.** Twice in one phase now (entry 12): encode the *meaning* of a state, not a value that currently correlates with it. And a fluid model's invariant ("the share is constant between events") must be enforced by the code's shape — snapshot, then mutate — not by hoping the iteration order is kind. Cross-checking an instrument with hand arithmetic before trusting its measurements is the cheapest audit this project does.

## 13. Two anti-flapping rules collided and stranded a preempted build the pool could no longer hold

**Found** by the Phase 3 step 3 cross-check agent (`PreemptedBuildNeverReneged_EvenWhenItFitsNowhere`, a probe pitting decision 8 against decision 19). The test passed — the code followed the spec — and the finding (F3) was that the spec itself builds a hang. Owner decided (decision 20); fixed in this commit, `scheduler/policy/reactive.go` and `scheduler/store/store.go`. **Related:** entry 11 (the identical end state — a queued build no claim will ever take, no lease, no reaper, no retry budget, the round hung — reached this time not by a missing filter but by two *correct* rules composed; entry 11's own fix created the move that this entry's rule then refused to undo), entry 12 (the same step's other blind spot in a placement-change guard).

**Symptom.** A hurting local build is preempted onto the one worker big enough for it (move #1, decision 8's quota spent). That worker dies or drains — the shrinking_pool scenario. The build now fits no live worker: the claim's memory filter skips it forever, and the reconsider's renege — decision 19's designated escape, "exactly how a stuck too-big build escapes the queue" — is refused because the build already moved once. It sits `queued` until a big worker happens to return; the round hangs on it.

**Cause.** Decision 8 (placement changes at most once, the anti-flapping rule) and decision 19 (fits-nowhere reneges out) were each argued correctly and never confronted: the flapping rule was written when every move was a *cost judgment* that a later tick might reverse, and the escape was written as if the stranded build's history didn't exist. The collision needs three steps — preempt, then shrink, then the refusal — which no single decision's "why" column walks through.

**Fix (decision 20, the impossibility escape).** A moved build may still renege when its memory fits no live worker (`QueueWaitS` +Inf from the queue replay). Impossibility is not a cost judgment, so the anti-flapping rationale does not apply to it; and no loop can form, because preemption still strictly requires a never-moved build — the longest possible lifetime is submit → preempt → escape-renege → local forever. The store's `RenegeToLocal` drops its moved guard (the policy enforces the judgment rule; the store keeps state and speculative guards), `PreemptToGPU` keeps its moved guard absolutely, and on an empty pool the escape stays shut (decision 19's cold-start clause).

**In production.** This is the shrinking_pool scenario crossed with the bimodal one: big builds are preempted onto the few big cards, and big cards are exactly what maintenance drains. One drain after one preemption and the round waits for hardware on an operator's timetable, with the scheduler reporting nothing wrong — every row is in a legal state.

**Lesson.** Two guards that are each right can compose into a deadlock; the question "what undoes this state if the world changes?" has to be asked per *state*, not per rule. And a cross-check test that passes can still be the most important finding of its round — this one pinned the spec's own behaviour and handed the conflict to the owner instead of silently picking a side.

## 12. A preempt in the settle window threw away a won speculation race

**Found** by the Phase 3 step 3 cross-check agent (`PreemptWhileCopyDoneButUnresolvedKeepsTheWin`, a deliberate probe of the gap between a copy finishing and `ResolveSpeculations` noticing). Fixed in this commit, `scheduler/store/store.go`. **Related:** entry 11 (the same family: a placement-change site whose guard had a blind spot; there the memory filter, here the definition of "the copy is racing"), entry 5 (work discarded because two actors disagreed about who owns a result).

**Symptom.** A shard's local build is racing its speculative GPU copy. The copy completes — the index exists. Before the next reconsider tick settles the race, a preempt of the primary is accepted: the primary becomes `gpu/queued`, `ResolveSpeculations` then finds no primary still running locally and returns nothing, and the finished index is simply forgotten. The build is claimed and built a **third** time, and the shard — whose backlog was the reason for preempting — waits a whole extra GPU build for an index that already existed.

**Cause.** The preempt guard refused a primary whose copy was "in flight", implemented as `state IN ('queued', 'leased')`. A copy in state `done` is neither — but it is not *gone*, it is a race already won that nobody has settled yet. "Live copy" was never defined to include done-but-unsettled; the guard encoded the intuition (a copy still running) instead of the invariant (the race has a possible or actual winner). The window is real on one scheduler — a copy can complete between a tick's resolve step and its snapshot read — and wider with two.

**Fix.** The guard is now `state <> 'failed'`: only a failed copy — a race with no possible winner left — frees the primary to move. In the settle window the preempt is refused, and the next tick's resolve marks the primary done from the copy. The policy-level refusal (step 1's "never preempt a build whose copy is in flight") was already there; this is the store making it unconditional, with the correct definition.

**In production.** Speculation exists for the shards hurting most; those are exactly the shards the reactive policy also wants to preempt. Every GPU-won race whose primary crossed the preempt threshold in the same tick window would silently double the shard's wait and burn a third build's worth of GPU — the policy's two tools cancelling each other.

**Lesson.** A guard that names states ("queued or leased") rots the moment a new state appears in the lifecycle; a guard that names the *meaning* ("any copy that could still win, or already has") survives. When a check exists in two layers (policy and store), the store's version is the one that must get the meaning right, because it is the one that runs during the race.

## 11. Preemption ignored the memory filter: a hurting build could be parked in a queue nobody can claim

**Found** by the Phase 3 step 1 cross-check agent (`Reactive_MemoryAwarePreempt`): a local build one byte too big for every live worker, with a convincing backlog, was preempted. Fixed in [`86d7099`](https://github.com/arqbca11/HNSWonGPUs/commit/86d7099), `scheduler/policy/reactive.go`. **Related:** entry 5 (the other build that could never finish and what the retry budget does about leases — but a *queued* unclaimable build has no lease to expire, so no reaper or budget ever touches it), entry 1 (also the cross-check's first run against a brand-new piece of the system).

**Symptom.** Pure policy level, caught before any deployment. The reactive policy's `reconsider`, given a local build whose modeled memory exceeded every live worker's capacity (or an empty pool) and a backlog that made migrating look good, returned a preempt action. Had the host applied it: the shard's CPU progress destroyed, the build placed in the GPU queue where `FOR UPDATE SKIP LOCKED` would skip it forever — no lease, no expiry, no reaper, no retry budget — and the round would never complete.

**Cause.** The memory filter ("a build whose modeled memory need exceeds every live worker's capacity must build locally") was implemented at the two *obvious* placement sites, submit (`Decide`) and speculation, and forgotten at the third: preemption. The filter lived as a shared helper for `Decide`'s input shape, and `reconsider`'s different input shape (workers as a slice, not a summarized pool) meant applying it there needed two lines of aggregation that nothing forced anyone to write.

**Fix.** `Reconsider` computes the largest live worker's capacity once per tick and never preempts a build that would fit no one — including when the pool is empty, which is deliberately *stricter* than submit's cold-start rule: a submit that queues on an empty pool has lost nothing if the pool stays cold, but a preemption has destroyed real CPU progress to wait for hardware that may not come.

**In production.** The bimodal and production-size scenarios make over-sized builds routine (10M × 1536 vectors need ~61 GB; 24 GB cards hold 5M). One such shard with a busy query stream would be preempted on the first reconsider tick, and the whole DDL round would hang on a build sitting `queued` forever — the exact "one slow shard holds up the round" failure the scheduler exists to prevent, manufactured by the scheduler itself.

**Lesson.** A constraint that guards a *decision* must be enforced at every site that makes that decision, and placement changes are decisions: the invariant is "no placement may select the GPU for a build that fits no live worker", not "submit checks the memory". When a rule lives in a helper shaped for one call site, every new call site is a chance to forget it — better to make the rule a property of the action (preempt/renege/speculate all route through one placement check) than of the caller. The cross-check found it by attacking the spec's words ("the memory filter is shared") at a site the implementer's own tests never exercised.

## 10. Deleting the scheduler pod stalled every shard for 10 s, though the replacement was ready in 1 s

**Found** by the step 4 cross-check agent's scheduler-deletion test (36 retry warnings, the last 10.5 s after the delete; the round 3.4 s slower than its twin). Reproduced with `KILL_SCHEDULER=1 scripts/k8s-round.sh`: 56 warnings, 10.5 s stall, replacement ready in 0.9 s. Fixed in [`56e4dbe`](https://github.com/arqbca11/HNSWonGPUs/commit/56e4dbe), `scheduler/cmd/scheduler/main.go`, `shard/client/client.go`. **Related:** entry 7 and entry 9 (the same family: a dependency vanishing for a moment; those two were handled by retrying, this one is about *how long one failed attempt takes* before the retry can happen), entry 8 (another case where a small fixed delay per operation was multiplied by many operations into a visible slowdown).

**Symptom.** `kubectl delete pod` on the scheduler during a 50-shard round. Kubernetes replaced the pod and it was ready within a second; the NodePort answered again within 1.2 s. Yet the shards logged `scheduler call failed, retrying` for ten seconds: a few `EOF`s at once, then dozens of client timeouts, the last 10.5 s after the deletion. Every build and query was correct; the round was simply three to five seconds longer than it should have been.

**Why it matters.** The whole point of running the scheduler on Kubernetes is that losing its pod costs about a second. A ten-second stall per deletion turns every rolling update, node drain and crash into a visible pause for fifty database shards, and the stall is invisible to anyone watching the cluster: pod ready, Service healthy, probes green. In production it shows up as a step in every shard's report latency on every deploy, with nothing in Kubernetes to blame.

**Cause.** Two races, one each side. When a pod is deleted, Kubernetes does two things at the same moment: it removes the pod from the Service's endpoints, and it sends the container SIGTERM. The endpoint removal has to reach the forwarding rules on every node, which takes up to a second or so. The scheduler, on SIGTERM, called `Shutdown` immediately and exited within milliseconds, so its IP address disappeared while nodes were still forwarding new connections to it. A connection to an address nobody owns gets no reply at all, not even a refusal; the client's SYN is retransmitted into silence. The shard client had a 10 s overall request timeout and no separate connection timeout, so each such connection sat for the full 10 s before the retry could re-dial, and by then the forwarding rules had long since pointed at the replacement. The six `EOF`s were the healthy case: idle keep-alive connections the scheduler closed properly on shutdown, retried at once.

**Fix (decision 29).** Scheduler: on SIGTERM it logs `shutdown requested; serving for the shutdown delay while endpoints update` and keeps serving for `SHUTDOWN_DELAY` (default 3 s) before draining. That is the in-process form of the standard `preStop: sleep` hook, used because the image has no shell. The readiness probe is irrelevant here; the endpoint is already gone. Client: a 2 s dial timeout (`DialTimeout`) separate from the 10 s request timeout, so a black-holed connection fails fast and the retry re-dials. After both: zero retry warnings on the same deletion, the round 7.4 s against 12.5 s. `TestBlackHoleDialFailsFastAndIsRetried` dials a reserved, unroutable address and shows an attempt is bounded by the dial timeout.

**Lesson.** "The pod was replaced in a second" is a statement about the control plane; the data plane has its own clocks. Endpoint propagation is eventually consistent, so a server must outlive its own removal from the Service by that propagation time; this is why nearly every production Deployment carries a `preStop` sleep. And a client's request timeout is not a connection timeout: one bounds a slow reply, the other bounds silence, and the two should differ by an order of magnitude.

---

## 9. Workers crashed on a fresh cluster: they connected before the schema existed

**Found** on the first deploy to a rebuilt kind cluster (Phase 2 step 3b). Fixed in [`2f501d0`](https://github.com/arqbca11/HNSWonGPUs/commit/2f501d0), `worker/kgpu_worker/worker.py`. **Related:** entry 7 (same class: a dependency not being there at a moment the code assumed it was; that one was the connection, this one is the schema), entry 5 (crash while an error could have been waited out).

**Symptom.** Both worker pods went to `CrashLoopBackOff` within seconds of the stack being applied, each with one restart, then came up fine. The previous container's log ended in `psycopg.errors.UndefinedTable: relation "workers" does not exist` from the first heartbeat's `INSERT INTO workers`.

**Why it matters.** Kubernetes restarted them and nothing was lost, which is why this was never seen in Compose and would be easy to wave away. But a crash loop on every fresh deploy is noise that hides real failures, the restart backoff doubles each time so a slow migration leaves workers out for minutes, and on a real GPU fleet a restart can mean re-initialising the device. In production this looks like a wave of worker restarts after every new environment or every migration that adds a table the worker touches, timed exactly to the deploy.

**Cause.** Compose started the worker only after the scheduler reported healthy (`depends_on: condition: service_healthy`), and the scheduler is healthy only after it has applied the migrations. That ordering was doing load-bearing work nobody had written down. Kubernetes has no startup ordering between Deployments: every pod starts when it can, and the worker connected to a Postgres that was up but empty. The worker's connect path already retried while Postgres was *down* (entry 7), but a missing table is a `ProgrammingError`, not a connection error, so it fell through to the crash.

**Fix (decision 25).** Registration treats `UndefinedTable` as "not yet": it logs `schema not ready, waiting for the scheduler to migrate` and retries with backoff up to 5 s, forever. Readiness stays 503 meanwhile, because no heartbeat has succeeded, so the pod is visibly not ready rather than crash-looping. The test `test_waits_for_schema_at_startup` hides the `workers` table, starts a worker, checks it is alive and unregistered after 1.5 s, restores the table and expects registration on the next retry.

**Lesson.** When a platform provides ordering (Compose `depends_on`, systemd `After=`), the code acquires hidden assumptions that the next platform will not honour. Kubernetes' stance is that every component must tolerate every other component not existing yet, and "the schema isn't there" is the same kind of transient as "the database isn't there." The general form: classify errors by what they mean (not yet, never, lost ownership), not by which exception class the driver chose.

---

## 8. Every simulated query ran 0.6 ms longer than modeled: a fixed overhead that scales with the time scale

**Found** by the step 7 cross-check's run-time fidelity test at 50 shards (and confirmed at 6). Fixed in [`2e601f3`](https://github.com/arqbca11/HNSWonGPUs/commit/2e601f3), `shard/sim/shard.go` (`runQuery`). **Related:** the step 5 cross-check's time-scale finding (the HTTP calls inside the measured interval), which decision 60 fixed; this is the residue underneath it.

**Symptom.** Across 31,087 queries in one round, every recorded run time (`finished_at - started_at`) was the scaled modeled duration plus about 0.56 ms (p50), never less. Aggregate run time was 105.5% of modeled. By size: queries under 1 ms ran at 1.77x, 1 to 5 ms at 1.19x, over 100 ms at 1.004x. The same 0.6 ms appeared at 6 shards, so it was not load.

**Why it matters.** The overhead is fixed per query, so it grows with the time scale: at 10x a median 12 ms query becomes 1.2 ms real plus 0.6, about 1.5x. The shard's CPU is then busier than the model says by tens of percent, queues grow faster than they should, and queue growth is exactly what the Phase 3 experiments measure. Every 10x number in the step 7 notes was taken with this inflation.

**Cause.** `runQuery` took `started = now`, slept the scaled duration with a relative timer, then took `finished = now`. A timer never wakes early and usually wakes a little late (Go timer granularity plus goroutine scheduling, about half a millisecond here), so the recorded run time was the sleep plus the slop, and the next query started only after that, so the slop accumulated into throughput.

**Fix (decision 65).** The CPU is a virtual clock. A query starts at the later of "the CPU is free" and "the query arrived", finishes exactly its scaled duration later, and both timestamps are recorded from that clock; the goroutine sleeps to the absolute finish deadline. A late wake-up no longer stretches the recorded run time, and it does not accumulate, because the next sleep is to an absolute time and is shorter by the same amount. The shard runs at most one slop behind real time.

**What it would have looked like in production.** Nothing: this is a simulator defect. What it would have looked like in the results: every policy comparison at high time scale would have run on shards 20 to 50 percent busier than the scenario said, and the conclusions about when local builds hurt would have been drawn from the wrong load.

**Lesson.** A relative sleep measures "at least this long". For anything whose duration is the measurement, keep a virtual clock, record from it, and use real sleeps only to pace it. And a fidelity claim ("sleeps exactly the scaled duration") needs a test that checks it against every record, not a lower bound.

---

## 7. Workers crashed when Postgres restarted

**Found** by the Compose cross-check agent's `docker compose restart postgres` test. Fixed in [`51716f8`](https://github.com/arqbca11/HNSWonGPUs/commit/51716f8), `worker/kgpu_worker/worker.py` and `store.py`. **Related:** entry 5 (same class: an exit path while holding a lease that nothing guarded).

**Symptom.** Postgres was restarted while one worker held a lease and the other was idle. Both worker processes died with a traceback and were restarted by the Docker restart policy. The idle one crashed inside `claim` with `psycopg.errors.AdminShutdown: terminating connection due to administrator command`. The leaseholder logged `heartbeat errored`, `renew errored`, then crashed after its build finished, in the `complete` call. The build was still recovered, by the reaper and the other worker as attempt 2, 17 s after the restart, and nothing completed twice.

**Why it matters.** Correctness held, but every database blip cost a full lease plus a process restart per worker, and under Compose a container stopped the wrong way is not restarted at all (see the study notes on restart policy), so one blip could leave the pool permanently short. In production the symptom is a fleet of worker pods restarting in unison every time the database fails over, with the builds table showing a burst of attempt increments.

**Cause.** Two unguarded paths. The main loop's `claim` had no error handling at all: it was outside any `try`, so an idle worker died on the first failed statement. The leaseholder's `complete` ran in `_publish`, after the `try` from bug 5 had ended, so a connection error there escaped too. And the renewer treated an *errored* renew as a transient to retry forever, with no notion that past the lease deadline ownership can no longer be assumed.

**Fix (decision 48).** A connection error while idle logs, reconnects with backoff, and continues. `complete`, `fail` and `release` run through one helper that retries on connection errors until the lease deadline (the last successful claim or renew plus the lease length) and then stops touching the row, logging it as lost, because by then the reaper may have requeued it. A renew that errors is not a rejection: it keeps retrying until the same deadline, after which the lease is assumed lost and the build is cancelled. The heartbeat reconnects. psycopg marks a broken connection `closed`, so each thread's store simply reconnects.

**Lesson.** "No exit while holding a lease" has to cover connection loss, not just exceptions from the build. And an error is not a rejection: a rejected renew means someone else owns the build; an errored renew means you don't know, and the only honest deadline for "I don't know" is the lease itself.

---

## 6. Concurrent duplicates of a round's last build were refused as "round full"

**Found** by the Compose cross-check agent. Fixed in [`422d722`](https://github.com/arqbca11/HNSWonGPUs/commit/422d722), the `SubmitBuild` hunk of `scheduler/store/store.go`. **Related:** entry 4 (the same snapshot-before-lock mechanism, one statement earlier).

**Symptom.** A round with two slots, one taken. Eight identical submits of the second shard's build fired at once. One got 201. The other seven got `409 round already has n_shards builds` instead of `200 created:false`. In a round with spare capacity the same burst was handled correctly every time. A shard retrying its submit after a client timeout, as the round's last shard, would have been told the round was full rather than given its stored decision.

**Cause.** After bug 4, `SubmitBuild` locked the round row and then counted builds in a fresh statement. But the duplicate check, `SELECT EXISTS … WHERE build_id = $1`, still ran *before* the lock. The seven losers evaluated it from a snapshot taken before the winner committed, saw no build, waited for the lock, and then counted one build in a one-slot round. Full. The fix for bug 4 had moved the count after the lock and left the existence check where it was.

**Fix.** Lock first, then every check in its own statement: existence, then finished, then count. A losing twin now sees the winner's row and returns "not inserted," which the API turns into the idempotent 200.

**Lesson.** When a fix moves one check behind a lock, look at every other check in the same transaction. Bug 4's fix was correct and incomplete, and the same cross-check method that found 4 found 6 three days later by testing a different race.

---

## 5. A builder exception crashed the worker with the lease held: a poison-pill build

**Found** by the cross-check agent running the worker as a black-box process. Fixed in [`9f206d0`](https://github.com/arqbca11/HNSWonGPUs/commit/9f206d0); the fix is the `_work` hunk of `worker/kgpu_worker/worker.py` plus `validate` in `config.py` and the exit-2 handling in `__main__.py`. (Bundled with the rest of that cross-check round; later bug fixes get their own commit.)

**Symptom.** With `COST_BANDWIDTH=0`, a documented configuration override, the worker claimed a build and then the whole process died with exit status 1 and a `ZeroDivisionError` traceback. No `failed` or `released` line was logged. The row stayed `state=leased, attempt=1, lease_owner=gpu-err`.

**Why it is worse than one crash.** The reaper would return the lease to the queue after it expired. A restarted worker, or any other worker with the same configuration, would claim the same build and crash the same way. Each cycle costs a lease length and a worker restart, and nothing ever marks the build failed. The build is a poison pill circulating through the pool forever, and under Kubernetes the symptom would be a worker Deployment in a crash loop with no obvious cause in the builds table.

**Cause.** The exception came from `estimate_s`, which the loop called to write the `claimed` log line, outside the `try` that guarded `build()`. The guard covered the step the implementer thought of as "the build" rather than everything that happens while the lease is held.

**Fix.** Two layers. Configuration is validated before the worker connects, so a value that would make arithmetic fail is refused with exit status 2 and one line naming the variable (decision 39). And the `try` now begins the instant the claim returns: any exception that is not a cancellation ends in the guarded `fail` with `TypeName: message` as the reason, the worker logs it, and the loop continues (decision 41). If even `fail` cannot be written, the worker says so and leaves the lease to the reaper.

**What it would have looked like in production.** On Kubernetes the worker pod exits 1 and is restarted with exponential backoff, so it shows `CrashLoopBackOff`. The pod log ends in a `ZeroDivisionError` traceback inside cost-model code, which reads as a code bug rather than a configuration one. On a different screen, the `builds` table shows one build whose `attempt` climbs by one every lease length, and the scheduler's reaper logs `lease expired` for the same `build_id` over and over. If the workers share a ConfigMap they all crash on the same build in turn, the GPU pool looks dead, and every other shard's build waits in a queue nobody drains. None of the three symptoms names the cause, and they live in three places: pod status, pod log, database.

What would make it visible: the `attempt` counter is the tell, since a healthy system loses a lease only when a worker dies, so `attempt` above 2 or 3 on any build is abnormal and worth an alert (Phase 2 metrics: `lease_expirations_total` by `build_id`, or a gauge of builds by attempt). The reaper's log line already carries `build_id`, `attempt` and `lease_owner`, so "same build, new attempt, freshly restarted owner, every thirty seconds" is a diagnosable pattern once you know to look. Startup validation is the cheapest layer: one clear line at deploy time, before any shared state is touched.

**The wider class, and the third fix.** Exception handling closes only half of this. A worker killed by the kernel for running out of memory on one oversized build raises nothing, so `fail` never runs, and that build cycles through the pool the same way, killing each worker that claims it. The defence that covers both halves is a retry budget: the reaper marks a build `failed` when its lease expires on attempt `MAX_ATTEMPTS` (default 5) instead of requeueing it, with the reason recorded (decision 42, commit [`7b13c7e`](https://github.com/arqbca11/HNSWonGPUs/commit/7b13c7e)). It is a dead-letter rule, independent of why the attempts failed.

**Lesson.** In a lease-based worker, the invariant is not "the build is guarded" but "while I hold a lease, no code path may exit without resolving it." The guard belongs at the claim, not at the step you happen to call the build. A process that can crash on a configured value should refuse the configuration at startup, before it can touch shared state. And any retry loop that can be entered by a failure you did not anticipate needs a budget.

**Related:** entry 3 (a rule enforced in one caller is not an invariant; here the guard was in one place too).

---

## 4. Lock-then-count race: a per-round cap that held for sequential submits and failed completely under concurrency

**Found** by the independent cross-check agent's concurrency test. Fixed in [`c3e7178`](https://github.com/arqbca11/HNSWonGPUs/commit/c3e7178), the `SubmitBuild` hunk of `scheduler/store/store.go`.

**Symptom.** Twelve concurrent `POST /builds` into a round created with `n_shards = 3` all returned 201. Postgres held twelve builds in the round. The same cap worked perfectly when submits arrived one at a time, which is why the implementer's tests, all sequential, passed.

**The code as it was.** `SubmitBuild` ran this inside a transaction, then inserted if the count was below the cap:

```sql
SELECT r.n_shards, r.finished_at,
       (SELECT COUNT(*) FROM builds WHERE round_id = r.round_id)   -- the count
FROM rounds r
WHERE r.round_id = $1
FOR UPDATE;                                                        -- the lock
```

The intent: lock the round's row so submits into one round serialise, count the builds already present, insert only if under the cap. It reads as atomic. It is not.

**What READ 2f501d0TED does.** Each statement takes its own snapshot at the moment it starts and sees exactly the rows committed before that moment. `FOR UPDATE` adds one refinement: if the row to be locked is held by another transaction, the statement blocks until that transaction ends, then re-fetches the latest version of *that one row* and rechecks the `WHERE` clause against it (EvalPlanQual). Nothing else in the statement is re-evaluated. A subquery against another table still answers from the snapshot taken at the start.

**The interleaving.** Two shards, cap 3, round 7 empty.

| Step | Transaction A | Transaction B |
| --- | --- | --- |
| 1 | Statement starts, snapshot taken: 0 builds in round 7 | |
| 2 | Locks round 7's row. Count = 0 | |
| 3 | | Statement starts, snapshot taken: still 0 builds (A has not committed) |
| 4 | | Tries to lock round 7's row, blocks behind A |
| 5 | Inserts the build for shard 1 | blocked |
| 6 | 2f501d0, lock released | blocked |
| 7 | | Acquires the lock. Rechecks round 7's row: unchanged, A never modified it. Returns `n_shards = 3`, count = 0, from the step-3 snapshot |
| 8 | | 0 < 3, inserts the build for shard 2 |

B's count was correct at step 3 and stale by step 7, and nothing in the statement's semantics ever refreshed it. With twelve transactions all starting before the first commit, every one snapshots zero builds, they serialise politely on the lock one after another, and all twelve insert. The lock did its job. The count was answered against the wrong moment.

**The fix.** Same transaction, same lock, but the count is its own statement, so it gets a fresh snapshot taken after the lock was acquired:

```sql
SELECT n_shards, finished_at FROM rounds WHERE round_id = $1 FOR UPDATE;  -- statement 1: lock
SELECT COUNT(*) FROM builds WHERE round_id = $1;                        -- statement 2: count
```

Replaying the table: B blocks at step 4 inside statement 1, acquires the lock at step 7, and only then starts statement 2, whose snapshot includes A's committed insert. Count = 1. The fourth transaction in a row of twelve sees 3 and is refused. After the fix the same test produced 3 inserts and 9 refusals, over HTTP and through the store.

Both halves are necessary. Lock without a fresh count is this bug. A fresh count without the lock lets two transactions both read 2, both insert, and end at 4.

**Alternatives not taken.** `SERIALIZABLE` isolation would have detected the read-write conflict and aborted one side with a serialization failure, correct but turning a rare condition into a retry loop. A database constraint cannot express a count cap directly without a trigger.

**Lesson.** Under READ 2f501d0TED, "lock, then read, then act" must be three statements, or at least the read must come after the lock in a statement of its own. A check folded into the locking statement is answered before the lock is granted. More generally: a test that exercises a concurrency promise must actually be concurrent. The promise in the design doc said "the row lock serialises submits per round so the cap holds"; the implementer tested the cap, the cross-checker tested the sentence.

---

## 3. A `stream_done` report from a shard with no build could complete a round

**Found** by the cross-check agent. Fixed in [`c3e7178`](https://github.com/arqbca11/HNSWonGPUs/commit/c3e7178) and [`412c189`](https://github.com/arqbca11/HNSWonGPUs/commit/412c189) (the foreign keys).

**Symptom.** Round of two shards. Both builds finished. Shard 0 reported its stream done, shard 1 was still streaming. A status row written for a shard 99 that had never submitted, with `stream_done = true`, made `FinishCompleteRounds` stamp the round finished.

**Cause.** The completion rule counted `shard_status` rows with `stream_done` and compared the count to `n_shards`. Nothing tied those rows to shards that had actually submitted builds. The HTTP report handler refused a report for a shard with no build, so no real client could produce the state, but the store method underneath did no such check, and the store is a public surface used by tests, the reaper, and the future reconsider loop. A guard that exists in one caller is not an invariant.

**Fix.** Three layers. The completion query joins `shard_status` to `builds` on `(round_id, shard_id)`, so only shards with a build count. `ReportStatus` and `RecordArrival` return `ErrNoBuild` (404 over HTTP) for a shard with no build. And, the next day, migration 0004 added `UNIQUE (round_id, shard_id)` on `builds` with foreign keys from `shard_status` and `shard_jobs` to it, so the orphan row cannot exist no matter who writes it; the store now gets that error from Postgres (SQLSTATE 23503) rather than checking first.

**Lesson.** Enforce a rule where every caller passes through, or in the schema, not in the one handler that happens to exist today.

---

## 2. Test packages sharing one database truncated each other mid-test

**Found** while triaging the first cross-check run. Fixed in [`5fdc4eb`](https://github.com/arqbca11/HNSWonGPUs/commit/5fdc4eb), `scripts/test-db.sh`.

**Symptom.** A test created a round, got `{"round_id": 19}` back, submitted a build into round 19, and received 404 "no such round." Another package's test reported a build it had just inserted as missing.

**Cause.** `go test ./...` runs different packages' test binaries in parallel. Every package's database tests begin with `TRUNCATE` on the shared throwaway Postgres. The api and crosscheck packages ran at the same time, and one package's `TRUNCATE` landed between another's two requests.

**Fix.** `scripts/test-db.sh` runs `go test -p 1`, one package at a time. Within a package, tests were already sequential.

**Lesson.** Package-level parallelism is a default worth knowing before pointing several packages at one external resource. The phantom 404 was easy to misread as an API bug.

---

## 1. Duplicate submit mixed the stored row with the repeat's request body

**Found** by the cross-check agent's first run. Fixed in [`5fdc4eb`](https://github.com/arqbca11/HNSWonGPUs/commit/5fdc4eb), `scheduler/api/api.go`.

**Symptom.** A shard retried `POST /builds` with a body that differed from the original. The response said `created: false` and returned the stored placement and memory, but the CPU and GPU time estimates were computed from the repeat's `n_vectors`. A shard retrying a local build would have slept for the wrong time.

**Cause.** The handler computed estimates from the request before discovering the build existed, then overwrote only some fields from the stored row.

**Fix.** On a duplicate, every field in the response is derived from the stored row. The doc comment on `Created` had already promised this; the code hadn't kept the promise.

**Lesson.** When an endpoint is idempotent, build the duplicate response from storage alone. Mixing request and storage is how two retries disagree with each other.
