-- Phase 3 step 3 (owner approved 2026-10-22): preemption, reneging and
-- speculation executed by the reconsider timer. Two new facts per build and
-- room for decision 5's speculative copy as a second builds row.

-- moved: placement already changed once after submit (preempt or renege set
-- it). Decision 8 says at most once; the policy checks the flag and the store
-- ops guard on it, so a second move is impossible even across two schedulers.
ALTER TABLE builds ADD COLUMN moved boolean NOT NULL DEFAULT false;

-- speculative: this row is a GPU copy of a local build (decision 5), racing
-- it. build_id is the primary's id plus ':spec'. A copy lives only on the GPU
-- side: queued, leased, then done (it won), or failed (cancelled after the
-- local build won, or its retry budget ran out).
ALTER TABLE builds ADD COLUMN speculative boolean NOT NULL DEFAULT false;
ALTER TABLE builds ADD CONSTRAINT builds_speculative_gpu
  CHECK (NOT speculative OR placement = 'gpu');

-- A copy shares its primary's (round_id, shard_id), which the bug-3 fix
-- (migration 0004) made UNIQUE with the shard_status/shard_jobs foreign keys
-- hanging off it. Re-anchor those FKs on builds.build_id (the primary key)
-- through a stored generated column: the expression is exactly the
-- idempotency-key convention (shard_id:round_id), so the database itself
-- derives "this shard's real build in this round" — it can never reference
-- the copy, whose id carries the ':spec' suffix. Bug 3's rule (a report must
-- come from a shard with a build) survives unweakened.
ALTER TABLE shard_status DROP CONSTRAINT shard_status_build_fk;
ALTER TABLE shard_jobs   DROP CONSTRAINT shard_jobs_build_fk;
ALTER TABLE builds       DROP CONSTRAINT builds_round_shard_unique;

ALTER TABLE shard_status ADD COLUMN build_id text
  GENERATED ALWAYS AS (shard_id::text || ':' || round_id::text) STORED
  REFERENCES builds (build_id);
ALTER TABLE shard_jobs ADD COLUMN build_id text
  GENERATED ALWAYS AS (shard_id::text || ':' || round_id::text) STORED
  REFERENCES builds (build_id);

-- What the dropped constraint really meant, restated: one real build per
-- shard per round, and at most one copy. (Partial unique indexes enforce it;
-- they just cannot anchor a foreign key, hence the generated columns above.)
CREATE UNIQUE INDEX builds_one_real_per_shard ON builds (round_id, shard_id)
  WHERE NOT speculative;
CREATE UNIQUE INDEX builds_one_copy_per_shard ON builds (round_id, shard_id)
  WHERE speculative;
