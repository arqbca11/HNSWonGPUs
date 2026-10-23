package crosscheck

// Helpers for the Phase 3 step 3 cross-check (reconsider_db_test.go,
// reconsider_tick_test.go). Written from docs/design/phase3-policies.md
// (decisions 14-19), migration 0006 and `go doc` of scheduler/store,
// scheduler/reconsider and scheduler/policy, without reading their source.

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	rc "github.com/arqbca11/HNSWonGPUs/scheduler/reconsider"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
	"github.com/jackc/pgx/v5/pgconn"
)

const rxDim int32 = 128

// Cost model v0 magnitudes at dim 128 (costmodel.Default(), printed by the
// tests themselves): 100k vectors build in 20.63 s on a CPU and 4.64 s on a
// GPU including 0.576 s of transfer; 2M vectors take 65.5 s on a GPU.
const (
	rxN    int64 = 100_000
	rxBigN int64 = 2_000_000
)

var rxCM = costmodel.Default()

func rxSecs(d time.Duration) float64 { return d.Seconds() }

// rxSubmit inserts a build through the store with a chosen placement and
// priority; mem_bytes is the cost model's, as the API would set it.
func (e *env) rxSubmit(t *testing.T, round int64, shard int32, n int64, placement string, prio float64) string {
	t.Helper()
	id := store.BuildID(shard, round)
	ok, err := e.st.SubmitBuild(e.ctx, store.Build{BuildID: id, RoundID: round, ShardID: shard, NVectors: n, Dim: rxDim,
		MemBytes: rxCM.GPUMemBytes(n, rxDim), Placement: placement, Priority: prio})
	if err != nil || !ok {
		t.Fatalf("SubmitBuild %s: ok=%v err=%v", id, ok, err)
	}
	t.Logf("submitted %s n=%d dim=%d mem=%d placement=%s priority=%g", id, n, rxDim, rxCM.GPUMemBytes(n, rxDim), placement, prio)
	return id
}

// rxReport writes a shard's load report plus one recorded query of meanMs
// (so the host's SQL-derived mean query duration is exactly meanMs,
// decision 18). seq 0 is upserted on repeats, so calling it again only
// changes the report.
func (e *env) rxReport(t *testing.T, round int64, shard int32, q, wni int32, prog float32, meanMs int64, streamDone bool) {
	t.Helper()
	var jobs []store.JobRecord
	if meanMs > 0 {
		jobs = []store.JobRecord{{Seq: 0, DurationMs: meanMs, NeedsIndex: false, ArrivedAt: time.Now()}}
	}
	err := e.st.ReportBatch(e.ctx, store.ShardStatus{RoundID: round, ShardID: shard, QueueDepth: q, WaitingNeedsIndex: wni,
		OldestWaitMs: 0, BuildProgress: prog, StreamDone: streamDone}, jobs)
	if err != nil {
		t.Fatalf("ReportBatch shard %d: %v", shard, err)
	}
	t.Logf("shard %d reports queue_depth=%d waiting_needs_index=%d build_progress=%g stream_done=%v (one recorded query of %d ms)", shard, q, wni, prog, streamDone, meanMs)
}

// rxAgeRound backdates the round's start so the SQL-derived arrival rate
// (arrivals / modeled round age, decision 18) is negligible: one arrival in
// an hour. Without it a one-query shard 50 ms into its round reads 20 q/s.
func (e *env) rxAgeRound(t *testing.T, round int64) {
	t.Helper()
	if _, err := e.pool.Exec(e.ctx, `UPDATE rounds SET started_at = now() - interval '1 hour' WHERE round_id = $1`, round); err != nil {
		t.Fatalf("age round: %v", err)
	}
	t.Logf("fixture: round %d started_at backdated by 1 h, so the derived arrival rate is ~1/3600 q/s", round)
}

func (e *env) rxCloseRound(t *testing.T, round int64) {
	t.Helper()
	if _, err := e.pool.Exec(e.ctx, `UPDATE rounds SET finished_at = now() WHERE round_id = $1`, round); err != nil {
		t.Fatalf("close round: %v", err)
	}
}

func (e *env) rxCopy(t *testing.T, primary string, prio float64) string {
	t.Helper()
	ok, err := e.st.SpeculateCopy(e.ctx, primary, prio)
	if err != nil || !ok {
		t.Fatalf("SpeculateCopy %s: ok=%v err=%v", primary, ok, err)
	}
	t.Logf("SpeculateCopy(%s, %g) -> true", primary, prio)
	return primary + ":spec"
}

func (e *env) rxRow(t *testing.T, label, id string) store.BuildRow {
	t.Helper()
	b := e.mustBuild(t, id)
	rxLogRow(t, label, b)
	return b
}

func rxLogRow(t *testing.T, label string, b store.BuildRow) {
	t.Helper()
	t.Logf("[db] %s: %s placement=%s state=%s attempt=%d priority=%g moved=%v speculative=%v lease_owner=%s fail_reason=%s started_at=%s finished_at=%s",
		label, b.BuildID, b.Placement, b.State, b.Attempt, b.Priority, b.Moved, b.Speculative, deref(b.LeaseOwner), deref(b.FailReason), tptr(b.StartedAt), tptr(b.FinishedAt))
}

func (e *env) rxHeartbeat(t *testing.T, id string, mem int64) {
	t.Helper()
	must(t, e.st.Heartbeat(e.ctx, id, mem, ""))
	t.Logf("worker %s heartbeats, mem %d", id, mem)
}

func (e *env) rxClaim(t *testing.T, worker string, mem int64, lease time.Duration) store.Claimed {
	t.Helper()
	c, ok, err := e.st.Claim(e.ctx, worker, mem, lease)
	if err != nil || !ok {
		t.Fatalf("Claim by %s: ok=%v err=%v", worker, ok, err)
	}
	t.Logf("%s claims %s attempt %d", worker, c.BuildID, c.Attempt)
	return c
}

// lockedBuf is a log sink safe for concurrent ticks.
type lockedBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuf) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *lockedBuf) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

func rxPolicy(t *testing.T, name string) policy.Policy {
	t.Helper()
	p, err := policy.ByName(name)
	if err != nil {
		t.Fatalf("ByName(%q): %v", name, err)
	}
	return p
}

// rxTick runs one reconsider.Tick with a real policy and returns the tick's
// log lines, each echoed into the test log.
func (e *env) rxTick(t *testing.T, polName string, scale float64) []string {
	t.Helper()
	var buf lockedBuf
	log := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	rc.Tick(e.ctx, e.st, rxPolicy(t, polName), rxCM, rc.Config{Interval: time.Second, StaleAfter: time.Minute, TimeScale: scale}, log)
	var lines []string
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if l != "" {
			lines = append(lines, l)
			t.Logf("  [tick %s scale %g] %s", polName, scale, l)
		}
	}
	if len(lines) == 0 {
		t.Logf("  [tick %s scale %g] (no log lines)", polName, scale)
	}
	return lines
}

func pgCode(err error) string {
	var pe *pgconn.PgError
	if errors.As(err, &pe) {
		return pe.Code
	}
	return ""
}

// rxRawBuild inserts a builds row by SQL, bypassing the store, to probe the
// schema itself.
func (e *env) rxRawBuild(id string, round int64, shard int32, placement, state string, speculative bool) error {
	_, err := e.pool.Exec(e.ctx, `INSERT INTO builds (build_id, round_id, shard_id, n_vectors, dim, mem_bytes, placement, state, speculative)
		VALUES ($1, $2, $3, 1000, 128, 1, $4, $5, $6)`, id, round, shard, placement, state, speculative)
	return err
}

// roadmapClaimSQL is the claim statement from docs/roadmap.md ("The four
// operations"), plus the memory filter every claim carries since Phase 1:
// the statement the Python worker runs, typed in from the spec.
const roadmapClaimSQL = `UPDATE builds
SET state = 'leased', attempt = attempt + 1, lease_owner = $1,
    lease_until = now() + interval '30 seconds', started_at = now()
WHERE build_id = (
  SELECT build_id FROM builds
  WHERE state = 'queued' AND placement = 'gpu' AND mem_bytes <= $2
  ORDER BY priority DESC, enqueued_at
  LIMIT 1
  FOR UPDATE SKIP LOCKED)
RETURNING build_id, attempt`
