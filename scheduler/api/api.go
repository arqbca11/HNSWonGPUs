// Package api is the scheduler's HTTP surface. Shards talk to it; workers
// never do (they use SQL directly, see design decision 4).
//
//	POST /rounds                                 start a round
//	GET  /rounds/{id}                            round status: builds, queries, shard reports
//	POST /builds                                 submit a build; returns placement and estimates
//	GET  /builds/{id}                            the build row
//	POST /builds/{id}/done                       a shard finished a local build
//	POST /jobs/{round}/{shard}                   a query arrived at the shard
//	POST /jobs/{round}/{shard}/{seq}/start       the query began running
//	POST /jobs/{round}/{shard}/{seq}/done        the query finished
//	POST /shards/{round}/{shard}/report          the shard's load report; reply carries the build's placement
//	GET  /workers                                registered workers and the live pool state
//	GET  /healthz                                200 when Postgres answers (readiness)
//	GET  /livez                                  200 when the process answers at all (liveness)
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/metrics"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Config struct {
	WorkerStaleAfter time.Duration // a worker not seen for this long is not in the pool
	// TimeScale converts wall-clock facts to modeled seconds (wall x scale)
	// before they meet cost-model durations: under FAKE_TIME_SCALE=s a worker
	// finishes a modeled 60 s build in 6 wall seconds, so a lease's elapsed
	// wall time understates its modeled progress by s. 0 means 1 (production).
	TimeScale float64
	// Draining is set by main after SIGTERM, for the window the pod keeps
	// serving while endpoint removal propagates (decision 29). A draining
	// scheduler answers but never stamps a round: a stamp here would record
	// the round's metrics in a process Prometheus has already stopped
	// scraping (decision 40). nil means never draining.
	Draining *atomic.Bool
}

type Server struct {
	st  *store.Store
	pol policy.Policy
	cm  costmodel.Model
	cfg Config
	log *slog.Logger
	mux *http.ServeMux
}

func New(st *store.Store, pol policy.Policy, cm costmodel.Model, cfg Config, log *slog.Logger) *Server {
	s := &Server{st: st, pol: pol, cm: cm, cfg: cfg, log: log, mux: http.NewServeMux()}
	s.mux.HandleFunc("POST /rounds", s.createRound)
	s.mux.HandleFunc("GET /rounds/{id}", s.getRound)
	s.mux.HandleFunc("POST /builds", s.submitBuild)
	s.mux.HandleFunc("GET /builds/{id}", s.getBuild)
	s.mux.HandleFunc("POST /builds/{id}/done", s.localBuildDone)
	s.mux.HandleFunc("POST /builds/{id}/fail", s.localBuildFail)
	s.mux.HandleFunc("POST /jobs/{round}/{shard}", s.jobArrived)
	s.mux.HandleFunc("POST /shards/{round}/{shard}/report", s.shardReport)
	s.mux.HandleFunc("POST /jobs/{round}/{shard}/{seq}/start", s.jobStart)
	s.mux.HandleFunc("POST /jobs/{round}/{shard}/{seq}/done", s.jobDone)
	s.mux.HandleFunc("GET /workers", s.workers)
	metrics.Init(pol.Name())
	s.mux.Handle("GET /metrics", promhttp.Handler())
	s.mux.HandleFunc("GET /healthz", s.healthz)
	s.mux.HandleFunc("GET /livez", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "alive"})
	})
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// ---- rounds ---------------------------------------------------------------

type CreateRoundRequest struct {
	Scenario string `json:"scenario"`
	Seed     int64  `json:"seed"`
	NShards  int32  `json:"n_shards"`
	Dataset  string `json:"dataset,omitempty"` // Phase 4 decision 15; "" = a modeled round
}

// LocalFailRequest is the body of POST /builds/{id}/fail: a local build the
// recall gate refused (Phase 4 decision 19).
type LocalFailRequest struct {
	Reason string `json:"reason"`
}

type CreateRoundResponse struct {
	RoundID int64 `json:"round_id"`
}

func (s *Server) createRound(w http.ResponseWriter, r *http.Request) {
	var req CreateRoundRequest
	if !decode(w, r, &req) {
		return
	}
	if req.NShards <= 0 {
		writeError(w, http.StatusBadRequest, "n_shards must be positive")
		return
	}
	if req.Scenario == "" {
		req.Scenario = "default"
	}
	id, err := s.st.CreateRoundWithDataset(r.Context(), req.Scenario, req.Seed, req.NShards, req.Dataset)
	if err != nil {
		s.internal(w, "create round", err)
		return
	}
	s.log.Info("round started", "round_id", id, "scenario", req.Scenario, "seed", req.Seed, "n_shards", req.NShards)
	writeJSON(w, http.StatusCreated, CreateRoundResponse{RoundID: id})
}

type RoundStatus struct {
	Round    store.Round         `json:"round"`
	Builds   []store.BuildRow    `json:"builds"`
	Jobs     []store.JobRow      `json:"jobs"`
	Statuses []store.ShardStatus `json:"shard_status"`
}

func (s *Server) getRound(w http.ResponseWriter, r *http.Request) {
	id, ok := pathInt64(w, r, "id")
	if !ok {
		return
	}
	ctx := r.Context()
	// Make the answer fresh: GPU completions happen via SQL and the reaper
	// tick may not have run since. Not while draining: the answer then comes
	// from what is already stamped, and the replacement's reaper stamps the
	// rest within a tick (decision 40).
	if s.cfg.Draining == nil || !s.cfg.Draining.Load() {
		fin, err := s.st.FinishCompleteRounds(ctx)
		if err != nil {
			s.internal(w, "finish rounds", err)
			return
		}
		metrics.ObserveFinishedRounds(fin)
	}
	rd, found, err := s.st.GetRound(ctx, id)
	if err != nil {
		s.internal(w, "get round", err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "no such round")
		return
	}
	builds, err := s.st.ListBuilds(ctx, id)
	if err != nil {
		s.internal(w, "list builds", err)
		return
	}
	jobs, err := s.st.ListJobs(ctx, id)
	if err != nil {
		s.internal(w, "list jobs", err)
		return
	}
	statuses, err := s.st.ListShardStatus(ctx, id)
	if err != nil {
		s.internal(w, "list shard status", err)
		return
	}
	if builds == nil {
		builds = []store.BuildRow{}
	}
	if jobs == nil {
		jobs = []store.JobRow{}
	}
	if statuses == nil {
		statuses = []store.ShardStatus{}
	}
	writeJSON(w, http.StatusOK, RoundStatus{Round: rd, Builds: builds, Jobs: jobs, Statuses: statuses})
}

// ---- builds ---------------------------------------------------------------

type SubmitBuildRequest struct {
	RoundID  int64 `json:"round_id"`
	ShardID  int32 `json:"shard_id"`
	NVectors int64 `json:"n_vectors"`
	Dim      int32 `json:"dim"`
}

type SubmitBuildResponse struct {
	BuildID   string  `json:"build_id"`
	Placement string  `json:"placement"`
	Priority  float64 `json:"priority"`
	Reason    string  `json:"reason"` // the policy's one-line reason; on a duplicate, a note that the existing decision was returned (the original reason is not stored)
	MemBytes  int64   `json:"mem_bytes"`
	Created   bool    `json:"created"` // false: this build_id already existed; every field but reason describes the existing row
	// Cost model estimates, so the shard can sleep for the right time on a local build.
	CPUBuildMs int64 `json:"cpu_build_ms"`
	GPUTotalMs int64 `json:"gpu_total_ms"`
	// The step 2 queue replay: how long a newly queued build would wait before
	// a worker starts it (decision 13), in whole milliseconds (truncated). It
	// describes the GPU path whatever placement was chosen. 0 with an empty
	// pool (the cold-start rule); -1 when no live worker's memory could ever
	// take this build. A pointer so a duplicate can omit it entirely (its
	// answer is the original decision, not a fresh estimate) while an empty
	// pool's genuine 0 still serializes — the cross-check caught the plain
	// int64 serving 0 for both.
	GPUQueueWaitMs *int64 `json:"gpu_queue_wait_ms,omitempty"`
}

func (s *Server) submitBuild(w http.ResponseWriter, r *http.Request) {
	var req SubmitBuildRequest
	if !decode(w, r, &req) {
		return
	}
	if req.RoundID <= 0 || req.ShardID < 0 || req.NVectors <= 0 || req.Dim <= 0 {
		writeError(w, http.StatusBadRequest, "round_id, n_vectors and dim must be positive; shard_id non-negative")
		return
	}
	ctx := r.Context()
	rd, found, err := s.st.GetRound(ctx, req.RoundID)
	if err != nil {
		s.internal(w, "get round", err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "no such round")
		return
	}
	buildID := store.BuildID(req.ShardID, req.RoundID)

	// Invariant 6 first: a resubmit returns the existing decision, even if the
	// round has since finished. Only a genuinely new build is refused below.
	if existing, found, err := s.st.GetBuild(ctx, buildID); err != nil {
		s.internal(w, "get build", err)
		return
	} else if found {
		s.writeDuplicate(w, existing)
		return
	}
	_ = rd // existence checked above; fullness and finish are enforced inside SubmitBuild under a row lock

	mem := s.cm.GPUMemBytes(req.NVectors, req.Dim)
	resp := SubmitBuildResponse{
		BuildID:    buildID,
		MemBytes:   mem,
		CPUBuildMs: s.cm.CPUBuild(req.NVectors, req.Dim).Milliseconds(),
		GPUTotalMs: s.cm.GPUTotal(req.NVectors, req.Dim).Milliseconds(),
	}

	pool, err := s.st.Pool(ctx, s.cfg.WorkerStaleAfter)
	if err != nil {
		s.internal(w, "pool state", err)
		return
	}
	spec := policy.BuildSpec{ShardID: req.ShardID, RoundID: req.RoundID, NVectors: req.NVectors, Dim: req.Dim, MemBytes: mem}
	// The host computes every estimate the policy sees (Phase 3 decision 9).
	est := policy.Estimates{
		LocalBuildS: s.cm.CPUBuild(req.NVectors, req.Dim).Seconds(),
		GPUBuildS:   s.cm.GPUBuild(req.NVectors, req.Dim).Seconds(),
		TransferS:   s.cm.Transfer(req.NVectors, req.Dim).Seconds(),
	}
	// The step 2 queue replay fills QueueWaitS: with an empty pool it stays 0
	// (the cold-start rule, phase 3 doc G1 — "no workers yet" must not scare
	// a policy off the queue), otherwise it is the wait this build would join
	// at (+Inf when its memory fits no live worker, the estimator's truth).
	queueWaitS := 0.0
	if pool.LiveWorkers > 0 {
		snap, err := s.st.GetQueueSnapshot(ctx, s.cfg.WorkerStaleAfter)
		if err != nil {
			s.internal(w, "queue snapshot", err)
			return
		}
		queueWaitS = s.queueWait(snap, buildID, mem, est)
	}
	dec := s.pol.Decide(policy.Input{Build: spec, Est: est,
		Pool: policy.PoolState{LiveWorkers: pool.LiveWorkers, LargestMem: pool.LargestMem, QueueWaitS: queueWaitS}})
	waitMs := int64(queueWaitS * 1000)
	if math.IsInf(queueWaitS, 1) {
		waitMs = -1
	}
	resp.GPUQueueWaitMs = &waitMs

	inserted, err := s.st.SubmitBuild(ctx, store.Build{
		BuildID: buildID, RoundID: req.RoundID, ShardID: req.ShardID, NVectors: req.NVectors,
		Dim: req.Dim, MemBytes: mem, Placement: dec.Placement, Priority: dec.Priority,
	})
	if errors.Is(err, store.ErrRoundFull) || errors.Is(err, store.ErrRoundFinished) {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	if err != nil {
		s.internal(w, "submit build", err)
		return
	}
	if !inserted {
		// Lost a race with a concurrent identical submit; ON CONFLICT DO NOTHING
		// guarantees exactly one insert, so describe the winner's row.
		existing, _, err := s.st.GetBuild(ctx, buildID)
		if err != nil {
			s.internal(w, "get build", err)
			return
		}
		s.writeDuplicate(w, existing)
		return
	}
	resp.Placement, resp.Priority, resp.Reason, resp.Created = dec.Placement, dec.Priority, dec.Reason, true
	metrics.PlacementDecisions.WithLabelValues(dec.Placement, s.pol.Name()).Inc()
	s.log.Info("build placed", "build_id", buildID, "round_id", req.RoundID, "attempt", 0,
		"shard_id", req.ShardID, "placement", dec.Placement, "priority", dec.Priority, "policy", s.pol.Name(),
		"reason", dec.Reason, "n_vectors", req.NVectors, "dim", req.Dim, "mem_bytes", mem,
		"live_workers", pool.LiveWorkers, "gpu_queue_wait_s", queueWaitS)
	writeJSON(w, http.StatusCreated, resp)
}

// queueWait turns a store snapshot into the estimator's inputs and asks what
// a newly queued build of memBytes would wait (decision 13: it joins at
// priority 0 behind the whole queue — exact for the FCFS baseline, an upper
// bound for a policy that would rank it higher). Per decision 9 the host
// computes every second here: a leased build's remaining time is its cost
// model total minus the modeled time since its claim (wall elapsed x
// TimeScale), floored at 0 (an overdue lease means "about to free or be
// reaped", either way soon).
func (s *Server) queueWait(snap store.QueueSnapshot, buildID string, memBytes int64, est policy.Estimates) float64 {
	scale := s.cfg.TimeScale
	if scale <= 0 {
		scale = 1
	}
	busy := map[string]float64{}
	for _, l := range snap.Leased {
		rem := s.cm.GPUTotal(l.NVectors, l.Dim).Seconds() - l.ElapsedS*scale
		if rem < 0 {
			rem = 0
		}
		if rem > busy[l.LeaseOwner] {
			busy[l.LeaseOwner] = rem
		}
	}
	workers := make([]policy.WorkerView, 0, len(snap.Workers))
	for _, w := range snap.Workers {
		workers = append(workers, policy.WorkerView{ID: w.WorkerID, Node: w.Node, MemBytes: w.MemBytes, BusyRemainingS: busy[w.WorkerID]})
	}
	queue := make([]policy.QueuedJob, 0, len(snap.Queued))
	for i, q := range snap.Queued { // snapshot is already in claim order
		queue = append(queue, policy.QueuedJob{
			BuildID: q.BuildID, Priority: q.Priority, Seq: i, MemBytes: q.MemBytes,
			GPUBuildS: s.cm.GPUBuild(q.NVectors, q.Dim).Seconds(),
			TransferS: s.cm.Transfer(q.NVectors, q.Dim).Seconds(),
		})
	}
	return policy.NewJobWait(workers, queue, policy.QueuedJob{
		BuildID: buildID, MemBytes: memBytes, GPUBuildS: est.GPUBuildS, TransferS: est.TransferS,
	})
}

// writeDuplicate answers a resubmit: 200, created=false, and every estimate
// computed from the stored row rather than the request body (which may
// differ; cross-check finding, 2026-10-02).
func (s *Server) writeDuplicate(w http.ResponseWriter, existing store.BuildRow) {
	s.log.Info("duplicate submit ignored", "build_id", existing.BuildID, "round_id", existing.RoundID,
		"attempt", existing.Attempt, "placement", existing.Placement)
	writeJSON(w, http.StatusOK, SubmitBuildResponse{
		BuildID:    existing.BuildID,
		Placement:  existing.Placement,
		Priority:   existing.Priority,
		Reason:     "duplicate submit; existing decision returned",
		MemBytes:   existing.MemBytes,
		Created:    false,
		CPUBuildMs: s.cm.CPUBuild(existing.NVectors, existing.Dim).Milliseconds(),
		GPUTotalMs: s.cm.GPUTotal(existing.NVectors, existing.Dim).Milliseconds(),
	})
}

func (s *Server) getBuild(w http.ResponseWriter, r *http.Request) {
	b, found, err := s.st.GetBuild(r.Context(), r.PathValue("id"))
	if err != nil {
		s.internal(w, "get build", err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "no such build")
		return
	}
	writeJSON(w, http.StatusOK, b)
}

// ---- query arrivals and shard reports --------------------------------------

type ArrivalRequest struct {
	Seq        int32 `json:"seq"`
	DurationMs int64 `json:"duration_ms"`
	NeedsIndex bool  `json:"needs_index"`
}

// jobArrived records a query arriving at a shard (decision 28).
func (s *Server) jobArrived(w http.ResponseWriter, r *http.Request) {
	round, ok := pathInt64(w, r, "round")
	if !ok {
		return
	}
	shard, ok := pathInt32(w, r, "shard")
	if !ok {
		return
	}
	var req ArrivalRequest
	if !decode(w, r, &req) {
		return
	}
	if req.Seq < 0 || req.DurationMs < 0 {
		writeError(w, http.StatusBadRequest, "seq and duration_ms must be non-negative")
		return
	}
	inserted, err := s.st.RecordArrival(r.Context(), round, shard, req.Seq, req.DurationMs, req.NeedsIndex)
	if errors.Is(err, store.ErrNoBuild) {
		writeError(w, http.StatusNotFound, err.Error()+"; submit first")
		return
	}
	if errors.Is(err, store.ErrStreamDone) {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	if err != nil {
		s.internal(w, "record arrival", err)
		return
	}
	code := http.StatusOK
	if inserted {
		code = http.StatusCreated
		s.log.Info("query arrived", "round_id", round, "shard_id", shard, "seq", req.Seq,
			"build_id", store.BuildID(shard, round), "duration_ms", req.DurationMs, "needs_index", req.NeedsIndex)
	}
	writeJSON(w, code, map[string]any{"round_id": round, "shard_id": shard, "seq": req.Seq, "created": inserted})
}

type ReportRequest struct {
	QueueDepth        int32   `json:"queue_depth"`
	WaitingNeedsIndex int32   `json:"waiting_needs_index"`
	OldestWaitMs      int64   `json:"oldest_wait_ms"`
	BuildProgress     float32 `json:"build_progress"`
	StreamDone        bool    `json:"stream_done"`
	// Jobs is the batch of query records that changed since the last report
	// (decision 60): new arrivals, and starts/finishes of earlier ones, with
	// the shard's own timestamps. Optional; the per-event job endpoints remain.
	Jobs []store.JobRecord `json:"jobs,omitempty"`
}

type BuildSummary struct {
	BuildID   string `json:"build_id"`
	Placement string `json:"placement"`
	State     string `json:"state"`
	Attempt   int32  `json:"attempt"`
}

type ReportResponse struct {
	Build BuildSummary `json:"build"`
}

// shardReport is the shard's sync point (decision 29): it tells the scheduler
// its load and learns the build's current placement in the same round trip.
func (s *Server) shardReport(w http.ResponseWriter, r *http.Request) {
	round, ok := pathInt64(w, r, "round")
	if !ok {
		return
	}
	shard, ok := pathInt32(w, r, "shard")
	if !ok {
		return
	}
	var req ReportRequest
	if !decode(w, r, &req) {
		return
	}
	if req.QueueDepth < 0 || req.WaitingNeedsIndex < 0 || req.WaitingNeedsIndex > req.QueueDepth ||
		req.OldestWaitMs < 0 || req.BuildProgress < 0 || req.BuildProgress > 1 {
		writeError(w, http.StatusBadRequest, "report fields out of range")
		return
	}
	for i, j := range req.Jobs {
		switch {
		case j.Seq < 0 || j.DurationMs < 0 || j.ArrivedAt.IsZero():
			writeError(w, http.StatusBadRequest, fmt.Sprintf("jobs[%d]: seq and duration_ms must be non-negative and arrived_at set", i))
			return
		case j.StartedAt != nil && j.StartedAt.Before(j.ArrivedAt):
			writeError(w, http.StatusBadRequest, fmt.Sprintf("jobs[%d]: started_at before arrived_at", i))
			return
		case j.FinishedAt != nil && (j.StartedAt == nil || j.FinishedAt.Before(*j.StartedAt)):
			writeError(w, http.StatusBadRequest, fmt.Sprintf("jobs[%d]: finished_at without started_at or before it", i))
			return
		}
	}
	if len(req.Jobs) > 10000 {
		writeError(w, http.StatusBadRequest, "jobs batch too large (max 10000)")
		return
	}
	ctx := r.Context()
	b, found, err := s.st.GetBuild(ctx, store.BuildID(shard, round))
	if err != nil {
		s.internal(w, "get build", err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "this shard has no build in this round; submit first")
		return
	}
	status := store.ShardStatus{
		RoundID: round, ShardID: shard, QueueDepth: req.QueueDepth, WaitingNeedsIndex: req.WaitingNeedsIndex,
		OldestWaitMs: req.OldestWaitMs, BuildProgress: req.BuildProgress, StreamDone: req.StreamDone,
	}
	if len(req.Jobs) > 0 {
		err = s.st.ReportBatch(ctx, status, req.Jobs)
	} else {
		err = s.st.ReportStatus(ctx, status)
	}
	if errors.Is(err, store.ErrNoBuild) {
		writeError(w, http.StatusNotFound, err.Error()+"; submit first")
		return
	}
	if errors.Is(err, store.ErrStreamDone) {
		writeError(w, http.StatusConflict, "batch brings new arrivals after stream_done")
		return
	}
	if err != nil {
		s.internal(w, "report", err)
		return
	}
	if len(req.Jobs) > 0 {
		s.log.Debug("report batch", "round_id", round, "shard_id", shard, "build_id", store.BuildID(shard, round), "records", len(req.Jobs))
	}
	writeJSON(w, http.StatusOK, ReportResponse{Build: BuildSummary{
		BuildID: b.BuildID, Placement: b.Placement, State: b.State, Attempt: b.Attempt,
	}})
}

func (s *Server) localBuildDone(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	ctx := r.Context()
	ok, err := s.st.CompleteLocal(ctx, id)
	if err != nil {
		s.internal(w, "complete local", err)
		return
	}
	if !ok {
		b, found, err := s.st.GetBuild(ctx, id)
		if err != nil {
			s.internal(w, "get build", err)
			return
		}
		if !found {
			writeError(w, http.StatusNotFound, "no such build")
			return
		}
		writeError(w, http.StatusConflict, "build is "+b.Placement+"/"+b.State+", not a running local build")
		return
	}
	b, _, _ := s.st.GetBuild(ctx, id)
	s.log.Info("local build done", "build_id", id, "round_id", b.RoundID, "attempt", b.Attempt, "shard_id", b.ShardID)
	writeJSON(w, http.StatusOK, map[string]string{"build_id": id, "state": "done"})
}

// localBuildFail: the shard's local build was refused by the recall gate (or
// could not run). The same guard as done: only a running local build.
func (s *Server) localBuildFail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req LocalFailRequest
	if !decode(w, r, &req) {
		return
	}
	if req.Reason == "" {
		writeError(w, http.StatusBadRequest, "reason is required")
		return
	}
	if len(req.Reason) > 500 {
		req.Reason = req.Reason[:500]
	}
	ctx := r.Context()
	ok, err := s.st.FailLocal(ctx, id, req.Reason)
	if err != nil {
		s.internal(w, "fail local", err)
		return
	}
	if !ok {
		b, found, err := s.st.GetBuild(ctx, id)
		if err != nil {
			s.internal(w, "get build", err)
			return
		}
		if !found {
			writeError(w, http.StatusNotFound, "no such build")
			return
		}
		writeError(w, http.StatusConflict, "build is "+b.Placement+"/"+b.State+", not a running local build")
		return
	}
	b, _, _ := s.st.GetBuild(ctx, id)
	s.log.Warn("local build failed", "build_id", id, "round_id", b.RoundID, "attempt", b.Attempt, "shard_id", b.ShardID, "reason", req.Reason)
	writeJSON(w, http.StatusOK, map[string]string{"build_id": id, "state": "failed"})
}

// ---- follow-up jobs -------------------------------------------------------

func (s *Server) jobStart(w http.ResponseWriter, r *http.Request) {
	s.jobEvent(w, r, s.st.StartJob, "started")
}
func (s *Server) jobDone(w http.ResponseWriter, r *http.Request) {
	s.jobEvent(w, r, s.st.FinishJob, "finished")
}

func (s *Server) jobEvent(w http.ResponseWriter, r *http.Request,
	op func(context.Context, int64, int32, int32) (bool, error), what string) {
	round, ok := pathInt64(w, r, "round")
	if !ok {
		return
	}
	shard, ok := pathInt32(w, r, "shard")
	if !ok {
		return
	}
	seq, ok := pathInt32(w, r, "seq")
	if !ok {
		return
	}
	done, err := op(r.Context(), round, shard, seq)
	if err != nil {
		s.internal(w, "job "+what, err)
		return
	}
	if !done {
		j, found, err := s.st.GetJob(r.Context(), round, shard, seq)
		if err != nil {
			s.internal(w, "get job", err)
			return
		}
		if !found {
			writeError(w, http.StatusNotFound, "no such job")
			return
		}
		writeError(w, http.StatusConflict, fmt.Sprintf("job cannot be %s: started_at=%v finished_at=%v", what, j.StartedAt, j.FinishedAt))
		return
	}
	s.log.Info("query "+what, "round_id", round, "shard_id", shard, "seq", seq,
		"build_id", store.BuildID(shard, round))
	writeJSON(w, http.StatusOK, map[string]any{"round_id": round, "shard_id": shard, "seq": seq, "event": what})
}

// ---- workers, health ------------------------------------------------------

// WorkerView is a registered worker plus whether the pool currently counts
// it: live means seen within WorkerStaleAfter. The list keeps stale workers
// so an operator can see who has gone quiet.
type WorkerView struct {
	store.Worker
	Live bool `json:"live"`
}

type WorkersResponse struct {
	Pool    store.PoolState `json:"pool"`
	Workers []WorkerView    `json:"workers"`
}

func (s *Server) workers(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	pool, err := s.st.Pool(ctx, s.cfg.WorkerStaleAfter)
	if err != nil {
		s.internal(w, "pool", err)
		return
	}
	ws, err := s.st.ListWorkers(ctx)
	if err != nil {
		s.internal(w, "list workers", err)
		return
	}
	views := make([]WorkerView, 0, len(ws))
	for _, x := range ws {
		views = append(views, WorkerView{Worker: x, Live: time.Since(x.LastSeen) <= s.cfg.WorkerStaleAfter})
	}
	writeJSON(w, http.StatusOK, WorkersResponse{Pool: pool, Workers: views})
}

func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.st.Ping(ctx); err != nil {
		writeError(w, http.StatusServiceUnavailable, "postgres: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ---- helpers --------------------------------------------------------------

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, "bad json: "+err.Error())
		return false
	}
	return true
}

func pathInt64(w http.ResponseWriter, r *http.Request, name string) (int64, bool) {
	v, err := strconv.ParseInt(r.PathValue(name), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, name+" must be an integer")
		return 0, false
	}
	return v, true
}

func pathInt32(w http.ResponseWriter, r *http.Request, name string) (int32, bool) {
	v, err := strconv.ParseInt(r.PathValue(name), 10, 32)
	if err != nil {
		writeError(w, http.StatusBadRequest, name+" must be a 32-bit integer")
		return 0, false
	}
	return int32(v), true
}

func (s *Server) internal(w http.ResponseWriter, what string, err error) {
	if errors.Is(err, context.Canceled) {
		return
	}
	s.log.Error(what+" failed", "err", err)
	writeError(w, http.StatusInternalServerError, what+" failed")
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}
