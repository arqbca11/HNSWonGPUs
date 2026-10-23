package crosscheck

// Phase 3 steps 4–6 cross-check: the discrete-event simulator, black-box.
// Spec: docs/design/phase3-policies.md decisions 12–25 (21 fidelity, 22 link
// sharing, 23 noise, 24 runner, 25 presets) and the steps 4–6 notes; the
// roadmap's Phase 3 section; CLAUDE.md invariants 8 and 10 and the modeling
// assumptions; `go doc -all ./experiments/sim` (its doc comments are spec).
// Pure: no database, no cluster, no env gate.
//
// Hand numbers used throughout (default cost model, see sim_helpers_test.go):
//   n=100k  dim=128 : CPU 20.6313  GPU 4.0631  xfer 0.5760  total 4.6391  mem 102.4 MB
//   n=150k  dim=128 : CPU 32.0358  GPU 5.2036  xfer 0.8640  total 6.0676
//   n=100k  dim=1536: CPU 247.58   GPU 26.758  xfer 6.2080  total 32.966

import (
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/experiments/sim"
	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
	"github.com/arqbca11/HNSWonGPUs/scheduler/costmodel"
	"github.com/arqbca11/HNSWonGPUs/scheduler/policy"
)

func TestSimSpec(t *testing.T) {
	big := int64(16 * gib)
	never := policy.StaticThreshold{N: 1 << 50} // every build local
	L := sxCPU(100_000, 128)
	T := sxGPUTotal(100_000, 128)

	t.Run("CostModelDefaultsAreTheDocumentedOnes", func(t *testing.T) {
		m := costmodel.Default()
		t.Logf("costmodel.Default() = %+v", m)
		if m.A != 1.4e-7 || m.Speedup != 10 || m.GPUOverhead != 2 || m.Bandwidth != 200e6 || m.M != 16 || m.MemFactor != 2 || m.BytesPerDim != 4 {
			t.Fatalf("defaults differ from the documented A=1.4e-7, Speedup=10, GPUOverhead=2, Bandwidth=200e6, M=16, MemFactor=2: every hand number below would be wrong")
		}
		sxExpect(t, "CPUBuild(100k,128)", m.CPUBuild(100_000, 128).Seconds(), L, "costmodel doc: A*n*ln n*dim")
		sxExpect(t, "GPUTotal(100k,128)", m.GPUTotal(100_000, 128).Seconds(), T, "costmodel doc: GPU build + transfer")
		if got := m.GPUMemBytes(100_000, 128); got != sxMem(100_000, 128) {
			t.Errorf("GPUMemBytes = %d, want %d", got, sxMem(100_000, 128))
		}
	})

	t.Run("LocalRoundByHand", func(t *testing.T) {
		// One shard, local: the CPU builds for L, queries queue behind it
		// (modeling assumption: a local build occupies the CPU), then drain
		// in arrival order. The local index is ready the moment the shard's
		// own build ends.
		w := sxWorkload(t, 128, 1, big, 2.5,
			sxShard(0, 100_000, sxQ(1, 2, false), sxQ(5, 3, true)))
		r := sxRun(t, "local, DDL at 0", w, never, sim.Config{})
		if !r.Completed {
			t.Fatal("round did not complete")
		}
		sxExpect(t, "RoundS", r.RoundS, L+5, "decision 30/21: latest build-or-query finish = build 20.6313 + 2 s + 3 s")
		sxExpectSet(t, "ShardFinishS", r.ShardFinishS, []float64{L + 5}, "Result doc: max(build finish, last query finish)")
		sxExpect(t, "StragglerLagS", r.StragglerLagS, 0, "one shard: last minus median = 0")
		if r.BuildsLocal != 1 || r.BuildsGPU != 0 || len(r.QueueWaits) != 0 {
			t.Errorf("want 1 local build and no queue waits (only claimed builds have one); got local=%d gpu=%d waits=%v", r.BuildsLocal, r.BuildsGPU, r.QueueWaits)
		}

		// The same shard receiving its DDL 5 s into the round: round time is
		// measured from round start, not from the shard's DDL.
		w2 := sxWorkload(t, 128, 1, big, 2.5,
			sxShard(5, 100_000, sxQ(6, 2, false), sxQ(10, 3, true)))
		r2 := sxRun(t, "local, DDL at 5", w2, never, sim.Config{})
		sxExpect(t, "RoundS with DDL at 5 s", r2.RoundS, 5+L+5, "modeling assumption: round begins at the DDL pattern's origin; staggered shards start late")
	})

	t.Run("GPURoundIndexQueryWaitsForNextReport", func(t *testing.T) {
		// always_gpu, one idle worker: claimed at 0, done at T. A free query
		// runs on arrival; an index query waits for the build AND for the
		// shard's next report (decision 21: the shard learns at its own next
		// report; decision 29: the report is the sync point).
		for _, poll := range []float64{0.5, 0.3, 2.0} {
			w := sxWorkload(t, 128, 1, big, 1,
				sxShard(0, 100_000, sxQ(1, 1, false), sxQ(2, 1, true)))
			r := sxRun(t, "always_gpu", w, policy.AlwaysGPU{}, sim.Config{PollS: poll})
			rep := sxNextReport(T, poll)
			t.Logf("PollS=%.1f: build done at %.4f, first report at/after it %.4f, index query runs %.4f..%.4f", poll, T, rep, rep, rep+1)
			sxExpect(t, "RoundS", r.RoundS, rep+1, "decision 21: report latency is in the model")
			if r.BuildsGPU != 1 || r.BuildsLocal != 0 {
				t.Errorf("want 1 GPU build, got gpu=%d local=%d", r.BuildsGPU, r.BuildsLocal)
			}
			sxExpectSet(t, "QueueWaits", r.QueueWaits, []float64{0}, "claimed at once by the idle worker")
		}
		// No queries at all: the shard's finish is the build's finish, not the
		// report that tells the shard (decision 30: build terminal).
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000))
		r := sxRun(t, "always_gpu, empty stream", w, policy.AlwaysGPU{}, sim.Config{})
		sxExpect(t, "RoundS (no queries)", r.RoundS, T, "round = latest real-build finish when nothing else runs")
	})

	t.Run("FCFSQueueWaitsAndInterpolatedMedian", func(t *testing.T) {
		// One worker, equal builds submitted together: FCFS (always_gpu,
		// priority 0, enqueued order). Finishes T,2T,3T,4T; waits 0,T,2T,3T.
		// Straggler lag = last - interpolated median = 4T - 2.5T = 1.5T.
		w := sxWorkload(t, 128, 1, big, 1,
			sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000))
		r := sxRun(t, "4 shards, 1 worker", w, policy.AlwaysGPU{}, sim.Config{})
		sxExpect(t, "RoundS", r.RoundS, 4*T, "FCFS on one worker")
		sxExpectSet(t, "QueueWaits", r.QueueWaits, []float64{0, T, 2 * T, 3 * T}, "Result doc: first claim minus submit")
		sxExpectSet(t, "ShardFinishS", r.ShardFinishS, []float64{T, 2 * T, 3 * T, 4 * T}, "one build per worker at a time")
		sxExpect(t, "StragglerLagS (even count)", r.StragglerLagS, 1.5*T, "decision 21: interpolated median (2T+3T)/2")
		if len(r.ShardFinishS) == 4 {
			inOrder := true
			for i := range r.ShardFinishS {
				if !sxNear(r.ShardFinishS[i], float64(i+1)*T) {
					inOrder = false
				}
			}
			t.Logf("observation: ShardFinishS indexed by shard ID and simultaneous submits enqueued in shard-ID order: %v (the docs do not say which order same-instant submits take)", inOrder)
		}

		w3 := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000))
		r3 := sxRun(t, "3 shards, 1 worker", w3, policy.AlwaysGPU{}, sim.Config{})
		sxExpect(t, "StragglerLagS (odd count)", r3.StragglerLagS, T, "3T - median 2T")
	})

	t.Run("ClaimPriorityThenEnqueuedOrder", func(t *testing.T) {
		// Scripted priorities. s0 (prio 0) takes the only worker at t=0 and
		// holds it until T. While it runs: s3 enqueues at 0.5 (prio 5), s2 at
		// 1.0 (prio 5), s1 at 1.5 (prio 1), s4 at 2.0 (prio 9). Decision 21's
		// claim mirror: priority DESC, then enqueued order, so at T the claims
		// go s4, s3, s2, s1 — finishing 2T, 3T, 4T, 5T.
		sc := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0, 1: 1, 2: 5, 3: 5, 4: 9})}
		w := sxWorkload(t, 128, 1, big, 1,
			sxShard(0, 100_000), sxShard(1.5, 100_000), sxShard(1.0, 100_000), sxShard(0.5, 100_000), sxShard(2.0, 100_000))
		r := sxRun(t, "scripted priorities", w, sc, sim.Config{})
		want := []float64{T, 5 * T, 4 * T, 3 * T, 2 * T}
		if len(r.ShardFinishS) != 5 {
			t.Fatalf("ShardFinishS has %d entries", len(r.ShardFinishS))
		}
		for i, wv := range want {
			sxExpect(t, "finish of shard "+itoa(int64(i)), r.ShardFinishS[i], wv, "decision 21: priority DESC, enqueued seq within")
		}
	})

	t.Run("PreemptedBuildKeepsSubmitTimeSeq", func(t *testing.T) {
		// s0 submits local at 0; s1 (400k, prio 100) holds the only worker
		// from 0 to T1; s2 (0.5 s) and s3 (1.0 s) queue at prio 5. At the first
		// tick the script preempts s0 at prio 5. If s0 keeps its insert-time
		// seq (decision 21, mirroring decision 31's enqueued_at), it claims
		// before s2 and s3 when s1 finishes: s0 at T1+T, s2 at T1+2T, s3 at
		// T1+3T. Its queue wait counts from its submit (0), not the preemption.
		T1 := sxGPUTotal(400_000, 128)
		sc := &sxScript{
			decide: sxGPUAt(map[int32]float64{1: 100, 2: 5, 3: 5}),
			recon: func(in policy.ReconsiderInput) []policy.Action {
				if b, ok := sxFindView(in.Builds, 0, false); ok && b.Phase == "local_running" && !b.Moved {
					return []policy.Action{{Kind: policy.ActPreempt, BuildID: b.BuildID, Priority: 5, Reason: "script"}}
				}
				return nil
			},
		}
		w := sxWorkload(t, 128, 1, big, 1,
			sxShard(0, 100_000), sxShard(0, 400_000), sxShard(0.5, 100_000), sxShard(1.0, 100_000))
		r := sxRun(t, "scripted preempt", w, sc, sim.Config{})
		t.Logf("T1 (400k GPU total) = %.4f", T1)
		if r.Preempts != 1 {
			t.Errorf("Preempts = %d, want 1", r.Preempts)
		}
		if len(r.ShardFinishS) == 4 {
			sxExpect(t, "shard 0 (preempted) finish", r.ShardFinishS[0], T1+T, "decision 21: seq kept across a preemption, so it claims first among prio 5")
			sxExpect(t, "shard 2 finish", r.ShardFinishS[2], T1+2*T, "enqueued at 0.5")
			sxExpect(t, "shard 3 finish", r.ShardFinishS[3], T1+3*T, "enqueued at 1.0")
		}
		sxExpectSet(t, "QueueWaits", r.QueueWaits, []float64{0, T1, T1 + T - 0.5, T1 + 2*T - 1.0}, "first claim minus SUBMIT (decision 31: enqueued_at keeps the submit time)")
	})

	t.Run("StaleAndForbiddenActionsAreSkipped", func(t *testing.T) {
		// The script asks, every tick, to preempt AND renege every build in
		// every phase. Guarded transitions (decision 21, mirroring decisions
		// 4, 8, 20, 31): a leased build is never reneged or preempted; a queued
		// GPU build cannot be "preempted"; once reneged (moved), the build is
		// never preempted (PreemptToGPU keeps its moved guard absolutely).
		sc := &sxScript{
			decide: sxGPUAt(map[int32]float64{0: 0, 1: 0}),
			recon: func(in policy.ReconsiderInput) []policy.Action {
				var acts []policy.Action
				for _, b := range in.Builds {
					acts = append(acts,
						policy.Action{Kind: policy.ActPreempt, BuildID: b.BuildID, Priority: 1, Reason: "script"},
						policy.Action{Kind: policy.ActRenege, BuildID: b.BuildID, Reason: "script"})
				}
				return acts
			},
		}
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000))
		r := sxRun(t, "preempt+renege everything", w, sc, sim.Config{})
		if !r.Completed {
			t.Fatal("round did not complete")
		}
		if r.Reneges != 1 || r.Preempts != 0 {
			t.Errorf("Reneges=%d Preempts=%d, want 1 and 0: only the queued s1 can be reneged; the leased s0 and the moved s1 never move again", r.Reneges, r.Preempts)
		} else {
			t.Logf("ok: exactly one renege (the queued build), zero preempts (leased never moved; reneged build never preempted)")
		}
		if r.BuildsGPU != 1 || r.BuildsLocal != 1 {
			t.Errorf("gpu=%d local=%d, want 1 and 1", r.BuildsGPU, r.BuildsLocal)
		}
		// s1 reneged at the first tick (2 s); the shard starts the local build
		// at its next report (decision 21). Report at the tick instant or the
		// next grid point: finish in {2+L, 2.5+L}.
		if len(r.ShardFinishS) == 2 {
			f := r.ShardFinishS[1]
			t.Logf("reneged shard finished at %.4f; 2+L=%.4f, 2.5+L=%.4f", f, 2+L, 2.5+L)
			if !(sxNear(f, 2+L) || sxNear(f, 2.5+L)) {
				t.Errorf("reneged shard finish %.4f is neither tick+L nor next report+L", f)
			}
		}
		if len(sc.ticks) > 0 {
			t.Logf("ticks observed by the policy: %d over a %.2f s round", len(sc.ticks), r.RoundS)
		}
	})

	t.Run("OneCopyEverAndNoPreemptWhileCopyInFlight", func(t *testing.T) {
		// Script: speculate shard 0 on every tick; once a copy is visible,
		// also ask to preempt shard 0. Decision 15: one copy ever per build;
		// a primary whose copy is in flight is never preempted.
		mk := func() *sxScript {
			return &sxScript{
				decide: sxLocal,
				recon: func(in policy.ReconsiderInput) []policy.Action {
					b, ok := sxFindView(in.Builds, 0, false)
					if !ok || b.Phase != "local_running" {
						return nil
					}
					acts := []policy.Action{{Kind: policy.ActSpeculate, BuildID: b.BuildID, Priority: 1, Reason: "script"}}
					if _, copyVisible := sxFindView(in.Builds, 0, true); copyVisible {
						acts = append(acts, policy.Action{Kind: policy.ActPreempt, BuildID: b.BuildID, Priority: 1, Reason: "script"})
					}
					return acts
				},
			}
		}
		// GPU wins (speedup 10): copy at tick 2, done 2+6.0676.
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000))
		r := sxRun(t, "speedup 10", w, mk(), sim.Config{})
		if r.Speculations != 1 || r.Preempts != 0 || r.SpecGPUWins != 1 {
			t.Errorf("speculations=%d preempts=%d gpuWins=%d, want 1, 0, 1", r.Speculations, r.Preempts, r.SpecGPUWins)
		}
		// Local wins (speedup 1): the copy is cancelled; the script keeps asking
		// for a copy every tick until the primary finishes. Still one ever.
		w2 := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000))
		r2 := sxRun(t, "speedup 1", w2, mk(), sim.Config{CostModel: sxModel(1)})
		if r2.Speculations != 1 || r2.Preempts != 0 || r2.SpecLocalWins != 1 {
			t.Errorf("speculations=%d preempts=%d localWins=%d, want 1, 0, 1", r2.Speculations, r2.Preempts, r2.SpecLocalWins)
		}
	})

	t.Run("SubmitSeesNewJobWaitColdStartAndInf", func(t *testing.T) {
		// Decision 13 as the simulator applies it (decision 21: submits mapped
		// like scheduler/api): Pool.QueueWaitS = NewJobWait — joining at the
		// back. Three equal builds at t=0, one worker: 0, T, 2T.
		sc := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0, 1: 0, 2: 0})}
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000))
		sxRun(t, "three submits", w, sc, sim.Config{})
		var got []float64
		for _, in := range sc.inputs {
			got = append(got, in.Pool.QueueWaitS)
			t.Logf("submit of shard %d: Pool=%+v Est=%+v Status.Valid=%v", in.Build.ShardID, in.Pool, in.Est, in.Status.Valid)
		}
		sxExpectSet(t, "submit-time QueueWaitS", got, []float64{0, T, 2 * T}, "decision 13: behind every queued job")
		if len(sc.inputs) > 0 {
			in := sc.inputs[0]
			sxExpect(t, "Est.LocalBuildS", in.Est.LocalBuildS, L, "sigma 0: the truth")
			sxExpect(t, "Est.GPUBuildS", in.Est.GPUBuildS, sxGPUBuild(100_000, 128, 10), "sigma 0: the truth")
			sxExpect(t, "Est.TransferS", in.Est.TransferS, sxXfer(100_000, 128), "sigma 0: the truth")
			if in.Build.MemBytes != sxMem(100_000, 128) || in.Pool.LargestMem != big || in.Pool.LiveWorkers != 1 {
				t.Errorf("BuildSpec/Pool mapping off: %+v %+v", in.Build, in.Pool)
			}
		}

		// Empty pool: cold-start rule keeps 0 (decision 13/19).
		sc0 := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0})}
		w0 := sxWorkload(t, 128, 0, big, 1, sxShard(0, 100_000))
		r0 := sxRun(t, "empty pool", w0, sc0, sim.Config{MaxModeledS: 200})
		if len(sc0.inputs) == 1 {
			sxExpect(t, "empty-pool QueueWaitS", sc0.inputs[0].Pool.QueueWaitS, 0, "cold-start rule")
			if sc0.inputs[0].Pool.LiveWorkers != 0 {
				t.Errorf("LiveWorkers = %d with an empty pool", sc0.inputs[0].Pool.LiveWorkers)
			}
		}
		if r0.Completed {
			t.Errorf("a GPU build with zero workers forever completed")
		}

		// Fits nowhere: +Inf (decision 13).
		scI := &sxScript{decide: sxLocal}
		wI := sxWorkload(t, 128, 1, 50_000_000, 1, sxShard(0, 100_000))
		sxRun(t, "fits nowhere", wI, scI, sim.Config{})
		if len(scI.inputs) == 1 && !math.IsInf(scI.inputs[0].Pool.QueueWaitS, 1) {
			t.Errorf("fits-nowhere QueueWaitS = %v, want +Inf", scI.inputs[0].Pool.QueueWaitS)
		} else {
			t.Logf("ok: a 102.4 MB build against a 50 MB worker sees QueueWaitS=+Inf")
		}
	})

	t.Run("NodeLinkSharingThreeModes", func(t *testing.T) {
		// Decision 22 with decision 12's fluid. Two 100k x 1536 builds on two
		// idle workers, transfer X=6.208, build G=26.758.
		X, G := sxXfer(100_000, 1536), sxGPUBuild(100_000, 1536, 10)
		t.Logf("X=%.4f G=%.4f", X, G)
		two := func() workload.Workload {
			return sxWorkload(t, 1536, 2, big, 1, sxShard(0, 100_000), sxShard(0, 100_000))
		}
		r := sxRun(t, "one node, sharing on", two(), policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: true})
		sxExpectSet(t, "finishes", r.ShardFinishS, []float64{2*X + G, 2*X + G}, "each transfer at 1/2 rate: both end at 2X")
		r = sxRun(t, "two nodes, sharing on", two(), policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 1, ShareNodeLinks: true})
		sxExpectSet(t, "finishes", r.ShardFinishS, []float64{X + G, X + G}, "a link each")
		r = sxRun(t, "one node, sharing off", two(), policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: false})
		sxExpectSet(t, "finishes", r.ShardFinishS, []float64{X + G, X + G}, "off = the fake worker that sleeps GPUTotal")

		// Staggered join: s1 arrives at 3 s while s0 is mid-transfer. Alone
		// 0..3, s0 has X-3 left; shared at 1/2 it ends at 3+2(X-3)=2X-3; s1 has
		// moved X-3 by then and finishes its last 3 s alone: 2X.
		ws := sxWorkload(t, 1536, 2, big, 1, sxShard(0, 100_000), sxShard(3, 100_000))
		r = sxRun(t, "staggered join, one node", ws, policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: true})
		if len(r.ShardFinishS) == 2 {
			sxExpect(t, "s0 finish", r.ShardFinishS[0], 2*X-3+G, "fluid: 1/k while k transfers run")
			sxExpect(t, "s1 finish", r.ShardFinishS[1], 2*X+G, "fluid: back to full rate after s0 leaves")
		}
	})

	t.Run("GPUExecutionIndependentOfReportAndTickGrid", func(t *testing.T) {
		// Execution is physics: with always_gpu (no reconsider actions) and no
		// queries, a build's finish cannot depend on how often shards report
		// or the policy ticks — those only change what policies SEE (decision
		// 21). Two 100k x 1536 transfers sharing one node, the second joining
		// at 1 s; then both starting together.
		X, G := sxXfer(100_000, 1536), sxGPUBuild(100_000, 1536, 10)
		for _, ddl1 := range []float64{1, 0} {
			var base []float64
			for _, grid := range [][2]float64{{0.5, 2}, {0.37, 2}, {0.5, 3.3}, {7, 50}} {
				w := sxWorkload(t, 1536, 2, big, 1, sxShard(0, 100_000), sxShard(ddl1, 100_000))
				r := sxRun(t, "s1 DDL "+itoa(int64(ddl1))+" PollS/ReconsiderS "+fmtF(grid[0])+"/"+fmtF(grid[1]), w, policy.AlwaysGPU{},
					sim.Config{WorkersPerNode: 2, ShareNodeLinks: true, PollS: grid[0], ReconsiderS: grid[1]})
				if base == nil {
					base = r.ShardFinishS
					continue
				}
				for i := range base {
					if !sxNear(base[i], r.ShardFinishS[i]) {
						t.Errorf("s1 DDL %.0f: shard %d finishes at %.4f with PollS/ReconsiderS %v but %.4f with 0.5/2 — GPU execution depends on the report grid", ddl1, i, r.ShardFinishS[i], grid, base[i])
					}
				}
			}
			if ddl1 == 0 {
				sxExpectSet(t, "simultaneous finishes", base, []float64{2*X + G, 2*X + G}, "decision 12 fluid: both transfers end at 2X, both builds then take G")
			} else {
				sxExpectSet(t, "staggered finishes", base, []float64{2*X - 1 + G, 2*X + G}, "decision 12 fluid, hand-replayed")
			}
		}
	})

	t.Run("SharingNeverSpeedsABuildSixEqualBuilds", func(t *testing.T) {
		// Property: sharing a link only slows transfers (decision 12). Six
		// equal 100k x 1536 builds, no queries, always_gpu, two workers on one
		// node. Off: pairs finish at (k+1)(X+G). On: each pair transfers
		// together (both end at 2X) then builds: (k+1)(2X+G). No shard may
		// finish earlier with sharing on, and none before one GPU build G.
		// (A first version used the uniform preset, whose finishes are set by
		// its query horizon and hide the GPU path.)
		X, G := sxXfer(100_000, 1536), sxGPUBuild(100_000, 1536, 10)
		mk := func() workload.Workload {
			var ss []workload.ShardWorkload
			for i := 0; i < 6; i++ {
				ss = append(ss, sxShard(0, 100_000))
			}
			return sxWorkload(t, 1536, 2, big, 1, ss...)
		}
		off := sxRun(t, "sharing off", mk(), policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: false})
		on := sxRun(t, "sharing on", mk(), policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: true})
		var wantOff, wantOn []float64
		for k := 1; k <= 3; k++ {
			wantOff = append(wantOff, float64(k)*(X+G), float64(k)*(X+G))
			wantOn = append(wantOn, float64(k)*(2*X+G), float64(k)*(2*X+G))
		}
		sxExpectSet(t, "sharing off", off.ShardFinishS, wantOff, "fake-worker model: pairs of X+G")
		sxExpectSet(t, "sharing on", on.ShardFinishS, wantOn, "fluid model: pairs of 2X+G")
		for i := range on.ShardFinishS {
			if on.ShardFinishS[i] < off.ShardFinishS[i]-sxEps {
				t.Errorf("shard %d finishes at %.4f with sharing ON but %.4f with it off — sharing made it faster", i, on.ShardFinishS[i], off.ShardFinishS[i])
			}
			if on.ShardFinishS[i] < G {
				t.Errorf("shard %d finished at %.4f, before even one GPU build (%.4f) could run", i, on.ShardFinishS[i], G)
			}
		}
	})

	t.Run("ReplayQueuePredictsTheSimulator", func(t *testing.T) {
		// Cross-validation: no noise, sharing on, five builds of different
		// sizes at t=0 on two workers sharing one node, always_gpu (FCFS).
		// Step 2's ReplayQueue over the t=0 queue should predict exactly what
		// the simulator then executes: decision 12's model IS decision 22's
		// execution when no worker is busy at the snapshot.
		sizes := []int64{100_000, 200_000, 50_000, 150_000, 80_000}
		var shards []workload.ShardWorkload
		var queue []policy.QueuedJob
		for i, n := range sizes {
			shards = append(shards, sxShard(0, n))
			queue = append(queue, policy.QueuedJob{BuildID: "s" + itoa(int64(i)), Priority: 0, Seq: i,
				GPUBuildS: sxGPUBuild(n, 1536, 10), TransferS: sxXfer(n, 1536), MemBytes: sxMem(n, 1536)})
		}
		workers := []policy.WorkerView{{ID: "w0", Node: "n0", MemBytes: big}, {ID: "w1", Node: "n0", MemBytes: big}}
		pred := policy.ReplayQueue(workers, queue)
		w := sxWorkload(t, 1536, 2, big, 1, shards...)
		r := sxRun(t, "5 builds, 2 workers, 1 node", w, policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: true})
		if len(r.ShardFinishS) != 5 {
			t.Fatalf("ShardFinishS len %d", len(r.ShardFinishS))
		}
		var want []float64
		for i := range sizes {
			e := pred["s"+itoa(int64(i))]
			want = append(want, e.DoneS)
			t.Logf("shard %d (%dk): replay start %.4f done %.4f | sim finish %.4f", i, sizes[i]/1000, e.StartS, e.DoneS, r.ShardFinishS[i])
		}
		sxExpectSet(t, "sim finishes vs replay DoneS", r.ShardFinishS, want, "decision 12 = decision 22 with sharing on, no noise")
		for i := range sizes {
			if !sxNear(r.ShardFinishS[i], want[i]) {
				t.Errorf("shard %d: sim %.6f vs replay %.6f", i, r.ShardFinishS[i], want[i])
			}
		}

		// Where they legitimately disagree: busy-worker opacity (decision 12's
		// pinned approximation). s1 arrives at 1 s while w0 is mid-transfer.
		// The replay at t=1 sees w0 as one opaque lump, so it predicts s1 at
		// full link rate: 1+X+G. The simulator executes the sharing: s0 has
		// X-1 left at t=1, shares until 1+2(X-1)=2X-1, s1 then has 1 s left:
		// s1 ends its transfer at 2X, done 2X+G.
		X, G := sxXfer(100_000, 1536), sxGPUBuild(100_000, 1536, 10)
		p2 := policy.ReplayQueue(
			[]policy.WorkerView{{ID: "w0", Node: "n0", MemBytes: big, BusyRemainingS: X + G - 1}, {ID: "w1", Node: "n0", MemBytes: big}},
			[]policy.QueuedJob{{BuildID: "s1", GPUBuildS: G, TransferS: X, MemBytes: sxMem(100_000, 1536)}})
		predS1 := 1 + p2["s1"].DoneS
		w2 := sxWorkload(t, 1536, 2, big, 1, sxShard(0, 100_000), sxShard(1, 100_000))
		r2 := sxRun(t, "s1 joins at 1 s", w2, policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: true})
		sxExpect(t, "replay's prediction for s1 (absolute)", predS1, 1+X+G, "busy worker opaque: no contention seen")
		if len(r2.ShardFinishS) == 2 {
			sxExpect(t, "sim's s1 finish", r2.ShardFinishS[1], 2*X+G, "execution shares the link with s0's in-flight transfer")
			t.Logf("legitimate disagreement = %.4f s (= X-1): the estimator's documented opacity, not a bug", r2.ShardFinishS[1]-predS1)
		}
		r3 := sxRun(t, "s1 joins at 1 s, sharing off", w2, policy.AlwaysGPU{}, sim.Config{WorkersPerNode: 2, ShareNodeLinks: false})
		if len(r3.ShardFinishS) == 2 {
			sxExpect(t, "sim's s1 finish with sharing off", r3.ShardFinishS[1], predS1, "with sharing off the opacity costs nothing: replay exact")
		}
	})

	t.Run("ReactivePreemptsHurtingLocalBuild", func(t *testing.T) {
		// 150k < DefaultThresholdN → local at submit. 20 one-second free
		// queries arrive in the first 0.2 s and pile up behind the CPU build.
		// Stay ≈ 32.04 + 20 = 52; migrate = max(20, 6.07) = 20 → preempt at a
		// tick. The CPU is freed at the shard's report (decision 21), the queue
		// drains 20 s from there; the GPU copy runs in parallel.
		var qs []workload.Query
		for i := 0; i < 20; i++ {
			qs = append(qs, sxQ(0.01*float64(i), 1, false))
		}
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000, qs...))
		r := sxRun(t, "reactive", w, policy.NewReactive(policy.ReactiveConfig{}), sim.Config{})
		if r.Preempts != 1 || r.BuildsGPU != 1 || r.BuildsLocal != 0 {
			t.Errorf("preempts=%d gpu=%d local=%d, want 1/1/0", r.Preempts, r.BuildsGPU, r.BuildsLocal)
		}
		if len(r.QueueWaits) == 1 {
			qw := r.QueueWaits[0]
			t.Logf("queue wait of the preempted build = %.4f: claim time minus submit (0) — equals the preempt tick when a worker is idle", qw)
			if math.Abs(qw-math.Round(qw/2)*2) > sxEps || qw < 2-sxEps {
				t.Errorf("queue wait %.4f is not a tick instant ≥ 2: was it measured from the preemption instead of the submit?", qw)
			}
			// RoundS: CPU freed at the report at/after the tick, + 20 s drain.
			tick := qw
			if !(sxNear(r.RoundS, tick+20) || sxNear(r.RoundS, tick+0.5+20)) {
				t.Errorf("RoundS %.4f, want tick+20 or next report+20 (tick %.1f)", r.RoundS, tick)
			} else {
				t.Logf("ok: RoundS %.4f = CPU freed at %.1f + 20 s of queued queries (stay-local would have been ≈ %.2f)", r.RoundS, r.RoundS-20, sxCPU(150_000, 128)+20)
			}
		} else {
			t.Errorf("QueueWaits = %v, want exactly one (the preempted build's claim)", r.QueueWaits)
		}
	})

	t.Run("NoPreemptOfBuildThatFitsNoWorker", func(t *testing.T) {
		// Bug 11's cure in the simulator: a hurting local build that fits no
		// live worker is never preempted. 200k x 128 needs 204.8 MB; the
		// worker has 100 MB. always_gpu also places it local (memory filter).
		var qs []workload.Query
		for i := 0; i < 20; i++ {
			qs = append(qs, sxQ(0.01*float64(i), 1, false))
		}
		// (First version also had a fitting 50k shard here; reactive rightly
		// preempted THAT one — a test error, so the reactive run is alone.)
		w := sxWorkload(t, 128, 1, 100_000_000, 1, sxShard(0, 200_000, qs...))
		r := sxRun(t, "reactive, too big", w, policy.NewReactive(policy.ReactiveConfig{}), sim.Config{})
		if r.Preempts != 0 || !r.Completed || r.BuildsLocal != 1 {
			t.Errorf("preempts=%d completed=%v local=%d: a fits-nowhere build must never be preempted (bug 11)", r.Preempts, r.Completed, r.BuildsLocal)
		}
		sxExpect(t, "too-big shard's finish", r.ShardFinishS[0], sxCPU(200_000, 128)+20, "builds locally to the end, then drains")
		w2 := sxWorkload(t, 128, 1, 100_000_000, 1, sxShard(0, 200_000), sxShard(0, 50_000))
		r2 := sxRun(t, "always_gpu, too big", w2, policy.AlwaysGPU{}, sim.Config{})
		if r2.BuildsLocal != 1 || r2.BuildsGPU != 1 {
			t.Errorf("always_gpu: local=%d gpu=%d, want 1/1 (memory filter at submit)", r2.BuildsLocal, r2.BuildsGPU)
		}
	})

	t.Run("SpeculationGPUWinsBackdatedAndIndexWaitsForResolve", func(t *testing.T) {
		// reactive_spec with an unreachable margin: no preempt, no renege; the
		// only move is a copy onto the idle worker when the queue is empty.
		// Copy at the first tick (2 s), done at 2+6.0676. Decision 16: the
		// primary is done at the copy's finish (backdated). Decision 17:
		// copies never appear in queue waits; the primary was never claimed.
		pol := policy.NewReactiveSpec(policy.ReactiveConfig{MarginS: 1e9})
		T150 := sxGPUTotal(150_000, 128)
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000))
		r := sxRun(t, "no queries", w, pol, sim.Config{})
		if r.Speculations != 1 || r.SpecGPUWins != 1 || r.SpecLocalWins != 0 {
			t.Errorf("spec=%d gpuWins=%d localWins=%d, want 1/1/0", r.Speculations, r.SpecGPUWins, r.SpecLocalWins)
		}
		sxExpect(t, "RoundS", r.RoundS, 2+T150, "decision 16: backdated to the copy's finish, not the resolving tick (10 s)")
		sxExpect(t, "WastedGPUS", r.WastedGPUS, 0, "the GPU copy won; nothing on a worker was wasted")
		if r.BuildsGPU != 1 || r.BuildsLocal != 0 || len(r.QueueWaits) != 0 {
			t.Errorf("gpu=%d local=%d waits=%v: want the primary counted on the GPU path and no queue wait (never claimed; copy invisible, decision 17)", r.BuildsGPU, r.BuildsLocal, r.QueueWaits)
		}

		// With an index query waiting: the shard can only run it once it
		// learns its build is done — after the tick that settles the race
		// (the first tick after 8.07 is 10) and at its own report.
		w2 := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000, sxQ(1, 1, true)))
		r2 := sxRun(t, "one index query", w2, pol, sim.Config{})
		t.Logf("index query finished with the round at %.4f; settle tick 10 → report at 10 or 10.5 → +1 s", r2.RoundS)
		if !(sxNear(r2.RoundS, 11) || sxNear(r2.RoundS, 11.5)) {
			t.Errorf("RoundS %.4f: want 11.0 or 11.5 (the race is settled at a tick, decision 14/16, and the shard learns at a report)", r2.RoundS)
		}
	})

	t.Run("SpeculationLocalWinsWastedWorkerSeconds", func(t *testing.T) {
		// Speedup 1: the copy (2 + 34.9) cannot beat the local build (32.04).
		// The local build wins at 32.04; the next tick (34) cancels the copy
		// and frees the worker (decision 21's simplification): 32 wasted
		// worker-seconds. The round ends at the local finish (decision 17:
		// the copy's cancel does not stretch it).
		pol := policy.NewReactiveSpec(policy.ReactiveConfig{MarginS: 1e9})
		L150 := sxCPU(150_000, 128)
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 150_000))
		r := sxRun(t, "speedup 1", w, pol, sim.Config{CostModel: sxModel(1)})
		if r.Speculations != 1 || r.SpecLocalWins != 1 || r.SpecGPUWins != 0 {
			t.Errorf("spec=%d localWins=%d gpuWins=%d, want 1/1/0", r.Speculations, r.SpecLocalWins, r.SpecGPUWins)
		}
		sxExpect(t, "RoundS", r.RoundS, L150, "the local build's own finish; the copy is invisible")
		sxExpect(t, "WastedGPUS", r.WastedGPUS, 34-2, "claimed at tick 2, cancelled at tick 34")
		if r.BuildsLocal != 1 || r.BuildsGPU != 0 || len(r.QueueWaits) != 0 {
			t.Errorf("local=%d gpu=%d waits=%v, want 1/0/none", r.BuildsLocal, r.BuildsGPU, r.QueueWaits)
		}
	})

	t.Run("SpeculationOneInFlightAtATime", func(t *testing.T) {
		// Two equal local builds, two idle workers. Decision 8 / ReactiveSpec
		// doc: one speculation per tick, never while one is in flight. Copy A
		// at tick 2 (done 8.07); ticks 4, 6, 8 see a copy in flight; tick 10
		// settles A (resolve-then-decide, step 3 G2) and copies B (done
		// 16.07). Both GPU wins, backdated.
		pol := policy.NewReactiveSpec(policy.ReactiveConfig{MarginS: 1e9})
		T150 := sxGPUTotal(150_000, 128)
		w := sxWorkload(t, 128, 2, big, 1, sxShard(0, 150_000), sxShard(0, 150_000))
		r := sxRun(t, "two locals, two workers", w, pol, sim.Config{})
		if r.Speculations != 2 || r.SpecGPUWins != 2 {
			t.Errorf("spec=%d gpuWins=%d, want 2/2", r.Speculations, r.SpecGPUWins)
		}
		sxExpectSet(t, "ShardFinishS", r.ShardFinishS, []float64{2 + T150, 10 + T150}, "second copy only after the first settles at tick 10")
	})

	t.Run("KilledWorkerLeaseRequeuesAfterLeaseS", func(t *testing.T) {
		// always_gpu, two workers, one build claimed by w0 at 0. At 2 s the
		// busiest worker dies; its lease requeues LeaseS (10) later, at 12;
		// the survivor claims at once: done 12 + T. Queue wait is the FIRST
		// claim (0).
		w := sxWorkload(t, 128, 2, big, 1, sxShard(0, 100_000))
		r := sxRun(t, "kill at 2, LeaseS 10", w, policy.AlwaysGPU{}, sim.Config{KillWorkerAtS: 2, LeaseS: 10})
		sxExpect(t, "RoundS", r.RoundS, 12+T, "decision 21: a dead worker's lease requeues after LeaseS")
		sxExpectSet(t, "QueueWaits", r.QueueWaits, []float64{0}, "first claim minus submit")

		// Only worker dies: empty pool, the requeued build holds its place
		// (decision 19) — under always_gpu and under reactive (no renege:
		// waits read 0). The hang shows as Completed=false at the cap.
		for _, p := range []policy.Policy{policy.AlwaysGPU{}, policy.NewReactive(policy.ReactiveConfig{})} {
			w1 := sxWorkload(t, 128, 1, big, 1, sxShard(0, 300_000))
			r1 := sxRun(t, "only worker killed", w1, p, sim.Config{KillWorkerAtS: 2, LeaseS: 10, MaxModeledS: 500})
			if r1.Completed || r1.Reneges != 0 {
				t.Errorf("%s: completed=%v reneges=%d, want an incomplete round and no renege (decision 19: empty pool holds)", p.Name(), r1.Completed, r1.Reneges)
			}
		}
	})

	t.Run("ShrunkenPoolMakesReactiveRenege", func(t *testing.T) {
		// Ten 300k builds (≥ threshold → GPU) on two workers. GPU total 10.51,
		// local 67.80. Untouched, the last wait is 4 totals (42 s): no renege.
		// Kill one worker at 1 s: the replayed waits of the tail exceed local
		// + margin and reneging reacts (decision 19's second clause).
		mk := func() workload.Workload {
			var ss []workload.ShardWorkload
			for i := 0; i < 10; i++ {
				ss = append(ss, sxShard(0, 300_000))
			}
			return sxWorkload(t, 128, 2, big, 1, ss...)
		}
		pol := policy.NewReactive(policy.ReactiveConfig{})
		r0 := sxRun(t, "no kill", mk(), pol, sim.Config{})
		r1 := sxRun(t, "kill at 1", mk(), pol, sim.Config{KillWorkerAtS: 1})
		if r0.Reneges != 0 {
			t.Errorf("no kill: reneges=%d, want 0", r0.Reneges)
		}
		if r1.Reneges < 1 || !r1.Completed || r1.BuildsLocal != r1.Reneges {
			t.Errorf("kill: reneges=%d completed=%v local=%d, want ≥1 renege, completed, local = reneges", r1.Reneges, r1.Completed, r1.BuildsLocal)
		}
	})

	t.Run("NoiseNeverTouchesExecution", func(t *testing.T) {
		// Decision 23: noise only on what policies see. Policies whose
		// decisions ignore estimates must produce identical rounds at any σ.
		for _, sigma := range []float64{0.6, 2} {
			w := sxWorkload(t, 128, 1, big, 2.5, sxShard(0, 100_000, sxQ(1, 2, false), sxQ(5, 3, true)))
			r := sxRun(t, "static local", w, never, sim.Config{EstimateSigma: sigma, NoiseSeed: 3})
			sxExpect(t, "local RoundS under noise", r.RoundS, L+5, "execution uses the truth")
			w4 := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000), sxShard(0, 100_000))
			r4 := sxRun(t, "always_gpu FCFS", w4, policy.AlwaysGPU{}, sim.Config{EstimateSigma: sigma, NoiseSeed: 3})
			sxExpectSet(t, "FCFS finishes under noise", r4.ShardFinishS, []float64{T, 2 * T, 3 * T, 4 * T}, "execution uses the truth")
		}
	})

	t.Run("NoiseThreeFactorsPersistAcrossTicks", func(t *testing.T) {
		// Record what the policy sees. Decision 23: three independent factors
		// per build (local, GPU build, transfer), drawn once and persisting
		// across ticks; also applied to worker busy-remaining.
		sc := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0})} // s0 GPU, s1 local
		w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000))
		sxRun(t, "sigma 0.5", w, sc, sim.Config{EstimateSigma: 0.5, NoiseSeed: 11})
		if len(sc.inputs) != 2 || len(sc.ticks) < 3 {
			t.Fatalf("inputs=%d ticks=%d", len(sc.inputs), len(sc.ticks))
		}
		est := map[int32]policy.Estimates{}
		for _, in := range sc.inputs {
			est[in.Build.ShardID] = in.Est
		}
		e := est[1]
		fL := e.LocalBuildS / L
		fG := e.GPUBuildS / sxGPUBuild(100_000, 128, 10)
		fX := e.TransferS / sxXfer(100_000, 128)
		t.Logf("shard 1 factors: local %.4f, gpu %.4f, transfer %.4f", fL, fG, fX)
		if sxNear(fL, fG) || sxNear(fG, fX) || sxNear(fL, fX) || sxNear(fL, 1) {
			t.Errorf("factors not independent per kind (decision 23): %.4f %.4f %.4f", fL, fG, fX)
		}
		if est[0] == est[1] {
			t.Errorf("two builds got identical noised estimates: factors must be per build")
		}
		for i, tk := range sc.ticks {
			if b, ok := sxFindView(tk.Builds, 1, false); ok && b.Est != e {
				t.Errorf("tick %d: shard 1's estimate %+v differs from its submit-time %+v (decision 23: persists across ticks)", i, b.Est, e)
			}
		}
		t.Logf("ok so far: shard 1's noised estimates identical on submit and on all %d ticks", len(sc.ticks))
		// Busy-remaining at the first tick: truth would be T - tick.
		tk := sc.ticks[0]
		for _, wv := range tk.Workers {
			if wv.BusyRemainingS > 0 {
				noisedTotal := est[0].GPUBuildS + est[0].TransferS
				t.Logf("first tick: worker %s BusyRemainingS=%.4f; truth-based T-2=%.4f; noisedTotal-2=%.4f", wv.ID, wv.BusyRemainingS, T-2, noisedTotal-2)
				if sxNear(wv.BusyRemainingS, T-2) {
					t.Errorf("busy-remaining is the truth: decision 23 says it is noised")
				}
			}
		}
		// Gap probe: WorkerView doc says BusyRemainingS 0 means idle. A
		// noised-short estimate runs out while the build still runs — what
		// does the policy see then?
		for i, tk := range sc.ticks {
			if i > 3 {
				break
			}
			for _, wv := range tk.Workers {
				t.Logf("tick #%d: worker %s BusyRemainingS=%.4f (the true build on it runs until %.4f)", i+1, wv.ID, wv.BusyRemainingS, T)
			}
		}
		// σ = 0: everything the policy sees is the truth.
		sc0 := &sxScript{decide: sxGPUAt(map[int32]float64{0: 0})}
		sxRun(t, "sigma 0", sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000), sxShard(0, 100_000)), sc0, sim.Config{NoiseSeed: 11})
		for _, in := range sc0.inputs {
			if !sxNear(in.Est.LocalBuildS, L) || !sxNear(in.Est.GPUBuildS, sxGPUBuild(100_000, 128, 10)) || !sxNear(in.Est.TransferS, sxXfer(100_000, 128)) {
				t.Errorf("sigma 0 estimate %+v is not the truth", in.Est)
			}
		}
		if len(sc0.ticks) > 0 {
			for _, wv := range sc0.ticks[0].Workers {
				if wv.BusyRemainingS > 0 {
					sxExpect(t, "sigma-0 busy-remaining at first tick", wv.BusyRemainingS, T-2, "truth minus elapsed (first tick at 2 s)")
				}
			}
		}
	})

	t.Run("NoiseMovesTheOracleLocalVsGPU", func(t *testing.T) {
		// A borderline build: speedup 1.15 makes GPU total 20.516 vs local
		// 20.631, no queries, idle worker, so the oracle picks GPU at σ=0.
		// If one factor per build scaled every estimate, the ratio would never
		// change and the oracle would never flip; with per-kind factors (and
		// the oracle noised like everyone, decision 23) it must flip for some
		// noise seeds.
		m := sxModel(1.15)
		t.Logf("local %.4f vs GPU %.4f", sxCPU(100_000, 128), sxCPU(100_000, 128)/1.15+2+sxXfer(100_000, 128))
		run := func(seed int64, wseed int64, sigma float64) sim.Result {
			w := sxWorkload(t, 128, 1, big, 1, sxShard(0, 100_000))
			w.Seed = wseed
			return sim.Run(w, policy.NewClairvoyant(sim.Futures(w)), sim.Config{CostModel: m, EstimateSigma: sigma, NoiseSeed: seed})
		}
		for s := int64(1); s <= 5; s++ {
			if r := run(s, 1, 0); r.BuildsGPU != 1 {
				t.Errorf("σ=0, noise seed %d: oracle went local; want GPU", s)
			}
		}
		pattern := func(wseed int64) (string, int) {
			b := make([]byte, 40)
			locals := 0
			for s := int64(1); s <= 40; s++ {
				if run(s, wseed, 0.5).BuildsLocal == 1 {
					b[s-1] = 'L'
					locals++
				} else {
					b[s-1] = 'G'
				}
			}
			return string(b), locals
		}
		p1, n1 := pattern(1)
		p2, _ := pattern(2)
		t.Logf("σ=0.5, noise seeds 1..40, workload seed 1: %s (%d local)", p1, n1)
		t.Logf("σ=0.5, noise seeds 1..40, workload seed 2: %s", p2)
		if n1 == 0 || n1 == 40 {
			t.Errorf("the oracle never changed its mind across 40 noise seeds: estimates are not noised per kind, or the oracle is exempt (decision 23)")
		}
		if p1 == p2 {
			t.Errorf("identical flip pattern for two workload seeds: the workload seed is not mixed into the noise (decision 23)")
		}
	})

	t.Run("DeterminismUnderNoiseAndKills", func(t *testing.T) {
		// Invariant 10 and sim.Run's doc: same workload, policy, config →
		// same Result, "whatever the host". Presets, all five policies, σ 0.6,
		// a kill; run twice sequentially and once concurrently.
		type cell struct {
			preset string
			pol    string
			cfg    sim.Config
		}
		var cells []cell
		for _, ps := range []string{"skewed", "staggered", "shrinking_pool", "bimodal"} {
			for _, pn := range []string{"always_gpu", "static_threshold", "reactive", "reactive_spec", "clairvoyant"} {
				cfg := sim.Config{EstimateSigma: 0.6, NoiseSeed: 5}
				if ps == "shrinking_pool" {
					cfg.KillWorkerAtS = 5
				}
				cells = append(cells, cell{ps, pn, cfg})
			}
		}
		runCell := func(c cell) (sim.Result, workload.Workload) {
			sc, err := workload.Preset(c.preset)
			if err != nil {
				t.Fatalf("preset %s: %v", c.preset, err)
			}
			w := workload.Generate(sc, 7)
			var p policy.Policy
			if c.pol == "clairvoyant" {
				p = policy.NewClairvoyant(sim.Futures(w))
			} else {
				p, _ = policy.ByName(c.pol)
			}
			return sim.Run(w, p, c.cfg), w
		}
		first := make([]sim.Result, len(cells))
		for i, c := range cells {
			r, w := runCell(c)
			sc, _ := workload.Preset(c.preset)
			if !reflect.DeepEqual(w, workload.Generate(sc, 7)) {
				t.Errorf("%s/%s: sim.Run mutated its workload (invariant 8: pure)", c.preset, c.pol)
			}
			first[i] = r
		}
		for i, c := range cells {
			r, _ := runCell(c)
			if !reflect.DeepEqual(r, first[i]) {
				t.Errorf("%s/%s: second run differs:\n%+v\n%+v", c.preset, c.pol, first[i], r)
			}
		}
		var wg sync.WaitGroup
		conc := make([]sim.Result, len(cells))
		for i, c := range cells {
			wg.Add(1)
			go func(i int, c cell) { defer wg.Done(); conc[i], _ = runCell(c) }(i, c)
		}
		wg.Wait()
		for i, c := range cells {
			if !reflect.DeepEqual(conc[i], first[i]) {
				t.Errorf("%s/%s: concurrent run differs from sequential", c.preset, c.pol)
			}
			t.Logf("%-14s %-16s completed=%v round=%.3f lag=%.3f gpu=%d local=%d pre=%d ren=%d spec=%d wasted=%.2f", c.preset, c.pol,
				first[i].Completed, first[i].RoundS, first[i].StragglerLagS, first[i].BuildsGPU, first[i].BuildsLocal,
				first[i].Preempts, first[i].Reneges, first[i].Speculations, first[i].WastedGPUS)
		}
		t.Logf("ok if no errors above: %d cells identical across two sequential runs and one concurrent run", len(cells))
	})

	t.Run("FuturesRelativeToDDLAndClamped", func(t *testing.T) {
		// sim.Futures doc: offsets relative to the shard's DDL; a pre-DDL
		// arrival clamps to 0.
		w := sxWorkload(t, 128, 1, big, 1,
			sxShard(5, 100_000, sxQ(3, 1, false), sxQ(8, 2, true)),
			sxShard(0, 100_000))
		f := sim.Futures(w)
		t.Logf("futures: %+v", f)
		got := f[0]
		if len(got) != 2 || !sxNear(got[0].ArriveS, 0) || !sxNear(got[1].ArriveS, 3) || !sxNear(got[1].DurS, 2) || !got[1].NeedsIndex || got[0].NeedsIndex {
			t.Errorf("shard 0 futures %+v, want [{0 1 false} {3 2 true}]", got)
		}
		if len(f[1]) != 0 {
			t.Errorf("shard 1 has no queries but futures %+v", f[1])
		}
	})

	t.Run("NewPresetsGenerateValidateComplete", func(t *testing.T) {
		names := workload.PresetNames()
		t.Logf("PresetNames = %v", names)
		if len(names) != 8 || names[6] != "high_dim" || names[7] != "production" {
			t.Errorf("want eight presets ending high_dim, production (roadmap order, decision 25)")
		}
		m := costmodel.Default()
		hd, err := workload.Preset("high_dim")
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("high_dim: dim=%d sizes=%+v pool=%+v", hd.Dim, hd.Sizes, hd.Pool)
		if hd.Dim != 1536 {
			t.Errorf("high_dim dim = %d", hd.Dim)
		}
		if !(m.GPUMemBytes(hd.Sizes.Max, hd.Dim) > hd.Pool.WorkerMemBytes && m.GPUMemBytes(hd.Sizes.Min, hd.Dim) <= hd.Pool.WorkerMemBytes) {
			t.Errorf("decision 25: high_dim's worker memory should leave the top of the size range fitting no worker but the bottom fitting (max needs %d, min %d, worker %d)",
				m.GPUMemBytes(hd.Sizes.Max, hd.Dim), m.GPUMemBytes(hd.Sizes.Min, hd.Dim), hd.Pool.WorkerMemBytes)
		}
		for seed := int64(1); seed <= 3; seed++ {
			w := workload.Generate(hd, seed)
			if err := w.Validate(); err != nil {
				t.Errorf("high_dim seed %d: %v", seed, err)
			}
			if !reflect.DeepEqual(w, workload.Generate(hd, seed)) {
				t.Errorf("high_dim seed %d not reproducible (invariant 10)", seed)
			}
			for _, pn := range []string{"always_gpu", "reactive_spec", "clairvoyant"} {
				var p policy.Policy
				if pn == "clairvoyant" {
					p = policy.NewClairvoyant(sim.Futures(w))
				} else {
					p, _ = policy.ByName(pn)
				}
				r := sxRun(t, "high_dim seed "+itoa(seed), w, p, sim.Config{})
				if !r.Completed {
					t.Errorf("high_dim seed %d under %s did not complete", seed, pn)
				}
				if r.BuildsLocal == 0 {
					var tooBig int
					for _, s := range w.Shards {
						if m.GPUMemBytes(s.NVectors, hd.Dim) > hd.Pool.WorkerMemBytes {
							tooBig++
						}
					}
					if tooBig > 0 {
						t.Errorf("%d builds fit no worker yet none ran locally", tooBig)
					}
				}
			}
		}

		pr, err := workload.Preset("production")
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("production: dim=%d sizes=%+v pool=%+v horizon=%v meanQ=%v", pr.Dim, pr.Sizes, pr.Pool, pr.Stream.Horizon, pr.Stream.MeanQueryDuration)
		if pr.Sizes.Min != 1_000_000 || pr.Sizes.Max != 10_000_000 || pr.Dim != 1536 || pr.Sizes.Kind != "zipf" {
			t.Errorf("decision 25: production is zipf 1M–10M x 1536")
		}
		w := workload.Generate(pr, 1)
		if err := w.Validate(); err != nil {
			t.Errorf("production: %v", err)
		}
		var maxN int64
		for _, s := range w.Shards {
			if s.NVectors > maxN {
				maxN = s.NVectors
			}
		}
		t.Logf("production seed 1: largest build %d vectors, local %.2f h, GPU mem %.1f GB vs worker %.1f GB", maxN, sxCPU(maxN, pr.Dim)/3600, float64(m.GPUMemBytes(maxN, pr.Dim))/1e9, float64(pr.Pool.WorkerMemBytes)/1e9)
		r := sxRun(t, "production seed 1, default Config", w, policy.NewReactive(policy.ReactiveConfig{}), sim.Config{})
		if !r.Completed {
			t.Errorf("production round under reactive with default Config did not complete (RoundS %.0f s = %.2f h): decision 21 says the safety cap is 24 h, sized for ~9.6 h local builds", r.RoundS, r.RoundS/3600)
		}
	})

	t.Run("SafetyCapDefaultIsDocumentedConsistently", func(t *testing.T) {
		// Zero workers forever, a GPU build: the round can only end at the
		// default cap. Decision 21 says 24 h; the Config field comment in
		// `go doc experiments/sim` says "default 6 h modeled". Observe which.
		w := sxWorkload(t, 128, 0, big, 1, sxShard(0, 100_000))
		r := sxRun(t, "zero workers, default cap", w, policy.AlwaysGPU{}, sim.Config{})
		t.Logf("incomplete round reports RoundS=%.0f s (%.2f h)", r.RoundS, r.RoundS/3600)
		if r.Completed {
			t.Fatalf("a GPU build with zero workers completed")
		}
		if r.BuildsGPU != 0 || r.BuildsLocal != 0 {
			t.Errorf("incomplete round counts gpu=%d local=%d, but the Result doc defines them as real builds that FINISHED on that path — this build never ran", r.BuildsGPU, r.BuildsLocal)
		}
		if r.RoundS == 0 {
			t.Logf("gap: an incomplete round reports RoundS=0 (and ShardFinishS %v); the Result doc does not say what RoundS holds when Completed=false — a consumer that forgets the completed column reads a hang as the fastest round", r.ShardFinishS)
		}
		// Decide the cap by physics: one local build of ~7 h must complete
		// under a 24 h cap (it would not under 6 h); one of ~25.4 h must not.
		for _, c := range []struct {
			n        int64
			complete bool
		}{{7_400_000, true}, {25_000_000, false}} {
			h := sxCPU(c.n, 1536) / 3600
			w := sxWorkload(t, 1536, 1, 1<<20, 1, sxShard(0, c.n))
			r := sxRun(t, "one local build of "+fmtF(math.Round(h*10)/10)+" h, default cap", w, never, sim.Config{})
			if r.Completed != c.complete {
				t.Errorf("a %.1f h local build: Completed=%v, want %v under decision 21's 24 h cap (the Config field comment says 6 h)", h, r.Completed, c.complete)
			} else if c.complete {
				sxExpect(t, "RoundS", r.RoundS, sxCPU(c.n, 1536), "the build itself")
			}
		}
		t.Logf("verdict: the cap behaves as decision 21's 24 h; `go doc experiments/sim` Config.MaxModeledS still says \"default 6 h modeled\" — stale")
	})
}
