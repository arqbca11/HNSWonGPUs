package crosscheck

// Phase 4 step 2 cross-check: a round's dataset (decision 15, migration 0007)
// and a local build that fails (decision 19: POST /builds/{id}/fail with
// {"reason"}; store.FailLocal "guarded like CompleteLocal"). Spec: the Phase 4
// design doc, the Phase 1 API table (POST /builds/{id}/done: 409 unless the
// build is a running local build), go doc of store/api/client.

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/arqbca11/HNSWonGPUs/scheduler/api"
	"github.com/arqbca11/HNSWonGPUs/scheduler/store"
	"github.com/arqbca11/HNSWonGPUs/shard/client"
)

func (e *env) roundJSON(t *testing.T, id int64) (map[string]any, int) {
	t.Helper()
	var out map[string]any
	code, body := e.do(t, "GET", "/rounds/"+itoa(id), nil, &out)
	if code != 200 {
		t.Logf("GET /rounds/%d: %d %s", id, code, body)
		return nil, code
	}
	return out["round"].(map[string]any), code
}

// Decision 15, migration 0007: POST /rounds takes an optional dataset; GET
// /rounds/{id} shows it; NULL (json null) is a modeled round.
func TestXcP4S2RoundDatasetOverHTTP(t *testing.T) {
	e := setup(t)
	cases := []struct {
		label string
		body  any
		want  any // nil = json null
	}{
		{"with dataset sift-128", api.CreateRoundRequest{Scenario: "xc", Seed: 1, NShards: 6, Dataset: "sift-128"}, "sift-128"},
		{"without dataset", api.CreateRoundRequest{Scenario: "xc", Seed: 1, NShards: 6}, nil},
		{"dataset \"\" explicitly", `{"scenario":"xc","seed":1,"n_shards":6,"dataset":""}`, nil},
		{"dataset null explicitly", `{"scenario":"xc","seed":1,"n_shards":6,"dataset":null}`, nil},
		{"gist", api.CreateRoundRequest{Scenario: "xc", Seed: 1, NShards: 50, Dataset: "gist-960-300k"}, "gist-960-300k"},
	}
	for _, c := range cases {
		var r api.CreateRoundResponse
		code, body := e.do(t, "POST", "/rounds", c.body, &r)
		if code != http.StatusCreated {
			t.Errorf("%s: POST /rounds %d %s", c.label, code, body)
			continue
		}
		rd, _ := e.roundJSON(t, r.RoundID)
		dsVal, present := rd["dataset"]
		stored := e.mustRound(t, r.RoundID)
		t.Logf("%s: round %d; GET /rounds dataset=%#v (key present %v); store Round.Dataset=%v", c.label, r.RoundID, dsVal, present, derefS(stored.Dataset))
		if !present {
			t.Errorf("%s: GET /rounds/{id} should carry the dataset key (null for a modeled round)", c.label)
		}
		if dsVal != c.want {
			t.Errorf("%s: dataset %#v, want %#v", c.label, dsVal, c.want)
		}
	}
	// A dataset that names nothing we have: the scheduler has no catalogue
	// (spec silent). Record what happens.
	var r api.CreateRoundResponse
	code, body := e.do(t, "POST", "/rounds", api.CreateRoundRequest{Scenario: "xc", Seed: 1, NShards: 7, Dataset: "sift-128"}, &r)
	t.Logf("POST /rounds sift-128 at n_shards 7 (no such split committed): %d %s  [spec silent: accepted rounds would fail at the first build]", code, strings.TrimSpace(body))
	code, body = e.do(t, "POST", "/rounds", api.CreateRoundRequest{Scenario: "xc", Seed: 1, NShards: 6, Dataset: "no/such dataset"}, &r)
	t.Logf("POST /rounds dataset \"no/such dataset\": %d %s  [spec silent]", code, strings.TrimSpace(body))
}

func derefS(s *string) string {
	if s == nil {
		return "<nil>"
	}
	return *s
}

// The shard client's CreateRoundWithDataset and the store's agree.
func TestXcP4S2ClientAndStoreCreateRoundWithDataset(t *testing.T) {
	e := setup(t)
	c := client.New(e.srv.URL)
	id, err := c.CreateRoundWithDataset(e.ctx, "xc", 3, 6, "sift-128")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.GetRound(e.ctx, id); err != nil {
		t.Fatal(err)
	}
	st := e.mustRound(t, id)
	t.Logf("note: client.Round (GET /rounds/{id} as the shard decodes it) has no Dataset field; read back through the store")
	id2, err := e.st.CreateRoundWithDataset(e.ctx, "xc", 3, 6, "")
	if err != nil {
		t.Fatal(err)
	}
	r2 := e.mustRound(t, id2)
	t.Logf("client: round %d dataset %v; store with \"\": round %d dataset %v", id, derefS(st.Dataset), id2, derefS(r2.Dataset))
	if st.Dataset == nil || *st.Dataset != "sift-128" {
		t.Errorf("decision 15: the client round should carry sift-128")
	}
	if r2.Dataset != nil {
		t.Errorf("store doc: \"\" means a modeled round (nil Dataset); got %q", *r2.Dataset)
	}
}

// The fail endpoint's guards: decision 19 and the API's own doc. Missing
// reason -> 400 (API table addition); a gpu build, an unknown id, a done or
// already failed build -> refused.
func TestXcP4S2FailEndpointGuards(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 8)
	local := e.submitLocal(t, round, 0)
	gpu := e.submit(t, round, 1).BuildID
	logBuild(t, "local at submit", e.mustBuild(t, local))
	logBuild(t, "gpu at submit", e.mustBuild(t, gpu))

	type step struct {
		label string
		id    string
		body  any
		want  int
	}
	steps := []step{
		{"no body", local, nil, 400},
		{"empty object", local, `{}`, 400},
		{"empty reason", local, api.LocalFailRequest{Reason: ""}, 400},
		{"not JSON", local, `reason=gate`, 400},
		{"gpu build", gpu, api.LocalFailRequest{Reason: "gate"}, 409},
		{"unknown id", "99:999999", api.LocalFailRequest{Reason: "gate"}, 404},
		{"running local, real reason", local, api.LocalFailRequest{Reason: "recall@10 0.84 < 0.90"}, 200},
		{"again", local, api.LocalFailRequest{Reason: "second"}, 409},
	}
	for _, s := range steps {
		code, body := e.do(t, "POST", "/builds/"+s.id+"/fail", s.body, nil)
		t.Logf("POST /builds/%s/fail (%s): %d %s", s.id, s.label, code, strings.TrimSpace(body))
		if code != s.want {
			t.Errorf("%s: got %d want %d", s.label, code, s.want)
		}
	}
	b := e.mustBuild(t, local)
	logBuild(t, "local after fail", b)
	t.Logf("[db] fail_reason=%q", derefS(b.FailReason))
	if b.State != "failed" || b.FailReason == nil || *b.FailReason != "recall@10 0.84 < 0.90" {
		t.Errorf("decision 19: the build is failed with the gate's reason; got state %s reason %v", b.State, derefS(b.FailReason))
	}
	if b.FinishedAt == nil {
		t.Errorf("a failed build should carry finished_at (terminal)")
	}
	code, _ := e.do(t, "POST", "/builds/"+local+"/done", nil, nil)
	t.Logf("POST /builds/%s/done after fail: %d (API table: 409 unless a running local build)", local, code)
	if code != 409 {
		t.Errorf("done after fail: %d, want 409", code)
	}
	// whitespace-only reason: spec silent
	l2 := e.submitLocal(t, round, 2)
	code, body := e.do(t, "POST", "/builds/"+l2+"/fail", api.LocalFailRequest{Reason: "   "}, nil)
	t.Logf("POST fail with a whitespace-only reason: %d %s  [spec silent; 'missing reason' is 400]", code, strings.TrimSpace(body))
	// done then fail
	l3 := e.submitLocal(t, round, 3)
	code, _ = e.do(t, "POST", "/builds/"+l3+"/done", nil, nil)
	code2, _ := e.do(t, "POST", "/builds/"+l3+"/fail", api.LocalFailRequest{Reason: "late gate"}, nil)
	b3 := e.mustBuild(t, l3)
	t.Logf("done (%d) then fail (%d): state %s reason %v", code, code2, b3.State, derefS(b3.FailReason))
	if code != 200 || code2 != 409 || b3.State != "done" || b3.FailReason != nil {
		t.Errorf("a done build cannot be failed afterwards")
	}
	// a preempted local build (now gpu/queued) cannot be failed by the shard
	l4 := e.submitLocal(t, round, 4)
	if ok, err := e.st.PreemptToGPU(e.ctx, l4, 10); err != nil || !ok {
		t.Fatalf("preempt: %v %v", ok, err)
	}
	code, _ = e.do(t, "POST", "/builds/"+l4+"/fail", api.LocalFailRequest{Reason: "gate after preempt"}, nil)
	b4 := e.mustBuild(t, l4)
	t.Logf("fail after preempt: %d; build %s/%s", code, b4.Placement, b4.State)
	if code != 409 || b4.State != "queued" {
		t.Errorf("a preempted build is no longer the shard's to fail (guarded like done)")
	}
}

// A long reason (600 chars) over HTTP. The brief says the shard truncates to
// 500; the API table says nothing about a limit.
func TestXcP4S2FailEndpointLongReason(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 5)
	reasons := []string{
		strings.Repeat("r", 499) + "Z",
		strings.Repeat("r", 599) + "Z",
		strings.Repeat("r", 19999) + "Z",
		strings.Repeat("é", 600),            // 2 bytes per character
		"x" + strings.Repeat("\u2603", 300), // 3 bytes per character, offset by one byte
	}
	for i, reason := range reasons {
		n := utf8.RuneCountInString(reason)
		id := e.submitLocal(t, round, int32(i))
		code, body := e.do(t, "POST", "/builds/"+id+"/fail", api.LocalFailRequest{Reason: reason}, nil)
		b := e.mustBuild(t, id)
		stored := derefS(b.FailReason)
		tb := strings.TrimSpace(body)
		t.Logf("reason of %d chars / %d bytes: %d %s; build %s; stored %d chars / %d bytes, valid UTF-8 %v",
			n, len(reason), code, tb[:min(90, len(tb))], b.State, utf8.RuneCountInString(stored), len(stored), utf8.ValidString(stored))
		if code != 200 || b.State != "failed" {
			t.Errorf("a gate refusal with a long reason must still fail the build (decision 19); got %d, state %s", code, b.State)
		}
		if code == 200 && stored != reason {
			t.Logf("  NOTE: accepted but stored %d of %d characters (the API table states no limit)", utf8.RuneCountInString(stored), n)
		}
	}
}

// FailLocal vs CompleteLocal racing on one build: both are guarded by
// placement and state (doc), so exactly one wins and the row agrees.
func TestXcP4S2FailLocalVersusCompleteLocalRace(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 100)
	wins := map[string]int{}
	for i := 0; i < 100; i++ {
		id := e.submitLocal(t, round, int32(i))
		var wg sync.WaitGroup
		var okF, okC bool
		var errF, errC error
		start := make(chan struct{})
		wg.Add(2)
		go func() { defer wg.Done(); <-start; okF, errF = e.st.FailLocal(context.Background(), id, "gate") }()
		go func() { defer wg.Done(); <-start; okC, errC = e.st.CompleteLocal(context.Background(), id) }()
		close(start)
		wg.Wait()
		if errF != nil || errC != nil {
			t.Fatalf("%s: errors %v %v", id, errF, errC)
		}
		b := e.mustBuild(t, id)
		switch {
		case okF && !okC && b.State == "failed":
			wins["fail"]++
		case okC && !okF && b.State == "done" && b.FailReason == nil:
			wins["done"]++
		default:
			wins["BAD"]++
			t.Errorf("%s: fail=%v done=%v, row state %s reason %v", id, okF, okC, b.State, derefS(b.FailReason))
		}
	}
	t.Logf("100 races of FailLocal vs CompleteLocal on a running local build: %v (exactly one winner each, row matches the winner)", wins)
}

// Round completion (Phase 1 decision 18, Phase 2 decision 30 per the brief):
// failed is terminal, so a round whose local build failed finishes once its
// shards are stream_done and their queries ran.
func TestXcP4S2FailedLocalBuildIsTerminalForTheRound(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 2)
	l0 := e.submitLocal(t, round, 0)
	l1 := e.submitLocal(t, round, 1)
	// shard 1 has a query that does not need the index; it runs.
	if _, err := e.st.RecordArrival(e.ctx, round, 1, 0, 100, false); err != nil {
		t.Fatal(err)
	}
	runJob(t, e, round, 1, 0)
	code, _ := e.do(t, "POST", "/builds/"+l0+"/done", nil, nil)
	code2, _ := e.do(t, "POST", "/builds/"+l1+"/fail", api.LocalFailRequest{Reason: "recall@10 0.50 < 0.90"}, nil)
	t.Logf("shard 0 done: %d; shard 1 failed: %d", code, code2)
	for s := int32(0); s < 2; s++ {
		var rep api.ReportResponse
		c, body := e.do(t, "POST", "/shards/"+itoa(round)+"/"+itoa(int64(s))+"/report", report(true), &rep)
		t.Logf("shard %d final report stream_done: %d, reply build %+v %s", s, c, rep.Build, strings.TrimSpace(body)[:0])
	}
	if _, err := e.st.FinishCompleteRounds(e.ctx); err != nil {
		t.Fatal(err)
	}
	r := e.mustRound(t, round)
	t.Logf("[db] round %d finished_at=%v", round, tptr(r.FinishedAt))
	if r.FinishedAt == nil {
		t.Errorf("a failed local build is terminal: the round should finish (n_shards builds done or failed, every shard stream_done)")
	}

	// The open question (Phase 1 doc, decision 54): a needs_index query on a
	// shard whose build failed can never start. Record what the round does.
	round2 := e.createRound(t, 1)
	l := e.submitLocal(t, round2, 0)
	if _, err := e.st.RecordArrival(e.ctx, round2, 0, 0, 100, true); err != nil {
		t.Fatal(err)
	}
	e.do(t, "POST", "/builds/"+l+"/fail", api.LocalFailRequest{Reason: "gate"}, nil)
	e.do(t, "POST", "/shards/"+itoa(round2)+"/0/report", report(true), nil)
	e.st.FinishCompleteRounds(e.ctx) //nolint:errcheck
	r2 := e.mustRound(t, round2)
	t.Logf("round %d: failed local build + one needs_index query never run, stream_done: finished_at=%v  [Phase 1 open question; decision 54 says such queries stay stuck]", round2, tptr(r2.FinishedAt))
}

var _ = store.BuildID

// The same multi-byte reason straight through the store: FailLocal (local)
// and Fail (a claimed gpu build, the worker's path for exit 3 per decision
// 18 — the Python worker writes over SQL, this is the Go store's equivalent).
func TestXcP4S2StoreFailReasonsMultiByte(t *testing.T) {
	e := setup(t)
	round := e.createRound(t, 4)
	reason := "x" + strings.Repeat("☃", 300) // 901 bytes; byte 500 falls inside a character
	l := e.submitLocal(t, round, 0)
	ok, err := e.st.FailLocal(e.ctx, l, reason)
	b := e.mustBuild(t, l)
	t.Logf("store.FailLocal(%d chars / %d bytes): ok=%v err=%v; build %s, stored %d bytes", utf8.RuneCountInString(reason), len(reason), ok, err, b.State, len(derefS(b.FailReason)))
	if err != nil || !ok || b.State != "failed" {
		t.Errorf("FailLocal with a valid UTF-8 reason must fail the build; got ok=%v err=%v state=%s", ok, err, b.State)
	}
	g := e.submit(t, round, 1).BuildID
	if err := e.st.Heartbeat(e.ctx, "xc-w", 1<<40, ""); err != nil {
		t.Fatal(err)
	}
	c, got, err := e.st.Claim(e.ctx, "xc-w", 1<<40, time.Minute)
	if err != nil || !got || c.BuildID != g {
		t.Fatalf("claim: %v %v %+v", got, err, c)
	}
	ok, err = e.st.Fail(e.ctx, g, c.Attempt, reason)
	b = e.mustBuild(t, g)
	t.Logf("store.Fail(gpu %s attempt %d, same reason): ok=%v err=%v; build %s, stored %d bytes", g, c.Attempt, ok, err, b.State, len(derefS(b.FailReason)))
	if err != nil || !ok || b.State != "failed" {
		t.Errorf("Fail with a valid UTF-8 reason must fail the build; got ok=%v err=%v state=%s", ok, err, b.State)
	}
}
