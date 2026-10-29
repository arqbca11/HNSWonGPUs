package crosscheck

// Phase 4 step 2 cross-check: the generator's split mode (decision 16), from
// the go doc of ApplySplit / LoadSplit / Splits / SizeDist and the committed
// JSON files under experiments/workload/splits/ (read here as the spec: the
// sizes the generator must reproduce). Pure: no database.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/arqbca11/HNSWonGPUs/experiments/workload"
)

type splitJSON struct {
	Dataset  string `json:"dataset"`
	Dim      int32  `json:"dim"`
	NShards  int    `json:"n_shards"`
	NVectors int64  `json:"n_vectors"`
	Shards   []struct {
		ShardID int32 `json:"shard_id"`
		Rank    int   `json:"rank"`
		Size    int64 `json:"size"`
	} `json:"shards"`
}

func readSplitJSON(t *testing.T, dataset string, n int) splitJSON {
	t.Helper()
	p := filepath.Join(repoRoot(t), "experiments", "workload", "splits", fmt.Sprintf("%s-%d.json", dataset, n))
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read %s: %v", p, err)
	}
	var s splitJSON
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("decode %s: %v", p, err)
	}
	return s
}

var p4Datasets = []string{"sift-128", "gist-960-300k"}

// Decision 16: Splits() names exactly the committed files.
func TestXcP4S2SplitsListsTheCommittedFiles(t *testing.T) {
	ents, err := os.ReadDir(filepath.Join(repoRoot(t), "experiments", "workload", "splits"))
	if err != nil {
		t.Fatal(err)
	}
	var files []string
	for _, e := range ents {
		if filepath.Ext(e.Name()) == ".json" {
			files = append(files, e.Name()[:len(e.Name())-5])
		}
	}
	got := append([]string(nil), workload.Splits()...)
	sort.Strings(got)
	sort.Strings(files)
	t.Logf("Splits() = %v; files on disk = %v", got, files)
	if !reflect.DeepEqual(got, files) {
		t.Errorf("decision 16: Splits() should list the committed splits; got %v want %v", got, files)
	}
}

// Decision 16 / ApplySplit doc: every preset, at 6 and 50, for both datasets,
// takes each shard's NVectors and the scenario's Dim from the committed split.
func TestXcP4S2ApplySplitEveryPresetMatchesTheJSON(t *testing.T) {
	for _, ds := range p4Datasets {
		for _, n := range []int{6, 50} {
			js := readSplitJSON(t, ds, n)
			want := map[int32]int64{}
			for _, s := range js.Shards {
				want[s.ShardID] = s.Size
			}
			for _, name := range workload.PresetNames() {
				sc, err := workload.PresetWithShards(name, n)
				if err != nil {
					t.Fatalf("PresetWithShards(%s,%d): %v", name, n, err)
				}
				origDim, origKind := sc.Dim, sc.Sizes.Kind
				if err := workload.ApplySplit(&sc, ds); err != nil {
					t.Errorf("ApplySplit(%s, %s) at %d: %v", name, ds, n, err)
					continue
				}
				w := workload.Generate(sc, 1)
				var sum int64
				bad := 0
				for _, sh := range w.Shards {
					sum += sh.NVectors
					if sh.NVectors != want[sh.ShardID] {
						bad++
						if bad <= 3 {
							t.Errorf("%s/%s/%d shard %d: NVectors %d, committed split says %d", name, ds, n, sh.ShardID, sh.NVectors, want[sh.ShardID])
						}
					}
				}
				verr := w.Validate()
				t.Logf("%-16s %-14s n=%2d: kind %s->%s, dim %d->%d, %d shards, sum %d (split %d), mismatches %d, Validate=%v",
					name, ds, n, origKind, sc.Sizes.Kind, origDim, sc.Dim, len(w.Shards), sum, js.NVectors, bad, verr)
				if sc.Sizes.Kind != "split" || sc.Sizes.Split != ds || sc.Dim != js.Dim || len(w.Shards) != n || sum != js.NVectors {
					t.Errorf("%s/%s/%d: want kind split, Split %s, dim %d, %d shards summing to %d", name, ds, n, ds, js.Dim, n, js.NVectors)
				}
				if verr != nil {
					t.Errorf("%s/%s/%d: generated workload does not validate: %v", name, ds, n, verr)
				}
			}
		}
	}
}

// Generate is pure (package doc): the same split scenario and seed give
// identical output, and the sizes do not depend on the seed (decision 9).
func TestXcP4S2SplitModeIsDeterministicAndSeedFreeInSizes(t *testing.T) {
	sc, _ := workload.PresetWithShards("skewed", 6)
	if err := workload.ApplySplit(&sc, "sift-128"); err != nil {
		t.Fatal(err)
	}
	a, b := workload.Generate(sc, 42), workload.Generate(sc, 42)
	c := workload.Generate(sc, 43)
	t.Logf("seed 42 twice: DeepEqual=%v", reflect.DeepEqual(a, b))
	if !reflect.DeepEqual(a, b) {
		t.Error("Generate must be deterministic")
	}
	for i := range a.Shards {
		t.Logf("shard %d: seed 42 n=%d, seed 43 n=%d; queries %d vs %d", i, a.Shards[i].NVectors, c.Shards[i].NVectors, len(a.Shards[i].Queries), len(c.Shards[i].Queries))
		if a.Shards[i].NVectors != c.Shards[i].NVectors {
			t.Errorf("decision 9: sizes are fixed by the split, not by the seed")
		}
	}
}

// Decision 16 says "the stream draws unchanged"; ApplySplit's doc says "the
// seed keeps governing the query streams". Is the stream for a seed the same
// in split mode as in the preset it was applied to? (Draw order per Generate's
// doc: sizes first, then DDL offsets, then the streams.)
func TestXcP4S2SplitModeStreamVersusPreset(t *testing.T) {
	for _, name := range []string{"skewed", "uniform", "staggered"} {
		pre, _ := workload.PresetWithShards(name, 6)
		spl := pre
		if err := workload.ApplySplit(&spl, "sift-128"); err != nil {
			t.Fatal(err)
		}
		a, b := workload.Generate(pre, 7), workload.Generate(spl, 7)
		sameQ, sameDDL, total := 0, 0, 0
		for i := range a.Shards {
			total++
			if reflect.DeepEqual(a.Shards[i].Queries, b.Shards[i].Queries) {
				sameQ++
			}
			if a.Shards[i].DDLOffset == b.Shards[i].DDLOffset {
				sameDDL++
			}
		}
		qa, qb := 0, 0
		for i := range a.Shards {
			qa += len(a.Shards[i].Queries)
			qb += len(b.Shards[i].Queries)
		}
		t.Logf("%s seed 7: shards with identical query streams %d/%d, identical DDL offsets %d/%d; total queries preset %d vs split %d",
			name, sameQ, total, sameDDL, total, qa, qb)
	}
	t.Logf("spec: decision 16 'the stream draws unchanged' is ambiguous between 'the same streams as the preset for a seed' and 'streams drawn the same way'; this test records which holds and does not assert")
}

// ApplySplit at a shard count with no committed split is an error, and the
// scenario is left as it was.
func TestXcP4S2ApplySplitRefusesUncommittedCountsAndDatasets(t *testing.T) {
	cases := []struct {
		ds string
		n  int
	}{{"sift-128", 1}, {"sift-128", 5}, {"sift-128", 100}, {"gist-960-300k", 7}, {"nope", 6}, {"", 6}, {"sift", 6}}
	for _, c := range cases {
		sc, err := workload.PresetWithShards("skewed", c.n)
		if err != nil {
			t.Fatalf("preset at %d: %v", c.n, err)
		}
		before := sc
		err = workload.ApplySplit(&sc, c.ds)
		t.Logf("ApplySplit(skewed@%d, %q): err=%v; scenario unchanged: %v", c.n, c.ds, err, reflect.DeepEqual(before, sc))
		if err == nil {
			t.Errorf("no committed split %s-%d: ApplySplit should fail", c.ds, c.n)
		}
		if !reflect.DeepEqual(before, sc) {
			t.Errorf("ApplySplit(%q,%d) failed but changed the scenario: %+v -> %+v", c.ds, c.n, before.Sizes, sc.Sizes)
		}
	}
}

// SizeDist doc: Kind "split" uses "the committed split of Split at the
// scenario's shard count". Built by hand with no such split, Generate (which
// returns no error) can only panic: the brief calls it a programming error.
func TestXcP4S2HandBuiltSplitKindWithoutAFile(t *testing.T) {
	for _, sd := range []workload.SizeDist{{Kind: "split", Split: "nope"}, {Kind: "split", Split: "sift-128"}} {
		sc, _ := workload.PresetWithShards("skewed", 7)
		sc.Sizes = sd
		var w workload.Workload
		var rec any
		func() {
			defer func() { rec = recover() }()
			w = workload.Generate(sc, 1)
		}()
		if rec == nil {
			t.Logf("SizeDist%+v at 7 shards: Generate returned without panicking; sizes %v; Validate=%v", sd, sizesOf(w), w.Validate())
			t.Errorf("a split that does not exist should not silently produce a workload")
		} else {
			t.Logf("SizeDist%+v at 7 shards: Generate panicked: %v", sd, rec)
		}
	}
}

// hour_build carries SizeDist.Straggler (one shard overridden). In split mode
// "sizes are fixed by the split": does the straggler survive ApplySplit?
func TestXcP4S2ApplySplitOnAStragglerPreset(t *testing.T) {
	sc, err := workload.PresetWithShards("hour_build", 6)
	if err != nil {
		t.Skipf("no hour_build preset: %v", err)
	}
	t.Logf("hour_build before: Sizes=%+v", sc.Sizes)
	if err := workload.ApplySplit(&sc, "sift-128"); err != nil {
		t.Fatal(err)
	}
	w := workload.Generate(sc, 1)
	js := readSplitJSON(t, "sift-128", 6)
	t.Logf("after ApplySplit: Sizes=%+v; generated sizes %v", sc.Sizes, sizesOf(w))
	for _, s := range js.Shards {
		if w.Shards[s.ShardID].NVectors != s.Size {
			t.Errorf("shard %d: %d, split says %d (decision 9: sizes fixed by the split)", s.ShardID, w.Shards[s.ShardID].NVectors, s.Size)
		}
	}
}

func sizesOf(w workload.Workload) []int64 {
	var out []int64
	for _, s := range w.Shards {
		out = append(out, s.NVectors)
	}
	return out
}
