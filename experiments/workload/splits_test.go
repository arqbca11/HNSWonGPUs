package workload

import "testing"

func TestSplitSizesFromTheCommittedManifest(t *testing.T) {
	t.Logf("Phase 4 decision 16: ApplySplit makes a scenario take every shard's size and the dimension from the committed split; the query streams are still the seed's.")
	if got := Splits(); len(got) != 4 {
		t.Fatalf("committed splits %v, want sift-128 and gist-960-300k at 6 and 50", got)
	}
	for _, n := range []int{6, 50} {
		sc := mustPreset(t, "skewed", n)
		if err := ApplySplit(&sc, "sift-128"); err != nil {
			t.Fatal(err)
		}
		m, _ := LoadSplit("sift-128", n)
		w := Generate(sc, 1)
		total := int64(0)
		for i, s := range w.Shards {
			if s.NVectors != m.Shards[i].Size {
				t.Fatalf("n=%d shard %d: %d vectors, split says %d", n, i, s.NVectors, m.Shards[i].Size)
			}
			total += s.NVectors
		}
		if total != 1_000_000 || sc.Dim != 128 || sc.Sizes.Kind != "split" || sc.Sizes.Max != m.Shards[0].Size && sc.Sizes.Max < m.Shards[0].Size {
			t.Fatalf("n=%d: total %d dim %d kind %q max %d", n, total, sc.Dim, sc.Sizes.Kind, sc.Sizes.Max)
		}
		if len(w.Shards[0].Queries) == 0 {
			t.Fatal("the stream must still be generated")
		}
		t.Logf("n=%d: sizes are the committed split's (sum 1,000,000; largest %d; dim 128); %d queries on shard 0 from seed 1", n, sc.Sizes.Max, len(w.Shards[0].Queries))
	}
	sc := mustPreset(t, "skewed", 7)
	if err := ApplySplit(&sc, "sift-128"); err == nil {
		t.Fatal("7 shards has no committed split: want an error")
	} else {
		t.Logf("no split at 7 shards: %v", err)
	}
	gist := mustPreset(t, "skewed", 6)
	if err := ApplySplit(&gist, "gist-960-300k"); err != nil || gist.Dim != 960 {
		t.Fatalf("gist: err %v dim %d", err, gist.Dim)
	}
}
