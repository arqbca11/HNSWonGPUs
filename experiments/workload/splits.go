package workload

import (
	"embed"
	"encoding/json"
	"fmt"
	"sort"
)

// The split manifests the builder wrote (python -m kgpu_builder split),
// copied into the repo so the Go side — the shard simulator, the scheduler's
// cost-model inputs, the discrete-event simulator — knows every shard's size
// offline and deterministically (Phase 4 decision 16). Each file is
// splits/<dataset>/<n>/manifest.json from the object store, verifiable
// against it by the sha256 it carries per object.
//
//go:embed splits/*.json
var splitFiles embed.FS

// SplitManifest is the subset of the builder's manifest the generator uses.
type SplitManifest struct {
	Dataset      string  `json:"dataset"`
	Dim          int32   `json:"dim"`
	NShards      int     `json:"n_shards"`
	NVectors     int64   `json:"n_vectors"`
	ZipfExponent float64 `json:"zipf_exponent"`
	PermSeed     int64   `json:"perm_seed"`
	Shards       []struct {
		ShardID int32 `json:"shard_id"`
		Rank    int   `json:"rank"`
		Offset  int64 `json:"offset"`
		Size    int64 `json:"size"`
	} `json:"shards"`
}

// Splits lists the committed splits as "<dataset>-<n>" names.
func Splits() []string {
	entries, _ := splitFiles.ReadDir("splits")
	var out []string
	for _, e := range entries {
		name := e.Name()
		out = append(out, name[:len(name)-len(".json")])
	}
	sort.Strings(out)
	return out
}

// LoadSplit returns the committed manifest for a dataset at n shards.
func LoadSplit(dataset string, n int) (SplitManifest, error) {
	b, err := splitFiles.ReadFile(fmt.Sprintf("splits/%s-%d.json", dataset, n))
	if err != nil {
		return SplitManifest{}, fmt.Errorf("workload: no committed split for dataset %q at %d shards (have %v)", dataset, n, Splits())
	}
	var m SplitManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return SplitManifest{}, fmt.Errorf("workload: split %s-%d: %w", dataset, n, err)
	}
	if m.NShards != n || len(m.Shards) != n {
		return SplitManifest{}, fmt.Errorf("workload: split %s-%d declares %d shards with %d entries", dataset, n, m.NShards, len(m.Shards))
	}
	for i, s := range m.Shards {
		if int(s.ShardID) != i || s.Size < 1 {
			return SplitManifest{}, fmt.Errorf("workload: split %s-%d: entry %d is shard %d with size %d", dataset, n, i, s.ShardID, s.Size)
		}
	}
	return m, nil
}

// ApplySplit makes the scenario take its sizes and dimension from the
// committed split of `dataset` at the scenario's shard count: the size
// distribution becomes Kind "split" (decision 9: with real data the sizes
// are fixed by the split, not drawn per round; the seed keeps governing the
// query streams and the DDL timing). The split's own rank assignment is used
// as is — it is the same method as the zipf draw, not the same permutation
// (cross-check G9).
func ApplySplit(sc *Scenario, dataset string) error {
	m, err := LoadSplit(dataset, sc.NShards)
	if err != nil {
		return err
	}
	lo, hi := m.Shards[0].Size, m.Shards[0].Size
	for _, s := range m.Shards {
		lo, hi = min(lo, s.Size), max(hi, s.Size)
	}
	sc.Sizes = SizeDist{Kind: "split", Split: dataset, Min: lo, Max: hi, ZipfExponent: m.ZipfExponent}
	sc.Dim = m.Dim
	return nil
}

// splitSizes is genSizes's "split" case: the committed sizes in shard order.
// ApplySplit validated the manifest, so a failure here is a programming
// error (a SizeDist built by hand with a Split that has no file).
func splitSizes(d SizeDist, n int) []int64 {
	m, err := LoadSplit(d.Split, n)
	if err != nil {
		panic(err)
	}
	out := make([]int64, n)
	for i, s := range m.Shards {
		out[i] = s.Size
	}
	return out
}
