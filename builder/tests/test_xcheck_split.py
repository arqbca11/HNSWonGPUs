"""Cross-check (black box): the Zipf split, written from the spec only.

Spec: Phase 4 decision 9 (docs/design/phase4-hnswlib.md), the split module's
docstring (rank r gets a share proportional to 1/r^s; largest = N/H(n, s);
sizes sum to N exactly, the rounding remainder goes to the top ranks; ranks
assigned to shard ids by a seeded permutation; each shard a contiguous slice
in shard-id order), experiments/workload/DESIGN.md's Zipf row, and the
"What the data looks like" paragraph of the design doc.

Every expected number here is computed by hand (exact fractions where the
exponent is an integer), not by calling the code under test.
"""

from __future__ import annotations

import hashlib
import io
import json
from fractions import Fraction

import numpy as np
import pytest

from kgpu_builder import split as S


# ---------------------------------------------------------------- reference


def hand_zipf(n_vectors: int, n_shards: int, s: float) -> list[int]:
    """The docstring's rule, written independently: floor of each rank's exact
    share, then the remainder one vector each to ranks 1, 2, ... ."""
    if float(s).is_integer():
        w = [Fraction(1, r ** int(s)) for r in range(1, n_shards + 1)]
        tot = sum(w)
        shares = [n_vectors * x / tot for x in w]
        floors = [int(x) for x in shares]  # Fraction floor for non-negatives
    else:
        w = [r ** -s for r in range(1, n_shards + 1)]
        tot = sum(w)
        floors = [int(n_vectors * x / tot) for x in w]
    rem = n_vectors - sum(floors)
    assert 0 <= rem < n_shards
    return [f + (1 if i < rem else 0) for i, f in enumerate(floors)]


def H(n: int, s: float) -> float:
    return sum(r ** -s for r in range(1, n + 1))


# ---------------------------------------------------------------- tests


def test_xcheck_zipf_1m_over_6_by_hand():
    """1M across 6 at s = 1: H(6,1) = 49/20 exactly, so the top share is
    1,000,000 * 20/49 = 408,163.27. Floors 408163, 204081, 136054, 102040,
    81632, 68027 sum to 999,997; the remainder 3 goes to ranks 1-3."""
    got = S.zipf_sizes(1_000_000, 6, 1.0)
    want = [408_164, 204_082, 136_055, 102_040, 81_632, 68_027]
    print(f"did: zipf_sizes(1_000_000, 6, 1.0)\n  expected (hand, remainder to top ranks): {want}\n  seen: {got}")
    assert got == want
    assert sum(got) == 1_000_000


def test_xcheck_design_doc_example_sizes_match_the_rule():
    """The design doc's "What the data looks like" lists SIFT's 6-way sizes.
    Its first draft said 408,163 / 204,082 / 136,054 / 102,041 / 81,633 /
    68,027 (largest-remainder rounding, which also sums to 1M) while the
    module docstring says the remainder goes to the TOP ranks; both could not
    be right (F1). The doc now carries the rule's numbers, pinned here."""
    doc = [408_164, 204_082, 136_055, 102_040, 81_632, 68_027]
    got = S.zipf_sizes(1_000_000, 6, 1.0)
    print(f"design doc says: {doc} (sum {sum(doc)})\n  code produces:   {got} (sum {sum(got)})")
    assert got == doc, "design doc 'What the data looks like' disagrees with zipf_sizes / the docstring rule"


@pytest.mark.parametrize(
    "n_vectors,n_shards,s",
    [
        (1_000_000, 50, 1.0),
        (300_000, 6, 1.0),
        (300_000, 50, 1.0),
        (1000, 3, 2.0),
        (999_999, 7, 2.0),
        (1_000_003, 6, 1.0),
        (10_007, 5, 0.0),
        (123_457, 9, 0.5),
        (123_457, 9, 1.5),
    ],
)
def test_xcheck_zipf_matches_hand_rule(n_vectors, n_shards, s):
    want = hand_zipf(n_vectors, n_shards, s)
    got = S.zipf_sizes(n_vectors, n_shards, s)
    top = n_vectors / H(n_shards, s)
    print(
        f"did: zipf_sizes({n_vectors}, {n_shards}, {s})\n"
        f"  N/H(n,s) = {top:.3f}; expected top {want[0]}, seen top {got[0]}\n"
        f"  expected {want[:8]}{'...' if n_shards > 8 else ''}\n  seen     {got[:8]}{'...' if n_shards > 8 else ''}"
    )
    assert sum(got) == n_vectors
    assert len(got) == n_shards
    assert all(a >= b for a, b in zip(got, got[1:])), "sizes by rank must be non-increasing"
    assert abs(got[0] - top) < 1 + 1e-9, "largest must be N/H(n,s) up to the remainder"
    assert got == want


def test_xcheck_zipf_edge_cases():
    """n = 1 takes everything; s = 0 is uniform with the remainder on the
    top ranks; N = n gives one vector each (tests.md: every shard at least
    one vector)."""
    cases = {
        (12345, 1, 1.0): [12345],
        (12345, 1, 0.0): [12345],
        (10, 3, 0.0): [4, 3, 3],
        (11, 3, 0.0): [4, 4, 3],
        (6, 6, 1.0): [1] * 6,
        (50, 50, 2.0): [1] * 50,
    }
    for (N, n, s), want in cases.items():
        got = S.zipf_sizes(N, n, s)
        print(f"zipf_sizes({N}, {n}, {s}): expected {want[:10]} seen {got[:10]}")
        assert got == want


def test_xcheck_zipf_small_total_never_empties_a_shard():
    """A shard with 0 vectors cannot be built. The pure floor rule would give
    zeros here (e.g. 20 across 6 at s=2: shares 13.1, 3.3, 1.5, 0.8, 0.5,
    0.4); the docs (tests.md) promise every shard at least one vector."""
    for N, n, s in [(20, 6, 2.0), (60, 50, 1.0), (100, 50, 2.0), (7, 4, 1.5)]:
        got = S.zipf_sizes(N, n, s)
        print(f"zipf_sizes({N}, {n}, {s}) -> {got}  (pure floor rule: {hand_zipf(N, n, s)})")
        assert sum(got) == N
        assert min(got) >= 1
        assert all(a >= b for a, b in zip(got, got[1:]))


def test_xcheck_zipf_fewer_vectors_than_shards():
    """N < n cannot give every shard a vector. The spec does not say what
    happens; record it. Acceptable: an exception. Not acceptable: sizes that
    do not sum to N or a negative size."""
    try:
        got = S.zipf_sizes(3, 6, 1.0)
    except Exception as e:  # noqa: BLE001
        print(f"zipf_sizes(3, 6, 1.0) raised {type(e).__name__}: {e}")
        return
    print(f"zipf_sizes(3, 6, 1.0) returned {got} (no error; spec silent)")
    assert sum(got) == 3 and min(got) >= 0


def test_xcheck_plan_ranks_offsets_and_recorded_fields():
    N, n = 1_000_000, 6
    m = S.plan("sift-128", 128, "euclidean", N, n, exponent=1.0, perm_seed=1)
    sizes = S.zipf_sizes(N, n, 1.0)
    print(f"did: plan(sift-128, N={N}, n={n}, s=1, seed=1)")
    for sh in m.shards:
        print(f"  shard {sh.shard_id}: rank {sh.rank} offset {sh.offset} size {sh.size}")
    assert (m.dataset, m.dim, m.metric, m.n_shards, m.n_vectors) == ("sift-128", 128, "euclidean", n, N)
    assert m.zipf_exponent == 1.0 and m.perm_seed == 1
    assert [sh.shard_id for sh in m.shards] == list(range(n)), "shards in shard-id order"
    assert sorted(sh.rank for sh in m.shards) == list(range(1, n + 1))
    for sh in m.shards:
        assert sh.size == sizes[sh.rank - 1], "a shard's size is its rank's Zipf size"
    off = 0
    for sh in m.shards:
        assert sh.offset == off, "contiguous offsets in shard-id order"
        off += sh.size
    assert off == N
    # The split already on disk (data/splits/sift-128/6, seed 1) puts rank 1
    # on shard 1; the plan must reproduce it.
    print(f"  expected (from the committed split): shard 1 holds rank 1 (408,164)")
    assert m.shards[1].rank == 1


def test_xcheck_plan_seed_only_permutes():
    """Different seeds change which shard holds which rank, never the sorted
    sizes (DESIGN.md: 'the seed only permutes Zipf sizes')."""
    N, n = 300_000, 50
    base = S.plan("d", 8, "euclidean", N, n, 1.0, perm_seed=1)
    again = S.plan("d", 8, "euclidean", N, n, 1.0, perm_seed=1)
    assert base == again, "same seed must replan identically"
    ranks_seen = set()
    for seed in range(1, 11):
        m = S.plan("d", 8, "euclidean", N, n, 1.0, perm_seed=seed)
        assert sorted((sh.size for sh in m.shards), reverse=True) == S.zipf_sizes(N, n, 1.0)
        ranks_seen.add(tuple(sh.rank for sh in m.shards))
    print(f"did: plan 50-way with seeds 1..10; distinct rank assignments: {len(ranks_seen)} of 10; sorted sizes identical for all")
    assert len(ranks_seen) == 10


def test_xcheck_plan_zipf_exponent_two_and_zero():
    for s in (0.0, 2.0):
        m = S.plan("d", 4, "euclidean", 1000, 3, exponent=s, perm_seed=5)
        sizes = sorted((sh.size for sh in m.shards), reverse=True)
        print(f"plan(1000 over 3, s={s}): sizes {sizes}, exponent recorded {m.zipf_exponent}")
        assert sizes == hand_zipf(1000, 3, s)
        assert m.zipf_exponent == s


def test_xcheck_write_split_bytes_sha_and_manifest(tmp_path):
    rng = np.random.default_rng(42)
    train = rng.standard_normal((997, 16)).astype(np.float32)
    m = S.plan("toy", 16, "euclidean", 997, 5, 1.0, perm_seed=3)
    out = S.write_split(train, m, tmp_path, log=lambda *a, **k: None)
    print(f"did: write_split of 997x16 float32 into 5 shards under {tmp_path}")
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    print(f"  files written: {files}")
    assert "manifest.json" in files
    for sh in out.shards:
        key = f"vectors/shard_{sh.shard_id}.npy"
        raw = (tmp_path / key).read_bytes()
        a = np.load(io.BytesIO(raw))
        assert a.dtype == np.float32 and a.shape == (sh.size, 16)
        assert np.array_equal(a, train[sh.offset : sh.offset + sh.size]), "contiguous slice"
        digest = hashlib.sha256(raw).hexdigest()
        assert out.sha256[key] == digest, f"sha256 of {key}"
        assert S.npy_bytes(train[sh.offset : sh.offset + sh.size]) == raw
    print(f"  manifest sha256 keys: {sorted(out.sha256)}; each matches the bytes on disk")
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    for field in ("dataset", "dim", "metric", "n_shards", "n_vectors", "zipf_exponent", "perm_seed", "shards", "sha256"):
        assert field in on_disk, f"manifest field {field}"
    for sh in on_disk["shards"]:
        assert set(sh) >= {"shard_id", "rank", "offset", "size"}
    rt = S.SplitManifest.from_json((tmp_path / "manifest.json").read_text())
    print(f"  manifest.json round-trips: {rt == out}")
    assert rt == out
    assert S.SplitManifest.from_json(out.to_json()) == out


def test_xcheck_write_split_from_h5py_dataset(tmp_path):
    """The docstring: train may be an h5py dataset. Same bytes as numpy."""
    import h5py

    rng = np.random.default_rng(7)
    train = rng.standard_normal((300, 8)).astype(np.float32)
    h5 = tmp_path / "t.hdf5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("train", data=train)
    m = S.plan("toy", 8, "euclidean", 300, 4, 1.0, perm_seed=9)
    a = S.write_split(train, m, tmp_path / "np", log=lambda *a, **k: None)
    with h5py.File(h5, "r") as f:
        b = S.write_split(f["train"], m, tmp_path / "h5", log=lambda *a, **k: None)
    print(f"did: the same split from a numpy array and from an h5py dataset\n  sha256 equal: {a.sha256 == b.sha256}")
    assert a.sha256 == b.sha256
