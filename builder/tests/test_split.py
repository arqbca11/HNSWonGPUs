"""The Zipf split, by hand. Pure."""

import numpy as np

from kgpu_builder import split as sp


def test_zipf_sizes_sum_and_shape():
    sizes = sp.zipf_sizes(1_000_000, 6, 1.0)
    # H(6,1) = 2.45; largest = 1M/2.45 = 408163, then /2, /3 ...
    print(f"1M across 6 shards, s=1: {sizes}")
    assert sum(sizes) == 1_000_000
    assert sizes == sorted(sizes, reverse=True)
    assert abs(sizes[0] - 408_163) <= 6  # the remainder lands on the top ranks
    assert abs(sizes[1] - sizes[0] / 2) <= 6 and abs(sizes[2] - sizes[0] / 3) <= 6
    s50 = sp.zipf_sizes(1_000_000, 50, 1.0)
    print(f"1M across 50 shards: largest {s50[0]}, smallest {s50[-1]}, sum {sum(s50)}")
    assert sum(s50) == 1_000_000 and 222_000 <= s50[0] <= 223_000 and s50[-1] >= 4_400


def test_zipf_exponent_zero_is_uniform_and_small_totals_are_valid():
    assert sp.zipf_sizes(100, 4, 0.0) == [25, 25, 25, 25]
    assert sp.zipf_sizes(7, 4, 1.0) == [4, 1, 1, 1] or sum(sp.zipf_sizes(7, 4, 1.0)) == 7
    assert min(sp.zipf_sizes(50, 50, 1.5)) >= 1 and sum(sp.zipf_sizes(50, 50, 1.5)) == 50


def test_plan_permutes_ranks_and_is_contiguous_and_seeded():
    m = sp.plan("sift-128", 128, "euclidean", 1_000_000, 6, 1.0, perm_seed=1)
    print("shard -> rank, offset, size:", [(s.shard_id, s.rank, s.offset, s.size) for s in m.shards])
    assert sorted(s.rank for s in m.shards) == [1, 2, 3, 4, 5, 6]
    off = 0
    for s in m.shards:
        assert s.offset == off
        off += s.size
    assert off == 1_000_000
    assert m == sp.plan("sift-128", 128, "euclidean", 1_000_000, 6, 1.0, perm_seed=1)
    assert [s.rank for s in m.shards] != [s.rank for s in sp.plan("sift-128", 128, "euclidean", 1_000_000, 6, 1.0, perm_seed=2).shards]


def test_write_split_round_trips_with_sha256(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.standard_normal((1000, 8)).astype(np.float32)
    m = sp.plan("toy", 8, "euclidean", 1000, 4, 1.0, perm_seed=3)
    m2 = sp.write_split(train, m, tmp_path, log=lambda *_: None)
    assert set(m2.sha256) == {f"vectors/shard_{i}.npy" for i in range(4)}
    back = sp.SplitManifest.from_json((tmp_path / "manifest.json").read_text())
    assert back == m2
    for s in m2.shards:
        x = np.load(tmp_path / f"vectors/shard_{s.shard_id}.npy")
        assert x.shape == (s.size, 8)
        np.testing.assert_array_equal(x, train[s.offset:s.offset + s.size])
    print("4 shards written, manifest round-trips, every shard equals its contiguous slice")
