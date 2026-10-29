"""Exact top-k against a naive loop, and recall@k by hand. Pure."""

import numpy as np

from kgpu_builder import groundtruth as gt


def test_exact_topk_matches_naive_loop_and_chunking_is_invisible():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3000, 16)).astype(np.float32)
    q = rng.standard_normal((700, 16)).astype(np.float32)
    got = gt.exact_topk(q, x, k=10, chunk=256)
    for i in range(0, 700, 97):
        d = ((x - q[i]) ** 2).sum(axis=1)
        want = np.argsort(d, kind="stable")[:10]
        assert got[i].tolist() == want.tolist(), (i, got[i], want)
    assert np.array_equal(got, gt.exact_topk(q, x, k=10, chunk=10_000))
    print("700 queries x 3000 vectors: chunked top-10 equals the naive loop at 8 probed queries and is chunk-size independent")


def test_k_is_clamped_to_shard_size():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    q = rng.standard_normal((3, 4)).astype(np.float32)
    got = gt.exact_topk(q, x, k=100)
    assert got.shape == (3, 5)
    for i in range(3):
        assert sorted(got[i].tolist()) == [0, 1, 2, 3, 4]


def test_recall_at_by_hand():
    truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    found = np.array([[3, 2, 9, 9], [8, 7, 6, 5]])
    assert gt.recall_at(found, truth, 4) == (2 + 4) / 8
    assert gt.recall_at(found, truth, 2) == (1 + 0) / 4
