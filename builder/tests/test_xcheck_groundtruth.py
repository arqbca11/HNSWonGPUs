"""Cross-check (black box): exact per-shard ground truth and recall_at.

Spec: Phase 4 decision 14 (exact top-100 by Euclidean distance over the
shard's vectors, local indices, int32 [n_queries, k]), the groundtruth
module's docstrings (nearest first; k clamped to the shard size; recall_at
= mean over queries of |found[:k] ∩ truth[:k]| / k).

The reference here is deliberately a different formula from the one the
module documents (|q|^2 + |x|^2 - 2 q.x in a matrix product): a direct
float64 difference-and-norm per query, which has no cancellation error.
Where ties exist the order among tied vectors is not specified, so ties are
checked by distance (the k-th distance and the multiset of distances), not
by index.
"""

from __future__ import annotations

import numpy as np
import pytest

from kgpu_builder.groundtruth import exact_topk, recall_at


def ref_dists(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    d = x.astype(np.float64) - q.astype(np.float64)[None, :]
    return np.sqrt((d * d).sum(axis=1))


def check_valid_topk(queries, vectors, got, k, label, exact_ties=False):
    """got must be k distinct local indices, nearest first by the float64
    reference, and the k it returns must be a true top-k (its distances equal
    the k smallest distances as a multiset)."""
    n = len(vectors)
    kk = min(k, n)
    assert got.shape == (len(queries), kk), f"{label}: shape {got.shape} != {(len(queries), kk)}"
    assert got.dtype == np.int32, f"{label}: dtype {got.dtype}"
    worst = 0
    for i, q in enumerate(queries):
        row = got[i]
        assert len(set(row.tolist())) == kk, f"{label}: query {i} has duplicate indices"
        assert row.min() >= 0 and row.max() < n
        d = ref_dists(q, vectors)
        dr = d[row]
        true_sorted = np.sort(d)[:kk]
        # nearest first
        if not np.all(np.diff(dr) >= -1e-9 * max(1.0, dr.max())):
            bad = int(np.argmax(np.diff(dr) < 0))
            raise AssertionError(f"{label}: query {i} not nearest-first at position {bad}: {dr[bad]} > {dr[bad+1]}")
        # a true top-k (allowing any order among exact ties)
        err = np.max(np.abs(dr - true_sorted))
        worst = max(worst, err)
        assert np.allclose(dr, true_sorted, rtol=0, atol=1e-9 * max(1.0, true_sorted.max())), (
            f"{label}: query {i} is not the exact top-{kk}; max distance error {err}"
        )
    return worst


def test_xcheck_topk_against_direct_float64_reference():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2500, 24)).astype(np.float32)
    q = rng.standard_normal((40, 24)).astype(np.float32)
    got = exact_topk(q, x, k=20, chunk=7)
    print("did: exact_topk 40 queries over 2500 random vectors (dim 24), k=20, chunk=7")
    w = check_valid_topk(q, x, got, 20, "random")
    # random continuous data has no ties: the indices themselves must match
    for i in range(len(q)):
        want = np.argsort(ref_dists(q[i], x), kind="stable")[:20]
        assert np.array_equal(got[i], want), f"query {i}: {got[i][:5]} vs {want[:5]}"
    print(f"  expected: indices of a float64 argsort of |x - q|; seen: identical for all 40 rows (max dist err {w:.2e})")


def test_xcheck_topk_is_exact_with_a_large_common_offset():
    """SIFT values are 0..255 with norms in the hundreds: |q|^2 + |x|^2 -
    2 q.x subtracts large, nearly equal numbers. Shift everything by 1000 so
    the squared norms are ~1.3e8 while neighbour distances differ by ~1:
    in float32 the expansion has a resolution of ~8 and would misorder.
    'Exact' (decision 14) means the order must still be right."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((3000, 128)).astype(np.float32)
    x = (base + 1000.0).astype(np.float32)
    q = (rng.standard_normal((25, 128)) * 0.5 + 1000.0).astype(np.float32)
    got = exact_topk(q, x, k=10)
    mism, hits = 0, 0
    for i in range(len(q)):
        want = np.argsort(ref_dists(q[i], x), kind="stable")[:10]
        if not np.array_equal(got[i], want):
            mism += 1
        hits += len(set(got[i].tolist()) & set(want.tolist()))
    print(f"did: 25 queries over 3000 vectors at offset 1000 (dim 128), k=10\n"
          f"  expected: the float64 reference order in every row\n"
          f"  seen: {mism} of 25 rows differ; recall@10 of the stored truth against the exact truth = {hits / 250:.3f}")
    check_valid_topk(q, x, got, 10, "offset-1000")


def test_xcheck_topk_sift_like_integer_vectors_with_ties():
    """Integer-valued vectors (like SIFT's) give integer squared distances,
    so ties at the boundary are common. Any tie order is acceptable; the
    returned set must be a true top-k by distance."""
    rng = np.random.default_rng(3)
    x = rng.integers(0, 4, size=(4000, 6)).astype(np.float32)
    q = rng.integers(0, 4, size=(30, 6)).astype(np.float32)
    got = exact_topk(q, x, k=50)
    check_valid_topk(q, x, got, 50, "integer-ties")
    d = ref_dists(q[0], x)
    print(f"did: 30 integer queries over 4000 integer vectors in {{0..3}}^6, k=50\n"
          f"  e.g. query 0: {np.sum(d <= np.sort(d)[49])} vectors at or within the 50th distance -> ties at the boundary\n"
          f"  expected: a valid top-50 nearest first (tie order unspecified); seen: valid")


def test_xcheck_topk_duplicate_vectors_and_the_query_itself():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    x[17] = x[3]
    x[150] = x[3]
    q = x[3:4].copy()
    got = exact_topk(q, x, k=5)
    print(f"did: query = shard vector 3, which is duplicated at 17 and 150\n  expected first three = {{3, 17, 150}} (distance 0); seen {got[0][:5].tolist()}")
    assert set(got[0][:3].tolist()) == {3, 17, 150}
    check_valid_topk(q, x, got, 5, "duplicates")


def test_xcheck_topk_k_larger_than_shard_and_single_vector_shard():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((7, 4)).astype(np.float32)
    q = rng.standard_normal((3, 4)).astype(np.float32)
    got = exact_topk(q, x, k=100)
    print(f"did: k=100 over a 7-vector shard -> shape {got.shape}; expected (3, 7), each row a permutation of 0..6 nearest first")
    check_valid_topk(q, x, got, 100, "k>n")
    one = exact_topk(q, x[:1], k=100)
    print(f"did: k=100 over a 1-vector shard -> {one.tolist()}; expected [[0],[0],[0]]")
    assert one.shape == (3, 1) and one.dtype == np.int32 and (one == 0).all()


def test_xcheck_topk_far_queries_still_get_k_answers():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((500, 16)).astype(np.float32)
    q = np.stack([np.full(16, 1e4, np.float32), np.full(16, -3e3, np.float32)])
    got = exact_topk(q, x, k=30)
    print(f"did: two queries far outside the data (1e4 and -3e3 per coordinate), k=30\n  seen shape {got.shape}; checking against float64 reference")
    check_valid_topk(q, x, got, 30, "far")


def test_xcheck_topk_chunk_sizes_do_not_change_the_answer():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1000, 12)).astype(np.float32)
    q = rng.standard_normal((53, 12)).astype(np.float32)
    outs = {c: exact_topk(q, x, k=15, chunk=c) for c in (1, 2, 52, 53, 54, 4096)}
    same = all(np.array_equal(outs[1], o) for o in outs.values())
    print(f"did: exact_topk with chunk 1, 2, 52, 53, 54, 4096 over 53 queries; all identical: {same}")
    assert same


def test_xcheck_topk_float64_and_float16_inputs():
    """The spec does not restrict dtypes; float64 input must still give the
    exact answer (no silent downcast that changes the order)."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal((800, 10))
    q = rng.standard_normal((10, 10))
    got = exact_topk(q, x, k=10)
    for i in range(10):
        assert np.array_equal(got[i], np.argsort(ref_dists(q[i], x), kind="stable")[:10])
    print("did: float64 vectors and queries, k=10; expected the float64 argsort; seen identical")


# ---------------------------------------------------------------- recall_at


def test_xcheck_recall_at_by_hand():
    truth = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    found = np.array([[3, 2, 9, 8], [4, 9, 9, 7]])
    r2 = recall_at(found, truth, 2)  # q0: {3,2}∩{0,1}=0; q1: {4,9}∩{4,5}=1 -> (0+0.5)/2
    r4 = recall_at(found, truth, 4)  # q0: 2/4; q1: {4,9,7}∩{4,5,6,7}=2 -> 2/4 (set: dup 9 counts once)
    print(f"recall@2: expected 0.25 seen {r2}; recall@4: expected 0.5 seen {r4}")
    assert r2 == pytest.approx(0.25)
    assert r4 == pytest.approx(0.5)


def test_xcheck_recall_at_duplicates_in_found_count_once():
    truth = np.array([[0, 1, 2]])
    found = np.array([[0, 0, 0]])
    r = recall_at(found, truth, 3)
    print(f"found [0,0,0] vs truth [0,1,2] at k=3: the docstring's ∩ is a set -> expected 1/3, seen {r}")
    assert r == pytest.approx(1 / 3)


def test_xcheck_recall_at_k_beyond_stored_width():
    """Ground truth stores 100 columns (decision 14). Asking recall@200 of a
    perfect answer: the docstring divides by k, so the result would be 0.5
    for a perfect index. Spec silent; record, and require it not to exceed 1
    nor crash silently into a wrong value above 1."""
    truth = np.tile(np.arange(100), (3, 1))
    found = np.tile(np.arange(250), (3, 1))
    try:
        r = recall_at(found, truth, 200)
    except Exception as e:  # noqa: BLE001
        print(f"recall_at(k=200, truth width 100) raised {type(e).__name__}: {e}")
        return
    print(f"recall_at(k=200) with truth width 100 and a perfect found list: {r} (spec silent; 0.5 means divide by k, 1.0 means clamp k)")
    assert 0.0 <= r <= 1.0


def test_xcheck_recall_at_found_narrower_than_k():
    """An index asked for 10 that returns only 4 (a tiny shard, say): spec
    silent. Record; it must stay within [0, 1]."""
    truth = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    found = np.array([[0, 1, 2, 3]])
    try:
        r = recall_at(found, truth, 10)
    except Exception as e:  # noqa: BLE001
        print(f"recall_at(found width 4, k=10) raised {type(e).__name__}: {e}")
        return
    print(f"recall_at(found width 4, k=10) = {r} (4 of 10 true neighbours found; divide-by-k gives 0.4)")
    assert r == pytest.approx(0.4)
