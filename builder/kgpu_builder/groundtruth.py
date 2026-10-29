"""Exact nearest neighbours of the held-out queries within ONE shard.

The recall gate (Phase 4 decision 4) asks: of a query's true top-10 among
THIS shard's vectors, how many does the new index return? So ground truth
is per shard, computed once by brute force: squared Euclidean distance
through a matrix product (|q|^2 + |x|^2 - 2 q.x), queries in chunks so a
10k x 410k distance block never materialises at once, then the top-k by
argpartition and a final sort. Indices are LOCAL to the shard (0..size-1).

Exact means exact (decision 14): the expansion is evaluated in float64.
In float32 it subtracts two large, nearly equal numbers and loses the
small differences between near neighbours — bug 16, found by the step's
cross-check with a float64 reference: vectors sitting near 1000 in every
coordinate got a top-10 that was 96 % wrong, and GIST's near-ties were
misordered inside the top-100. Ties (equal distances) may come back in
any order.
"""

from __future__ import annotations

import numpy as np


def exact_topk(queries: np.ndarray, vectors: np.ndarray, k: int = 100, chunk: int = 1024) -> np.ndarray:
    """int32 [n_queries, k]: local indices of each query's k nearest vectors,
    nearest first, by Euclidean distance, computed in float64. k is clamped
    to the shard size."""
    q = np.ascontiguousarray(queries, dtype=np.float64)
    x = np.ascontiguousarray(vectors, dtype=np.float64)
    n = x.shape[0]
    kk = min(k, n)
    x_norm = (x * x).sum(axis=1)
    out = np.empty((q.shape[0], kk), dtype=np.int32)
    for start in range(0, q.shape[0], chunk):
        qc = q[start:start + chunk]
        d = (qc * qc).sum(axis=1)[:, None] + x_norm[None, :] - 2.0 * (qc @ x.T)
        if kk < n:
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        else:
            part = np.tile(np.arange(n), (qc.shape[0], 1))
        dpart = np.take_along_axis(d, part, axis=1)
        order = np.argsort(dpart, axis=1, kind="stable")
        out[start:start + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def recall_at(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean over queries of |found[:k] ∩ truth[:k]| / k. k must not exceed
    either array's width (asking for recall@200 against a top-100 truth is
    a mistake, not a 0.5): ValueError otherwise (cross-check G2)."""
    if found.shape[0] != truth.shape[0]:
        raise ValueError(f"{found.shape[0]} found rows vs {truth.shape[0]} truth rows")
    if k < 1 or k > found.shape[1] or k > truth.shape[1]:
        raise ValueError(f"k={k} exceeds the stored width (found {found.shape[1]}, truth {truth.shape[1]})")
    hits = 0
    for f, t in zip(found[:, :k], truth[:, :k]):
        hits += len(set(f.tolist()) & set(t.tolist()))
    return hits / (found.shape[0] * k)
