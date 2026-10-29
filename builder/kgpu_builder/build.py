"""The real build: hnswlib on one shard's vectors, gated by recall, published
manifest-last (Phase 4 decisions 2, 3, 4, 17).

    run_build(cfg, store, spec, progress) -> manifest dict

Steps, in order: fetch the shard's vectors, the dataset's held-out queries
and the shard's exact ground truth from the store; build the HNSW graph in
ten add_items batches, reporting progress after each; search the index for
every query and compute recall@k against the truth; refuse (GateFailed) if
it is below the threshold; otherwise serialise the index, PUT index.bin,
then PUT manifest.json. Nothing is written to the store before the gate
passes, and the manifest is the last write, so a reader that sees the
manifest knows the index beside it is complete and gated (invariant 7). A
kill at any point leaves at most an orphan index.bin under this attempt's
own prefix.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable

import hnswlib
import numpy as np

from .groundtruth import recall_at
from .store import Store

BATCHES = 10


@dataclass(frozen=True)
class BuildSpec:
    build_id: str
    attempt: int
    dataset: str
    n_shards: int
    shard: int
    threads: int = 2
    m: int = 16
    ef_construction: int = 200
    recall_k: int = 10
    ef_search: int = 100
    recall_min: float = 0.9


class GateFailed(Exception):
    """Recall below the threshold: the build is not published."""

    def __init__(self, recall: float, k: int, threshold: float):
        self.recall, self.k, self.threshold = recall, k, threshold
        super().__init__(f"recall@{k} {recall:.4f} < {threshold:.2f}")


def artifact_prefix(build_id: str, attempt: int) -> str:
    return f"artifacts/{build_id}/{attempt}"


def build_index(vectors: np.ndarray, m: int, ef_construction: int, threads: int,
                progress: Callable[[float], None] | None = None) -> hnswlib.Index:
    """An HNSW index over the vectors (L2), added in BATCHES batches so the
    caller sees progress; labels are the local row indices."""
    n, dim = vectors.shape
    idx = hnswlib.Index(space="l2", dim=int(dim))
    idx.init_index(max_elements=int(n), M=m, ef_construction=ef_construction, random_seed=1)
    step = max(1, -(-n // BATCHES))
    for start in range(0, n, step):
        end = min(n, start + step)
        idx.add_items(vectors[start:end], np.arange(start, end), num_threads=threads)
        if progress:
            progress(end / n)
    return idx


def gate(idx: hnswlib.Index, queries: np.ndarray, truth: np.ndarray, k: int, ef_search: int, threads: int) -> float:
    idx.set_ef(max(ef_search, k))
    labels, _ = idx.knn_query(queries, k=k, num_threads=threads)
    return recall_at(labels.astype(np.int32), truth, k)


def run_build(store: Store, spec: BuildSpec, progress: Callable[[float], None] | None = None,
              phase: Callable[[str], None] | None = None) -> dict:
    split = f"splits/{spec.dataset}/{spec.n_shards}"
    t0 = time.time()
    vectors = store.get_npy(f"{split}/vectors/shard_{spec.shard}.npy")
    queries = store.get_npy(f"datasets/{spec.dataset}/queries.npy")
    truth = store.get_npy(f"{split}/ground_truth/shard_{spec.shard}.npy")
    fetch_s = time.time() - t0
    if phase:
        phase("build")
    t1 = time.time()
    idx = build_index(vectors, spec.m, spec.ef_construction, spec.threads, progress)
    build_s = time.time() - t1
    if phase:
        phase("gate")
    t2 = time.time()
    recall = gate(idx, queries, truth, spec.recall_k, spec.ef_search, spec.threads)
    gate_s = time.time() - t2
    if recall < spec.recall_min:
        raise GateFailed(recall, spec.recall_k, spec.recall_min)
    if phase:
        phase("publish")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "index.bin")
        idx.save_index(path)
        with open(path, "rb") as f:
            blob = f.read()
    prefix = artifact_prefix(spec.build_id, spec.attempt)
    t3 = time.time()
    store.put_bytes(f"{prefix}/index.bin", blob)
    manifest = {
        "status": "done",
        "kind": "hnswlib",
        **asdict(spec),
        "n_vectors": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "n_queries": int(queries.shape[0]),
        "index_bytes": len(blob),
        "index_sha256": hashlib.sha256(blob).hexdigest(),
        "index_key": f"{prefix}/index.bin",
        "recall": recall,
        "fetch_s": fetch_s,
        "build_s": build_s,
        "gate_s": gate_s,
        "upload_s": time.time() - t3,
        "hnswlib_version": getattr(hnswlib, "__version__", "unknown"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    store.put_json(f"{prefix}/manifest.json", manifest)  # last (invariant 7)
    return manifest
