"""The GPU build path (Phase 5 decisions 2, 10): cuVS CAGRA on the card,
converted to an HNSW graph that the ordinary hnswlib loads.

    build_index_cagra(vectors, m, ef_construction, threads, progress) -> hnswlib.Index

Three steps. (1) `cagra.build` builds the CAGRA graph on the GPU with a graph
degree of 2·M (CAGRA's degree counts both directions where HNSW's M counts
one). (2) `hnsw.from_cagra(hnsw.IndexParams(hierarchy="cpu", M=m,
ef_construction=ef_construction, num_threads=threads), index)` adds the
upper layers on the CPU; this hierarchy setting is the one whose saved file
"is compatible with the original hnswlib library" — the default writes a
base-layer-only file only cuVS's own wrapper can read. (3) `hnsw.save` writes
the file and STOCK hnswlib loads it back: that load is the proof the query
path (decision 2: CPU hnswlib) can read what the GPU built, so a file it
could not read fails the build here, before the gate and before anything
reaches the store. The returned object is an ordinary hnswlib.Index, so the
gate, the save and the publish in build.run_build do not change.

cuVS needs CUDA and a card. Where it is not importable (every laptop and
kind run) `cuvs_available()` is False and `build_index_cagra` raises
CuvsUnavailable with one line; the CLI turns that into a failed build with
a reason, and the tests of this module skip.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import hnswlib
import numpy as np


class CuvsUnavailable(RuntimeError):
    """cuVS (and a GPU) is not available in this process."""


def cuvs_available() -> bool:
    try:
        import cupy  # noqa: F401
        from cuvs.neighbors import cagra, hnsw  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means no GPU path here
        return False
    return True


def cuvs_version() -> str:
    try:
        from importlib.metadata import version
        for name in ("cuvs-cu12", "cuvs-cu13", "cuvs"):
            try:
                return f"{name} {version(name)}"
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def build_index_cagra(vectors: np.ndarray, m: int, ef_construction: int, threads: int,
                      progress: Callable[[float], None] | None = None) -> hnswlib.Index:
    """CAGRA on the GPU, converted to HNSW, saved, and loaded back by stock hnswlib."""
    if not cuvs_available():
        raise CuvsUnavailable("cuVS is not installed in this environment (no GPU build path): "
                              "run the worker's GPU image (worker/Dockerfile.gpu) or use --builder hnswlib")
    import cupy as cp
    from cuvs.neighbors import cagra, hnsw

    n, dim = vectors.shape
    data = cp.asarray(np.ascontiguousarray(vectors, dtype=np.float32))
    params = cagra.IndexParams(metric="sqeuclidean", graph_degree=2 * m, intermediate_graph_degree=4 * m)
    index = cagra.build(params, data)
    if progress:
        progress(0.4)
    hparams = hnsw.IndexParams(hierarchy="cpu", M=m, ef_construction=ef_construction, num_threads=threads,
                               metric="sqeuclidean")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cagra-hnsw.bin")
        hnsw_index = hnsw.from_cagra(hparams, index, temporary_index_path=os.path.join(d, "cagra-tmp.bin"))
        if progress:
            progress(0.8)
        hnsw.save(path, hnsw_index)
        del hnsw_index, index, data
        # the proof (decision 10): the ordinary hnswlib reads the file
        idx = hnswlib.Index(space="l2", dim=int(dim))
        idx.load_index(path, max_elements=int(n))
    if idx.get_current_count() != n:
        raise RuntimeError(f"the converted index holds {idx.get_current_count()} of {n} vectors")
    if progress:
        progress(1.0)
    return idx
