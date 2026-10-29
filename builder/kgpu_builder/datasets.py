"""The datasets this project uses, as the ann-benchmarks HDF5 files describe
them: a ``train`` array (the vectors to index), a ``test`` array (held-out
queries) and ``neighbors``/``distances`` (the exact top-100 of each query
over the WHOLE train set, which we do not use: our ground truth is per
shard, see groundtruth.py).

Phase 4 roadmap: SIFT-128 (1M x 128, Euclidean) is the main set — its 128
dimensions are the presets' Dim. The high-dimension set is a 300k subset
of GIST-960 (1M x 960, Euclidean): the OpenAI 1536-d sets are no longer
served by ann-benchmarks.com (checked 2026-10-28), and 960 sits between
the 768 and 1536 the roadmap names. (Phase 4 decision 10.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    dim: int
    metric: str  # "euclidean"
    size_bytes: int  # the HDF5 file, for the download check
    train_limit: int | None = None  # take only the first N train vectors (the GIST subset)

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


DATASETS: dict[str, Dataset] = {
    "sift-128": Dataset(
        name="sift-128",
        url="http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        dim=128,
        metric="euclidean",
        size_bytes=525_128_288,
    ),
    "gist-960-300k": Dataset(
        name="gist-960-300k",
        url="http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        dim=960,
        metric="euclidean",
        size_bytes=3_844_648_288,
        train_limit=300_000,
    ),
}


def get(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}") from None
