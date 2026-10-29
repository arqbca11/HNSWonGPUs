"""Splitting a dataset across n shards with Zipf-distributed sizes.

The rule mirrors the Go workload generator's rank-size Zipf (experiments/
workload DESIGN.md): the shard of rank r gets a share proportional to
1/r^s. Here the total is fixed by the dataset, so the largest shard is
N / H(n, s) with H the generalised harmonic number, and the sizes sum to
N exactly (the rounding remainder goes to the top ranks). Ranks are
assigned to shard ids by a seeded permutation, as the generator does, so
the largest shard is not always shard 0. Each shard is a CONTIGUOUS slice
of the train set in shard-id order; the manifest records every shard's
offset, size and rank, so the split is reproducible and a shard's local
vector index maps back to the dataset's row.

Roadmap, Phase 4 data: with real data the size distribution is fixed by
the split, not drawn per round; the exponent is a property of the split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    rank: int  # 1 = largest
    offset: int  # first train row
    size: int


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    dim: int
    metric: str
    n_shards: int
    n_vectors: int
    zipf_exponent: float
    perm_seed: int
    shards: list[ShardSpec]
    sha256: dict[str, str]  # object name -> sha256 of the .npy bytes: vectors/ by write_split, ground_truth/ by the groundtruth command
    ground_truth_k: int = 0  # the top-k the ground truth holds (0 until the groundtruth command ran)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "SplitManifest":
        d = json.loads(s)
        d["shards"] = [ShardSpec(**x) for x in d["shards"]]
        return SplitManifest(**d)


def zipf_sizes(n_vectors: int, n_shards: int, exponent: float) -> list[int]:
    """Sizes by rank (index 0 = rank 1), summing to n_vectors exactly."""
    if n_shards < 1 or n_vectors < n_shards:
        raise ValueError(f"cannot split {n_vectors} vectors across {n_shards} shards")
    # Every shard holds at least one vector: a share that floors to 0 is
    # bumped to 1 and the deficit taken back from the top ranks (cross-check
    # G1). Only totals within a small multiple of n are affected — 60 across
    # 50 gives [9, 3, 1, 1, ...] against shares of [13.3, 6.7, 4.4, ...] —
    # never the real datasets.
    w = np.array([1.0 / (r ** exponent) for r in range(1, n_shards + 1)])
    raw = w / w.sum() * n_vectors
    sizes = np.floor(raw).astype(np.int64)
    sizes = np.maximum(sizes, 1)
    rem = n_vectors - int(sizes.sum())
    # hand the remainder (positive or, after the floor-of-1 bump, possibly
    # negative) to the top ranks, one vector at a time, never below 1
    i = 0
    while rem != 0:
        if rem > 0:
            sizes[i % n_shards] += 1
            rem -= 1
        else:
            j = i % n_shards
            if sizes[j] > 1:
                sizes[j] -= 1
                rem += 1
        i += 1
    assert int(sizes.sum()) == n_vectors
    return [int(x) for x in sizes]


def plan(dataset: str, dim: int, metric: str, n_vectors: int, n_shards: int,
         exponent: float = 1.0, perm_seed: int = 1) -> SplitManifest:
    sizes = zipf_sizes(n_vectors, n_shards, exponent)
    perm = np.random.default_rng(perm_seed).permutation(n_shards)  # perm[shard] = rank-1
    shards = []
    offset = 0
    for shard in range(n_shards):
        rank = int(perm[shard]) + 1
        size = sizes[rank - 1]
        shards.append(ShardSpec(shard_id=shard, rank=rank, offset=offset, size=size))
        offset += size
    assert offset == n_vectors
    return SplitManifest(dataset=dataset, dim=dim, metric=metric, n_shards=n_shards, n_vectors=n_vectors,
                         zipf_exponent=exponent, perm_seed=perm_seed, shards=shards, sha256={})


def npy_bytes(a: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def write_split(train: np.ndarray, m: SplitManifest, out_dir: Path, *, log=print) -> SplitManifest:
    """Write vectors/shard_k.npy per shard and manifest.json under out_dir.

    ``train`` may be an h5py dataset or a numpy array; each shard's slice
    is read once. Returns the manifest with every object's sha256 filled in.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vectors").mkdir(exist_ok=True)
    sha: dict[str, str] = {}
    for s in m.shards:
        x = np.asarray(train[s.offset:s.offset + s.size], dtype=np.float32)
        assert x.shape == (s.size, m.dim), (x.shape, s)
        b = npy_bytes(x)
        name = f"vectors/shard_{s.shard_id}.npy"
        (out_dir / name).write_bytes(b)
        sha[name] = hashlib.sha256(b).hexdigest()
        log(f"  shard {s.shard_id:2d}: rank {s.rank:2d}, {s.size:8d} vectors, {len(b) / 1e6:7.1f} MB -> {name}")
    m2 = SplitManifest(**{**asdict(m), "shards": m.shards, "sha256": sha})
    (out_dir / "manifest.json").write_text(m2.to_json())
    return m2
