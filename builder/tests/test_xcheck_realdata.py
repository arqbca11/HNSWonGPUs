"""Cross-check (black box, read-only): the four real splits on disk.

data/splits/<dataset>/<n>/ for sift-128 and gist-960-300k at 6 and 50
shards, data/datasets/<dataset>/, and the source HDF5 files under
data/ann-benchmarks/. Nothing here writes to data/.

Spec: decision 9 (sizes by the Zipf rule, contiguous slices, sha256 per
object, layout), decision 10 (GIST: the first 300k train rows, all 1,000
queries), decision 14 (ground truth int32 [n_queries, 100], local indices,
exact by Euclidean distance), the design doc's "What the data looks like".
Skips when the data directory is absent.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "splits").is_dir(), reason="no data/splits on this machine")

SETS = {
    "sift-128": dict(h5="sift-128-euclidean.hdf5", dim=128, n=1_000_000, nq=10_000),
    "gist-960-300k": dict(h5="gist-960-euclidean.hdf5", dim=960, n=300_000, nq=1_000),
}
SPLITS = [(d, n) for d in SETS for n in (6, 50)]


def hand_zipf(N, n):  # s = 1, exact fractions, remainder to the top ranks
    w = [Fraction(1, r) for r in range(1, n + 1)]
    t = sum(w)
    f = [int(N * x / t) for x in w]
    rem = N - sum(f)
    return [v + (1 if i < rem else 0) for i, v in enumerate(f)]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def manifest(ds, n):
    return json.loads((DATA / "splits" / ds / str(n) / "manifest.json").read_text())


def dists64(q, x, block=50_000):
    q = q.astype(np.float64)
    out = np.empty(len(x))
    for s in range(0, len(x), block):
        d = x[s : s + block].astype(np.float64) - q
        out[s : s + block] = np.einsum("ij,ij->i", d, d)
    return out


@pytest.mark.parametrize("ds,n", SPLITS)
def test_xcheck_real_manifest_sizes_ranks_offsets(ds, n):
    m = manifest(ds, n)
    N = SETS[ds]["n"]
    want = hand_zipf(N, n)
    by_rank = {s["rank"]: s["size"] for s in m["shards"]}
    got = [by_rank[r] for r in range(1, n + 1)]
    print(f"{ds}/{n}: n_vectors={m['n_vectors']} s={m['zipf_exponent']} seed={m['perm_seed']}\n"
          f"  sizes by rank (seen)     {got[:6]}...{got[-2:]}\n  hand Zipf, top-remainder {want[:6]}...{want[-2:]}")
    assert m["n_vectors"] == N and m["n_shards"] == n and m["dim"] == SETS[ds]["dim"] and m["metric"] == "euclidean"
    assert got == want
    assert [s["shard_id"] for s in m["shards"]] == list(range(n))
    off = 0
    for s in m["shards"]:
        assert s["offset"] == off
        off += s["size"]
    assert off == N


@pytest.mark.parametrize("ds,n", SPLITS)
def test_xcheck_real_vectors_shape_dtype_and_source_rows(ds, n):
    import h5py

    m = manifest(ds, n)
    sd = DATA / "splits" / ds / str(n)
    with h5py.File(DATA / "ann-benchmarks" / SETS[ds]["h5"], "r") as f:
        train = f["train"]
        for s in m["shards"]:
            a = np.load(sd / "vectors" / f"shard_{s['shard_id']}.npy", mmap_mode="r")
            assert a.dtype == np.float32 and a.shape == (s["size"], SETS[ds]["dim"]), f"shard {s['shard_id']}: {a.shape} {a.dtype}"
            o, z = s["offset"], s["size"]
            assert np.array_equal(a[0], train[o]) and np.array_equal(a[-1], train[o + z - 1]), f"shard {s['shard_id']} endpoints"
            mid = z // 2
            assert np.array_equal(a[mid], train[o + mid])
        if ds == "gist-960-300k":
            last = [s for s in m["shards"] if s["offset"] + s["size"] == 300_000][0]
            a = np.load(sd / "vectors" / f"shard_{last['shard_id']}.npy", mmap_mode="r")
            assert np.array_equal(a[-1], train[299_999]), "GIST uses the FIRST 300k rows"
    print(f"{ds}/{n}: every shard file is float32 [size, dim] per the manifest; first, middle and last rows equal the HDF5 train rows at offset (contiguous slices of the source)")


@pytest.mark.parametrize("ds,n", SPLITS)
def test_xcheck_real_sha256_of_some_objects(ds, n):
    m = manifest(ds, n)
    sd = DATA / "splits" / ds / str(n)
    sizes = {f"vectors/shard_{s['shard_id']}.npy": s["size"] for s in m["shards"]}
    small = min(sizes, key=sizes.get)
    pick = [small] if n == 50 or ds == "sift-128" else [small, max(sizes, key=sizes.get)]
    for k in pick:
        h = sha(sd / k)
        print(f"{ds}/{n} {k}: manifest {m['sha256'][k][:16]}…  on disk {h[:16]}…")
        assert h == m["sha256"][k]
    assert set(m["sha256"]) >= set(sizes), "a sha256 for every vectors object"


@pytest.mark.parametrize("ds", list(SETS))
def test_xcheck_real_queries_match_hdf5_and_dataset_manifest(ds):
    import h5py

    q = np.load(DATA / "datasets" / ds / "queries.npy")
    dm = json.loads((DATA / "datasets" / ds / "manifest.json").read_text())
    with h5py.File(DATA / "ann-benchmarks" / SETS[ds]["h5"], "r") as f:
        test = f["test"][:]
    print(f"{ds}: queries.npy {q.shape} {q.dtype}; HDF5 test {test.shape}; dataset manifest n_queries={dm['n_queries']} n_train_used={dm['n_train_used']}")
    assert np.array_equal(q, test) and q.shape == (SETS[ds]["nq"], SETS[ds]["dim"])
    assert dm["n_queries"] == SETS[ds]["nq"] and dm["n_train_used"] == SETS[ds]["n"]
    assert dm["sha256"]["queries.npy"] == sha(DATA / "datasets" / ds / "queries.npy")


@pytest.mark.parametrize("ds,n", SPLITS)
def test_xcheck_real_ground_truth_shapes(ds, n):
    m = manifest(ds, n)
    sd = DATA / "splits" / ds / str(n)
    for s in m["shards"]:
        gt = np.load(sd / "ground_truth" / f"shard_{s['shard_id']}.npy", mmap_mode="r")
        assert gt.dtype == np.int32 and gt.shape == (SETS[ds]["nq"], 100), f"shard {s['shard_id']}: {gt.shape} {gt.dtype}"
        assert int(gt.min()) >= 0 and int(gt.max()) < s["size"]
    print(f"{ds}/{n}: every ground_truth/shard_k.npy is int32 [{SETS[ds]['nq']}, 100] with values in [0, shard size)")


@pytest.mark.parametrize(
    "ds,n,which,nq",
    [
        ("sift-128", 6, "smallest", 40),
        ("sift-128", 6, "largest", 3),
        ("sift-128", 50, "smallest", 200),
        ("gist-960-300k", 6, "largest", 5),
        ("gist-960-300k", 50, "smallest", 300),
    ],
)
def test_xcheck_real_ground_truth_rows_are_exact(ds, n, which, nq):
    """Recompute rows of the stored truth by a float64 difference norm.
    SIFT distances are integers, so ties occur: compare distance by
    position (any order among exactly tied vectors is accepted)."""
    m = manifest(ds, n)
    s = (min if which == "smallest" else max)(m["shards"], key=lambda s: s["size"])
    sd = DATA / "splits" / ds / str(n)
    x = np.load(sd / "vectors" / f"shard_{s['shard_id']}.npy", mmap_mode="r")
    gt = np.load(sd / "ground_truth" / f"shard_{s['shard_id']}.npy", mmap_mode="r")
    q = np.load(DATA / "datasets" / ds / "queries.npy", mmap_mode="r")
    idx = np.linspace(0, len(q) - 1, nq).astype(int)
    bad, set10, set100, worst = [], 0, 0, 0.0
    for i in idx:
        d = dists64(q[i], x)
        e = np.argsort(d, kind="stable")[:100]
        g = np.asarray(gt[i])
        want, got = d[e], d[g]
        worst = max(worst, float(np.max(np.abs(got - want) / np.maximum(want, 1e-300))))
        if (d[g[:10]] > d[e[9]]).any():  # a stored top-10 member farther than the true 10th
            set10 += 1
        if (got > d[e[99]]).any():
            set100 += 1
        if not np.allclose(got, want, rtol=1e-12, atol=0):
            pos = int(np.argmax(~np.isclose(got, want, rtol=1e-12, atol=0)))
            bad.append((int(i), pos, float(got[pos]), float(want[pos])))
    print(f"{ds}/{n} {which} shard {s['shard_id']} ({s['size']} vectors): {nq} queries recomputed in float64 (squared distances)\n"
          f"  rows whose top-100 distance profile differs from the exact one: {len(bad)}  e.g. (query, position, stored, exact) {bad[:2]}\n"
          f"  rows whose top-10 SET is wrong (what the recall gate reads): {set10}; rows whose top-100 set is wrong: {set100}\n"
          f"  worst relative distance error at a position: {worst:.2e}")
    assert set10 == 0, "the recall gate's top-10 differs from the exact top-10"
    assert not bad, "decision 14 promises exact ground truth; near-ties are misordered"


CLUSTER = __import__("os").environ.get("XCHECK_CLUSTER_S3")


@pytest.mark.skipif(not CLUSTER, reason="XCHECK_CLUSTER_S3 unset (a port-forward to the kind cluster's objectstore)")
def test_xcheck_cluster_bucket_mirrors_local_splits_read_only():
    """Read-only: the cluster's kgpu bucket (filled by scripts/phase4-data.sh)
    holds every local object under the same key with the same size, and the
    same manifests. Default keys from builder/README.md."""
    import os

    from kgpu_builder.config import Config
    from kgpu_builder.store import Store

    st = Store(Config(data_dir=DATA, s3_endpoint=CLUSTER,
                      s3_access_key=os.environ.get("XCHECK_CLUSTER_KEY", "kgpu"),
                      s3_secret_key=os.environ.get("XCHECK_CLUSTER_SECRET", "kgpu-secret"),
                      s3_bucket="kgpu", s3_region="us-east-1"))
    remote = set(st.list(""))
    local = {p.relative_to(DATA).as_posix() for p in (DATA / "splits").rglob("*") if p.is_file()}
    local |= {p.relative_to(DATA).as_posix() for p in (DATA / "datasets").rglob("*") if p.is_file()}
    missing = sorted(local - remote)
    extra = sorted(k for k in remote - local if not k.startswith("artifacts/"))
    print(f"did: listed the cluster bucket kgpu at {CLUSTER}: {len(remote)} objects; local objects {len(local)}\n"
          f"  local objects missing remotely: {missing[:5]} ({len(missing)})\n  remote objects with no local file: {extra[:5]} ({len(extra)})")
    assert not missing
    wrong = [k for k in sorted(local) if st.size(k) != (DATA / k).stat().st_size]
    print(f"  objects whose remote size differs from the local file: {wrong[:5]} ({len(wrong)})")
    assert not wrong
    for ds, n in SPLITS:
        assert st.get_json(f"splits/{ds}/{n}/manifest.json") == manifest(ds, n)
    print("  the four split manifests are identical remotely")
