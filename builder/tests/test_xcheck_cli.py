"""Cross-check (black box): the data pipeline's command line.

Spec: `python -m kgpu_builder --help` (download fetches the HDF5 resumably and
verifies its size; split writes vectors per shard + manifest under
KGPU_DATA_DIR/splits/<dataset>/<n>/; groundtruth writes exact top-k per
shard; upload pushes queries, split and ground truth to the object store
under the same layout; verify checks every object is present with the
manifest's size/sha256; all = the four in order), builder/README.md and
config's docstring (KGPU_DATA_DIR, S3_*), Phase 4 decisions 9, 10, 14.

The catalogue knows only sift-128 and gist-960-300k. The CLI reads the HDF5
from KGPU_DATA_DIR/ann-benchmarks/<catalogue filename> (seen on disk under
data/ann-benchmarks/), so a tiny synthetic file with the catalogue's name in
a temp data dir drives split and groundtruth without the real 525 MB file.
Network is blocked for every subprocess by a dead HTTP proxy, so nothing in
these tests can download the real datasets.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[1]
DEAD = "http://127.0.0.1:9"
SIFT_FILE = "sift-128-euclidean.hdf5"
SIFT_SIZE = 525_128_288  # the catalogue's size_bytes for sift-128 (help(datasets))


def run(args, data_dir, extra_env=None, timeout=120):
    env = dict(os.environ)
    env.update(
        KGPU_DATA_DIR=str(data_dir),
        http_proxy=DEAD, HTTP_PROXY=DEAD, https_proxy=DEAD, HTTPS_PROXY=DEAD, no_proxy="", NO_PROXY="",
    )
    for k in ("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET"):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    try:
        p = subprocess.run(
            [sys.executable, "-m", "kgpu_builder", *args], cwd=BUILDER, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT"
    return p.returncode, p.stdout, p.stderr


def tail(s, n=6):
    lines = [l for l in s.strip().splitlines() if l.strip()]
    return "\n      ".join(lines[-n:])


def make_sift_like(data_dir: Path, n_train=1000, n_test=20, seed=0):
    import h5py

    rng = np.random.default_rng(seed)
    train = rng.integers(0, 60, (n_train, 128)).astype(np.float32)
    test = rng.integers(0, 60, (n_test, 128)).astype(np.float32)
    (data_dir / "ann-benchmarks").mkdir(parents=True, exist_ok=True)
    with h5py.File(data_dir / "ann-benchmarks" / SIFT_FILE, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
    return train, test


def ref_topk(q, x, k):
    out = []
    for qi in q:
        d = ((x.astype(np.float64) - qi.astype(np.float64)) ** 2).sum(1)
        out.append(d)
    return np.array(out)


# ---------------------------------------------------------------- catalogue / download


def test_xcheck_cli_unknown_dataset_is_a_clear_error(tmp_path):
    rc, out, err = run(["split", "glove-100", "6"], tmp_path)
    print(f"did: split glove-100 6 (not in the catalogue)\n  expected: non-zero exit naming the known datasets (help: 'Datasets: sift-128, gist-960-300k')\n"
          f"  seen: rc={rc}\n      {tail(out + err)}")
    assert rc != 0
    assert "sift-128" in (out + err) and "gist-960-300k" in (out + err)
    assert "Traceback" not in err, "a user error should not surface as a Python traceback"


def test_xcheck_cli_download_refuses_an_oversized_file(tmp_path):
    """A file of the right name but LARGER than the catalogue's size cannot be
    resumed; download must refuse rather than accept or truncate it."""
    p = tmp_path / "ann-benchmarks" / SIFT_FILE
    p.parent.mkdir(parents=True)
    with open(p, "wb") as f:
        f.truncate(SIFT_SIZE + 1)  # sparse; costs no disk
    rc, out, err = run(["download", "sift-128"], tmp_path, timeout=60)
    print(f"did: download sift-128 with a {SIFT_SIZE + 1}-byte file in place (catalogue: {SIFT_SIZE})\n"
          f"  expected: non-zero exit, file not accepted\n  seen: rc={rc}\n      {tail(out + err)}")
    assert rc != 0


def test_xcheck_cli_download_accepts_a_complete_file_without_network(tmp_path):
    p = tmp_path / "ann-benchmarks" / SIFT_FILE
    p.parent.mkdir(parents=True)
    with open(p, "wb") as f:
        f.truncate(SIFT_SIZE)
    rc, out, err = run(["download", "sift-128"], tmp_path, timeout=60)
    print(f"did: download sift-128 with a file of exactly the catalogue's size and the network blocked\n"
          f"  expected: rc 0, no fetch (size verified)\n  seen: rc={rc}\n      {tail(out + err)}")
    assert rc == 0
    assert p.stat().st_size == SIFT_SIZE


def test_xcheck_cli_download_with_a_truncated_file_tries_to_resume(tmp_path):
    """A truncated file is what 'resumable' is for: download should try to
    fetch the rest (and here fail, the network being blocked) — never accept
    the short file as complete."""
    p = tmp_path / "ann-benchmarks" / SIFT_FILE
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\0" * 1000)
    rc, out, err = run(["download", "sift-128"], tmp_path, timeout=30)
    size = p.stat().st_size
    print(f"did: download sift-128 with a 1000-byte file in place, network blocked by a dead proxy\n"
          f"  expected: non-zero exit (resume fails), short file not declared complete\n"
          f"  seen: rc={rc}, file now {size} bytes\n      {tail(out + err)}")
    assert rc != 0
    assert size != SIFT_SIZE


# ---------------------------------------------------------------- split + groundtruth


def test_xcheck_cli_split_and_groundtruth_on_a_synthetic_file(tmp_path):
    train, test = make_sift_like(tmp_path)
    rc, out, err = run(["split", "sift-128", "6", "--zipf", "2", "--seed", "7"], tmp_path)
    print(f"did: split sift-128 6 --zipf 2 --seed 7 on a synthetic 1000x128 file\n  rc={rc}\n      {tail(out + err, 8)}")
    assert rc == 0
    sd = tmp_path / "splits" / "sift-128" / "6"
    m = json.loads((sd / "manifest.json").read_text())
    print(f"  manifest: n_vectors={m['n_vectors']} zipf_exponent={m['zipf_exponent']} perm_seed={m['perm_seed']} "
          f"sizes by shard={[s['size'] for s in m['shards']]} ranks={[s['rank'] for s in m['shards']]}")
    assert m["n_vectors"] == 1000 and m["zipf_exponent"] == 2.0 and m["perm_seed"] == 7
    # s = 2 over 6: H = 1 + 1/4 + 1/9 + 1/16 + 1/25 + 1/36 = 5369/3600;
    # shares 670.5, 167.6, 74.5, 41.9, 26.8, 18.6 -> floors 670,167,74,41,26,18 (996), +1 to ranks 1-4
    want = [671, 168, 75, 42, 26, 18]
    assert sorted((s["size"] for s in m["shards"]), reverse=True) == want, f"hand Zipf s=2: {want}"
    for s in m["shards"]:
        a = np.load(sd / "vectors" / f"shard_{s['shard_id']}.npy")
        assert np.array_equal(a, train[s["offset"] : s["offset"] + s["size"]])
        raw = (sd / "vectors" / f"shard_{s['shard_id']}.npy").read_bytes()
        assert m["sha256"][f"vectors/shard_{s['shard_id']}.npy"] == hashlib.sha256(raw).hexdigest()
    q = np.load(tmp_path / "datasets" / "sift-128" / "queries.npy")
    assert np.array_equal(q, test), "datasets/<ds>/queries.npy is the HDF5's test array"
    dm = json.loads((tmp_path / "datasets" / "sift-128" / "manifest.json").read_text())
    assert dm["sha256"]["queries.npy"] == hashlib.sha256((tmp_path / "datasets/sift-128/queries.npy").read_bytes()).hexdigest()
    print("  vectors are the contiguous train slices, sha256 match, queries.npy = the HDF5 test array")

    rc, out, err = run(["groundtruth", "sift-128", "6", "--k", "15"], tmp_path)
    print(f"did: groundtruth sift-128 6 --k 15\n  rc={rc}")
    assert rc == 0
    for s in m["shards"]:
        gt = np.load(sd / "ground_truth" / f"shard_{s['shard_id']}.npy")
        x = train[s["offset"] : s["offset"] + s["size"]]
        assert gt.dtype == np.int32 and gt.shape == (20, min(15, s["size"]))
        d = ref_topk(test, x, 15)
        for i in range(20):
            got_d = d[i][gt[i]]
            assert np.array_equal(got_d, np.sort(d[i])[: gt.shape[1]]), f"shard {s['shard_id']} query {i}"
    print("  each ground_truth/shard_k.npy is int32 [20, 15], local indices, the exact top-15 by float64 distance (ties by distance)")


def test_xcheck_cli_split_manifest_covers_ground_truth(tmp_path):
    """Decision 9: 'a manifest per split records offset, size, rank and the
    sha256 of every object'; the store docstring puts ground_truth/shard_k.npy
    under the split. So after groundtruth, the split's manifest (or some
    manifest) should carry the ground-truth objects' sha256, or verify
    cannot check them."""
    make_sift_like(tmp_path)
    assert run(["split", "sift-128", "6"], tmp_path)[0] == 0
    assert run(["groundtruth", "sift-128", "6"], tmp_path)[0] == 0
    sd = tmp_path / "splits" / "sift-128" / "6"
    m = json.loads((sd / "manifest.json").read_text())
    files = sorted(p.relative_to(sd).as_posix() for p in sd.rglob("*.npy"))
    missing = [f for f in files if f not in m["sha256"]]
    print(f"did: split + groundtruth; objects under the split: {len(files)}; sha256 entries in manifest.json: {len(m['sha256'])}\n"
          f"  objects with no sha256 anywhere in the manifest: {missing}")
    assert not missing


def test_xcheck_cli_split_is_deterministic(tmp_path):
    make_sift_like(tmp_path)
    assert run(["split", "sift-128", "6"], tmp_path)[0] == 0
    a = json.loads((tmp_path / "splits/sift-128/6/manifest.json").read_text())
    assert run(["split", "sift-128", "6"], tmp_path)[0] == 0
    b = json.loads((tmp_path / "splits/sift-128/6/manifest.json").read_text())
    print(f"did: split sift-128 6 twice; manifests identical (incl. sha256): {a == b}")
    assert a == b


def test_xcheck_cli_groundtruth_before_split_fails(tmp_path):
    make_sift_like(tmp_path)
    rc, out, err = run(["groundtruth", "sift-128", "6"], tmp_path)
    print(f"did: groundtruth with no split on disk\n  expected: non-zero exit\n  seen: rc={rc}\n      {tail(out + err, 3)}")
    assert rc != 0


def test_xcheck_cli_gist_train_limit_with_a_small_file(tmp_path):
    """gist-960-300k uses the first 300k train rows (decision 10). With a
    file holding fewer rows the spec is silent; record what happens."""
    import h5py

    rng = np.random.default_rng(1)
    (tmp_path / "ann-benchmarks").mkdir()
    with h5py.File(tmp_path / "ann-benchmarks" / "gist-960-euclidean.hdf5", "w") as f:
        f.create_dataset("train", data=rng.random((500, 960), dtype=np.float32))
        f.create_dataset("test", data=rng.random((5, 960), dtype=np.float32))
    rc, out, err = run(["split", "gist-960-300k", "6"], tmp_path)
    nv = None
    mp = tmp_path / "splits/gist-960-300k/6/manifest.json"
    if mp.exists():
        nv = json.loads(mp.read_text())["n_vectors"]
    print(f"did: split gist-960-300k 6 on a 500-row file (train_limit 300,000)\n  seen: rc={rc}, manifest n_vectors={nv}\n      {tail(out + err, 3)}")
    assert rc != 0 or nv == 500


def test_xcheck_cli_all_cannot_run_on_a_custom_file(tmp_path):
    """'all' starts with download, which checks the catalogue's size; so the
    full pipeline cannot be driven on a synthetic file. Record (gap)."""
    make_sift_like(tmp_path)
    rc, out, err = run(["all", "sift-128", "6"], tmp_path, timeout=60)
    print(f"did: all sift-128 6 on the synthetic 1000-row file, network blocked\n  seen: rc={rc}\n      {tail(out + err, 3)}")


# ---------------------------------------------------------------- upload + verify (needs a store)

S3 = os.environ.get("TEST_S3_ENDPOINT")


@pytest.mark.skipif(not S3, reason="TEST_S3_ENDPOINT unset")
def test_xcheck_cli_upload_verify_and_tamper(tmp_path):
    from kgpu_builder.config import Config
    from kgpu_builder.store import Store

    bucket = "xcheck-cli-" + uuid.uuid4().hex[:8]
    s3env = dict(
        S3_ENDPOINT=S3, S3_ACCESS_KEY=os.environ.get("TEST_S3_ACCESS_KEY", ""),
        S3_SECRET_KEY=os.environ.get("TEST_S3_SECRET_KEY", ""), S3_BUCKET=bucket,
        no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost",
    )
    make_sift_like(tmp_path)
    assert run(["split", "sift-128", "6"], tmp_path)[0] == 0
    assert run(["groundtruth", "sift-128", "6"], tmp_path)[0] == 0
    rc, out, err = run(["upload", "sift-128", "6"], tmp_path, s3env)
    print(f"did: upload sift-128 6 to a fresh bucket {bucket}\n  rc={rc}\n      {tail(out + err, 4)}")
    assert rc == 0
    st = Store(Config(data_dir=tmp_path, s3_endpoint=S3, s3_access_key=s3env["S3_ACCESS_KEY"],
                      s3_secret_key=s3env["S3_SECRET_KEY"], s3_bucket=bucket, s3_region="us-east-1"))
    keys = sorted(st.list(""))
    print(f"  bucket holds {len(keys)} objects: {keys}")
    want = {"datasets/sift-128/queries.npy", "datasets/sift-128/manifest.json", "splits/sift-128/6/manifest.json"}
    want |= {f"splits/sift-128/6/vectors/shard_{k}.npy" for k in range(6)}
    want |= {f"splits/sift-128/6/ground_truth/shard_{k}.npy" for k in range(6)}
    assert want <= set(keys), f"missing: {sorted(want - set(keys))}"
    for k in sorted(want):
        local = (tmp_path / k).read_bytes()
        assert st.get_bytes(k) == local, f"{k} differs from the local file"
    print("  every object equals its local file byte for byte (decision 9 layout)")

    rc, out, err = run(["verify", "sift-128", "6"], tmp_path, s3env)
    print(f"did: verify on the intact upload: rc={rc} (expected 0)\n      {tail(out + err, 3)}")
    assert rc == 0

    # tamper a vectors object: same size, different bytes
    key = "splits/sift-128/6/vectors/shard_2.npy"
    b = bytearray(st.get_bytes(key))
    b[-1] ^= 0xFF
    st.put_bytes(key, bytes(b))
    rc, out, err = run(["verify", "sift-128", "6"], tmp_path, s3env)
    print(f"did: flipped the last byte of {key} (size unchanged) and ran verify\n  expected: non-zero exit\n  seen: rc={rc}\n      {tail(out + err, 3)}")
    st.put_bytes(key, (tmp_path / key).read_bytes())
    assert rc != 0

    # tamper a ground-truth object
    key = "splits/sift-128/6/ground_truth/shard_1.npy"
    b = bytearray(st.get_bytes(key))
    b[-1] ^= 0xFF
    st.put_bytes(key, bytes(b))
    rc_gt, out, err = run(["verify", "sift-128", "6"], tmp_path, s3env)
    print(f"did: flipped the last byte of {key} and ran verify\n  expected: non-zero exit (help: 'every object present with the manifest's size/sha256')\n  seen: rc={rc_gt}\n      {tail(out + err, 3)}")
    st.put_bytes(key, (tmp_path / key).read_bytes())

    # delete the queries
    key = "datasets/sift-128/queries.npy"
    st.delete(key)  # an exact key: delete_prefix names a directory (G7)
    rc_q, out, err = run(["verify", "sift-128", "6"], tmp_path, s3env)
    print(f"did: deleted {key} and ran verify\n  expected: non-zero exit\n  seen: rc={rc_q}\n      {tail(out + err, 3)}")
    st.delete_prefix("")
    assert rc_gt != 0, "verify did not notice a corrupted ground-truth object"
    assert rc_q != 0, "verify did not notice missing queries"
