"""python -m kgpu_builder <command> — Phase 4 step 1's data pipeline.

    download  <dataset>                      fetch the HDF5 (resumable), verify size
    split     <dataset> <n_shards> [--zipf S] [--seed P]
                                             write the split locally: vectors per shard + manifest
    groundtruth <dataset> <n_shards> [--k 100]
                                             exact top-k per shard for the dataset's queries
    upload    <dataset> <n_shards>           push queries, split, ground truth to the object store
    verify    <dataset> <n_shards>           every object present with the manifest's size/sha256
    all       <dataset> <n_shards>           the four above in order

Datasets: sift-128, gist-960-300k. The HDF5 files live under
KGPU_DATA_DIR/ann-benchmarks/<file from the catalogue>; a file placed there
by hand (a different dataset under a catalogue name, for a test) drives
split, groundtruth, upload and verify, but `download` and `all` check its
size against the catalogue first. Local outputs under KGPU_DATA_DIR/splits/
<dataset>/<n_shards>/; the object store gets the same layout under splits/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
from pathlib import Path

import h5py
import numpy as np

from . import datasets, download as dl, groundtruth, split as sp
from .config import Config
from .store import NotFound, Store


def _log(msg: str, **_):
    print(msg, flush=True)


def _open_train(ds: datasets.Dataset, cfg: Config):
    path = cfg.data_dir / "ann-benchmarks" / ds.filename
    if not path.exists():
        raise SystemExit(f"{path} missing: run `download {ds.name}` first")
    f = h5py.File(path, "r")
    train = f["train"]
    if train.ndim != 2 or train.shape[1] != ds.dim:
        raise SystemExit(f"{path}: train is {train.shape}, the catalogue says dim {ds.dim}")
    if ds.train_limit is not None and train.shape[0] < ds.train_limit:
        raise SystemExit(f"{path}: {train.shape[0]} train rows, fewer than the {ds.train_limit} the catalogue takes for {ds.name}")
    n = train.shape[0] if ds.train_limit is None else ds.train_limit
    return f, train, n


def cmd_download(ds, cfg, args):
    dl.download(ds, cfg.data_dir, log=_log)


def split_dir(ds, cfg, n):
    return cfg.data_dir / "splits" / ds.name / str(n)


def cmd_split(ds, cfg, args):
    f, train, n = _open_train(ds, cfg)
    with f:
        m = sp.plan(ds.name, ds.dim, ds.metric, n, args.n_shards, exponent=args.zipf, perm_seed=args.seed)
        _log(f"{ds.name}: {n} vectors x {ds.dim} across {args.n_shards} shards, zipf s={args.zipf}, rank permutation seed {args.seed}")
        t0 = time.time()
        m = sp.write_split(train, m, split_dir(ds, cfg, args.n_shards), log=_log)
        q = np.asarray(f["test"], dtype=np.float32)
        qdir = cfg.data_dir / "datasets" / ds.name
        qdir.mkdir(parents=True, exist_ok=True)
        qb = sp.npy_bytes(q)
        (qdir / "queries.npy").write_bytes(qb)
        (qdir / "manifest.json").write_text(json.dumps({
            "dataset": ds.name, "source": ds.url, "dim": ds.dim, "metric": ds.metric,
            "n_train_used": n, "n_train_in_file": int(train.shape[0]), "n_queries": int(q.shape[0]),
            "sha256": {"queries.npy": hashlib.sha256(qb).hexdigest()},
        }, indent=2))
    _log(f"split written in {time.time() - t0:.1f} s: {split_dir(ds, cfg, args.n_shards)}; {q.shape[0]} queries")


def cmd_groundtruth(ds, cfg, args):
    d = split_dir(ds, cfg, args.n_shards)
    m = sp.SplitManifest.from_json((d / "manifest.json").read_text())
    q = np.load(cfg.data_dir / "datasets" / ds.name / "queries.npy", allow_pickle=False)
    (d / "ground_truth").mkdir(exist_ok=True)
    t0 = time.time()
    sha = dict(m.sha256)
    for s in m.shards:
        x = np.load(d / f"vectors/shard_{s.shard_id}.npy", allow_pickle=False)
        ts = time.time()
        gt = groundtruth.exact_topk(q, x, k=args.k)
        b = sp.npy_bytes(gt)
        name = f"ground_truth/shard_{s.shard_id}.npy"
        (d / name).write_bytes(b)
        sha[name] = hashlib.sha256(b).hexdigest()
        _log(f"  shard {s.shard_id:2d}: {x.shape[0]:8d} vectors, {q.shape[0]} queries, top-{gt.shape[1]} exact in {time.time() - ts:5.1f} s")
    # the manifest now covers the ground truth too (cross-check F3/F4: verify
    # used to report it intact without a hash to check) and records k (G8)
    m = sp.SplitManifest(**{**sp.asdict(m), "shards": m.shards, "sha256": sha, "ground_truth_k": args.k})
    (d / "manifest.json").write_text(m.to_json())
    _log(f"ground truth for {len(m.shards)} shards in {time.time() - t0:.1f} s; manifest updated with their sha256 and k={args.k}")


def cmd_upload(ds, cfg, args):
    d = split_dir(ds, cfg, args.n_shards)
    m = sp.SplitManifest.from_json((d / "manifest.json").read_text())
    st = Store(cfg)
    st.ensure_bucket()
    prefix = f"splits/{ds.name}/{args.n_shards}"
    total = 0
    t0 = time.time()
    qdir = cfg.data_dir / "datasets" / ds.name
    for name in ("queries.npy", "manifest.json"):
        b = (qdir / name).read_bytes()
        st.put_bytes(f"datasets/{ds.name}/{name}", b)
        total += len(b)
    for s in m.shards:
        for sub in ("vectors", "ground_truth"):
            p = d / sub / f"shard_{s.shard_id}.npy"
            if not p.exists():
                raise SystemExit(f"{p} missing: run split and groundtruth first")
            b = p.read_bytes()
            st.put_bytes(f"{prefix}/{sub}/shard_{s.shard_id}.npy", b)
            total += len(b)
            _log(f"  put {prefix}/{sub}/shard_{s.shard_id}.npy ({len(b) / 1e6:.1f} MB)")
    st.put_bytes(f"{prefix}/manifest.json", (d / "manifest.json").read_bytes(), "application/json")
    dt = time.time() - t0
    _log(f"uploaded {total / 1e6:.1f} MB in {dt:.1f} s ({total / 1e6 / max(dt, 1e-9):.0f} MB/s) to {cfg.s3_endpoint}/{cfg.s3_bucket}/{prefix}")


def cmd_verify(ds, cfg, args):
    """Every object the manifests name must be present with the recorded
    sha256 — vectors, ground truth and the queries alike. A missing object
    is a reported problem, not a traceback (cross-check F4, G5)."""
    st = Store(cfg)
    prefix = f"splits/{ds.name}/{args.n_shards}"
    m = sp.SplitManifest.from_json(st.get_bytes(f"{prefix}/manifest.json").decode())
    bad = 0
    expected = set(m.sha256)
    for s in m.shards:
        for sub_ in ("vectors", "ground_truth"):
            name = f"{sub_}/shard_{s.shard_id}.npy"
            if name not in expected:
                _log(f"  {prefix}/{name}: NO HASH IN THE MANIFEST (run groundtruth again)")
                bad += 1
    for name, want in sorted(m.sha256.items()):
        try:
            b = st.get_bytes(f"{prefix}/{name}")
        except NotFound:
            _log(f"  {prefix}/{name}: MISSING")
            bad += 1
            continue
        ok = hashlib.sha256(b).hexdigest() == want
        bad += not ok
        _log(f"  {prefix}/{name}: {len(b) / 1e6:7.1f} MB sha256 {'ok' if ok else 'MISMATCH'}")
    try:
        dm = st.get_json(f"datasets/{ds.name}/manifest.json")
        qb = st.get_bytes(f"datasets/{ds.name}/queries.npy")
    except NotFound as e:
        _log(f"  datasets/{ds.name}: MISSING ({e})")
        raise SystemExit(f"verify: {bad + 1} problem(s)")
    qok = hashlib.sha256(qb).hexdigest() == dm["sha256"]["queries.npy"]
    bad += not qok
    _log(f"  datasets/{ds.name}/queries.npy: {len(qb) / 1e6:.1f} MB sha256 {'ok' if qok else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"verify: {bad} problem(s)")
    nv = sum(1 for n in m.sha256 if n.startswith("vectors/"))
    ng = sum(1 for n in m.sha256 if n.startswith("ground_truth/"))
    _log(f"verify: {nv} vector objects, {ng} ground-truth objects (top-{m.ground_truth_k}) and the queries all present with their recorded sha256")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m kgpu_builder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "split", "groundtruth", "upload", "verify", "all"):
        sp_ = sub.add_parser(name)
        sp_.add_argument("dataset")
        if name != "download":
            sp_.add_argument("n_shards", type=int)
        if name in ("split", "all"):
            sp_.add_argument("--zipf", type=float, default=1.0)
            sp_.add_argument("--seed", type=int, default=1)
        if name in ("groundtruth", "all"):
            sp_.add_argument("--k", type=int, default=100)
    args = p.parse_args(argv)
    cfg = Config.from_env()
    ds = datasets.get(args.dataset)
    steps = {"download": cmd_download, "split": cmd_split, "groundtruth": cmd_groundtruth, "upload": cmd_upload, "verify": cmd_verify}
    try:
        if args.cmd == "all":
            for name in ("download", "split", "groundtruth", "upload", "verify"):
                _log(f"== {name}")
                steps[name](ds, cfg, args)
        else:
            steps[args.cmd](ds, cfg, args)
    except FileNotFoundError as e:
        raise SystemExit(f"{args.cmd}: {e.filename or e} not found — run the earlier steps first (download, split, groundtruth)") from None
    except NotFound as e:
        raise SystemExit(f"{args.cmd}: object not in the store: {e}") from None
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"{args.cmd}: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())
