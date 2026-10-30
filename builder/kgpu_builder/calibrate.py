"""Calibration (Phase 4 steps 3 and 5): measure the real build over a grid of
sizes and thread counts, then fit the cost model's constants and keep the
residuals (decisions 6, 23, 24).

A calibration POINT is one real build: the first `size` vectors of a split's
largest shard (so every size is a prefix of the same data), built with the
same hnswlib parameters a real build uses, gated against an exact top-10
computed for that prefix on a sample of the held-out queries. Each point
runs in its own process (the CLI's `calibrate-point`), so its peak memory
(ru_maxrss) is its own and the previous point's allocations do not inflate
it: the whole shard's array while it is read (freed before the build), the
prefix, its index, and at most ~100 MB for the exact truth. The parent
writes one CSV row per point.

The FIT: the cost model says CPU build time = A · n · ln(n) · dim. For each
thread count, A is the least-squares slope through the origin of time
against n·ln(n)·dim; the residual of a point is measured / fitted, and the
estimate-error sigma is the standard deviation of ln(measured / fitted) —
the same log-normal factor the simulator applies (Phase 3 decision 23). A
free-exponent fit t = c · n^p · dim is reported beside it, so a curve
steeper than n·ln(n) is visible as p > 1. Between two thread counts the
ratio of their A's is what the cost model calls Speedup when the "GPU" is
a CPU worker with more threads (roadmap, Phase 4). The transfer term comes
from the points' fetch times: bytes / seconds.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from .build import build_index, gate
from .groundtruth import exact_topk
from .store import Store

POINT_FIELDS = ["dataset", "n_shards", "shard", "size", "dim", "threads", "m", "ef_construction", "repeat",
                "fetch_s", "fetch_bytes", "build_s", "gate_s", "recall", "index_bytes", "maxrss_bytes", "host", "at"]


@dataclass(frozen=True)
class Point:
    dataset: str
    n_shards: int
    shard: int
    size: int
    dim: int
    threads: int
    m: int
    ef_construction: int
    repeat: int
    fetch_s: float
    fetch_bytes: int
    build_s: float
    gate_s: float
    recall: float
    index_bytes: int
    maxrss_bytes: int
    host: str
    at: str


def largest_shard(store: Store, dataset: str, n_shards: int) -> int:
    m = store.get_json(f"splits/{dataset}/{n_shards}/manifest.json")
    return max(m["shards"], key=lambda s: s["size"])["shard_id"]


def run_point(store: Store, dataset: str, n_shards: int, shard: int, size: int, threads: int,
              m: int = 16, ef_construction: int = 200, repeat: int = 0, n_queries: int = 500) -> Point:
    """One calibration build, in THIS process (the CLI runs it in a child)."""
    t0 = time.time()
    vectors = store.get_npy(f"splits/{dataset}/{n_shards}/vectors/shard_{shard}.npy")
    fetch_s = time.time() - t0
    fetch_bytes = vectors.nbytes
    if size > vectors.shape[0]:
        raise ValueError(f"size {size} exceeds shard {shard}'s {vectors.shape[0]} vectors")
    x = np.ascontiguousarray(vectors[:size])
    del vectors
    queries = store.get_npy(f"datasets/{dataset}/queries.npy")[:n_queries]
    # small query chunks: the exact truth's float64 distance block must not
    # set the process's memory high-water mark (32 x 400k x 8 B = 100 MB),
    # so peak RSS reflects the prefix and its index, plus at most that
    truth = exact_topk(queries, x, k=10, chunk=32)
    t1 = time.time()
    idx = build_index(x, m, ef_construction, threads)
    build_s = time.time() - t1
    t2 = time.time()
    recall = gate(idx, queries, truth, 10, 100, threads)
    gate_s = time.time() - t2
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.bin")
        idx.save_index(p)
        index_bytes = os.path.getsize(p)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024  # Linux reports KiB; macOS bytes
    return Point(dataset, n_shards, shard, size, int(x.shape[1]), threads, m, ef_construction, repeat,
                 fetch_s, fetch_bytes, build_s, gate_s, recall, index_bytes, int(rss), os.uname().nodename,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_grid(cmd: list[str], env: dict[str, str], dataset: str, n_shards: int, shard: int, sizes: list[int],
             threads: list[int], repeats: int, out_csv: str, log=print) -> list[Point]:
    """Every (size, threads, repeat) as a child process; rows appended to out_csv
    as they come, so an interrupted grid keeps what it measured."""
    new = not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    points: list[Point] = []
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        if new:
            w.writeheader()
        for th in threads:
            for size in sizes:
                for r in range(repeats):
                    argv = cmd + ["calibrate-point", dataset, str(n_shards), "--shard", str(shard), "--size", str(size),
                                  "--threads", str(th), "--repeat", str(r)]
                    p = subprocess.run(argv, capture_output=True, text=True, env=env)
                    if p.returncode != 0:
                        raise RuntimeError(f"calibrate-point failed ({p.returncode}): {p.stderr.strip()[-400:]}")
                    pt = Point(**json.loads(p.stdout))
                    points.append(pt)
                    w.writerow(asdict(pt))
                    f.flush()
                    log(f"  {dataset} shard {shard} size {size:7d} x {pt.dim} threads {th} repeat {r}: build {pt.build_s:7.2f} s, "
                        f"recall@10 {pt.recall:.3f}, index {pt.index_bytes / 1e6:6.1f} MB, peak rss {pt.maxrss_bytes / 1e6:6.0f} MB, fetch {pt.fetch_bytes / 1e6 / max(pt.fetch_s, 1e-9):5.0f} MB/s")
    return points


# ---- the fit ---------------------------------------------------------------

@dataclass(frozen=True)
class Fit:
    threads: int
    n_points: int
    a: float          # seconds per n·ln(n)·dim, least squares through the origin
    sigma: float      # std of ln(measured / fitted): the simulator's log-normal factor
    max_ratio: float  # worst measured / fitted
    min_ratio: float
    power_c: float    # free-exponent fit t = c · n^p · dim
    power_p: float
    power_sigma: float
    recall_min: float
    bandwidth_bps: float  # median fetch bytes / s over the points


def nlogn(n: np.ndarray) -> np.ndarray:
    return n * np.log(n)


def fit_points(points: list[dict]) -> list[Fit]:
    fits = []
    for th in sorted({int(p["threads"]) for p in points}):
        ps = [p for p in points if int(p["threads"]) == th]
        n = np.array([float(p["size"]) for p in ps])
        d = np.array([float(p["dim"]) for p in ps])
        t = np.array([float(p["build_s"]) for p in ps])
        x = nlogn(n) * d
        a = float((x * t).sum() / (x * x).sum())  # through the origin: t = a·x
        ratio = t / (a * x)
        sigma = float(np.std(np.log(ratio))) if len(ps) > 1 else 0.0
        # power law: ln(t/d) = ln c + p ln n
        X = np.vstack([np.ones_like(n), np.log(n)]).T
        coef, *_ = np.linalg.lstsq(X, np.log(t / d), rcond=None)
        c, pexp = float(math.exp(coef[0])), float(coef[1])
        pratio = t / (c * n ** pexp * d)
        bw = sorted(float(p["fetch_bytes"]) / max(float(p["fetch_s"]), 1e-9) for p in ps)
        fits.append(Fit(th, len(ps), a, sigma, float(ratio.max()), float(ratio.min()), c, pexp,
                        float(np.std(np.log(pratio))) if len(ps) > 2 else 0.0,
                        min(float(p["recall"]) for p in ps), bw[len(bw) // 2]))
    return fits


def residual_table(points: list[dict], fits: list[Fit]) -> list[str]:
    by = {f.threads: f for f in fits}
    lines = [f"{'threads':>7} {'size':>8} {'dim':>5} {'measured':>9} {'fitted':>9} {'ratio':>6} {'power':>9} {'recall':>7}"]
    for p in sorted(points, key=lambda p: (int(p["threads"]), int(p["size"]), int(p["dim"]))):
        f = by[int(p["threads"])]
        n, d, t = float(p["size"]), float(p["dim"]), float(p["build_s"])
        fitted = f.a * n * math.log(n) * d
        power = f.power_c * n ** f.power_p * d
        lines.append(f"{int(p['threads']):>7} {int(p['size']):>8} {int(d):>5} {t:>9.2f} {fitted:>9.2f} {t / fitted:>6.2f} {power:>9.2f} {float(p['recall']):>7.3f}")
    return lines


def env_overrides(fits: list[Fit], shard_threads: int, worker_threads: int, overhead_s: float) -> dict[str, str]:
    """The COST_* block for a cluster whose shards build with shard_threads and
    whose workers with worker_threads (both CPU: the 'GPU' speedup is the
    ratio of the two fitted constants)."""
    by = {f.threads: f for f in fits}
    if shard_threads not in by or worker_threads not in by:
        raise ValueError(f"need fits for {shard_threads} and {worker_threads} threads; have {sorted(by)}")
    a_shard, a_worker = by[shard_threads].a, by[worker_threads].a
    return {
        "COST_A": f"{a_shard:.4g}",
        "COST_SPEEDUP": f"{a_shard / a_worker:.3f}",
        "COST_GPU_OVERHEAD": f"{overhead_s:.2f}",
        "COST_BANDWIDTH": f"{by[worker_threads].bandwidth_bps:.4g}",
        "KGPU_ESTIMATE_SIGMA": f"{max(by[shard_threads].sigma, by[worker_threads].sigma):.3f}",
    }
