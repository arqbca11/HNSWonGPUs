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
thread count, A is fitted IN LOG SPACE — the geometric mean of
measured / (n·ln(n)·dim) over the points, i.e. least squares of
ln(measured) on ln(x) with a unit slope — because the error that matters to
the scheduler is multiplicative: a build twice as long as estimated is the
same mistake at every size. (A linear-space least squares through the
origin is dominated by the largest builds and leaves most points below
the line: the step 3 cross-check measured a −0.2 bias in the log residuals
of such a fit, G1.) The residual of a point is measured / fitted; with the
log-space A the log residuals are centred, and sigma — their standard
deviation (population, ddof 0) — is also their root-mean-square: the
log-normal factor the simulator applies (Phase 3 decision 23). The
linear-space A is printed beside it for reference. A free-exponent fit
t = c · n^p · dim is reported when at least two distinct sizes exist (n/a
otherwise), so a curve steeper than n·ln(n) is visible as p > 1. Between
two thread counts the ratio of their A's is what the cost model calls
Speedup when the "GPU" is a CPU worker with more threads (roadmap, Phase
4). The transfer term is the median (the middle value, averaging the two
middles for an even count) of the worker points' fetch bytes / seconds;
the overhead is, unless given, the median over the worker points of
fetch + gate seconds — the per-build fixed cost the build time excludes.

Grid details the tests asked about: sizes and thread counts run in the
order given; the largest shard is the manifest's largest `size`; sizes
below 2 are refused up front, and every size is checked against the shard
before any point runs; `fetch_bytes` is the whole shard's raw vector bytes
(rows × dim × 4), fetched in full for every point; with fewer than 500
held-out queries all of them are used; a point whose recall is below the
gate is still recorded (this is a measurement, not a build); a rerun into
an existing CSV continues the `repeat` numbering per (size, threads) and
refuses a file whose header is not the point schema. `maxrss_bytes` is in
bytes on every platform (ru_maxrss is KiB on Linux and bytes on macOS; the
conversion is done here).
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
    if size < 2:
        raise ValueError(f"size {size}: a calibration point needs at least 2 vectors")
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
    if any(s < 2 for s in sizes):
        raise ValueError(f"sizes must be at least 2, got {sizes}")
    # the existing file: its header must be the point schema, and repeats
    # continue the numbering per (size, threads) (cross-check F7, G4)
    existing: dict[tuple[int, int], int] = {}
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    if not new:
        with open(out_csv, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != POINT_FIELDS:
                raise ValueError(f"{out_csv} exists with a different header ({rd.fieldnames[:3]}...); choose another --out")
            for row in rd:
                key = (int(row["size"]), int(row["threads"]))
                existing[key] = max(existing.get(key, -1), int(row["repeat"]))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    points: list[Point] = []
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        if new:
            w.writeheader()
        for th in threads:
            for size in sizes:
                for r in range(repeats):
                    rep = existing.get((size, th), -1) + 1 + r
                    argv = cmd + ["calibrate-point", dataset, str(n_shards), "--shard", str(shard), "--size", str(size),
                                  "--threads", str(th), "--repeat", str(rep)]
                    p = subprocess.run(argv, capture_output=True, text=True, env=env)
                    if p.returncode != 0:
                        raise RuntimeError(f"calibrate-point failed ({p.returncode}): {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.stdout.strip()[-200:]}")
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
    a: float          # seconds per n·ln(n)·dim, fitted in log space (geometric mean of t/x)
    a_linear: float   # the same by linear least squares through the origin, for reference
    sigma: float      # std (= RMS, the log residuals are centred) of ln(measured / fitted): the simulator's log-normal factor
    bias_linear: float  # mean ln(measured / (a_linear·x)): what the linear fit hides (cross-check G1)
    max_ratio: float  # worst measured / fitted
    min_ratio: float
    power_c: float | None  # free-exponent fit t = c · n^p · dim; None with fewer than two distinct sizes
    power_p: float | None
    power_sigma: float | None
    recall_min: float
    bandwidth_bps: float  # median fetch bytes / s over the points
    overhead_s: float     # median fetch + gate seconds over the points


def nlogn(n: np.ndarray) -> np.ndarray:
    return n * np.log(n)


def fit_points(points: list[dict]) -> list[Fit]:
    fits = []
    for th in sorted({int(p["threads"]) for p in points}):
        ps = [p for p in points if int(p["threads"]) == th]
        n = np.array([float(p["size"]) for p in ps])
        d = np.array([float(p["dim"]) for p in ps])
        t = np.array([float(p["build_s"]) for p in ps])
        if (n < 2).any() or (t <= 0).any():
            raise ValueError(f"threads {th}: every point needs size >= 2 and build_s > 0")
        x = nlogn(n) * d
        logr = np.log(t / x)
        a = float(math.exp(logr.mean()))                  # log space: the geometric mean of t/x
        a_lin = float((x * t).sum() / (x * x).sum())      # linear space, for reference
        ratio = t / (a * x)
        sigma = float(np.std(np.log(ratio)))              # centred by construction; population std = RMS
        bias_lin = float(np.mean(np.log(t / (a_lin * x))))
        c = pexp = psig = None
        if len(set(n.tolist())) >= 2:
            X = np.vstack([np.ones_like(n), np.log(n)]).T  # ln(t/d) = ln c + p ln n
            coef, *_ = np.linalg.lstsq(X, np.log(t / d), rcond=None)
            c, pexp = float(math.exp(coef[0])), float(coef[1])
            psig = float(np.std(np.log(t / (c * n ** pexp * d)))) if len(set(n.tolist())) > 2 else 0.0
        bw = float(np.median([float(p.get("fetch_bytes", 0)) / max(float(p.get("fetch_s", 0)), 1e-9) for p in ps]))
        oh = float(np.median([float(p.get("fetch_s", 0)) + float(p.get("gate_s", 0)) for p in ps]))
        fits.append(Fit(th, len(ps), a, a_lin, sigma, bias_lin, float(ratio.max()), float(ratio.min()), c, pexp, psig,
                        min(float(p["recall"]) for p in ps), bw, oh))
    return fits


def residual_table(points: list[dict], fits: list[Fit]) -> list[str]:
    by = {f.threads: f for f in fits}
    lines = [f"{'threads':>7} {'size':>8} {'dim':>5} {'measured':>9} {'fitted':>9} {'ratio':>6} {'power':>9} {'recall':>7}"]
    for p in sorted(points, key=lambda p: (int(p["threads"]), int(p["size"]), int(p["dim"]))):
        f = by[int(p["threads"])]
        n, d, t = float(p["size"]), float(p["dim"]), float(p["build_s"])
        fitted = f.a * n * math.log(n) * d
        power = f"{f.power_c * n ** f.power_p * d:>9.2f}" if f.power_p is not None else f"{'n/a':>9}"
        lines.append(f"{int(p['threads']):>7} {int(p['size']):>8} {int(d):>5} {t:>9.2f} {fitted:>9.2f} {t / fitted:>6.2f} {power} {float(p['recall']):>7.3f}")
    return lines


def env_overrides(fits: list[Fit], shard_threads: int, worker_threads: int, overhead_s: float | None = None) -> dict[str, str]:
    """The COST_* block for a cluster whose shards build with shard_threads and
    whose workers with worker_threads (both CPU: the 'GPU' speedup is the
    ratio of the two fitted constants). overhead_s None means the measured
    median fetch + gate seconds of the worker points (cross-check G2)."""
    by = {f.threads: f for f in fits}
    if shard_threads not in by or worker_threads not in by:
        raise ValueError(f"need fits for {shard_threads} and {worker_threads} threads; have {sorted(by)}")
    a_shard, a_worker = by[shard_threads].a, by[worker_threads].a
    oh = by[worker_threads].overhead_s if overhead_s is None else overhead_s
    return {
        "COST_A": f"{a_shard:.4g}",
        "COST_SPEEDUP": f"{a_shard / a_worker:.3f}",
        "COST_GPU_OVERHEAD": f"{oh:.2f}",
        "COST_BANDWIDTH": f"{by[worker_threads].bandwidth_bps:.4g}",
        "KGPU_ESTIMATE_SIGMA": f"{max(by[shard_threads].sigma, by[worker_threads].sigma):.3f}",
    }
