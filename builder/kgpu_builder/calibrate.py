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

POINT_FIELDS_V0 = ["dataset", "n_shards", "shard", "size", "dim", "threads", "m", "ef_construction", "repeat",
                   "fetch_s", "fetch_bytes", "build_s", "gate_s", "recall", "index_bytes", "maxrss_bytes", "host", "at"]
# v1 (Phase 5 step 1): the point also times the rest of the worker's path —
# the save (write + sha256) and the upload to the store — and records how
# many queries the gate ran, so the gate cost is per query. A CSV with the
# v0 header still reads; its v1 columns are absent.
POINT_FIELDS_V1 = POINT_FIELDS_V0 + ["gate_queries", "save_s", "upload_s", "upload_bytes"]
# Phase 5 (the GPU session): which builder made the point — hnswlib on the
# CPU or cagra on the GPU — so one file can hold both sides of the speedup.
POINT_FIELDS = POINT_FIELDS_V1 + ["builder"]
ACCEPTED_HEADERS = (POINT_FIELDS, POINT_FIELDS_V1, POINT_FIELDS_V0)


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
    gate_queries: int = 0
    save_s: float = 0.0     # write the index to disk and sha256 it (v1)
    upload_s: float = 0.0   # put it in the store under a scratch key, then delete it (v1; 0 when not asked)
    upload_bytes: int = 0
    builder: str = "hnswlib"   # or cagra (Phase 5)


def largest_shard(store: Store, dataset: str, n_shards: int) -> int:
    m = store.get_json(f"splits/{dataset}/{n_shards}/manifest.json")
    return max(m["shards"], key=lambda s: s["size"])["shard_id"]


def run_point(store: Store, dataset: str, n_shards: int, shard: int, size: int, threads: int,
              m: int = 16, ef_construction: int = 200, repeat: int = 0, n_queries: int = 500,
              upload_prefix: str | None = None, builder: str = "hnswlib") -> Point:
    """One calibration build, in THIS process (the CLI runs it in a child).
    upload_prefix, when given, is a store prefix the saved index is uploaded
    under (`<prefix>/<host>-<size>-<threads>-<repeat>.bin`) and deleted from
    again, so the worker's upload is timed on the real path."""
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
    if builder == "cagra":
        from .cagra import build_index_cagra
        idx = build_index_cagra(x, m, ef_construction, threads)
    else:
        idx = build_index(x, m, ef_construction, threads)
    build_s = time.time() - t1
    t2 = time.time()
    recall = gate(idx, queries, truth, 10, 100, threads)
    gate_s = time.time() - t2
    import hashlib
    import tempfile
    upload_s = 0.0
    upload_bytes = 0
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.bin")
        t3 = time.time()
        idx.save_index(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        save_s = time.time() - t3   # what a real build pays before it can publish (build.run_build)
        index_bytes = os.path.getsize(p)
        if upload_prefix:
            key = f"{upload_prefix.rstrip('/')}/{os.uname().nodename}-{size}-{threads}-{repeat}.bin"
            t4 = time.time()
            upload_bytes = store.put_file(key, p)
            upload_s = time.time() - t4
            store.delete(key)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024  # Linux reports KiB; macOS bytes
    return Point(dataset, n_shards, shard, size, int(x.shape[1]), threads, m, ef_construction, repeat,
                 fetch_s, fetch_bytes, build_s, gate_s, recall, index_bytes, int(rss), os.uname().nodename,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 int(queries.shape[0]), save_s, upload_s, int(upload_bytes), builder)


def run_grid(cmd: list[str], env: dict[str, str], dataset: str, n_shards: int, shard: int, sizes: list[int],
             threads: list[int], repeats: int, out_csv: str, log=print, upload_prefix: str | None = None,
             builder: str = "hnswlib") -> list[Point]:
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
                raise ValueError(f"{out_csv} exists with a different header ({rd.fieldnames[:3]}...; an older calibration CSV cannot take new rows); choose another --out")
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
                                  "--threads", str(th), "--repeat", str(rep), "--builder", builder] + (["--upload-prefix", upload_prefix] if upload_prefix else [])
                    p = subprocess.run(argv, capture_output=True, text=True, env=env)
                    if p.returncode != 0:
                        raise RuntimeError(f"calibrate-point failed ({p.returncode}): {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.stdout.strip()[-200:]}")
                    pt = Point(**json.loads(p.stdout))
                    points.append(pt)
                    w.writerow(asdict(pt))
                    f.flush()
                    log(f"  {dataset} shard {shard} size {size:7d} x {pt.dim} threads {th} repeat {r}: build {pt.build_s:7.2f} s, "
                        f"recall@10 {pt.recall:.3f}, index {pt.index_bytes / 1e6:6.1f} MB, peak rss {pt.maxrss_bytes / 1e6:6.0f} MB, fetch {pt.fetch_bytes / 1e6 / max(pt.fetch_s, 1e-9):5.0f} MB/s"
                        + (f", save {pt.save_s:5.2f} s, upload {pt.upload_bytes / 1e6 / max(pt.upload_s, 1e-9):4.0f} MB/s" if pt.upload_bytes else f", save {pt.save_s:5.2f} s"))
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


# ---- the v1 fit (Phase 5 step 1, decision 6) --------------------------------
#
# cost model v1: build = A · n^P · ln(n) · (dim + D0). For each thread count
# the three constants are fitted in log space: for a candidate D0, ln(t) −
# ln(ln n) − ln(dim + D0) is linear in ln(n) with intercept ln A and slope P
# (an ordinary least squares); D0 is the candidate on a geometric grid from 0
# to 5000 whose residuals are smallest. The gate constant is the median over
# the points of gate_s / (queries · ln n · (dim + D0)). The bandwidths are
# medians of bytes / seconds over the points that have them (fetch; and, for
# v1 CSVs, save and upload). The speedup between the shard's and the worker's
# thread counts is the geometric mean, over the worker's points, of the
# shard model's build time over the worker's measured one — not the ratio of
# the two A's, which would ignore that P and D0 differ between the fits.


@dataclass(frozen=True)
class FitV1:
    threads: int
    n_points: int
    distinct_sizes: int   # fewer than 2: P and D0 are not identifiable and are reported as v0's (1, 0)
    a: float
    p: float
    d0: float
    sigma: float            # std (= RMS) of ln(measured / fitted), the simulator's log-normal factor
    min_ratio: float
    max_ratio: float
    gate_per_query: float   # s per (query · ln n · (dim + D0)); 0 when no point ran a gate
    gate_queries: int       # the gate's query count on the points (0 for v0 CSVs, which used 500)
    bandwidth_in: float     # median fetch bytes/s
    bandwidth_out: float | None   # median upload bytes/s; None when no point uploaded
    save_bandwidth: float | None  # median index bytes / save seconds; None for v0 CSVs
    recall_min: float
    fixed_s: float = 0.0  # v2 (Phase 5 decision 14): a per-build fixed cost fitted with the shape (0 unless asked)

    def build_s(self, n: float, dim: float) -> float:
        return self.fixed_s + self.a * n ** self.p * math.log(n) * (dim + self.d0)


def fit_points_v1(points: list[dict], fixed: bool = False, drop_warmup: bool = False) -> list[FitV1]:
    """fixed=True (v2, the GPU side): the shape is fitted with a per-build
    fixed term g0 — build = g0 + A·n^P·ln(n)·(dim + D0) — searched on a grid
    like D0 (the residuals are of the remaining time). drop_warmup=True leaves
    out each thread count's first point in CSV order: a process's first CUDA
    build pays a one-off warm-up (5–6 s on the L4) that a long-running worker
    pays once, not per build."""
    fits = []
    for th in sorted({int(p["threads"]) for p in points}):
        ps = [p for p in points if int(p["threads"]) == th]
        if drop_warmup and len(ps) > 1:
            # the process's first build is the earliest MEASURED point, not the
            # first row read: several CSVs arrive in argument (glob) order
            # (the v2 cross-check's A). `at` is the point's UTC timestamp.
            first = min(range(len(ps)), key=lambda i: (str(ps[i].get("at") or ""), i))
            ps = ps[:first] + ps[first + 1:]
        n = np.array([float(p["size"]) for p in ps])
        d = np.array([float(p["dim"]) for p in ps])
        t = np.array([float(p["build_s"]) for p in ps])
        if (n < 2).any() or (t <= 0).any():
            raise ValueError(f"threads {th}: every point needs size >= 2 and build_s > 0")
        distinct = len(set(n.tolist()))
        best = None
        g0_grid = np.linspace(0.0, float(t.min()) * 0.98, 120) if fixed and distinct >= 2 else [0.0]   # the floor's resolution: <1 % of the smallest time
        for g0 in g0_grid:
            tt = t - g0
            for d0 in np.concatenate([[0.0], np.geomspace(1.0, 5000.0, 400)]):
                y = np.log(tt) - np.log(np.log(n)) - np.log(d + d0)
                if distinct >= 2:
                    X = np.vstack([np.ones_like(n), np.log(n)]).T
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                else:
                    coef = np.array([float(np.mean(y - 1.0 * np.log(n))), 1.0])   # one size: P is 1 by convention
                pred = g0 + math.exp(coef[0]) * n ** coef[1] * np.log(n) * (d + d0)
                s = float(np.std(np.log(t / pred)))   # the residual of the WHOLE time, fixed term included
                if best is None or s < best[0] - 1e-12:
                    best = (s, float(d0), float(math.exp(coef[0])), float(coef[1]), float(g0))
        sigma, d0, a, pexp, g0 = best
        if distinct < 2:
            # one size cannot tell P from A, nor one dim D0 from A: the fit
            # reports v0's shape (P 1, D0 0) and the caller labels it (G6)
            d0, pexp = 0.0, 1.0
            a = float(math.exp(np.mean(np.log(t) - np.log(np.log(n)) - np.log(d))))
            sigma = float(np.std(np.log(t) - np.log(a * n * np.log(n) * d)))
        fitted = g0 + a * n ** pexp * np.log(n) * (d + d0)
        ratio = t / fitted
        gq = [int(float(p.get("gate_queries") or 0)) for p in ps]
        gqs = sorted(set(gq))
        q = gqs[-1] if gqs and gqs[-1] > 0 else 500   # v0 CSVs ran 500 queries (run_point's default)
        gate = [float(p.get("gate_s") or 0) / (q * math.log(float(p["size"])) * (float(p["dim"]) + d0)) for p in ps if float(p.get("gate_s") or 0) > 0]
        bw_in = float(np.median([float(p.get("fetch_bytes", 0)) / max(float(p.get("fetch_s", 0)), 1e-9) for p in ps]))
        ups = [float(p["upload_bytes"]) / float(p["upload_s"]) for p in ps if float(p.get("upload_bytes") or 0) > 0 and float(p.get("upload_s") or 0) > 0]
        saves = [float(p["index_bytes"]) / float(p["save_s"]) for p in ps if float(p.get("save_s") or 0) > 0]
        fits.append(FitV1(th, len(ps), distinct, a, pexp, d0, sigma, float(ratio.min()), float(ratio.max()),
                          float(np.median(gate)) if gate else 0.0, q if gqs and gqs[-1] > 0 else 0,
                          bw_in, float(np.median(ups)) if ups else None, float(np.median(saves)) if saves else None,
                          min(float(p["recall"]) for p in ps), g0))
    return fits


def speedup_v1(points: list[dict], shard: FitV1, worker_threads: int) -> tuple[float, float]:
    """The geometric mean over the worker's points of shard-model build time /
    worker measured build time, and the spread (std of the logs)."""
    ps = [p for p in points if int(p["threads"]) == worker_threads]
    if not ps:
        raise ValueError(f"no points at {worker_threads} threads")
    logs = [math.log(shard.build_s(float(p["size"]), float(p["dim"])) / float(p["build_s"])) for p in ps]
    return float(math.exp(np.mean(logs))), float(np.std(logs))


def env_overrides_v1(points: list[dict], fits: list[FitV1], shard_threads: int, worker_threads: int,
                     overhead_s: float | None = None, worker_points: list[dict] | None = None,
                     worker_fits: list[FitV1] | None = None, gate_queries: int = 10000) -> dict[str, str]:
    """The COST_* block for cost model v1 (COST_MODEL=calibrated plus these
    overrides). The shard side's A, P, D0 and gate; the speedup as the
    geometric mean above; bandwidths from the worker's points; the overhead
    is the given constant or 1.5 s (process start and the manifest: what the
    terms do not cover; measured by the live rounds, decision 37)."""
    by = {f.threads: f for f in fits}
    wby = {f.threads: f for f in (worker_fits if worker_fits is not None else fits)}
    if shard_threads not in by or worker_threads not in wby:
        raise ValueError(f"need a shard fit at {shard_threads} threads (have {sorted(by)}) and a worker fit at {worker_threads} (have {sorted(wby)})")
    sh, wk = by[shard_threads], wby[worker_threads]
    sp, spread = speedup_v1(worker_points if worker_points is not None else points, sh, worker_threads)
    out = {
        "COST_MODEL": "calibrated",
        "COST_A": f"{sh.a:.4g}",
        "COST_P": f"{sh.p:.3f}",
        "COST_D0": f"{sh.d0:.0f}",
        "COST_SPEEDUP": f"{sp:.3f}",
        "COST_GPU_OVERHEAD": f"{1.5 if overhead_s is None else overhead_s:.2f}",
        "COST_GATE": f"{sh.gate_per_query:.4g}",
        "COST_GATE_QUERIES": f"{gate_queries}",
        "COST_BANDWIDTH": f"{wk.bandwidth_in:.4g}",
    }
    if wk.bandwidth_out is not None:
        out["COST_BANDWIDTH_OUT"] = f"{wk.bandwidth_out:.4g}"
    if wk.save_bandwidth is not None:
        out["COST_SAVE_BANDWIDTH"] = f"{wk.save_bandwidth:.4g}"
    out["KGPU_ESTIMATE_SIGMA"] = f"{max(sh.sigma, wk.sigma, spread):.3f}"
    return out


def env_overrides_v2(points: list[dict], fits: list[FitV1], shard_threads: int, worker_threads: int,
                     worker_points: list[dict], worker_fits: list[FitV1], gate_queries: int = 10000,
                     overhead_s: float | None = None) -> dict[str, str]:
    """The COST_* block of cost model v2 (Phase 5 decision 14): the shard's
    shape as in v1, and the worker's OWN shape — fixed term, A, P, D0 and gate
    — instead of a speedup. Bandwidths from the worker's points. The sigma is
    the larger of the two sides' fit errors: each side's estimate now has its
    own, and no ratio spread is involved."""
    by = {f.threads: f for f in fits}
    wby = {f.threads: f for f in worker_fits}
    if shard_threads not in by or worker_threads not in wby:
        raise ValueError(f"need a shard fit at {shard_threads} threads (have {sorted(by)}) and a worker fit at {worker_threads} (have {sorted(wby)})")
    sh, wk = by[shard_threads], wby[worker_threads]
    out = {
        "COST_MODEL": "calibrated",
        "COST_A": f"{sh.a:.4g}",
        "COST_P": f"{sh.p:.3f}",
        "COST_D0": f"{sh.d0:.0f}",
        "COST_GATE": f"{sh.gate_per_query:.4g}",
        "COST_GATE_QUERIES": f"{gate_queries}",
        "COST_WORKER_FIXED": f"{wk.fixed_s:.3f}",
        "COST_WORKER_A": f"{wk.a:.4g}",
        "COST_WORKER_P": f"{wk.p:.3f}",
        "COST_WORKER_D0": f"{wk.d0:.0f}",
        "COST_WORKER_GATE": f"{wk.gate_per_query:.4g}",
        "COST_GPU_OVERHEAD": f"{1.5 if overhead_s is None else overhead_s:.2f}",
        "COST_BANDWIDTH": f"{wk.bandwidth_in:.4g}",
    }
    if wk.bandwidth_out is not None:
        out["COST_BANDWIDTH_OUT"] = f"{wk.bandwidth_out:.4g}"
    if wk.save_bandwidth is not None:
        out["COST_SAVE_BANDWIDTH"] = f"{wk.save_bandwidth:.4g}"
    out["KGPU_ESTIMATE_SIGMA"] = f"{max(sh.sigma, wk.sigma):.3f}"
    return out


def residual_table_v1(points: list[dict], fits: list[FitV1]) -> list[str]:
    by = {f.threads: f for f in fits}
    lines = [f"{'threads':>7} {'size':>8} {'dim':>5} {'measured':>9} {'v1 fit':>9} {'ratio':>6} {'recall':>7}"]
    for p in sorted(points, key=lambda p: (int(p["threads"]), int(p["size"]), int(p["dim"]))):
        f = by[int(p["threads"])]
        n, d, t = float(p["size"]), float(p["dim"]), float(p["build_s"])
        fitted = f.build_s(n, d)
        lines.append(f"{int(p['threads']):>7} {int(p['size']):>8} {int(d):>5} {t:>9.2f} {fitted:>9.2f} {t / fitted:>6.2f} {float(p['recall']):>7.3f}")
    return lines
