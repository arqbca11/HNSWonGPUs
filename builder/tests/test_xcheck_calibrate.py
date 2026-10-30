"""Cross-check (black box) of Phase 4 step 3: the calibration method.

Spec only: roadmap Phase 4 "Calibration method (laptop, a rehearsal)";
docs/design/phase4-hnswlib.md decisions 6, 23, 24 and "Implementation notes ->
Step 3"; docs/speedup.md evidence 6; the `kgpu_builder.calibrate` module
docstring and the CLI's --help; the cost model's docstrings (Go
`costmodel.Model`, Python `kgpu_worker.costmodel`). The implementation
(calibrate.py, cli.py, test_calibrate.py, plot_calibration.py) was NOT read.

Everything is driven as a process: `python -m kgpu_builder fit|calibrate|
calibrate-point` and `uv run python plot_calibration.py`. The fit tests are
pure (CSV files written here); the grid tests need a throwaway RustFS
(TEST_S3_ENDPOINT) and write toy splits with the public Store.

Reference arithmetic (decision 24, the module docstring), computed here by hand:
  x = n * ln(n) * dim                      per point
  A = sum(x*t) / sum(x^2)                  per thread count, through the origin
  ratio = t / (A*x);  sigma = std(ln ratio)
  power law: least squares of ln(t/dim) = ln c + p * ln n
  COST_A = A_shard;  COST_SPEEDUP = A_shard / A_worker
  COST_BANDWIDTH = median(fetch_bytes / fetch_s)  (which points: see the tests)
  KGPU_ESTIMATE_SIGMA = max(sigma_shard, sigma_worker)   (evidence 6: 0.245)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "builder"
ANALYSIS = REPO / "experiments" / "analysis"
CAL = REPO / "experiments" / "results" / "calibration"
SIFT_CSV = CAL / "laptop-sift-128-6.csv"
GIST_CSV = CAL / "laptop-gist-960-300k-6.csv"

S3 = os.environ.get("TEST_S3_ENDPOINT")
needs_s3 = pytest.mark.skipif(not S3, reason="TEST_S3_ENDPOINT unset")

with open(SIFT_CSV) as _f:
    HEADER = _f.readline().strip().split(",")  # the committed point schema


# ------------------------------------------------------------------ helpers


def point(size, threads, build_s, dim=128, fetch_s=1.0, fetch_bytes=200_000_000, repeat=0,
          recall=0.99, dataset="xfit"):
    return {"dataset": dataset, "n_shards": 6, "shard": 1, "size": size, "dim": dim, "threads": threads,
            "m": 16, "ef_construction": 200, "repeat": repeat, "fetch_s": fetch_s, "fetch_bytes": fetch_bytes,
            "build_s": build_s, "gate_s": 0.1, "recall": recall, "index_bytes": 1, "maxrss_bytes": 1,
            "host": "xhost", "at": "2026-10-30T00:00:00Z"}


def write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def x_of(n, dim):
    return n * math.log(n) * dim


def ref_fit(points: list[dict]) -> dict[int, dict]:
    """Decision 24 by hand, per thread count."""
    out = {}
    for th in sorted({int(p["threads"]) for p in points}):
        ps = [p for p in points if int(p["threads"]) == th]
        n = np.array([float(p["size"]) for p in ps])
        d = np.array([float(p["dim"]) for p in ps])
        t = np.array([float(p["build_s"]) for p in ps])
        x = n * np.log(n) * d
        a = float(np.exp(np.log(t / x).mean()))  # amended decision 24: the log-space fit (geometric mean of t/x)
        lr = np.log(t / (a * x))
        X = np.column_stack([np.ones_like(n), np.log(n)])
        coef, *_ = np.linalg.lstsq(X, np.log(t / d), rcond=None)
        pres = np.log(t / d) - X @ coef
        bw = np.array([float(p["fetch_bytes"]) / float(p["fetch_s"]) for p in ps])
        rec = np.array([float(p["recall"]) for p in ps])
        out[th] = dict(n=len(ps), a=a, ratio_min=float(np.exp(lr).min()), ratio_max=float(np.exp(lr).max()),
                       sigma_pop=float(np.std(lr)), sigma_samp=float(np.std(lr, ddof=1)) if len(ps) > 1 else float("nan"),
                       rms=float(np.sqrt((lr ** 2).mean())), mean_log=float(lr.mean()),
                       c=float(math.exp(coef[0])), p=float(coef[1]), psigma=float(np.std(pres)),
                       bw_median=float(np.median(bw)), bw_min=float(bw.min()), bw_max=float(bw.max()),
                       recall_min=float(rec.min()))
    return out


LINE = re.compile(
    r"threads (\d+): (\d+) points; A = ([^\s)]+) s per n·ln\(n\)·dim \(ratio ([^\s)]+)\.\.([^\s)]+), "
    r"sigma ([^\s)]+)\); power law t = ([^\s)]+) · n\^([^\s)]+) · dim \(sigma ([^\s)]+)\); "
    r"min recall ([-\d.]+); fetch (\d+) MB/s")


def run_fit(csv_path: Path, *extra: str) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-m", "kgpu_builder", "fit", str(csv_path), *extra]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=BUILDER, timeout=120)
    print(f"ran: kgpu_builder fit {csv_path.name} {' '.join(extra)} -> exit {p.returncode}")
    return p


def parse_fit(stdout: str) -> tuple[dict[int, dict], dict[str, str], list[str]]:
    fits, env, table = {}, {}, []
    for line in stdout.splitlines():
        m = LINE.search(line)
        if m:
            g = m.groups()
            fits[int(g[0])] = dict(n=int(g[1]), a=float(g[2]), ratio_min=float(g[3]), ratio_max=float(g[4]),
                                   sigma=float(g[5]), c=float(g[6]), p=float(g[7]), psigma=float(g[8]),
                                   recall_min=float(g[9]), fetch_mbs=int(g[10]))
            continue
        m = re.match(r"\s+((?:COST|KGPU)_[A-Z_]+)=(\S+)$", line)
        if m:
            env[m.group(1)] = m.group(2)
            continue
        if re.match(r"\s+\d+\s+\d+\s+\d+\s", line):
            table.append(line)
    return fits, env, table


def close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b))


def one_line_error(p: subprocess.CompletedProcess) -> bool:
    err = (p.stderr + p.stdout).strip()
    return p.returncode != 0 and "Traceback" not in err and len(err.splitlines()) <= 2


# ------------------------------------------------------------------ the fit, by hand


def test_xc_fit_constant_power_law_and_ratios_by_hand(tmp_path):
    """Decision 24: A through the origin, ratios measured/fitted, the power law beside it."""
    rng = np.random.default_rng(7)
    rows = []
    for th, a_true in ((2, 9e-8), (8, 4e-8)):
        for n in (10_000, 30_000, 100_000, 200_000, 400_000):
            for rep in range(2):
                t = a_true * x_of(n, 128) * math.exp(rng.normal(0, 0.2))
                rows.append(point(n, th, t, repeat=rep, fetch_s=1.0 + rng.random()))
    f = write_csv(tmp_path / "pts.csv", rows)
    p = run_fit(f)
    print(p.stdout)
    assert p.returncode == 0
    got, env, table = parse_fit(p.stdout)
    ref = ref_fit(rows)
    for th in (2, 8):
        g, r = got[th], ref[th]
        print(f"threads {th}: A printed {g['a']:.4e} vs hand {r['a']:.4e}; ratio {g['ratio_min']}..{g['ratio_max']} "
              f"vs {r['ratio_min']:.3f}..{r['ratio_max']:.3f}; p {g['p']} vs {r['p']:.4f}; c {g['c']} vs {r['c']:.3e}; "
              f"power sigma {g['psigma']} vs {r['psigma']:.4f}; n {g['n']} vs {r['n']}")
        assert g["n"] == r["n"] == 10
        assert close(g["a"], r["a"], 6e-4), "A = sum(x t)/sum(x^2), x = n ln n dim (decision 24)"
        assert abs(g["ratio_min"] - r["ratio_min"]) <= 0.006 and abs(g["ratio_max"] - r["ratio_max"]) <= 0.006
        assert abs(g["p"] - r["p"]) <= 6e-4, "power exponent: least squares of ln(t/dim) on ln n"
        assert close(g["c"], r["c"], 0.02)
        assert abs(g["psigma"] - r["psigma"]) <= 6e-4
    print(f"residual table: {len(table)} rows for {len(rows)} points")
    assert len(table) == len(rows), "one residual row per point"


def test_xc_fit_sigma_is_population_or_sample(tmp_path):
    """Decision 24 says 'the standard deviation of ln(measured/fitted)'; which estimator? Three points
    make population (ddof 0) and sample (ddof 1) differ by sqrt(3/2)."""
    rows = [point(10_000, 2, 1e-7 * x_of(10_000, 128) * math.exp(0.4)),
            point(50_000, 2, 1e-7 * x_of(50_000, 128) * math.exp(-0.3)),
            point(200_000, 2, 1e-7 * x_of(200_000, 128) * math.exp(0.05))]
    p = run_fit(write_csv(tmp_path / "s.csv", rows))
    got, _, _ = parse_fit(p.stdout)
    r = ref_fit(rows)[2]
    print(f"printed sigma {got[2]['sigma']}; population std {r['sigma_pop']:.4f}; sample std {r['sigma_samp']:.4f}; "
          f"RMS of ln ratio (no mean removed) {r['rms']:.4f}; mean ln ratio {r['mean_log']:+.4f}")
    which = "population" if abs(got[2]["sigma"] - r["sigma_pop"]) <= 6e-4 else (
        "sample" if abs(got[2]["sigma"] - r["sigma_samp"]) <= 6e-4 else "neither")
    print(f"=> fit's sigma is the {which} standard deviation")
    assert which != "neither", "sigma must be a standard deviation of ln(measured/fitted)"


def test_xc_fit_env_block_by_hand(tmp_path):
    """Decision 24: COST_A = shard A, COST_SPEEDUP = A ratio, COST_BANDWIDTH = median fetch bytes/s,
    COST_GPU_OVERHEAD = --overhead; KGPU_ESTIMATE_SIGMA = max of the pair's sigmas (evidence 6).
    The shard's points get a wildly different bandwidth and a LARGER sigma than the worker's; a third
    thread count (4) with an even larger sigma must not leak into the block."""
    rng = np.random.default_rng(3)
    rows = []
    sizes = (10_000, 40_000, 160_000, 320_000, 640_000)
    for n in sizes:  # shard, 2 threads, noisy, slow fetch (10 MB/s)
        rows.append(point(n, 2, 1.2e-7 * x_of(n, 128) * math.exp(rng.normal(0, 0.35)), fetch_s=20.0))
    worker_fetch = [1.0, 1.25, 2.0, 4.0, 5.0]  # 200, 160, 100, 50, 40 MB/s: median 100 MB/s, mean 110
    for n, fs in zip(sizes, worker_fetch):  # worker, 8 threads, quiet
        rows.append(point(n, 8, 4e-8 * x_of(n, 128) * math.exp(rng.normal(0, 0.05)), fetch_s=fs))
    for n in sizes:  # a third configuration, very noisy
        rows.append(point(n, 4, 6e-8 * x_of(n, 128) * math.exp(rng.normal(0, 0.9)), fetch_s=20.0))
    p = run_fit(write_csv(tmp_path / "env.csv", rows), "--shard-threads", "2", "--worker-threads", "8",
                "--overhead", "3.5")
    print(p.stdout)
    _, env, _ = parse_fit(p.stdout)
    ref = ref_fit(rows)
    bw_all = float(np.median([200e6 / (r["fetch_s"]) for r in rows]))
    bw_pair = float(np.median([200e6 / r["fetch_s"] for r in rows if r["threads"] in (2, 8)]))
    print(f"hand: A2 {ref[2]['a']:.4e} A8 {ref[8]['a']:.4e} speedup {ref[2]['a'] / ref[8]['a']:.4f}; "
          f"sigma2 {ref[2]['sigma_pop']:.4f} sigma8 {ref[8]['sigma_pop']:.4f} sigma4 {ref[4]['sigma_pop']:.4f}; "
          f"median bandwidth: worker {ref[8]['bw_median']:.4e}, shard {ref[2]['bw_median']:.4e}, "
          f"shard+worker {bw_pair:.4e}, all points {bw_all:.4e}")
    print(f"printed block: {env}")
    assert close(float(env["COST_A"]), ref[2]["a"], 6e-4)
    assert close(float(env["COST_SPEEDUP"]), ref[2]["a"] / ref[8]["a"], 6e-4)
    assert float(env["COST_GPU_OVERHEAD"]) == pytest.approx(3.5)
    assert float(env["KGPU_ESTIMATE_SIGMA"]) == pytest.approx(max(ref[2]["sigma_pop"], ref[8]["sigma_pop"]), abs=6e-4), \
        "the sigma of the shard/worker pair, not the third thread count's"
    bw = float(env["COST_BANDWIDTH"])
    src = {"worker points": ref[8]["bw_median"], "shard points": ref[2]["bw_median"],
           "shard+worker points": bw_pair, "all points": bw_all}
    which = [k for k, v in src.items() if close(bw, v, 6e-4)]
    print(f"=> COST_BANDWIDTH {bw:.4e} is the median over: {which or 'none of the candidates'}")
    assert which == ["worker points"], "COST_BANDWIDTH from the worker's points (evidence 6's 1.84e8)"


def test_xc_fit_bandwidth_median_of_even_count(tmp_path):
    """'the median fetch bytes/s' with an even number of worker points: the middle two's mean?"""
    rows = [point(n, 2, 1e-7 * x_of(n, 128)) for n in (10_000, 20_000)]
    rows += [point(n, 8, 5e-8 * x_of(n, 128), fetch_s=fs)
             for n, fs in zip((10_000, 20_000, 40_000, 80_000), (1.0, 2.0, 4.0, 8.0))]  # 200, 100, 50, 25 MB/s
    p = run_fit(write_csv(tmp_path / "even.csv", rows))
    _, env, _ = parse_fit(p.stdout)
    bw = float(env["COST_BANDWIDTH"])
    print(f"worker bandwidths 200/100/50/25 MB/s -> COST_BANDWIDTH {bw:.4e} "
          f"(numpy median 7.5e7; lower middle 5e7; upper middle 1e8)")
    assert bw == pytest.approx(7.5e7, rel=1e-3)


def test_xc_fit_repeats_are_separate_points(tmp_path):
    """Repeats at one size are separate points (decision 23's CSV row per point; decision 24's least
    squares over points). Unequal repeat counts make 'separate points' and 'mean per size' differ."""
    rows = [point(10_000, 2, 1e-7 * x_of(10_000, 128) * f, repeat=i) for i, f in enumerate((0.5, 0.6, 0.7))]
    rows += [point(400_000, 2, 1e-7 * x_of(400_000, 128) * 1.3)]
    p = run_fit(write_csv(tmp_path / "rep.csv", rows))
    got, _, table = parse_fit(p.stdout)
    r = ref_fit(rows)[2]
    xs = [x_of(10_000, 128), x_of(400_000, 128)]
    ts = [np.mean([row["build_s"] for row in rows[:3]]), rows[3]["build_s"]]
    a_mean = sum(x * t for x, t in zip(xs, ts)) / sum(x * x for x in xs)
    print(f"printed: {got[2]['n']} points, A {got[2]['a']:.4e}; hand, every repeat a point: {r['a']:.4e}; "
          f"hand, repeats averaged per size: {a_mean:.4e}; residual rows {len(table)}")
    assert got[2]["n"] == 4 and len(table) == 4
    assert close(got[2]["a"], r["a"], 6e-4)
    assert abs(got[2]["sigma"] - r["sigma_pop"]) <= 6e-4 or abs(got[2]["sigma"] - r["sigma_samp"]) <= 6e-4


def test_xc_fit_mixed_dims_in_one_file(tmp_path):
    """Evidence 6 fits SIFT (dim 128) and GIST (dim 960) together; with exact t = A n ln n dim the fit must
    recover A exactly whatever the mix of dims (x carries dim per point)."""
    rows = [point(n, th, (1e-7 if th == 2 else 5e-8) * x_of(n, d), dim=d, dataset=f"d{d}")
            for th in (2, 8) for d in (128, 960) for n in (10_000, 50_000, 120_000)]
    p = run_fit(write_csv(tmp_path / "mixed.csv", rows))
    print(p.stdout)
    got, env, _ = parse_fit(p.stdout)
    for th, a in ((2, 1e-7), (8, 5e-8)):
        print(f"threads {th}: A {got[th]['a']} (true {a}), ratio {got[th]['ratio_min']}..{got[th]['ratio_max']}, "
              f"sigma {got[th]['sigma']}")
        assert close(got[th]["a"], a, 6e-4) and got[th]["sigma"] == 0.0
        assert got[th]["ratio_min"] == got[th]["ratio_max"] == 1.0
    assert float(env["COST_SPEEDUP"]) == pytest.approx(2.0, rel=1e-3)


def test_xc_fit_fitted_column_is_the_worker_cost_model(tmp_path):
    """The block is meant to be pasted as COST_*: with COST_A the cost model's cpu_build_s(n, dim) (the
    worker's mirror of the Go Model: A * n * ln(n) * dim) must equal the residual table's 'fitted'."""
    rows = [point(n, th, (1.1e-7 if th == 2 else 5e-8) * x_of(n, 128) * f)
            for th in (2, 8) for n, f in ((50_000, 0.8), (200_000, 1.1), (400_000, 1.05))]
    p = run_fit(write_csv(tmp_path / "cm.csv", rows))
    got, env, table = parse_fit(p.stdout)
    a = float(env["COST_A"])
    code = ("import json,sys; from kgpu_worker.costmodel import Model; m=Model(a=float(sys.argv[1])); "
            "print(json.dumps([m.cpu_build_s(n,128) for n in (50000,200000,400000)]))")
    q = subprocess.run(["uv", "run", "python", "-c", code, str(a)], capture_output=True, text=True,
                       cwd=REPO / "worker", timeout=120)
    model = json.loads(q.stdout)
    fitted = [float(l.split()[4]) for l in table if l.split()[0] == "2"]
    print(f"COST_A {a}: worker cost model cpu_build_s {['%.2f' % v for v in model]}; fit's fitted column {fitted}")
    assert fitted == pytest.approx(model, abs=0.006 + 6e-4 * max(model))


# ------------------------------------------------------------------ the fit, degenerate inputs


def test_xc_fit_single_point(tmp_path):
    """One point: A is defined (t/x), sigma of one residual is 0 (population), and a power law through one
    size is undetermined: the fit must not print a made-up exponent."""
    p = run_fit(write_csv(tmp_path / "one.csv", [point(1000, 2, 0.5, dim=16)]))
    print(p.stdout + p.stderr)
    got, env, _ = parse_fit(p.stdout)
    assert p.returncode == 0 and 2 in got
    a = 0.5 / x_of(1000, 16)
    print(f"A printed {got[2]['a']} vs t/x {a:.4e}; sigma {got[2]['sigma']}; power exponent printed {got[2]['p']}")
    assert close(got[2]["a"], a, 6e-4)
    assert got[2]["sigma"] == 0.0
    assert "COST_A" not in env, "no worker fit, no environment block"
    assert not math.isfinite(got[2]["p"]), \
        f"a power law from ONE size is undetermined, yet fit prints p = {got[2]['p']} (decision 24 reads p > 1 as a finding)"


def test_xc_fit_repeats_at_one_size_only(tmp_path):
    """Several points but one distinct size: A and sigma are fine, the exponent is still undetermined."""
    rows = [point(20_000, th, (1e-7 if th == 2 else 5e-8) * x_of(20_000, 128) * f, repeat=i)
            for th in (2, 8) for i, f in enumerate((0.9, 1.0, 1.1))]
    p = run_fit(write_csv(tmp_path / "onesize.csv", rows))
    print(p.stdout + p.stderr)
    got, env, _ = parse_fit(p.stdout)
    print(f"exponents printed: 2 threads p={got[2]['p']}, 8 threads p={got[8]['p']}")
    assert p.returncode == 0 and "COST_A" in env
    assert not math.isfinite(got[2]["p"]) and not math.isfinite(got[8]["p"]), \
        "one distinct size cannot determine an exponent; a finite p here is a fabricated number"


def test_xc_fit_two_thread_counts_one_point_each(tmp_path):
    rows = [point(1000, 2, 0.5, dim=16, fetch_s=0.01, fetch_bytes=192000),
            point(1000, 8, 0.25, dim=16, fetch_s=0.02, fetch_bytes=192000)]
    p = run_fit(write_csv(tmp_path / "two.csv", rows))
    print(p.stdout)
    _, env, _ = parse_fit(p.stdout)
    print(f"block {env}: speedup 0.5/0.25 = 2; sigma of single points 0; bandwidth of the worker point 9.6e6, "
          f"of the shard point 1.92e7")
    assert float(env["COST_SPEEDUP"]) == pytest.approx(2.0, rel=1e-3)
    assert float(env["KGPU_ESTIMATE_SIGMA"]) == 0.0
    assert float(env["COST_BANDWIDTH"]) == pytest.approx(9.6e6, rel=1e-3), "COST_BANDWIDTH from the worker's points"


@pytest.mark.parametrize("content", ["", "HEADER"], ids=["empty-file", "header-only"])
def test_xc_fit_no_points_refused_in_one_line(tmp_path, content):
    f = tmp_path / "e.csv"
    f.write_text(",".join(HEADER) + "\n" if content == "HEADER" else "")
    p = run_fit(f)
    print(f"stdout {p.stdout!r} stderr {p.stderr!r}")
    assert one_line_error(p)


def test_xc_fit_non_numeric_build_s_refused_in_one_line(tmp_path):
    rows = [point(1000, 2, 0.5, dim=16), point(2000, 2, 1.1, dim=16)]
    rows[1]["build_s"] = "abc"
    p = run_fit(write_csv(tmp_path / "bad.csv", rows))
    err = p.stderr.strip().splitlines()
    print(f"exit {p.returncode}; stderr {len(err)} lines; last: {err[-1] if err else ''!r}")
    assert p.returncode != 0
    assert one_line_error(p), "a malformed row should be named in one line, not a Python traceback"


def test_xc_fit_size_one_point(tmp_path):
    """A point of size 1 has n ln n = 0: its fitted time is 0 and its ratio undefined. The fit should
    refuse or skip it, not crash."""
    rows = [point(1, 2, 0.001, dim=16), point(1000, 2, 0.5, dim=16), point(2000, 2, 1.2, dim=16)]
    p = run_fit(write_csv(tmp_path / "n1.csv", rows))
    err = p.stderr.strip().splitlines()
    print(f"exit {p.returncode}; stderr tail {err[-1:] }")
    assert "Traceback" not in p.stderr, "size 1 (x = 0) crashes the residual table"


def test_xc_fit_requested_pair_missing(tmp_path):
    """Fits exist for 2 and 8 threads, but the block is asked for 2 and 4: no block, a message naming why."""
    rows = [point(n, th, (1e-7 if th == 2 else 5e-8) * x_of(n, 128)) for th in (2, 8) for n in (10_000, 50_000)]
    p = run_fit(write_csv(tmp_path / "pair.csv", rows), "--worker-threads", "4")
    print(p.stdout + p.stderr)
    got, env, _ = parse_fit(p.stdout)
    print(f"exit {p.returncode}; per-thread fits printed for {sorted(got)}; block keys {sorted(env)}")
    assert set(got) == {2, 8}
    assert not env, "no COST_* block for a pair the file does not have"
    assert "4" in (p.stdout + p.stderr)


# ------------------------------------------------------------------ the committed rehearsal (evidence 6)


def test_xc_rehearsal_csvs_reproduce_evidence6(tmp_path):
    sift, gist = read_rows(SIFT_CSV), read_rows(GIST_CSV)
    print(f"committed: SIFT {len(sift)} rows, GIST {len(gist)} rows (Step 3 notes: 40 rows); header == schema: "
          f"{list(sift[0]) == HEADER and list(gist[0]) == HEADER}")
    assert len(sift) + len(gist) == 40
    both = sift + gist
    for name, rows, lo, hi in (("SIFT", sift, 10_000, 400_000), ("GIST", gist, 10_000, 120_000)):
        sizes = sorted({int(r["size"]) for r in rows})
        reps = {(r["size"], r["threads"]): 0 for r in rows}
        for r in rows:
            reps[(r["size"], r["threads"])] += 1
        print(f"{name}: sizes {sizes}, threads {sorted({int(r['threads']) for r in rows})}, repeats per point "
              f"{set(reps.values())}, shard {set(r['shard'] for r in rows)}")
        assert sizes[0] == lo and sizes[-1] == hi and set(reps.values()) == {2}
        for th in ("2", "8"):
            rec = [float(r["recall"]) for r in sorted(rows, key=lambda r: int(r["size"])) if r["threads"] == th]
            print(f"  {name} {th} threads: recall at smallest size {rec[:2]}, at largest {rec[-2:]}")
    ref = ref_fit(both)
    for th in (2, 8):
        r = ref[th]
        print(f"hand, threads {th}: {r['n']} points, A {r['a']:.4e}, ratio {r['ratio_min']:.2f}..{r['ratio_max']:.2f}, "
              f"sigma pop {r['sigma_pop']:.3f} (sample {r['sigma_samp']:.3f}; RMS incl. bias {r['rms']:.3f}, "
              f"mean ln ratio {r['mean_log']:+.3f}), p {r['p']:.3f}, power sigma {r['psigma']:.3f}, "
              f"fetch median {r['bw_median']:.4e} B/s (range {r['bw_min'] / 1e6:.0f}..{r['bw_max'] / 1e6:.0f} MB/s)")
    both_csv = write_csv(tmp_path / "both.csv", both)
    p = run_fit(both_csv)
    got, env, _ = parse_fit(p.stdout)
    print(f"fit printed: {got}\nblock: {env}")
    # evidence 6 / Step 3 table: every check reported, then one assert
    bad = []
    def check(ok: bool, what: str) -> None:
        print(f"  {'ok ' if ok else 'BAD'} {what}")
        if not ok:
            bad.append(what)
    check(f"{ref[2]['a']:.2e}" == "6.08e-08" and f"{ref[8]['a']:.2e}" == "2.97e-08",
          f"A 6.08e-8 / 2.97e-8 (hand {ref[2]['a']:.3e} / {ref[8]['a']:.3e})")
    check(round(ref[2]["a"] / ref[8]["a"], 2) == 2.05, f"speedup 2.05 (hand {ref[2]['a'] / ref[8]['a']:.3f})")
    check(round(ref[2]["p"], 2) == 1.27 and round(ref[8]["p"], 2) == 1.30, f"p 1.27 / 1.30 (hand {ref[2]['p']:.3f} / {ref[8]['p']:.3f})")
    check(round(ref[2]["sigma_pop"], 2) == 0.22 and round(ref[8]["sigma_pop"], 2) == 0.25,
          f"sigma 0.22 / 0.25 (hand {ref[2]['sigma_pop']:.3f} / {ref[8]['sigma_pop']:.3f})")
    check(round(ref[2]["psigma"], 2) == 0.06 and round(ref[8]["psigma"], 2) == 0.05,
          f"power sigma 0.06 / 0.05 (hand {ref[2]['psigma']:.3f} / {ref[8]['psigma']:.3f})")
    for th, lo, hi in ((2, "0.69", "1.49"), (8, "0.70", "1.52")):
        check(f"{ref[th]['ratio_min']:.2f}" == lo and f"{ref[th]['ratio_max']:.2f}" == hi,
              f"{th} threads ratio {lo}..{hi} (hand {ref[th]['ratio_min']:.3f}..{ref[th]['ratio_max']:.3f})")
    allbw = [float(r["fetch_bytes"]) / float(r["fetch_s"]) / 1e6 for r in both]
    check(150 <= min(allbw) and max(allbw) <= 215,
          f"'Fetch from the store ran at 150-215 MB/s' (every point: {min(allbw):.0f}..{max(allbw):.0f} MB/s)")
    for th in (2, 8):
        check(close(got[th]["a"], ref[th]["a"], 6e-4) and abs(got[th]["p"] - ref[th]["p"]) <= 6e-4
              and abs(got[th]["sigma"] - ref[th]["sigma_pop"]) <= 6e-4, f"fit CLI agrees with the hand numbers at {th} threads")
    check(env.get("COST_A") == "6.08e-08" and env.get("COST_SPEEDUP") == "2.050" and env.get("KGPU_ESTIMATE_SIGMA") == "0.245",
          f"the CLI block matches the Step 3 notes' COST_A 6.08e-08, SPEEDUP 2.050, SIGMA 0.245 ({env})")
    w = sorted(float(r["fetch_bytes"]) / float(r["fetch_s"]) for r in both if r["threads"] == "8")
    check(close(float(env["COST_BANDWIDTH"]), ref[8]["bw_median"], 6e-4),
          f"COST_BANDWIDTH {env['COST_BANDWIDTH']} is the median of the worker points' bytes/s "
          f"(median {ref[8]['bw_median']:.4e}; upper middle of the 20 {w[10]:.4e}; lower middle {w[9]:.4e}; "
          f"shard points' median {ref[2]['bw_median']:.4e})")
    assert not bad, f"{len(bad)} mismatch(es): {bad}"


# ------------------------------------------------------------------ the grid (decision 23), against a store


class Toy:
    """A fresh bucket; shards written by hand with a split manifest in the committed format."""

    def __init__(self, tmp: Path, dataset: str = "xcal", n_shards: int = 3, dim: int = 16, n_queries: int = 600):
        from kgpu_builder.config import Config
        from kgpu_builder.groundtruth import exact_topk
        from kgpu_builder.store import Store
        self.exact_topk = exact_topk
        self.bucket = "xcc-" + uuid.uuid4().hex[:10]
        self.cfg = Config(data_dir=tmp, s3_endpoint=S3, s3_access_key=os.environ.get("TEST_S3_ACCESS_KEY", ""),
                          s3_secret_key=os.environ.get("TEST_S3_SECRET_KEY", ""), s3_bucket=self.bucket,
                          s3_region="us-east-1")
        self.st = Store(self.cfg)
        self.st.ensure_bucket()
        self.dataset, self.n, self.dim = dataset, n_shards, dim
        self.rng = np.random.default_rng(11)
        self.q = self.rng.standard_normal((n_queries, dim)).astype(np.float32)
        self.st.put_npy(f"datasets/{dataset}/queries.npy", self.q)
        self.tmp = tmp

    def put_shards(self, arrays: list[np.ndarray], truths: list[np.ndarray] | None = None) -> None:
        shards, off = [], 0
        for k, v in enumerate(arrays):
            self.st.put_npy(f"splits/{self.dataset}/{self.n}/vectors/shard_{k}.npy", v)
            gt = truths[k] if truths else self.exact_topk(self.q, v, 100)
            self.st.put_npy(f"splits/{self.dataset}/{self.n}/ground_truth/shard_{k}.npy", gt)
            shards.append({"shard_id": k, "rank": 0, "offset": off, "size": int(v.shape[0])})
            off += int(v.shape[0])
        for rank, k in enumerate(sorted(range(len(arrays)), key=lambda k: -arrays[k].shape[0]), 1):
            shards[k]["rank"] = rank
        self.st.put_json(f"splits/{self.dataset}/{self.n}/manifest.json",
                         {"dataset": self.dataset, "dim": self.dim, "metric": "euclidean", "n_shards": self.n,
                          "n_vectors": off, "zipf_exponent": 1.0, "perm_seed": 1, "shards": shards,
                          "ground_truth_k": 100})
        print(f"(bucket {self.bucket}: {self.dataset} split {self.n}, shard sizes "
              f"{[a.shape[0] for a in arrays]} x {self.dim}, {len(self.q)} queries)")

    def env(self) -> dict[str, str]:
        e = {k: v for k, v in os.environ.items() if not k.startswith("S3_")}
        e.update(S3_ENDPOINT=S3, S3_ACCESS_KEY=self.cfg.s3_access_key, S3_SECRET_KEY=self.cfg.s3_secret_key,
                 S3_BUCKET=self.bucket, S3_REGION="us-east-1")
        return e

    def argv(self, cmd: str, *args: str) -> list[str]:
        return [sys.executable, "-m", "kgpu_builder", cmd, self.dataset, str(self.n), *args]

    def run(self, cmd: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
        argv = self.argv(cmd, *args)
        t0 = time.monotonic()
        p = subprocess.run(argv, capture_output=True, text=True, env=self.env(), cwd=BUILDER, timeout=timeout)
        print(f"ran: kgpu_builder {' '.join(argv[3:])} -> exit {p.returncode} in {time.monotonic() - t0:.1f}s")
        return p

    def close(self) -> None:
        self.st.delete_prefix("")


@pytest.fixture
def toy(tmp_path):
    t = Toy(tmp_path)
    yield t
    t.close()


def standard_shards(toy: Toy, nan_tail: bool = False, wrong_truth: bool = False) -> None:
    """Shard 1 is the largest (3000) and distinctive: rows 2000.. are NaN when nan_tail, so any point that
    is not a prefix of it (a sample, a suffix, another shard's rows) is visible."""
    r = toy.rng
    a0 = r.standard_normal((1500, toy.dim)).astype(np.float32)
    a1 = r.standard_normal((3000, toy.dim)).astype(np.float32)
    if nan_tail:
        a1[2000:] = np.nan
    a2 = r.standard_normal((2500, toy.dim)).astype(np.float32)
    truths = None
    if wrong_truth:
        truths = [toy.exact_topk(toy.q, a, 100) for a in (a0, a1, a2)]
        truths[1] = np.full_like(truths[1], 2999)  # every neighbour the same, NaN, row: deliberately wrong
    toy.put_shards([a0, a1, a2], truths)


@needs_s3
def test_xc_grid_schema_and_row_order(toy, tmp_path):
    standard_shards(toy)
    out = tmp_path / "grid.csv"
    p = toy.run("calibrate", "--sizes", "1000,500", "--threads", "2,1", "--repeats", "2", "--out", str(out))
    print(p.stdout[-1500:])
    assert p.returncode == 0
    lines = out.read_text().splitlines()
    print(f"header: {lines[0]}")
    assert lines[0].split(",") == HEADER, "same schema as the committed rehearsal CSVs"
    rows = read_rows(out)
    order = [(int(r["threads"]), int(r["size"]), int(r["repeat"])) for r in rows]
    print(f"row order (threads, size, repeat): {order}")
    assert len(rows) == 2 * 2 * 2
    assert len(set(order)) == 8, "every (size, threads, repeat) once"
    thr_seq = [o[0] for o in order]
    assert thr_seq == sorted(thr_seq, key=lambda t: [2, 1].index(t)) or thr_seq == sorted(thr_seq), \
        "threads outermost"
    for th in (1, 2):
        sub = [o for o in order if o[0] == th]
        for s in (500, 1000):
            reps = [o[2] for o in sub if o[1] == s]
            assert reps == [0, 1], "repeats innermost, increasing"
    sizes_seen = [o[1] for o in order[:4:2]]
    print(f"=> sizes within a thread count run in the order {sizes_seen} (asked 1000,500): "
          f"{'as given' if sizes_seen == [1000, 500] else 'sorted'}; threads asked 2,1, run {sorted(set(thr_seq), key=thr_seq.index)}")
    for r in rows:
        assert int(r["dim"]) == 16 and int(r["m"]) == 16 and int(r["ef_construction"]) == 200, \
            "the parameters a real build uses (decision 17: M 16, ef_construction 200)"
        assert float(r["build_s"]) > 0 and float(r["gate_s"]) > 0 and float(r["fetch_s"]) > 0
        assert int(r["index_bytes"]) > 0 and int(r["maxrss_bytes"]) > 0 and r["host"] and r["at"]


@needs_s3
def test_xc_grid_prefix_of_largest_shard_gated_on_its_own_truth(toy, tmp_path):
    """Decision 23: every point is the first `size` vectors of the split's LARGEST shard, gated by an exact
    top-10 computed for the prefix. Shard 1 (3000, largest) has NaN rows from 2000 on and a stored ground
    truth that is deliberately wrong (every neighbour = row 2999)."""
    standard_shards(toy, nan_tail=True, wrong_truth=True)
    out = tmp_path / "prefix.csv"
    p = toy.run("calibrate", "--sizes", "500,2000", "--threads", "2", "--out", str(out))
    print(p.stdout[-800:] + p.stderr[-800:])
    assert p.returncode == 0
    rows = read_rows(out)
    for r in rows:
        print(f"size {r['size']}: shard {r['shard']}, fetch_bytes {r['fetch_bytes']} (shard 1's vectors are "
              f"{3000 * 16 * 4} B raw; shard 2's {2500 * 16 * 4}), recall {r['recall']}")
        assert r["shard"] == "1", "the largest shard"
        assert int(r["fetch_bytes"]) in (3000 * 16 * 4, toy.st.size(f"splits/{toy.dataset}/3/vectors/shard_1.npy"))
        assert float(r["recall"]) >= 0.9, \
            "a clean prefix scored against its own exact top-10 (not the stored, wrong, truth; no NaN tail rows)"


@needs_s3
def test_xc_grid_shard_override(toy, tmp_path):
    standard_shards(toy)
    out = tmp_path / "ovr.csv"
    p = toy.run("calibrate", "--shard", "2", "--sizes", "800", "--threads", "1", "--out", str(out))
    rows = read_rows(out)
    print(f"--shard 2: rows {[(r['shard'], r['size'], r['fetch_bytes']) for r in rows]}")
    assert p.returncode == 0 and len(rows) == 1
    assert rows[0]["shard"] == "2"
    assert int(rows[0]["fetch_bytes"]) in (2500 * 16 * 4, toy.st.size(f"splits/{toy.dataset}/3/vectors/shard_2.npy"))


@needs_s3
def test_xc_point_json_has_exactly_the_csv_fields(toy, tmp_path):
    standard_shards(toy)
    p = toy.run("calibrate-point", "--shard", "1", "--size", "700", "--threads", "1", "--repeat", "4")
    print(f"stdout: {p.stdout.strip()}")
    assert p.returncode == 0
    doc = json.loads(p.stdout)
    print(f"JSON keys == CSV header (same order): {list(doc) == HEADER}; missing {set(HEADER) - set(doc)}, "
          f"extra {set(doc) - set(HEADER)}")
    assert list(doc) == HEADER
    assert doc["size"] == 700 and doc["threads"] == 1 and doc["repeat"] == 4 and doc["shard"] == 1
    for k in ("n_shards", "size", "dim", "threads", "m", "ef_construction", "repeat", "fetch_bytes",
              "index_bytes", "maxrss_bytes"):
        assert isinstance(doc[k], int), f"{k} is an integer"


@needs_s3
def test_xc_grid_size_beyond_shard(toy, tmp_path):
    """A size beyond the shard is refused in one line (Step 3 tests table). In a grid, what was measured
    before it stays (decision 23: rows appended as they come)."""
    standard_shards(toy)
    out = tmp_path / "beyond.csv"
    q = toy.run("calibrate-point", "--shard", "2", "--size", "2501", "--threads", "1")
    print(f"calibrate-point size 2501 of 2500: exit {q.returncode}, stderr {q.stderr.strip()!r}")
    p = toy.run("calibrate", "--shard", "2", "--sizes", "600,2600", "--threads", "1", "--out", str(out))
    rows = read_rows(out) if out.exists() else []
    err = p.stderr.strip().splitlines()
    print(f"grid: exit {p.returncode}; stderr {len(err)} lines, last {err[-1:]}; rows kept {[r['size'] for r in rows]}")
    print("=> " + ("refused before measuring anything" if not rows else "the 600 point was measured and kept; the "
                   "oversize point was discovered only when its turn came"))
    assert q.returncode != 0 and "Traceback" not in q.stderr and len(q.stderr.strip().splitlines()) == 1
    assert p.returncode != 0
    assert [r["size"] for r in rows] in ([], ["600"]), "no row for the impossible size"
    assert "Traceback" not in p.stderr, "the grid relays the refusal as a Python traceback, not one line"


@needs_s3
def test_xc_grid_peak_rss_is_per_point(toy, tmp_path):
    """Decision 23: one child process per point so its peak memory is its own. A small point run after a
    large one must report a small peak."""
    big = toy.rng.standard_normal((120_000, 32)).astype(np.float32)
    toy.dim = 32
    toy.q = toy.rng.standard_normal((600, 32)).astype(np.float32)
    toy.st.put_npy(f"datasets/{toy.dataset}/queries.npy", toy.q)
    toy.put_shards([big[:1000], big])
    out = tmp_path / "rss.csv"
    p = toy.run("calibrate", "--sizes", "120000,1000", "--threads", "4,2", "--out", str(out), timeout=900)
    assert p.returncode == 0
    rows = read_rows(out)
    seq = [(int(r["threads"]), int(r["size"]), int(r["maxrss_bytes"]), int(r["index_bytes"])) for r in rows]
    for th, s, rss, ib in seq:
        print(f"threads {th} size {s:>6}: peak RSS {rss / 2**20:7.1f} MiB, index {ib / 2**20:6.1f} MiB")
    big_rows = [r for r in seq if r[1] == 120_000]
    small_rows = [r for r in seq if r[1] == 1000]
    after_big = [i for i, r in enumerate(seq) if r[1] == 1000 and any(b[1] == 120_000 for b in seq[:i])]
    print(f"small points that ran after a large one: rows {after_big}")
    assert after_big, "the grid order put a small point after a large one"
    for b in big_rows:
        assert b[2] > b[3], "peak RSS (bytes) exceeds the index's bytes: the unit is bytes on this platform"
    for s in small_rows:
        assert s[2] < min(b[2] for b in big_rows) - 20 * 2**20, "a small point's peak is its own, not the large one's"
    assert max(s[2] for s in small_rows) <= 1.3 * min(s[2] for s in small_rows)
    print(f"platform {sys.platform}: maxrss_bytes looked like bytes here (ru_maxrss is bytes on macOS, KiB on Linux)")


@needs_s3
def test_xc_grid_interrupted_keeps_rows_and_rerun_appends(toy, tmp_path):
    """Decision 23: rows appended as they come, so an interrupted grid keeps what it measured. Kill the
    parent (SIGKILL) mid-grid, count rows, rerun into the same --out."""
    standard_shards(toy)
    out = tmp_path / "intr.csv"
    argv = toy.argv("calibrate", "--sizes", "500", "--threads", "1", "--repeats", "40", "--out", str(out))
    proc = subprocess.Popen(argv, env=toy.env(), cwd=BUILDER, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    deadline = time.monotonic() + 120
    n = 0
    while time.monotonic() < deadline:
        n = len(read_rows(out)) if out.exists() else 0
        if n >= 3:
            break
        time.sleep(0.05)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    at_kill = len(read_rows(out))
    time.sleep(3)
    pg = subprocess.run(["pgrep", "-g", str(proc.pid), "-f", "kgpu_builder"], capture_output=True, text=True).stdout
    orphans = pg.split()
    after = read_rows(out)
    raw = out.read_text()
    print(f"parent killed with {at_kill} rows on disk; 3 s later {len(after)} rows; orphan children still running: "
          f"{len(orphans)}; file ends with a newline: {raw.endswith(chr(10))}")
    assert 3 <= at_kill < 40, "rows were on disk before the grid finished"
    for r in after:
        assert None not in r and all(r[k] not in (None, "") for k in HEADER), "every kept row is complete"
        assert r["repeat"].isdigit()
    assert [int(r["repeat"]) for r in after] == list(range(len(after)))
    p = toy.run("calibrate", "--sizes", "500", "--threads", "1", "--repeats", "1", "--out", str(out))
    lines = out.read_text().splitlines()
    final = read_rows(out)
    headers = sum(1 for l in lines if l.split(",")[:2] == HEADER[:2])
    print(f"rerun into the same CSV: exit {p.returncode}; rows {len(after)} -> {len(final)}; header lines {headers}; "
          f"repeat column now {[r['repeat'] for r in final]}")
    assert p.returncode == 0
    assert len(final) == len(after) + 1, "appended, not overwritten"
    assert headers == 1, "the header is written once"
    print("=> note: the rerun's point is repeat 0 again: (size, threads, repeat) is no longer a unique key in the file"
          if final[-1]["repeat"] == "0" else "=> the rerun continued the repeat numbering")


@needs_s3
def test_xc_grid_existing_csv_with_other_header(toy, tmp_path):
    """Spec silent: --out names an existing CSV with a different header. Acceptable: refuse, or leave a file
    whose every row parses under its own header. Not acceptable: point rows under a foreign header."""
    standard_shards(toy)
    out = tmp_path / "other.csv"
    out.write_text("a,b,c\n1,2,3\n")
    p = toy.run("calibrate", "--sizes", "500", "--threads", "1", "--out", str(out))
    text = out.read_text()
    print(f"exit {p.returncode}; stderr tail {p.stderr.strip().splitlines()[-1:]}; file now:\n{text}")
    if p.returncode != 0:
        assert text == "a,b,c\n1,2,3\n", "refused and left the file alone"
        return
    rows = read_rows(out)
    assert all(None not in r for r in rows) and list(rows[0]) == HEADER, \
        "the file mixes a foreign header with point rows: neither fit nor the plot can read it as points"


# ------------------------------------------------------------------ the plot


def run_plot(*csvs: Path, out: Path) -> subprocess.CompletedProcess:
    p = subprocess.run(["uv", "run", "python", "plot_calibration.py", *map(str, csvs), "-o", str(out)],
                       capture_output=True, text=True, cwd=ANALYSIS, timeout=300)
    print(f"plot {[c.name for c in csvs]} -> exit {p.returncode}; stdout {p.stdout.strip()[-200:]!r}; "
          f"png {out.exists() and out.stat().st_size} bytes")
    return p


def is_png(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_xc_plot_committed_csvs(tmp_path):
    for csvs, name in (((SIFT_CSV, GIST_CSV), "both"), ((SIFT_CSV,), "sift"), ((GIST_CSV,), "gist")):
        out = tmp_path / f"{name}.png"
        p = run_plot(*csvs, out=out)
        assert p.returncode == 0 and is_png(out)


def test_xc_plot_synthetic_and_one_series(tmp_path):
    rows = [point(n, th, (1e-7 if th == 2 else 5e-8) * x_of(n, 16) * (1 + 0.01 * i), dim=16, repeat=i)
            for th in (2, 8) for n in (500, 1000, 2000) for i in range(2)]
    toy_csv = write_csv(tmp_path / "toy.csv", rows)
    out = tmp_path / "toy.png"
    assert run_plot(toy_csv, out=out).returncode == 0 and is_png(out)
    one = write_csv(tmp_path / "one_series.csv", [r for r in rows if r["threads"] == 2])
    out1 = tmp_path / "one.png"
    p = run_plot(one, out=out1)
    assert p.returncode == 0 and is_png(out1)


def test_xc_plot_empty_csv(tmp_path):
    """Spec silent. `fit` refuses a CSV with no points; the plot is recorded here for comparison."""
    f = tmp_path / "empty.csv"
    f.write_text(",".join(HEADER) + "\n")
    out = tmp_path / "empty.png"
    p = run_plot(f, out=out)
    print(f"stderr tail: {p.stderr.strip().splitlines()[-1:] }")
    print("=> " + ("wrote an empty figure and exited 0 (fit, on the same file, exits 1)" if p.returncode == 0
                   else "refused"))
    assert "Traceback" not in p.stderr
