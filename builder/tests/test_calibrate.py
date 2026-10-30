"""The fit, by hand on synthetic points; a two-point grid on a toy split in
the throwaway store through the CLI (needs TEST_S3_ENDPOINT)."""

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from kgpu_builder import calibrate as cal


def synth(a, threads, sizes, dim=128, noise_sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        t = a * n * math.log(n) * dim * math.exp(noise_sigma * rng.standard_normal())
        rows.append({"size": n, "dim": dim, "threads": threads, "build_s": t, "recall": 0.99, "fetch_bytes": n * dim * 4, "fetch_s": n * dim * 4 / 150e6})
    return rows


def test_fit_recovers_the_constant_exactly_without_noise():
    rows = synth(1.4e-7, 2, [10_000, 50_000, 200_000]) + synth(1.4e-7 / 2.5, 8, [10_000, 50_000, 200_000])
    fits = {f.threads: f for f in cal.fit_points(rows)}
    print({th: (round(f.a, 12), round(f.sigma, 6), round(f.power_p, 4)) for th, f in fits.items()})
    assert abs(fits[2].a - 1.4e-7) < 1e-12 and fits[2].sigma < 1e-9 and fits[2].min_ratio > 0.999999
    assert abs(fits[8].a - 1.4e-7 / 2.5) < 1e-12
    env = cal.env_overrides(list(fits.values()), 2, 8, 2.0)
    print(env)
    assert env["COST_A"] == "1.4e-07" and env["COST_SPEEDUP"] == "2.500" and env["COST_GPU_OVERHEAD"] == "2.00"
    assert abs(float(env["COST_BANDWIDTH"]) - 150e6) < 1e3 and env["KGPU_ESTIMATE_SIGMA"] == "0.000"


def test_fit_sigma_is_the_std_of_the_log_residuals():
    rows = synth(1.4e-7, 2, [10_000 * k for k in range(1, 41)], noise_sigma=0.3, seed=1)
    f = cal.fit_points(rows)[0]
    print(f"40 points with log-normal noise sigma 0.3: fitted sigma {f.sigma:.3f}, A {f.a:.4g}, power exponent {f.power_p:.3f}")
    assert 0.2 < f.sigma < 0.4 and 0.9 < f.power_p < 1.2


def test_fit_sees_a_steeper_curve_in_the_power_exponent():
    rows = [{"size": n, "dim": 128, "threads": 2, "build_s": 1e-9 * n ** 1.3 * 128, "recall": 0.99, "fetch_bytes": 1, "fetch_s": 1} for n in [10_000, 30_000, 100_000, 300_000]]
    f = cal.fit_points(rows)[0]
    print(f"t ∝ n^1.3: power fit p = {f.power_p:.3f}; the n·ln(n) fit's ratios span {f.min_ratio:.2f}..{f.max_ratio:.2f}")
    assert abs(f.power_p - 1.3) < 0.02 and f.max_ratio / f.min_ratio > 1.3


def test_env_overrides_needs_both_thread_counts():
    f = cal.fit_points(synth(1e-7, 2, [10_000, 20_000]))
    with pytest.raises(ValueError):
        cal.env_overrides(f, 2, 8, 2.0)


@pytest.fixture
def toy():
    ep = os.environ.get("TEST_S3_ENDPOINT")
    if not ep:
        pytest.skip("TEST_S3_ENDPOINT not set; run scripts/test-db.sh")
    from kgpu_builder.config import Config
    from kgpu_builder.store import Store
    env = {"S3_ENDPOINT": ep, "S3_ACCESS_KEY": os.environ["TEST_S3_ACCESS_KEY"],
           "S3_SECRET_KEY": os.environ["TEST_S3_SECRET_KEY"], "S3_BUCKET": "kgpu-test-cal"}
    st = Store(Config.from_env(env))
    st.ensure_bucket(); st.delete_prefix("")
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6000, 32)).astype(np.float32)
    st.put_npy("splits/toy/2/vectors/shard_0.npy", x[:2000])
    st.put_npy("splits/toy/2/vectors/shard_1.npy", x[2000:])
    st.put_json("splits/toy/2/manifest.json", {"shards": [{"shard_id": 0, "size": 2000}, {"shard_id": 1, "size": 4000}]})
    st.put_npy("datasets/toy/queries.npy", rng.standard_normal((100, 32)).astype(np.float32))
    return env


def test_calibrate_grid_through_the_cli(toy, tmp_path):
    out = tmp_path / "cal.csv"
    r = subprocess.run([sys.executable, "-m", "kgpu_builder", "calibrate", "toy", "2", "--sizes", "1000,4000", "--threads", "1,2", "--out", str(out)],
                       env={**os.environ, **toy}, capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-300:])
    assert r.returncode == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4 and [int(x["shard"]) for x in rows] == [1, 1, 1, 1], "the largest shard (1, 4000 vectors) is the source"
    assert [(int(x["threads"]), int(x["size"])) for x in rows] == [(1, 1000), (1, 4000), (2, 1000), (2, 4000)]
    for x in rows:
        assert float(x["recall"]) > 0.9 and float(x["build_s"]) > 0 and int(x["index_bytes"]) > 0 and int(x["maxrss_bytes"]) > 1e6 and int(x["dim"]) == 32
    r = subprocess.run([sys.executable, "-m", "kgpu_builder", "fit", str(out), "--shard-threads", "1", "--worker-threads", "2"], capture_output=True, text=True)
    print(r.stdout[-900:])
    assert r.returncode == 0 and "COST_A=" in r.stdout and "COST_SPEEDUP=" in r.stdout and "threads 1:" in r.stdout
    r = subprocess.run([sys.executable, "-m", "kgpu_builder", "calibrate-point", "toy", "2", "--shard", "0", "--size", "5000", "--threads", "1"],
                       env={**os.environ, **toy}, capture_output=True, text=True)
    assert r.returncode == 1 and "exceeds" in (r.stderr + r.stdout)
    print("a size beyond the shard is refused:", (r.stderr + r.stdout).strip().splitlines()[-1][:100])
