"""Cross-check (black box) of Phase 5 step 1 on the builder: the v1 calibration point and `fit --v1`.

Spec only: docs/design/phase5-gpu.md decisions 6-8 and "Step 1 (GPU-free): cost model v1"; docs/speedup.md
evidence 9; Phase 4 decisions 35 and 37; `python -m kgpu_builder fit|calibrate|calibrate-point --help`; the
Store's public methods. calibrate.py, cli.py and test_calibrate.py were NOT read. The toy-split helpers are
reused from the earlier cross-check, tests/test_xcheck_calibrate.py.

The fit tests are pure (CSV files written here, plus the committed calibration CSVs). The point/grid tests need
a throwaway RustFS (TEST_S3_ENDPOINT, TEST_S3_ACCESS_KEY, TEST_S3_SECRET_KEY) and skip without it.

v1 by hand (decision 6): build = A·n^P·ln(n)·(dim+D0); gate = G·Q·ln(n)·(dim+D0) (Q = the point's gate_queries);
fetch = fetch_bytes/fetch_s; upload = upload_bytes/upload_s; save = index_bytes/save_s;
COST_SPEEDUP = geometric mean over the worker's points of (shard model build time / worker measured build time).
"""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_xcheck_calibrate as xc  # noqa: E402  (the earlier cross-check's helpers: Toy, standard_shards, HEADER)

CAL = xc.CAL
BUILDER = xc.BUILDER
needs_s3 = xc.needs_s3

V1LINE = re.compile(
    r"v1 threads (\d+): (\d+) points; build = ([^\s]+) · n\^([^\s]+) · ln\(n\) · \(dim \+ ([^\s)]+)\) "
    r"\(ratio ([^\s.]+\.?\d*)\.\.([^\s,]+), sigma ([^\s)]+)\); gate ([^\s]+) s per query·ln\(n\)·\(dim\+D0\) on (\d+) queries; "
    r"fetch (\S+) MB/s, upload (\S+)(?: MB/s)?, save (\S+)(?: MB/s)?; min recall ([-\d.]+)")


def run_fit(*args: str) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-m", "kgpu_builder", "fit", *map(str, args)]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=BUILDER, timeout=300)
    print(f"ran: kgpu_builder fit {' '.join(Path(a).name if '/' in str(a) else str(a) for a in args)} -> exit {p.returncode}")
    return p


def parse_v1(stdout: str) -> tuple[list[dict], dict[str, str]]:
    lines, env, in_v1 = [], {}, False
    for line in stdout.splitlines():
        m = V1LINE.search(line)
        if m:
            g = m.groups()
            lines.append(dict(threads=int(g[0]), n=int(g[1]), a=float(g[2]), p=float(g[3]), d0=float(g[4]),
                              ratio_lo=float(g[5]), ratio_hi=float(g[6]), sigma=float(g[7]), gate=float(g[8]),
                              gate_q=int(g[9]), fetch=g[10], upload=g[11], save=g[12], recall=float(g[13])))
            continue
        if line.startswith("v1 COST_*"):
            in_v1 = True
            continue
        if in_v1:
            m = re.match(r"\s+((?:COST|KGPU)_[A-Z_0-9]+)=(\S+)$", line)
            if m:
                env[m.group(1)] = m.group(2)
            elif line.strip():
                in_v1 = False
    return lines, env


def v1_point(size, dim, threads, build_s, gate_s, gate_q, fetch_bw, upload_bw, save_bw, dataset="xv1", repeat=0):
    idx = size * (dim * 4 + 2 * 16 * 4)
    vec = size * dim * 4
    return {"dataset": dataset, "n_shards": 6, "shard": 1, "size": size, "dim": dim, "threads": threads, "m": 16,
            "ef_construction": 200, "repeat": repeat, "fetch_s": vec / fetch_bw, "fetch_bytes": vec,
            "build_s": build_s, "gate_s": gate_s, "recall": 0.99, "index_bytes": idx, "maxrss_bytes": 1,
            "host": "xhost", "at": "2026-10-31T00:00:00Z", "gate_queries": gate_q, "save_s": idx / save_bw,
            "upload_s": idx / upload_bw, "upload_bytes": idx}


def no_traceback(p: subprocess.CompletedProcess) -> bool:
    return "Traceback" not in p.stderr and "Traceback" not in p.stdout


# ------------------------------------------------------------------ fit --v1, pure


def test_xc_v1_fit_reproduces_the_cloud_block():
    shard = [CAL / "kgpu-shards-sift-128-6.csv", CAL / "kgpu-shards-gist-960-300k-6.csv"]
    worker = [CAL / "worker-sift-128-6.csv", CAL / "worker-gist-960-300k-6.csv"]
    p = run_fit(*shard, "--v1", "--worker-csv", *worker, "--gate-queries", "10000")
    v1, env = parse_v1(p.stdout)
    for line in p.stdout.splitlines():
        if line.startswith("v1 ") or line.startswith("  COST_") or line.startswith("  KGPU_"):
            print("  | " + line)
    print("spec (step 1 results, speedup.md evidence 9, brief): A 2.3e-09, P 1.220, D0 187, speedup 2.126, gate 1.787e-08, "
          "bandwidth 1.573e+08, sigma 0.113; shard fit sigma 0.079 at 2 threads, worker 1.53e-9·n^1.198·(dim+142) sigma 0.091")
    assert p.returncode == 0, p.stderr[-2000:]
    want = {"COST_MODEL": "calibrated", "COST_A": "2.3e-09", "COST_P": "1.220", "COST_D0": "187", "COST_SPEEDUP": "2.126",
            "COST_GATE": "1.787e-08", "COST_GATE_QUERIES": "10000", "COST_BANDWIDTH": "1.573e+08", "KGPU_ESTIMATE_SIGMA": "0.113"}
    bad = {k: (env.get(k), v) for k, v in want.items() if env.get(k) != v}
    print(f"v1 block: {env}\n=> mismatches (got, want): {bad}")
    assert not bad
    sh = [l for l in v1 if l["threads"] == 2]
    assert sh and sh[0]["sigma"] == pytest.approx(0.079, abs=0.0015) and sh[0]["d0"] == 187
    wk = [l for l in v1 if l["threads"] == 8 and abs(l["d0"] - 142) < 1]
    print(f"v1 lines: {[(l['threads'], l['a'], l['p'], l['d0'], l['sigma']) for l in v1]}")
    assert wk and wk[0]["a"] == pytest.approx(1.53e-9, rel=0.01) and wk[0]["p"] == pytest.approx(1.198, abs=0.001)
    assert wk[0]["sigma"] == pytest.approx(0.091, abs=0.0015)
    t8 = [l for l in v1 if l["threads"] == 8]
    print(f"note: {len(t8)} 'v1 threads 8' lines (one from the shard CSVs' 8-thread points, one from --worker-csv); "
          "the output does not label which file each comes from")
    print(f"note: the per-thread lines report the gate 'on {sh[0]['gate_q']} queries' although these v0-schema CSVs have "
          "no gate_queries column: the fit assumes a query count for v0 points (not written in the docs)")
    for k in ("COST_SAVE_BANDWIDTH", "COST_BANDWIDTH_OUT", "COST_SHARD_USES_STORE"):
        print(f"note: {k} in the v1 block: {env.get(k, 'absent')} (v0 CSVs carry no save/upload timing)")


def _synthetic(tmp_path: Path):
    """Shard (2 threads) and worker (8 threads, another file) points generated exactly from known v1 constants."""
    S = dict(a=3.0e-9, p=1.2, d0=100.0, g=2.0e-8, q=500, fetch=1.0e8, upload=4.0e7, save=5.0e8)
    W = dict(a=1.2e-9, p=1.18, d0=60.0, g=9.0e-9, q=500, fetch=1.5e8, upload=6.0e7, save=8.0e8)
    sizes = [10_000, 20_000, 50_000, 100_000, 200_000, 400_000]
    dims = [128, 960]

    def rows(c, threads):
        out = []
        for n in sizes:
            for d in dims:
                b = c["a"] * n ** c["p"] * math.log(n) * (d + c["d0"])
                g = c["g"] * c["q"] * math.log(n) * (d + c["d0"])
                out.append(v1_point(n, d, threads, b, g, c["q"], c["fetch"], c["upload"], c["save"]))
        return out

    srows, wrows = rows(S, 2), rows(W, 8)
    sp = xc.write_csv(tmp_path / "shard.csv", srows)
    wp = xc.write_csv(tmp_path / "worker.csv", wrows)
    ratios = [(S["a"] * r["size"] ** S["p"] * math.log(r["size"]) * (r["dim"] + S["d0"])) / r["build_s"] for r in wrows]
    speedup = math.exp(sum(map(math.log, ratios)) / len(ratios))
    return S, W, sp, wp, speedup, ratios


def test_xc_v1_fit_recovers_known_constants(tmp_path):
    S, W, sp, wp, speedup, ratios = _synthetic(tmp_path)
    print(f"synthetic shard (2 threads): {S}\nsynthetic worker (8 threads, --worker-csv): {W}")
    print(f"speedup by hand = geomean(shard model / worker measured) over the worker's {len(ratios)} points = {speedup:.4f} "
          f"(per point {min(ratios):.3f}..{max(ratios):.3f}); a ratio of A's would give {S['a'] / W['a']:.3f}")
    p = run_fit(sp, "--v1", "--worker-csv", wp, "--gate-queries", "10000")
    v1, env = parse_v1(p.stdout)
    for line in p.stdout.splitlines():
        if line.startswith("v1 ") or line.startswith("  COST_") or line.startswith("  KGPU_"):
            print("  | " + line)
    assert p.returncode == 0 and no_traceback(p), p.stderr[-2000:]
    sh = next(l for l in v1 if l["threads"] == 2)
    wk = [l for l in v1 if l["threads"] == 8]
    print(f"shard line: {sh}\nworker line(s): {wk}")
    print("spec (decision 8): A, P, D0 fitted in log space with a 1-D search on D0; the gate per query; the three bandwidths; "
          "COST_SPEEDUP the geometric mean (decision 6)")
    assert sh["d0"] == pytest.approx(S["d0"], abs=3), "D0 recovered (search resolution unknown: +-3)"
    assert sh["p"] == pytest.approx(S["p"], abs=0.01)
    assert sh["a"] == pytest.approx(S["a"], rel=0.1)
    assert sh["sigma"] < 0.02, "exact data: residuals ~0"
    assert sh["gate"] == pytest.approx(S["g"], rel=0.05), "the gate per query·ln(n)·(dim+D0), with Q from the gate_queries column"
    assert sh["gate_q"] == 500
    assert wk and wk[0]["d0"] == pytest.approx(W["d0"], abs=3) and wk[0]["p"] == pytest.approx(W["p"], abs=0.01)
    assert float(env["COST_A"]) == pytest.approx(S["a"], rel=0.1)
    assert float(env["COST_P"]) == pytest.approx(S["p"], abs=0.01)
    assert float(env["COST_D0"]) == pytest.approx(S["d0"], abs=3)
    assert float(env["COST_SPEEDUP"]) == pytest.approx(speedup, rel=0.02), \
        f"COST_SPEEDUP {env['COST_SPEEDUP']} vs geomean {speedup:.4f} (ratio of A's {S['a'] / W['a']:.3f})"
    assert float(env["COST_GATE"]) == pytest.approx(S["g"], rel=0.05)
    assert env["COST_GATE_QUERIES"] == "10000", "--gate-queries is the block's COST_GATE_QUERIES"
    assert float(env["COST_BANDWIDTH"]) == pytest.approx(W["fetch"], rel=0.01), \
        "decision 7: the fetch is the worker node's median"
    print(f"bandwidths in the block: fetch {env.get('COST_BANDWIDTH')}, upload {env.get('COST_BANDWIDTH_OUT', 'absent')}, "
          f"save {env.get('COST_SAVE_BANDWIDTH', 'absent')} (shard side generated at upload {S['upload']:.3g} save {S['save']:.3g}; "
          f"worker at upload {W['upload']:.3g} save {W['save']:.3g}; which side the fit uses for upload/save is not written down)")
    assert "COST_BANDWIDTH_OUT" in env and "COST_SAVE_BANDWIDTH" in env, \
        "v1 points carry upload and save timings; decision 8: the fit emits the three bandwidths"
    assert float(env["COST_BANDWIDTH_OUT"]) in (pytest.approx(W["upload"], rel=0.01), pytest.approx(S["upload"], rel=0.01))
    assert float(env["COST_SAVE_BANDWIDTH"]) in (pytest.approx(W["save"], rel=0.01), pytest.approx(S["save"], rel=0.01))
    print(f"KGPU_ESTIMATE_SIGMA={env.get('KGPU_ESTIMATE_SIGMA')} (the cloud block's equals the speedup spread; here every "
          f"ratio's log spread is {float(np.std(np.log(ratios))):.3f})")


def test_xc_v1_fit_gate_queries_default(tmp_path):
    _, _, sp, wp, _, _ = _synthetic(tmp_path)
    p = run_fit(sp, "--v1", "--worker-csv", wp)
    _, env = parse_v1(p.stdout)
    print(f"without --gate-queries: COST_GATE_QUERIES={env.get('COST_GATE_QUERIES', 'absent')} "
          "(help: 'how many held-out queries a real build's gate runs'; the default is not documented)")
    assert p.returncode == 0 and no_traceback(p)


def test_xc_v1_fit_single_size_does_not_crash(tmp_path):
    rows = [v1_point(50_000, d, t, 1.0 + 0.1 * r, 0.05, 500, 1e8, 4e7, 5e8, repeat=r) for d in (128,) for t in (2, 8) for r in (0, 1)]
    csvp = xc.write_csv(tmp_path / "one_size.csv", rows)
    p = run_fit(csvp, "--v1")
    print(p.stdout[-1500:] + p.stderr[-1500:])
    print("spec (brief): a v1 fit on a single-size CSV must not crash (P and D0 are not identifiable from one size and one dim)")
    assert no_traceback(p), "a traceback is a crash"
    assert p.returncode == 0 or len(p.stderr.strip().splitlines()) == 1, "either a fit or a one-line refusal"


def test_xc_v1_fit_single_point(tmp_path):
    csvp = xc.write_csv(tmp_path / "one.csv", [v1_point(50_000, 128, 2, 1.5, 0.05, 500, 1e8, 4e7, 5e8)])
    p = run_fit(csvp, "--v1", "--shard-threads", "2", "--worker-threads", "2")
    print(p.stdout[-1200:] + p.stderr[-1200:])
    assert no_traceback(p)


def test_xc_v1_fit_reads_v0_and_mixed_schemas(tmp_path):
    print("spec (decision 8): a v0 CSV still reads, its v1 columns absent; fit reads both schemas")
    p = run_fit(CAL / "laptop-sift-128-6.csv", "--v1")
    v1, env = parse_v1(p.stdout)
    print(f"v0 laptop CSV alone: exit {p.returncode}; v1 lines {[(l['threads'], l['upload'], l['save']) for l in v1]}; block {env}")
    assert p.returncode == 0 and no_traceback(p) and v1
    p = run_fit(CAL / "laptop-sift-128-6.csv", "--v1", "--worker-csv", CAL / "laptop-v1-sift-128-6.csv")
    v1, env = parse_v1(p.stdout)
    print(f"v0 shard CSV + v1 worker CSV: exit {p.returncode}; v1 lines {[(l['threads'], l['fetch'], l['upload'], l['save']) for l in v1]}; block {env}")
    assert p.returncode == 0 and no_traceback(p), p.stderr[-1500:]
    p = run_fit(CAL / "laptop-sift-128-6.csv", CAL / "laptop-v1-sift-128-6.csv", "--v1")
    print(f"both schemas in one fit: exit {p.returncode}; {p.stderr.strip()[-300:]}")
    assert p.returncode == 0 and no_traceback(p)


def test_xc_v1_fit_laptop_dimension_offset_is_zero():
    p = run_fit(CAL / "laptop-sift-128-6.csv", CAL / "laptop-gist-960-300k-6.csv", "--v1")
    v1, env = parse_v1(p.stdout)
    for l in v1:
        print(f"laptop v1 threads {l['threads']}: A {l['a']:.4g} P {l['p']} D0 {l['d0']} sigma {l['sigma']}")
    print("spec (step 1 results): the laptop's v1 fit of its own data gives D0 = 0 (Phase 4 decision 35: the dimension term is the machine's)")
    assert p.returncode == 0 and v1
    assert all(l["d0"] == 0 for l in v1)


# ------------------------------------------------------------------ the extended point, against a store


@pytest.fixture
def toy(tmp_path):
    t = xc.Toy(tmp_path)
    yield t
    t.close()


@needs_s3
def test_xc_point_times_save_and_upload_and_deletes(toy, tmp_path):
    xc.standard_shards(toy)
    prefix = "xcheck/upl-" + toy.bucket
    before = set(toy.st.list(""))
    p = toy.run("calibrate-point", "--shard", "1", "--size", "2000", "--threads", "1", "--upload-prefix", prefix)
    print(f"stdout: {p.stdout.strip()}\nstderr tail: {p.stderr.strip()[-400:]}")
    assert p.returncode == 0
    doc = json.loads(p.stdout)
    print("spec (decision 8): after the build and gate the point saves (write + sha256) and uploads the index under the "
          "prefix, deletes it, and records gate_queries; the JSON is the v1 CSV row")
    assert list(doc) == xc.HEADER
    assert doc["save_s"] > 0 and doc["upload_s"] > 0
    assert doc["upload_bytes"] == doc["index_bytes"], "what is uploaded is the saved index"
    assert doc["index_bytes"] >= 2000 * (16 * 4 + 2 * 16 * 4) * 0.5
    print(f"gate_queries {doc['gate_queries']} (the toy split has {len(toy.q)} held-out queries)")
    assert 0 < doc["gate_queries"] <= len(toy.q)
    after = set(toy.st.list(""))
    left = sorted(after - before)
    print(f"keys under {prefix!r} afterwards: {toy.st.list(prefix)}; new keys anywhere in the bucket: {left}")
    assert toy.st.list(prefix) == [] and left == [], "the uploaded index is deleted afterwards"


@needs_s3
def test_xc_point_without_upload_prefix(toy):
    xc.standard_shards(toy)
    p = toy.run("calibrate-point", "--shard", "1", "--size", "1000", "--threads", "1")
    assert p.returncode == 0
    doc = json.loads(p.stdout)
    print(f"no --upload-prefix: save_s {doc['save_s']!r}, upload_s {doc['upload_s']!r}, upload_bytes {doc['upload_bytes']!r} "
          "(what an untimed upload records is not written down)")
    assert doc["save_s"] not in (None, "") and float(doc["save_s"]) > 0, "the save is timed regardless of the upload"


@needs_s3
def test_xc_grid_uploads_under_scratch_by_default_and_cleans_up(toy, tmp_path):
    xc.standard_shards(toy)
    out = tmp_path / "grid.csv"
    host = socket.gethostname()
    p = toy.run("calibrate", "--sizes", "500,1500", "--threads", "1", "--out", str(out))
    print(p.stdout[-800:] + p.stderr[-600:])
    assert p.returncode == 0
    rows = xc.read_rows(out)
    print(f"rows: {[(r['size'], r['save_s'], r['upload_s'], r['upload_bytes'], r['gate_queries']) for r in rows]}")
    print("spec (decision 8): the upload is on by default in the grid, under calibration/scratch/<host>/, deleted afterwards")
    assert all(float(r["upload_s"]) > 0 and int(r["upload_bytes"]) == int(r["index_bytes"]) for r in rows)
    leftover = toy.st.list("calibration/")
    print(f"keys under calibration/ afterwards: {leftover} (host {host})")
    assert leftover == []
    out2 = tmp_path / "noup.csv"
    p = toy.run("calibrate", "--sizes", "500", "--threads", "1", "--out", str(out2), "--no-upload")
    rows = xc.read_rows(out2)
    print(f"--no-upload: {[(r['save_s'], r['upload_s'], r['upload_bytes']) for r in rows]}")
    assert p.returncode == 0 and rows
    assert rows[0]["upload_s"] in ("", "0", "0.0") or float(rows[0]["upload_s"]) == 0


@needs_s3
def test_xc_grid_v1_csv_feeds_fit_v1(toy, tmp_path):
    xc.standard_shards(toy)
    out = tmp_path / "g.csv"
    p = toy.run("calibrate", "--sizes", "500,1000,2000", "--threads", "1", "--out", str(out))
    assert p.returncode == 0
    f = run_fit(out, "--v1", "--shard-threads", "1", "--worker-threads", "1")
    v1, env = parse_v1(f.stdout)
    print(f"fit --v1 on a freshly measured toy grid: exit {f.returncode}; lines {v1}; block {env}; stderr {f.stderr.strip()[-400:]}")
    assert f.returncode == 0 and no_traceback(f) and v1
    assert v1[0]["upload"] != "n/a" and v1[0]["save"] != "n/a", "a v1 grid carries upload and save rates into the fit"


def test_xc_v1_fit_on_points_without_upload(tmp_path):
    """A grid run with --no-upload records upload_s 0 and upload_bytes 0 (seen in the grid test). Its rate is 0/0."""
    rows = []
    for n in (10_000, 50_000, 200_000):
        for d in (128, 960):
            for t in (2, 8):
                r = v1_point(n, d, t, 3e-9 * n ** 1.2 * math.log(n) * (d + 100) / (1 if t == 2 else 2.5), 0.05, 500,
                             1e8, 4e7, 5e8)
                r["upload_s"], r["upload_bytes"] = 0.0, 0
                rows.append(r)
    csvp = xc.write_csv(tmp_path / "noupload.csv", rows)
    p = run_fit(csvp, "--v1")
    v1, env = parse_v1(p.stdout)
    print(f"exit {p.returncode}; v1 lines {[(l['threads'], l['fetch'], l['upload'], l['save']) for l in v1]}; "
          f"block COST_BANDWIDTH_OUT={env.get('COST_BANDWIDTH_OUT', 'absent')}; stderr {p.stderr.strip()[-600:]}")
    print("expected: no crash, the upload reported as unmeasured (n/a), and no COST_BANDWIDTH_OUT of 0 (which the model reads as "
          "'same as the fetch'), inf or nan")
    assert no_traceback(p) and p.returncode == 0
    out = env.get("COST_BANDWIDTH_OUT")
    assert out is None or (math.isfinite(float(out)) and float(out) > 0)
    assert all(l["upload"] in ("n/a",) or (l["upload"].replace(".", "").isdigit() and float(l["upload"]) > 0) for l in v1)
