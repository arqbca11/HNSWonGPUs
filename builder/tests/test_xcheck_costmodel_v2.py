"""Cross-check (black box) of cost model v2 on the builder: `fit --v2 --fixed --drop-warmup`.

Spec only: docs/design/phase5-gpu.md decisions 6-8, 13, 14 and "The first GPU session"; `python -m kgpu_builder
fit --help`; the brief. calibrate.py, cli.py and test_calibrate.py were NOT read. Helpers are reused from the
earlier cross-checks (tests/test_xcheck_calibrate.py, tests/test_xcheck_costmodel_v1.py).

v2's worker side by hand (decision 14): build_s = F + A·n^P·ln(n)·(dim+D0) on the worker's points; --fixed fits F
on a grid like D0 (worker side only); --drop-warmup leaves out "the process's first point" per thread count (the
one-off CUDA warm-up); without --worker-csv, --v2 prints no v2 block; KGPU_ESTIMATE_SIGMA is the larger of the two
sides' fit errors.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_xcheck_calibrate as xc  # noqa: E402
import test_xcheck_costmodel_v1 as v1x  # noqa: E402

CAL = xc.CAL
run_fit = v1x.run_fit
no_traceback = v1x.no_traceback

WLINE = re.compile(
    r"v1 threads (\d+): (\d+) points; build = (?:([\d.e+-]+) s \+ )?([^\s]+) · n\^([^\s]+) · ln\(n\) · \(dim \+ ([^\s)]+)\) "
    r"\(ratio [^,]+, sigma ([^\s)]+)\); gate ([^\s]+) s per query.*\((shard|worker) side\)")

L4_BLOCK = {"COST_WORKER_FIXED": "1.154", "COST_WORKER_A": "1.295e-08", "COST_WORKER_P": "0.891", "COST_WORKER_D0": "154",
            "COST_WORKER_GATE": "2.068e-08", "COST_BANDWIDTH": "1.388e+08", "COST_BANDWIDTH_OUT": "1.007e+08",
            "COST_SAVE_BANDWIDTH": "2.914e+08", "KGPU_ESTIMATE_SIGMA": "0.097"}


def parse(stdout: str) -> tuple[list[dict], dict[str, str] | None]:
    lines, v2, in_v2 = [], None, False
    for line in stdout.splitlines():
        m = WLINE.search(line)
        if m:
            g = m.groups()
            lines.append(dict(threads=int(g[0]), n=int(g[1]), fixed=float(g[2]) if g[2] else None, a=float(g[3]),
                              p=float(g[4]), d0=float(g[5]), sigma=float(g[6]), gate=float(g[7]), side=g[8], raw=line))
            continue
        if line.startswith("v2 COST_*"):
            in_v2, v2 = True, {}
            continue
        if in_v2:
            mm = re.match(r"\s+((?:COST|KGPU)_[A-Z_0-9]+)=(\S+)$", line)
            if mm:
                v2[mm.group(1)] = mm.group(2)
            elif line.strip():
                in_v2 = False
    return lines, v2


def show(p) -> None:
    for line in p.stdout.splitlines():
        if line.startswith(("v1 threads", "v2 ", "  COST_WORKER", "  KGPU_", "  COST_BANDWIDTH", "  COST_SAVE", "(no ")):
            print("  | " + line)


SHARD = sorted(CAL.glob("kgpu-shards-*.csv"))   # what the shell's glob gives: gist-960-300k before sift-128
GPU = sorted(CAL.glob("gpu-l4-*.csv"))
FLAGS = ("--worker-threads", "6", "--v1", "--v2", "--fixed", "--drop-warmup")


# ------------------------------------------------------------------ the committed CSVs


def _check_block(v2: dict | None) -> dict:
    return {k: ((v2 or {}).get(k), v) for k, v in L4_BLOCK.items() if (v2 or {}).get(k) != v}


def test_xc_v2_fit_reproduces_the_l4_block_as_documented():
    print("spec (decision 14, brief): `fit <shard csvs> --worker-csv <worker csvs> --worker-threads 6 --v1 --v2 --fixed "
          "--drop-warmup` on kgpu-shards-*.csv against gpu-l4-*.csv prints COST_WORKER_FIXED=1.154 ... KGPU_ESTIMATE_SIGMA=0.097")
    print(f"the globs, as a shell expands them: shards {[p.name for p in SHARD]}, worker {[p.name for p in GPU]}")
    p = run_fit(*SHARD, "--worker-csv", *GPU, *FLAGS)
    show(p)
    lines, v2 = parse(p.stdout)
    assert p.returncode == 0 and no_traceback(p), p.stderr[-1500:]
    bad = _check_block(v2)
    print(f"=> v2 block mismatches (got, documented): {bad}")
    assert v2 is not None and not bad


def test_xc_v2_fit_does_not_depend_on_the_csv_order():
    print("spec (decision 14): --drop-warmup leaves out 'the process's first point' (the one-off CUDA warm-up); which file is "
          "named first on the command line is not part of the spec, so both orders must give the same block")
    blocks = {}
    for label, w in (("gist first", GPU), ("sift first", list(reversed(GPU)))):
        p = run_fit(*SHARD, "--worker-csv", *w, *FLAGS)
        lines, v2 = parse(p.stdout)
        wl = [l for l in lines if l["side"] == "worker"]
        print(f"{label} ({[x.name for x in w]}): worker line {wl[0]['raw'][:140] if wl else None}")
        print(f"   v2 block {v2}")
        blocks[label] = v2
    import csv
    for f in GPU:
        rows = list(csv.DictReader(open(f)))
        print(f"{f.name}: first row by file order size {rows[0]['size']} build_s {float(rows[0]['build_s']):.2f} at {rows[0]['at']}; "
              f"earliest 'at' {min(r['at'] for r in rows)}")
    diff = {k: (blocks['gist first'].get(k), blocks['sift first'].get(k)) for k in L4_BLOCK
            if blocks['gist first'].get(k) != blocks['sift first'].get(k)}
    print(f"=> keys that change with the order of --worker-csv: {diff}")
    assert not diff


def test_xc_v2_fit_with_the_warmup_file_first_matches_the_block():
    w = sorted(GPU, key=lambda f: "sift" not in f.name)  # sift-128 (the process's first points) first
    p = run_fit(*SHARD, "--worker-csv", *w, *FLAGS)
    show(p)
    lines, v2 = parse(p.stdout)
    wl = [l for l in lines if l["side"] == "worker"]
    print("spec (brief): the worker-side line 'v1 threads 6: 19 points; build = 1.15 s + 1.295e-08 · n^0.891 · ln(n) · (dim + 154) "
          "... (worker side)' and the v2 block")
    assert p.returncode == 0 and no_traceback(p)
    assert wl, "no worker-side line"
    l = wl[0]
    print(f"worker line parsed: threads {l['threads']} points {l['n']} fixed {l['fixed']} A {l['a']} P {l['p']} D0 {l['d0']} sigma {l['sigma']}")
    assert (l["threads"], l["n"], l["fixed"], l["a"], l["p"], l["d0"], l["sigma"]) == (6, 19, 1.15, 1.295e-08, 0.891, 154, 0.097)
    bad = _check_block(v2)
    print(f"=> v2 block mismatches: {bad}; shard side sigma {[x['sigma'] for x in lines if x['side'] == 'shard' and x['threads'] == 2]} "
          f"(KGPU_ESTIMATE_SIGMA is the larger of the two sides')")
    assert not bad
    print(f"note: the v2 block starts with COST_MODEL={v2.get('COST_MODEL')} (not calibrated-l4); pasted as an environment it "
          "reaches the L4 block only because it sets every field the two blocks differ in")


# ------------------------------------------------------------------ synthetic data of the v2 shape


W2 = dict(fixed=1.2, a=1.0e-8, p=0.9, d0=150.0, g=2.0e-8, q=500, fetch=1.4e8, upload=1.0e8, save=2.9e8)
S2 = dict(a=3.0e-9, p=1.2, d0=100.0, g=2.0e-8, q=500, fetch=1.0e8, upload=4.0e7, save=5.0e8)
SIZES = [10_000, 20_000, 50_000, 100_000, 200_000, 400_000]
DIMS = [128, 960]


def _rows(c, threads, fixed=0.0, warmup=None, dataset="xv2", t0=0):
    out, k = [], t0
    for d in DIMS:
        for n in SIZES:
            b = fixed + c["a"] * n ** c["p"] * math.log(n) * (d + c["d0"])
            g = c["g"] * c["q"] * math.log(n) * (d + c["d0"])
            r = v1x.v1_point(n, d, threads, b, g, c["q"], c["fetch"], c["upload"], c["save"], dataset=f"{dataset}-{d}")
            r["at"] = f"2026-11-02T08:{k // 60:02d}:{k % 60:02d}Z"
            k += 1
            out.append(r)
    if warmup is not None:
        out[0]["build_s"] += warmup
    return out


def _files(tmp_path, wrows_by_file, srows=None):
    sp = xc.write_csv(tmp_path / "shard.csv", srows if srows is not None else _rows(S2, 2))
    wps = [xc.write_csv(tmp_path / f"worker{i}.csv", r) for i, r in enumerate(wrows_by_file)]
    return sp, wps


def _fit(sp, wps, *extra):
    p = run_fit(sp, "--worker-csv", *wps, "--shard-threads", "2", "--worker-threads", "6", "--v1", "--v2", *extra)
    show(p)
    lines, v2 = parse(p.stdout)
    wl = [l for l in lines if l["side"] == "worker" and l["threads"] == 6]
    return p, wl[0] if wl else None, v2, lines


def test_xc_v2_fixed_recovers_the_floor_and_the_shape(tmp_path):
    sp, wps = _files(tmp_path, [_rows(W2, 6, fixed=W2["fixed"])])
    print(f"synthetic worker (6 threads, exact): build = {W2['fixed']} + {W2['a']}·n^{W2['p']}·ln(n)·(dim+{W2['d0']}); gate {W2['g']}")
    p, wl, v2, _ = _fit(sp, wps, "--fixed")
    print(f"spec (decision 14, brief): with --fixed the fit recovers the floor, A, P and D0. Worker line: {wl}; v2 block {v2}")
    assert p.returncode == 0 and no_traceback(p) and wl and v2
    print(f"recovered: floor {v2['COST_WORKER_FIXED']} (true {W2['fixed']}), A {v2['COST_WORKER_A']} (true {W2['a']}, "
          f"{float(v2['COST_WORKER_A']) / W2['a'] - 1:+.0%}), P {v2['COST_WORKER_P']} (true {W2['p']}), D0 {v2['COST_WORKER_D0']} "
          f"(true {W2['d0']}), sigma {wl['sigma']}")
    print("tolerances: the floor's grid resolution is not specified (decision 14: 'on a grid like D0'); on exact data the grid "
          "lands within ~1% of the floor, and A, which trades against P (n^0.016 is ~20% at 1e5), within 25%")
    assert float(v2["COST_WORKER_FIXED"]) == pytest.approx(W2["fixed"], rel=0.02)
    assert float(v2["COST_WORKER_P"]) == pytest.approx(W2["p"], abs=0.025)
    assert float(v2["COST_WORKER_D0"]) == pytest.approx(W2["d0"], abs=6)
    assert float(v2["COST_WORKER_A"]) == pytest.approx(W2["a"], rel=0.25)
    assert float(v2["COST_WORKER_GATE"]) == pytest.approx(W2["g"], rel=0.05)
    assert wl["sigma"] < 0.02, "exact data: residuals ~0"


def test_xc_v2_without_fixed_does_not_recover(tmp_path):
    sp, wps = _files(tmp_path, [_rows(W2, 6, fixed=W2["fixed"])])
    _, wf, v2f, _ = _fit(sp, wps, "--fixed")
    p, wl, v2, _ = _fit(sp, wps)
    print(f"without --fixed: worker line {wl}\n  v2 block {v2}\nwith --fixed: sigma {wf and wf['sigma']}")
    print("spec (brief): on data with a floor, the fit without --fixed must not recover the shape (its sigma is larger)")
    assert p.returncode == 0 and no_traceback(p) and wl and v2
    assert float(v2.get("COST_WORKER_FIXED", "0")) == 0, "no floor is fitted without --fixed"
    assert wl["sigma"] > wf["sigma"] + 0.02
    assert not (abs(float(v2["COST_WORKER_P"]) - W2["p"]) <= 0.02 and abs(float(v2["COST_WORKER_D0"]) - W2["d0"]) <= 5)


def test_xc_v2_fixed_is_the_worker_side_only(tmp_path):
    sp, wps = _files(tmp_path, [_rows(W2, 6, fixed=W2["fixed"])])
    _, _, v2a, la = _fit(sp, wps)
    _, _, v2b, lb = _fit(sp, wps, "--fixed")
    sa = [l for l in la if l["side"] == "shard" and l["threads"] == 2]
    sb = [l for l in lb if l["side"] == "shard" and l["threads"] == 2]
    print(f"shard line without --fixed: {sa[0]['raw'][:150] if sa else None}\nshard line with --fixed:    {sb[0]['raw'][:150] if sb else None}")
    print("spec (brief): --fixed fits a floor on the worker side only")
    assert sa and sb and sb[0]["fixed"] is None and sa[0]["raw"] == sb[0]["raw"]
    assert all(v2a[k] == v2b[k] for k in ("COST_A", "COST_P", "COST_D0", "COST_GATE"))


def test_xc_v2_drop_warmup_drops_the_first_point(tmp_path):
    rows = _rows(W2, 6, fixed=W2["fixed"], warmup=5.6)
    sp, wps = _files(tmp_path, [rows])
    print(f"synthetic worker with a 5.6 s warm-up on its first point ({rows[0]['size']} x {rows[0]['dim']}: {rows[0]['build_s']:.2f} s)")
    _, keep, v2k, _ = _fit(sp, wps, "--fixed")
    p, drop, v2d, _ = _fit(sp, wps, "--fixed", "--drop-warmup")
    print(f"--fixed: {keep and (keep['n'], keep['sigma'])}, block {v2k}\n--fixed --drop-warmup: {drop and (drop['n'], drop['sigma'])}, block {v2d}")
    print("spec (decision 14): --drop-warmup leaves out the worker side's first point per thread count")
    assert p.returncode == 0 and drop and keep
    assert drop["n"] == keep["n"] - 1 == len(rows) - 1
    assert drop["sigma"] < 0.02 < keep["sigma"]
    assert float(v2d["COST_WORKER_FIXED"]) == pytest.approx(W2["fixed"], rel=0.02)


def test_xc_v2_drop_warmup_drops_the_chronologically_first_point(tmp_path):
    """Two worker files from one process: the second file named holds the process's first point (earliest `at`)."""
    late = _rows(W2, 6, fixed=W2["fixed"], dataset="late", t0=600)
    early = _rows(W2, 6, fixed=W2["fixed"], dataset="early", t0=0)
    early = [dict(r, size=r["size"] + 1) for r in early]  # distinct points
    for r in early:
        n, d = r["size"], r["dim"]
        r["build_s"] = W2["fixed"] + W2["a"] * n ** W2["p"] * math.log(n) * (d + W2["d0"])
    early[0]["build_s"] += 5.6
    sp, wps = _files(tmp_path, [late, early])
    print(f"worker files in argument order: late (at 08:10..) then early (at 08:00.., its first point carries the 5.6 s warm-up)")
    p, wl, v2, _ = _fit(sp, wps, "--fixed", "--drop-warmup")
    print(f"worker line {wl}\nv2 block {v2}")
    print("spec (decision 14): the dropped point is 'the process's first point' (the one-off CUDA warm-up), i.e. the earliest "
          "measured; the CSVs record `at` per point")
    assert p.returncode == 0 and wl
    assert wl["sigma"] < 0.02, "the warm-up point was kept: the fit dropped the first row in argument order, not the first measured"


# ------------------------------------------------------------------ edges


def test_xc_v2_no_block_without_worker_csv(tmp_path):
    srows = _rows(S2, 2) + _rows(W2, 6, fixed=W2["fixed"])
    sp = xc.write_csv(tmp_path / "both.csv", srows)
    p = run_fit(sp, "--shard-threads", "2", "--worker-threads", "6", "--v1", "--v2", "--fixed")
    lines, v2 = parse(p.stdout)
    print(p.stdout[-600:] + p.stderr[-400:])
    print(f"spec (brief, --help): without --worker-csv, --v2 prints no v2 block. v2 block found: {v2}")
    assert no_traceback(p)
    assert v2 is None


def test_xc_v2_single_size_worker_csv_with_fixed_does_not_crash(tmp_path):
    wrows = [v1x.v1_point(50_000, 128, 6, 2.0 + 0.01 * r, 0.05, 500, 1.4e8, 1e8, 2.9e8, repeat=r) for r in range(3)]
    sp, wps = _files(tmp_path, [wrows])
    for extra in (("--fixed",), ("--fixed", "--drop-warmup")):
        p, wl, v2, _ = _fit(sp, wps, *extra)
        print(f"{extra}: exit {p.returncode}; worker line {wl and wl['raw']}; v2 {v2}; stderr {p.stderr.strip()[-300:]}")
        print("spec (brief): a single-size worker CSV with --fixed must not crash")
        assert no_traceback(p)
        assert p.returncode == 0 or len(p.stderr.strip().splitlines()) == 1, "either a fit or a one-line refusal"
    # one point only, which --drop-warmup removes
    sp, wps = _files(tmp_path, [wrows[:1]])
    p, wl, v2, _ = _fit(sp, wps, "--fixed", "--drop-warmup")
    print(f"one worker point and --drop-warmup: exit {p.returncode}; {p.stdout.strip()[-300:]} {p.stderr.strip()[-300:]}")
    assert no_traceback(p)
