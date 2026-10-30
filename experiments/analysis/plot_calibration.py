"""Measured build time against the fitted cost model, per thread count, on a
log-log axis, with the residual ratio below — so the shape error of the
n·ln(n) curve is visible, not hidden in one constant (Phase 4 decision 23).

    uv run python plot_calibration.py ../results/calibration/laptop-sift-128-6.csv -o cal.png
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotlib import load  # noqa: E402


def fit_a(points):
    # the same log-space fit as kgpu_builder.calibrate: the geometric mean of t/x
    return math.exp(sum(math.log(p["t"] / p["x"]) for p in points) / len(points))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("-o", "--out", default="calibration.png")
    args = ap.parse_args()
    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"plot_calibration: no points in {' '.join(args.csv)}")
    series = defaultdict(list)
    for r in rows:
        n, d, t = int(r["size"]), int(r["dim"]), float(r["build_s"])
        series[(r["dataset"], int(r["threads"]))].append({"n": n, "x": n * math.log(n) * d, "t": t, "d": d})
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for (ds, th), pts in sorted(series.items()):
        pts.sort(key=lambda p: p["n"])
        a = fit_a(pts)
        xs = [p["n"] for p in pts]
        ax.plot(xs, [p["t"] for p in pts], "o-", label=f"{ds}, {th} threads: measured")
        ax.plot(xs, [a * p["x"] for p in pts], "--", label=f"{ds}, {th} threads: fit A={a:.3g}")
        rx.plot(xs, [p["t"] / (a * p["x"]) for p in pts], "o-", label=f"{ds}, {th} threads")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("build time (s)"); ax.set_title("hnswlib build time: measured vs A·n·ln(n)·dim")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
    rx.axhline(1.0, color="k", lw=0.8)
    rx.set_xlabel("vectors"); rx.set_ylabel("measured / fitted"); rx.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}: {len(rows)} points in {len(series)} series")


if __name__ == "__main__":
    main()
