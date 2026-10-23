"""p99 round completion against estimate-error sigma, one line per policy
(Phase 3 step 6, the roadmap's second plot: how fragile the cost model is,
and whether speculation covers for it).

    uv run python plot_sigma.py sigma.csv [more.csv ...] -o sigma.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotlib import group, load, pct, policies_in


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("-o", "--out", default="sigma.png")
    args = ap.parse_args()

    rows = load(args.csv)
    pols = policies_in(rows)
    sigmas = sorted({float(r["sigma"]) for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})
    g = group(rows, ("scenario", "policy", "sigma"))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 3.8), squeeze=False)
    for i, sc in enumerate(scenarios):
        ax = axes[0][i]
        for p in pols:
            ys = [pct(g.get((sc, p, fmt(s)), []), 0.99) for s in sigmas]
            ax.plot(sigmas, ys, "o-", label=p.replace("static_threshold:", "thresh "))
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel("estimate-error sigma (log-normal)")
        ax.set_ylabel("p99 round completion, modeled s")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Cost-model fragility: p99 round time vs estimate error")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}: sigmas {sigmas} x {len(pols)} policies over {len(rows)} rows")


def fmt(x: float) -> str:
    """Match the runner's FormatFloat(x, 'g', 10): 0 -> "0", 0.3 -> "0.3"."""
    s = f"{x:.10g}"
    return s


if __name__ == "__main__":
    main()
