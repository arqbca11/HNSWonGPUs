"""p50 and p99 round completion time per policy, one panel per scenario
(Phase 3 step 6, the roadmap's first plot).

    uv run python plot_rounds.py rounds.csv [more.csv ...] -o rounds.png
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
    ap.add_argument("-o", "--out", default="rounds.png")
    args = ap.parse_args()

    rows = load(args.csv)
    pols = policies_in(rows)
    scenarios = sorted({r["scenario"] for r in rows})
    g = group(rows, ("scenario", "policy"))

    ncols = min(4, len(scenarios))
    nrows = (len(scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for i, sc in enumerate(scenarios):
        ax = axes[i // ncols][i % ncols]
        xs = range(len(pols))
        p50 = [pct(g.get((sc, p), []), 0.5) for p in pols]
        p99 = [pct(g.get((sc, p), []), 0.99) for p in pols]
        n = [len(g.get((sc, p), [])) for p in pols]
        ax.bar(xs, p50, width=0.6, label="p50", color="#4477aa")
        ax.plot(xs, p99, "v", color="#ee6677", markersize=9, label="p99")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([p.replace("static_threshold:", "thresh ") for p in pols], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{sc}  (n={max(n) if n else 0} rounds)", fontsize=10)
        ax.set_ylabel("round completion, modeled s")
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(scenarios), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Round completion time per policy (bar p50, marker p99)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}: {len(scenarios)} scenarios x {len(pols)} policies over {len(rows)} rows")


if __name__ == "__main__":
    main()
