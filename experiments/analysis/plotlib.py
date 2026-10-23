"""Shared bits for the sweep plots: CSV loading and the project's percentile.

The percentile matches the Go side (timeline.pct and the runner):
sorted[int(p * (n-1))] — so a number in a plot can be traced to the exact
CSV rows that produced it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

# The policies in a fixed, roadmap order, so every figure's legend reads the same.
POLICY_ORDER = ["always_gpu", "static_threshold:200000", "reactive", "reactive_spec", "clairvoyant"]


def load(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with Path(p).open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[int(p * (len(s) - 1))]


def group(rows: list[dict[str, str]], keys: tuple[str, ...], value: str = "round_s") -> dict[tuple, list[float]]:
    out: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("completed") != "true":
            continue  # an incomplete round has no honest round time
        out[tuple(r[k] for k in keys)].append(float(r[value]))
    return out


def policies_in(rows: list[dict[str, str]]) -> list[str]:
    seen = {r["policy"] for r in rows}
    ordered = [p for p in POLICY_ORDER if p in seen]
    return ordered + sorted(seen - set(ordered))
