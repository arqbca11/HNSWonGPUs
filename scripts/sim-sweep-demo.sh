#!/usr/bin/env bash
# Phase 3 steps 5-6 demo: a small seeded sweep through the discrete-event
# simulator, and the roadmap's two plots over it. Fully reproducible: the
# same seeds give byte-identical CSVs. The committed copies live in
# experiments/results/demo/; rerun this after changing the simulator, the
# policies or the cost model, and commit the diff.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/demo
mkdir -p "$OUT"

echo "== sweep: seven scenarios x five policies, 20 seeded rounds each"
go run ./experiments/cmd/sweep \
  -scenarios uniform,skewed,bimodal,downstream_heavy,staggered,high_dim,production \
  -policies all -rounds 20 -out "$OUT/rounds.csv"

echo "== sweep: shrinking pool (the busiest worker dies at t=5)"
go run ./experiments/cmd/sweep \
  -scenarios shrinking_pool -policies all -rounds 20 -kill_at 5 -out "$OUT/shrinking.csv"

echo "== sweep: estimate-error sigma 0 / 0.3 / 0.6 on skewed and high_dim"
go run ./experiments/cmd/sweep \
  -scenarios skewed,high_dim -policies all -sigma 0,0.3,0.6 -rounds 20 -out "$OUT/sigma.csv"

echo "== plots"
(cd experiments/analysis && uv sync -q && \
  uv run python plot_rounds.py ../results/demo/rounds.csv ../results/demo/shrinking.csv -o ../results/demo/rounds.png && \
  uv run python plot_sigma.py ../results/demo/sigma.csv -o ../results/demo/sigma.png)

echo "done: $OUT (rounds.csv, shrinking.csv, sigma.csv, rounds.png, sigma.png)"
