#!/usr/bin/env bash
# The FULL policy comparison (Phase 3's "done when"), on the simulator step 7
# validated: the roadmap's eight scenarios x five policies at 50 seeds, then
# every sweep axis of the roadmap's table varied one at a time around the
# skewed and high_dim presets at 30 seeds. One CSV per sweep under
# experiments/results/full/ (committed; the plots beside them). ~15 minutes.
#
#   scripts/sim-sweep-full.sh            # everything
#   SEEDS=10 scripts/sim-sweep-full.sh   # a quick pass
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/full
SEEDS="${SEEDS:-50}"; AXIS_SEEDS="${AXIS_SEEDS:-30}"
mkdir -p "$OUT"
run() { echo "== $1"; shift; go run ./experiments/cmd/sweep -rounds "$AXIS_SEEDS" "$@" 2>&1 | tail -1; }
echo "== baseline: eight scenarios x five policies x $SEEDS seeds (link sharing on: the Phase 5 pool, decision 22)"
go run ./experiments/cmd/sweep -scenarios uniform,skewed,bimodal,downstream_heavy,staggered,high_dim,production,hour_build -policies all -rounds "$SEEDS" -out "$OUT/baseline.csv" 2>&1 | tail -1
go run ./experiments/cmd/sweep -scenarios shrinking_pool -policies all -rounds "$SEEDS" -kill_at 5 -out "$OUT/baseline-shrinking.csv" 2>&1 | tail -1
run "workers 1/2/4"               -scenarios skewed,high_dim -policies all -workers 1,2,4 -out "$OUT/axis-workers.csv"
run "workers per node 1/2"        -scenarios skewed,high_dim -policies all -workers 2 -workers_per_node 1,2 -out "$OUT/axis-workers-per-node.csv"
run "zipf 0.5/1/1.5"              -scenarios skewed,high_dim -policies all -zipf 0.5,1,1.5 -out "$OUT/axis-zipf.csv"
run "sigma 0/0.3/0.6"             -scenarios skewed,high_dim -policies all -sigma 0,0.3,0.6 -out "$OUT/axis-sigma.csv"
run "load 0.5/1/2"                -scenarios skewed,high_dim -policies all -load 0.5,1,2 -out "$OUT/axis-load.csv"
run "needs_index 0/0.5/1"         -scenarios skewed,high_dim -policies all -needs_index 0,0.5,1 -out "$OUT/axis-needs-index.csv"
run "shards 6/50"                 -scenarios skewed,high_dim -policies all -shards 50 -workers 3 -out "$OUT/axis-shards-50.csv"
run "dim 128/768/1536"            -scenarios skewed -policies all -dim 128,768,1536 -out "$OUT/axis-dim.csv"
run "speedup 10/40"               -scenarios skewed,high_dim -policies all -speedup 10,40 -out "$OUT/axis-speedup.csv"
echo "== plots"
(cd experiments/analysis && uv sync -q && \
  uv run python plot_rounds.py ../results/full/baseline.csv ../results/full/baseline-shrinking.csv -o ../results/full/baseline.png && \
  uv run python plot_sigma.py ../results/full/axis-sigma.csv -o ../results/full/sigma.png)
echo "done: $OUT"
