#!/usr/bin/env bash
# CHUNK 1/4 — FT-plain × 5 seeds + eval. The headline; cheapest and most certain. No render.
# Est.: FT corpus ~20-30 min + 5×(train ~2-4 min + eval ~5-15 min). Run this first.
cd "$(dirname "$0")/../../.." || exit 1          # → kevin/
source scripts/dev/official_kpi/_lib.sh
echo "===== ft_plain  start $(date '+%F %H:%M')  EPISODES=$EPISODES ====="

FT=data/dataset_official_ft
gen_corpus "$FT" --record commands

for s in 0 1 2 3 4; do
  run "train_ft_s$s" "$RUNS/official_ft_s$s/checkpoint.pt" \
    uv run kvn train "$FT" --seed "$s" --action-rate-weight 100 --name "official_ft_s$s"
  run "eval_ft_s$s" "runs/eval_official_ft_s$s/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --residual-checkpoint "$RUNS/official_ft_s$s/checkpoint.pt" --out-dir "runs/eval_official_ft_s$s"
done
echo "===== ft_plain  done  $(date '+%F %H:%M') ====="
