#!/usr/bin/env bash
# FT-plain × 5 seeds at BATCH SIZE 2 — de-confounds the plain-vs-DAgger comparison.
# Corpus already exists; this only trains + evals. No render.
cd "$(dirname "$0")/../../.." || exit 1          # → kevin/
source scripts/dev/official_kpi/_lib.sh
echo "===== ft_plain_b2  start $(date '+%F %H:%M') ====="

FT=data/dataset_official_ft
[ -e "$FT/metadata.json" ] || { echo "MISSING corpus $FT — abort"; exit 2; }

for s in 0 1 2 3 4; do
  run "train_ft_b2_s$s" "$RUNS/official_ft_b2_s$s/checkpoint.pt" \
    uv run kvn train "$FT" --seed "$s" --action-rate-weight 100 --batch-size 2 \
      --name "official_ft_b2_s$s"
  run "eval_ft_b2_s$s" "runs/eval_official_ft_b2_s$s/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --residual-checkpoint "$RUNS/official_ft_b2_s$s/checkpoint.pt" \
      --out-dir "runs/eval_official_ft_b2_s$s"
done
echo "===== ft_plain_b2  done  $(date '+%F %H:%M') ====="
