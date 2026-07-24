#!/usr/bin/env bash
# CHUNK 2/4 — FT-DAgger × 5 (rollout-seed axis) + eval. Needs ft_plain (base = official_ft_s0).
# Each run = 3 DAgger rounds (rollout gen + retrain) → more expensive than ft_plain. No render.
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dev/official_kpi/_lib.sh
echo "===== ft_dagger  start $(date '+%F %H:%M') ====="

FT=data/dataset_official_ft
gen_corpus "$FT" --record commands                 # no-op if ft_plain already made it
require_base "$RUNS/official_ft_s0/checkpoint.pt" "ft_plain.sh"

for i in 0 1 2 3 4; do
  rms=$((200 + i))                                 # rollout-master-seed = the DAgger seed axis
  run "dag_ft_s$i" "$RUNS/dag_ft_s$i/dagger_round2/checkpoint.pt" \
    uv run python scripts/dagger.py --base "$FT" \
      --checkpoint "$RUNS/official_ft_s0/checkpoint.pt" \
      --aggregate "data/dagger_official_ft_s$i" --runs-root "$RUNS/dag_ft_s$i" \
      --rounds 3 --n-rollout 40 --rollout-master-seed "$rms" \
      --action-rate-weight 100 --eval-seeds "$EVAL_SEEDS" --error-scale "$ERR"
  run "eval_dag_ft_s$i" "runs/eval_official_dag_ft_s$i/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --residual-checkpoint "$RUNS/dag_ft_s$i/dagger_round2/checkpoint.pt" \
      --out-dir "runs/eval_official_dag_ft_s$i"
done
echo "===== ft_dagger  done  $(date '+%F %H:%M') ====="
