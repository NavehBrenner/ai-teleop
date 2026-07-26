#!/usr/bin/env bash
# CHUNK 4/4 — Vision-DAgger × 3 + eval. THE MOST EXPENSIVE: renders during on-policy rollouts
# AND during eval. Needs vision_plain (corpus + base = official_vis_s0). Run last, only if it fits.
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dev/official_kpi/_lib.sh
echo "===== vision_dagger  start $(date '+%F %H:%M') ====="

VIS=data/dataset_official_vision
gen_corpus "$VIS" --record all --render-every 20   # no-op if vision_plain already made it
require_base "$RUNS/official_vis_s0/checkpoint.pt" "vision_plain.sh"

for i in 0 1 2; do
  rms=$((300 + i))
  run "dag_vis_s$i" "$RUNS/dag_vis_s$i/dagger_round4/checkpoint.pt" \
    uv run python scripts/dagger.py --base "$VIS" \
      --checkpoint "$RUNS/official_vis_s0/checkpoint.pt" \
      --aggregate "data/dagger_official_vis_s$i" --runs-root "$RUNS/dag_vis_s$i" \
      --rounds 5 --n-rollout 40 --rollout-master-seed "$rms" --render-every 20 \
      --action-rate-weight 100 --eval-seeds "$EVAL_SEEDS" --error-scale "$ERR"
  run "eval_dag_vis_s$i" "runs/eval_official_dag_vis_s$i/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --vision-checkpoint "$RUNS/dag_vis_s$i/dagger_round4/checkpoint.pt" \
      --out-dir "runs/eval_official_dag_vis_s$i"
done
echo "===== vision_dagger  done  $(date '+%F %H:%M') ====="
