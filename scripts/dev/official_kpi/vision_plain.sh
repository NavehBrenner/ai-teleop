#!/usr/bin/env bash
# CHUNK 3/4 — Vision-plain × 3 seeds (frozen encoder, batch-2) + eval. THE LONG POLE.
# The vision corpus RENDERS (~10 s/episode measured) → 1000 ep ≈ 2.5-3 h before any training,
# then render-bound evals. Do NOT launch this unless it fits before the deadline. Lower EPISODES
# (e.g. EPISODES=400) if you need it to fit one slot.
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dev/official_kpi/_lib.sh
echo "===== vision_plain  start $(date '+%F %H:%M')  EPISODES=$EPISODES ====="

VIS=data/dataset_official_vision
gen_corpus "$VIS" --record all --render-every 20   # the render-bound step

for s in 0 1 2; do
  run "train_vis_s$s" "$RUNS/official_vis_s$s/checkpoint.pt" \
    uv run kvn train "$VIS" --vision --freeze-image-encoder --num-workers 4 --batch-size 2 \
      --seed "$s" --action-rate-weight 100 --name "official_vis_s$s"
  run "eval_vis_s$s" "runs/eval_official_vis_s$s/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --vision-checkpoint "$RUNS/official_vis_s$s/checkpoint.pt" --out-dir "runs/eval_official_vis_s$s"
done
echo "===== vision_plain  done  $(date '+%F %H:%M') ====="
