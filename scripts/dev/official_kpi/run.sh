#!/usr/bin/env bash
# Official KPI run — the seeded, distribution-reported checkpoints the project is submitted against.
#
# Recipe (decided with Naveh, 2026-07-25):
#   FT residual  : 5 training seeds (0..4)          × {plain, DAgger}
#   Vision       : 3 training seeds (0..2)          × {plain, DAgger}   (frozen encoder — 8.6 GB GPU)
#   every checkpoint evaluated at 100 paired eval seeds, error_scale 0.4, and reported as a DISTRIBUTION.
#
# Design: RESUMABLE and FAIL-SOFT. Each stage skips if its output already exists, so this survives
# interruption and can be re-launched across several overnight slots. One bad stage logs and the
# next independent stage still runs. Front-loads the certain, cheap, high-value FT headline first;
# the render-bound vision arms come after; DAgger (expensive) last.
#
# Launch (from kevin/):   bash scripts/dev/official_kpi/run.sh   (nohup / background for overnight)
# Watch:                  tail -f outputs/policy/official_kpi_logs/*.log
#
# ponytail: pure orchestration of existing commands (kvn gen/train/evaluate, scripts/dagger.py) —
# no new training/eval infra. Tunables are the two vars below.
set -u

EPISODES=1000            # corpus size ("REAL scale"); drop to ~400 if the vision render/eval overruns
ERR=0.4                  # in-band operating point (chamfer-contact band)
EVAL_SEEDS=100           # paired eval seeds per checkpoint — the power the distribution needs
FT_SEEDS="0 1 2 3 4"     # 5 training seeds for FT
VIS_SEEDS="0 1 2"        # 3 training seeds for vision
LOGDIR=outputs/policy/official_kpi_logs
RUNS=outputs/policy/runs
mkdir -p "$LOGDIR"

# run <logname> <marker-path-that-exists-when-done> <cmd...>
# Skips if the marker exists; else runs, logging to $LOGDIR/<logname>.log.
run () {
  local name="$1" marker="$2"; shift 2
  if [ -e "$marker" ]; then echo "SKIP  $name (have $marker)"; return 0; fi
  echo ">>>>  $name  ::  $*"
  if "$@" >"$LOGDIR/$name.log" 2>&1; then
    echo "DONE  $name"
  else
    echo "FAIL  $name (see $LOGDIR/$name.log) — continuing"
  fi
}

echo "===== Official KPI run  $(cat /proc/loadavg | cut -d' ' -f1) ====="

# ---- Stage 1: FT corpus (no images → fast) ---------------------------------------------------
FT_CORPUS=data/dataset_official_ft
run gen_ft "$FT_CORPUS/metadata.json" \
  uv run kvn gen --episodes "$EPISODES" --seed 100 --record commands --no-baseline --out "$FT_CORPUS"

# ---- Stage 2: FT-plain × 5 seeds + eval (the headline — front-loaded) ------------------------
for s in $FT_SEEDS; do
  run "train_ft_s$s" "$RUNS/official_ft_s$s/checkpoint.pt" \
    uv run kvn train "$FT_CORPUS" --seed "$s" --action-rate-weight 100 --name "official_ft_s$s"
  run "eval_ft_s$s" "runs/eval_official_ft_s$s/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --residual-checkpoint "$RUNS/official_ft_s$s/checkpoint.pt" --out-dir "runs/eval_official_ft_s$s"
done

# ---- Stage 3: Vision corpus (images → render-bound, the long pole) ---------------------------
VIS_CORPUS=data/dataset_official_vision
run gen_vision "$VIS_CORPUS/metadata.json" \
  uv run kvn gen --episodes "$EPISODES" --seed 100 --record all --render-every 20 --no-baseline --out "$VIS_CORPUS"

# ---- Stage 4: Vision-plain × 3 seeds (frozen encoder, batch-2) + eval ------------------------
for s in $VIS_SEEDS; do
  run "train_vis_s$s" "$RUNS/official_vis_s$s/checkpoint.pt" \
    uv run kvn train "$VIS_CORPUS" --vision --freeze-image-encoder --num-workers 4 --batch-size 2 \
      --seed "$s" --action-rate-weight 100 --name "official_vis_s$s"
  run "eval_vis_s$s" "runs/eval_official_vis_s$s/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --vision-checkpoint "$RUNS/official_vis_s$s/checkpoint.pt" --out-dir "runs/eval_official_vis_s$s"
done

# ---- Stage 5: FT-DAgger × 5 (rollout-seed axis; base = ft_s0) + final-round eval -------------
# dagger.py evals internally each round, but we re-eval the final round via the same harness for a
# comparable trials.csv. --rounds 3 → round dirs 0,1,2; the final is dagger_round2.
for i in $FT_SEEDS; do
  rms=$((200 + i))
  run "dag_ft_s$i" "$RUNS/dag_ft_s$i/dagger_round2/checkpoint.pt" \
    uv run python scripts/dagger.py --base "$FT_CORPUS" \
      --checkpoint "$RUNS/official_ft_s0/checkpoint.pt" \
      --aggregate "data/dagger_official_ft_s$i" --runs-root "$RUNS/dag_ft_s$i" \
      --rounds 3 --n-rollout 40 --rollout-master-seed "$rms" \
      --action-rate-weight 100 --eval-seeds "$EVAL_SEEDS" --error-scale "$ERR"
  run "eval_dag_ft_s$i" "runs/eval_official_dag_ft_s$i/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --residual-checkpoint "$RUNS/dag_ft_s$i/dagger_round2/checkpoint.pt" \
      --out-dir "runs/eval_official_dag_ft_s$i"
done

# ---- Stage 6: Vision-DAgger × 3 (most expensive — renders during rollouts) -------------------
for i in $VIS_SEEDS; do
  rms=$((300 + i))
  run "dag_vis_s$i" "$RUNS/dag_vis_s$i/dagger_round2/checkpoint.pt" \
    uv run python scripts/dagger.py --base "$VIS_CORPUS" \
      --checkpoint "$RUNS/official_vis_s0/checkpoint.pt" \
      --aggregate "data/dagger_official_vis_s$i" --runs-root "$RUNS/dag_vis_s$i" \
      --rounds 3 --n-rollout 40 --rollout-master-seed "$rms" --render-every 20 \
      --action-rate-weight 100 --eval-seeds "$EVAL_SEEDS" --error-scale "$ERR"
  run "eval_dag_vis_s$i" "runs/eval_official_dag_vis_s$i/trials.csv" \
    uv run kvn evaluate pair --seeds "$EVAL_SEEDS" --error-scale "$ERR" \
      --vision-checkpoint "$RUNS/dag_vis_s$i/dagger_round2/checkpoint.pt" \
      --out-dir "runs/eval_official_dag_vis_s$i"
done

# ---- Stage 7: aggregate every eval into per-recipe distributions -----------------------------
echo ">>>>  aggregate"
uv run python scripts/dev/official_kpi/aggregate.py | tee "$LOGDIR/aggregate.txt"
echo "===== done. Distributions in $LOGDIR/aggregate.txt ====="
