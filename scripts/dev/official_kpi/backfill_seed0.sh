#!/usr/bin/env bash
# Backfill rich KPIs for seed-0 vision-DAgger rounds 0-3.
# Seed 0 ran under old dagger code (no per-round trials.csv). Round 4's rich KPI
# already exists in runs/eval_official_dag_vis_s0/. This re-evals the earlier
# rounds' checkpoints on the SAME held-out seed-0 walls (--seeds 100, err 0.4).
# Resumable: skips a round whose out-dir already has trials.csv.
set -u
cd "$(dirname "$0")/../../.."   # -> kevin/
LOG=outputs/policy/official_kpi_logs
mkdir -p "$LOG"
for r in 0 1 2 3; do
  ckpt=outputs/policy/runs/dag_vis_s0/dagger_round${r}/checkpoint.pt
  out=runs/backfill_dag_vis_s0_r${r}
  if [ -e "$out/trials.csv" ]; then echo "SKIP  r$r (have $out/trials.csv)"; continue; fi
  echo ">>>>  backfill r$r :: $ckpt"
  t0=$(date +%s)
  if uv run kvn evaluate pair --seeds 100 --error-scale 0.4 \
       --vision-checkpoint "$ckpt" --out-dir "$out" \
       >"$LOG/backfill_s0_r$r.log" 2>&1; then
    echo "DONE  r$r  ($(( $(date +%s) - t0 ))s)"
  else
    echo "FAIL  r$r  (see $LOG/backfill_s0_r$r.log)"
  fi
done
echo "ALL DONE"
