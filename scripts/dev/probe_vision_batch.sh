#!/usr/bin/env bash
# Find the max vision train batch size that fits the 8 GB GPU (frozen encoder).
# Runs 1 real epoch of `kvn train` per batch size on the existing 300-ep vision
# corpus; a batch that OOMs fails on its first step, one that fits finishes fast.
# Tries plain fp32 first, then --amp (mixed precision) which should push higher.
# Run from kevin/ on an idle GPU:  bash scripts/dev/probe_vision_batch.sh
cd "$(dirname "$0")/../.." || exit 1

VIS=data/dataset_vision
ROOT=outputs/policy/runs/_batch_probe
mkdir -p "$ROOT"

probe () {                       # probe <batch> <extra-flags...>
  local bs="$1"; shift
  local tag="bs${bs}${1:+_amp}"
  local log="$ROOT/$tag.log"
  timeout 420 uv run kvn train "$VIS" --vision --freeze-image-encoder \
    --num-workers 0 --batch-size "$bs" --epochs 1 --seed 0 \
    --runs-root "$ROOT" --name "$tag" "$@" >"$log" 2>&1
  local rc=$?
  if grep -qiE 'out of memory|CUDA error|cuda.*alloc' "$log"; then
    echo "  batch $bs ${1:+(amp)} → OOM"
  elif [ $rc -eq 124 ]; then
    echo "  batch $bs ${1:+(amp)} → FIT (timeout mid-epoch, no OOM)"
  elif [ $rc -eq 0 ]; then
    local vram; vram=$(grep -oiE 'peak.*(GB|MiB)|max_memory[^ ]*' "$log" | head -1)
    echo "  batch $bs ${1:+(amp)} → FIT (epoch done) $vram"
  else
    echo "  batch $bs ${1:+(amp)} → rc=$rc (see $log)"
  fi
}

echo "=== fp32 ==="
for bs in 8 16 32 64; do probe "$bs"; done
echo "=== amp (mixed precision) ==="
for bs in 32 64 128; do probe "$bs" --amp; done
echo "done $(date '+%H:%M:%S')"
