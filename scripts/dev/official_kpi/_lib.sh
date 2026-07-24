# Shared helpers for the official-KPI chunk scripts. SOURCED by each chunk, not run directly.
# Each chunk is self-contained, resumable (skips finished steps) and self-timing, so a session
# can run ONE chunk, read its per-step seconds, estimate the next, and decide.
set -u

LOGDIR=outputs/policy/official_kpi_logs
RUNS=outputs/policy/runs
EPISODES=${EPISODES:-1000}      # corpus size; lower for a smaller/faster run
ERR=${ERR:-0.4}                 # in-band operating point
EVAL_SEEDS=${EVAL_SEEDS:-100}   # paired eval seeds per checkpoint
STOP_BY=${STOP_BY:-}            # optional epoch-seconds: no NEW step starts at/after it
mkdir -p "$LOGDIR"

# run <name> <marker-that-exists-when-done> <cmd...>
run () {
  local name="$1" marker="$2"; shift 2
  if [ -e "$marker" ]; then echo "SKIP  $name (have $marker)"; return 0; fi
  if [ -n "$STOP_BY" ] && [ "$(date +%s)" -ge "$STOP_BY" ]; then
    echo "STOP  $name — past deadline $(date -d "@$STOP_BY" '+%F %H:%M'); not starting new work"
    return 9
  fi
  local t0; t0=$(date +%s)
  echo ">>>>  $name  ::  $*"
  if "$@" >"$LOGDIR/$name.log" 2>&1; then
    echo "DONE  $name  ($(( $(date +%s) - t0 ))s)"
  else
    echo "FAIL  $name  (see $LOGDIR/$name.log) — continuing"
  fi
}

# gen_corpus <dir> <extra gen args...> — idempotent (skips if metadata.json exists)
gen_corpus () {
  local dir="$1"; shift
  run "gen_$(basename "$dir")" "$dir/metadata.json" \
    uv run kvn gen --episodes "$EPISODES" --seed 100 --no-baseline --out "$dir" "$@"
}

# require_base <checkpoint-path> <hint> — fail a DAgger chunk early if its base is missing
require_base () {
  [ -e "$1" ] && return 0
  echo "MISSING base checkpoint: $1"; echo "   → run $2 first."; exit 2
}
