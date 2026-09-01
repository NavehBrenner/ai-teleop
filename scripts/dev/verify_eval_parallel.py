"""Verify parallel eval (`run_paired_batch`) == sequential, bit-identical.

Each paired trial is seed-determined ((master_seed, episode_index) — no shared RNG,
no order dependence), so fanning trials across processes must give identical KPIs.
Runs the same 4 seeds workers=1 vs workers=4 and asserts every KPI field matches;
also checks the `success` flags reproduce the banked seed-0 eval. Run from kevin/:

    uv run python scripts/dev/verify_eval_parallel.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.eval.ablation import TrialConfigSpec, run_paired_batch

CHECKPOINT = "outputs/policy/runs/official_ft_s0/checkpoint.pt"
BANKED_CSV = Path("runs/eval_official_ft_s0/trials.csv")  # the banked seed-0..99 eval
SEEDS = [0, 1, 2, 3]
ERROR_SCALE = 0.4  # official operating point (ft_plain.sh)


def _run(workers: int) -> list[dict[str, object]]:
    specs = [
        TrialConfigSpec(label="human_only", checkpoint=None),
        TrialConfigSpec(label="residual", checkpoint=CHECKPOINT, device="cuda"),
    ]
    batch = run_paired_batch(
        SEEDS, specs, workers=workers, master_seed=0, operator_error_scale=ERROR_SCALE
    )
    return [kpis.to_dict() for results in batch for kpis in results.values()]


def main() -> int:
    seq, par = _run(1), _run(4)
    assert len(seq) == len(par), f"row count {len(seq)} vs {len(par)}"
    for i, (s, p) in enumerate(zip(seq, par, strict=True)):
        assert s == p, f"trial row {i} differs:\n seq={s}\n par={p}"

    banked_note = "(banked eval absent, skipped)"
    if BANKED_CSV.exists():
        with BANKED_CSV.open() as handle:
            banked = {
                (int(r["seed"]), r["config_label"]): r["success"] for r in csv.DictReader(handle)
            }
        for row in par:
            key = (int(str(row["seed"])), str(row["config_label"]))
            if key in banked:
                got, want = str(row["success"]), banked[key]
                assert str(got).lower() == str(want).lower(), f"{key}: {got} vs banked {want}"
        banked_note = "and reproduces banked seed-0 successes"

    print(f"OK: parallel eval == sequential for {len(SEEDS)} seeds {banked_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
