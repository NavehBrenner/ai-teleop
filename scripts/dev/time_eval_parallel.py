"""Time sequential vs 8-way parallel eval throughput (clean-machine measurement).

Prints wall-seconds for an N-seed paired eval at workers=1 and workers=8, and the
speedup. Feeds the seeds-3/4 estimate. Run from kevin/ on an idle machine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.eval.ablation import TrialConfigSpec, run_paired_batch  # noqa: E402

CHECKPOINT = "outputs/policy/runs/official_ft_s0/checkpoint.pt"
N = 32
SPECS = [
    TrialConfigSpec(label="human_only", checkpoint=None),
    TrialConfigSpec(label="residual", checkpoint=CHECKPOINT, device="cuda"),
]


def _time(workers: int) -> float:
    start = time.time()
    run_paired_batch(
        list(range(N)), SPECS, workers=workers, master_seed=0, operator_error_scale=0.4
    )
    return time.time() - start


def main() -> int:
    seq = _time(1)
    par = _time(8)
    print(f"eval {N} seeds │ seq(w1)={seq:.0f}s  par(w8)={par:.0f}s  speedup={seq / par:.1f}x")
    print(f"  per-trial: seq={seq / N:.1f}s  par={par / N:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
