"""Verify parallel DAgger rollout gen == sequential (LAB parallelization).

Rollouts are seed-determined (``_human_seed`` / ``episode_wall_seed`` — no shared
RNG, no order dependence), so fanning them across processes must produce
bit-identical episodes. This runs the *same* round-0 rollouts three ways and
asserts equality of every saved array:

  1. sequential (workers=1) — the original code path
  2. parallel   (workers=4) — the new path
  3. the banked seed-0 corpus on disk (the real official run's round 0)

Fast: 4 rollouts, no retrain/eval. Run from kevin/:
    uv run python scripts/dev/verify_dagger_parallel.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.dagger import _run_round_rollouts
from ai_teleop.data.trajectory import episode_npz_path

BASE = Path("data/dataset_official_ft")
CHECKPOINT = Path("outputs/policy/runs/official_ft_s0/checkpoint.pt")
BANKED = Path("data/dagger_official_ft_s0/runs")  # seed-0 aggregate (rollout_master_seed 200)
MASTER_SEED = 200  # ft_dagger.sh: seed i=0 -> rms 200
N = 4  # first four round-0 rollouts is plenty to catch a divergence
DEVICE = "cuda"


def _load(runs_dir: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(episode_npz_path(runs_dir, index)) as data:
        return {key: data[key] for key in data.files}


def _assert_same(label: str, left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert left.keys() == right.keys(), f"{label}: key mismatch {left.keys()} vs {right.keys()}"
    for key, value in left.items():
        assert np.array_equal(value, right[key]), f"{label}: array '{key}' differs"


def _run(tag: str, workers: int) -> Path:
    config = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))["config"]
    runs_dir = Path(f"outputs/policy/runs/_verify_parallel_{tag}/runs")
    if runs_dir.parent.exists():
        shutil.rmtree(runs_dir.parent)
    runs_dir.mkdir(parents=True)
    _run_round_rollouts(
        checkpoint=CHECKPOINT,
        config=config,
        runs_dir=runs_dir,
        round_index=0,
        n_rollout=N,
        master_seed=MASTER_SEED,
        render_every=None,  # F/T base — no wrist capture
        device=DEVICE,
        workers=workers,
    )
    return runs_dir


def main() -> int:
    seq = _run("seq", workers=1)
    par = _run("par", workers=4)

    for i in range(N):
        index = 1_000_000 + i  # dagger_episode_index(round 0, rollout i)
        s, p = _load(seq, index), _load(par, index)
        _assert_same(f"seq-vs-par rollout {i}", s, p)
        if BANKED.exists():
            _assert_same(f"par-vs-banked rollout {i}", p, _load(BANKED, index))

    banked = (
        "and reproduces banked seed-0" if BANKED.exists() else "(banked seed-0 absent, skipped)"
    )
    print(f"OK: parallel == sequential for {N} rollouts {banked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
