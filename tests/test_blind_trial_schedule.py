"""The two pieces of the blinded human trial that would fail silently if wrong.

A broken schedule (unbalanced arms, or one that ignores its seed) and a broken assist
scale (0 that still corrects) both produce a session that *looks* fine and reports a
meaningless number. Neither is visible from the output, so both get a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dev"))

from blind_trial import BLOCK, schedule  # noqa: E402
from run_episode import _ScaledAssist  # noqa: E402

from ai_teleop.common import Command, Observation
from ai_teleop.domain import Delta


class _ConstantAssist:
    """Returns a fixed non-zero Δ, so scaling is observable."""

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        return Delta(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 4.0)


def test_every_block_is_balanced() -> None:
    assignments = schedule(60, seed=1)
    assert len(assignments) == 60
    for start in range(0, 60, BLOCK):
        block = assignments[start : start + BLOCK]
        assert sum(block) == len(block) // 2, f"block at {start} is unbalanced: {block}"


def test_partial_final_block_stays_within_one_trial() -> None:
    # 10 trials = two full blocks + a half block; the arms must not drift apart.
    assignments = schedule(10, seed=7)
    assert abs(sum(assignments) - sum(not a for a in assignments)) <= 1


def test_schedule_depends_on_its_seed() -> None:
    assert schedule(40, seed=1) == schedule(40, seed=1)  # reproducible
    assert schedule(40, seed=1) != schedule(40, seed=2)  # and actually seeded


def test_assist_scale_zero_zeroes_every_channel() -> None:
    delta = _ScaledAssist(_ConstantAssist(), 0.0).get_delta(None, None)  # type: ignore[arg-type]
    assert not np.any(delta.delta_position)
    assert not np.any(delta.delta_orientation)
    assert delta.delta_grip_force == 0.0


def test_assist_scale_one_is_the_identity() -> None:
    inner = _ConstantAssist().get_delta(None, None)  # type: ignore[arg-type]
    scaled = _ScaledAssist(_ConstantAssist(), 1.0).get_delta(None, None)  # type: ignore[arg-type]
    assert np.array_equal(scaled.delta_position, inner.delta_position)
    assert np.array_equal(scaled.delta_orientation, inner.delta_orientation)
    assert scaled.delta_grip_force == inner.delta_grip_force
