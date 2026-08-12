"""The two pieces of the blinded human trial that would fail silently if wrong.

A broken schedule (unbalanced arms, or one that ignores its seed) and a broken assist
scale (0 that still corrects) both produce a session that *looks* fine and reports a
meaningless number. Neither is visible from the output, so both get a test.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dev"))

from blind_trial import (  # noqa: E402
    _OPERATOR_STATUS,
    BLOCK,
    _StatusLine,
    run_child,
    schedule,
)
from run_episode import _ScaledAssist  # noqa: E402

from ai_teleop.common import Command, Observation
from ai_teleop.domain import Delta


class _ConstantAssist:
    """Returns a fixed non-zero Δ, so scaling is observable."""

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        return Delta(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), 4.0)


class _VisionAssist(_ConstantAssist):
    """Stands in for a vision-conditioned checkpoint, which advertises `use_vision`."""

    use_vision = True


class _TerminalLike(io.StringIO):
    """A StringIO that claims to be a TTY, so the status line actually draws into it."""

    def isatty(self) -> bool:
        return True


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


def test_scaling_a_vision_policy_keeps_its_modality() -> None:
    """The scale-0 arm must still request wrist frames.

    `main` turns on the env's wrist capture from this flag. If the wrapper hides it, a
    vision checkpoint gets no frames on the control arm and raises on its first step —
    the blinded vision trial has no control arm at all.
    """
    assert _ScaledAssist(_VisionAssist(), 0.0).use_vision is True


def test_scaling_an_ft_policy_does_not_invent_a_modality() -> None:
    # An F/T-only provider has no `use_vision` at all; wrapping must not add one.
    assert _ScaledAssist(_ConstantAssist(), 0.0).use_vision is False


# Verbatim from a real `run_episode.py --policy tf` run against a vision checkpoint. The
# centering lines are what `calibrate_neutral` emits when its console is a pipe (captured by
# driving it with a fake hand source); the rest are the lines that would tell the operator
# which arm they are on.
_CENTERING_LOG_LINES = [
    "2026-08-04 16:13:53,997 INFO  [ai_teleop.vision_input] centering — hold an open palm "
    "still for 3s to set neutral",
    "2026-08-04 16:13:53,997 INFO  [ai_teleop.vision_input] centering in 3s — hold still",
    "2026-08-04 16:13:53,998 INFO  [ai_teleop.vision_input] centering reset — open palm lost "
    "for >0.30s; hold again",
    "2026-08-04 16:13:54,000 INFO  [ai_teleop.vision_input] centering complete — neutral set",
]
_ARM_REVEALING_LOG_LINES = [
    "15:58:20 INFO  [episode] vision checkpoint — enabling wrist capture every 20 ticks",
    "15:58:20 INFO  [episode] Running 400 steps with scripted input + tf policy (Ctrl-C)...",
    '  File "scripts/run_episode.py", line 156, in get_delta  # _ScaledAssist',
    "15:58:20 INFO  [episode] assist_scale=0.0",
]


def test_operator_whitelist_passes_every_centering_state() -> None:
    for line in _CENTERING_LOG_LINES:
        found = _OPERATOR_STATUS.search(line)
        assert found is not None, f"operator would not see: {line}"
        # The log prefix is stripped, so the rendered line reads as a status, not a log.
        assert found.group(1).startswith("centering")


def test_operator_whitelist_hides_anything_naming_the_arm() -> None:
    """The blind rests on this: no line that distinguishes the arms may reach the terminal.

    Includes a traceback frame, because an unhandled exception names `_ScaledAssist` on the
    scale-0 arm and `LearnedResidual` on the scale-1 arm — an arm leak on the one stream the
    centering spinner also uses, which is why the parent whitelists instead of un-redirecting.
    """
    for line in _ARM_REVEALING_LOG_LINES:
        assert _OPERATOR_STATUS.search(line) is None, f"operator would see: {line}"


def test_status_line_is_inert_when_stdout_is_not_a_terminal() -> None:
    # A piped session (CI, or `| tee`) must not get spinner escape codes in its output.
    stream = io.StringIO()  # StringIO.isatty() is False
    with _StatusLine(stream) as status:
        status.update("centering in 3s — hold still")
        status.update("centering complete — neutral set")
    assert stream.getvalue() == ""


def test_status_line_marks_completion_and_tells_the_operator_to_move() -> None:
    stream = _TerminalLike()
    with _StatusLine(stream) as status:
        status.update("centering complete — neutral set")
    written = stream.getvalue()
    assert "✓ centering complete — neutral set" in written
    assert "you can move now" in written


def test_run_child_logs_everything_but_shows_only_centering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end over a real subprocess: the log is complete, the terminal is not.

    Uses a stand-in child rather than `run_episode.py` so it runs without cameras or a
    robot, but exercises the actual pipe, encoding and whitelist that carry a live trial.
    """
    child = tmp_path / "fake_episode.py"
    child.write_text(
        "import sys\n"
        "for line in [\n"
        "    'INFO [episode] vision checkpoint — enabling wrist capture every 20 ticks',\n"
        "    'INFO [vision_input] centering in 1s — hold still',\n"
        "    'INFO [vision_input] centering complete — neutral set',\n"
        "    'INFO [episode] Running with tf policy, assist_scale=0.0',\n"
        "]:\n"
        "    print(line)\n",
        encoding="utf-8",
    )
    console_path = tmp_path / "console.log"

    assert run_child([sys.executable, str(child)], console_path) == 0

    logged = console_path.read_text(encoding="utf-8")
    assert "assist_scale=0.0" in logged  # the log keeps everything, including the arm
    assert "enabling wrist capture" in logged
    assert "centering complete" in logged

    shown = capsys.readouterr().out
    assert "assist_scale" not in shown  # ...the terminal never sees it
    assert "wrist capture" not in shown
    assert "tf policy" not in shown


def test_assist_scale_one_is_the_identity() -> None:
    inner = _ConstantAssist().get_delta(None, None)  # type: ignore[arg-type]
    scaled = _ScaledAssist(_ConstantAssist(), 1.0).get_delta(None, None)  # type: ignore[arg-type]
    assert np.array_equal(scaled.delta_position, inner.delta_position)
    assert np.array_equal(scaled.delta_orientation, inner.delta_orientation)
    assert scaled.delta_grip_force == inner.delta_grip_force
