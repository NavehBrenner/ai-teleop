"""``--record-hand`` must capture the episode, not the centering that preceded it.

The sibling of ``test_hand_tracker_sensor_health.py``, for the same window and the same
reason. Startup centering (``calibrate_neutral``) runs *before the sim is stepped* and is
wall-clock timed — the operator has to find the cameras, present an open palm and hold it
still, with any drop restarting the hold — so it routinely outlasts a short episode.

Recording from construction produced a 2026-08-20 take whose video was 26.1 s long against
6.79 s of trajectory: roughly two-thirds of it was the tracker reporting ``no hand`` while
the robot sat motionless. That is bad footage, and worse, the clip's duration bears no
relation to the ``episode.npz`` it is meant to be cut beside — pairing them appears to show
the arm ignoring the hand for most of the run.

``StereoHandSource.__init__`` needs stereohand and a calibration file, so these build the
instance state directly; the gate is the whole contract under test.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pytest

from ai_teleop.input.hand_tracker import StereoHandSource

LOGGER = "ai_teleop.hand_tracker"


class _FakeReading:
    present = False
    landmarks = None


class _FakeTracker:
    """Stands in for stereohand's tracker, counting the frames it was asked to draw."""

    def __init__(self) -> None:
        self.render_calls = 0

    def render_step(self) -> np.ndarray:
        self.render_calls += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def poll(self) -> None:
        pass

    def read(self) -> _FakeReading:
        return _FakeReading()


def _source(*, record_path: str | None, recording: bool) -> StereoHandSource:
    """A StereoHandSource wired for `read()`'s window-pump branch and nothing else."""
    source = object.__new__(StereoHandSource)
    source._tracker = _FakeTracker()  # type: ignore[assignment]
    source._show_window = True
    source._last_pump = 0.0  # far enough in the past that the pump fires immediately
    source._record_path = record_path
    source._recording = recording
    source._writer = None
    source._reads = 0
    source._absent = 0
    source._fresh = 0
    source._first_read = 0.0
    source._prev_landmarks = None
    source._mark = None
    source._pumps = 0
    source._first_pump = 0.0
    source._record_started = 0.0
    source._recorded_frames = 0
    return source


def _written_frames(source: StereoHandSource, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture what `read()` hands to the video writer, without opening one."""
    written: list[Any] = []
    monkeypatch.setattr(
        StereoHandSource, "_write_frame", lambda _self, frame: written.append(frame)
    )
    source.read()
    return written


def test_unarmed_recording_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The centering phase: configured to record, not yet armed, so no frames land."""
    source = _source(record_path="hand.mp4", recording=False)
    assert _written_frames(source, monkeypatch) == []


def test_armed_recording_writes_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `start_recording()` the same call records."""
    source = _source(record_path="hand.mp4", recording=False)
    source.start_recording()
    assert len(_written_frames(source, monkeypatch)) == 1


def test_start_recording_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(record_path="hand.mp4", recording=False)
    source.start_recording()
    source.start_recording()
    assert len(_written_frames(source, monkeypatch)) == 1


def test_arming_without_a_record_path_still_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arming is not itself a request to record."""
    source = _source(record_path=None, recording=False)
    source.start_recording()
    assert _written_frames(source, monkeypatch) == []


def test_close_warns_when_recording_was_requested_but_never_armed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing file must be announced, not discovered later.

    Gating on an explicit call means a caller that forgets it gets no video; that is the
    one regression this design could introduce, so it is made loud.
    """
    source = _source(record_path="hand.mp4", recording=False)
    source._tracker = _FakeTrackerWithClose()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        source.close()

    assert any("never armed" in record.getMessage() for record in caplog.records)


def test_close_is_quiet_when_no_recording_was_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _source(record_path=None, recording=False)
    source._tracker = _FakeTrackerWithClose()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        source.close()

    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


class _FakeTrackerWithClose(_FakeTracker):
    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# The recording must be stamped with the rate it achieves, not the one it aims for
# --------------------------------------------------------------------------


def test_pump_rate_is_unmeasurable_before_enough_samples() -> None:
    """No evidence yet — the writer falls back to the nominal rate rather than guess."""
    source = _source(record_path="hand.mp4", recording=False)
    source._pumps = 5
    source._first_pump = time.monotonic() - 1.0
    assert source._measured_pump_fps() is None


def test_pump_rate_reflects_what_was_achieved(monkeypatch: pytest.MonkeyPatch) -> None:
    """60 pumps over 3 s is 20 fps, not the 30 the interval nominally allows.

    `_WINDOW_PUMP_INTERVAL_S` is a floor on the gap between pumps, not a promise: the
    pump fires from `read()`, so it slips whenever the control loop is busy. A take on
    2026-08-20 achieved 21.3 fps, was stamped 30, and played 1.41x quick.
    """
    source = _source(record_path="hand.mp4", recording=False)
    source._pumps = 61
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    source._first_pump = 97.0

    measured = source._measured_pump_fps()
    assert measured is not None
    assert measured == pytest.approx(20.0)


def test_measured_rate_never_exceeds_the_nominal_ceiling() -> None:
    """A sanity property of the gate: pumps can come in late, never early."""
    source = _source(record_path="hand.mp4", recording=False)
    source._pumps = 1000
    source._first_pump = time.monotonic() - 1000 * (1.0 / 30.0)
    measured = source._measured_pump_fps()
    assert measured is not None
    assert measured <= 30.0 + 1e-6


def test_close_warns_when_the_stamped_rate_does_not_match(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A file whose timestamps are a guess must say so — nothing downstream can tell."""
    source = _source(record_path="hand.mp4", recording=True)
    source._tracker = _FakeTrackerWithClose()  # type: ignore[assignment]
    source._writer = _FakeWriter()
    # Stamped from a 30 fps estimate, but only 21 frames landed in a second.
    source._pumps = 0  # too few to measure -> `_log_recording_rate` compares against nominal
    source._recorded_frames = 22
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    source._record_started = 99.0

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        source.close()

    assert any("will play" in record.getMessage() for record in caplog.records)


class _FakeWriter:
    def release(self) -> None:
        pass
