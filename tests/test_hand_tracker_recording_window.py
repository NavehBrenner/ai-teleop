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
