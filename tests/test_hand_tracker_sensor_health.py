"""Sensor health must describe the episode, not the centering that preceded it.

The counters run from the first `read()`, which is during startup centering — a phase
that can outlast a short episode and in which the operator's hand is legitimately in and
out of frame. Left blended, a clean run reports figures that read as a sensor fault: a
2026-08-20 session logged **36% drop-out and 11.9 fresh fps** over a window that was
three-quarters centering, while the episode inside it released the clutch zero times and
a controlled probe on the same rig measured 0.48% drop-out.

`StereoHandSource.__init__` needs stereohand and a calibration file, so these build the
counter state directly — the windowing arithmetic is the whole contract under test.
"""

from __future__ import annotations

import logging

import pytest

from ai_teleop.input.hand_tracker import StereoHandSource

# get_logger namespaces under the package logger, so caplog must target the full name.
LOGGER = "ai_teleop.hand_tracker"


def _source(*, first_read: float, reads: int, absent: int, fresh: int) -> StereoHandSource:
    """A StereoHandSource carrying only its sensor-health counters."""
    source = object.__new__(StereoHandSource)
    source._first_read = first_read
    source._reads = reads
    source._absent = absent
    source._fresh = fresh
    source._mark = None
    return source


def _health_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if "sensor health" in record.getMessage()
    ]


def test_unmarked_reports_a_single_window(caplog: pytest.LogCaptureFixture) -> None:
    """Without a mark the old behaviour stands: one line, everything since first read."""
    source = _source(first_read=0.0, reads=100, absent=50, fresh=100)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        source._log_sensor_health()

    lines = _health_lines(caplog)
    assert len(lines) == 1
    assert "startup centering" not in lines[0]
    assert "50% drop-out" in lines[0]


def test_marked_window_excludes_the_centering_phase(caplog: pytest.LogCaptureFixture) -> None:
    """The regression this exists for: a clean episode behind a messy centering phase.

    Setup: 900 reads, 360 absent (40%). Episode: 100 reads, 1 absent (1%). Blended that
    is 36% — the figure that looked like a fault. The headline must report the episode.
    """
    source = _source(first_read=0.0, reads=1000, absent=361, fresh=500)
    source._mark = (30.0, 900, 360, 250)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        source._log_sensor_health()

    lines = _health_lines(caplog)
    assert len(lines) == 2

    episode = next(line for line in lines if "startup centering" not in line)
    assert "1% drop-out" in episode
    assert "100 reads" in episode
    assert "36% drop-out" not in episode

    centering = next(line for line in lines if "startup centering" in line)
    assert "40% drop-out" in centering
    assert "900 reads" in centering


def test_fresh_fps_is_measured_over_the_marked_window_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fresh-fps is deflated by the same blend, since absent reads never count as fresh.

    250 fresh over the episode's 10 s is 25 fps — the camera rate. Measured against the
    full 40 s window it would read 12.5 and look like the tracker had halved.
    """
    source = _source(first_read=0.0, reads=1000, absent=361, fresh=500)
    source._mark = (30.0, 900, 360, 250)

    monotonic_end = 40.0
    with caplog.at_level(logging.INFO, logger=LOGGER):
        StereoHandSource._log_window("sensor health", 30.0, monotonic_end, 100, 1, 250)

    line = _health_lines(caplog)[0]
    assert "25.0 fresh fps" in line


def test_mark_snapshots_rather_than_resets() -> None:
    """The setup phase stays reportable — the mark must not zero the counters."""
    source = _source(first_read=0.0, reads=900, absent=360, fresh=250)
    source.mark_measurement_start()

    assert source._reads == 900, "counters were reset instead of snapshotted"
    assert source._mark is not None
    _, mark_reads, mark_absent, mark_fresh = source._mark
    assert (mark_reads, mark_absent, mark_fresh) == (900, 360, 250)


def test_empty_window_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """A mark placed after the last read must not divide by zero."""
    source = _source(first_read=0.0, reads=900, absent=360, fresh=250)
    source._mark = (30.0, 900, 360, 250)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        source._log_sensor_health()

    lines = _health_lines(caplog)
    assert all("startup centering" in line for line in lines)
