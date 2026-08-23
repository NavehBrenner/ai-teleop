"""Stereo hand sensor — metric 3-D landmarks → hand-pose readings.

The off-the-shelf **sensor** layer per `docs/design-document.md` §4.4: two calibrated webcams
feed the :mod:`stereohand` package, which triangulates each frame into 21 *metric*
3-D landmarks, and we distill those into a small typed :class:`HandReading` (wrist
position in metres, an orientation estimate, an open/close grip proxy, and a
``present`` flag). It is deliberately *pure sensing*: no robot, no
:class:`Command`, no calibration, no clutch — all of that lives one layer up in
:class:`ai_teleop.input.vision_input.VisionInput`.

Two halves:

- :func:`reading_from_landmarks` — the deterministic landmark→reading math. No
  camera, no stereohand import; this is what the unit tests exercise with
  synthetic landmark sets.
- :class:`StereoHandSource` — the live path. It adapts
  :class:`stereohand.StereoHandTracker` (capture + MediaPipe-per-view +
  triangulation on a background thread) to the non-blocking ``read() ->
  HandReading`` seam, returning ``present=False`` when the hand is missing in
  either view (never raises mid-stream).

``stereohand`` is imported lazily inside the live class so the pure function (and
its tests) work without the ``stereo-input`` extra installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from ai_teleop.common.geometry import mat3_to_quat
from ai_teleop.common.log import get_logger

log = get_logger("hand_tracker")

# MediaPipe Hands landmark indices we use (of the 21-point model).
_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_MIDDLE_FINGERTIP = 12
_PINKY_MCP = 17
_FINGERTIPS = (8, 12, 16, 20)  # index, middle, ring, pinky tips (thumb excluded)
_FINGER_MCPS = (5, 9, 13, 17)  # their knuckles (for the open-palm recenter test)

# Empirical open/close bounds for the fingertip-spread ratio (tip→wrist distance
# over hand scale). Fist ≈ 1.0, flat open hand ≈ 2.4 — hand-tuned; recalibrate
# from operator feel if the grip proxy reads hot or cold.
_GRIP_RATIO_CLOSED = 1.0
_GRIP_RATIO_OPEN = 2.4

# The cv2 preview only needs ~30 Hz, and its imshow/waitKey is an expensive WSLg round-trip
# on the main thread that steals CPU/GIL from the tracking thread. Throttle by *wall-clock
# time*, not by a read-count stride: read() is called at wildly different rates (≈200 Hz in the
# centering loop, ≈100-160 Hz in the render-paced control loop), so a fixed stride yields ~10 fps
# in some phases and starves tracking in others. A time gate gives a stable preview rate
# regardless of caller cadence. The hand reading itself is taken every call (cheap — it just
# returns the background thread's latest).
_WINDOW_PUMP_INTERVAL_S = 1.0 / 30.0


@dataclass(frozen=True)
class HandReading:
    """One frame of hand sensing, in the metric stereo-rig (left-camera) frame.

    Lives here (not in ``common/``) because only the input layer consumes it.

    Attributes
    ----------
    position:
        Shape (3,) — the wrist landmark's true metric xyz (metres) in the
        rectified left-camera frame: x right, y down, z = depth (away from the
        camera). Real triangulated depth, no proxy. The strategy layer maps this
        to robot workspace.
    orientation:
        Shape (4,) unit quaternion (w, x, y, z) — a hand-frame estimate fit to the
        3-D palm landmarks. Observable from two views; the strategy may still filter it.
    open_close:
        Grip proxy in [0, 1]: 0 = closed fist, 1 = flat open hand.
    present:
        False when no hand was detected this frame (drop-out).
    point_direction:
        Shape (2,) — the in-plane direction the hand points (wrist → middle
        fingertip), in the camera xy-plane (x right, y down), scaled by hand size
        so its *magnitude* shrinks as the hand angles into the camera
        (foreshortening). Position-independent: it's *where the hand points*, not
        where it sits. Drives in-plane steering in ``rate`` mode; the shrinking
        magnitude blends in-plane motion out as the forward component takes over.
    forwardness:
        How much the hand points *into* the camera (≈ "forward"): ~0 when pointing
        across the image plane, larger as the fingertips angle toward the lens.
        From the fingertips' depth (negative z = toward camera). The strategy
        dead-zones it and drives a gentle forward creep.
    recenter_pose:
        True when this frame is an open palm held square to the camera — the pose
        the startup centering (:func:`ai_teleop.input.calibrate_neutral`) requires the
        operator to hold still to set the neutral anchor. The hold timing lives in the
        calibration routine; this is just the per-frame pose test.

    The ``rate`` mode reads the open/close grip to pick a gesture — an open hand
    steers (+ creeps forward), a fist drives back — so it is robust where a
    finger-extension test fails: ``open_close`` is built from 3-D landmark
    distances and stays "open" even when the hand foreshortens toward the camera.
    """

    position: np.ndarray
    orientation: np.ndarray
    open_close: float
    present: bool
    point_direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    forwardness: float = 0.0
    recenter_pose: bool = False


_ABSENT = HandReading(
    position=np.zeros(3),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    open_close=0.0,
    present=False,
)


def reading_from_landmarks(landmarks: np.ndarray) -> HandReading:
    """Convert a (21, 3) metric hand landmark array to a :class:`HandReading`.

    Pure and camera-free — the unit-tested core. ``landmarks`` is the hand's 21
    points as real metric (x, y, z) rows in the rectified left-camera frame (the
    triangulated output of :mod:`stereohand`). ``position`` is the true wrist xyz;
    the derived signals (grip, pointing, orientation) are scale-invariant ratios.
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[_WRIST]

    # Apparent hand size (wrist→middle-finger MCP) — normalizes the grip ratio and
    # the pointing vector so both are distance-invariant.
    hand_scale = float(np.linalg.norm(landmarks[_MIDDLE_MCP] - wrist))
    position = wrist[:3].copy()
    if hand_scale < 1e-6:
        return HandReading(position, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, True)

    tip_spread = float(np.mean([np.linalg.norm(landmarks[t] - wrist) for t in _FINGERTIPS]))
    ratio = tip_spread / hand_scale
    open_close = (ratio - _GRIP_RATIO_CLOSED) / (_GRIP_RATIO_OPEN - _GRIP_RATIO_CLOSED)
    open_close = float(np.clip(open_close, 0.0, 1.0))

    orientation = _palm_orientation(landmarks)

    # In-plane pointing: wrist → middle fingertip, scaled by hand size so the
    # vector is distance-invariant and *shrinks* as the hand angles into the
    # camera (the fingertip foreshortens toward the wrist). Kept un-normalized on
    # purpose: the magnitude blends in-plane steering out as forwardness rises.
    point_direction = (landmarks[_MIDDLE_FINGERTIP, :2] - wrist[:2]) / hand_scale
    # Forwardness: fingertips angled toward the camera ⇒ negative tip z ⇒ positive.
    # Wrist-relative, to remove the camera-frame depth offset.
    forwardness = -float(np.mean(landmarks[list(_FINGERTIPS), 2] - wrist[2]))

    return HandReading(
        position,
        orientation,
        open_close,
        present=True,
        point_direction=point_direction,
        forwardness=forwardness,
        recenter_pose=_palm_open_facing(landmarks),
    )


def _palm_open_facing(landmarks: np.ndarray) -> bool:
    """True when the hand is open and roughly square to the camera (the recenter pose).

    Ported from stereohand's renderer: ≥3 fingers extended (tip→wrist clearly longer
    than knuckle→wrist) and the palm-plane normal roughly aligned with the camera's
    z-axis (the squareness test reads metric landmark depth).
    """
    wrist = landmarks[_WRIST]
    extended = sum(
        np.linalg.norm(landmarks[tip] - wrist) > 1.4 * np.linalg.norm(landmarks[mcp] - wrist)
        for tip, mcp in zip(_FINGERTIPS, _FINGER_MCPS, strict=True)
    )
    if extended < 3:
        return False
    normal = np.cross(landmarks[_INDEX_MCP] - wrist, landmarks[_PINKY_MCP] - wrist)
    norm = float(np.linalg.norm(normal))
    return norm > 0 and abs(float(normal[2])) > 0.7 * norm


def _palm_orientation(landmarks: np.ndarray) -> np.ndarray:
    """Estimate a hand-frame quaternion from the palm landmarks.

    Builds an orthonormal frame: +y points wrist→middle-MCP (finger direction),
    +x points across the knuckles (index→pinky MCP), +z is the palm normal. Rough
    — adequate for the coarse teleop signal, and the strategy filters/ignores it.
    """
    wrist = landmarks[_WRIST]
    forward = landmarks[_MIDDLE_MCP] - wrist
    across = landmarks[_PINKY_MCP] - landmarks[_INDEX_MCP]

    y_axis = forward / (np.linalg.norm(forward) + 1e-9)
    z_axis = np.cross(across, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_axis = z_axis / z_norm
    x_axis = np.cross(y_axis, z_axis)

    quat = mat3_to_quat(np.column_stack([x_axis, y_axis, z_axis]))
    mujoco.mju_normalize4(quat)
    return quat


class StereoHandSource:
    """Two-webcam stereo tracker → metric :class:`HandReading` (the live sensor).

    Adapts :class:`stereohand.StereoHandTracker` (which triangulates metric
    ``(21, 3)`` landmarks from two calibrated webcams) to the non-blocking
    ``read() -> HandReading`` seam by running :func:`reading_from_landmarks` on the
    triangulated landmarks — so :class:`~ai_teleop.input.vision_input.VisionInput`
    gets real metric depth and an observable orientation (enable
    ``track_orientation`` for true 6-DoF mirroring).

    ``stereohand`` is imported lazily (the ``stereo-input`` extra) so the pure
    landmark math and its tests don't need it installed.
    """

    def __init__(
        self,
        calibration_path: str,
        *,
        left: int | str = 0,
        right: int | str = 2,
        show_window: bool = False,
        max_fps: int | Literal["cam"] = "cam",
        max_skew_s: float = 0.02,
        record_path: str | None = None,
    ) -> None:
        from stereohand import RenderConfig, StereoCalibration, StereoHandTracker

        calibration = StereoCalibration.load(calibration_path)
        self._show_window = show_window
        self._last_pump = 0.0  # wall-clock of the last cv2 window pump (see read())
        # Operator-side demo footage: the composite frame render_step() already draws
        # (both camera feeds + the 3-D skeleton), written straight to a video file. The
        # robot side is *not* captured here — render it offline from the recorded episode
        # (scripts/dev/render_trajectory.py), which costs the live loop nothing.
        self._record_path = record_path
        self._writer: Any = None
        # Recording is *armed* separately from being configured — see `start_recording()`.
        # Startup centering runs before the sim is stepped and routinely outlasts a short
        # episode, so capturing from construction puts minutes of an operator waving at the
        # cameras ahead of the few seconds anyone wants to watch.
        self._recording = False
        # Pump bookkeeping, used to stamp the recording with the frame rate it actually
        # achieves rather than the one it aims for -- see `_measured_pump_fps`.
        self._pumps = 0
        self._first_pump = 0.0
        self._record_started = 0.0
        self._recorded_frames = 0
        # Sensor-health counters (logged on close). The control loop polls read() far faster
        # than the cameras produce frames, so what matters for teleop feel is the *effective*
        # rate: how often a genuinely new landmark arrives, and how often the hand drops out.
        self._reads = 0
        self._absent = 0
        self._fresh = 0
        self._prev_landmarks: np.ndarray | None = None
        self._first_read = 0.0
        # Counters as of `mark_measurement_start()`, so the headline figures describe the
        # phase the caller cares about rather than everything since the tracker opened.
        # Startup centering can run *longer than the episode* and the operator's hand is
        # legitimately in and out of frame throughout it, which inflates drop-out and
        # deflates fresh-fps enough to look like a sensor fault. None ⇒ never marked.
        self._mark: tuple[float, int, int, int] | None = None
        # recenter=True only drives the renderer's open-palm countdown HUD, a handy visual
        # while the operator holds the startup-centering pose; kevin times the hold itself in
        # calibrate_neutral, and the renderer's origin offset is a no-op for us.
        #
        # max_skew_s: the two cameras run on independent, uncoordinated capture threads
        # (stereohand's StereoCapture), so a pair is only fused if their capture timestamps
        # land within this tolerance of each other -- unrelated to whether MediaPipe detects
        # a hand at all. It is an *alignment-quality* knob: how simultaneous the two views
        # must be for triangulating them to mean anything.
        #
        # It is NOT a drop-out remedy, and this comment used to say it was. The advice was
        # "if sensor-health shows high drop-out, measure your skew and raise this", from a
        # measurement of 88% of pairs rejected at 0.02s. The rejections were real; the
        # conclusion was not. stereohand's tracker woke on a *different* predicate than the
        # one read() enforced (`max(ts_left, ts_right)`, which advances when EITHER camera
        # ticks, vs "both within skew"), so it kept stepping on pairs that were then
        # rejected -- and a rejection published `_ABSENT`, i.e. "the hand is gone". That is
        # what produced 78% drop-out and a clutch releasing twice a second on a hand sitting
        # still in frame. Raising max_skew_s only widened the gate enough to hide the
        # mismatch, while trading away cross-view alignment during fast hand motion.
        #
        # Fixed in stereohand v0.2.0: one shared `pair_status()` predicate, and
        # `_ABSENT` published only for a genuinely stalled camera (`max_age_s`). On >= 0.2.0
        # a high drop-out is no longer evidence about skew at all -- look at per-view
        # detection (lighting, shared field of view) instead. Leave this at the default
        # unless triangulation *accuracy* is the complaint.
        self._tracker = StereoHandTracker.open(
            calibration,
            left=left,
            right=right,
            max_fps=max_fps,
            max_skew_s=max_skew_s,
            render=show_window,
            render_config=RenderConfig() if show_window else None,
        )
        log.info(
            "stereo hand tracker started (calib %s, cameras %r/%r)",
            calibration_path,
            left,
            right,
        )

    def read(self) -> HandReading:
        # cv2 GUI must be pumped from the main (control-loop) thread, but only needs ~30 Hz —
        # pump on a wall-clock interval, not every call (see the interval constant). The hand
        # reading below is taken every call regardless (cheap — the background thread's latest).
        if self._show_window:
            now = time.monotonic()
            if now - self._last_pump >= _WINDOW_PUMP_INTERVAL_S:
                self._last_pump = now
                if self._first_pump == 0.0:
                    self._first_pump = now
                self._pumps += 1
                # stereohand's split renderer draws in render_step() but flushes the imshow
                # buffer to screen only in poll(); without the poll() the window stays blank.
                frame = self._tracker.render_step()
                self._tracker.poll()
                if self._recording and self._record_path is not None and frame is not None:
                    self._write_frame(frame)
        reading = self._tracker.read()
        if self._reads == 0:
            self._first_read = time.monotonic()
        self._reads += 1
        if not reading.present:
            self._absent += 1
            return _ABSENT
        # A genuinely new frame changed the landmarks; identical arrays = the background
        # thread hasn't produced a new pair yet (we polled faster than the cameras run).
        if self._prev_landmarks is None or not np.array_equal(
            reading.landmarks, self._prev_landmarks
        ):
            self._fresh += 1
            self._prev_landmarks = reading.landmarks
        return reading_from_landmarks(reading.landmarks)

    _MIN_PUMPS_TO_MEASURE = 30

    def _measured_pump_fps(self) -> float | None:
        """The rate the window pump *achieves*, or None before there is enough evidence.

        ``_WINDOW_PUMP_INTERVAL_S`` is a floor on the gap between pumps, not a guarantee
        of the rate: the pump fires from ``read()``, so it slips whenever the control loop
        is busy, and it can only ever come in *at or below* the nominal 30 Hz. Stamping a
        recording with the nominal rate therefore makes it play back too fast — a
        2026-08-20 take achieved 21.3 fps (the figure the preview itself displays), was
        written as 30, and ran 1.41x quick, which also read as a physically impossible
        1.41x real-time factor when the clip was measured against its own trajectory.

        Measured across every pump since the first, which by the time recording is armed
        means the whole centering phase — the same camera, MediaPipe and compositing work
        the episode does, so it is a good estimate of the rate about to be achieved. The
        episode adds physics and control on top, so expect the true rate to come in a
        little lower still; :meth:`close` reports the actual figure and complains if the
        two disagree enough to matter.
        """
        if self._pumps < self._MIN_PUMPS_TO_MEASURE:
            return None
        elapsed = time.monotonic() - self._first_pump
        if elapsed <= 0.0:
            return None
        return (self._pumps - 1) / elapsed

    def _write_frame(self, frame: np.ndarray) -> None:
        """Append one composite frame to the recording, opening the writer on the first.

        The writer is opened lazily because its frame size has to match, and the composite's
        size isn't known until stereohand has drawn one. Encoding happens on the control
        thread but only at the pump rate (30 Hz), on a frame that was composited anyway —
        capture, MediaPipe and triangulation all run on stereohand's background thread and
        are untouched. If a recorded session ever shows a worse fresh-fps than an unrecorded
        one, hand the write to a queue + writer thread; it was not worth it up front.
        """
        import cv2

        if self._writer is None:
            path = self._record_path
            assert path is not None  # only called while recording is on
            height, width = frame.shape[:2]
            nominal_fps = 1.0 / _WINDOW_PUMP_INTERVAL_S
            measured = self._measured_pump_fps()
            fps = measured if measured is not None else nominal_fps
            for tag in ("mp4v", "avc1", "MJPG"):
                writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*tag), fps, (width, height))
                if writer.isOpened():
                    self._writer = writer
                    log.info(
                        "recording hand view → %s (codec %s, %.1f fps %s)",
                        self._record_path,
                        tag,
                        fps,
                        "measured" if measured is not None else "nominal — too few pumps yet",
                    )
                    break
                writer.release()
            else:
                log.error("could not open a video writer for %s — not recording", self._record_path)
                self._record_path = None
                return
        self._writer.write(frame)
        self._recorded_frames += 1

    def set_renderer_origin(self, origin: np.ndarray) -> None:
        """Center the preview's 3-D skeleton view on ``origin`` (metric left-camera frame).

        No-op when the preview window is off (there's no renderer to update). Used to pin the
        renderer's origin to the operator-set neutral from startup centering.
        """
        if self._show_window:
            self._tracker.set_renderer_origin((
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
            ))

    def _log_sensor_health(self) -> None:
        """Report the marked window, and the setup phase before it as a separate line."""
        now = time.monotonic()
        if self._mark is None:
            self._log_window(
                "sensor health", self._first_read, now, self._reads, self._absent, self._fresh
            )
            return

        mark_time, mark_reads, mark_absent, mark_fresh = self._mark
        self._log_window(
            "sensor health",
            mark_time,
            now,
            self._reads - mark_reads,
            self._absent - mark_absent,
            self._fresh - mark_fresh,
        )
        # Reported, never folded in: a hand out of frame during centering is expected, so
        # this number is not evidence about the sensor and must not be read as such.
        if mark_reads > 0:
            self._log_window(
                "sensor health (startup centering, expected to be worse)",
                self._first_read,
                mark_time,
                mark_reads,
                mark_absent,
                mark_fresh,
            )

    @staticmethod
    def _log_window(
        label: str, start: float, end: float, reads: int, absent: int, fresh: int
    ) -> None:
        if reads <= 0:
            return
        elapsed = end - start
        effective_fps = fresh / elapsed if elapsed > 0 else 0.0
        log.info(
            "%s: %d reads over %.1fs — %.1f fresh fps (new landmarks), %.0f%% drop-out. "
            "Teleop tracks the hand at the fresh-fps rate, not the loop rate.",
            label,
            reads,
            elapsed,
            effective_fps,
            100.0 * absent / reads,
        )

    def start_recording(self) -> None:
        """Arm ``record_path`` — call once the operator is actually driving the arm.

        The sibling of :meth:`mark_measurement_start`, for the same reason and on the same
        boundary. Startup centering (:func:`ai_teleop.input.calibrate_neutral`) runs
        *before the sim is stepped* and is wall-clock timed: the operator has to find the
        cameras, present an open palm and hold it still, and any drop restarts the hold. It
        routinely outlasts the episode. Recording from construction therefore produces a
        video that is mostly a person waving at a webcam with a motionless robot, and whose
        duration has no relationship to the trajectory beside it — so pairing the two clips
        appears to show the arm ignoring the hand for most of the run.

        Gating here also makes the operator video and ``episode.npz`` cover the *same*
        window by construction, which is what lets them be cut side by side (see
        ``scripts/dev/take_sync_report.py``).

        Idempotent. Recording is only ever armed by an explicit call, so a caller that
        configures ``record_path`` and never calls this gets a warning at
        :meth:`close` rather than a silently empty file.
        """
        self._recording = True
        self._record_started = time.monotonic()
        self._recorded_frames = 0

    def mark_measurement_start(self) -> None:
        """Start the window `close()` reports on — call once the real work begins.

        Without this, sensor health covers everything since the first `read()`, which
        includes startup centering. That phase can outlast a short episode and the
        operator's hand is in and out of frame throughout it by design, so its absences
        dominate the aggregate: a clean run reported 36% drop-out and 11.9 fresh fps
        over a window that was three-quarters centering, while the episode inside it
        released the clutch zero times. The counters are snapshotted rather than reset,
        so the setup phase is still reported — just separately, where it cannot be
        mistaken for a measurement of the episode.
        """
        self._mark = (time.monotonic(), self._reads, self._absent, self._fresh)

    def _log_recording_rate(self) -> None:
        """Report the rate the recording actually captured at, and flag a bad stamp.

        The writer's frame rate is fixed when it opens, from an estimate taken before the
        episode began (:meth:`_measured_pump_fps`). This is the same number measured over
        the window that was really recorded, so a disagreement means the file's playback
        speed is wrong by that ratio — worth saying out loud, because nothing downstream
        can tell from the file alone that its timestamps are a guess.
        """
        elapsed = time.monotonic() - self._record_started
        if self._recorded_frames < 2 or elapsed <= 0.0:
            log.info("hand view saved → %s (%d frames)", self._record_path, self._recorded_frames)
            return

        achieved = (self._recorded_frames - 1) / elapsed
        stamped = self._measured_pump_fps() or 1.0 / _WINDOW_PUMP_INTERVAL_S
        log.info(
            "hand view saved → %s (%d frames, %.2f s, %.1f fps achieved)",
            self._record_path,
            self._recorded_frames,
            elapsed,
            achieved,
        )
        if abs(achieved - stamped) / achieved > 0.05:
            log.warning(
                "hand view is stamped ~%.1f fps but captured at %.1f — it will play %.2fx "
                "off; re-time it before cutting it beside the episode",
                stamped,
                achieved,
                stamped / achieved,
            )

    def close(self) -> None:
        if self._reads > 0:
            self._log_sensor_health()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._log_recording_rate()
        elif self._record_path is not None:
            # Configured but never armed (or armed with the window off, which produces no
            # composite to write). Say so — the alternative is a caller discovering the
            # missing file after the operator has gone home.
            log.warning(
                "no hand view written to %s — recording was never armed "
                "(start_recording()) or the camera window was off",
                self._record_path,
            )
        self._tracker.close()
        log.info("stereo hand tracker stopped")

    def __enter__(self) -> StereoHandSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
