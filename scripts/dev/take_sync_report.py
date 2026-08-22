"""Report whether a free-play take's two halves can be cut side by side (LAB-125).

A take records the operator half (``hand.mp4``, written frame-by-frame on the **wall
clock**) and the robot half (``episode.npz``, a trajectory in **sim time**, rendered
offline later by ``render_trajectory.py``). Those are the same run, so pairing them is
honest — but they only *line up* if the session held real time. A session at 0.85x turns
10 s of hand video into 8.5 s of robot video, and the pair drifts visibly by the end of
the shot.

This measures the drift instead of assuming it, and reports the render cadence that
cancels it: ``render_trajectory.py --every N --fps F`` covers ``n_steps / N`` frames in
``n_steps / (N * F)`` seconds, so matching the hand clip's duration is a matter of
choosing the pair that makes that equal. Re-timing the robot render to the wall clock the
episode actually ran at is an alignment, not a fabrication — the trajectory is unchanged
and every frame still comes from the step it was recorded at.

    uv run python scripts/dev/take_sync_report.py outputs/lab125/hand_takes/take_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data.trajectory import load_episode  # noqa: E402

log = get_logger("take-sync")

SIM_DT = 0.002  # one control tick, seconds (500 Hz)


def _probe_video(path: Path) -> tuple[float, int]:
    """Return ``(duration_seconds, frame_count)`` for a video, via the bundled ffmpeg."""
    import imageio_ffmpeg

    ffprobe = Path(imageio_ffmpeg.get_ffmpeg_exe()).with_name("ffprobe.exe")
    if ffprobe.exists():
        result = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=duration,nb_read_frames,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        return float(stream["duration"]), int(stream["nb_read_frames"])

    # `imageio-ffmpeg` ships the ffmpeg binary but not ffprobe, so read the container
    # through its own plugin instead. `nframes` is often inf/unreliable for a
    # variable-rate capture, so count the frames rather than trust the header.
    import imageio.v3 as iio

    metadata = iio.immeta(path, plugin="FFMPEG")
    frames = sum(1 for _ in iio.imiter(path, plugin="FFMPEG"))
    return frames / float(metadata["fps"]), frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", type=Path, help="a take directory (episode.npz + hand.mp4)")
    parser.add_argument(
        "--every",
        type=int,
        default=25,
        help="candidate render decimation to evaluate (default 25, render_trajectory's own)",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    episode = arguments.take / "episode.npz"
    hand = arguments.take / "hand.mp4"
    for path in (episode, hand):
        if not path.exists():
            raise SystemExit(f"{path} is missing — is {arguments.take} a completed take?")

    columns, metadata = load_episode(episode)
    n_steps = int(metadata["n_steps"])
    sim_seconds = n_steps * SIM_DT
    wall_seconds, hand_frames = _probe_video(hand)

    realtime_factor = sim_seconds / wall_seconds
    log.info("take            : %s", arguments.take)
    log.info("outcome         : %s", metadata["terminal_reason"])
    log.info("trajectory      : %d steps = %.2f s of sim time", n_steps, sim_seconds)
    log.info("hand.mp4        : %d frames = %.2f s of wall clock", hand_frames, wall_seconds)
    log.info("real-time factor: %.3fx", realtime_factor)

    # render_trajectory: n_steps/every frames, played at fps -> n_steps/(every*fps) seconds.
    frames = n_steps // arguments.every
    naive_fps = 1.0 / (arguments.every * SIM_DT)
    matched_fps = frames / wall_seconds
    log.info(
        "at --every %d: %d frames; --fps %.1f plays sim-time (%.2f s), "
        "--fps %.2f matches hand.mp4 (%.2f s)",
        arguments.every,
        frames,
        naive_fps,
        frames / naive_fps,
        matched_fps,
        wall_seconds,
    )

    drift = abs(sim_seconds - wall_seconds)
    if drift < 0.25:
        log.info("verdict         : within %.2f s — cut side by side as-is", drift)
    else:
        log.warning(
            "verdict         : %.2f s of drift over the take — render with --fps %.2f "
            "to pair without desync",
            drift,
            matched_fps,
        )


if __name__ == "__main__":
    main()
