"""Render a recorded episode into presentable video, offline.

`run_episode.py --record images` writes the 224x224 wrist frames the *policy* is fed —
the right thing for training, useless as footage. And rendering a third-person view live
puts a MuJoCo render call inside the 500 Hz control loop, which is what makes a
`--input vision` session lag.

So: record commands live (cheap), render here afterwards (no real-time pressure). Replay
rebuilds the exact recorded scene and controller from the episode's own metadata and
reproduces the run to the step, so what this renders is the episode that happened, not a
re-simulation of it.

Pairs with `--record-hand`, which captures the operator half — two clips of the same
trial, cut side by side later.

Read-only apart from its output. Run from kevin/:

    uv run python scripts/dev/render_trajectory.py runs/blind_trial/trial_007
    uv run python scripts/dev/render_trajectory.py runs/take_1 --wrist --every 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from record_comparison_grid import PANEL, label, make_camera, save_animation  # noqa: E402

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data import load_episode  # noqa: E402
from ai_teleop.sim.scene import EnvConfig, SimEnv  # noqa: E402
from ai_teleop.sim.scene_source import resolve_scene_path  # noqa: E402

log = get_logger("render-trajectory")


def render(episode_dir: Path, args: argparse.Namespace) -> Path:
    """Replay the recorded commands and render frames; returns the written video path."""
    npz = episode_dir / "episode.npz" if episode_dir.is_dir() else episode_dir
    columns, metadata = load_episode(npz)

    scene_path = resolve_scene_path(
        generated=bool(metadata.get("generated_wall")),
        wall_seed=metadata.get("wall_seed"),
        distractors=metadata.get("distractors"),
    )
    environment = SimEnv(
        str(scene_path),
        render_mode="headless",
        camera_height=PANEL,
        camera_width=PANEL,
        config=EnvConfig(wall_seed=metadata.get("wall_seed")),
    )
    observation = environment.reset()
    # The recorded controller config, not this script's defaults: replaying the commands
    # through a differently-clamped controller drives a different physical episode.
    controller = Controller(
        environment,
        max_dpos_per_step=float(metadata.get("max_dpos", 0.025)),
        joint_damping=float(metadata.get("joint_damping", 4.0)),
    )

    camera = make_camera(args)
    renderer = mujoco.Renderer(environment.model, height=PANEL, width=PANEL)
    frames: list[np.ndarray] = []
    positions, quaternions, grips = (
        columns["cmd_position"],
        columns["cmd_quaternion"],
        columns["cmd_grip"],
    )
    from ai_teleop.common import Command

    for step in range(len(positions)):
        controller.compute(
            observation,
            Command(positions[step].copy(), quaternions[step].copy(), float(grips[step])),
        )
        environment.step()
        observation = environment.get_observation()
        if step % args.every:
            continue
        renderer.update_scene(environment.data, camera=camera)
        frame = renderer.render().copy()
        if args.wrist:
            wrist = np.asarray(
                Image.fromarray(environment.render_wrist_camera()).resize(
                    (PANEL, PANEL), Image.Resampling.NEAREST
                )
            )
            frame = np.hstack([frame, wrist])
        frames.append(frame)
    renderer.close()
    environment.close()

    if args.caption:
        frames = [np.asarray(label(frame, args.caption)) for frame in frames]

    output = args.out or episode_dir.with_suffix("") / "robot.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_animation(frames, output, args.fps)
    log.info(
        "%d frames → %s  [recorded outcome: %s]",
        len(frames),
        output,
        metadata.get("terminal_reason", "unknown"),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="episode folder or episode.npz")
    parser.add_argument("--out", type=Path, default=None, help="output .mp4/.gif")
    parser.add_argument("--every", type=int, default=25, help="render one frame per N steps")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--wrist", action="store_true", help="append the wrist view beside it")
    parser.add_argument("--caption", default=None, help="text banner burnt into every frame")
    parser.add_argument("--cam-azimuth", type=float, default=-40.0)
    parser.add_argument("--cam-elevation", type=float, default=-18.0)
    parser.add_argument("--cam-distance", type=float, default=1.6)
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    if not arguments.episode.exists():
        log.error("no such episode: %s", arguments.episode)
        raise SystemExit(1)
    render(arguments.episode, arguments)


if __name__ == "__main__":
    main()
