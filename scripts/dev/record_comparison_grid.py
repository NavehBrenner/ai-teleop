"""Record a 2x2 comparison clip: Expert vs No-assist, third-person + wrist cam.

Runs two episodes through the M3/M4 seam with the *same* scripted operator
(same seed) so the only difference is the assist source:

    row 0 : ANALYTICAL EXPERT  [ third-person | wrist camera ]
    row 1 : NO ASSIST          [ third-person | wrist camera ]

Row 0 is the *analytical, privileged-information* expert -- it is handed the target
pose -- and NOT the trained residual policy. The panels say so, because the two are
routinely conflated and the trained policy's success-rate lift was retracted.

Each panel is labelled. The third-person camera is placed behind the wall,
looking back toward the arm. Output defaults to MP4 (seekable / pausable);
pass --format gif for a GIF instead. Pass --generated-wall to run on a freshly
procedurally-generated wall instead of the static task scene. --max-dpos raises
the controller's per-step command clamp (the free-space approach-speed knob).

Run: uv run python scripts/dev/record_comparison_grid.py
     uv run python scripts/dev/record_comparison_grid.py --format gif
     uv run python scripts/dev/record_comparison_grid.py --generated-wall --wall-seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_teleop.control import Controller
from ai_teleop.domain import NoAssist, apply_delta
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import resolve_scene_path

# Font families to try, in order, per weight. DejaVu is the Linux/WSL name this project
# was written against; it does not exist on Windows, where every caption in every clip was
# silently falling back to `load_default()` — a *bitmap* font that ignores the size it is
# handed and has no em-dash, so text rendered tiny and `—` came out as a box.
_FONT_CANDIDATES: dict[bool, tuple[str, ...]] = {
    False: ("DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"),
    True: ("DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"),
}


def resolve_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A real scalable font at ``size``, whatever platform this is rendering on.

    Shared with the other clip renderers (``render_trajectory.py``, ``build_demo_cut.py``)
    so captions look the same everywhere and no caller re-invents the fallback chain.
    """
    for candidate in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        # Pillow >= 10.1 honours `size` here and returns a real scalable face.
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - only on Pillow < 10.1
        return ImageFont.load_default()


_FONT = resolve_font(22, bold=True)
_SUBTITLE_FONT = resolve_font(14)

# Burnt into every assisted panel. See `label()` for why this qualifier is not optional.
_EXPERT_SUBTITLE = "privileged-info controller, not the trained policy"
_NO_ASSIST_SUBTITLE = "scripted operator alone"

PANEL = 480
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
STEM = "comparison_grid"


def make_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, 0.0, 0.5]
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth  # behind-the-wall view by default
    cam.elevation = args.cam_elevation
    return cam


def _progress(prefix: str, done: int, total: int) -> None:
    """In-place ASCII progress bar on one line; newline when complete."""
    filled = int(28 * done / total)
    bar = "█" * filled + "░" * (28 - filled)
    end = "\n" if done >= total else ""
    print(f"\r  {prefix:11} [{bar}] {done:3d}/{total} frames", end=end, flush=True)


def run_views(
    scene_path: Path,
    assist,
    seed: int,
    steps: int,
    every: int,
    cam: mujoco.MjvCamera,
    max_dpos: float,
    progress_label: str = "",
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Run one episode; return (third_person_frames, wrist_frames, final_mm)."""
    env = SimEnv(str(scene_path), render_mode="headless", camera_height=PANEL, camera_width=PANEL)
    observation = env.reset()
    controller = Controller(env, max_dpos_per_step=max_dpos)
    target = observation.hole_poses[0][:3].copy()  # task goal: hole_0
    human = ScriptedNoisyHuman(
        np.concatenate([target, controller.home_pose[3:]]),
        position_bias_std=0.012,
        orientation_bias_std=np.deg2rad(4),
        seed=seed,
    )
    third = mujoco.Renderer(env.model, height=PANEL, width=PANEL)
    third_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    total_frames = len(range(0, steps, every))
    for t in range(steps):
        base = human.get_command(observation)
        command = apply_delta(base, assist.get_delta(observation, base))
        controller.compute(observation, command)
        env.step()
        observation = env.get_observation()
        if t % every == 0:
            third.update_scene(env.data, camera=cam)
            third_frames.append(third.render().copy())
            wrist = env.render_wrist_camera()
            wrist_frames.append(
                np.asarray(Image.fromarray(wrist).resize((PANEL, PANEL), Image.Resampling.NEAREST))
            )
            if progress_label:
                _progress(progress_label, len(third_frames), total_frames)
    third.close()
    final_mm = float(np.linalg.norm(observation.peg_pose[:3] - target)) * 1000
    env.close()
    return third_frames, wrist_frames, final_mm


def label(frame: np.ndarray, text: str, subtitle: str | None = None) -> Image.Image:
    """Burn a caption into a panel; ``subtitle`` adds a smaller qualifying line.

    The subtitle exists because "EXPERT" alone is misread. This grid's assisted row is
    the **analytical, privileged-information** expert — it is handed the target pose —
    and *not* the trained residual policy, whose success-rate lift was measured and
    retracted (``docs/results/phase-1/noise-floor-per-kpi.md``). A viewer who reads the
    row as "the AI assistance" takes away a claim the project explicitly withdrew, so
    the qualifier travels with the frame rather than living in a caption someone might
    drop when the clip is re-cut.
    """
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    banner = 34 if subtitle is None else 56
    draw.rectangle([0, 0, PANEL, banner], fill=(0, 0, 0, 170))
    draw.text((10, 7), text, fill=(255, 255, 255, 255), font=_FONT)
    if subtitle is not None:
        draw.text((10, 34), subtitle, fill=(220, 220, 220, 255), font=_SUBTITLE_FONT)
    return img


def save_animation(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """Write `frames` (list of HxWx3 uint8) as MP4 or GIF, inferred from suffix."""
    if path.suffix == ".mp4":
        import imageio.v3 as iio

        # even dimensions required by the H.264 encoder.
        h, w = frames[0].shape[:2]
        stack = np.stack(frames)[:, : h - (h % 2), : w - (w % 2)]
        iio.imwrite(path, stack, fps=fps, codec="libx264", quality=8)
    else:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument(
        "--format",
        choices=("mp4", "gif"),
        default="mp4",
        help="output container (mp4 is seekable/pausable; default)",
    )
    ap.add_argument("--fps", type=float, default=20.0, help="playback frame rate")
    ap.add_argument(
        "--max-dpos",
        type=float,
        default=0.025,
        help="controller per-step command clamp in metres (approach-speed knob)",
    )
    ap.add_argument(
        "--generated-wall",
        action="store_true",
        help="run on a freshly generated wall instead of the static scene",
    )
    ap.add_argument("--wall-seed", type=int, default=7)
    ap.add_argument("--distractors", type=int, default=None)
    ap.add_argument("--cam-azimuth", type=float, default=-40.0)
    ap.add_argument("--cam-elevation", type=float, default=-18.0)
    ap.add_argument("--cam-distance", type=float, default=1.6)
    args = ap.parse_args()

    scene_path = resolve_scene_path(
        generated=args.generated_wall,
        wall_seed=args.wall_seed,
        distractors=args.distractors,
    )
    cam = make_camera(args)

    print("rendering 2 episodes:")
    ex_third, ex_wrist, ex_mm = run_views(
        scene_path, Expert(), args.seed, args.steps, args.every, cam, args.max_dpos, "expert"
    )
    na_third, na_wrist, na_mm = run_views(
        scene_path, NoAssist(), args.seed, args.steps, args.every, cam, args.max_dpos, "no-assist"
    )
    print(f"expert final peg->hole = {ex_mm:.0f} mm   no-assist = {na_mm:.0f} mm")

    n = min(len(ex_third), len(na_third))
    grid_frames: list[np.ndarray] = []
    for i in range(n):
        top = np.concatenate(
            [
                np.asarray(
                    label(ex_third[i], "ANALYTICAL EXPERT  -  third person", _EXPERT_SUBTITLE)
                ),
                np.asarray(
                    label(ex_wrist[i], "ANALYTICAL EXPERT  -  wrist camera", _EXPERT_SUBTITLE)
                ),
            ],
            axis=1,
        )
        bottom = np.concatenate(
            [
                np.asarray(label(na_third[i], "NO ASSIST  -  third person", _NO_ASSIST_SUBTITLE)),
                np.asarray(label(na_wrist[i], "NO ASSIST  -  wrist camera", _NO_ASSIST_SUBTITLE)),
            ],
            axis=1,
        )
        grid_frames.append(np.concatenate([top, bottom], axis=0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{STEM}_{args.seed}_{args.wall_seed}.{args.format}"
    print(f"encoding {n} frames → {args.format} ...")
    save_animation(grid_frames, out, args.fps)
    Image.fromarray(grid_frames[int(n * 0.85)]).save(OUT_DIR / f"{STEM}.still.png")
    print(f"wrote {out}  ({n} frames, {grid_frames[0].shape[1]}x{grid_frames[0].shape[0]})")


if __name__ == "__main__":
    main()
