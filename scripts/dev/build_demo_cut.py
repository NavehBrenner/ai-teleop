"""Assemble the submission demo video from its already-rendered parts (LAB-125).

``submission-handoffs/video-cut.md`` is the authority on this cut; this script is its
executable form, so the captions are reviewable in the diff rather than living in someone's
editor. Everything it consumes is produced elsewhere and committed or reproducible:

    scripts/dev/render_trajectory.py        the blinded takes and the free-play robot halves
    scripts/dev/record_comparison_grid.py   the analytical-expert vs no-assist grid
    docs/results/phase-1/*.png              the results stills

Four segments, and the honesty rules that shape them:

1. **Live teleoperation** — free-play takes, operator view beside robot view. These are a
   demonstration of the tracking pipeline and *not* measured trials; every caption says so.
   The two halves are the same run (``--record-hand`` and ``--record commands`` from one
   invocation) and, since the recording window was fixed to start with the episode, they
   cover the same span with no re-timing. The one thing that must never happen here is
   pairing a hand take with a robot take from *different* runs.
2. **Blinded trial takes** — four of the 32 pre-registered trials, chosen for legibility
   and deliberately balanced two assisted / two unassisted, two seated / two force-abort.
   Their captions are burnt in by ``render_trajectory.py``.
3. **Assist on/off** — the analytical, privileged-information expert against no assist.
   Labelled as such in the frames: it is *not* the trained residual policy.
4. **Results** — what the measurements actually support, which is a documented negative.

No segment claims a success-rate lift or a contact-force reduction; both were measured and
retracted. See ``docs/conclusions.md``.

    uv run python scripts/dev/build_demo_cut.py
    uv run python scripts/dev/build_demo_cut.py --out assets/media/demo.mp4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from record_comparison_grid import resolve_font  # noqa: E402

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("demo-cut")

WIDTH, HEIGHT = 1280, 720
FPS = 20.0
BACKGROUND = (16, 16, 18)

DEFAULT_OUT = Path("assets/media/demo.mp4")
TAKES = Path("outputs/lab125/hand_takes")
BLIND = Path("outputs/lab125")
RESULTS = Path("docs/results/phase-1")


# One fallback chain for every clip renderer -- see `record_comparison_grid.resolve_font`.
# Naming DejaVu alone meant Windows silently got a bitmap font that ignores its size.
_font = resolve_font


@dataclass(frozen=True)
class Card:
    """A title card: a heading, and lines of smaller body text under it."""

    heading: str
    lines: tuple[str, ...] = ()
    seconds: float = 3.0


def _card_frame(card: Card) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 210), card.heading, fill=(245, 245, 245), font=_font(44, bold=True))
    for index, line in enumerate(card.lines):
        draw.text((80, 300 + index * 42), line, fill=(190, 190, 195), font=_font(25))
    return np.asarray(image)


def _fit(frame: np.ndarray, *, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple]:
    """Scale ``frame`` to fit ``box`` (x, y, w, h), preserving aspect; return it centred."""
    x, y, width, height = box
    image = Image.fromarray(frame)
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return resized, (x + (width - size[0]) // 2, y + (height - size[1]) // 2)


def _caption(image: Image.Image, text: str, sub: str | None) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, HEIGHT - 78, WIDTH, HEIGHT], fill=(0, 0, 0, 190))
    draw.text((28, HEIGHT - 68), text, fill=(255, 255, 255), font=_font(24, bold=True))
    if sub:
        draw.text((28, HEIGHT - 36), sub, fill=(200, 200, 205), font=_font(19))


def _read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [np.asarray(frame) for frame in iio.imiter(path, plugin="FFMPEG")]


def _compose(
    panels: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    caption: tuple[str, str | None],
) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    for panel, box in zip(panels, boxes, strict=True):
        fitted, position = _fit(panel, box=box)
        canvas.paste(fitted, position)
    _caption(canvas, *caption)
    return np.asarray(canvas)


def _emit_card(writer, card: Card) -> int:
    frame = _card_frame(card)
    count = int(card.seconds * FPS)
    for _ in range(count):
        writer.append_data(frame)
    return count


def _emit_pair(writer, take: Path, caption: str, sub: str) -> int:
    """One free-play take: operator view left, robot view right, same run."""
    hand = _read_frames(
        take / ("hand_retimed.mp4" if (take / "hand_retimed.mp4").exists() else "hand.mp4")
    )
    robot = _read_frames(take / "robot.mp4")
    count = min(len(hand), len(robot))
    for index in range(count):
        writer.append_data(
            _compose(
                [hand[index], robot[index]],
                [(20, 60, 700, 560), (740, 60, 520, 560)],
                (caption, sub),
            )
        )
    return count


def _emit_clip(writer, path: Path, caption: str, sub: str) -> int:
    frames = _read_frames(path)
    for frame in frames:
        writer.append_data(_compose([frame], [(40, 30, WIDTH - 80, HEIGHT - 130)], (caption, sub)))
    return len(frames)


def _emit_still(writer, path: Path, seconds: float, caption: str, sub: str) -> int:
    import imageio.v3 as iio

    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    still = np.asarray(Image.fromarray(iio.imread(path)).convert("RGB"))
    frame = _compose([still], [(40, 30, WIDTH - 80, HEIGHT - 130)], (caption, sub))
    count = int(seconds * FPS)
    for _ in range(count):
        writer.append_data(frame)
    return count


def build(out: Path, quality: int) -> None:
    import imageio.v2 as iio2

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio2.get_writer(
        str(out), fps=FPS, codec="libx264", quality=quality, macro_block_size=8
    )
    total = 0
    try:
        total += _emit_card(
            writer,
            Card(
                "AI-assisted teleoperation for peg-in-hole insertion",
                (
                    "A human gives coarse 6-DoF commands; a vision-conditioned residual policy",
                    "supplies micro-corrections. MuJoCo, Franka Emika Panda.",
                    "OpenU 20973 — Workshop in Autonomous Systems Simulation.",
                ),
                seconds=4.0,
            ),
        )

        total += _emit_card(
            writer,
            Card(
                "1 — Live teleoperation",
                (
                    "Two webcams to metric 3-D hand tracking to the arm.",
                    "Free-play demonstrations of the tracking pipeline — not measured trials.",
                    "Operator view and robot view are the same run, recorded together.",
                ),
            ),
        )
        total += _emit_pair(
            writer,
            TAKES / "take_2",
            "Free play — operator view (left) and robot view (right), one run",
            "Not a trial. Hand tracking at 21 fps, control loop at 498 Hz, 1.00x real time. Peg seated.",
        )
        total += _emit_pair(
            writer,
            TAKES / "take_3",
            "Free play — the same setup, a run that fails",
            "Not a trial. Terminated by the 30 N force cap without seating. Shown unedited.",
        )

        total += _emit_card(
            writer,
            Card(
                "2 — The blinded human-operator trial",
                (
                    "32 pre-registered trials, assist on or off, hidden from the operator.",
                    "Four shown here, picked for legibility: two assisted, two not;",
                    "two seated, two aborted. Different walls, single runs, not a comparison.",
                ),
                seconds=4.0,
            ),
        )
        for name, caption, sub in (
            (
                "trial_20",
                "Blinded trial 20 — assist OFF",
                "Seated in 3.5 s. Single run on wall seed 20.",
            ),
            (
                "trial_18",
                "Blinded trial 18 — assist ON",
                "Seated in 3.6 s. Single run on wall seed 18.",
            ),
            (
                "trial_05",
                "Blinded trial 05 — assist OFF",
                "Force abort at 32.6 N. Single run on wall seed 5.",
            ),
            (
                "trial_08",
                "Blinded trial 08 — assist ON",
                "Force abort at 36.5 N. Measured force is not bounded by the 30 N cap.",
            ),
        ):
            total += _emit_clip(writer, BLIND / f"{name}.mp4", caption, sub)

        total += _emit_card(
            writer,
            Card(
                "3 — Assistance on and off",
                (
                    "The analytical, privileged-information expert against no assist,",
                    "same scripted operator and seed. This is NOT the trained policy —",
                    "the expert is handed the target pose.",
                ),
            ),
        )
        total += _emit_clip(
            writer,
            Path("outputs/comparison_grid_2_7.mp4"),
            "Analytical expert (top) vs no assist (bottom) — same operator, same seed",
            "Privileged-info controller, not the trained residual. Expert seats at 2 mm; no-assist ends 30 mm out.",
        )

        total += _emit_card(
            writer,
            Card(
                "4 — What the measurements support",
                (
                    "The project closed as a documented negative, and the video says so.",
                    "No success-rate lift is established; no force reduction is either.",
                ),
                seconds=4.0,
            ),
        )
        total += _emit_still(
            writer,
            RESULTS / "success_rate_spread.png",
            7.0,
            "Per-recipe success against the human-only baseline",
            "No recipe clears a 20-31 pp training-seed noise floor. 9 of 21 checkpoints sit above baseline — a coin flip.",
        )
        total += _emit_card(
            writer,
            Card(
                "What can be claimed",
                (
                    "The residual is clamped by construction: +/-3 cm, +/-10 deg, +/-5 N per step,",
                    "and the backbone can command no more than 12.5 N of restoring force.",
                    "Measured contact force is NOT bounded — it reaches 77.86 N.",
                    "The only established effects are two costs: slower, and rougher.",
                ),
                seconds=7.0,
            ),
        )
    finally:
        writer.close()

    size_mb = out.stat().st_size / 1e6
    log.info("wrote %s — %d frames, %.1f s, %.1f MB", out, total, total / FPS, size_mb)
    if size_mb > 10:
        log.warning("%.1f MB is over the ~10 MB budget for inline GitHub playback", size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quality", type=int, default=6, help="imageio/x264 quality, 0-10")
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)
    build(arguments.out, arguments.quality)


if __name__ == "__main__":
    main()
