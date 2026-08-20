"""Exploratory blinded playtest of the *vision* residual — curiosity, not the record.

This is **not** the pre-registered human trial (`blind_trial.py` +
`docs/results/human-trial-protocol.md`). It deliberately does the thing the protocol
forbids: it loads the **best-scoring** vision-DAgger checkpoint (seed 1 / round 4, 62%
success, the +12 pp seed-noise outlier), so nothing it produces is an unbiased estimate of
anything and no result from it may be reported as a finding. Its only purpose is to let the
operator *feel* whether the vision policy assists them, blinded, and see how well they could
tell the arm apart.

10 trials by default, half with the residual applied and half with it discarded, in a
random order held only in memory. You guess on/off/unsure after each; the assignment and
your guess statistics are revealed only at the end. Episodes are recorded (harmless), but
the artifacts are throwaway — they live under `runs/vision_play/`, which is gitignored.

Run from kevin/ (Windows-native — the cameras are not visible from WSL):

    uv run python scripts/dev/vision_play.py --stereo-calib ..\\stereohand\\stereo_calib.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for blind_trial

from blind_trial import GUESSES, run_trial, schedule  # noqa: E402

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("vision-play")

# The best-scoring vision-DAgger checkpoint (official_kpi_tables.md §6): seed 1, round 4,
# 62% success / +12 pp. It is the seed-noise outlier — chosen here precisely because this is
# the exploratory feel-test, NOT the pre-registered trial, which uses the seed-stable
# FT-DAgger checkpoint for exactly the reasons this one is disqualified from the record.
BEST_VISION_CHECKPOINT = Path("docs/results/checkpoints/vision/dagger/seed_1/round_4/checkpoint.pt")


def report(rows: list[tuple[bool, str, str]]) -> None:
    """Reveal the assignment and score the operator's guesses. Descriptive only."""
    print("\n" + "=" * 52)
    print("  REVEAL - exploratory vision playtest (not a finding)")
    print("=" * 52)

    print("\nPer-arm outcomes:")
    print("| arm | trials | successes | rate |")
    print("|---|---|---|---|")
    for label, wanted in (("assist ON ", True), ("assist OFF", False)):
        arm = [r for r in rows if r[0] == wanted]
        seated = sum(outcome == "success" for _, outcome, _ in arm)
        rate = 100.0 * seated / len(arm) if arm else 0.0
        print(f"| {label} | {len(arm)} | {seated} | {rate:.0f}% |")

    decided = [r for r in rows if r[2] != "unsure"]
    correct = sum((guess == "on") == assist_on for assist_on, _, guess in decided)
    unsure = len(rows) - len(decided)
    print("\nYour guesses:")
    print(f"  {len(rows)} trials  |  {unsure} unsure  |  {len(decided)} decided")
    if decided:
        print(
            f"  {correct}/{len(decided)} decided guesses correct "
            f"({100.0 * correct / len(decided):.0f}%) -- 50% is chance."
        )

    print("\nConfusion (rows = truth, cols = your guess):")
    print("|        | said ON | said OFF | unsure |")
    print("|---|---|---|---|")
    for label, wanted in (("was ON ", True), ("was OFF", False)):
        guesses = [guess for assist_on, _, guess in rows if assist_on == wanted]
        print(
            f"| {label} | {guesses.count('on')} | {guesses.count('off')} | "
            f"{guesses.count('unsure')} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-calib", required=True)
    parser.add_argument("--cameras", nargs=2, default=["0", "1"])
    parser.add_argument("--checkpoint", type=Path, default=BEST_VISION_CHECKPOINT)
    parser.add_argument("--trials", type=int, default=10, help="total trials (default 10)")
    parser.add_argument(
        "--cam",
        choices=["main", "wrist"],
        default="main",
        help="Viewer camera: 'main' free camera (default) or 'wrist' for the robot's-eye POV.",
    )
    parser.add_argument(
        "--no-cam-window",
        action="store_true",
        help="Drop the cv2 hand preview. Its pump runs inside StereoHandSource.read(), so it "
        "is charged to the control loop -- measured 0.982 -> 0.257 ms/step on the `input` "
        "phase, which is what takes a vision checkpoint from 0.68x to 0.99x real time. "
        "Trade-offs: no operator-side hand preview, and `hand.mp4` is not written (the "
        "recording is made from the frame the preview composites).",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/vision_play"))
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--max-dpos", type=float, default=None)
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    if not arguments.checkpoint.exists():
        parser.error(f"checkpoint not found: {arguments.checkpoint}")
    arguments.out.mkdir(parents=True, exist_ok=True)

    # Fresh random order every run so the blind is not memorisable between sessions; kept in
    # memory only, never written to disk, so there is nothing to accidentally peek at.
    assignments = schedule(arguments.trials, random.Random().randrange(1 << 30))

    rows: list[tuple[bool, str, str]] = []
    for i, assist_on in enumerate(assignments):
        log.info("--- trial %d/%d ---", i + 1, arguments.trials)
        outcome = run_trial(i, assist_on, i, arguments.out, arguments)
        log.info("outcome: %s", outcome)
        answer = ""
        while answer not in GUESSES:
            answer = input("Was the assist on? [y/n/u] ").strip().lower()[:1]
        rows.append((assist_on, outcome, GUESSES[answer]))

    report(rows)


if __name__ == "__main__":
    main()
