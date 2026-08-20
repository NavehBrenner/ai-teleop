"""Blinded A/B session: does the trained residual help a *human* operator?

Every KPI in this project was measured against `ScriptedNoisyHuman`, a model of an
operator. This is the only place a real one closes the loop. Each trial runs one live
teleoperation episode with the residual's output either applied or discarded, chosen by a
hidden schedule, and the operator does not learn which until the session is unsealed.

**Read `docs/results/human-trial-protocol.md` before running.** It fixes the checkpoint,
the trial count and the analysis in advance, and states what this design can and cannot
resolve — at 16 trials per arm the smallest difference it can detect is ~45 pp, while the
scripted study's effects are single-digit pp. It is a proxy-validity check and a source of
unselected footage, not a hypothesis test.

How the blind holds:

- Both arms run `--policy tf --assist-scale {1,0}`, never `--policy noassist`. Scale 0
  still loads the checkpoint, still runs the forward pass, still captures wrist frames —
  it only discards the Δ. Startup time and per-step cost are identical, so the arm cannot
  be inferred from how the session behaves.
- The child's stdout/stderr go to a per-trial `console.log`, so no line naming the policy
  reaches the terminal. The one exception is the startup centering state ("centering in
  3s — hold still" … "centering complete"), re-rendered by the parent as a live spinner so
  the operator knows when to hold the palm still and when they may start moving. That is a
  *whitelist* of one known-safe family (`_OPERATOR_STATUS`), not a filter over unsafe
  lines — centering finishes before the sim is stepped, so it cannot differ between arms.
- Assignment is **block-randomized**: within each block of 4 trials, exactly 2 on and 2
  off. Guarantees balance and spreads both arms evenly across the operator's fatigue and
  learning curve, which plain coin-flipping does not.
- The operator records a guess after every trial. If the guesses land at chance, the blind
  held and the write-up can say so; if they do not, the write-up says that instead.

Every trial gets a fresh wall seed. Re-running one wall with and without the assist would
be the stronger paired design, but a human remembers the wall — the memory contaminates
the second run, so the arms are unpaired and wall difficulty is randomized instead.

Run from kevin/ (Windows-native — the cameras are not visible from WSL):

    uv run python scripts/dev/blind_trial.py --stereo-calib ..\\stereohand\\stereo_calib.json
    uv run python scripts/dev/blind_trial.py --practice 4 --trials 32   # the defaults
    uv run python scripts/dev/blind_trial.py --unseal      # after every trial is done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import TracebackType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data import load_episode  # noqa: E402

log = get_logger("blind-trial")

# Pre-specified in the protocol, NOT selected on outcome: lowest-numbered training seed,
# final DAgger round. Picking the best-scoring checkpoint instead is the selection bias the
# project already documented (docs/self-evaluation.md, "pre-register the reporting rule") --
# here that would be vision/dagger/seed_1/round_4 (+12 pp at p=0.036), which the rule
# excludes.
#
# The family is Vision-DAgger rather than F/T-DAgger, amended 2026-08-12 before any
# recorded trial existed (protocol §0). An F/T-only residual acts on contact force, and
# this rig has no haptic feedback -- its other effects are +0.14-0.27 s on a 7.83 s task
# and +2.0 pp success, neither perceptible. The blinding check would have been measuring
# the absence of a channel rather than the absence of an effect. The vision residual acts
# on free-space approach, which the operator can see.
DEFAULT_CHECKPOINT = Path("docs/results/checkpoints/vision/dagger/seed_0/round_4/checkpoint.pt")

BLOCK = 4  # trials per randomization block; must be even so each block splits evenly
GUESSES = {"y": "on", "n": "off", "u": "unsure"}

# The ONLY child output that reaches the operator's terminal: the startup centering states
# from `vision_input.calibrate_neutral` ("centering in 3s — hold still", "centering reset —
# ...", "centering complete — neutral set"). The operator has to know when to hold the palm
# still and when they may start moving, and that cue is otherwise buried in the redirected
# log along with every line naming the policy.
#
# This is a **whitelist**, which is why it is safe where the module docstring rejects a
# filter. A blacklist has to anticipate every line that could leak the arm; a whitelist
# shows one known-safe family and drops everything else, so a new log line in the child is
# hidden by default rather than exposed by default. Nothing matching this pattern differs
# between the arms — centering runs before the sim is stepped, and the assist has not been
# consulted yet.
_OPERATOR_STATUS = re.compile(r"(centering[^\r\n]*)")

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # matches run_episode's centering line


class _StatusLine:
    """One self-overwriting spinner line, driven by the child's centering log.

    The child's own spinner only draws when its console is a TTY (see
    `vision_input.calibrate_neutral`), and here that console is a pipe — so the child falls
    back to per-second log lines and the parent animates them instead. Redrawing from this
    side is what keeps the spinner moving *between* the child's 1 Hz countdown ticks;
    without it the line would sit frozen for a second at a time and read as a hang.

    Silently inert when the parent's own stdout is not a TTY, so a piped session still
    produces clean output.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._live = self._stream.isatty()
        self._text: str | None = None
        self._done = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _StatusLine:
        if self._live:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._live and self._text is not None and not self._done:
                self._stream.write("\r\x1b[K")  # clear a half-drawn line
                self._stream.flush()

    def update(self, text: str) -> None:
        """Show `text`. 'centering complete' finishes the line and stops the animation."""
        with self._lock:
            self._text = text
            if "complete" in text:
                self._done = True
                if self._live:
                    # Tick, newline, then the one cue the operator actually acts on. The
                    # arm is live from here: everything before this point required holding
                    # the palm still, everything after is teleoperation.
                    self._stream.write(f"\r✓ {text}\x1b[K\n  → you can move now\n")
                    self._stream.flush()

    def _animate(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._text is not None and not self._done:
                    frame = _SPINNER_FRAMES[int(time.monotonic() * 10) % len(_SPINNER_FRAMES)]
                    self._stream.write(f"\r{frame} {self._text}\x1b[K")
                    self._stream.flush()
            self._stop.wait(0.1)


def schedule(n_trials: int, seed: int) -> list[bool]:
    """Block-randomized assist assignment: exactly half of every block of `BLOCK` is on.

    A trailing partial block is balanced as evenly as it can be, so the arms stay within
    one trial of each other however the session is cut short.
    """
    rng = random.Random(seed)
    assignments: list[bool] = []
    while len(assignments) < n_trials:
        remaining = n_trials - len(assignments)
        size = min(BLOCK, remaining)
        block = [True] * (size // 2) + [False] * (size - size // 2)
        rng.shuffle(block)
        assignments.extend(block)
    return assignments


def run_child(command: list[str], console_path: Path) -> int:
    """Run `command`, log every line to `console_path`, surface only centering states.

    The seam the blind now rests on, extracted so it can be tested without a robot: every
    line the child writes is captured to the log and **nothing** reaches the terminal except
    what `_OPERATOR_STATUS` admits.
    """
    # The child's own stdout encoding follows its locale, which on Windows is cp1252 — and
    # the status lines it writes are full of em dashes and arrows. Pin it to UTF-8 so the
    # child cannot die encoding its own log, and so this end decodes what it actually sent.
    child_environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    with console_path.open("w", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,  # line-buffered: the countdown must reach the operator as it happens
            env=child_environment,
        )
        assert process.stdout is not None
        with _StatusLine() as status:
            for line in process.stdout:
                console.write(line)
                found = _OPERATOR_STATUS.search(line)
                if found:
                    status.update(found.group(1))
        return process.wait()


def run_trial(
    index: int, assist_on: bool, wall_seed: int, out_dir: Path, args: argparse.Namespace
) -> str:
    """Run one episode as a child process; return its terminal reason.

    The child's output is still redirected wholesale — every line goes to the trial's
    `console.log` and none reaches the terminal on its own. See :func:`run_child`.
    """
    trial_dir = out_dir / f"trial_{index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_episode.py"),
        "--input",
        "vision",
        "--stereo-calib",
        args.stereo_calib,
        "--cameras",
        args.cameras[0],
        args.cameras[1],
        "--wall-seed",
        str(wall_seed),
        "--policy",
        "tf",
        "--checkpoint",
        str(args.checkpoint),
        "--assist-scale",
        "1" if assist_on else "0",
        # Goes to the trial's console.log, never to the operator's terminal (the whitelist
        # passes only centering lines). Both arms run the policy in full — scale 0 discards
        # the Δ but still pays for it — so the timings are identical either way and reveal
        # nothing about the assignment. Worth having: the per-episode real-time factor is
        # the only record of whether the loop actually kept pace under the full live load
        # (cameras + tracker + viewer), which no offline measurement covers.
        "--profile",
        "--record",
        "commands",
        "--record-out",
        str(trial_dir),
        "--cam",
        # Optional on the caller: `vision_play.py` drives run_trial with its own namespace
        # and does not offer the flag, so an absent one means the protocol's default view.
        getattr(args, "cam", "main"),
    ]
    if args.gain is not None:
        command += ["--gain", str(args.gain)]
    if args.max_dpos is not None:
        command += ["--max-dpos", str(args.max_dpos)]
    # Same optional-on-the-caller pattern as --cam: absent means the protocol's default
    # (preview on). The cv2 pump lives inside StereoHandSource.read(), so it is charged to
    # the control loop's `input` phase -- dropping it is what buys a vision checkpoint its
    # real-time budget back. Note it also silences `--record-hand`: the operator-side video
    # is written from the frame the preview composites, so no window means no footage.
    if getattr(args, "no_cam_window", False):
        command += ["--no-cam-window"]
    else:
        # Only meaningful with the preview on: the operator-side video is written from the
        # frame the preview composites, so passing it without a window promises footage that
        # is never written. Since 2026-08-20 the trials run without the preview and the hand
        # footage comes from separate free-play takes.
        command += ["--record-hand", str(trial_dir / "hand.mp4")]

    run_child(command, trial_dir / "console.log")

    episode = trial_dir / "episode.npz"
    if not episode.exists():
        return "no_episode"
    _, metadata = load_episode(episode)
    return str(metadata.get("terminal_reason", "unknown"))


def unseal(out_dir: Path) -> None:
    """Join the sealed assignments to the recorded outcomes and print the result.

    Descriptive only: counts and rates per arm, plus how well the operator guessed. No
    p-value — see the protocol for why this design cannot support one.
    """
    sealed = json.loads((out_dir / "assignments.json").read_text())
    with (out_dir / "trials.csv").open() as handle:
        trials = list(csv.DictReader(handle))

    print(f"\n{len(trials)} trials  ·  checkpoint {sealed['checkpoint']}\n")
    print("| Arm | trials | successes | rate |")
    print("|---|---|---|---|")
    for label, wanted in (("assist ON", True), ("assist OFF", False)):
        arm = [t for t in trials if sealed["assignments"][int(t["trial"])] == wanted]
        seated = sum(t["outcome"] == "success" for t in arm)
        rate = 100.0 * seated / len(arm) if arm else 0.0
        print(f"| {label} | {len(arm)} | {seated} | {rate:.1f}% |")

    guessed = [t for t in trials if t["guess"] != "unsure"]
    correct = sum((t["guess"] == "on") == sealed["assignments"][int(t["trial"])] for t in guessed)
    if guessed:
        print(
            f"\nBlinding: {correct}/{len(guessed)} guesses correct "
            f"({100.0 * correct / len(guessed):.0f}%) — 50% is chance."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-calib", required=False, default=None)
    parser.add_argument(
        "--cameras",
        nargs=2,
        default=["1", "2"],
        help="left/right camera indices. Defaults to the dev rig's pairing (1 2, the "
        "built-in webcam being 0); verify with stereohand's scripts/view_cameras.py, as "
        "Windows can renumber devices across reboots",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trials", type=int, default=32, help="recorded trials (default 32)")
    parser.add_argument(
        "--practice",
        type=int,
        default=4,
        help="unrecorded warm-up trials run first and discarded, declared in advance so "
        "dropping them is not a choice made after seeing them (default 4)",
    )
    parser.add_argument(
        "--cam",
        choices=["main", "wrist"],
        default="wrist",
        help="Viewer camera for every trial: 'wrist' the robot's-eye POV (default, what "
        "the protocol's fixed parameters assume since the 2026-08-20 amendment) or "
        "'main' for the free camera. The operator's view changes the task's difficulty, "
        "so it is held constant across a session and recorded in assignments.json rather "
        "than changed trial to trial.",
    )
    parser.add_argument(
        "--cam-window",
        dest="cam_window",
        action="store_true",
        help="show the operator's camera preview during trials. Off by default since the "
        "2026-08-20 amendment: the cv2 pump runs at 30 Hz inside the control loop and cost "
        "~1.4 ms/step measured (0.55x real-time with it, ~0.90x without). It also silences "
        "--record-hand, since the operator-side video is written from the frame the preview "
        "composites -- that footage is now taken in separate free-play takes instead.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/blind_trial"))
    parser.add_argument("--schedule-seed", type=int, default=20260804)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--max-dpos", type=float, default=None)
    parser.add_argument(
        "--unseal", action="store_true", help="report the finished session and stop"
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)
    # run_trial reads the negative form (shared with vision_play.py, which has no flag).
    arguments.no_cam_window = not arguments.cam_window

    if arguments.unseal:
        unseal(arguments.out)
        return
    if not arguments.stereo_calib:
        parser.error("--stereo-calib is required (or pass --unseal to report a finished session)")
    if not arguments.checkpoint.exists():
        parser.error(f"checkpoint not found: {arguments.checkpoint}")

    arguments.out.mkdir(parents=True, exist_ok=True)
    assignments = schedule(arguments.trials, arguments.schedule_seed)
    (arguments.out / "assignments.json").write_text(
        json.dumps(
            {
                "checkpoint": str(arguments.checkpoint),
                "schedule_seed": arguments.schedule_seed,
                "practice": arguments.practice,
                "cam": arguments.cam,
                "assignments": assignments,
            },
            indent=2,
        )
    )
    log.info(
        "sealed schedule → %s. Do NOT open it until the session is done.",
        arguments.out / "assignments.json",
    )

    for i in range(arguments.practice):
        log.info("--- practice %d/%d (discarded) ---", i + 1, arguments.practice)
        run_trial(i, bool(i % 2), 9000 + i, arguments.out / "practice", arguments)

    results_path = arguments.out / "trials.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "wall_seed", "outcome", "guess"])
        for i, assist_on in enumerate(assignments):
            log.info("--- trial %d/%d ---", i + 1, arguments.trials)
            outcome = run_trial(i, assist_on, i, arguments.out, arguments)
            log.info("outcome: %s", outcome)
            answer = ""
            while answer not in GUESSES:
                answer = input("Was the assist on? [y/n/u] ").strip().lower()[:1]
            writer.writerow([i, i, outcome, GUESSES[answer]])
            handle.flush()  # a session interrupted halfway still has every trial so far

    log.info("done — %s. Report with: --unseal --out %s", results_path, arguments.out)


if __name__ == "__main__":
    main()
