"""Blinded A/B session: does the trained residual help a *human* operator?

Every KPI in this project was measured against `ScriptedNoisyHuman`, a model of an
operator. This is the only place a real one closes the loop. Each trial runs one live
teleoperation episode with the residual's output either applied or discarded, chosen by a
hidden schedule, and the operator does not learn which until the session is unsealed.

**Read `docs/results/human-trial-protocol.md` before running.** It fixes the checkpoint,
the trial count and the analysis in advance, and states what this design can and cannot
resolve — at 30 trials per arm the smallest difference it can detect is ~33 pp, while the
scripted study's effects are single-digit pp. It is a proxy-validity check and a source of
unselected footage, not a hypothesis test.

How the blind holds:

- Both arms run `--policy tf --assist-scale {1,0}`, never `--policy noassist`. Scale 0
  still loads the checkpoint, still runs the forward pass, still captures wrist frames —
  it only discards the Δ. Startup time and per-step cost are identical, so the arm cannot
  be inferred from how the session behaves.
- The child's stdout/stderr go to a per-trial `console.log`, so no line naming the policy
  reaches the terminal.
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
    uv run python scripts/dev/blind_trial.py --practice 10 --trials 60
    uv run python scripts/dev/blind_trial.py --unseal      # after every trial is done
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data import load_episode  # noqa: E402

log = get_logger("blind-trial")

# Pre-specified in the protocol, NOT selected on outcome. FT-DAgger is the family whose
# success rate moves least across training seeds (2 pp, against 27-31 pp for plain BC), so
# it is the arm least dependent on the seed lottery — a reason that stands without looking
# at any human data. Picking the best-scoring checkpoint instead is the selection bias the
# project already documented (docs/self-evaluation.md, "pre-register the reporting rule").
DEFAULT_CHECKPOINT = Path("docs/results/checkpoints/ft/dagger/seed_0/round_2/checkpoint.pt")

BLOCK = 4  # trials per randomization block; must be even so each block splits evenly
GUESSES = {"y": "on", "n": "off", "u": "unsure"}


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


def run_trial(
    index: int, assist_on: bool, wall_seed: int, out_dir: Path, args: argparse.Namespace
) -> str:
    """Run one episode as a child process; return its terminal reason.

    The child's output is redirected wholesale rather than filtered — a filter has to
    anticipate every line that could name the policy, a redirect does not.
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
        "--record",
        "commands",
        "--record-out",
        str(trial_dir),
        "--record-hand",
        str(trial_dir / "hand.mp4"),
    ]
    if args.gain is not None:
        command += ["--gain", str(args.gain)]
    if args.max_dpos is not None:
        command += ["--max-dpos", str(args.max_dpos)]

    with (trial_dir / "console.log").open("w") as console:
        subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=False)

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
    parser.add_argument("--cameras", nargs=2, default=["0", "1"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trials", type=int, default=60, help="recorded trials (default 60)")
    parser.add_argument(
        "--practice",
        type=int,
        default=10,
        help="unrecorded warm-up trials run first and discarded, declared in advance so "
        "dropping them is not a choice made after seeing them (default 10)",
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
