"""Per-arm KPIs for the blinded human trial — protocol §6's secondary reporting.

§6 fixes the analysis in advance: success rate per arm with Wilson intervals and the
difference with its interval (primary, **no p-value**), then time-to-insert, peak contact
force and trajectory jerk per arm as *descriptive only*, then the blinding check. This
computes all of it from the session on disk.

The KPIs are recovered by replaying each trial through the **canonical**
:class:`TrialObserver` rather than a fast reimplementation, so these numbers are computed
by the same calculator that produced every other KPI in the project. `replay_kpis` itself
cannot be used: it reads the eval-trace schema (`base_cmd_*`), while the blind-trial
harness records episodes (`cmd_*`). The observer never reads the command — only the
observation — so reconstructing the observation stream is sufficient and exact.

Reads `assignments.json`, so run it only after the session is finished and unsealed.

    uv run python scripts/dev/trial_kpis.py --out runs/blind_trial
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_teleop.common import Command, Observation  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.domain import Delta  # noqa: E402
from ai_teleop.eval.observer import TrialObserver  # noqa: E402
from ai_teleop.eval.schema import TrialKPIs  # noqa: E402

log = get_logger("lab120-kpis")

Z = 1.96


def wilson(successes: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval — §6's specified interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(on: tuple[int, int], off: tuple[int, int]) -> tuple[float, float, float]:
    """Difference of two proportions with Newcombe's Wilson-based interval.

    The textbook normal-approximation interval is unusable at n=16 per arm; Newcombe
    composes the two Wilson intervals instead and behaves at small n and extreme rates,
    which is why §6 asks for Wilson in the first place.
    """
    x_on, n_on = on
    x_off, n_off = off
    p_on, p_off = x_on / n_on, x_off / n_off
    low_on, high_on = wilson(x_on, n_on)
    low_off, high_off = wilson(x_off, n_off)
    difference = p_on - p_off
    lower = difference - math.sqrt((p_on - low_on) ** 2 + (high_off - p_off) ** 2)
    upper = difference + math.sqrt((high_on - p_on) ** 2 + (p_off - low_off) ** 2)
    return difference, lower, upper


def kpis_for_episode(path: Path, index: int) -> tuple[TrialKPIs, dict[str, Any], float | None]:
    """Observer KPIs, the episode's own metadata, and time-to-insert.

    Three sources, deliberately: the observer owns force / jerk / near-miss, the episode's
    ``terminal_reason`` owns the outcome, and time-to-insert is read off the trajectory.

    The observer's *outcome* is unusable here and the reason is structural, not a bug.
    ``TrialObserver`` requires seating to hold for ``sustained_duration_s`` (0.05 s) before
    it records SUCCESS, while ``run_episode`` terminates the instant the seating criterion
    is met -- ``step_success`` fires exactly once, on the final step. The sustained window
    can therefore never elapse inside a recorded episode, so every seated trial replays as
    the observer's default TIMEOUT. Feeding these episodes to the ablation harness's
    calculator asks it a question the recording cannot answer.
    """
    with np.load(path, allow_pickle=True) as data:
        columns = {key: data[key] for key in data.files if key != "metadata"}
        metadata = data["metadata"].item()
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    observer = TrialObserver(
        seed=index,
        success_depth=metadata["success_depth"],
        lateral_tolerance=metadata["lateral_tolerance"],
        force_cap=metadata["force_cap"],
    )
    for step in range(int(columns["step"].shape[0])):
        observation = Observation(
            joint_positions=columns["joint_positions"][step],
            joint_velocities=columns["joint_velocities"][step],
            ee_pose=columns["ee_pose"][step],
            wrist_ft=columns["wrist_ft"][step],
            gripper_width=float(columns["gripper_width"][step]),
            peg_pose=columns["peg_pose"][step],
            # Only the target hole is logged, so the observer reads this single row.
            hole_poses=columns["target_hole_pose"][step][None, :],
            sim_time=float(columns["sim_time"][step]),
        )
        # The observer reads neither of these; they exist to satisfy the callback shape.
        base_command = Command(
            columns["cmd_position"][step],
            columns["cmd_quaternion"][step],
            float(columns["cmd_grip"][step]),
        )
        delta = Delta(
            columns["delta_position"][step],
            columns["delta_orientation"][step],
            float(columns["delta_grip"][step]),
        )
        if observer(int(columns["step"][step]), observation, base_command, delta, command=None):
            break

    # The episode stops at seating, so the sim-time of the one `step_success` tick *is* the
    # time to insert. None when the trial never seated.
    seated = np.flatnonzero(np.asarray(columns["step_success"]) > 0)
    time_to_insert = float(columns["sim_time"][seated[0]]) if seated.size else None
    return observer.result(), metadata, time_to_insert


def _summarise(values: list[float]) -> str:
    if not values:
        return "n/a"
    array = np.asarray(values, dtype=float)
    return f"{array.mean():.2f} (median {np.median(array):.2f}, n={len(array)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/blind_trial"))
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    session = arguments.out
    rows = list(csv.DictReader((session / "trials.csv").open()))
    sealed = json.loads((session / "assignments.json").read_text(encoding="utf-8"))
    assignments: list[bool] = sealed["assignments"]

    per_trial: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["trial"])
        kpis, metadata, time_to_insert = kpis_for_episode(
            session / f"trial_{index:03d}" / "episode.npz", index
        )
        per_trial.append({
            "trial": index,
            "assist_on": bool(assignments[index]),
            "wall_seed": int(row["wall_seed"]),
            # The outcome of record: run_episode's own terminal_reason, the same
            # determination every interactive episode in this project uses.
            "outcome": row["outcome"],
            "guess": row["guess"],
            "time_to_insert_s": time_to_insert,
            "duration_s": kpis.duration_s,
            # Kept for the instrument note below -- not the outcome of record.
            "observer_outcome": kpis.outcome.value,
            "kpis": kpis.to_dict(),
        })
        assert metadata["terminal_reason"] == row["outcome"], (
            f"trial {index}: trials.csv and the episode disagree on the outcome"
        )

    (session / "kpis.json").write_text(json.dumps(per_trial, indent=2) + "\n", encoding="utf-8")

    # Flat per-trial table, matching docs/results/phase-1/official-evals/: one row per
    # trial, every column the published analysis reads. This is the committed artifact --
    # the analysis below is reproducible from it without the trajectories.
    columns = [
        "trial",
        "assist_on",
        "wall_seed",
        "outcome",
        "guess",
        "time_to_insert_s",
        "duration_s",
        "peak_contact_force",
        "jerk_integral",
        "contact_events",
        "min_tip_hole_distance_mm",
        "penetration_at_closest_mm",
        "lateral_error_at_closest_mm",
        "n_steps",
        "observer_outcome",
    ]
    with (session / "kpis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in per_trial:
            flat = {**entry["kpis"], **{k: v for k, v in entry.items() if k != "kpis"}}
            writer.writerow({name: flat.get(name) for name in columns})

    on = [entry for entry in per_trial if entry["assist_on"]]
    off = [entry for entry in per_trial if not entry["assist_on"]]

    print(f"\n## Primary — success rate ({len(per_trial)} trials)\n")
    print("| Arm | successes | rate | Wilson 95% |")
    print("|---|---|---|---|")
    counts = {}
    for label, arm in (("assist ON", on), ("assist OFF", off)):
        seated = sum(entry["outcome"] == "success" for entry in arm)
        counts[label] = (seated, len(arm))
        low, high = wilson(seated, len(arm))
        print(
            f"| {label} | {seated}/{len(arm)} | {seated / len(arm):.1%} | [{low:.1%}, {high:.1%}] |"
        )
    difference, lower, upper = newcombe(counts["assist ON"], counts["assist OFF"])
    print(
        f"\n**Difference {100 * difference:+.1f} pp**, "
        f"Newcombe 95% [{100 * lower:+.1f}, {100 * upper:+.1f}] pp. "
        "No p-value, per §6.1."
    )

    print("\n## Secondary — descriptive only (§6.2)\n")
    print("| KPI | assist ON | assist OFF |")
    print("|---|---|---|")
    # (label, key, from the entry itself rather than the observer block)
    for label, key, top_level in (
        ("Time to insert (s), seated trials only", "time_to_insert_s", True),
        ("Episode duration (s), all trials", "duration_s", True),
        ("Peak contact force (N)", "peak_contact_force", False),
        ("Trajectory jerk (∫|jerk| dt)", "jerk_integral", False),
        ("Closest tip–hole distance (mm)", "min_tip_hole_distance_mm", False),
        ("Lateral error at closest (mm)", "lateral_error_at_closest_mm", False),
    ):
        cells = []
        for arm in (on, off):
            source = (entry if top_level else entry["kpis"] for entry in arm)
            values = [block[key] for block in source if block[key] is not None]
            cells.append(_summarise(values))
        print(f"| {label} | {cells[0]} | {cells[1]} |")
    print(
        "\nJerk is an integral over the episode, so read it against the duration row "
        "rather than on its own."
    )

    print("\n## Outcomes\n")
    print("| Arm | " + " | ".join(sorted({e["outcome"] for e in per_trial})) + " |")
    print("|---|" + "---|" * len({e["outcome"] for e in per_trial}))
    outcomes = sorted({e["outcome"] for e in per_trial})
    for label, arm in (("assist ON", on), ("assist OFF", off)):
        tally = Counter(entry["outcome"] for entry in arm)
        print(f"| {label} | " + " | ".join(str(tally.get(name, 0)) for name in outcomes) + " |")

    print("\n## Blinding check (§6.3)\n")
    guessed = [entry for entry in per_trial if entry["guess"] != "unsure"]
    correct = sum((entry["guess"] == "on") == entry["assist_on"] for entry in guessed)
    low, high = wilson(correct, len(guessed))
    tally = Counter(entry["guess"] for entry in per_trial)
    print(
        f"{correct}/{len(guessed)} guesses correct = {correct / len(guessed):.0%}, "
        f"Wilson 95% [{low:.0%}, {high:.0%}] against a 50% chance line.  "
        f"Guesses: {tally.get('on', 0)} on / {tally.get('off', 0)} off / "
        f"{tally.get('unsure', 0)} unsure."
    )
    disagreements = Counter(
        (entry["outcome"], entry["observer_outcome"])
        for entry in per_trial
        if entry["outcome"] != entry["observer_outcome"]
    )
    if disagreements:
        print("\n## Instrument note — the observer cannot score these episodes\n")
        print(
            "`TrialObserver` requires seating to hold for 0.05 s before it records SUCCESS. "
            "`run_episode` terminates the instant the seating criterion is met, so "
            "`step_success` fires once, on the final step, and the sustained window never "
            "elapses inside the recording. Every seated trial therefore replays as the "
            "observer's default TIMEOUT.\n"
        )
        print("| harness `terminal_reason` | observer verdict | trials |")
        print("|---|---|---|")
        for (harness, replayed), count in sorted(disagreements.items()):
            print(f"| {harness} | {replayed} | {count} |")
        print(
            "\nThe outcome of record is the harness's, which is what every interactive "
            "episode in this project uses. Force, jerk and the near-miss trio are "
            "unaffected -- they accumulate over the observation stream and do not depend "
            "on the verdict. **Cross-study caveat:** the scripted results were scored by "
            "the observer's sustained criterion, so these rates are not directly "
            "comparable to them. Within this session both arms are scored identically, so "
            "the comparison that matters here is unaffected."
        )
    log.info("per-trial KPIs → %s", session / "kpis.json")


if __name__ == "__main__":
    main()
