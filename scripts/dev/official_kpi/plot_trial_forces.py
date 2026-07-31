"""Per-trial peak contact force, by arm and by outcome — the distribution behind the mean.

Every other figure in this directory plots a distribution over **training seeds**, where one
point is a seed's *mean over 100 trials*. That answers "how much does retraining move the
average?". This one plots the **individual trials**, which answers a different question: how
much does one insertion attempt differ from the next, and does the mean describe any of them?

It does not. Peak contact force is **bimodal** — a seated cluster around 15 N and a
force-abort cluster around 37 N — so the ~24 N mean sits in the empty space between them and
describes no trial that actually ran. The figure also places the two thresholds that matter:
the impedance controller's commanded-force bound (stiffness × the per-step command clamp) and
the watchdog's abort threshold.

Read-only. Run from kevin/:  uv run python scripts/dev/official_kpi/plot_trial_forces.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kpi_data import BASELINE_CONFIG, DEFAULT_RUNS_ROOT, load_recipes  # noqa: E402
from plot_kpis import (  # noqa: E402
    DEFAULT_FIGURE_DIR,
    RECIPE_ORDER,
    apply_plain_style,
    finish,
)

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("trial-forces")

# Stiffness (N/m) on the translational axes and the per-step command clamp (m), both from
# `control/backbone.py`. Their product is the largest restoring force the controller can
# *command* — a bound on the command, which is not the same thing as a bound on the
# measured contact reaction.
STIFFNESS_TCP = (400.0, 400.0, 500.0)
MAX_DPOS_PER_STEP = 0.025
COMMAND_FORCE_BOUND = sum((k * MAX_DPOS_PER_STEP) ** 2 for k in STIFFNESS_TCP) ** 0.5

# The eval observer aborts a trial whose contact force exceeds this (`eval/observer.py`).
FORCE_ABORT_THRESHOLD = 30.0

OUTCOME_COLORS = {
    "success": "#2c7bb6",
    "timeout": "#999999",
    "force_abort": "#d7191c",
}


def trials_by_arm(recipes: list) -> dict[str, list[tuple[float, str]]]:
    """Every trial's (peak force, outcome), keyed by arm label.

    `human_only` is pooled across every eval directory: it uses no checkpoint, so those
    are repeated runs of one arm rather than different arms, and pooling them is what
    gives its per-trial distribution its full sample.
    """
    collected: dict[str, list[tuple[float, str]]] = {}
    for recipe in recipes:
        for point in recipe.points:
            collected.setdefault(BASELINE_CONFIG, []).extend(
                (float(t.peak_contact_force), str(t.outcome)) for t in point.baseline_trials
            )
            collected.setdefault(recipe.label, []).extend(
                (float(t.peak_contact_force), str(t.outcome)) for t in point.treatment_trials
            )
    return collected


def draw_panel(axes, values: list[tuple[float, str]], title: str) -> None:
    """One arm's per-trial histogram, stacked by outcome, with both thresholds marked."""
    outcomes = [name for name in OUTCOME_COLORS if any(o == name for _, o in values)]
    axes.hist(
        [[force for force, outcome in values if outcome == name] for name in outcomes],
        bins=36,
        range=(0.0, 80.0),
        stacked=True,
        color=[OUTCOME_COLORS[name] for name in outcomes],
        label=[f"{name} (n={sum(1 for _, o in values if o == name)})" for name in outcomes],
        edgecolor="white",
        linewidth=0.3,
    )
    axes.axvline(COMMAND_FORCE_BOUND, color="#1a9641", linestyle="-", linewidth=1.8)
    axes.axvline(FORCE_ABORT_THRESHOLD, color="#000000", linestyle="--", linewidth=1.6)
    forces = [force for force, _ in values]
    axes.set_title(
        f"{title}\nmean {mean(forces):.1f} N · min {min(forces):.1f} · "
        f"max {max(forces):.1f} · SD {pstdev(forces):.1f}",
        fontsize=11,
    )
    axes.set_xlabel("Peak contact force (N)")
    axes.set_ylabel("trials")
    axes.legend(fontsize=8, frameon=False)
    axes.grid(True, alpha=0.4)
    axes.set_axisbelow(True)


def plot_trial_forces(recipes: list, path: Path) -> Path:
    collected = trials_by_arm(recipes)
    arms = [BASELINE_CONFIG] + [r.label for r in recipes]
    columns = 3
    rows = -(-len(arms) // columns)
    figure: Figure
    figure, grid = plt.subplots(rows, columns, figsize=(6.0 * columns, 4.4 * rows), squeeze=False)
    flat = list(grid.ravel())
    for axes, arm in zip(flat, arms, strict=False):
        draw_panel(axes, collected[arm], arm)
    for axes in flat[len(arms) :]:
        axes.set_axis_off()
    figure.suptitle(
        "Peak contact force per trial — the mean describes no trial that ran\n"
        f"green: commanded-force bound {COMMAND_FORCE_BOUND:.1f} N (stiffness × command clamp)   "
        f"·   black dashed: {FORCE_ABORT_THRESHOLD:.0f} N watchdog abort",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return finish(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)
    apply_plain_style()

    recipes = load_recipes(RECIPE_ORDER, arguments.runs_root)
    ordered = [recipes[label] for label in RECIPE_ORDER if label in recipes]
    if not ordered:
        log.error("no finished official evals under %s", arguments.runs_root)
        raise SystemExit(1)

    collected = trials_by_arm(ordered)
    for arm, values in collected.items():
        forces = [f for f, _ in values]
        counts = Counter(o for _, o in values)
        log.info(
            "%-22s n=%4d  mean %5.2f  min %5.2f  max %5.2f  SD %5.2f  %s",
            arm,
            len(forces),
            mean(forces),
            min(forces),
            max(forces),
            pstdev(forces),
            dict(counts),
        )
    written = plot_trial_forces(ordered, arguments.figure_dir / "trial_force_distribution.png")
    log.info("wrote %s", written)


if __name__ == "__main__":
    main()
