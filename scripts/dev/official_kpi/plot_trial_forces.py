"""Per-trial KPI distributions, pooled over training seeds — what the means average over.

Every other figure in this directory plots a distribution over **training seeds**, where one
point is a seed's *mean over 100 trials*. That answers "how much does retraining move the
average?". This one plots the **individual trials**, pooled across every checkpoint of a recipe,
which answers a different question: how much does one insertion attempt differ from the next,
and does the mean describe any of them?

Often it does not. Peak contact force is **bimodal** — a seated cluster around 15 N and a
force-abort cluster around 37 N — so the ~24 N mean sits in the empty space between them and
describes no trial that actually ran. Jerk is bimodal for a different reason (a run that aborts
early accumulates less corrective motion), and time-to-insert exists only on trials that seated.

Pooling is deliberate here and **only** legitimate for this question. A recipe's trials are
5 clusters of 100 sharing a checkpoint, not 500 independent draws, so nothing on these figures
is a recipe-level effect estimate — for that, the correct n is the number of training seeds, and
the seed-level figures own it. These figures describe trial-level *structure*: where the modes
sit, and where the thresholds fall relative to them.

Read-only. Run from kevin/:  uv run python scripts/dev/official_kpi/plot_trial_forces.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kpi_data import (  # noqa: E402
    BASELINE_CONFIG,
    DEFAULT_RUNS_ROOT,
    METRICS,
    MetricSpec,
    load_recipes,
)
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

log = get_logger("trial-kpis")

# Stiffness (N/m) on the translational axes and the per-step command clamp (m), both from
# `control/backbone.py`. Their product is the largest restoring force the controller can
# *command* — a bound on the command, which is not the same thing as a bound on the
# measured contact reaction.
STIFFNESS_TCP = (400.0, 400.0, 500.0)
MAX_DPOS_PER_STEP = 0.025
COMMAND_FORCE_BOUND = sum((k * MAX_DPOS_PER_STEP) ** 2 for k in STIFFNESS_TCP) ** 0.5

# The eval observer aborts a trial whose contact force exceeds this (`eval/observer.py`).
FORCE_ABORT_THRESHOLD = 30.0

# Marked as vertical reference lines on the force panels only — no other KPI has a
# threshold the controller or the harness enforces.
FORCE_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (COMMAND_FORCE_BOUND, "#1a9641", "-"),
    (FORCE_ABORT_THRESHOLD, "#000000", "--"),
)

# Values span three decades, so a linear axis collapses every panel into one bar.
LOG_SCALE_METRICS = frozenset({"jerk_integral"})

OUTCOME_COLORS = {
    "success": "#2c7bb6",
    "timeout": "#999999",
    "force_abort": "#d7191c",
}

# The continuous KPIs, in reporting order; the success rate has no per-trial distribution.
TRIAL_METRICS = tuple(metric for metric in METRICS if metric.continuous_index is not None)


def trials_by_arm(recipes: list, metric: MetricSpec) -> dict[str, list[tuple[float, str]]]:
    """Every trial's (metric value, outcome), keyed by arm label.

    `human_only` is pooled across every eval directory: it uses no checkpoint, so those
    are repeated runs of one arm rather than different arms, and pooling them is what
    gives its per-trial distribution its full sample. A success-only KPI is undefined on
    a miss and stored as ``None`` there — those trials are dropped, so each panel's n is
    the number of trials on which the KPI exists, not the number that ran.
    """
    collected: dict[str, list[tuple[float, str]]] = {}

    def add(label: str, trials) -> None:
        collected.setdefault(label, []).extend(
            (float(value), str(trial.outcome))
            for trial in trials
            if (value := getattr(trial, metric.key)) is not None
        )

    for recipe in recipes:
        for point in recipe.points:
            add(BASELINE_CONFIG, point.baseline_trials)
            add(recipe.label, point.treatment_trials)
    return collected


def bin_edges(values: list[float], metric: MetricSpec, bins: int = 36) -> np.ndarray:
    """Bin edges spanning every arm's data, log-spaced where the metric is log-scaled."""
    low, high = min(values), max(values)
    if metric.key in LOG_SCALE_METRICS:
        return np.logspace(np.log10(max(low, 1e-3)), np.log10(high), bins + 1)
    # Force keeps its historical fixed window so the two thresholds land in the same place
    # on every panel; other linear metrics span their own data with a little headroom.
    if metric.key == "peak_contact_force":
        return np.linspace(0.0, 80.0, bins + 1)
    return np.linspace(0.0, high * 1.05, bins + 1)


def draw_panel(
    axes: Axes,
    values: list[tuple[float, str]],
    title: str,
    metric: MetricSpec,
    edges: np.ndarray,
    *,
    note: str = "",
) -> None:
    """One arm's per-trial histogram, stacked by outcome, with any thresholds marked."""
    outcomes = [name for name in OUTCOME_COLORS if any(o == name for _, o in values)]
    axes.hist(
        [[value for value, outcome in values if outcome == name] for name in outcomes],
        bins=edges,
        stacked=True,
        color=[OUTCOME_COLORS[name] for name in outcomes],
        label=[f"{name} (n={sum(1 for _, o in values if o == name)})" for name in outcomes],
        edgecolor="white",
        linewidth=0.3,
    )
    if metric.key == "peak_contact_force":
        for level, color, style in FORCE_THRESHOLDS:
            axes.axvline(level, color=color, linestyle=style, linewidth=1.7)
    if metric.key in LOG_SCALE_METRICS:
        axes.set_xscale("log")
    observed = [value for value, _ in values]
    axes.set_title(
        f"{title}\nmean {mean(observed):.1f} · min {min(observed):.1f} · "
        f"max {max(observed):.1f} · SD {pstdev(observed):.1f}{note}",
        fontsize=11,
    )
    axes.set_xlabel(metric.axis_label)
    axes.set_ylabel("trials")
    axes.legend(fontsize=8, frameon=False)
    axes.grid(True, alpha=0.4)
    axes.set_axisbelow(True)


def caption(metric: MetricSpec) -> str:
    """The figure's subtitle — the thresholds for force, the sampling rule elsewhere."""
    if metric.key == "peak_contact_force":
        return (
            f"green: commanded-force bound {COMMAND_FORCE_BOUND:.1f} N "
            f"(stiffness × command clamp)   ·   "
            f"black dashed: {FORCE_ABORT_THRESHOLD:.0f} N watchdog abort"
        )
    if metric.success_only:
        return "defined only on trials that seated — each panel's n is its seated count"
    if metric.key in LOG_SCALE_METRICS:
        return "logarithmic axis — the values span three decades"
    return metric.direction_note


def plot_trial_metric(recipes: list, metric: MetricSpec, path: Path) -> Path:
    collected = trials_by_arm(recipes, metric)
    arms = [BASELINE_CONFIG] + [r.label for r in recipes]
    edges = bin_edges([value for arm in arms for value, _ in collected[arm]], metric)
    columns = 3
    rows = -(-len(arms) // columns)
    figure: Figure
    figure, grid = plt.subplots(rows, columns, figsize=(6.0 * columns, 4.4 * rows), squeeze=False)
    flat = list(grid.ravel())
    # The baseline uses no checkpoint, so its pooled sample is one eval repeated once per
    # checkpoint — identical walls, identical values. Saying so on the panel stops its n
    # from being read as an independent sample the policy arms do not have.
    repeats = sum(len(recipe.points) for recipe in recipes)
    for axes, arm in zip(flat, arms, strict=False):
        note = (
            f"\n{repeats} identical repeats of {len(collected[arm]) // repeats} trials"
            if arm == BASELINE_CONFIG
            else ""
        )
        draw_panel(axes, collected[arm], arm, metric, edges, note=note)
    for axes in flat[len(arms) :]:
        axes.set_axis_off()
    figure.suptitle(
        f"{metric.label} per trial, pooled over training seeds — "
        f"what the recipe mean averages over\n{caption(metric)}",
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

    for metric in TRIAL_METRICS:
        collected = trials_by_arm(ordered, metric)
        log.info("--- %s", metric.axis_label)
        for arm, values in collected.items():
            observed = [value for value, _ in values]
            counts = Counter(outcome for _, outcome in values)
            log.info(
                "%-22s n=%4d  mean %8.2f  min %8.2f  max %9.2f  SD %8.2f  %s",
                arm,
                len(observed),
                mean(observed),
                min(observed),
                max(observed),
                pstdev(observed),
                dict(counts),
            )
        written = plot_trial_metric(
            ordered, metric, arguments.figure_dir / f"trial_{metric.key}_distribution.png"
        )
        log.info("wrote %s", written)


if __name__ == "__main__":
    main()
