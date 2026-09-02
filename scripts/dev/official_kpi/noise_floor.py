"""The training-seed noise floor for **every** reported KPI, not just the success rate.

The project rejects a success-rate effect when it is smaller than the spread that retraining
the *same recipe* under a different training seed produces. That test is only meaningful if
it is applied to every metric a claim is made about — otherwise a continuous KPI can be
reported as a finding under a standard the headline metric would fail.

This prints the floor per recipe per metric: the range of the paired Δ across that recipe's
training seeds. A treatment effect smaller than its own recipe family's floor is not
established, whatever its sign or its p-value.

Read-only. Run from kevin/:  uv run python scripts/dev/official_kpi/noise_floor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kpi_data import (
    BASELINE_CONFIG,
    DEFAULT_RUNS_ROOT,
    METRICS,
    MetricSpec,
    Recipe,
    load_recipes,
)
from plot_kpis import RECIPE_ORDER

from ai_teleop.common.log import (
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("noise-floor")


def paired_deltas(recipe: Recipe, metric: MetricSpec) -> list[float]:
    """One paired Δ (treatment − baseline) per training seed, over the matched trials."""
    deltas = []
    for point in recipe.points:
        baseline = {trial.seed: trial for trial in point.baseline_trials}
        if metric.continuous_index is None:  # the success rate is a rate, not a per-trial mean
            treated = 100.0 * sum(t.success for t in point.treatment_trials)
            treated /= len(point.treatment_trials)
            base = 100.0 * sum(t.success for t in point.baseline_trials)
            base /= len(point.baseline_trials)
            deltas.append(treated - base)
            continue
        pairs = [
            (getattr(trial, metric.key), getattr(baseline[trial.seed], metric.key))
            for trial in point.treatment_trials
            if trial.seed in baseline
        ]
        usable = [(a, b) for a, b in pairs if a is not None and b is not None]
        if usable:
            deltas.append(mean(a - b for a, b in usable))
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    recipes = load_recipes(RECIPE_ORDER, arguments.runs_root)
    ordered = [recipes[label] for label in RECIPE_ORDER if label in recipes]
    if not ordered:
        log.error("no finished official evals under %s", arguments.runs_root)
        raise SystemExit(1)

    for metric in METRICS:
        if metric.key == "contact_events":
            continue
        print(f"\n### {metric.axis_label} — paired Δ vs {BASELINE_CONFIG}, per training seed\n")
        print("| Recipe | seeds | per-seed Δ | mean | **floor (range)** |")
        print("|---|---|---|---|---|")
        for recipe in ordered:
            deltas = paired_deltas(recipe, metric)
            if not deltas:
                continue
            draws = ", ".join(f"{d:+.2f}" for d in sorted(deltas))
            span = max(deltas) - min(deltas)
            print(
                f"| {recipe.label} | {len(deltas)} | {draws} | {mean(deltas):+.2f} | **{span:.2f}** |"
            )


if __name__ == "__main__":
    main()
