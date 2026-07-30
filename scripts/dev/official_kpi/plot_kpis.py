"""The official multi-seed run as figures — plain matplotlib interval charts.

Four figures, all under ``docs/results/phase-1/``:

1. ``success_rate_spread.png`` — the headline. Paired Δ success rate per recipe: the
   mean over training seeds, whiskers to the lowest and highest seed, one dot per seed.
2. ``kpi_spread_by_recipe.png`` — the same interval chart for **all five** metrics, in
   absolute units, against the ``human_only`` line.
3. ``dagger_vs_plain.png`` — per modality and per metric, the three arms side by side:
   ``human_only``, plain BC, DAgger.
4. ``dagger_rounds_vision.png`` — every metric as a function of DAgger round, one line
   per training seed (vision only — the F/T arm has no per-round eval).

Deliberately conventional: ``Axes.errorbar`` with asymmetric min/max whiskers, white
background, black text, no bespoke visual language. The whiskers are the **observed
range over training seeds**, not a confidence interval — with 3–5 seeds a standard error
would imply a precision the run does not have, and the range is the quantity the LAB-114
noise-floor argument is actually about.

Statistics are not computed here; :mod:`kpi_data` reads them out of
:mod:`ai_teleop.eval.report`. This module only lays them out.

Run from kevin/:  uv run python scripts/dev/official_kpi/plot_kpis.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kpi_data import (  # noqa: E402
    DEFAULT_POLICY_RUNS_ROOT,
    DEFAULT_RUNS_ROOT,
    METRICS,
    SUCCESS_METRIC,
    Recipe,
    Spread,
    baseline_spread,
    compare_recipes,
    cross_recipe_spread,
    delta_spread,
    load_recipes,
    load_vision_dagger_rounds,
    marginal_value,
    treatment_spread,
)

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("official-kpi-figures")

DEFAULT_FIGURE_DIR = Path("docs/results/phase-1")

RECIPE_ORDER = (
    "FT plain",
    "FT plain (batch 2)",
    "FT DAgger",
    "Vision plain",
    "Vision DAgger",
)
# (modality, arm labels in plotting order, DAgger recipe) — the §3 comparison.
DAGGER_ARMS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("F/T", ("FT plain", "FT plain (batch 2)"), "FT DAgger"),
    ("Vision", ("Vision plain",), "Vision DAgger"),
)

# Plain, high-contrast defaults: a GitHub reader may be on either theme, so the figure
# paints an opaque white surface with black text instead of going transparent.
MARKER_COLOR = "#1f4e9c"
SEED_COLOR = "#8a8a8a"
BASELINE_COLOR = "#b03030"
# How far left of an interval the individual training seeds are drawn (x data units).
SEED_OFFSET = 0.17
# Fraction of the data range left free above/below so annotations never hit the frame.
HEADROOM = 0.16


def apply_plain_style() -> None:
    """Conventional matplotlib on a white surface with black text."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.size": 11,
        "grid.color": "#cccccc",
    })


def whiskers(spreads: Sequence[Spread]) -> list[list[float]]:
    """``yerr`` for :meth:`Axes.errorbar` — distance from the mean down to the minimum
    and up to the maximum, i.e. the observed range drawn honestly rather than symmetrised.

    Clamped at zero only to absorb float error: a spread whose seeds are all identical
    (``human_only``, which uses no checkpoint) can compute a mean a machine epsilon below
    its own minimum, and ``errorbar`` rejects a negative ``yerr``.
    """
    return [
        [max(0.0, spread.mean - spread.minimum) for spread in spreads],
        [max(0.0, spread.maximum - spread.mean) for spread in spreads],
    ]


def draw_intervals(
    axes: Axes,
    positions: Sequence[float],
    spreads: Sequence[Spread],
    *,
    label: str | None = None,
    color: str = MARKER_COLOR,
    show_seeds: bool = True,
) -> None:
    """Mean marker + min/max whiskers at each x position, with the raw seeds beside them.

    The individual seeds sit a fixed step to the left of the interval rather than on it:
    a seed that happens to equal the mean, the minimum or the maximum would otherwise
    vanish under the marker, and the count of dots is what tells the reader how thin the
    distribution behind a mean actually is.
    """
    if show_seeds:
        for position, spread in zip(positions, spreads, strict=True):
            axes.plot(
                [position - SEED_OFFSET] * spread.n,
                list(spread.values),
                linestyle="none",
                marker="o",
                markersize=4,
                color=SEED_COLOR,
                alpha=0.9,
                zorder=2,
            )
    axes.errorbar(
        list(positions),
        [spread.mean for spread in spreads],
        yerr=whiskers(spreads),
        fmt="s",
        markersize=7,
        color=color,
        ecolor=color,
        elinewidth=1.6,
        capsize=7,
        capthick=1.6,
        linestyle="none",
        label=label,
        zorder=3,
    )


def annotate_counts(axes: Axes, positions: Sequence[float], spreads: Sequence[Spread]) -> None:
    """Print the per-seed trial count along the axis floor for a success-only KPI.

    Pinned to the axes floor rather than to the data, so the label cannot drift into the
    tick labels or into a neighbouring interval as the y-scale changes.
    """
    for position, spread in zip(positions, spreads, strict=True):
        text = (
            f"n={spread.min_count}"
            if spread.min_count == spread.max_count
            else f"n={spread.min_count}–{spread.max_count}"
        )
        axes.annotate(
            text,
            (position, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )


def flatten_degenerate_axis(axes: Axes, values: Sequence[float], note: str) -> bool:
    """When every value is identical, stop the autoscale inventing a fake ±0.05 y-range.

    ``contact_events`` is exactly 1 on every arm of every recipe — a real finding (the
    scoring counts one contact episode per trial), but left to autoscale it renders as a
    zoomed-in flatline that reads like noise. Returns whether the axis was flattened.
    """
    if not values or max(values) != min(values):
        return False
    level = values[0]
    axes.set_ylim(level - 1.0, level + 1.0)
    axes.annotate(
        note,
        (0.5, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    return True


def add_headroom(axes: Axes, *, top: float = HEADROOM, bottom: float = HEADROOM) -> None:
    """Widen the y limits by a fraction of the drawn range so annotations have room."""
    low, high = axes.get_ylim()
    span = high - low
    axes.set_ylim(low - bottom * span, high + top * span)


def finish(figure: Figure, path: Path) -> Path:
    """Save to ``path`` on an opaque white surface and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return path


def tick_labels(recipes: Sequence[Recipe]) -> list[str]:
    """Recipe name over its seed count and batch size — the context the mean needs."""
    return [f"{r.label}\nn={r.n_seeds} · batch {r.batch_size}" for r in recipes]


# ---------------------------------------------------------------------------
# Figure 1 — the headline: paired Δ success rate
# ---------------------------------------------------------------------------


def plot_success_rate_spread(recipes: Sequence[Recipe], path: Path) -> Path:
    """Paired Δ success rate per recipe, mean with min/max over training seeds."""
    spreads = [(r, delta_spread(r, SUCCESS_METRIC)) for r in recipes]
    usable = [(r, s) for r, s in spreads if s is not None]
    figure, axes = plt.subplots(figsize=(9.0, 5.5))
    positions = list(range(len(usable)))
    draw_intervals(axes, positions, [s for _, s in usable])
    axes.axhline(0.0, color=BASELINE_COLOR, linewidth=1.4, linestyle="--")
    axes.annotate(
        "human_only baseline (Δ = 0)",
        (-0.55, 0.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color=BASELINE_COLOR,
    )
    for position, (_, spread) in zip(positions, usable, strict=True):
        axes.annotate(
            f"mean {spread.mean:+.1f} pp\nrange {spread.width:.0f} pp",
            (position, spread.maximum),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(tick_labels([r for r, _ in usable]))
    axes.set_xlim(-0.6, len(usable) - 0.4)
    add_headroom(axes, top=0.22, bottom=0.06)
    axes.set_ylabel("Δ insertion success rate vs human_only (percentage points)")
    axes.set_title("Insertion success: every recipe's spread over training seeds swallows its mean")
    axes.grid(True, axis="y", alpha=0.5)
    axes.set_axisbelow(True)
    axes.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=SEED_COLOR, label="one training seed"
            ),
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=MARKER_COLOR,
                label="mean over seeds (whiskers: min–max)",
            ),
        ],
        loc="lower right",
        framealpha=1.0,
    )
    figure.tight_layout()
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Figure 2 — all five metrics, absolute units, against the baseline
# ---------------------------------------------------------------------------


def _metric_grid(figsize: tuple[float, float]) -> tuple[Figure, list[Axes], Axes]:
    """A 2×3 grid for the five metrics; the sixth cell is emptied to hold the legend."""
    figure, grid = plt.subplots(2, 3, figsize=figsize)
    flat = list(grid.ravel())
    spare = flat[len(METRICS)]
    spare.set_axis_off()
    return figure, flat, spare


def draw_baseline(axes: Axes, level: float, *, digits: int = 2) -> None:
    """The ``human_only`` reference line, labelled on the line rather than in a legend.

    A legend box in a small multiple lands wherever matplotlib finds room, which in these
    panels is on top of the data or the ``n=`` labels. Writing the value onto the line is
    both fixed and closer to what it describes.
    """
    axes.axhline(level, color=BASELINE_COLOR, linewidth=1.4, linestyle="--", zorder=1)
    axes.annotate(
        f"human_only = {level:.{digits}f}",
        (0.015, level),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        color=BASELINE_COLOR,
        # An opaque patch so the label stays readable where an interval passes behind it.
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
    )


def interval_legend(axes: Axes, *, extra_baseline: bool = True) -> None:
    """The shared key for every interval chart, drawn into an otherwise empty cell."""
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=SEED_COLOR, label="one training seed"
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=MARKER_COLOR,
            label="mean over training seeds\n(whiskers: lowest–highest seed)",
        ),
    ]
    if extra_baseline:
        handles.append(
            plt.Line2D([], [], linestyle="--", color=BASELINE_COLOR, label="human_only baseline")
        )
    axes.legend(handles=handles, loc="center", frameon=False, fontsize=11)


def plot_kpi_spread_by_recipe(recipes: Sequence[Recipe], path: Path) -> Path:
    """One interval chart per metric: recipe means with min/max over training seeds."""
    figure, panels, spare = _metric_grid((15.0, 8.6))
    for axes, metric in zip(panels, METRICS, strict=False):
        pairs = [(r, treatment_spread(r, metric)) for r in recipes]
        usable = [(r, s) for r, s in pairs if s is not None]
        if not usable:
            axes.set_visible(False)
            continue
        positions = list(range(len(usable)))
        spreads = [s for _, s in usable]
        draw_intervals(axes, positions, spreads)
        baselines = [
            b for _, b in ((r, baseline_spread(r, metric)) for r, _ in usable) if b is not None
        ]
        baseline_mean: float | None = None
        if baselines:
            baseline_mean = sum(b.mean for b in baselines) / len(baselines)
        every_value = [v for spread in spreads for v in spread.values]
        if baseline_mean is not None:
            every_value.append(baseline_mean)
        flat = flatten_degenerate_axis(
            axes, every_value, "identical on every arm\n(and on human_only)"
        )
        if not flat:
            add_headroom(axes, top=0.12, bottom=0.20 if metric.success_only else 0.10)
        if baseline_mean is not None:
            draw_baseline(axes, baseline_mean)
        if metric.success_only:
            annotate_counts(axes, positions, spreads)
        axes.set_xticks(positions)
        axes.set_xticklabels([r.label for r, _ in usable], rotation=30, ha="right")
        axes.set_ylabel(metric.axis_label)
        axes.set_title(f"{metric.label} ({metric.direction_note})")
        axes.grid(True, axis="y", alpha=0.5)
        axes.set_axisbelow(True)
    interval_legend(spare)
    figure.suptitle(
        "All five KPIs per recipe — mean over training seeds, whiskers to the lowest and "
        "highest seed",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Figure 3 — human_only vs plain BC vs DAgger
# ---------------------------------------------------------------------------


def plot_dagger_vs_plain(recipes: dict[str, Recipe], path: Path) -> Path:
    """Per modality (rows) and metric (columns): the three arms side by side.

    The ``human_only`` arm has no training seed, so it is drawn as a single point with no
    whisker — that absence is informative, not an omission.
    """
    modalities = [
        (modality, plains, dagger)
        for modality, plains, dagger in DAGGER_ARMS
        if dagger in recipes and any(label in recipes for label in plains)
    ]
    figure, grid = plt.subplots(
        len(modalities), len(METRICS), figsize=(4.0 * len(METRICS), 4.4 * len(modalities))
    )
    rows = grid if len(modalities) > 1 else [grid]
    for row, (modality, plain_labels, dagger_label) in zip(rows, modalities, strict=True):
        dagger = recipes[dagger_label]
        plain_arms = [recipes[label] for label in plain_labels if label in recipes]
        arms = [*plain_arms, dagger]
        # The paired Δ is quoted against the plain arm trained at the *same* batch size —
        # the only one that isolates DAgger from the optimisation-noise confound.
        reference = next(
            (arm for arm in plain_arms if arm.batch_size == dagger.batch_size), plain_arms[0]
        )
        comparisons = compare_recipes(reference, dagger)
        for axes, metric in zip(row, METRICS, strict=True):
            arm_spreads = [(a, treatment_spread(a, metric)) for a in arms]
            usable = [(a, s) for a, s in arm_spreads if s is not None]
            baseline = baseline_spread(dagger, metric)
            positions: list[float] = []
            labels: list[str] = []
            if baseline is not None:
                axes.errorbar(
                    [0.0],
                    [baseline.mean],
                    yerr=whiskers([baseline]),
                    fmt="D",
                    markersize=8,
                    color=BASELINE_COLOR,
                    ecolor=BASELINE_COLOR,
                    capsize=7,
                    linestyle="none",
                    zorder=3,
                )
                axes.axhline(
                    baseline.mean, color=BASELINE_COLOR, linewidth=1.0, linestyle="--", zorder=1
                )
                positions.append(0.0)
                labels.append("human_only")
            offset = len(positions)
            arm_positions = [float(offset + index) for index in range(len(usable))]
            if usable:
                draw_intervals(axes, arm_positions, [s for _, s in usable])
            positions += arm_positions
            labels += [
                ("plain BC" if a.regime == "plain BC" else "DAgger") + f"\nbatch {a.batch_size}"
                for a, _ in usable
            ]
            every_value = [v for _, s in usable for v in s.values]
            if baseline is not None:
                every_value += list(baseline.values)
            flat = flatten_degenerate_axis(axes, every_value, "identical on every arm")
            if not flat:
                add_headroom(axes, top=0.24, bottom=0.20 if metric.success_only else 0.10)
            if metric.success_only and usable:
                annotate_counts(axes, arm_positions, [s for _, s in usable])
            cross = cross_recipe_spread(comparisons, metric)
            if cross is not None:
                digits = 1 if metric is SUCCESS_METRIC else 2
                axes.annotate(
                    f"paired DAgger − plain (batch {reference.batch_size}), per training seed:\n"
                    f"mean {cross.mean:+.{digits}f}  "
                    f"[{cross.minimum:+.{digits}f}, {cross.maximum:+.{digits}f}]",
                    (0.5, 0.985),
                    xycoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color="#333333",
                )
            axes.set_xticks(positions)
            axes.set_xticklabels(labels, rotation=20, ha="right")
            axes.set_xlim(-0.6, len(positions) - 0.4)
            axes.set_ylabel(metric.axis_label)
            axes.set_title(f"{modality} · {metric.label}\n({metric.direction_note})", fontsize=11)
            axes.grid(True, axis="y", alpha=0.5)
            axes.set_axisbelow(True)
    figure.legend(
        handles=[
            plt.Line2D(
                [], [], marker="D", linestyle="none", color=BASELINE_COLOR, label="human_only"
            ),
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=SEED_COLOR, label="one training seed"
            ),
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=MARKER_COLOR,
                label="mean over training seeds (whiskers: lowest–highest seed)",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "Did DAgger move anything? human_only vs plain BC vs DAgger, per modality and KPI",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Figure 4 — DAgger across rounds (vision only)
# ---------------------------------------------------------------------------


def plot_dagger_rounds(runs_root: Path, policy_runs_root: Path, path: Path) -> Path | None:
    """Each metric against DAgger round, one line per training seed.

    One line per seed on purpose: the round axis is noisy *within* a single training run,
    so an average over seeds would smooth away the very thing the figure exists to show.
    """
    rounds = load_vision_dagger_rounds(runs_root, policy_runs_root)
    if not rounds:
        log.warning("no vision DAgger round evals found — skipping the round figure")
        return None
    seeds = sorted({entry.training_seed for entry in rounds})
    figure, panels, spare = _metric_grid((15.0, 8.6))
    for axes, metric in zip(panels, METRICS, strict=False):
        drawn: list[float] = []
        for seed in seeds:
            entries = sorted(
                (e for e in rounds if e.training_seed == seed), key=lambda e: e.round_index
            )
            values = [
                (e.round_index, marginal_value(e.point.treatment, metric).value) for e in entries
            ]
            defined = [(index, value) for index, value in values if value is not None]
            if not defined:
                continue
            drawn += [value for _, value in defined]
            axes.plot(
                [index for index, _ in defined],
                [value for _, value in defined],
                marker="o",
                markersize=5,
                linewidth=1.6,
                label=f"train seed {seed}",
            )
        baselines = [marginal_value(e.point.baseline, metric).value for e in rounds]
        defined_baselines = [b for b in baselines if b is not None]
        baseline_mean: float | None = None
        if defined_baselines:
            baseline_mean = sum(defined_baselines) / len(defined_baselines)
            drawn.append(baseline_mean)
        flat = flatten_degenerate_axis(axes, drawn, "identical in every round\n(and on human_only)")
        if not flat:
            add_headroom(axes, top=0.10, bottom=0.10)
        if baseline_mean is not None:
            draw_baseline(axes, baseline_mean)
        axes.set_xticks(sorted({e.round_index for e in rounds}))
        axes.set_xlabel("DAgger round")
        axes.set_ylabel(metric.axis_label)
        axes.set_title(f"{metric.label} ({metric.direction_note})")
        axes.grid(True, alpha=0.5)
        axes.set_axisbelow(True)
    handles, labels = panels[0].get_legend_handles_labels()
    spare.legend(
        handles=[
            *handles,
            plt.Line2D([], [], linestyle="--", color=BASELINE_COLOR, label="human_only baseline"),
        ],
        labels=[*labels, "human_only baseline"],
        loc="center",
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "Vision DAgger across rounds — the round axis is a lottery within a single training "
        "seed (the F/T arm has no per-round eval)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def log_spreads(recipes: Sequence[Recipe]) -> None:
    """Echo every recipe × metric distribution so a run is self-documenting in the log."""
    for recipe in recipes:
        for metric in METRICS:
            spread = treatment_spread(recipe, metric)
            delta = delta_spread(recipe, metric)
            if spread is None:
                continue
            log.info(
                "%-20s %-24s mean %.2f [%.2f, %.2f] over %d seeds (n/seed %d–%d) | paired Δ %s",
                recipe.label,
                metric.label,
                spread.mean,
                spread.minimum,
                spread.maximum,
                spread.n,
                spread.min_count,
                spread.max_count,
                f"{delta.mean:+.2f} [{delta.minimum:+.2f}, {delta.maximum:+.2f}]"
                if delta is not None
                else "—",
            )


def log_dagger_effect(recipes: dict[str, Recipe]) -> None:
    """Echo the DAgger-vs-plain paired deltas — the direct answer to "did it do anything"."""
    for modality, plain_labels, dagger_label in DAGGER_ARMS:
        dagger = recipes.get(dagger_label)
        if dagger is None:
            continue
        for plain_label in plain_labels:
            plain = recipes.get(plain_label)
            if plain is None:
                continue
            comparisons = compare_recipes(plain, dagger)
            for metric in METRICS:
                cross = cross_recipe_spread(comparisons, metric)
                if cross is None:
                    continue
                log.info(
                    "%-7s %-24s %s − %s: mean Δ %+.2f [%+.2f, %+.2f] over %d training seeds",
                    modality,
                    metric.label,
                    dagger_label,
                    plain_label,
                    cross.mean,
                    cross.minimum,
                    cross.maximum,
                    cross.n,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 official-run KPI figures.")
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Directory holding eval sets."
    )
    parser.add_argument(
        "--policy-runs-root",
        type=Path,
        default=DEFAULT_POLICY_RUNS_ROOT,
        help="Directory holding training run folders (per-round DAgger evals).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Where the PNGs are written (default: {DEFAULT_FIGURE_DIR}).",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)
    apply_plain_style()

    recipes = load_recipes(RECIPE_ORDER, arguments.runs_root)
    if not recipes:
        log.error("no finished official evals under %s — nothing to plot", arguments.runs_root)
        raise SystemExit(1)
    ordered = [recipes[label] for label in RECIPE_ORDER if label in recipes]
    log_spreads(ordered)
    log_dagger_effect(recipes)

    written = [
        plot_success_rate_spread(ordered, arguments.figure_dir / "success_rate_spread.png"),
        plot_kpi_spread_by_recipe(ordered, arguments.figure_dir / "kpi_spread_by_recipe.png"),
        plot_dagger_vs_plain(recipes, arguments.figure_dir / "dagger_vs_plain.png"),
        plot_dagger_rounds(
            arguments.runs_root,
            arguments.policy_runs_root,
            arguments.figure_dir / "dagger_rounds_vision.png",
        ),
    ]
    for figure_path in written:
        if figure_path is not None:
            log.info("wrote %s", figure_path)


if __name__ == "__main__":
    main()
