"""Within one training seed — the policy against the operator, trial by trial.

Every other figure in this directory plots a distribution over **training seeds**, where one
point is a seed's *mean over 100 trials*. That answers "how much does retraining move the
average?". These figures hold the checkpoint fixed and plot the **individual trials**, which
answers the complementary question: *does this one policy beat the human operator, wall by
wall?* Two views, both for every reported KPI:

**View A — absolute per-trial distributions** (``within_seed_trials_<kpi>.png``). One panel per
training seed; in each, ``human_only`` and the policy over the same 100 walls. This is the shape
the seed-level mean is computed from, and for peak force that shape is bimodal — the mean lands
in the trough between the modes and describes no trial that ran.

**View B — paired per-trial deltas** (``within_seed_delta_<kpi>.png``). The same panels, but of
``policy − human_only`` computed **on the same wall** (matched by the eval-wall ``seed``). Wall
difficulty cancels exactly, so the spread that remains is what the policy did, not what it was
handed. Zero and the mean are marked.

**Success is boolean**, so a per-trial delta would be −1/0/+1 and a histogram of it says nothing.
It gets the McNemar discordance instead (``within_seed_success_mcnemar.png``): per seed, how many
walls the policy seated that the human missed, against how many it lost — the two counts whose
difference *is* the paired Δ, and whose split is what the exact test is computed on.

Two populations, where the KPI has two. ``peak_contact_force`` and ``jerk_integral`` are recorded
on every trial, so each panel draws the full matched set as an outline and the both-arms-seated
subset filled inside it — the subset definition is :func:`kpi_data.matched_trial_pairs`, the same
one the seed-level population split uses. ``time_to_insert_s`` exists only where both arms
seated, so it has one population and an ``n`` well under 100, printed in every panel.

**What these figures are not.** Fixing the seed removes the noise floor from the *picture*, not
from the *result*: a checkpoint that wins here is one draw from a recipe whose seeds span 20–31 pp
of success rate (``docs/results/noise-floor.md``). Every panel is a lottery ticket; the row of
panels is the lottery. The captions and the reading are in ``docs/results/within-seed.md`` §5.6.3.

Statistics are not computed here — the means and p-values come out of
:mod:`ai_teleop.eval.report` through :mod:`kpi_data`. This module lays out the raw trials behind
them.

Read-only, no sim, no network. Run from kevin/:

    uv run python scripts/dev/official_kpi/plot_within_seed.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, floor, log10
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kpi_data import (
    DEFAULT_RUNS_ROOT,
    METRICS,
    SUCCESS_METRIC,
    EvalPoint,
    MetricSpec,
    Recipe,
    load_recipes,
    matched_trial_pairs,
    paired_value,
    seated_pair_means,
)
from plot_kpis import (
    BASELINE_COLOR,
    DEFAULT_FIGURE_DIR,
    MARKER_COLOR,
    RECIPE_ORDER,
    add_headroom,
    apply_plain_style,
    finish,
)

from ai_teleop.common.log import (
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("within-seed")

# Every reported KPI except the success rate, which is boolean and gets its own figure.
CONTINUOUS_METRICS: tuple[MetricSpec, ...] = tuple(
    metric for metric in METRICS if metric is not SUCCESS_METRIC
)

BIN_COUNT = 32
# A success-only KPI is defined on ~25–45 walls per seed rather than 100, so the same binning
# would render it as a comb of ones. Half the bins, stated here rather than tuned per figure.
SPARSE_BIN_COUNT = BIN_COUNT // 2
# One rule for every axis in this module: a quantity whose largest magnitude is more than this
# multiple of its typical (median) magnitude is heavy-tailed and gets a logarithmic axis —
# logarithmic for a strictly positive quantity, symmetric-log for a signed one. ∫|jerk| per
# trial is 180× its median and collapses to a single bar on a linear axis; the paired
# time-to-insert delta is 50×. Peak force (3×) and its delta are not. Measured, not assumed
# per metric — and it changes only where the axis bends, never which trials are drawn.
HEAVY_TAIL_RATIO = 25.0

SEATED_ALPHA = 0.32
ZERO_COLOR = "#000000"


# ---------------------------------------------------------------------------
# Shared axes — one binning per metric, so panels can be read against each other
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricAxis:
    """The x-axis every panel of one metric shares: same bin edges, same scale.

    Built from the pooled values across every recipe and seed, so the outermost bins are the
    observed extremes and nothing falls outside the axis. Comparing two panels of one figure
    is then a comparison of shapes rather than of autoscaled boxes.
    """

    edges: tuple[float, ...]
    scale: str  # "linear" | "log" | "symlog"
    linthresh: float | None = None

    def apply(self, axes: Axes) -> None:
        """Put the scale and the full binned range onto one panel."""
        if self.scale == "log":
            axes.set_xscale("log")
        elif self.scale == "symlog" and self.linthresh is not None:
            axes.set_xscale("symlog", linthresh=self.linthresh)
            # Decade ticks either side of zero, chosen explicitly: the default symlog locator
            # also ticks the linear stretch, and with a sub-unit bend that prints two labels
            # on top of each other at the origin.
            decades = [
                10.0**exponent
                for exponent in range(
                    ceil(log10(self.linthresh)), floor(log10(abs(self.edges[-1]))) + 1
                )
            ]
            axes.set_xticks([-decade for decade in reversed(decades)] + [0.0] + decades)
        axes.set_xlim(self.edges[0], self.edges[-1])


def _linear_edges(low: float, high: float, count: int) -> tuple[float, ...]:
    step = (high - low) / count
    return tuple(low + step * index for index in range(count + 1))


def _geometric_edges(low: float, high: float, count: int) -> tuple[float, ...]:
    ratio = (high / low) ** (1.0 / count)
    return tuple(low * ratio**index for index in range(count + 1))


def _symmetric_log_edges(limit: float, linthresh: float, count: int) -> tuple[float, ...]:
    """Bins that are linear across zero and geometric out to ±``limit``.

    A signed heavy-tailed quantity (the jerk delta) has no useful linear binning: one seed's
    tail sets the range and every other panel collapses into the centre bin. Splitting the
    bins the way the symlog axis splits the scale keeps the bulk resolved *and* keeps the tail
    on the same axis instead of off it.
    """
    core = max(4, count // 4)
    wing = max(2, (count - core) // 2)
    negative = [-edge for edge in reversed(_geometric_edges(linthresh, limit, wing))]
    linear = list(_linear_edges(-linthresh, linthresh, core))
    positive = list(_geometric_edges(linthresh, limit, wing))
    return tuple(negative[:-1] + linear + positive[1:])


def _pooled_absolute_values(recipes: Sequence[Recipe], metric: MetricSpec) -> list[float]:
    """Every value of ``metric`` that View A will draw, both arms, every recipe and seed."""
    values: list[float] = []
    for recipe in recipes:
        for point in recipe.points:
            for baseline, treatment in matched_trial_pairs(point, metric, seated_only=False):
                values += [baseline, treatment]
    return values


def _pooled_deltas(recipes: Sequence[Recipe], metric: MetricSpec) -> list[float]:
    """Every paired delta View B will draw, every recipe and seed."""
    return [
        treatment - baseline
        for recipe in recipes
        for point in recipe.points
        for baseline, treatment in matched_trial_pairs(point, metric, seated_only=False)
    ]


def _bin_count(metric: MetricSpec) -> int:
    return SPARSE_BIN_COUNT if metric.success_only else BIN_COUNT


def _typical_magnitude(values: Sequence[float]) -> float:
    """Median absolute value over the non-zero entries — 0.0 when there are none."""
    magnitudes = [abs(value) for value in values if value != 0.0]
    return median(magnitudes) if magnitudes else 0.0


def _heavy_tailed(values: Sequence[float]) -> bool:
    """Whether the extreme is far enough above the typical magnitude to need a log axis."""
    typical = _typical_magnitude(values)
    return typical > 0.0 and max(abs(value) for value in values) / typical > HEAVY_TAIL_RATIO


def build_absolute_axis(recipes: Sequence[Recipe], metric: MetricSpec) -> MetricAxis | None:
    """View A's shared axis for one metric, logarithmic when the values are heavy-tailed."""
    values = _pooled_absolute_values(recipes, metric)
    if not values:
        return None
    low, high = min(values), max(values)
    if low > 0.0 and _heavy_tailed(values):
        return MetricAxis(_geometric_edges(low, high, _bin_count(metric)), "log")
    return MetricAxis(_linear_edges(low, high, _bin_count(metric)), "linear")


def build_delta_axis(recipes: Sequence[Recipe], metric: MetricSpec) -> MetricAxis | None:
    """View B's shared axis — symmetric-log when the deltas are heavy-tailed.

    The bend is placed at the *median* absolute delta, so the linear stretch covers the bulk of
    the walls and the logarithmic wings cover the outliers that would otherwise flatten it. The
    decision is taken on the deltas themselves: a metric can be well-behaved in absolute terms
    and heavy-tailed in its difference, which is exactly the case for time-to-insert.
    """
    deltas = _pooled_deltas(recipes, metric)
    if not deltas:
        return None
    limit = max(abs(delta) for delta in deltas)
    linthresh = _typical_magnitude(deltas)
    if _heavy_tailed(deltas) and limit > linthresh > 0.0:
        return MetricAxis(
            _symmetric_log_edges(limit, linthresh, _bin_count(metric)), "symlog", linthresh
        )
    return MetricAxis(_linear_edges(min(deltas), max(deltas), _bin_count(metric)), "linear")


# ---------------------------------------------------------------------------
# Layout — recipes down the page, training seeds across it
# ---------------------------------------------------------------------------


PANEL_SIZE = (3.3, 2.55)
# Vertical budget, in inches, for the title and each caption line above the grid. Held in
# inches rather than axes fractions because the figures differ in height, and a fraction that
# frames the 12.75-inch grids leaves the 4.6-inch success figure with a title inside its panels.
TITLE_INCHES = 0.42
CAPTION_LINE_INCHES = 0.24


def training_seed_columns(recipes: Sequence[Recipe]) -> list[int]:
    """Every training seed that appears in any recipe, ascending — one column each."""
    return sorted({
        point.training_seed
        for recipe in recipes
        for point in recipe.points
        if point.training_seed is not None
    })


def panel_for(recipe: Recipe, training_seed: int) -> EvalPoint | None:
    """That recipe's eval point for one training seed, or ``None`` if it has no such seed."""
    return recipe.by_training_seed().get(training_seed)


def add_caption(figure: Figure, title: str, lines: Sequence[str]) -> float:
    """Title plus a wrapped caption under it; returns the top of the area left for the grid.

    The caption is a separate text block rather than more lines of ``suptitle`` so it can be
    set smaller and so a long sentence can be broken where it reads well instead of running
    off both edges of the canvas.
    """
    height = figure.get_figheight()
    figure.suptitle(title, fontsize=13, y=1.0 - 0.16 / height)
    caption_top = 1.0 - (0.16 + TITLE_INCHES) / height
    figure.text(
        0.5,
        caption_top,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    return caption_top - (len(lines) * CAPTION_LINE_INCHES + 0.12) / height


def draw_seed_grid(
    recipes: Sequence[Recipe],
    axis: MetricAxis,
    *,
    draw: Callable[[Axes, EvalPoint], None],
    x_label: str,
    row_unit: str,
) -> Figure:
    """Recipes down the page, training seeds across it, every panel on the same axes.

    Shared axes on purpose: the question a reader brings here is whether *this* seed differs
    from the one beside it, and independently autoscaled panels answer it wrongly by making
    every distribution fill its own box. A recipe with fewer seeds leaves its trailing cells
    empty, and the column's x labels move up to whichever panel is lowest in it.
    """
    columns = training_seed_columns(recipes)
    figure, grid = plt.subplots(
        len(recipes),
        len(columns),
        figsize=(PANEL_SIZE[0] * len(columns), PANEL_SIZE[1] * len(recipes)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    lowest_in_column: dict[int, Axes] = {}
    for row, recipe in zip(grid, recipes, strict=True):
        drawn: list[Axes] = []
        for column_index, training_seed in enumerate(columns):
            axes = row[column_index]
            point = panel_for(recipe, training_seed)
            if point is None:
                axes.set_axis_off()
                continue
            draw(axes, point)
            style_panel(axes, axis)
            lowest_in_column[column_index] = axes
            drawn.append(axes)
        if drawn:
            drawn[0].set_ylabel(f"{recipe.label}\n{row_unit}", fontsize=9)
    for axes, training_seed in zip(grid[0], columns, strict=True):
        axes.set_title(f"training seed {training_seed}", fontsize=10)
    for axes in lowest_in_column.values():
        axes.set_xlabel(x_label, fontsize=9)
        axes.tick_params(labelbottom=True)
    if lowest_in_column:
        # Shared y, so one call sets every panel — room for the per-panel numbers on top.
        add_headroom(next(iter(lowest_in_column.values())), top=0.34, bottom=0.0)
    return figure


def label_panel(axes: Axes, text: str) -> None:
    """The panel's own numbers, pinned to its top-right corner over an opaque patch."""
    axes.annotate(
        text,
        (0.975, 0.955),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color="#222222",
        # Above the mean markers: the numbers are the panel's headline and a vertical rule
        # through them is harder to read than a rule interrupted by them.
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )


def draw_histogram(
    axes: Axes,
    values: Sequence[float],
    axis: MetricAxis,
    *,
    color: str,
    filled: bool,
) -> None:
    """One population's histogram — outline for the full set, filled for a subset of it."""
    if not values:
        return
    axes.hist(
        list(values),
        bins=list(axis.edges),
        histtype="stepfilled" if filled else "step",
        color=color,
        alpha=SEATED_ALPHA if filled else 1.0,
        linewidth=0.0 if filled else 1.3,
    )


def style_panel(axes: Axes, axis: MetricAxis) -> None:
    axis.apply(axes)
    axes.grid(True, alpha=0.35)
    axes.set_axisbelow(True)
    axes.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# View A — absolute per-trial distributions, one panel per training seed
# ---------------------------------------------------------------------------


def draw_absolute_panel(axes: Axes, point: EvalPoint, metric: MetricSpec, axis: MetricAxis) -> None:
    """``human_only`` and the policy over the same walls, with the seated subset filled."""
    matched = matched_trial_pairs(point, metric, seated_only=False)
    seated = matched_trial_pairs(point, metric, seated_only=True)
    draw_histogram(axes, [b for b, _ in matched], axis, color=BASELINE_COLOR, filled=False)
    draw_histogram(axes, [t for _, t in matched], axis, color=MARKER_COLOR, filled=False)
    if not metric.success_only:
        draw_histogram(axes, [b for b, _ in seated], axis, color=BASELINE_COLOR, filled=True)
        draw_histogram(axes, [t for _, t in seated], axis, color=MARKER_COLOR, filled=True)

    baseline_mean, treatment_mean, count = seated_pair_means(point, metric, seated_only=False)
    lines = []
    if baseline_mean is not None and treatment_mean is not None:
        lines.append(f"human {baseline_mean:.2f} · policy {treatment_mean:.2f}  (n={count})")
    if not metric.success_only:
        seated_baseline, seated_treatment, seated_count = seated_pair_means(
            point, metric, seated_only=True
        )
        if seated_baseline is not None and seated_treatment is not None:
            lines.append(
                f"seated: {seated_baseline:.2f} · {seated_treatment:.2f}  (n={seated_count})"
            )
    label_panel(axes, "\n".join(lines))


def plot_absolute_distributions(
    recipes: Sequence[Recipe], metric: MetricSpec, axis: MetricAxis, path: Path
) -> Path:
    """View A for one metric: recipes down the page, training seeds across it."""
    figure = draw_seed_grid(
        recipes,
        axis,
        draw=lambda axes, point: draw_absolute_panel(axes, point, metric, axis),
        x_label=metric.axis_label,
        row_unit="trials",
    )
    figure.legend(
        handles=[
            plt.Line2D([], [], color=BASELINE_COLOR, linewidth=1.6, label="human_only"),
            plt.Line2D([], [], color=MARKER_COLOR, linewidth=1.6, label="policy"),
        ],
        loc="lower center",
        ncols=2,
        frameon=False,
        fontsize=11,
    )
    population_note = (
        "The KPI is defined only where both arms seated, so each panel has one population "
        "and an n well under 100."
        if metric.success_only
        else "Outline: all 100 matched walls. Filled: the walls where both arms seated."
    )
    top = add_caption(
        figure,
        f"{metric.label} per trial, within one training seed — {metric.direction_note}",
        [
            f"Each panel is one checkpoint against the operator over the same held-out walls. "
            f"{population_note}",
            "A panel that looks like a win is one draw from a recipe whose training seeds span "
            "the 20–31 pp noise floor.",
        ],
    )
    figure.tight_layout(rect=(0, 0.035, 1, top))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# View B — paired per-trial deltas, same wall on both arms
# ---------------------------------------------------------------------------


def draw_delta_panel(axes: Axes, point: EvalPoint, metric: MetricSpec, axis: MetricAxis) -> None:
    """``policy − human_only`` per matched wall, with zero and both populations' means."""
    matched = matched_trial_pairs(point, metric, seated_only=False)
    seated = matched_trial_pairs(point, metric, seated_only=True)
    draw_histogram(axes, [t - b for b, t in matched], axis, color=MARKER_COLOR, filled=False)
    if not metric.success_only:
        draw_histogram(axes, [t - b for b, t in seated], axis, color=MARKER_COLOR, filled=True)
    axes.axvline(0.0, color=ZERO_COLOR, linestyle="--", linewidth=1.2, zorder=4)

    # The mean and the p-value are the report's paired statistics, not a re-derivation: for an
    # always-on KPI they are computed over exactly these matched walls, and for a success-only
    # KPI over exactly these both-seated ones.
    paired = paired_value(point.paired, metric)
    lines: list[str] = []
    if paired.delta is not None:
        axes.axvline(paired.delta, color=MARKER_COLOR, linewidth=1.8, zorder=5)
        significance = f" p={paired.p_value:.3f}" if paired.p_value is not None else ""
        lines.append(f"Δ mean {paired.delta:+.2f} (n={paired.n_pairs}){significance}")
    if not metric.success_only:
        seated_baseline, seated_treatment, seated_count = seated_pair_means(
            point, metric, seated_only=True
        )
        if seated_baseline is not None and seated_treatment is not None:
            seated_delta = seated_treatment - seated_baseline
            axes.axvline(seated_delta, color=MARKER_COLOR, linewidth=1.8, linestyle=":", zorder=5)
            lines.append(f"seated Δ {seated_delta:+.2f} (n={seated_count})")
    won = sum(1 for base, treat in matched if treat < base)
    if metric.lower_is_better and matched:
        lines.append(f"policy lower on {won}/{len(matched)} walls")
    label_panel(axes, "\n".join(lines))


def plot_paired_deltas(
    recipes: Sequence[Recipe], metric: MetricSpec, axis: MetricAxis, path: Path
) -> Path:
    """View B for one metric — the same wall on both arms, so wall difficulty cancels."""
    unit = f" ({metric.unit})" if metric.unit else ""
    figure = draw_seed_grid(
        recipes,
        axis,
        draw=lambda axes, point: draw_delta_panel(axes, point, metric, axis),
        x_label=f"policy − human_only, same wall{unit}",
        row_unit="matched walls",
    )
    handles = [
        plt.Line2D([], [], color=ZERO_COLOR, linestyle="--", linewidth=1.2, label="no difference"),
        plt.Line2D([], [], color=MARKER_COLOR, linewidth=1.8, label="mean Δ (all matched walls)"),
    ]
    if not metric.success_only:
        handles.append(
            plt.Line2D(
                [], [], color=MARKER_COLOR, linewidth=1.8, linestyle=":", label="mean Δ (seated)"
            )
        )
    figure.legend(
        handles=handles, loc="lower center", ncols=len(handles), frameon=False, fontsize=11
    )
    better = "left of zero is better" if metric.lower_is_better else "right of zero is better"
    top = add_caption(
        figure,
        f"Paired {metric.label.lower()} delta per wall, within one training seed — {better}",
        [
            "Each count is one held-out wall run by both arms, so wall difficulty cancels "
            "exactly and the width that remains is the policy's own trial-to-trial spread.",
            "The per-panel mean and p-value are the paired statistics the KPI board reports; "
            "a mean near zero over a wide distribution is a policy that trades wins for losses.",
        ],
    )
    figure.tight_layout(rect=(0, 0.035, 1, top))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Success — boolean, so the discordant walls rather than a distribution of ±1
# ---------------------------------------------------------------------------


def draw_discordance_panel(axes: Axes, recipe: Recipe, columns: Sequence[int]) -> None:
    """Per seed: walls the policy won (up) against walls it lost (down), and the net.

    A per-trial success "delta" can only be −1, 0 or +1, so its histogram is three bars that
    carry no more information than the counts. The counts are the honest object: they are what
    the exact McNemar test is computed on, and their difference *is* the paired Δ.
    """
    positions: list[int] = []
    for offset, training_seed in enumerate(columns):
        point = panel_for(recipe, training_seed)
        if point is None:
            continue
        success = point.paired.success
        positions.append(offset)
        axes.bar(offset, success.treatment_only, width=0.62, color=MARKER_COLOR, zorder=3)
        axes.bar(offset, -success.baseline_only, width=0.62, color=BASELINE_COLOR, zorder=3)
        net = success.treatment_only - success.baseline_only
        axes.plot([offset], [net], marker="D", markersize=6, color="#000000", zorder=4)
        significance = f"\np={success.p_value:.3f}" if success.p_value is not None else ""
        axes.annotate(
            f"{net:+d}{significance}",
            (offset, success.treatment_only),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes.axhline(0.0, color=ZERO_COLOR, linewidth=1.0)
    axes.set_xticks(list(range(len(columns))))
    axes.set_xticklabels([str(seed) for seed in columns])
    axes.set_xlim(-0.7, len(columns) - 0.3)
    axes.set_xlabel("training seed", fontsize=9)
    axes.set_title(recipe.label, fontsize=11)
    axes.grid(True, axis="y", alpha=0.4)
    axes.set_axisbelow(True)


def plot_success_discordance(recipes: Sequence[Recipe], path: Path) -> Path:
    """The success KPI's within-seed view: discordant walls per training seed."""
    columns = training_seed_columns(recipes)
    figure, grid = plt.subplots(
        1, len(recipes), figsize=(3.6 * len(recipes), 5.6), squeeze=False, sharey=True
    )
    for axes, recipe in zip(grid[0], recipes, strict=True):
        draw_discordance_panel(axes, recipe, columns)
    grid[0][0].set_ylabel("discordant walls (of 100)", fontsize=10)
    add_headroom(grid[0][0], top=0.34, bottom=0.02)
    figure.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=MARKER_COLOR,
                label="policy seated, human missed",
            ),
            plt.Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                color=BASELINE_COLOR,
                label="human seated, policy missed",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                color="#000000",
                label="net = paired Δ (percentage points)",
            ),
        ],
        loc="lower center",
        ncols=3,
        frameon=False,
        fontsize=10,
    )
    top = add_caption(
        figure,
        "Insertion success within one training seed — the walls the two arms disagree on",
        [
            "A boolean KPI has no per-trial distribution, so this is the discordant split the "
            "exact McNemar test is computed on: every checkpoint both wins and loses walls.",
            "The net (diamond) is that seed's paired Δ in percentage points; read down a "
            "panel, not across one bar.",
        ],
    )
    figure.tight_layout(rect=(0, 0.1, 1, top))
    return finish(figure, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def log_within_seed(recipes: Sequence[Recipe]) -> None:
    """Echo every per-seed paired result so a run is self-documenting in the log."""
    for recipe in recipes:
        for point in recipe.points:
            success = point.paired.success
            log.info(
                "%-20s seed %s  success: policy-only %2d  human-only %2d  net %+3d pp  p=%s",
                recipe.label,
                point.training_seed,
                success.treatment_only,
                success.baseline_only,
                success.treatment_only - success.baseline_only,
                f"{success.p_value:.4f}" if success.p_value is not None else "—",
            )
            for metric in CONTINUOUS_METRICS:
                paired = paired_value(point.paired, metric)
                seated_baseline, seated_treatment, seated_count = seated_pair_means(
                    point, metric, seated_only=True
                )
                matched = matched_trial_pairs(point, metric, seated_only=False)
                won = sum(1 for base, treat in matched if treat < base)
                seated_delta = (
                    None
                    if seated_baseline is None or seated_treatment is None
                    else seated_treatment - seated_baseline
                )
                log.info(
                    "  %-24s Δ %s (n=%3d, p=%s)  seated Δ %s (n=%3d)  policy lower on %d/%d",
                    metric.label,
                    f"{paired.delta:+8.3f}" if paired.delta is not None else "       —",
                    paired.n_pairs,
                    f"{paired.p_value:.4f}" if paired.p_value is not None else "—",
                    f"{seated_delta:+8.3f}" if seated_delta is not None else "       —",
                    seated_count,
                    won,
                    len(matched),
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-seed per-trial figures for the official run."
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Directory holding eval sets."
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
    ordered = [recipes[label] for label in RECIPE_ORDER if label in recipes]
    if not ordered:
        log.error("no finished official evals under %s — nothing to plot", arguments.runs_root)
        raise SystemExit(1)
    log_within_seed(ordered)

    written: list[Path] = [
        plot_success_discordance(ordered, arguments.figure_dir / "within_seed_success_mcnemar.png")
    ]
    for metric in CONTINUOUS_METRICS:
        absolute = build_absolute_axis(ordered, metric)
        if absolute is None:
            log.warning("%s has no matched trials — skipping", metric.label)
            continue
        written.append(
            plot_absolute_distributions(
                ordered,
                metric,
                absolute,
                arguments.figure_dir / f"within_seed_trials_{metric.key}.png",
            )
        )
        delta = build_delta_axis(ordered, metric)
        if delta is None:
            continue
        written.append(
            plot_paired_deltas(
                ordered, metric, delta, arguments.figure_dir / f"within_seed_delta_{metric.key}.png"
            )
        )
    for figure_path in written:
        log.info("wrote %s", figure_path)


if __name__ == "__main__":
    main()
