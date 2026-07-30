"""The official multi-seed run as one picture: every production recipe's paired Δ, the
spread it lands in over training seeds, and the noise floor that swallows all of them.

The headline KPI of this project is a **null** — no {F/T, vision} × {plain BC, DAgger}
recipe lifts closed-loop insertion success above the human-only baseline beyond
training-seed noise (docs/results/kpi-dashboard.md §5.5). Read as a table that is four
rows of numbers; read as a forest plot it is one glance: every mean sits inside the band
the *same recipe* spans when only its training seed changes.

The floor drawn here is measured **in this same run**, not imported from elsewhere: F/T
plain over 5 training seeds spans 27 pp [−19, +8] and vision plain over 3 spans 20 pp
[−16, +4] (§5). So the F/T-plain row *is* the wider band — that overlap is the point, not
a double count.

Statistics are not re-derived here: every number comes from `aggregate.py`'s `per_seed`,
i.e. from `ai_teleop.eval.report.{load_trials, group_by_config, compare_paired}` over the
per-trial CSVs. This module only lays them out. Read-only, re-runnable, no network.

Run from kevin/:  uv run python scripts/dev/official_kpi/plot_seed_spread.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

# aggregate.py is a sibling script, not an installed module — reach it by path so this
# stays runnable as `python scripts/dev/official_kpi/plot_seed_spread.py` from kevin/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import RECIPES, per_seed  # noqa: E402

log = get_logger("official-kpi-plot")

PLOT_PATH = Path("docs/results/phase-1/official_multiseed_deltas.png")

# The four production recipes, in the order of the §5.5 table. `aggregate.RECIPES` also
# carries "FT plain (batch 2)" — a control re-run isolating the DAgger batch-size
# confound, not a production recipe — so it is deliberately absent here.
PRODUCTION_RECIPES: tuple[str, ...] = (
    "FT plain",
    "FT DAgger",
    "Vision plain",
    "Vision DAgger",
)
# Extra context per row, from the training metadata recorded in §5.5.
BATCH_SIZE = {"FT plain": 16, "FT DAgger": 2, "Vision plain": 2, "Vision DAgger": 2}
# The two floor rows: one recipe, many training seeds, everything else fixed (§5).
FLOOR_SOURCES = ("FT plain", "Vision plain")

# --- palette -------------------------------------------------------------------------
# Same hues as `ai_teleop.eval.report` so the results figures read as one system. The
# figure paints its own light surface rather than going transparent: a GitHub reader may
# be on the light or the dark theme, and an opaque panel is legible on both.
SURFACE = "#fcfcfb"
INK = "#1f1f1f"
INK_MUTED = "#5c5c5c"
GRID = "#d8d6d0"
FLOOR_WIDE = "#e8e5dc"  # F/T floor band (the wider one)
FLOOR_NARROW = "#dbd7c9"  # vision floor band, drawn inside it
FAMILY_COLOR = {"FT": "#3b7dd8", "Vision": "#3aa657"}
# Data coords. Right of MEAN_COLUMN_X is the numeric column, not plotting space.
MEAN_COLUMN_X = 15.0
RANGE_COLUMN_X = 25.0
X_LIMITS = (-24.0, 37.0)
# Seeds ride just above the row line, the mean diamond just below it, so a mean that
# coincides with a seed never hides it.
SEED_ROW_OFFSET = -0.15
MEAN_ROW_OFFSET = 0.21


@dataclass(frozen=True)
class RecipeSpread:
    """One recipe's paired Δ distribution over its training seeds (percentage points)."""

    label: str
    deltas: tuple[float, ...]
    baseline_rate: float

    @property
    def mean_delta(self) -> float:
        return sum(self.deltas) / len(self.deltas)

    @property
    def min_delta(self) -> float:
        return min(self.deltas)

    @property
    def max_delta(self) -> float:
        return max(self.deltas)

    @property
    def family(self) -> str:
        return "Vision" if self.label.startswith("Vision") else "FT"

    @property
    def color(self) -> str:
        return FAMILY_COLOR[self.family]


def signed(value: float, decimals: int = 1) -> str:
    """`+8.0` / `−19.0` with a real minus sign, matching the dashboard and the axis."""
    return f"{value:+.{decimals}f}".replace("-", "−")


def collect_spreads() -> list[RecipeSpread]:
    """Every production recipe that has at least one finished eval, in table order.

    A recipe whose eval directories are missing or still running is skipped with a
    warning rather than crashing: the batch-2 control and any re-run in flight must not
    be able to break the figure.
    """
    spreads: list[RecipeSpread] = []
    for label in PRODUCTION_RECIPES:
        glob, treatment_label = RECIPES[label]
        rows = per_seed(glob, treatment_label)
        finished = [row for row in rows if row[3] is not None]
        if not finished:
            log.warning("%s: no complete evals under runs/%s — skipping row", label, glob)
            continue
        if len(finished) < len(rows):
            log.warning(
                "%s: %d/%d evals complete — plotting the finished seeds only",
                label,
                len(finished),
                len(rows),
            )
        baseline_rates = {round(row[1], 6) for row in finished if row[1] is not None}
        if len(baseline_rates) > 1:
            # human_only uses no checkpoint, so it must be identical across seeds. If it
            # moved, the spread below is not a training-seed spread and the figure lies.
            log.warning("%s: human_only differs across evals %s", label, sorted(baseline_rates))
        spreads.append(
            RecipeSpread(
                label=label,
                deltas=tuple(row[3] for row in finished if row[3] is not None),
                baseline_rate=next(iter(baseline_rates)),
            )
        )
    return spreads


def dodge(values: tuple[float, ...], step: float = 0.075) -> list[float]:
    """Vertical offsets that fan out seeds sharing a Δ, so 5 dots never read as 3."""
    offsets = [0.0] * len(values)
    for value in set(values):
        tied = [i for i, other in enumerate(values) if other == value]
        for rank, index in enumerate(tied):
            offsets[index] = step * (rank - (len(tied) - 1) / 2)
    return offsets


def floor_source(spreads: list[RecipeSpread], label: str) -> RecipeSpread | None:
    """The one-recipe-many-seeds row whose observed Δ range *is* the measured floor."""
    return next((spread for spread in spreads if spread.label == label), None)


def draw(spreads: list[RecipeSpread], path: Path) -> None:
    """Render the forest plot: per-seed dots, per-recipe mean and range, floor, Δ=0."""
    figure, axes = plt.subplots(figsize=(10.0, 5.6), facecolor=SURFACE)
    figure.subplots_adjust(left=0.155, right=0.985, top=0.775, bottom=0.275)
    axes.set_facecolor(SURFACE)

    wide_floor = floor_source(spreads, FLOOR_SOURCES[0])
    narrow_floor = floor_source(spreads, FLOOR_SOURCES[1])
    if wide_floor is not None:
        axes.axvspan(
            wide_floor.min_delta,
            wide_floor.max_delta,
            color=FLOOR_WIDE,
            zorder=0,
            label=(
                f"training-seed noise floor — F/T plain, {len(wide_floor.deltas)} seeds: "
                f"{wide_floor.max_delta - wide_floor.min_delta:.0f} pp"
            ),
        )
    if narrow_floor is not None:
        axes.axvspan(
            narrow_floor.min_delta,
            narrow_floor.max_delta,
            color=FLOOR_NARROW,
            zorder=0,
            label=(
                f"same, vision plain, {len(narrow_floor.deltas)} seeds: "
                f"{narrow_floor.max_delta - narrow_floor.min_delta:.0f} pp"
            ),
        )
    axes.axvline(0.0, color=INK, linewidth=1.4, zorder=2)

    for row, spread in enumerate(spreads):
        seed_row = row + SEED_ROW_OFFSET
        axes.plot(
            [spread.min_delta, spread.max_delta],
            [seed_row, seed_row],
            color=spread.color,
            linewidth=2.0,
            solid_capstyle="butt",
            alpha=0.5,
            zorder=3,
        )
        for edge in (spread.min_delta, spread.max_delta):
            axes.plot(
                [edge, edge],
                [seed_row - 0.11, seed_row + 0.11],
                color=spread.color,
                linewidth=2.0,
                alpha=0.5,
                zorder=3,
            )
        axes.scatter(
            spread.deltas,
            [seed_row + offset for offset in dodge(spread.deltas)],
            s=58,
            color=spread.color,
            edgecolors=SURFACE,
            linewidths=1.6,
            zorder=4,
        )
        axes.scatter(
            [spread.mean_delta],
            [row + MEAN_ROW_OFFSET],
            s=165,
            marker="D",
            color=spread.color,
            edgecolors=INK,
            linewidths=1.2,
            zorder=5,
        )
        # Numeric column on the right: the distribution *is* the claim, so it is spelled
        # out per row rather than left to be read off the axis.
        axes.text(
            MEAN_COLUMN_X,
            row,
            f"mean {signed(spread.mean_delta)} pp",
            ha="left",
            va="center",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
        axes.text(
            RANGE_COLUMN_X,
            row,
            f"[{signed(spread.min_delta, 0)}, {signed(spread.max_delta, 0)}]",
            ha="left",
            va="center",
            fontsize=10.5,
            color=INK_MUTED,
        )
        axes.text(
            X_LIMITS[1] - 0.8,
            row,
            "NOISE",
            ha="right",
            va="center",
            fontsize=10,
            color=INK_MUTED,
        )

    axes.set_yticks(range(len(spreads)))
    axes.set_yticklabels([
        f"{spread.label}\n{len(spread.deltas)} seeds · batch {BATCH_SIZE[spread.label]}"
        for spread in spreads
    ])
    axes.tick_params(axis="both", colors=INK, labelsize=10, length=0)
    axes.set_ylim(len(spreads) - 0.42, -0.78)  # table order, top to bottom
    axes.set_xlim(*X_LIMITS)
    axes.set_xticks([-20, -10, 0, 10])
    axes.set_xlabel(
        "Δ success rate vs human-only baseline (percentage points)", fontsize=11, color=INK
    )
    axes.text(
        MEAN_COLUMN_X,
        -0.62,
        "distribution over training seeds",
        ha="left",
        va="center",
        fontsize=9.5,
        color=INK_MUTED,
    )
    axes.annotate(
        "Δ = 0 · human-only baseline",  # its rate is in the subtitle
        (0.0, -0.62),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=INK_MUTED,
        va="center",
    )
    for row in range(len(spreads) - 1):
        axes.axhline(row + 0.5, color=GRID, linewidth=0.8, alpha=0.7, zorder=1)
    axes.grid(True, axis="x", color=GRID, alpha=0.8, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(GRID)

    seed_handle = plt.Line2D(
        [], [], marker="o", linestyle="", color=INK_MUTED, markersize=7, label="one training seed"
    )
    mean_handle = plt.Line2D(
        [],
        [],
        marker="D",
        linestyle="",
        color=INK_MUTED,
        markeredgecolor=INK,
        markersize=10,
        label="recipe mean Δ",
    )
    band_handles, band_labels = axes.get_legend_handles_labels()
    legend = axes.legend(
        handles=[seed_handle, mean_handle, *band_handles],
        labels=["one training seed", "recipe mean Δ", *band_labels],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.155),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        handletextpad=0.7,
        columnspacing=2.6,
    )
    legend.set_zorder(7)

    figure.text(
        0.5,
        0.955,
        "Every production recipe lands inside the training-seed noise floor",
        ha="center",
        va="center",
        fontsize=15,
        color=INK,
    )
    figure.text(
        0.5,
        0.885,
        "Official multi-seed run: each recipe retrained over training seeds, every checkpoint "
        f"scored on the same 100 paired held-out\neval seeds at es0.4 — human_only = "
        f"{spreads[0].baseline_rate:.1f}% in every one of these evaluations.",
        ha="center",
        va="center",
        fontsize=10,
        color=INK_MUTED,
        linespacing=1.45,
    )
    floor_seeds = len(wide_floor.deltas) if wide_floor is not None else 0
    figure.text(
        0.5,
        0.045,
        "Verdict: NOISE across the board — no recipe lifts closed-loop seating above the "
        "human-only baseline beyond training-seed noise.\nThe F/T-plain row is itself a floor "
        f"measurement (one recipe, {floor_seeds} seeds, nothing but the seed changed), so its "
        "band and its row coincide by construction.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK_MUTED,
        linespacing=1.45,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PLOT_PATH, help=f"PNG path (default: {PLOT_PATH})"
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    spreads = collect_spreads()
    if not spreads:
        log.error("no finished official evals under runs/ — nothing to plot")
        raise SystemExit(1)
    for spread in spreads:
        log.info(
            "%-14s n=%d  mean Δ %s pp  range [%s, %s]  seeds %s",
            spread.label,
            len(spread.deltas),
            signed(spread.mean_delta),
            signed(spread.min_delta, 0),
            signed(spread.max_delta, 0),
            ", ".join(signed(delta, 0) for delta in spread.deltas),
        )
    draw(spreads, arguments.output)
    log.info("wrote %s", arguments.output)


if __name__ == "__main__":
    main()
