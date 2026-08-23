"""The official multi-seed run as markdown tables — all five KPIs, not just success.

The headline of this project is a null on **one** metric (closed-loop seating success).
That is not the same statement as "nothing moved": the four continuous KPIs
(``time_to_insert_s``, ``peak_contact_force``, ``jerk_integral``)
are measured on the same trials and have their own answers, and collapsing all five into
the success rate hides them. This script prints every recipe × every metric as a
distribution over training seeds (mean + observed range), the paired comparison against
``human_only``, the paired comparison of DAgger against plain BC, the raw per-seed draws,
and the vision DAgger round sweep.

Nothing is re-derived: every statistic comes from :mod:`ai_teleop.eval.report` via
:mod:`kpi_data`. The output is markdown on stdout (paste-ready for
``docs/results/kpi-dashboard.md``) and, unless ``--no-write``, the same text saved beside
the figures.

Run from kevin/:  uv run python scripts/dev/official_kpi/kpi_tables.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kpi_data import (  # noqa: E402
    DEFAULT_POLICY_RUNS_ROOT,
    DEFAULT_RUNS_ROOT,
    METRICS,
    SUCCESS_METRIC,
    MetricSpec,
    Recipe,
    Spread,
    baseline_spread,
    compare_recipes,
    cross_recipe_spread,
    delta_spread,
    load_recipes,
    load_vision_dagger_rounds,
    marginal_value,
    paired_p_values,
    paired_value,
    treatment_spread,
)

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)

log = get_logger("official-kpi-tables")

DEFAULT_OUTPUT = Path("docs/results/phase-1/official_kpi_tables.md")

# Table order: the four production recipes, then the batch-2 control that exists only to
# strip the batch-size confound out of the F/T DAgger comparison.
RECIPE_ORDER = (
    "FT plain",
    "FT plain (batch 2)",
    "FT DAgger",
    "Vision plain",
    "Vision DAgger",
)
# (modality, plain-BC recipes to compare against, DAgger recipe).
DAGGER_ARMS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("F/T", ("FT plain", "FT plain (batch 2)"), "FT DAgger"),
    ("Vision", ("Vision plain",), "Vision DAgger"),
)


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def decimals_for(metric: MetricSpec) -> int:
    """Display precision — one decimal for a percentage, two for a physical KPI."""
    return 1 if metric is SUCCESS_METRIC else 2


def number(value: float | None, decimals: int) -> str:
    """A fixed-point cell, or an em dash when the metric is undefined here."""
    return "—" if value is None else f"{value:.{decimals}f}"


def signed(value: float | None, decimals: int) -> str:
    """A signed cell — deltas always carry their sign so direction is never guessed."""
    return "—" if value is None else f"{value:+.{decimals}f}"


def spread_cell(spread: Spread | None, metric: MetricSpec) -> str:
    """``mean [min, max]`` — the distribution is the claim, so both parts are printed."""
    if spread is None:
        return "—"
    digits = decimals_for(metric)
    if spread.n == 1:
        return number(spread.mean, digits)
    return f"{number(spread.mean, digits)} [{number(spread.minimum, digits)}, {number(spread.maximum, digits)}]"


def delta_cell(spread: Spread | None, metric: MetricSpec) -> str:
    """``mean [min, max]`` for a signed delta."""
    if spread is None:
        return "—"
    digits = decimals_for(metric)
    if spread.n == 1:
        return signed(spread.mean, digits)
    return (
        f"{signed(spread.mean, digits)} "
        f"[{signed(spread.minimum, digits)}, {signed(spread.maximum, digits)}]"
    )


def count_cell(spread: Spread | None) -> str:
    """Trials behind each seed's value — a range when the seeds disagree on ``n``.

    For a success-only KPI the count is the number of *seated* trials, which at these
    success rates is a fraction of the 100 evaluated. A mean over 30 trials and a mean
    over 100 do not deserve the same weight, and this column is what says which is which.
    """
    if spread is None:
        return "—"
    if spread.min_count == spread.max_count:
        return str(spread.min_count)
    return f"{spread.min_count}–{spread.max_count}"


def p_cell(p_values: Sequence[float]) -> str:
    """The per-seed p-values as a range — one recipe yields one p per training seed."""
    if not p_values:
        return "—"
    formatted = [("<0.001" if p < 1e-3 else f"{p:.3f}") for p in (min(p_values), max(p_values))]
    return (
        formatted[0]
        if len(p_values) == 1 or formatted[0] == formatted[1]
        else (f"{formatted[0]}–{formatted[1]}")
    )


def escape_cell(text: str) -> str:
    """Escape pipes so a cell cannot invent a column.

    ``Trajectory jerk (∫|jerk|)`` is a real metric label, and its bare pipes split the
    header into more columns than the rows have — GitHub then renders the whole table as
    text. Escaping at the table builder covers every caller, so no label has to remember.
    """
    return text.replace("|", "\\|")


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """A GitHub-flavored markdown table from a header and its rows."""
    lines = [
        "| " + " | ".join(escape_cell(cell) for cell in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def baseline_across(recipes: Sequence[Recipe], metric: MetricSpec) -> Spread | None:
    """The ``human_only`` arm pooled over every eval directory in the run.

    It uses no checkpoint, so it *should* be one number repeated. Pooling it and printing
    the range is how a drifting baseline would announce itself instead of hiding.
    """
    spreads = [baseline_spread(recipe, metric) for recipe in recipes]
    values = tuple(v for spread in spreads if spread is not None for v in spread.values)
    counts = tuple(spread.min_count for spread in spreads if spread is not None)
    return Spread(values=values, counts=counts) if values else None


def overview_table(recipes: Sequence[Recipe]) -> str:
    """Every metric × every recipe, mean over training seeds with the observed range."""
    header = ["Metric", "dir.", "`human_only`", *(r.label for r in recipes)]
    rows: list[list[str]] = []
    for metric in METRICS:
        cells = [
            f"**{metric.axis_label}**",
            "↓" if metric.lower_is_better else "↑",
            spread_cell(baseline_across(recipes, metric), metric),
            *(spread_cell(treatment_spread(recipe, metric), metric) for recipe in recipes),
        ]
        rows.append(cells)
        # The n row matters only where it is not simply the trial count.
        if metric.success_only:
            rows.append([
                "  ↳ n (seated trials)",
                "",
                count_cell(baseline_across(recipes, metric)),
                *(count_cell(treatment_spread(recipe, metric)) for recipe in recipes),
            ])
    return table(header, rows)


def per_metric_paired_tables(recipes: Sequence[Recipe]) -> list[tuple[str, str]]:
    """One table per metric: each recipe against ``human_only``, paired by eval seed."""
    tables: list[tuple[str, str]] = []
    for metric in METRICS:
        header = [
            "Recipe",
            "seeds",
            "batch",
            "`human_only`",
            "treatment",
            "paired Δ (mean [min, max])",
            "n/seed",
            "per-seed p",
        ]
        rows: list[list[str]] = []
        for recipe in recipes:
            deltas = delta_spread(recipe, metric)
            rows.append([
                recipe.label,
                str(recipe.n_seeds),
                str(recipe.batch_size),
                spread_cell(baseline_spread(recipe, metric), metric),
                spread_cell(treatment_spread(recipe, metric), metric),
                delta_cell(deltas, metric),
                count_cell(deltas),
                p_cell(paired_p_values(recipe, metric)),
            ])
        title = f"{metric.axis_label} — {metric.direction_note}"
        tables.append((title, table(header, rows)))
    return tables


def dagger_tables(recipes: dict[str, Recipe]) -> list[tuple[str, str]]:
    """Per modality: ``human_only`` vs plain BC vs DAgger, with the paired statistics.

    The last three columns are the actual answer to "did DAgger produce any result at
    all": the DAgger checkpoint and the plain checkpoint of the *same training seed*
    scored on the *same* eval walls, compared trial-for-trial.
    """
    tables: list[tuple[str, str]] = []
    for modality, plain_labels, dagger_label in DAGGER_ARMS:
        dagger = recipes.get(dagger_label)
        if dagger is None:
            continue
        for plain_label in plain_labels:
            plain = recipes.get(plain_label)
            if plain is None:
                continue
            comparisons = compare_recipes(plain, dagger)
            header = [
                "Metric",
                "dir.",
                "`human_only`",
                plain_label,
                dagger_label,
                "paired Δ (DAgger − plain)",
                "n/seed",
                "per-seed p",
            ]
            rows: list[list[str]] = []
            for metric in METRICS:
                cross = cross_recipe_spread(comparisons, metric)
                p_values = [
                    p
                    for _, comparison in comparisons
                    if (p := paired_value(comparison, metric).p_value) is not None
                ]
                rows.append([
                    f"**{metric.axis_label}**",
                    "↓" if metric.lower_is_better else "↑",
                    spread_cell(baseline_spread(dagger, metric), metric),
                    spread_cell(treatment_spread(plain, metric), metric),
                    spread_cell(treatment_spread(dagger, metric), metric),
                    delta_cell(cross, metric),
                    count_cell(cross),
                    p_cell(p_values),
                ])
            title = (
                f"{modality} — {dagger_label} vs {plain_label} "
                f"({len(comparisons)} training seeds matched)"
            )
            tables.append((title, table(header, rows)))
    return tables


def per_seed_tables(recipes: Sequence[Recipe]) -> list[tuple[str, str]]:
    """The raw draws: every training seed's own value, per metric.

    These are the numbers a single-checkpoint report would have quoted one of. Printing
    all of them is the cheapest possible defence against picking a favourite.
    """
    seeds = sorted({
        point.training_seed
        for recipe in recipes
        for point in recipe.points
        if point.training_seed is not None
    })
    tables: list[tuple[str, str]] = []
    for metric in METRICS:
        digits = decimals_for(metric)
        header = ["Recipe", *(f"seed {seed}" for seed in seeds), "mean", "range"]
        rows: list[list[str]] = []
        for recipe in recipes:
            by_seed = recipe.by_training_seed()
            cells = []
            for seed in seeds:
                point = by_seed.get(seed)
                value = marginal_value(point.treatment, metric).value if point else None
                cells.append(number(value, digits))
            spread = treatment_spread(recipe, metric)
            rows.append([
                recipe.label,
                *cells,
                number(spread.mean, digits) if spread else "—",
                number(spread.width, digits) if spread else "—",
            ])
        tables.append((f"{metric.axis_label} — per training seed", table(header, rows)))
    return tables


def rounds_table(runs_root: Path, policy_runs_root: Path) -> str:
    """Vision DAgger, every round of every training seed, on the same held-out walls."""
    rounds = load_vision_dagger_rounds(runs_root, policy_runs_root)
    header = [
        "train seed",
        "round",
        *(metric.axis_label for metric in METRICS),
        "paired Δ success (pp)",
        "p",
    ]
    rows: list[list[str]] = []
    for entry in rounds:
        success_delta = paired_value(entry.point.paired, SUCCESS_METRIC)
        rows.append([
            str(entry.training_seed),
            str(entry.round_index),
            *(
                number(marginal_value(entry.point.treatment, metric).value, decimals_for(metric))
                for metric in METRICS
            ),
            signed(success_delta.delta, 1),
            p_cell([success_delta.p_value] if success_delta.p_value is not None else []),
        ])
    return table(header, rows)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


SEATED_ONLY_NOTE = (
    "> **Why this section exists.** `peak_contact_force` and `jerk_integral` are recorded on "
    "*every* trial, seated or not, so their means above mix two populations: the runs that "
    "inserted and the runs that ran out of budget or tripped the force cap. A treatment that "
    "fails differently — say, by loading harder against the wall before giving up — would move "
    "the all-trials mean without changing anything about how it behaves when it succeeds. "
    "Restricting to trials where **both** arms seated the same wall separates the two. "
    "`time_to_insert_s` is seated-only by definition and appears here for completeness."
)


def seated_only_tables(recipes: Sequence[Recipe]) -> list[tuple[str, str]]:
    """Each always-on KPI over all trials, then over the both-arms-seated subset.

    The subset is the same pairing used everywhere else — matched eval wall, both arms
    seated — so the two columns differ only in which trials they admit, never in how the
    numbers are computed.
    """
    tables: list[tuple[str, str]] = []
    for metric in METRICS:
        if metric.key == SUCCESS_METRIC.key or metric.success_only:
            continue
        header = [
            "Recipe",
            "all trials: baseline",
            "all trials: treatment",
            "all trials: Δ",
            "seated-only: baseline",
            "seated-only: treatment",
            "seated-only: Δ",
            "n seated pairs",
        ]
        rows: list[list[str]] = []
        for recipe in recipes:
            all_base, all_treat, seat_base, seat_treat, pairs = [], [], [], [], 0
            for point in recipe.points:
                by_seed = {t.seed: t for t in point.baseline_trials}
                for treatment in point.treatment_trials:
                    baseline = by_seed.get(treatment.seed)
                    if baseline is None:
                        continue
                    base_value = getattr(baseline, metric.key)
                    treat_value = getattr(treatment, metric.key)
                    all_base.append(base_value)
                    all_treat.append(treat_value)
                    if baseline.success and treatment.success:
                        seat_base.append(base_value)
                        seat_treat.append(treat_value)
                        pairs += 1
            if not all_base:
                continue
            rows.append([
                recipe.label,
                f"{mean(all_base):.2f}",
                f"{mean(all_treat):.2f}",
                f"{mean(all_treat) - mean(all_base):+.2f}",
                f"{mean(seat_base):.2f}" if seat_base else "—",
                f"{mean(seat_treat):.2f}" if seat_treat else "—",
                f"{mean(seat_treat) - mean(seat_base):+.2f}" if seat_base else "—",
                str(pairs),
            ])
        if rows:
            tables.append((f"{metric.axis_label} — {metric.direction_note}", table(header, rows)))
    return tables


SURVIVORSHIP_NOTE = (
    "> **For `time_to_insert_s`, read the paired Δ column — not the difference between the "
    "two mean columns. They can disagree in sign.** Each arm's mean is taken over *its own* "
    "successful trials, and the two arms do not seat the same set of walls, so the marginal "
    "means describe two different populations. The paired Δ is restricted to the walls where "
    "**both** arms seated, which is the only subset on which a difference is a difference. "
    "Where a treatment's marginal mean beats the baseline while its paired Δ says it is "
    "slower, the marginal gap is survivorship: an arm that seats fewer walls is seating the "
    "easier ones. Every other KPI is over all 100 trials and has no such gap."
)

FT_DAGGER_ROUNDS_NOTE = (
    "**The F/T DAgger arm has no per-round evaluation.** Its round checkpoints under "
    "`outputs/policy/runs/dag_ft_s*/dagger_round*/` carry weights and training history but "
    "no `trials.csv` — the intermediate rounds were never scored on the held-out walls, and "
    "backfilling them was only ever done for vision. So the round sweep below is "
    "vision-only, and no claim about how the F/T arm evolved across rounds can be made from "
    "this data."
)


def build_markdown(recipes: dict[str, Recipe], runs_root: Path, policy_runs_root: Path) -> str:
    """Assemble every table into one markdown document."""
    ordered = [recipes[label] for label in RECIPE_ORDER if label in recipes]
    parts: list[str] = [
        "# Official multi-seed run — full KPI tables",
        "",
        "Generated by `scripts/dev/official_kpi/kpi_tables.py`; every number is a function of "
        "the per-trial CSVs, computed through `ai_teleop.eval.report`. A recipe is reported as "
        "**mean over training seeds `[min, max]`**, never as a single checkpoint.",
        "",
        "`↓` = lower is better, `↑` = higher is better. `time_to_insert_s` is defined only on "
        "trials that seated, so its `n` is smaller than the 100 evaluated and is printed "
        "wherever it appears.",
        "",
        "`contact_events` counts rising edges of the contact force past its floor "
        "(hysteresis-debounced, `eval/observer.py`). At this operating point it is **1 on "
        "every trial of every arm, including `human_only`** — the approach makes one "
        "sustained contact and stays in it. It is therefore recorded but **not reported**; "
        "re-add it to `CONTINUOUS_KPIS` if the operating point changes.",
        "",
        "## 1. Every metric, every recipe",
        "",
        overview_table(ordered),
        "",
        "",
        SURVIVORSHIP_NOTE,
        "",
        "## 2. Each recipe against the `human_only` baseline (paired by eval seed)",
    ]
    for title, body in per_metric_paired_tables(ordered):
        parts += ["", f"### {title}", "", body]

    parts += ["", "## 3. Plain BC vs DAgger, side by side with the baseline"]
    for title, body in dagger_tables(recipes):
        parts += ["", f"### {title}", "", body]

    parts += ["", "## 4. The raw per-training-seed draws"]
    for title, body in per_seed_tables(ordered):
        parts += ["", f"### {title}", "", body]

    parts += ["", "## 5. All trials vs seated-only, for the always-on KPIs", "", SEATED_ONLY_NOTE]
    for title, body in seated_only_tables(ordered):
        parts += ["", f"### {title}", "", body]

    parts += [
        "",
        "## 6. DAgger across rounds (vision only)",
        "",
        FT_DAGGER_ROUNDS_NOTE,
        "",
        rounds_table(runs_root, policy_runs_root),
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-KPI markdown tables for the official run.")
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
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Markdown path ({DEFAULT_OUTPUT})."
    )
    parser.add_argument("--no-write", action="store_true", help="Print to stdout only.")
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    recipes = load_recipes(RECIPE_ORDER, arguments.runs_root)
    if not recipes:
        log.error("no finished official evals under %s — nothing to tabulate", arguments.runs_root)
        raise SystemExit(1)
    for recipe in recipes.values():
        log.info("%-20s %d training seeds", recipe.label, recipe.n_seeds)

    markdown = build_markdown(recipes, arguments.runs_root, arguments.policy_runs_root)
    print(markdown)  # noqa: T201 — the tables are this script's output, not status
    if not arguments.no_write:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(markdown, encoding="utf-8")
        log.info("wrote %s", arguments.output)


if __name__ == "__main__":
    main()
