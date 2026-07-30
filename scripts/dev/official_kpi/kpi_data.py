"""The official multi-seed run, loaded once and shared by the tables and the figures.

Every number this project reports about the official run is a function of the per-trial
CSVs under ``runs/`` and ``outputs/policy/runs/``. This module is the single place that
reads them, and it re-derives **nothing**: success rates, per-KPI central tendency, the
McNemar success split and the Wilcoxon per-KPI deltas all come out of
:mod:`ai_teleop.eval.report` (``load_trials`` → ``group_by_config`` → ``summarize_config``
/ ``compare_paired``). What is added here is only *layout*: which directories belong to
which recipe, which training seed each one is, and how a metric is addressed uniformly
whether it is the success rate or one of the four continuous KPIs.

Two things the caller must respect, because they are easy to get wrong:

* **A recipe is a distribution, not a checkpoint.** One eval directory is one training
  seed's draw. The reportable quantity is mean + observed range over those seeds
  (LAB-114); a single directory's number is a lottery ticket, not a measurement.
* **``time_to_insert_s`` is success-only.** It exists only on trials that seated, so its
  ``n`` is a fraction of the trial count and differs per config. Every accessor here
  returns that ``n`` alongside the value so no table can quietly print a mean over four
  trials next to a mean over a hundred.

Read-only, no sim, no network.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

# `aggregate.py` owns the recipe → eval-directory mapping; importing it keeps one source
# of truth for the globs. Sibling script, not an installed module, hence the path import
# performed by the entry-point scripts before they import this module.
from aggregate import RECIPES

from ai_teleop.eval.report import (
    CONTINUOUS_KPIS,
    ConfigSummary,
    PairedComparison,
    compare_paired,
    group_by_config,
    load_trials,
    summarize_config,
)
from ai_teleop.eval.schema import TrialKPIs

BASELINE_CONFIG = "human_only"
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_POLICY_RUNS_ROOT = Path("outputs/policy/runs")

# Training seed lives in the eval directory's suffix: eval_official_ft_b2_s3 -> 3.
_TRAINING_SEED_PATTERN = re.compile(r"_s(\d+)$")


# ---------------------------------------------------------------------------
# Metrics — the success rate and the four continuous KPIs, addressed uniformly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One reportable metric, whether it is the headline rate or a continuous KPI.

    ``continuous_index`` is the position in :data:`ai_teleop.eval.report.CONTINUOUS_KPIS`,
    or ``None`` for the success rate (which the report API models separately, as a
    McNemar contingency rather than a mean). ``success_only`` and ``lower_is_better`` are
    copied straight off the underlying :class:`~ai_teleop.eval.report.KpiSpec` so the
    flags cannot drift from the definitions the statistics were computed under.
    """

    key: str
    label: str
    unit: str
    lower_is_better: bool
    success_only: bool
    continuous_index: int | None

    @property
    def axis_label(self) -> str:
        """Metric name with its unit, for a figure axis or a table header."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    @property
    def direction_note(self) -> str:
        """``lower is better`` / ``higher is better`` — spelled out, never inferred."""
        return "lower is better" if self.lower_is_better else "higher is better"


SUCCESS_METRIC = MetricSpec(
    key="success_rate",
    label="Success rate",
    unit="%",
    lower_is_better=False,
    success_only=False,
    continuous_index=None,
)

METRICS: tuple[MetricSpec, ...] = (SUCCESS_METRIC,) + tuple(
    MetricSpec(
        key=spec.attribute,
        label=spec.label,
        unit=spec.unit,
        lower_is_better=spec.lower_is_better,
        success_only=spec.success_only,
        continuous_index=index,
    )
    for index, spec in enumerate(CONTINUOUS_KPIS)
)


@dataclass(frozen=True)
class MetricValue:
    """A metric's value for one config, with the trial count it was averaged over."""

    value: float | None
    n: int


def marginal_value(summary: ConfigSummary, metric: MetricSpec) -> MetricValue:
    """One config's value for ``metric`` — success rate in percent, else the KPI mean."""
    if metric.continuous_index is None:
        return MetricValue(100.0 * summary.success_rate, summary.n_trials)
    stat = summary.kpi_stats[metric.continuous_index]
    return MetricValue(stat.mean, stat.n)


@dataclass(frozen=True)
class PairedValue:
    """A paired treatment − baseline delta for one metric, with its ``n`` and p-value."""

    delta: float | None
    n_pairs: int
    p_value: float | None


def paired_value(comparison: PairedComparison, metric: MetricSpec) -> PairedValue:
    """The paired delta for ``metric`` — percentage points for success, else KPI units."""
    if metric.continuous_index is None:
        success = comparison.success
        return PairedValue(100.0 * success.rate_delta, success.n_pairs, success.p_value)
    stat = comparison.kpi_stats[metric.continuous_index]
    return PairedValue(stat.mean_delta, stat.n_pairs, stat.p_value)


# ---------------------------------------------------------------------------
# One eval directory = one training seed's checkpoint on the shared eval walls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPoint:
    """One finished eval set: a baseline arm, a treatment arm, and their pairing.

    Both arms come out of the *same* ``trials.csv`` and share eval-wall seeds, so the
    pairing removes operator and scene variance exactly. ``training_seed`` is the seed
    the *checkpoint* was trained under — the only thing that varies across the points of
    one recipe.
    """

    directory: Path
    training_seed: int | None
    baseline: ConfigSummary
    treatment: ConfigSummary
    paired: PairedComparison
    treatment_trials: tuple[TrialKPIs, ...]
    baseline_trials: tuple[TrialKPIs, ...]

    @property
    def name(self) -> str:
        return self.directory.name


def _training_seed_of(directory: Path) -> int | None:
    """The training seed encoded in an eval directory's name, or ``None`` if absent."""
    match = _TRAINING_SEED_PATTERN.search(directory.name)
    return int(match.group(1)) if match else None


def load_eval_point(directory: Path, treatment_config: str) -> EvalPoint | None:
    """Load one eval directory, or ``None`` when it has not produced both arms yet.

    An in-flight or aborted run is skipped rather than raising: the figures must stay
    regenerable while a re-run is still filling in seeds.
    """
    csv_path = directory / "trials.csv"
    if not csv_path.is_file():
        return None
    grouped = group_by_config(load_trials(csv_path))
    if BASELINE_CONFIG not in grouped or treatment_config not in grouped:
        return None
    baseline_trials = grouped[BASELINE_CONFIG]
    treatment_trials = grouped[treatment_config]
    return EvalPoint(
        directory=directory,
        training_seed=_training_seed_of(directory),
        baseline=summarize_config(BASELINE_CONFIG, baseline_trials),
        treatment=summarize_config(treatment_config, treatment_trials),
        paired=compare_paired(
            baseline_trials,
            treatment_trials,
            baseline_label=BASELINE_CONFIG,
            treatment_label=treatment_config,
        ),
        treatment_trials=tuple(treatment_trials),
        baseline_trials=tuple(baseline_trials),
    )


# ---------------------------------------------------------------------------
# Recipes — a named set of eval points that differ only in their training seed
# ---------------------------------------------------------------------------

# Per-recipe context that is not in the CSVs: it comes from the training metadata
# recorded in docs/results/kpi-dashboard.md §5.5 and the run scripts beside this file.
# `modality` is the observation the policy sees; `regime` is how it was trained.
RECIPE_METADATA: dict[str, tuple[str, str, int]] = {
    # label: (modality, regime, batch size)
    "FT plain": ("F/T", "plain BC", 16),
    "FT plain (batch 2)": ("F/T", "plain BC", 2),
    "FT DAgger": ("F/T", "DAgger", 2),
    "Vision plain": ("Vision", "plain BC", 2),
    "Vision DAgger": ("Vision", "DAgger", 2),
}


@dataclass(frozen=True)
class Recipe:
    """One recipe's eval points — the distribution that is the reportable unit."""

    label: str
    modality: str
    regime: str
    batch_size: int
    treatment_config: str
    points: tuple[EvalPoint, ...]

    @property
    def n_seeds(self) -> int:
        return len(self.points)

    @property
    def short_label(self) -> str:
        """Recipe name over its seed count and batch — a two-line tick label."""
        return f"{self.label}\nn={self.n_seeds} seeds · batch {self.batch_size}"

    def by_training_seed(self) -> dict[int, EvalPoint]:
        """Eval points keyed by training seed, dropping any point without one."""
        return {p.training_seed: p for p in self.points if p.training_seed is not None}


@dataclass(frozen=True)
class Spread:
    """A metric's distribution over one recipe's training seeds — the claim, not a draw."""

    values: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def minimum(self) -> float:
        return min(self.values)

    @property
    def maximum(self) -> float:
        return max(self.values)

    @property
    def width(self) -> float:
        """Observed range — how far two runs of the *same* recipe landed apart."""
        return self.maximum - self.minimum

    @property
    def deviation(self) -> float:
        """Sample standard deviation over the training seeds; 0.0 for a single seed.

        Drawn as the box height in the KPI figures. Sample (``ddof=1``) rather than
        population: these seeds are a sample of the recipe's possible draws, not the
        whole of it.
        """
        return stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def min_count(self) -> int:
        """Smallest per-seed trial count behind the mean (matters for success-only KPIs)."""
        return min(self.counts)

    @property
    def max_count(self) -> int:
        return max(self.counts)


def _spread(values: Sequence[MetricValue]) -> Spread | None:
    """Collapse per-seed values into a distribution, or ``None`` if none are defined."""
    defined = [v for v in values if v.value is not None]
    if not defined:
        return None
    return Spread(
        values=tuple(v.value for v in defined if v.value is not None),
        counts=tuple(v.n for v in defined),
    )


def treatment_spread(recipe: Recipe, metric: MetricSpec) -> Spread | None:
    """The recipe's own value for ``metric``, spread over its training seeds."""
    return _spread([marginal_value(point.treatment, metric) for point in recipe.points])


def baseline_spread(recipe: Recipe, metric: MetricSpec) -> Spread | None:
    """The ``human_only`` arm across the same evals — it uses no checkpoint, so it should
    be constant; a non-zero width here means the comparison is not clean."""
    return _spread([marginal_value(point.baseline, metric) for point in recipe.points])


def delta_spread(recipe: Recipe, metric: MetricSpec) -> Spread | None:
    """The paired treatment − ``human_only`` delta, spread over training seeds."""
    paired = [paired_value(point.paired, metric) for point in recipe.points]
    return _spread([MetricValue(p.delta, p.n_pairs) for p in paired])


def paired_p_values(recipe: Recipe, metric: MetricSpec) -> tuple[float, ...]:
    """Every per-seed p-value for ``metric`` vs the baseline, in seed order."""
    values = [paired_value(point.paired, metric).p_value for point in recipe.points]
    return tuple(v for v in values if v is not None)


def load_recipes(labels: Sequence[str], runs_root: Path = DEFAULT_RUNS_ROOT) -> dict[str, Recipe]:
    """Load every named recipe that has at least one finished eval, in the given order."""
    recipes: dict[str, Recipe] = {}
    for label in labels:
        glob, treatment_config = RECIPES[label]
        points = [
            point
            for directory in sorted(runs_root.glob(glob))
            if (point := load_eval_point(directory, treatment_config)) is not None
        ]
        if not points:
            continue
        modality, regime, batch_size = RECIPE_METADATA[label]
        recipes[label] = Recipe(
            label=label,
            modality=modality,
            regime=regime,
            batch_size=batch_size,
            treatment_config=treatment_config,
            points=tuple(points),
        )
    return recipes


# ---------------------------------------------------------------------------
# Cross-recipe pairing — DAgger vs plain BC, matched by training seed
# ---------------------------------------------------------------------------


def compare_recipes(baseline: Recipe, treatment: Recipe) -> list[tuple[int, PairedComparison]]:
    """Pair two recipes' treatment arms by training seed, then by eval seed within it.

    Both arms were scored on the *same* 100 eval walls, so for a fixed training seed the
    two checkpoints can be compared trial-for-trial with the same paired machinery used
    against ``human_only``. That is the only way to ask "did DAgger move anything relative
    to plain BC" with the operator and scene variance removed; comparing the two marginal
    means would fold the eval-wall variance back in.

    Training seeds present in only one recipe are dropped.
    """
    baseline_points = baseline.by_training_seed()
    treatment_points = treatment.by_training_seed()
    comparisons: list[tuple[int, PairedComparison]] = []
    for training_seed in sorted(set(baseline_points) & set(treatment_points)):
        comparisons.append((
            training_seed,
            compare_paired(
                baseline_points[training_seed].treatment_trials,
                treatment_points[training_seed].treatment_trials,
                baseline_label=baseline.label,
                treatment_label=treatment.label,
            ),
        ))
    return comparisons


def cross_recipe_spread(
    comparisons: Sequence[tuple[int, PairedComparison]], metric: MetricSpec
) -> Spread | None:
    """The per-training-seed paired delta between two recipes, as a distribution."""
    paired = [paired_value(comparison, metric) for _, comparison in comparisons]
    return _spread([MetricValue(p.delta, p.n_pairs) for p in paired])


# ---------------------------------------------------------------------------
# DAgger rounds — vision only; the F/T arm was never evaluated per round
# ---------------------------------------------------------------------------

DAGGER_ROUNDS = (0, 1, 2, 3, 4)
VISION_DAGGER_SEEDS = (0, 1, 2)
VISION_TREATMENT_CONFIG = "vision"

# Why seed 0 is special: its intermediate rounds were re-evaluated after the fact
# (LAB-112 backfill) into `runs/backfill_dag_vis_s0_r{0..3}`, and its final round is the
# official eval directory itself. Seeds 1 and 2 kept their per-round evals in the
# training run folder. Same 100 held-out walls in all three cases.
_SEED0_BACKFILL_ROUNDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class RoundPoint:
    """One DAgger round of one training seed, scored on the shared eval walls."""

    training_seed: int
    round_index: int
    point: EvalPoint


def round_directory(
    training_seed: int, round_index: int, runs_root: Path, policy_runs_root: Path
) -> Path:
    """Where a vision-DAgger round's ``trials.csv`` lives, per the layout note above."""
    if training_seed == 0:
        if round_index in _SEED0_BACKFILL_ROUNDS:
            return runs_root / f"backfill_dag_vis_s0_r{round_index}"
        return runs_root / "eval_official_dag_vis_s0"
    return policy_runs_root / f"dag_vis_s{training_seed}" / f"dagger_round{round_index}"


def load_vision_dagger_rounds(
    runs_root: Path = DEFAULT_RUNS_ROOT, policy_runs_root: Path = DEFAULT_POLICY_RUNS_ROOT
) -> list[RoundPoint]:
    """Every vision-DAgger round that has an eval, ordered by (training seed, round)."""
    rounds: list[RoundPoint] = []
    for training_seed in VISION_DAGGER_SEEDS:
        for round_index in DAGGER_ROUNDS:
            directory = round_directory(training_seed, round_index, runs_root, policy_runs_root)
            point = load_eval_point(directory, VISION_TREATMENT_CONFIG)
            if point is not None:
                rounds.append(RoundPoint(training_seed, round_index, point))
    return rounds
