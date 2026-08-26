"""Aggregate the official-run eval sets into per-recipe DISTRIBUTIONS (the seed-variance discipline:
report mean + range over training seeds, never a single checkpoint). Read-only; re-runnable.

Run from kevin/:  uv run python scripts/dev/official_kpi/aggregate.py
"""

from pathlib import Path

from ai_teleop.eval.report import compare_paired, group_by_config, load_trials

RUNS = Path("runs")
# recipe label -> (glob of eval dirs, treatment config label)
RECIPES = {
    "FT plain": ("eval_official_ft_s*", "residual"),
    "FT plain (batch 2)": ("eval_official_ft_b2_s*", "residual"),
    "FT DAgger": ("eval_official_dag_ft_s*", "residual"),
    "Vision plain": ("eval_official_vis_s*", "vision"),
    "Vision DAgger": ("eval_official_dag_vis_s*", "vision"),
}


# One finished (or unfinished) eval directory: (dir name, human%, treat%, paired Δpp, p).
# The four value slots are None together when the eval has not produced both arms yet.
SeedRow = tuple[str, float | None, float | None, float | None, float | None]


def per_seed(glob: str, treat: str) -> list[SeedRow]:
    """Return [(seed_dir, human%, treat%, paired_Δpp, p)] for every finished eval in the glob."""
    rows: list[SeedRow] = []
    for d in sorted(RUNS.glob(glob)):
        csv = d / "trials.csv"
        if not csv.is_file():
            continue
        grouped = group_by_config(load_trials(csv))
        if "human_only" not in grouped or treat not in grouped:
            rows.append((d.name, None, None, None, None))
            continue
        comp = compare_paired(
            grouped["human_only"],
            grouped[treat],
            baseline_label="human_only",
            treatment_label=treat,
        )
        mc = comp.success
        rows.append((
            d.name,
            mc.baseline_rate * 100,
            mc.treatment_rate * 100,
            mc.rate_delta * 100,
            mc.p_value,
        ))
    return rows


def main() -> None:
    print("=" * 78)
    print("OFFICIAL KPI RUN — per-recipe distribution over training seeds (es 0.4, 100 eval seeds)")
    print("=" * 78)
    for label, (glob, treat) in RECIPES.items():
        rows = per_seed(glob, treat)
        done = [r for r in rows if r[3] is not None]
        print(f"\n### {label}   ({len(done)}/{len(rows)} evals complete)")
        if not rows:
            print("    (no eval dirs yet)")
            continue
        for name, h, t, d, p in rows:
            if d is None or h is None or t is None:
                print(f"    {name:32} (incomplete)")
            else:
                print(f"    {name:32} human {h:4.1f}%  {treat} {t:4.1f}%  Δ {d:+5.1f} pp  p={p}")
        if done:
            deltas = [r[3] for r in done if r[3] is not None]
            treats = [r[2] for r in done if r[2] is not None]
            print(
                f"    --> DISTRIBUTION: mean Δ {sum(deltas) / len(deltas):+.1f} pp, "
                f"range [{min(deltas):+.1f}, {max(deltas):+.1f}]  |  "
                f"{treat} success {min(treats):.1f}–{max(treats):.1f}% "
                f"(human_only {done[0][1]:.1f}%)"
            )
    print(
        "\nRead every row against the training-seed noise floor measured in this same run "
        "\n(F/T plain, 5 seeds: 27 pp; vision plain, 3 seeds: 20 pp). The DISTRIBUTION line is "
        "\nthe claim; a single seed's Δ is one draw. See docs/results/kpi-dashboard.md."
    )


if __name__ == "__main__":
    main()
