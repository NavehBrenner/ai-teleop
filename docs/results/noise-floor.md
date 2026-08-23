# The Noise Floor — what retraining alone moves

**The between-seed question:** train one fixed recipe several times, changing only the
training seed, and see how far the outcomes land apart. That spread is this project's
measurement resolution, and it turns out to be larger than any treatment effect measured
here — which is why every claim is reported as a distribution rather than a checkpoint.

This file also carries the official multi-seed run: all production recipes, retrained
across seeds, on 100 paired held-out eval walls. That is the measurement the project's
headline rests on.

**Related:** [experiment ledger](experiment-ledger.md) · [KPI board](kpi-board.md) ·
[within-seed](within-seed.md) · [mechanisms](mechanisms.md) · [index](kpi-dashboard.md)

---

## 5. The noise floor — how it was measured

Every comparison in this document is a difference between two success rates. Before any
difference can mean anything, one number has to be established: **how far apart can two runs of
the *same* recipe land?** That is the floor, and it is measured, not assumed.

**The design.** Fix a recipe — same corpus, same hyperparameters, same everything. Train it N
times, varying only the **training seed**, which controls weight initialization and batch
shuffling order. Evaluate every resulting checkpoint on the **same** 100 paired held-out eval
seeds against `human_only`. Any spread in the outcome is attributable to training randomness and
nothing else.

**On the official corpus** (`dataset_official_ft` 1000 episodes / `dataset_official_vision` 500, es0.4, 100 paired
eval seeds — the same data every production recipe was trained on):

| Recipe | train seed | treatment success | paired Δ | p |
|---|---|---|---|---|
| F/T | 0 | 47.0% | −3.0 pp | 0.711 |
| F/T | 1 | **31.0%** | **−19.0 pp** | **0.0009** |
| F/T | 2 | 58.0% | +8.0 pp | 0.115 |
| F/T | 3 | 47.0% | −3.0 pp | 0.678 |
| F/T | 4 | 45.0% | −5.0 pp | 0.458 |
| **F/T spread** | 5 seeds | 31–58% | **27 pp** [−19, +8] | |
| vision | 0 | 37.0% | −13.0 pp | 0.041 |
| vision | 1 | 34.0% | −16.0 pp | 0.009 |
| vision | 2 | 54.0% | +4.0 pp | 0.503 |
| **vision spread** | 3 seeds | 34–54% | **20 pp** [−16, +4] | |

**The environment is exonerated.** `human_only` uses no checkpoint and returned **exactly 50.0%
in every one of these evaluations** — identical walls, operator, controller config, step budget
and scoring. The checkpoint is the only variable.

**Seed 1 is the point.** On its own it reads as a strongly significant regression, p=0.0009,
produced by nothing but a different random initialization. Read seed 1 alone and you would
report a broken recipe; read seed 2 alone and you would report an +8 pp win. Neither is true.
The same trap fires positive elsewhere — vision-DAgger seed 1 (§5.5) reads **+12 pp at p=0.036**
while its two sibling seeds both read −4.

**Corroborated on independent data.** The floor was first measured (2026-07-22) on
`dataset_10` — a 5× smaller, 200-episode corpus with a different master seed — giving a **18 pp**
spread over five seeds (paired Δ −15 to +3, records in `phase-1/lab114/`). Two corpora, two
generations of the pipeline, the same order of magnitude. The floor is a property of the task
and the recipe, not of any one dataset.

**Why it existed unmeasured for so long.** Training was **unseeded** until 2026-07-23:
`torch.manual_seed` was never called, and `--seed` reached only the train/val split — so weight
init and batch order came from OS entropy while each run folder faithfully recorded a seed and a
git commit. Two runs of the same command produced different models and the artifacts could not
show it. Fixed in one line, with a train-twice-identical-weights regression test. Results
predating that fix are single unrepeatable draws and are marked as history throughout this
document.

**A free result worth keeping.** Across the five `dataset_10` seeds, `best_val_loss` vs
closed-loop success is Spearman **ρ = −0.82** (p=0.089, n=5): offline loss is directionally
predictive *within one recipe*, the opposite of its behavior *across* interventions ([negative results](mechanisms.md#6-negative-results)).
Selecting the best-val checkpoint of a fixed recipe is therefore fine; tuning *recipes* by val
loss is not.

![val loss vs closed-loop success across seeds](phase-1/lab114_val_loss_vs_success.png)

---

## 5.5 The official multi-seed run — the definitive measurement

§5 above established the measurement resolution; this section is the measurement the project
stands on. Per the D-6 mandate: fresh official corpora (**1000 episodes** F/T, **500** vision,
generated separately), each of the four production recipes **retrained over multiple seeds** and
reported as a **distribution**, evaluated on 100 paired held-out seeds at the es0.4 operating
point. Seed families are disjoint by construction, so every Δ below is genuine held-out **test**
(corpus master-seed 100; DAgger rollouts 300/301/302; eval walls seed 0). Re-aggregate with
`scripts/dev/official_kpi/aggregate.py`.

| Recipe | seeds (n) | batch | `human_only` | treatment success | **mean Δ** | range | verdict vs floor |
|---|---|---|---|---|---|---|---|
| **FT plain** | 5 | 16 | 50.0% | 31–58% | **−4.4 pp** | [−19, +8] | **NOISE** |
| **FT plain** *(batch-2 control)* | 5 | 2 | 50.0% | 29–60% | **−3.4 pp** | [−21, +10] | **NOISE** |
| **FT DAgger** | 5 | 2 | 50.0% | 51–53% | **+2.0 pp** | [+1, +3] | **NOISE** (all 5 seeds ≥ 0) |
| **Vision plain** | 3 | 2 | 50.0% | 34–54% | **−8.3 pp** | [−16, +4] | **NOISE** |
| **Vision DAgger** | 3 | 2 | 50.0% | 46–62% | **+1.3 pp** | [−4, +12] | **NOISE** |

![interval chart of paired Δ success rate per recipe: mean over training seeds with whiskers to the lowest and highest seed](phase-1/success_rate_spread.png)

***Figure 1 — the headline.** Paired Δ success rate against `human_only`, one interval per
recipe: the square is the mean over training seeds, the whiskers reach the lowest and highest
seed, and each grey dot is one seed's own Δ. **What to conclude:** every whisker crosses Δ = 0
except FT DAgger's, and FT DAgger's mean (+2.0 pp) sits far inside the 27–31 pp span that the
*same* plain recipe covers when only its training seed changes. No recipe is distinguishable
from the baseline at this measurement's resolution. The one thing that does change is the
*width*: 27 pp and 31 pp for the plain arms, 2 pp for FT DAgger. Regenerate with
`uv run python scripts/dev/official_kpi/plot_kpis.py` ([provenance](kpi-dashboard.md#9-provenance--how-to-regenerate)).*

**No recipe clears the floor.** The FT-plain row *is* the floor — it is the same
one-recipe-many-seeds measurement as §5, and its 27 pp spread swallows every mean in the table.
The honest reading of all four rows is one sentence: **on the seeded measurement, none of
{F/T, vision} × {plain BC, DAgger} lifts closed-loop seating above the human-only baseline
beyond training-seed noise.**

Two second-order structure notes:

- **DAgger tightens the distribution without moving its center — and it is DAgger, not the batch
  size.** Both DAgger arms show a narrower seed spread (FT: [−19,+8]→[+1,+3]; vision:
  [−16,+4]→[−4,+12]) around a mean a hair above zero, and FT DAgger is the only arm where all
  five seeds land non-negative.

  This was **confounded when first observed**: FT plain trained at batch 16 and FT DAgger at
  batch 2, so the two arms differed in *two* variables, and an 8× smaller batch is itself a large
  change to optimization noise. The **batch-2 control row above resolves it.** Retraining FT
  plain at batch 2 — every other knob held — gives mean **−3.4 pp, range [−21, +10]**: a **31 pp
  spread, slightly *wider* than at batch 16**, and a near-identical center. Batch size moves
  neither the spread nor the center. The collapse from a 27–31 pp spread to 2 pp is therefore
  attributable to DAgger.

  It remains a **tighter draw of the same null, not a win** — +2 pp is inside the floor and every
  per-seed p is ≥ 0.68. What DAgger buys is *reliability*, not lift: it makes the recipe's outcome
  predictable without making it better. (Contrast the small-corpus DAgger collapse in [negative results](mechanisms.md#6-negative-results), which was
  dominated by force-abort states the bounded expert couldn't relabel — the larger, cleaner corpus
  removes the degradation but adds no lift.)
- **The noise lives on *both* axes.** Re-evaluating each intermediate DAgger round on the same
  100 held-out walls (per-round `trials.csv`, the vision backfill) shows the paired Δ swinging
  *within a single training seed* across rounds — seed-0 vision-DAgger ran **−1 → −28 → −12 → +8
  → −4** over rounds 0–4, a 36 pp range with no trend (Figure 4). A single round's checkpoint is
  as much a lottery draw as a single seed. Reported KPI is the **pre-committed final round**,
  never max-over-rounds (that is the optimistic-selection bias re-introduced on a new
  axis).

**This is the arc's closing measurement.** It does not overturn [what stands](mechanisms.md#7-what-still-stands) — the standing positives never
rested on a success rate. It is a multi-seed null on a fresh corpus. See
[`further-exploration.md`](further-exploration.md) for what (if anything) is worth further
compute.

---

