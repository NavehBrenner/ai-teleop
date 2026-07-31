# KPI Dashboard — the M5→M7 experiment ledger

The consolidated record of every training and evaluation experiment behind the policy arc
(M5 first behavioral clone → M6 F/T residual → M7 vision + DAgger): what was tried, its
config, its measured result, and why it did or did not work. It exists because the numbers
that decide the project's story were scattered across `outputs/policy/runs/`, `runs/eval*/`,
and a private wiki, with no single document that reconciles them.

**Every number here is recomputed from the raw artifacts**, not copied from prose —
per-trial CSVs re-aggregated through `ai_teleop.eval.report`, training configs read from each
run's committed `metadata.json`. The reconstruction script and the exact paths are in
[§8](#8-provenance--how-to-regenerate). Where an artifact disagrees with the wiki, both values
are logged and the artifact wins.

> ## Read this first — the noise floor
>
> **Retraining one fixed recipe, changing only the training seed, moves closed-loop success by
> more than any treatment in this document.** Measured on the official corpus
> ([§5](#5-the-noise-floor--how-it-was-measured)): the F/T recipe over five seeds spans **27 pp**
> of paired Δ (−19 to +8); the vision recipe over three seeds spans **20 pp** (−16 to +4). An
> earlier, 5× smaller corpus gave **18 pp** over five seeds — the same order, on different data.
>
> The environment contributes none of it. The `human_only` arm uses no checkpoint and returns
> **exactly 50.0%** in all twenty-one official evaluations — identical walls, operator, controller
> and budget. The spread is training randomness alone.
>
> That spread is this project's **measurement resolution**, and three rules follow:
>
> 1. **A single checkpoint is not a measurement.** Every closed-loop claim is reported as a
>    distribution over ≥3 training seeds, each carrying its `n`. A margin smaller than the seed
>    spread *plus* the eval-sampling interval at that n (±20 pp at n=20; ±10 pp at n=100) is a
>    **draw, not a finding** — flagged `NOISE` in the verdict column, never `WIN`/`REGRESSION`.
> 2. **A p-value inside one checkpoint pair says nothing about the recipe.** In the official run,
>    F/T seed 1 reads −19 pp at **p=0.0009** and vision-DAgger seed 1 reads +12 pp at **p=0.036**
>    — significant results pointing in *opposite* directions inside the same recipe families.
>    Significance there describes those two arms, not the recipe that produced them.
> 3. **The project's standing positive results are the bounded-force guarantee and the
>    mechanism findings** ([§7](#7-what-still-stands)) — *not* a success-rate lift, which on the
>    seeded measurement is not established.
>
> **Success is not the only KPI.** The other four — time-to-insert, peak contact force, contact
> events, jerk — are measured on the same trials and answer separately;
> [§5.6](#56-the-full-kpi-board--what-the-success-rate-alone-hid) reports all five per recipe,
> and the short version is that DAgger buys *reliability and a couple of newtons*, not seating.
>
> **Training was unseeded before 2026-07-23** — `torch.manual_seed` was absent and `--seed`
> reached only the train/val split, so weight init and batch order came from OS entropy while the
> run folder recorded a seed and a git commit. It is seeded now, with a
> train-twice-identical-weights test. Every pre-2026-07-23 number in this document is one
> unrepeatable draw and is listed as **history, not evidence**.
>
> One more, from finding H-11: **never compare an `expert_success_rate` to a residual success
> rate.** They are different actors, scored by different rules, at different difficulty — see
> the actor column in [§3](#3-training-runs-m5m7) and the note in [§2.3](#23-why-the-human-only-baseline-moves).

---

## 1. What the arc was trying to do

The policy is a **residual**: a human (here a scripted noisy operator) gives coarse 6-DoF
commands, and a behavioral-cloning-trained network adds a clamped micro-correction
(±2 cm / ±10° / ±5 N per step) on top of an always-on impedance backbone. The arc asked, in
order:

- **M5** — can a GRU clone the analytical expert's corrections at all? (offline BC)
- **M6** — does the F/T-only residual lift *closed-loop* insertion success over human-only?
- **M7** — does adding **vision** raise the success ceiling into the free-space regime, and can
  **DAgger** or a **better expert** push past the F/T ceiling?

**The answer to both closed-loop questions is no**, on the seeded multi-seed measurement (§5.5):
neither the F/T residual nor the vision residual lifts closed-loop success above the human-only
baseline beyond training-seed noise, with or without DAgger. What the arc contributes instead is
the bounded-force guarantee and a mechanism-level account of *why* per-step imitation cannot lift
closed-loop seating on this task (§6, §7). The numbers for both are below.

---

## 2. Operating-point ledger

Every closed-loop number is only interpretable against *which corpus trained the policy* and
*which difficulty the eval ran at*. These two tables are the key; the ledgers in §3–§4 point
back to them.

### 2.1 Corpus lineage (`data/dataset_*/metadata.json`)

| Corpus | Fingerprint | Created | n_ep | Schema | `expert_success_rate` | Role |
|---|---|---|---|---|---|---|
| `dataset_1` | `290f1750` | 2026-06-16 | 200 | 1.0 | — | M5 first BC (`lab34_baseline`). Old geometry/schema — **not comparable** to later corpora. |
| `dataset_9` | `54dccad9` | 2026-07-06 | 200 | 2.0 | 71.5% | Trained the 2026-07-07 M6 run. **Overwritten in place** — see caveat below. |
| `dataset_10` | `54dccad9` | 2026-07-22 | 200 | 2.0 | 71.5% | Regeneration of `dataset_9`'s config; trains every LAB-101/114 run. |
| `dataset_vision` | `de0eeb3b` | 2026-07-07 | 300 | 2.0 | 72.3% | All M6/M7 F/T + vision ablations (`ftonly_*`, `vision_*`). |
| `dagger_ft_agg` | `de0eeb3b` base | 2026-07-10 | 340→420 | 2.0 | — | Aggregated on-policy corpus, grows per DAgger round. |

**Two caveats the fingerprints hide:**

- **`dataset_9` and `dataset_10` share the fingerprint `54dccad9` but are not the same data**
  (finding G-4 / H-B). The fingerprint hashes *config*, not *code*; regenerating `dataset_9`'s
  config under 2026-07-22 code changed 35 of 200 trajectories (34 by a median of 1 step, one
  flipped baseline outcome, corpus baseline 22.5% → 23.0%). The original 2026-07-06 episode
  files were then **overwritten in place** by that regeneration — proven byte-for-byte by
  `scripts/dev/lab114_corpus_identity.py` — so `data/dataset_9/` now holds `dataset_10`'s
  arrays under `dataset_9`'s stale manifest. **The corpus that trained the 2026-07-07 M6 run no longer
  exists on disk.**
- **A "code era" column is load-bearing.** `dataset_0`/`dataset_1` (schema 1.0) already drift —
  their manifests predate `generated_walls` entering the fingerprint payload (finding C-1a).
  Do not trust byte-identical regeneration of any pre-LAB-91 corpus.

### 2.2 Eval operating points (`runs/eval*/trials.csv`)

| Eval set | `error_scale` | Seeds | Regime | `human_only` | Notes |
|---|---|---|---|---|---|
| `runs/eval/` (LAB-53) | 0.4 | 100 | in-band | **31.0%** | Older step-budget era (pre-LAB-100); zero force-aborts. |
| `eval_ftgate_es0p4` | 0.4 | 20 | in-band | 35.0% | M7 F/T gate ablation (3 arms). |
| `eval_ftgate_es1p0` | 1.0 | 20 | flat-wall | 15.0% | " |
| `eval_stageC_band04` | 0.4 | 20 | in-band | 35.0% | M7 Stage-C vision ablation (3 arms). |
| `eval_stageC` | 1.0 | 20 | flat-wall | 15.0% | " |
| `band_scale0.4` *(committed)* | 0.4 | 30 | in-band | **36.7%** | The 2026-07-07 M6 30-seed slice. |
| `flatwall_scale1.0` *(committed)* | 1.0 | 30 | flat-wall | 20.0% | That run's ceiling-check control. |
| `eval_lab101_band100*` | 0.4 | 100 | in-band | **50.0%** | LAB-101 reproduction (both ar0/ar100). |
| `eval_lab114_*` (×10) | 0.4 | 100 | in-band | **50.0%** | The seed-variance + H-B/H-C study. |

### 2.3 Why the `human_only` baseline moves

The three "contradictory" human baselines quoted across old docs — **36.7 / 31 / 50 / 35 / 15**
— are one number at different operating points, not a discrepancy:

- **50.0%** is the true in-band (es0.4) baseline, measured at 100 seeds, five independent times
  in LAB-114, all *exactly* 50.0% (the arm uses no checkpoint, so it is bit-stable).
- **36.7%** is that same baseline on the **hard 30-seed slice** (seeds 0–29) the 2026-07-07 run
  happened to draw: 36.7% on 0–29 vs 55.7% on 30–99.
- **31.0%** is the LAB-53 run at an **older step-budget era** (pre-LAB-100) — a different
  contact regime, not a different sample.
- **35% / 15%** are the 20-seed es0.4 / es1.0 baselines (M7 sets); es1.0 lands on the flat wall
  where nobody has a lateral lever, so everyone drops.

The stale `outputs/policy/kpi_report/kpi_comparison.json` (human **15.0%**, vision residual
`"PENDING"`) is a fourth operating point *and* an un-refreshed artifact; Phase-3 stage 3C
retires it. The lesson: **a bare success rate is meaningless without its (corpus, error_scale,
seed-count, step-budget-era) tuple.**

---

## 3. Training runs (M5→M7)

Reconstructed from `outputs/policy/runs/<name>/metadata.json`. `val` is `best_val_loss` at
`best_epoch/epochs_run`. **All GPU, seed 0, unless noted.** Offline val loss is *within-recipe*
predictive but **anti-predictive across interventions** (LAB-106) — do not rank recipes by it.

| Run | Date | Corpus (fp) | Config delta vs F/T baseline | `val` (epoch) | Closed-loop | Verdict |
|---|---|---|---|---|---|---|
| `lab34_baseline` | 06-18 | `dataset_1` (`290f`) | M5 first BC, schema-1.0 task | 0.00042 (12) | — no committed eval | offline milestone |
| `ftonly_baseline_lab82` | 07-07 | `dataset_vision` (`de0e`) | F/T residual, no action-rate penalty | 0.00149 (17) | see §4 (es0.4 20s) | M6/M7 F/T baseline |
| `ftonly_ar30` | 07-08 | `dataset_vision` | + action-rate penalty ×30 | 0.00116 (28) | — | jerk-reduction sweep |
| `ftonly_ar100` | 07-08 | `dataset_vision` | + action-rate penalty ×100 | 0.00140 (23) | 40% (es0.4, 20s) | the "old-ar100" M7 arm |
| `ftonly_wpos10_wd` | 07-10 | `dataset_vision` | pos-loss ×10 + weight-decay | 0.00169 (17) | — | LAB-106 offline fix (1/2) |
| `ftonly_gate_wpos10_wd` | 07-10 | `dataset_vision` | ↑ + **`command_ee_delta`** feedback feature + gate | 0.00135 (26) | **10%** (es0.4, 20s) | **REGRESSION** — see §6 |
| `vision_frozen_lab82` | 07-07 | `dataset_vision` | + vision, frozen MobileNetV3 encoder | 0.00107 (26) | — | best offline val of the arc — see caveat |
| `vision_frozen_ar100` | 07-08 | `dataset_vision` | ↑ + action-rate ×100 | 0.00123 (19) | — | |
| `vision_stageC` | 07-10 | `dataset_vision` | vision, **encoder unfrozen** (Stage C) | 0.00161 (16) | 40% in / 10% out (20s) | **NULL** — see §6 |
| `dagger_round0` | 07-10 | `dagger_ft_agg` (340) | on-policy relabel, round 0 | 0.00209 (20) | 40% (es0.4, 20s) | DAgger start |
| `dagger_round1` | 07-10 | `dagger_ft_agg` (380) | round 1 | 0.00214 (9) | 30% | **REGRESSION** (§6) |
| `dagger_round2` | 07-10 | `dagger_ft_agg` (420) | round 2 | 0.00194 (13) | 15% | **REGRESSION** (§6) |
| `probe_b2` | 07-07 | `dataset_vision_probe` (10) | vision batch-2 smoke probe | 0.00702 (2) | — | fits-in-8GB probe only |
| `lab101_ft_ar0_ds10` | 07-22 | `dataset_10` (`54dc`) | F/T recipe, GPU repro, ar0 | 0.00130 (22) | **−4.0 pp** (100s) | §5 |
| `lab101_ft_ar100_ds10` | 07-22 | `dataset_10` | ↑ + action-rate ×100 | 0.00178 (13) | **−9.0 pp** (100s) | §5 |
| `lab114_seed{0..4}` | 07-22 | `dataset_10` | F/T recipe, seeds 0–4 (**seeded**) | 0.00117–0.00197 | −15…+3 pp | §5 the spread |
| `lab114_ds9_seed{0..3}` | 07-22 | `dataset_9` | H-B corpus arm — **identical to `_seed`** | ≡ `lab114_seed*` | ≡ | H-B unanswerable |
| `lab114_cpu_seed0` | 07-22 | `dataset_10` | H-C device arm, CPU | 0.00144 (21) | −1.0 pp (100s) | H-C null |

**The offline-val trap, stated once.** `vision_frozen_lab82` has the *best* val loss of the
whole arc (0.00107) and is a closed-loop non-improver; the F/T recipe's own five seeds
span 18 pp of success at val losses 0.00117–0.00197. **A lower validation loss did not buy
closed-loop success across these interventions** — the central M7 mechanism (LAB-106,
[§7](#7-what-still-stands)).

---

## 4. Closed-loop experiment ledger

Reconstructed from the per-trial CSVs via `compare_paired`. Δ is paired (McNemar exact p);
`b/c` is discordant wins/losses. **Verdict uses the noise floor (§5):** `NOISE` = |Δ| within the
20–27 pp training-seed spread + the eval interval at that n.

| Eval set | Op. point | Arm | Success | Paired Δ (n, p) | Verdict |
|---|---|---|---|---|---|
| `flatwall_scale1.0` | es1.0, 30s | residual | 20.0% vs 20.0% | +0.0 pp (30, p=1.0) | flat-wall ceiling control (expected) |
| `runs/eval/` (LAB-53) | es0.4, 100s, old budget | residual | 43.0% vs 31.0% | +12.0 pp (100, p=0.043) | **NOISE** — inside the floor; unseeded training (H-7) |
| `eval_ftgate_es0p4` | es0.4, 20s | ar100 (`residual`) | 40.0% vs 35.0% | +5.0 pp (20, p=1.0) | **NOISE** |
| `eval_ftgate_es0p4` | es0.4, 20s | `command_ee_delta` (`ftonly`) | **10.0%** vs 35.0% | −25.0 pp (20, p=0.125) | **REGRESSION** (mechanism, §6) |
| `eval_ftgate_es1p0` | es1.0, 20s | ar100 / gate | 20% / 15% vs 15% | ≤+5 pp (20, p=1.0) | **NOISE** (flat wall) |
| `eval_stageC_band04` | es0.4, 20s | ftonly / **vision** | 40% / **40%** vs 35% | +5 / +5 pp (20, p=1.0) | **NULL** — vision ties F/T (§6) |
| `eval_stageC` | es1.0, 20s | ftonly / **vision** | 20% / **10%** vs 15% | +5 / −5 pp (20, p=1.0) | **NOISE** (margin < floor) |
| `eval_lab101_band100_ar0` | es0.4, 100s, `dataset_10` | residual | 46.0% vs 50.0% | **−4.0 pp** (100, p=0.557) | reproduction — §5 |
| `eval_lab101_band100` | es0.4, 100s, `dataset_10` | residual (ar100) | 41.0% vs 50.0% | **−9.0 pp** (100, p=0.136) | reproduction — §5 |
| `eval_lab114_seed2` | es0.4, 100s | residual | 35.0% vs 50.0% | **−15.0 pp** (100, **p=0.008**) | a *significant* regression from nothing but a seed — §5 |
| `eval_lab114_seed4` | es0.4, 100s | residual | 53.0% vs 50.0% | +3.0 pp (100, p=0.74) | the other end of the same spread |
| **`eval_official_ft_s*`** | es0.4, 100s, official corpus | residual ×5 seeds | 31–58% vs 50.0% | **mean −4.4 pp** [−19,+8] | **NOISE** (distribution — §5.5) |
| **`eval_official_dag_ft_s*`** | es0.4, 100s, official | residual ×5 seeds | 51–53% vs 50.0% | **mean +2.0 pp** [+1,+3] | **NOISE** (tightest — §5.5) |
| **`eval_official_vis_s*`** | es0.4, 100s, official | vision ×3 seeds | 34–54% vs 50.0% | **mean −8.3 pp** [−16,+4] | **NOISE** (§5.5) |
| **`eval_official_dag_vis_s*`** | es0.4, 100s, official | vision ×3 seeds | 46–62% vs 50.0% | **mean +1.3 pp** [−4,+12] | **NOISE** (§5.5) |

The Stage-C DAgger rounds (`eval` per round, 20s es0.4) read **40% → 30% → 15%** across rounds
0–2 — the round-to-round steps are inside the noise floor, but the round-0-to-round-3 drop and
the parallel rollout-success decline (0.325 → 0.25) are the real signal (§6).

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

**On the official corpus** (`dataset_official_ft` / `_vision`, 1000 episodes, es0.4, 100 paired
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

**Corroborated on independent data.** The floor was first measured (LAB-114, 2026-07-22) on
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
predictive *within one recipe*, the opposite of its behavior *across* interventions (§6).
Selecting the best-val checkpoint of a fixed recipe is therefore fine; tuning *recipes* by val
loss is not.

![val loss vs closed-loop success across seeds](phase-1/lab114_val_loss_vs_success.png)

---

## 5.5 The official multi-seed run — the definitive measurement

§5 established the measurement resolution; this section is the measurement the project stands
on. Per the D-6 mandate: a fresh **~1000-episode** official corpus (F/T
and vision, separate), each of the four production recipes **retrained over multiple seeds** and
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
`uv run python scripts/dev/official_kpi/plot_kpis.py` (§8).*

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
  predictable without making it better. (Contrast the small-corpus DAgger collapse in §6, which was
  dominated by force-abort states the bounded expert couldn't relabel — the larger, cleaner corpus
  removes the degradation but adds no lift.)
- **The noise lives on *both* axes.** Re-evaluating each intermediate DAgger round on the same
  100 held-out walls (per-round `trials.csv`, LAB-112 backfill) shows the paired Δ swinging
  *within a single training seed* across rounds — seed-0 vision-DAgger ran **−1 → −28 → −12 → +8
  → −4** over rounds 0–4, a 36 pp range with no trend (Figure 4). A single round's checkpoint is
  as much a lottery draw as a single seed. Reported KPI is the **pre-committed final round**,
  never max-over-rounds (that is the LAB-114 optimistic-selection bias re-introduced on a new
  axis).

**This is the arc's closing measurement.** It does not overturn §7 — the standing positives never
rested on a success rate — and it converts the documented negative from "one unreproducible
checkpoint" into a rigorous, multi-seed, fresh-corpus null. See
[`further-exploration.md`](further-exploration.md) for what (if anything) is worth further
compute.

---

## 5.6 The full KPI board

Success is one of the KPIs the harness records; `time_to_insert_s`, `peak_contact_force` and
`jerk_integral` are measured on the same trials of the same evals. (`contact_events` is recorded
but not reported — it reads exactly 1 on every trial of every arm, `human_only` included, so at
this operating point it separates nothing.) The complete
board — every recipe × every metric, mean over training seeds with the observed range, the paired
comparison against `human_only`, the paired DAgger-vs-plain comparison, and the raw per-seed
draws — is generated into
[`phase-1/official_kpi_tables.md`](phase-1/official_kpi_tables.md) by
`scripts/dev/official_kpi/kpi_tables.py`. The summary:

| Metric | dir. | `human_only` | FT plain | FT plain (b2) | FT DAgger | Vision plain | Vision DAgger |
|---|---|---|---|---|---|---|---|
| **Success rate (%)** | ↑ | 50.0 | 45.6 [31.0, 58.0] | 46.6 [29.0, 60.0] | 52.0 [51.0, 53.0] | 41.7 [34.0, 54.0] | 51.3 [46.0, 62.0] |
| **Time to insert (s)** | ↓ | 7.83 | 7.91 [7.12, 8.42] | 7.47 [7.28, 7.79] | 7.28 [6.88, 7.47] | 7.46 [7.06, 8.05] | 7.62 [6.85, 8.02] |
| ↳ n (seated trials) | | 50 | 31–58 | 29–60 | 51–53 | 34–54 | 46–62 |
| **Peak contact force (N)** | ↓ | 23.97 | 25.79 [23.93, 29.02] | 24.26 [21.21, 28.80] | 22.92 [21.66, 24.07] | 24.68 [23.37, 25.58] | 23.49 [20.92, 25.18] |
| **∫\|jerk\|** | ↓ | 45.60 | 70.10 [59.62, 95.18] | 52.69 [46.60, 61.89] | 85.12 [45.08, 231.74] | 57.25 [46.01, 78.70] | 48.03 [45.84, 50.04] |

![four interval charts, one per reported KPI, showing each recipe's mean and min/max over training seeds against the human_only line](phase-1/kpi_spread_by_recipe.png)

***Figure 2 — every KPI, every recipe.** Absolute units, the dashed red line the `human_only`
value on the same walls. **What to conclude:** the seed spread that swallows the success rate is
not a success-rate phenomenon — it is present in *every* continuous KPI, and in jerk it is
larger still. Note the `n=` labels under *Time to insert*: that KPI is defined only on trials
that seated, so its means rest on 29–62 trials, not 100.*

**Time to insert — every treatment is slower, and the marginal means say the opposite.**
The paired Δ (both arms seated the same wall) is positive for all five recipes: **+0.14 s to
+0.27 s** on a 7.83 s baseline. Yet three recipes show a *lower* marginal mean than
`human_only` — `FT plain (batch 2)` reads 7.47 s against 7.83 s while its paired Δ is **+0.27 s
slower**. That gap is survivorship: an arm that seats fewer walls is seating the easier ones, and
its mean is over a different population. Read the paired column; the marginal column is not a
comparison.

**Peak contact force — the one KPI where DAgger moves the mean, not just the spread.**
Plain BC *raises* peak force above the baseline (FT plain **+1.83 N** paired, per-seed p reaching
<0.001); both DAgger arms *lower* it (FT DAgger **−1.04 N**, Vision DAgger **−0.48 N**). Compared
directly against plain BC at matched training seed, **FT DAgger is gentler on all five seeds**
(mean **−2.87 N**, range [−5.93, −0.37]); against the batch-2 control the mean is **−1.34 N**
[−5.70, +0.45], so the sign is not unanimous once batch size is held. Vision: **−1.19 N**
[−4.66, +1.82]. The direction is consistent, the magnitude is a couple of newtons, and none of
it approaches the abort threshold. The eval observer ends a trial
above **30 N** (`DEFAULT_FORCE_CAP`, `eval/observer.py`); that terminates the trial rather than
bounding the force, and the peak recorded on such a trial reaches 77.86 N (§5.6.2). 23.97 N is
the `human_only` mean, not a bound.

**Jerk — the smoothness cost is real, still positive everywhere, and the FT-DAgger number is one
seed.** Every treatment raises ∫|jerk| above `human_only`; none lowers it. Vision DAgger is
nearest flat (**+2.43** paired, range [+0.24, +4.44] — statistically non-zero, practically ~5%).
Two things visible only per seed:

- **FT DAgger's 85.12 mean is a single outlier.** Its five seeds are 45.08 / 52.64 / **231.74** /
  47.96 / 48.19. Four of them sit at or below the batch-2 plain arm; one seed is 4.5× the others.
  Quoting 85.1 as "FT DAgger's jerk" describes no checkpoint that exists.
- **Batch size, not just DAgger, moves jerk.** FT plain at batch 16 gives 70.10; the *same recipe*
  at batch 2 gives 52.69. Part of the smoothness story previously credited to DAgger is an
  optimisation-noise effect the batch-2 control isolates.

### 5.6.1 The same KPIs on the success group only

`peak_contact_force` and `jerk_integral` are recorded on **every** trial, so every number above
mixes two populations: the runs that inserted, and the runs that ran out of budget or tripped the
force cap. A treatment that merely *fails* differently — loading harder against the wall before
giving up — moves those means without changing anything about how it behaves while seating.
Restricting to walls where **both** arms seated separates the two. (Success rate is degenerate on
that subset by construction, and time-to-insert is already seated-only, so only these two metrics
have two populations to compare.)

![two rows by two columns: peak contact force and jerk, each over all matched walls and over the both-arms-seated subset](phase-1/kpi_population_split.png)

***Figure 5 — the two always-on KPIs by population.** Top row all matched walls, bottom row the
seated subset; the dashed line is `human_only` *for that population*, which moves too. **What to
conclude:** the population, not the treatment, is the larger effect on both metrics — and it moves
them in opposite directions.*

**Peak force nearly halves on the success group: 15.46 N against 23.97 N for `human_only`.** The
high all-trials figure is mostly the failures, which include force-cap trips. 30 N is where the watchdog *aborts*, not a force the
system cannot reach — see §5.6.2. **DAgger's advantage survives the split** — FT DAgger −1.04 N over all trials and −0.59 N
seated-only, Vision DAgger −0.48 N and −1.05 N — so it is a property of how the controller seats,
not of how often it fails.

**Jerk moves the other way: `human_only` rises from 45.60 to 64.90 on the seated subset.** A
successful insertion involves more corrective motion near the hole; a failed run often aborts
early having accumulated less. So the smoothness cost is *understated* by the all-trials view.
Two recipes change character under the split: **Vision DAgger goes from +2.43 to −0.21**, i.e.
indistinguishable from the human when both seat, while **FT plain (batch 2) improves from +7.09 to
+3.80** and **FT DAgger worsens from +39.52 to +47.03** (still one outlier seed).

The general point is worth stating once: for any KPI recorded on all trials, *which trials are in
the average* is a design decision, and here it is worth more than any treatment effect in the
table. Both populations are reported in
[`phase-1/official_kpi_tables.md`](phase-1/official_kpi_tables.md) §5.

**Contact events — recorded, not reported.** Exactly **1.00** on every trial of every arm,
`human_only` included. The metric counts hysteresis-debounced rising edges past the contact floor,
and at this operating point the approach makes one sustained contact and stays in it, so it
separates nothing. It stays in the recorded schema and is dropped from the reported set; a lower
force floor or a bouncing regime would make it informative again.

### 5.6.2 The distribution behind the force mean

Every chart above plots a distribution over **training seeds**, where one point is a seed's mean
over 100 trials — it answers *how much does retraining move the average?* This one plots the
**individual trials**, which answers a different question and gives a different answer.

![six panels, one per arm, each a histogram of per-trial peak contact force stacked by outcome, with the commanded-force bound and the watchdog threshold marked](phase-1/trial_force_distribution.png)

***Figure 6 — peak contact force per trial, by arm and outcome.** Green line: the ≈18.9 N
commanded-force bound (stiffness × command clamp). Black dashed: the 30 N watchdog abort.
**What to conclude:** the distribution is bimodal and the mean falls between its two modes.*

**The mean describes no trial that ran.** Peak force is **bimodal** — a seated cluster below
~20 N and a force-abort cluster above 30 N — so `human_only`'s 23.97 N mean sits in the trough
between them. Its per-trial spread is **8.60 to 54.70 N, SD 12.51**, against a spread across
training seeds of **exactly zero** (it uses no checkpoint, so all 21 evals return the identical
number). Those are two different variances and only one of them is what the other figures show.

**The 30 N line is visibly a cut, not a ceiling.** Successes and timeouts stop dead at it and
force-aborts begin there, because exceeding it *is* what makes a trial a `force_abort`. The
recorded peak on those trials runs to **77.86 N** — the force spikes within the tick before the
watchdog fires.

**The ≈18.9 N commanded-force bound is not a bound on the measurement.** 33% of *successful*
trials sit above it. The quasi-static `K·Δx` argument bounds what the controller can *ask for*;
the F/T sensor reads the contact reaction, which carries impact transients and the full distal
load. Both statements are true and they are about different quantities — §7 now separates them.

**The policies' worst impacts are harder than the human's** (max 64–78 N against 54.70 N) while
their *rate* of hard impacts is lower for both DAgger arms. Fewer bad contacts, but a heavier
tail when one happens.

Regenerate with `uv run python scripts/dev/official_kpi/plot_trial_forces.py`.

### Did DAgger produce any result at all?

![two rows (F/T, vision) by four columns (KPI) of interval charts comparing human_only, plain BC and DAgger](phase-1/dagger_vs_plain.png)

***Figure 3 — the three arms side by side.** Rows are modality, columns are KPI; each panel puts
`human_only` (red diamond), plain BC and DAgger on one axis, and prints the paired
DAgger − plain delta per training seed above the data. **What to conclude:** on success, DAgger's
interval collapses onto the baseline rather than rising above it — reliability, not lift. On peak
force it sits below both the baseline and plain BC in both modalities. On jerk it is the tightest
arm in vision and the widest in F/T (that one outlier seed). On time-to-insert and contact events
it changes nothing.*

The honest three-line answer to "did DAgger do anything":

1. **On success — no.** +6.4 pp against FT plain and +9.7 pp against Vision plain sound large, but
   the per-seed range is [−6, +20] and [−8, +28]; against the batch-2 control it is +5.4 pp
   [−7, +22]. Every one of those intervals contains zero, and the *absolute* DAgger rate (52.0%,
   51.3%) is inside the noise band around the 50.0% baseline.
2. **On seed-to-seed variance — yes, and this is the real finding.** FT DAgger's five seeds land
   in a **2 pp** band where the same recipe without DAgger spans 27 pp (batch 16) or 31 pp
   (batch 2). Same corpus, same batch size, same everything but the on-policy relabeling. DAgger
   makes the outcome *predictable* without making it *better*.
3. **On smoothness and contact force — partly, with a caveat.** Peak force drops consistently
   relative to plain BC; jerk drops in vision (−9.22 paired vs plain) and the batch-2 control
   shows part of the F/T improvement is batch size, not DAgger.

### DAgger across rounds

![four panels, one per reported KPI, each plotting the metric against DAgger round with one line per training seed](phase-1/dagger_rounds_vision.png)

***Figure 4 — vision DAgger, every round, every seed.** One line per training seed so the reader
sees the round axis is noisy *within* a seed. **What to conclude:** there is no round trend to
select on. Seed 0 runs 49 → 22 → 38 → 58 → 46 % success (paired Δ −1 → −28 → −12 → +8 → −4, a
36 pp swing inside one training run); seed 1 ends on its best round (+12 pp, p=0.036) and seed 2
on its worst. Peak force and jerk wander the same way. Reporting the pre-committed final round is
what keeps this from becoming a max-over-rounds selection bias.*

**The F/T DAgger arm has no per-round evaluation at all.** Its round checkpoints under
`outputs/policy/runs/dag_ft_s*/dagger_round*/` hold weights and training history but no
`trials.csv` — the intermediate rounds were never scored on the held-out walls, and the LAB-112
backfill covered vision only. Nothing about how the F/T arm evolved across rounds can be claimed
from this run; the gap is stated rather than filled.

---

## 6. Negative results

Surfacing the failures is an explicit goal of this document — each is a mechanism, not just a
missing win.

- **Action-rate penalty — works exactly as designed, and it is *not* the arc's problem.**
  `ar0` vs `ar100` on `dataset_10` are indistinguishable on success (46% vs 41%, inside the
  floor) but jerk drops **153.6 → 85.7** (p<1e-15). The penalty buys smoothness at no success
  cost — which *retires* the "apply the action-rate penalty" candidate:
  it was already applied and does nothing to success.
- **The offline-fix collapse (`command_ee_delta`) — REGRESSION, and the sharpest mechanism.**
  Adding a `(command − ee_position)` **feedback feature** + pos-loss ×10 drove offline error
  below the zero-Δ baseline for the first time (7.6 → 3.5 mm) — and closed-loop success
  **collapsed to 10%** (`ftonly_gate_wpos10_wd`, es0.4, vs 35% human), with *more* force-aborts.
  Online, the policy amplifies its own tracking error into wall-slams. **A more accurate
  imitator is a worse controller** — offline BC fidelity is *anti*-correlated with closed-loop
  success on this task (LAB-106).
- **DAgger degrades, it doesn't rescue — REGRESSION.** Three F/T rounds on the ar100 base:
  **40% → 30% → 15%** (rollout success 0.325 → 0.25). Mechanism: the policy's rollouts are
  dominated by force-abort states, and the **bounded analytical expert cannot demonstrate a
  recovery** from a peg pinned at the force cap — so each round aggregates more failure states
  labeled with passive Δ, and the clone gets more passive. DAgger's founding premise (a
  competent expert on visited states) is structurally violated.
- **Stage-C vision fine-tune — NULL.** Unfreezing the image encoder (`vision_stageC`) ties
  F/T-only in-band (40% vs 40%, es0.4) and loses out-of-band (10% vs 20%, es1.0, inside the
  floor). Vision carries little marginal signal because **the operator command already proxies
  the hole location** (LAB-77 identifiability); the free-space correction the clone would learn
  is ≈0 by construction.
- **A better analytical expert — REFUTED (LAB-108).** Five expert knobs meant to prevent the
  slam were all inert; the expert's own ceiling stayed at ~73.3%. The binding constraint is
  operator-originated, pre-contact force-abort, which a bounded residual cannot fix.

---

## 7. What still stands

Two classes of result do **not** rest on a sampled success rate, so LAB-114 leaves them intact.
These are the project's standing positives.

- **The force argument, stated precisely.** Three things are true by construction, and one
  commonly-assumed fourth is **not**:

  1. **The residual is clamped** — ±3 cm / ±10° / ±5 N per step, applied *before* the controller
     sees the augmented command (`domain/delta.py`). A maximally wrong network cannot enlarge its
     own authority.
  2. **The commanded restoring force is bounded at ≈18.9 N.** The impedance backbone's
     translational stiffness is `[400, 400, 500]` N/m and the per-step command clamp is 0.025 m
     (`control/backbone.py`), so `‖K·Δx‖ ≤ 18.9 N` is the most force the controller can ever
     *ask* for.
  3. **No trial continues past 30 N** — the eval observer aborts it (`eval/observer.py`).
  4. **Measured contact force is *not* bounded.** The wrist F/T sensor reads the contact
     *reaction*, which includes impact transients the quasi-static `K·Δx` argument says nothing
     about. **1712 of 4200 official trials exceed 30 N, reaching 77.86 N** — every one of them a
     `force_abort`, the overshoot occurring within the tick before the watchdog fires. 33% of
     *successful* trials also exceed 18.9 N. See [§5.6.2](#562-the-distribution-behind-the-force-mean).

  What the measurements *do* support is a comparison rather than a bound, and two independent
  metrics agree on it: **DAgger lowers both the mean peak force and the force-abort rate; plain
  BC at batch 16 raises both.**

  | Arm | mean peak force vs baseline | force-abort rate (baseline 41.0%) |
  |---|---|---|
  | FT DAgger | **−1.04 N** | **36.4%** (−4.6 pp) |
  | Vision DAgger | **−0.48 N** | **38.3%** (−2.7 pp) |
  | FT plain (batch 2) | +0.30 N | 39.0% (−2.0 pp) |
  | Vision plain | +0.71 N | 44.3% (+3.3 pp) |
  | FT plain | +1.83 N | 45.2% (+4.2 pp) |

  Two measures that could have disagreed do not, which is what makes this the arc's most solid
  positive result — stronger than the success-rate null and, unlike it, consistent in sign.
- **The mechanism findings**, each theory or a byte-identical/exact probe:
  - **Identifiability ceiling** (LAB-77) — the operator command proxies the hole; a no-vision
    residual cannot lift success outside the chamfer band. A structurally-flat flat-wall delta
    is a *result*, not a failure.
  - **Far-field gating failure** (LAB-106) — trained GRUs emit a ~5.6 mm correction floor
    across the ~60% free-space steps where the expert is exactly zero.
  - **Offline/closed-loop anti-correlation** (LAB-106) — fixing offline BC error made
    closed-loop worse; only a closed-loop ablation is a valid signal here.
  - **The bounded-expert/DAgger argument** (LAB-105/106) — on-policy relabeling can only teach
    what the expert can perform, and it cannot un-jam a force-aborted peg.

An honest engineering summary: **Phase 1 delivers an assist whose authority is bounded by
construction and which measurably reduces contact force and force-aborts, plus a mechanized
account of why per-step imitation cannot lift closed-loop seating success on this task. The
success-rate *lift* is, on the seeded measurement, not established.**

---

## 8. Provenance & how to regenerate

Every table above is a pure function of committed artifacts. Re-aggregate any eval set:

```bash
# One eval set → success + paired McNemar + per-KPI Wilcoxon, from raw per-trial rows.
uv run python scripts/report_results.py --trials runs/eval_lab101_band100_ar0/trials.csv

# The committed Phase-1 records (30-seed slice, flat-wall, seed-variance, H-C) live here:
ls docs/results/phase-1/*.csv docs/results/phase-1/lab114/

# The §5.5 official multi-seed success distributions (all recipes, over training seeds):
uv run python scripts/dev/official_kpi/aggregate.py       # reads runs/eval_official_*

# The §5.6 full-KPI board — every recipe × every metric, as markdown on stdout:
uv run python scripts/dev/official_kpi/kpi_tables.py      # → docs/results/phase-1/official_kpi_tables.md

# Figures 1–5 — the same statistics as plain matplotlib box charts:
uv run python scripts/dev/official_kpi/plot_kpis.py       # → docs/results/phase-1/*.png
```

All three read the eval CSVs through `ai_teleop.eval.report` (`load_trials` → `group_by_config` →
`summarize_config` / `compare_paired`) and re-derive no statistic of their own;
`scripts/dev/official_kpi/kpi_data.py` is the shared loader. Both entry points take
`--runs-root` / `--policy-runs-root` if the eval sets live elsewhere.

The §5.5/§5.6 official-run eval sets (`runs/eval_official_*`, `runs/backfill_dag_vis_s0_r*`) and
per-round DAgger `trials.csv` (`outputs/policy/runs/dag_*_s*/dagger_round*/`) are **local
artifacts, gitignored** like the rest of `runs/`/`outputs/` — the committed record is the tables,
`phase-1/official_kpi_tables.md` and the four figures, plus the read-only scripts (regenerable
only while the eval dirs still exist locally). The chunk
scripts that produced them are `scripts/dev/official_kpi/*.sh` (self-resumable, self-timing).

Training configs are each run's committed `outputs/policy/runs/<name>/metadata.json`. Post-G1
runs carry a `checkpoint_sha256`; the two pre-G1 checkpoints behind published numbers are
committed under `docs/results/phase-1/checkpoints/` (retention policy: that dir's README).

**Three provenance gaps this ledger inherits, stated so no reader trusts a number past them:**

| Gap | Finding | Consequence |
|---|---|---|
| The 2026-07-07 M6 **checkpoint** is gone | H-8 (`outputs/` gitignored) | 70.0% cannot be re-evaluated. |
| That run's **corpus** was overwritten in place | H-B / G-4 | `dataset_9`'s trajectories are unrecoverable. |
| `dataset_0`/`dataset_1` fingerprints predate `generated_walls` | C-1a | Pre-LAB-91 corpora do not regenerate byte-identically. |

The reconstruction that built §3–§4 is a read-only sweep over `outputs/policy/runs/`,
`runs/eval*/`, and `data/dataset_*/` — see the LAB-114 evidence scripts
(`scripts/dev/lab114_corpus_identity.py`, `lab114_weight_distance.py`) and the report audit
(`scripts/dev/lab42_report_audit.py`).

---

**See also:** [`architecture-tour.md`](../guides/architecture-tour.md) (where each of
these modules lives).
