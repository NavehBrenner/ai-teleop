# The KPI Board — every recipe, every metric

Success rate is one of the KPIs the harness records. This file reports all of them per
recipe, as distributions over training seeds: the headline board, the same metrics
restricted to the trials that actually seated, the plain-BC-versus-DAgger comparison, and
DAgger's behaviour across its rounds.

Read [the noise floor](noise-floor.md) first — it sets the resolution every number here
has to clear.

**Related:** [experiment ledger](experiment-ledger.md) · [noise floor](noise-floor.md) ·
[within-seed](within-seed.md) · [mechanisms](mechanisms.md) · [index](kpi-dashboard.md)

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
bounding the force, and the peak recorded on such a trial reaches 77.86 N ([within-seed.md](within-seed.md)). 23.97 N is
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

**Peak force nearly halves on the success group: 14.77–15.55 N against 23.97 N for `human_only`**
(a range, not a constant — the seated subset differs per recipe, so each row carries its own
baseline). The
high all-trials figure is mostly the failures, which include force-cap trips. 30 N is where the watchdog *aborts*, not a force the
system cannot reach — see [within-seed.md](within-seed.md). **DAgger's advantage survives the split** — FT DAgger −1.04 N over all trials and −0.59 N
seated-only, Vision DAgger −0.48 N and −1.05 N — so it is a property of how the controller seats,
not of how often it fails.

**Jerk moves the other way: `human_only` rises from 45.60 to 57.69–64.90 on the seated subset**
(per-recipe, since the subset differs). A
successful insertion involves more corrective motion near the hole; a failed run often aborts
early having accumulated less. So the smoothness cost is *understated* by the all-trials view.
Two recipes change character under the split: **Vision DAgger goes from +2.43 to −0.21**, i.e.
indistinguishable from the human when both seat, while **FT plain (batch 2) improves from +7.09 to
+3.80** and **FT DAgger worsens from +39.52 to +47.03** (still one outlier seed).

The general point is worth stating once: for any KPI recorded on all trials, *which trials are in
the average* is a design decision, and here it is worth more than any treatment effect in the
table. Both populations are reported in
[`phase-1/official_kpi_tables.md`](phase-1/official_kpi_tables.md).

**Contact events — recorded, not reported.** Exactly **1.00** on every trial of every arm,
`human_only` included. The metric counts hysteresis-debounced rising edges past the contact floor,
and at this operating point the approach makes one sustained contact and stays in it, so it
separates nothing. It stays in the recorded schema and is dropped from the reported set; a lower
force floor or a bouncing regime would make it informative again.

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

