# Conclusions

What this project set out to do, what it delivered, what it measured, and what the
measurements support. The evidence lives in [`docs/results/`](results/kpi-dashboard.md);
this page is the reading of it.

---

## 1. The question

Can a vision-conditioned residual policy, trained by behavioral cloning on an analytical
expert, make a human operator measurably better at precision peg-in-hole insertion?

"Better" was defined before the experiments ran, as five success criteria in the design
document ([§8](design-document.md#8-evaluation-criteria)). Two are about insertion
success rate, one about contact force, one about the software, one about the operator
interface.

The system is a **residual assist**: an operator issues coarse 6-DoF commands, an always-on
impedance backbone tracks them, and a trained network adds a clamped micro-correction on top
at 500 Hz. The operator in every reported measurement is a *scripted* noisy operator — open
loop, seeded, reproducible — which is what makes a paired comparison possible at all.

---

## 2. What was delivered

A complete, runnable system, and this part met its criteria.

- **The simulation and control stack** — procedurally generated walls, a Franka Panda with a
  compliant peg, a Cartesian impedance backbone, a three-layer safety envelope with a
  trip-and-lock force watchdog.
- **Two swappable seams** — `InputStrategy` (scripted operator ↔ live stereo hand tracking)
  and `AssistProvider` (none ↔ analytical expert ↔ trained policy). Both are `Protocol`s;
  switching either is a one-argument change. The headline experiment is literally the same
  code path run twice with a different argument.
- **The full imitation pipeline** — corpus generation, behavioral cloning (F/T-only and
  vision-conditioned), DAgger on-policy relabelling, and a paired evaluation harness that
  emits per-trial CSVs.
- **A live teleoperation path** — metric 3D hand tracking from two webcams, split out as the
  standalone [`stereohand`](https://github.com/NavehBrenner/stereohand) package and
  integrated back as a dependency.
- **The measurement infrastructure** — every number in the results is recomputed from
  committed artifacts by a script — the 4200 raw per-trial rows behind every figure are in
  [`results/phase-1/official-evals/`](results/phase-1/official-evals/), and every checkpoint
  behind a published number is committed and runnable from a clean clone.

---

## 3. What the measurements say

### 3.1 Insertion success rate — no lift

The definitive measurement is the official multi-seed run: each production recipe retrained
across training seeds, every checkpoint evaluated on the **same 100 held-out walls**, paired
against the same operator.

| Recipe | seeds | mean Δ vs human-only | range across seeds |
|---|---|---|---|
| F/T plain BC (batch 16) | 5 | −4.4 pp | [−19, +8] — **27 pp** |
| F/T plain BC (batch 2) | 5 | −3.4 pp | [−21, +10] — **31 pp** |
| F/T DAgger | 5 | **+2.0 pp** | [+1, +3] — **2 pp** |
| Vision plain BC | 3 | −8.3 pp | [−16, +4] — **20 pp** |
| Vision DAgger | 3 | **+1.3 pp** | [−4, +12] |

**No recipe lifts insertion success beyond the resolution of the measurement.** The two
DAgger recipes are positive in the mean; neither margin approaches the spread that retraining
the plain recipe alone produces. Across all 21 checkpoints, 9 are above the baseline and 12
below.

That spread is the central methodological finding. Retraining a fixed *plain-BC* recipe,
changing **only the training seed**, moves the paired outcome by 20–31 pp; the two DAgger
families are tighter (2 pp and 16 pp) around means that are themselves near zero. The environment contributes
none of it: the `human_only` arm uses no checkpoint and returns **exactly 50.0%** in all 21
evaluations — identical walls, operator, controller and budget. An independent, five-times
smaller corpus reproduces the same order of magnitude (18 pp).

The consequence is stated once and obeyed everywhere: **a single checkpoint is not a
measurement of its recipe.** Inside the F/T plain recipe, seed 1 reads a significant regression
(−19 pp, p=0.0009) while seed 2 reads +8 pp; across recipes, vision-DAgger seed 1 reads a
significant +12 pp (p=0.036) while its two sibling seeds both read −4.

→ [The noise floor](results/noise-floor.md)

### 3.2 Contact force — no established effect

DAgger's recipe-level means are negative (FT DAgger −1.04 N, Vision DAgger −0.48 N) and plain
BC's are positive. Applying the same noise-floor test the success rate is judged by, that
pattern does not survive.

| Recipe | mean Δ peak force | its own training-seed floor |
|---|---|---|
| F/T plain BC (batch 16) | +1.83 N | 5.10 N |
| F/T plain BC (batch 2) | +0.30 N | 7.59 N |
| F/T DAgger | −1.04 N | 2.41 N |
| Vision plain BC | +0.71 N | 2.21 N |
| Vision DAgger | −0.48 N | 4.26 N |

Every mean is smaller than the spread its own recipe produces by retraining alone. Counting
signs across all 21 checkpoints, **11 of 21 are above the baseline** — indistinguishable from
the success rate's 9 of 21. The two DAgger arms disagree with each other at seed level: F/T
DAgger is gentler on 4 of 5 seeds, Vision DAgger *rougher* on 2 of 3. No per-seed Wilcoxon for
F/T DAgger peak force reaches p = 0.05.

Mean force and force-abort rate are not independent confirmation of each other — the abort rate
is a threshold count of the same quantity on the same trials.

Two things remain true and are worth separating from the null. Paired wall by wall, the policy
is the gentler arm on only 31–58 of 100 walls across all 21 checkpoints (41–58 restricted to
the checkpoints whose mean actually dropped), so even the negative means are carried by a
minority of walls. And the policies' **worst single impacts are harder than the operator's**
(64–78 N against 54.70 N).

### 3.3 Trajectory smoothness — a real cost, and the clearest effect in the project

**20 of 21 checkpoints are rougher than the operator alone.** That near-unanimous sign across
every recipe, modality and batch size is stronger evidence than any mean clearing a floor, and
it is the only KPI besides time-to-insert where the direction is established at all. The
magnitude is not: the per-seed spread reaches 186 on F/T DAgger, driven by one outlier seed.

It is also a per-wall property rather than an average — F/T plain BC is the rougher arm on
**79–95 of 100 walls**, and `Vision DAgger`, whose seed-level mean reads nearly flat, is still
rougher on 64–71. The action-rate penalty reduces the cost substantially (jerk 153.6 → 85.7,
p<1e-15) without removing it; whether it costs success is untested — the single checkpoint pair
behind that claim moved the point estimate 46% → 41%, inside the floor either way.

### 3.4 Time to insert — a real cost, small in absolute terms

**20 of 21 checkpoints are slower**, by +0.14 s to +0.27 s on a 7.83 s baseline. The direction
is established; the magnitude is a fraction of a second inside a distribution running from
1.4 s to nearly 18 s.

This KPI is only interpretable as a **survivorship-conditioned** number: it exists only on
trials that succeeded, so it compares the trials each arm happened to win. The same effect
runs the other way on the other KPIs — restricting to the walls both arms seated moves the
operator's own peak force from 23.97 N down to **14.77–15.55 N** and its jerk *up* from 45.60
to **57.69–64.90**, because the low-jerk trials are the ones that aborted early. (Both are
ranges, not constants: the seated subset depends on which walls the policy also seated, so
each recipe's comparison carries its own baseline.)

### 3.5 What a near-zero Δ actually conceals

Holding a checkpoint fixed and comparing wall by wall gives an answer no mean shows: **every
checkpoint flips the outcome on 20–37 walls of 100, in both directions at once.** F/T DAgger
flips 34, 23, 28, 23 and 24 walls while netting +2, +1, +2, +3 and +2 pp.

The policy is not inert. It is doing a great deal, and trading wins for losses at close to
even odds.

→ [Within a single seed](results/within-seed.md) · [The KPI board](results/kpi-board.md)

---

## 4. What was not possible, and why

The negative result is not one failure. It is a sequence of levers, each tried, each measured,
each with an identified mechanism — and the mechanisms are the transferable part.

**The learning target is not the problem.** A standardized linear ridge on the same F/T
observables the policy sees predicts the expert's gated correction on held-out data to
**2.36 mm**, against a zero-correction baseline of 4.91 mm (held-out R² 0.87 mean over the
three axes). A linear model beats the zero baseline by a factor of two. The target is
learnable from what the policy is given.

**The trained objective is what fails.** The same quantity, learned by the GRU, scores
**7.63 mm** — worse than predicting zero (4.75 mm on that evaluation). The error
decomposition locates it precisely: the expert is structurally **exactly zero** beyond
15 cm from the hole, which is **123797 of 209143 held-out steps (59%)**, and the network
emits a **5.64 mm** correction floor across them. Its near- and close-field errors do beat
the zero baseline (8.78 vs 9.42 mm; 12.02 vs 13.56 mm). The whole deficit is the free-space
floor: the objective averages a regression loss over a population dominated by steps whose
correct answer is "do nothing", and the network under-fits the gate rather than the
correction.

*(Both figures are reproduced in [`results/phase-1/probes/`](results/phase-1/probes/).)*

**Better imitation makes a worse controller.** Adding a `(command − ee_position)` feedback
feature drove offline error below the zero baseline for the first time (7.6 → 3.5 mm) and
closed-loop success **collapsed to 10%**, with more force-aborts. Online, the policy amplifies
its own tracking error into wall impacts. On this task, offline behavioral-cloning fidelity is
*anti*-correlated with closed-loop success — which means only a closed-loop ablation is a
valid signal.

**Vision adds little marginal signal.** The operator's command already proxies the hole
location, so the free-space correction a clone would learn from images is ≈0 by construction.
Unfreezing the image encoder ties F/T-only in-band and loses out-of-band.

**DAgger cannot rescue it, and the reason is structural.** The policy's rollouts are dominated
by force-abort states, and the bounded analytical expert **cannot demonstrate a recovery** from
a peg pinned at the force cap. Each round therefore aggregates more failure states labelled
with a passive correction, and the clone gets more passive. DAgger's founding premise — a
competent expert on the visited states — is violated.

**A better expert is not available either.** Five expert knobs meant to prevent the impact were
inert across ~30 settings; the expert's own ceiling stayed at ~73.3%. The binding constraint is
an operator-originated, *pre-contact* impact, which a bounded residual cannot fix.

→ [Mechanisms](results/mechanisms.md)

---

## 5. What the architecture guarantees

Three statements hold by construction, and a commonly-assumed fourth does not.

1. **The residual is clamped** to ±3 cm / ±10° / ±5 N per step, applied before the controller
   sees the augmented command. A maximally wrong network cannot enlarge its own authority.
2. **The commanded restoring force is bounded at 12.5 N.** The clamp applies to the Euclidean
   *norm* of the position delta (‖Δx‖ ≤ 0.025 m), so the bound is `λ_max·‖Δx‖` = 500 × 0.025.
3. **No trial continues past 30 N** — the evaluation observer aborts it. This bounds how long a
   breach lasts, not whether one happens.
4. **Measured contact force is not bounded.** The wrist sensor reads the contact *reaction*,
   including impact transients the quasi-static `K·Δx` argument does not cover, and the
   commanded wrench also carries a damping term no clamp bounds: 1712 of 4200 trials exceed
   30 N, reaching 77.86 N, and 58% of *successful* trials exceed 12.5 N.

The bound is on the assist's **authority**, which is the property a safety argument needs, and
it holds without reference to any measurement. It is not a bound on what the robot feels.

---

## 6. Verdict against the stated criteria

The design document's five original success criteria are quoted verbatim and marked in
[§8](design-document.md#8-evaluation-criteria). In summary: **two met, two not met, one in
progress.** The two not met are both success-rate criteria — one of them partly met on its
force clause, which holds as a bound on the assist's authority but not on measured contact.
The one in progress is the submission deliverables themselves.

The project does not demonstrate that a behavioral-cloned residual improves insertion success
for a human operator on this task, nor that it reduces contact force. It demonstrates, with a
stated measurement resolution, that it does neither — that it costs a slower and rougher
trajectory — and it identifies why.

---

## 7. What would move it

The mechanisms rule out the levers that were tried, and they also predict which remaining ones
could matter. The strongest is **contact-recovery control**: the binding constraint is a
pre-contact impact that ends the trial, and no per-step imitation of a bounded expert can
address it, because the expert cannot demonstrate the recovery. That is a control problem, not
a supervision problem.

Two others follow from §4 directly: an objective that models the far-field gate explicitly
rather than averaging over it, and a competence signal that lets the assist defer when it
cannot localize the target.

→ [Further exploration](results/further-exploration.md)

---

**See also:** [design document](design-document.md) · [results index](results/kpi-dashboard.md)
· [architecture tour](guides/architecture-tour.md) · [policy guide](guides/policy-guide.md)
