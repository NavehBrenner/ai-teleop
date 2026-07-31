# Within a Single Seed — the policy against the operator, trial by trial

**The within-seed question, and it is a different one.** Every other results file plots a
distribution over *training seeds*, where one point is a seed's mean across 100 trials —
answering *how much does retraining move the average?* This file holds the checkpoint
fixed and looks at the individual trials, answering *how does this one policy compare to
the human operator, wall by wall?*

The two answers differ, and the most visible way is that a mean can describe no trial that
ran: peak contact force is bimodal, so its average falls in the trough between the two
modes.

**Related:** [experiment ledger](experiment-ledger.md) · [noise floor](noise-floor.md) ·
[KPI board](kpi-board.md) · [mechanisms](mechanisms.md) · [index](kpi-dashboard.md)

---

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
load. Both statements are true and they are about different quantities — [what stands](mechanisms.md#7-what-still-stands) now separates them.

**The policies' worst impacts are harder than the human's** (max 64–78 N against 54.70 N) while
their *rate* of hard impacts is lower for both DAgger arms. Fewer bad contacts, but a heavier
tail when one happens.

Regenerate with `uv run python scripts/dev/official_kpi/plot_trial_forces.py`.

