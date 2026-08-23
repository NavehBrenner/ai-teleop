# Further Exploration — what was tried, what it measured, and what might actually work

The imitation-learning arc (M5 → M7) closed as a **documented negative**: on the seeded
multi-seed measurement, no recipe lifts closed-loop insertion success above the human-only
baseline beyond training-seed noise ([the official multi-seed run](noise-floor.md#55-the-official-multi-seed-run--the-definitive-measurement)).

That is a result, not an abandonment — and it is only useful if it says *where the wall is* and
*what is on the other side of it*. This document does both: the levers that were tried and
measured inert, and the ones the mechanism findings predict could still move the number.

## What was tried, and what it measured

Each of these was a real experiment with committed artifacts, not a plan that was skipped.

| Lever | What it was | Outcome |
|---|---|---|
| **Plain behavioral cloning (F/T)** | GRU over command + force/torque history, cloning the analytical expert's per-step Δ | **Inert.** 5 training seeds, mean Δ −4.4 pp, spread 27 pp — that spread *is* the noise floor |
| **Vision conditioning** | Wrist-camera CNN early-fused into the same GRU; the M7 premise | **Inert.** Mean Δ −8.3 pp over 3 seeds. Vision never beat F/T-only at any operating point |
| **DAgger (on-policy relabel)** | Let the policy act, query the expert at the visited states, aggregate, retrain — the textbook fix for BC covariate shift | **Inert on success.** F/T +2.0 pp, vision +1.3 pp, both inside the floor. It *does* collapse the training-seed spread (27 pp → 2 pp), confirmed against a batch-size control — see below |
| **Action-rate penalty** | Squared first-difference of the predicted Δ in the BC loss, to remove the residual's jerk cost | **Worked — on the wrong axis.** Jerk falls substantially (153.6 → 85.7 on the matched ar0/ar100 pair) without reaching human level: every recipe still reads a positive paired Δ, and 20 of 21 checkpoints are rougher than the operator. Smoothness is solved; success is untouched |
| **A better analytical expert** | Five knobs swept on the expert that generates the training labels — the ceiling the clone imitates toward | **Refuted.** All five inert; the expert ceiling sits at ~73%, and the binding constraint proved to be operator-side pre-contact force-abort, not expert quality |
| **Scaling to 100 paired eval seeds** | More statistical power on the headline comparison | **Done — it *is* the measurement.** More eval seeds tighten the interval around a null; they do not move it |
| **Frozen vs fine-tuned image encoder** | Stage-C unfreeze of the CNN, to test whether frozen features were the bottleneck | **Inert.** On the 6167 near-hole frames (d < 0.15 m) a linear probe decodes depth from the frozen features (z R² = 0.88) but not lateral offset (x R² = −0.36, y R² = +0.06); unfreezing did not fix closed-loop success ([probe output](phase-1/probes/perception-probe.md)) |

Every lever inside imitation learning was measured. [Negative results](mechanisms.md#6-negative-results)
and [what stands](mechanisms.md#7-what-still-stands) explain *why* per-step imitation cannot lift
closed-loop seating on this task — an
identifiability ceiling, a far-field gating floor, and an anti-correlation between offline BC
fidelity and closed-loop success.

## The one thing DAgger does buy

DAgger does not lift success, but it **collapses the seed spread** — F/T goes from a 27 pp range
across training seeds to 2 pp, with all five seeds non-negative. When first observed this was
confounded (plain trained at batch 16, DAgger at batch 2), so it was tested: retraining F/T plain
at batch 2 with every other knob held gives a **31 pp** spread — slightly *wider* than at batch 16,
with a near-identical center. Batch size moves neither. The tightening is DAgger's.

That is worth knowing for any follow-on work. It means on-policy relabelling makes this recipe
*reliable* without making it *better*: the outcome stops depending on the training seed, and
settles on a value inside the noise floor. A method that removes variance without moving the mean
is telling you the mean is the wall.

## What might actually work

These are outside imitation learning by construction, because imitation is exhausted. Ordered by
how strongly the measured mechanisms predict they would help.

### 1. Contact-recovery control — the strongest candidate

**What it is.** A state machine that detects a jam (force rising without progress), retracts,
re-aligns, and retries — rather than a policy trying to clone its way around the jam.

**Why the mechanisms predict it works.** The binding constraint identified by the expert sweep is
an **operator-side pre-contact force-abort**: the episode dies because the peg loads against the
wall before it is aligned. Nothing in a per-step residual can undo a jam that has already
happened — a residual can only nudge the *next* command, never back out and start over. Contact
recovery attacks that failure directly instead of cloning around it.

**Honest cost.** A new subsystem and a new failure taxonomy, not a training run. Its
lateral-authority half was already measured inert; the full retract-and-retry loop is untested.
This is a new arc, not a follow-up experiment.

### 2. Reinforcement learning against a contact-aware reward

**Why it might work.** Every method tried here optimizes a *per-step imitation* objective, and
this project's central finding is that per-step fidelity is anti-correlated with closed-loop
success. RL optimizes the closed-loop objective directly — precisely the mismatch imitation
cannot escape.

**Honest cost.** Weeks, not hours: a reward function, a sim training loop, and tuning. It was
scoped out of this project at the start for exactly that reason. High variance, genuine upside.

### 3. Making the scripted operator contact-aware

**Why it is listed, and why it is not recommended.** An operator that backs off before jamming
would raise everyone's success rate. But the residual's entire job is to fix a *given* operator —
improving the operator moves the goalposts rather than clearing them, and every baseline in this
project would need re-measuring. It changes the benchmark, not the result.

## What does not need revisiting

- **More seeds, or a bigger corpus, of the same recipes.** Measures the null more precisely;
  does not change it.
- **The action-rate penalty.** Applied, works, orthogonal to success.
- **A better expert.** Refuted across five swept knobs, with the real binding constraint located.

---

**See also:** [`kpi-dashboard.md`](kpi-dashboard.md) — every experiment with its config and
measured result; [negative results](mechanisms.md#6-negative-results) for the mechanism findings, [what stands](mechanisms.md#7-what-still-stands) for what still stands.
