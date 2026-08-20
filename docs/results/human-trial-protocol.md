# Human-operator trial — protocol, fixed in advance

**Status: pre-registered, not yet run.** Everything below — the checkpoint, the trial
count, the analysis, the reporting rule — is fixed *before* any data exists. This file is
committed first and is not edited after the session starts; the results go in a separate
document that links back here.

That ordering is the whole point. The project's own post-mortem
([self-evaluation](../self-evaluation.md) §5) names *"pre-register the reporting rule, not
just the metric"* as a lesson learned the expensive way: the KPIs were defined up front but
how results would be *selected* was not, which is what let a single lucky checkpoint become
a headline. This is that lesson applied once, deliberately, on the record.

**Related:** [mechanisms](mechanisms.md) · [noise floor](noise-floor.md) ·
[per-KPI floors](phase-1/noise-floor-per-kpi.md) · [conclusions](../conclusions.md)

---

## 0. Amendment — 2026-08-12, before any recorded trial

`runs/blind_trial/` did not exist when this was written; no recorded trial had been run.
Amending *before* data is what pre-registration permits — the rule it exists to enforce is
that the design cannot be edited once results are visible. The superseded values are kept
below so the change is auditable rather than silent.

**What changed:** the checkpoint moves from the **F/T-DAgger** family to the
**Vision-DAgger** family; recorded trials 60 → **32**; practice 10 → **8**.

**Why.** The design's one live measurement other than the outcome is the blinding check —
whether the operator can tell the arms apart. On an F/T-only residual that question is
close to answered in advance, because the operator has no sensory channel to any of its
effects:

| Channel | F/T-DAgger's measured effect | Available to the operator? |
|---|---|---|
| Contact force | −1.04 N, against its own 2.41 N seed floor (null) | **No** — the rig has no haptic feedback; commands come from stereo hand tracking |
| Time to insert | +0.14 to +0.27 s on a 7.83 s baseline | **No** — 2–3%, inside a per-trial distribution running 1.4–18 s |
| Success rate | +2.0 pp | **No** — 0.6 trials out of 16 per arm |
| Trajectory smoothness | rougher on 20 of 21 checkpoints | Marginal, and the only one |

Spending a session confirming chance-level guessing against that table measures the
absence of a channel, not the absence of an effect. The Vision-DAgger residual acts on
free-space approach, which is visible in the viewer and is the modality the project is
actually about — so the guess has something to be a guess *about*.

**The deeper reason the original choice was wrong.** F/T-DAgger was selected for seed
stability (2 pp spread), which is the correct guard for an *effect estimate*. §2 states
that this design cannot produce one. The selection rule was therefore optimizing against a
bias that could not occur here, and it paid for that with the only property the session can
actually examine.

**What did not change — the anti-selection rule.** The superseded checkpoint was seed 0,
final DAgger round (F/T-DAgger has rounds 0–2, so `round_2` *was* its last). The
replacement applies that identical rule to the new family: lowest-numbered training seed,
final round → `vision/dagger/seed_0/round_4`. It is **not** the best-scoring vision
checkpoint. `vision/dagger/seed_1/round_4` is (62% success, +12 pp at p = 0.036) and is
excluded by the rule, as it must be — it is the exact outlier
[§5.5](../conclusions.md) documents.

**Two costs, disclosed rather than argued away:**

- **Seed stability is lost.** Vision-DAgger's success rate spans 16 pp across its three
  training seeds ([−4, +12]) against F/T-DAgger's 2 pp. This checkpoint is one draw from
  that spread, and no result here generalizes to the recipe. That matters less than it
  looks, because §2 already forbids reading this session as an effect estimate — but it
  is a real reduction in what a positive result could mean.
- **The rule lands on seed 0's best round.** Seed 0's paired Δ reads −1 → −28 → −12 → +8
  across rounds, so "final round" selects +8. The rule was fixed before that sequence was
  consulted and is inherited unchanged from the superseded checkpoint; it was not chosen
  to land there. Stated here so the reader can judge that for themselves.

**Trial count.** 32 recorded (8 full blocks of 4) and 8 practice, ≈ 20 minutes. §2's table
was never a case *for* 60 — at 30 per arm it resolves ~33 pp against single-digit effects.
Halving trials halves nothing that was load-bearing, and the session's real products
(human-in-the-loop evidence, proxy validity, unselected footage) do not scale with n.

## 0.1 Amendment — 2026-08-20, still before any recorded trial

`runs/blind_trial/` did not exist when this was written either. Same standing as §0: the
rule pre-registration enforces is that the design cannot be edited once results are
visible, and none are.

**What changed:** the viewer camera moves from `--cam main` to **`--cam wrist`** (the
robot's-eye POV). `blind_trial.py`'s default follows.

**Why.** The requirement this row encodes is that the viewpoint be **fixed**, not that it
be the free camera — §3 says a different viewpoint "would need its own session to compare
against", and this is the only session, so either choice satisfies it as long as it is
declared in advance and held for every trial. Given that, the operator's familiarity
decides it: the rehearsal and the `--gain` / `--max-dpos` tuning were both done under the
wrist view, and switching to the free camera would enter the session on a viewpoint
nobody has practised.

**What this does not trade away.** The footage is unaffected —
`scripts/dev/render_trajectory.py` rebuilds a third-person view offline from the recorded
trajectory, so what the operator watched live does not constrain what the demo shows.
Real-time performance is unaffected too: the rehearsal ran **1.00× with 45.5% of the loop
in `sleep`** under `--cam wrist`, i.e. with headroom to spare.

**The cost, disclosed.** §0 chose the Vision-DAgger family because its residual acts on
free-space approach, "which is visible in the viewer". The wrist camera shows that phase
from the robot's own vantage rather than from outside, which is a *different* view of the
same behaviour, not an absent one — but it is a narrower field of view, and it may make
the task harder. The practice block's floor check is what catches that, and it runs before
any recorded trial either way.

**Superseded value:** `--cam main` (free camera), as §3's table read until 2026-08-20.

## 0.2 Amendment — 2026-08-20, still before any recorded trial

`runs/blind_trial/` held a practice block and one incomplete trial when this was written;
`trials.csv` had **zero rows**. Nothing recorded exists, so the §0 standing applies again.

**What changed:** the operator's camera-preview window is **off** during trials
(`blind_trial.py` no longer passes it, and no longer requests `--record-hand`), and
practice trials go 8 → **4**.

**Why the preview goes.** It was the single largest cost in the loop and it was not
buying the measurement anything. Measured on the rehearsal and the practice block:

| | ms/step |
|---|---|
| `input` with the preview (30 Hz cv2 pump, compositing both feeds + the 3-D skeleton, flushing to screen, writing the `hand.mp4` frame) | **1.454** |
| `input` with no preview (same checkpoint, scripted operator) | 0.044 |

That is the difference between **0.55× and ~0.90× real time**. The practice block ran at
0.50–0.59×, i.e. the operator drove a half-speed sim — a different task from every other
measurement in this project. Nothing else came close: the viewer camera is 0.27 ms/step,
and the policy's own cost (`assist` ≈ 0.95, `observe` ≈ 0.77) is as designed, with the
CUDA graph confirmed capturing on all eight practice trials.

**What this costs, and how it is paid.** `hand.mp4` is written from the frame the preview
composites, so no preview means no operator-side footage from the trials. That footage is
a *demo* artifact, not a measurement: the property worth protecting is that the
**robot-side** takes are unselected, and they still are. The hand-side video is taken
afterwards in 2–3 separate free-play recordings. Any published cut must therefore never
imply a hand take and a robot take are the same trial.

**The real cost, disclosed.** The operator loses live visual confirmation that tracking is
alive, including during centering, and the eight practice trials already run were run
*with* the preview — so the acclimatisation they bought is to a slightly different task.
That is the main reason practice is not dropped further than 4.

**Why practice 8 → 4.** The block's two jobs are acclimatisation and the floor/ceiling
check. The operator has now run eight trials on this rig today, so acclimatisation is
largely banked; 4 is enough to re-establish feel under the changed conditions and to
re-run the floor/ceiling check, which **must** be re-run because the earlier 5/8 (62.5%)
was measured with the preview on and at half real-time.

**Superseded values:** preview window on with `--record-hand` per trial; practice 8.

## 1. The question

Every number in this project was measured against `ScriptedNoisyHuman` — a *model* of an
operator, open-loop by construction. This is the only place a real human closes the loop.

> Does the trained residual change closed-loop insertion outcomes for a human operator
> driving the arm through stereo hand tracking?

It is a **proxy-validity check**, not a re-run of the Phase-1 measurement. The scripted
study answered the effect question with 4200 trials; no human session can improve on that.
What it can do is test whether the proxy's *conclusion* survives contact with a real
operator, whose error distribution differs from the scripted one in ways the model does not
capture.

## 2. What this design can and cannot resolve

Stated first, because it decides how the result may be read.

Assuming a 50% baseline (the scripted `human_only` rate), two-sided α = 0.05, 80% power:

| Recorded trials | Per arm | Smallest difference detectable |
|---|---|---|
| **32** | **16** | **~45 pp** |
| 40 | 20 | ~40 pp |
| 60 (superseded, §0) | 30 | ~33 pp |
| 80 | 40 | ~30 pp |

The effects the scripted study measured are **single-digit pp**. This design cannot see
them. A null result here is therefore *uninformative about effects of that size*, and will
not be reported as evidence of absence — only an effect large enough to clear the table
above would be a finding, and none is expected.

The measurement is worth taking anyway for two reasons that do not depend on power: it is
the only human-in-the-loop evidence in the project, and it produces **unselected footage**
(§7).

## 3. Fixed parameters

| Parameter | Value | Why this one |
|---|---|---|
| Checkpoint | `docs/results/checkpoints/vision/dagger/seed_0/round_4/checkpoint.pt` | Lowest-numbered training seed, final DAgger round — a rule fixed without reference to any outcome, inherited unchanged from the superseded F/T checkpoint (§0). **Not** the best-scoring checkpoint: `vision/dagger/seed_1/round_4` is, and selecting on outcome and then measuring is the bias this project already documented. Vision rather than F/T because the F/T residual has no operator-perceptible channel, so the blinding check would have been answered in advance — see §0 for the full argument and its costs. |
| Recorded trials | 32 (16 per arm) | §2. Eight full blocks of 4 |
| Practice trials | 4, unrecorded, discarded | Declared here so discarding them is not a choice made after seeing them. Amended 2026-08-20 from 8 — protocol §0.2 |
| Wall seeds | Trial *i* uses wall seed *i*; practice uses 9000 + *i* | Fresh wall every trial, no overlap between practice and record |
| Assignment | Block-randomized, `BLOCK = 4`, seed `20260804` | Exactly half of every block of 4 is assisted |
| Viewer camera | `--cam wrist` (robot's-eye POV) | The operator's viewpoint changes the task's difficulty, so it is fixed for the whole session rather than varied per trial. Either viewpoint satisfies that; the operator rehearsed and tuned under this one. Amended 2026-08-20 from `--cam main` before any recorded trial — see [§0.1](#01-amendment--2026-08-20-still-before-any-recorded-trial) |
| Operator | Naveh Brenner, blinded | |

## 4. Design

**Unpaired, fresh wall per trial.** Re-running one wall with and without the assist is the
stronger paired design and it is unavailable here: a human remembers the wall, and the
memory contaminates the second attempt. Wall difficulty is randomized across arms instead,
which costs power and buys validity.

**Block randomization, not coin-flipping.** Within each block of 4 trials exactly 2 are
assisted. This guarantees arm balance and, more importantly, spreads both arms evenly
across the operator's fatigue and learning curve — a real confound over 32 trials, and one
that plain randomization only controls in expectation.

**Blinding.** Both arms run `--policy tf` with `--assist-scale 1` or `--assist-scale 0`.
Scale 0 still loads the checkpoint, still runs the forward pass, still captures wrist
frames; it discards only the Δ that reaches the command. Startup time and per-step cost are
identical, so the arm cannot be inferred from how the session behaves — which
`--policy noassist` would leak immediately. Each trial's console output is redirected to a
per-trial log, so no line naming the policy reaches the terminal.

**One cue survives the redirect.** The startup centering states are re-rendered on the
operator's terminal as a live spinner, so they know when to hold the open palm still and
when the arm is live. It is a whitelist of that one message family, and centering completes
*before the sim is stepped* — the assist has not been consulted yet, so the cue is identical
in both arms by construction. Everything else, including any traceback, stays in the trial's
`console.log`.

**Blinding is measured, not assumed.** The operator records a guess after every trial (on /
off / unsure). If the guesses land at chance, the write-up says the blind held; if they do
not, the write-up says that instead and the result is qualified accordingly. The residual
does perturb the arm, so partial unblinding is a real possibility rather than a formality.

## 5. Abort conditions

Checked against the 8 practice trials, before any recorded trial runs:

- **Floor** — near-0% practice success means the operator cannot seat the peg by hand at
  all, and the study measures the interface rather than the assist.
- **Ceiling** — near-100% leaves no room for an effect in either direction.

On either, the session is abandoned, the reason is published, and the footage is kept.

## 6. Analysis and reporting rule

Fixed here, applied exactly as written:

1. **Primary:** success rate per arm, with Wilson 95% intervals, and the difference with
   its interval. Reported as an estimate with uncertainty, alongside the detectable-effect
   figure from §2. **No p-value is computed** — at this n it would only invite a null to be
   read as evidence of absence.
2. **Secondary, descriptive only:** time-to-insert, peak contact force and trajectory jerk,
   per arm. These inherit the same power limits, and the two costs already established in
   the scripted study (slower, rougher — 20 of 21 checkpoints each) are the prior.
3. **Blinding check:** fraction of correct guesses against the 50% chance line.
4. **Every trial that runs is reported.** No trial is dropped for being unrepresentative,
   messy, or spoiled by a bad clutch. If a trial is unusable for a mechanical reason (a
   camera dropout, a crash), the reason is logged at the time and reported as a count.

**The result is published whichever way it comes out**, including "worse with the assist",
which the two established costs make a live possibility.

## 7. Footage

Every trial records its command trajectory and its operator-side video, so demo footage is
a by-product rather than a separate shoot. Because assignment is hidden and randomized,
clips are **unselected** — the honest version of a with/without comparison, which a
same-wall pair driven by a non-blinded operator could never be.

Any published clip is captioned as what it is: different walls, single runs, not a
comparison.

## 8. Running it

```bash
# free play first — tune --gain / --max-dpos before anything is recorded
uv run kvn episode --input vision --stereo-calib <calib.json> --cameras 0 1 --max-steps 0

uv run python scripts/dev/blind_trial.py --stereo-calib <calib.json> --cameras 0 1
uv run python scripts/dev/blind_trial.py --unseal --out runs/blind_trial
```

Windows-native: WSL2 has no UVC driver, so the cameras are not visible there.

Do not open `runs/blind_trial/assignments.json` until every trial is done.
