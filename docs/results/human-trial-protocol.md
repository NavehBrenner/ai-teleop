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
| 40 | 20 | ~40 pp |
| **60** | **30** | **~33 pp** |
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
| Checkpoint | `docs/results/checkpoints/ft/dagger/seed_0/round_2/checkpoint.pt` | F/T-DAgger is the family whose success rate moves least across training seeds — 2 pp, against 27–31 pp for plain BC — so it depends least on the seed lottery. That reason stands without looking at any human data. **Not** the best-scoring checkpoint: selecting on outcome and then measuring is the bias this project already documented. |
| Recorded trials | 60 (30 per arm) | §2 |
| Practice trials | 10, unrecorded, discarded | Declared here so discarding them is not a choice made after seeing them |
| Wall seeds | Trial *i* uses wall seed *i*; practice uses 9000 + *i* | Fresh wall every trial, no overlap between practice and record |
| Assignment | Block-randomized, `BLOCK = 4`, seed `20260804` | Exactly half of every block of 4 is assisted |
| Viewer camera | `--cam main` (free camera) | The operator's viewpoint changes the task's difficulty, so it is fixed for the whole session rather than varied per trial. `--cam wrist` (robot's-eye POV) is a different task and would need its own session to compare against |
| Operator | Naveh Brenner, blinded | |

## 4. Design

**Unpaired, fresh wall per trial.** Re-running one wall with and without the assist is the
stronger paired design and it is unavailable here: a human remembers the wall, and the
memory contaminates the second attempt. Wall difficulty is randomized across arms instead,
which costs power and buys validity.

**Block randomization, not coin-flipping.** Within each block of 4 trials exactly 2 are
assisted. This guarantees arm balance and, more importantly, spreads both arms evenly
across the operator's fatigue and learning curve — a real confound over 60 trials, and one
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

Checked against the 10 practice trials, before any recorded trial runs:

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
