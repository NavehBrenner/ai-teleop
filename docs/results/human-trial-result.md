# Blinded human-operator trial — result

Session run 2026-08-20. Companion to [`human-trial-protocol.md`](human-trial-protocol.md),
which fixed the checkpoint, the trial count, the analysis and the reporting rule **before**
any data existed. This document applies §6 exactly as written and adds nothing to it.

Raw data: [`human-trial/trials.csv`](human-trial/trials.csv) (the session record),
[`human-trial/assignments.json`](human-trial/assignments.json) (the unsealed schedule),
[`human-trial/kpis.csv`](human-trial/kpis.csv) (per-trial KPIs). Regenerate the analysis with
`uv run python scripts/dev/trial_kpis.py --out <session>`.

---

## 1. What this session is and is not

Every other KPI in this project was measured against `ScriptedNoisyHuman` — a *model* of an
operator. This is the only measurement in which a real person closes the loop.

It is **not** an effect estimate, and §2 said so in advance: at 16 trials per arm the
smallest difference this design can resolve is **~45 pp**, while the effects the scripted
study measured are single-digit pp. It is a proxy-validity check and a source of unselected
footage. A null here is not evidence of absence, and a positive difference smaller than
~45 pp is not evidence of a benefit.

**32 recorded trials, 0 excluded.** No mechanical failures, no crashes, no camera dropouts,
so §6.4's exclusion count is zero.

## 2. Primary — success rate (§6.1)

| Arm | successes | rate | Wilson 95% |
|---|---|---|---|
| assist **ON** | 13/16 | **81.2%** | [57.0%, 93.4%] |
| assist **OFF** | 11/16 | **68.8%** | [44.4%, 85.8%] |

**Difference: +12.5 pp, Newcombe 95% [−17.2, +39.7] pp.** No p-value, per §6.1 — at this n
it would only invite a null to be read as evidence of absence.

**The interval spans zero.** The observed difference is consistent with no effect and
equally consistent with a substantial benefit; it is well inside the ~45 pp this design
cannot resolve. Nothing here establishes that the residual helps a human operator, and
nothing here establishes that it does not.

The direction does **not** match the scripted study: on this same checkpoint
(`vision/dagger/seed_0/round_4`) the scripted paired Δ is **−4.0 pp** (p=0.572,
[official KPI tables](phase-1/official_kpi_tables.md) §6). At n=16 per arm neither number
resolves anything, so the disagreement is no more meaningful than agreement would have been —
but it must not be reported as corroboration.

### The headline depends on one contested trial

**Trial 31** (assist ON) is recorded as `success` by the harness, but the canonical
`TrialObserver` scores it `force_abort` — its peak contact force reached **31.59 N** against
the 30 N cap. It is the only trial where the two disagree in a way that changes an outcome
rather than a label (see §5).

| | ON | OFF | difference |
|---|---|---|---|
| As recorded | 13/16 = 81.2% | 11/16 = 68.8% | **+12.5 pp** |
| If trial 31 is scored `force_abort` | 12/16 = 75.0% | 11/16 = 68.8% | **+6.2 pp** |

A single trial moving the headline by half is what n=16 per arm looks like. Reported here
rather than resolved, because the protocol fixed the scoring rule in advance and the
harness's `terminal_reason` is that rule.

## 3. Secondary — descriptive only (§6.2)

These inherit the same power limits. The prior from the scripted study is the two
established costs: **slower and rougher**, on 20 of 21 checkpoints each.

| KPI | assist ON | assist OFF |
|---|---|---|
| Time to insert (s), seated trials only | 5.07 (median 4.76, n=13) | 3.60 (median 3.62, n=11) |
| Episode duration (s), all trials | 4.78 (median 4.71, n=16) | 4.31 (median 3.83, n=16) |
| Peak contact force (N) | 21.40 (median 22.18, n=16) | 22.68 (median 21.08, n=16) |
| Trajectory jerk (∫\|jerk\| dt) | 68.35 (median 64.27, n=16) | 54.78 (median 46.39, n=16) |
| Trajectory jerk **per second** | 14.19 (median 14.12) | 12.96 (median 12.90) |
| Closest tip–hole distance (mm) | 7.35 (median 5.97, n=16) | 8.57 (median 6.47, n=16) |
| Lateral error at closest (mm) | 6.30 (median 5.41, n=16) | 7.92 (median 5.96, n=16) |

**Both established costs are directionally present.** Insertion took ~1.5 s longer with the
assist on (5.07 vs 3.60 s), and the trajectory was rougher. Jerk is an *integral*, so the
raw +25% partly reflects the longer episodes; normalised per second it is +9.5%, which is
the honest version of the roughness figure and the one to quote.

No interval is given for these because §6.2 declares them descriptive. At n=16 per arm they
are consistent with the scripted prior rather than independent confirmation of it.

## 4. Blinding check (§6.3) — the blind held

**11/20 guesses correct = 55%, Wilson 95% [34%, 74%]**, against a 50% chance line. The
interval spans chance.

Guesses split **9 on / 11 off / 12 unsure**. The operator declined to guess on 12 of 32
trials, which is the stronger signal: it is what an intact blind looks like from the inside.

This is what makes the proxy-validity reading meaningful. `--assist-scale 0` keeps the
checkpoint load, the forward pass and the wrist capture and discards only the Δ, so the arms
were indistinguishable in startup and per-step cost — confirmed in the session data, where
real-time factor ran 0.81/0.81 on the off arm and 0.86/0.80 on the on arm during practice.

## 5. Instrument note — the observer cannot score these episodes

`TrialObserver` requires seating to hold for **0.05 s** before it records SUCCESS.
`run_episode` terminates the *instant* the seating criterion is met — `step_success` fires
exactly once, on the final step — so the sustained window can never elapse inside a recorded
episode.

| harness `terminal_reason` | observer verdict | trials |
|---|---|---|
| `success` | `timeout` | 23 |
| `success` | `force_abort` | 1 |
| `force_abort` | `force_abort` | 8 |

Replaying these episodes through the ablation harness's calculator asks it a question the
recording cannot answer. The outcome of record is therefore the harness's `terminal_reason`,
which is what every interactive episode in this project uses. Force, jerk and the near-miss
trio are unaffected — they accumulate over the observation stream and do not depend on the
verdict.

**Cross-study caveat.** The scripted results were scored by the observer's *sustained*
criterion; this session's rates come from `run_episode`'s *instantaneous* one, which is
strictly more permissive. **These rates are not directly comparable to the scripted study's
50% `human_only` baseline.** Within this session both arms are scored identically, so the
comparison that matters here is unaffected.

## 6. Force

Nine of 32 trials recorded a peak contact force above the 30 N cap, up to **36.51 N**. That
is consistent with the documented position and worth restating precisely: exceeding 30 N is
what *aborts* a trial, but the crossing value itself overshoots, so the **measured** force is
not bounded by the cap. Any claim that contact force stays under a figure is false — the
project's own retraction on this point stands, and this session reproduces it.

What remains true and is unaffected: the residual is clamped by construction (±3 cm / ±10° /
±5 N per step), and the assist can move the *commanded* restoring force by no more than 15 N.
(This session ran the vision default `--max-dpos 0.3`, under which the *total* commanded
restoring force is bounded at 150 N, not the 12.5 N an earlier revision published.)

## 7. Session log

| | |
|---|---|
| Date | 2026-08-20, Windows-native |
| Checkpoint | `docs/results/checkpoints/vision/dagger/seed_0/round_4/checkpoint.pt` |
| `--gain` / `--max-dpos` | defaults (1.0 / 0.3) — settled in free play, unchanged for every trial |
| Viewer | `--cam wrist`, fixed for the session (protocol §0.1) |
| Camera preview | off (protocol §0.2) |
| Cameras | `1 2` |
| Schedule seed | 20260804, block-randomized, BLOCK=4 |
| Real-time factor | 0.80–0.86× |
| Sensor health | 0% drop-out, 33–38 fresh fps, all trials |
| Trials excluded | 0 |

**Practice block (4 trials, unrecorded, discarded):** 3 success / 1 force_abort = 75%.
Neither abort condition triggered — not near-0 (floor), not near-100 (ceiling). At n=4 this
check is weak: 3/4 has a 95% CI of roughly [19%, 99%]. It rules out the gross failure modes,
which is what §5 asks of it, and nothing more.

Worth recording against §2's power table, which assumed a ~50% baseline: the operator ran
nearer 70–75% unassisted, which compresses the room above for any benefit to appear. This
does not change the conclusion — the design already could not resolve under ~45 pp — but it
means the real detectable effect was, if anything, larger than the table states.

### Amendments, both before any recorded trial

- **§0 (2026-08-12)** — checkpoint F/T-DAgger → Vision-DAgger; trials 60 → 32; practice 10 → 8.
- **§0.1 (2026-08-20)** — viewer camera `--cam main` → `--cam wrist`.
- **§0.2 (2026-08-20)** — camera preview off during trials; practice 8 → 4.

§0.2 was made after a first practice block ran at **0.50–0.59× real time**: the 30 Hz cv2
preview pump cost 1.454 ms/step against 0.044 without it, and removing it took the session to
0.80–0.86×. That first practice block is superseded and its data is not part of this result.

## 8. What this session delivered

- **A real human closed the loop** — the only such measurement in the project.
- **A measured, intact blind**, so the proxy-validity reading means something.
- **32 unselected robot-side takes** for demo footage, since assignment was hidden and
  randomized throughout.

What it did not deliver, by design, is an effect estimate.
