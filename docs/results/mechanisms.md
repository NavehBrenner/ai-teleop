# Mechanisms — why it does not work, and what stands

The negative results with their explanations, and the claims that survive them. The
success-rate null is a measurement; this file is the account of *why* per-step imitation
cannot lift closed-loop seating on this task, which is the part that transfers.

It also states precisely what the architecture does and does not guarantee about contact
force.

**Related:** [experiment ledger](experiment-ledger.md) · [noise floor](noise-floor.md) ·
[KPI board](kpi-board.md) · [within-seed](within-seed.md) ·
[further exploration](further-exploration.md) · [index](kpi-dashboard.md)

---

## 6. Negative results

Each is a mechanism, not just a missing win.

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
  success on this task.
- **DAgger degrades, it doesn't rescue — REGRESSION.** Three F/T rounds on the ar100 base:
  **40% → 30% → 15%** (rollout success 0.325 → 0.25). Mechanism: the policy's rollouts are
  dominated by force-abort states, and the **bounded analytical expert cannot demonstrate a
  recovery** from a peg pinned at the force cap — so each round aggregates more failure states
  labeled with passive Δ, and the clone gets more passive. DAgger's founding premise (a
  competent expert on visited states) is structurally violated.
- **Stage-C vision fine-tune — NULL.** Unfreezing the image encoder (`vision_stageC`) ties
  F/T-only in-band (40% vs 40%, es0.4) and loses out-of-band (10% vs 20%, es1.0, inside the
  floor). Vision carries little marginal signal because **the operator command already proxies
  the hole location** (the identifiability finding); the free-space correction the clone would learn
  is ≈0 by construction.
- **A better analytical expert — REFUTED.** Five expert knobs meant to prevent the
  slam were all inert; the expert's own ceiling stayed at ~73.3%. The binding constraint is
  operator-originated, pre-contact force-abort, which a bounded residual cannot fix.

---

## 7. What still stands

One class of result does **not** rest on a sampled measurement — the architectural bound. The
rest of this section applies the noise-floor test to every KPI, not just the success rate,
because a claim about a continuous metric is subject to exactly the same standard.

- **The bound on the assist's authority, stated precisely.** Three things are true by
  construction, and a commonly-assumed fourth is **not**:

  1. **The residual is clamped** — ±3 cm / ±10° / ±5 N per step, applied *before* the controller
     sees the augmented command (`domain/delta.py`). A maximally wrong network cannot enlarge its
     own authority.
  2. **The commanded restoring force is bounded at 12.5 N.** The backbone clamps the **Euclidean
     norm** of the position delta to 0.025 m (`control/backbone.py`), so with translational
     stiffness `[400, 400, 500]` N/m the most the controller can ever *ask* for is
     `λ_max·‖Δx‖ = 500 × 0.025 = 12.5 N`. (Taking each axis at the full clamp independently gives
     18.9 N, but that describes a `Δx` of norm 0.043 m, which the clamp makes unreachable — a
     valid bound, 51% loose.) The bound holds in `ACTIVE` and `HOLD`; `PARK` returns the home
     pose directly and bypasses the clamp.
  3. **No trial continues past 30 N** — the eval observer aborts it (`eval/observer.py`).
  4. **Measured contact force is *not* bounded.** The wrist F/T sensor reads the contact
     *reaction*, which includes impact transients the quasi-static `K·Δx` argument says nothing
     about, and the commanded wrench also carries a damping term `−D·ẋ` that no command clamp
     bounds. **1712 of 4200 official trials exceed 30 N, reaching 77.86 N** — every one of them a
     `force_abort`, the overshoot occurring within the tick before the watchdog fires. 58% of
     *successful* trials exceed the 12.5 N commanded bound. See [within-seed.md](within-seed.md).

- **The measured effects, against each metric's own floor.** Retraining one recipe with a
  different training seed moves *every* KPI, not just success. The floor per recipe per metric is
  in [`phase-1/noise-floor-per-kpi.md`](phase-1/noise-floor-per-kpi.md). Counting the sign of all
  21 checkpoints' paired Δ:

  | KPI | checkpoints above baseline | reading |
  |---|---|---|
  | Success rate | **9 / 21** | no effect — a coin flip |
  | Peak contact force | **11 / 21** | **no effect — a coin flip** |
  | Time to insert | **20 / 21** | a real cost (slower) |
  | Trajectory jerk | **20 / 21** | a real cost (rougher) |

  **The force reduction under DAgger is not established.** FT DAgger's −1.04 N mean sits inside
  its own family's 2.41 N seed spread and well inside plain BC's (5.10 N at batch 16, 7.59 N at
  batch 2); no per-seed Wilcoxon clears p=0.05. The two DAgger arms disagree at seed level — FT
  DAgger is negative on 4 of 5 seeds, Vision DAgger *positive* on 2 of 3. Mean force and
  force-abort rate are not independent corroboration either: both are computed from the same
  trials, and the abort rate is a threshold count of the same quantity.

  What *is* established is the direction of the two costs. Twenty of twenty-one checkpoints are
  slower and rougher than the operator alone — a near-unanimous sign across every recipe,
  modality and batch size, which is a stronger form of evidence than any mean clearing a floor.
  Their magnitudes remain seed-dependent (jerk's floor reaches 186 on FT DAgger).
- **The mechanism findings**, each theory or a byte-identical/exact probe:
  - **Identifiability ceiling** — the operator command proxies the hole; a no-vision
    residual cannot lift success outside the chamfer band. A structurally-flat flat-wall delta
    is a *result*, not a failure.
  - **Far-field gating failure** — trained GRUs emit a **5.64 mm** correction floor
    across the 123797 of 209143 held-out steps (59%) beyond `d_far` where the expert is
    exactly zero. Near and close in, the same GRU beats the zero baseline (8.78 vs 9.42 mm;
    12.02 vs 13.56 mm) — the entire offline deficit is the free-space floor
    ([probe output](phase-1/probes/error-decomposition.md)).
  - **Offline/closed-loop anti-correlation** — fixing offline BC error made
    closed-loop worse; only a closed-loop ablation is a valid signal here.
  - **The bounded-expert/DAgger argument** — on-policy relabeling can only teach
    what the expert can perform, and it cannot un-jam a force-aborted peg.

The engineering summary: **Phase 1 delivers an assist whose authority is bounded by
construction, and a mechanized account of why per-step imitation cannot lift closed-loop
seating on this task. On the seeded measurement it establishes no benefit — neither a
success-rate lift nor a contact-force reduction — and two costs, a slower and rougher
trajectory, whose direction is near-unanimous across all 21 checkpoints.**

---

