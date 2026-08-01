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
     *successful* trials also exceed 18.9 N. See [within-seed.md](within-seed.md).

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

