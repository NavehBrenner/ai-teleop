# Go-Forward Decision — the policy-training arc (D-6, LAB-112)

The go/no-go on the M5→M7 imitation arc, decided on the numbers in the
[KPI dashboard](../results/kpi-dashboard.md), not on intent. This is the last decision of
LAB-42: what is the project's honest claim, and is any further compute worth spending against
it?

## The premise changed — read this before the candidates

The Phase-4 plan (`PROJECT-REVIEW.md`) was written to answer *"which cheap win strengthens the
Phase-1 positive?"*. **There is no longer a success-rate positive to strengthen.** Two findings
landed after that plan:

1. **LAB-114** — training was unseeded; the +33.3 pp headline is one lucky draw of an 18 pp
   noise band, its checkpoint and corpus gone (dashboard §5).
2. **The official multi-seed run (LAB-112)** — a fresh ~1000-episode corpus, all four production
   recipes, each retrained over seeds, 100 paired held-out eval seeds (dashboard §5.5):

   | Recipe | seeds | mean Δ | range | verdict |
   |---|---|---|---|---|
   | F/T plain BC | 5 | −4.4 pp | [−19, +8] | NOISE |
   | F/T + DAgger | 5 | +2.0 pp | [+1, +3] | NOISE (tightest) |
   | Vision plain BC | 3 | −8.3 pp | [−16, +4] | NOISE |
   | Vision + DAgger | 3 | +1.3 pp | [−4, +12] | NOISE |

   **No recipe lifts closed-loop seating above the human baseline beyond training-seed noise**,
   on data that did not exist when LAB-114 ran. The null is a property of the task.

So the question is not "which win to bank" but **"close the arc, or open a new one?"** — and a
new one is by definition out of imitation-learning, because every imitation lever has now been
measured.

## Candidates, costed against the current evidence

Budget frame: **~4 active human-hours + 2–4 unattended overnight compute slots** remain before
the D2 review; solo project, final deadline 2026-08-31. "P(lift)" = probability the candidate
moves closed-loop success above the noise floor, given what §5.5/§6 already measured.

| Candidate | Status on current evidence | Eng. effort | Full cost | P(lift) | Verdict |
|---|---|---|---|---|---|
| **Scale Phase-1 to 100 seeds (GPU)** | **DONE — it *is* the −4.4 pp null** (§5.5 FT plain). Not a pending win. | 0 | spent | — | **retired: already the measurement** |
| **Action-rate penalty on the headline run** | **DONE — applied; jerk 45.6→48.0 (Vision+DAgger), no success cost** (§5.5, §6). Solves smoothness, not success. | 0 | spent | ~0 | **retired: solved, orthogonal to success** |
| **Better analytical expert** | **REFUTED — LAB-108: five knobs inert, expert ceiling ~73%; binding constraint is operator-side pre-contact force-abort.** | — | — | ~0 | **retired: refuted** |
| **More seeds / bigger corpus of the same recipes** | Tightens the CI around a null already tight enough to call. | ~0 human, 2–4 slots | compute | ~0 | **no: measures the null more precisely, doesn't change it** |
| **Contact-recovery state machine** (detect jam → retract → re-align) | Out of imitation. Its lateral-authority half was measured **inert** pre-LAB-108; the full retract-and-retry loop is **untested**. A new controller + new failure taxonomy. | ≫ 4 hrs | new subsystem, new eval | low–med, unquantified | **defer: real lever, but a new arc, not a close-out task** |
| **Operator-side fix** (make the scripted human contact-aware) | Untested. Changes the **task**, not the policy — the residual's job is to fix a *given* operator; an easier operator moves the goalposts. | ≫ 4 hrs | re-baseline everything | n/a | **out of scope: redefines the benchmark** |
| **RL / reward fine-tune** | Explicitly scoped out 2026-07-10. Needs a reward, a sim training loop, tuning — a research arc, not a task. Genuinely out-of-imitation, plausible P(lift), but weeks not hours. | weeks | new arc | med, high variance | **out of scope for LAB-42; candidate for a *future* project** |
| **Close the arc as an honest negative** | Zero cost. §6/§7 already give a mechanism-level account (identifiability ceiling, far-field gating floor, offline/closed-loop anti-correlation, bounded-expert/DAgger); §7 the standing force-bound guarantee. | 0 | 0 | n/a | **the recommendation ↓** |

## Recommendation

**Close the M5→M7 policy-training arc as a rigorous, documented negative.** The evidence for
closing rather than spending more compute:

- **Every imitation lever is exhausted and measured**, not merely untried: plain BC and DAgger,
  F/T and vision, the action-rate penalty, the better expert, the 100-seed scale-up. The two
  candidates the old plan called "cheap wins" are *done* and both landed on the null.
- **The remaining P(lift) candidates are all out-of-imitation new arcs** (contact-recovery
  controller, RL, operator redesign) — each ≫ the 4-hour budget, and RL was already scoped out
  for exactly this deadline. Starting one now trades a finished, honest result for a fourth
  likely-failed experiment under time pressure.
- **The project's deliverables do not depend on a success lift.** The standing positives —
  the **bounded-force guarantee** (proven by construction, never exceeded across any official
  trial) and the **mechanism findings** — are what the arc actually contributes, and LAB-114
  leaves them fully intact. A mechanized "*why per-step imitation cannot lift closed-loop seating
  on this task*" is a stronger scientific result than an unreproducible +33 pp.

**Spend the remaining compute on nothing further for this arc.** If any overnight slots are used
at all, the only defensible use is *more eval seeds on the existing checkpoints* to tighten the
null's CI for the writeup — and even that changes no decision.

The one genuinely promising out-of-imitation lever, **contact-recovery control**, is worth
recording as **future work** (not LAB-42 work): the mechanism findings predict *why* it might
help where imitation can't — it addresses the force-abort jam directly rather than cloning around
it — but it is a new subsystem and a new arc.

## Decision

**Verdict (Naveh, 2026-07-29): the M5→M7 policy-training arc is CLOSED as a documented negative.**
No further compute on it. The project's standing deliverables are the bounded-force guarantee and
the mechanism findings (dashboard §7); the success-rate lift is, on the seeded measurement, not
established and will not be pursued further within LAB-42. Contact-recovery control is recorded as
**future work**, outside this arc.

**Follow-on issue state (D-7), enacted in Linear:**

- **LAB-42** — rewritten as the durable Phases 0–4 checklist; arc marked complete with the §5.5
  null pasted in.
- **Close** (Done): LAB-79 (M7 spec, already auto-closed), LAB-101 (100-seed scale-up — done, it
  is the null), LAB-106 (offline/closed-loop anti-correlation — mechanism captured), LAB-108
  (better-expert — refuted, captured).
- **Close/re-scope:** LAB-99 and LAB-107 per the close-out (findings folded into the dashboard).
- **Open:** `future work: contact-recovery control` — carries the mechanism rationale for why it
  might succeed where imitation cannot.
