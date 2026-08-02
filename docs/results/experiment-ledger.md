# Experiment Ledger — what was run, on what data

Provenance for every number in the results: which corpus trained which policy, which
difficulty each evaluation ran at, and what each run measured. Nothing here is a
conclusion — it is the key that makes the other files interpretable, because a
closed-loop number means nothing without its operating point.

**Related:** [noise floor](noise-floor.md) · [KPI board](kpi-board.md) ·
[within-seed](within-seed.md) · [mechanisms](mechanisms.md) · [index](kpi-dashboard.md)

---

## 2. Operating-point ledger

Every closed-loop number is only interpretable against *which corpus trained the policy* and
*which difficulty the eval ran at*. These two tables are the key; the ledgers in §3 and §4 point back to them.

### 2.1 Corpus lineage (`data/dataset_*/metadata.json`)

| Corpus | Fingerprint | Created | n_ep | Schema | `expert_success_rate` | Role |
|---|---|---|---|---|---|---|
| `dataset_1` | `290f1750` | 2026-06-16 | 200 | 1.0 | — | M5 first BC (`lab34_baseline`). Old geometry/schema — **not comparable** to later corpora. |
| `dataset_9` | `54dccad9` | 2026-07-06 | 200 | 2.0 | 71.5% | Trained the 2026-07-07 M6 run. **Overwritten in place** — see caveat below. |
| `dataset_10` | `54dccad9` | 2026-07-22 | 200 | 2.0 | 71.5% | Regeneration of `dataset_9`'s config; trains every LAB-101/114 run. |
| `dataset_vision` | `de0eeb3b` | 2026-07-07 | 300 | 2.0 | 72.3% | All M6/M7 F/T + vision ablations (`ftonly_*`, `vision_*`). |
| `dagger_ft_agg` | `de0eeb3b` base | 2026-07-10 | 340→420 | 2.0 | — | Aggregated on-policy corpus, grows per DAgger round. |

**Two caveats the fingerprints hide:**

- **`dataset_9` and `dataset_10` share the fingerprint `54dccad9` but are not the same data**
  (finding G-4 / H-B). The fingerprint hashes *config*, not *code*; regenerating `dataset_9`'s
  config under 2026-07-22 code changed 35 of 200 trajectories (34 by a median of 1 step, one
  flipped baseline outcome, corpus baseline 22.5% → 23.0%). The original 2026-07-06 episode
  files were then **overwritten in place** by that regeneration — proven byte-for-byte by
  `scripts/dev/lab114_corpus_identity.py` — so `data/dataset_9/` now holds `dataset_10`'s
  arrays under `dataset_9`'s stale manifest. **The corpus that trained the 2026-07-07 M6 run no longer
  exists on disk.**
- **A "code era" column is load-bearing.** `dataset_0`/`dataset_1` (schema 1.0) already drift —
  their manifests predate `generated_walls` entering the fingerprint payload (finding C-1a).
  Do not trust byte-identical regeneration of any pre-LAB-91 corpus.

### 2.2 Eval operating points (`runs/eval*/trials.csv`)

| Eval set | `error_scale` | Seeds | Regime | `human_only` | Notes |
|---|---|---|---|---|---|
| `runs/eval/` (LAB-53) | 0.4 | 100 | in-band | **31.0%** | Older step-budget era (pre-LAB-100); zero force-aborts. |
| `eval_ftgate_es0p4` | 0.4 | 20 | in-band | 35.0% | M7 F/T gate ablation (3 arms). |
| `eval_ftgate_es1p0` | 1.0 | 20 | flat-wall | 15.0% | " |
| `eval_stageC_band04` | 0.4 | 20 | in-band | 35.0% | M7 Stage-C vision ablation (3 arms). |
| `eval_stageC` | 1.0 | 20 | flat-wall | 15.0% | " |
| `band_scale0.4` *(committed)* | 0.4 | 30 | in-band | **36.7%** | The 2026-07-07 M6 30-seed slice. |
| `flatwall_scale1.0` *(committed)* | 1.0 | 30 | flat-wall | 20.0% | That run's ceiling-check control. |
| `eval_lab101_band100*` | 0.4 | 100 | in-band | **50.0%** | LAB-101 reproduction (both ar0/ar100). |
| `eval_lab114_*` (×10) | 0.4 | 100 | in-band | **50.0%** | The seed-variance + H-B/H-C study. |

### 2.3 Why the `human_only` baseline moves

The three "contradictory" human baselines quoted across old docs — **36.7 / 31 / 50 / 35 / 15**
— are one number at different operating points, not a discrepancy:

- **50.0%** is the true in-band (es0.4) baseline, measured at 100 seeds, five independent times
  in LAB-114, all *exactly* 50.0% (the arm uses no checkpoint, so it is bit-stable).
- **36.7%** is that same baseline on the **hard 30-seed slice** (seeds 0–29) the 2026-07-07 run
  happened to draw: 36.7% on 0–29 vs 55.7% on 30–99.
- **31.0%** is the LAB-53 run at an **older step-budget era** (pre-LAB-100) — a different
  contact regime, not a different sample.
- **35% / 15%** are the 20-seed es0.4 / es1.0 baselines (M7 sets); es1.0 lands on the flat wall
  where nobody has a lateral lever, so everyone drops.

The stale `outputs/policy/kpi_report/kpi_comparison.json` (human **15.0%**, vision residual
`"PENDING"`) is a fourth operating point *and* an un-refreshed artifact; Phase-3 stage 3C
retires it. The lesson: **a bare success rate is meaningless without its (corpus, error_scale,
seed-count, step-budget-era) tuple.**

---

## 3. Training runs (M5→M7)

Reconstructed from `outputs/policy/runs/<name>/metadata.json`. `val` is `best_val_loss` at
`best_epoch/epochs_run`. **All GPU, seed 0, unless noted.** Offline val loss is *within-recipe*
predictive but **anti-predictive across interventions** (LAB-106) — do not rank recipes by it.

| Run | Date | Corpus (fp) | Config delta vs F/T baseline | `val` (epoch) | Closed-loop | Verdict |
|---|---|---|---|---|---|---|
| `lab34_baseline` | 06-18 | `dataset_1` (`290f`) | M5 first BC, schema-1.0 task | 0.00042 (12) | — no committed eval | offline milestone |
| `ftonly_baseline_lab82` | 07-07 | `dataset_vision` (`de0e`) | F/T residual, no action-rate penalty | 0.00149 (17) | see [the closed-loop ledger](experiment-ledger.md#4-closed-loop-experiment-ledger) (es0.4 20s) | M6/M7 F/T baseline |
| `ftonly_ar30` | 07-08 | `dataset_vision` | + action-rate penalty ×30 | 0.00116 (28) | — | jerk-reduction sweep |
| `ftonly_ar100` | 07-08 | `dataset_vision` | + action-rate penalty ×100 | 0.00140 (23) | 40% (es0.4, 20s) | the "old-ar100" M7 arm |
| `ftonly_wpos10_wd` | 07-10 | `dataset_vision` | pos-loss ×10 + weight-decay | 0.00169 (17) | — | LAB-106 offline fix (1/2) |
| `ftonly_gate_wpos10_wd` | 07-10 | `dataset_vision` | ↑ + **`command_ee_delta`** feedback feature + gate | 0.00135 (26) | **10%** (es0.4, 20s) | **REGRESSION** — see [negative results](mechanisms.md#6-negative-results) |
| `vision_frozen_lab82` | 07-07 | `dataset_vision` | + vision, frozen MobileNetV3 encoder | 0.00107 (26) | — | best offline val of the arc — see caveat |
| `vision_frozen_ar100` | 07-08 | `dataset_vision` | ↑ + action-rate ×100 | 0.00123 (19) | — | |
| `vision_stageC` | 07-10 | `dataset_vision` | vision, **encoder unfrozen** (Stage C) | 0.00161 (16) | 40% in / 10% out (20s) | **NULL** — see [negative results](mechanisms.md#6-negative-results) |
| `dagger_round0` | 07-10 | `dagger_ft_agg` (340) | on-policy relabel, round 0 | 0.00209 (20) | 40% (es0.4, 20s) | DAgger start |
| `dagger_round1` | 07-10 | `dagger_ft_agg` (380) | round 1 | 0.00214 (9) | 30% | **REGRESSION** ([negative results](mechanisms.md#6-negative-results)) |
| `dagger_round2` | 07-10 | `dagger_ft_agg` (420) | round 2 | 0.00194 (13) | 15% | **REGRESSION** ([negative results](mechanisms.md#6-negative-results)) |
| `probe_b2` | 07-07 | `dataset_vision_probe` (10) | vision batch-2 smoke probe | 0.00702 (2) | — | fits-in-8GB probe only |
| `lab101_ft_ar0_ds10` | 07-22 | `dataset_10` (`54dc`) | F/T recipe, GPU repro, ar0 | 0.00130 (22) | **−4.0 pp** (100s) | [the noise floor](noise-floor.md) |
| `lab101_ft_ar100_ds10` | 07-22 | `dataset_10` | ↑ + action-rate ×100 | 0.00178 (13) | **−9.0 pp** (100s) | [the noise floor](noise-floor.md) |
| `lab114_seed{0..4}` | 07-22 | `dataset_10` | F/T recipe, seeds 0–4 (**seeded**) | 0.00117–0.00197 | −15…+3 pp | [the noise floor](noise-floor.md) the spread |
| `lab114_ds9_seed{0..3}` | 07-22 | `dataset_9` | H-B corpus arm — **identical to `_seed`** | ≡ `lab114_seed*` | ≡ | H-B unanswerable |
| `lab114_cpu_seed0` | 07-22 | `dataset_10` | H-C device arm, CPU | 0.00144 (21) | −1.0 pp (100s) | H-C null |

**The offline-val trap, stated once.** `vision_frozen_lab82` has the *best* val loss of the
whole arc (0.00107) and is a closed-loop non-improver; the F/T recipe's own five seeds
span 18 pp of success at val losses 0.00117–0.00197. **A lower validation loss did not buy
closed-loop success across these interventions** — the central M7 mechanism (LAB-106,
[what stands](mechanisms.md#7-what-still-stands)).

---

## 4. Closed-loop experiment ledger

Reconstructed from the per-trial CSVs via `compare_paired`. Δ is paired (McNemar exact p);
`b/c` is discordant wins/losses. **Verdict uses the noise floor ([the noise floor](noise-floor.md)):** `NOISE` = |Δ| within the
20–31 pp training-seed spread + the eval interval at that n.

| Eval set | Op. point | Arm | Success | Paired Δ (n, p) | Verdict |
|---|---|---|---|---|---|
| `flatwall_scale1.0` | es1.0, 30s | residual | 20.0% vs 20.0% | +0.0 pp (30, p=1.0) | flat-wall ceiling control (expected) |
| `runs/eval/` (LAB-53) | es0.4, 100s, old budget | residual | 43.0% vs 31.0% | +12.0 pp (100, p=0.043) | **NOISE** — inside the floor; unseeded training (H-7) |
| `eval_ftgate_es0p4` | es0.4, 20s | ar100 (`residual`) | 40.0% vs 35.0% | +5.0 pp (20, p=1.0) | **NOISE** |
| `eval_ftgate_es0p4` | es0.4, 20s | `command_ee_delta` (`ftonly`) | **10.0%** vs 35.0% | −25.0 pp (20, p=0.125) | **REGRESSION** (mechanism, [negative results](mechanisms.md#6-negative-results)) |
| `eval_ftgate_es1p0` | es1.0, 20s | ar100 / gate | 20% / 15% vs 15% | ≤+5 pp (20, p=1.0) | **NOISE** (flat wall) |
| `eval_stageC_band04` | es0.4, 20s | ftonly / **vision** | 40% / **40%** vs 35% | +5 / +5 pp (20, p=1.0) | **NULL** — vision ties F/T ([negative results](mechanisms.md#6-negative-results)) |
| `eval_stageC` | es1.0, 20s | ftonly / **vision** | 20% / **10%** vs 15% | +5 / −5 pp (20, p=1.0) | **NOISE** (margin < floor) |
| `eval_lab101_band100_ar0` | es0.4, 100s, `dataset_10` | residual | 46.0% vs 50.0% | **−4.0 pp** (100, p=0.557) | reproduction — [the noise floor](noise-floor.md) |
| `eval_lab101_band100` | es0.4, 100s, `dataset_10` | residual (ar100) | 41.0% vs 50.0% | **−9.0 pp** (100, p=0.136) | reproduction — [the noise floor](noise-floor.md) |
| `eval_lab114_seed2` | es0.4, 100s | residual | 35.0% vs 50.0% | **−15.0 pp** (100, **p=0.008**) | a *significant* regression from nothing but a seed — [the noise floor](noise-floor.md) |
| `eval_lab114_seed4` | es0.4, 100s | residual | 53.0% vs 50.0% | +3.0 pp (100, p=0.74) | the other end of the same spread |
| **`eval_official_ft_s*`** | es0.4, 100s, official corpus | residual ×5 seeds | 31–58% vs 50.0% | **mean −4.4 pp** [−19,+8] | **NOISE** (distribution — [the official multi-seed run](noise-floor.md#55-the-official-multi-seed-run--the-definitive-measurement)) |
| **`eval_official_dag_ft_s*`** | es0.4, 100s, official | residual ×5 seeds | 51–53% vs 50.0% | **mean +2.0 pp** [+1,+3] | **NOISE** (tightest — [the official multi-seed run](noise-floor.md#55-the-official-multi-seed-run--the-definitive-measurement)) |
| **`eval_official_vis_s*`** | es0.4, 100s, official | vision ×3 seeds | 34–54% vs 50.0% | **mean −8.3 pp** [−16,+4] | **NOISE** ([the official multi-seed run](noise-floor.md#55-the-official-multi-seed-run--the-definitive-measurement)) |
| **`eval_official_dag_vis_s*`** | es0.4, 100s, official | vision ×3 seeds | 46–62% vs 50.0% | **mean +1.3 pp** [−4,+12] | **NOISE** ([the official multi-seed run](noise-floor.md#55-the-official-multi-seed-run--the-definitive-measurement)) |

The Stage-C DAgger rounds (`eval` per round, 20s es0.4) read **40% → 30% → 15%** across rounds
0–2 — the round-to-round steps are inside the noise floor, but the round-0-to-round-2 drop and
the parallel rollout-success decline (0.325 → 0.25) are the real signal ([negative results](mechanisms.md#6-negative-results)).

---

