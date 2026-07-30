# Trained checkpoints

Every trained policy behind a number in [`../kpi-dashboard.md`](../kpi-dashboard.md), committed
so the results can be *run*, not only read. Training runs live in the gitignored
`outputs/policy/runs/`; this tree is the durable copy, laid out by **methodology** rather than by
run name.

## Run one

```bash
kvn episode --policy tf --checkpoint docs/results/checkpoints/ft/bc/seed_0/checkpoint.pt
```

Any checkpoint below works with that command — the file carries its own `PolicyConfig`, so a
**vision** checkpoint loads through the same `--policy tf` path and switches on the wrist camera
by itself. Add `--headless` for a one-line summary instead of the viewer. To reproduce a
dashboard row rather than watch one episode, see
[`../../guides/policy-guide.md`](../../guides/policy-guide.md).

## Layout

```
ft/                     force/torque-only  (Phase 1)
  bc/seed_{0..4}/                    plain behavioral cloning
  bc_batch2/seed_{0..4}/             the same recipe at batch 2 — the confound control
  dagger/seed_{0..4}/round_{0..2}/   DAgger, every round retained
vision/                 wrist camera + force/torque  (Phase 2)
  bc/seed_{0..2}/
  dagger/seed_{0..2}/round_{0..4}/
legacy/                 pre-official runs that back an earlier documented number
```

| Family | Checkpoints | Size |
|---|---|---|
| `ft/bc` | 5 | 3.8 MB |
| `ft/bc_batch2` | 5 | 3.8 MB |
| `ft/dagger` | 15 | 12 MB |
| `vision/bc` | 3 | 15 MB |
| `vision/dagger` | 15 | 74 MB |
| `legacy` | 24 | 31 MB |

Each leaf holds `checkpoint.pt`, the `metadata.json` recording the exact corpus fingerprint and
training config, `history.json`, and the training curve as `history.png`.

## Two things to know before quoting one

**A single checkpoint is not a measurement.** Retraining any of these recipes with a different
training seed moves closed-loop success by 20–27 pp — the reason each family ships as a *set* of
seeds rather than a best-of. Whichever one you load, it is one draw from a distribution, and the
distribution is what [`../kpi-dashboard.md`](../kpi-dashboard.md) §5–§5.5 reports.

**Every DAgger round is retained; the reported round is the last one.** Round-to-round Δ swings
36 pp *within a single training seed* with no trend (vision seed 0 ran −1 → −28 → −12 → +8 → −4).
The rounds are here so the trajectory is inspectable, not so a good one can be picked — selecting
the best round is the same optimistic-selection bias that produced an earlier headline this
project had to retract.

## `legacy/`

Runs from before the official multi-seed campaign, kept because the dashboard's experiment ledger
(§3, §4) cites them: the `lab114_*` seed sweep that first measured the noise floor, the
`lab101_*` reproduction attempts, and the `ftonly_*` / `vision_*` / `dagger_round*` ablations from
M6–M7. Some predate seeded training (before 2026-07-23) and are therefore not reproducible from
their recorded config — the dashboard marks which.

Not committed: `_batch_probe`, `_verify_parallel_*` and `probe_b2`, which are parallelism and
memory-fit tooling artefacts backing no documented number.
