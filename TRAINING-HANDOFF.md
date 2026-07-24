# ▶ TRAINING HANDOFF — the official KPI run, chunked & deadline-bound

> **Working file — never commit** (ignored via `.git/info/exclude`). For the session that runs the
> official checkpoints while Naveh sleeps. Durable plan: `PROJECT-REVIEW.md` (Phase 4) +
> `docs/results/kpi-dashboard.md`.

## What this is

The seeded, **distribution-reported** checkpoints the project is submitted against (decided with
Naveh 2026-07-25). Report every recipe as a **distribution over training seeds**, never a single
checkpoint — that single-checkpoint mistake is what LAB-114 exposed.

Split into **four independent, resumable, self-timing chunks** so they can be run one at a time,
overnight, with a decision between each:

| # | Chunk script | What | Cost |
|---|---|---|---|
| 1 | `ft_plain.sh` | FT-plain ×5 seeds + eval | **cheap, certain** — no render. Run first. |
| 2 | `ft_dagger.sh` | FT-DAgger ×5 + eval | moderate — 3 DAgger rounds each, no render |
| 3 | `vision_plain.sh` | Vision-plain ×3 + eval | **long pole** — corpus renders ~10 s/ep (~3 h) + render-bound eval |
| 4 | `vision_dagger.sh` | Vision-DAgger ×3 + eval | most expensive — renders in rollouts *and* eval |

All on **master** (`scripts/dev/official_kpi/`). All command forms smoke-tested; both guards
(skip-if-done, stop-at-deadline) tested.

## The two hard rules

1. **Nothing running past 2026-07-26 13:00.** Export the deadline so no chunk starts new work past it:
   ```bash
   export STOP_BY=$(date -d '2026-07-26 13:00' +%s)
   ```
   Every step checks `STOP_BY` and refuses to *start* past it. **But an in-flight step still
   finishes**, and a vision step can be 30–60 min — so do NOT launch a chunk you can't expect to
   finish comfortably before 13:00. Leave margin.
2. **Don't run during Naveh's working hours.** Run chunks overnight; don't start one that would
   spill into the workday. If in doubt, defer to the next night. (GPU can't be cleanly capped to
   "50%" here — MuJoCo render + torch share it — so *scheduling*, not throttling, is the control.)

## The loop you run — one chunk, measure, decide

For each chunk, in order (1→4):

1. **Estimate** from the previous chunk's per-step seconds (each `DONE name (Ns)` line is logged).
   Priors to sanity-check against: FT corpus ~20–30 min; FT train ~2–4 min/seed; FT eval (100
   seeds) ~5–15 min; **vision corpus ~2.5–3 h**; vision train ~10–30 min/seed; vision eval
   ~30–60 min/seed. DAgger ≈ 3× a train plus rollout gen.
2. **Decide:** launch only if it finishes with margin before 13:00 **and** clears the workday. Else
   stop and leave it for the next slot — the chunks are resumable, so nothing is lost.
3. **Launch in the background** (chunks run longer than a foreground tool call):
   ```bash
   export STOP_BY=$(date -d '2026-07-26 13:00' +%s)
   nohup bash scripts/dev/official_kpi/ft_plain.sh > outputs/policy/official_kpi_logs/CHUNK.ft_plain.log 2>&1 &
   ```
   (launch it as a background Bash task and wait for the completion notification).
4. **When it finishes**, read the log's `DONE …(Ns)` timings, run the aggregate to see the
   distribution so far, then return to step 1 for the next chunk:
   ```bash
   uv run python scripts/dev/official_kpi/aggregate.py
   ```

**Resumability:** every step skips if its output exists, so a killed/relaunched chunk continues
where it stopped. Safe to re-run any chunk. Lower `EPISODES` (e.g. `export EPISODES=400`) if a
vision chunk won't fit a slot — the recipe is unchanged, just a smaller corpus.

**Hard backstop (optional):** to guarantee nothing survives 13:00 even mid-step, arm once:
```bash
( sleep $(( $(date -d '2026-07-26 13:00' +%s) - $(date +%s) )); pkill -f official_kpi; pkill -f 'uv run kvn'; pkill -f dagger.py ) &
```
(A killed step just leaves an incomplete run folder with no checkpoint — the resume logic re-does it.)

## Dependencies between chunks

- `ft_dagger` needs `ft_plain` (base = `official_ft_s0`); it errors early if missing.
- `vision_dagger` needs `vision_plain` (corpus + base = `official_vis_s0`).
- `ft_*` and `vision_*` are otherwise independent — the FT and vision corpora are separate.

## When chunks are done — hand back

Fold `aggregate.py`'s per-recipe distributions into `docs/results/kpi-dashboard.md` (a new
*Official run* section) and `docs/results/phase-1-results.md`, as **mean ± range over seeds** with
each row's n and the 18 pp floor beside it. Expected (per the mechanism findings): all arms cluster
near the human baseline, no significant lift — making the documented negative **official and
rigorous**. Standing positives (bounded-force guarantee + mechanisms) hold regardless of sign.

Then Phase-4 close-out (`PROJECT-REVIEW.md`): D-6 `go-forward.md` (decision record — ready to
write), D-7 (rewrite LAB-42), delete the working files.
