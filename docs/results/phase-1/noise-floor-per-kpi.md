# The training-seed noise floor, per KPI

Every reported metric's paired Δ against `human_only`, one draw per training seed, with
the range across seeds. **That range is the floor the metric's own treatment effects must
clear.** A recipe whose mean Δ is smaller than its family's floor is not distinguishable
from a re-draw of the same recipe.

Regenerate: `uv run python scripts/dev/official_kpi/noise_floor.py`
Source: the committed per-trial CSVs in [`official-evals/`](official-evals/).

### Success rate (%) — paired Δ vs human_only, per training seed

| Recipe | seeds | per-seed Δ | mean | **floor (range)** |
|---|---|---|---|---|
| FT plain | 5 | -19.00, -5.00, -3.00, -3.00, +8.00 | -4.40 | **27.00** |
| FT plain (batch 2) | 5 | -21.00, -5.00, -1.00, +0.00, +10.00 | -3.40 | **31.00** |
| FT DAgger | 5 | +1.00, +2.00, +2.00, +2.00, +3.00 | +2.00 | **2.00** |
| Vision plain | 3 | -16.00, -13.00, +4.00 | -8.33 | **20.00** |
| Vision DAgger | 3 | -4.00, -4.00, +12.00 | +1.33 | **16.00** |

### Time to insert (s) — paired Δ vs human_only, per training seed

| Recipe | seeds | per-seed Δ | mean | **floor (range)** |
|---|---|---|---|---|
| FT plain | 5 | -0.05, +0.06, +0.17, +0.18, +0.40 | +0.15 | **0.46** |
| FT plain (batch 2) | 5 | +0.05, +0.12, +0.27, +0.35, +0.58 | +0.27 | **0.53** |
| FT DAgger | 5 | +0.06, +0.09, +0.15, +0.17, +0.21 | +0.14 | **0.14** |
| Vision plain | 3 | +0.01, +0.08, +0.55 | +0.21 | **0.54** |
| Vision DAgger | 3 | +0.04, +0.32, +0.36 | +0.24 | **0.31** |

### Peak contact force (N) — paired Δ vs human_only, per training seed

| Recipe | seeds | per-seed Δ | mean | **floor (range)** |
|---|---|---|---|---|
| FT plain | 5 | -0.04, +0.91, +1.43, +1.76, +5.06 | +1.83 | **5.10** |
| FT plain (batch 2) | 5 | -2.76, -1.01, -0.07, +0.48, +4.83 | +0.30 | **7.59** |
| FT DAgger | 5 | -2.30, -1.75, -0.87, -0.41, +0.11 | -1.04 | **2.41** |
| Vision plain | 3 | -0.60, +1.12, +1.61 | +0.71 | **2.21** |
| Vision DAgger | 3 | -3.04, +0.39, +1.22 | -0.48 | **4.26** |

### Trajectory jerk (∫|jerk|) — paired Δ vs human_only, per training seed

| Recipe | seeds | per-seed Δ | mean | **floor (range)** |
|---|---|---|---|---|
| FT plain | 5 | +14.02, +18.50, +19.21, +21.22, +49.58 | +24.51 | **35.56** |
| FT plain (batch 2) | 5 | +1.00, +1.01, +1.88, +15.27, +16.29 | +7.09 | **15.29** |
| FT DAgger | 5 | -0.52, +2.36, +2.59, +7.04, +186.14 | +39.52 | **186.66** |
| Vision plain | 3 | +0.41, +1.44, +33.10 | +11.65 | **32.68** |
| Vision DAgger | 3 | +0.24, +2.62, +4.44 | +2.43 | **4.20** |
