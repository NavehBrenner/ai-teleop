# Delta-target audit — is the BC position target learnable at all?

Regenerate: `uv run python scripts/dev/delta_target_audit.py`
Captured: 2026-08-02 · corpus `dataset_vision`

```
11:26:48 INFO  [lab106audit] episodes used: 300 | active near-hole steps pooled: 502699
11:26:48 INFO  [lab106audit] [A] delta |lateral|/|full| on active steps: 0.896 (─→ label is ~all lateral)
11:26:48 INFO  [lab106audit] [A] zero-Δ baseline |delta| over ALL steps: 5.06 mm (offline eval reported ~4.75)
11:26:48 INFO  [lab106audit] [A] active-step delta per world axis std (mm): [7.55 6.91 8.09]
11:26:48 INFO  [lab106audit] [B] |ee - command| lateral, near-hole: mean 10.66 mm, p90 20.70 mm  (small ⇒ tip≈cmd)
11:26:48 INFO  [lab106audit] [B] per-episode lateral operator error (hole-cmd): cross-ep std [ 5.61 19.55 22.23] mm, |mean| [ 2.72  0.34 15.44] mm
11:26:49 INFO  [lab106audit] [C] F/T-observables → delta LATERAL, held-out R² per axis: [0.073 0.973 0.921] (mean 0.656)
11:26:49 INFO  [lab106audit] [C] F/T-observables → delta FULL,    held-out R² per axis: [0.858 0.979 0.934] (mean 0.924)
11:26:49 INFO  [lab106audit] [C] +PRIVILEGED hole/peg → delta LATERAL R² per axis: [0.311 0.974 0.922] (mean 0.736)  [ceiling]
11:26:50 INFO  [lab106audit] [C2] ALL-steps obs→gated-delta held-out R² per axis: [0.825 0.968 0.822] (mean 0.872)
11:26:50 INFO  [lab106audit] [C2] ALL-steps held-out |err| mm: linear-probe 2.36  vs  zero-Δ 4.91  (offline eval: F/T 7.74)
11:26:50 INFO  [lab106audit] [D] hole world-position across 300 walls: std [20.6 93.7 97.3] mm, range [117.6 357.7 375.2] mm
```
