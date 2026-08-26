# Error decomposition — where the GRU's offline position error lives

Regenerate: `uv run python scripts/dev/error_decomp.py`
Captured: 2026-08-02 · corpus `dataset_vision` (held-out val split)

```
11:25:46 INFO  [error-decomp] val episodes: 60 (seed 0, val_fraction 0.2)
11:25:46 WARNING [residual_policy] checkpoint config carries retired key(s) use_tanh_head — ignoring them
11:25:58 INFO  [error-decomp] ── ftonly_baseline_lab82  (command_ee_delta=False) ──
11:25:58 INFO  [error-decomp]   far d>=0.15      n= 123797 │ GRU |err|  5.64 mm │ zero-Δ  0.00 mm │ GRU |pred|  5.64 mm
11:25:58 INFO  [error-decomp]   near 0.05-0.15   n=  39586 │ GRU |err|  8.78 mm │ zero-Δ  9.42 mm │ GRU |pred|  6.29 mm
11:25:58 INFO  [error-decomp]   close d<0.05     n=  45760 │ GRU |err| 12.02 mm │ zero-Δ 13.56 mm │ GRU |pred| 10.01 mm
11:25:58 INFO  [error-decomp]   ALL steps        n= 209143 │ GRU |err|  7.63 mm │ zero-Δ  4.75 mm  ← offline pos
11:25:58 WARNING [residual_policy] checkpoint config carries retired key(s) use_tanh_head — ignoring them
11:26:12 INFO  [error-decomp] ── ftonly_wpos10_wd  (command_ee_delta=False) ──
11:26:12 INFO  [error-decomp]   far d>=0.15      n= 123797 │ GRU |err|  3.83 mm │ zero-Δ  0.00 mm │ GRU |pred|  3.83 mm
11:26:12 INFO  [error-decomp]   near 0.05-0.15   n=  39586 │ GRU |err|  8.93 mm │ zero-Δ  9.42 mm │ GRU |pred|  5.34 mm
11:26:12 INFO  [error-decomp]   close d<0.05     n=  45760 │ GRU |err| 11.50 mm │ zero-Δ 13.56 mm │ GRU |pred|  9.90 mm
11:26:12 INFO  [error-decomp]   ALL steps        n= 209143 │ GRU |err|  6.47 mm │ zero-Δ  4.75 mm  ← offline pos
11:26:12 WARNING [residual_policy] checkpoint config carries retired key(s) use_tanh_head — ignoring them
11:26:24 INFO  [error-decomp] ── ftonly_gate_wpos10_wd  (command_ee_delta=True) ──
11:26:24 INFO  [error-decomp]   far d>=0.15      n= 123797 │ GRU |err|  2.81 mm │ zero-Δ  0.00 mm │ GRU |pred|  2.81 mm
11:26:24 INFO  [error-decomp]   near 0.05-0.15   n=  39586 │ GRU |err|  3.74 mm │ zero-Δ  9.42 mm │ GRU |pred|  9.09 mm
11:26:24 INFO  [error-decomp]   close d<0.05     n=  45760 │ GRU |err|  4.97 mm │ zero-Δ 13.56 mm │ GRU |pred| 13.76 mm
11:26:24 INFO  [error-decomp]   ALL steps        n= 209143 │ GRU |err|  3.46 mm │ zero-Δ  4.75 mm  ← offline pos
```
