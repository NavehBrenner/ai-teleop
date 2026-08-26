# Perception probe — is hole position linearly decodable from the frozen encoder?

Regenerate: `uv run python scripts/dev/perception_probe.py data/dataset_vision`
Captured: 2026-08-02

```
collected 13431 frames from 80 episodes

[all frames]
  x: R²=-0.618  RMSE=5.87 cm
  y: R²=+0.451  RMSE=4.38 cm
  z: R²=+0.951  RMSE=3.00 cm
  mean R²=+0.261  mean RMSE=4.42 cm

near-hole frames (d < 0.15 m): 6167 / 13431

[near-hole frames]
  x: R²=-0.356  RMSE=1.32 cm
  y: R²=+0.062  RMSE=1.37 cm
  z: R²=+0.878  RMSE=1.71 cm
  mean R²=+0.195  mean RMSE=1.47 cm

target spread (dead-decoder RMSE): [ 4.55  5.63 13.37] cm per axis
```
