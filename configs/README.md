# configs/

**Empty by design.** The project configures runs through typed dataclasses in code
(`policy/config.py`, `sim/scenegen/config.py`, …) plus CLI flags, not through config files —
so a run's settings travel with its `metadata.json` rather than with a separate YAML that can
drift from it.

This directory is kept as the place a run-time config file would go if one is ever needed.
