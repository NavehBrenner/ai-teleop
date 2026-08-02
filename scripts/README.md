# scripts/

Executable entry points for the project workflow. Prefer the **`kvn` CLI** as the
front door — it dispatches to these scripts and forwards their flags
(`kvn <command> --help`). See [`../docs/guides/cli.md`](../docs/guides/cli.md). Running a script
file directly with `uv run python scripts/<name>.py` still works.

One-time project setup (not a `kvn` command — it bootstraps `kvn` itself):

- `setup.sh` / `setup.ps1`     — create the venv, install the package, enable hooks, install the
  `kvn` launcher on PATH. Use the PowerShell one on Windows-native.

Scripts with a `kvn` command:

- `view_generated_wall.py`     (`kvn sim`)      — generate / view a procedural wall.
- `smoke_test_sim.py`          (`kvn smoke`)    — load scene, dump sensors + wrist-cam PNG.
- `run_episode.py`             (`kvn episode`)  — one end-to-end episode, any input × any assist.
- `dev_harness_controller.py`  (`kvn harness`)  — backbone-controller dev harness.
- `generate_dataset.py`        (`kvn gen`)      — unattended behavioral-cloning data generation.
- `train_policy.py`            (`kvn train`)    — BC training → a deployable checkpoint.
- `evaluate.py`                (`kvn evaluate`) — paired ablation + difficulty sweep → trials.csv.

Scripts run directly (not registered in `APP_COMMANDS`):

- `dagger.py`                  — on-policy DAgger relabelling rounds.
- `report_results.py`          — KPI tables + paired statistics from a `trials.csv`.

`dev/` holds one-off investigation scripts kept for provenance; `dev/official_kpi/` regenerates
every published results table and figure.

When you add a runnable script here, register it in `APP_COMMANDS` in
`../src/ai_teleop/cli.py` so it gets a `kvn` command.
