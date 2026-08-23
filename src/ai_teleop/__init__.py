"""AI-assisted robotic teleoperation for precision peg-in-hole insertion.

Top-level package. See submodules:

- domain   : interfaces (Protocols) and core dataclasses (Observation, Command, ...).
- sim      : MuJoCo scene wrapper, sensor reading, rendering.
- control  : backbone controller (operational-space IK, impedance) — the always-on substrate.
- input    : input strategies (vision, keyboard, scripted noisy-human) behind a common interface.
- expert   : analytical privileged-info expert (data-generation supervisor).
- policy   : residual correction policy (BC-trained neural network).
- data     : data-generation pipeline, dataset loaders.
- eval     : evaluation harness (passive observer, KPI computation, ablation orchestration).
- common   : shared utilities (logging, types, math).
"""

__version__ = "0.0.1"

# Make the console tolerant of non-ASCII output before anything can write any.
#
# An import side effect, deliberately, and at the package root because that is the only
# place that covers every entry point: this is a property of the *process*, not of any one
# parser or logger. Wiring it into `add_logging_arguments` alone left 45 scripts in
# `scripts/dev/` uncovered — `record_comparison_grid.py` died on the `█` in its progress
# bar, printed with a bare `print()`, having never touched the project logger.
#
# On Windows `sys.stdout` gets the console code page (cp1252 on a stock en-US install) in
# strict mode, so a single `Δ`, `█` or `→` is fatal — and these are spread across help
# text, progress bars and log messages. The call is idempotent and silent on failure; see
# `common.log.ensure_console_encoding`.
from ai_teleop.common.log import ensure_console_encoding as _ensure_console_encoding

_ensure_console_encoding()
