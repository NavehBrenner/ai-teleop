"""Clamp a process-parallel worker count to what free RAM allows.

Each MuJoCo+torch rollout/eval worker peaks ~1.7 GB RSS (measured 2026-07-25), and
this box is a ~15 GB-capped WSL shared with the editor and the CLI. Launching 8
such workers overcommitted memory and the kernel OOM-killer took out the Claude
Code session (observed twice). So any requested worker count is clamped here to
what ``MemAvailable`` actually permits — a guard against crashing the machine, not
a tuning knob. Raise the ceiling by giving WSL more RAM (``.wslconfig``), not by
editing these constants.
"""

from __future__ import annotations

from pathlib import Path

from ai_teleop.common.log import get_logger

log = get_logger("resources")

PER_WORKER_GB = 2.0  # ~1.7 GB measured peak RSS + margin
RESERVE_GB = 3.0  # kept free for the parent process, editor, CLI and page cache


def _available_gb() -> float:
    """Current ``MemAvailable`` in GiB, or the reserve (→ 1 worker) if unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return RESERVE_GB


def safe_worker_count(requested: int) -> int:
    """Largest worker count ≤ ``requested`` that fits free RAM (always ≥ 1).

    ``requested <= 1`` short-circuits to the sequential path untouched.
    """
    if requested <= 1:
        return 1
    budget = max(0.0, _available_gb() - RESERVE_GB)
    allowed = max(1, min(requested, int(budget / PER_WORKER_GB)))
    if allowed < requested:
        log.warning(
            "capping workers %d → %d (%.1f GB available, ~%.1f GB/worker) to avoid OOM",
            requested,
            allowed,
            _available_gb(),
            PER_WORKER_GB,
        )
    return allowed
