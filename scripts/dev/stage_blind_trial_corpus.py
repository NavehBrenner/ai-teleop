"""Stage the 32 blinded-trial episodes into ``data/blind_trial/``.

The session wrote them to ``runs/blind_trial/trial_NNN/`` (gitignored, and holding a
per-trial ``console.log`` beside each trajectory). This copies **only** the
``episode.npz`` into the per-episode folder layout every other corpus uses
(``runs/episode_NNNNN/episode.npz``, ``trajectory.EPISODE_DIR_TEMPLATE``) — so the
destination directories are clean *by construction* rather than by a ``.gitignore``
re-exclude, which is the trap the corpus staging hit with 12,432 wrist frames.

Trial ordering is preserved: ``trial_007`` becomes ``episode_00007``, so the manifest's
``episode_index`` still joins against the published ``docs/results/human-trial/kpis.csv``
``trial`` column and against ``assignments.json``.

    uv run python scripts/dev/stage_blind_trial_corpus.py --check   # verify, copy nothing
    uv run python scripts/dev/stage_blind_trial_corpus.py           # copy
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data.trajectory import EPISODE_NPZ_NAME, episode_npz_path  # noqa: E402

log = get_logger("stage-blind-trial")

DEFAULT_SOURCE = Path("runs/blind_trial")
DEFAULT_TARGET = Path("data/blind_trial")
EXPECTED_TRIALS = 32


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the staged copy matches the source byte-for-byte; copy nothing",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    sources = sorted(arguments.source.glob("trial_*/" + EPISODE_NPZ_NAME))
    if len(sources) != EXPECTED_TRIALS:
        raise SystemExit(
            f"expected {EXPECTED_TRIALS} trials under {arguments.source}, found {len(sources)}"
        )

    runs = arguments.target / "runs"
    for index, source in enumerate(sources):
        # trial_NNN -> episode_NNNNN, and the index must be the trial number itself.
        trial_number = int(source.parent.name.removeprefix("trial_"))
        if trial_number != index:
            raise SystemExit(f"trial numbering is not contiguous: {source.parent.name} at {index}")

        target = episode_npz_path(runs, index)
        if arguments.check:
            if not target.exists():
                raise SystemExit(f"{target} is missing — rerun without --check")
            if _digest(target) != _digest(source):
                raise SystemExit(f"{target} differs from {source}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    total_bytes = sum(episode_npz_path(runs, i).stat().st_size for i in range(len(sources)))
    log.info(
        "%s %d episodes (%.1f MB) %s %s",
        "verified" if arguments.check else "staged",
        len(sources),
        total_bytes / 1e6,
        "in" if arguments.check else "into",
        runs,
    )


if __name__ == "__main__":
    main()
