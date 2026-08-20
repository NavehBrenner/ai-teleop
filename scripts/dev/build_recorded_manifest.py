"""Build ``data/recorded/metadata.json`` from the recorded episodes themselves.

Every other corpus in ``data/`` commits a manifest written *by* the generator that
produced it. The recorded corpus has no generator — 54 live teleoperation episodes,
recorded 2026-07-03 — so its manifest is derived here, from what the episode files
already carry, and this script is the audit trail for that derivation.

It is deliberately strict: every field the manifest reports as corpus-wide is checked
to be identical across all episodes first, and a disagreement is an error rather than
an average. A summary that quietly averaged over a corpus that is not uniform would be
worse than no summary.

The manifest uses the *same* ``ResBCDatasetMetadata`` shape as the generated corpora —
one contract, not two — with the generation-only fields carrying explicit
not-applicable values: ``master_seed`` is ``RECORDED_MASTER_SEED``, ``fingerprint`` and
``config.expert_d_far`` are null, and ``expert`` is an empty rate block. The human's own
outcomes go in ``baseline_no_assist``, because a no-assist operator *is* the unassisted
arm that field names.

    uv run python scripts/dev/build_recorded_manifest.py            # write the manifest
    uv run python scripts/dev/build_recorded_manifest.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data.schema import RECORDED_MASTER_SEED  # noqa: E402
from ai_teleop.data.trajectory import load_episode  # noqa: E402

log = get_logger("recorded-manifest")

DEFAULT_CORPUS = Path("data/recorded")
SCHEMA_VERSION = "2.0"

# When the session happened, stated rather than inferred. This was read from the episode
# files' mtime until 2026-08-20, which made `--check` unpassable on every machine but the
# one that recorded them: git does not preserve mtime, so a fresh clone re-derives the
# checkout time and reports the committed manifest stale. Nothing *inside* an episode
# records when it was recorded, so the honest options are a stated constant or no field at
# all -- and `generated_at` is required by the schema. The value is the one the original
# mtime derivation produced, so the committed manifest is unchanged.
RECORDED_AT = "2026-07-03T07:15:50.242902+00:00"

# Per-episode metadata keys that must agree across the whole corpus for a single
# corpus-level `config` block to be truthful. `max_steps` is not among them: it is not
# stamped per episode and is recovered from the timeouts instead (see `_infer_max_steps`).
_CONSTANT_KEYS = (
    "source",
    "policy",
    "generated_wall",
    "scene",
    "max_dpos",
    "joint_damping",
    "force_cap",
    "success_depth",
    "lateral_tolerance",
)


def _infer_max_steps(episodes: list[dict[str, Any]]) -> int:
    """Recover the session's step budget from the episodes that hit it.

    A timeout is by definition an episode that ran out of budget, so its ``n_steps``
    *is* the budget. Requiring every timeout to agree is what makes this an observation
    rather than a guess; a corpus with no timeout cannot support the claim at all.
    """
    timeouts = {entry["n_steps"] for entry in episodes if entry["terminal_reason"] == "timeout"}
    if not timeouts:
        raise SystemExit("no timeouts in the corpus — the step budget is not recoverable")
    if len(timeouts) > 1:
        raise SystemExit(f"timeouts disagree on the step budget: {sorted(timeouts)}")
    return timeouts.pop()


def build(corpus: Path) -> dict[str, Any]:
    """Derive the corpus manifest from every ``episode.npz`` under ``corpus/runs/``."""
    paths = sorted(corpus.glob("runs/*/episode.npz"))
    if not paths:
        raise SystemExit(f"no episodes under {corpus}/runs/")

    per_episode = [(path, load_episode(path)[1]) for path in paths]

    # Every corpus-wide claim is checked before it is made.
    for key in _CONSTANT_KEYS:
        values = {json.dumps(metadata.get(key)) for _, metadata in per_episode}
        if len(values) != 1:
            raise SystemExit(f"{key} is not constant across the corpus: {sorted(values)}")
    common = per_episode[0][1]

    if common["policy"] != "noassist":
        raise SystemExit(
            f"expected an unassisted corpus, found policy={common['policy']!r} — the "
            "manifest reports these episodes as the no-assist arm and that would be false"
        )

    episodes: list[dict[str, Any]] = [
        {
            "episode_index": index,
            "file": path.relative_to(corpus).as_posix(),
            "n_steps": int(metadata["n_steps"]),
            "target_hole_index": int(metadata["target_hole_index"]),
            # No master seed to root a scene seed in, and no seed behind the operator:
            # a person drove these. Both are null rather than absent so the entry keeps
            # the shape a generated one has.
            "scene_seed": None,
            "human_seed": None,
            "wall_seed": metadata.get("wall_seed"),
            "terminal_reason": metadata["terminal_reason"],
            "success": bool(metadata["episode_success"]),
        }
        for index, (path, metadata) in enumerate(per_episode)
    ]

    counts = Counter(entry["terminal_reason"] for entry in episodes)
    successes = sum(entry["success"] for entry in episodes)

    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": RECORDED_MASTER_SEED,
        "n_episodes": len(episodes),
        "generated_at": RECORDED_AT,
        # No GenerationConfig produced this corpus, so there is nothing to hash.
        "fingerprint": None,
        "config": {
            "max_steps": _infer_max_steps(episodes),
            "max_dpos": common["max_dpos"],
            # The expert never ran: these are unassisted human episodes. The scripted-
            # human and expert-brake knobs are NotRequired and simply absent, which is
            # already this schema's way of saying a knob did not apply.
            "expert_d_far": None,
            "success_depth": common["success_depth"],
            "lateral_tolerance": common["lateral_tolerance"],
            "force_cap": common["force_cap"],
            "scene": "generated" if common["generated_wall"] else common["scene"],
            "joint_damping": common["joint_damping"],
        },
        # No privileged-expert arm exists here.
        "expert": {"counts": {}, "success_rate": None},
        # The operator *is* the unassisted arm, so their outcomes belong in the field
        # that names it — not in `expert`, and not in a third block invented for humans.
        "baseline_no_assist": {
            "counts": dict(sorted(counts.items())),
            "success_rate": successes / len(episodes),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and compare against the committed manifest; write nothing",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    manifest = build(arguments.corpus)
    target = arguments.corpus / "metadata.json"
    rendered = json.dumps(manifest, indent=2) + "\n"

    if arguments.check:
        if not target.exists():
            raise SystemExit(f"{target} does not exist")
        if target.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{target} is stale — rerun without --check")
        log.info("%s matches the episodes on disk", target)
        return

    target.write_text(rendered, encoding="utf-8")
    baseline = manifest["baseline_no_assist"]
    log.info(
        "wrote %s — %d episodes, human success %.1f%% (%s)",
        target,
        manifest["n_episodes"],
        100.0 * baseline["success_rate"],
        ", ".join(f"{key} {value}" for key, value in baseline["counts"].items()),
    )


if __name__ == "__main__":
    main()
