"""Build ``data/blind_trial/metadata.json`` from the blinded-trial episodes (LAB-125).

The sibling of ``build_recorded_manifest.py``, for the other corpus in this project that
no seed can rebuild: the 32 trials of the pre-registered blinded human-operator session
(2026-08-20, ``docs/results/human-trial-result.md``). Same reason for existing — a
recorded corpus has no generator, so its manifest is *derived*, and this script is the
audit trail for the derivation — and the same strictness: every corpus-wide claim is
checked across all episodes before it is made.

One structural difference drives everything below. ``data/recorded/`` is a single
unassisted arm; this corpus is **two-armed**, and the arm is carried by ``assist_scale``
(1.0 / 0.0) rather than by ``policy``. Both arms ran at ``policy="tf"`` deliberately:
scale 0 still loads the checkpoint and still runs the forward pass, which is what made
the arms indistinguishable to the operator. So ``policy`` is constant here, and reading
it alone would report all 32 episodes as assisted.

Consequently the two arms get two aggregate blocks — the unassisted half in
``baseline_no_assist``, the assisted half in ``assisted_residual`` — and neither is
folded into ``expert``, which names the *analytical, privileged-info* expert that never
ran here. A single corpus-wide success rate is not written at all: it would average over
episodes that did not share a condition, and the pre-registered analysis deliberately
reports two arms with an interval instead.

    uv run python scripts/dev/build_blind_trial_manifest.py            # write the manifest
    uv run python scripts/dev/build_blind_trial_manifest.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
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

log = get_logger("blind-trial-manifest")

DEFAULT_CORPUS = Path("data/blind_trial")
SCHEMA_VERSION = "2.0"

# The published per-trial outcomes. Cross-checking against them is what makes this
# manifest a *description* of the trial that was analysed, rather than a second,
# independently-drifting account of it.
PUBLISHED_KPIS = Path("docs/results/human-trial/kpis.csv")

# When the session ran, stated rather than inferred from file mtime — git does not
# preserve mtime, so an mtime derivation makes `--check` unpassable on any fresh clone
# (the trap `build_recorded_manifest.py` sat in until 2026-08-20). Nothing inside an
# episode records its wall-clock time, so this is stated once, here.
RECORDED_AT = "2026-08-20T14:36:27.907115+00:00"

# The step budget the trials ran under. `build_recorded_manifest.py` recovers this from
# the episodes that timed out, which cannot work here: no trial timed out (every one
# ended in `success` or `force_abort`). So it is the live default the child
# `run_episode.py` processes fell back to — `blind_trial.py` passes no `--max-steps` —
# and `_check_budget_unreached` verifies the corpus is at least *consistent* with it.
LIVE_MAX_STEPS = 5000

EXPECTED_EPISODES = 32
ASSISTED = 1.0
UNASSISTED = 0.0

# Per-episode keys that must agree across the whole corpus. `assist_scale` is pointedly
# *not* here — it is the treatment, and a corpus where it did not vary would be a failed
# blind rather than a uniform corpus (see `_split_arms`).
_CONSTANT_KEYS = (
    "source",
    "policy",
    "seed",
    "generated_wall",
    "scene",
    "distractors",
    "max_dpos",
    "joint_damping",
    "force_cap",
    "success_depth",
    "lateral_tolerance",
)


def _check_budget_unreached(episodes: list[dict[str, Any]]) -> int:
    """Confirm no episode ran to the step budget, then report it.

    A timeout would mean the budget is *observable* in the data and this constant is the
    wrong way to get it; it would also contradict the recorded terminal reasons. Either
    way, the manifest should not quietly report a number the episodes disagree with.
    """
    overruns = [entry for entry in episodes if entry["n_steps"] >= LIVE_MAX_STEPS]
    if overruns:
        raise SystemExit(
            f"{len(overruns)} episodes reached the assumed {LIVE_MAX_STEPS}-step budget — "
            "it is not the budget these trials ran under"
        )
    return LIVE_MAX_STEPS


def _split_arms(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition into (assisted, unassisted), rejecting anything that is neither."""
    scales = {entry["assist_scale"] for entry in episodes}
    if not scales <= {ASSISTED, UNASSISTED}:
        raise SystemExit(f"unexpected assist_scale values: {sorted(scales)}")
    if scales != {ASSISTED, UNASSISTED}:
        raise SystemExit(
            f"expected a two-armed corpus, found only assist_scale={sorted(scales)} — "
            "a manifest reporting two arms from one would be false"
        )
    return (
        [entry for entry in episodes if entry["assist_scale"] == ASSISTED],
        [entry for entry in episodes if entry["assist_scale"] == UNASSISTED],
    )


def _rate_block(arm: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``{counts-by-terminal-reason, success_rate}`` aggregate over one arm."""
    counts = Counter(entry["terminal_reason"] for entry in arm)
    return {
        "counts": dict(sorted(counts.items())),
        "success_rate": sum(entry["success"] for entry in arm) / len(arm),
    }


def _cross_check_published(episodes: list[dict[str, Any]], kpis: Path) -> None:
    """Every episode must agree with the published per-trial row it corresponds to.

    The manifest is derived from the ``.npz`` files, and the analysis was computed from
    the same session, so a disagreement means the staged corpus is not the corpus that
    was analysed — the one failure this whole exercise exists to make impossible.
    """
    if not kpis.exists():
        log.warning("%s not found — skipping the cross-check against the published result", kpis)
        return

    with open(kpis, encoding="utf-8", newline="") as handle:
        published = {int(row["trial"]): row for row in csv.DictReader(handle)}

    if sorted(published) != [entry["episode_index"] for entry in episodes]:
        raise SystemExit(
            f"{kpis} covers trials {sorted(published)}, which the corpus does not match"
        )

    for entry in episodes:
        row = published[entry["episode_index"]]
        index = entry["episode_index"]
        published_assist = row["assist_on"].strip().lower() == "true"
        if (entry["assist_scale"] == ASSISTED) != published_assist:
            raise SystemExit(
                f"trial {index}: assist_scale={entry['assist_scale']} contradicts the "
                f"published assist_on={row['assist_on']!r}"
            )
        if entry["terminal_reason"] != row["outcome"]:
            raise SystemExit(
                f"trial {index}: terminal_reason={entry['terminal_reason']!r} contradicts "
                f"the published outcome={row['outcome']!r}"
            )
        if entry["wall_seed"] != int(row["wall_seed"]):
            raise SystemExit(
                f"trial {index}: wall_seed={entry['wall_seed']} contradicts the published "
                f"{row['wall_seed']}"
            )
    log.info("cross-checked all %d episodes against %s", len(episodes), kpis)


def build(corpus: Path, kpis: Path) -> dict[str, Any]:
    """Derive the corpus manifest from every ``episode.npz`` under ``corpus/runs/``."""
    paths = sorted(corpus.glob("runs/*/episode.npz"))
    if len(paths) != EXPECTED_EPISODES:
        raise SystemExit(
            f"expected {EXPECTED_EPISODES} episodes under {corpus}/runs/, found {len(paths)}"
        )

    per_episode = [(path, load_episode(path)[1]) for path in paths]

    # Every corpus-wide claim is checked before it is made.
    for key in _CONSTANT_KEYS:
        values = {json.dumps(metadata.get(key)) for _, metadata in per_episode}
        if len(values) != 1:
            raise SystemExit(f"{key} is not constant across the corpus: {sorted(values)}")
    common = per_episode[0][1]

    if common["policy"] != "tf":
        raise SystemExit(
            f"expected the blinded harness's policy=tf, found {common['policy']!r} — both "
            "arms run the checkpoint and differ only in assist_scale, so any other value "
            "means these are not the blinded trials"
        )

    episodes: list[dict[str, Any]] = [
        {
            "episode_index": index,
            "file": path.relative_to(corpus).as_posix(),
            "n_steps": int(metadata["n_steps"]),
            "target_hole_index": int(metadata["target_hole_index"]),
            # A person drove these: no master seed to root a scene seed in, and no seed
            # behind the operator. Null rather than absent, so the entry keeps the shape
            # a generated one has.
            "scene_seed": None,
            "human_seed": None,
            "wall_seed": metadata.get("wall_seed"),
            "assist_scale": float(metadata["assist_scale"]),
            "terminal_reason": metadata["terminal_reason"],
            "success": bool(metadata["episode_success"]),
        }
        for index, (path, metadata) in enumerate(per_episode)
    ]

    _cross_check_published(episodes, kpis)
    assisted, unassisted = _split_arms(episodes)

    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": RECORDED_MASTER_SEED,
        "n_episodes": len(episodes),
        "generated_at": RECORDED_AT,
        # No GenerationConfig produced this corpus, so there is nothing to hash.
        "fingerprint": None,
        "config": {
            "max_steps": _check_budget_unreached(episodes),
            "max_dpos": common["max_dpos"],
            # The analytical expert never ran. The assisted arm is a *trained* residual
            # and is reported in `assisted_residual`, not here.
            "expert_d_far": None,
            "success_depth": common["success_depth"],
            "lateral_tolerance": common["lateral_tolerance"],
            "force_cap": common["force_cap"],
            "scene": "generated" if common["generated_wall"] else common["scene"],
            "joint_damping": common["joint_damping"],
        },
        # No privileged-expert arm exists here — see the module docstring.
        "expert": {"counts": {}, "success_rate": None},
        "assisted_residual": _rate_block(assisted),
        "baseline_no_assist": _rate_block(unassisted),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--kpis", type=Path, default=PUBLISHED_KPIS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and compare against the committed manifest; write nothing",
    )
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)

    manifest = build(arguments.corpus, arguments.kpis)
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
    assisted = manifest["assisted_residual"]
    unassisted = manifest["baseline_no_assist"]
    log.info(
        "wrote %s — %d episodes; assist on %.1f%% (n=%d), off %.1f%% (n=%d)",
        target,
        manifest["n_episodes"],
        100.0 * assisted["success_rate"],
        sum(assisted["counts"].values()),
        100.0 * unassisted["success_rate"],
        sum(unassisted["counts"].values()),
    )


if __name__ == "__main__":
    main()
