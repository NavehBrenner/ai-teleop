"""The blinded-trial corpus must stay the corpus that was analysed.

``data/blind_trial/`` is the second corpus here that no seed can rebuild (after
``data/recorded/`` — see ``test_recorded_corpus.py`` for the regeneration guard those
two share). What is specific to this one is that it is **two-armed**: 16 assisted and
16 unassisted trials, distinguished by ``assist_scale`` and *not* by ``policy``, which
is ``"tf"`` for all 32 because both arms load the checkpoint and run the forward pass.

Two failure modes get tests, because neither is visible from reading the manifest:

1. **An aggregate that averages over both arms.** The published result is two rates and
   an interval that spans zero; one blended number would read as a result the trial did
   not produce.
2. **A derivation that cannot be re-run.** The manifest is *derived* from the episodes,
   and the script that derives it advertises ``--check`` as the audit trail. Until
   2026-08-20 that check compared a committed constant against a value read from file
   **mtime**, which git does not preserve — so it passed only on the machine that
   recorded the episodes and reported "stale" on every fresh clone. Both corpora now
   state their timestamp, and both derivations are pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai_teleop.data.generate import GenerationConfig, NotRegenerableError
from ai_teleop.data.schema import RECORDED_MASTER_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))

BLIND_TRIAL_CORPUS = REPO_ROOT / "data" / "blind_trial"
BLIND_TRIAL_MANIFEST = BLIND_TRIAL_CORPUS / "metadata.json"
RECORDED_CORPUS = REPO_ROOT / "data" / "recorded"
RECORDED_MANIFEST = RECORDED_CORPUS / "metadata.json"

ASSISTED = 1.0
UNASSISTED = 0.0

requires_manifest = pytest.mark.skipif(
    not BLIND_TRIAL_MANIFEST.exists(), reason="the blind-trial manifest is not present"
)


def _manifest() -> dict:
    return json.loads(BLIND_TRIAL_MANIFEST.read_text(encoding="utf-8"))


@requires_manifest
def test_committed_manifest_declares_itself_unregenerable() -> None:
    """A human drove these; nothing may simulate over them."""
    manifest = _manifest()
    assert manifest["master_seed"] == RECORDED_MASTER_SEED
    assert manifest["fingerprint"] is None
    assert manifest["n_episodes"] == len(manifest["episodes"])
    with pytest.raises(NotRegenerableError):
        GenerationConfig.from_metadata(manifest)  # type: ignore[arg-type]


@requires_manifest
def test_the_two_arms_are_reported_separately() -> None:
    """The assisted arm must not be folded into ``expert`` or into a blended rate.

    ``expert`` names the analytical, privileged-info expert, which never ran here — the
    assisted arm is a *trained* residual. Reporting one under the other's name would
    overstate what produced the numbers.
    """
    manifest = _manifest()
    assert manifest["expert"] == {"counts": {}, "success_rate": None}

    assisted = manifest["assisted_residual"]
    unassisted = manifest["baseline_no_assist"]
    assert sum(assisted["counts"].values()) == 16
    assert sum(unassisted["counts"].values()) == 16
    assert (
        sum(assisted["counts"].values()) + sum(unassisted["counts"].values())
        == (manifest["n_episodes"])
    )


@requires_manifest
def test_arm_aggregates_match_the_per_episode_entries() -> None:
    """Recompute both blocks from the episode list rather than trusting them."""
    manifest = _manifest()
    for block, scale in (("assisted_residual", ASSISTED), ("baseline_no_assist", UNASSISTED)):
        arm = [entry for entry in manifest["episodes"] if entry["assist_scale"] == scale]
        successes = sum(entry["success"] for entry in arm)
        assert manifest[block]["success_rate"] == pytest.approx(successes / len(arm))

        counts: dict[str, int] = {}
        for entry in arm:
            counts[entry["terminal_reason"]] = counts.get(entry["terminal_reason"], 0) + 1
        assert manifest[block]["counts"] == counts


@requires_manifest
def test_manifest_matches_the_published_headline() -> None:
    """13/16 and 11/16 — the numbers ``docs/results/human-trial-result.md`` reports.

    The manifest and the write-up are derived from the same session by different code, so
    this is the check that they still describe the same 32 trials.
    """
    manifest = _manifest()
    assert manifest["assisted_residual"]["counts"]["success"] == 13
    assert manifest["baseline_no_assist"]["counts"]["success"] == 11


@requires_manifest
def test_every_episode_named_by_the_manifest_is_committed() -> None:
    """The point of the ``.gitignore`` exception: the files are actually there."""
    manifest = _manifest()
    missing = [
        entry["file"]
        for entry in manifest["episodes"]
        if not (BLIND_TRIAL_CORPUS / entry["file"]).exists()
    ]
    assert not missing, f"manifest names {len(missing)} episodes that are not on disk"


@pytest.mark.skipif(
    not (BLIND_TRIAL_CORPUS / "runs").exists(), reason="the blind-trial episodes are not present"
)
def test_blind_trial_derivation_is_reproducible() -> None:
    """``--check`` must pass anywhere, not only where the episodes were recorded.

    This is the mtime regression: derive the manifest again from the committed episodes
    and require it byte-for-byte. It fails if any field is read from the filesystem
    rather than from the data or a stated constant.
    """
    from build_blind_trial_manifest import PUBLISHED_KPIS, build

    rederived = build(BLIND_TRIAL_CORPUS, REPO_ROOT / PUBLISHED_KPIS)
    assert json.dumps(rederived, indent=2) + "\n" == BLIND_TRIAL_MANIFEST.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(
    not (RECORDED_CORPUS / "runs").exists(), reason="the recorded episodes are not present"
)
def test_recorded_derivation_is_reproducible() -> None:
    """The same guard for ``data/recorded/``, whose derivation carried the bug."""
    from build_recorded_manifest import build

    rederived = build(RECORDED_CORPUS)
    assert json.dumps(rederived, indent=2) + "\n" == RECORDED_MANIFEST.read_text(encoding="utf-8")
