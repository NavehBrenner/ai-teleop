"""A recorded corpus must never be regenerated over.

Generated corpora commit only ``metadata.json``; the trajectories are rebuilt on demand,
so the loader's ``download=True`` default happily simulates any episode it finds missing.
``data/recorded/`` inverts that — 54 live teleoperation episodes, no generating seed, the
files are the only copy — and it now carries a manifest of its own, which arms that same
gap-filler against it.

These tests pin the refusal, and pin that the refusal happens *before* anything is
written. The failure mode being guarded is not an exception; it is a successful-looking
run that leaves scripted episodes in the directory that held the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_teleop.data.dataset import OfflineResidualBCDataset
from ai_teleop.data.generate import GenerationConfig, NotRegenerableError, regenerate_from_metadata
from ai_teleop.data.schema import RECORDED_MASTER_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = REPO_ROOT / "data" / "recorded" / "metadata.json"


def _recorded_manifest(n_episodes: int = 2) -> dict[str, Any]:
    """A manifest in the shape ``build_recorded_manifest.py`` writes."""
    return {
        "schema_version": "2.0",
        "master_seed": RECORDED_MASTER_SEED,
        "n_episodes": n_episodes,
        "generated_at": "2026-07-03T07:15:50.242902+00:00",
        "fingerprint": None,
        "config": {
            "max_steps": 5000,
            "max_dpos": 0.3,
            "expert_d_far": None,
            "success_depth": 0.015,
            "lateral_tolerance": 0.006,
            "force_cap": 30.0,
            "scene": "generated",
            "joint_damping": 1.5,
        },
        "expert": {"counts": {}, "success_rate": None},
        "baseline_no_assist": {"counts": {"success": n_episodes}, "success_rate": 1.0},
        "episodes": [
            {
                "episode_index": index,
                "file": f"runs/episode_{index:05d}/episode.npz",
                "n_steps": 100,
                "target_hole_index": 0,
                "scene_seed": None,
                "human_seed": None,
                "wall_seed": 200 + index,
                "terminal_reason": "success",
                "success": True,
            }
            for index in range(n_episodes)
        ],
    }


def test_generation_config_refuses_a_recorded_manifest() -> None:
    """``from_metadata`` is the choke point every regeneration path runs through."""
    with pytest.raises(NotRegenerableError, match="recorded from a live operator"):
        GenerationConfig.from_metadata(_recorded_manifest())  # type: ignore[arg-type]


def test_null_expert_config_is_refused_rather_than_defaulted() -> None:
    """A null ``expert_d_far`` must not resolve to some legacy default and proceed.

    The neighbouring optional knobs *do* fall back to legacy values, which is correct for
    an old generated manifest and wrong here: there was no expert at all.
    """
    manifest = _recorded_manifest()
    manifest["master_seed"] = 0  # get past the seed guard, to reach the config guard
    with pytest.raises(NotRegenerableError, match="expert_d_far"):
        GenerationConfig.from_metadata(manifest)  # type: ignore[arg-type]


def test_regenerate_refuses_and_writes_nothing(tmp_path: Path) -> None:
    """The point of the guard: the corpus directory is untouched, not half-rebuilt."""
    corpus = tmp_path / "recorded"
    corpus.mkdir()
    (corpus / "metadata.json").write_text(json.dumps(_recorded_manifest()), encoding="utf-8")

    with pytest.raises(NotRegenerableError):
        regenerate_from_metadata(corpus / "metadata.json")

    assert not (corpus / "runs").exists(), "generation ran despite the refusal"
    assert [path.name for path in corpus.iterdir()] == ["metadata.json"]


def test_loader_refuses_missing_recorded_episodes(tmp_path: Path) -> None:
    """``download=True`` is the default, so the ordinary load path must refuse too."""
    corpus = tmp_path / "recorded"
    corpus.mkdir()
    (corpus / "metadata.json").write_text(json.dumps(_recorded_manifest()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="cannot be rebuilt"):
        OfflineResidualBCDataset(corpus)

    assert not (corpus / "runs").exists()


def test_generated_manifest_still_regenerates() -> None:
    """The guard must not fire on the corpora that legitimately rebuild themselves."""
    manifest = _recorded_manifest()
    manifest["master_seed"] = 0
    manifest["config"]["expert_d_far"] = 0.15

    config = GenerationConfig.from_metadata(manifest)  # type: ignore[arg-type]
    assert config.seed == 0
    assert config.expert_d_far == 0.15


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="the recorded corpus manifest is not present"
)
def test_committed_recorded_manifest_declares_itself_unregenerable() -> None:
    """Whatever else it says, the committed manifest must carry the sentinel.

    Its episodes are gitignored like every other corpus's, so this checks the manifest
    itself rather than the corpus — the derivation is checked by
    ``build_recorded_manifest.py --check``, which needs the episodes on disk.
    """
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["master_seed"] == RECORDED_MASTER_SEED
    assert manifest["fingerprint"] is None
    assert manifest["config"]["expert_d_far"] is None
    assert manifest["expert"]["counts"] == {}
    assert manifest["n_episodes"] == len(manifest["episodes"])
    with pytest.raises(NotRegenerableError):
        GenerationConfig.from_metadata(manifest)
