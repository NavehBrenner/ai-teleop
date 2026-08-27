"""DAgger — close the BC imitation gap by on-policy expert relabel.

The M7 headline is blocked by a behavioral-cloning **covariate-shift** gap, not a
corpus mismatch: the clone drifts into states the privileged expert never
demonstrated and its corrections there are confidently wrong (deployed 20% vs
expert-ceiling 65% on the eval walls). DAgger is the one idea that fixes this —
**let the policy act, so it visits its own drift states, and query the expert for
the correct label at those states** — then aggregate those relabeled states into
the corpus and retrain. Batched form: rollout → relabel → aggregate → retrain,
repeated a few rounds.

This module owns the *new* mechanism only; everything else is reused:

* **Rollout + relabel** (:func:`rollout_and_relabel`) drives the shared
  ``sim.runner.run_episode`` with the learned policy as the acting ``assist`` and
  the analytical :class:`~ai_teleop.expert.Expert` as the *label provider* on
  ``data.step_callbacks.EpisodeLogger`` — so each visited state is recorded with
  the expert's correction as the BC target (the on-policy relabel). The vision
  policy's own rendered wrist frame is saved as-is (no second render).
* **Aggregation** (:func:`seed_aggregate` / :func:`append_summaries`) grows a
  dataset dir whose manifest unions the seed corpus (symlinked in, no copy) with
  each round's relabeled episodes — DAgger's data aggregation, in the exact
  on-disk schema ``data.dataset`` already loads.
* **Retrain / re-ablate** (:func:`run_dagger`) shells the aggregate through the
  existing ``train_policy`` and ``eval`` paths unchanged.

DAgger episodes roll out on a **distinct wall family** (``rollout_master_seed``,
default 105) from both the corpus (seed 82) and the held-out eval walls (seed 0),
so the eval walls stay clean and the rounds add wall diversity on top of the
on-policy states.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ai_teleop.common.log import get_logger
from ai_teleop.common.resources import safe_worker_count
from ai_teleop.control import Controller
from ai_teleop.data.generate import (
    DEFAULT_DELTA_CLAMP,
    DEFAULT_EXPERT_BRAKE_GAIN,
    DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
    DEFAULT_EXPERT_D_FAR,
    DEFAULT_FORCE_CAP,
    DEFAULT_JOINT_DAMPING,
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_MAX_DPOS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SPEED_LOGNORMAL_MEDIAN,
    DEFAULT_SPEED_LOGNORMAL_SIGMA,
    DEFAULT_SUCCESS_DEPTH,
    _episode_summary,
)
from ai_teleop.data.schema import DatasetConfig, EpisodeSpec, EpisodeSummary
from ai_teleop.data.step_callbacks import EpisodeLogger
from ai_teleop.data.trajectory import (
    SCHEMA_VERSION,
    TerminalReason,
    episode_imgs_dir,
    episode_npz_path,
)
from ai_teleop.domain.interfaces import AssistProvider
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode

log = get_logger("dagger")

_TARGET_HOLE_INDEX = 0

# Wall family for the on-policy rollouts — distinct from the corpus (82) and the
# held-out eval walls (0), so aggregation adds new walls and eval stays clean.
DEFAULT_ROLLOUT_MASTER_SEED = 105

# DAgger episode indices live far above any corpus index so they never collide
# with the symlinked seed corpus (0..N). round r, rollout i → this + r*BLOCK + i.
_DAGGER_INDEX_BASE = 1_000_000
_DAGGER_ROUND_BLOCK = 10_000


def _human_seed(master_seed: int, episode_index: int) -> int:
    """Deterministic per-episode operator seed — the shared ``(master, index)``
    derivation used by data generation and the eval harness."""
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])


def dagger_episode_index(round_index: int, rollout_index: int) -> int:
    """Collision-free episode index for a DAgger-relabeled episode."""
    return _DAGGER_INDEX_BASE + round_index * _DAGGER_ROUND_BLOCK + rollout_index


def _summaries_through_round(aggregate_dir: Path, last_round: int) -> list[EpisodeSummary]:
    """The aggregate manifest reduced to the base corpus + DAgger episodes from
    rounds ``<= last_round``. Used to resume a crashed run: rounds after the last
    *checkpointed* one may have written rollout episodes (and grown the manifest)
    without training, so those are dropped here to avoid double-counting when the
    resumed round re-rolls them (rollouts overwrite by index — see ``rollout_and_relabel``)."""
    metadata = json.loads((aggregate_dir / "metadata.json").read_text(encoding="utf-8"))
    kept: list[EpisodeSummary] = []
    for summary in metadata["episodes"]:
        index = summary["episode_index"]
        # base corpus episode, or a DAgger rollout from a round we're keeping
        is_base = index < _DAGGER_INDEX_BASE
        if is_base or (index - _DAGGER_INDEX_BASE) // _DAGGER_ROUND_BLOCK <= last_round:
            kept.append(summary)
    return kept


def expert_from_config(config: Mapping[str, Any]) -> Expert:
    """Rebuild the corpus's expert from its ``metadata.json`` config, so the
    relabels are drawn from the *same* teacher the corpus was cloned from."""
    return Expert(
        target_hole_index=_TARGET_HOLE_INDEX,
        d_far=float(config.get("expert_d_far", DEFAULT_EXPERT_D_FAR)),
        brake_gain=float(config.get("expert_brake_gain", DEFAULT_EXPERT_BRAKE_GAIN)),
        brake_lead_floor=float(
            config.get("expert_brake_lead_floor", DEFAULT_EXPERT_BRAKE_LEAD_FLOOR)
        ),
        max_delta_position=float(config.get("delta_clamp", DEFAULT_DELTA_CLAMP)),
    )


def rollout_and_relabel(
    *,
    policy: AssistProvider,
    expert: AssistProvider,
    runs_dir: Path,
    dagger_index: int,
    master_seed: int,
    rollout_index: int,
    config: Mapping[str, Any],
    render_every: int | None,
    generated_walls: bool = True,
) -> EpisodeSummary:
    """Run one on-policy rollout, relabel every visited state with ``expert``, and
    write it as ``runs_dir/episode_<dagger_index>/`` in the corpus schema.

    ``policy`` is the acting assist (the current learned residual); ``expert`` is
    the label provider whose correction becomes the BC target. With
    ``render_every`` set (an int) the env's wrist capture is enabled at that cadence
    so a vision policy can act, and the saved frames are the ones it saw; ``None``
    leaves capture off (F/T-only — no frames, for fast tests). Returns the episode's
    manifest summary.
    """
    max_dpos = float(config.get("max_dpos", DEFAULT_MAX_DPOS))
    joint_damping = float(config.get("joint_damping", DEFAULT_JOINT_DAMPING))
    max_steps = int(config.get("max_steps", DEFAULT_MAX_STEPS))
    # Expert + operator knobs this rollout runs under, read from the corpus config
    # exactly as `expert_from_config` reads them. Bound to locals because they are
    # both *used* here (the operator's speed draw) and *stamped* into the episode
    # metadata below — a DAgger episode carries the same spec as a generated one.
    expert_d_far = float(config.get("expert_d_far", DEFAULT_EXPERT_D_FAR))
    expert_brake_gain = float(config.get("expert_brake_gain", DEFAULT_EXPERT_BRAKE_GAIN))
    expert_brake_lead_floor = float(
        config.get("expert_brake_lead_floor", DEFAULT_EXPERT_BRAKE_LEAD_FLOOR)
    )
    delta_clamp = float(config.get("delta_clamp", DEFAULT_DELTA_CLAMP))
    speed_lognormal_median = float(
        config.get("speed_lognormal_median", DEFAULT_SPEED_LOGNORMAL_MEDIAN)
    )
    speed_lognormal_sigma = float(
        config.get("speed_lognormal_sigma", DEFAULT_SPEED_LOGNORMAL_SIGMA)
    )
    success_depth = float(config.get("success_depth", DEFAULT_SUCCESS_DEPTH))
    lateral_tolerance = float(config.get("lateral_tolerance", DEFAULT_LATERAL_TOLERANCE))
    force_cap = float(config.get("force_cap", DEFAULT_FORCE_CAP))

    wall_seed = episode_wall_seed(master_seed, rollout_index) if generated_walls else None
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(
            environment, max_dpos_per_step=max_dpos, joint_damping=joint_damping
        )
        home_quaternion = controller.home_pose[3:]
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        ft_bias = observation.wrist_ft.copy()

        # The vision policy needs a live wrist frame each control step; enable the
        # env's rate-limited capture and save that same frame (no second render).
        if render_every is not None:
            environment.enable_wrist_capture(render_every)

        human = ScriptedNoisyHuman(
            np.concatenate([target_position, home_quaternion]),
            seed=_human_seed(master_seed, rollout_index),
            speed_lognormal_median=speed_lognormal_median,
            speed_lognormal_sigma=speed_lognormal_sigma,
        )
        if hasattr(policy, "reset"):
            policy.reset()  # fresh GRU hidden state + F/T bias for this episode

        imgs_dir = episode_imgs_dir(runs_dir, dagger_index)
        imgs_dir.mkdir(parents=True, exist_ok=True)
        logger = EpisodeLogger(
            ft_bias,
            controller,
            target_hole_index=_TARGET_HOLE_INDEX,
            success_depth=success_depth,
            lateral_tolerance=lateral_tolerance,
            force_cap=force_cap,
            label_provider=expert,
            save_observation_frame=render_every is not None,
            imgs_dir=imgs_dir if render_every is not None else None,
            render_every=render_every or 1,
        )
        run_episode(
            environment, controller, human, policy, max_steps=max_steps, step_callback=logger
        )

        path = episode_npz_path(runs_dir, dagger_index)
        # The full corpus spec, not a subset: these episodes land in the same
        # aggregate manifest as generated ones, so they carry the same keys. Every
        # value below is what this rollout actually ran under — the expert knobs come
        # from the same `config` that built the relabeling expert (`expert_from_config`),
        # and the speed draw from the same config that seeded the operator. Before
        # G-2 this blob was an untyped dict that silently omitted `expert_d_far`
        # (required) and the five optional knobs, so every DAgger episode on disk
        # violated the declared schema.
        episode_metadata: EpisodeSpec = {
            "source": "dagger",
            "policy": "learned_residual",
            "master_seed": master_seed,
            "episode_index": dagger_index,
            "scene_seed": [master_seed, rollout_index],
            "human_seed": _human_seed(master_seed, rollout_index),
            "fingerprint": "dagger",  # rollout-derived, not seed-regenerable
            "max_dpos": max_dpos,
            "joint_damping": joint_damping,
            "expert_d_far": expert_d_far,
            "expert_brake_gain": expert_brake_gain,
            "expert_brake_lead_floor": expert_brake_lead_floor,
            "delta_clamp": delta_clamp,
            "speed_lognormal_median": speed_lognormal_median,
            "speed_lognormal_sigma": speed_lognormal_sigma,
            "target_hole_index": _TARGET_HOLE_INDEX,
            "generated_wall": generated_walls,
            "wall_seed": wall_seed,
            "terminal_reason": logger.terminal_reason.value,
            "episode_success": logger.terminal_reason is TerminalReason.SUCCESS,
            "success_depth": success_depth,
            "lateral_tolerance": lateral_tolerance,
            "force_cap": force_cap,
        }
        logger.recorder.save(path, metadata=episode_metadata)
        return _episode_summary(path, episode_metadata, n_steps=len(logger.recorder))
    finally:
        environment.close()


def _link_episode(source: Path, link: Path) -> None:
    """Point ``link`` at the episode directory ``source``, by symlink or by copy.

    A symlink is the intent — the seed corpus is read-only here, so a link costs
    nothing while a copy costs the whole corpus on disk. But **Windows refuses
    symlinks to unprivileged processes** unless Developer Mode is on, raising
    ``OSError: [WinError 1314] A required privilege is not held by the client``.
    That is an environment property, not a bad argument, so it cannot be
    pre-flighted portably — hence try-and-fall-back rather than a capability check.

    The copy is a correct substitute: the loader only ever *reads* through this
    path, so it resolves the same episode either way. It is slower and costs disk,
    which is worth one warning rather than a silent surprise on a large corpus.
    """
    if link.exists() or link.is_symlink():  # is_symlink also catches a broken link,
        return  # which exists() reports as absent and os.symlink then rejects.
    try:
        os.symlink(source, link, target_is_directory=True)
    except OSError as error:
        shutil.copytree(source, link)
        log.warning(
            "symlink unavailable (%s) — copied %s instead. This costs disk on a large "
            "corpus; on Windows, enabling Developer Mode restores the linked path.",
            error.__class__.__name__,
            link.name,
        )


def seed_aggregate(base_dir: str | Path, aggregate_dir: str | Path) -> list[EpisodeSummary]:
    """Seed the aggregate corpus from ``base_dir`` and return its episode summaries.

    The seed corpus's episode folders are **symlinked** (not copied) into
    ``aggregate/runs/`` — the loader resolves the dataset-relative ``file`` paths
    through the links, so a 300-episode image corpus costs 300 symlinks, not a
    multi-GB copy. Where symlinks are unavailable the episodes are copied instead
    (see :func:`_link_episode`), which costs disk but reads identically. The
    aggregate ``metadata.json`` starts as the base manifest;
    :func:`append_summaries` extends its ``episodes`` list per round.
    """
    base_dir = Path(base_dir)
    aggregate_dir = Path(aggregate_dir)
    aggregate_runs = aggregate_dir / "runs"
    aggregate_runs.mkdir(parents=True, exist_ok=True)

    base_metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    for summary in base_metadata["episodes"]:
        name = Path(summary["file"]).parent.name  # episode_NNNNN
        source = (base_dir / "runs" / name).resolve()
        _link_episode(source, aggregate_runs / name)

    (aggregate_dir / "metadata.json").write_text(json.dumps(base_metadata, indent=2) + "\n")
    return list(base_metadata["episodes"])


def append_summaries(
    aggregate_dir: str | Path,
    all_summaries: list[EpisodeSummary],
    *,
    config: DatasetConfig | Mapping[str, object] | None = None,
) -> None:
    """Rewrite the aggregate manifest's ``episodes`` list to ``all_summaries``.

    ``all_summaries`` is the full union (seed + every round so far), so this is
    idempotent per round. The dataset loader reads only ``episodes`` (+ counts),
    so nothing else needs to change for the retrain to see the aggregated corpus.
    """
    aggregate_dir = Path(aggregate_dir)
    metadata = json.loads((aggregate_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["episodes"] = all_summaries
    metadata["n_episodes"] = len(all_summaries)
    metadata["schema_version"] = SCHEMA_VERSION
    if config is not None:
        metadata["config"] = dict(config)
    (aggregate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


# Rollouts are the dominant cost and each one is fully determined by
# (master_seed, rollout_index) — no shared RNG, no order dependence (see
# `_human_seed` / `episode_wall_seed`) — so they parallelize with bit-identical
# results. Processes (not threads): MuJoCo + the GRU policy hold per-episode state
# that must not be shared. Spawn (not fork) so each worker builds its own CUDA
# context cleanly. Opt-in via `DAGGER_ROLLOUT_WORKERS` / `--rollout-workers`;
# default 1 keeps the original sequential path.
_WORKER: dict[str, Any] = {}


def _default_rollout_workers() -> int:
    raw = os.environ.get("DAGGER_ROLLOUT_WORKERS")
    return int(raw) if raw else 1


def _init_rollout_worker(
    checkpoint: str, config: Mapping[str, Any], render_every: int | None, device: str
) -> None:
    """Per-process setup: load the round's policy + rebuild the expert once, reused
    across every rollout this worker runs (the pool is created fresh each round)."""
    from ai_teleop.policy import LearnedResidual

    _WORKER["policy"] = LearnedResidual.from_checkpoint(Path(checkpoint), device=device)
    _WORKER["expert"] = expert_from_config(config)
    _WORKER["config"] = config
    _WORKER["render_every"] = render_every


def _rollout_task(
    round_index: int, rollout_index: int, runs_dir: str, master_seed: int
) -> tuple[int, EpisodeSummary]:
    summary = rollout_and_relabel(
        policy=_WORKER["policy"],
        expert=_WORKER["expert"],
        runs_dir=Path(runs_dir),
        dagger_index=dagger_episode_index(round_index, rollout_index),
        master_seed=master_seed,
        rollout_index=rollout_index,
        config=_WORKER["config"],
        render_every=_WORKER["render_every"],
    )
    return rollout_index, summary


def _run_round_rollouts(
    *,
    checkpoint: Path,
    config: Mapping[str, Any],
    runs_dir: Path,
    round_index: int,
    n_rollout: int,
    master_seed: int,
    render_every: int | None,
    device: str,
    workers: int,
) -> list[EpisodeSummary]:
    """The round's ``n_rollout`` on-policy rollouts, returned in ``rollout_index``
    order. ``workers <= 1`` runs the original sequential loop; more fans the
    rollouts across processes (identical results — each rollout is seed-determined)."""
    workers = safe_worker_count(workers)
    if workers <= 1:
        from ai_teleop.policy import LearnedResidual

        policy = LearnedResidual.from_checkpoint(checkpoint, device=device)
        expert = expert_from_config(config)
        summaries: list[EpisodeSummary] = []
        for rollout_index in range(n_rollout):
            summary = rollout_and_relabel(
                policy=policy,
                expert=expert,
                runs_dir=runs_dir,
                dagger_index=dagger_episode_index(round_index, rollout_index),
                master_seed=master_seed,
                rollout_index=rollout_index,
                config=config,
                render_every=render_every,
            )
            summaries.append(summary)
            log.info(
                "  rollout %3d │ %6d steps │ %s",
                rollout_index,
                summary["n_steps"],
                summary["terminal_reason"],
            )
        return summaries

    ordered: list[EpisodeSummary | None] = [None] * n_rollout
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_rollout_worker,
        initargs=(str(checkpoint), config, render_every, device),
    ) as pool:
        futures = [
            pool.submit(_rollout_task, round_index, i, str(runs_dir), master_seed)
            for i in range(n_rollout)
        ]
        for future in as_completed(futures):
            rollout_index, summary = future.result()
            ordered[rollout_index] = summary
            log.info(
                "  rollout %3d │ %6d steps │ %s",
                rollout_index,
                summary["n_steps"],
                summary["terminal_reason"],
            )
    return [summary for summary in ordered if summary is not None]


def run_dagger(
    *,
    base_dir: str | Path,
    checkpoint: str | Path,
    aggregate_dir: str | Path,
    runs_root: str | Path = "outputs/policy/runs",
    rounds: int = 1,
    n_rollout: int = 40,
    rollout_master_seed: int = DEFAULT_ROLLOUT_MASTER_SEED,
    render_every: int = 20,
    device: str = "cuda",
    epochs: int = 40,
    batch_size: int = 2,
    action_rate_weight: float = 100.0,
    eval_seeds: int = 20,
    eval_master_seed: int = 0,
    error_scale: float = 1.0,
    rollout_workers: int | None = None,
) -> list[dict[str, object]]:
    """Full batched-DAgger loop; returns a per-round result record.

    Each round: roll out ``n_rollout`` episodes of the current policy (relabeled by
    the corpus expert) onto the aggregate, retrain the frozen-encoder vision policy
    on the union at ``action_rate_weight`` (the Stage-A smoothness win), re-ablate
    on the held-out eval walls, and carry the new checkpoint into the next round.
    Reuses ``train_policy`` and the eval harness unchanged.
    """
    # Heavy / torch-only imports are lazy so the sim-only rollout path (and its
    # tests) need neither torch nor a checkpoint on disk.
    from ai_teleop.policy import LossConfig, PolicyConfig, TrainConfig
    from ai_teleop.policy.residual_policy import load_checkpoint
    from ai_teleop.policy.run_artifacts import CHECKPOINT_NAME
    from ai_teleop.policy.train import train_policy

    base_dir = Path(base_dir)
    aggregate_dir = Path(aggregate_dir)
    config = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))["config"]

    # Modality is read from the base checkpoint: an F/T base needs no wrist
    # render (much faster rounds) and must retrain F/T-only, not vision. render_every
    # is passed to the rollout only for a vision policy (None ⇒ no capture).
    use_vision = load_checkpoint(Path(checkpoint)).config.use_vision
    effective_render_every = render_every if use_vision else None

    # Crash-resume: if earlier rounds already trained a checkpoint on disk, continue
    # from the next round instead of restarting from the base (each vision round is
    # hours). The last checkpointed round's corpus state is rebuilt from the manifest;
    # a fresh run (no round checkpoints) seeds the aggregate from base as before.
    runs_root_path = Path(runs_root)
    resume_from = 0
    for round_index in reversed(range(rounds)):
        if (runs_root_path / f"dagger_round{round_index}" / CHECKPOINT_NAME).exists():
            resume_from = round_index + 1
            break
    if resume_from == 0:
        all_summaries = seed_aggregate(base_dir, aggregate_dir)
        current_checkpoint = Path(checkpoint)
    else:
        all_summaries = _summaries_through_round(aggregate_dir, resume_from - 1)
        current_checkpoint = runs_root_path / f"dagger_round{resume_from - 1}" / CHECKPOINT_NAME
        log.info(
            "resuming DAgger at round %d (rounds 0–%d already checkpointed) │ "
            "corpus %d episodes │ from %s",
            resume_from,
            resume_from - 1,
            len(all_summaries),
            current_checkpoint,
        )
    results: list[dict[str, object]] = []
    workers = _default_rollout_workers() if rollout_workers is None else rollout_workers

    for round_index in range(resume_from, rounds):
        log.info(
            "round %d │ rolling out %d episodes with %s (%d worker%s)",
            round_index,
            n_rollout,
            current_checkpoint,
            workers,
            "" if workers == 1 else "s",
        )
        round_summaries = _run_round_rollouts(
            checkpoint=current_checkpoint,
            config=config,
            runs_dir=aggregate_dir / "runs",
            round_index=round_index,
            n_rollout=n_rollout,
            master_seed=rollout_master_seed,
            render_every=effective_render_every,
            device=device,
            workers=workers,
        )
        all_summaries.extend(round_summaries)
        successes = sum(int(bool(summary["success"])) for summary in round_summaries)
        append_summaries(aggregate_dir, all_summaries, config=config)
        log.info(
            "round %d │ policy rollout success %d/%d │ aggregate now %d episodes",
            round_index,
            successes,
            n_rollout,
            len(all_summaries),
        )

        # Retrain on the aggregate. Vision rounds reuse the frozen-encoder recipe and
        # decode frames in worker processes; an F/T-only base trains F/T-only.
        trained = train_policy(
            aggregate_dir,
            config=PolicyConfig(use_vision=use_vision, freeze_image_encoder=use_vision),
            loss_config=LossConfig(weight_action_rate=action_rate_weight),
            train_config=TrainConfig(epochs=epochs),
            runs_root=runs_root,
            name=f"dagger_round{round_index}",
            batch_size=batch_size,
            num_workers=4 if use_vision else 0,
            device=device,
        )
        current_checkpoint = trained.checkpoint_path

        ablation = _reablate(
            current_checkpoint,
            seeds=eval_seeds,
            master_seed=eval_master_seed,
            error_scale=error_scale,
            device=device,
            out_dir=Path(current_checkpoint).parent,
        )
        log.info(
            "round %d │ eval @ error_scale %.2f │ human %.0f%% │ vision %.0f%%",
            round_index,
            error_scale,
            100 * ablation["human_only"],
            100 * ablation["vision"],
        )
        results.append({
            "round": round_index,
            "checkpoint": str(current_checkpoint),
            "rollout_success": successes / n_rollout if n_rollout else None,
            "aggregate_episodes": len(all_summaries),
            **ablation,
        })
    return results


def _reablate(
    checkpoint: str | Path,
    *,
    seeds: int,
    master_seed: int,
    error_scale: float,
    device: str,
    out_dir: Path | None = None,
) -> dict[str, float]:
    """Paired human-only vs vision ablation on the held-out eval walls; returns
    each config's success rate. Thin reuse of ``eval.ablation.run_paired``. When
    ``out_dir`` is given, also dumps the full per-trial KPIs (force, jerk, time —
    already computed here, else discarded) to ``out_dir/trials.csv`` so every round
    gets the same rich metrics as the final ``evaluate pair``, not just success%."""
    from ai_teleop.eval.ablation import TrialConfigSpec, run_paired_batch

    specs = [
        TrialConfigSpec(label="human_only", checkpoint=None),
        TrialConfigSpec(label="vision", checkpoint=str(checkpoint), device=device),
    ]
    successes = {spec.label: 0 for spec in specs}
    rows: list[dict[str, object]] = []
    batch = run_paired_batch(
        list(range(seeds)), specs, master_seed=master_seed, operator_error_scale=error_scale
    )
    for results in batch:
        for label, kpis in results.items():
            successes[label] += int(kpis.success)
            rows.append(kpis.to_dict())
    if out_dir is not None and rows:
        csv_path = Path(out_dir) / "trials.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {label: successes[label] / seeds for label in successes}
