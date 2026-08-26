"""``LearnedResidual`` — the trained Phase-1 residual as an ``AssistProvider``.

This is the deployment side of M5: it wraps a trained checkpoint so the learned
policy slots into the M3 seam exactly where ``NoAssist`` / ``Expert`` do, with **no
edit to the runner, input strategy, or controller** (the dependency-inversion
property the seam exists to provide). ``run_episode`` calls ``get_delta`` each tick;
the wrapper advances the GRU hidden state by one ``model.step`` and returns a
``clamp_delta``'d ``Delta``.

Two correctness details that must mirror the M4 training pipeline exactly, or the
policy sees a different input distribution than it trained on (silent covariate
shift):

1. **F/T bias subtraction.** ``data.generate`` logs ``wrist_ft - ft_bias`` where
   ``ft_bias`` is the *raw* wrist F/T captured at the episode's reset
   (``generate.py``). The runtime ``Observation.wrist_ft`` is raw, so the wrapper
   re-captures that bias on the first observation of each episode and subtracts it.
2. **Stream assembly.** The per-step command / F/T / proprioception vectors are
   built identically to ``data.dataset.extract_training_episode`` (same column
   order, same quaternion→6D map) and then z-scored with the checkpoint's stored
   train-set normalization. ``tests/test_residual_policy.py`` asserts this matches
   the loader to guard against drift.

**Per-episode reset.** The GRU hidden state and the F/T bias must reset between
episodes, but ``AssistProvider.get_delta`` has no reset signal. Rather than add a
reset hook to the shared runner (which would weaken the "drop-in, no runner edit"
guarantee), the wrapper is **self-resetting**: it watches ``observation.sim_time``
(monotonic within an episode, reset toward 0 at episode start) and clears its state
when it sees the clock jump backwards. An explicit :meth:`reset` is also exposed for
callers that prefer to be explicit (e.g. the M6 ablation orchestration).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ai_teleop.common.command import Command
from ai_teleop.common.geometry import quat_to_6d
from ai_teleop.common.log import get_logger
from ai_teleop.common.observation import Observation
from ai_teleop.data.dataset import INPUT_STREAMS, NormStats
from ai_teleop.data.images import normalize_frame
from ai_teleop.data.trajectory import SCHEMA_VERSION as DATA_SCHEMA_VERSION
from ai_teleop.domain import Delta, clamp_delta
from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.losses import LossConfig
from ai_teleop.policy.model import ResidualPolicy

# Bumped when the checkpoint payload layout changes (independent of the data schema).
POLICY_CHECKPOINT_VERSION = "1.0"

log = get_logger("residual_policy")

# A backward jump in sim_time larger than this signals a new episode (the clock is
# monotonic within an episode and resets toward 0 at reset).
_EPISODE_RESET_SIM_TIME_DROP = 1e-6

# Iterations to run before capturing the encoder's CUDA graph. Capture records the kernels
# a forward pass launches, so anything one-off (cuDNN picking algorithms, allocating its
# workspace) must already have happened or it gets baked into every replay.
_GRAPH_WARMUP_ITERATIONS = 3


class _GraphedImageEncoder:
    """The image encoder as a single CUDA-graph replay — one launch instead of ~200.

    ``mobilenet_v3_small`` at batch 1 is **launch**-bound, not compute-bound. Measured on
    an RTX 4070 Laptop: a batch of 16 frames costs the same wall-time as a batch of 1
    (7.7 ms vs 8.3 ms), because the GPU is idle between ~200 tiny kernel launches. That is
    why moving the encoder to the GPU as-is barely helps (15.5 ms on CPU → 12.2 ms on CUDA)
    and why lowering precision does not help either — there is no arithmetic to speed up.
    Capturing the forward pass into a graph replays every kernel from one launch instead:

        CPU fp32 15.5 ms | CUDA fp32 12.2 ms | **CUDA graph 0.54 ms** | CUDA graph fp16 0.90 ms

    (fp16 is *slower* than fp32 here — half precision adds conversion work to a model that
    was never arithmetic-bound.)

    A graph replays into fixed memory, so this owns one input and one output buffer for its
    lifetime: :meth:`__call__` copies the frame in and clones the embedding out. Both are
    negligible next to the ~11 ms the replay saves, and the clone is what makes the returned
    embedding safe to cache across ticks — without it the caller's cached tensor would be
    silently overwritten by the next replay.
    """

    def __init__(self, encoder: nn.Module, device: torch.device) -> None:
        self._static_input = torch.zeros(1, 3, 224, 224, device=device)

        # Capture with TF32 off. PyTorch enables it for convolutions by default, and on this
        # backbone it costs three orders of magnitude of agreement with the CPU encoder
        # (max |cpu - cuda| 7.7e-4 with TF32, 7.0e-7 without — TF32 keeps 10 mantissa bits)
        # to save 0.26 ms on one tick in twenty. That trade is backwards for a policy whose
        # eval numbers were produced elsewhere: the fast path should change where the CNN
        # runs, not what it computes. The graph records the kernels chosen here, so the
        # global is restored afterwards and training in the same process is unaffected.
        previous_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            # Warm up on a side stream — the capture API requires it, and it is what moves
            # the one-off cuDNN setup out of the recorded region.
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(_GRAPH_WARMUP_ITERATIONS):
                    encoder(self._static_input)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph), torch.no_grad():
                self._static_output = encoder(self._static_input)
        finally:
            torch.backends.cudnn.allow_tf32 = previous_tf32

    def __call__(self, image: Tensor) -> Tensor:
        self._static_input.copy_(image)
        self._graph.replay()
        return self._static_output.clone()


def best_image_encoder_device() -> str | None:
    """``"cuda"`` when this box can run the vision fast path, else ``None`` (stay on CPU).

    Live callers use this to opt in without hard-coding a device; batch callers that already
    place the whole model themselves have no need for it.
    """
    return "cuda" if torch.cuda.is_available() else None


@dataclass
class LoadedCheckpoint:
    """A checkpoint reconstructed from disk: an eval-mode model + its provenance."""

    model: ResidualPolicy
    config: PolicyConfig
    norm_stats: NormStats
    loss_config: LossConfig | None
    train_history: dict[str, list[float]] | None
    policy_checkpoint_version: str
    data_schema_version: str


def save_checkpoint(
    path: str | Path,
    *,
    model: ResidualPolicy,
    config: PolicyConfig,
    norm_stats: NormStats,
    loss_config: LossConfig | None = None,
    train_history: dict[str, list[float]] | None = None,
) -> None:
    """Serialize weights + normalization stats + hyperparameters + schema versions.

    Everything needed to rebuild a deployable policy from disk: the model weights,
    the train-set normalization (so inference normalizes identically), the model
    and loss hyperparameters, and both schema versions (to flag a stale checkpoint
    against a changed corpus / payload).
    """
    payload = {
        "policy_checkpoint_version": POLICY_CHECKPOINT_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "norm_stats": {
            "mean": {stream: norm_stats.mean[stream] for stream in INPUT_STREAMS},
            "std": {stream: norm_stats.std[stream] for stream in INPUT_STREAMS},
        },
        "loss_config": asdict(loss_config) if loss_config is not None else None,
        "train_history": train_history,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _policy_config_from_payload(payload_config: dict) -> PolicyConfig:
    """Rebuild a :class:`PolicyConfig` from a checkpoint's serialized config dict.

    Tolerates keys the current :class:`PolicyConfig` no longer defines, so retiring a
    knob does not strand every checkpoint trained before the removal. (New keys already
    round-trip: every field is defaulted precisely so an older, narrower payload loads.)
    Dropped keys are logged rather than silently swallowed — a *weight-bearing* field
    disappearing would change the model, and that should be visible in the run log.
    """
    known = {field.name for field in fields(PolicyConfig)}
    dropped = sorted(set(payload_config) - known)
    if dropped:
        log.warning(
            "checkpoint config carries retired key(s) %s — ignoring them", ", ".join(dropped)
        )
    return PolicyConfig(**{key: value for key, value in payload_config.items() if key in known})


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Rebuild a :class:`LoadedCheckpoint` (eval-mode model + provenance) from disk."""
    # weights_only=False: the payload carries our own config/stats dicts, not just
    # tensors. The file is a first-party training artifact, so this is trusted.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    config = _policy_config_from_payload(payload["config"])
    norm_stats = NormStats(mean=payload["norm_stats"]["mean"], std=payload["norm_stats"]["std"])

    model = ResidualPolicy(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    loss_config_payload = payload.get("loss_config")
    loss_config = LossConfig(**loss_config_payload) if loss_config_payload is not None else None

    return LoadedCheckpoint(
        model=model,
        config=config,
        norm_stats=norm_stats,
        loss_config=loss_config,
        train_history=payload.get("train_history"),
        policy_checkpoint_version=payload.get("policy_checkpoint_version", "unknown"),
        data_schema_version=payload.get("data_schema_version", "unknown"),
    )


class LearnedResidual:
    """Trained Phase-1 residual as a stateful, real-time ``AssistProvider``.

    Build from a checkpoint with :meth:`from_checkpoint`, or pass a model +
    normalization stats directly (handy in tests). Reused across episodes safely:
    state auto-resets on a detected ``sim_time`` restart (or call :meth:`reset`).
    """

    def __init__(
        self,
        model: ResidualPolicy,
        norm_stats: NormStats,
        *,
        device: str = "cpu",
        image_encoder_device: str | None = None,
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        # Stash normalization as device tensors so per-step z-scoring is allocation-free.
        self._mean = {stream: norm_stats.mean[stream].to(self._device) for stream in INPUT_STREAMS}
        self._std = {stream: norm_stats.std[stream].to(self._device) for stream in INPUT_STREAMS}

        # The CNN is the only part worth accelerating, and it wants the *opposite* placement
        # from the rest: it is ~30x faster as a CUDA graph, while the batch-1 GRU is ~3x
        # faster on the CPU than on CUDA (0.28 ms vs 0.77 ms — again launch overhead, on a
        # model far too small to fill a GPU). So the two halves are placed independently
        # rather than the whole policy being moved to one device.
        self._image_device = self._device
        self._encode_image: Callable[[Tensor], Tensor] | None = None
        if self._model.image_encoder is not None:
            self._setup_image_encoder(image_encoder_device)

        self._hidden: Tensor | None = None
        self._ft_bias: np.ndarray | None = None
        self._last_sim_time: float | None = None
        # Held wrist frame -> its encoder output; see :meth:`_image_embedding`.
        self._cached_frame: np.ndarray | None = None
        self._cached_embedding: Tensor | None = None

    def _setup_image_encoder(self, image_encoder_device: str | None) -> None:
        """Place the CNN and, on CUDA, capture it into a graph. Falls back, never raises.

        Every failure here is a *performance* failure, not a correctness one — the plain
        encoder still produces the same embedding — so a box without CUDA, or a driver that
        refuses capture, degrades to the CPU path with a warning instead of taking the run
        down. Graph capture is the fragile step (it needs a capture-capable stream and is the
        first thing to break under an unusual driver), so it is guarded separately from the
        device move.
        """
        assert self._model.image_encoder is not None
        encoder = self._model.image_encoder

        if image_encoder_device is None:
            self._encode_image = encoder
            return

        try:
            self._image_device = torch.device(image_encoder_device)
            encoder.to(self._image_device)
        except (RuntimeError, AssertionError) as error:
            log.warning(
                "could not place the image encoder on %s (%s) — staying on %s",
                image_encoder_device,
                error,
                self._device,
            )
            self._image_device = self._device
            encoder.to(self._device)
            self._encode_image = encoder
            return

        if self._image_device.type != "cuda":
            self._encode_image = encoder
            return

        try:
            self._encode_image = _GraphedImageEncoder(encoder, self._image_device)
            log.info("image encoder: CUDA graph captured (vision fast path)")
        except RuntimeError as error:
            # Un-graphed CUDA is only ~20% better than CPU for this backbone, but it is not
            # worse, so keep the placement and lose only the capture.
            log.warning("CUDA graph capture failed (%s) — using un-graphed CUDA", error)
            self._encode_image = encoder

    @property
    def use_vision(self) -> bool:
        """Whether this checkpoint conditions on the wrist image (Phase-2 vision).

        The eval harness reads this (duck-typed, no policy import) to decide whether
        to enable the env's wrist-camera capture for the trial.
        """
        return self._model.config.use_vision

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        image_encoder_device: str | None = None,
    ) -> LearnedResidual:
        """Load a trained checkpoint into a deployable provider.

        ``image_encoder_device`` accelerates only the vision branch — see
        :meth:`_setup_image_encoder`. Leave it ``None`` (the default) for batch work, where
        throughput comes from running many episodes at once and per-tick latency is
        irrelevant; pass :func:`best_image_encoder_device` for a live run, where it is the
        difference between hitting the 2 ms control budget and not.
        """
        loaded = load_checkpoint(path, map_location=device)
        return cls(
            loaded.model,
            loaded.norm_stats,
            device=device,
            image_encoder_device=image_encoder_device,
        )

    def reset(self) -> None:
        """Clear the GRU hidden state and F/T bias — call at episode start.

        A no-op-safe equivalent happens automatically on a detected ``sim_time``
        restart, so callers driving a single ``run_episode`` need not call this.
        """
        self._hidden = None
        self._ft_bias = None
        self._last_sim_time = None
        # A fresh episode may allocate its first frame at the address the last one ended
        # on, so identity alone could hand the new episode the old embedding.
        self._cached_frame = None
        self._cached_embedding = None

    def _is_new_episode(self, observation: Observation) -> bool:
        """True on the first call ever, or when ``sim_time`` jumps backward (reset)."""
        if self._last_sim_time is None:
            return True
        return observation.sim_time < self._last_sim_time - _EPISODE_RESET_SIM_TIME_DROP

    def _assemble_streams(
        self, observation: Observation, command: Command
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the raw per-step (command, F/T, proprioception) vectors.

        Mirrors ``data.dataset.extract_training_episode`` for a single step: same
        column order, same quaternion→6D map, and the **bias-subtracted** F/T
        (``self._ft_bias`` is captured on the episode's first observation).
        """
        assert self._ft_bias is not None  # set by get_delta before this is called

        command_vector = np.concatenate([
            command.target_position,
            quat_to_6d(command.target_quaternion),
        ])  # (9,)
        force_torque_vector = observation.wrist_ft - self._ft_bias  # (6,) bias-subtracted
        proprioception_parts: list[np.ndarray] = [
            observation.ee_pose[:3],
            quat_to_6d(observation.ee_pose[3:7]),
            observation.joint_positions,
            observation.joint_velocities,
            np.array([observation.gripper_width]),
        ]
        # error-decomposition: mirror extract_training_episode's command_ee_delta feature exactly
        # (same trailing position in the proprioception stream).
        if self._model.config.use_command_ee_delta:
            proprioception_parts.append(command.target_position - observation.ee_pose[:3])  # (3,)
        proprioception_vector = np.concatenate(proprioception_parts)  # (24,) or (27,)
        return command_vector, force_torque_vector, proprioception_vector

    def _normalized_step_tensor(self, stream: str, vector: np.ndarray) -> Tensor:
        """Z-score one stream and shape it ``(1, dim)`` for ``model.step``."""
        raw = torch.as_tensor(vector, dtype=torch.float32, device=self._device)
        return ((raw - self._mean[stream]) / self._std[stream]).unsqueeze(0)

    def _image_tensor(self, observation: Observation) -> Tensor:
        """The wrist frame as a normalized ``(1, 3, 224, 224)`` tensor for ``model.step``.

        Uses the *same* ``normalize_frame`` the training loader uses, so the encoder
        sees the channel statistics it trained on.

        ponytail: live frames are raw renders; the corpus frames went through a JPEG
        q90 round-trip. Near-lossless at 224², so no re-encode here — revisit only if
        a train/deploy gap shows up.
        """
        if observation.wrist_image is None:
            raise ValueError(
                "vision policy (use_vision=True) requires Observation.wrist_image, but it is "
                "None — enable the env's wrist capture (SimEnv.enable_wrist_capture) for this run"
            )
        return normalize_frame(observation.wrist_image).unsqueeze(0).to(self._image_device)

    def _image_embedding(self, observation: Observation) -> Tensor:
        """The wrist frame's encoder output, reusing the last one while the frame is held.

        The env renders a new wrist frame every ``render_every`` ticks (~25 Hz) and returns
        the *same array object* in between, while this runs at the 500 Hz control rate. So
        ~19 of every 20 ticks would otherwise push an identical frame through the CNN again.

        Measured on native Windows before this cache existed: the whole assist took
        15.2 ms/step and 90.9% of the loop, holding the sim at 0.12x real-time — and since
        the viewer's floor rate is counted in *sim*-seconds, that showed up as roughly
        1 fps on screen. Deployment was the only path still re-encoding; training already
        hoists the CNN out of the recurrent loop the same way (``forward``'s
        ``image_embedding``).

        Keyed on object identity, not content: the env hands back the held frame itself, so
        ``is`` is exact and free, where hashing 224x224x3 every tick would not be. A frame
        mutated in place would defeat it — nothing does that, and the cost of being wrong
        is a stale embedding for one render interval, not a crash.
        """
        frame = observation.wrist_image
        if frame is None:
            raise ValueError(
                "vision policy (use_vision=True) requires Observation.wrist_image, but it is "
                "None — enable the env's wrist capture (SimEnv.enable_wrist_capture) for this run"
            )
        if self._cached_embedding is not None and frame is self._cached_frame:
            return self._cached_embedding
        assert self._encode_image is not None
        # Own the no_grad here rather than inheriting `get_delta`'s: a cached tensor that
        # carried an autograd graph would pin it for the whole render interval.
        with torch.no_grad():
            # `.to` is a no-op when the encoder shares the GRU's device (the default).
            embedding = self._encode_image(self._image_tensor(observation)).to(self._device)
        self._cached_frame = frame
        self._cached_embedding = embedding
        return embedding

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        """Advance the policy one step and return the clamped correction Δ.

        ``command`` is the **base** operator command (pre-Δ) the seam hands in —
        exactly the ``cmd_*`` the training corpus logged.
        """
        if self._is_new_episode(observation):
            self.reset()
        if self._ft_bias is None:
            self._ft_bias = np.asarray(observation.wrist_ft, dtype=np.float64).copy()
        self._last_sim_time = observation.sim_time

        command_vector, force_torque_vector, proprioception_vector = self._assemble_streams(
            observation, command
        )
        command_tensor = self._normalized_step_tensor("command", command_vector)
        force_torque_tensor = self._normalized_step_tensor("force_torque", force_torque_vector)
        proprioception_tensor = self._normalized_step_tensor(
            "proprioception", proprioception_vector
        )
        image_embedding = self._image_embedding(observation) if self.use_vision else None

        with torch.no_grad():
            raw_delta, self._hidden = self._model.step(
                command_tensor,
                force_torque_tensor,
                proprioception_tensor,
                image_embedding=image_embedding,
                hidden=self._hidden,
            )
        delta = raw_delta.squeeze(0).cpu().numpy()  # (7,)

        return clamp_delta(
            Delta(
                delta_position=delta[0:3].astype(np.float64),
                delta_orientation=delta[3:6].astype(np.float64),
                delta_grip_force=float(delta[6]),
            )
        )
