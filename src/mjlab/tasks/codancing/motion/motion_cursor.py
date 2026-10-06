"""The motion cursor's lifecycle home (cfg + the pure helpers the command calls).

The cursor is the per-env playback state ``(motion_id, motion_time)``. Its
lifecycle is INIT (the true episode reset: what state is written at episode
start) -> ADVANCE (``motion_time += dt``, owned by the command) -> FINISH
(clip end). Control flow stays command-driven on purpose: ``mode="reset"``
events fire before the command samples the cursor and never fire on the
command's mid-episode clip-end resample, so an EventTerm cannot do
cursor-dependent state init (the mjlab tracking task reaches the same
conclusion).

BUILT here: the SOURCE (``ClipSourceCfg`` with one ``ClipCursorSampleCfg`` per
resample moment, saying WHICH clip);
ONE MODE per resample moment on :class:`MotionCursorCfg` (``episode_reset``
default | keyframe | rsi, ``clip_end`` terminate | chain), each fully
deciding that moment's state write, so a mode that samples a phase always
writes the reference at that phase and an incoherent (sampling, write) pair
cannot be written; the ``rsi`` sub-config (:class:`RsiCfg`, whose sampler
weight dict ``phase_sampler.PhaseSampler`` consumes); ONE perturbation stage
with three named ranges, applied on true episode resets only
(:func:`apply_reset_perturbation`); and the unified write primitive
``write_state(entity, EntityStateTarget, env_ids)`` every reset-time write
path goes through. Layering: clip-specific selection vocabulary lives in the
clip source; the modes are source-agnostic; algorithm routing (e.g. the
discriminator group) lives on the AMP cfg -- the cursor only exposes behavior
signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.entity import Entity

__all__ = [
  "ClipCursorSampleCfg",
  "ClipSourceCfg",
  "EntityStateTarget",
  "MotionCursorCfg",
  "RsiCfg",
  "apply_reset_perturbation",
  "validate_cursor_modes",
  "write_state",
]


@dataclass(kw_only=True)
class ClipCursorSampleCfg:
  """WHICH clip the CLIP source lands on at ONE resample moment.

  The WHERE (the phase inside the clip) lives on the mode: ``default`` and
  ``keyframe`` start at 0, ``rsi`` samples a phase, ``chain`` starts the
  incoming clip at 0.
  """

  # weighted random over the tracked pool | per-env round-robin (each env
  # loops through the tracked pool in manifest order) | env i plays clip
  # (i mod N) | pinned by manifest name.
  clip_selection: Literal["random", "alternate", "assign", "pin"] = "random"
  pin_clip: str | None = None


@dataclass(kw_only=True)
class ClipSourceCfg:
  """The recorded-clip reference source, with per-moment clip selection.

  The two blocks are independent on purpose: training draws a random clip at
  the episode reset while a sequenced medley uses
  ``clip_end.clip_selection="alternate"``; eval pins or assigns.
  """

  episode_reset: ClipCursorSampleCfg = field(default_factory=ClipCursorSampleCfg)
  clip_end: ClipCursorSampleCfg = field(default_factory=ClipCursorSampleCfg)


@dataclass(kw_only=True)
class RsiCfg:
  """Parameters of the `rsi` episode-reset mode.

  `start_prob` is the fraction of resets forced to phase 0 IN THE REFERENCE
  pose; `default_prob` the fraction that start from the robot's default
  standing pose at phase 0 (what `episode_reset: default` writes); the rest
  of the resets draw a phase from the sampler. Both compose with any sampler
  and sum to at most 1 (checked at construct). `sampler` is a WEIGHT DICT, not
  an enum: `{uniform: 1.0}` is
  plain uniform RSI, and terms add (see phase_sampler.py for the pipeline and
  the three silent-failure rules it enforces).
  `kernel_size: 1` disables the backward spread entirely (identity kernel;
  `kernel_lambda` then has no effect). Every sampler key is declared (zero =
  off) so the composed tree, which is in struct mode, takes plain overrides
  such as `...rsi.sampler.force_event=0.9`; Hydra merges dict nodes key by
  key, so an overlay re-weighting one term must restate `uniform` as well.
  """

  start_prob: float = 0.0
  default_prob: float = 0.0
  sampler: dict[str, float] = field(
    default_factory=lambda: {"uniform": 1.0, "force_event": 0.0, "failure_binned": 0.0}
  )
  kernel_lambda: float = 0.8
  kernel_size: int = 5
  ema_alpha: float = 0.001


@dataclass(kw_only=True)
class MotionCursorCfg:
  """The per-env motion cursor's whole life, on the command.

  One mode per resample moment. ``episode_reset`` decides what state a TRUE
  reset writes: ``default`` is the robot's default pose with the cursor at
  0, ``keyframe`` the reference at frame 0, ``rsi`` the reference at a
  sampled phase (a mode that samples a phase writes the reference at that
  phase, by definition). ``clip_end`` decides what a mid-episode clip end
  does: ``terminate`` writes nothing (the ``motion_clip_end`` termination
  truncates when composed; without it the cursor freezes on the last frame),
  ``chain`` keeps the robot where it stands and translates the incoming
  reference stream so its continuity body continues from where the outgoing
  clip stopped. The perturbation ranges apply on true episode resets only, after
  whatever the mode wrote; all-zero defaults leave the reset exactly as the mode
  wrote it.
  """

  source: ClipSourceCfg = field(default_factory=ClipSourceCfg)
  episode_reset: Literal["default", "keyframe", "rsi"] = "default"
  clip_end: Literal["terminate", "chain"] = "terminate"
  rsi: RsiCfg = field(default_factory=RsiCfg)
  # x y z roll pitch yaw ranges (dict shape of core's reset_root_state_uniform).
  root_pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  root_velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)


def validate_cursor_modes(cfg: MotionCursorCfg) -> None:
  """Reject unknown mode strings by name at construct time."""
  if cfg.episode_reset not in ("default", "keyframe", "rsi"):
    raise ValueError(
      f"motion_cursor.episode_reset={cfg.episode_reset!r}; use default | "
      "keyframe | rsi."
    )
  if cfg.clip_end not in ("terminate", "chain"):
    raise ValueError(f"motion_cursor.clip_end={cfg.clip_end!r}; use terminate | chain.")
  rsi = cfg.rsi
  if not (
    0.0 <= rsi.start_prob <= 1.0
    and 0.0 <= rsi.default_prob <= 1.0
    and rsi.start_prob + rsi.default_prob <= 1.0 + 1e-9
  ):
    raise ValueError(
      f"motion_cursor.rsi: start_prob={rsi.start_prob} and default_prob="
      f"{rsi.default_prob} must each lie in [0, 1] and sum to at most 1 (the "
      "remaining resets draw a phase from the sampler)."
    )


@dataclass(kw_only=True)
class EntityStateTarget:
  """One entity-state write: the fields to set (``None`` = leave untouched).

  The unified write target every reset-time path builds: the human write,
  the mode resets (``_write_reset_state``) and the reset perturbation.
  ``joint_ids`` selects a joint subset for the joint fields.
  """

  root_pose: torch.Tensor | None = None  # (N, 7) pos + quat (wxyz)
  root_vel: torch.Tensor | None = None  # (N, 6) lin + ang, world frame
  joint_pos: torch.Tensor | None = None  # (N, J) or (N, len(joint_ids))
  joint_vel: torch.Tensor | None = None  # paired with joint_pos
  joint_ids: torch.Tensor | None = None  # joint subset for the joint fields


def write_state(
  entity: "Entity",
  target: EntityStateTarget,
  env_ids: torch.Tensor,
) -> None:
  """Write ``target``'s populated fields into ``entity`` for ``env_ids``.

  Field order is load-bearing: root pose BEFORE root velocity, because the
  velocity write converts the world-frame angular velocity to the body frame
  using the quat currently in qpos (``entity/data.py::write_root_velocity``).
  Writing both fields here is therefore bitwise-identical to one
  ``write_root_state_to_sim`` call (which delegates in the same order).
  """
  if (target.joint_pos is None) != (target.joint_vel is None):
    raise ValueError(
      "EntityStateTarget.joint_pos and joint_vel must be set together (the "
      "joint write is one position+velocity sim write)."
    )
  if target.root_pose is not None:
    entity.write_root_link_pose_to_sim(target.root_pose, env_ids=env_ids)
  if target.root_vel is not None:
    entity.write_root_link_velocity_to_sim(target.root_vel, env_ids=env_ids)
  if target.joint_pos is not None and target.joint_vel is not None:
    entity.write_joint_state_to_sim(
      target.joint_pos,
      target.joint_vel,
      joint_ids=target.joint_ids,
      env_ids=env_ids,
    )


_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def apply_reset_perturbation(
  robot: "Entity",
  env_ids: torch.Tensor,
  *,
  root_pose_range: dict[str, tuple[float, float]],
  root_velocity_range: dict[str, tuple[float, float]],
  joint_position_range: tuple[float, float],
  base_root_pose: torch.Tensor | None = None,
  base_root_vel: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """Perturb the just-written reset state. TRUE episode resets only.

  Offsets add to the CURRENT (just-written) state, so the spawn stays a
  nearby plausible state whatever the mode wrote; pass `base_root_pose` /
  `base_root_vel` whenever a root write just happened (live xpos reads are
  stale until the next `sim.forward()`). The rotation delta composes in the
  local frame (euler -> quat multiply, the core events' order); joint
  offsets clamp to the soft limits like core. Pose is written BEFORE
  velocity because the velocity write converts the angular part with the
  quat currently in qpos. All-zero ranges return without a single write,
  keeping unperturbed resets byte-identical. Returns the offsets drawn, by
  name (`root_pose` (N, 6), `root_vel` (N, 6), `joint_pos` (N, J) before the
  limit clamp); empty when nothing was written.
  """
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  def _active(ranges: dict[str, tuple[float, float]]) -> bool:
    unknown = set(ranges) - set(_POSE_AXES)
    if unknown:
      raise ValueError(
        f"unknown perturbation axis {sorted(unknown)}; the ranges are keyed by "
        f"{_POSE_AXES}."
      )
    return any(lo != 0.0 or hi != 0.0 for lo, hi in ranges.values())

  jr = joint_position_range
  joints_active = jr[0] != 0.0 or jr[1] != 0.0
  out: dict[str, torch.Tensor] = {}
  if not _active(root_pose_range) and not _active(root_velocity_range):
    if not joints_active:
      return out
  device = str(robot.data.root_link_pose_w.device)
  n = len(env_ids)
  if _active(root_pose_range):
    ranges = torch.tensor(
      [root_pose_range.get(a, (0.0, 0.0)) for a in _POSE_AXES], device=device
    )
    off = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device)
    out["root_pose"] = off
    pose = (
      base_root_pose
      if base_root_pose is not None
      else robot.data.root_link_pose_w[env_ids]
    ).clone()
    pose[:, :3] += off[:, :3]
    dq = quat_from_euler_xyz(off[:, 3], off[:, 4], off[:, 5])
    pose[:, 3:7] = quat_mul(pose[:, 3:7], dq)
    write_state(robot, EntityStateTarget(root_pose=pose), env_ids)
  if _active(root_velocity_range):
    vr = torch.tensor(
      [root_velocity_range.get(a, (0.0, 0.0)) for a in _POSE_AXES], device=device
    )
    voff = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device)
    out["root_vel"] = voff
    vel = (
      base_root_vel
      if base_root_vel is not None
      else robot.data.root_link_vel_w[env_ids]
    )
    write_state(robot, EntityStateTarget(root_vel=vel + voff), env_ids)
  if joints_active:
    jp = robot.data.joint_pos[env_ids].clone()
    joff = sample_uniform(jr[0], jr[1], jp.shape, device)
    out["joint_pos"] = joff
    jp += joff
    limits = robot.data.soft_joint_pos_limits[env_ids]
    jp = jp.clamp_(limits[..., 0], limits[..., 1])
    write_state(
      robot,
      EntityStateTarget(joint_pos=jp, joint_vel=robot.data.joint_vel[env_ids]),
      env_ids,
    )
  return out
