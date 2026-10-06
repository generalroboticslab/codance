"""Observation functions for the codancing task: the robot reference, the
robot's own body state, the partner, and the compliant-contact channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .body_sets import model_order_indexes
from .commands import CodancingComposedCommand, HumanSource, get_codancing_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "human_body_pos_b",
  "robot_ref_anchor_pos_b",
  "robot_ref_anchor_ori_b",
  "robot_ref_joint_state",
  "robot_body_ori_b",
  "robot_body_pos_b",
  "active_clip_obs",
  "desired_stiffness_log_multi",
  "forcefield_stiffness_log_multi",
  "compliance_applied_wrench_multi",
  "compliance_desired_wrench_multi",
]


def robot_ref_anchor_pos_b(
  env: ManagerBasedRlEnv, command_name: str, ref_face: str = "served"
) -> torch.Tensor:
  """Reference anchor position in the robot anchor frame; ``ref_face`` selects
  the face to read (served / free)."""
  command = get_codancing_command(env, command_name)
  ref_pos, ref_quat = command.pose_match_anchor_pose_w_for(ref_face)

  pos, _ = subtract_frame_transforms(
    command.pose_match_anchor_live_pos_w,
    command.pose_match_anchor_live_quat_w,
    ref_pos,
    ref_quat,
  )

  return pos.view(env.num_envs, -1)


def robot_ref_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str, ref_face: str = "served"
) -> torch.Tensor:
  """Reference anchor orientation in the robot anchor frame; ``ref_face`` selects
  the face to read (served / free)."""
  command = get_codancing_command(env, command_name)
  ref_pos, ref_quat = command.pose_match_anchor_pose_w_for(ref_face)

  _, ori = subtract_frame_transforms(
    command.pose_match_anchor_live_pos_w,
    command.pose_match_anchor_live_quat_w,
    ref_pos,
    ref_quat,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_ref_joint_state(
  env: ManagerBasedRlEnv,
  command_name: str,
  ref_face: str = "served",
  include_velocity: bool = True,
  body_set: str | None = None,
) -> torch.Tensor:
  """Reference joint state (pos + vel).

  ``ref_face`` selects the face to read: "served" (default) = the tracking target
  the provider serves (the adapted face for an augmented pool); "free" = the
  original un-yielded face (SoftMimic's actor view, augmented pools only). Rewards
  and the critic always track the served face.

  ``include_velocity=False`` emits joint positions only (width J instead of 2J).
  Meant for the adapted face: the augmented joint velocities are finite
  differences of per-frame IK solutions, and their measured second difference
  reaches 2 to 4 times the original face's at p99.9 (worst at the reset boundary
  at event ends); SoftMimic never observes either face's joint velocities either.
  The free face's velocities come straight from the original clip, are clean, and
  are kept by default.

  ``body_set`` selects the scope: None (default) = the full joint vector; a
  registry set name narrows it to the joints **owned** by that set's bodies (a
  sparse set gives sparse joints, a set listing every body of a region gives all
  of its joints). The scope is thus spelled out in this one parameter, so the
  config reads clearly."""
  command = get_codancing_command(env, command_name)
  if not include_velocity:
    return command.robot_ref_joint_pos_for(ref_face, body_set)
  return command.robot_ref_joint_state_for(ref_face, body_set)


def robot_body_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  num_bodies = len(body_indexes)
  pos_b, _ = subtract_frame_transforms(
    command.pose_match_anchor_live_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.pose_match_anchor_live_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  num_bodies = len(body_indexes)
  _, ori_b = subtract_frame_transforms(
    command.pose_match_anchor_live_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.pose_match_anchor_live_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _robot_body_pose_w(
  command: CodancingComposedCommand, body_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """One robot body's live world pose (MODEL-order lookup in the entity)."""
  idx = command.robot.body_names.index(body_name)
  return (
    command.robot.data.body_link_pos_w[:, idx],
    command.robot.data.body_link_quat_w[:, idx],
  )


def human_body_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
  human_source: HumanSource = "auto",
  base_body_name: str = "torso_link",
) -> torch.Tensor:
  """Human body positions projected into a NAMED robot link's frame.

  ``base_body_name`` selects the robot link the partner observation is
  expressed in, independent of the pose-match anchor: both default to the
  torso, but moving the tracking anchor must not silently move the
  partner-observation frame (nor the other way round).
  """
  command = get_codancing_command(env, command_name)
  human_body_names = command.human_body_names
  body_indices = [human_body_names.index(name) for name in body_names]
  num_bodies = len(body_indices)
  base_pos, base_quat = _robot_body_pose_w(command, base_body_name)
  pos_b, _ = subtract_frame_transforms(
    base_pos[:, None, :].repeat(1, num_bodies, 1),
    base_quat[:, None, :].repeat(1, num_bodies, 1),
    command.get_human_body_pos_w(body_indices, human_source),
    command.get_human_body_quat_w(body_indices, human_source),
  )
  return pos_b.view(env.num_envs, -1)


def active_clip_obs(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Active motion pair index as observation (one-hot or scalar).

  Returns the output of ``CodancingComposedCommand.active_clip_obs``.
  When ``active_clip_obs_mode`` is ``"none"`` this should not be registered
  as an observation term -- the env config is responsible for gating this.
  """
  command = get_codancing_command(env, command_name)
  obs = command.active_clip_obs
  if obs is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  return obs


# ── SoftMimic compliant-contact observations ───────────────────────────────────
# SoftMimic's core asymmetry (arXiv:2510.17792): the actor observes the
# **original** reference + the **desired stiffness**, but the reward targets the
# **adapted** target, which forces the policy to **infer** the external force from
# its proprioceptive history instead of being fed the adapted target directly.
# Hence:
#   * the actor only adds `desired_stiffness_log_multi` (same phase + different K ->
#     different yield, so without K it is ambiguous); it observes neither the
#     applied force (implicit compliance) nor the adapted per-body targets.
#     Proprioceptive history comes from the native history_length.
#   * critic privileged: applied + desired force/torque (world frame).


# ── multi-link compliance observations ─────────────────────────────────────────────
# One block per contact slot, so the width grows K-fold. Flattened **slot-major,
# component-minor** (flatten(1) after stack/cat): K=1 gives actor 2 / critic 6,
# K=2 gives actor 4 / critic 12 (per term).


def _multi_k(command) -> int:
  """The provider's slot count K (single-contact provider / no provider -> 1)."""
  return int(getattr(command._contact_provider, "_K", 1))


def _slot_of_side(command, side: str) -> int:
  """Index of the ``side`` (left / right) wrist among the K slots.

  Resolved from the provider's slot names without assuming a slot order: the slot
  names come from the contact NPZ's ``slot_link_names``, and a different rc can
  change their order or count. Raises unless exactly one slot matches, instead of
  falling back to slot 0: when the two wrists get different stiffnesses, silently
  setting it on the other hand gives results that look perfectly normal, only the
  data is wrong.
  """
  names = list(getattr(command._contact_provider, "_slot_link_names", None) or [])
  hits = [i for i, name in enumerate(names) if side in name]
  if len(hits) != 1:
    raise ValueError(
      f"per-hand desired_stiffness needs exactly one '{side}' slot, "
      f"found {len(hits)} in {names}; drop the per-hand knob and use "
      "fixed_lin / fixed_rot, which apply to every slot."
    )
  return hits[0]


def desired_stiffness_log_multi(
  env: ManagerBasedRlEnv,
  command_name: str = "codancing",
  fixed_lin: float | None = None,
  fixed_rot: float | None = None,
  fixed_lin_left: float | None = None,
  fixed_rot_left: float | None = None,
  fixed_lin_right: float | None = None,
  fixed_rot_right: float | None = None,
) -> torch.Tensor:
  """Actor observation: per-slot log stiffness [log(K_rob), log(K_rot_rob)],
  flattened (N,2K).

  SoftMimic semantics: the free frames are nonzero per slot (between events the
  next event's K, after the last event a placeholder 140, zerowrench a
  held-then-resampled random value), so the observation carries no contact
  timing; the +1e-6 guards log(0).

  The stiffness can be pinned to a constant at eval time, matching deployment: on
  the robot, desired_stiffness is a fixed value from the config, and
  the clip's K schedule belongs to training only. Pinning it at eval gives the
  yield under a hand drag a definite K to compare against. Two tiers of knobs,
  both taking plain stiffness values (N/m and N·m/rad), both None = read the clip
  schedule as is:

  * ``fixed_lin`` / ``fixed_rot``: global, the same value for every slot.
  * ``fixed_lin_left`` / ``fixed_rot_left`` / ``fixed_lin_right`` / ``fixed_rot_right``:
    per hand, overriding the global value of the same channel. Giving the two
    wrists different stiffnesses makes one hand the control: the soft/stiff
    difference shows within a single drag, removing every variable between two
    separate runs.
  """
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, 2 * _multi_k(command), device=env.device)
  eps = 1e-6
  k_lin, k_rot = sig.robot_stiffness, sig.robot_rot_stiffness
  if fixed_lin is not None:
    k_lin = torch.full_like(k_lin, fixed_lin)
  if fixed_rot is not None:
    k_rot = torch.full_like(k_rot, fixed_rot)
  per_hand = (
    ("left", fixed_lin_left, fixed_rot_left),
    ("right", fixed_lin_right, fixed_rot_right),
  )
  if any(v is not None for _, lin, rot in per_hand for v in (lin, rot)):
    # k_lin/k_rot may still be the provider's cached tensors here; editing them in
    # place would corrupt other readers in the same step.
    k_lin, k_rot = k_lin.clone(), k_rot.clone()
    for side, lin, rot in per_hand:
      if lin is None and rot is None:
        continue
      slot = _slot_of_side(command, side)
      if lin is not None:
        k_lin[:, slot] = lin
      if rot is not None:
        k_rot[:, slot] = rot
  stacked = torch.stack(
    [
      torch.log(k_lin + eps),
      torch.log(k_rot + eps),
    ],
    dim=-1,
  )  # (N,K,2)
  return stacked.flatten(1)  # (N,2K) slot-major, component-minor


def forcefield_stiffness_log_multi(
  env: ManagerBasedRlEnv, command_name: str = "codancing"
) -> torch.Tensor:
  """Critic privileged observation: per-slot log **environment** stiffness
  [log(K_env), log(K_rot_env)], flattened (N,2K).

  Paired with the actor's ``desired_stiffness_log_multi`` (the robot's commanded
  K): that one is "how much compliance was requested", this one is "how stiff a
  spring the robot is pushing against". SoftMimic's critic sees both.

  **Critic only.** Unlike the robot K channel, K_env is zero outside events by
  design (measured exactly 0.0 outside events), so log(K+eps) is constantly
  log(1e-6) ≈ -13.8 on free frames and jumps to log(K) inside events: exactly the
  contact-timing cue that was deliberately taken away from the actor. The critic
  is privileged and never deployed, so it is harmless there; to feed this to the
  actor, it must first be changed to hold / next-event fill like the robot K."""
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, 2 * _multi_k(command), device=env.device)
  eps = 1e-6
  stacked = torch.stack(
    [
      torch.log(sig.forcefield_stiffness + eps),
      torch.log(sig.forcefield_rot_stiffness + eps),
    ],
    dim=-1,
  )  # (N,K,2)
  return stacked.flatten(1)  # (N,2K) slot-major, component-minor


def compliance_applied_wrench_multi(
  env: ManagerBasedRlEnv, command_name: str = "codancing"
) -> torch.Tensor:
  """Critic privileged observation: per-slot **applied** wrench
  [force(3), torque(3)], flattened (N,6K)."""
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, 6 * _multi_k(command), device=env.device)
  return torch.cat([sig.applied_force, sig.applied_torque], dim=-1).flatten(1)


def compliance_desired_wrench_multi(
  env: ManagerBasedRlEnv, command_name: str = "codancing"
) -> torch.Tensor:
  """Critic privileged observation: per-slot **desired** (scripted) wrench
  [force(3), torque(3)], flattened (N,6K)."""
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, 6 * _multi_k(command), device=env.device)
  return torch.cat([sig.desired_force, sig.desired_torque], dim=-1).flatten(1)
