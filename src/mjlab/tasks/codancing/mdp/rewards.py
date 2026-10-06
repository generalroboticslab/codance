"""Reward functions for the codancing task: reference tracking, partner
following, compliance force matching, and contact penalties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
)

from .body_sets import model_order_indexes
from .commands import CodancingComposedCommand, get_codancing_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "partner_contact_penalty",
  "robot_right_foot_in_front_of_human_left_foot_height_aligned_exp",
  "robot_left_foot_in_front_of_human_right_foot_height_aligned_exp",
  "robot_global_ref_anchor_position_error_exp",
  "robot_global_ref_anchor_orientation_error_exp",
  "robot_relative_ref_body_position_error_exp",
  "robot_relative_ref_body_orientation_error_exp",
  "robot_global_ref_body_linear_velocity_error_exp",
  "robot_global_ref_body_angular_velocity_error_exp",
  "compliance_force_match_exp_multi",
  "compliance_torque_match_exp_multi",
  "forced_link_pos_emphasis_exp",
  "forced_link_ori_emphasis_exp",
]


def partner_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Fraction of ``sensor_name``'s slots in contact (the robot-against-partner
  sensor): positive while the robot touches the partner, zero otherwise."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.found is None:
    return torch.zeros(env.num_envs, device=env.device)

  found = data.found.view(env.num_envs, -1)
  contact_count = (found > 0).float().sum(dim=-1)
  max_contacts = found.shape[-1]

  return contact_count / max(max_contacts, 1)


def _deadband_sq_error(error: torch.Tensor, deadband: float) -> torch.Tensor:
  """Flat reward inside the deadband (no millimeter-level correction gradient);
  outside it, re-square using (dist - deadband)."""
  if deadband <= 0.0:
    return error
  dist = torch.sqrt(error + 1e-12)
  return torch.square(torch.clamp(dist - deadband, min=0.0))


def _foot_front_height_follow_exp(
  robot_foot_pos_w: torch.Tensor,
  human_foot_pos_w: torch.Tensor,
  offset_xy: tuple[float, float],
  std: float,
  deadband: float = 0.0,
  track_z: bool = True,
) -> torch.Tensor:
  target_pos_w = human_foot_pos_w.clone()
  target_pos_w[:, 0] += offset_xy[0]
  target_pos_w[:, 1] += offset_xy[1]
  error_xy = torch.sum(
    torch.square(robot_foot_pos_w[:, :2] - target_pos_w[:, :2]),
    dim=-1,
  )
  if track_z:
    error_xy = error_xy + torch.square(robot_foot_pos_w[:, 2] - human_foot_pos_w[:, 2])
  error = _deadband_sq_error(error_xy, deadband)
  return torch.exp(-error / std**2)


def robot_right_foot_in_front_of_human_left_foot_height_aligned_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  offset_xy: tuple[float, float],
  std: float = 0.3,
  deadband: float = 0.0,
  track_z: bool = True,
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  return _foot_front_height_follow_exp(
    robot_foot_pos_w=command.robot_right_foot_pos_w,
    human_foot_pos_w=command.human_left_foot_pos_w,
    offset_xy=offset_xy,
    std=std,
    deadband=deadband,
    track_z=track_z,
  )


def robot_left_foot_in_front_of_human_right_foot_height_aligned_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  offset_xy: tuple[float, float],
  std: float = 0.3,
  deadband: float = 0.0,
  track_z: bool = True,
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  return _foot_front_height_follow_exp(
    robot_foot_pos_w=command.robot_left_foot_pos_w,
    human_foot_pos_w=command.human_right_foot_pos_w,
    offset_xy=offset_xy,
    std=std,
    deadband=deadband,
    track_z=track_z,
  )


def robot_global_ref_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  error = torch.sum(
    torch.square(
      command.pose_match_anchor_ref_pos_w - command.pose_match_anchor_live_pos_w
    ),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def robot_global_ref_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  error = (
    quat_error_magnitude(
      command.pose_match_anchor_ref_quat_w, command.pose_match_anchor_live_quat_w
    )
    ** 2
  )
  return torch.exp(-error / std**2)


def robot_relative_ref_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  error = torch.sum(
    torch.square(
      command.robot_ref_body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def robot_relative_ref_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  error = (
    quat_error_magnitude(
      command.robot_ref_body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def robot_global_ref_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  error = torch.sum(
    torch.square(
      command.robot_ref_body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def robot_global_ref_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  body_indexes = model_order_indexes(body_names, command.robot.body_names)
  error = torch.sum(
    torch.square(
      command.robot_ref_body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


# ============================================================
# SoftMimic compliance force-match rewards (the training signal in reactive mode)
# ============================================================
# SoftMimic pairs the reactive force with a force/torque match reward: it matches
# the **applied** wrench (sim buffer) to the **desired/baked** wrench. It only has a
# gradient under reactive: feedforward replays the baked force verbatim, so applied
# ≡ desired and the gradient is 0; under reactive, applied = K_env·(setpoint−live),
# which equals the desired force only when the robot settles at the adapted pose,
# so this reward drives the policy to **yield to the force**. Without it the
# reactive force has no training signal. Outside events both are 0 -> exp(0)=1,
# neutral.


# ── multi-link compliance force/torque match ───────────────────────────────────────
# Masked mean over the active slots: the per-slot exp(-err) is averaged only over
# in_event slots, and a step with every slot free gives 1 (neutral, like
# exp(0) outside events), keeping the per-contact [0,1] spread.
# ``sum`` is an ablation only (total compliance work; all free -> 0, which is not
# neutral).


def _reduce_over_k(
  per_slot: torch.Tensor, mask: torch.Tensor, reduce: str
) -> torch.Tensor:
  """(N,K) per-slot reward -> (N,): masked_mean (default) / sum (ablation).
  mask = in_event."""
  m = mask.float()
  if reduce == "sum":
    return (per_slot * m).sum(dim=-1)
  denom = m.sum(dim=-1)
  num = (per_slot * m).sum(dim=-1)
  return torch.where(denom > 0, num / denom.clamp(min=1.0), torch.ones_like(num))


def _forced_link_names(
  command: CodancingComposedCommand, body_names: tuple[str, ...] | None
) -> tuple[str, ...]:
  """The links the contact script forces, from the provider's slot table.

  ``body_names`` None resolves to the provider's ``slot_link_names``, so the
  emphasis always follows the links actually being pushed when a pool changes
  its forced links. An explicit list must MATCH the provider (order-free):
  accepting a mismatch would silently emphasize links nothing is pushing
  while ignoring the ones that are.
  """
  prov = command._contact_provider
  slots = list(getattr(prov, "_slot_link_names", None) or []) if prov else []
  if not slots:
    raise ValueError(
      "forced-link emphasis needs a multi-link augmented pool: every clip's "
      "contact_file must carry slot_link_names."
    )
  if body_names is None:
    return tuple(slots)
  if sorted(body_names) != sorted(slots):
    raise ValueError(
      f"forced-link emphasis body_names {sorted(body_names)} do not match "
      f"the provider's slot links {sorted(slots)}; drop the explicit list "
      "(None follows the provider) or fix it."
    )
  return tuple(body_names)


def forced_link_pos_emphasis_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Pose-match emphasis on the FORCED links (position half).

  The same computation as ``robot_relative_ref_body_position_error_exp``,
  narrowed to the links the contact script forces (see
  ``_forced_link_names``): extra tracking weight where the wrench acts, on
  top of the general upper-body tracking group.
  """
  command = get_codancing_command(env, command_name)
  return robot_relative_ref_body_position_error_exp(
    env, command_name, std, _forced_link_names(command, body_names)
  )


def forced_link_ori_emphasis_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Pose-match emphasis on the FORCED links (orientation half)."""
  command = get_codancing_command(env, command_name)
  return robot_relative_ref_body_orientation_error_exp(
    env, command_name, std, _forced_link_names(command, body_names)
  )


def compliance_force_match_exp_multi(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  reduce: str = "masked_mean",
) -> torch.Tensor:
  """multi-link applied vs desired force match: per-slot exp(-‖Δ‖²/std²),
  masked_mean over the active slots."""
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, device=env.device)
  error = torch.sum(
    torch.square(sig.applied_force - sig.desired_force), dim=-1
  )  # (N,K)
  per_slot = torch.exp(-error / std**2)
  return _reduce_over_k(per_slot, sig.in_event, reduce)


def compliance_torque_match_exp_multi(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  reduce: str = "masked_mean",
) -> torch.Tensor:
  """multi-link applied vs desired torque match: per-slot exp(-‖Δ‖²/std²),
  masked_mean over the active slots."""
  command = get_codancing_command(env, command_name)
  sig = command.compliance_signals_multi()
  if sig is None:
    return torch.zeros(env.num_envs, device=env.device)
  error = torch.sum(
    torch.square(sig.applied_torque - sig.desired_torque), dim=-1
  )  # (N,K)
  per_slot = torch.exp(-error / std**2)
  return _reduce_over_k(per_slot, sig.in_event, reduce)
