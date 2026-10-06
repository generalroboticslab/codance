from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import HumanSource, get_codancing_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "robot_anchor_ori_deviation",
  "robot_anchor_pos_z_low",
  "robot_foot_far_from_human_foot",
  "motion_clip_end",
]


def motion_clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Truncate the episode when the active clip ends -- a clean clip boundary
  (fresh RSI, cleared obs history) instead of the in-episode resample.

  Reads the command's per-step ``clip_just_ended`` pulse. Selecting this term
  (the ``env.terminations.motion_clip_end`` entry; null it to hand the clip end
  to ``motion_cursor.clip_end``) is the SOLE config for clip-end reset: the
  cursor resolves the term's presence and steps aside (skips its in-episode
  resample) when it is active. Pair with ``time_out=True`` -- a clip
  boundary is a natural episode end, not a failure (no penalty / value-bootstrap)."""
  command = get_codancing_command(env, command_name)
  return command.clip_just_ended()


def robot_anchor_pos_z_low(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  return command.pose_match_anchor_live_pos_w[:, 2] < threshold


def robot_anchor_ori_deviation(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  error = quat_error_magnitude(
    command.pose_match_anchor_live_quat_w, command.pose_match_anchor_ref_quat_w
  )
  return error > threshold


def robot_foot_far_from_human_foot(
  env: ManagerBasedRlEnv,
  command_name: str,
  offset_xy: tuple[float, float],
  threshold: float,
  human_source: HumanSource = "auto",
) -> torch.Tensor:
  command = get_codancing_command(env, command_name)
  target_left = command.get_human_left_foot_pos_w(human_source).clone()
  target_right = command.get_human_right_foot_pos_w(human_source).clone()
  target_left[:, 0] += offset_xy[0]
  target_left[:, 1] += offset_xy[1]
  target_right[:, 0] += offset_xy[0]
  target_right[:, 1] += offset_xy[1]
  left_dist = torch.norm(
    command.robot_left_foot_pos_w[:, :2] - target_left[:, :2], dim=-1
  )
  right_dist = torch.norm(
    command.robot_right_foot_pos_w[:, :2] - target_right[:, :2], dim=-1
  )
  return torch.logical_or(left_dist > threshold, right_dist > threshold)
