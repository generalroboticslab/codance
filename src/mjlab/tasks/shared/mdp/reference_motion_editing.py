"""Motion NPZ helpers behind ``prepare reverse``.

Load a clip, flip its frame order, and recompute every velocity channel from
the reversed positions and orientations (never sign-flipped), so the reversed
clip plays backward with consistent dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul


@dataclass
class _EditableMotionData:
  source_path: str
  arrays: dict[str, np.ndarray]

  @property
  def root_pos(self) -> np.ndarray:
    return self.arrays["root_pos"]

  @property
  def root_quat(self) -> np.ndarray:
    return self.arrays["root_quat"]

  @property
  def joint_pos(self) -> np.ndarray:
    return self.arrays["joint_pos"]

  def validate_required_keys(self) -> None:
    required_keys = (
      "root_pos",
      "root_quat",
      "root_lin_vel",
      "root_ang_vel",
      "joint_pos",
      "joint_vel",
    )
    for key in required_keys:
      if key not in self.arrays:
        raise ValueError(f"Motion file `{self.source_path}` is missing key `{key}`.")


def _load_motion_npz(motion_file: str) -> _EditableMotionData:
  with np.load(motion_file) as loaded_data:
    arrays = {key: np.array(value, copy=True) for key, value in loaded_data.items()}
  _populate_missing_root_arrays(arrays, motion_file)
  motion_data = _EditableMotionData(source_path=motion_file, arrays=arrays)
  motion_data.validate_required_keys()
  return motion_data


def _populate_missing_root_arrays(
  arrays: dict[str, np.ndarray], source_path: str
) -> None:
  """Fill missing `root_*` arrays from `body_*` arrays when possible."""
  has_body_pos = "body_pos_w" in arrays
  has_body_quat = "body_quat_w" in arrays
  has_body_lin_vel = "body_lin_vel_w" in arrays
  has_body_ang_vel = "body_ang_vel_w" in arrays

  if "root_pos" not in arrays:
    if not has_body_pos:
      raise ValueError(
        f"Motion file `{source_path}` is missing `root_pos` and `body_pos_w`."
      )
    arrays["root_pos"] = np.array(arrays["body_pos_w"][:, 0, :], copy=True)

  if "root_quat" not in arrays:
    if not has_body_quat:
      raise ValueError(
        f"Motion file `{source_path}` is missing `root_quat` and `body_quat_w`."
      )
    arrays["root_quat"] = np.array(arrays["body_quat_w"][:, 0, :], copy=True)

  num_frames = int(arrays["root_pos"].shape[0])
  dtype = arrays["root_pos"].dtype

  if "root_lin_vel" not in arrays:
    if has_body_lin_vel:
      arrays["root_lin_vel"] = np.array(arrays["body_lin_vel_w"][:, 0, :], copy=True)
    else:
      arrays["root_lin_vel"] = np.zeros((num_frames, 3), dtype=dtype)

  if "root_ang_vel" not in arrays:
    if has_body_ang_vel:
      arrays["root_ang_vel"] = np.array(arrays["body_ang_vel_w"][:, 0, :], copy=True)
    else:
      arrays["root_ang_vel"] = np.zeros((num_frames, 3), dtype=dtype)


def _reverse_motion_in_place(motion_data: _EditableMotionData) -> None:
  for key in (
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    if key in motion_data.arrays:
      motion_data.arrays[key] = motion_data.arrays[key][::-1].copy()


def _extract_motion_fps(motion_data: _EditableMotionData) -> float:
  fps_array = motion_data.arrays.get("fps")
  if fps_array is None:
    raise ValueError(
      f"Motion file `{motion_data.source_path}` is missing `fps`, cannot recompute derived kinematics."
    )
  fps_value = float(np.asarray(fps_array).reshape(-1)[0])
  if fps_value <= 0.0:
    raise ValueError(
      f"Motion file `{motion_data.source_path}` has invalid `fps` value `{fps_value}`."
    )
  return fps_value


def _compute_linear_velocity(values: np.ndarray, dt: float) -> np.ndarray:
  num_frames = int(values.shape[0])
  if num_frames <= 1:
    return np.zeros_like(values, dtype=np.float32)
  values_tensor = torch.as_tensor(values, dtype=torch.float32)
  velocity_tensor = torch.gradient(values_tensor, spacing=dt, dim=0)[0]
  return velocity_tensor.cpu().numpy()


def _compute_so3_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
  num_frames = int(quats_wxyz.shape[0])
  if num_frames <= 1:
    return np.zeros((num_frames, 3), dtype=np.float32)

  quat_tensor = torch.as_tensor(quats_wxyz, dtype=torch.float32)
  if num_frames == 2:
    q_rel = quat_mul(quat_tensor[1:2], quat_conjugate(quat_tensor[0:1]))
    omega = axis_angle_from_quat(q_rel) / dt
    return omega.repeat(2, 1).cpu().numpy()

  q_prev = quat_tensor[:-2]
  q_next = quat_tensor[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return omega.cpu().numpy()


def _compute_body_so3_angular_velocity(
  body_quats_wxyz: np.ndarray, dt: float
) -> np.ndarray:
  num_bodies = int(body_quats_wxyz.shape[1])
  body_ang_velocities: list[np.ndarray] = []
  for body_index in range(num_bodies):
    body_ang_velocities.append(
      _compute_so3_angular_velocity(body_quats_wxyz[:, body_index, :], dt)
    )
  return np.stack(body_ang_velocities, axis=1)


def _recompute_derived_kinematics_in_place(motion_data: _EditableMotionData) -> None:
  fps = _extract_motion_fps(motion_data)
  dt = 1.0 / fps

  motion_data.arrays["root_lin_vel"] = _compute_linear_velocity(
    motion_data.root_pos, dt
  )
  motion_data.arrays["root_ang_vel"] = _compute_so3_angular_velocity(
    motion_data.root_quat, dt
  )
  motion_data.arrays["joint_vel"] = _compute_linear_velocity(motion_data.joint_pos, dt)

  if "body_pos_w" in motion_data.arrays:
    motion_data.arrays["body_lin_vel_w"] = _compute_linear_velocity(
      motion_data.arrays["body_pos_w"], dt
    )
  if "body_quat_w" in motion_data.arrays:
    motion_data.arrays["body_ang_vel_w"] = _compute_body_so3_angular_velocity(
      motion_data.arrays["body_quat_w"], dt
    )
