"""Flat-pool motion registry with continuous-time slicing.

One entity's clips are concatenated into a single tensor per field plus a
``length_starts`` prefix-sum offset table, the usual layout of a motion
library (the clips of one pool back to back, sliced by offset). A
per-env ``(motion_id, motion_time)`` pair is sliced by ``get_frames`` with
linear interpolation (positions / velocities / joints) and slerp (rotations),
using each clip's own ``dt = 1/fps`` so playback is fps-correct and decoupled
from the control rate. When ``fps`` equals the control rate (the codancing
default: 50 Hz npz, 50 Hz control), a query at ``t = step * control_dt`` lands
exactly on integer frame ``step`` (blend 0), so the cursor advances exactly one
frame per step, while other rates stay supported.

The registry is built FROM the per-clip ``MotionLoader``s (which also feed the
AMP expert pool); it owns the pool the per-env cursor reads.
Phase choice lives elsewhere: the registry is a pure data store (frames,
lengths, slicing); the command places the cursor and ``phase_sampler.py``
draws ``rsi`` phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.codancing.motion.loaders import MotionLoader
from mjlab.utils.lab_api.math import normalize


@dataclass
class MotionFrames:
  """Reference state at ``(motion_id, motion_time)`` -- one row per query.

  ``body_*_w`` are the RAW full body set (the consumer slices with its own body
  index sets); ``None`` when the clips carry no body arrays.
  """

  root_pos: torch.Tensor  # (N, 3)
  root_quat: torch.Tensor  # (N, 4) wxyz
  root_lin_vel: torch.Tensor  # (N, 3)
  root_ang_vel: torch.Tensor  # (N, 3)
  joint_pos: torch.Tensor  # (N, J)
  joint_vel: torch.Tensor  # (N, J)
  body_pos_w: torch.Tensor | None  # (N, B, 3)
  body_quat_w: torch.Tensor | None  # (N, B, 4)
  body_lin_vel_w: torch.Tensor | None  # (N, B, 3)
  body_ang_vel_w: torch.Tensor | None  # (N, B, 3)

  def root_state(self) -> torch.Tensor:
    """``(N, 13)`` = [pos(3), quat(4), lin_vel(3), ang_vel(3)]."""
    return torch.cat(
      [self.root_pos, self.root_quat, self.root_lin_vel, self.root_ang_vel], dim=-1
    )


def _slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
  """Batched quaternion slerp. ``q0``/``q1`` ``(M, 4)`` wxyz, ``blend`` ``(M, 1)``."""
  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)  # shortest path
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  near = sin_theta < 1e-6  # parallel -> lerp fallback
  w0 = torch.where(near, 1.0 - blend, torch.sin((1.0 - blend) * theta) / sin_theta)
  w1 = torch.where(near, blend, torch.sin(blend * theta) / sin_theta)
  return normalize(w0 * q0 + w1 * q1)


class MotionRegistry:
  """Flat pool of one entity's clips, sliced by continuous time."""

  def __init__(
    self, loaders: list[MotionLoader], control_dt: float, device: str
  ) -> None:
    if not loaders:
      raise ValueError("MotionRegistry needs at least one clip.")
    self.device = device
    self.num_clips = len(loaders)

    num_frames = torch.tensor(
      [loader.time_step_total for loader in loaders],
      dtype=torch.long,
      device=device,
    )
    self.motion_num_frames = num_frames
    starts = torch.zeros(self.num_clips, dtype=torch.long, device=device)
    if self.num_clips > 1:
      starts[1:] = torch.cumsum(num_frames, dim=0)[:-1]
    self.length_starts = starts

    default_fps = 1.0 / control_dt
    fps = torch.tensor(
      [float(loader.fps) if loader.fps else default_fps for loader in loaders],
      dtype=torch.float32,
      device=device,
    )
    self.motion_dt = 1.0 / fps
    # Clip length in seconds: a 1-frame clip (pose) has length 0.
    self.motion_lengths = (num_frames.float() - 1.0).clamp(min=0.0) * self.motion_dt

    # Flat per-field pools (concatenate every clip's frames).
    self.root_pos = torch.cat([loader.root_pos for loader in loaders], dim=0)
    self.root_quat = torch.cat([loader.root_quat for loader in loaders], dim=0)
    self.root_lin_vel = torch.cat([loader.root_lin_vel for loader in loaders], dim=0)
    self.root_ang_vel = torch.cat([loader.root_ang_vel for loader in loaders], dim=0)
    self.joint_pos = torch.cat([loader.joint_pos for loader in loaders], dim=0)
    self.joint_vel = torch.cat([loader.joint_vel for loader in loaders], dim=0)

    have_body = all(loader._body_pos_w is not None for loader in loaders)
    self._body_pos_w = (
      torch.cat([loader._body_pos_w for loader in loaders], dim=0)  # type: ignore[arg-type]
      if have_body
      else None
    )
    self._body_quat_w = (
      torch.cat([loader._body_quat_w for loader in loaders], dim=0)  # type: ignore[arg-type]
      if have_body
      else None
    )
    self._body_lin_vel_w = (
      torch.cat([loader._body_lin_vel_w for loader in loaders], dim=0)  # type: ignore[arg-type]
      if have_body
      else None
    )
    self._body_ang_vel_w = (
      torch.cat([loader._body_ang_vel_w for loader in loaders], dim=0)  # type: ignore[arg-type]
      if have_body
      else None
    )

  @property
  def has_body(self) -> bool:
    return self._body_pos_w is not None

  def _frame_blend(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(motion_id, motion_time)`` -> (global frame0, global frame1, blend)."""
    dt = self.motion_dt[motion_ids]
    num_frames = self.motion_num_frames[motion_ids]
    length = self.motion_lengths[motion_ids]
    t = motion_times.clamp(min=0.0).minimum(length)
    frame_pos = t / dt
    # De-drift: a continuous cursor advanced by `+= control_dt` accumulates
    # float error (``0.02`` is not exactly representable), so an intended
    # integer frame lands a hair off. Snap near-integer positions to the exact
    # frame -> byte-identical when fps == control rate, while genuine sub-frame
    # queries (blend well away from 0/1) still interpolate.
    nearest = frame_pos.round()
    frame_pos = torch.where((frame_pos - nearest).abs() < 1e-4, nearest, frame_pos)
    f0 = frame_pos.floor().long().clamp(max=num_frames - 1)
    f1 = (f0 + 1).clamp(max=num_frames - 1)
    blend = (frame_pos - f0.float()).clamp(0.0, 1.0).unsqueeze(-1)
    g0 = self.length_starts[motion_ids] + f0
    g1 = self.length_starts[motion_ids] + f1
    return g0, g1, blend

  def get_frames(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrames:
    """Blended reference frames for per-env ``(motion_id, motion_time)``."""
    g0, g1, blend = self._frame_blend(motion_ids, motion_times)

    def lerp(field: torch.Tensor) -> torch.Tensor:
      return (1.0 - blend) * field[g0] + blend * field[g1]

    root_quat = _slerp(self.root_quat[g0], self.root_quat[g1], blend)

    body_pos = body_quat = body_lin = body_ang = None
    if self._body_pos_w is not None:
      bb = blend.unsqueeze(1)  # (N, 1, 1) for (N, B, ·)
      assert self._body_quat_w is not None
      assert self._body_lin_vel_w is not None
      assert self._body_ang_vel_w is not None
      body_pos = (1.0 - bb) * self._body_pos_w[g0] + bb * self._body_pos_w[g1]
      body_lin = (1.0 - bb) * self._body_lin_vel_w[g0] + bb * self._body_lin_vel_w[g1]
      body_ang = (1.0 - bb) * self._body_ang_vel_w[g0] + bb * self._body_ang_vel_w[g1]
      q0 = self._body_quat_w[g0]
      q1 = self._body_quat_w[g1]
      n, b = q0.shape[0], q0.shape[1]
      body_quat = _slerp(
        q0.reshape(-1, 4),
        q1.reshape(-1, 4),
        blend.unsqueeze(1).expand(n, b, 1).reshape(-1, 1),
      ).reshape(n, b, 4)

    return MotionFrames(
      root_pos=lerp(self.root_pos),
      root_quat=root_quat,
      root_lin_vel=lerp(self.root_lin_vel),
      root_ang_vel=lerp(self.root_ang_vel),
      joint_pos=lerp(self.joint_pos),
      joint_vel=lerp(self.joint_vel),
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin,
      body_ang_vel_w=body_ang,
    )
