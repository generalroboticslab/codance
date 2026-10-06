"""One npz motion loader for every entity (the g1 robot and the g1 human): the
same npz layout, root/joint getters plus raw body views.

``MotionLoader`` is a **pure reader** -- it does no preprocessing; the npz a run
loads is expected to already be in its final form.

Motion npz layout (per frame; ``num_joints`` is the robot's DoF count -- G1 joints
are 1-DoF, so the joint arrays are ``num_joints`` wide, not ``num_joints * 4``):
  - root_pos/root_quat/root_lin_vel/root_ang_vel (each optional; fall back to
    ``body_*_w[:, 0]`` -- body 0 is the root body)
  - joint_pos: (num_frames, num_joints)
  - joint_vel: (num_frames, num_joints)
  - body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w: (num_frames, num_bodies, ·)
    (optional; required when index-sliced views are used)
  - body_names / joint_names       -- name-based indexing (optional)

IMPORTANT (positional body ordering): the npz body axis is assumed to be MODEL
order (``Entity.body_names``, worldbody dropped), the one body index space of
the runtime. ``check_body_order`` turns that assumption into a per-clip check
wherever the npz records ``body_names``.
"""

from __future__ import annotations

import numpy as np
import torch


def check_body_order(
  npz_body_names: list[str] | None,
  expected_body_names: tuple[str, ...] | list[str],
  motion_file: str,
) -> None:
  """Assert an npz's recorded body order matches the model's body order.

  Every body index in the runtime is MODEL order (``Entity.body_names`` ==
  the npz body rows, worldbody dropped); a same-count npz exported from a
  differently-ordered model would silently pair wrong bodies everywhere.
  The check fires when the npz records ``body_names``; a file without names is
  taken to be in model order, as documented at the loader.
  """
  if npz_body_names is None:
    return
  if list(npz_body_names) != list(expected_body_names):
    raise ValueError(
      f"{motion_file}: the npz body_names disagree with the model body order; "
      "every body index assumes npz rows == model bodies (worldbody dropped). "
      f"npz: {list(npz_body_names)} model: {list(expected_body_names)}. "
      "Re-export the motion file against the current model."
    )


def _maybe_names(data, key: str) -> list[str] | None:
  return [str(n) for n in data[key]] if key in data else None


class MotionLoader:
  """Loads and provides access to one entity's motion npz (a pure reader).

  Body arrays are exposed raw, in the npz's own (model) order.
  """

  def __init__(
    self,
    motion_file: str,
    *,
    device: str = "cpu",
  ) -> None:
    data = np.load(motion_file)

    def _t(key: str) -> torch.Tensor:
      return torch.tensor(data[key], dtype=torch.float32, device=device)

    # Root state (fall back to body 0 -- the root body).
    self.root_pos = _t("root_pos") if "root_pos" in data else _t("body_pos_w")[:, 0]
    self.root_quat = _t("root_quat") if "root_quat" in data else _t("body_quat_w")[:, 0]
    self.root_lin_vel = (
      _t("root_lin_vel") if "root_lin_vel" in data else _t("body_lin_vel_w")[:, 0]
    )
    self.root_ang_vel = (
      _t("root_ang_vel") if "root_ang_vel" in data else _t("body_ang_vel_w")[:, 0]
    )

    # Joint state.
    self.joint_pos = _t("joint_pos")
    self.joint_vel = _t("joint_vel")

    # Body state (raw; sliced lazily via the index sets below).
    self._body_pos_w = _t("body_pos_w") if "body_pos_w" in data else None
    self._body_quat_w = _t("body_quat_w") if "body_quat_w" in data else None
    self._body_lin_vel_w = _t("body_lin_vel_w") if "body_lin_vel_w" in data else None
    self._body_ang_vel_w = _t("body_ang_vel_w") if "body_ang_vel_w" in data else None

    # Optional name-based indexing metadata.
    self.body_names = _maybe_names(data, "body_names")
    self.joint_names = _maybe_names(data, "joint_names")
    # Clip frame rate (Hz); drives continuous-time slicing in MotionRegistry.
    # ``None`` -> the registry falls back to the control rate (1 frame / step).
    self.fps: float | None = (
      float(data["fps"].reshape(-1)[0]) if "fps" in data else None
    )

    self.time_step_total = self.root_pos.shape[0]
    self.device = device

  # -- Root / joint accessors (the human / playback surface) ------------------

  def get_root_state(self, time_steps: torch.Tensor) -> torch.Tensor:
    """Root state ``(num_envs, 13)`` = [pos(3), quat(4), lin_vel(3), ang_vel(3)]."""
    return torch.cat(
      [
        self.root_pos[time_steps],
        self.root_quat[time_steps],
        self.root_lin_vel[time_steps],
        self.root_ang_vel[time_steps],
      ],
      dim=-1,
    )

  def get_joint_state(
    self, time_steps: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.joint_pos[time_steps], self.joint_vel[time_steps]

  # -- Body views (raw, model order) ------------------------------------------

  @property
  def body_pos_w(self) -> torch.Tensor | None:
    return self._body_pos_w

  @property
  def body_quat_w(self) -> torch.Tensor | None:
    return self._body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor | None:
    return self._body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor | None:
    return self._body_ang_vel_w
