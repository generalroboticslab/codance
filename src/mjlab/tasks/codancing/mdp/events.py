"""Event functions for the codancing task: the compliance force field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.codancing.motion.contact_reference import (
  delta_quat,
  reactive_wrench,
  reanchor_points,
  reanchor_quats,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "apply_compliance_forcefield_multi",
]


_FORCEFIELD_CHOICES = {
  "force_response": ("reactive", "feedforward"),
  "reanchor_rotation": ("yaw", "full_quat"),
  "reanchor_mode": ("rising_edge", "per_frame"),
}


def _check_forcefield_choices(**values: str) -> None:
  """Value gate for the force field's three string knobs (at the config -> code
  entry point).

  All three branches are ``if x == "a": ... else: <other branch>``, so a
  misspelled value does not raise, it **silently** falls into the other branch:
  ``reanchor_mode: perframe`` would run as rising_edge and mislabel any
  comparison of the two modes. Stopping it here makes a config error fail on the
  spot instead of producing mislabeled runs.
  """
  for key, value in values.items():
    allowed = _FORCEFIELD_CHOICES[key]
    if value not in allowed:
      raise ValueError(
        f"compliance forcefield `{key}={value!r}` is not one of {allowed}; "
        "an unrecognised value would silently pick the other branch."
      )


def apply_compliance_forcefield_multi(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  command_name: str = "codancing",
  force_scale: float = 1.0,
  force_response: str = "reactive",
  reanchor_rotation: str = "yaw",
  anchor_body: str = "",
  reanchor_mode: str = "rising_edge",
  robot_cfg: SceneEntityCfg | None = None,
) -> None:
  """STEP event: applies K SoftMimic contact wrenches to K forced links at once.

  Computes the reactive wrench per slot (re-anchored with each slot's frozen anchor), then
  **assign-scatters it into pairwise-distinct bodies** (slot links are unique by
  name and asserted pairwise distinct -> the per-slot sim-buffer readback is
  exact, with no double counting). Needs multi-link clips (a contact file with
  ``slot_link_names``)."""
  from .commands import get_codancing_command  # local: import cycle

  _check_forcefield_choices(
    force_response=force_response,
    reanchor_rotation=reanchor_rotation,
    reanchor_mode=reanchor_mode,
  )
  del env_ids  # step mode fires for every env; per-env cursors are read below
  robot: Entity = env.scene[(robot_cfg or SceneEntityCfg("robot")).name]
  command = get_codancing_command(env, command_name)
  prov = command._contact_provider
  if prov is None or getattr(prov, "_slot_link_names", None) is None:
    raise ValueError(
      "the compliance_forcefield event (apply_compliance_forcefield_multi) needs "
      "multi-link augmented clips: every clip's contact_file must carry "
      "`slot_link_names` (the multilink augmenter writes them). None do."
    )

  # Per-slot fixed link -> robot body id (resolved by name once, cache shared with
  # the command). Assert pairwise distinct (the precondition for the
  # assign-scatter). Read it into a local before the None check: same semantics,
  # but clean type narrowing (otherwise pyright thinks it may still be None).
  cached = command._compliance_slot_body_ids
  if cached is None:
    cached = prov.slot_link_robot_ids(list(robot.body_names))
    valid = cached[cached >= 0]
    assert valid.numel() == valid.unique().numel(), (
      "the forced links must map to distinct robot bodies (one contact slot per body)."
    )
    command._compliance_slot_body_ids = cached
  slot_body_ids: torch.Tensor = cached  # (K,)

  mid, mt = command._motion_id, command._motion_time
  cs = prov.contact_state_multi(mid, mt)  # (N,K,·)
  cur_events = prov.event_indices_at(mid, mt)  # (N,K)
  n, k = cur_events.shape
  body_id = slot_body_ids.view(1, -1).expand(n, k)  # (N,K)
  in_event = (cur_events >= 0) & (body_id >= 0)  # (N,K)

  # Per-slot rising-edge freeze of the robot vs reference anchors (per link);
  # the read side goes through the unified multi-link view.
  command.update_forcefield_anchor_multi(
    cur_events, anchor_body, reanchor_rotation, reanchor_mode
  )
  r_pos, r_quat, ref_pos, ref_quat = command.frozen_anchor_slots()

  if force_response == "reactive":
    ar = torch.arange(n, device=env.device).view(-1, 1)  # (N,1)
    bidx = body_id.clamp(min=0)  # (N,K)
    live_pos = robot.data.body_link_pos_w[ar, bidx]  # (N,K,3)
    live_quat = robot.data.body_link_quat_w[ar, bidx]  # (N,K,4)
    # setpoint is in the clip frame (no origin); the frozen ref anchor is in world
    # -> add the reference origin (incl. the chain offset) before re-anchoring.
    set_pos_w = reanchor_points(
      cs.setpoint_pos + command._ref_origins.unsqueeze(1),
      r_pos,
      r_quat,
      ref_pos,
      ref_quat,
      reanchor_rotation,
    )
    set_quat_w = reanchor_quats(cs.setpoint_quat, r_quat, ref_quat, reanchor_rotation)
    # Stash per slot: the visualization draws exactly this copy per slot (K slots
    # are not recomputed either).
    command.stash_forcefield_setpoint(set_pos_w)
    f_w, tau_w = reactive_wrench(
      live_pos,
      live_quat,
      set_pos_w,
      set_quat_w,
      cs.forcefield_stiffness,
      cs.forcefield_rot_stiffness,
    )
  else:  # feedforward: replay the baked wrench, frozen-yaw re-anchor (open loop).
    yd = delta_quat(r_quat, ref_quat, reanchor_rotation)
    f_w = quat_apply(yd, cs.force)
    tau_w = quat_apply(yd, cs.torque)

  n_body = robot.data.body_link_pos_w.shape[1]
  forces = torch.zeros(env.num_envs, n_body, 3, device=env.device)
  torques = torch.zeros(env.num_envs, n_body, 3, device=env.device)
  # Per-slot assign-scatter (K is small); slot links are pairwise distinct -> no
  # body collisions, so plain assignment is exact.
  for s in range(k):
    rows = torch.nonzero(in_event[:, s], as_tuple=False).flatten()
    if rows.numel() > 0:
      cols = body_id[rows, s]
      forces[rows, cols] = f_w[rows, s] * force_scale
      torques[rows, cols] = tau_w[rows, s] * force_scale
  robot.write_external_wrench_to_sim(forces, torques)
