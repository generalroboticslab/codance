"""Frozen spawn inspection: the force field at frame 0, and what the spawn is.

Training evaluates the force-field STEP event in the same env step as a reset,
on the spawn state and before the first observation (`ManagerBasedRlEnv.step`:
physics, terminations, rewards, reset, command compute, step events,
observations). The standalone `env.reset()` behind the viewer's reset key
runs no step events, so right after it the force arrows and the
external-force buffer still describe the previous episode.
`evaluate_forcefield_at_spawn` closes that gap by running exactly those terms
once; `spawn_summary` reports the spawn: clip, phase, whether the rsi coin
sampled it, the perturbation drawn, and per contact slot the scripted versus
applied force and the frozen anchor delta.
"""

from __future__ import annotations

from typing import Any

from mjlab.tasks.codancing.mdp.commands import get_codancing_command

FORCEFIELD_FUNCS = frozenset({"apply_compliance_forcefield_multi"})


def evaluate_forcefield_at_spawn(env: Any) -> list[str]:
  """Run the force-field step terms once on the current state; return their names.

  Only those terms: the other step events and the interval events (pushes)
  would move the state being inspected. The
  per-slot anchor freeze happens inside the call exactly as on training's
  reset step (the reset cleared the slot event ids), so a following env step
  keeps that anchor instead of freezing again.
  """
  env = getattr(env, "unwrapped", env)
  manager = env.event_manager
  ran: list[str] = []
  for names in manager.active_terms.values():
    for name in names:
      cfg = manager.get_term_cfg(name)
      if cfg.mode != "step":
        continue
      if getattr(cfg.func, "__name__", "") in FORCEFIELD_FUNCS:
        cfg.func(env, None, **cfg.params)
        ran.append(name)
  return ran


def spawn_summary(env: Any, env_idx: int = 0) -> dict[str, Any]:
  """The spawn of env `env_idx` as plain values (see `format_spawn`)."""
  env = getattr(env, "unwrapped", env)
  command = get_codancing_command(env)
  e = env_idx
  info: dict[str, Any] = {
    "clip": command.active_clip_names[e],
    "mode": command.cfg.motion_cursor.episode_reset,
    "t0": float(command._rsi_spawn_time[e].item()),
    "sampled": bool(command._rsi_spawn_sampled[e].item()),
    "cursor": float(command._motion_time[e].item()),
    "pose_offset": [float(v) for v in command._reset_perturb_pose[e].tolist()],
    "joint_offset_max": float(command._reset_perturb_joint_max[e].item()),
    "slots": [],
  }
  prov = command._contact_provider
  slot_bodies = command._compliance_slot_body_ids
  if prov is None or slot_bodies is None or not hasattr(prov, "contact_state_multi"):
    return info
  ids = command._motion_id[e : e + 1]
  times = command._motion_time[e : e + 1]
  cs = prov.contact_state_multi(ids, times)
  events = prov.event_indices_at(ids, times)[0]  # (K,)
  applied = command.robot.data.body_external_force[e]  # (bodies, 3)
  anchor_delta = (
    command._ff_multi_robot_anchor_pos_w[e] - command._ff_multi_ref_anchor_pos_w[e]
  ).norm(dim=-1)  # (K,)
  for k in range(int(slot_bodies.numel())):
    body = int(slot_bodies[k].item())
    active = bool(events[k].item() >= 0) and body >= 0
    info["slots"].append(
      {
        "active": active,
        "scripted_n": float(cs.force[0, k].norm().item()) if active else 0.0,
        "applied_n": float(applied[body].norm().item()) if active else 0.0,
        "anchor_delta_m": float(anchor_delta[k].item()) if active else 0.0,
      }
    )
  return info


def format_spawn(info: dict[str, Any]) -> str:
  """One line for a status row or a still's caption."""
  if info["sampled"]:
    how = "sampled"
  elif info["mode"] == "rsi":
    how = "start"
  else:
    how = str(info["mode"])
  x, y, _z, _roll, _pitch, yaw = info["pose_offset"]
  pert = ""
  if any(abs(v) > 0.0 for v in info["pose_offset"]):
    pert += f" pert dx{x:+.2f} dy{y:+.2f} yaw{yaw:+.2f}"
  if info["joint_offset_max"] > 0.0:
    pert += f" dq{info['joint_offset_max']:.2f}"
  slots = " ".join(
    (
      f"s{k}:{s['scripted_n']:.0f}/{s['applied_n']:.0f}N "
      f"d{s['anchor_delta_m'] * 100.0:.1f}cm"
      if s["active"]
      else f"s{k}:-"
    )
    for k, s in enumerate(info["slots"])
  )
  # Three decimals: the contact schedule is sampled hold-left per frame, so
  # two draws 10 ms apart can sit on different sides of an event boundary.
  line = f"{info['clip']} t={info['t0']:.3f}s ({how}){pert}"
  return f"{line} | {slots}" if slots else line
