"""Generated natural-language description of a session's visualization.

Drift prevention: the description is DERIVED from the same inputs the
renderer consumes (the resolved ghost flags, the resolved force-channel set,
`viz_palette`'s marker tables, the force event's response/re-anchor knobs, and
the pool's face topology), written into every play-session manifest and
printed at session start, so "which force is plotted on which body" follows the
renderer instead of being written by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mjlab.tasks.codancing import viz_palette


def _get(entry: Any, key: str, default: Any = None) -> Any:
  """Read a key from a dict entry or an attribute from an object entry (the
  motion manifest arrives as either, depending on the compose path)."""
  if isinstance(entry, dict):
    return entry.get(key, default)
  return getattr(entry, key, default)


def is_twin_pool(motions: list[Any]) -> bool:
  """True when every entry's served (robot) file IS its free file: the pool
  plays the ORIGINAL face and the two ghost sources coincide."""
  return bool(motions) and all(
    _get(m, "free_motion_file") is not None
    and _get(m, "robot_motion_file") == _get(m, "free_motion_file")
    for m in motions
  )


# What each channel draws: (quantity source, face, attach point). The drawer
# gates on the channel names; this table says what those names MEAN. It is the
# spec half of `_draw_compliance_force_arrow`: every channel in
# `viz_palette.FORCE_CHANNELS` needs a line here, or describing an explicit
# `force_channels` list raises KeyError.
MARKER_SPEC: dict[str, tuple[str, str, str]] = {
  "scripted_wrench": (
    "the force recorded in the contact NPZ",
    "served",
    "the SERVED GHOST's forced link (slot-colored wrist marks the link)",
  ),
  "applied_wrench": (
    "the sim buffer (body_external_force), the authoritative force number",
    "live",
    "the LIVE ROBOT's forced link",
  ),
  "setpoint_script": (
    "the setpoint recorded in the clip, never re-anchored",
    "served",
    "a world-frame sphere",
  ),
  "setpoint_active": (
    "the force event's stashed setpoint",
    "mode-dependent",
    "a world-frame sphere",
  ),
  "setpoint_per_frame": (
    "the setpoint re-anchored with the CURRENT anchor (viz only, physics "
    "never uses it)",
    "viz-only",
    "a world-frame sphere + spring arrow",
  ),
  "data_arrow": (
    "the would-be spring force toward the NEVER-re-anchored script setpoint "
    "(viz only, physics never uses it)",
    "viz-only",
    "an arrow from the LIVE forced link",
  ),
  "reanchor_transform": (
    "current vs frozen anchor headings (dim = frozen at event start) plus the "
    "blend base (robot XY, reference Z)",
    "both anchors",
    "heading arrows + a dot",
  ),
  "reanchor_base_per_frame": (
    "the per-frame counterpart of the frozen blend base, plus the hairlines tying it "
    "to what it carries; the gap to the frozen one IS the anchor drift since "
    "the event's rising edge",
    "viz-only",
    "a ball at (live anchor XY, reference anchor Z) + connectors",
  ),
}


@dataclass(frozen=True)
class VizInputs:
  """Everything the description depends on, resolved at session build."""

  debug_vis_enabled: bool
  robot_ref_ghost: bool
  free_ref_ghost: bool
  human_ref_ghost: bool
  robot_ref_ghost_bodies: tuple[str, ...]  # () -> the whole model
  free_ref_ghost_bodies: tuple[str, ...] | None  # None -> borrows the served
  force_channels: tuple[str, ...]
  force_style: str  # arrow | spring
  force_response: str | None  # reactive | feedforward | None (no force event)
  reanchor_mode: str | None
  twin_pool: bool  # every clip serves its FREE face (_track_original)
  slots: int | None = None  # K, when known


def _word(channel: str) -> str:
  return viz_palette.MARKER_COLOR_WORD.get(channel, "colored")


def _bodies_label(bodies: tuple[str, ...]) -> str:
  """A ghost body list as prose: () is the whole model, else a body count."""
  return "whole model" if not bodies else f"{len(bodies)} bodies"


def describe_visualization(v: VizInputs) -> str:
  """Deterministic, systematically ordered description of what renders."""
  if not v.debug_vis_enabled:
    return "debug visualization disabled: no ghosts, no force markers."
  lines: list[str] = []

  ghosts: list[str] = []
  if v.robot_ref_ghost:
    face = (
      "ORIGINAL (served == free: the pool tracks the unforced clip)"
      if v.twin_pool
      else "SERVED"
    )
    ghosts.append(
      f"robot reference ghost ({_bodies_label(v.robot_ref_ghost_bodies)}): "
      f"the {face} face, "
      "the exact reference the metric suite measures against"
    )
  if v.free_ref_ghost:
    bodies = (
      _bodies_label(v.free_ref_ghost_bodies)
      if v.free_ref_ghost_bodies is not None
      else f"{_bodies_label(v.robot_ref_ghost_bodies)} (borrowed)"
    )
    ghosts.append(
      f"free reference ghost ({bodies}): FREE joints on the SERVED root, so "
      "its gap to the served ghost is the joint-space yield (root shift "
      "understated)"
    )
  if v.human_ref_ghost:
    ghosts.append("human partner ghost: the partner's reference pose")
  lines.append(
    f"{len(ghosts)} ghost(s): " + "; ".join(ghosts) if ghosts else "no ghosts."
  )

  slot_note = f" per slot (K={v.slots})" if v.slots and v.slots > 1 else ""
  for channel in viz_palette.FORCE_CHANNELS:
    if channel not in v.force_channels:
      continue
    quantity, face, attach = MARKER_SPEC[channel]
    line = f"{channel} [{_word(channel)}]: {quantity}{slot_note}, drawn at {attach}"
    if channel == "setpoint_active" and v.reanchor_mode:
      line += (
        f"; with reanchor_mode={v.reanchor_mode} this IS the {v.reanchor_mode} face"
      )
    if channel == "applied_wrench" and v.force_response == "feedforward":
      line += (
        "; feedforward stashes no setpoint, so it renders as a plain "
        "direction arrow from the live link"
      )
    elif channel != "reanchor_transform" and face not in ("live",):
      pass  # face already stated inside quantity/attach text
    lines.append(line)
  if not any(c in v.force_channels for c in viz_palette.FORCE_CHANNELS):
    lines.append("no force markers.")

  if any(
    c in v.force_channels
    for c in ("scripted_wrench", "applied_wrench", "setpoint_per_frame")
  ):
    if v.force_style == "spring":
      lines.append(
        "style=spring: a thin cylinder from link to setpoint IS the geometry; "
        "a fixed-length arrow gives direction only."
      )
    else:
      lines.append(
        "style=arrow: arrow length is |F| x 0.02 m/N with a log knee, NOT "
        "geometry (it may overshoot its own setpoint sphere)."
      )

  if v.twin_pool:
    lines.append(
      "CAUTION: served == free on every clip (the pool tracks the unforced "
      "reference), so the free ghost coincides with the served ghost and the "
      "compliance yield reads 0."
    )
  lines.append(
    "note: recomputed force channels read 15 to 25 percent high on fast "
    "ramps (step-mid vs post-step sampling); the sim buffer is authoritative."
  )
  return "\n".join(lines)


def inputs_from_session(env_cfg: object, motions: list) -> VizInputs:
  """Resolve :class:`VizInputs` from a built env cfg + the motion manifest.

  Reads the SAME command-cfg attributes `run_exec.apply_debug_vis` writes and
  the same event params the force event will receive, so the description
  cannot disagree with the render.
  """
  from mjlab.tasks.codancing.mdp.commands import get_codancing_command_cfg

  cmd = get_codancing_command_cfg(env_cfg)
  events = getattr(env_cfg, "events", {}) or {}
  force_response = None
  reanchor_mode = None
  for term in events.values():
    params = getattr(term, "params", None) or {}
    if "force_response" in params:
      force_response = params.get("force_response")
      reanchor_mode = params.get("reanchor_mode")
      break

  twin_pool = is_twin_pool(motions)
  resolved_channels = tuple(
    sorted(
      viz_palette.expand_force_channels(
        force_arrow=getattr(cmd, "debug_vis_compliance_force_arrow", False),
        reanchor=getattr(cmd, "debug_vis_reanchor", False),
        explicit=getattr(cmd, "debug_vis_force_channels", None),
      )
    )
  )
  free_bodies = getattr(cmd, "debug_vis_free_ref_ghost_bodies", None)
  return VizInputs(
    debug_vis_enabled=bool(getattr(cmd, "debug_vis", False)),
    robot_ref_ghost=bool(getattr(cmd, "debug_vis_robot_ref_ghost", False)),
    free_ref_ghost=bool(getattr(cmd, "debug_vis_free_ref_ghost", False)),
    human_ref_ghost=bool(getattr(cmd, "debug_vis_human_ref_ghost", False)),
    robot_ref_ghost_bodies=tuple(
      getattr(cmd, "debug_vis_robot_ref_ghost_bodies", ()) or ()
    ),
    free_ref_ghost_bodies=None if free_bodies is None else tuple(free_bodies),
    force_channels=tuple(resolved_channels),
    force_style=str(getattr(cmd, "debug_vis_compliance_force_style", "arrow")),
    force_response=force_response,
    reanchor_mode=reanchor_mode,
    twin_pool=twin_pool,
  )
