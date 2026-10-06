from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.tasks.codancing.mdp.body_sets import body_set_indices
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.tasks.codancing.mdp.commands import CodancingComposedCommand

MetricReferenceFrame = Literal["global", "anchor_aligned", "both"]
MetricSuiteName = Literal["robot_ref_tracking", "compliance", "partner", "clip_drag"]
# A comparison pair (a, b): each side is one of three pose faces. Every pair
# records under the explicit `robot_ref/{a}_vs_{b}/{group}/...` namespace.

_COMPLIANCE_METRIC_KEYS: tuple[str, ...] = (
  "compliance/applied_force_mag",
  "compliance/desired_force_mag",
  "compliance/force_error",
)

# Per contact slot, what a clip's own force event says about the realized
# stiffness (see compute_clip_drag_metrics). Two slots: the multilink pools'
# left and right wrist; a one-slot pool leaves slot 1 at zero.
_CLIP_DRAG_SLOTS = 2
_CLIP_DRAG_FIELDS = (
  "in_event",
  "phase",
  "told",
  "kenv",
  "force",
  "yield_free",
  "yield_served",
)
_CLIP_DRAG_KEYS: tuple[str, ...] = tuple(
  f"clip_drag/k{i}_{field}"
  for i in range(_CLIP_DRAG_SLOTS)
  for field in _CLIP_DRAG_FIELDS
)
# Partner-relative scalars (paired pools; zeros without a human stream).
_PARTNER_METRIC_KEYS: tuple[str, ...] = (
  "partner/foot_follow_xy",
  "partner/foot_follow_xy_left",
  "partner/foot_follow_xy_right",
  "partner/pelvis_dist",
  "partner/pelvis_dist_xy",
)


def log_safe_metric_name(metric_name: str) -> str:
  return metric_name.replace("/", "_")


def iter_metric_suite_keys(
  suite_names: tuple[str, ...],
  reference_frame: MetricReferenceFrame,
  body_groups: tuple[str, ...],
  compare: tuple[tuple[str, str], ...] = (("served", "live"),),
) -> tuple[str, ...]:
  keys: list[str] = []
  frames = _expand_reference_frames(reference_frame)
  for suite_name in suite_names:
    if suite_name == "compliance":
      keys.extend(_COMPLIANCE_METRIC_KEYS)
      continue
    if suite_name == "partner":
      keys.extend(_PARTNER_METRIC_KEYS)
      continue
    if suite_name == "clip_drag":
      keys.extend(_CLIP_DRAG_KEYS)
      continue
    if suite_name != "robot_ref_tracking":
      raise ValueError(f"Unsupported codancing metric suite: {suite_name!r}")
    for pair in compare:
      _validate_pair(pair)
      for body_group in body_groups:
        for frame in frames:
          if frame == "anchor_aligned" and not _pair_has_relative_buffers(pair):
            continue
          keys.extend(_robot_ref_tracking_keys(body_group, frame, (pair[0], pair[1])))
  return tuple(keys)


def compute_metric_suite(
  command: CodancingComposedCommand,
  suite_names: tuple[str, ...],
  reference_frame: MetricReferenceFrame,
  body_groups: tuple[str, ...],
  compare: tuple[tuple[str, str], ...] = (("served", "live"),),
) -> dict[str, torch.Tensor]:
  metrics: dict[str, torch.Tensor] = {}
  for suite_name in suite_names:
    if suite_name == "compliance":
      metrics.update(compute_compliance_metrics(command))
      continue
    if suite_name == "partner":
      metrics.update(compute_partner_metrics(command))
      continue
    if suite_name == "clip_drag":
      metrics.update(compute_clip_drag_metrics(command))
      continue
    if suite_name != "robot_ref_tracking":
      raise ValueError(f"Unsupported codancing metric suite: {suite_name!r}")
    metrics.update(
      compute_robot_ref_tracking_metrics(
        command=command,
        reference_frame=reference_frame,
        body_groups=body_groups,
        compare=compare,
      )
    )
  return metrics


def _pair_has_relative_buffers(pair: tuple[str, str]) -> bool:
  """anchor_aligned needs precomputed relative buffers: they exist for the
  served face (always; rewards read them) and the free face (computed when a
  [free, live] pair is configured on an augmented pool). The live side is
  always the alignment destination, so other pairs stay global-frame."""
  return tuple(pair) in (("served", "live"), ("free", "live"))


def _validate_pair(pair: tuple[str, str]) -> None:
  a, b = pair
  for face in (a, b):
    if face not in ("live", "served", "free"):
      raise ValueError(
        f"Unsupported metric face {face!r} in pair {pair!r}; "
        "use 'live', 'served' or 'free'."
      )
  if a == b:
    raise ValueError(f"Metric compare pair {pair!r} compares a face to itself.")


def _face_body_frames(
  command: CodancingComposedCommand, face: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """World-frame (pos, quat) body tensors for one face."""
  if face == "live":
    return command.robot_body_pos_w, command.robot_body_quat_w
  if face == "served":
    return command.robot_ref_body_pos_w, command.robot_ref_body_quat_w
  if face == "free":
    gather = command._gather_robot_motion_tensor
    pos = gather("body_pos_w", face="free") + command._ref_origins[:, None, :]
    return pos, gather("body_quat_w", face="free")
  raise ValueError(f"Unsupported metric face: {face!r}")


def compute_robot_ref_tracking_metrics(
  command: CodancingComposedCommand,
  reference_frame: MetricReferenceFrame,
  body_groups: tuple[str, ...],
  compare: tuple[tuple[str, str], ...] = (("served", "live"),),
) -> dict[str, torch.Tensor]:
  metrics: dict[str, torch.Tensor] = {}
  for pair in compare:
    _validate_pair(pair)
    a, b = pair
    for body_group in body_groups:
      body_indexes = list(
        body_set_indices(command.cfg, body_group, command.robot.body_names)
      )
      for frame in _expand_reference_frames(reference_frame):
        if frame == "anchor_aligned" and not _pair_has_relative_buffers(pair):
          continue
        prefix = f"robot_ref/{a}_vs_{b}/{body_group}/{frame}"
        if frame == "global":
          pos_a, quat_a = _face_body_frames(command, a)
          pos_b, quat_b = _face_body_frames(command, b)
          metrics[f"{prefix}_pos"] = torch.norm(
            pos_a[:, body_indexes] - pos_b[:, body_indexes], dim=-1
          ).mean(dim=-1)
          metrics[f"{prefix}_rot"] = quat_error_magnitude(
            quat_a[:, body_indexes], quat_b[:, body_indexes]
          ).mean(dim=-1)
        elif frame == "anchor_aligned":
          if a == "served":
            rel_pos = command.robot_ref_body_pos_relative_w
            rel_quat = command.robot_ref_body_quat_relative_w
          else:  # ("free", "live"): the free face's own relative buffers.
            rel_pos = command.robot_free_body_pos_relative_w
            rel_quat = command.robot_free_body_quat_relative_w
            assert rel_pos is not None and rel_quat is not None, (
              "[free, live] anchor_aligned needs the free reference's relative "
              "buffers: augmented pool (free_motion_file + contact_file) only."
            )
          metrics[f"{prefix}_pos"] = torch.norm(
            rel_pos[:, body_indexes] - command.robot_body_pos_w[:, body_indexes],
            dim=-1,
          ).mean(dim=-1)
          metrics[f"{prefix}_rot"] = quat_error_magnitude(
            rel_quat[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
          ).mean(dim=-1)
        else:
          raise ValueError(f"Unsupported metric reference frame: {frame!r}")
  return metrics


def compute_compliance_metrics(
  command: CodancingComposedCommand,
) -> dict[str, torch.Tensor]:
  """Per-env instantaneous compliance scalars of the K contact slots, snapshot per
  step into ``command.metrics`` so the eval CSV gets
  per-clip-bucketable columns. The force means are masked over the active
  slots, so the zero force on free frames never dilutes them.
  Reads the same buffers as the force arrows, so the numbers agree."""
  num_envs = command._motion_id.shape[0]
  sig = command.compliance_signals_multi()
  if sig is not None:
    return {
      "compliance/applied_force_mag": _masked_mean_mag(sig.applied_force, sig.in_event),
      "compliance/desired_force_mag": _masked_mean_mag(sig.desired_force, sig.in_event),
      "compliance/force_error": _masked_mean_mag(
        sig.applied_force - sig.desired_force, sig.in_event
      ),
    }
  device = command._motion_id.device
  zeros = torch.zeros(num_envs, device=device)
  return {key: zeros.clone() for key in _COMPLIANCE_METRIC_KEYS}


def _foot_follow_offset_xy(command: CodancingComposedCommand) -> tuple[float, float]:
  """The XY offset the foot-follow reward puts between a robot foot and the
  partner's crossed foot, read from that reward term when the env has one
  (the paired policy configs use (-0.5, 0)), else that value."""
  rewards = getattr(getattr(command._env, "cfg", None), "rewards", None) or {}
  for name in ("robot_left_foot_front", "robot_right_foot_front"):
    term = rewards.get(name) if isinstance(rewards, dict) else None
    params = getattr(term, "params", None)
    if params and "offset_xy" in params:
      x, y = params["offset_xy"]
      return float(x), float(y)
  return (-0.5, 0.0)


def compute_clip_drag_metrics(
  command: CodancingComposedCommand,
) -> dict[str, torch.Tensor]:
  """Per env and contact slot, the clip's own force event read as a drag: the
  applied force magnitude, and the slot link's displacement along that force
  away from the FREE reference (the un-adapted trajectory, so the whole yield
  the pull produced) and away from the SERVED reference (the adapted one, so
  what tracking left over), both anchor-aligned so the torso's own motion
  cancels; with the told and environment stiffness and the event phase (0
  free, 1 ramp up, 2 hold, 3 ramp down). Zero outside events, below 0.5 N,
  and on pools without a contact script. An offline pass slices the events
  and reads K_eff over each hold."""
  num_envs = command._motion_id.shape[0]
  device = command._motion_id.device
  zeros = torch.zeros(num_envs, device=device)
  sig = command.compliance_signals_multi()
  slot_ids = getattr(command, "_compliance_slot_body_ids", None)
  free = getattr(command, "robot_free_body_pos_relative_w", None)
  if sig is None or slot_ids is None or free is None:
    return {key: zeros.clone() for key in _CLIP_DRAG_KEYS}
  n, k = sig.in_event.shape
  rows = torch.arange(n, device=device).view(-1, 1)
  body = slot_ids.clamp(min=0).view(1, -1).expand(n, k)
  live = command.robot.data.body_link_pos_w[rows, body]  # (N,K,3)
  served = command.robot_ref_body_pos_relative_w[rows, body]
  mag = sig.applied_force.norm(dim=-1)  # (N,K)
  direction = sig.applied_force / mag.clamp(min=1e-6).unsqueeze(-1)
  gate = (sig.in_event & (mag > 0.5)).float()
  yield_free = ((live - free[rows, body]) * direction).sum(-1) * gate
  yield_served = ((live - served) * direction).sum(-1) * gate
  out: dict[str, torch.Tensor] = {}
  for i in range(_CLIP_DRAG_SLOTS):
    if i >= k:
      for field in _CLIP_DRAG_FIELDS:
        out[f"clip_drag/k{i}_{field}"] = zeros.clone()
      continue
    out[f"clip_drag/k{i}_in_event"] = sig.in_event[:, i].float()
    out[f"clip_drag/k{i}_phase"] = sig.phase[:, i].float()
    out[f"clip_drag/k{i}_told"] = sig.robot_stiffness[:, i] * gate[:, i]
    out[f"clip_drag/k{i}_kenv"] = sig.forcefield_stiffness[:, i] * gate[:, i]
    out[f"clip_drag/k{i}_force"] = mag[:, i] * gate[:, i]
    out[f"clip_drag/k{i}_yield_free"] = yield_free[:, i]
    out[f"clip_drag/k{i}_yield_served"] = yield_served[:, i]
  return out


def compute_partner_metrics(
  command: CodancingComposedCommand,
) -> dict[str, torch.Tensor]:
  """Per-env partner-relative scalars, m: each foot's XY error against its
  foot-follow target (the partner's CROSSED foot plus the reward's XY offset,
  the pairing the policy trains on) and their mean, and the pelvis-to-pelvis
  distance in 3D and in the ground plane. Zeros on a solo pool."""
  num_envs = command._motion_id.shape[0]
  device = command._motion_id.device
  if not command._human_motions:
    zeros = torch.zeros(num_envs, device=device)
    return {key: zeros.clone() for key in _PARTNER_METRIC_KEYS}
  offset = torch.tensor(_foot_follow_offset_xy(command), device=device)

  def xy_error(robot_foot: torch.Tensor, human_foot: torch.Tensor) -> torch.Tensor:
    return torch.norm(robot_foot[:, :2] - (human_foot[:, :2] + offset), dim=-1)

  left = xy_error(command.robot_left_foot_pos_w, command.human_right_foot_pos_w)
  right = xy_error(command.robot_right_foot_pos_w, command.human_left_foot_pos_w)
  robot_pelvis = command.robot.data.body_link_pos_w[
    :, command.robot.body_names.index("pelvis")
  ]
  human_pelvis = command.get_human_body_pos_w(
    [command.human_body_names.index("pelvis")], "auto"
  )[:, 0]
  gap = human_pelvis - robot_pelvis
  return {
    "partner/foot_follow_xy": 0.5 * (left + right),
    "partner/foot_follow_xy_left": left,
    "partner/foot_follow_xy_right": right,
    "partner/pelvis_dist": gap.norm(dim=-1),
    "partner/pelvis_dist_xy": gap[:, :2].norm(dim=-1),
  }


def _expand_reference_frames(
  reference_frame: MetricReferenceFrame,
) -> tuple[Literal["global", "anchor_aligned"], ...]:
  if reference_frame == "both":
    return ("global", "anchor_aligned")
  if reference_frame in ("global", "anchor_aligned"):
    return (reference_frame,)
  raise ValueError(f"Unsupported metric reference frame: {reference_frame!r}")


def _robot_ref_tracking_keys(
  body_group: str,
  reference_frame: Literal["global", "anchor_aligned"],
  pair: tuple[str, str] = ("served", "live"),
) -> tuple[str, ...]:
  prefix = f"robot_ref/{pair[0]}_vs_{pair[1]}/{body_group}/{reference_frame}"
  return (f"{prefix}_pos", f"{prefix}_rot")


# ── Masked means over the active contact slots ──────────────────────────────


def masked_slot_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Masked mean of per-slot scalars over the active slots (N,K)->(N,); all free
  -> 0."""
  m = mask.float()
  denom = m.sum(dim=-1)
  return torch.where(
    denom > 0, (values * m).sum(dim=-1) / denom.clamp(min=1.0), torch.zeros_like(denom)
  )


def _masked_mean_mag(vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Masked mean of per-slot vector magnitudes over the active slots
  (N,K,3)->(N,); all free -> 0."""
  return masked_slot_mean(vec.norm(dim=-1), mask)
