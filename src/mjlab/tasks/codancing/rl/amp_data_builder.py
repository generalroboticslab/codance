"""Build the expert AMP data pool from the codancing command's loaded motions.

Produces a ``TensorAmpData`` (the rsl_rl task-blind expert protocol) with one
window pool per discriminator group, computed through the SAME
``compute_amp_features`` as the policy ``amp`` obs term -- so policy and expert
features cannot drift. The reference motions are already resident on the
command's per-pair ``MotionLoader``s, so the builder reads them directly instead
of re-loading the npz.

The ``disc_mode`` maps motions -> ``disc_group`` (the single co-design knob, the
same one rsl_rl's discriminator reads):
  - ``single``               : all motions pooled into group 0 (one discriminator).
  - ``multi_head``/``separate`` : motion i -> group i (K = number of motions; the
    index matches ``command.active_disc_group`` so policy windows route to the
    same group).
"""

from __future__ import annotations

import torch
from rsl_rl.storage.amp_data import TensorAmpData

from mjlab.tasks.codancing.mdp.amp_features import AmpFeatureCfg, compute_amp_features
from mjlab.tasks.codancing.mdp.commands import get_codancing_command
from mjlab.tasks.codancing.motion.library import amp_group_assignment


def _group_assignment(disc_mode: str, command) -> list[int]:
  """Map each clip to a discriminator-group index.

  ``single`` pools everything into one group; ``multi_head``/``separate`` use
  the per-clip ``amp_group`` LUT (default: group == clip index), the same
  mapping ``command.active_disc_group`` routes windows with. Clips are dropped
  from the expert pool by ``_skip_expert`` at the window-collection site
  (per-clip ``amp_expert: false``, or the bulk
  ``amp_feature.expert_exclude_adapted`` switch).
  """
  if disc_mode == "single":
    return [0] * len(command._clips)
  if disc_mode in ("multi_head", "separate"):
    return list(amp_group_assignment(command._clips))
  raise NotImplementedError(
    f"disc_mode={disc_mode!r} not supported (use single/multi_head/separate)"
  )


def _skip_expert(clip, feat: AmpFeatureCfg) -> bool:
  """True when this clip must stay out of the expert pool.

  Two gates: the clip's own ``amp_expert: false``, and the bulk
  ``expert_exclude_adapted`` switch, which drops any clip whose served motion
  file differs from its free (original) one -- the structural signature of an
  offline force-augmented entry. Originals (equal paths, or no free face at
  all) always pass the second gate.
  """
  if not getattr(clip, "amp_expert", True):
    return True
  if getattr(feat, "expert_exclude_adapted", False):
    free = getattr(clip, "free_motion_file", None)
    if free is not None and str(free) != str(clip.robot_motion_file):
      return True
  return False


def _sliding_windows(features: torch.Tensor, num_frames: int) -> torch.Tensor:
  """``(T, D)`` per-frame features -> ``(num_windows, num_frames, D)`` of all
  contiguous windows (exhaustive; left-padded with the first frame if T < F)."""
  num_steps, feat_dim = features.shape
  if num_steps < num_frames:
    pad = features[:1].expand(num_frames - num_steps, feat_dim)
    return torch.cat([pad, features], dim=0).unsqueeze(0)
  return features.unfold(0, num_frames, 1).permute(0, 2, 1).contiguous()


def _motion_features(
  loader, body_idx: torch.Tensor, anchor_idx: int, feat: AmpFeatureCfg, device
) -> torch.Tensor:
  """Per-frame AMP features ``(T, feature_dim)`` for one MotionLoader,
  sliced to the AMP bodies + anchor in the loader's full-body (30) ordering."""
  pos_full = loader._body_pos_w.to(device)  # (T, 30, 3)
  quat_full = loader._body_quat_w.to(device)  # (T, 30, 4)
  idx = body_idx.to(device)
  body_pos = pos_full[:, idx]
  body_quat = quat_full[:, idx]
  anchor_pos = pos_full[:, anchor_idx : anchor_idx + 1]
  anchor_quat = quat_full[:, anchor_idx]
  lin = (
    loader._body_lin_vel_w.to(device)[:, idx] if "lin_vel" in feat.components else None
  )
  ang = (
    loader._body_ang_vel_w.to(device)[:, idx] if "ang_vel" in feat.components else None
  )
  return compute_amp_features(
    body_pos,
    body_quat,
    anchor_pos,
    anchor_quat,
    use_anchor=feat.use_anchor,
    components=feat.components,
    body_lin_vel=lin,
    body_ang_vel=ang,
  )


def build_amp_data(env, command_name: str = "codancing", device=None) -> TensorAmpData:
  """Build the per-``disc_group`` expert ``TensorAmpData`` from the command's
  loaded reference motions."""
  command = get_codancing_command(env, command_name)
  feat = command.cfg.amp_feature
  assert feat is not None, "command.cfg.amp_feature must be set to build amp_data"
  device = device if device is not None else env.device

  names = list(env.scene["robot"].body_names)
  body_idx = torch.tensor([names.index(n) for n in feat.body_names], dtype=torch.long)
  anchor_idx = names.index(feat.anchor_body_name)

  loaders = command._robot_motions
  clips = command._clips
  assignment = _group_assignment(feat.disc_mode, command)
  num_groups = max(assignment) + 1
  group_windows: list[list[torch.Tensor]] = [[] for _ in range(num_groups)]
  skipped = 0
  for clip, group, loader in zip(clips, assignment, loaders, strict=True):
    if _skip_expert(clip, feat):
      skipped += 1
      continue
    feats = _motion_features(loader, body_idx, anchor_idx, feat, device)
    group_windows[group].append(_sliding_windows(feats, feat.num_frames))

  empty = [g for g, windows in enumerate(group_windows) if not windows]
  if empty:
    raise ValueError(
      f"AMP expert pool empty for disc group(s) {empty} after skipping "
      f"{skipped}/{len(clips)} clip(s) (amp_expert=false / "
      "expert_exclude_adapted); the discriminator would train on nothing."
    )
  if skipped:
    print(f"[INFO] AMP expert pool: skipped {skipped}/{len(clips)} clip(s)")
  pools = [torch.cat(windows, dim=0) for windows in group_windows]
  return TensorAmpData(pools, device)
