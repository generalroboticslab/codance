"""AMP observation terms (task-blind AMP).

Two observation groups feed the rsl_rl AMP algorithm:
  - ``amp``: the per-step AMP feature window from ``compute_amp_features`` over
    the live robot AMP bodies. The ``amp`` ObservationGroupCfg uses
    ``history_length=num_frames`` + ``flatten_history_dim=False`` so
    ``obs["amp"]`` is ``(num_envs, num_frames, feature_dim)``.
  - ``amp_group``: the per-env discriminator-group routing index
    (``command.active_disc_group``) so ``obs["amp_group"]`` is ``(num_envs, 1)``.

Both read the declarative ``AmpFeatureCfg`` carried on the codancing command
cfg, so the policy obs features come from the SAME ``compute_amp_features`` as
the expert ``amp_data`` (no drift).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.codancing.mdp.amp_features import AmpFeatureCfg, compute_amp_features
from mjlab.tasks.codancing.mdp.commands import (
  CodancingComposedCommand,
  get_codancing_command,
)

__all__ = ["amp_features_obs", "amp_group_obs"]


def _resolve_amp_indices(
  command: CodancingComposedCommand, robot, feat: AmpFeatureCfg, device
) -> tuple[torch.Tensor, int]:
  """Resolve (AMP body indices, anchor index) into ``robot.body_names``, cached
  on the command (stable for the env's lifetime)."""
  cached = getattr(command, "_amp_feature_idx", None)
  if cached is not None:
    return cached
  names = list(robot.body_names)
  body_idx = torch.tensor(
    [names.index(n) for n in feat.body_names], dtype=torch.long, device=device
  )
  anchor_idx = names.index(feat.anchor_body_name)
  command._amp_feature_idx = (body_idx, anchor_idx)  # type: ignore[attr-defined]
  return body_idx, anchor_idx


def amp_features_obs(
  env: ManagerBasedRlEnv, command_name: str = "codancing"
) -> torch.Tensor:
  """Per-step AMP feature vector ``(num_envs, feature_dim)`` over the live robot
  AMP bodies (the ``amp`` group stacks ``num_frames`` of these)."""
  command = get_codancing_command(env, command_name)
  feat = command.cfg.amp_feature
  assert feat is not None, "command.cfg.amp_feature must be set for the amp obs term"
  robot = env.scene["robot"]
  body_idx, anchor_idx = _resolve_amp_indices(command, robot, feat, env.device)
  data = robot.data
  body_pos = data.body_link_pos_w[:, body_idx]
  body_quat = data.body_link_quat_w[:, body_idx]
  anchor_pos = data.body_link_pos_w[:, anchor_idx : anchor_idx + 1]
  anchor_quat = data.body_link_quat_w[:, anchor_idx]
  lin = data.body_link_lin_vel_w[:, body_idx] if "lin_vel" in feat.components else None
  ang = data.body_link_ang_vel_w[:, body_idx] if "ang_vel" in feat.components else None
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


def amp_group_obs(
  env: ManagerBasedRlEnv, command_name: str = "codancing"
) -> torch.Tensor:
  """Per-env discriminator-group routing index ``(num_envs, 1)`` (fwd/rev/...).

  The rsl_rl AMP algorithm reads this as ``obs["amp_group"]`` and re-casts to
  long; small integer group ids survive the float obs round-trip exactly."""
  command = get_codancing_command(env, command_name)
  return command.active_disc_group.reshape(-1, 1).float()
