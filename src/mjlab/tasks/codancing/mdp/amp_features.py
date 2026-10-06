"""Shared AMP feature computation for codancing.

A single pure ``compute_amp_features`` used by BOTH the policy-side obs term
(live robot body state) and the expert-side ``amp_data`` builder (reference
motion frames), so the two can never drift.

The feature is a per-body, anchor-relative stack of declarative ``components``
(default ``pos`` + ``ori6d``: 3 + 6 = 9 floats/body). The
caller selects the AMP bodies + anchor and passes already-sliced tensors; this
function is layout-agnostic (works for a live ``(num_envs, B, ...)`` step or an
expert ``(num_frames_total, B, ...)`` motion tensor -- the leading dim ``L`` is
just the batch).

``AmpFeatureCfg`` is the declarative spec carried on the codancing command cfg
(YAML-serializable); it drives the obs term, the
``amp_data`` expert pool, and the construct-time feature-dim resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import matrix_from_quat

# Per-component float width per body.
_COMPONENT_DIMS: dict[str, int] = {"pos": 3, "ori6d": 6, "lin_vel": 3, "ang_vel": 3}

# The implemented AMP feature mask modes. "static" keeps every configured body
# active on every frame; a config naming any other mode fails at build time
# instead of silently running the static path.
_MASK_MODES: tuple[str, ...] = ("static",)


@dataclass(kw_only=True)
class AmpFeatureCfg:
  """Declarative AMP feature spec (lives on the codancing command cfg).

  Attributes:
    body_names: AMP bodies, in feature order; must be non-empty (the build
      rejects an empty list). The AMP policy configs pass a registry set,
      ``${body_sets.amp_lower_body_with_torso}``.
    anchor_body_name: Body whose frame positions/orientations are expressed in
      (when ``use_anchor``).
    use_anchor: Express body pos/ori (and velocities) relative to the anchor
      frame; otherwise world frame.
    components: Ordered feature components per body. Each of ``pos`` (3),
      ``ori6d`` (6, first two columns of the rotation matrix), ``lin_vel`` (3),
      ``ang_vel`` (3). Default ``("pos", "ori6d")``.
    num_frames: Number of stacked consecutive frames (history) per AMP window;
      drives BOTH the ``amp`` obs-group ``history_length`` and the ``amp_data``
      window length (one source of truth).
    disc_mode: The per-motion discriminator mode -- the SAME value the runner's
      ``algorithm.amp.disc_mode`` carries (env reads it via interpolation, so
      the expert grouping and the discriminator can't disagree). ``single`` =>
      one expert pool / one discriminator group; ``multi_head``/``separate`` =>
      one group per motion.
    mask_mode: Which bodies are active in the feature. ``"static"`` (the only
      mode) keeps every body in ``body_names`` active on every frame, so the
      feature is exactly ``len(body_names) * per_body_dim`` floats. A partial
      body set (lower body, upper body, ...) is selected by listing a subset in
      ``body_names``.
  """

  body_names: tuple[str, ...] = ()
  anchor_body_name: str = "torso_link"
  use_anchor: bool = True
  components: tuple[str, ...] = ("pos", "ori6d")
  num_frames: int = 5
  disc_mode: str = "single"
  mask_mode: str = "static"
  # Explicit opt-in for keeping the anchor among the obs bodies: its relative
  # pos/ori6d are the constant identity (9 dims of padding), which is normally
  # a config accident the compose-time guard rejects.
  allow_anchor_in_bodies: bool = False
  # Keep offline force-augmented clips OUT of the expert pool: drop every clip
  # whose served motion differs from its free (original) motion, so the
  # discriminator's style comes from original legs only. Structural (compares
  # the two file paths), so it needs no naming convention and is a no-op on
  # track_original pools (served == free). Tracking, foot-follow, and the
  # policy-side amp obs are untouched; per-clip `amp_expert: false` still
  # excludes a clip regardless of this switch.
  expert_exclude_adapted: bool = False

  def __post_init__(self) -> None:
    for c in self.components:
      if c not in _COMPONENT_DIMS:
        raise ValueError(
          f"unknown AMP feature component {c!r}; expected one of {tuple(_COMPONENT_DIMS)}"
        )
    if self.mask_mode not in _MASK_MODES:
      raise ValueError(
        f"unknown AMP feature mask_mode {self.mask_mode!r}; only {_MASK_MODES} "
        "is implemented. Select a partial body set with body_names instead."
      )
    if self.disc_mode not in ("single", "multi_head", "separate"):
      raise ValueError(
        f"AMP feature disc_mode must be one of ('single', 'multi_head', "
        f"'separate'), got {self.disc_mode!r}."
      )

  def per_body_dim(self) -> int:
    return sum(_COMPONENT_DIMS[c] for c in self.components)

  def feature_dim(self, num_bodies: int | None = None) -> int:
    n = num_bodies if num_bodies is not None else len(self.body_names)
    return n * self.per_body_dim()


def compute_amp_features(
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  anchor_pos: torch.Tensor,
  anchor_quat: torch.Tensor,
  *,
  use_anchor: bool = True,
  components: tuple[str, ...] = ("pos", "ori6d"),
  body_lin_vel: torch.Tensor | None = None,
  body_ang_vel: torch.Tensor | None = None,
) -> torch.Tensor:
  """Compute the per-body AMP feature stack.

  Args:
    body_pos: ``(L, B, 3)`` selected AMP-body world positions.
    body_quat: ``(L, B, 4)`` selected AMP-body world orientations (wxyz).
    anchor_pos: ``(L, 3)`` or ``(L, 1, 3)`` anchor world position.
    anchor_quat: ``(L, 4)`` anchor world orientation (wxyz).
    use_anchor: Express features in the anchor frame (vs world).
    components: Ordered components per body (see :class:`AmpFeatureCfg`).
    body_lin_vel: ``(L, B, 3)`` world linear velocities (required iff
      ``"lin_vel"`` in ``components``).
    body_ang_vel: ``(L, B, 3)`` world angular velocities (required iff
      ``"ang_vel"`` in ``components``).

  Returns:
    ``(L, B * per_body_dim)`` feature tensor, body-major then component-order.
  """
  num_steps, num_bodies = body_pos.shape[0], body_pos.shape[1]
  if anchor_pos.dim() == 2:
    anchor_pos = anchor_pos.unsqueeze(1)  # (L, 1, 3)

  body_rot = matrix_from_quat(body_quat.reshape(-1, 4)).reshape(
    num_steps, num_bodies, 3, 3
  )
  if use_anchor:
    anchor_rot_inv = matrix_from_quat(anchor_quat).transpose(-1, -2)  # (L, 3, 3)
    rel_pos = torch.einsum("lij,lbj->lbi", anchor_rot_inv, body_pos - anchor_pos)
    rel_rot = torch.einsum("lij,lbjk->lbik", anchor_rot_inv, body_rot)
  else:
    anchor_rot_inv = None
    rel_pos = body_pos
    rel_rot = body_rot

  def _to_frame(vec: torch.Tensor) -> torch.Tensor:
    if use_anchor:
      assert anchor_rot_inv is not None
      return torch.einsum("lij,lbj->lbi", anchor_rot_inv, vec)
    return vec

  feats: list[torch.Tensor] = []
  for c in components:
    if c == "pos":
      feats.append(rel_pos)
    elif c == "ori6d":
      feats.append(rel_rot[..., :2].reshape(num_steps, num_bodies, 6))
    elif c == "lin_vel":
      if body_lin_vel is None:
        raise ValueError("components includes 'lin_vel' but body_lin_vel is None")
      feats.append(_to_frame(body_lin_vel))
    elif c == "ang_vel":
      if body_ang_vel is None:
        raise ValueError("components includes 'ang_vel' but body_ang_vel is None")
      feats.append(_to_frame(body_ang_vel))
    else:
      raise ValueError(f"unknown AMP feature component {c!r}")

  per_body = torch.cat(feats, dim=-1)  # (L, B, per_body_dim)
  return per_body.reshape(num_steps, -1)  # (L, B * per_body_dim)
