"""The codancing command: the motion cursor over the clip pool, the served and
free reference faces, the partner, the compliance signals and the reference
visualization. A CommandTerm, so its state carries across steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.codancing import viz_palette
from mjlab.tasks.codancing.mdp.amp_features import AmpFeatureCfg
from mjlab.tasks.codancing.mdp.body_sets import (
  body_set_indices,
  body_set_names,
  model_order_indexes,
)
from mjlab.tasks.codancing.mdp.metrics import (
  MetricReferenceFrame,
  MetricSuiteName,
  compute_metric_suite,
  iter_metric_suite_keys,
  log_safe_metric_name,
)
from mjlab.tasks.codancing.motion.library import (
  MotionClip,
  PairedMotionClip,
  amp_group_assignment,
)
from mjlab.tasks.codancing.motion.loaders import (
  MotionLoader,
  check_body_order,
)
from mjlab.tasks.codancing.motion.motion_cursor import (
  ClipCursorSampleCfg,
  ClipSourceCfg,
  EntityStateTarget,
  MotionCursorCfg,
  apply_reset_perturbation,
  validate_cursor_modes,
  write_state,
)
from mjlab.tasks.codancing.motion.phase_sampler import PhaseSampler
from mjlab.tasks.codancing.motion.reference_sources import (
  ClipReferenceProvider,
  ReferenceProvider,
)
from mjlab.tasks.codancing.motion.registry import MotionFrames, MotionRegistry
from mjlab.tasks.shared.mdp.command_debug_vis import (
  CoordAxisColors,
  CoordFrameLayer,
  CoordFrameStyle,
  anchor_coord_frame,
  draw_coord_frames,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Source selector for the unified `get_human_*` accessors and any MDP term
# that reads human poses. "auto" follows `cfg.load_human_entity`; "entity"
# forces FK reads (errors if entity not loaded); "motion" always reads from
# the motion file (useful for ablations and side-by-side overlap checks).
HumanSource = Literal["auto", "entity", "motion"]

__all__ = [
  "CodancingComposedCommand",
  "CodancingComposedCommandCfg",
  "get_codancing_command",
  "get_codancing_command_cfg",
]


@dataclass
class ComplianceSignalsMulti:
  """multi-link per-step compliance quantities (shared by the _multi consumers in
  obs/rewards/metrics), one entry per contact slot. All in the world frame,
  zeroed outside events."""

  in_event: torch.Tensor  # (N,K) bool
  active: torch.Tensor  # (N,K) bool
  phase: torch.Tensor  # (N,K) long
  robot_stiffness: torch.Tensor  # (N,K)
  robot_rot_stiffness: torch.Tensor  # (N,K)
  forcefield_stiffness: torch.Tensor  # (N,K) K_env, linear
  forcefield_rot_stiffness: torch.Tensor  # (N,K) K_rot_env, rotational
  applied_force: torch.Tensor  # (N,K,3)
  applied_torque: torch.Tensor  # (N,K,3)
  desired_force: torch.Tensor  # (N,K,3)
  desired_torque: torch.Tensor  # (N,K,3)


# All compliance markers share one set of sizes. The three setpoint faces coincide under perfect tracking and are told
# apart by "baked = wide and translucent / re-anchored = thin and solid" (the
# contract and the reasoning behind every value live in viz_palette).
# The color of each channel is decided by viz_palette; do not copy it here:
# `uv run python src/mjlab/tasks/codancing/viz_palette.py` prints the palette.
# The two styles have separate roles, one primitive per job:
#   arrow  = the **magnitude** channel. Its length follows the force (a fixed m/N
#            gain + a soft-compression knee), so even a few newtons are drawn. It
#            **may cross** the setpoint sphere: it expresses how large the force
#            is and does not claim to be geometry; geometry is the spring's job.
#   spring = the **geometry** channel. The thin cylinder spans exactly to the
#            setpoint and never beyond.
# Every value and its rationale live in viz_palette.SIZES (in the same place as the
# colors, with the same per-medium overrides); here we only resolve the ones this
# module uses into video-medium constants, and the drawing code keeps reading the
# constants.
_FORCE_ARROW_SCALE = viz_palette.VIDEO.size("force_arrow_scale")
_FORCE_ARROW_KNEE_LEN = viz_palette.VIDEO.size("force_arrow_knee_len")
_SPRING_RADIUS = viz_palette.VIDEO.size("spring_connector_radius")
_SPRING_BALL_SCALE = viz_palette.VIDEO.size("spring_ball_scale")
_DIR_ARROW_LEN = viz_palette.VIDEO.size("dir_arrow_len")
_DIR_ARROW_CLEAR = viz_palette.VIDEO.size("dir_arrow_clear")
_DIR_ARROW_WIDTH = viz_palette.VIDEO.size("dir_arrow_width")
_SCRIPT_SETPOINT_SPHERE_RADIUS = viz_palette.VIDEO.size("setpoint_ball_baked")
_LIVE_SETPOINT_SPHERE_RADIUS = viz_palette.VIDEO.size("setpoint_ball_live")
_BAKED_ARROW_WIDTH = viz_palette.VIDEO.size("wrench_arrow_baked")
_LIVE_ARROW_WIDTH = viz_palette.VIDEO.size("wrench_arrow_live")
_REANCHOR_BASE_SPHERE_RADIUS = viz_palette.VIDEO.size("reanchor_base_ball")
_REANCHOR_CONNECTOR_RADIUS = viz_palette.VIDEO.size("reanchor_connector_radius")
_REANCHOR_HEADING_LEN = viz_palette.VIDEO.size("reanchor_heading_len")
_REANCHOR_HEADING_WIDTH = viz_palette.VIDEO.size("reanchor_heading_width")


def resolve_color(value: Any, where: str) -> tuple[float, float, float, float] | None:
  """A config color -> a 4-tuple, or None when unset.

  A string is a palette name (``"85_slot_a"``); a sequence is a literal rgba,
  whose arity is checked here so a three-element color fails at build time
  rather than painting a wrong alpha.
  """
  if value is None:
    return None
  if isinstance(value, str):
    try:
      return viz_palette.VIDEO.rgba(value)
    except KeyError as exc:
      raise ValueError(f"{where}: {exc.args[0]}") from None
  parts = [float(c) for c in value]
  if len(parts) != 4:
    raise ValueError(f"{where} needs 4 numbers (r, g, b, a), got {len(parts)}.")
  return (parts[0], parts[1], parts[2], parts[3])


def _ghost_rgba(value: Any, where: str) -> tuple[float, float, float, float]:
  """A face's ghost color from config: a palette name or a literal rgba. Unlike
  :func:`resolve_color` this one is not optional, because every face has a
  default."""
  rgba = resolve_color(value, where)
  if rgba is None:
    raise ValueError(f"{where} must be set (a palette name or an rgba).")
  return rgba


def _coord_axis_colors(value: Any, where: str) -> CoordAxisColors | None:
  """A config color set -> one rgb triple per arrow, a
  ``viz_palette.COORD_AXIS_SETS`` name, or None when unset."""
  if value is None:
    return None
  if isinstance(value, str):
    try:
      return viz_palette.COORD_AXIS_SETS[value]
    except KeyError:
      raise ValueError(
        f"{where}: unknown coord axis set {value!r}; "
        f"valid: {sorted(viz_palette.COORD_AXIS_SETS)}"
      ) from None
  axes = [[float(c) for c in axis] for axis in value]
  if len(axes) != 3 or any(len(axis) != 3 for axis in axes):
    raise ValueError(
      f"{where} needs 3 rgb triples, one per coord axis (x, y, z); "
      f"got {[len(axis) for axis in axes]}."
    )
  return tuple((axis[0], axis[1], axis[2]) for axis in axes)


# Every face `_drawable_faces` can build -- the palette's face table IS the
# roster. Config validates against it, so a misspelled face name in
# `body_styles` is caught instead of ignored.
_DRAWABLE_FACE_NAMES = frozenset(viz_palette.FACE_STYLE)


def _face_ghost(face_name: str) -> tuple[float, float, float, float]:
  """One face's default ghost rgba, from the palette."""
  return viz_palette.VIDEO.rgba(viz_palette.FACE_STYLE[face_name].ghost)


def _anchor_axis_len(face_name: str) -> float:
  """The arrow length for a face's one-body anchor layer."""
  length = viz_palette.FACE_STYLE[face_name].anchor_axis_len
  assert length is not None, f"face {face_name!r} draws no anchor layer"
  return length


def _face_axes(face_name: str) -> CoordAxisColors | None:
  """One face's default coord-axis colors, or None for the visualizer's own
  full-strength rgb (which is the live face's convention)."""
  axes = viz_palette.FACE_STYLE[face_name].coord_axis
  return None if axes is None else viz_palette.COORD_AXIS_SETS[axes]


@dataclass(frozen=True)
class BodyStyle:
  """How ONE body deviates from its face's default styling.

  Recorded per body name so a config states exactly which bodies it repaints and
  in which representation. A field left None keeps the face's own value, so
  ``{"left_wrist_yaw_link": BodyStyle(ghost_color=...)}`` repaints that link's
  ghost geoms and leaves its coord frame alone. Ghost color carries its own
  alpha (rgba); a coord frame's opacity is separate (``coord_axis_alpha``),
  because its colors are rgb only.

  Set from config as a nested dict, face name -> body name -> these fields:

      body_styles:
        desired:
          left_wrist_yaw_link: {ghost_color: [0.95, 0.2, 0.2, 0.9]}
  """

  ghost_color: tuple[float, float, float, float] | None = None
  coord_axis_colors: CoordAxisColors | None = None
  coord_axis_alpha: float | None = None


@dataclass(frozen=True)
class _DrawableFace:
  """One drawable reference face and the two ways it can be shown.

  A face is one pose of the tracked bodies (raw reference, that reference
  carried onto the live anchor, the free face, the live robot). It can be drawn
  as a translucent GHOST or as per-body COORD FRAMES; both read the same
  buffers, so the face carries both and the two toggles pick the representation
  instead of every new face inventing its own pair of draw methods.

  Ghost fields (``root_*`` + ``joint_pos``) describe a qpos; coord-frame fields
  (``body_*``) describe per-body world poses. ``body_*`` is None for a face with
  no per-body truth (the free ghost is a synthetic mix of the served root and the
  free joints), and such a face can only be a ghost.

  ``ghost_bodies`` is a body-name list (a ``${body_sets.*}`` reference in
  config; empty = the whole model); ``coord_frame_body_indices`` is the same
  list as model-order column indices, carried PER FACE. Pick a sparse set
  when frames are on: a list of every body of a region draws a coord frame on
  each of them.

  "Frame" here always means a COORD frame (three drawn arrows), never a clip frame
  (a motion time index) or a video frame; see ``command_debug_vis``.
  """

  name: str  # coord-frame label prefix; cache key for the ghost model
  root_pos_w: torch.Tensor  # (N,3)
  root_quat_w: torch.Tensor  # (N,4)
  joint_pos: torch.Tensor  # (N,J)
  body_pos_w: torch.Tensor | None  # (N,B,3)
  body_quat_w: torch.Tensor | None  # (N,B,4)
  ghost_color: tuple[float, float, float, float]  # rgba; alpha drives the mesh
  ghost_bodies: tuple[str, ...]  # bodies the ghost model keeps; () = all
  coord_frame_body_indices: tuple[int, ...]  # model-order column indices
  coord_axis_colors: CoordAxisColors | None  # None = visualizer default rgb
  coord_axis_length: float  # arrow length, meters
  coord_axis_alpha: float  # opacity for this face's arrows, 0 to 1
  body_styles: dict[str, BodyStyle]  # per-body overrides, keyed by body name
  ghost: bool
  coord_frames: bool


@dataclass
class ComplianceMark:
  """Everything drawable for one contact slot in this frame.

  The record the force visualization draws. Each quantity is defined in one
  place only: script quantities come from the provider (the same cursor as the
  physics), the applied wrench from the sim buffer (never recomputed), and the
  re-anchored setpoint goes through the force event's same pure function
  ``reanchor_points`` + the same frozen anchors ``_ff_*`` + the same rotation
  knob. So what is drawn is exactly what the physics uses.
  """

  slot: int  # contact slot index
  k_env: float  # K_env (N/m)
  script_force: np.ndarray  # (3,) script force (WORLD), = cs.force
  script_setpoint: np.ndarray  # (3,) baked setpoint channel in world (+ env origin)
  # (3,) the re-anchored setpoint **stashed** by the force event this step = the
  # point the reactive force actually pulls toward. Not recomputed: read straight
  # from `_ff_setpoint_w`. None when there is no force event this step (no event
  # term wired / feedforward force law): there is no spring at all then, so
  # drawing an "equilibrium point" would mislead.
  setpoint_rise: np.ndarray | None
  setpoint_per_frame: np.ndarray  # (3,) setpoint re-anchored per frame (viz-only face)
  # (3,) the forced link on the reference ghost; None if not in the reference body set.
  ref_link: np.ndarray | None
  # (3,) the forced link on the live robot; None if it is not on the robot.
  live_link: np.ndarray | None
  # (3,) the force actually applied this step (sim buffer); present iff live_link is.
  applied_force: np.ndarray | None


def force_arrow_end(start: np.ndarray, force: np.ndarray) -> np.ndarray:
  """End point of the arrow-style force arrow = start + force × visual gain, with
  the length **soft-compressed** logarithmically above the knee.

  Below the knee it is plain linear (in normal operation lengths compare directly);
  above it, the length grows as ``knee·(1+ln(len/knee))``, **monotonically**, so
  the order still reads when several arrows exceed the knee at once. See the roles
  described at the constants: crossing the setpoint is allowed; exact geometry is
  the job of the spring style's thin cylinder."""
  v = np.asarray(force, dtype=float) * _FORCE_ARROW_SCALE
  n = float(np.linalg.norm(v))
  if n <= _FORCE_ARROW_KNEE_LEN or n < 1e-9:
    return start + v
  compressed = _FORCE_ARROW_KNEE_LEN * (1.0 + math.log(n / _FORCE_ARROW_KNEE_LEN))
  return start + v * (compressed / n)


class CodancingComposedCommand(CommandTerm):
  cfg: "CodancingComposedCommandCfg"
  _env: ManagerBasedRlEnv

  def __init__(
    self, cfg: "CodancingComposedCommandCfg", env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.robot_asset_name]
    self._env_origins = env.scene.env_origins
    # Per-env world XY translation of the whole reference stream (robot
    # reference, human ghost, contact setpoints), written ONLY by the clip-end
    # `chain` mode and cleared at every episode reset. xy only: z never enters,
    # and there is no yaw term.
    self._ref_anchor_offset = torch.zeros(env.num_envs, 3, device=env.device)

    self.human: Entity | None
    self._human_body_names: tuple[str, ...]
    # Standalone g1 spec, set only when entity is skipped. Used to lazily
    # compile a ghost mj_model in `_draw_human_ghost_mesh` (worth keeping the
    # spec around -- it's a few hundred KB and saves a re-parse on first viz).
    self._human_standalone_spec: mujoco.MjSpec | None = None
    if cfg.load_human_entity:
      human_entity: Entity = env.scene[cfg.human_asset_name]
      self.human = human_entity
      self._human_body_names = human_entity.body_names
    else:
      self.human = None
      # Drop worldbody at index 0 to mirror Entity.body_names -- the npz's
      # body_pos_w array follows the same convention.
      from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
        get_spec as _get_g1_human_spec,
      )

      self._human_standalone_spec = _get_g1_human_spec()
      self._human_body_names = tuple(
        b.name.split("/")[-1] for b in self._human_standalone_spec.bodies[1:]
      )

    # fmt:off
    self._human_pelvis_index = self._human_body_names.index(cfg.g1_human_pelvis_body_name)
    self._human_left_hand_index = self._human_body_names.index(cfg.g1_human_left_hand_body_name)
    self._human_right_hand_index = self._human_body_names.index(cfg.g1_human_right_hand_body_name)
    self._human_left_foot_index = self._human_body_names.index(cfg.g1_human_left_foot_body_name)
    self._human_right_foot_index = self._human_body_names.index(cfg.g1_human_right_foot_body_name)
    # fmt:on

    # One anchor body serves the pose-match purpose on BOTH sides (the live
    # robot and the reference); live and reference share the one model-order
    # index space, so a single index reads both. Other purposes (the wrench
    # anchors) are free to pick their own body via the force event's
    # `anchor_body`.
    self._pose_match_anchor_index = self.robot.body_names.index(
      cfg.pose_match_anchor_body_name
    )
    self._robot_left_foot_index = self.robot.body_names.index(
      cfg.robot_left_foot_body_name
    )
    self._robot_right_foot_index = self.robot.body_names.index(
      cfg.robot_right_foot_body_name
    )
    # fmt:on

    # The single clip-end signal: a clean per-step pulse, True only on the step
    # a clip ends (every env that hits clip-end), cleared at the start of the
    # next `_update_command`. The `motion_clip_end` termination reads it to
    # truncate the episode at clip-end.
    self._clip_just_ended = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    # Per-env "this clip already ended" latch: set when the pulse fires,
    # cleared wherever the cursor is written (reset, chain), so the pulse
    # fires exactly once per clip even when the cursor keeps running past the
    # end (`terminate` with no truncation) or starts past it (a paired pool
    # whose human clip is shorter than the robot clip, with rsi sampling from
    # the robot clock).
    self._clip_end_latched = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    # Lazily resolved on the first update: does a `motion_clip_end` termination
    # reset us at clip-end? If so the cursor steps aside (skips the in-episode
    # resample) and lets that termination do the reset; else it resamples. The
    # termination's PRESENCE is the sole config; the command can't read the
    # termination manager at __init__ (built after the command manager), hence
    # the lazy check.
    self._clip_end_resets: bool | None = None

    # ONE body index space: MODEL order. Positions in `robot.body_names` ==
    # the motion NPZ body rows == the registry's raw `body_*_w` columns ==
    # the columns of every `robot_body_*_w` / `robot_ref_*` / `*_relative_w`
    # tensor (worldbody dropped; model column 0 is the root body by MJCF
    # construction). The npz half of the equality is checked per clip where
    # the npz records `body_names` (see `check_body_order`, the single
    # invariant). Joint indices are entity joint order everywhere (npz joint
    # columns follow it too).

    # -- The clip pool: typed MotionLibrary clips (exec injects motion.motions) --
    self._clips: list[MotionClip | PairedMotionClip] = list(cfg.clips)
    if not self._clips:
      raise ValueError(
        "CodancingComposedCommandCfg.clips is empty. Clips are injected by the "
        "exec layer from the `motion.motions` manifest; the command cannot run "
        "without a clip pool."
      )
    paired_flags = [isinstance(c, PairedMotionClip) for c in self._clips]
    if any(paired_flags) and not all(paired_flags):
      raise ValueError(
        "clips must be homogeneous: all paired (human + robot) or all robot-only."
      )
    is_paired = paired_flags[0]
    if cfg.require_human_motion and not is_paired:
      raise ValueError(
        f"partner_mode={cfg.partner_mode!r} needs paired clips (human + robot), "
        "but the clip pool is robot-only. Pure robot imitation is "
        "env.commands.codancing.partner_mode=none."
      )
    if not cfg.require_human_motion and is_paired:
      raise ValueError(
        "partner_mode=none is pure robot imitation, but the clip pool carries a "
        "human side. Drop the human files or set partner_mode to imagined or g1."
      )
    # Per-clip loaders. The human side exists only for paired clips -- the
    # empty `_human_motions` list is the runtime marker every human-touching
    # path gates on. Loaders are pure readers: ground-alignment / offsets are
    # baked offline (prepare).
    self._human_motions: list[MotionLoader] = []
    self._robot_motions: list[MotionLoader] = []
    self._clip_placement_offsets: list[tuple[float, float, float]] = []
    for clip in self._clips:
      if isinstance(clip, PairedMotionClip):
        human_loader = MotionLoader(clip.human_motion_file, device=env.device)
        check_body_order(
          human_loader.body_names, self._human_body_names, clip.human_motion_file
        )
        self._human_motions.append(human_loader)
        self._clip_placement_offsets.append(clip.placement_offset)
      else:
        self._clip_placement_offsets.append((0.0, 0.0, 0.0))
      robot_loader = MotionLoader(clip.robot_motion_file, device=env.device)
      # Every body index is MODEL order (npz rows == Entity.body_names); an npz
      # that records body_names turns that assumption into a check, one without
      # names is taken to be in model order.
      check_body_order(
        robot_loader.body_names, tuple(self.robot.body_names), clip.robot_motion_file
      )
      self._robot_motions.append(robot_loader)
    # The SELECTION DOMAIN is the trackable subset: `trackable: false` entries
    # are style-corpus-only (their windows feed the AMP expert pools below, but
    # random / alternate / pin never land on them). build_motion_library
    # validates the manifest; clips constructed directly skip it, so re-assert
    # here.
    trackable_ids = [i for i, c in enumerate(self._clips) if c.trackable]
    if not trackable_ids:
      raise ValueError(
        "every clip is `trackable: false` -- the cursor has nothing to track."
      )
    self._trackable_clip_ids = torch.tensor(
      trackable_ids, dtype=torch.long, device=env.device
    )
    weights = torch.tensor(
      [self._clips[i].weight for i in trackable_ids],
      device=env.device,
      dtype=torch.float32,
    )
    self._clip_weights = weights / weights.sum()
    # alternate's cyclic next-tracked LUT (rows of untracked clips are
    # unreachable; they point at the first tracked id defensively).
    next_trackable = torch.full(
      (len(self._clips),), trackable_ids[0], dtype=torch.long, device=env.device
    )
    for pos, clip_id in enumerate(trackable_ids):
      next_trackable[clip_id] = trackable_ids[(pos + 1) % len(trackable_ids)]
    self._next_trackable_id = next_trackable
    # Per-clip AMP discriminator-group LUT (the `amp_group` label axis).
    # Default = clip index (group == clip, what the multi_head and separate
    # discriminator modes rely on); shared labels collapse K as the pool scales.
    self._disc_group_of_clip = torch.tensor(
      amp_group_assignment(self._clips), dtype=torch.long, device=env.device
    )
    if len(self._clips) > 1:
      print(
        f"[INFO] Motion library: {len(self._clips)} clips, "
        f"names={[c.name for c in self._clips]}, "
        f"weights={self._clip_weights.tolist()}"
      )

    # The per-clip placement offsets as one device tensor (indexed per env at
    # every reset, so the pool crosses to the device once, not per reset batch).
    self._clip_placement_offsets_t = torch.tensor(
      self._clip_placement_offsets, device=env.device, dtype=torch.float32
    )
    self._validate_and_log_human_source_diagnostics()

    # Per-env active clip index (the cursor's clip half).
    self._motion_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cursor = cfg.motion_cursor
    if (
      isinstance(cursor.source, ClipSourceCfg)
      and cursor.source.episode_reset.clip_selection == "alternate"
    ):
      # Per-env round-robin: each env advances from ITS OWN previous clip at
      # every resample, so every env loops through the tracked pool in
      # manifest order no matter when it resets. Seeding at the last tracked
      # clip makes each env's first episode play the first tracked clip.
      self._motion_id.fill_(int(self._trackable_clip_ids[-1].item()))
    # One mode per resample moment; reject unknown values up front so a
    # misspelled mode never falls silently into another branch.
    self._clip_index_by_name: dict[str, int] = {
      c.name: i for i, c in enumerate(self._clips)
    }
    validate_cursor_modes(cursor)
    if isinstance(cursor.source, ClipSourceCfg):
      for moment, sampling in (
        ("episode_reset", cursor.source.episode_reset),
        ("clip_end", cursor.source.clip_end),
      ):
        if sampling.clip_selection == "pin":
          if sampling.pin_clip is None:
            raise ValueError(
              f"motion_cursor.source.{moment}.clip_selection='pin' needs "
              "pin_clip (a manifest clip name)."
            )
          if sampling.pin_clip not in self._clip_index_by_name:
            raise ValueError(
              f"pin_clip={sampling.pin_clip!r} is not in the clip pool "
              f"(names={list(self._clip_index_by_name)})."
            )
          if not self._clips[self._clip_index_by_name[sampling.pin_clip]].trackable:
            raise ValueError(
              f"pin_clip={sampling.pin_clip!r} is `trackable: false` (a "
              "style-corpus-only entry) -- pin a trackable clip."
            )

    # Flat-pool registries: continuous-time reads at the per-env cursor. Built
    # from the per-clip loaders (which remain the AMP-expert / IK source); the
    # registry owns the pool the cursor reads. fps == control rate, so a query at
    # `t = step * control_dt` lands on integer frame `step`.
    self._control_dt = float(self._env.step_dt)
    self._robot_registry = MotionRegistry(
      self._robot_motions, self._control_dt, env.device
    )
    self._human_registry = (
      MotionRegistry(self._human_motions, self._control_dt, env.device)
      if self._human_motions
      else None
    )
    # The per-step reference PRODUCER: the clip registry.
    self._reference_provider: ReferenceProvider = ClipReferenceProvider(
      self._robot_registry
    )

    # Compliance provider (SoftMimic AUGMENTED clips): when the clips carry their
    # own free + contact data, build a ContactReferenceProvider that SHARES the
    # served (adapted) registry and adds the FREE face (original ghost) + the
    # contact script (force arrow + the online ForceField event). The CLIP is
    # self-describing -- debug-viz + the force event just read this; no data
    # lives in the viz/event config. None unless clips are augmented.
    self._contact_provider: Any = None
    augmented = [
      c
      for c in self._clips
      if getattr(c, "contact_file", None) and getattr(c, "free_motion_file", None)
    ]
    if augmented:
      if len(augmented) != len(self._clips):
        raise ValueError(
          "Compliance clips need free_motion_file + contact_file on EVERY clip "
          "(the cursor's motion_id indexes the whole pool); only "
          f"{len(augmented)}/{len(self._clips)} carry them."
        )
      from mjlab.tasks.codancing.motion.contact_reference import (
        build_contact_provider_from_clips,
      )

      self._contact_provider = build_contact_provider_from_clips(
        self._clips,
        self._robot_registry,
        self._control_dt,
        str(env.device),
        expected_body_names=tuple(self.robot.body_names),
      )
      # A twin pool (every clip serves its FREE face, e.g. a _track_original
      # overlay) silently degenerates every face-comparison surface: the free
      # ghost z-fights the served ghost at the same pose and compliance yield
      # reads 0 forever. Announce it.
      if all(
        getattr(c, "robot_motion_file", None) == getattr(c, "free_motion_file", None)
        for c in self._clips
      ):
        print(
          "[WARN] every clip serves its free (unforced) reference (a "
          "_track_original pool): the free ghost coincides with the served ghost "
          "and the compliance yield is 0 by construction."
        )

    # The `rsi` phase sampler: built once from the clip lengths (and the
    # contact schedule's hold windows when the pool carries one). The command
    # owns cursor placement; the sampler only draws phases.
    self._phase_sampler: PhaseSampler | None = None
    if cursor.episode_reset == "rsi":
      hold = None
      if self._contact_provider is not None:
        hold = self._contact_provider.hold_windows()
        if len(hold) == 0:
          hold = None  # an empty event table is "no schedule", not a no-op
      self._phase_sampler = PhaseSampler(
        self._robot_registry.motion_lengths,
        cursor.rsi.sampler,
        kernel_lambda=cursor.rsi.kernel_lambda,
        kernel_size=cursor.rsi.kernel_size,
        ema_alpha=cursor.rsi.ema_alpha,
        hold_windows=hold,
        device=str(env.device),
      )
    # Spawn inspection bookkeeping (rsi_inspect): the phase this env's episode
    # started at, did the rsi coin sample it, and what the last reset's
    # perturbation drew (root-pose offsets x y z roll pitch yaw, largest joint
    # offset). Written on every true episode reset, never read back by the
    # reset itself.
    self._rsi_spawn_time = torch.zeros(env.num_envs, device=env.device)
    self._rsi_spawn_sampled = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self._reset_perturb_pose = torch.zeros(env.num_envs, 6, device=env.device)
    self._reset_perturb_joint_max = torch.zeros(env.num_envs, device=env.device)
    self._is_episode_reset = False
    # The single per-env playback cursor: `_motion_id` (which clip) + `_motion_time`
    # (seconds into it). The command consumes the typed MotionLibrary clips
    # directly; selection / start-time / init policy live on
    # `cfg.motion_cursor`.
    self._motion_time = torch.zeros(env.num_envs, device=env.device)
    # The pause counter: while > 0, `_advance_cursor` freezes `motion_time`, so
    # the human entity and the robot reference stand still together (one shared
    # cursor). Written by the viewer's cursor-pause toggle; cleared on
    # the true episode reset; drains by one per step.
    self._pause_steps_left = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    # Per-env, per-slot rising-edge re-anchoring state of the multi-link online force
    # field; K comes from the provider.
    ff_k = (
      int(getattr(self._contact_provider, "_K", 1)) if self._contact_provider else 1
    )
    self._ff_multi_event_id = torch.full(
      (env.num_envs, ff_k), -1, dtype=torch.long, device=env.device
    )
    self._ff_multi_robot_anchor_pos_w = torch.zeros(
      env.num_envs, ff_k, 3, device=env.device
    )
    self._ff_multi_robot_anchor_quat_w = torch.zeros(
      env.num_envs, ff_k, 4, device=env.device
    )
    self._ff_multi_robot_anchor_quat_w[..., 0] = 1.0
    self._ff_multi_ref_anchor_pos_w = torch.zeros(
      env.num_envs, ff_k, 3, device=env.device
    )
    self._ff_multi_ref_anchor_quat_w = torch.zeros(
      env.num_envs, ff_k, 4, device=env.device
    )
    self._ff_multi_ref_anchor_quat_w[..., 0] = 1.0
    # The re-anchored setpoint **actually used** by the force event this step
    # (multi-link unified (N,K,3), env origin included). The visualization draws
    # exactly this copy: not "recomputed with the same function and the same
    # anchors" but the same data, so "what is drawn is what the physics uses" holds
    # by construction, not because two implementations happen to agree (the same
    # reason the applied wrench is read from the sim buffer). Only the reactive
    # force law has a setpoint; feedforward replays the baked wrench and has no
    # spring equilibrium at all, so nothing is written then, _ff_setpoint_valid
    # stays False, and the visualization skips that sphere (no spring, no
    # equilibrium point).
    self._ff_setpoint_w = torch.zeros(env.num_envs, ff_k, 3, device=env.device)
    self._ff_setpoint_valid = False
    # Active config written by the force event every step, read by the force
    # visualization.
    self._ff_reanchor_rotation = "yaw"
    self._ff_anchor_body = ""
    self._ff_reanchor_mode = "rising_edge"
    # Cache-tier convention: an attribute spelled `*_cache` is a PER-STEP
    # freshness cache, cleared by `_invalidate_per_step_caches` whenever the
    # cursor or the robot state moves; build-once tables (body/joint ids,
    # pools, compiled ghost models) are NOT called cache and are never
    # invalidated.
    self._robot_ref_cache: MotionFrames | None = None
    self._robot_free_cache: MotionFrames | None = None
    self._human_ref_cache: MotionFrames | None = None
    # multi-link per-step compliance cache (read by obs / rewards / metrics,
    # computed once) + the robot body id of each slot's fixed link (resolved by
    # name once, a build-once table).
    self._compliance_multi_cache: ComplianceSignalsMulti | None = None
    self._compliance_slot_body_ids: torch.Tensor | None = None
    # fmt:off
    # Allocated here because rewards read robot_ref_body_pos_relative_w before the
    # first command update, so the very first reward step reads these zeros.
    num_model_bodies = len(self.robot.body_names)
    self.robot_ref_body_pos_relative_w = torch.zeros(self.num_envs, num_model_bodies, 3, device=self.device)
    self.robot_ref_body_quat_relative_w = torch.zeros(self.num_envs, num_model_bodies, 4, device=self.device)
    self.robot_ref_body_quat_relative_w[:, :, 0] = 1.0
    # fmt:on
    # FREE-face relative buffers: computed EVERY step on augmented pools
    # (None only when the pool has no free face). Unconditional on purpose: a
    # predicate gating them would have to enumerate every reader, and a reader
    # missing from it would silently render nothing while the video exits 0.
    # One extra anchor-aligned transform per step rules that failure out.
    self.robot_free_body_pos_relative_w: torch.Tensor | None = None
    self.robot_free_body_quat_relative_w: torch.Tensor | None = None

    # Reference ghosts: one recolored standalone g1 model per drawn face,
    # compiled from `self.robot.cfg.spec_fn()` (NOT a deepcopy of the composed
    # scene) and handed to `visualizer.add_ghost_mesh` -- the SAME path as the
    # human ghost. Standalone => never passes the collision-viz spec_fn, so the
    # visual geoms stay in the default-visible groups and the backend's
    # read-only MjvOption renders them as-is (no geom-group fiddling). Built
    # lazily per face in `_draw_face_ghost`, keyed by face name (a face's
    # color and body subset never change mid-run), with its qpos scratch
    # alongside -- a build-once table, never invalidated (deliberately not
    # spelled `*_cache`: that suffix marks the per-step tier).
    self._face_ghost_models: dict[str, tuple[mujoco.MjModel, np.ndarray]] = {}
    # Body-set -> entity joint ids for body-scoped joint reads, plus the
    # standalone robot model the mapping resolves against. Build-once tables;
    # the model compiles on the first narrowed read only.
    self._body_set_joint_id_table: dict[str, torch.Tensor] = {}
    self._standalone_robot_model: mujoco.MjModel | None = None
    self._robot_ref_ghost_color = cfg.robot_ref_ghost_color
    self._debug_vis_info_logged = False

    self._human_ghost_warning_logged: bool = False
    # No-entity human ghost: a recolored copy of the standalone g1 model, driven
    # from motion data each frame and handed to `visualizer.add_ghost_mesh` -- the
    # SAME path the robot reference ghost uses, so it renders on every backend
    # (native viewer, offscreen recorder, viser). The visualizer builds the per-model MjData
    # matching whatever model it gets, so the human's smaller nq is fine; the model
    # + a reusable qpos buffer are the only state we keep.
    self._human_ref_ghost_model: mujoco.MjModel | None = None
    self._human_ref_ghost_qpos: np.ndarray | None = None

    # Resolve every metric body group ONCE at construct: an unknown group name
    # (or a member outside the model) fails the dry run here, not at the
    # first metric computation.
    for body_group in cfg.metric_body_groups:
      body_set_indices(cfg, body_group, self.robot.body_names)
    for metric_key in iter_metric_suite_keys(
      suite_names=cfg.metric_suites,
      reference_frame=cfg.metric_reference_frame,
      body_groups=cfg.metric_body_groups,
      compare=cfg.metric_compare,
    ):
      self.metrics[log_safe_metric_name(metric_key)] = torch.zeros(
        self.num_envs, device=self.device
      )
    # `self.metrics` is the per-step snapshot the eval recorders read; nothing
    # about it reaches the training log (see `reset`).

  def _validate_and_log_human_source_diagnostics(self) -> None:
    """Validate body-count parity between the human source(s) and motion files,
    then emit a one-time diagnostic block covering both entity-mode and
    motion-mode indexing.

    Always runs (regardless of `cfg.load_human_entity`). The point is to make
    indexing drift visible at a glance -- including future changes that compose
    specs differently or change `Entity.body_names` semantics.
    """
    if not self._human_motions:
      print(
        "[Codancing human-source] partner-free mode: no human reference "
        "(pure robot imitation)."
      )
      return
    cfg = self.cfg
    n_spec = len(self._human_body_names)

    # Per-loader shape information & parity check.
    motion_shapes: list[tuple[str, tuple[int, ...] | None]] = []
    for i, loader in enumerate(self._human_motions):
      if loader.body_pos_w is not None:
        shape = tuple(loader.body_pos_w.shape)
        motion_shapes.append((f"pair[{i}]", shape))
        if shape[1] != n_spec:
          path = self._clips[i].human_motion_file  # type: ignore[union-attr]
          raise ValueError(
            f"[Codancing human-source] Body count mismatch for {path!r}: "
            f"motion.body_pos_w has {shape[1]} bodies but the human spec "
            f"(after dropping worldbody) has {n_spec}. "
            f"Re-export the motion file against the current g1.xml."
          )
      else:
        motion_shapes.append((f"pair[{i}]", None))
        if not cfg.load_human_entity:
          path = self._clips[i].human_motion_file  # type: ignore[union-attr]
          raise ValueError(
            f"[Codancing human-source] load_human_entity=False requires "
            f"body_pos_w in the motion file, but pair {i} ({path!r}) has none."
          )

    resolved_indices = (
      f"pelvis={self._human_pelvis_index} "
      f"({self._human_body_names[self._human_pelvis_index]!r}), "
      f"left_hand={self._human_left_hand_index} "
      f"({self._human_body_names[self._human_left_hand_index]!r}), "
      f"right_hand={self._human_right_hand_index} "
      f"({self._human_body_names[self._human_right_hand_index]!r}), "
      f"left_foot={self._human_left_foot_index} "
      f"({self._human_body_names[self._human_left_foot_index]!r}), "
      f"right_foot={self._human_right_foot_index} "
      f"({self._human_body_names[self._human_right_foot_index]!r})"
    )

    lines: list[str] = ["[Codancing human-source diagnostics] (logged once)"]
    lines.append(f"  load_human_entity: {cfg.load_human_entity}")

    # Entity-side indexing.
    if self.human is not None:
      entity_body_names = self.human.body_names
      entity_body_ids = self.human.indexing.body_ids.tolist()
      lines.append("  -- Entity-side indexing --")
      lines.append("    entity.spec is per-entity (NOT composed scene): True")
      lines.append(
        f"    entity.body_names (local, len={len(entity_body_names)}): "
        f"{list(entity_body_names)}"
      )
      probe = self._human_pelvis_index
      lines.append(
        f"    entity.indexing.body_ids[pelvis={probe}] "
        f"= global={entity_body_ids[probe]} "
        f"(name in composed scene: "
        f"{self._env.sim.mj_model.body(entity_body_ids[probe]).name!r})"
      )
      lines.append(f"    resolved local indices (entity-mode): {resolved_indices}")
    else:
      lines.append("  -- Entity-side indexing -- n/a (entity not loaded)")

    # Motion-source indexing (always printed).
    lines.append("  -- Motion-source indexing --")
    lines.append(
      "    body_names_source: "
      + (
        "g1_constants.get_spec (worldbody dropped at [1:])"
        if not cfg.load_human_entity
        else "self.human.body_names (entity per-entity spec, worldbody already dropped)"
      )
    )
    lines.append(f"    human_body_names (len={n_spec}): {list(self._human_body_names)}")
    for label, shape in motion_shapes:
      lines.append(f"    motion file body_pos_w shape ({label}): {shape}")
    lines.append(f"    resolved local indices (motion-mode): {resolved_indices}")

    # Cross-check.
    lines.append("  -- Cross-check --")
    motion_n_match = all(
      shape is not None and shape[1] == n_spec for _, shape in motion_shapes
    )
    lines.append(f"    n_spec == n_motion (all pairs): {motion_n_match}")
    if self.human is not None:
      n_entity = len(self.human.body_names)
      lines.append(f"    n_spec == n_entity (entity loaded): {n_spec == n_entity}")
    else:
      lines.append("    n_spec == n_entity: n/a (entity not loaded)")
    lines.append(
      "  IMPORTANT: index lookup assumes the motion file's body order matches "
      "g1.xml's body order at retarget time. If you see odd reward values or "
      "visual offsets, re-export the motion file from the current g1.xml."
    )
    print("\n".join(lines))

  @property
  def command(self) -> torch.Tensor:
    joint_pos, joint_vel = self._gather_human_joint_state()
    return torch.cat([joint_pos, joint_vel], dim=1)

  @property
  def active_clip_obs(self) -> torch.Tensor | None:
    """Optional observation of active motion pair index per env.

    The one-hot width spans ALL clips, including ``trackable: false`` style
    corpus entries -- selection never lands on those, so their columns are
    simply never hot (harmless dead columns).
    """
    if self.cfg.active_clip_obs_mode == "none":
      return None
    # Pair count is the robot/clip count (one loader per clip), not the human
    # one -- robot-only multi-clip mode has no human loaders.
    if self.cfg.active_clip_obs_mode == "one_hot":
      n = len(self._robot_motions)
      return torch.nn.functional.one_hot(self._motion_id, n).float()
    if self.cfg.active_clip_obs_mode == "scalar":
      n = len(self._robot_motions)
      return (self._motion_id.float() / max(n - 1, 1)).unsqueeze(-1)
    return None

  # -- Unified human-source accessors ---------------------------------------
  #
  # Coordinate-frame contract: the entity's `body_link_pos_w` includes env-
  # origin offsets (the entity is placed in the composed scene), while the
  # motion file's `body_pos_w` was recorded for a single env and does not.
  # Motion-mode pos reads therefore add `_ref_origins` (= env origin + the
  # chain offset); quaternions and velocities are translation-invariant.

  @property
  def _ref_origins(self) -> torch.Tensor:
    """World origin of the reference stream per env ``(N, 3)``: the scene env
    origin plus the clip-end chain offset. EVERY read that places clip data
    (robot reference, human ghost, contact setpoints) in the world
    adds this, so one switch moves the whole stream together; the robot's own
    scene spawn keeps `_env_origins`."""
    return self._env_origins + self._ref_anchor_offset

  _HUMAN_BODY_PARTS = {
    "pelvis": "_human_pelvis_index",
    "left_hand": "_human_left_hand_index",
    "right_hand": "_human_right_hand_index",
    "left_foot": "_human_left_foot_index",
    "right_foot": "_human_right_foot_index",
  }

  def _resolve_human_source(self, source: HumanSource) -> Literal["entity", "motion"]:
    if source == "auto":
      return "entity" if self.cfg.load_human_entity else "motion"
    if source == "entity" and self.human is None:
      raise RuntimeError(
        "human_source='entity' but the human entity is not loaded "
        "(cfg.load_human_entity=False). Use source='motion' or source='auto'."
      )
    return source

  def _gather_human_body_attr(self, attr_name: str, body_idx: int) -> torch.Tensor:
    """One human body's `body_*_w` at the cursor (full env). Does NOT add env
    origins; callers that need world-frame positions add `_env_origins`.
    """
    raw = getattr(self._human_ref_clip_frames(), attr_name)
    assert raw is not None, (
      f"registry missing `{attr_name}`; load_human_entity=False requires "
      "retarget motion files with body_*."
    )
    return raw[:, body_idx]

  def _get_human_field(
    self, field: str, body_idx: int, source: HumanSource
  ) -> torch.Tensor:
    """Read one human body's `pos_w / quat_w / lin_vel_w / ang_vel_w` from the
    selected source. Adds env origins to positions in motion mode (see contract
    note above)."""
    if self._resolve_human_source(source) == "entity":
      assert self.human is not None
      return getattr(self.human.data, f"body_link_{field}")[:, body_idx]
    motion = self._gather_human_body_attr(f"body_{field}", body_idx)
    return motion + self._ref_origins if field == "pos_w" else motion

  def _get_human_field_for_part(
    self, part: str, field: str, source: HumanSource
  ) -> torch.Tensor:
    return self._get_human_field(
      field, getattr(self, self._HUMAN_BODY_PARTS[part]), source
    )

  # Public per-part accessors used by mdp.{observations,rewards,terminations}
  # via `command.get_human_<part>_<field>_w(source=...)`. Callers may override
  # the source per-term with `params={"human_source": "motion"}` for ablations.
  # The human has NO anchor concept: an observation names the bodies it reads
  # per term (`human_body_*_b` with explicit `body_names`, e.g. a
  # `${body_sets.human_*}` set); rewards resolve one body by name.

  def get_human_left_foot_pos_w(self, source: HumanSource = "auto") -> torch.Tensor:
    return self._get_human_field_for_part("left_foot", "pos_w", source)

  def get_human_right_foot_pos_w(self, source: HumanSource = "auto") -> torch.Tensor:
    return self._get_human_field_for_part("right_foot", "pos_w", source)

  def get_human_body_pos_w(
    self,
    body_indices: Sequence[int] | torch.Tensor,
    source: HumanSource = "auto",
  ) -> torch.Tensor:
    """Positions of arbitrary human bodies as `(num_envs, len(indices), 3)`.
    Indices are local to `self.human_body_names`.
    """
    return self._stack_human_bodies("pos_w", body_indices, source)

  def get_human_body_quat_w(
    self,
    body_indices: Sequence[int] | torch.Tensor,
    source: HumanSource = "auto",
  ) -> torch.Tensor:
    """Quaternions of arbitrary human bodies as `(num_envs, len(indices), 4)`."""
    return self._stack_human_bodies("quat_w", body_indices, source)

  def _stack_human_bodies(
    self,
    field: str,
    body_indices: Sequence[int] | torch.Tensor,
    source: HumanSource,
  ) -> torch.Tensor:
    if isinstance(body_indices, torch.Tensor):
      idx_list = body_indices.tolist()
    else:
      idx_list = list(body_indices)
    if self._resolve_human_source(source) == "entity":
      assert self.human is not None
      return getattr(self.human.data, f"body_link_{field}")[:, idx_list, :]
    return torch.stack(
      [self._get_human_field(field, int(i), source) for i in idx_list], dim=1
    )

  @property
  def human_body_names(self) -> tuple[str, ...]:
    return self._human_body_names

  _HUMAN_TRACKED_PARTS = (
    "pelvis",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
  )

  def _get_human_tracked_body_pose_w(
    self, source: HumanSource
  ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Poses + labels for the human coord-frame viz: the named parts."""
    indices = [
      getattr(self, self._HUMAN_BODY_PARTS[p]) for p in self._HUMAN_TRACKED_PARTS
    ]
    pos = self._stack_human_bodies("pos_w", indices, source)
    quat = self._stack_human_bodies("quat_w", indices, source)
    return pos, quat, self._HUMAN_TRACKED_PARTS

  # The feet at the default source ("auto"), read by the foot-follow rewards and
  # metrics. Terms that need another source call `get_human_*_foot_pos_w`.

  @property
  def human_left_foot_pos_w(self) -> torch.Tensor:
    return self.get_human_left_foot_pos_w()

  @property
  def human_right_foot_pos_w(self) -> torch.Tensor:
    return self.get_human_right_foot_pos_w()

  @property
  def pose_match_anchor_live_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self._pose_match_anchor_index]

  @property
  def pose_match_anchor_live_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self._pose_match_anchor_index]

  @property
  def active_disc_group(self) -> torch.Tensor:
    """Per-env discriminator-group index, shape ``(num_envs,)`` long.

    Read by the ``amp_group`` obs term to route AMP windows to the right
    discriminator group in the task-blind rsl_rl AMP API. The mapping
    follows ``amp_feature.disc_mode``: ``single`` pools every motion into group 0;
    ``multi_head``/``separate`` route through the per-clip ``disc_group`` LUT
    (default: group == clip index), matching the expert pooling from
    ``build_amp_data``.
    """
    feat = self.cfg.amp_feature
    if feat is None or feat.disc_mode == "single":
      return torch.zeros_like(self._motion_id)
    return self._disc_group_of_clip[self._motion_id]

  # The `robot_body_*_w` family is the LIVE robot, full body set in MODEL
  # order: the same columns as every robot_ref_* tensor and `robot.data`.
  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w

  @property
  def robot_left_foot_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self._robot_left_foot_index]

  @property
  def robot_right_foot_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self._robot_right_foot_index]

  @property
  def robot_ref_joint_pos(self) -> torch.Tensor:
    return self._gather_robot_motion_tensor("joint_pos")

  @property
  def robot_ref_joint_state(self) -> torch.Tensor:
    joint_pos = self._gather_robot_motion_tensor("joint_pos")
    joint_vel = self._gather_robot_motion_tensor("joint_vel")
    return torch.cat([joint_pos, joint_vel], dim=1)

  # -- Face-selectable reads -------------------------------------------------
  # Every `robot_ref_*` property reads the SERVED face (the tracking target;
  # for augmented pools that is the ADAPTED clip) -- rewards, metrics, events,
  # and visualization stay there. Observation terms that declare a face call
  # these instead; `ref_face: free` gives SoftMimic's actor view (the ORIGINAL
  # reference) while everything else keeps tracking the adapted one.

  def _body_set_joint_ids(self, body_set: str) -> torch.Tensor:
    """Entity joint indices for the joints OWNED by the named set's bodies.

    Each body contributes the joints attached to it (its ``body_jntadr``
    entries; the root's free joint never counts), so a sparse set gives the
    sparse joints and a set listing every body of a region gives all of its
    joints.
    Resolved against a standalone compile of the robot spec (unprefixed
    names), returned in entity joint order, memoized per set name
    (build-once).
    """
    cached = self._body_set_joint_id_table.get(body_set)
    if cached is not None:
      return cached
    body_names = body_set_names(self.cfg, body_set)
    model = self._standalone_robot_model
    if model is None:
      model = self.robot.cfg.spec_fn().compile()
      self._standalone_robot_model = model
    joint_names = list(self.robot.joint_names)
    ids: list[int] = []
    for name in body_names:
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
      assert bid >= 0, f"body set {body_set!r} body {name!r} not in the model"
      for offset in range(int(model.body_jntnum[bid])):
        jid = int(model.body_jntadr[bid]) + offset
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
          continue
        ids.append(joint_names.index(model.joint(jid).name))
    result = torch.tensor(sorted(ids), dtype=torch.long, device=self.device)
    self._body_set_joint_id_table[body_set] = result
    return result

  def robot_ref_joint_state_for(
    self, face: str = "served", body_set: str | None = None
  ) -> torch.Tensor:
    """Joint pos + vel of the requested reference face at the cursor.

    ``body_set`` None (default) keeps the whole joint vector; a registry set
    name narrows both halves to the joints its bodies own (an index slice; the
    faces are computed unconditionally anyway).
    """
    joint_pos = self._gather_robot_motion_tensor("joint_pos", face=face)
    joint_vel = self._gather_robot_motion_tensor("joint_vel", face=face)
    if body_set is not None:
      joint_ids = self._body_set_joint_ids(body_set)
      joint_pos = joint_pos[:, joint_ids]
      joint_vel = joint_vel[:, joint_ids]
    return torch.cat([joint_pos, joint_vel], dim=1)

  def robot_ref_joint_pos_for(
    self, face: str = "served", body_set: str | None = None
  ) -> torch.Tensor:
    """Joint positions only of the requested reference face (no velocity half)."""
    joint_pos = self._gather_robot_motion_tensor("joint_pos", face=face)
    if body_set is not None:
      joint_pos = joint_pos[:, self._body_set_joint_ids(body_set)]
    return joint_pos

  def pose_match_anchor_pose_w_for(
    self, face: str = "served"
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor body world pose (pos, quat) of the requested reference face."""
    pos = (
      self._gather_robot_motion_tensor("body_pos_w", face=face)
      + self._ref_origins[:, None, :]
    )[:, self._pose_match_anchor_index]
    quat = self._gather_robot_motion_tensor("body_quat_w", face=face)[
      :, self._pose_match_anchor_index
    ]
    return pos, quat

  @property
  def robot_ref_body_pos_w(self) -> torch.Tensor:
    return (
      self._gather_robot_motion_tensor("body_pos_w") + self._ref_origins[:, None, :]
    )

  @property
  def robot_ref_body_quat_w(self) -> torch.Tensor:
    return self._gather_robot_motion_tensor("body_quat_w")

  @property
  def robot_ref_body_lin_vel_w(self) -> torch.Tensor:
    return self._gather_robot_motion_tensor("body_lin_vel_w")

  @property
  def robot_ref_body_ang_vel_w(self) -> torch.Tensor:
    return self._gather_robot_motion_tensor("body_ang_vel_w")

  @property
  def pose_match_anchor_ref_pos_w(self) -> torch.Tensor:
    return self.robot_ref_body_pos_w[:, self._pose_match_anchor_index]

  @property
  def pose_match_anchor_ref_quat_w(self) -> torch.Tensor:
    return self.robot_ref_body_quat_w[:, self._pose_match_anchor_index]

  # -- Cursor reads via the flat-pool registries (one (clip, time) cursor) -----

  def _robot_ref_clip_frames(self) -> MotionFrames:
    """Full-env robot reference frames at the cursor (cached, invalidated on move)."""
    if self._robot_ref_cache is None:
      self._robot_ref_cache = self._reference_provider.get_reference(
        self._motion_id, self._motion_time
      )
    return self._robot_ref_cache

  def _robot_free_clip_frames(self) -> MotionFrames:
    """Full-env FREE (original) face frames at the cursor (cached like served).

    The face knob: augmented pools carry two poses per frame, the
    served ADAPTED face (tracking target) and the FREE face (the un-adapted
    original). Observation terms that declare ``ref_face: free`` read this one;
    everything else keeps the served face.
    """
    if self._contact_provider is None:
      raise ValueError(
        "ref_face='free' needs augmented clips: every clip must carry "
        "free_motion_file + contact_file (they build the "
        "ContactReferenceProvider). This pool has none."
      )
    cache = self._robot_free_cache
    if cache is None:
      cache = self._contact_provider.get_free(self._motion_id, self._motion_time)
      self._robot_free_cache = cache
    return cache

  def _human_ref_clip_frames(self) -> MotionFrames:
    """Full-env human reference frames at the cursor (cached, invalidated on move)."""
    assert self._human_registry is not None
    if self._human_ref_cache is None:
      self._human_ref_cache = self._human_registry.get_frames(
        self._motion_id, self._motion_time
      )
    return self._human_ref_cache

  def _invalidate_per_step_caches(self) -> None:
    """Clear EVERY per-step cache (the `*_cache` tier): the three cursor-keyed
    frame caches and the two compliance-signal caches. Call after anything
    that moves the cursor or rewrites the robot state mid-step. The caches
    hold RAW clip frames (origins and the chain offset are added at the
    read sites), so the offset write needs no extra invalidation; a future
    world-frame cache must key on `_ref_anchor_offset`."""
    self._robot_ref_cache = None
    self._robot_free_cache = None
    self._human_ref_cache = None
    self._compliance_multi_cache = None

  def compliance_signals_multi(self) -> ComplianceSignalsMulti | None:
    """multi-link per-step compliance quantities (shared by the _multi obs / rewards /
    metrics, cached per step).

    None without a contact provider or when it has no slot link names (the
    multi-link force event needs them). The applied wrench is read per slot from
    the sim's world-frame buffer (slot links are pairwise distinct -> exact
    per-slot reads, no double counting); the desired wrench is re-anchored with
    each slot's frozen anchors. Read **after** the force event, so what you see is
    what is applied."""
    prov = self._contact_provider
    if prov is None or getattr(prov, "_slot_link_names", None) is None:
      return None
    if self._compliance_multi_cache is not None:
      return self._compliance_multi_cache
    from mjlab.tasks.codancing.motion.contact_reference import delta_quat

    mid, mt = self._motion_id, self._motion_time
    cs = prov.contact_state_multi(mid, mt)  # (N,K,·)
    cur_events = prov.event_indices_at(mid, mt)  # (N,K)
    if self._compliance_slot_body_ids is None:
      self._compliance_slot_body_ids = prov.slot_link_robot_ids(
        list(self.robot.body_names)
      )
    sbi = self._compliance_slot_body_ids  # (K,)
    assert sbi is not None
    n, k = cur_events.shape
    ar = torch.arange(n, device=self.device).view(-1, 1)  # (N,1)
    body_id = sbi.view(1, -1).expand(n, k)  # (N,K) fixed body per slot
    in_event = (cur_events >= 0) & (body_id >= 0)  # (N,K)
    gate = in_event.unsqueeze(-1).float()  # (N,K,1)
    bidx = body_id.clamp(min=0)  # (N,K)
    # Applied wrench: read per slot from the sim's world-frame buffer (written by the
    # force event this step); zeroed outside events.
    applied_force = self.robot.data.body_external_force[ar, bidx] * gate  # (N,K,3)
    applied_torque = self.robot.data.body_external_torque[ar, bidx] * gate
    # Desired (scripted) wrench: re-anchored per slot with each slot's frozen anchors
    # (the same rotation knob as the force law / viz).
    yd = delta_quat(
      self._ff_multi_robot_anchor_quat_w,  # (N,K,4)
      self._ff_multi_ref_anchor_quat_w,
      self._ff_reanchor_rotation,
    )
    desired_force = quat_apply(yd, cs.force) * gate  # (N,K,3)
    desired_torque = quat_apply(yd, cs.torque) * gate
    sig = ComplianceSignalsMulti(
      in_event=in_event,
      active=cs.active,
      phase=cs.phase,
      robot_stiffness=cs.robot_stiffness,
      robot_rot_stiffness=cs.robot_rot_stiffness,
      forcefield_stiffness=cs.forcefield_stiffness,
      forcefield_rot_stiffness=cs.forcefield_rot_stiffness,
      applied_force=applied_force,
      applied_torque=applied_torque,
      desired_force=desired_force,
      desired_torque=desired_torque,
    )
    self._compliance_multi_cache = sig
    return sig

  def compliance_marks(self, env_idx: int = 0) -> list[ComplianceMark]:
    """Per-slot contact marks of one env in this frame: the read path of the
    force visualization (a single path for every K>=1).

    Records are emitted only for active slots (an empty list outside events). The
    script setpoint is read straight from the baked channel, **not** back-derived
    as ``ref_link + F/K``: the reference face is the ADAPTED face (already yielded
    by F/K_rob), so back-deriving would add a whole extra yield (3 to 5 cm) and put
    the sphere beyond the point the spring actually pulls toward.
    """
    prov = self._contact_provider
    if prov is None:
      return []
    from mjlab.tasks.codancing.motion.contact_reference import reanchor_points

    sl = slice(env_idx, env_idx + 1)
    cs, events = prov.contact_state_slots(self._motion_id[sl], self._motion_time[sl])
    k = int(events.shape[1])
    if k == 0 or not bool((events >= 0).any()):
      return []
    origin = self._ref_origins[sl].unsqueeze(1)  # (1,1,3)
    set_w = cs.setpoint_pos + origin  # script (data) face: clip frame -> world
    rot = self._ff_reanchor_rotation
    # rise face = the copy **stashed** by the force event this step, used as is. It
    # is already in the world frame, so the env origin is **not added again**
    # (that would be off by one env origin with several envs). Without a force
    # event the whole face is None.
    rise = self._ff_setpoint_w[sl] if self._ff_setpoint_valid else None
    # per_frame is a visualization-only comparison face (the physics does not use
    # it), so this face has to be recomputed.
    cur = tuple(t[sl].unsqueeze(1).expand(-1, k, -1) for t in self.anchor_pose("wrench_match_anchor_per_frame"))  # fmt: skip
    per_frame = reanchor_points(set_w, cur[0], cur[1], cur[2], cur[3], rot)

    robot_names = list(self.robot.body_names)
    marks: list[ComplianceMark] = []
    for s in range(k):
      ev = int(events[0, s].item())
      if ev < 0:
        continue
      names = prov.event_link_names
      name = names[ev] if 0 <= ev < len(names) else ""
      ref_link = live_link = applied = None
      if name in robot_names:
        body_id = robot_names.index(name)
        ref_link = self.robot_ref_body_pos_w[env_idx, body_id].cpu().numpy()
        live_link = self.robot.data.body_link_pos_w[env_idx, body_id].cpu().numpy()
        applied = self.robot.data.body_external_force[env_idx, body_id].cpu().numpy()
      marks.append(
        ComplianceMark(
          slot=s,
          k_env=float(cs.forcefield_stiffness[0, s].item()),
          script_force=cs.force[0, s].cpu().numpy(),
          script_setpoint=set_w[0, s].cpu().numpy(),
          setpoint_rise=None if rise is None else rise[0, s].cpu().numpy(),
          setpoint_per_frame=per_frame[0, s].cpu().numpy(),
          ref_link=ref_link,
          live_link=live_link,
          applied_force=applied,
        )
      )
    return marks

  # -- Episode reset and the logged metric means ----------------------------

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    # A new episode never starts paused.
    self._pause_steps_left[env_ids] = 0
    # Clear the online force-field re-anchoring: -1 makes an env restarted
    # mid-event re-freeze its anchors at the next rising edge (native reset hook).
    self._ff_multi_event_id[env_ids] = -1  # the per-slot re-anchoring too
    # Mark the true episode reset: `_resample_command` also fires on the
    # mid-episode clip-end chain, which never re-places the robot and never
    # perturbs.
    self._is_episode_reset = True
    try:
      super().reset(env_ids)
    finally:
      self._is_episode_reset = False
    # The base class would report the termination-step means of `self.metrics`
    # as `Metrics/codancing/*` training curves. The metrics are an evaluation
    # tool (the eval recorders read `self.metrics` every step and the offline
    # analyzer summarizes them), so training logs none of them.
    return {}

  def _gather_human_root_state(
    self, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Human root state ``(N, 13)`` at the cursor (subset when ``env_ids`` given)."""
    assert self._human_registry is not None
    if env_ids is None:
      return self._human_ref_clip_frames().root_state()
    return self._human_registry.get_frames(
      self._motion_id[env_ids], self._motion_time[env_ids]
    ).root_state()

  def _chain_anchor_pos(self, env_ids: torch.Tensor) -> torch.Tensor:
    """RAW (no origins) root position ``(n, 3)`` of the body the clip-end
    chain keeps continuous: the human when the pool is paired (its ghost
    is what the actor observes), else the robot reference. Same rule as the
    clip-end clock in `_update_command`. Raw on purpose: the before/after
    reads carry the same offset, so only their difference matters."""
    if self._human_registry is not None:
      return self._gather_human_root_state(env_ids)[:, :3]
    return self._reference_provider.get_reference(
      self._motion_id[env_ids], self._motion_time[env_ids]
    ).root_pos

  def _gather_human_joint_state(
    self, env_ids: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    assert self._human_registry is not None
    frames = (
      self._human_ref_clip_frames()
      if env_ids is None
      else self._human_registry.get_frames(
        self._motion_id[env_ids], self._motion_time[env_ids]
      )
    )
    return frames.joint_pos, frames.joint_vel

  def _gather_robot_motion_tensor(
    self,
    attr_name: str,
    env_ids: torch.Tensor | None = None,
    face: str = "served",
  ) -> torch.Tensor:
    """Robot reference at the cursor: joint state, or a sliced body-group view.

    ``face`` picks the pose face behind the read: "served" (default) is what
    the provider serves as the tracking target (the ADAPTED clip for augmented
    pools); "free" is the ORIGINAL un-adapted face (augmented pools only) and
    is full-env only (it backs observation terms, never per-env state writes).
    """
    if face == "free":
      assert env_ids is None, "free-face reads are full-env (observation) only."
      frames = self._robot_free_clip_frames()
    elif face == "served":
      frames = (
        self._robot_ref_clip_frames()
        if env_ids is None
        else self._robot_registry.get_frames(
          self._motion_id[env_ids], self._motion_time[env_ids]
        )
      )
    else:
      raise ValueError(f"Unknown reference face {face!r}; use 'served' or 'free'.")
    if attr_name in ("joint_pos", "joint_vel"):
      return getattr(frames, attr_name)
    assert attr_name in (
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    )
    raw = getattr(frames, attr_name)
    assert raw is not None, f"registry has no body arrays for `{attr_name}`."
    return raw  # full MODEL-order body set, no gather

  def _select_clips_for_resample(
    self, env_ids: torch.Tensor, sampling: ClipCursorSampleCfg | None
  ) -> None:
    """Assign clip indices for resampling envs (the moment's sampling block)."""
    # Key on the robot/clip count so selection works in robot-only multi-clip
    # mode (partner_mode=none has no human loaders).
    if len(self._robot_motions) <= 1 or sampling is None:
      return
    if sampling.clip_selection == "random":
      pick = torch.multinomial(
        self._clip_weights.expand(len(env_ids), -1),
        num_samples=1,
      ).squeeze(-1)
      self._motion_id[env_ids] = self._trackable_clip_ids[pick]
    elif sampling.clip_selection == "alternate":
      # Per-env round-robin (vectorized): advance each resampling env from its
      # own previous clip, cycling the TRACKED subset. Lockstep batches still
      # cycle in unison; scattered resets keep every env's ordered loop intact.
      self._motion_id[env_ids] = self._next_trackable_id[self._motion_id[env_ids]]
    elif sampling.clip_selection == "assign":
      # Enforced pool coverage: env i always plays tracked clip
      # (i mod N), so with num_envs a multiple of the pool size every clip
      # holds exactly num_envs/N envs and per-clip sample counts are exact.
      # The analyzer's coverage assertion is the check on the other end.
      self._motion_id[env_ids] = self._trackable_clip_ids[
        env_ids % len(self._trackable_clip_ids)
      ]
    else:  # "pin" -- validated at construct.
      assert sampling.pin_clip is not None
      self._motion_id[env_ids] = self._clip_index_by_name[sampling.pin_clip]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    cursor = self.cfg.motion_cursor
    sampling: ClipCursorSampleCfg | None = None
    if isinstance(cursor.source, ClipSourceCfg):
      sampling = (
        cursor.source.episode_reset
        if self._is_episode_reset
        else cursor.source.clip_end
      )
    if self._is_episode_reset:
      self._reset_episode(env_ids, sampling)
      return
    # Mid-episode clip end. `terminate` writes nothing: with the
    # motion_clip_end termination composed the episode truncates on the
    # `clip_just_ended` pulse; without it the cursor runs past the end and
    # `get_frames` clamps, so the reference freezes on the last frame (the
    # play-with-cleared-terminations case).
    if cursor.clip_end == "terminate":
      return
    self._chain(env_ids, sampling)

  def _reset_episode(
    self, env_ids: torch.Tensor, sampling: ClipCursorSampleCfg | None
  ) -> None:
    """The TRUE episode reset: note the failed episodes, pick the clip and the
    phase per `episode_reset`, write the state, perturb, forward."""
    cursor = self.cfg.motion_cursor
    if self._phase_sampler is not None:
      # The rsi sampler's failure-by-death-location table, fed BEFORE the clip
      # and time are overwritten. Only real episodes count: at the initial
      # env.reset() every env "resets" with a zero-length episode.
      real = self._env.episode_length_buf[env_ids] > 0
      failed = self._env.termination_manager.terminated[env_ids] & real
      if bool(failed.any()):
        self._phase_sampler.note_failures(
          self._motion_id[env_ids][failed], self._motion_time[env_ids][failed]
        )
    # A true episode reset re-anchors the reference stream to identity.
    self._ref_anchor_offset[env_ids] = 0.0
    self._select_clips_for_resample(env_ids, sampling)
    # Which envs take the reference pose (True) and which the default standing
    # pose (False) when the state is written below.
    on_reference = torch.full(
      (len(env_ids),),
      cursor.episode_reset in ("keyframe", "rsi"),
      dtype=torch.bool,
      device=self.device,
    )
    if cursor.episode_reset == "rsi":
      assert self._phase_sampler is not None
      t = self._phase_sampler.sample(self._motion_id[env_ids])
      sampled = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
      if cursor.rsi.start_prob > 0.0 or cursor.rsi.default_prob > 0.0:
        # Three-way mix per reset: `default_prob` of them start from the
        # default standing pose at phase 0, `start_prob` at phase 0 in the
        # reference pose, the rest at the sampled phase (reference pose).
        u = torch.rand(len(env_ids), device=self.device)
        at_default = u < cursor.rsi.default_prob
        at_start = ~at_default & (u < cursor.rsi.default_prob + cursor.rsi.start_prob)
        t = torch.where(at_default | at_start, torch.zeros_like(t), t)
        sampled = ~(at_default | at_start)
        on_reference = ~at_default
      self._motion_time[env_ids] = t
      self._rsi_spawn_sampled[env_ids] = sampled
    else:
      self._motion_time[env_ids] = 0.0
      self._rsi_spawn_sampled[env_ids] = False
    self._clip_end_latched[env_ids] = False
    self._rsi_spawn_time[env_ids] = self._motion_time[env_ids]
    self._invalidate_per_step_caches()
    # The cursor just moved, and on a reset `randomize_terrain` may have moved
    # the env to another patch entirely, so the cached ground shift now
    # describes the wrong place. Both writes below read `_ref_origins`, so
    # re-measure before them: leaving the step-start value in place put the
    # reset spawn off by a measured mean 2.08 cm / max 6.48 cm on 8 cm terrain,
    # most of the compensation's own magnitude.
    self._write_human_state(env_ids)
    base_pose = torch.zeros(len(env_ids), 7, device=self.device)
    base_vel = torch.zeros(len(env_ids), 6, device=self.device)
    for on_ref in (True, False):
      mask = on_reference if on_ref else ~on_reference
      if bool(mask.any()):
        base_pose[mask], base_vel[mask] = self._write_reset_state(
          env_ids[mask], on_reference=on_ref
        )
    offsets = apply_reset_perturbation(
      self.robot,
      env_ids,
      root_pose_range=cursor.root_pose_range,
      root_velocity_range=cursor.root_velocity_range,
      joint_position_range=cursor.joint_position_range,
      base_root_pose=base_pose,
      base_root_vel=base_vel,
    )
    self._reset_perturb_pose[env_ids] = offsets.get(
      "root_pose", torch.zeros(len(env_ids), 6, device=self.device)
    )
    joint_offsets = offsets.get("joint_pos")
    self._reset_perturb_joint_max[env_ids] = (
      joint_offsets.abs().amax(dim=1) if joint_offsets is not None else 0.0
    )
    # Actuator targets / internal state, after every write.
    self.robot.reset(env_ids=env_ids)
    self._env.sim.forward()
    # The state write moved robot and cursor; refresh the anchor-aligned
    # buffers NOW: the metric suite reads them on this very step, before
    # `_update_command` would rewrite them one step too late.
    self.update_relative_reference_frames()

  def _chain(self, env_ids: torch.Tensor, sampling: ClipCursorSampleCfg | None) -> None:
    """The clip-end `chain`: the robot is never moved; the incoming reference
    stream is translated in world xy so the continuity body (the human root
    when the pool is paired, else the robot reference root) continues from
    where the outgoing clip's last frame left it."""
    before = self._chain_anchor_pos(env_ids)
    self._select_clips_for_resample(env_ids, sampling)
    self._motion_time[env_ids] = 0.0
    self._clip_end_latched[env_ids] = False
    # Both reads are raw clip frames under the same (pre-chain) offset, so
    # their xy difference is exactly the translation that lands the new clip's
    # continuity body where the outgoing one stopped. z never enters.
    after = self._chain_anchor_pos(env_ids)
    self._ref_anchor_offset[env_ids, :2] += (before - after)[:, :2]
    self._invalidate_per_step_caches()
    self._write_human_state(env_ids)
    self.robot.reset(env_ids=env_ids)
    self._env.sim.forward()
    self.update_relative_reference_frames()

  def _advance_cursor(self) -> None:
    """Advance `motion_time` by one control step, honoring the pause counter.

    Paused envs hold their time -- a frozen
    cursor never crosses the clip end, so no clip-end resample fires while
    paused -- and the counter drains by one per step.
    """
    paused = self._pause_steps_left > 0
    self._motion_time += self._control_dt * (~paused).float()
    self._pause_steps_left[paused] -= 1

  def _body_anchor_pose(
    self, name: str
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The **current** two-sided pose of a body by name: (live pos, live quat,
    reference pos, reference quat), all envs (N,·). The pose-match anchor body goes
    through its dedicated properties (the same tensors, one name lookup); other
    bodies are resolved by name in both the live model and the reference body set
    (the SoftMimic variant, e.g. pelvis)."""
    if name == self.cfg.pose_match_anchor_body_name:
      return (
        self.pose_match_anchor_live_pos_w,
        self.pose_match_anchor_live_quat_w,
        self.pose_match_anchor_ref_pos_w,
        self.pose_match_anchor_ref_quat_w,
      )
    rb = self.robot.body_names.index(name)
    return (
      self.robot.data.body_link_pos_w[:, rb],
      self.robot.data.body_link_quat_w[:, rb],
      self.robot_ref_body_pos_w[:, rb],
      self.robot_ref_body_quat_w[:, rb],
    )

  def anchor_pose(
    self, spec: str
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve an anchor by purpose name: (live pos, live quat, reference pos,
    reference quat), all envs (N,·).

    The full catalog of anchors, each name saying which computation it serves; new
    consumers take their anchor from here instead of inventing their own lookup
    (get the anchor body wrong in one place and the force direction falls out of
    frame with the tracking reward):

    * ``pose_match_anchor``: the anchor of the per-body pose rewards
      (``pose_match_anchor_body_name``), current pose on both sides.
    * ``wrench_match_anchor_per_frame``: the force law's re-anchor point,
      re-capturing the current pose on **every active frame**; anchor body = the
      force event's ``anchor_body`` (empty = the pose-match anchor body). The
      anchors frozen at each slot's rising edge are ``frozen_anchor_slots``.
    * ``amp_feature_anchor``: the anchor body of the AMP feature frame
      (``amp_feature.anchor_body_name``), current pose on both sides.

    ``per_frame`` is also a value of the config's ``reanchor_mode: rising_edge |
    per_frame``, so the config value and the anchor it selects use the same
    word."""
    if spec == "pose_match_anchor":
      return self._body_anchor_pose(self.cfg.pose_match_anchor_body_name)
    if spec == "wrench_match_anchor_per_frame":
      return self._body_anchor_pose(
        self._ff_anchor_body or self.cfg.pose_match_anchor_body_name
      )
    if spec == "amp_feature_anchor":
      feat = self.cfg.amp_feature
      if feat is None:
        raise ValueError(
          "anchor_pose('amp_feature_anchor') needs cfg.amp_feature (no AMP "
          "wiring on this command)."
        )
      return self._body_anchor_pose(feat.anchor_body_name)
    raise ValueError(
      f"unknown anchor spec {spec!r}; use pose_match_anchor, "
      "wrench_match_anchor_per_frame or amp_feature_anchor."
    )

  def frozen_anchor_slots(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The anchors frozen at each slot's rising edge, (N,K,·): robot anchor
    position and quaternion, reference anchor position and quaternion.

    The same ``_ff_multi_*`` state **the force event wrote this step**, so the
    visualization re-anchors with exactly the anchors the physics used."""
    return (
      self._ff_multi_robot_anchor_pos_w,
      self._ff_multi_robot_anchor_quat_w,
      self._ff_multi_ref_anchor_pos_w,
      self._ff_multi_ref_anchor_quat_w,
    )

  def stash_forcefield_setpoint(self, setpoint_w: torch.Tensor) -> None:
    """The force event hands over the re-anchored setpoint (N,K,3) it **actually
    used** this step (world frame, env origin included).

    The visualization only reads this copy and never recomputes it, so "what is
    drawn is the point the physics pulls toward" holds by construction.
    **The reading side must never add the env origin again**: what is stored here
    is already in the world frame, and adding it again would be off by one env
    origin with multiple envs (invisible with a single env at the origin)."""
    self._ff_setpoint_w = setpoint_w
    self._ff_setpoint_valid = True

  def update_forcefield_anchor_multi(
    self,
    cur_events: torch.Tensor,  # (N,K)
    anchor_body: str,
    reanchor_rotation: str,
    reanchor_mode: str = "rising_edge",
  ) -> None:
    """Per-slot rising-edge re-anchoring of the multi-link online force field (per
    link: each slot freezes at its own rising edge, independently).

    The anchor body is shared across slots (default: the pose-match anchor body,
    (N,·) broadcast to (N,K,·)); the freeze timing comes from each slot's
    held-event change (cur_events != _ff_multi_event_id). Inactive slots (-1) are
    not written and their event_id goes to -1."""
    self._ff_reanchor_rotation = reanchor_rotation
    self._ff_anchor_body = anchor_body
    self._ff_reanchor_mode = reanchor_mode
    rpos, rquat, refpos, refquat = self.anchor_pose("wrench_match_anchor_per_frame")
    k = cur_events.shape[1]
    rpos_k = rpos.unsqueeze(1).expand(-1, k, -1)  # shared across slots -> (N,K,·)
    rquat_k = rquat.unsqueeze(1).expand(-1, k, -1)
    refpos_k = refpos.unsqueeze(1).expand(-1, k, -1)
    refquat_k = refquat.unsqueeze(1).expand(-1, k, -1)
    if reanchor_mode == "per_frame":
      need = cur_events >= 0
    else:
      need = (cur_events >= 0) & (cur_events != self._ff_multi_event_id)
    if bool(need.any()):
      self._ff_multi_robot_anchor_pos_w[need] = rpos_k[need]
      self._ff_multi_robot_anchor_quat_w[need] = rquat_k[need]
      self._ff_multi_ref_anchor_pos_w[need] = refpos_k[need]
      self._ff_multi_ref_anchor_quat_w[need] = refquat_k[need]
    self._ff_multi_event_id = torch.where(
      cur_events >= 0, cur_events, torch.full_like(cur_events, -1)
    )

  def update_relative_reference_frames(self) -> None:
    """Recompute the anchor-aligned reference buffers from the CURRENT state.

    These are plain attributes, not lazily derived: any state write that moves
    the robot or the cursor must re-run this, or the next reader compares
    against the pre-write anchor. Called every step from `_update_command`, and
    from `_resample_command` right after the reset state write, because the
    metric suite reads these buffers on the reset step BEFORE `_update_command`
    runs; without the reset-path call every clip's metric bucket eats one
    cross-episode sample at its own start. (Upstream's tracking task keeps the
    same second call site: `update_relative_body_poses` after `reset_to_frame`.)
    """
    (
      self.robot_ref_body_pos_relative_w,
      self.robot_ref_body_quat_relative_w,
    ) = self._anchor_aligned(
      self.robot_ref_body_pos_w,
      self.robot_ref_body_quat_w,
      self.pose_match_anchor_ref_pos_w,
      self.pose_match_anchor_ref_quat_w,
    )
    # FREE-face relative buffers, computed UNCONDITIONALLY on augmented pools
    # so no reader can exist without its buffer (see the __init__ note): the
    # anchor is the free face's OWN anchor body, so the transform also cancels
    # the augmenter's root shift.
    if self._contact_provider is not None:
      free_pos = (
        self._gather_robot_motion_tensor("body_pos_w", face="free")
        + self._ref_origins[:, None, :]
      )
      free_quat = self._gather_robot_motion_tensor("body_quat_w", face="free")
      (
        self.robot_free_body_pos_relative_w,
        self.robot_free_body_quat_relative_w,
      ) = self._anchor_aligned(
        free_pos,
        free_quat,
        free_pos[:, self._pose_match_anchor_index],
        free_quat[:, self._pose_match_anchor_index],
      )

  def _anchor_aligned(
    self,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map reference body poses into the anchor-aligned frame: the yaw-only
    rotation and xy translation that carry the given reference anchor onto the
    live robot anchor (z keeps the reference anchor's height)."""
    count = body_pos_w.shape[1]
    assert count > 0, "body count must be greater than 0"
    ref_pos = ref_anchor_pos_w[:, None, :].repeat(1, count, 1)
    ref_quat = ref_anchor_quat_w[:, None, :].repeat(1, count, 1)
    rob_pos = self.pose_match_anchor_live_pos_w[:, None, :].repeat(1, count, 1)
    rob_quat = self.pose_match_anchor_live_quat_w[:, None, :].repeat(1, count, 1)
    delta_pos_w = rob_pos
    delta_pos_w[..., 2] = ref_pos[..., 2]
    delta_ori_w = yaw_quat(quat_mul(rob_quat, quat_inv(ref_quat)))
    pos_rel = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - ref_pos)
    quat_rel = quat_mul(delta_ori_w, body_quat_w)
    return pos_rel, quat_rel

  def _update_command(self) -> None:
    # The clip-end pulse is a single step: clear last step's before detecting.
    self._clip_just_ended[:] = False
    # Resolve once: does a `motion_clip_end` termination reset us at clip-end?
    # (The termination manager exists by the first command compute.)
    if self._clip_end_resets is None:
      self._clip_end_resets = (
        "motion_clip_end" in self._env.termination_manager.active_terms
      )
    self._advance_cursor()

    # Per-env clip-end check against the active clip's frame count (the human
    # clip is the clock when present, else the robot reference): a clip plays
    # `num_frames` steps, then the env resamples.
    if self._human_registry is not None:
      num_frames = self._human_registry.motion_num_frames[self._motion_id].float()
    else:
      num_frames = self._reference_provider.num_frames(self._motion_id)
    end = num_frames * self._control_dt
    # Fire once per clip (the latch): under `terminate` with no truncation
    # composed the cursor keeps running past the end (the reference freezes on
    # the last frame), and a pulse re-firing every step would inflate every
    # clip-end consumer's count.
    ended = (self._motion_time >= end) & ~self._clip_end_latched
    env_ids = torch.where(ended)[0]
    if env_ids.numel() > 0:
      self._clip_end_latched[env_ids] = True
      self._clip_just_ended[env_ids] = True
      # When a `motion_clip_end` termination is active it truncates this step
      # (a clean boundary: the env's reset re-RSIs, cleared obs history), so the
      # cursor steps aside. Otherwise the `clip_end` mode decides: `chain`
      # resamples in-episode, `terminate` freezes on the last frame.
      if not self._clip_end_resets:
        self._resample_command(env_ids)
    self._invalidate_per_step_caches()

    self.update_relative_reference_frames()

    env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    self._write_human_state(env_ids)
    # Playback overwrote the human entity's qpos above, and mjlab forwards only
    # BEFORE command_manager.compute() (mj_step staleness), so without this the
    # live human-body observation (human_body_pos_b, critic)
    # would read the stale pre-write pose, lagging the playback by one step.
    # Refresh derived quantities here, command-side (like the _resample_command
    # forwards), so the mjlab env step needs no change.
    self._env.sim.forward()

  def _write_human_state(self, env_ids: torch.Tensor) -> None:
    # When the human entity is skipped (cfg.load_human_entity=False), motion
    # data is consumed directly via the unified `get_human_*` accessors; there
    # is no entity to write to, and callers need no check of their own.
    if self.human is None:
      return
    root_state = self._gather_human_root_state(env_ids).clone()
    root_state[:, :3] += self._ref_origins[env_ids]
    joint_pos, joint_vel = self._gather_human_joint_state(env_ids)
    write_state(
      self.human,
      EntityStateTarget(
        root_pose=root_state[:, :7],
        root_vel=root_state[:, 7:],
        joint_pos=joint_pos,
        joint_vel=joint_vel,
      ),
      env_ids,
    )

  def _write_reset_state(
    self, env_ids: torch.Tensor, on_reference: bool
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the robot's episode-reset state for `env_ids`; return the root
    pose and velocity written (the perturbation stacks on them, because live
    xpos reads are stale until the next forward).

    `on_reference=False` (`default`): the default pose at the scene
    placement. `on_reference=True` (`keyframe` / `rsi`): the reference at
    the cursor. In paired and imagined mode the root XY is the HUMAN root at
    the cursor plus the manifest `placement_offset` in BOTH cases (the same
    anchoring foot_follow enforces every step; the robot and human streams
    were retargeted separately, so their raw root delta is retargeting noise,
    measured drifting between -0.09 and -0.43 m in x on the 20260224_001
    pair), while z, orientation, joint state and velocities come from the
    robot reference. Solo pools have no human stream and take the full
    reference root. Command-driven, NOT a reset EventTerm, on purpose: the
    reference reads the per-env cursor sampled inside this command's reset,
    and mode="reset" events fire BEFORE command.reset (stale cursor) and
    never fire on the mid-episode clip-end chain.
    """
    root_state = self.robot.data.default_root_state[env_ids].clone()
    if not self._human_motions:
      # Partner-free mode: no human anchor to spawn relative to. The robot
      # reference is consumed anchor-relative (yaw-aligned deltas in
      # `_update_command`), so the env origin is the natural spawn.
      root_state[:, :2] += self._env_origins[env_ids, :2]
    else:
      human_root = self._gather_human_root_state(env_ids)[:, :3].clone()
      human_root += self._ref_origins[env_ids]
      offsets = self._clip_placement_offsets_t
      # Per-env offset from the active clip (one row -> broadcast).
      offset = offsets[self._motion_id[env_ids]] if len(offsets) > 1 else offsets[0]
      root_state[:, :2] = (human_root + offset)[:, :2]  # only set x and y
    if on_reference:
      ref = self._reference_provider.get_reference(
        self._motion_id[env_ids], self._motion_time[env_ids]
      )
      if self._human_motions:
        # XY keeps the human-anchored arrangement computed above; z carries
        # the reference height plus the stream origin's z (terrain shift).
        root_state[:, 2] = ref.root_pos[:, 2] + self._ref_origins[env_ids, 2]
      else:
        # Solo pool: the reference stream is the only world.
        root_state[:, :3] = ref.root_pos + self._ref_origins[env_ids]
      root_state[:, 3:7] = ref.root_quat
      root_state[:, 7:10] = ref.root_lin_vel
      root_state[:, 10:13] = ref.root_ang_vel
      joint_pos, joint_vel = ref.joint_pos.clone(), ref.joint_vel.clone()
    else:
      joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
      joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
    target = EntityStateTarget(root_pose=root_state[:, :7], root_vel=root_state[:, 7:])
    if self.robot.is_articulated:
      target.joint_pos, target.joint_vel = joint_pos, joint_vel
    write_state(self.robot, target, env_ids)
    return root_state[:, :7], root_state[:, 7:]

  def _update_metrics(self) -> None:
    metric_values = compute_metric_suite(
      command=self,
      suite_names=self.cfg.metric_suites,
      reference_frame=self.cfg.metric_reference_frame,
      body_groups=self.cfg.metric_body_groups,
      compare=self.cfg.metric_compare,
    )
    for metric_key, metric_value in metric_values.items():
      self.metrics[log_safe_metric_name(metric_key)] = metric_value
    if self._phase_sampler is not None:
      self._phase_sampler.decay()

  def clip_just_ended(self) -> torch.Tensor:
    """Per-env pulse, True only on the step a clip ends (every clip-end, whether
    it resamples in-episode or is reset by the `motion_clip_end` termination).
    Cleared at the start of the next update. The sole clip-end signal, read by
    the termination."""
    return self._clip_just_ended

  @property
  def active_clip_names(self) -> tuple[str, ...]:
    # One device sync for the whole batch: the eval recorder calls this every
    # step, and a per-env `.item()` loop would sync once per env.
    ids = self._motion_id.tolist()
    return tuple(self._clips[i].name or str(i) for i in ids)

  def _debug_vis_impl(self, visualizer) -> None:
    """Visualize the codancing reference: robot ghost/frames and/or human ghost.

    Each draw is gated by its own independent bool (``debug_vis_*_ref_ghost`` /
    ``debug_vis_*_ref_coord_frames``). Robot and human blocks are independent --
    turning robot off (both robot bools False) must not suppress the human, and
    vice versa.
    """
    if self._robot_registry.has_body:
      faces = self._drawable_faces()
      for face in faces:
        if face.ghost:
          self._draw_face_ghost(visualizer, face)

      if self._resolved_force_channels() and self._contact_provider is not None:
        self._draw_compliance_force_arrow(visualizer)

      layers = self._coord_frame_layers(faces)
      if layers:
        draw_coord_frames(visualizer, layers)

    # Human reference is independent of the robot-ref toggles. Partner-free mode
    # has no human reference; skip silently so a generic overlay stays harmless.
    if (
      self.cfg.debug_vis_human_ref_ghost or self.cfg.debug_vis_human_ref_coord_frames
    ) and self._human_motions:
      self._draw_human_ref_vis(visualizer)

  def _drawable_faces(self) -> list[_DrawableFace]:
    """Every drawable reference face, with its two representations resolved.

    A face is one pose of the tracked bodies. It can be drawn as a translucent
    GHOST (a whole recolored robot, readable when the faces sit close together)
    or as per-body coordinate FRAMES (exact, but 7 bodies x N faces of axis
    coordinate frames turns into hay once the offsets are small). Both come off the same
    buffers, so a face carries both and the two per-face toggles pick the
    representation. Adding a face is one entry here, not a new draw method.

    The transported faces are why a ghost works at all: the anchor map is a
    RIGID transform (yaw + xy, reference anchor's z), so carrying a face onto
    the live anchor moves its root and leaves every joint angle alone. Index 0
    of the body tensors is the root, so the transported root reads straight off
    the same relative buffers the frames use -- no second derivation, and the
    ghost cannot drift from the frames.
    """
    faces: list[_DrawableFace] = []
    ref_ghost_color = _ghost_rgba(
      self.cfg.debug_vis_robot_ref_ghost_color or self._robot_ref_ghost_color,
      "robot_ref_ghost_color",
    )

    def frame_idx(bodies: tuple[str, ...]) -> tuple[int, ...]:
      # Coord frames follow the face's ghost body list (model-order columns);
      # an empty list (whole model) draws every body's coord frame.
      names = self.robot.body_names
      if not bodies:
        return tuple(range(len(names)))
      return tuple(model_order_indexes(bodies, names))

    served_bodies = tuple(self.cfg.debug_vis_robot_ref_ghost_bodies)
    served_frame_idx = frame_idx(served_bodies)
    faces.append(
      _DrawableFace(
        name="reference",
        root_pos_w=self.robot_ref_body_pos_w[:, 0],
        root_quat_w=self.robot_ref_body_quat_w[:, 0],
        joint_pos=self.robot_ref_joint_pos,
        body_pos_w=self.robot_ref_body_pos_w,
        body_quat_w=self.robot_ref_body_quat_w,
        ghost_color=ref_ghost_color,
        ghost_bodies=served_bodies,
        coord_frame_body_indices=served_frame_idx,
        coord_axis_colors=_face_axes("reference"),
        coord_axis_length=viz_palette.FACE_STYLE["reference"].coord_axis_len,
        coord_axis_alpha=viz_palette.FACE_STYLE["reference"].coord_axis_alpha,
        body_styles=self._body_styles_for("reference"),
        ghost=self.cfg.debug_vis_robot_ref_ghost,
        coord_frames=self.cfg.debug_vis_robot_ref_coord_frames,
      )
    )
    # The ADAPTED face carried onto the LIVE anchor: BeyondMimic's "Desired
    # Body". Its gap to `reference` is the anchor error, and its gap to
    # `current` is the anchor-centered tracking error the reward sees.
    faces.append(
      _DrawableFace(
        name="desired",
        root_pos_w=self.robot_ref_body_pos_relative_w[:, 0],
        root_quat_w=self.robot_ref_body_quat_relative_w[:, 0],
        joint_pos=self.robot_ref_joint_pos,
        body_pos_w=self.robot_ref_body_pos_relative_w,
        body_quat_w=self.robot_ref_body_quat_relative_w,
        ghost_color=_ghost_rgba(
          self.cfg.robot_ref_desired_ghost_color, "robot_ref_desired_ghost_color"
        ),
        ghost_bodies=served_bodies,
        coord_frame_body_indices=served_frame_idx,
        coord_axis_colors=_face_axes("desired"),
        coord_axis_length=viz_palette.FACE_STYLE["desired"].coord_axis_len,
        coord_axis_alpha=viz_palette.FACE_STYLE["desired"].coord_axis_alpha,
        body_styles=self._body_styles_for("desired"),
        ghost=self.cfg.debug_vis_robot_ref_desired_ghost,
        coord_frames=self.cfg.debug_vis_robot_ref_desired_coord_frames,
      )
    )
    # FREE-face draws need the augmented clip; skip silently when a generic
    # overlay leaves them on against an unaugmented pool.
    if self._contact_provider is not None:
      free = self._robot_free_clip_frames()
      # The free ghost: SERVED root + FREE joints, so its gap to `reference` is
      # purely the joint-space yield (root shift understated). None = borrow
      # the served subset; an empty list is an explicit whole-model choice, so
      # test for None, not truth.
      free_bodies = (
        tuple(self.cfg.debug_vis_free_ref_ghost_bodies)
        if self.cfg.debug_vis_free_ref_ghost_bodies is not None
        else served_bodies
      )
      faces.append(
        _DrawableFace(
          name="free",
          root_pos_w=self.robot_ref_body_pos_w[:, 0],
          root_quat_w=self.robot_ref_body_quat_w[:, 0],
          joint_pos=free.joint_pos,
          body_pos_w=None,  # synthetic mix of two faces: no per-body truth
          body_quat_w=None,
          ghost_color=_ghost_rgba(
            self.cfg.debug_vis_free_ref_ghost_color or self.cfg.free_ref_ghost_color,
            "free_ref_ghost_color",
          ),
          ghost_bodies=free_bodies,
          coord_frame_body_indices=frame_idx(free_bodies),
          coord_axis_colors=_face_axes("free"),
          coord_axis_length=viz_palette.FACE_STYLE["free"].coord_axis_len,
          coord_axis_alpha=viz_palette.FACE_STYLE["free"].coord_axis_alpha,
          body_styles=self._body_styles_for("free"),
          ghost=self.cfg.debug_vis_free_ref_ghost,
          coord_frames=False,
        )
      )
      pos, quat = (
        self.robot_free_body_pos_relative_w,
        self.robot_free_body_quat_relative_w,
      )
      # An augmented pool computes these on every `update_relative_reference_
      # frames` (unconditionally), and the construct-time
      # resample runs before any draw -- so with a provider they exist.
      assert pos is not None and quat is not None, (
        "free-face relative buffers missing on an augmented pool; "
        "update_relative_reference_frames must have run before any draw."
      )
      faces.append(
        _DrawableFace(
          name="desired_free",
          root_pos_w=pos[:, 0],
          root_quat_w=quat[:, 0],
          joint_pos=free.joint_pos,
          body_pos_w=pos,
          body_quat_w=quat,
          ghost_color=_ghost_rgba(
            self.cfg.free_ref_desired_ghost_color, "free_ref_desired_ghost_color"
          ),
          ghost_bodies=served_bodies,
          coord_frame_body_indices=served_frame_idx,
          coord_axis_colors=_face_axes("desired_free"),
          coord_axis_length=viz_palette.FACE_STYLE["desired_free"].coord_axis_len,
          coord_axis_alpha=viz_palette.FACE_STYLE["desired_free"].coord_axis_alpha,
          body_styles=self._body_styles_for("desired_free"),
          ghost=self.cfg.debug_vis_free_ref_desired_ghost,
          coord_frames=self.cfg.debug_vis_free_ref_desired_coord_frames,
        )
      )
    elif (
      self.cfg.debug_vis_free_ref_desired_ghost
      or self.cfg.debug_vis_free_ref_desired_coord_frames
    ):
      raise ValueError(
        "session.debug_vis.free_ref_desired_ghost/coord_frames draw the FREE "
        "reference, which only exists on augmented clips (free_motion_file + "
        "contact_file); this manifest has none."
      )
    # The live robot: frames only, since the solid robot already IS its ghost.
    faces.append(
      _DrawableFace(
        name="current",
        root_pos_w=self.robot_body_pos_w[:, 0],
        root_quat_w=self.robot_body_quat_w[:, 0],
        joint_pos=self.robot.data.joint_pos,
        body_pos_w=self.robot_body_pos_w,
        body_quat_w=self.robot_body_quat_w,
        ghost_color=_face_ghost("current"),
        ghost_bodies=served_bodies,
        coord_frame_body_indices=served_frame_idx,
        coord_axis_colors=_face_axes("current"),  # None: visualizer default rgb
        coord_axis_length=viz_palette.FACE_STYLE["current"].coord_axis_len,
        coord_axis_alpha=viz_palette.FACE_STYLE["current"].coord_axis_alpha,
        body_styles=self._body_styles_for("current"),
        ghost=False,
        coord_frames=self.cfg.debug_vis_robot_ref_coord_frames,
      )
    )
    return faces

  def _body_styles_for(self, face_name: str) -> dict[str, BodyStyle]:
    """Per-body style overrides declared for one face, normalized to records.

    Config carries a nested dict (face name -> body name -> fields), so this is
    where a typo becomes a clear error instead of a silently unstyled body: an
    unknown face name or an unknown field name raises, since either one means a
    figure quietly renders without the highlight it asked for.
    """
    declared = self.cfg.debug_vis_body_styles
    if not declared:
      return {}
    unknown_faces = set(declared) - _DRAWABLE_FACE_NAMES
    if unknown_faces:
      raise ValueError(
        f"debug_vis.body_styles names unknown drawn references "
        f"{sorted(unknown_faces)}; valid names: {sorted(_DRAWABLE_FACE_NAMES)}"
      )
    fields = {"ghost_color", "coord_axis_colors", "coord_axis_alpha"}
    styles: dict[str, BodyStyle] = {}
    for body_name, raw in (declared.get(face_name) or {}).items():
      spec = dict(raw)
      unknown = set(spec) - fields
      if unknown:
        raise ValueError(
          f"debug_vis.body_styles[{face_name!r}][{body_name!r}] has unknown "
          f"keys {sorted(unknown)}; valid: {sorted(fields)}"
        )
      where = f"debug_vis.body_styles[{face_name!r}][{body_name!r}]"
      styles[body_name] = BodyStyle(
        ghost_color=resolve_color(spec.get("ghost_color"), f"{where}.ghost_color"),
        coord_axis_colors=_coord_axis_colors(
          spec.get("coord_axis_colors"), f"{where}.coord_axis_colors"
        ),
        coord_axis_alpha=(
          None
          if spec.get("coord_axis_alpha") is None
          else float(spec["coord_axis_alpha"])
        ),
      )
    return styles

  def _coord_frame_layers(self, faces: list[_DrawableFace]) -> list[CoordFrameLayer]:
    """Coord-frame layers for whichever faces asked to be drawn as coord frames.

    Each face carries its own coord-frame body indices, so a face can pair a
    full-mesh ghost with sparse frames. Axis lengths grow toward the live face
    so the stack stays legible when the faces coincide (an exact kinematic pin
    collapses them onto each other).
    """
    body_names = self.robot.body_names
    layers = [
      CoordFrameLayer(
        f.name,
        tuple(body_names[i] for i in f.coord_frame_body_indices),
        f.body_pos_w[:, list(f.coord_frame_body_indices)],
        f.body_quat_w[:, list(f.coord_frame_body_indices)],
        f.coord_axis_length,
        f.coord_axis_colors,
        f.coord_axis_alpha,
        {
          body: CoordFrameStyle(style.coord_axis_colors, style.coord_axis_alpha)
          for body, style in f.body_styles.items()
          if style.coord_axis_colors is not None or style.coord_axis_alpha is not None
        },
      )
      for f in faces
      if f.coord_frames and f.body_pos_w is not None and f.body_quat_w is not None
    ]
    # The tracking anchors ride with the base reference-vs-current draw: a body's
    # coord frame is only readable against the anchor it was measured from.
    if self.cfg.debug_vis_robot_ref_coord_frames:
      layers += [
        anchor_coord_frame(
          "reference",
          "pose_match_anchor",
          self.pose_match_anchor_ref_pos_w,
          self.pose_match_anchor_ref_quat_w,
          _anchor_axis_len("reference"),
          _face_axes("reference"),
        ),
        anchor_coord_frame(
          "current",
          "pose_match_anchor",
          self.pose_match_anchor_live_pos_w,
          self.pose_match_anchor_live_quat_w,
          _anchor_axis_len("current"),
        ),
      ]
    if layers and not self._debug_vis_info_logged:
      self._debug_vis_info_logged = True
      print(
        "[INFO] Codancing debug vis robot coord frames:",
        {
          "anchor": self.cfg.pose_match_anchor_body_name,
          "layers": {layer.name: list(layer.body_names) for layer in layers},
        },
      )
    return layers

  def _draw_face_ghost(self, visualizer, face: _DrawableFace) -> None:
    """Draw one face as a translucent ghost through ``add_ghost_mesh``.

    The ghost is a standalone g1 model compiled from ``self.robot.cfg.spec_fn()``
    (NOT a deepcopy of the composed scene), so it never passes the collision-viz
    spec_fn: its visual geoms stay in the default-visible groups and the
    backend's read-only ``MjvOption`` renders it as-is (no per-call geom-group
    fiddling). Driven from the face's root pose + joint angles via the
    standalone qpos layout (``[:3]`` root pos, ``[3:7]`` root quat, ``[7:]``
    joints) -- the same path the imagined-human ghost uses, so it renders
    identically on the native viewer, the offscreen recorder and viser.

    Models and qpos scratch are cached per face name: the compile is the
    expensive part, and a face's colors / body subset are static config.
    """
    cached = self._face_ghost_models.get(face.name)
    if cached is None:
      model = self._build_ghost_model(
        self.robot.cfg.spec_fn(),
        face.ghost_color,
        face.ghost_bodies,
        {
          body: style.ghost_color
          for body, style in face.body_styles.items()
          if style.ghost_color is not None
        },
      )
      cached = (model, np.zeros(model.nq))
      self._face_ghost_models[face.name] = cached
    model, qpos = cached
    env_idx = visualizer.env_idx
    # Positions already include the env origin (the reference buffers add it).
    qpos[:3] = face.root_pos_w[env_idx].cpu().numpy()
    qpos[3:7] = face.root_quat_w[env_idx].cpu().numpy()
    n_joint = model.nq - 7
    qpos[7 : 7 + n_joint] = face.joint_pos[env_idx].cpu().numpy()[:n_joint]
    visualizer.add_ghost_mesh(
      qpos, model, alpha=float(face.ghost_color[3]), label=face.name
    )

  def _resolved_force_channels(self) -> frozenset[str]:
    """The named force-viz channels this render should draw.

    An explicit `debug_vis.force_channels` list wins; None expands the two
    booleans: `compliance_force_arrow` -> the four base markers, `reanchor` ->
    the per-frame face, the data comparison arrow, the re-anchor transform
    extras and `reanchor_base_per_frame` (the per-frame twin of the FROZEN base
    ball and its connector, offset from the frozen one by however far
    per-frame re-anchoring has drifted). Every marker belongs to exactly one
    channel, so any subset composes a figure.
    """
    return viz_palette.expand_force_channels(
      force_arrow=self.cfg.debug_vis_compliance_force_arrow,
      reanchor=self.cfg.debug_vis_reanchor,
      explicit=self.cfg.debug_vis_force_channels,
    )

  def _draw_compliance_force_arrow(self, visualizer) -> None:
    """Draw the SoftMimic contact overlay, **per slot** (K>=1; single contact is just
    K=1, the same code).

    All quantities come from ``compliance_marks()`` (the only read path), so what
    is drawn is exactly what the physics uses. Each active contact draws two sets:

      * **reference (ghost) link**: the ``scripted_wrench`` arrow (script force
        ``cs.force``) + the ``setpoint_script`` sphere (the script setpoint, read
        straight from the baked channel, not back-derived as ``ref_link + F/K``:
        the reference face is the ADAPTED face already yielded by F/K_rob, so
        back-deriving would add a whole extra yield). Follows the cursor, clean,
        and correct regardless of how good the policy is. The forced link itself
        gets no marker sphere: the reference ghost's wrists are lit in
        the slot colors via ``body_styles`` in the config, so link identity is
        conveyed by color.
      * **live forced link**: the ``applied_wrench`` arrow (the applied force
        ``robot.data.body_external_force``, sim buffer, never recomputed) + the
        ``setpoint_active`` sphere (the **re-anchored setpoint** = the script
        setpoint re-anchored into the live frame with the frozen anchors
        ``_ff_*`` = the point the reactive spring pulls toward). The arrow points
        from the live link to that sphere, force = ``K_env·(setpoint−live)``. A
        zero-action policy **still receives the online force**: after the robot
        falls, the live link is far from the re-anchored setpoint -> a large
        force, a long arrow pointing at that sphere, obvious at a glance (which is
        exactly why it receives such a large force).

    Size/transparency contract (both halves live in ``viz_palette``: color names
    carry alpha, sizes are in ``SIZES``): **baked/un-re-anchored** faces
    (``scripted_wrench`` / ``setpoint_script`` / ``data_arrow``) are drawn **large
    and translucent**; **live/re-anchored** faces (``applied_wrench`` /
    ``setpoint_active`` / ``setpoint_per_frame`` and ``per_frame_arrow``) are drawn
    **thin and solid**. When the two families overlap (frames where the frozen
    anchor is the identity, rise == script), the thin solid stroke sits on top of
    the wide translucent one and both stay distinguishable.

    multi-link clips (both wrists pushed at once) draw two groups per frame,
    independently. For a well-trained policy live ≈ reference -> the two sets
    nearly coincide and can be compared directly. No-op outside events.
    """
    channels = self._resolved_force_channels()
    marks = self.compliance_marks(visualizer.env_idx)
    if channels & {"reanchor_transform", "reanchor_base_per_frame"}:
      self._draw_reanchor_frames(visualizer, marks, channels)
    rgba = viz_palette.MARKER_RGBA
    for m in marks:
      # Script force on the reference (ghost) link; the script setpoint is a world
      # point and does not depend on the ghost body set.
      if "scripted_wrench" in channels and m.ref_link is not None:
        self._draw_force_marker(
          visualizer,
          m.ref_link,
          m.script_setpoint,
          m.script_force,
          m.k_env,
          color=rgba["scripted_wrench"],
          width=_BAKED_ARROW_WIDTH,
          channel="scripted_wrench",
        )
      if "setpoint_script" in channels:
        visualizer.add_sphere(
          m.script_setpoint,
          self._setpoint_radius(_SCRIPT_SETPOINT_SPHERE_RADIUS),
          color=rgba["setpoint_script"],
          label="setpoint",
        )

      if m.live_link is None or m.applied_force is None:
        continue
      # Applied force on the live forced link + the re-anchored setpoint (where the
      # reactive spring pulls).
      rise = m.setpoint_rise
      if "applied_wrench" in channels:
        if rise is not None:
          self._draw_force_marker(
            visualizer,
            m.live_link,
            rise,
            m.applied_force,
            m.k_env,
            color=rgba["applied_wrench"],
            width=_LIVE_ARROW_WIDTH,
            channel="applied_wrench",
          )
        else:
          # No stashed setpoint (feedforward replay stashes none): the wrench
          # is still real, so draw it as a plain direction arrow from the live
          # link instead of vanishing; the applied wrench stays visible in
          # every response mode.
          visualizer.add_arrow(
            m.live_link,
            force_arrow_end(m.live_link, m.applied_force),
            color=rgba["applied_wrench"],
            width=_LIVE_ARROW_WIDTH,
            label="applied_wrench",
          )
      if "setpoint_active" in channels and rise is not None:
        visualizer.add_sphere(
          rise,
          self._setpoint_radius(_LIVE_SETPOINT_SPHERE_RADIUS),
          color=rgba["setpoint_active"],
          # The stashed face IS whatever the re-anchor mode produced; never
          # hardcode "rising edge", which would be false under per_frame.
          label=f"setpoint_{self._ff_reanchor_mode}",
        )

      # Three-face setpoint comparison (debug_vis_reanchor): next to active
      # (= `_ff_*`, by default the rising_edge anchors frozen at the rising edge),
      # give data (= the baked setpoint, **not re-anchored**, the same point as the
      # sphere above) and per_frame (= re-anchored with the **current** anchors)
      # one spring arrow `K_env·(setpoint−live)` each. Under perfect tracking the
      # re-anchoring is the identity -> three spheres and three arrows overlap;
      # when live departs from the reference, per_frame follows the drift every
      # frame, rising_edge stays locked at the rising edge, and data is not
      # re-anchored -> the three separate, and the gaps are the differences
      # between the three choices.
      if "data_arrow" in channels:
        f_data = m.k_env * (m.script_setpoint - m.live_link)
        self._draw_force_marker(
          visualizer,
          m.live_link,
          m.script_setpoint,
          f_data,
          m.k_env,
          color=rgba["data_arrow"],
          width=_BAKED_ARROW_WIDTH,
          channel="data_arrow",
        )
      if "setpoint_per_frame" in channels:
        visualizer.add_sphere(
          m.setpoint_per_frame,
          self._setpoint_radius(_LIVE_SETPOINT_SPHERE_RADIUS),
          color=rgba["setpoint_per_frame"],
          label="setpoint_per_frame",
        )
        f_pf = m.k_env * (m.setpoint_per_frame - m.live_link)
        self._draw_force_marker(
          visualizer,
          m.live_link,
          m.setpoint_per_frame,
          f_pf,
          m.k_env,
          color=rgba["per_frame_arrow"],
          width=_LIVE_ARROW_WIDTH,
          channel="per_frame_arrow",
        )

  def _spring_style(self) -> bool:
    return self.cfg.debug_vis_compliance_force_style == "spring"

  def _setpoint_radius(self, base: float) -> float:
    """Setpoint sphere radius: smaller in the spring style, where the sphere is only
    an end marker and the thin cylinder shows the spring geometry."""
    return base * _SPRING_BALL_SCALE if self._spring_style() else base

  def _draw_force_marker(
    self,
    visualizer,
    start: np.ndarray,
    setpoint: np.ndarray,
    force: np.ndarray,
    k_env: float,
    color: tuple[float, float, float, float],
    width: float = 0.02,
    channel: str = "",
  ) -> None:
    """Draw "this link receives this force and is pulled toward that setpoint", in
    the style set by ``compliance_force_style``.

    Both styles draw the same quantities; they differ only in which primitives
    carry **geometry** and **magnitude**:

      * ``arrow`` (default): one arrow from the
        link toward the setpoint, length ``|F|/K_env`` and capped at the setpoint.
        One primitive expresses direction, magnitude and where it pulls at once.
      * ``spring``: a thin cylinder = the spring itself (link to setpoint), exact
        and never overshooting, and a line has no minimum size: when the stretch
        under a stiff ``K_env`` is only about 2 cm, the arrow style's arrow is
        buried entirely inside the setpoint sphere, while the thin cylinder stays
        readable. The force gets a separate **fixed-length** arrow for direction
        only, drawn push-style behind the link (tail at
        ``live − F̂·(len+clear)``, tip stopping at ``live − F̂·clear``), offset from
        the cylinder. This style does not draw the magnitude.
    """
    if not self._spring_style():
      visualizer.add_arrow(
        start, force_arrow_end(start, force), color=color, width=width, label=channel
      )
      return
    visualizer.add_cylinder(start, setpoint, _SPRING_RADIUS, color, label=channel)
    n = float(np.linalg.norm(force))
    if n < 1e-9:
      return
    unit = np.asarray(force, dtype=float) / n
    visualizer.add_arrow(
      start - unit * (_DIR_ARROW_LEN + _DIR_ARROW_CLEAR),
      start - unit * _DIR_ARROW_CLEAR,
      color=color,
      width=_DIR_ARROW_WIDTH,
      label=channel,
    )

  def _draw_reanchor_frames(
    self, visualizer, marks: list[ComplianceMark], channels: frozenset[str]
  ) -> None:
    """Draw the **re-anchor transform itself**, not just its three output points
    (drawn once per env when ``debug_vis_reanchor`` is on).

    What re-anchoring does: it carries "the setpoint's offset relative to the
    reference anchor" from the clip frame into the robot frame, **using yaw only**,
    with Z kept from the reference. Otherwise none of these three things is visible
    in the frame: only the three spheres show, not the transform that placed them
    there. So this draws:

      * one **heading arrow** each for the reference anchor and the robot anchor
        (``reanchor_ref`` / ``reanchor_robot``): the angle between them is the yaw
        delta Δyaw. Heading arrows instead of coordinate triads are deliberate: a
        triad suggests all three axes take part, but ``yaw_quat`` inside
        ``delta_quat`` already drops pitch/roll, so only yaw enters the transform.
      * a **dim copy** of the same pair drawn at the frozen anchors (``_ff_*``,
        captured at the event's rising edge): the angle difference between the
        bright and dim pairs is the whole difference between ``rising_edge`` and
        ``per_frame``. Under perfect tracking (kinematically pinned replay) all four
        overlap = the transform is the identity, **which is exactly why the three
        setpoint spheres collapse into one point**, not a drawing bug.
      * ``reanchor_base`` = the blended base point
        ``(robot_anchor.xy, ref_anchor.z)``, with a thin vertical line back to the
        robot anchor: nothing else in the frame shows the rule "XY from the robot,
        Z from the reference".
      * two thin cylinders per slot: frozen reference anchor -> script setpoint,
        and blended base point -> re-anchored setpoint. **The same offset vector,
        differing by one Δyaw**: if the two differ in length, the rotation is
        wrong, a correctness check you can judge by eye.

    The anchors are frozen per slot; only the **primary slot**'s pair is drawn here
    (the anchor body is shared anyway); each slot's own frozen anchor already shows
    in the position of its own ``setpoint_active`` sphere.
    """
    if not marks:
      return
    sl = slice(visualizer.env_idx, visualizer.env_idx + 1)
    cur_rp, cur_rq, cur_fp, cur_fq = (
      t[sl] for t in self.anchor_pose("wrench_match_anchor_per_frame")
    )
    fr = [t[sl][:, marks[0].slot] for t in self.frozen_anchor_slots()]
    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

    def heading(pos, quat, channel: str) -> None:
      """Draw a heading arrow from the anchor along its **yaw** direction (yaw only,
      matching what the transform actually uses).

      The colors avoid the robot's body color: the robot is white, and a near-white
      arrow vanishes into the torso."""
      d = quat_apply(yaw_quat(quat), x_axis)[0].cpu().numpy()
      p0 = pos[0].cpu().numpy()
      visualizer.add_arrow(
        p0,
        p0 + d * _REANCHOR_HEADING_LEN,
        color=viz_palette.MARKER_RGBA[channel],
        width=_REANCHOR_HEADING_WIDTH,
        label=channel,
      )

    heading(cur_fp, cur_fq, "reanchor_ref")  # reference anchor (current)
    heading(cur_rp, cur_rq, "reanchor_robot")  # robot anchor (current), blue
    heading(fr[2], fr[3], "reanchor_ref_frozen")  # reference anchor (frozen)
    heading(fr[0], fr[1], "reanchor_robot_frozen")  # robot anchor (frozen), dark blue

    def base_pair(robot_anchor, ref_anchor, ball, connector, name: str):
      """Gold sphere + thin rod for the blended base point: XY from the robot
      anchor, Z from the reference anchor.

      Written once because the frozen set and the per-frame set follow **the same
      rule** with different anchors: how far apart the two spheres are is how much
      "locked at the rising edge" and "recomputed every frame" differ. Under
      perfect tracking the re-anchoring is the identity and the two sets coincide.
      """
      base = robot_anchor.clone()
      base[..., 2] = ref_anchor[..., 2]
      base_np = base[0].cpu().numpy()
      visualizer.add_sphere(
        base_np, _REANCHOR_BASE_SPHERE_RADIUS, color=ball, label=name
      )
      visualizer.add_cylinder(
        base_np,
        robot_anchor[0].cpu().numpy(),
        _REANCHOR_CONNECTOR_RADIUS,
        color=connector,
        label=name,
      )
      return base_np

    # The frozen set (the same anchors as the rise face).
    base_np = base_pair(
      fr[0],
      fr[2],
      viz_palette.MARKER_RGBA["reanchor_base"],
      viz_palette.MARKER_RGBA["reanchor_base_connector"],
      "reanchor_base",
    )
    # The per-frame set: the same sphere and rod with the **current** anchors, so it
    # drifts with the robot, plus two thin cylinders from this sphere to the
    # per-frame setpoints, mirroring the frozen set one to one:
    # `reanchor_offset_active` runs from the frozen base point to the rise setpoint,
    # and here from the per-frame base point to the per-frame setpoint. Without
    # these two cylinders the per-frame sphere would appear out of nowhere, with no
    # sign of which base point it was re-anchored to.
    if "reanchor_base_per_frame" in channels:
      pf_base = base_pair(
        cur_rp,
        cur_fp,
        viz_palette.MARKER_RGBA["reanchor_base_per_frame"],
        viz_palette.MARKER_RGBA["reanchor_base_per_frame_connector"],
        "reanchor_base_per_frame",
      )
      offset_rgba = viz_palette.MARKER_RGBA["reanchor_offset_per_frame"]
      data_rgba = viz_palette.MARKER_RGBA["reanchor_offset_data_per_frame"]
      cur_ref_np = cur_fp[0].cpu().numpy()
      for m in marks:
        # After the transport: current base point -> per-frame setpoint sphere.
        visualizer.add_cylinder(
          pf_base,
          m.setpoint_per_frame,
          _REANCHOR_CONNECTOR_RADIUS,
          offset_rgba,
          label="reanchor_base_per_frame",
        )
        # Before the transport: the reference ghost's **current** anchor -> the
        # same baked setpoint_script sphere. The two cylinders are **the same
        # vector**, equal in length every frame, differing only by one Δyaw: that
        # is exactly what re-anchoring does. Mirrors the frozen set's
        # `reanchor_offset_data` one to one, only with the current anchor instead
        # of the rising-edge one.
        visualizer.add_cylinder(
          cur_ref_np,
          m.script_setpoint,
          _REANCHOR_CONNECTOR_RADIUS,
          data_rgba,
          label="reanchor_offset_data_per_frame",
        )

    # Per-slot offset comparison: the two thin cylinders are the same vector
    # differing by one Δyaw. Cylinders rather than arrows, so they are not confused
    # with the force arrows.
    ref_np = fr[2][0].cpu().numpy()
    for m in marks:
      if m.setpoint_rise is None:
        continue
      visualizer.add_cylinder(
        ref_np,
        m.script_setpoint,
        _REANCHOR_CONNECTOR_RADIUS,
        viz_palette.MARKER_RGBA["reanchor_offset_data"],
        label="reanchor_offset_data",
      )
      visualizer.add_cylinder(
        base_np,
        m.setpoint_rise,
        _REANCHOR_CONNECTOR_RADIUS,
        viz_palette.MARKER_RGBA["reanchor_offset_active"],
        label="reanchor_offset_active",
      )

  def _draw_human_ref_vis(self, visualizer) -> None:
    """Render the human reference visualization (ghost mesh and/or body frames).

    The ghost is only supported when `cfg.load_human_entity` is False -- the
    standalone g1 spec is compiled into a small mj_model dedicated to ghost
    rendering and driven from motion data each step. In the entity-loaded
    case, the live entity already shows the human, so the ghost is skipped
    with a one-time hint.
    """
    if self.cfg.debug_vis_human_ref_ghost:
      if self.cfg.load_human_entity:
        if not self._human_ghost_warning_logged:
          self._human_ghost_warning_logged = True
          print(
            "[INFO] debug_vis_human_ref_ghost=true is a no-op when "
            "load_human_entity=True (the live entity already shows the human). "
            "Use debug_vis_human_ref_coord_frames=true for an overlap-check overlay."
          )
      else:
        self._draw_human_ghost_mesh(visualizer)

    if self.cfg.debug_vis_human_ref_coord_frames:
      self._draw_human_ref_coord_frames(visualizer)

  def _draw_human_ghost_mesh(self, visualizer) -> None:
    """Draw the imagined human as a translucent ghost through the shared
    ``visualizer.add_ghost_mesh`` API -- the SAME path the robot reference ghost
    uses, so it renders identically on the native viewer, the offscreen recorder
    (eval / train-eval video), and viser.

    The human is a separate, smaller g1 model than the composed scene; the
    visualizer builds the per-model MjData matching whatever model it is handed,
    so the differing ``nq`` is fine and no ``scn`` access or per-viewer
    special-casing is needed here. The recolor + body-group filtering ride on the (cached) ghost
    model's per-geom rgba (``_build_ghost_model``).
    """
    if self._human_ref_ghost_model is None:
      assert self._human_standalone_spec is not None, (
        "Human ghost requested but standalone spec wasn't built (only available "
        "when cfg.load_human_entity=False)."
      )
      self._human_ref_ghost_model = self._build_ghost_model(
        self._human_standalone_spec,
        self._human_ghost_rgba(),
        tuple(self.cfg.debug_vis_human_ref_ghost_bodies),
      )
      self._human_ref_ghost_qpos = np.zeros(self._human_ref_ghost_model.nq)

    assert self._human_ref_ghost_qpos is not None

    env_idx = visualizer.env_idx
    ts = (self._motion_time[env_idx : env_idx + 1] / self._control_dt).round().long()
    pair_idx = (
      int(self._motion_id[env_idx].item()) if len(self._human_motions) > 1 else 0
    )
    loader = self._human_motions[pair_idx]
    ts = ts.clamp(max=loader.time_step_total - 1)  # RSI/non-control fps may overrun
    root_state = loader.get_root_state(ts)[0].cpu().numpy()
    joint_pos, _ = loader.get_joint_state(ts)
    joint_pos_np = joint_pos[0].cpu().numpy()

    qpos = self._human_ref_ghost_qpos
    qpos[:3] = root_state[:3] + self._ref_origins[env_idx].cpu().numpy()
    qpos[3:7] = root_state[3:7]
    n_joint = self._human_ref_ghost_model.nq - 7
    qpos[7 : 7 + n_joint] = joint_pos_np[:n_joint]

    visualizer.add_ghost_mesh(
      qpos,
      self._human_ref_ghost_model,
      alpha=self._human_ghost_rgba()[3],
      label="human_ref",
    )

  def _human_ghost_rgba(self) -> tuple[float, float, float, float]:
    """The imagined partner's ghost color, session override first."""
    return _ghost_rgba(
      self.cfg.debug_vis_human_ref_ghost_color or self.cfg.human_ref_ghost_color,
      "human_ref_ghost_color",
    )

  def _build_ghost_model(
    self,
    spec: mujoco.MjSpec,
    color: tuple[float, float, float, float],
    ghost_bodies: tuple[str, ...],
    body_colors: dict[str, tuple[float, float, float, float]] | None = None,
  ) -> mujoco.MjModel:
    """Compile a copy of a standalone g1 spec, recolor for translucency, and
    keep only the geoms of the ``ghost_bodies`` body-name list (empty = the
    whole model; geoms outside the list get alpha=0, effectively hidden). The
    lists come from the ``body_sets`` registry (every body of a region for a
    connected mesh; a sparse reward set renders as disconnected fragments).
    Collision geoms are likewise hidden so the ghost is **self-describing** --
    visual-only under every backend's default options without relying on a
    renderer's geom-group filter.

    Shared by the robot reference ghost (``self.robot.cfg.spec_fn()``) and the
    imagined-human ghost (``self._human_standalone_spec``): both are standalone
    g1 models that never pass the collision-viz spec_fn, so their visual geoms
    stay in the default-visible groups (no group-4 parking). Standalone-spec
    body names are unprefixed (e.g. `pelvis`), so `mj_name2id` lookups are
    direct; a listed name absent from the model is skipped (the human model
    shares the g1 topology, so in practice the lists match).

    ``body_colors`` repaints named bodies only: a body listed there uses its own
    rgba, every other allowed body keeps ``color``. Body filtering still
    applies first, so naming a body outside ``ghost_bodies`` does not
    resurrect it (that would silently contradict the subset the caller chose) --
    but it does WARN, since the caller asked for a highlight and would otherwise
    get a figure quietly missing it.
    """
    spec_copy = spec.copy()
    model = spec_copy.compile()
    color_arr = np.array(color, dtype=np.float32)
    # Not a color: alpha 0 is how a geom outside `ghost_bodies` is hidden, and
    # the rgb under it is never seen.
    hidden_color = np.zeros(4, dtype=np.float32)

    def _names_to_body_ids(names: tuple[str, ...]) -> set[int]:
      ids: set[int] = set()
      for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
          ids.add(bid)
      return ids

    allowed = _names_to_body_ids(tuple(ghost_bodies)) if ghost_bodies else None

    # A geom shows the ghost color only if it is BOTH in the allowed body group
    # AND a visual geom; collision geoms (contype/conaffinity set) are hidden via
    # alpha 0. This keeps the ghost visual-only under viser too, whose ghost path
    # filters on alpha rather than geom group and would otherwise draw the ghost-
    # colored collision capsules. native / offscreen also
    # hide the group-3 collision via the default MjvOption, so this is a second,
    # renderer-independent guard there -- the same self-describing principle as
    # the main-scene collision viz (`apply_scene_geom_viz`).
    # Per-body overrides resolved to body ids once, so the geom loop stays a
    # dict lookup and an unknown body name fails loudly rather than silently
    # painting nothing.
    override_by_body_id: dict[int, np.ndarray] = {}
    for body_name, body_color in (body_colors or {}).items():
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
      if bid < 0:
        raise ValueError(
          f"debug_vis.body_styles names body {body_name!r}, which is not in the "
          "ghost model. Use a standalone-spec body name (unprefixed, e.g. "
          "'left_wrist_yaw_link')."
        )
      override_by_body_id[bid] = np.array(body_color, dtype=np.float32)
    # Named but filtered out by the body group: the override can never show, so
    # say so rather than rendering a figure that is silently missing its
    # highlight. A warning, not an error: one styles block is often shared
    # across panels that deliberately show different body subsets.
    dropped = sorted(
      name
      for name, bid in (
        (n, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n))
        for n in (body_colors or {})
      )
      if allowed is not None and bid not in allowed
    )
    if dropped:
      print(
        f"[WARN] debug_vis.body_styles names {dropped}, which the ghost body "
        f"list {list(ghost_bodies)} filters out of this ghost; those overrides "
        "draw nothing. Widen the list (e.g. [] for the whole model) or drop "
        "the entries."
      )

    for geom_idx in range(model.ngeom):
      body_id = int(model.geom_bodyid[geom_idx])
      in_allowed = allowed is None or body_id in allowed
      is_collider = bool(model.geom_contype[geom_idx]) or bool(
        model.geom_conaffinity[geom_idx]
      )
      if in_allowed and not is_collider:
        model.geom_rgba[geom_idx] = override_by_body_id.get(body_id, color_arr)
      else:
        model.geom_rgba[geom_idx] = hidden_color
      model.geom_matid[geom_idx] = -1
    return model

  def _draw_human_ref_coord_frames(self, visualizer) -> None:
    from mjlab.utils.lab_api.math import matrix_from_quat

    pos_w, quat_w, body_labels = self._get_human_tracked_body_pose_w(
      self.cfg.debug_vis_human_source
    )
    env_idx = visualizer.env_idx
    pos_np = pos_w[env_idx].cpu().numpy()
    rotm_np = matrix_from_quat(quat_w[env_idx]).cpu().numpy()
    for i, label in enumerate(body_labels):
      visualizer.add_frame(
        position=pos_np[i],
        rotation_matrix=rotm_np[i],
        scale=0.10,
        label=f"human_ref_{label}",
      )


@dataclass(kw_only=True)
class CodancingComposedCommandCfg(CommandTermCfg):
  robot_asset_name: str = "robot"
  human_asset_name: str = "g1_human"

  # Lift the whole reference stream onto the local terrain height instead of
  # leaving it at the constant `env_origins.z` baked in at build time.
  #
  # Clips are recorded on flat ground, so on procedural terrain the reference
  # sits at a height the ground does not have. Everything that compares a live
  # world z against a reference world z then reads the terrain as tracking
  # error: the foot-front reward's `error_z` (rewards.py) and the upper-body
  # reward via `_anchor_aligned` (z is deliberately NOT re-anchored there).
  # Measured cost at std 0.25 / 0.3: h=2cm loses 0.64% / 0.44%, h=4cm loses
  # 2.53% / 1.76%, h=10cm loses 14.8% / 10.5% -- past which the reward is
  # mostly penalising the robot for standing on a slope.
  #
  # OFF by default: it changes reward semantics, so a run with it on is not
  # comparable to one without; the released checkpoints were trained with it
  # off.
  #
  # The construction: one per-env vertical shift, taken under the reference
  # ROOT and applied to every body, so the pose's own geometry (a jump stays a
  # jump) is untouched.

  # Single source of truth for the partner axis. Every human-side toggle is
  # derived from this (see the `load_human_entity` / `require_human_motion`
  # properties below), so the two flags can never desync.
  #   "g1"       -- a live human ENTITY is in the scene (full cross-entity
  #                 contacts / sensors); the human is also driven by a motion.
  #   "imagined" -- no human entity; the human REFERENCE comes from the motion
  #                 file only (FK-free human reads).
  #   "none"     -- pure robot imitation; no human reference at all (the exec
  #                 layer rejects a human motion file and runs partner-free).
  partner_mode: Literal["none", "imagined", "g1"] = "g1"

  # The typed clip pool (MotionLibrary clips): per clip, the motion files,
  # weight, name, placement_offset. Injected by the exec layer from the
  # `motion.motions` manifest; never composed from yaml.
  clips: tuple[MotionClip | PairedMotionClip, ...] = ()

  # The motion cursor's whole life: reference SOURCE (the union; per-moment
  # clip selection lives on the clip source), one MODE per resample moment
  # (episode_reset: default | keyframe | rsi, clip_end: terminate | chain),
  # the `rsi` sampler parameters, and the episode-reset perturbation ranges.
  # All scalar leaves -> struct-mode overridable.
  motion_cursor: MotionCursorCfg = field(default_factory=MotionCursorCfg)

  # Optional observation of active motion pair index:
  #   "none": no extra obs (default)
  #   "one_hot": append one-hot vector of length num_pairs
  #   "scalar": append normalized scalar index
  active_clip_obs_mode: Literal["none", "one_hot", "scalar"] = "none"

  # Declarative AMP feature spec (task-blind AMP). When set, drives the `amp` +
  # `amp_group` obs terms, the per-disc_group expert `amp_data`, and
  # construct-time feature-dim resolution. None => no AMP wiring (plain PPO).
  amp_feature: AmpFeatureCfg | None = None

  # The human has NO anchor field (it is observed, it tracks nothing): each
  # observation term names the bodies it reads (`human_body_*_b` with
  # explicit `body_names`, e.g. a `${body_sets.human_*}` set) and rewards
  # name their body per term. Only the named PARTS below stay as fields (the
  # part accessors and the coord-frame viz read them).
  g1_human_pelvis_body_name: str = "pelvis"
  g1_human_left_hand_body_name: str = "left_wrist_yaw_link"
  g1_human_right_hand_body_name: str = "right_wrist_yaw_link"
  g1_human_left_foot_body_name: str = "left_ankle_roll_link"
  g1_human_right_foot_body_name: str = "right_ankle_roll_link"

  # The dense per-body pose reward's anchor, named by the computation it
  # serves. ONE field for both sides: the live and reference anchors of the
  # same purpose must be the same body (the anchor-aligned transform pairs
  # corresponding points), so a single name states that by construction. The
  # wrench-match anchors (the force events' `anchor_body`) and the AMP
  # feature anchor (`amp_feature.anchor_body_name`) are separate purposes
  # with their own config; see `anchor_pose`.
  pose_match_anchor_body_name: str = "torso_link"
  robot_left_foot_body_name: str = "left_ankle_roll_link"
  robot_right_foot_body_name: str = "right_ankle_roll_link"

  # The body-set registry (set name -> body-name list). Composed runs inject
  # the WHOLE `conf/body_sets/g1.yaml` registry (`body_sets: ${body_sets}` in
  # the catalog).
  # `body_set_indices` resolves a set name to model-order column indices
  # through this: the ONE vocabulary for every name-keyed body-scoped
  # consumer (metric_body_groups, body-scoped reference reads).
  body_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)

  metric_suites: tuple[MetricSuiteName, ...] = ("robot_ref_tracking",)
  metric_body_groups: tuple[str, ...] = ("torso",)
  metric_reference_frame: MetricReferenceFrame = "both"
  # Comparison pairs over the three pose faces (live / served / free); every
  # pair records under `robot_ref/{a}_vs_{b}/...` keys. `[free, live]` tracks
  # against the ORIGINAL clip without swapping the pool and also records
  # anchor_aligned
  # (its relative buffers compute per step when the pair is configured on an
  # augmented pool); `[served, free]` is the per-step yield magnitude,
  # global-frame only.
  metric_compare: tuple[tuple[str, str], ...] = (("served", "live"),)

  resampling_time_range: tuple[float, float] = field(
    default_factory=lambda: (1.0e9, 1.0e9)
  )

  # visualization
  # debug_vis is already defined in CommandTermCfg (overall master switch). Each
  # reference draws two INDEPENDENT things -- a translucent ghost mesh and/or
  # desired-vs-current coordinate frames -- each gated by its own bool (no
  # tri-state mode, no per-ref master). Draw nothing = both False.
  debug_vis_robot_ref_ghost: bool = True
  """Draw the robot reference as a translucent ghost mesh."""
  debug_vis_robot_ref_coord_frames: bool = False
  """Draw reference-vs-current robot body/anchor COORD frames (arrows)."""
  debug_vis_robot_ref_desired_ghost: bool = False
  """Draw the ADAPTED reference carried onto the LIVE anchor as a GHOST.

  The ghost form of `robot_ref_desired_coord_frames`. Prefer it when the offsets are
  small: a whole translucent body separates from the reference ghost at a glance,
  where several faces of arrows pile onto each other."""
  debug_vis_robot_ref_desired_coord_frames: bool = False
  """Draw the ADAPTED reference transported onto the LIVE anchor.

  The third face in the anchor-centered decomposition: `robot_ref_coord_frames` draws
  the raw reference and the live body, this draws the reference carried onto the
  live anchor, so the reference-to-desired gap is the anchor error and the
  desired-to-current gap is the anchor-centered tracking error the reward sees.
  All three coincide under an exact pin -- perturb the pin to separate them."""
  debug_vis_free_ref_desired_ghost: bool = False
  """Draw the FREE reference carried onto the live anchor as a GHOST."""
  debug_vis_free_ref_desired_coord_frames: bool = False
  """Draw the FREE (un-yielded) reference transported onto its OWN anchor.

  Same construction on the free face, whose anchor also cancels the augmenter's
  root shift. Against `robot_ref_desired_coord_frames` the gap is the compliance yield
  expressed in the tracking frame."""
  # Session-tier overrides written at render build by run_exec.apply_debug_vis;
  # None keeps the env-tier colors below, and a None bodies override makes the
  # free ghost borrow the served ghost's subset.
  debug_vis_robot_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  debug_vis_free_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  debug_vis_human_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  debug_vis_free_ref_ghost_bodies: tuple[str, ...] | None = None
  # Named force-viz channels; None expands the two booleans below
  # (`debug_vis_compliance_force_arrow`, `debug_vis_reanchor`).
  debug_vis_force_channels: tuple[str, ...] | None = None
  # Per-body style overrides: face name -> body name -> {ghost_color,
  # coord_axis_colors}. Empty/None = every body takes its face's default. See
  # `BodyStyle`; validated in `_body_styles_for` (unknown face or key raises).
  debug_vis_body_styles: dict[str, dict[str, dict[str, Any]]] | None = None
  debug_vis_robot_ref_ghost_bodies: tuple[str, ...] = ()
  """Body-name list the robot reference visualization keeps; () = the whole
  model.

  A plain ``${body_sets.<name>}`` reference in config: the registry is the one
  place body selections are named. Ghost mode renders
  via geom_rgba transparency on a standalone g1 model (compiled from the
  robot's ``spec_fn``); list every body of the region there, since a sparse
  reward set renders as disconnected fragments. Coord-frame mode draws one
  coord frame per listed body (the body tensors carry every model body), so
  with frames on prefer a sparse set."""
  robot_ref_ghost_color: tuple[float, float, float, float] | str = _face_ghost(
    "reference"
  )

  # SoftMimic compliance viz -- TOGGLES ONLY (the data is on the CLIP). When the
  # served reference is a force-ADAPTED clip carrying free_motion_file +
  # contact_file, the command builds `self._contact_provider`; these flags draw:
  #   free_ref_ghost  -> the FREE (un-forced) reference as a SECOND ghost at the
  #     served root with the FREE joints -> the gap IS the upper-body yield.
  #   compliance_force_arrow -> the scripted wrench as an arrow at the REFERENCE
  #     forced link (on the ghost, visible regardless of the policy) + the applied
  #     wrench / re-anchored setpoint at the LIVE link. Drawn PER SLOT, so a
  #     multi-link (multi-link) clip draws all K contacts; single contact is K=1.
  # Both no-op when no clip is augmented. No file paths here on purpose.
  debug_vis_free_ref_ghost: bool = False
  free_ref_ghost_color: tuple[float, float, float, float] | str = _face_ghost("free")
  # Transported-face ghost colors, hue-matched to the coord-frame colors so the two
  # representations of a face read as the same face. Alpha is deliberately below
  # the reference ghost's: a transported face usually overlaps it.
  robot_ref_desired_ghost_color: tuple[float, float, float, float] | str = _face_ghost(
    "desired"
  )
  free_ref_desired_ghost_color: tuple[float, float, float, float] | str = _face_ghost(
    "desired_free"
  )
  debug_vis_compliance_force_arrow: bool = False
  # Drawing style: arrow (default) | spring. See compliance_force_style in
  # config/run_cfg.py.
  debug_vis_compliance_force_style: str = "arrow"
  # Three-face setpoint comparison: next to active (= `_ff_*`, frozen at the rising
  # edge under the default rising_edge), also draw data (= the original
  # un-re-anchored setpoint) and per_frame (= re-anchored with the current
  # anchors), each with a spring arrow. The gaps between the three = the "no
  # re-anchor vs rising edge vs per frame" difference, which appears only when
  # live departs from the reference (under perfect tracking the three spheres
  # overlap). Off by default (to limit clutter); turn it on to compare.
  debug_vis_reanchor: bool = False

  # Human reference visualization (parallel to robot_ref_*).
  debug_vis_human_ref_ghost: bool = False
  """Draw the human reference as a translucent ghost mesh. Off by default: when
  `load_human_entity` is True the live entity already shows the human (the ghost
  is then a no-op); when False, the ghost is the imagined partner driven by the
  motion file (source via `debug_vis_human_source`)."""
  debug_vis_human_ref_coord_frames: bool = False
  """Draw human reference body coordinate frames (overlap-check overlay)."""
  debug_vis_human_ref_ghost_bodies: tuple[str, ...] = ()
  """Body-name list the human ghost keeps; () = the whole model (see
  ``debug_vis_robot_ref_ghost_bodies``)."""
  # The color and why it is that color live with it, in
  # `viz_palette.HUMAN_GHOST_COLOR`. Repeating either here is how the two go
  # out of step.
  human_ref_ghost_color: tuple[float, float, float, float] | str = viz_palette.rgba(
    viz_palette.HUMAN_GHOST_COLOR, "video"
  )
  debug_vis_human_source: Literal["auto", "entity", "motion"] = "auto"
  """Where to source body poses for the human ghost. "auto" follows
  `load_human_entity` (entity → live entity poses, motion → motion-file poses).
  "entity" forces FK reads (errors if entity not loaded). "motion" always reads
  from the motion file even if the entity is loaded -- useful for verifying that
  the motion data matches what the entity is doing (the two ghosts should
  overlap perfectly)."""

  @property
  def load_human_entity(self) -> bool:
    """A live human ENTITY exists only for partner_mode=g1 (derived from it)."""
    return self.partner_mode == "g1"

  @property
  def require_human_motion(self) -> bool:
    """A human REFERENCE motion exists for every partner except none."""
    return self.partner_mode != "none"

  def build(self, env: ManagerBasedRlEnv) -> CodancingComposedCommand:
    return CodancingComposedCommand(self, env)


def get_codancing_command(
  env: Any, command_name: str = "codancing"
) -> CodancingComposedCommand:
  """Typed accessor for the codancing command term (single narrowing point).

  ``CommandManager.get_term`` is typed as the base ``CommandTerm``, so every MDP
  term / script / viewer that needs codancing-specific state would otherwise
  repeat a ``cast(CodancingComposedCommand, ...)``. This runtime-checked accessor
  centralizes that narrowing. ``env`` may be a raw ``ManagerBasedRlEnv`` or any
  wrapper exposing ``command_manager`` (pass ``env.unwrapped`` for wrappers).
  """
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, CodancingComposedCommand), (
    f"Command {command_name!r} is {type(command).__name__}, "
    "expected CodancingComposedCommand."
  )
  return command


def get_codancing_command_cfg(
  env_cfg: Any, command_name: str = "codancing"
) -> CodancingComposedCommandCfg:
  """Typed accessor for the codancing command *cfg* (single narrowing point)."""
  command_cfg = env_cfg.commands[command_name]
  assert isinstance(command_cfg, CodancingComposedCommandCfg), (
    f"Command cfg {command_name!r} is {type(command_cfg).__name__}, "
    "expected CodancingComposedCommandCfg."
  )
  return command_cfg
