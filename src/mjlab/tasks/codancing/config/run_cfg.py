"""Typed execution configs for the three root blocks of a composed run.

A composed run tree carries three root blocks beyond ``env`` / ``runner``,
split by freeze tier:

* ``motion:`` -- FROZEN tier. The reference-motion selection the env depends
  on; embedded in every checkpoint so frozen rebuilds reproduce the env.
* ``eval:`` -- FROZEN tier. The dedicated-eval measurement spec (rollout/clip
  length, recorder enable, env overlays); embedded so offline ``run eval``
  reproduces every recipe from the checkpoint alone.
* ``session:`` -- LIVE-ONLY tier. Everything that describes the *host process*
  of an invocation (viewer, policy source, recorders, GPUs, log destinations).
  Stripped from frozen snapshots -- it structurally cannot leak into a later
  invocation -- and grafted fresh (dataclass defaults) when a checkpoint is
  loaded as a config anchor.

The invariant: frozen tiers determine the measured trajectory and travel with
the checkpoint; ``session`` only determines where/how the computation runs and
is rebuilt per invocation, like argv.

This module is the single source of truth for the typed view; the predeclared
blocks in ``conf/run.yaml`` mirror these defaults.
It is kept free of compose/executor imports so both ``compose_run`` (builds
the typed view) and ``run_exec`` (consumes it) can depend on it without
cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.tasks.shared.wrapper.scene import EntityGeomVizCfg

ViewerKind = Literal["auto", "native", "native-frozen", "viser", "none"]


@dataclass(kw_only=True)
class EvalMetricCfg:
  """An eval recipe's recording knobs. Every eval writes the per-(step, env)
  rollout table, and ``eval_analysis`` computes the summaries from it offline.
  The metric *suite* is owned by the command
  (``env.commands.codancing.metric_*``)."""

  settle_steps: int = 0
  """The analyzer's settle window: rows within this many steps after an env
  reset are masked there; every row is still recorded."""


@dataclass(kw_only=True)
class SessionMetricCfg:
  """A play session's recording knobs, opt-in per invocation."""

  enabled: bool = False
  """Write the per-(step, env) rollout table (``<prefix>-rollout-envs.csv``,
  the analyzer's input) beside the session's clips."""
  num_steps: int | None = None
  """Headless session length in env steps; None falls back to the video
  window. Sessions run a fixed length (there are no in-process stop
  conditions)."""


@dataclass
class MotionCfg:
  """Motion-data selection (FROZEN tier; resolved at execution time)."""

  # The clip pool: a clip-manifest YAML path (str) or the clip entries written
  # inline (list of dicts, same keys as the manifest schema). Single-clip runs
  # use a one-entry manifest or list.
  motions: str | list[dict[str, Any]] | None = None


@dataclass
class EvalVideoCfg:
  """The EVAL recorder's enable (FROZEN tier) -- records the eval clip.

  Deliberately JUST an ``enabled`` switch, parallel to
  ``session.video.enabled``. The eval clip length is NOT here: it is the
  recipe's ROLLOUT length (its ``length_s`` / ``length_steps``), because the
  eval clip IS the bounded rollout (and the rollout runs for metrics even with
  video off). So a recipe has no ``video.length_s``; use its ``length_*``. (The
  session recorder is the other one: ``session.video.{enabled, length_*,
  interval_steps}`` tapes your live train/play session.)
  """

  enabled: bool = True


@dataclass
class EvalRecipeCfg:
  """One eval RECIPE: a complete, self-contained eval spec (env overlays +
  rollout length + recorder + metric).

  ``compose_run`` freezes one eval tree per recipe into every checkpoint, so a
  checkpoint reproduces each condition offline.
  Fields are flat and explicit -- recipes do NOT inherit from one another (a few
  repeated lines per recipe, over an implicit cross-recipe merge).
  """

  overlays: list[str] = field(default_factory=lambda: ["play"])
  """Overlays composed (on top of the train selection) into THIS recipe's
  dedicated eval env at compose time only; the tree is frozen with the
  checkpoint, so later changes go through ``--overlay``. The
  default ``["play"]`` gives play semantics (infinite episode, no obs
  corruption, perturbation DR dropped)."""
  num_envs: int = 1
  length_s: float | None = None
  """Time-based rollout length (s; simulated == playback). Wins over
  ``length_steps`` when set; converted via :func:`length_steps_from`."""
  length_steps: int = 300
  """Bounded rollout length (env steps) -- the eval CLIP length equals the
  rollout length (the clip IS the rollout)."""
  video: EvalVideoCfg = field(default_factory=EvalVideoCfg)
  """The EVAL recorder enable for this recipe (length is ``length_*`` above).
  Resolution is taken from the eval env's ``env.viewer.width/height``."""
  metric: EvalMetricCfg = field(default_factory=EvalMetricCfg)
  """Per-recipe recording knobs: ``settle_steps`` is the analyzer's settle
  window. The per-env table is always written; the rollout is bounded by
  ``length_*``."""
  debug_vis: DebugVisCfg | None = None
  """Per-recipe reference-ghost / frames draw (session-level, render-only, like
  ``session.debug_vis``, it never changes the measured trajectory). ``None``
  (default) INHERITS ``session.debug_vis``; a block REPLACES it WHOLESALE (no
  field merge -- matching the flat, no-inherit recipe model). So one recipe can
  force both ghosts on while another sets ``enabled: false`` to draw nothing,
  each independent of the session default.
  Grafted onto the command at eval build time (``run_exec.apply_debug_vis``,
  resolved as ``recipe.debug_vis or session.debug_vis``); not meaningfully frozen
  (the eval tree is session-stripped on snapshot)."""
  viz: VizCfg | None = None
  """Per-recipe main-scene geom decoration (collision capsules / hidden visual),
  same inherit-or-replace rule as ``debug_vis``: ``None`` inherits ``session.viz``;
  a block replaces it. Resolved as ``recipe.viz or session.viz`` at eval build. So
  one recipe can show collision capsules in its clip while ``default`` stays
  clean."""


@dataclass
class EvalCfg:
  """The run's eval spec (FROZEN tier): the overlay-group gate + a dict of
  recipes.

  ``compose_run`` composes a dedicated eval env PER RECIPE for every train run
  and freezes each resolved tree into the checkpoint, so ANY checkpoint
  reproduces every condition offline via ``run eval [--recipe NAME|all]``.
  Top-level holds only the overlay-group gate (``allowed_overlay_groups``);
  everything else is per-recipe (see :class:`EvalRecipeCfg`).
  """

  allowed_overlay_groups: list[str] = field(
    default_factory=lambda: ["play", "video", "motion_cursor"]
  )
  """The overlay groups allowed in ``overlays``: the eval tree describes a
  measurement, so only groups that shape the measurement compose into it (``play``, the
  ``video/*`` resolutions and the ``motion_cursor/*`` episode-reset / clip-end
  recipes). Compose rejects others with an explanatory error: a motion-, runner- or
  session-shaping overlay would be frozen but inert. Allowlisting only PERMITS a
  group in ``overlays`` (the default eval env stays ``[play]``). The list is
  ordinary config: the run file may override it, or per run
  ``--override 'eval.allowed_overlay_groups=[...]'``; the motion/runner
  consistency gate still backstops whatever is allowed. ``video/*`` overlays
  that set ``session.*`` keys are inert in the eval tree (it is
  session-stripped); use a recipe's ``video.enabled`` / ``length_*`` knobs."""
  recipes: dict[str, EvalRecipeCfg] = field(
    default_factory=lambda: {"default": EvalRecipeCfg()}
  )
  """Named eval recipes; a ``default`` is required. Offline ``run eval`` runs
  ``default`` unless ``--recipe NAME`` / ``--recipe all``, with each recipe's
  artifacts under ``eval_sessions/{recipe}/``. Recipes are flat + explicit (no
  cross-recipe inheritance)."""


@dataclass
class SessionVideoCfg:
  """The SESSION recorder -- tapes the live train/play session you watch.

  This is one of the run's TWO recorders; the other is the EVAL recorder
  (each recipe's ``video.enabled`` + ``length_*``, FROZEN tier, records the
  eval clip). This one has its own length AND a train cadence because it samples
  clips from a continuous session; the eval recorder has neither (the eval clip
  IS the bounded rollout). Session-tier: never frozen into a checkpoint.
  """

  enabled: bool = False
  length_s: float | None = None
  """Time-based clip length in seconds; wins over ``length_steps`` when set."""
  length_steps: int = 200
  interval_steps: int = 2000
  """Train cadence: env steps between clips (recorder step-trigger units)."""


@dataclass
class ForkCfg:
  """The fork verb's load behavior (SESSION tier; see ``run fork``).

  WHAT a fork loads is runner-owned config (``runner.load.fork``; override
  leaves bare, e.g.
  ``runner.load.fork.discriminator=false`` forks an AMP run with a fresh
  discriminator); this block only carries HOW strictly state dicts apply.
  """

  strict: bool = True


@dataclass
class DebugVisCfg:
  """Per-invocation debug visualization (SESSION tier; never frozen).

  Drawing a reference ghost does not change the measured trajectory, so debug
  viz is a host-process choice, not part of the frozen env tree. Applied onto
  the command at play/eval build time (see run_exec). The human-ghost SOURCE
  derives from partner_mode (auto -> entity for g1, motion otherwise)."""

  enabled: bool = False
  # Two orthogonal draws per reference (ghost mesh + coordinate frames); each is
  # an independent bool (no tri-state mode, no per-ref master). Off = both False.
  robot_ref_ghost: bool = True
  robot_ref_coord_frames: bool = False
  # The anchor-centered decomposition (BeyondMimic fig. A(i)): robot_ref_coord_frames
  # draws the raw reference + the live body, and these add the same bodies
  # carried onto an anchor -- adapted face onto the LIVE anchor, free face onto
  # its own. The faces coincide while the robot tracks the reference exactly.
  # Each transported face has BOTH representations; ghost for a readable
  # separation, frames for exact per-body poses. Ghost is the one to reach for
  # when the offsets are small (several faces of arrows pile up on each other).
  robot_ref_desired_ghost: bool = False
  robot_ref_desired_coord_frames: bool = False
  free_ref_desired_ghost: bool = False
  free_ref_desired_coord_frames: bool = False
  # Body-name lists (a `${body_sets.<name>}` reference in an overlay); an empty
  # list keeps the whole model. Ghosts want a `*_tree` set: the sparse reward
  # sets render as disconnected fragments.
  robot_ref_ghost_bodies: tuple[str, ...] = ()
  human_ref_ghost: bool = False
  human_ref_coord_frames: bool = False
  human_ref_ghost_bodies: tuple[str, ...] = ()
  human_source: str = "auto"  # auto | entity | motion
  # SoftMimic compliance viz -- TOGGLES ONLY (the free mesh + force data live on
  # the augmented CLIP, not here). free_ref_ghost overlays the un-forced
  # reference beside the served (force-adapted) ghost (the gap IS the yield);
  # compliance_force_arrow draws the scripted wrench at the reference forced
  # link. Both no-op unless the clips are augmented (carry free + contact).
  free_ref_ghost: bool = False
  # Session-tier ghost color / body-subset overrides. A color is a `viz_palette`
  # name (`"50_green_vivid"`) or a literal rgba. None keeps the env-tier value
  # (`env.commands.codancing.*_color`), so a config that sets colors at the env
  # tier stays authoritative unless a session explicitly overrides it.
  # free_ref_ghost_bodies None = borrow the served ghost's subset.
  robot_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  free_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  human_ref_ghost_color: tuple[float, float, float, float] | str | None = None
  free_ref_ghost_bodies: tuple[str, ...] | None = None
  compliance_force_arrow: bool = False
  # How the force/contact overlay is drawn. Both styles draw the same quantities;
  # they only assign "geometry" and "magnitude" to different primitives:
  #   arrow  -- the setpoint is a sphere, the force is one arrow pointing at that
  #             sphere (length |F|/K_env, capped at the sphere).
  #   spring -- a thin cylinder = the spring itself (link to setpoint); the force
  #             gets a separate fixed-length arrow that shows direction only,
  #             drawn offset.
  # Under a stiff K_env the stretch can be as small as 2 cm, so the arrow style's
  # arrow is buried entirely inside the sphere; spring stays readable.
  compliance_force_style: str = "arrow"
  # Also draw the original pre-re-anchor setpoint with the force arrows (makes the
  # rising-edge-frozen re-anchoring visible); off by default.
  reanchor: bool = False
  # Named force-viz channels: any subset of `viz_palette.FORCE_CHANNELS`. Every
  # marker has exactly one channel, so any subset composes a figure. None =
  # expand the two booleans: compliance_force_arrow -> scripted_wrench,
  # applied_wrench, setpoint_script, setpoint_active; reanchor ->
  # setpoint_per_frame, data_arrow, reanchor_transform, reanchor_base_per_frame.
  force_channels: tuple[str, ...] | None = None
  # Per-body style overrides, face name -> body name -> {ghost_color,
  # coord_axis_colors, coord_axis_alpha}. Repaints named bodies only;
  # everything else keeps its face default. Use it to pick out the bodies a
  # figure is about.
  body_styles: dict[str, dict[str, dict[str, Any]]] | None = None


@dataclass
class SceneVizCfg:
  """Main-scene render-only geom decoration (SESSION tier; never frozen).

  ``geoms`` is a list of :class:`EntityGeomVizCfg` entries -- per entity (prefix),
  WHICH visual meshes + collision geoms to SHOW (rest of the entity hidden) + the
  collision color -- grafted onto the env's ``scene.spec_fn`` at build time
  (``run_exec.apply_scene_viz``), the same way ``apply_debug_vis`` grafts the ghost
  flags. ``None`` (default) is a byte-identical no-op. Collision viz is thus a
  per-invocation render choice on ANY checkpoint. Touches only
  ``geom.group`` / ``geom.rgba`` -- never the measured trajectory."""

  geoms: list[EntityGeomVizCfg] | None = None
  body_colors: dict[str, tuple[float, float, float, float] | str] | None = None
  """Composed body name (entity prefix included, e.g.
  ``robot/left_wrist_yaw_link``) -> rgba, or a ``viz_palette`` name
  (``"85_slot_a"``). Recolors every geom of the named bodies on the LIVE
  entities (material dropped so the color shows) -- the live-robot counterpart
  of ``debug_vis.body_styles``, which takes the same names. The two stay
  separate mechanisms because they paint different things (this one edits the
  compiled spec's geoms, that one restyles a drawn ghost); they share the
  palette, not the code path. Render-only; an unknown body name fails the
  build."""
  show_contacts: bool = False
  """Global contact-point/force arrows (a render-only ``MjvOption`` flag, all envs
  -- no per-env). Wired to the native viewer's vopt (C-key toggles it) and the
  offscreen recorder's scene option. Contacts are dynamic, so unlike collision
  geoms they can't bake into the model -- this stays a single render flag."""
  show_sensor_lines: bool = True
  """Rangefinder rays (``mjVIS_RANGEFINDER``): the yellow lines of the
  hand-to-thigh distance sensors that MuJoCo draws by default. ``false`` drops them from the
  OFFSCREEN recorder's scene option (render-only; the sensors keep measuring)
  -- for figures where the rays only add clutter."""
  show_world_frame: bool = True
  """Draw the world-origin coordinate frame (``MjvOption.frame = mjFRAME_WORLD``).
  The native viewers always draw it; this carries the same flag into the OFFSCREEN
  recorder's scene option so captured videos show the origin frame too. Default on
  -- a render-only choice, never touches the trajectory."""


@dataclass
class VizCfg:
  """Render-only visualization block (SESSION tier). ``scene`` decorates the main
  scene's geoms (collision capsules / hidden visual); reference ghosts stay on the
  parallel ``debug_vis`` block. Default empty -> byte-identical."""

  scene: SceneVizCfg = field(default_factory=SceneVizCfg)


@dataclass
class LaunchCfg:
  """Where/how this invocation runs (SESSION tier; set in a run file's
  ``session.launch`` block or with ``--override session.launch.*``)."""

  log_root: str = "logs/rsl_rl"
  log_dir_prefix: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  torchrunx_log_dir: str | None = None
  device: str | None = None


@dataclass
class SessionCfg:
  """Everything describing the host process of ONE invocation (SESSION tier).

  Never frozen into checkpoints (``_build_bundle`` strips it before the
  snapshot) and grafted fresh from these defaults when a checkpoint anchors a
  run -- so a train-time launch choice (GPUs, log dirs, recorders, play-metric
  recording) can never silently govern a later play/eval on another machine.
  """

  agent_type: Literal["zero", "random", "trained"] = "trained"
  viewer: ViewerKind = "auto"
  checkpoint_file: str | None = None
  play_sessions_root: str | None = None
  """Root the play-session folder is created under (a timestamped subdir is
  still made per session). ``None`` (default) routes it to the checkpoint's run
  dir for trained agents and to ``logs/play_sessions`` for dummy agents and the
  frozen viewer. Set it to collect a session inside a caller-owned folder."""
  # Where the recorded frame's camera + scene option come from: "fixed" (the
  # offscreen renderer's default camera) or "live" (track a live native viewer).
  camera_source: Literal["fixed", "live"] = "fixed"
  video: SessionVideoCfg = field(default_factory=SessionVideoCfg)
  # Play-mode metric recording. The metric *suite* is owned by the command (env
  # tree); this only governs a play session's recording -- explicit opt-in per
  # invocation.
  metric: SessionMetricCfg = field(default_factory=SessionMetricCfg)
  fork: ForkCfg = field(default_factory=ForkCfg)
  launch: LaunchCfg = field(default_factory=LaunchCfg)
  # Per-invocation debug visualization (reference ghosts / frames). Applied
  # onto the command at play/eval build time (run_exec.apply_debug_vis); a
  # default `enabled=False` is a no-op, so the command keeps its frozen
  # defaults. Drawing a ghost never changes the measured trajectory, so this is
  # a host-process choice, not a frozen env knob.
  debug_vis: DebugVisCfg = field(default_factory=DebugVisCfg)
  # Per-invocation main-scene geom decoration (collision capsules / hidden visual).
  # Grafted onto scene.spec_fn at build time (run_exec.apply_scene_viz); a default
  # empty `scene.geoms` is a no-op. Render-only, never frozen.
  viz: VizCfg = field(default_factory=VizCfg)


@dataclass
class ExecCfg:
  """The typed view of a composed run's three root blocks (+ the algo-owned
  ``load_specs`` table, threaded here so every executor path sees it)."""

  motion: MotionCfg = field(default_factory=MotionCfg)
  eval: EvalCfg = field(default_factory=EvalCfg)
  session: SessionCfg = field(default_factory=SessionCfg)
  # WHAT a session/fork loads from a checkpoint, per purpose -- RUNNER-OWNED
  # config (`runner.load.{session,fork}`): each runner block
  # declares its own module vocabulary (PPO has no discriminator key; AMP
  # does), so applicability is struct-checked at compose and travels with the
  # frozen tree. Override leaves bare: `runner.load.session.discriminator=true`.
  load_specs: dict[str, dict[str, bool]] = field(default_factory=dict)

  @staticmethod
  def from_blocks(
    motion: Mapping[str, Any] | None,
    eval_block: Mapping[str, Any] | None,
    session: Mapping[str, Any] | None,
    load_specs: Mapping[str, Any] | None = None,
  ) -> ExecCfg:
    """Build the typed view from the (MISSING-dropped) root-block dicts.

    Strict on purpose: an unknown key raises ``TypeError`` so schema drift
    between the config tree and these dataclasses surfaces immediately.
    """
    return ExecCfg(
      motion=MotionCfg(**dict(motion or {})),
      eval=_eval_from_block(eval_block or {}),
      session=_session_from_block(session or {}),
      load_specs={
        str(purpose): {str(k): bool(v) for k, v in dict(spec or {}).items()}
        for purpose, spec in dict(load_specs or {}).items()
      },
    )


def _entity_geom_viz_from_block(item: Mapping[str, Any]) -> EntityGeomVizCfg:
  e = dict(item)
  color = e.get("collision_color")
  return EntityGeomVizCfg(
    prefix=str(e["prefix"]),
    show_visual=tuple(e.get("show_visual") or ()),
    show_collision=tuple(e.get("show_collision") or ()),
    collision_color=tuple(color) if color is not None else None,
    collision_alpha=e.get("collision_alpha"),
  )


def _viz_from_block(data: Mapping[str, Any]) -> VizCfg:
  d = dict(data)
  scene_raw = dict(d.pop("scene", {}) or {})
  geoms_raw = scene_raw.pop("geoms", None)
  geoms = (
    [_entity_geom_viz_from_block(e) for e in geoms_raw]
    if geoms_raw is not None
    else None
  )
  scene = SceneVizCfg(geoms=geoms, **scene_raw)
  return VizCfg(scene=scene, **d)


def _eval_recipe_from_block(data: Mapping[str, Any]) -> EvalRecipeCfg:
  d = dict(data)
  video_raw = d.pop("video", {})
  if not isinstance(video_raw, Mapping):
    raise TypeError(
      f"eval recipe video must be a block, got {video_raw!r}. The recorder enable "
      "is video.enabled (parallel to session.video.enabled); the eval clip LENGTH "
      "is length_* (the rollout length), not video.length_*."
    )
  metric = EvalMetricCfg(**dict(d.pop("metric", {}) or {}))
  # `None`/absent -> inherit session.debug_vis at build; a block -> construct a
  # full DebugVisCfg that REPLACES the session block wholesale (no field merge).
  debug_vis_raw = d.pop("debug_vis", None)
  debug_vis = DebugVisCfg(**dict(debug_vis_raw)) if debug_vis_raw is not None else None
  viz_raw = d.pop("viz", None)
  viz = _viz_from_block(viz_raw) if viz_raw is not None else None
  return EvalRecipeCfg(
    video=EvalVideoCfg(**dict(video_raw or {})),
    metric=metric,
    debug_vis=debug_vis,
    viz=viz,
    **d,
  )


def _eval_from_block(data: Mapping[str, Any]) -> EvalCfg:
  d = dict(data)
  recipes_raw = d.pop("recipes", {}) or {}
  if not isinstance(recipes_raw, Mapping):
    raise TypeError(
      f"eval.recipes must be a mapping of name -> recipe block, got {recipes_raw!r}."
    )
  recipes = {
    str(name): _eval_recipe_from_block(block or {})
    for name, block in recipes_raw.items()
  }
  if "default" not in recipes:
    raise ValueError(
      "eval.recipes must include a 'default' recipe -- it is the offline "
      "`run eval` default and the single-eval baseline."
    )
  return EvalCfg(recipes=recipes, **d)


def _session_from_block(data: Mapping[str, Any]) -> SessionCfg:
  d = dict(data)
  video = SessionVideoCfg(**dict(d.pop("video", {}) or {}))
  metric = SessionMetricCfg(**dict(d.pop("metric", {}) or {}))
  fork = ForkCfg(**dict(d.pop("fork", {}) or {}))
  launch = LaunchCfg(**dict(d.pop("launch", {}) or {}))
  debug_vis = DebugVisCfg(**dict(d.pop("debug_vis", {}) or {}))
  viz = _viz_from_block(d.pop("viz", {}) or {})
  return SessionCfg(
    video=video,
    metric=metric,
    fork=fork,
    launch=launch,
    debug_vis=debug_vis,
    viz=viz,
    **d,
  )


def length_steps_from(
  length_s: float | None, length_steps: int, step_dt: float, *, what: str
) -> int:
  """Resolve a ``length_s`` / ``length_steps`` pair to env steps.

  ``length_s`` (simulated seconds == playback seconds, since clips play back
  at ``render_fps = 1/step_dt``) wins when non-null -- setting it is an
  explicit opt-in, the base default is null. One INFO line makes the
  conversion visible.
  """
  if length_s is not None:
    assert step_dt > 0, f"step_dt must be positive, got {step_dt}."
    steps = max(1, round(length_s / step_dt))
    print(f"[INFO] {what}: length_s={length_s}s -> {steps} steps (step_dt={step_dt}).")
    return steps
  return length_steps


def env_step_dt(env_cfg: Any) -> float:
  """Control dt of an env cfg: ``decimation * physics timestep``."""
  return float(env_cfg.decimation) * float(env_cfg.sim.mujoco.timestep)
