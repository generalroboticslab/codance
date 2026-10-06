"""Config-driven executor for composed codancing runs.

``compose_run`` builds a ``RunBundle`` (env cfg + runner cfg + runner class +
runtime knobs) and this module *runs* it -- training or play -- from the bundle
alone.

The *construction layer* is declarative:

* ``resolve_viewer_kind`` turns ``viewer == "auto"`` into ``native`` when a
  display is set and ``viser`` when none is.
* ``CodancingPlaySession`` owns build + run for every play mode (streaming,
  frozen, headless, viser): it picks the viewer object, the policy, the camera
  source, and the recorder, and runs all play-mode validation in one place.
* ``RecorderSpec`` builds the recorder from a small typed spec. A
  ``camera_source`` (fixed / live) selects ``CameraVideoRecorder`` over the
  plain ``VideoRecorder``, so train / play / frozen recording share one path.

Run-shaping choices (viewer kind, frozen env semantics, collision viz, motion
selection, eval metric) are data: the env cfg already carries the
overlay-applied semantics, and ``ExecCfg`` (see ``config/run_cfg``) carries the
three non-env root blocks -- frozen-tier ``motion`` / ``eval`` plus the
live-only ``session`` describing THIS invocation's host process.
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from rsl_rl.algorithms.amp_contract import AMP_GROUP_KEY, AMP_OBS_KEY
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.codancing.config.compose_run import (
  build_eval_env_cfg,
  eval_recipe_trees,
)
from mjlab.tasks.codancing.config.freeze import read_checkpoint_meta
from mjlab.tasks.codancing.config.run_cfg import (
  DebugVisCfg,
  EvalRecipeCfg,
  ExecCfg,
  LaunchCfg,
  MotionCfg,
  SessionCfg,
  SessionVideoCfg,
  ViewerKind,
  VizCfg,
  env_step_dt,
  length_steps_from,
)
from mjlab.tasks.codancing.eval_analysis import analyze
from mjlab.tasks.codancing.mdp.commands import (
  get_codancing_command_cfg,
  resolve_color,
)
from mjlab.tasks.codancing.rollout_record import EnvRowsRecorder
from mjlab.tasks.shared.utils import (
  setup_wandb_runtime_dir,
  start_training_meta_wandb_updater,
  start_wandb_symlink_watcher,
  timestamped_run_dir_name,
  write_git_recovery_snapshot,
  write_training_meta,
)
from mjlab.tasks.shared.utils.logging import tee_stdout_stderr
from mjlab.tasks.shared.utils.motion import resolve_motion_path
from mjlab.tasks.shared.wrapper.manager_based_rl_env import ManagerBasedRlEnvWrapper
from mjlab.tasks.shared.wrapper.rollout import (
  RolloutSink,
  observe_step,
)
from mjlab.tasks.shared.wrapper.scene import (
  append_spec_fn,
  apply_body_colors,
  apply_scene_geom_viz,
)
from mjlab.tasks.shared.wrapper.viewer import (
  CameraSource,
  CameraVideoRecorder,
  CodancingNativeFrozenViewer,
  CodancingNativeStreamingViewer,
  CodancingViserViewer,
  FixedCfgCameraSource,
  LiveViewerCameraSource,
)
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

__all__ = [
  "launch_training",
  "run_resume",
  "run_fork",
  "run_play",
  "run_eval",
  "apply_debug_vis",
  "enforce_resume_allowlist",
  "resolve_fork_load_cfg",
]


def _warn_if_command_metrics_empty(env_cfg: Any, *, context: str) -> None:
  """Warn early when a metric-collecting rollout will snapshot nothing.

  ``command.metrics`` is the product of ``metric_suites`` x ``metric_body_groups``
  (see ``iter_metric_suite_keys``); emptying either yields no metric keys, so the
  rollout runs but records empty metric columns. Launch sites call this so the
  user is told *before* the rollout.
  """
  command_cfg = get_codancing_command_cfg(env_cfg)
  if command_cfg.metric_suites and command_cfg.metric_body_groups:
    return
  print(
    f"[WARN] {context}: command.metrics will be empty "
    f"(metric_suites={tuple(command_cfg.metric_suites)}, "
    f"metric_body_groups={tuple(command_cfg.metric_body_groups)}) -- the rollout "
    "will run but log no metric summary. Set a metric suite in the run file "
    "(env.commands.codancing.metric_suites / metric_body_groups)."
  )


# ---------------------------------------------------------------------------
# Shared: motion-file resolution. The eval-metric suite is owned by the command
# (env cfg), never copied here.
# ---------------------------------------------------------------------------


def _resolve_motion(env_cfg: Any, motion: MotionCfg) -> None:
  """Resolve the motion manifest and hand the typed clips to the command.

  Runs at execution time so registry/wandb path resolution stays out of pure
  composition. The library parse is the single home for the schema +
  partner-axis contract; the command consumes the clips directly.
  """
  assert env_cfg.commands is not None
  command = get_codancing_command_cfg(env_cfg)

  from mjlab.tasks.codancing.motion import build_motion_library

  library = build_motion_library(
    motion,
    require_human_motion=command.require_human_motion,
    resolve_path=resolve_motion_path,
  )
  command.clips = library.clips
  _wire_foot_follow_offset(env_cfg, library)

  if command.active_clip_obs_mode != "none":
    from mjlab.managers import ObservationTermCfg
    from mjlab.tasks.codancing.mdp import observations as codancing_obs

    obs_term = ObservationTermCfg(
      func=codancing_obs.active_clip_obs,
      params={"command_name": "codancing"},
    )
    assert env_cfg.observations is not None
    for group_name, group in env_cfg.observations.items():
      # The AMP contract groups have a fixed width handshake with the expert
      # data (obs["amp"] must match build_amp_data's feature_dim;
      # obs["amp_group"] is the per-env disc-group index) -- the clip term is
      # policy/critic conditioning only, so injecting it there would desync
      # the discriminator from the expert features.
      if group_name in (AMP_OBS_KEY, AMP_GROUP_KEY):
        continue
      group.terms["active_clip_obs"] = obs_term


def _wire_foot_follow_offset(env_cfg: Any, library: Any) -> None:
  """Wire the foot-follow `offset_xy` from the clips' `foot_follow_offset`.

  The manifest is the single per-clip home for the footfall-target offset.
  The reward/termination params are
  per-run statics, so a mixed-offset pool fails fast; clips that omit the
  field keep the task default.
  """
  from mjlab.tasks.codancing.motion.library import PairedMotionClip

  offsets = {
    c.foot_follow_offset
    for c in library.clips
    if isinstance(c, PairedMotionClip) and c.foot_follow_offset is not None
  }
  if not offsets:
    return
  if len(offsets) > 1:
    raise ValueError(
      f"Clips disagree on foot_follow_offset ({sorted(offsets)}); the foot "
      "reward/termination offset_xy is one value per run."
    )
  offset_xy = next(iter(offsets))
  for term_name in ("robot_right_foot_front", "robot_left_foot_front"):
    term = (env_cfg.rewards or {}).get(term_name)
    if term is not None and "offset_xy" in term.params:
      term.params["offset_xy"] = offset_xy
  term = (env_cfg.terminations or {}).get("robot_foot_far_from_human_foot")
  if term is not None and "offset_xy" in term.params:
    term.params["offset_xy"] = offset_xy
  print(f"[INFO] foot_follow_offset from the manifest: {offset_xy}")


def apply_debug_vis(env_cfg: Any, debug_vis: DebugVisCfg) -> None:
  """Write the SESSION-tier debug-viz request onto the command at build time.

  Debug visualization (reference ghost mesh / coordinate frames) lives in the
  ``session`` tier, not the frozen env tree -- drawing a ghost does not change
  the measured trajectory, so it is a per-invocation host choice. The command
  carries the frozen-default debug-viz fields; this grafts the session values
  onto them at every RENDER build -- play (``CodancingPlaySession.run``) and
  offline ``run eval`` -- but never onto the frozen tree itself (the
  checkpoint's embedded env/eval trees stay viz-free; each render path applies
  THIS invocation's session). An ``enabled=False`` request turns the command's
  master draw switch off.

  The human-ghost SOURCE (``human_source``) stays on the command's existing
  ``auto`` derivation from ``partner_mode`` unless overridden. Fail fast when a
  human ghost is requested but the command has no human to draw
  (``partner_mode == "none"``).
  """
  # An explicit ``enabled: false`` means OFF: it turns the master draw switch
  # off (codancing bakes ``debug_vis=true`` + the robot ghost on), so the
  # render is clean.
  command = get_codancing_command_cfg(env_cfg)
  if not debug_vis.enabled:
    command.debug_vis = False
    return
  if debug_vis.compliance_force_style not in ("arrow", "spring"):
    raise ValueError(
      "session.debug_vis.compliance_force_style must be 'arrow' or 'spring', got "
      f"{debug_vis.compliance_force_style!r}. 'arrow' draws one arrow per contact "
      "capped at its setpoint; 'spring' splits the spring geometry (a thin cylinder "
      "link -> setpoint) from the force direction (a fixed-length offset arrow), "
      "which stays readable when a stiff K_env makes the stretch only a couple of cm."
    )
  human_requested = debug_vis.human_ref_ghost or debug_vis.human_ref_coord_frames
  if human_requested and command.partner_mode == "none":
    raise ValueError(
      "session.debug_vis.human_ref_ghost/frames requests a human reference, but "
      "the command's partner_mode is 'none' (no human exists to draw). Use a "
      "paired config (partner_mode != none), or drop the human-ref request "
      "(session.debug_vis.human_ref_ghost=false, human_ref_coord_frames=false)."
    )
  # Set the master switch and EVERY sub-flag explicitly, so the command's baked
  # defaults never leak into the requested draw. Each DebugVisCfg field except
  # `enabled` maps 1:1 onto a command `debug_vis_<name>` attribute.
  command.debug_vis = True
  for f in fields(DebugVisCfg):
    if f.name != "enabled":
      setattr(command, f"debug_vis_{f.name}", getattr(debug_vis, f.name))


def apply_scene_viz(env_cfg: Any, viz: VizCfg) -> None:
  """Graft the SESSION-tier main-scene geom decoration onto the env at build time.

  The render-only counterpart of :func:`apply_debug_vis` for the MAIN scene
  (collision capsules / hidden visual, body colors). Appends spec_fns (which run
  before compile, exactly like any ``scene.spec_fns`` entry) carrying THIS
  invocation's directives -- so collision-viz works on ANY checkpoint at
  play/eval build and is never frozen into the tree. Empty ``viz.scene.geoms``
  and ``viz.scene.body_colors`` append nothing. Touches only ``geom.group`` /
  ``geom.rgba``, so it never changes the measured trajectory; applied at the
  same render builds as ``apply_debug_vis`` (play / offline eval)."""
  entries = viz.scene.geoms
  if entries:
    append_spec_fn(env_cfg, functools.partial(apply_scene_geom_viz, entries=entries))
  body_colors = viz.scene.body_colors
  if body_colors:
    resolved = {
      body: resolve_color(color, f"viz.scene.body_colors[{body!r}]")
      for body, color in body_colors.items()
    }
    append_spec_fn(env_cfg, functools.partial(apply_body_colors, body_colors=resolved))


# ---------------------------------------------------------------------------
# Resolvers: checkpoint specs, artifact names, the viewer kind.
# ---------------------------------------------------------------------------


def _link_checkpoint_cache_to_local_home(run_id: str, home_dir: Path) -> None:
  """Point ``logs/wandb_checkpoints/<run_id>`` at the run's local log home.

  With the cache entry as a symlink, the cache path keeps working as a
  shortcut while checkpoint bytes and every checkpoint-anchored session
  (``play_sessions/``, ``eval_sessions/``) live in the real run dir.
  """
  shortcut = Path("logs") / "wandb_checkpoints" / run_id
  if shortcut.is_symlink():
    if shortcut.resolve() == home_dir.resolve():
      return
    shortcut.unlink()
  elif shortcut.exists():
    # A real dir here may hold downloads or session artifacts; never silently
    # replace user data with a link.
    print(
      f"[INFO] {shortcut} is a real directory; keeping it (move its contents "
      f"into {home_dir} and delete it to adopt the shortcut)."
    )
    return
  shortcut.parent.mkdir(parents=True, exist_ok=True)
  shortcut.symlink_to(Path(os.path.relpath(home_dir, shortcut.parent)))
  print(f"[INFO] checkpoint cache shortcut: {shortcut} -> {home_dir}")


def _latest_local_checkpoint(home_dir: Path) -> str | None:
  """Numerically-latest ``model_<it>.pt`` name in the local home, or None."""
  import re

  its = [
    int(m.group(1))
    for p in home_dir.glob("model_*.pt")
    if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))
  ]
  return f"model_{max(its)}.pt" if its else None


def resolve_checkpoint_spec(spec: str) -> Path:
  """Resolve a checkpoint *spec* to a local ``.pt`` path.

  A local path passes through; otherwise the spec is a wandb run path
  (``entity/project/run_id``, optionally ``:model_<it>.pt``). The run's wandb
  config names its ``log_dir`` (cwd-relative, so it transfers across
  machines). Resolution order per source of the bytes:

  1. the local home (the synced run dir; also linked as the cache shortcut),
  2. the wandb-uploaded checkpoint files (only runs trained with
     ``runner.upload_model=true`` have them).

  A bare spec resolves "latest" against wandb's file list, else against the
  local home. A failed wandb file listing reports "nothing here" and the chain
  continues; only when every source comes up empty does resolution fail --
  with a message naming what was tried.
  """
  from mjlab.utils.os import get_wandb_checkpoint_path

  path = Path(spec)
  if path.exists():
    return path
  run_path, _, checkpoint_name = spec.partition(":")
  if run_path.count("/") == 2:  # entity/project/run_id
    import re

    import wandb

    run_id = run_path.rsplit("/", 1)[-1]
    try:
      run = wandb.Api().run(run_path)
    except Exception as exc:  # noqa: BLE001 -- name the failing lookup
      raise FileNotFoundError(
        f"Cannot query wandb for run path {run_path!r}: {exc!r}"
      ) from exc
    log_dir = run.config.get("log_dir")
    home_dir = Path(log_dir) if log_dir else None

    if home_dir is not None and home_dir.is_dir():
      _link_checkpoint_cache_to_local_home(run_id, home_dir)

    # The run's uploaded checkpoint names, fetched lazily AT MOST once: a
    # local hit must resolve without touching the wandb file API at all.
    wandb_files: list[str] | None = None

    def uploaded() -> list[str]:
      nonlocal wandb_files
      if wandb_files is None:
        try:
          wandb_files = [
            f.name
            for f in run.files(pattern="model_%.pt")
            if re.fullmatch(r"model_\d+\.pt", f.name)
          ]
        except Exception:
          # A run with no uploads can crash the file paginator (its GraphQL
          # page comes back empty); an unreachable API must degrade the same
          # way -- this source reports "nothing here", the chain continues.
          wandb_files = []
      return wandb_files

    # Which checkpoint? Explicit name wins; a bare spec asks each source for
    # its latest: wandb uploads, then the local sync.
    name = checkpoint_name or None
    if name is None and uploaded():
      name = max(uploaded(), key=lambda x: int(x.split("_")[1].split(".")[0]))
    if name is None and home_dir is not None and home_dir.is_dir():
      name = _latest_local_checkpoint(home_dir)
    if name is None:
      raise FileNotFoundError(
        f"Run {run_path} has no checkpoints visible anywhere: no uploaded "
        "model_*.pt on wandb (runner.upload_model is off by default) and no "
        f"synced local run dir{f' at {log_dir}' if log_dir else ''}."
      )

    # Where are the bytes? Local home, then the wandb upload.
    if home_dir is not None and (home_dir / name).exists():
      local = home_dir / name
      print(f"[INFO] checkpoint spec {spec!r} -> {local} (local run dir).")
      return local
    if name in uploaded():
      resolved, was_cached = get_wandb_checkpoint_path(
        Path("logs"), Path(run_path), name
      )
      print(
        f"[INFO] checkpoint spec {spec!r} -> {resolved} "
        f"({'cache' if was_cached else 'downloaded from wandb'})."
      )
      return resolved
    raise FileNotFoundError(
      f"Checkpoint {name} of run {run_path} is not in the local run dir"
      f"{f' ({log_dir})' if log_dir else ''} and is not among the wandb uploads "
      "(runner.upload_model=true runs only). Sync the run dir or pick an "
      "available checkpoint."
    )
  raise FileNotFoundError(
    f"Checkpoint spec {spec!r} is neither an existing local path nor a wandb "
    "run path (entity/project/run_id[:model_<it>.pt])."
  )


def _now(fmt: str = "%Y%m%d_%H%M%S") -> str:
  """Current wall-clock timestamp, used for log-dir / video file names."""
  return datetime.now().strftime(fmt)


def _unique_path(path: Path) -> Path:
  """``path`` if free, else the first ``<stem>-<n><suffix>`` that is.

  The wall-clock stamps in run-artifact names are second-resolution, so two
  invocations in the same second (sweep loops forking one checkpoint, scripted
  back-to-back resumes/evals) would otherwise silently overwrite each other.
  """
  if not path.exists():
    return path
  for n in itertools.count(1):
    candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    if not candidate.exists():
      return candidate
  raise AssertionError("unreachable")


def _mkdir_unique(path: Path) -> Path:
  """Create ``path`` (parents ok), uniquifying with ``-<n>`` on collision.

  Same rationale as ``_unique_path``; ``mkdir(exist_ok=False)`` is the
  collision *detector* (atomic), the suffix retry is the resolution -- two
  same-second forks both get a dir instead of the second crashing.
  """
  for n in itertools.count():
    candidate = path if n == 0 else path.with_name(f"{path.name}-{n}")
    try:
      candidate.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
      continue
    return candidate
  raise AssertionError("unreachable")


def _viewer_auto(requested: str) -> str:
  """Resolve ``viewer="auto"`` to a concrete kind from the display environment.

  A display (X11 / Wayland) -> the interactive ``native`` viewer; headless ->
  the browser-based ``viser`` viewer. Any non-``auto`` value passes through.
  """
  if requested != "auto":
    return requested
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def resolve_viewer_kind(session: SessionCfg) -> ViewerKind:
  """Resolve the session's (possibly ``auto``) viewer kind to a concrete one."""
  return cast(ViewerKind, _viewer_auto(session.viewer))


# ---------------------------------------------------------------------------
# Recorder spec (a typed, lazily built VideoRecorder construction).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecorderSpec:
  """Declarative spec for the offscreen recorder (built lazily on env).

  Captures the per-context naming + length so train / play / frozen recording
  share one builder instead of near-identical inline constructions. The
  ``step_trigger`` is derived from ``interval`` (``None`` -> record once at
  step 0; otherwise every ``interval`` steps). When ``camera_source`` is set the
  builder returns a :class:`CameraVideoRecorder` (fixed / live
  camera); otherwise the plain :class:`VideoRecorder` (train default).
  """

  video_folder: Path
  video_length: int
  name_prefix: str | None = None
  interval: int | None = None
  disable_logger: bool = True
  camera_source: CameraSource | None = None

  def step_trigger(self) -> Callable[[int], bool]:
    interval = self.interval
    if interval is None:
      return lambda step: step == 0
    return lambda step: step % interval == 0

  def build(self, env: ManagerBasedRlEnv) -> VideoRecorder:
    kwargs: dict[str, Any] = dict(
      env=env,
      video_folder=self.video_folder,
      step_trigger=self.step_trigger(),
      video_length=self.video_length,
      disable_logger=self.disable_logger,
    )
    if self.name_prefix is not None:
      kwargs["name_prefix"] = self.name_prefix
    if self.camera_source is not None:
      return CameraVideoRecorder(camera_source=self.camera_source, **kwargs)
    return VideoRecorder(**kwargs)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _dim_str(shape: Any) -> str:
  """Render an obs-term shape tuple as ``3`` / ``2x3`` (unknown -> ``?``)."""
  try:
    return "x".join(str(int(x)) for x in shape) or "0"
  except (TypeError, ValueError):
    return "?"


def _obs_grouping(env: Any) -> dict[str, dict[str, str]]:
  """Per-obs-group signatures for wandb grouping, from the built observation
  manager. For each observation group (actor, critic, ...) emits a nested
  ``{group: {dim, single_step, history}}`` -- wandb auto-flattens it to columns
  ``grouping.obs.<group>.dim`` / ``.single_step`` / ``.history`` -- each a
  ``"name:VAL|..."`` string in the policy's input concatenation order (so a
  reorder shows up), groupable/filterable in a way a Hydra choice label cannot
  express (it names only one composition slot, not the terms):

  * ``dim`` -- each term's stacked/flattened width, i.e. what the network input
    actually contains.
  * ``single_step`` -- each term's PER-FRAME width, before history stacking, so
    runs group by obs CONTENT independent of the history. This is the DEFAULT
    group-by axis: ``dim`` collides across base×history (e.g. 6x5 == 10x3 == 30),
    ``single_step`` does not.
  * ``history`` -- each term's history length (``0`` = not stacked), so runs
    group by "how much history" per term. All three list the SAME terms in the
    SAME order, so you can read across them.

  Per term ``dim == single_step x max(history, 1)``, so the three overlap -- each
  is kept (the injectivity backstop) because the wandb GUI groups a column, it
  cannot divide two. Reads the resolved per-term cfgs (post group-level
  override), the only source of ``history_length``. Assumes a built env; the
  caller guards a raise.
  """
  mgr = env.unwrapped.observation_manager
  term_dims = mgr.group_obs_term_dim
  term_cfgs = mgr._group_obs_term_cfgs  # parallel to active_terms; has history_length
  out: dict[str, dict[str, str]] = {}
  for group_name, term_names in mgr.active_terms.items():
    group = str(group_name)
    dim_parts, single_step_parts, history_parts = [], [], []
    for name, dim, cfg in zip(
      term_names,
      term_dims.get(group_name, []),
      term_cfgs.get(group_name, []),
      strict=False,
    ):
      history_length = getattr(cfg, "history_length", 0)
      dim_parts.append(f"{name}:{_dim_str(dim)}")
      single_step_parts.append(f"{name}:{math.prod(dim) // (history_length or 1)}")
      history_parts.append(f"{name}:{history_length}")
    out[group] = {
      "dim": "|".join(dim_parts),
      "single_step": "|".join(single_step_parts),
      "history": "|".join(history_parts),
    }
  return out


def _build_grouping(composition: dict[str, Any] | None, env: Any) -> dict[str, Any]:
  """Assemble the wandb ``train_cfg.grouping`` block: the Hydra composition
  (``choices`` / ``config_name`` featured, ``launch`` provenance) plus nested
  per-obs-group signatures under ``obs``. The composition is ALWAYS logged
  (building ``dict(composition or {})`` cannot raise); the obs block is
  best-effort on top, so a shape surprise drops only that block, never the
  featured ``choices`` axis, and never aborts training. Tolerates
  ``composition=None`` (resume / raw ``--config`` / fork)."""
  grouping: dict[str, Any] = dict(composition or {})
  try:
    grouping["obs"] = _obs_grouping(env)
  except Exception as exc:  # noqa: BLE001 -- obs is best-effort; never kill training
    print(f"[WARN] wandb obs grouping skipped: {exc!r}")
  return grouping


def _wandb_run_group(composition: dict[str, Any] | None) -> str | None:
  """This run's variant name for wandb's native ``group`` (so seeds cluster):
  the single-file ``config_name``. ``None`` when it is not known (resume / raw
  ``--config``), so ``group`` is left unset."""
  variant = (composition or {}).get("config_name")
  return str(variant) if variant else None


def _wants_offscreen_render(video_enabled: bool, rank: int) -> bool:
  """Whether this rank needs the ``rgb_array`` offscreen renderer. Only rank 0
  records the session video, so only rank 0 needs it; gating on rank keeps a
  multi-GPU run from building an unused EGL context per non-zero rank."""
  return bool(video_enabled) and rank == 0


def _eval_camera_source(viz: VizCfg | None) -> CameraSource | None:
  """The offscreen camera source for an eval clip: the world-frame / contact
  render flags from the resolved scene viz, so eval clips carry the same scene
  decorations play wires instead of a bare recorder."""
  if viz is None:
    return None
  return FixedCfgCameraSource(
    show_contacts=viz.scene.show_contacts,
    show_world_frame=viz.scene.show_world_frame,
    show_sensor_lines=viz.scene.show_sensor_lines,
  )


def _run_train(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  log_dir: Path,
  resolved_cfg_yaml: str | None = None,
  resolved_eval_cfg_yaml: str | None = None,
  resume_from: Path | None = None,
  fork_from: Path | None = None,
  fork_load_cfg: dict[str, bool] | None = None,
  fork_strict: bool = True,
  log_basename: str = "train",
  composition: dict[str, Any] | None = None,
) -> None:
  """One training process (train / resume / fork share this body).

  ``resume_from`` = exact pickup: full restore (weights + optimizer +
  iteration; rsl_rl ``load_cfg=None``) -- the optimizer's param-group LRs and
  Adam moments come back; with an adaptive-KL schedule the controller's LR
  restarts from the frozen config value and re-adapts within a few updates,
  with a fixed schedule the config LR is the LR. ``fork_from`` = weights-only
  init via ``fork_load_cfg`` (fresh optimizer, iteration 0). In all cases
  ``runner.max_iterations`` is the run's TOTAL budget: the learn loop runs
  ``max_iterations - current_learning_iteration`` more iterations.
  """
  assert not (resume_from and fork_from), "resume_from and fork_from are exclusive"
  session = exec_cfg.session
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = agent_cfg.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    cuda_devices = cuda_visible.split(",")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = cuda_devices[local_rank]
    device = f"cuda:{local_rank}"
    seed = agent_cfg.seed + rank

  log_name = f"{log_basename}.log" if rank == 0 else f"{log_basename}_rank{rank}.log"
  teardown_tee = tee_stdout_stderr(log_dir / log_name)

  configure_torch_backends()
  agent_cfg.seed = seed
  env_cfg.seed = seed
  _resolve_motion(env_cfg, exec_cfg.motion)

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env: Any = ManagerBasedRlEnvWrapper(
    cfg=env_cfg,
    device=device,
    render_mode="rgb_array"
    if _wants_offscreen_render(session.video.enabled, rank)
    else None,
  )

  # Everything past env construction runs guarded: a crash anywhere below
  # (runner construct, the learn loop itself -- NaN, OOM, Ctrl-C) must still
  # flush the in-progress video clip + release the offscreen GL context
  # (env.close()) and restore the tee'd stdio (teardown_tee()), matching the
  # play/frozen/eval session paths.
  try:
    if _wants_offscreen_render(session.video.enabled, rank):
      env = RecorderSpec(
        video_folder=Path(log_dir) / "videos" / "train",
        video_length=_session_video_length_steps(session.video, env_cfg),
        interval=session.video.interval_steps,
      ).build(cast(ManagerBasedRlEnv, env))

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # Log a `train_cfg.grouping` block (store_config -> _json_safe) so runs are
    # groupable by what distinguishes them (see `_build_grouping`): the Hydra
    # composition is always logged, the nested per-obs-group signatures
    # `grouping.obs.<group>.{dim,single_step,history}` are best-effort on top.
    # Sits OUTSIDE store_config's own guard; never aborts a run.
    train_cfg = asdict(agent_cfg)
    train_cfg["grouping"] = _build_grouping(composition, env)
    runner: Any = cast(Any, runner_cls or OnPolicyRunner)(
      env, train_cfg, str(log_dir), device
    )
    # Freeze contract: carry the resolved tree so every saved checkpoint embeds it
    # (plus the dedicated eval env trees, so offline `run eval` reproduces them).
    runner.resolved_cfg_yaml = resolved_cfg_yaml
    if hasattr(runner, "resolved_eval_cfg_yaml"):
      runner.resolved_eval_cfg_yaml = resolved_eval_cfg_yaml
    # Train-invocation provenance (informational; the freeze contract above is
    # untouched -- loads still graft a fresh session). JSON round-trip turns
    # dataclass tuples into lists so the embedded yaml stays plain-typed.
    if hasattr(runner, "train_session_yaml"):
      import json

      import yaml as _yaml

      runner.train_session_yaml = _yaml.safe_dump(
        {"session": json.loads(json.dumps(asdict(session), default=str))},
        sort_keys=False,
      )
      runner.train_command = invocation_command()
    add_wandb_tags(list(agent_cfg.wandb_tags))
    # Cluster same-variant runs (seeds) under wandb's native `group`, named after
    # this run's config file. Read natively by the lazy wandb.init, and only
    # when a config name is known.
    _variant = _wandb_run_group(composition)
    if _variant:
      os.environ.setdefault("WANDB_RUN_GROUP", _variant)
    runner.add_git_repo_to_log(__file__)
    if resume_from is not None:
      # Full restore: weights + optimizer + iteration (rsl_rl default load_cfg).
      runner.load(str(resume_from), map_location=device)
      print(
        f"[INFO] resume: restored {resume_from.name} at iteration "
        f"{runner.current_learning_iteration}."
      )
    elif fork_from is not None:
      # Weights init only (per-algorithm load spec; optimizer fresh, iteration 0).
      runner.load(
        str(fork_from),
        load_cfg=fork_load_cfg,
        strict=fork_strict,
        map_location=device,
      )
      print(f"[INFO] fork: initialized weights from {fork_from} ({fork_load_cfg}).")

    if rank == 0:
      write_git_recovery_snapshot(log_dir=log_dir, repository_file_path=__file__)
      agent_cfg_dict = asdict(agent_cfg)
      agent_cfg_dict["log_dir_prefix"] = session.launch.log_dir_prefix or ""
      # The returned stable identity is stamped onto the runner so every
      # checkpoint's `infos['run_meta']` self-describes the same run.
      runner.run_meta = write_training_meta(
        log_dir=log_dir,
        task_id=agent_cfg.experiment_name,
        train_script=__spec__.name if __spec__ else __name__,
        agent_cfg=agent_cfg_dict,
        repository_file_path=__file__,
        play_script="mjlab.scripts.run",
      )
      start_training_meta_wandb_updater(log_dir)
      dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
      dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
      # Also dump the frozen resolved tree alongside the split param files (the
      # checkpoint embeds it too; this is the human-readable mirror).
      if resolved_cfg_yaml is not None:
        (log_dir / "params" / "resolved.yaml").write_text(resolved_cfg_yaml)
      # Log the dedicated eval envs' resolved trees too -- eval config is then as
      # reproducible as the train tree (each recipe's overlays, e.g. play DR
      # semantics, are captured here). Always present (always composed).
      if resolved_eval_cfg_yaml is not None:
        # One readable file per recipe (the embedded checkpoint keeps the whole
        # mapping); the manifest references resolved_eval.{recipe}.yaml.
        for _rname, _tree in eval_recipe_trees(resolved_eval_cfg_yaml).items():
          (log_dir / "params" / f"resolved_eval.{_rname}.yaml").write_text(_tree)
      # Record THIS invocation's session too -- NOT part of the freeze contract
      # (frozen trees deliberately carry no session); purely a debugging record of
      # what host-process knobs this train run actually used.
      dump_yaml(
        log_dir / "params" / "session.yaml",
        {
          "_note": "Session settings of this invocation; not frozen into checkpoints.",
          "session": asdict(session),
        },
      )
      if session.launch.log_dir_prefix:
        start_wandb_symlink_watcher(log_dir=log_dir)

    # `runner.max_iterations` is the TOTAL budget; the learn loop runs the
    # remainder past the (possibly restored) current iteration.
    current = int(getattr(runner, "current_learning_iteration", 0))
    remaining = int(agent_cfg.max_iterations) - current
    if remaining <= 0:
      raise ValueError(
        f"checkpoint is already at iteration {current} >= runner.max_iterations "
        f"({agent_cfg.max_iterations}); to train longer, resume with "
        "`--override runner.max_iterations=<larger total>`."
      )
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
  finally:
    env.close()
    teardown_tee()


def _session_video_length_steps(video: SessionVideoCfg, env_cfg: Any) -> int:
  """Resolve the G1 (train/play) recorder's clip length in env steps."""
  return length_steps_from(
    video.length_s, video.length_steps, env_step_dt(env_cfg), what="session.video"
  )


def launch_training(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  resolved_cfg_yaml: str | None = None,
  *,
  resolved_eval_cfg_yaml: str | None = None,
  composition: dict[str, Any] | None = None,
) -> None:
  """Create the log dir (with optional prefix symlink) and run training.

  The log dir is ``<log_root>/<experiment>/<timestamp>_<run_name>``;
  ``resolved_cfg_yaml`` (the frozen tree) is threaded to ``_run_train`` so the
  runner embeds it into every checkpoint.
  ``resolved_eval_cfg_yaml`` is the dedicated, separately-composed eval env's
  resolved tree (see ``compose_run``); it is logged + frozen so offline
  ``run eval`` can reproduce it.
  """
  if bool(getattr(agent_cfg, "resume", False)):
    raise ValueError(
      "`run train` does not read runner.resume. Use `run resume` with an "
      "explicit checkpoint: `run resume --ckpt <run>/model_<it>.pt`."
    )
  launch = exec_cfg.session.launch
  log_root_path = Path(launch.log_root) / agent_cfg.experiment_name
  log_dir_name = timestamped_run_dir_name(log_root_path)
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  log_dir = log_root_path / log_dir_name

  if launch.log_dir_prefix:
    log_dir_prefix_path = Path(launch.log_dir_prefix).expanduser().resolve()
    # Fail fast when the prefix root is absent (e.g. an external disk not
    # mounted in a container): silently creating it inside the container fs
    # would strand the run's data. The prefix root must pre-exist; only the run
    # subtree below it is created here.
    if not log_dir_prefix_path.is_dir():
      raise FileNotFoundError(
        f"session.launch.log_dir_prefix does not exist: {log_dir_prefix_path}. "
        "Mount/create it first, or unset it (the base default is null -> logs go "
        "under session.launch.log_root)."
      )
    wandb_runtime_dir, wandb_local_dir_created = setup_wandb_runtime_dir(
      log_dir_prefix_path
    )
    print(f"[INFO] W&B runtime dir redirected to: {wandb_runtime_dir}")
    if wandb_local_dir_created:
      print("[INFO] W&B local dir created: wandb/")

    actual_log_dir = log_dir_prefix_path / log_root_path / log_dir_name
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_root_path.mkdir(parents=True, exist_ok=True)
    if log_dir.exists() or log_dir.is_symlink():
      if log_dir.is_symlink():
        if log_dir.resolve() != actual_log_dir:
          raise ValueError(
            "log_dir already exists with a different target: "
            f"{log_dir} -> {log_dir.resolve()}, expected {actual_log_dir}"
          )
      else:
        raise ValueError(f"log_dir already exists & is not a symlink: {log_dir}")
    else:
      os.symlink(actual_log_dir, log_dir)
  else:
    log_root_path.mkdir(parents=True, exist_ok=True)

  _spawn_train_processes(
    env_cfg,
    agent_cfg,
    runner_cls,
    exec_cfg,
    log_dir,
    resolved_cfg_yaml,
    resolved_eval_cfg_yaml,
    composition=composition,
  )


def _setup_gpu_env(launch: LaunchCfg) -> int:
  """Set CUDA/MuJoCo env vars from the launch GPU selection; return GPU count."""
  selected_gpus, num_gpus = select_gpus(launch.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  return num_gpus


def _ensure_headless_gl() -> None:
  """Default ``MUJOCO_GL`` to EGL for windowless offscreen rendering.

  Train workers get this via ``_setup_gpu_env``; the eval and headless/viser
  play paths render offscreen too (recorder, eval video), and on a remote
  machine with a stale ``DISPLAY`` the process default would crash GLFW init
  before the rollout starts. An explicit user ``MUJOCO_GL`` always wins.
  """
  os.environ.setdefault("MUJOCO_GL", "egl")


def _spawn_train_processes(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  log_dir: Path,
  resolved_cfg_yaml: str | None,
  resolved_eval_cfg_yaml: str | None,
  resume_from: Path | None = None,
  fork_from: Path | None = None,
  fork_load_cfg: dict[str, bool] | None = None,
  fork_strict: bool = True,
  log_basename: str = "train",
  composition: dict[str, Any] | None = None,
) -> None:
  """Run ``_run_train`` directly (single GPU) or via torchrunx (multi-GPU)."""
  launch = exec_cfg.session.launch
  num_gpus = _setup_gpu_env(launch)
  train_args = (
    env_cfg,
    agent_cfg,
    runner_cls,
    exec_cfg,
    log_dir,
    resolved_cfg_yaml,
    resolved_eval_cfg_yaml,
    resume_from,
    fork_from,
    fork_load_cfg,
    fork_strict,
    log_basename,
    composition,
  )
  if num_gpus <= 1:
    _run_train(*train_args)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if launch.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = launch.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
      + ("MUJOCO*", "WANDB*", "MJLAB*"),
    ).run(
      _run_train,
      env_cfg,
      agent_cfg,
      runner_cls,
      exec_cfg,
      log_dir,
      resolved_cfg_yaml,
      resolved_eval_cfg_yaml,
      resume_from,
      fork_from,
      fork_load_cfg,
      fork_strict,
      log_basename,
      composition,
    )


# ---------------------------------------------------------------------------
# Resume / fork (the two checkpoint-anchored training verbs)
# ---------------------------------------------------------------------------

# Training-trajectory-neutral delta prefixes a `resume` may apply. session.* is
# execution context (the machine may differ); eval.* is the frozen no-grad
# measurement spec; the runner entries are duration/cadence/logging-only.
# Everything else fails fast -- changing the conditions is what `fork` is for.
# A recipe's overlays are additionally rejected (compose-time-only; see
# `is_recipe_overlays_path`).
RESUME_ALLOWED_PREFIXES: tuple[str, ...] = (
  "session.",
  "eval.",
  "runner.max_iterations",
  "runner.save_interval",
  "runner.wandb_",
  "runner.logger",
  "runner.upload_model",
)


def is_recipe_overlays_path(path: str) -> bool:
  """True for ``eval.recipes.<name>.overlays`` or a path below it.

  A recipe's overlays are consumed at live compose only: its eval env tree is
  frozen with the checkpoint, so a later change could never take effect.
  """
  parts = path.split(".")
  return len(parts) >= 4 and parts[:2] == ["eval", "recipes"] and parts[3] == "overlays"


def reject_recipe_overlay_deltas(changed: Sequence[str]) -> None:
  """Fail fast when a frozen anchor's deltas change any recipe's overlays.

  ``changed`` is the leaf diff of the frozen tree with and without this
  invocation's deltas (``resume_delta_paths``), so every spelling that reaches
  an overlay list is caught: the list itself, a whole-recipe mapping, a new
  recipe.
  """
  offenders = [path for path in changed if is_recipe_overlays_path(path)]
  if offenders:
    raise SystemExit(
      f"these changes touch eval recipe overlays: {offenders}. Recipe overlays "
      "are applied when the config is composed: the eval env trees are already "
      "frozen in the checkpoint, so the change could never take effect. Change "
      "the eval env with --overlay instead; to change a recipe's overlays, "
      "compose a new run with `fork --ckpt ... --config FILE`."
    )


def enforce_resume_allowlist(changed: Sequence[str]) -> None:
  """Fail fast unless every changed dotted path is resume-neutral."""
  offenders = [
    path
    for path in changed
    if is_recipe_overlays_path(path) or not path.startswith(RESUME_ALLOWED_PREFIXES)
  ]
  if offenders:
    raise ValueError(
      "resume means picking the run up under EXACTLY the same conditions; "
      f"these deltas change them: {offenders}. Allowed prefixes: "
      f"{list(RESUME_ALLOWED_PREFIXES)} (minus eval.recipes.<name>.overlays). "
      "To train under changed conditions starting from these weights, use "
      "`run fork --ckpt ... [--overlay/--override ...]` instead."
    )


def _resolve_load_cfg(
  load_specs: Mapping[str, Mapping[str, bool]], purpose: str, *, what: str
) -> dict[str, bool]:
  """Look up the runner-owned purpose spec -> ``runner.load(load_cfg=...)``.

  The vocabulary is per-algo (PPO has no ``discriminator`` key; AMP does), so
  applicability and typos are struct-checked at compose -- override leaves
  bare (``runner.load.session.discriminator=true``). Here only the purpose
  entry itself can be missing (a runner.load block that omits it).
  """
  if purpose not in load_specs:
    raise ValueError(
      f"{what}: the composed runner declares no `runner.load.{purpose}` spec "
      f"(has: {sorted(load_specs) or 'none'}). Declare it in the run file's "
      "runner.load block (see conf/decoupled.yaml for the shape)."
    )
  return {str(k): bool(v) for k, v in load_specs[purpose].items()}


def resolve_fork_load_cfg(exec_cfg: ExecCfg) -> dict[str, bool]:
  """The fork weights-load spec: the runner-owned ``runner.load.fork`` table.

  Values come from the composed runner's own load table (e.g. AMP keeps the
  trained discriminator; ``runner.load.fork.discriminator=false`` forks a fresh
  one).
  """
  return _resolve_load_cfg(exec_cfg.load_specs, "fork", what="runner.load (fork)")


def _write_run_manifest(
  log_dir: Path,
  filename: str,
  *,
  kind: str,
  checkpoint: Path,
  deltas: Mapping[str, Any],
) -> Path:
  manifest = {
    "kind": kind,
    "created": _now("%Y-%m-%d_%H-%M-%S"),
    "checkpoint": _path_forms(checkpoint, base=log_dir),
    "deltas": {k: list(v) for k, v in deltas.items()},
  }
  # Manifests are an audit record -- never silently overwrite an earlier one
  # (same-second stamps collide; see _unique_path).
  out = _unique_path(log_dir / filename)
  dump_yaml(out, manifest)
  return out


def run_resume(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  *,
  checkpoint_file: str,
  resolved_cfg_yaml: str | None,
  resolved_eval_cfg_yaml: str | None = None,
  deltas: Mapping[str, Any] | None = None,
) -> None:
  """Exact pickup of a run: SAME run dir, full restore, allowed-diff enforced.

  The caller (run.py) enforces the allowed-diff spec on the frozen text before
  assembly; by the time this runs, the deltas are known resume-neutral. LR
  semantics on restore: the optimizer state (param-group LRs + Adam moments)
  comes back via ``runner.load``; with an adaptive-KL schedule the
  controller's Python-side LR restarts from the frozen config value and
  re-adapts within a few updates, with a fixed schedule the config LR applies.
  """
  ckpt = Path(checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
  log_dir = ckpt.parent
  manifest = _write_run_manifest(
    log_dir,
    f"resume_manifest-{_now()}.yaml",
    kind="resume",
    checkpoint=ckpt,
    deltas=deltas or {},
  )
  print(f"[INFO] resume: continuing {log_dir} (manifest: {manifest.name})")
  _spawn_train_processes(
    env_cfg,
    agent_cfg,
    runner_cls,
    exec_cfg,
    log_dir,
    resolved_cfg_yaml,
    resolved_eval_cfg_yaml,
    resume_from=ckpt,
    log_basename="resume",
  )


def fork_log_dir(checkpoint: Path, exec_cfg: ExecCfg, run_name: str = "") -> Path:
  """The fork run dir: nested under the parent run by default.

  ``<parent_run>/forks/<stamp>[_run_name]`` keeps variant runs discoverable
  next to the parent checkpoint (and follows the parent onto another disk when
  the parent dir is a prefix symlink). Overriding
  ``session.launch.log_root`` opts into the standard flat layout instead.
  """
  launch = exec_cfg.session.launch
  stamp = _now("%Y-%m-%d_%H-%M-%S")
  if run_name:
    stamp += f"_{run_name}"
  if launch.log_root != LaunchCfg().log_root:
    return Path(launch.log_root) / stamp
  return checkpoint.parent / "forks" / stamp


def run_fork(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  *,
  checkpoint_file: str,
  resolved_cfg_yaml: str | None,
  resolved_eval_cfg_yaml: str | None = None,
  deltas: Mapping[str, Any] | None = None,
  composition: dict[str, Any] | None = None,
) -> None:
  """A NEW run starting from the checkpoint's weights (fresh optimizer, iter 0).

  Config is whatever the caller assembled (frozen anchor + deltas, or a
  ``--config`` recompose where the checkpoint supplies weights only). The fork
  embeds its own frozen trees like any train run, so `eval --ckpt` / `resume
  --ckpt` on its checkpoints work directly.
  """
  ckpt = Path(checkpoint_file)
  if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
  load_cfg = resolve_fork_load_cfg(exec_cfg)
  log_dir = _mkdir_unique(
    fork_log_dir(ckpt, exec_cfg, getattr(agent_cfg, "run_name", "") or "")
  )
  _write_run_manifest(
    log_dir,
    "fork_manifest.yaml",
    kind="fork",
    checkpoint=ckpt,
    deltas=deltas or {},
  )
  print(f"[INFO] fork: new run at {log_dir} (weights from {ckpt.name})")
  _spawn_train_processes(
    env_cfg,
    agent_cfg,
    runner_cls,
    exec_cfg,
    log_dir,
    resolved_cfg_yaml,
    resolved_eval_cfg_yaml,
    fork_from=ckpt,
    fork_load_cfg=load_cfg,
    fork_strict=exec_cfg.session.fork.strict,
    composition=composition,
  )


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------


def run_bounded_rollout(
  env: Any,
  policy: Any,
  *,
  total_steps: int,
  observer: Any = None,
  progress_desc: str | None = None,
) -> None:
  """No-grad rollout core shared by headless play and offline eval.

  Steps ``env`` with ``policy`` for ``total_steps`` env steps; the optional
  ``RolloutSink`` records each step through the shared ``observe_step`` entry
  point. The caller owns the ``env`` lifecycle (build + close) and the policy.
  """
  from tqdm import tqdm

  obs = env.get_observations()
  for _ in tqdm(range(total_steps), desc=progress_desc or "rollout", unit="step"):
    with torch.no_grad():
      actions = policy(obs)
    obs = env.step(actions)[0]
    observe_step(observer, env)


def session_record_yaml(session: Any) -> str:
  """The ``resolved_session.yaml`` payload (--dry-run and the session-dir
  archive write the same artifact): the live session record, stamped live-only
  so nobody mistakes it for one of the frozen trees."""
  record = OmegaConf.to_yaml(OmegaConf.create(asdict(session)))
  return (
    "# Session settings of this invocation; not part of the frozen trees.\n" + record
  )


@dataclass
class CodancingPlaySession:
  """Owns build + run for every play mode (streaming, frozen, headless, viser).

  One place picks the viewer object, policy, camera source, and recorder, and
  runs all play-mode validation -- ``run_play`` just delegates here. Streaming /
  viser / headless share the "policy viewer" build (env -> recorder ->
  RslRlVecEnvWrapper -> policy); native-frozen builds its own env + zero-policy
  viewer. The metric recorders (one shared ``RolloutSink``) plug into the
  viewers and the headless loop identically.
  """

  env_cfg: Any
  agent_cfg: Any
  runner_cls: type | None
  exec_cfg: ExecCfg
  device: str = field(init=False, default="cpu")
  resolved_viewer: ViewerKind = field(init=False, default=cast(ViewerKind, "none"))
  camera_source: CameraSource | None = field(init=False, default=None)
  # The per-(step, env) rollout table (`session.metric.enabled`), built with
  # the env.
  env_rows_recorder: Any = field(init=False, default=None)
  # The composed run's resolved yaml (threaded from the CLI bundle); archived
  # into the session dir beside the manifest so a session folder answers
  # "what exactly ran" without re-composing.
  resolved_cfg_yaml: str | None = field(init=False, default=None)
  run_meta: Mapping[str, Any] | None = field(init=False, default=None)
  checkpoint_path: Path | None = field(init=False, default=None)

  @property
  def session(self) -> SessionCfg:
    return self.exec_cfg.session

  def run(self) -> None:
    configure_torch_backends()
    self.device = self.session.launch.device or (
      "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    _resolve_motion(self.env_cfg, self.exec_cfg.motion)
    # Only the train path sets the env seed; default a play env to the runner
    # seed so two identical sessions draw the same clips/DR, and an explicit
    # `--override env.seed=N` (predeclared in the catalog) wins.
    if getattr(self.env_cfg, "seed", None) is None:
      self.env_cfg.seed = getattr(self.agent_cfg, "seed", None)
    apply_debug_vis(self.env_cfg, self.session.debug_vis)
    apply_scene_viz(self.env_cfg, self.session.viz)
    self.resolved_viewer = resolve_viewer_kind(self.session)
    if self.resolved_viewer in ("none", "viser"):
      _ensure_headless_gl()
    self._validate()
    # A loaded checkpoint names its run (the embedded `run_meta`) in the session
    # manifest; a zero-action play has no run.
    self.checkpoint_path = (
      resolve_checkpoint_spec(self.session.checkpoint_file)
      if self.session.checkpoint_file
      else None
    )
    self.run_meta = (
      read_checkpoint_meta(self.checkpoint_path).run_meta
      if self.checkpoint_path is not None
      else None
    )
    self.camera_source = self._build_camera_source()

    print(f"[INFO] Playing with viewer: {self.resolved_viewer}")
    if self.resolved_viewer == "native-frozen":
      self._run_frozen()
    else:
      self._run_policy_viewer()

  # -- validation + shared builders --------------------------------------

  def _validate(self) -> None:
    if self.session.metric.enabled:
      _warn_if_command_metrics_empty(self.env_cfg, context="play session.metric")
    if self.session.camera_source == "live":
      if not self.session.video.enabled:
        raise ValueError(
          "session.camera_source='live' requires session.video.enabled=True."
        )
      if self.resolved_viewer not in ("native", "native-frozen"):
        raise ValueError(
          "session.camera_source='live' needs a live native viewer "
          f"(session.viewer=native or native-frozen), got {self.resolved_viewer}."
        )

  def _build_camera_source(self) -> CameraSource | None:
    if not self.session.video.enabled:
      return None
    if self.session.camera_source == "live":
      return LiveViewerCameraSource()  # viewer attached after it is built.
    # Fixed camera: carry the global contacts + world-frame flags so the offscreen
    # recorder matches the native viewer (collision viz bakes into the model).
    return FixedCfgCameraSource(
      show_contacts=self.session.viz.scene.show_contacts,
      show_world_frame=self.session.viz.scene.show_world_frame,
      show_sensor_lines=self.session.viz.scene.show_sensor_lines,
    )

  def _build_env_with_recorder(self, video_folder: Path, name_prefix: str) -> Any:
    """Build the play env, wrapped for recording when session.video is on.

    Shared by the policy-viewer and frozen paths (only folder/prefix differ).
    """
    render_mode = "rgb_array" if self.session.video.enabled else None
    env: Any = ManagerBasedRlEnvWrapper(
      cfg=self.env_cfg, device=self.device, render_mode=render_mode
    )
    # The rollout table lands next to this session's clips and carries the SAME
    # name prefix they do, so one folder holds the mp4s and their rows.
    if self.session.metric.enabled:
      self.env_rows_recorder = EnvRowsRecorder(
        out_path=video_folder / f"{name_prefix}-rollout-envs.csv"
      )
    if self.session.video.enabled:
      env = RecorderSpec(
        video_folder=video_folder,
        video_length=_session_video_length_steps(self.session.video, self.env_cfg),
        name_prefix=name_prefix,
        camera_source=self.camera_source,
      ).build(cast(ManagerBasedRlEnv, env))
    return env

  def _build_policy(
    self, env: Any, effective_agent: str, dummy_mode: bool, resume_path: Path | None
  ) -> Any:
    if dummy_mode:
      action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
      policy_device = env.unwrapped.device
      if effective_agent == "zero":

        class PolicyZero:
          def __call__(self, obs) -> torch.Tensor:
            del obs
            return torch.zeros(action_shape, device=policy_device)

        return PolicyZero()

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=policy_device) - 1

      return PolicyRandom()

    runner: Any = cast(Any, self.runner_cls or OnPolicyRunner)(
      env, asdict(self.agent_cfg), device=self.device
    )
    assert resume_path is not None
    load_cfg = resolve_session_load_cfg(self.exec_cfg)
    runner.load(str(resume_path), load_cfg=load_cfg, map_location=self.device)
    return runner.get_inference_policy(device=self.device)

  def _session_observer(self) -> RolloutSink | None:
    """The session's recorder (the rollout table) as a sink, or None.

    The viewers take a single :class:`RolloutSink`. The same sink is reused
    at teardown for ``finalize``, whose errors propagate (play fails loudly).
    """
    return self.env_rows_recorder

  # -- per-mode runners ---------------------------------------------------

  def _run_policy_viewer(self) -> None:
    """Streaming native / viser / headless: build env + recorder + policy."""
    effective_agent = self.session.agent_type
    if self.session.checkpoint_file is None and self.session.agent_type == "trained":
      effective_agent = "zero"
      print("[INFO] No session.checkpoint_file provided, using zero-action policy.")
    dummy_mode = effective_agent in {"zero", "random"}

    log_dir: Path | None = None
    resume_path: Path | None = None
    if not dummy_mode:
      assert self.session.checkpoint_file is not None
      resume_path = resolve_checkpoint_spec(self.session.checkpoint_file)
      log_dir = resume_path.parent

    if not dummy_mode:
      assert log_dir is not None and resume_path is not None
      # One dir per play session: a session emits data (rollout table) alongside
      # its clips, and a dir outside `videos/` keeps play artifacts out of the
      # training video sweep's rglob should this run later resume.
      sessions_root = (
        Path(self.session.play_sessions_root)
        if self.session.play_sessions_root
        else log_dir / "play_sessions"
      )
      video_folder = sessions_root / timestamped_run_dir_name(sessions_root)
      video_prefix = resume_path.stem
    else:
      # Dummy sessions have no run dir to live in, but they dump the same
      # session-shaped set (manifest, clips, rollout table), so each gets its own
      # dir under the global root instead of prefix-named flat files.
      sessions_root = (
        Path(self.session.play_sessions_root)
        if self.session.play_sessions_root
        else Path("logs") / "play_sessions"
      )
      video_folder = sessions_root / timestamped_run_dir_name(sessions_root)
      video_prefix = effective_agent
    from mjlab.tasks.codancing.session_describe import describe_session
    from mjlab.tasks.codancing.viz_describe import (
      describe_visualization,
      inputs_from_session,
    )

    # The parsed clips `_resolve_motion` put on the command (`motion.motions`
    # itself is the manifest's path).
    motions = list(get_codancing_command_cfg(self.env_cfg).clips or [])
    session_text = describe_session(
      env_cfg=self.env_cfg,
      session=self.session,
      motions=motions,
      agent_type=effective_agent,
      checkpoint_spec=None if dummy_mode else self.session.checkpoint_file,
      visualization=describe_visualization(inputs_from_session(self.env_cfg, motions)),
    )
    print("[session] " + session_text.replace("\n", "\n[session] "))
    write_play_session_manifest(
      out_dir=video_folder,
      filename=f"{video_prefix}.yaml",
      agent_type=effective_agent,
      checkpoint=resume_path,
      checkpoint_spec=None if dummy_mode else self.session.checkpoint_file,
      run_meta=self.run_meta,
      session_description=session_text,
    )
    # The FULL resolved trees, archived beside the manifest: the env
    # tree as composed (when the CLI threaded it) and the live session record.
    if self.resolved_cfg_yaml:
      (video_folder / "resolved_config.yaml").write_text(self.resolved_cfg_yaml)
    (video_folder / "resolved_session.yaml").write_text(
      session_record_yaml(self.session)
    )
    env = self._build_env_with_recorder(video_folder, video_prefix)

    env = RslRlVecEnvWrapper(env, clip_actions=self.agent_cfg.clip_actions)
    policy = self._build_policy(env, effective_agent, dummy_mode, resume_path)

    try:
      if self.resolved_viewer == "viser":
        observer = self._session_observer()
        CodancingViserViewer(env, policy, step_observer=observer).run()
      elif self.resolved_viewer == "none":
        self._run_headless(env, policy)
      else:
        observer = self._session_observer()
        viewer = CodancingNativeStreamingViewer(
          env,
          policy,
          show_contacts=self.session.viz.scene.show_contacts,
          step_observer=observer,
        )
        if isinstance(self.camera_source, LiveViewerCameraSource):
          self.camera_source.attach_viewer(viewer)
        viewer.run()
    finally:
      # Sink finalize does file I/O and can raise -- the env/GL teardown must
      # run regardless, so it gets its own finally.
      try:
        observer = self._session_observer()
        if observer is not None:
          observer.finalize()
      finally:
        env.close()

  def _run_headless(self, env: Any, policy: Any) -> None:
    """Headless step loop for offscreen video capture (no GUI)."""
    observer = self._session_observer()
    # Sessions run a FIXED length (there are no in-process stop conditions):
    # `session.metric.num_steps` when set, else the requested capture length
    # (the session exists to produce the clip, so the clip bounds the session).
    total_steps = self.session.metric.num_steps or _session_video_length_steps(
      self.session.video, self.env_cfg
    )

    run_bounded_rollout(
      env,
      policy,
      total_steps=total_steps,
      observer=observer,
      progress_desc="Headless capture",
    )

  def _run_frozen(self) -> None:
    """Native manual-stepping viewer (frozen env semantics arrive via overlay)."""
    from mjlab.viewer import EnvProtocol

    sessions_root = (
      Path(self.session.play_sessions_root)
      if self.session.play_sessions_root
      else Path("logs") / "play_sessions"
    )
    video_folder = sessions_root / timestamped_run_dir_name(sessions_root)
    video_prefix = "frozen"
    write_play_session_manifest(
      out_dir=video_folder,
      filename=f"{video_prefix}.yaml",
      # The frozen viewer steps the env manually; no policy acts, whatever
      # session.agent_type says (a --ckpt here anchors the WORLD only).
      agent_type="none",
      checkpoint=self.checkpoint_path,
      checkpoint_spec=self.session.checkpoint_file,
      run_meta=self.run_meta,
    )
    env = self._build_env_with_recorder(video_folder, video_prefix)

    observer = self._session_observer()

    viewer = CodancingNativeFrozenViewer(
      env=cast(EnvProtocol, env),
      # frame_rate omitted: the frozen viewer owns its own display-refresh default
      # (symmetric with the streaming viewer, which is also built without one).
      show_contacts=self.session.viz.scene.show_contacts,
      step_observer=observer,
    )
    if isinstance(self.camera_source, LiveViewerCameraSource):
      self.camera_source.attach_viewer(viewer)

    try:
      viewer.run()
    finally:
      # Same shape as the streaming path: a finalize failure must not skip
      # the env/GL teardown.
      try:
        observer = self._session_observer()
        if observer is not None:
          observer.finalize()
      finally:
        env.close()


def run_play(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  resolved_cfg_yaml: str | None = None,
) -> None:
  """Delegate to :class:`CodancingPlaySession` (build + run owned there)."""
  session = CodancingPlaySession(
    env_cfg=env_cfg,
    agent_cfg=agent_cfg,
    runner_cls=runner_cls,
    exec_cfg=exec_cfg,
  )
  session.resolved_cfg_yaml = resolved_cfg_yaml
  session.run()


def resolve_session_load_cfg(exec_cfg: ExecCfg) -> dict[str, bool]:
  """The play/eval checkpoint-load spec: the runner-owned ``runner.load.session``.

  E.g. AMP's session spec pins ``discriminator: false`` (actor-only
  inference; the rsl_rl absent-key default would silently load it) -- opt in
  per run with the bare ``--override runner.load.session.discriminator=true``.
  """
  return _resolve_load_cfg(exec_cfg.load_specs, "session", what="runner.load (session)")


def select_eval_recipes(
  resolved_eval_cfg_yaml: str | None, recipe: str
) -> dict[str, str]:
  """The frozen recipe trees ``--recipe`` selects: the named one, or every one
  with ``all``. ``run.py`` calls it before the ``--dry-run`` exit too, so a dry
  run rejects the same recipe names the real run does.
  """
  # Boundary: eval mode FAITHFULLY reproduces the frozen eval recipes, so it
  # requires the dedicated eval envs (composed from each recipe's overlays and
  # frozen into the checkpoint). Without them, fail loudly and point at play,
  # the free-form "watch/record a policy" path.
  if resolved_eval_cfg_yaml is None:
    raise SystemExit(
      "eval mode reproduces the frozen eval recipes, but this checkpoint carries "
      "no dedicated eval env (its run trained without one: a fully resolved "
      "--config run with no eval tree, or a failed eval compose).\n"
      "To freely play/inspect this checkpoint instead, compose a play session:\n"
      "  uv run python -m mjlab.scripts.run play --ckpt <checkpoint.pt>"
    )
  trees = eval_recipe_trees(resolved_eval_cfg_yaml)
  if recipe == "all":
    return trees
  if recipe not in trees:
    raise SystemExit(
      f"--recipe {recipe!r} is not a frozen eval recipe of this checkpoint "
      f"({sorted(trees)}). Use one of those names or `--recipe all`."
    )
  return {recipe: trees[recipe]}


def run_eval(
  env_cfg: Any,
  agent_cfg: Any,
  runner_cls: type | None,
  exec_cfg: ExecCfg,
  *,
  resolved_eval_cfg_yaml: str | None,
  checkpoint_file: str,
  recipe: str = "default",
  deviations: Mapping[str, Sequence[str]] | None = None,
) -> None:
  """Reproduce a checkpoint's frozen eval recipe(s) offline.

  Each selected recipe runs :func:`_run_eval_session` (``run_bounded_rollout``
  + the recorded per-step fact table) with the checkpoint's policy. The eval env
  is built here (via ``build_eval_env_cfg``) from ``resolved_eval_cfg_yaml`` --
  the dedicated, play-semantics eval tree, read from the checkpoint's embedded
  freeze slot when loaded frozen. It is composed + frozen for every train run;
  it is missing only when a run trained without one (a raw ``--config`` run
  with no eval tree, or a failed eval compose), in which case eval fails loudly
  and points at play. Writes the rollout tables, a summary CSV and an
  eval-session manifest under ``<checkpoint log dir>/eval_sessions/<recipe>/``.
  """
  configure_torch_backends()
  _ensure_headless_gl()  # eval renders offscreen (video) with no window.
  device = exec_cfg.session.launch.device or (
    "cuda:0" if torch.cuda.is_available() else "cpu"
  )
  trees = select_eval_recipes(resolved_eval_cfg_yaml, recipe)

  checkpoint = Path(checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  log_dir = checkpoint.parent

  def make_policy(env: Any) -> Any:
    runner: Any = cast(Any, runner_cls or OnPolicyRunner)(
      env, asdict(agent_cfg), device=device
    )
    load_cfg = resolve_session_load_cfg(exec_cfg)
    runner.load(str(checkpoint), load_cfg=load_cfg, map_location=device)
    return runner.get_inference_policy(device=device)

  for recipe_name in trees:
    # Build this recipe's eval env lazily from its frozen tree. The tree is
    # session-stripped, so the debug-viz request comes from THIS invocation:
    # per-recipe `recipe.debug_vis` when set, else the session fallback
    # (--override session.debug_vis.* / a debug_vis overlay), grafted here.
    base_cfg = build_eval_env_cfg(trees[recipe_name])
    _resolve_motion(base_cfg, exec_cfg.motion)
    # Same seeding rule as play: offline eval envs default to the runner seed.
    if getattr(base_cfg, "seed", None) is None:
      base_cfg.seed = getattr(agent_cfg, "seed", None)
    _recipe = exec_cfg.eval.recipes[recipe_name]
    apply_debug_vis(base_cfg, _recipe.debug_vis or exec_cfg.session.debug_vis)
    apply_scene_viz(base_cfg, _recipe.viz or exec_cfg.session.viz)
    _warn_if_command_metrics_empty(base_cfg, context=f"eval [{recipe_name}]")
    # Uniquify under the recipe's eval-session dir so two same-second evals of
    # one checkpoint never overwrite each other's manifest/metrics pair.
    name_prefix = _unique_path(
      log_dir / "eval_sessions" / recipe_name / f"{checkpoint.stem}-{_now()}.yaml"
    ).stem
    _run_eval_session(
      eval_env_cfg=base_cfg,
      agent_cfg=agent_cfg,
      recipe=exec_cfg.eval.recipes[recipe_name],
      recipe_name=recipe_name,
      device=device,
      log_dir=log_dir,
      make_policy=make_policy,
      checkpoint=checkpoint,
      name_prefix=name_prefix,
      deviations=deviations,
      viz=_recipe.viz or exec_cfg.session.viz,
    )
    print(
      "[INFO] Wrote eval-session manifest: "
      f"{log_dir / 'eval_sessions' / recipe_name / name_prefix}.yaml"
    )


# ---------------------------------------------------------------------------
# Eval sessions and session manifests
# ---------------------------------------------------------------------------


def _latest_file(folder: Path, pattern: str) -> Path | None:
  """Newest match for ``pattern`` in ``folder``, or ``None``."""
  matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
  return matches[-1] if matches else None


def _existing(path: Path) -> Path | None:
  """``path`` if it exists, else ``None`` (so manifest path-forms stay clean)."""
  return path if path.exists() else None


def _path_forms(path: Path | None, *, base: Path) -> dict[str, str] | None:
  """All the path forms a downstream reproduction might want for ``path``.

  ``abs`` is the resolved absolute path; ``rel_to_log_dir`` is relative to the
  experiment log dir (the portable form when the whole run folder is moved);
  ``rel_to_cwd`` is relative to the current working dir (handy for a copy-paste
  command). ``None`` when ``path`` is missing.
  """
  if path is None:
    return None
  resolved = path.resolve()
  forms = {"abs": str(resolved), "rel_to_cwd": os.path.relpath(resolved, Path.cwd())}
  try:
    forms["rel_to_log_dir"] = str(resolved.relative_to(base.resolve()))
  except ValueError:
    pass
  return forms


def invocation_command() -> list[str]:
  """This process's CLI as a canonical list: one item per flag, values quoted.

  Joining with spaces reproduces a runnable command. Shared by the play/eval
  session manifests and the checkpoint-embedded train provenance.
  """
  command = ["uv run python -m mjlab.scripts.run"]
  for token in sys.argv[1:]:
    if token.startswith("--"):
      command.append(shlex.quote(token))
    else:
      command[-1] += " " + shlex.quote(token)
  return command


def write_play_session_manifest(
  *,
  out_dir: Path,
  filename: str,
  agent_type: str,
  checkpoint: Path | None,
  checkpoint_spec: str | None,
  run_meta: Mapping[str, Any] | None,
  session_description: str | None = None,
) -> Path:
  """Write a play session's manifest: the invocation that produced the dir.

  The play twin of :func:`write_eval_session_manifest`. Eval reconstructs a
  reproduce command from recorded deviations; play records its verbatim argv --
  every ``--override`` / ``--overlay`` that shaped the artifacts, which the
  frozen trees do not preserve. ``command`` is a list, one item per flag (each
  ``--flag`` starts an item, values stay with their flag), so the yaml reads one
  override per line; join with spaces to run it. Written before the env builds,
  so a crashed session still names its command. ``checkpoint_spec`` is the
  reference the session ran with -- for a wandb spec that is the cache/shortcut
  path it resolved to at the CLI (the as-typed spec sits in ``command``), kept
  beside ``checkpoint``'s fully symlink-resolved forms; ``run_meta`` is the
  checkpoint's embedded identity (task / run name / wandb ids / commit).
  """
  manifest: dict[str, Any] = {
    "created": _now("%Y-%m-%d_%H-%M-%S"),
    "command": invocation_command(),
    "agent_type": agent_type,
    "checkpoint": (
      None if checkpoint is None else _path_forms(checkpoint, base=checkpoint.parent)
    ),
    "checkpoint_spec": checkpoint_spec,
    "run": None if run_meta is None else dict(run_meta),
    # ONE generated description (session_describe, with the viz_describe text
    # as its final section): what the whole session IS and what renders.
    # Regenerated from the executed config, so it cannot drift.
    "session_description": session_description,
  }
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / filename
  dump_yaml(out_path, manifest)
  return out_path


def write_eval_session_manifest(
  *,
  out_dir: Path,
  filename: str,
  log_dir: Path,
  checkpoint: Path | None,
  recipe: str,
  eval_config: Path | None,
  video: Path | None,
  knobs: Mapping[str, Any],
  summary: Mapping[str, Any],
  deviations: Mapping[str, Sequence[str]] | None = None,
) -> Path:
  """Write a per-eval-session manifest: exactly what was evaluated + how to redo.

  Captures the checkpoint actually used, the eval env config tree, the produced
  video, the eval knobs, and the metric summary -- each path in absolute /
  log-dir-relative / cwd-relative forms -- so a single checkpoint's eval can be
  reproduced later (``run eval --ckpt <checkpoint> --recipe <recipe>``, every
  argument shell-quoted).

  ``deviations`` records this session's deltas from the frozen contract
  (overlays/overrides) with a stable shape: empty lists = a FAITHFUL
  reproduction of the frozen recipe; non-empty = a modified measurement,
  and the reproduce command re-applies them so the modified session is itself
  reproducible.
  """
  deviations = deviations or {}
  recorded_deviations = {
    "overlays": list(deviations.get("overlays", ())),
    "overrides": list(deviations.get("overrides", ())),
  }
  manifest: dict[str, Any] = {
    "created": _now("%Y-%m-%d_%H-%M-%S"),
    "checkpoint": _path_forms(checkpoint, base=log_dir),
    "eval_config": _path_forms(eval_config, base=log_dir),
    "video": _path_forms(video, base=log_dir),
    "knobs": dict(knobs),
    "deviations": recorded_deviations,
    "metrics_summary": {k: dict(v) for k, v in summary.items()},
  }
  if checkpoint is not None:
    delta_args = "".join(
      f" --overlay {shlex.quote(name)}" for name in recorded_deviations["overlays"]
    ) + "".join(
      f" --override {shlex.quote(ov)}" for ov in recorded_deviations["overrides"]
    )
    faithful = not (recorded_deviations["overlays"] or recorded_deviations["overrides"])
    ckpt_abs = shlex.quote(str(checkpoint.resolve()))
    manifest["reproduce"] = {
      "note": (
        "Rebuilds the eval env from the checkpoint's embedded eval tree"
        + ("." if faithful else " and re-applies this session's deviations.")
      ),
      "command": (
        "uv run python -m mjlab.scripts.run eval "
        f"--ckpt {ckpt_abs} --recipe {shlex.quote(recipe)}{delta_args}"
      ),
    }
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / filename
  dump_yaml(out_path, manifest)
  return out_path


def _run_eval_session(
  *,
  eval_env_cfg: Any,
  agent_cfg: Any,
  recipe: EvalRecipeCfg,
  recipe_name: str,
  device: str,
  log_dir: Path,
  make_policy: Callable[[Any], Any],
  checkpoint: Path,
  name_prefix: str,
  deviations: Mapping[str, Sequence[str]] | None = None,
  viz: VizCfg | None = None,
) -> Mapping[str, Any]:
  """One offline eval session: build env -> rollout -> analysis -> manifest.

  Builds a fresh ``num_envs``-shrunk eval env (optionally wrapped for video),
  hands it to ``make_policy``, runs a bounded no-grad rollout that records the
  per-step fact table, summarizes that table offline, writes an eval-session
  manifest, and always closes the env.
  """
  import copy

  pe = recipe
  cfg = copy.deepcopy(eval_env_cfg)
  cfg.scene.num_envs = pe.num_envs
  # One knob: the bounded-rollout length IS the eval-clip length (`length_s`
  # wins over `length_steps` when set; resolved against the eval env's dt).
  length_steps = length_steps_from(
    pe.length_s, pe.length_steps, env_step_dt(cfg), what="eval"
  )
  # Every artifact of THIS recipe's session (clip, tables, summary, manifest) is
  # namespaced under `eval_sessions/{recipe}/`, so recipes never collide and
  # each recipe's history is self-contained.
  session_dir = log_dir / "eval_sessions" / recipe_name
  render_mode = "rgb_array" if pe.video.enabled else None
  base_env: Any = ManagerBasedRlEnvWrapper(
    cfg=cfg, device=device, render_mode=render_mode
  )
  recorded: Any = base_env
  if pe.video.enabled:
    recorded = RecorderSpec(
      video_folder=session_dir,
      video_length=length_steps,
      name_prefix=name_prefix,
      camera_source=_eval_camera_source(viz),
    ).build(cast(ManagerBasedRlEnv, base_env))
  env = RslRlVecEnvWrapper(recorded, clip_actions=agent_cfg.clip_actions)

  # The per-(step, env) fact table is ALWAYS written in an eval session: it is
  # the single source the summaries and the manifest numbers are computed from
  # (offline, after the rollout).
  env_rows_recorder = EnvRowsRecorder(
    out_path=session_dir / f"{name_prefix}.rollout-envs.csv"
  )
  observer = env_rows_recorder
  closed = False
  try:
    policy = make_policy(env)
    run_bounded_rollout(
      env, policy, total_steps=length_steps, observer=observer, progress_desc="eval"
    )
    observer.finalize()
    # Summaries and the manifest numbers are computed OFFLINE from the
    # just-written fact table (summary CSV + analysis.json land beside it).
    results: dict[str, Any] = analyze(
      env_rows_recorder.path,
      settle=pe.metric.settle_steps,
      recipe=recipe_name,
    )
    # Close BEFORE naming the clip: the recorder writes the mp4 on close. Set
    # `closed` before the lookup so a lookup failure never re-triggers close in
    # `finally`. Name-prefixed so it matches THIS session's clip, not just the
    # newest file in the dir.
    env.close()
    closed = True
    video = (
      _latest_file(session_dir, f"{name_prefix}*.mp4") if pe.video.enabled else None
    )
    write_eval_session_manifest(
      out_dir=session_dir,
      filename=f"{name_prefix}.yaml",
      log_dir=log_dir,
      checkpoint=checkpoint,
      recipe=recipe_name,
      eval_config=_existing(log_dir / "params" / f"resolved_eval.{recipe_name}.yaml"),
      video=video,
      knobs={
        "recipe": recipe_name,
        "num_envs": pe.num_envs,
        "length_steps": length_steps,
        "length_s": pe.length_s,
        "video": pe.video.enabled,
        "overlays": list(pe.overlays),
      },
      summary=results.get("summary", {}),
      deviations=deviations,
    )
    return results
  finally:
    if not closed:
      env.close()
