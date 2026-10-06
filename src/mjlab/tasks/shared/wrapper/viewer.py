"""Shared viewers, per-env contact visualizer, step observer, and recording.

Layers (all viewer-task agnostic, no codancing imports):
- contacts are a single GLOBAL ``MjvOption`` flag (no per-env): the native viewers
  set it on their ``viewer.opt`` (C-key toggles), and the offscreen recorder gets
  it via a contacts-enabled ``scene_option`` from :class:`FixedCfgCameraSource`.
  Collision viz is baked into the model, so it needs no render option.
- the rollout-sink protocol (``RolloutSink``, in ``rollout.py``) + ``observe_step``
  helper, driving the native + viser play wrappers below.
- ``CameraSource`` protocol (+ fixed / live implementations) and
  ``CameraVideoRecorder``: one recording path subclassing core ``VideoRecorder``.
- ``CodancingNativeStreamingViewer`` / ``CodancingNativeFrozenViewer``: the two
  sibling native viewers (continuous vs manual-stepping), both composing the
  visualizer + observer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, cast, runtime_checkable

import mediapy as media
import mujoco
import numpy as np
import torch
import viser

from mjlab.tasks.shared.wrapper.rollout import RolloutSink, observe_step
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import (
  EnvProtocol,
  NativeMujocoViewer,
  PolicyProtocol,
  VerbosityLevel,
  ViserPlayViewer,
)
from mjlab.viewer.native.keys import KEY_C, KEY_D, KEY_F8, KEY_H, KEY_R, KEY_SPACE
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer

# Secondary-env catmask for offscreen mjv_addGeoms: include DECOR so per-env
# contact/force visualization renders even when that env is not the primary.
_SECONDARY_ENV_CATMASK = (
  mujoco.mjtCatBit.mjCAT_DYNAMIC.value | mujoco.mjtCatBit.mjCAT_DECOR.value
)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
  """A frame as contiguous ``uint8`` RGB (float [0,1] -> [0,255]), the conversion the
  core recorder applies at encode time, done per frame here because frames are
  streamed."""
  frame = np.asarray(frame)
  if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
  return frame


# Controlled motion-cursor pause ("H"): one shared toggle for every codancing
# viewer (native key, viser hotkey/button) so the two paths never drift.
_ACTION_TOGGLE_CURSOR_PAUSE = "toggle_cursor_pause"
# F8: a fresh episode reset, frozen at frame 0 with the force field evaluated
# on the spawn the way training's reset step leaves it (rsi_inspect). A
# function key because MuJoCo's viewer binds every letter to a visualization
# toggle (N recolors bodies by constraint island, and that toggle fires even
# with the UI hidden); F8, 9 and Insert were verified free by screen diff.
_ACTION_INSPECT_SPAWN = "inspect_spawn"
# "Hold until toggled back": the counter form of an indefinite pause
# (~231 sim-days at 50 Hz; episode resets clear it per env).
_CURSOR_PAUSE_HOLD_STEPS = 1_000_000_000


def toggle_codancing_cursor_pause(env: Any) -> bool:
  """Toggle the codancing motion-cursor pause; return the NEW paused state.

  Freezes / unfreezes the shared human+robot reference cursor while the POLICY
  keeps running, as if the partner had stopped. Distinct from the
  viewer's whole-sim pause (SPACE / the viser Pause button), which freezes time
  entirely. Mutates the command's per-env ``_pause_steps_left`` counter; episode
  resets clear it per env.

  Must run on the viewer's main loop (dispatched via the action queue), not a GUI
  callback thread, so it never races env stepping.
  """
  from mjlab.tasks.codancing.mdp.commands import get_codancing_command

  command = get_codancing_command(env.unwrapped)
  if bool((command._pause_steps_left > 0).any()):
    command._pause_steps_left.zero_()
    return False
  command._pause_steps_left.fill_(_CURSOR_PAUSE_HOLD_STEPS)
  return True


def _log_cursor_pause(viewer: Any, paused: bool) -> None:
  """Shared one-liner for the cursor-pause toggle (native key + viser hotkey)."""
  if paused:
    viewer.log(
      "[INFO] Motion cursor PAUSED -- the policy keeps running against the "
      "frozen reference; H resumes",
      VerbosityLevel.INFO,
    )
  else:
    viewer.log("[INFO] Motion cursor resumed (H toggles)", VerbosityLevel.INFO)


# The rollout-sink protocol + ``observe_step`` live in ``rollout.py`` (they drive
# the headless loop too, not only viewers); imported above and used by the native
# and viser play wrappers below.


# ---------------------------------------------------------------------------
# Camera source strategy for recording: where the offscreen frame's camera +
# scene option come from (fixed default camera / live viewer).
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsOverrideRender(Protocol):
  """Narrow protocol for the recorder: an env exposing ``render_with_overrides``
  (so ``CameraVideoRecorder`` is typed precisely, no casts)."""

  @property
  def unwrapped(self) -> Any: ...

  def render_with_overrides(
    self,
    camera: str | mujoco.MjvCamera | None = None,
    scene_option: mujoco.MjvOption | None = None,
  ) -> np.ndarray: ...


CamOpt = tuple[
  str | mujoco.MjvCamera | None,
  mujoco.MjvOption | None,
]


@runtime_checkable
class CameraSource(Protocol):
  """Supplies the ``(camera, scene_option)`` override for a recorded frame."""

  def frame_camera(self, step: int) -> CamOpt: ...


class FixedCfgCameraSource:
  """Use the offscreen renderer's default camera; supply a global contacts option.

  ``show_contacts`` builds a one-off ``MjvOption`` with the contact-point/force
  flags set, returned as the frame's ``scene_option`` so the offscreen recorder
  draws contacts (collision viz bakes into the model, so it needs no option).
  ``show_world_frame`` adds the world-origin coordinate frame (``frame = mjFRAME_WORLD``) to
  the SAME option, so captured videos match the native viewer (which always draws
  it). ``show_sensor_lines=False`` drops the rangefinder rays
  (``mjVIS_RANGEFINDER``, the distance-sensor lines), which
  MuJoCo draws by default. All flags at their defaults -> ``(None, None)``
  (base ``VideoRecorder`` behavior)."""

  def __init__(
    self,
    show_contacts: bool = False,
    show_world_frame: bool = False,
    show_sensor_lines: bool = True,
  ) -> None:
    self._opt: mujoco.MjvOption | None = None
    if show_contacts or show_world_frame or not show_sensor_lines:
      opt = mujoco.MjvOption()
      if show_contacts:
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
      if show_world_frame:
        opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value
      if not show_sensor_lines:
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
      self._opt = opt

  def frame_camera(self, step: int) -> CamOpt:
    del step
    return (None, self._opt)


class LiveViewerCameraSource:
  """Track a live native viewer's camera + scene option.

  ``native_viewer`` is a ``CodancingNative*Viewer`` whose ``.viewer`` is the live
  mujoco passive handle; streaming and frozen viewers share this source.
  """

  def __init__(self, native_viewer: Any = None) -> None:
    self._native_viewer = native_viewer
    self._cam = mujoco.MjvCamera()

  def attach_viewer(self, native_viewer: Any) -> None:
    self._native_viewer = native_viewer

  def frame_camera(self, step: int) -> CamOpt:
    del step
    nv = self._native_viewer
    if nv is None or getattr(nv, "viewer", None) is None:
      return (None, None)
    self._sync_camera_from(nv.viewer.cam)
    return (self._cam, nv.viewer.opt)

  def _sync_camera_from(self, viewer_cam: mujoco.MjvCamera) -> None:
    self._cam.type = viewer_cam.type
    self._cam.fixedcamid = viewer_cam.fixedcamid
    self._cam.trackbodyid = viewer_cam.trackbodyid
    self._cam.lookat[:] = viewer_cam.lookat
    self._cam.distance = viewer_cam.distance
    self._cam.azimuth = viewer_cam.azimuth
    self._cam.elevation = viewer_cam.elevation
    self._cam.orthographic = viewer_cam.orthographic


class OffscreenRendererWrapper(OffscreenRenderer):
  """Offscreen renderer adding an optional ``scene_option`` override to the base.

  Collision viz is baked into the model (no per-env render options); contacts are a
  single GLOBAL flag carried by the recorder's ``scene_option`` (a contacts-enabled
  ``MjvOption`` from the camera source, :class:`FixedCfgCameraSource`). With no
  override the base default option is used. Primary + neighbor envs render with the
  SAME option (there is no per-env collision/contact selection)."""

  def update(
    self,
    data: Any,
    debug_vis_callback: Callable[[Any], None] | None = None,
    camera: str | mujoco.MjvCamera | None = None,
    scene_option: mujoco.MjvOption | None = None,
  ) -> None:
    """Render the primary env + nearest neighbors with ONE option (``scene_option``
    when given, else the base default ``self._opt``). Mirrors the base nworld guard
    / env_idx clamp / per-world model-field sync / neighbor loop."""
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    nworld = int(data.nworld)
    if nworld <= 0:
      return

    env_idx = max(0, min(int(self._cfg.env_idx), nworld - 1))
    option = scene_option if scene_option is not None else self._opt

    self._sync_model_fields(env_idx)
    if self._model.nq > 0:
      self._data.qpos[:] = data.qpos[env_idx].cpu().numpy()
      self._data.qvel[:] = data.qvel[env_idx].cpu().numpy()
    if self._model.nmocap > 0:
      self._data.mocap_pos[:] = data.mocap_pos[env_idx].cpu().numpy()
      self._data.mocap_quat[:] = data.mocap_quat[env_idx].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)
    cam = camera if camera is not None else self._cam
    self._renderer.update_scene(self._data, camera=cam, scene_option=option)

    # Note: update_scene() resets the scene each frame, so no need to manually clear.
    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._model, env_idx=self._cfg.env_idx
      )
      debug_vis_callback(visualizer)

    # Add nearest neighboring environments as geoms for context (same option).
    for i in self._get_extra_env_ids(nworld, env_idx):
      self._sync_model_fields(i)
      if self._model.nq > 0:
        self._data.qpos[:] = data.qpos[i].cpu().numpy()
        self._data.qvel[:] = data.qvel[i].cpu().numpy()
      if self._model.nmocap > 0:
        self._data.mocap_pos[:] = data.mocap_pos[i].cpu().numpy()
        self._data.mocap_quat[:] = data.mocap_quat[i].cpu().numpy()
      mujoco.mj_forward(self._model, self._data)
      mujoco.mjv_addGeoms(
        self._model,
        self._data,
        option,
        self._pert,
        _SECONDARY_ENV_CATMASK,
        self._renderer.scene,
      )
    self._sync_model_fields(env_idx)


class CodancingNativeFrozenViewer(NativeMujocoViewer):
  """MuJoCo viewer that keeps simulation frozen and steps on demand.

  Sets the global contact-viz flag on its ``viewer.opt`` (C-key toggles) and drives
  an optional :class:`RolloutSink` (through ``observe_step`` in ``_execute_step``),
  so frozen single-steps record metrics like streaming play.
  """

  _ACTION_TOGGLE_CONTACTS = "FROZEN_TOGGLE_CONTACTS"
  _ACTION_TOGGLE_DEBUG_VIS = "FROZEN_TOGGLE_DEBUG_VIS"

  def __init__(
    self,
    env: EnvProtocol,
    policy: Optional[PolicyProtocol] = None,
    frame_rate: float = 60.0,
    key_callback: Optional[Callable[[int], None]] = None,
    plot_cfg=None,
    enable_perturbations: bool = True,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
    show_contacts: bool = True,
    step_observer: RolloutSink | None = None,
  ):
    super().__init__(
      env=env,
      policy=policy or self._build_zero_policy(env),
      frame_rate=frame_rate,
      key_callback=key_callback,
      plot_cfg=plot_cfg,
      enable_perturbations=enable_perturbations,
      verbosity=verbosity,
    )
    self.show_contacts = show_contacts
    self._observer = step_observer

  @staticmethod
  def _build_zero_policy(env: EnvProtocol) -> PolicyProtocol:
    def _zero_policy(obs: torch.Tensor) -> torch.Tensor:
      del obs
      num_envs = env.unwrapped.num_envs
      action_dim = env.unwrapped.action_manager.total_action_dim
      return torch.zeros((num_envs, action_dim), device=env.device)

    return _zero_policy

  def setup(self) -> None:
    super().setup()
    if self.viewer is not None:
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.show_contacts
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.show_contacts
    # Start frozen: reset + pause here (base `run()` calls `setup()` then loops),
    # so the viewer opens paused and only steps on demand (SPACE).
    self.reset_environment()

  def _execute_step(self) -> bool:
    # Frozen single-steps funnel through here too, so metrics are recorded on
    # each real step exactly like streaming play.
    ok = super()._execute_step()
    if ok:
      observe_step(self._observer, self.env)
    return ok

  def reset_environment(self) -> None:
    super().reset_environment()
    self.pause()

  def _safe_key_callback(self, key: int) -> None:
    delegated_to_super = False
    if key == KEY_SPACE:
      self.request_single_step()
    elif key == KEY_R:
      self.request_reset()
    elif key == KEY_D:
      self.request_action("CUSTOM", self._ACTION_TOGGLE_DEBUG_VIS)
    elif key == KEY_C:
      self.request_action("CUSTOM", self._ACTION_TOGGLE_CONTACTS)
    else:
      super()._safe_key_callback(key)
      delegated_to_super = True

    if self.user_key_callback and not delegated_to_super:
      try:
        self.user_key_callback(key)
      except Exception as error:
        self.log(f"[WARN] user key_callback raised: {error}", VerbosityLevel.INFO)

  def _handle_custom_action(self, action, payload) -> bool:
    if payload == self._ACTION_TOGGLE_CONTACTS:
      # Global contact toggle: flip the live viewer's vopt flags (all envs render
      # the same option; there is no per-env selection).
      self.show_contacts = not self.show_contacts
      if self.viewer is not None:
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.show_contacts
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.show_contacts
      self.log(
        f"[INFO] Contact visualization {'shown' if self.show_contacts else 'hidden'}",
        VerbosityLevel.INFO,
      )
      return True
    if payload == self._ACTION_TOGGLE_DEBUG_VIS:
      self._show_debug_vis = not self._show_debug_vis
      self.log(
        f"[INFO] Debug visualization {'shown' if self._show_debug_vis else 'hidden'}",
        VerbosityLevel.INFO,
      )
      return True
    return super()._handle_custom_action(action, payload)


class CodancingNativeStreamingViewer(NativeMujocoViewer):
  """Native viewer for continuous play, the sibling of the frozen viewer.

  Sets the global contact-viz flag on its ``viewer.opt`` and drives an optional
  :class:`RolloutSink` (through ``observe_step`` in ``_execute_step``), so the
  metric recorders record play.

  Extra key: ``H`` toggles a CONTROLLED motion-cursor pause -- the policy
  keeps running while the human / robot reference freeze. Distinct from the
  base viewer's SPACE, which pauses
  the whole sim. The viser viewer binds the same toggle to its ``H`` hotkey /
  button (both via :func:`toggle_codancing_cursor_pause`).

  Extra key: ``F8`` is the frozen spawn inspection. A full episode reset (the
  rsi draw, the anchored write, the perturbation), then the sim pauses and
  the force field is evaluated on that spawn exactly as training's reset step
  does, so the setpoint and force arrows show frame 0 of the new episode
  rather than the previous one. The Spawn row of the status panel and a
  terminal line describe the draw; SPACE then steps from it. ``R`` keeps its
  plain reset-and-run behavior.
  """

  def __init__(
    self,
    *args: Any,
    show_contacts: bool = False,
    step_observer: RolloutSink | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.show_contacts = show_contacts
    self._observer = step_observer
    # Live mouse-drag wrench, for the status overlay and the terminal echo.
    # Worth surfacing because the number is otherwise invisible: a drag is
    # the only disturbance in the viewer with no configured magnitude, so
    # "how hard did I just push it" cannot be answered from the config or
    # the session recording, and comparing a hand push on hardware against a
    # drag in sim needs both sides in newtons. Lives here, not in core: the
    # readout is exact without a core hook because mjv_applyPerturbForce
    # ASSIGNS into xfrc_applied within each call (engine_vis_interact.c:
    # mju_copy3 for the force, mju_cross for the torque), so the base
    # method's own call right after ours rewrites the same row with the same
    # wrench. A ROTATE-only perturbation writes only the torque half;
    # harmless, the row is cleared every frame.
    self._pert_force_n = 0.0
    self._pert_torque_nm = 0.0
    self._pert_body = ""
    self._pert_next_echo = 0.0
    self._spawn_row = "-"

  def sync_viewer_to_env(self) -> None:
    v = self.viewer
    if v is not None and self.mjm is not None and self.mjd is not None:
      pert = v.perturb
      if pert.active != 0 and pert.select > 0:
        mujoco.mjv_applyPerturbForce(self.mjm, self.mjd, pert)
        body_id = pert.select
        self._report_perturb(
          self.mjd.xfrc_applied[body_id, :3].copy(),
          self.mjd.xfrc_applied[body_id, 3:].copy(),
          body_id,
        )
      else:
        self._pert_force_n = 0.0
        self._pert_torque_nm = 0.0
        self._pert_body = ""
    super().sync_viewer_to_env()

  def _report_perturb(
    self, force: np.ndarray, torque: np.ndarray, body_id: int
  ) -> None:
    """Record this frame's drag wrench and echo it, at most 4 times a second.

    Throttled because the drag lasts for hundreds of frames and an unthrottled
    echo buries whatever else the run is printing. The overlay updates every
    frame regardless; only the terminal line is rate-limited.
    """
    self._pert_force_n = float(np.linalg.norm(force))
    self._pert_torque_nm = float(np.linalg.norm(torque))
    assert self.mjm is not None
    self._pert_body = (
      mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
    )
    now = time.perf_counter()
    if now >= self._pert_next_echo:
      self._pert_next_echo = now + 0.25
      print(
        f"[drag] {self._pert_body}: {self._pert_force_n:7.1f} N  "
        f"{self._pert_torque_nm:6.2f} N.m",
        flush=True,
      )

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    # A copy of NativeMujocoViewer._set_status_overlay
    # (src/mjlab/viewer/native/viewer.py) plus the Drag row: `set_texts` takes
    # one text blob, so a row added to the base panel must be mirrored here.
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    # The drag row is always present, not only while dragging: a row that
    # appears and vanishes shifts every line under it, and the reader is
    # usually watching this panel to compare one drag against the next.
    drag = (
      f"{self._pert_force_n:.1f} N / {self._pert_torque_nm:.2f} N.m ({self._pert_body})"
      if self._pert_body
      else "-"
    )
    text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\nDrag\nSpawn"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)\n"
      f"{drag}\n"
      f"{self._spawn_row}"
    )
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      text_1,
      text_2,
    )
    viewer.set_texts(overlay)

  def _safe_key_callback(self, key: int) -> None:
    if key == KEY_H:
      self.request_action("CUSTOM", _ACTION_TOGGLE_CURSOR_PAUSE)
      return
    if key == KEY_F8:
      self.request_action("CUSTOM", _ACTION_INSPECT_SPAWN)
      return
    super()._safe_key_callback(key)

  def _handle_custom_action(self, action: Any, payload: object | None) -> bool:
    if payload == _ACTION_TOGGLE_CURSOR_PAUSE:
      _log_cursor_pause(self, toggle_codancing_cursor_pause(self.env))
      return True
    if payload == _ACTION_INSPECT_SPAWN:
      self._inspect_spawn()
      return True
    return super()._handle_custom_action(action, payload)

  def _inspect_spawn(self) -> None:
    """Reset, pause, evaluate the force field on the spawn, describe it.

    Runs on the main loop (queued by the F8 key) so the env mutation never
    races stepping. The readout must not take the viewer down with it, so a
    failure lands in the Spawn row instead of raising.
    """
    from mjlab.tasks.codancing.rsi_inspect import (
      evaluate_forcefield_at_spawn,
      format_spawn,
      spawn_summary,
    )

    self.reset_environment()
    self.pause()
    ran: list[str] = []
    try:
      ran = evaluate_forcefield_at_spawn(self.env)
      self._spawn_row = format_spawn(spawn_summary(self.env, self.env_idx))
    except Exception as error:  # noqa: BLE001 -- a readout, not the sim
      self._spawn_row = f"unavailable ({error})"
    detail = f"  (force field at frame 0: {', '.join(ran)})" if ran else ""
    self.log(f"[spawn] {self._spawn_row}{detail}", VerbosityLevel.INFO)

  def setup(self) -> None:
    super().setup()
    if self.viewer is not None:
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.show_contacts
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.show_contacts

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if ok:
      observe_step(self._observer, self.env)
    return ok


class CodancingViserViewer(ViserPlayViewer):
  """Viser play viewer that drives the rollout sinks, mirroring the native
  wrappers. Core ``ViserPlayViewer`` inherits the base loop but never calls
  ``observe_step``; this task-layer subclass adds it so a viser play session
  records the metric sinks like the native and headless paths. It also exposes
  the controlled motion-cursor pause as an ``H`` hotkey + button (the viser twin
  of the native viewer's ``H`` key), so a browser-hosted play session can freeze
  the dance on demand. No core edits: every hook lives at the task layer."""

  def __init__(
    self,
    *args: Any,
    step_observer: RolloutSink | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self._observer = step_observer

  def setup(self) -> None:
    super().setup()
    self._add_cursor_pause_control()

  def _add_cursor_pause_control(self) -> None:
    """Expose the controlled motion-cursor pause in viser: an ``H`` hotkey
    (command palette) plus a clickable button, both queuing the same toggle the
    native ``H`` key uses. Queued (not run inline) so the env mutation lands on
    the main loop, never the viser callback thread. The hotkey API is
    experimental; degrade to button-only if this viser lacks it."""

    def _request_toggle() -> None:
      self.request_action("CUSTOM", _ACTION_TOGGLE_CURSOR_PAUSE)

    button = self._server.gui.add_button(
      "Pause dance (H)",
      icon=viser.Icon.PLAYER_PAUSE,
      hint="Freeze the reference "
      "cursor; the policy keeps running. Same as the native H key.",
    )
    button.on_click(lambda _: _request_toggle())
    try:
      command = self._server.gui.add_command(
        "Pause / resume dance", hotkey="H", icon=viser.Icon.PLAYER_PAUSE
      )
      command.on_trigger(lambda _: _request_toggle())
    except Exception:  # noqa: BLE001 -- experimental hotkey API; button still works
      self.log(
        "[INFO] viser hotkey API unavailable; use the 'Pause dance' button.",
        VerbosityLevel.INFO,
      )

  def _handle_custom_action(self, action: Any, payload: object | None) -> bool:
    if payload == _ACTION_TOGGLE_CURSOR_PAUSE:
      _log_cursor_pause(self, toggle_codancing_cursor_pause(self.env))
      return True
    return super()._handle_custom_action(action, payload)

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if ok:
      observe_step(self._observer, self.env)
    return ok


class CameraVideoRecorder(VideoRecorder):
  """``VideoRecorder`` whose frame camera + scene option come from a
  :class:`CameraSource` (one recording path for fixed / live cameras).

  Subclasses the core recorder so triggers, naming, length, and saving are
  inherited unchanged; only ``_record_frame`` differs: it renders env[0] via the
  env's ``render_with_overrides`` with the source's per-step ``(camera,
  scene_option)``. A :class:`FixedCfgCameraSource` yields ``(None, None)`` --
  identical to the base recorder's ``render()`` -- while a
  :class:`LiveViewerCameraSource` tracks a live native viewer. The wrapped env
  must satisfy :class:`SupportsOverrideRender`.
  """

  def __init__(
    self,
    *args: Any,
    camera_source: CameraSource | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self._camera_source: CameraSource = camera_source or FixedCfgCameraSource()
    # Streaming encode: frames go straight to an incremental ffmpeg writer instead
    # of a whole-clip RAM buffer, so memory is O(1) in clip length (a long / high-res
    # clip would otherwise buffer GBs). `current_video_frames` holds only per-frame
    # placeholders (for the base class's length/stop check); the pixels are streamed.
    self._writer: media.VideoWriter | None = None
    self._stream_path: Path | None = None

  def _start_recording(self) -> None:
    super()._start_recording()
    # Frames stream straight to the final file, O(1) memory. The writer opens
    # lazily on the first frame (its shape).
    assert self.current_video_path is not None
    self._stream_path = self.current_video_path
    self._writer = None
    # Capture the current state as the FIRST frame, so the clip starts at
    # observation_0 / step 0 -- the core recorder otherwise only records inside
    # `step()`, after the first action, skipping the initial pose. This runs
    # before `env.step()` in the base step loop, so the env is still at the state
    # recording began from. That frame is only observation_0 if the env JUST
    # reset; validate it (episode_length_buf[0] == 0) and fail loudly otherwise,
    # rather than label a mid-episode frame as step 0. Every CameraVideoRecorder
    # records once at step 0 on a fresh env, so this is a guard, not a branch.
    reset_step = self._recorded_env_step()
    if reset_step not in (0, None):
      raise RuntimeError(
        "CameraVideoRecorder began recording on a non-reset env "
        f"(episode_length_buf[0]={reset_step}); the first frame is captured as "
        "observation_0 / step 0, so recording must start right after a reset. "
        "A mid-episode recorder must use the plain VideoRecorder."
      )
    if reset_step == 0:
      self._record_frame()

  def _recorded_env_step(self) -> int | None:
    """The recorded env (env[0]) step from ``episode_length_buf``, or ``None`` when
    the env does not expose it (then the reset frame is skipped, not forced)."""
    buf = getattr(self._wrapped_env.unwrapped, "episode_length_buf", None)
    if buf is None:
      return None
    return int(buf.reshape(-1)[0].item())

  def _finish_recording(self) -> None:
    # Reset-first / idempotent: snapshot the resources, then clear ALL recorder state
    # BEFORE any fallible encode. So if `close()` raises, the base `close()`'s
    # re-invocation sees `is_recording=False` and is a clean no-op --
    # instead of re-entering and running the base encode over the `None` placeholder
    # buffer (a TypeError that would mask the real error and skip the env close).
    writer = self._writer
    final_path = self.current_video_path
    self.is_recording = False
    self.current_video_frames = []
    self.current_video_path = None
    self._writer = None
    self._stream_path = None
    self.video_count += 1
    self.trigger_type = None
    if writer is None:
      return  # nothing streamed (render disabled / no frames).
    writer.close()
    if not self.disable_logger:
      print(f"[INFO] Saved video to {final_path}")

  def close(self) -> None:
    """Finalize any open recording, then ALWAYS close the wrapped env -- so an encode
    failure in ``_finish_recording`` can never leak the env's renderer / GPU context
    (the base ``close`` skips the env close if finalize raises)."""
    try:
      if self.is_recording:
        self._finish_recording()
    finally:
      self._wrapped_env.close()

  def _open_writer(self, path: Path, shape: tuple[int, int]) -> media.VideoWriter:
    """Open an incremental ffmpeg writer -- manual ``__enter__`` since it lives across
    many frames, not a ``with`` block."""
    fps = float(self._wrapped_env.metadata.get("render_fps", 30))
    writer = media.VideoWriter(str(path), shape=shape, fps=fps)
    writer.__enter__()
    return writer

  def _record_frame(self) -> None:
    if self._wrapped_env.render_mode != "rgb_array":
      return
    env = cast(SupportsOverrideRender, self._wrapped_env.unwrapped)
    camera, scene_option = self._camera_source.frame_camera(self.step_count)
    frame = env.render_with_overrides(camera=camera, scene_option=scene_option)
    if frame is None:
      return
    rgb_frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    self._write_stream_frame(np.asarray(rgb_frame))

  def _write_stream_frame(self, frame: np.ndarray) -> None:
    """Stream one frame to the incremental writer (opened lazily on the first frame's
    shape). Records a ``None`` placeholder in ``current_video_frames`` so the base
    class's length/stop check counts it -- the pixels themselves are not buffered."""
    frame = _to_uint8(frame)
    if self._writer is None:
      assert self._stream_path is not None
      self._writer = self._open_writer(self._stream_path, frame.shape[:2])
    self._writer.add_image(np.ascontiguousarray(frame))
    self.current_video_frames.append(None)
