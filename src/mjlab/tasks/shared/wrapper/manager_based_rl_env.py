from dataclasses import dataclass

import mujoco

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.shared.wrapper.viewer import OffscreenRendererWrapper


class ManagerBasedRlEnvWrapper(ManagerBasedRlEnv):
  """Manager-based RL env whose offscreen renderer can take camera overrides."""

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    # Build everything with base behavior first.
    super().__init__(cfg=cfg, device=device, render_mode=render_mode, **kwargs)

    # Then switch to wrapped offscreen renderer for rgb_array mode.
    self.render_mode = render_mode
    if self._offline_renderer is not None:
      self._offline_renderer.close()
      self._offline_renderer = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRendererWrapper(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt

  def render_with_overrides(
    self,
    camera: str | mujoco.MjvCamera | None = None,
    scene_option: mujoco.MjvOption | None = None,
  ):
    """Render via the offscreen renderer with optional camera/option overrides.

    Used by CameraVideoRecorder (via a CameraSource) to render the scene from a
    live-viewer camera without disturbing the env's primary `render()` path.
    """
    if self._offline_renderer is None:
      raise ValueError("Offline renderer not initialized")
    debug_callback = (
      self.update_visualizers if hasattr(self, "update_visualizers") else None
    )
    self._offline_renderer.update(
      self.sim.data,
      debug_vis_callback=debug_callback,
      camera=camera,
      scene_option=scene_option,
    )
    return self._offline_renderer.render()


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfgWrapper(ManagerBasedRlEnvCfg):
  """The codancing env cfg: the base cfg under the name every config and every
  checkpoint's frozen tree targets."""

  # `viewer` is the base `ViewerConfig` (collision viz is model-baked; contacts
  # are a session-tier global flag).
