"""Reusable single-frame G1 rendering.

Poses the real G1 from an adapted motion NPZ and renders it: color chosen bodies, add
overlay geometry to the scene (spheres, spring capsules, force arrows), whiten the
background and crop to the robot, then project world points to pixel coordinates, so
that arrows / labels drawn afterwards in Figma or matplotlib land exactly on the
render.

Recipes (the section comments below refer to them):
  A = render + whiten + crop, B = in-scene overlay geometry, C = camera-to-pixel
  projection.

Frame conventions (clips from this pipeline):
  joint_pos (T, 29): the 29 actuated joints, in G1_JOINT_NAMES order.
  body_pos_w / body_quat_w (T, 30, .): model body order with the world body dropped,
    so clip body 0 is the pelvis (free-joint root). Quaternions are wxyz.

Renders offscreen with EGL unless MUJOCO_GL names another backend
(CUDA_VISIBLE_DEVICES picks the GPU).
"""

from __future__ import annotations

import os
import sys
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# Clip joint columns follow G1_JOINT_NAMES (the single source of truth, in the
# augmenter). Only that constant is imported; the small qpos-index helper is kept
# here so this figure tool does not depend on a private symbol.
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
  sys.path.insert(0, _SCRIPTS)
from augment_multilink import G1_JOINT_NAMES  # noqa: E402
from palette import PAL  # noqa: E402

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML  # noqa: E402

# Colors and primitive sizes shared by all the figures, from the png medium of the
# repo palette (rgba in 0..1; sizes in meters).
SLOT_BLUE = PAL.PNG.rgba("slot_a")
SLOT_PINK = PAL.PNG.rgba("slot_b")
RED = PAL.PNG.rgba("crimson")
SLATE = PAL.PNG.rgba("slate")
# Default wrist colors (left wrist = slot 0 blue, right wrist = slot 1 pink).
WRIST_COLORS = {"left_wrist_yaw_link": SLOT_BLUE, "right_wrist_yaw_link": SLOT_PINK}


def qpos_index(model, joint_name: str) -> int:
  """Index of a joint in qpos, by name (model.jnt_qposadr[mj_name2id(...JOINT...)])."""
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  return int(model.jnt_qposadr[jid])


class G1FrameRenderer:
  """Render single frames of an adapted G1 clip, with coloring, overlays, projection.

  Typical use::

      clip = ("data/compliant/rc/20260830_comz_small_002/"
              "waltz_20260224_001_multilink_forcefield_seed0.npz")
      r = G1FrameRenderer(clip, body_colors=WRIST_COLORS)
      img, meta = r.render_frame(100, anchor_bodies=["left_wrist_yaw_link",
                                                     "right_wrist_yaw_link"])
      # img is the cropped RGB array on white; meta["anchors"] gives the wrist
      # pixel positions **in crop coordinates**, ready for placing an arrow in
      # Figma.
  """

  def __init__(
    self,
    clip_npz: str,
    *,
    width: int = 900,
    height: int = 1100,
    body_colors: dict[str, tuple] | None = None,
    root_body_index: int = 0,
    ambient: float = 0.42,
    diffuse: float = 0.72,
    xml: str | None = None,
    background: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    model: mujoco.MjModel | None = None,
  ) -> None:
    self.width, self.height = width, height
    self._root = root_body_index
    # An injected model brings its own background (e.g. the ground + skybox of
    # build_g1_scene_model), not a solid color, so _solid_bg is None -> crop_solid
    # refuses and whiten_crop takes the difference path.
    self._solid_bg = None if model is not None else background
    if model is not None:
      # Use the caller's compiled model as is (its qpos layout must match the bare
      # G1 exactly: joints looked up by name, root at [0:7]).
      self.model = model
    elif background is not None:
      # Inject a solid-color skybox (via MjSpec, the XML file is untouched): the
      # background renders directly in the target color, so a single render gives a
      # clean background, without even the background holes enclosed by the robot.
      # The default pure white is the canvas color of the figure assets, and
      # antialiased edges blend toward white, melting right into it. Assumes the
      # model has no skybox of its own (g1.xml has ntex=0); otherwise pass
      # background=None.
      spec = mujoco.MjSpec.from_file(str(xml or G1_XML))
      tex = spec.add_texture()
      tex.name = "clip_viz_bg"
      tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
      tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
      tex.rgb1[:] = background
      tex.rgb2[:] = background
      tex.width = 32
      tex.height = 32
      self.model = spec.compile()
    else:
      self.model = mujoco.MjModel.from_xml_path(str(xml or G1_XML))
    # The offscreen framebuffer must be at least the render size (a common crash on
    # a first run).
    buf = max(width, height, 1400)
    self.model.vis.global_.offwidth = buf
    self.model.vis.global_.offheight = buf
    self.model.vis.headlight.ambient[:] = ambient
    self.model.vis.headlight.diffuse[:] = diffuse
    for name, rgba in (body_colors or {}).items():
      self._color_body(name, rgba)
    self.data = mujoco.MjData(self.model)
    self.renderer = mujoco.Renderer(self.model, height=height, width=width)
    self.camera = mujoco.MjvCamera()
    clip = np.load(clip_npz, allow_pickle=False)
    self._jp = clip["joint_pos"]
    self._bp = clip["body_pos_w"]
    self._bq = clip["body_quat_w"]

  # -- setup ----------------------------------------------------------------
  def _color_body(self, body_name: str, rgba) -> None:
    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
      raise ValueError(f"unknown body {body_name!r}")
    for g in range(self.model.ngeom):
      if self.model.geom_bodyid[g] == bid:
        self.model.geom_rgba[g] = rgba
        self.model.geom_matid[g] = -1  # drop the material so the rgba shows

  @property
  def num_frames(self) -> int:
    return int(self._jp.shape[0])

  # -- pose + camera --------------------------------------------------------
  def set_pose(self, frame: int) -> None:
    """Write a clip frame into qpos (root from body 0, joints by name), then forward."""
    self.data.qpos[:3] = self._bp[frame, self._root]
    self.data.qpos[3:7] = self._bq[frame, self._root]  # wxyz
    for j, nm in enumerate(G1_JOINT_NAMES):
      self.data.qpos[qpos_index(self.model, nm)] = self._jp[frame, j]
    mujoco.mj_forward(self.model, self.data)

  def look(
    self,
    *,
    distance: float = 2.7,
    azimuth: float = 155.0,
    elevation: float = -8.0,
    lookat: np.ndarray | None = None,
    dz: float = 0.10,
  ) -> None:
    """Aim the camera (framing the pelvis by default) and update the render scene."""
    self.camera.lookat[:] = (
      np.asarray(lookat, float)
      if lookat is not None
      else self.data.qpos[:3] + np.array([0.0, 0.0, dz])
    )
    self.camera.distance = distance
    self.camera.azimuth = azimuth
    self.camera.elevation = elevation
    self.renderer.update_scene(self.data, self.camera)

  def body_xpos(self, body_name: str) -> np.ndarray:
    """World position of a body in the current pose (for anchors / overlays)."""
    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(self.data.xpos[bid])

  # -- in-scene overlay geometry (recipe B) ---------------------------------
  def _next_geom(self):
    scn = self.renderer.scene
    if scn.ngeom >= scn.maxgeom:
      raise RuntimeError(f"scene geom budget exhausted ({scn.maxgeom})")
    return scn.geoms[scn.ngeom]

  def add_sphere(
    self, pos, radius: float = PAL.PNG.size("probe_ball"), rgba=RED
  ) -> None:
    g = self._next_geom()
    mujoco.mjv_initGeom(
      g,
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.array([radius, 0, 0], float),
      np.asarray(pos, float),
      np.eye(3).flatten(),
      np.asarray(rgba, np.float32),
    )
    self.renderer.scene.ngeom += 1

  def add_connector(
    self,
    p0,
    p1,
    *,
    radius: float = PAL.PNG.size("probe_connector_radius"),
    rgba=SLATE,
    arrow: bool = False,
  ) -> None:
    """A world-frame capsule (arrow=False) or arrow (arrow=True) from p0 to p1."""
    gtype = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    g = self._next_geom()
    mujoco.mjv_initGeom(
      g,
      gtype,
      np.zeros(3),
      np.zeros(3),
      np.eye(3).flatten(),
      np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(g, gtype, radius, np.asarray(p0, float), np.asarray(p1, float))
    self.renderer.scene.ngeom += 1

  def add_spring_device(
    self,
    live,
    setpoint,
    force,
    *,
    sphere_radius: float = PAL.PNG.size("spring_device_ball"),
    spring_radius: float = PAL.PNG.size("spring_connector_radius"),
    arrow_len: float = PAL.PNG.size("dir_arrow_len"),
    arrow_clear: float = PAL.PNG.size("dir_arrow_clear"),
    arrow_radius: float = PAL.PNG.size("dir_arrow_width"),
  ) -> None:
    """The spring device: live ball (slate), setpoint ball (red), spring, force arrow.

    The force arrow is drawn **pushing**: its tail behind the wrist, its tip stopping
    just short of it (tail = live - F_hat*(len+clear), tip = live - F_hat*clear).
    Under the spring law F is collinear with (setpoint - live), so an arrow drawn
    from live along F would lie on the same ray as the spring capsule; the pushing
    form keeps the two apart.
    """
    live = np.asarray(live, float)
    setpoint = np.asarray(setpoint, float)
    F = np.asarray(force, float)
    self.add_sphere(setpoint, radius=sphere_radius, rgba=RED)
    self.add_sphere(live, radius=sphere_radius, rgba=SLATE)
    self.add_connector(live, setpoint, radius=spring_radius, rgba=SLATE)
    fmag = float(np.linalg.norm(F))
    if fmag > 1e-9:
      fu = F / fmag
      tail = live - fu * (arrow_len + arrow_clear)
      tip = live - fu * arrow_clear
      self.add_connector(tail, tip, radius=arrow_radius, rgba=RED, arrow=True)

  # -- render + post-processing (recipe A) ----------------------------------
  def render(self) -> np.ndarray:
    return self.renderer.render()

  def whiten_crop(
    self,
    img: np.ndarray,
    *,
    pad: int = 24,
    tol: int = 8,
    crop: bool = True,
    bg_img: np.ndarray | None = None,
  ):
    """Whiten the background, optionally cropping to the foreground bounding box.

    Returns (rgb, (x0, y0), foreground_mask_full). With ``bg_img`` (a robot-free
    background render from the same camera, see render_background) the foreground
    comes from a per-pixel difference: exact, and it also whitens background holes
    **enclosed** by the robot (such as the gap between the legs, which border
    connectivity cannot reach). Otherwise it falls back to the corner color + a
    border-connected fill (scipy.ndimage.label). ``crop=False`` whitens without
    cropping (equal-size tiles, e.g. for comparing camera candidates).
    """
    if bg_img is not None:
      diff = np.abs(img.astype(np.int16) - bg_img.astype(np.int16)).sum(axis=-1)
      fg = diff > tol * 3
      bg = ~fg
    else:
      from scipy import ndimage

      base = img.astype(np.int16)
      corner = base[2, 2]
      near = np.abs(base - corner[None, None, :]).sum(axis=-1) <= tol * 3
      # scipy-stubs picks the int-returning overload of label; at runtime this is
      # always (array, count).
      lbl, _ = ndimage.label(near)  # pyright: ignore[reportGeneralTypeIssues]
      border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
      border.discard(0)
      bg = np.isin(lbl, list(border)) if border else np.zeros_like(near)
      fg = ~bg
    out = img.copy()
    out[bg] = 255
    ys, xs = np.where(fg)
    if not crop or ys.size == 0:
      return out, (0, 0), fg
    h, w = img.shape[:2]
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w)
    return out[y0:y1, x0:x1], (x0, y0), fg

  def crop_solid(self, img: np.ndarray, *, pad: int = 24):
    """Crop a solid-skybox render: foreground = pixels unlike the background color.

    No whitening is needed. Returns (rgb, (x0, y0), foreground_mask_full), the same
    shape as whiten_crop. Highlight pixels inside the robot that happen to equal the
    background color only affect the mask, not the bounding box, so they are
    harmless for white-background assets.
    """
    assert self._solid_bg is not None, "crop_solid needs the solid background= mode"
    bg = np.round(np.asarray(self._solid_bg) * 255).astype(np.uint8)
    fg = (img != bg[None, None, :]).any(axis=-1)
    ys, xs = np.where(fg)
    if ys.size == 0:
      return img, (0, 0), fg
    h, w = img.shape[:2]
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w)
    return img[y0:y1, x0:x1], (x0, y0), fg

  def render_background(self) -> np.ndarray:
    """Robot-free background render from the same camera.

    The robot is moved far away and the scene redrawn, which also clears the in-scene
    overlay geometry. A per-pixel difference against the foreground frame gives an
    exact foreground mask (whiten_crop's ``bg_img``). The scene needs a fresh update
    afterwards (render_frame re-poses every frame anyway).
    """
    saved = self.data.qpos.copy()
    self.data.qpos[2] -= 500.0
    mujoco.mj_forward(self.model, self.data)
    self.renderer.update_scene(self.data, self.camera)
    bg = self.renderer.render().copy()
    self.data.qpos[:] = saved
    mujoco.mj_forward(self.model, self.data)
    return bg

  def project(self, points) -> list[tuple[float, float]]:
    """Project world points to full-frame pixels with the current camera (recipe C)."""
    gl = self._gl_camera()
    pos, fwd, up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
    right = np.cross(fwd, up)
    half_h = (gl.frustum_top - gl.frustum_bottom) / 2.0
    half_w = half_h * (self.width / self.height)
    cx = gl.frustum_center
    out = []
    for p in points:
      v = np.asarray(p, float) - pos
      xc, yc, zc = v @ right, v @ up, v @ fwd
      xn = ((xc * gl.frustum_near / zc) - cx) / half_w
      yn = (yc * gl.frustum_near / zc) / half_h
      out.append(((xn + 1) / 2 * self.width, (1 - (yn + 1) / 2) * self.height))
    return out

  def _gl_camera(self):
    """The camera this render actually uses.

    ``scene.camera`` holds two cameras, one per eye; even in mono
    (``scene.stereo == 0``) their ``pos`` differ by an interpupillary baseline, and
    mjr_render uses their average. Taking ``camera[0]`` alone shifts the eye sideways
    by half that baseline, and projected pixel x coordinates all come out offset
    (measured ~16px at 760px width).
    """
    return mujoco.mjv_averageCamera(
      self.renderer.scene.camera[0], self.renderer.scene.camera[1]
    )

  # -- one-call convenience entry point -------------------------------------
  def render_frame(
    self,
    frame: int,
    *,
    look_kw: dict | None = None,
    overlays: Callable[["G1FrameRenderer"], None] | None = None,
    anchor_bodies: list[str] | None = None,
    anchors: dict[str, np.ndarray] | None = None,
    whiten: bool = True,
    crop: bool = True,
    pad: int = 24,
  ) -> tuple[np.ndarray, dict]:
    """Pose -> aim -> overlay -> render -> whiten/crop -> project anchors.

    ``overlays(self)`` may call add_sphere / add_connector to draw in-scene geometry.
    ``anchor_bodies`` lists bodies to project; their world positions are read
    **after posing**, so they are correct for this frame. ``anchors`` maps labels to
    raw world points the caller has already computed. Both end up in
    meta["anchors"], as pixel positions **in the cropped image** (offset already
    subtracted).
    """
    self.set_pose(frame)
    self.look(**(look_kw or {}))
    if overlays is not None:
      overlays(self)
    world = dict(anchors or {})
    for name in anchor_bodies or []:
      world[name] = self.body_xpos(name)
    img = self.render().copy()
    offset = (0, 0)
    if whiten:
      if self._solid_bg is not None:
        # Solid-skybox mode: the background already has the target color, so only
        # crop to the bounding box (single render).
        if crop:
          img, offset, _ = self.crop_solid(img, pad=pad)
      else:
        # Raw-background mode: whiten by differencing against a robot-free
        # background (exact foreground, including enclosed background holes).
        img, offset, _ = self.whiten_crop(
          img, pad=pad, crop=crop, bg_img=self.render_background()
        )
    meta: dict = {
      "frame": frame,
      "size": [int(img.shape[1]), int(img.shape[0])],
      "offset": list(offset),
    }
    if world:
      names = list(world)
      px = self.project([world[n] for n in names])
      meta["anchors"] = {
        n: [round(x - offset[0], 1), round(y - offset[1], 1)]
        for n, (x, y) in zip(names, px, strict=True)
      }
    return img, meta


def _demo() -> None:
  """Render one real frame: both wrists colored + a setpoint ball + projected wrist
  anchors.

  Exercises recipes A, B and C on the seed clip (skips cleanly if the clip is
  missing). Run: uv run python scripts/clip_viz/g1_render.py
  """
  import imageio.v2 as iio

  clip = os.path.join(
    _SCRIPTS,
    os.pardir,
    "data/compliant/rc/20260830_comz_small_002",
    "waltz_20260224_001_multilink_forcefield_seed0.npz",
  )
  clip = os.path.abspath(clip)
  if not os.path.exists(clip):
    print(f"skipping demo: {clip} not found")
    return
  r = G1FrameRenderer(clip, body_colors=WRIST_COLORS)

  def overlays(rr: G1FrameRenderer) -> None:
    # Overlays run after posing, so body_xpos is correct for this frame.
    wl = rr.body_xpos("left_wrist_yaw_link")
    rr.add_sphere(
      wl + np.array([0.0, 0.0, 0.05]), radius=PAL.PNG.size("probe_ball"), rgba=RED
    )
    rr.add_connector(
      wl,
      wl + np.array([0.0, 0.0, 0.05]),
      radius=PAL.PNG.size("probe_connector_radius"),
      rgba=RED,
      arrow=True,
    )

  img, meta = r.render_frame(
    100,
    overlays=overlays,
    anchor_bodies=["left_wrist_yaw_link", "right_wrist_yaw_link"],
  )
  import tempfile

  out = os.path.join(tempfile.gettempdir(), "clip_viz_demo_frame100.png")
  iio.imwrite(out, img)
  w, h = meta["size"]
  assert img.ndim == 3 and w > 50 and h > 50, "render is empty / too small"
  for name, (x, y) in meta["anchors"].items():
    assert 0 <= x <= w and 0 <= y <= h, (
      f"anchor {name} outside the crop: {(x, y)} vs {(w, h)}"
    )
  print(f"OK: rendered frame 100 -> {out} ({w}x{h}); anchors {meta['anchors']}")


if __name__ == "__main__":
  _demo()
