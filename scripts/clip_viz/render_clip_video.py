"""Render a whole adapted clip to mp4, labeling on every frame which link is being
forced.

The video uses the ``G1FrameRenderer`` recipes (pose / camera / in-scene overlays)
plus a banner at the top that prints each frame's forced links and |F|. Use it to
check at a glance whether an augmentation looks right: is the force applied to the
intended wrist, does it ramp up and down, does the pose hold together.

The forced link can be marked in **two styles**, chosen with ``--decor``:

* ``spring`` (default) = the spring device
  (``G1FrameRenderer.add_spring_device``): live ball + setpoint ball + spring
  capsule + pushing arrow. The extra setpoint end shows the gap between "where the
  environment pulls the wrist" and where the wrist actually is, i.e. the compliance
  itself. Best for explaining the mechanism.
* ``arrow`` = one big orange arrow from the wrist along the force, with length
  proportional to |F|. The most direct view of "who is pushed, which way, how
  hard", good for a quick pass over whether things are right.

The labels come from the **recorded** contact NPZ (the final forces after clamping /
backoff at generation time), not from re-running the scheduler with the same seed
and guessing, so the numbers on screen are the forces actually applied, and there
is no need to re-align seed / duration / mode.

The full pipeline (augment, convert, render; the duration is always the whole reference, and when --motion is an original reference
NPZ its fps key is picked up automatically; do not feed the augmenter's output back
in as input):

    uv run python scripts/augment_multilink.py motion \\
        --motion data/reference_motion_edits_g1_g1_waltz/20260224_001_robot.npz \\
        --out adapted.csv --contact-out contact.npz
    uv run python scripts/qpos_csv_to_motion_npz.py \\
        --input-file adapted.csv --output-file clip.npz --input-fps 50
    MUJOCO_GL=egl uv run python scripts/clip_viz/render_clip_video.py \\
        --clip clip.npz --contact contact.npz --out clip.mp4

Without ``--contact`` it is a plain motion video (no labels, no spring device).
Single-contact (single-link) contact NPZs work too: ``ClipEvidence`` presents them
as a K=1 view, so single-link augmented clips use the same command.

By default the camera follows the root (the default behavior of ``look()``), since
a traveling dance would otherwise walk out of frame; ``--no-track`` fixes it at the
root's mean position over the clip. The bridge (qpos_csv_to_motion_npz.py) outputs
one frame fewer than the contact log (velocities by finite difference), so the two
are front-aligned and the smaller frame count is used.

Self-check (renders a few frames of the seed clip and asserts that the overlays /
banner really follow the contact data):
    MUJOCO_GL=egl uv run python scripts/clip_viz/render_clip_video.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must come before importing mujoco

import mediapy as media
import numpy as np
from clip_evidence import ClipEvidence
from g1_render import SLOT_BLUE, SLOT_PINK, WRIST_COLORS, G1FrameRenderer
from palette import PAL

# Slot colors share their source with WRIST_COLORS (slot 0 left wrist blue, slot 1
# right wrist pink), so the names in the banner match the colored wrists in the
# frame. Cycled when there are more slots than colors.
_SLOT_RGBA = [SLOT_BLUE, SLOT_PINK]
_BANNER_H = 58  # banner height (pixels)
# Banner font sizes follow the banner height, not the palette (see the note at
# FONT_FAMILY in viz_palette.py).
_BANNER_FONT_PX_BIG = 21
_BANNER_FONT_PX_SMALL = 16
_FONTS: dict = {}
# Force arrow (--decor arrow): orange, length = |F| x ARROW_SCALE meters, shaft
# radius ARROW_RADIUS (both from the palette), sized for the 4 to 23 N the CoM budget
# holds the forces to: a 0.014 m/N scale would draw them only 0.06 to 0.32 m long,
# and a 0.028 m shaft would turn them into an orange blob.
# Note: **whether an arrow is foreshortened depends on the force direction and the
# camera azimuth**. Forces mostly push along world x, and the default azimuth=155
# looks almost straight along it, so even a long arrow shrinks to a dot; only a side
# view (--azimuth around 100) shows direction and length.
ARROW_SCALE = PAL.PNG.size("force_arrow_scale")
ARROW_RADIUS = PAL.PNG.size("wrench_arrow_baked")
ARROW_RGBA = PAL.PNG.rgba("orange_bright")
# Default output location of the clip tools (this module's default_out, also used by
# plot_force_curve.py and plot_stiffness_curve.py): out/ next to this script,
# ignored by clip_viz/.gitignore. --out overrides it per call.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def short_link(name: str) -> str:
  """left_wrist_yaw_link -> L-wrist (fits on one banner line)."""
  s = name.removesuffix("_link").replace("left_", "L-").replace("right_", "R-")
  for axis in ("_yaw", "_roll", "_pitch"):
    s = s.replace(axis, "")
  return s


def _fonts():
  """The banner's two font faces, from the palette. Loaded once.

  Same typeface as the force and stiffness plots. Resolving through
  the palette instead of hardcoding absolute paths under
  ``/usr/share/fonts/truetype/dejavu`` keeps the banner in step when the font
  family changes; that path also exists only on some Linux installs.
  """
  if not _FONTS:
    from PIL import ImageFont

    try:
      _FONTS["big"] = ImageFont.truetype(PAL.font_file("bold"), _BANNER_FONT_PX_BIG)
      _FONTS["small"] = ImageFont.truetype(PAL.font_file(), _BANNER_FONT_PX_SMALL)
    except OSError:
      _FONTS["big"] = _FONTS["small"] = ImageFont.load_default()
  return _FONTS["big"], _FONTS["small"]


def _setup_lines(run_meta_raw) -> list[str]:
  """run_meta.sampler (the augmenter's record of its settings) -> banner setup
  lines. A contact NPZ without it (the hosted clips carry no run_meta) gives an
  empty list and no setup lines. Every knob is drawn (defaults included), so the
  video alone tells the full settings the data was generated with."""
  import json

  try:
    sp = json.loads(str(run_meta_raw)).get("sampler")
  except (TypeError, ValueError):
    return []
  if not sp:
    return []

  def num(v, nd=2):
    return "off" if v is None else f"{v:.{nd}g}"

  return [
    (
      f"single {num(sp.get('single_prob'))}  oppose {num(sp.get('oppose_prob'))}  "
      f"counter {num(sp.get('counter_prob'))}  vertical {num(sp.get('vertical_scale'))}  "
      f"guided {'yes' if sp.get('guided') else 'no'}  "
      f"async {'yes' if sp.get('async_singles') else 'no'}"
    ),
    (
      f"disp<={num(sp.get('max_disp'))}  load_vel {num(sp.get('max_load_vel'))}  "
      f"com {num(sp.get('com_cost_xy'))}/{num(sp.get('com_cost_z'), 1)}  "
      f"cap {num(sp.get('com_cap'))}  pin {'yes' if sp.get('pin_free_wrists') else 'no'}  "
      f"torque {('<=' + num(sp.get('max_rot_disp'))) + 'rad' if sp.get('with_torque') else 'no'}"
    ),
  ]


def draw_banner(
  img: np.ndarray,
  t: float,
  frame: int,
  rows: list[dict],
  setup: list[str] | None = None,
) -> np.ndarray:
  """Draw the banner at the top of a frame: time / frame number + forced links (in
  slot colors) with |F| + small audit tags (the row's ``tag`` key,
  intent·branch[·solo][·r<backoff rotation deg>]); FREE when there is no contact.
  ``setup`` (optional) holds the self-described knob lines, appended below; the
  banner grows with the number of lines."""
  from PIL import Image, ImageDraw

  big, small = _fonts()
  pim = Image.fromarray(img)
  dr = ImageDraw.Draw(pim, "RGBA")
  bar_h = _BANNER_H + 19 * len(setup or [])
  dr.rectangle((0, 0, pim.width, bar_h), fill=(20, 24, 30, 210))
  for i, line in enumerate(setup or []):
    dr.text((8, _BANNER_H - 4 + 19 * i), line, font=small, fill=(165, 178, 200, 255))
  dr.text((8, 5), f"t={t:5.2f}s   frame {frame}", font=small, fill=(190, 205, 235, 255))
  if not rows:
    dr.text((8, 27), "FREE  (no contact)", font=big, fill=(130, 210, 130, 255))
    return np.asarray(pim)[:, :, :3]
  x = 8.0
  dr.text((x, 27), "FORCING:", font=big, fill=(235, 235, 235, 255))
  x += dr.textlength("FORCING:  ", font=big)
  for i, row in enumerate(rows):
    if i:
      dr.text((x, 27), "+", font=big, fill=(170, 170, 170, 255))
      x += dr.textlength("+  ", font=big)
    text = f"{short_link(row['link'])} {row['fmag']:.0f}N  "
    rgba = _SLOT_RGBA[row["slot"] % len(_SLOT_RGBA)]
    dr.text((x, 27), text, font=big, fill=tuple(int(255 * c) for c in rgba))
    x += dr.textlength(text, font=big)
    tag = row.get("tag")
    if tag:  # audit tag (intent·branch[·solo][·r<deg>]): smaller font, muted color
      dr.text((x, 31), f"{tag}  ", font=small, fill=(185, 195, 215, 255))
      x += dr.textlength(f"{tag}  ", font=small)
  return np.asarray(pim)[:, :, :3]


def _frame_range(spec: str | None, n: int) -> range:
  """ "START:STOP[:STEP]" -> range; without a spec, the whole clip [0, n)."""
  if not spec:
    return range(n)
  parts = (spec.split(":") + ["", ""])[:3]
  start, stop, step = (
    int(p) if p else d for p, d in zip(parts, (0, n, 1), strict=True)
  )
  return range(start, min(stop, n), step)


def default_out(clip: str, ext: str = ".mp4") -> str:
  """Default output: out/<clip name><ext>, or **next to the clip** if it is in out/.

  The second rule keeps the outputs of one generation run together: when the
  ref/adapted/contact/motion files sit under out/<name>/, the video and the curve
  plots belong in that same subdirectory, not back at the out/ root.
  """
  stem = os.path.splitext(os.path.basename(clip))[0]
  clip_dir = os.path.dirname(os.path.abspath(clip))
  inside = os.path.commonpath([clip_dir, OUT_DIR]) == OUT_DIR
  return os.path.join(clip_dir if inside else OUT_DIR, f"{stem}{ext}")


def render(
  clip: str,
  out: str | None = None,
  *,
  contact: str | None = None,
  fps: float | None = None,
  frames: str | None = None,
  width: int = 640,
  height: int = 800,
  distance: float = 3.0,
  azimuth: float = 155.0,
  elevation: float = -8.0,
  dz: float = 0.15,
  track: bool = True,
  banner: bool = True,
  decor: str = "spring",
  scene: bool = False,
) -> int:
  """Render a clip (optionally with contact labels) to mp4; return frames written.

  An empty ``out`` means out/<clip name>.mp4. Frames stream into the encoder one at
  a time (the clip is never buffered whole), so memory does not grow with clip
  length.

  ``decor`` picks what marks a forced link: ``spring`` = the spring device
  (live/setpoint balls + capsule + pushing arrow, so the compliant
  displacement is visible); ``arrow`` = one big orange arrow along the force (the
  most direct view of "who is pushed, how hard"). Both draw something only when
  ``contact`` is given.
  """
  if decor not in ("spring", "arrow"):
    raise ValueError(f"decor must be spring / arrow, got {decor!r}")
  scene_model = None
  if scene:
    # Scene model with a ground plane and a gradient skybox (the augmenter's own;
    # its qpos layout matches the bare G1 exactly). Imported only when needed: this
    # chain pulls in the whole large augmentation script plus mink.
    import sys

    _s = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _s not in sys.path:
      sys.path.insert(0, _s)
    from softmimic_mink_augment import build_g1_scene_model

    scene_model = build_g1_scene_model(width, height)
  out = out or default_out(clip)
  meta = np.load(clip, allow_pickle=False)
  fps = float(fps or np.asarray(meta["fps"]).reshape(-1)[0])
  ev = ClipEvidence(contact) if contact else None
  # Audit channels (the multilink augmenter writes them): the banner adds a small
  # tag per forced hand, "intent·branch[·solo][·r<backoff rotation deg>]". An NPZ
  # without them gets no tag.
  book = None
  if contact:
    d_con = np.load(contact, allow_pickle=False)
    if "intent" in d_con:
      book = {
        key: np.asarray(d_con[key])
        for key in ("intent", "intent_final", "guide_branch", "backoff_rot_deg")
      }
  setup = _setup_lines(d_con["run_meta"]) if contact and "run_meta" in d_con else []
  renderer = G1FrameRenderer(
    clip, width=width, height=height, body_colors=WRIST_COLORS, model=scene_model
  )
  # The bridge output is one frame shorter than the contact log (finite-difference
  # velocities); front-align and take the smaller count.
  n = renderer.num_frames if ev is None else min(renderer.num_frames, ev.T)
  look: dict = {
    "distance": distance,
    "azimuth": azimuth,
    "elevation": elevation,
    "dz": dz,
  }
  if not track:
    root = np.asarray(meta["body_pos_w"])[:n, 0]
    look["lookat"] = root.mean(axis=0) + np.array([0.0, 0.0, dz])
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  written = 0
  with media.VideoWriter(out, (height, width), fps=fps) as writer:
    for fr in _frame_range(frames, n):
      rows = [row for row in ev.frame(fr) if row["active"]] if ev else []
      if book is not None:
        bfr = min(fr, len(book["intent"]) - 1)
        for row in rows:
          s = int(row["slot"])
          intent = str(book["intent"][bfr, s])
          if not intent:
            continue
          # Intent abbreviations (along/open/twist -> al/op/tw): with tags on both
          # rows the full names do not fit in 640px.
          parts = [
            {"along": "al", "open": "op", "twist": "tw", "single": "sg"}.get(
              intent, intent
            )
          ]
          branch = str(book["guide_branch"][bfr, s])
          if branch:
            parts.append(branch)
          if str(book["intent_final"][bfr, s]) == "single":
            parts.append("solo")
          deg = float(book["backoff_rot_deg"][bfr, s])
          if deg > 0:
            parts.append(f"r{deg:.0f}")
          row["tag"] = "·".join(parts)

      def overlays(rr: G1FrameRenderer, rows=rows) -> None:
        # live wrist position = model FK at the posed frame (same source as the clip).
        for row in rows:
          live = rr.body_xpos(row["link"])
          force = np.asarray(row["force"], float)
          if decor == "arrow":
            # One big orange arrow from the wrist along the force, length
            # proportional to |F|: the most direct view of "who is pushed, which
            # way, how hard"; the small balls + capsule of the spring device are not
            # eye-catching enough in motion.
            rr.add_connector(
              live,
              live + force * ARROW_SCALE,
              radius=ARROW_RADIUS,
              rgba=ARROW_RGBA,
              arrow=True,
            )
          else:
            # The spring device: live ball + setpoint ball +
            # spring + pushing arrow. The extra setpoint end shows the gap between
            # "where the environment pulls the wrist" and where the wrist actually
            # is, i.e. the compliance itself.
            rr.add_spring_device(live, np.asarray(row["setpoint_pos"], float), force)

      img, _ = renderer.render_frame(
        fr, look_kw=look, overlays=overlays if rows else None, whiten=False
      )
      if banner:
        img = draw_banner(img, fr / fps, fr, rows, setup)
      writer.add_image(np.ascontiguousarray(img))
      written += 1
  print(f"wrote {written} frames @ {fps:g}fps -> {out}")
  return written


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "--clip",
    required=True,
    help="adapted motion NPZ (output of qpos_csv_to_motion_npz.py)",
  )
  ap.add_argument(
    "--contact",
    default=None,
    help="contact NPZ: enables per-frame labels + spring-device overlays",
  )
  ap.add_argument(
    "--out", default=None, help=f"output mp4 path; empty = {OUT_DIR}/<clip name>.mp4"
  )
  ap.add_argument("--fps", type=float, default=None, help="default: the clip's own fps")
  ap.add_argument(
    "--frames", default=None, metavar="START:STOP[:STEP]", help="frame slice"
  )
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=800)
  ap.add_argument("--distance", type=float, default=3.0)
  ap.add_argument("--azimuth", type=float, default=155.0)
  ap.add_argument("--elevation", type=float, default=-8.0)
  ap.add_argument(
    "--dz", type=float, default=0.15, help="z offset of lookat relative to the root"
  )
  ap.add_argument(
    "--no-track",
    action="store_true",
    help="fix the camera at the root's mean position (default: follow the root)",
  )
  ap.add_argument("--no-banner", action="store_true", help="do not draw the top banner")
  ap.add_argument(
    "--decor", choices=["spring", "arrow"], default="spring",
    help="how forced links are marked: spring = spring device (default), "
    "arrow = big orange arrow along the force",
  )  # fmt: skip
  ap.add_argument(
    "--scene", action="store_true",
    help="use the scene model with ground + skybox (default: robot only on white, "
    "where it appears to float)",
  )  # fmt: skip
  args = ap.parse_args()
  render(
    args.clip,
    args.out,
    contact=args.contact,
    fps=args.fps,
    frames=args.frames,
    width=args.width,
    height=args.height,
    distance=args.distance,
    azimuth=args.azimuth,
    elevation=args.elevation,
    dz=args.dz,
    track=not args.no_track,
    banner=not args.no_banner,
    decor=args.decor,
    scene=args.scene,
  )


def _demo() -> None:
  """End-to-end self-check: the overlays and banner must really follow the contact
  data (skips cleanly if the clip is missing).

  Run: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run python \\
      scripts/clip_viz/render_clip_video.py
  """
  import tempfile

  root = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      os.pardir,
      os.pardir,
      "data/compliant/rc/20260830_comz_small_002",
    )
  )
  clip = os.path.join(root, "waltz_20260224_001_multilink_forcefield_seed0.npz")
  contact = os.path.join(root, "waltz_20260224_001_multilink_contact_seed0.npz")
  if not os.path.exists(clip) or not os.path.exists(contact):
    print(f"skipping demo: {clip} not found")
    return
  ev = ClipEvidence(contact)
  forcing = next(f for f in range(ev.T) if all(ev.active(f)))
  free = next(f for f in range(ev.T) if not any(ev.active(f)))
  d = tempfile.mkdtemp()

  def run(tag: str, fr: int, *, contact_npz: str | None, banner: bool) -> np.ndarray:
    path = os.path.join(d, f"{tag}.mp4")
    render(
      clip,
      path,
      contact=contact_npz,
      frames=f"{fr}:{fr + 2}",
      width=320,
      height=400,
      banner=banner,
    )
    return np.asarray(media.read_video(path)).astype(int)

  # Forcing frame: with contact it must draw the spring device on top of the render
  # without contact; idle frame: the two must be identical (nothing should be drawn
  # without a contact). Asserting both sides catches both "drew nothing" and "drew
  # where it should not".
  on = run("forcing_on", forcing, contact_npz=contact, banner=False)
  off = run("forcing_off", forcing, contact_npz=None, banner=False)
  free_on = run("free_on", free, contact_npz=contact, banner=False)
  free_off = run("free_off", free, contact_npz=None, banner=False)
  banner_on = run("banner", forcing, contact_npz=contact, banner=True)

  assert on.shape == (2, 400, 320, 3), f"wrong frame count / size: {on.shape}"
  assert np.abs(on - off).max() > 25, "forcing frame did not draw the spring device"
  assert np.abs(free_on - free_off).max() <= 2, "idle frame should draw no overlay"
  strip = slice(0, _BANNER_H)
  assert np.abs(banner_on[:, strip] - on[:, strip]).max() > 25, "banner not drawn"
  print(f"OK: forcing=f{forcing} free=f{free}; overlays + banner follow contact ({d})")


if __name__ == "__main__":
  import sys

  # No arguments = run the self-check (the g1_render.py convention); with arguments
  # = the real CLI.
  _demo() if len(sys.argv) == 1 else main()
