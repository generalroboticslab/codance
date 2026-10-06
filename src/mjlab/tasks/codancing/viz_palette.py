"""One palette home for every color this repo draws.

The 3D drawer, the offline renderer and the matplotlib figures all read THIS
module, never inline literals, so the mp4, the still and the figures agree by
construction.

NAMES CARRY OPACITY. A color name is ``<opacity>_<color>``, where the number is
percent OPAQUE: ``10_cyan`` is alpha 0.10 (a faint wash), ``90_cyan`` is alpha
0.90 (nearly solid). A bare ``cyan`` is fully opaque, which is what plot code
normally wants, since matplotlib takes alpha as its own argument.

ONE ROLE, PER-MEDIUM VALUES. The same named role often needs a different actual
value per medium: a color that separates well against the 3D scene's blue floor
is too light on a white figure background. So a name carries a base value plus
optional per-medium overrides, and ``_`` is the fallback for any medium not
named. The common case is one value; divergence is opt-in, which is what keeps
the table easy to tweak.

The media are drawing surfaces, not file extensions:

* ``video`` -- the mjlab 3D recorder / viewer, and anything rendering through
  ``env.render()``.
* ``png`` -- the standalone ``g1_render`` offline renderer, which poses a bare
  G1 on a paper-white background with no scene around it.
* ``matplotlib`` -- figure plots.

SINGLE THEME. Medium is the only axis; there is no light-versus-dark pair.
"""

from dataclasses import dataclass

Rgb = tuple[float, float, float]
Rgba = tuple[float, float, float, float]

MEDIA: tuple[str, ...] = ("video", "png", "matplotlib")

# name -> {medium: rgb}, where "_" is the value for every medium not named.
# Alpha is NOT here: it rides in the name (`45_orange`), so one entry serves
# every opacity a caller needs.
COLORS: dict[str, dict[str, Rgb]] = {
  # ── neutrals and surfaces ──────────────────────────────────────────────────
  "ink": {"_": (0.121569, 0.141176, 0.188235)},  # #1f2430
  "ink_soft": {"_": (0.168627, 0.184314, 0.239216)},  # #2b2f3d
  "ink_muted": {"_": (0.352941, 0.392157, 0.447059)},  # #5a6472
  "surface": {"_": (0.988235, 0.988235, 0.984314)},  # #fcfcfb
  # Grid and event-band shade, composited at alpha 0.35 over `surface`: it
  # reaches 21 of 255 against the surface in rendered pixels, enough to read
  # next to a slot event band, while staying neutral gray. Surface shade, not a
  # validated series color.
  "band": {"_": (0.737255, 0.745098, 0.760784)},  # #bcbec2
  "white": {"_": (1.0, 1.0, 1.0)},
  "slate": {"_": (0.17, 0.20, 0.25)},
  "slate_mid": {"_": (0.36, 0.38, 0.42)},
  "slate_pale": {"_": (0.55, 0.58, 0.62)},
  # ── slot identity (which contact) ─────────────────────────────────────────
  # The one role that genuinely diverges by medium: in 3D a slot color is worn
  # by a whole wrist link against a blue floor, so it is pale enough to read as
  # "that hand"; on a white figure the same identity has to carry a 1 px line,
  # so it is the deepened pair that passes the six color-blindness checks
  # (protan 11.7 / normal 25.1 against the surface).
  "slot_a": {"_": (0.60, 0.71, 0.86), "matplotlib": (0.180392, 0.427451, 0.705882)},
  "slot_b": {"_": (0.93, 0.62, 0.70), "matplotlib": (0.768627, 0.247059, 0.419608)},
  # ── figure series ─────────────────────────────────────────────────────────
  "blue": {"_": (0.164706, 0.470588, 0.839216)},  # #2a78d6
  "green": {"_": (0.105882, 0.686275, 0.478431)},  # #1baf7a
  "amber": {"_": (0.929412, 0.631373, 0.0)},  # #eda100
  "amber_dark": {"_": (0.690196, 0.486275, 0.164706)},  # #b07c2a
  "pink": {"_": (0.909804, 0.482353, 0.643137)},  # #e87ba4
  "violet": {"_": (0.290196, 0.227451, 0.654902)},  # #4a3aa7
  "violet_soft": {"_": (0.478431, 0.360784, 0.776471)},  # #7a5cc6
  "orange": {"_": (0.921569, 0.407843, 0.203922)},  # #eb6834
  "red": {"_": (0.847059, 0.2, 0.250980)},  # #d83340
  "red_dark": {"_": (0.760784, 0.250980, 0.164706)},  # #c2402a
  "mint": {"_": (0.933333, 0.956863, 0.925490)},  # #eef4ec
  # ── in-scene marker hues ──────────────────────────────────────────────────
  # The scripted force arrow. On the offline renderer it is drawn on a bare G1
  # against paper white with no scene behind it, so it can afford to be more
  # saturated than the version that has to sit over a blue checkerboard floor.
  "orange_bright": {"_": (1.0, 0.55, 0.1), "png": (1.0, 0.45, 0.0)},
  "cyan": {"_": (0.2, 0.8, 0.95)},
  "cyan_bright": {"_": (0.3, 0.9, 1.0)},
  # The human partner, paired with the salmon `free` face as the two people of a
  # dancing pair. The muted value reads as its own color on a white figure but
  # not over the dark blue checkerboard, where it also stops separating from the
  # green `reference` ghost, so `video` takes the brighter, bluer cyan that
  # already passed that check for the three-ghost set above.
  "teal": {"_": (0.372549, 0.666667, 0.729412), "video": (0.2, 0.75, 0.9)},
  "magenta": {"_": (1.0, 0.4, 0.85)},
  "magenta_deep": {"_": (1.0, 0.3, 0.85)},
  "magenta_soft": {"_": (0.95, 0.45, 0.85)},
  "yellow": {"_": (1.0, 0.95, 0.2)},
  "yellow_pale": {"_": (1.0, 0.97, 0.45)},
  "gold": {"_": (1.0, 0.85, 0.3)},
  "green_soft": {"_": (0.5, 0.7, 0.5)},
  # Three-ghost separation set. When a scene holds the robot's reference, its
  # free face AND the imagined partner, the muted face defaults blur into each
  # other and into the floor, so a partner profile swaps in these: a greener
  # green, a redder red, and a cyan that stays brighter than the blue
  # checkerboard.
  "green_vivid": {"_": (0.4, 0.85, 0.4)},
  "red_deep": {"_": (0.9, 0.3, 0.3)},
  "cyan_deep": {"_": (0.2, 0.75, 0.9)},
  "green_bright": {"_": (0.45, 0.95, 0.5)},
  "green_dark": {"_": (0.3, 0.6, 0.35)},
  "blue_bright": {"_": (0.35, 0.45, 1.0)},
  "blue_dark": {"_": (0.2, 0.26, 0.6)},
  "amber_bright": {"_": (0.95, 0.65, 0.2)},
  "red_soft": {"_": (0.9, 0.45, 0.45)},
  "crimson": {"_": (0.69, 0.07, 0.15)},
}

# ── in-scene geometry ─────────────────────────────────────────────────────────
# Sizes live here for the same reason colors do: an in-scene marker is a size AND
# a color, the two halves of one contract (a wide translucent baked arrow under a
# thin solid live one), and splitting them across two files would let them
# drift apart. Same per-medium shape as `COLORS`, and the
# divergence is just as real: `png` is the standalone `g1_render` renderer, which
# frames one bare G1 with nothing else in shot, so its scaffolding can be finer
# than anything that has to stay readable next to three ghosts and a floor.
#
# Meters, except the two named `_scale` (a ratio) and `force_arrow_scale`
# (meters per newton). NOT here: sizes that vary per call within one figure (the
# ratios inside one drawing routine) -- those are arguments, and their defaults
# are the entries below.
SIZES: dict[str, dict[str, float]] = {
  # The force arrow is the MAGNITUDE channel: length follows |F| so a 5 N push
  # still draws. Linear below the knee (5 to 25 N is untouched), logarithmic
  # above it, so a few-hundred-newton spike compresses instead of drawing metres
  # of arrow. Not a hard cap: three arrows over the cap would clip to identical
  # lengths and stop being comparable.
  "force_arrow_scale": {"_": 0.02, "png": 0.030},
  "force_arrow_knee_len": {"_": 0.5},
  # The spring style is the GEOMETRY channel: the connector spans exactly link to
  # setpoint, and a fixed-length arrow behind the link carries direction only.
  # Offset is mandatory -- under the spring law F is colinear with
  # (setpoint - live), so an arrow drawn from the link along F lies on the connector.
  "spring_connector_radius": {"_": 0.006, "png": 0.003},
  "spring_ball_scale": {"_": 0.28},  # ratio: setpoint ball shrinks in this style
  "dir_arrow_len": {"_": 0.15},
  "dir_arrow_clear": {"_": 0.055},
  "dir_arrow_width": {"_": 0.009},
  # The size half of the baked-vs-live contract (the alpha half is in
  # MARKER_COLOR): BAKED/un-reanchored faces draw LARGE and WIDE, LIVE/reanchored
  # faces SMALL and THIN, so the two families stay tellable even on a frame where
  # a pair coincides exactly.
  "setpoint_ball_baked": {"_": 0.07},
  "setpoint_ball_live": {"_": 0.022},
  "wrench_arrow_baked": {"_": 0.035, "png": 0.020},
  "wrench_arrow_live": {"_": 0.012},
  # Re-anchor scaffolding: a ball at the blended base point and hairlines out to
  # what it carries. Both families draw at these sizes, so the frozen and
  # per-frame pair differ only in color.
  "reanchor_base_ball": {"_": 0.03},
  "reanchor_connector_radius": {"_": 0.005},
  "reanchor_heading_len": {"_": 0.35},
  "reanchor_heading_width": {"_": 0.012},
  # Offline-renderer overlay defaults. `spring_device_ball` is that renderer's
  # endpoint ball; the recorder's counterpart is not a constant but a derivation
  # (`setpoint_ball_baked` x `spring_ball_scale`), so the two are named
  # separately rather than pretending to be one number.
  "spring_device_ball": {"_": 0.008},
  "probe_ball": {"_": 0.012},
  "probe_connector_radius": {"_": 0.004},
}


def size(name: str, medium: str) -> float:
  """One in-scene dimension on one medium. Meters, unless the name says
  otherwise (a ``_scale`` is a ratio; ``force_arrow_scale`` is m per N)."""
  if medium not in MEDIA:
    raise KeyError(f"unknown medium {medium!r}; valid: {MEDIA}")
  try:
    values = SIZES[name]
  except KeyError:
    raise KeyError(f"unknown size {name!r}; valid: {sorted(SIZES)}") from None
  return values.get(medium, values["_"])


# The one typeface every text surface writes in, in its two faces: Regular for
# body text, Bold wherever a surface already asks for emphasis (the figure
# module's title block and column headers). Sizes are NOT here: a video banner
# size is in pixels against a known frame height and a matplotlib size is in
# points against a figure DPI, so a shared scale would have to model different
# units to say anything. The family alone is what makes a run's videos and plots
# read as one document.
#
# Not installed by the repo: put the two OFL files in a user font dir
# (`~/.local/share/fonts/OpenSans/`), run `fc-cache -f`, and clear matplotlib's
# font cache. `font_file` below says so when they are missing. Install the
# `latin` subset rather than `latin-ext`: the Fontsource split puts basic ASCII
# in `latin` ONLY, so a `latin-ext` file alone renders every digit and letter of
# a text line as tofu, and installing both under one family leaves matplotlib
# picking between two files that claim the same (family, weight).
FONT_FAMILY = "Open Sans"
FONT_WEIGHTS: dict[str, str] = {"regular": "normal", "bold": "bold"}

_font_warned: set[tuple[str, str]] = set()


def font_file(weight: str = "regular") -> str:
  """The TrueType file for one of the two faces, for PIL's ``ImageFont``.

  Warns ONCE per face when the family is not installed. matplotlib's resolver
  silently substitutes DejaVu Sans and mentions it only on a logger nobody turns
  on, so without this a missing font looks exactly like a font that worked --
  and every surface would go on agreeing with every other surface about the
  wrong typeface.
  """
  from matplotlib import font_manager

  try:
    mpl_weight = FONT_WEIGHTS[weight]
  except KeyError:
    raise KeyError(
      f"unknown font weight {weight!r}; valid: {sorted(FONT_WEIGHTS)}"
    ) from None
  prop = font_manager.FontProperties(family=FONT_FAMILY, weight=mpl_weight)
  path = font_manager.findfont(prop)
  key = (FONT_FAMILY, weight)
  if key not in _font_warned:
    _font_warned.add(key)
    got = font_manager.get_font(path).family_name
    if got.lower() != FONT_FAMILY.lower():
      print(
        f"[WARN] font {FONT_FAMILY!r} {weight} is not installed; falling back to "
        f"{got!r} ({path}). Every text surface still agrees with every other "
        "one, but the run is not written in the typeface the palette names. "
        "Install the family (a user font dir is enough: ~/.local/share/fonts, "
        "then `fc-cache -f`) and clear matplotlib's font cache."
      )
  return path


def _split(name: str) -> tuple[str, float]:
  """``"45_orange"`` -> ``("orange", 0.45)``; a bare name is fully opaque."""
  head, sep, tail = name.partition("_")
  if sep and head.isdigit():
    return tail, int(head) / 100.0
  return name, 1.0


def rgb(name: str, medium: str) -> Rgb:
  """The rgb this name resolves to on this medium (opacity in the name is
  ignored; use :func:`rgba` or :func:`hex` to consume it)."""
  if medium not in MEDIA:
    raise KeyError(f"unknown medium {medium!r}; valid: {MEDIA}")
  color, _ = _split(name)
  try:
    values = COLORS[color]
  except KeyError:
    raise KeyError(
      f"unknown color {color!r} (from name {name!r}); valid: {sorted(COLORS)}"
    ) from None
  return values.get(medium, values["_"])


def rgba(name: str, medium: str) -> Rgba:
  """rgb plus the alpha the name carries -- what mujoco's ``geom_rgba`` and the
  debug visualizer take."""
  color, alpha = _split(name)
  r, g, b = rgb(color, medium)
  return (r, g, b, alpha)


def hex(name: str, medium: str) -> str:  # noqa: A001
  """``"#rrggbb"`` -- what matplotlib takes. Opacity in the name is dropped
  (matplotlib carries alpha as its own argument), so pass a bare name here."""
  return "#" + "".join(f"{round(c * 255):02x}" for c in rgb(name, medium))


# ── one view per medium ───────────────────────────────────────────────────────
# The medium is the half of every lookup that a call site repeats, so each
# medium gets one bound view, and a call still says which surface it is for:
# `MATPLOTLIB.hex("ink")` names it in the view rather than in a string argument.
#
# Prefer the bound view. The two-argument functions above stay public for the
# one case it cannot serve: a medium chosen at runtime (`describe(medium)`).


@dataclass(frozen=True)
class Medium:
  """The palette's emitters bound to one drawing surface."""

  name: str

  def rgb(self, color: str) -> Rgb:
    return rgb(color, self.name)

  def rgba(self, color: str) -> Rgba:
    return rgba(color, self.name)

  def hex(self, color: str) -> str:
    return hex(color, self.name)

  def size(self, dimension: str) -> float:
    return size(dimension, self.name)


VIDEO = Medium("video")
PNG = Medium("png")
MATPLOTLIB = Medium("matplotlib")


# ── figure roles ──────────────────────────────────────────────────────────────
# Face identity (which force / which setpoint face) is constant across every
# panel; slot identity colors subplot titles and header chips, never lines.
SLOT: tuple[str, ...] = ("slot_a", "slot_b")


# ── 3D (in-scene) marker contract ─────────────────────────────────────────────
# Alpha carries half of the baked-vs-live contract (the size half is `SIZES`
# above): BAKED/un-reanchored
# faces (scripted wrench, script setpoint, data arrow) are translucent,
# LIVE/reanchored faces (applied wrench, rise + per-frame setpoints and the
# per-frame arrow) are fully opaque -- a thin solid stroke stays readable on
# top of a wide translucent one when the two faces coincide exactly (identity
# frozen anchor). The forced link itself has no marker: the reference ghost's
# wrist wears the slot color via `body_styles` instead.
MARKER_COLOR: dict[str, str] = {
  "scripted_wrench": "45_orange_bright",
  "applied_wrench": "cyan",
  "setpoint_script": "35_magenta",
  "setpoint_active": "cyan_bright",
  "setpoint_per_frame": "yellow",
  "per_frame_arrow": "yellow",
  "data_arrow": "45_magenta_deep",
  # The blended base point's ball (the connector under it reuses this rgb at 0.6 of
  # the alpha). Bright = the CURRENT per-frame anchor, matching the heading
  # arrows' bright-is-current / dark-is-frozen contract; the frozen base keeps
  # its own dimmer gold.
  "reanchor_base_per_frame": "95_yellow_pale",
  "reanchor_base": "90_gold",
  # Each base ball's hairline connector, and the offset connectors that tie a
  # base point to the setpoint it carries. All translucent: they are scaffolding
  # for reading the transform, not faces in their own right.
  "reanchor_base_connector": "55_gold",
  "reanchor_base_per_frame_connector": "57_yellow_pale",
  "reanchor_offset_data": "55_magenta",
  "reanchor_offset_active": "55_cyan_bright",
  "reanchor_offset_per_frame": "55_yellow",
  "reanchor_offset_data_per_frame": "55_magenta_deep",
  # The re-anchor transform itself: two heading arrows at the CURRENT anchors
  # and their dimmer twins at the frozen ones, so bright-vs-dark reads as
  # per-frame-vs-rising-edge without a legend.
  "reanchor_ref": "95_green_bright",
  "reanchor_robot": "95_blue_bright",
  "reanchor_ref_frozen": "50_green_dark",
  "reanchor_robot_frozen": "50_blue_dark",
}
MARKER_RGBA: dict[str, Rgba] = {k: rgba(v, "video") for k, v in MARKER_COLOR.items()}
MARKER_COLOR_WORD: dict[str, str] = {
  "scripted_wrench": "orange",
  "applied_wrench": "cyan",
  "setpoint_script": "magenta",
  "setpoint_active": "cyan",
  "setpoint_per_frame": "yellow",
  "data_arrow": "magenta",
  "reanchor_base_per_frame": "yellow",
}


def expand_force_channels(
  *, force_arrow: bool, reanchor: bool, explicit: object = None
) -> frozenset[str]:
  """The channel set a render draws: an explicit list, or the two boolean
  toggles expanded.

  ONE copy on purpose: the drawer and the session description both need this
  answer, and two copies could disagree about it.
  """
  if explicit is not None:
    unknown = set(explicit) - set(FORCE_CHANNELS)
    if unknown:
      raise ValueError(
        f"Unknown force-viz channels {sorted(unknown)}; valid: {FORCE_CHANNELS}"
      )
    return frozenset(explicit)
  channels: set[str] = set()
  if force_arrow:
    channels |= {
      "scripted_wrench",
      "applied_wrench",
      "setpoint_script",
      "setpoint_active",
    }
  if reanchor:
    channels |= {
      "setpoint_per_frame",
      "data_arrow",
      "reanchor_transform",
      "reanchor_base_per_frame",
    }
  return frozenset(channels)


# The named channels a session may request (`debug_vis.force_channels`); None
# in config expands the two boolean toggles (see `expand_force_channels` and
# `_resolved_force_channels` on the command). Every marker has exactly one
# channel, so any subset composes a figure.
FORCE_CHANNELS: tuple[str, ...] = (
  "scripted_wrench",
  "applied_wrench",
  "setpoint_script",
  "setpoint_active",
  "setpoint_per_frame",
  "data_arrow",
  "reanchor_transform",
  "reanchor_base_per_frame",
)


# ── reference faces ───────────────────────────────────────────────────────────
# Two kinds of name are involved and they are not interchangeable. A GHOST color
# is one rgba, so it uses the `<opacity>_<color>` grammar above. A COORD-FRAME
# color is three rgb triples (one per arrow) with opacity carried separately,
# because both viewer backends build each arrow's rgba as this color plus the
# frame's single alpha. So coord-axis sets get their own table and the alpha
# rides beside them in `FACE_STYLE`.
#
# The LIVE face passes None (the visualizer's own full-strength rgb), so every
# softened triple below reads as "a reference face" at a glance and a still image
# says which face without a legend.
COORD_AXIS_SETS: dict[str, tuple[Rgb, ...]] = {
  "reference_axes": ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0)),
  "desired_axes": ((1.0, 0.65, 0.2), (0.7, 0.9, 0.25), (0.4, 0.6, 1.0)),
  "free_axes": ((1.0, 0.4, 0.85), (0.75, 0.5, 1.0), (0.5, 0.8, 1.0)),
  # The plain x-red / y-green / z-blue convention, for a figure that marks one
  # body's own frame rather than a reference face. Deliberately NOT a face
  # color: it says "these are axes", not "this is that pose".
  "marker_axes": ((0.85, 0.10, 0.10), (0.10, 0.70, 0.20), (0.10, 0.35, 0.95)),
}


# Default styling per drawable face. Ghost alpha is deliberately below the
# reference ghost's on a transported face, since a transported face usually
# overlaps it. `coord_axis` is None where a face has no coord frames, and then
# `coord_axis_alpha` has no referent; `current` has neither a ghost of its own
# (the live robot is its own mesh) nor softened axes (it uses the visualizer's
# default full-strength rgb). `coord_axis_len` is the face's own arrow length --
# overlapping faces read better when the live one is drawn longest, so a reader
# can tell the stack apart where they coincide -- and `anchor_axis_len` is the
# one-body anchor layer that rides along with it. `body_styles` (per-body, in
# config) overrides individual bodies on top.
@dataclass(frozen=True)
class FaceStyle:
  ghost: str  # palette name, carries its own alpha
  coord_axis: str | None = None  # a COORD_AXIS_SETS key
  coord_axis_alpha: float = 1.0
  coord_axis_len: float = 0.10  # meters; the live face draws longest
  anchor_axis_len: float | None = None  # the face's anchor layer, None = none


FACE_STYLE: dict[str, FaceStyle] = {
  "reference": FaceStyle("50_green_soft", "reference_axes", 0.75, 0.08, 0.10),
  "desired": FaceStyle("45_amber_bright", "desired_axes", 0.90, 0.10),
  "free": FaceStyle("50_red_soft", coord_axis_len=0.10),
  "desired_free": FaceStyle("40_magenta_soft", "free_axes", 0.90, 0.10),
  "current": FaceStyle("white", coord_axis_len=0.12, anchor_axis_len=0.15),
}
# The imagined partner is not one of the robot's reference faces, so it is named
# separately. Teal, deliberately far from BOTH the green robot ghost and the
# default blue checkerboard floor: a blue ghost blends into the floor and the
# overlapping green robot ghost and reads as "not drawn". The per-medium split
# that keeps that true while still pairing with the salmon `free` face lives on
# `teal` itself.
HUMAN_GHOST_COLOR = "50_teal"


# ── what is this actually drawing? ────────────────────────────────────────────


def describe(medium: str = "video") -> str:
  """The palette resolved, as text.

  Exists so that nothing else has to write a color down in order to tell a
  reader what they are looking at. A comment that says "green is the reference
  ghost" is a copy of this table, and it goes wrong silently the first time
  somebody edits a face; a comment that says "run this" cannot.

      uv run python src/mjlab/tasks/codancing/viz_palette.py [medium]
  """
  out = [f"palette, medium={medium!r}  (media: {', '.join(MEDIA)})", ""]
  out.append("FACES  (session.debug_vis draws these; body_styles overrides per body)")
  for face, style in FACE_STYLE.items():
    axes = "none" if style.coord_axis is None else style.coord_axis
    out.append(
      f"  {face:14s} ghost {style.ghost:18s} {hex(style.ghost, medium)}"
      f"  alpha {rgba(style.ghost, medium)[3]:.2f}"
      f"  axes {axes} @ {style.coord_axis_len} m"
    )
  out.append(
    f"  {'human partner':14s} ghost {HUMAN_GHOST_COLOR:18s} "
    f"{hex(HUMAN_GHOST_COLOR, medium)}"
  )
  out.append("")
  out.append("FORCE MARKERS  (session.debug_vis.force_channels picks the subset)")
  for channel, name in MARKER_COLOR.items():
    word = MARKER_COLOR_WORD.get(channel, "")
    r, g, b, a = rgba(name, medium)
    out.append(
      f"  {channel:30s} {name:24s} {hex(name, medium)}  alpha {a:.2f}"
      + (f"  ({word} in the session description)" if word else "")
    )
  out.append("")
  out.append("FIGURE SERIES")
  for i, name in enumerate(SLOT):
    out.append(f"  slot {i:<10d} {name:18s} {hex(name, 'matplotlib')}")
  out.append("")
  out.append(f"SIZES, meters unless the name says otherwise (medium={medium!r})")
  for name in SIZES:
    out.append(f"  {name:26s} {size(name, medium)}")
  out.append("")
  out.append(f"FONT  {FONT_FAMILY}, faces: {', '.join(FONT_WEIGHTS)}")
  return "\n".join(out)


if __name__ == "__main__":
  import sys

  print(describe(sys.argv[1] if len(sys.argv) > 1 else "video"))
