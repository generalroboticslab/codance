"""Plot a contact NPZ's per-slot |F| against frame number as a PNG, with event spans
shaded and each event's peak labeled.

Rather than path data for Figma / SVG (see ClipEvidence.curve_path), this draws a
plot to read directly, answering "what did the added force actually look like":
does it follow the ramp-up / hold / ramp-down profile, are the two hands in sync,
how large is the peak, did a backoff tear a dip between events. It pairs with
`render_clip_video.py`: that one shows the pose, this one the numbers.

Run (no GL needed):
    rc=data/compliant/rc/20260830_comz_small_002
    uv run python scripts/clip_viz/plot_force_curve.py \\
        --contact $rc/waltz_20260224_001_multilink_contact_seed0.npz

Without --out the PNG goes to out/<contact file name>.png; if the contact NPZ itself
is inside out/, it goes next to it. Accepts both multi-link and single-contact NPZs
(single-contact is shown as a K=1 view whose link changes per event, and the legend
reads "(varies)").

Colors: the slot colors are **deepened** versions of the wrist pair used in 3D (left
wrist blue / right wrist pink). The light pair works as a surface fill in the
renders, but as lines on white it fails the legibility checks (measured: protan
ΔE 5.9, normal vision ΔE 13.1, contrast 2.0:1; with protanopia the two lines are
nearly indistinguishable). Deepened, all six checks pass (protan 11.7 / normal 25.1 /
contrast ≥3:1), while the blue/pink identity mapping stays, so readers can still
match each curve to the colored hand in the render.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot: render without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from clip_evidence import ClipEvidence  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from palette import PAL, figure_defaults  # noqa: E402
from render_clip_video import default_out  # noqa: E402

figure_defaults()
# Slot colors: deepened versions of the 3D wrist pair (see the module docstring for
# the checks). Taken in fixed slot order, never cycled.
SLOT_LINE = [PAL.MATPLOTLIB.hex(name) for name in PAL.SLOT]
INK = PAL.MATPLOTLIB.hex("ink")
INK_MUTED = PAL.MATPLOTLIB.hex("ink_muted")
BAND = PAL.MATPLOTLIB.hex("band")


def _slot_color(k: int) -> str:
  """Line color of slot k. Past the end of the palette, fall back to the neutral ink
  instead of inventing a new hue."""
  return SLOT_LINE[k] if k < len(SLOT_LINE) else INK_MUTED


# Frame ticks are picked only from this fixed ladder, never by matplotlib itself.
_TICK_LADDER = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


def tick_step(n_frames: int, target: int = 9) -> int:
  """Tick spacing from the fixed ladder whose tick count is closest to target.

  The fixed ladder puts **two clips of similar length on the same spacing**: for
  400-frame and 487-frame clips, matplotlib's automatic ticks would pick 50 and 100
  respectively, so side by side the ticks do not line up and the reader has to
  convert in their head.
  """
  return min(_TICK_LADDER, key=lambda s: abs(n_frames / s - target))


def draw_force_panel(
  ax, ev: ClipEvidence, *, xmax: int | None = None, marks: Sequence[int] = ()
) -> None:
  """Draw per-slot |F|(t) + event shading + per-event peak labels on one axis.

  A reusable axis-level piece: the standalone force plot and the combined figure of
  `plot_stiffness_curve.py` share this drawing code, so their force panels always
  look the same. A zero-force (zero-wrench) clip is drawn as a flat line at 0 rather
  than left blank: in the combined figure "stiffness moves while force stays zero"
  is the point, and an empty panel would read as "not measured", while a flat zero
  line reads as "measured, and it is zero".

  ``marks`` draws a vertical marker labeled with its frame number at each given
  frame: when still frames accompany the plot, readers need to place each render at
  its exact spot on this time axis, otherwise "how large is the force in this
  frame" is guesswork.
  """
  t = np.arange(ev.T)
  spans = sorted({(e["f_start"], e["f_end"]) for e in ev.events()})
  for i, (f0, f1) in enumerate(spans):
    ax.axvspan(f0, f1, color=BAND, zorder=0, label="event" if i == 0 else None)

  for k in range(ev.K):
    mag = ev.force_curve(k)
    ax.plot(t, mag, color=_slot_color(k), lw=2.0, zorder=3, label=ev.slot_link_names[k])
    # Selective direct labels: one peak per event, not a number on every frame.
    for f0, f1 in spans:
      seg = mag[int(f0) : int(f1) + 1]
      if seg.max() < 1e-6:
        continue
      j = int(f0) + int(seg.argmax())
      ax.annotate(
        f"{seg.max():.0f}N",
        (t[j], mag[j]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
        color=INK_MUTED,  # values in ink, not in the series color
        zorder=4,
      )

  ax.set_ylabel("|F|  (N)", fontsize=9, color=INK_MUTED)
  hi = float(xmax) if xmax is not None else float(t[-1])
  ax.set_xlim(0, hi)
  ax.xaxis.set_major_locator(MultipleLocator(tick_step(int(hi))))
  # 15% headroom at the top: peak labels are drawn offset upward and would be clipped
  # by the axis at the upper bound.
  peak = max((float(ev.force_curve(k).max()) for k in range(ev.K)), default=0.0)
  ax.set_ylim(0, peak * 1.15 or 1.0)
  # Markers sit above the curves and below the labels: a marker is a "look at this
  # frame" pointer, so the curves must not hide it and it must not cover the values.
  for f in marks:
    ax.axvline(f, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=3.5)
    ax.annotate(
      f"f{int(f)}",
      (f, ax.get_ylim()[1]),
      textcoords="offset points",
      xytext=(3, -9),
      ha="left",
      fontsize=8,
      color=INK,
      zorder=4,
    )

  ax.grid(axis="y", color=BAND, lw=0.8, zorder=1)  # keep the grid in the background
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(BAND)
  ax.tick_params(colors=INK_MUTED, labelsize=8)
  ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")


def plot(
  contact: str,
  out: str | None = None,
  *,
  xmax: int | None = None,
  marks: Sequence[int] = (),
) -> str:
  """Plot per-slot |F| against **frame number** + event shading; return the PNG path.

  The x axis is frames, not seconds: everything else in this pipeline speaks in
  frames (the event table's f_start/f_end are frame indices, the contact NPZ is
  indexed by frame, the video banner prints the frame number), and the contact NPZ
  carries no fps field, so converting to seconds would mean guessing one from
  outside, and a wrong guess shifts the whole time axis. Frame numbers line up with
  what the file itself holds and bring in no outside assumption.

  ``xmax`` forces the x-axis upper bound: give two clips of **different lengths**
  the same value when comparing them side by side, so the same frame number lands on
  the same pixel (matching the tick spacing alone is not enough; with different
  ranges the positions still drift apart).
  """
  out = out or default_out(contact, ".png")
  ev = ClipEvidence(contact)
  fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
  draw_force_panel(ax, ev, xmax=xmax, marks=marks)
  ax.set_xlabel("frame", fontsize=9, color=INK_MUTED)
  ax.set_title(os.path.basename(contact), fontsize=10, color=INK, loc="left", pad=10)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  fig.savefig(out, facecolor="white")
  plt.close(fig)
  n_events = len({(e["f_start"], e["f_end"]) for e in ev.events()})
  print(f"wrote {out}  (T={ev.T} K={ev.K} events={n_events})")
  return out


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "--contact", required=True, help="contact NPZ (multi-link or single-contact)"
  )
  ap.add_argument(
    "--out",
    default=None,
    help="output PNG; default: next to the contact file or in the out/ root",
  )
  ap.add_argument(
    "--xmax", type=int, default=None,
    help="force the x-axis upper bound (frames); give two clips of different "
    "lengths the same value to compare them side by side",
  )  # fmt: skip
  ap.add_argument(
    "--marks", default=None,
    help="comma-separated frame numbers: a vertical marker + frame label each, to "
    "place rendered stills on the time axis",
  )  # fmt: skip
  args = ap.parse_args()
  marks = [int(x) for x in args.marks.split(",")] if args.marks else ()
  plot(args.contact, args.out, xmax=args.xmax, marks=marks)


def _demo() -> None:
  """Self-check: plot the seed clip and assert the event shading and the curves
  really land on the image (skipped if the file is missing)."""
  import tempfile

  root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data/compliant")
  )
  contact = os.path.join(
    root, "rc/20260830_comz_small_002/waltz_20260224_001_multilink_contact_seed0.npz"
  )
  if not os.path.exists(contact):
    print(f"skipping demo: {contact} not found")
    return
  import imageio.v2 as iio

  out = os.path.join(tempfile.mkdtemp(), "curve.png")
  plot(contact, out)
  img = iio.imread(out)[:, :, :3]
  assert img.shape[0] > 200 and img.shape[1] > 800, f"wrong image size: {img.shape}"
  # Both slot colors and the band color must really appear in the pixels (drawn, not
  # an empty plot).
  for name, hexc in (("slot0", SLOT_LINE[0]), ("slot1", SLOT_LINE[1]), ("band", BAND)):
    rgb = np.array([int(hexc[i : i + 2], 16) for i in (1, 3, 5)])
    hits = int((np.abs(img.astype(int) - rgb).sum(axis=-1) < 30).sum())
    assert hits > 20, f"{name} ({hexc}) barely appears in the image: {hits} pixels"
  print(f"OK: {out}")


if __name__ == "__main__":
  import sys

  _demo() if len(sys.argv) == 1 else main()
