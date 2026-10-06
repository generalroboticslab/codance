"""Plot a contact NPZ's force and per-slot stiffness in one PNG: the |F| panel on top,
four panels below for robot stiffness and force-field (environment) stiffness
(position + rotation, raw values), on a shared frame axis with event spans shaded.

One figure shows what each stiffness channel means, and makes the **decorrelation**
of force and stiffness directly visible: force rises and falls inside the event
bands, while the robot stiffness curve stays flat across an event's **start** (its
value switches only where an event ends, and falls to the 140/1 placeholder after
the last event); the force-field stiffness K_env is nonzero only inside events (it
is the gain of the reactive spring); in zero-wrench clips force and K_env stay zero
and the robot stiffness is a random staircase with 2 to 5 s holds. Only raw values
are plotted, no log: the trends match the log transform, and a value off the
dashed 140 placeholder line is easy to spot on the raw panels. The force panel
shares its drawing code with
`plot_force_curve.py` (draw_force_panel), so the two always look the same.

Run (no GL needed; the contact NPZs of a whole pool can go in at once):
    uv run python scripts/clip_viz/plot_stiffness_curve.py \\
        --out-dir data/compliant/rc/<pool>/work \\
        data/compliant/rc/<pool>/waltz_*_contact*.npz

For a single file, --out gives the exact output; with neither option the PNG goes
to the out/ root, or next to the contact file when that is inside out/ (as in the
force plot). With --out-dir the output name is the contact file name with
`_contact` replaced by `_stiffness`. Colors / axes / event bands are the same as in
the force plot (SLOT_LINE already passed the legibility checks).
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # must precede pyplot: render without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from clip_evidence import ClipEvidence  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from plot_force_curve import (  # noqa: E402
  BAND,
  INK,
  INK_MUTED,
  _slot_color,
  draw_force_panel,
  tick_step,
)
from render_clip_video import default_out  # noqa: E402

# Free-frame placeholder values (augment_multilink.FREE_FRAME_*; frames after the
# last event and slots without events sit on this line).
FREE_K, FREE_K_ROT = 140.0, 1.0


def _panel(ax, ev: ClipEvidence, *, rotational: bool, env: bool) -> None:
  """One panel: per-slot stiffness curves + event shading.

  The robot side also gets the 140/1 placeholder reference line. env=True draws the
  force-field (environment) stiffness: by design it is nonzero only inside events
  (the gain of the reactive spring, whose force is always zero outside events) and
  has no placeholder value, so no reference line is drawn."""
  t = np.arange(ev.T)
  spans = sorted({(e["f_start"], e["f_end"]) for e in ev.events()})
  for f0, f1 in spans:
    ax.axvspan(f0, f1, color=BAND, zorder=0)
  if not env:
    ref = FREE_K_ROT if rotational else FREE_K
    ax.axhline(ref, color=INK_MUTED, lw=0.8, ls="--", alpha=0.6, zorder=2)
  for k in range(ev.K):
    curve = ev.stiffness_curve(k, rotational=rotational, env=env)
    ax.plot(
      t, curve, color=_slot_color(k), lw=1.8, zorder=3, label=ev.slot_link_names[k]
    )
  base = ("K_rot" if rotational else "K") + ("_env" if env else "")
  unit = "N*m/rad" if rotational else "N/m"
  ax.set_ylabel(f"{base}  ({unit})", fontsize=9, color=INK_MUTED)
  ax.set_xlim(0, float(t[-1]))
  ax.xaxis.set_major_locator(MultipleLocator(tick_step(ev.T)))
  ax.grid(axis="y", color=BAND, lw=0.8, zorder=1)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(BAND)
  ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot(contact: str, out: str | None = None) -> str:
  """Draw the combined panels: |F| on the top row, then K / K_env / K_rot /
  K_rot_env (raw values) below, five rows stacked with **equal width and a shared
  axis**.

  Equal width is the point: the force event bands and the stiffness curves must
  line up vertically, so one glance confirms "where force rises and falls the robot
  stiffness does not move, and K_env is nonzero only inside events". The x axis is
  frames, for the same reason as in the force plot: the contact NPZ has no fps and
  the event table itself is in frame indices. Only the force panel has a legend
  (slot colors are the same across the figure); rows other than the bottom one hide
  their x tick labels.
  """
  out = out or default_out(contact, ".png")
  ev = ClipEvidence(contact)
  fig, axes = plt.subplots(5, 1, figsize=(12, 10.5), dpi=160, sharex=True)
  draw_force_panel(axes[0], ev)
  axes[0].set_title(
    os.path.basename(contact), fontsize=10, color=INK, loc="left", pad=10
  )
  for ax, (rotational, env) in zip(
    axes[1:], [(False, False), (False, True), (True, False), (True, True)], strict=True
  ):
    _panel(ax, ev, rotational=rotational, env=env)
  for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
  axes[-1].set_xlabel("frame", fontsize=9, color=INK_MUTED)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  fig.savefig(out, facecolor="white")
  plt.close(fig)
  n_events = len({(e["f_start"], e["f_end"]) for e in ev.events()})
  print(f"wrote {out}  (T={ev.T} K={ev.K} events={n_events})")
  return out


def _derived_name(contact: str) -> str:
  """Contact file name -> stiffness plot name: `_contact` becomes `_stiffness`."""
  stem = os.path.splitext(os.path.basename(contact))[0]
  return stem.replace("_contact", "_stiffness") + ".png"


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "contacts",
    nargs="+",
    help="contact NPZs (multi-link or single-contact), one or more",
  )
  ap.add_argument(
    "--out-dir",
    default=None,
    help="output directory, names derived from the contact files; for a whole pool",
  )
  ap.add_argument(
    "--out", default=None, help="exact output PNG; only valid with a single input"
  )
  args = ap.parse_args()
  if args.out is not None and len(args.contacts) > 1:
    ap.error("--out takes a single input only; use --out-dir for several inputs")
  for contact in args.contacts:
    out = args.out or (
      os.path.join(args.out_dir, _derived_name(contact)) if args.out_dir else None
    )
    plot(contact, out)


def _demo() -> None:
  """Self-check: plot the seed-0 contact clip of the downloaded pool and assert
  the slot colors and the shading really land in the pixels (skipped if the file
  is missing)."""
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
  from plot_force_curve import SLOT_LINE

  out = os.path.join(tempfile.mkdtemp(), "stiffness.png")
  plot(contact, out)
  img = iio.imread(out)[:, :, :3]
  assert img.shape[0] > 400 and img.shape[1] > 1000, f"wrong image size: {img.shape}"
  for name, hexc in (("slot0", SLOT_LINE[0]), ("slot1", SLOT_LINE[1]), ("band", BAND)):
    rgb = np.array([int(hexc[i : i + 2], 16) for i in (1, 3, 5)])
    hits = int((np.abs(img.astype(int) - rgb).sum(axis=-1) < 30).sum())
    assert hits > 20, f"{name} ({hexc}) barely appears in the image: {hits} pixels"
  print(f"OK: {out}")


if __name__ == "__main__":
  import sys

  _demo() if len(sys.argv) == 1 else main()
