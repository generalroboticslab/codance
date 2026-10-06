"""The repo palette, reached without importing `mjlab.tasks`.

`src/mjlab/tasks/codancing/viz_palette.py` is the single home for every color
this repo draws (its docstring has the name grammar and the media). Importing it
the ordinary way would run `mjlab/tasks/__init__.py`, which imports every task
package: ~5 s and a torch load that a figure script has no use for, plus a
failure mode where an unrelated task's registration takes the figure down with
it. The palette module itself imports nothing outside the standard library, so
loading it from its own file is exact -- same file, same values the sim viz
reads, no package side effects.

    from palette import PAL
    INK = PAL.MATPLOTLIB.hex("ink")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE = (
  Path(__file__).resolve().parents[2]
  / "src"
  / "mjlab"
  / "tasks"
  / "codancing"
  / "viz_palette.py"
)

_spec = importlib.util.spec_from_file_location("mjlab_viz_palette", _SOURCE)
if _spec is None or _spec.loader is None:  # pragma: no cover - packaging error
  raise ImportError(f"cannot load the palette from {_SOURCE}")
PAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PAL)


def figure_defaults() -> None:
  """Point matplotlib at the palette's typeface (family only; sizes are
  per-surface). ``fontweight="bold"`` then picks up its Bold face. Call once at
  import time in a plotting script; the explicit resolve is what surfaces the
  palette's warning when the family is not installed."""
  import matplotlib

  matplotlib.rcParams["font.family"] = PAL.FONT_FAMILY
  # Embed real fonts, not Type 3. Matplotlib's default (pdf.fonttype 3) writes
  # each glyph as a little PDF drawing program, which some venues' PDF checkers
  # reject. 42 means TrueType, so the text stays selectable and the figure needs
  # no post-processing. Set here rather than per script, so every figure that
  # calls figure_defaults is compliant by default.
  matplotlib.rcParams["pdf.fonttype"] = 42
  matplotlib.rcParams["ps.fonttype"] = 42
  PAL.font_file()
