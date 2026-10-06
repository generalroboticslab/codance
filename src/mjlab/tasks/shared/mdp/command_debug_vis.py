"""Shared debug visualization for robot reference commands (coordinate frames).

NAMING. "Frame" is overloaded across this repo and this module deliberately uses
only one of its senses:

* **coord frame** (this module): a position + orientation drawn as three colored
  arrows. Every name here says ``coord_`` so it never reads as one of the others.
* **clip frame**: an integer time index into a motion clip (``num_frames``,
  ``event_f_start``, the ``per_frame`` re-anchor mode). Nothing here.
* **video frame**: one rendered RGB image (the recorder). Nothing here.
* **reference frame** in the measurement sense (``metric_reference_frame``:
  global vs anchor-aligned). Related, but that is the metrics vocabulary.

"Axis" is likewise overloaded (array reduction axis, rotation axis, plot axis),
so a coord frame's three arrows are always ``coord_axis_*``.

One concept: a :class:`CoordFrameLayer` is a labeled set of coord frames drawn
as a unit. Bodies and anchors are the same thing (an anchor is a one-body
layer), so a caller adds a face or an anchor by appending a layer, never by
growing this module's signature. Which layers make up a given task's reference
is that task's business; this module only draws them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  import torch

  from mjlab.viewer.debug_visualizer import DebugVisualizer

CoordAxisColors = tuple[tuple[float, float, float], ...]
"""One RGB color per arrow, in (x, y, z) order.

RGB only, no alpha: both viewer backends build each arrow's rgba as this color
plus the frame's single ``alpha``, so transparency is a property of the whole
coord frame rather than of one arrow. Set it via ``coord_axis_alpha``.
"""

# The per-face color sets themselves are NOT here: which faces a task draws and
# what they look like is that task's business, so they live in its palette
# (`codancing/viz_palette.py`, `COORD_AXIS_SETS`) and arrive through the fields
# below. This module only draws what it is handed.


@dataclass(frozen=True)
class CoordFrameStyle:
  """How ONE body's coord frame deviates from its layer's default.

  A field left None keeps the layer's value, so a style can change a body's
  color without touching its transparency and the other way round.
  """

  colors: CoordAxisColors | None = None
  alpha: float | None = None


@dataclass(frozen=True)
class CoordFrameLayer:
  """One labeled set of coordinate frames, drawn as a unit.

  Args:
    name: Label prefix; each coord frame is labeled ``f"{name}_{body_name}"``.
    body_names: One name per body, in the tensors' body order.
    pos_w: (num_envs, num_bodies, 3) world positions.
    quat_w: (num_envs, num_bodies, 4) world quaternions (wxyz).
    coord_axis_length: Arrow length, in meters. Overlapping faces read better
      when the live face is drawn longest, so a reader can tell the stack apart
      where the faces coincide.
    coord_axis_colors: The layer's default arrow colors, or None for the
      visualizer default (full-strength rgb), which is the live face's
      convention.
    coord_axis_alpha: Opacity for this layer's arrows, 0 to 1. Applies to all
      three arrows of a coord frame together (see :data:`CoordAxisColors`). Use
      it to push background faces back so the face a figure is about stays
      readable through them. NOTE: on the viser backend all arrows in the scene
      share one alpha, so a per-layer value there is not independent; the native
      and offscreen backends honor it per frame.
    body_styles: Per-body overrides, keyed by body name. A body listed here uses
      its own color and/or alpha; every body absent from the dict keeps the
      layer's. Use it to pick out the bodies a figure is about without
      recoloring the whole layer.
  """

  name: str
  body_names: tuple[str, ...]
  pos_w: torch.Tensor
  quat_w: torch.Tensor
  coord_axis_length: float = 0.08
  coord_axis_colors: CoordAxisColors | None = None
  coord_axis_alpha: float = 1.0
  body_styles: dict[str, CoordFrameStyle] = field(default_factory=dict)


def anchor_coord_frame(
  name: str,
  anchor_name: str,
  pos_w: torch.Tensor,
  quat_w: torch.Tensor,
  coord_axis_length: float = 0.1,
  coord_axis_colors: CoordAxisColors | None = None,
  coord_axis_alpha: float = 1.0,
) -> CoordFrameLayer:
  """A one-body layer from the ``(num_envs, ·)`` tensors anchors are stored as.

  Anchors are named, not implied: pass which anchor this is (``tracking``,
  ``forcefield``, ...) so a label survives a scene that draws more than one.
  """
  return CoordFrameLayer(
    name=name,
    body_names=(anchor_name,),
    pos_w=pos_w[:, None],
    quat_w=quat_w[:, None],
    coord_axis_length=coord_axis_length,
    coord_axis_colors=coord_axis_colors,
    coord_axis_alpha=coord_axis_alpha,
  )


def draw_coord_frames(
  visualizer: DebugVisualizer, layers: Sequence[CoordFrameLayer]
) -> None:
  """Draw every layer's per-body coordinate frames.

  Coord frames only: this helper never touches the backend's ``MjvOption``
  (ghost-mesh rendering is each command's own concern, via
  ``visualizer.add_ghost_mesh``).
  """
  env_idx = visualizer.env_idx
  for layer in layers:
    pos = layer.pos_w[env_idx].cpu().numpy()
    rotm = matrix_from_quat(layer.quat_w[env_idx]).cpu().numpy()
    for i, body_name in enumerate(layer.body_names):
      style = layer.body_styles.get(body_name)
      colors = layer.coord_axis_colors if style is None else style.colors
      alpha = layer.coord_axis_alpha if style is None else style.alpha
      visualizer.add_frame(
        position=pos[i],
        rotation_matrix=rotm[i],
        scale=layer.coord_axis_length,
        label=f"{layer.name}_{body_name}",
        alpha=layer.coord_axis_alpha if alpha is None else alpha,
        # `add_frame`'s own parameter keeps the core viewer's spelling; the
        # coord_ prefix stops at this module's boundary.
        axis_colors=layer.coord_axis_colors if colors is None else colors,
      )
