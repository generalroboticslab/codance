"""Shared scene wrappers for task-specific MuJoCo spec customization.

Spec functions (``Callable[[MjSpec], None]``) are composed declaratively: the
catalog lists named entries under ``scene.spec_fns`` and ``compose_run`` chains
them into one ``scene.spec_fn`` via :func:`chain_spec_fns`. Code that adds a fn
at run time (the executor's viz styling) appends it with :func:`append_spec_fn`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg

SpecFn = Callable[[mujoco.MjSpec], None]


def chain_spec_fns(fns: Sequence[SpecFn | None]) -> SpecFn | None:
  """Compose spec_fns left-to-right into one (``None`` entries are skipped).

  Returns ``None`` when nothing is active and the single fn unchanged when only
  one is active.
  """
  active = [fn for fn in fns if fn is not None]
  if not active:
    return None
  if len(active) == 1:
    return active[0]

  def chained(spec: mujoco.MjSpec) -> None:
    for fn in active:
      fn(spec)

  return chained


def append_spec_fn(cfg: ManagerBasedRlEnvCfg, fn: SpecFn) -> None:
  """Append ``fn`` after the cfg's existing ``scene.spec_fn`` (declarative
  composition for imperative call sites)."""
  cfg.scene.spec_fn = chain_spec_fns([cfg.scene.spec_fn, fn])


Rgba = tuple[float, float, float, float]

# Default-visible geom group. ``mujoco.MjvOption`` defaults to geomgroup
# ``[1,1,1,0,0,0]`` and viser to ``[T,T,T,F,F,F]``, so a geom placed here renders
# under every backend's DEFAULT options -- no per-renderer group flipping.
_DEFAULT_VISIBLE_GEOM_GROUP = 0


@dataclass(frozen=True)
class EntityGeomVizCfg:
  """One entity's two-axis main-scene viz (render-only, self-describing).

  WITHIN this entity (``prefix``), the composed model shows EXACTLY the listed
  visual meshes (``show_visual``) + collision geoms (``show_collision``, recolored
  ``collision_color``); every OTHER geom of the entity is hidden (alpha 0). So the
  two user-facing concepts are positive whitelists -- "which visual to show" +
  "which collision to show". Self-describing (alpha + a default-visible group), so
  all three backends render it under their DEFAULT options -- no viewer flag, no
  group-3/4 convention, no per-renderer code, no per-env vopt.

  The modes fall out of the two lists::

      capsules (today):    show_visual=()     show_collision=<all>
      capsules-over-mesh:  show_visual=<all>  show_collision=<all>  collision_alpha<1
      lower capsules:      show_visual=<upper> show_collision=<lower>

  Touches only ``geom.group`` / ``geom.rgba`` (render-only; physics reads
  ``contype``/``conaffinity``), so it never changes the measured trajectory."""

  prefix: str
  """Entity scope (e.g. ``"robot/"``): only geoms under it are touched; the rest of
  the scene (floor, other entities) is left alone."""
  show_visual: tuple[str, ...] = ()
  """Visual meshes kept visible (by ``meshname``); every other visual mesh of the
  entity is hidden. ``()`` hides all visual (the pure-capsule view)."""
  show_collision: tuple[str, ...] = ()
  """Collision geoms shown (by ``name``): moved into the default-visible group and
  recolored ``collision_color``."""
  collision_color: Rgba | None = None
  """Color for the shown collision geoms (a palette key resolved upstream)."""
  collision_alpha: float | None = None
  """Opacity override for the shown collision geoms; ``< 1`` overlays translucent
  capsules on the visual mesh (pair with a non-empty ``show_visual``)."""


def apply_scene_geom_viz(
  spec: mujoco.MjSpec, entries: Sequence[EntityGeomVizCfg]
) -> None:
  """Apply per-entity two-axis scene viz to a spec's geoms, in place.

  For each entry, scan the entity's geoms once (``geom.name`` / ``geom.meshname``
  under ``prefix``): a geom in ``show_collision`` is moved to the default-visible
  group + recolored (the entire collision/visual-viz algorithm); a geom in
  ``show_visual`` is left as-is; any OTHER geom of the entity is hidden (alpha 0).
  Identifiers that don't resolve are simply absent from the scan (an overlay may
  target a superset of what an asset has)."""
  for entry in entries:
    keep_visual = frozenset(entry.show_visual)
    show_collision = frozenset(entry.show_collision)
    color = (
      None
      if entry.collision_color is None
      else np.array(entry.collision_color, dtype=np.float32)
    )
    for geom in spec.geoms:
      if not (
        geom.name.startswith(entry.prefix) or geom.meshname.startswith(entry.prefix)
      ):
        continue
      if geom.name in show_collision or geom.meshname in show_collision:
        rgba = (
          color.copy() if color is not None else np.array(geom.rgba, dtype=np.float32)
        )
        if entry.collision_alpha is not None:
          rgba[3] = entry.collision_alpha
        geom.rgba = rgba
        geom.group = _DEFAULT_VISIBLE_GEOM_GROUP
      elif geom.name in keep_visual or geom.meshname in keep_visual:
        continue  # keep visible as-is
      else:
        rgba = np.array(geom.rgba, dtype=np.float32)
        rgba[3] = 0.0
        geom.rgba = rgba


def apply_body_colors(
  spec: mujoco.MjSpec, body_colors: Mapping[str, Sequence[float]]
) -> None:
  """Recolor every geom of the named bodies, in place (render-only).

  ``body_colors`` maps composed body names (entity prefix included, e.g.
  ``"robot/left_wrist_yaw_link"``) to an rgba. Each matched body's geoms take
  the color and drop their material (a material's own rgba wins over
  ``geom.rgba`` at render time, so the color would not show otherwise).
  Groups are untouched: hidden collision geoms stay hidden. A name matching
  no body raises, since a silently unstyled body is easy to miss in a
  render.
  """
  missing = set(body_colors)
  for body in spec.bodies:
    rgba = body_colors.get(body.name)
    if rgba is None:
      continue
    missing.discard(body.name)
    for geom in body.geoms:
      geom.rgba = np.array(rgba, dtype=np.float32)
      geom.material = ""
  if missing:
    raise ValueError(
      f"viz.scene.body_colors names unknown bodies: {sorted(missing)}. "
      "Use the composed name (entity prefix included, e.g. "
      "'robot/left_wrist_yaw_link')."
    )
