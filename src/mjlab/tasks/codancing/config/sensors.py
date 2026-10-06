"""Wrapper-layer sensors for codancing envs.

``GeomFromToSensor`` exposes MuJoCo's ``mjSENS_GEOMFROMTO`` (the closest-points
segment between two geoms) as a declarative ``SensorCfg`` so the hand-to-thigh
distance sensors live in the data catalog instead of an imperative ``spec_fn``
closure. It reuses ``BuiltinSensor``'s ``initialize`` / ``_compute_data`` (a plain
``sensordata`` view); only ``edit_spec`` differs because the core
``BuiltinSensorCfg`` type map does not cover ``GEOMFROMTO`` and its ``ref`` is
restricted to frame sensors. Kept here (wrapper layer) rather than in
``mjlab.sensor`` to avoid touching the core sensor package.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import BuiltinSensor, ObjRef
from mjlab.sensor.sensor import SensorCfg


@dataclass
class GeomFromToSensorCfg(SensorCfg):
  """A ``mjSENS_GEOMFROMTO`` sensor between two geoms (``obj`` -> ``ref``)."""

  obj: ObjRef | None = None
  """The geom the segment starts from."""
  ref: ObjRef | None = None
  """The geom the segment goes to."""
  cutoff: float = 0.0
  """When positive, clamps the absolute value of the sensor output."""

  @property
  def prefixed_name(self) -> str:
    """The sensor name with the ``obj`` entity prefix if applicable."""
    if self.obj is not None and self.obj.entity is not None:
      return f"{self.obj.entity}/{self.name}"
    return self.name

  def build(self) -> GeomFromToSensor:
    return GeomFromToSensor(self)


class GeomFromToSensor(BuiltinSensor):
  """GEOMFROMTO sensor: reuses ``BuiltinSensor`` data plumbing, custom edit_spec."""

  def __init__(self, cfg: GeomFromToSensorCfg) -> None:
    # Reuse BuiltinSensor's _name / sensordata-view plumbing; carry our own cfg
    # since it is not a BuiltinSensorCfg.
    super().__init__(cfg=None, name=cfg.prefixed_name)
    self._fromto_cfg = cfg

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities
    cfg = self._fromto_cfg
    assert cfg.obj is not None and cfg.ref is not None, (
      "GeomFromToSensorCfg requires both obj and ref geoms."
    )
    scene_spec.add_sensor(
      name=cfg.prefixed_name,
      type=mujoco.mjtSensor.mjSENS_GEOMFROMTO,
      objtype=mujoco.mjtObj.mjOBJ_GEOM,
      objname=cfg.obj.prefixed_name(),
      reftype=mujoco.mjtObj.mjOBJ_GEOM,
      refname=cfg.ref.prefixed_name(),
      cutoff=cfg.cutoff if cfg.cutoff > 0 else None,
    )
