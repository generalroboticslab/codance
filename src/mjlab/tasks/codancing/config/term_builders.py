"""Term-building primitives: thin YAML sugar over native Hydra ``instantiate``.

The compose engine merges manager term nodes via Hydra config groups; this module
only *expands* a compact term node into the native ``_target_`` form that
``instantiate`` expects, then builds it. ``instantiate`` does the real work.

Sugar (per term node):
* ``func: a.b.c``           -> ``func: {_target_: hydra_zen.funcs.get_obj, path: a.b.c}``
* ``noise: {n_min, n_max}`` -> a ``UniformNoiseCfg`` ``_target_`` node
Everything else (``params``, ``weight``, ``mode``, ``interval_range_s``,
``time_out``, ...) is copied through; nested ``_target_`` nodes inside ``params``
(e.g. ``SceneEntityCfg``) are built by ``instantiate`` recursively. A node that
already carries a ``_target_`` is passed through untouched.

The only Python leaves are the ``mdp.*`` callables a term points at; noise is
fully explicit per term (a term either has a ``noise`` block or it does not).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from hydra_zen import instantiate
from omegaconf import OmegaConf

_OBS_TERM_TARGET = "mjlab.managers.observation_manager.ObservationTermCfg"
_REWARD_TERM_TARGET = "mjlab.managers.reward_manager.RewardTermCfg"
_EVENT_TERM_TARGET = "mjlab.managers.event_manager.EventTermCfg"
_TERMINATION_TERM_TARGET = "mjlab.managers.termination_manager.TerminationTermCfg"
_CURRICULUM_TERM_TARGET = "mjlab.managers.curriculum_manager.CurriculumTermCfg"
_METRICS_TERM_TARGET = "mjlab.managers.metrics_manager.MetricsTermCfg"
_RECORDER_TERM_TARGET = "mjlab.managers.recorder_manager.RecorderTermCfg"
_GET_OBJ_TARGET = "hydra_zen.funcs.get_obj"
_UNOISE_TARGET = "mjlab.utils.noise.UniformNoiseCfg"


def _prepare(node: Any, default_target: str | None) -> dict[str, Any]:
  """Expand a sugary term node into a native Hydra ``_target_`` config.

  Pure shape rewriting -- ``instantiate`` does the real work. Nodes already
  carrying a ``_target_`` are returned as-is (only ``_convert_`` is ensured).
  """
  container = OmegaConf.to_container(node, resolve=True)
  assert isinstance(container, dict)
  data = cast("dict[str, Any]", container)

  spec: dict[str, Any]
  if "_target_" in data:
    spec = dict(data)
  else:
    assert default_target is not None, "sugary node needs a default target"
    spec = {"_target_": default_target}
    for key, value in data.items():
      if key == "func":
        spec["func"] = (
          value
          if isinstance(value, Mapping)
          else {"_target_": _GET_OBJ_TARGET, "path": value}
        )
      elif key == "noise":
        spec["noise"] = (
          value
          if isinstance(value, Mapping) and "_target_" in value
          else {"_target_": _UNOISE_TARGET, **value}
        )
      else:
        spec[key] = value
  # Return plain dicts/lists (and built nested _target_ nodes), not DictConfig.
  spec.setdefault("_convert_", "all")
  return spec


def _build_terms(nodes: Any, default_target: str | None = None) -> dict[str, Any]:
  """Instantiate a mapping of term nodes (name -> node).

  A ``null`` node passes through as ``None``: every manager's ``_prepare_terms``
  skips None terms by convention, so ``env.<manager>.<term>: null`` (overlay or
  ``--override``) is the per-term disable switch, with no group-list restating.
  """
  return {
    name: None if node is None else instantiate(_prepare(node, default_target))
    for name, node in nodes.items()
  }
