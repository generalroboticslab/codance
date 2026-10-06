"""The one resolver from body selections to model-order body indices.

``conf/body_sets/g1.yaml`` is the source of the NAMES; the composed config
carries the whole registry onto the command cfg (``body_sets: ${body_sets}``
in the catalog). Two resolution forms live here, one per order semantic:

* ``body_set_indices``: a registry SET name to indices, in the set's own
  declaration order (metric groups; every registry set declares in model
  order, so gather order is stable).
* ``model_order_indexes``: an explicit body-name LIST to indices, in MODEL
  order regardless of how the list is spelled (the ``body_names`` term
  parameters in rewards/observations, coord-frame body filters).

Both raise on an unknown name instead of silently shrinking the selection,
and every name-keyed body-scoped consumer resolves through here, so there is
exactly one vocabulary. Deliberately NOT routed through here: resolutions
whose COLUMN ORDER is itself the contract, the AMP feature body order
(checkpoint-frozen feature columns), the partner obs ``body_names`` order
(checkpoint-frozen obs columns), and the force events' per-slot link order
(slot correspondence). Those stay positional ``names.index`` loops at their
call sites.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class BodySetSource(Protocol):
  """The one cfg field the resolver reads (any command cfg satisfies it)."""

  @property
  def body_sets(self) -> Mapping[str, Sequence[str]]: ...


def body_set_names(cfg: BodySetSource, name: str) -> tuple[str, ...]:
  """The named registry set's body names (raises on an unknown set name)."""
  try:
    return tuple(cfg.body_sets[name])
  except KeyError:
    raise ValueError(
      f"unknown body set {name!r}; the registry carries "
      f"{sorted(cfg.body_sets)}. Add the set to conf/body_sets/g1.yaml "
      "(or cfg.body_sets on a Python-built cfg)."
    ) from None


def model_order_indexes(
  body_names: Sequence[str], model_body_names: Sequence[str]
) -> list[int]:
  """Column indexes of the named bodies, in MODEL order.

  Filter order (independent of how the list is spelled), so reductions over
  the gathered columns are stable across config spellings. Unknown names
  raise instead of silently shrinking the selection.
  """
  wanted = set(body_names)
  missing = wanted - set(model_body_names)
  if missing:
    raise ValueError(
      f"body names not in the model: {sorted(missing)} "
      f"(model bodies: {list(model_body_names)})."
    )
  return [i for i, name in enumerate(model_body_names) if name in wanted]


def body_set_indices(
  cfg: BodySetSource, name: str, body_names: Sequence[str]
) -> tuple[int, ...]:
  """Indices into ``body_names`` (entity model order) for the named set.

  Returned in the set's own declaration order, which for every registry set
  matches the model-order restriction, so reductions over the gathered
  columns are stable.
  """
  set_names = body_set_names(cfg, name)
  known = list(body_names)
  missing = [n for n in set_names if n not in known]
  if missing:
    raise ValueError(
      f"body set {name!r} names bodies outside the model: {missing} "
      f"(model bodies: {known})."
    )
  return tuple(known.index(n) for n in set_names)
