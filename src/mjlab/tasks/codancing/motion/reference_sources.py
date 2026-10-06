"""Pluggable reference sources: the per-step target PRODUCER behind the cursor.

A ``ReferenceProvider`` turns the cursor ``(motion_id, motion_time)`` into the
reference ``MotionFrames`` the command tracks. :class:`ClipReferenceProvider`
wraps the ``MotionRegistry`` (byte-identical to reading the registry directly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from mjlab.tasks.codancing.motion.registry import MotionFrames

if TYPE_CHECKING:
  from mjlab.tasks.codancing.motion.registry import MotionRegistry

__all__ = [
  "ClipReferenceProvider",
  "ReferenceProvider",
]


class ReferenceProvider(Protocol):
  """The per-step reference producer behind the cursor."""

  def get_reference(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrames:
    """Reference frames at the cursor, one row per query."""
    ...

  def num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Per-env clip length in frames (the clip-end clock); always finite."""
    ...


class ClipReferenceProvider:
  """The recorded-clip source: a thin view over the ``MotionRegistry``."""

  def __init__(self, registry: "MotionRegistry") -> None:
    self._registry = registry

  def get_reference(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrames:
    return self._registry.get_frames(motion_ids, motion_times)

  def num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self._registry.motion_num_frames[motion_ids].float()
