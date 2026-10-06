"""The rollout-sink protocol, shared by every play mode.

A :class:`RolloutSink` is anything that observes a rollout step by step and may
write something out at the end: the rollout table recorder. It is NOT
viewer-specific -- the same sinks run under the headless
:func:`run_bounded_rollout`, the native viewers, and the viser viewer -- so the
protocol lives here, away from the viewer classes that merely happen to drive
it.

The protocol has two steps:

- ``on_step(env)`` -- called after each env step (record / snapshot).
- ``finalize()``   -- write the output file.

No codancing imports -- this is a leaf.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
  "RolloutSink",
  "observe_step",
]


@runtime_checkable
class RolloutSink(Protocol):
  """Post-step observer that may write an artifact at the end of a rollout."""

  def on_step(self, env: Any) -> None: ...
  def finalize(self) -> None: ...


def observe_step(sink: RolloutSink | None, env: Any) -> None:
  """Run one observe cycle.

  The single entry point for the native viewers (in ``_execute_step``) and the
  headless loop, so streaming / frozen / headless share identical step logic.
  """
  if sink is not None:
    sink.on_step(env)
