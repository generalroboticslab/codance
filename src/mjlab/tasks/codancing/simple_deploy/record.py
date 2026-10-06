"""Per-run record: every tick's state/obs/action/command/timing to NPZ plus
a JSON identity file (artifact, clip, stiffness, host, git, gains).

Providers add channels the runner does not see itself (the VICON source's
world-frame poses); ``float64`` names the channels that must not be squeezed
to float32 (a wall-clock epoch loses 2 minutes of resolution there)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np


class RunRecorder:
  def __init__(
    self, root: str | Path, identity: dict, *, float64: tuple[str, ...] = ("t_wall",)
  ) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    self.dir = Path(root) / stamp
    self.dir.mkdir(parents=True, exist_ok=True)
    identity = dict(identity)
    identity.update(
      host=platform.node(),
      git_head=_git_head(),
      stamp=stamp,
    )
    (self.dir / "run.json").write_text(json.dumps(identity, indent=2, default=str))
    self._rows: dict[str, list[np.ndarray]] = {}
    self._events: list[tuple[float, str]] = []
    self._float64 = set(float64)
    self._providers: list[Callable[[], dict[str, np.ndarray | float]]] = []

  def add_provider(self, fn: Callable[[], dict[str, np.ndarray | float]]) -> None:
    """Register a per-tick channel source, read at every ``tick``."""
    self._providers.append(fn)

  def tick(self, **channels: np.ndarray | float | int) -> None:
    for fn in self._providers:
      channels.update(fn())
    for key, value in channels.items():
      self._rows.setdefault(key, []).append(np.asarray(value))

  def event(self, message: str) -> None:
    self._events.append((time.time(), message))
    print(f"[deploy] {message}")

  def save(self) -> Path:
    arrays = {
      k: np.stack(v).astype(np.float64 if k in self._float64 else np.float32)
      for k, v in self._rows.items()
      if v
    }
    path = self.dir / "ticks.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[call-overload]
    (self.dir / "events.json").write_text(json.dumps(self._events, indent=2))
    return path


def _git_head() -> str:
  try:
    out = subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
    )
    return out.stdout.strip()
  except Exception:
    return "unknown"
